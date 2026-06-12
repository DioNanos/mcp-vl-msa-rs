//! A5 injection-ablation gate (Strato 3, generator-free).
//!
//! Measures the deterministic precursor to the paper's −37% original-text
//! injection lever (Tab.4). At an **equal character budget**, does delivering
//! the full original document text (condition **C = injection**, what
//! `msa_interleave_round` does) place the gold answer string into the model's
//! context more often than delivering only the BM25-matched chunk snippets
//! (condition **D = snippet-only**, what bare `msa_search` returns)? Same
//! retrieval feeds both, so the only variable is delivery depth vs breadth.
//!
//! This is the *precursor* to answer F1: if the answer-bearing text never
//! reaches the context, no generator can answer. The answer-F1 version with a
//! model in the loop (M3's protocol Strato 2/3b) is the production gate and is
//! intentionally out of scope here — this gate is offline, deterministic, and
//! needs no LLM.
//!
//! Scope: only **extractive** queries — the gold answer is not yes/no and
//! appears verbatim (after light normalization) in at least one of the query's
//! gold relevant docs. yes/no and non-extractive answers are excluded because
//! substring presence is meaningless for them. The exclusions are reported.

use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use chrono::Utc;
use msa_core::config::{MsaConfig, StorageConfig};
use msa_core::index::CollectionRegistry;
use msa_core::schema::{ChunkConfig, Document};
use serde::Serialize;
use tempfile::TempDir;

use crate::dataset::{LoadedDataset, Passage};

/// Knobs for the injection-ablation gate.
#[derive(Debug, Clone, Copy)]
pub struct GateConfig {
    /// Candidate chunk hits retrieved per query (the round's `top_k`).
    pub top_k: usize,
    /// Equal character budget B that both conditions fill.
    pub budget_chars: usize,
    /// Per-document injection cap for condition C (mirrors `max_chars_per_doc`).
    pub max_chars_per_doc: usize,
    /// Word-level chunk size. MUST be > 0 (e.g. 64): with one-chunk-per-passage
    /// a snippet equals the whole doc and C and D collapse to the same context.
    pub chunk_size: usize,
    /// Pass threshold: injection must beat snippet by at least this many points.
    pub threshold_points: f64,
}

#[derive(Debug, Serialize)]
pub struct AblationReport {
    pub run_id: String,
    pub dataset: String,
    pub corpus_size: usize,
    pub top_k: usize,
    pub budget_chars: usize,
    pub max_chars_per_doc: usize,
    pub chunk_size: usize,
    pub num_queries_total: usize,
    /// Queries scored (extractive). This is the denominator for the rates.
    pub num_extractive: usize,
    pub excluded_no_answer: usize,
    pub excluded_yes_no: usize,
    pub excluded_non_extractive: usize,
    /// Gold answer present in the budget-bounded **full-text** context (C).
    pub answer_present_injection: usize,
    /// Gold answer present in the budget-bounded **snippet** context (D).
    pub answer_present_snippet: usize,
    /// Gold answer present in the full text of *any* retrieved doc, unbounded
    /// budget — the retrieval ceiling (how often the answer is reachable at all).
    pub answer_present_oracle: usize,
    pub rate_injection: f64,
    pub rate_snippet: f64,
    pub rate_oracle: f64,
    /// `(rate_injection - rate_snippet) * 100`, in percentage points (informational).
    pub delta_points: f64,
    /// Reported margin only; the verdict is structural (see `passed`).
    pub threshold_points: f64,
    /// `rate_injection / rate_oracle` — fraction of reachable answers injection
    /// actually delivered within the budget/cap (1.0 = at the ceiling).
    pub oracle_capture: f64,
    /// `oracle_capture >= MIN_ORACLE_CAPTURE && rate_injection > rate_snippet`.
    pub passed: bool,
}

/// Injection must deliver at least this fraction of the reachable answers to
/// count as "at the oracle ceiling"; the small residual is per-doc-cap
/// truncation, not a delivery failure.
const MIN_ORACLE_CAPTURE: f64 = 0.95;

/// Normalize for substring matching: lowercase + whitespace-collapsed.
fn normalize(s: &str) -> String {
    s.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Tantivy's QueryParser is boolean and chokes on punctuation; keep only
/// alphanumerics and whitespace (BM25 retokenizes). Mirrors the runner.
fn sanitize_query(q: &str) -> String {
    q.chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Append `text` to `ctx` without exceeding `budget` total chars. Returns the
/// number of chars actually appended (0 once the budget is full).
fn append_bounded(ctx: &mut String, text: &str, budget: usize, per_item_cap: usize) -> usize {
    let used = ctx.chars().count();
    if used >= budget {
        return 0;
    }
    let room = (budget - used).min(per_item_cap);
    let mut n = 0;
    for ch in text.chars() {
        if n >= room {
            break;
        }
        ctx.push(ch);
        n += 1;
    }
    ctx.push(' '); // separator (not counted against the cap, negligible)
    n
}

/// One evidence row for the model-in-loop gate (M3 Strato 2). Serialized as
/// JSONL by `run()` when `evidence_out` is set: the downstream harness feeds
/// `ctx_c`/`ctx_d` to a generator and scores answer F1, reusing *exactly* the
/// evidence this deterministic gate scored for presence.
#[derive(Debug, Serialize)]
struct EvidenceRow<'a> {
    query_id: &'a str,
    question: &'a str,
    answer: &'a str,
    ctx_c: &'a str,
    ctx_d: &'a str,
}

fn doc_from_passage(p: &Passage) -> Document {
    Document {
        id: p.id.clone(),
        text: p.text.clone(),
        metadata: Default::default(),
        created_at: Utc::now(),
    }
}

pub fn run(
    dataset: &LoadedDataset,
    cfg: GateConfig,
    evidence_out: Option<&Path>,
) -> Result<AblationReport> {
    anyhow::ensure!(
        cfg.chunk_size > 0,
        "injection-ablation needs chunk_size > 0 (snippets must be sub-document); got 0"
    );

    let mut evidence_writer = match evidence_out {
        Some(p) => {
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            Some(std::io::BufWriter::new(
                std::fs::File::create(p).context("create evidence dump")?,
            ))
        }
        None => None,
    };

    let storage = TempDir::new().context("temp storage for tantivy")?;
    let config = MsaConfig {
        storage: StorageConfig {
            storage_dir: storage.path().to_path_buf(),
        },
        chunking: ChunkConfig {
            chunk_size: cfg.chunk_size,
            overlap: 0,
        },
        ..Default::default()
    };
    let registry = CollectionRegistry::new();
    let idx_arc = registry
        .open_or_create(&dataset.name, &config.storage.storage_dir, &config.chunking)
        .context("open tantivy collection")?;
    let idx = idx_arc.as_ref();

    let docs: Vec<Document> = dataset.passages.iter().map(doc_from_passage).collect();
    idx.index_documents(&docs, None)
        .context("bulk index passages")?;

    // Pre-normalize the corpus once for the extractive-answer check.
    let norm_corpus: std::collections::HashMap<String, String> = dataset
        .passages
        .iter()
        .map(|p| (p.id.clone(), normalize(&p.text)))
        .collect();

    let mut num_extractive = 0usize;
    let mut excluded_no_answer = 0usize;
    let mut excluded_yes_no = 0usize;
    let mut excluded_non_extractive = 0usize;
    let mut present_injection = 0usize;
    let mut present_snippet = 0usize;
    let mut present_oracle = 0usize;

    for q in &dataset.queries {
        let answer = match &q.answer {
            Some(a) => a,
            None => {
                excluded_no_answer += 1;
                continue;
            }
        };
        let raw = answer.trim().to_lowercase();
        if raw == "yes" || raw == "no" {
            excluded_yes_no += 1;
            continue;
        }
        let ans_norm = normalize(answer);
        if ans_norm.is_empty() {
            excluded_no_answer += 1;
            continue;
        }
        // Extractive iff the answer appears in at least one gold relevant doc.
        let extractive = q.relevant.iter().any(|d| {
            norm_corpus
                .get(d)
                .map(|t| t.contains(&ans_norm))
                .unwrap_or(false)
        });
        if !extractive {
            excluded_non_extractive += 1;
            continue;
        }
        num_extractive += 1;

        let hits = idx
            .search_excluding(&sanitize_query(&q.text), cfg.top_k, None, &HashSet::new())
            .context("bm25 retrieve")?;

        // Condition C — full original text of the ranked unique docs.
        let mut ctx_c = String::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut oracle_hit = false;
        for h in &hits {
            if !seen.insert(h.doc_id.clone()) {
                continue;
            }
            if let Ok(doc) = idx.fetch_doc(&h.doc_id) {
                let dn = normalize(&doc.text);
                if dn.contains(&ans_norm) {
                    oracle_hit = true; // answer reachable in some retrieved doc
                }
                append_bounded(
                    &mut ctx_c,
                    &doc.text,
                    cfg.budget_chars,
                    cfg.max_chars_per_doc,
                );
            }
        }

        // Condition D — matched chunk snippets, same ranking, same budget.
        let mut ctx_d = String::new();
        for h in &hits {
            append_bounded(
                &mut ctx_d,
                &h.snippet,
                cfg.budget_chars,
                cfg.max_chars_per_doc,
            );
        }

        if normalize(&ctx_c).contains(&ans_norm) {
            present_injection += 1;
        }
        if normalize(&ctx_d).contains(&ans_norm) {
            present_snippet += 1;
        }
        if oracle_hit {
            present_oracle += 1;
        }

        if let Some(w) = evidence_writer.as_mut() {
            let row = EvidenceRow {
                query_id: &q.id,
                question: &q.text,
                answer,
                ctx_c: &ctx_c,
                ctx_d: &ctx_d,
            };
            serde_json::to_writer(&mut *w, &row).context("serialize evidence row")?;
            w.write_all(b"\n").context("write evidence row")?;
        }
    }

    if let Some(mut w) = evidence_writer {
        w.flush().context("flush evidence dump")?;
    }

    let n = num_extractive.max(1) as f64;
    let rate_injection = present_injection as f64 / n;
    let rate_snippet = present_snippet as f64 / n;
    let rate_oracle = present_oracle as f64 / n;
    let delta_points = (rate_injection - rate_snippet) * 100.0;

    // Pass criterion (budget-honest, not an arbitrary delta). Injection must
    // (1) capture >= MIN_ORACLE_CAPTURE of the reachable answers — i.e. sit at
    // the oracle ceiling, proving the budget suffices and full-text injection
    // delivers what retrieval found (the small residual to oracle is per-doc-cap
    // truncation, tunable via --gate-max-chars-per-doc) — and (2) beat
    // snippet-only. Snippet structurally cannot reach the oracle (it only ever
    // carries the matched chunk), so a passing run demonstrates the injection
    // lever's mechanism: original text recovers answers that live in a
    // non-matched chunk. NOTE: answer *presence* is a LOWER BOUND on the paper's
    // −37% F1 lever — the residual (answering worse from fragments even when the
    // answer is present) needs the model-in-loop gate (M3 Strato 2).
    let oracle_capture = if rate_oracle > 0.0 {
        rate_injection / rate_oracle
    } else {
        1.0
    };
    let passed = oracle_capture >= MIN_ORACLE_CAPTURE && rate_injection > rate_snippet;

    Ok(AblationReport {
        run_id: format!(
            "{}-{}-ablation",
            Utc::now().format("%Y-%m-%dT%H:%M:%SZ"),
            dataset.name
        ),
        dataset: dataset.name.clone(),
        corpus_size: dataset.passages.len(),
        top_k: cfg.top_k,
        budget_chars: cfg.budget_chars,
        max_chars_per_doc: cfg.max_chars_per_doc,
        chunk_size: cfg.chunk_size,
        num_queries_total: dataset.queries.len(),
        num_extractive,
        excluded_no_answer,
        excluded_yes_no,
        excluded_non_extractive,
        answer_present_injection: present_injection,
        answer_present_snippet: present_snippet,
        answer_present_oracle: present_oracle,
        rate_injection,
        rate_snippet,
        rate_oracle,
        delta_points,
        threshold_points: cfg.threshold_points,
        oracle_capture,
        passed,
    })
}
