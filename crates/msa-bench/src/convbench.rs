//! Conversational memory bench over LongMemEval-S (design 2026-06-05 v2,
//! audit APPROVE: `CONV_BENCH_DESIGN_V2`).
//!
//! Per-question haystacks: each question carries ~50 chat sessions with
//! dates; we ingest them as deterministic TURN-level "capsule proxies"
//! (short = first 120 chars of the turn, rich = `enrich`-composed detail
//! up to 800 chars — NO LLM in the loop: this measures the index, not the
//! summarizer) and measure retrieval per question category under:
//!
//! - C1 `bm25-short` — baseline (pre-rich Vivling memory)
//! - C2 `bm25-rich` — rich capsules
//! - C3a `recency` — C2 + exponential recency prior (bench-only knob);
//!   reads ONLY on knowledge-update (NOT temporal!)
//! - C3b `oracle-band` — C2 restricted to sessions whose DATE falls within
//!   the gold sessions' date band (±1 day). Uses gold TIMESTAMPS, never gold
//!   session ids: an upper bound for time-aware retrieval (audit note:
//!   filtering on the answer docs themselves would be answer leak).
//! - C4 `hybrid` — C2 with dense+BM25 mix (run on a subset, embeddings are
//!   the cost driver) [wired by the caller]
//!
//! Output: one JSON line per (question, condition) with hit ranks and
//! evidence coverage; aggregation/CI happens downstream (Python report).

use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use chrono::NaiveDate;
use chrono::NaiveDateTime;
use msa_core::config::MsaConfig;
use msa_core::config::StorageConfig;
use msa_core::consolidate::{
    self, ConsolidateOptions, DeterministicConcatBackend, DistillBackend, DistillError,
    SessionKeySource,
};
use msa_core::enrich;
use msa_core::index::CollectionRegistry;
use msa_core::schema::{ChunkConfig, ChunkHit, Document};
use serde::Deserialize;
use serde::Serialize;
use tempfile::TempDir;

const SHORT_CHARS: usize = 120;
const RICH_DETAIL_CHARS: usize = 800;
const TOP_K: usize = 10;
const RECENCY_HALF_LIFE_DAYS: f32 = 30.0;
const ORACLE_BAND_SLACK_DAYS: i64 = 1;
/// Retrieve more than TOP_K before re-ranking so the recency prior has
/// candidates to promote.
const RERANK_POOL: usize = 50;

// ------------------------------------------------------------------ input

#[derive(Debug, Deserialize)]
struct LmeQuestion {
    question_id: String,
    question_type: String,
    question: String,
    #[allow(dead_code)]
    #[serde(default)]
    answer: serde_json::Value,
    question_date: String,
    haystack_dates: Vec<String>,
    haystack_session_ids: Vec<String>,
    haystack_sessions: Vec<Vec<LmeTurn>>,
    answer_session_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct LmeTurn {
    role: String,
    content: String,
}

// ----------------------------------------------------------------- output

#[derive(Debug, Serialize)]
struct QuestionResult<'a> {
    question_id: &'a str,
    question_type: &'a str,
    condition: &'a str,
    /// rank (0-based) of the first hit whose session is a gold session.
    first_gold_rank: Option<usize>,
    /// gold sessions covered within TOP_K / total gold sessions.
    evidence_covered: usize,
    evidence_total: usize,
    /// candidate pool size after any condition filter (oracle band).
    pool_sessions: usize,
    /// cons-* conditions: digests written for this question's collection.
    #[serde(skip_serializing_if = "Option::is_none")]
    digests_written: Option<usize>,
}

fn parse_date(s: &str) -> Option<NaiveDateTime> {
    // haystack/question dates look like "2023/05/20 (Sat) 02:21" or similar;
    // be liberal: take the leading YYYY/MM/DD.
    let head: String = s.chars().take(10).collect();
    NaiveDate::parse_from_str(&head, "%Y/%m/%d")
        .ok()
        .and_then(|d| d.and_hms_opt(0, 0, 0))
}

fn turn_text_short(turn: &LmeTurn) -> String {
    let collapsed: String = turn
        .content
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    collapsed.chars().take(SHORT_CHARS).collect()
}

fn turn_text_rich(turn: &LmeTurn) -> String {
    let collapsed: String = turn
        .content
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let summary: String = collapsed.chars().take(SHORT_CHARS).collect();
    let detail: String = collapsed.chars().skip(SHORT_CHARS).collect();
    let sanitized = enrich::sanitize_detail(&detail);
    enrich::compose_rich_text(&summary, &sanitized, RICH_DETAIL_CHARS).text
}

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

/// Session id for a chunk-hit doc id of form `s<idx>::t<idx>`.
fn session_of(doc_id: &str) -> &str {
    doc_id.split("::").next().unwrap_or(doc_id)
}

pub struct ConvBenchArgs<'a> {
    pub dataset_json: &'a Path,
    pub output_jsonl: &'a Path,
    pub limit_questions: Option<usize>,
    /// run only these conditions (None = all bm25 conditions).
    pub conditions: Option<Vec<String>>,
    /// C4 hybrid: dense weight (design: 0.3). BM25 pool is re-ranked by
    /// `alpha*bm25_norm + (1-alpha)*cosine` over the top candidates only —
    /// embeddings are the cost driver, the corpus is never embedded.
    pub hybrid_alpha: f32,
    #[cfg(feature = "bench-ollama")]
    pub ollama: Option<(String, String, usize, usize, usize)>, // url, model, dim, num_ctx, num_thread
    /// In-process ONNX encoder (no server). Takes precedence over `ollama`
    /// when set — used when the local Ollama server cannot load embedding
    /// models (e.g. huge OLLAMA_CONTEXT_LENGTH set for cloud relays).
    #[cfg(feature = "bench-fastembed")]
    pub fastembed_model: Option<String>,
    /// Dump per-question evidence (top-K rich capsule texts from the
    /// `bm25-rich` condition) for the model-in-loop harness, Strato-2 style.
    pub emit_evidence: Option<std::path::PathBuf>,
}

pub fn run(args: &ConvBenchArgs) -> Result<()> {
    let file = File::open(args.dataset_json).context("open longmemeval json")?;
    let questions: Vec<LmeQuestion> =
        serde_json::from_reader(BufReader::new(file)).context("parse longmemeval json")?;
    let total = questions.len();
    let take = args.limit_questions.unwrap_or(total).min(total);

    let wanted = |c: &str| {
        args.conditions
            .as_ref()
            .map(|v| v.iter().any(|x| x == c))
            .unwrap_or(true)
    };

    if let Some(parent) = args.output_jsonl.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut out = BufWriter::new(File::create(args.output_jsonl).context("create output")?);
    let mut evidence_out = match args.emit_evidence.as_ref() {
        Some(p) => {
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            Some(BufWriter::new(
                File::create(p).context("create evidence dump")?,
            ))
        }
        None => None,
    };

    for (qi, q) in questions.iter().take(take).enumerate() {
        // Abstention questions (id suffix `_abs`) are reported but flagged by
        // category downstream; we keep them out of nothing here — the report
        // separates them.
        let storage = TempDir::new().context("temp storage")?;
        let config = MsaConfig {
            storage: StorageConfig {
                storage_dir: storage.path().to_path_buf(),
            },
            chunking: ChunkConfig {
                // One chunk per capsule: capsules are <=920 chars, the chunk
                // is the retrieval unit exactly like the vivling index.
                chunk_size: 512,
                overlap: 0,
            },
            ..Default::default()
        };
        let registry = CollectionRegistry::new();

        let session_dates: Vec<Option<NaiveDateTime>> =
            q.haystack_dates.iter().map(|d| parse_date(d)).collect();
        let question_date = parse_date(&q.question_date);

        // Gold session date band (oracle C3b): timestamps of gold sessions
        // ±slack. Built from DATES only — gold ids are never used to filter.
        let gold_ids: HashSet<&str> = q.answer_session_ids.iter().map(String::as_str).collect();
        let gold_dates: Vec<NaiveDateTime> = q
            .haystack_session_ids
            .iter()
            .zip(session_dates.iter())
            .filter(|(sid, _)| gold_ids.contains(sid.as_str()))
            .filter_map(|(_, d)| *d)
            .collect();
        let band = if gold_dates.is_empty() {
            None
        } else {
            let min = gold_dates.iter().min().unwrap().date()
                - chrono::Days::new(ORACLE_BAND_SLACK_DAYS as u64);
            let max = gold_dates.iter().max().unwrap().date()
                + chrono::Days::new(ORACLE_BAND_SLACK_DAYS as u64);
            Some((min, max))
        };

        // Map internal session index -> (external id, date, in_band).
        struct SessInfo<'a> {
            // Kept for evidence-dump symmetry; not read in the scoring path.
            #[allow(dead_code)]
            ext_id: &'a str,
            date: Option<NaiveDateTime>,
            in_band: bool,
        }
        let sess_infos: Vec<SessInfo> = q
            .haystack_session_ids
            .iter()
            .enumerate()
            .map(|(i, sid)| {
                let date = session_dates.get(i).copied().flatten();
                let in_band = match (band, date) {
                    (Some((lo, hi)), Some(d)) => d.date() >= lo && d.date() <= hi,
                    _ => band.is_none(),
                };
                SessInfo {
                    ext_id: sid,
                    date,
                    in_band,
                }
            })
            .collect();

        // Build the two indexes (short, rich) over all turns.
        let mut docs_short: Vec<Document> = Vec::new();
        let mut docs_rich: Vec<Document> = Vec::new();
        for (si, session) in q.haystack_sessions.iter().enumerate() {
            let created = sess_infos[si]
                .date
                .map(|d| d.and_utc())
                .unwrap_or_else(chrono::Utc::now);
            for (ti, turn) in session.iter().enumerate() {
                if turn.content.trim().is_empty() {
                    continue;
                }
                // Match the vivling ingestion: assistant + user turns both
                // become capsules (the capsule source is assistant-visible
                // text; user turns carry the requests the memory must recall).
                let _ = &turn.role;
                let doc_id = format!("s{si}::t{ti}");
                let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
                metadata.insert("session".into(), serde_json::json!(si));
                docs_short.push(Document {
                    id: doc_id.clone(),
                    text: turn_text_short(turn),
                    metadata: metadata.clone(),
                    created_at: created,
                });
                docs_rich.push(Document {
                    id: doc_id,
                    text: turn_text_rich(turn),
                    metadata,
                    created_at: created,
                });
            }
        }

        let idx_short = registry
            .open_or_create("short", &config.storage.storage_dir, &config.chunking)
            .context("open short")?;
        idx_short
            .index_documents(&docs_short, None)
            .context("index short")?;
        let idx_rich = registry
            .open_or_create("rich", &config.storage.storage_dir, &config.chunking)
            .context("open rich")?;
        idx_rich
            .index_documents(&docs_rich, None)
            .context("index rich")?;

        let query = sanitize_query(&q.question);
        let gold_sessions: HashSet<String> = gold_ids
            .iter()
            .filter_map(|gid| {
                q.haystack_session_ids
                    .iter()
                    .position(|s| s == gid)
                    .map(|i| format!("s{i}"))
            })
            .collect();

        let score = |hits: &[ChunkHit], condition: &str, pool_sessions: usize| -> QuestionResult {
            let mut seen_sessions: Vec<&str> = Vec::new();
            for h in hits {
                let s = session_of(&h.doc_id);
                if !seen_sessions.contains(&s) {
                    seen_sessions.push(s);
                }
            }
            let first_gold_rank = seen_sessions
                .iter()
                .position(|s| gold_sessions.contains(*s));
            let covered = seen_sessions
                .iter()
                .take(TOP_K)
                .filter(|s| gold_sessions.contains(**s))
                .count();
            QuestionResult {
                question_id: &q.question_id,
                question_type: &q.question_type,
                condition: Box::leak(condition.to_string().into_boxed_str()),
                first_gold_rank,
                evidence_covered: covered,
                evidence_total: gold_sessions.len(),
                pool_sessions,
                digests_written: None,
            }
        };

        let empty = HashSet::new();
        let n_sessions = q.haystack_sessions.len();

        if wanted("bm25-short") {
            let hits = idx_short.search_excluding(&query, RERANK_POOL, None, &empty)?;
            let r = score(
                &hits[..hits.len().min(RERANK_POOL)],
                "bm25-short",
                n_sessions,
            );
            writeln!(out, "{}", serde_json::to_string(&r)?)?;
        }
        if wanted("bm25-rich") {
            let hits = idx_rich.search_excluding(&query, RERANK_POOL, None, &empty)?;
            let r = score(&hits, "bm25-rich", n_sessions);
            writeln!(out, "{}", serde_json::to_string(&r)?)?;
            if let Some(w) = evidence_out.as_mut() {
                // Top-K capsule texts, first hit per capsule, fetch_doc text
                // (the same delivery shape as the vivling recall section).
                let mut seen: HashSet<&str> = HashSet::new();
                let mut evidence: Vec<serde_json::Value> = Vec::new();
                for h in hits.iter() {
                    if evidence.len() >= TOP_K {
                        break;
                    }
                    if !seen.insert(h.doc_id.as_str()) {
                        continue;
                    }
                    if let Ok(doc) = idx_rich.fetch_doc(&h.doc_id) {
                        let si: usize = session_of(&h.doc_id)[1..].parse().unwrap_or(usize::MAX);
                        let date = sess_infos
                            .get(si)
                            .and_then(|s| s.date)
                            .map(|d| d.date().to_string());
                        evidence.push(serde_json::json!({
                            "text": doc.text,
                            "session": si,
                            "date": date,
                        }));
                    }
                }
                let row = serde_json::json!({
                    "question_id": q.question_id,
                    "question_type": q.question_type,
                    "question": q.question,
                    "question_date": q.question_date,
                    "answer": q.answer,
                    "evidence": evidence,
                });
                writeln!(w, "{row}")?;
            }
        }
        if wanted("recency") {
            let mut hits = idx_rich.search_excluding(&query, RERANK_POOL, None, &empty)?;
            if let Some(qd) = question_date {
                for h in hits.iter_mut() {
                    let si: usize = session_of(&h.doc_id)[1..].parse().unwrap_or(0);
                    if let Some(d) = sess_infos.get(si).and_then(|s| s.date) {
                        let age_days = (qd - d).num_days().max(0) as f32;
                        let decay = 0.5f32.powf(age_days / RECENCY_HALF_LIFE_DAYS);
                        h.score *= 0.2 + 0.8 * decay; // prior, not a hard gate
                    }
                }
                hits.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            let r = score(&hits, "recency", n_sessions);
            writeln!(out, "{}", serde_json::to_string(&r)?)?;
        }
        if wanted("oracle-band") {
            let hits = idx_rich.search_excluding(&query, RERANK_POOL, None, &empty)?;
            let in_band_pool = sess_infos.iter().filter(|s| s.in_band).count();
            let filtered: Vec<ChunkHit> = hits
                .into_iter()
                .filter(|h| {
                    let si: usize = session_of(&h.doc_id)[1..].parse().unwrap_or(usize::MAX);
                    sess_infos.get(si).map(|s| s.in_band).unwrap_or(false)
                })
                .collect();
            let r = score(&filtered, "oracle-band", in_band_pool);
            writeln!(out, "{}", serde_json::to_string(&r)?)?;
        }

        // C-cons-*: consolidation pass between ingest and query (design
        // 2026-06-06 v3 §3). Each condition gets its OWN collection so the
        // rich index stays pristine for the other conditions.
        let mut run_cons = |name: &str, backend: &dyn DistillBackend| -> Result<()> {
            let idx = registry
                .open_or_create(name, &config.storage.storage_dir, &config.chunking)
                .with_context(|| format!("open {name}"))?;
            idx.index_documents(&docs_rich, None)
                .context("index cons")?;
            let opts = ConsolidateOptions {
                session_key: SessionKeySource::DocIdPrefix {
                    separator: "::".into(),
                },
                // bench: consolidate everything eligible, runtime caps don't apply
                max_candidates: usize::MAX,
                ..ConsolidateOptions::default()
            };
            let cands = consolidate::select_candidates(&idx, &opts)?;
            let mut digests_written = 0usize;
            for c in &cands {
                match consolidate::consolidate(&idx, c, backend, &opts) {
                    Ok(_) => digests_written += 1,
                    Err(e) => eprintln!("[convbench] {name} distill error ({}): {e}", c.key),
                }
            }

            let hits = idx.search_excluding(&query, RERANK_POOL, None, &empty)?;

            // Retrieval scoring with digest expansion: a digest hit covers the
            // sessions of its members (that is precisely its job).
            let mut seen_sessions: Vec<String> = Vec::new();
            let mut push_sess = |s: String| {
                if !seen_sessions.contains(&s) {
                    seen_sessions.push(s);
                }
            };
            for h in &hits {
                if consolidate::is_digest(&h.metadata) {
                    for m in consolidate::digest_member_ids(&h.metadata) {
                        push_sess(session_of(&m).to_string());
                    }
                } else {
                    push_sess(session_of(&h.doc_id).to_string());
                }
            }
            let first_gold_rank = seen_sessions
                .iter()
                .position(|s| gold_sessions.contains(s.as_str()));
            let covered = seen_sessions
                .iter()
                .take(TOP_K)
                .filter(|s| gold_sessions.contains(s.as_str()))
                .count();
            let r = QuestionResult {
                question_id: &q.question_id,
                question_type: &q.question_type,
                condition: Box::leak(name.to_string().into_boxed_str()),
                first_gold_rank,
                evidence_covered: covered,
                evidence_total: gold_sessions.len(),
                pool_sessions: n_sessions,
                digests_written: Some(digests_written),
            };
            writeln!(out, "{}", serde_json::to_string(&r)?)?;

            // Evidence for the reading gate, with the serving supersede policy
            // (mirror of the interleave round assembler): if a digest is in
            // the top-K picks, its members are covered and backfilled.
            if let Some(w) = evidence_out.as_mut() {
                let mut picked: Vec<&ChunkHit> = Vec::new();
                let mut seen_docs: HashSet<&str> = HashSet::new();
                for h in &hits {
                    if seen_docs.insert(h.doc_id.as_str()) {
                        picked.push(h);
                    }
                }
                let covered_ids: HashSet<String> = picked
                    .iter()
                    .take(TOP_K)
                    .filter(|h| consolidate::is_digest(&h.metadata))
                    .flat_map(|h| consolidate::digest_member_ids(&h.metadata))
                    .collect();
                let mut evidence: Vec<serde_json::Value> = Vec::new();
                for h in &picked {
                    if evidence.len() >= TOP_K {
                        break;
                    }
                    if covered_ids.contains(h.doc_id.as_str()) {
                        continue; // superseded by a digest in the same pick set
                    }
                    let Ok(doc) = idx.fetch_doc(&h.doc_id) else {
                        continue;
                    };
                    if consolidate::is_digest(&h.metadata) {
                        evidence.push(serde_json::json!({
                            "text": doc.text,
                            "session": serde_json::Value::Null,
                            "date": serde_json::Value::Null,
                            "kind": "digest",
                        }));
                    } else {
                        let si: usize = session_of(&h.doc_id)[1..].parse().unwrap_or(usize::MAX);
                        let date = sess_infos
                            .get(si)
                            .and_then(|s| s.date)
                            .map(|d| d.date().to_string());
                        evidence.push(serde_json::json!({
                            "text": doc.text,
                            "session": si,
                            "date": date,
                        }));
                    }
                }
                let row = serde_json::json!({
                    "question_id": q.question_id,
                    "question_type": q.question_type,
                    "question": q.question,
                    "question_date": q.question_date,
                    "answer": q.answer,
                    "condition": name,
                    "evidence": evidence,
                });
                writeln!(w, "{row}")?;
            }
            Ok(())
        };

        if wanted("cons-det") {
            run_cons("cons-det", &DeterministicConcatBackend)?;
        }
        if wanted("cons-llm") {
            let backend = GlmDistill::from_env()
                .context("cons-llm requires ZAI_API_KEY (and optional GLM_MODEL / ZAI_BASE_URL)")?;
            run_cons("cons-llm", &backend)?;
        }

        #[cfg(feature = "bench-fastembed")]
        if wanted("hybrid") {
            if let Some(model_name) = args.fastembed_model.as_ref() {
                let inner = crate::fastembed_adapter::FastembedEncoder::try_new(
                    crate::fastembed_adapter::parse_model(model_name)?,
                )
                .context("fastembed encoder")?;
                let (qp, dp) = crate::prefix_encoder::default_prefixes(model_name);
                let enc = crate::prefix_encoder::PrefixedEncoder::new(inner, qp, dp);
                let reordered = dense_rerank(&enc, args.hybrid_alpha, &idx_rich, &query, &empty)?;
                let r = score(&reordered, "hybrid", n_sessions);
                writeln!(out, "{}", serde_json::to_string(&r)?)?;
            }
        }

        #[cfg(feature = "bench-ollama")]
        if wanted("hybrid-ollama") {
            if let Some((url, model, dim, num_ctx, num_thread)) = args.ollama.as_ref() {
                let client =
                    msa_core::embeddings::OllamaClient::new(url.clone(), model.clone(), *dim)
                        .context("ollama client")?
                        .with_num_ctx(*num_ctx)
                        .with_num_thread(*num_thread);
                let (qp, dp) = crate::prefix_encoder::default_prefixes(model);
                let enc = crate::prefix_encoder::PrefixedEncoder::new(client, qp, dp);
                let reordered = dense_rerank(&enc, args.hybrid_alpha, &idx_rich, &query, &empty)?;
                let r = score(&reordered, "hybrid-ollama", n_sessions);
                writeln!(out, "{}", serde_json::to_string(&r)?)?;
            }
        }

        if (qi + 1) % 25 == 0 {
            eprintln!("[convbench] {}/{take} domande", qi + 1);
            out.flush().ok();
        }
    }
    out.flush().ok();
    if let Some(mut w) = evidence_out {
        use std::io::Write as _;
        w.flush().ok();
    }
    eprintln!(
        "[convbench] done: {take} domande -> {}",
        args.output_jsonl.display()
    );
    Ok(())
}

/// Dense re-rank of the BM25 pool (C4): embeds the query and the top
/// candidates only — the corpus is never embedded.
#[cfg(any(feature = "bench-fastembed", feature = "bench-ollama"))]
fn dense_rerank<E: msa_core::embeddings::DenseEncoder>(
    enc: &E,
    alpha: f32,
    idx: &msa_core::index::MsaIndex,
    query: &str,
    exclude: &HashSet<String>,
) -> Result<Vec<ChunkHit>> {
    use msa_core::embeddings::cosine_unit;
    use msa_core::embeddings::EmbedInputKind;
    const DENSE_TOP: usize = 20;

    let hits = idx.search_excluding(query, RERANK_POOL, None, exclude)?;
    let pool: Vec<&ChunkHit> = hits.iter().take(DENSE_TOP).collect();
    if pool.is_empty() {
        return Ok(Vec::new());
    }
    let qv = enc
        .encode_batch_kind(EmbedInputKind::Query, &[query])
        .context("embed query")?
        .remove(0);
    let bm_max = pool.iter().map(|h| h.score).fold(f32::MIN, f32::max);
    let bm_min = pool.iter().map(|h| h.score).fold(f32::MAX, f32::min);
    let span = (bm_max - bm_min).max(1e-6);
    let texts: Vec<String> = pool
        .iter()
        .map(|h| {
            idx.fetch_doc(&h.doc_id)
                .map(|d| d.text)
                .unwrap_or_else(|_| h.snippet.clone())
        })
        .collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let dvs = enc
        .encode_batch_kind(EmbedInputKind::Passage, &refs)
        .context("embed docs")?;
    let mut rescored: Vec<(f32, &ChunkHit)> = pool
        .iter()
        .zip(dvs.iter())
        .map(|(h, dv)| {
            let dense = cosine_unit(&qv, dv);
            let bm = (h.score - bm_min) / span;
            (alpha * bm + (1.0 - alpha) * dense, *h)
        })
        .collect();
    rescored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(rescored.into_iter().map(|(_, h)| h.clone()).collect())
}

/// GLM distillation backend for `cons-llm` (Anthropic-compat endpoint, same
/// as scripts/convbench_model_gate.py). Sync (ureq) — bench-only code path.
struct GlmDistill {
    base_url: String,
    api_key: String,
    model: String,
}

impl GlmDistill {
    fn from_env() -> Result<Self> {
        let api_key = std::env::var("ZAI_API_KEY").context("ZAI_API_KEY not set")?;
        Ok(Self {
            base_url: std::env::var("ZAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.z.ai/api/anthropic".to_string()),
            api_key,
            model: std::env::var("GLM_MODEL").unwrap_or_else(|_| "glm-4.6".to_string()),
        })
    }
}

impl DistillBackend for GlmDistill {
    fn distill(
        &self,
        _key: &str,
        member_texts: &[String],
        max_chars: usize,
    ) -> std::result::Result<String, DistillError> {
        let notes = member_texts
            .iter()
            .enumerate()
            .map(|(i, t)| format!("[note {}] {}", i + 1, t))
            .collect::<Vec<_>>()
            .join("\n\n");
        let prompt = format!(
            "Consolidate these related notes from different chat sessions into ONE \
             factual digest of at most {max_chars} characters. Merge duplicates; keep \
             every distinct fact; make counts, totals, dates and quantities explicit \
             (e.g. \"mentioned 3 times: ...\"); no commentary, no preamble.\n\n{notes}"
        );
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        });
        let mut last_err = String::new();
        for attempt in 0..3 {
            if attempt > 0 {
                std::thread::sleep(std::time::Duration::from_secs(2 << attempt));
            }
            let resp = ureq::post(&url)
                .set("Authorization", &format!("Bearer {}", self.api_key))
                .set("x-api-key", &self.api_key)
                .set("anthropic-version", "2023-06-01")
                .set("content-type", "application/json")
                .send_string(&body.to_string());
            match resp {
                Ok(r) => {
                    let v: serde_json::Value = r
                        .into_json()
                        .map_err(|e| DistillError::Backend(e.to_string()))?;
                    let text = v["content"][0]["text"]
                        .as_str()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if text.is_empty() {
                        last_err = format!("empty completion: {v}");
                        continue;
                    }
                    return Ok(text);
                }
                Err(e) => last_err = e.to_string(),
            }
        }
        Err(DistillError::Backend(last_err))
    }
}
