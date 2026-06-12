//! Loaders for the MSA-RAG-BENCHMARKS bundle. Supports `hotpotqa` and
//! `2wikimultihopqa` initially. Other tasks share the same on-disk
//! layout so adding them is a one-line dispatch change.
//!
//! Expected layout on disk:
//!
//! ```text
//! datasets/<bench>/
//!   corpus.jsonl        # one passage per line: {"_id": str, "text": str, ...}
//!   queries.jsonl       # one query per line:   {"_id": str, "text": str}
//!   qrels/test.tsv      # TREC qrels:           query_id 0 doc_id rel
//! ```
//!
//! The MSA paper distributes its benchmark dump in this BEIR-compatible
//! shape; the download script normalizes whatever HuggingFace ships into
//! it. If a future MSA dataset uses a different layout (e.g. parquet
//! shards) we add a branch here without breaking the runner.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

/// One row of the corpus collection.
#[derive(Debug, Clone)]
pub struct Passage {
    pub id: String,
    pub text: String,
}

/// One query with its set of relevant doc ids and, when available, the gold
/// answer string (used by the injection-ablation gate; absent in pure BEIR
/// retrieval datasets).
#[derive(Debug, Clone)]
pub struct EvalQuery {
    pub id: String,
    pub text: String,
    pub relevant: Vec<String>,
    pub answer: Option<String>,
}

/// Complete dataset slice loaded into memory.
pub struct LoadedDataset {
    pub name: String,
    pub passages: Vec<Passage>,
    pub queries: Vec<EvalQuery>,
}

#[derive(Debug, Deserialize)]
struct CorpusRow {
    #[serde(rename = "_id")]
    id: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct QueryRow {
    #[serde(rename = "_id")]
    id: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct AnswerRow {
    #[serde(rename = "_id")]
    id: String,
    answer: String,
}

/// Load `<root>/<bench>/{corpus,queries,qrels/test}` honoring the per-side
/// limits (sampled deterministically from the head of the files).
pub fn load(
    root: &Path,
    bench: &str,
    limit_queries: Option<usize>,
    limit_corpus: Option<usize>,
) -> Result<LoadedDataset> {
    let dir = root.join(bench);
    if !dir.is_dir() {
        anyhow::bail!(
            "dataset dir {} not found — run scripts/download-bench-datasets.sh first",
            dir.display()
        );
    }

    let passages = load_corpus(&dir.join("corpus.jsonl"), limit_corpus)
        .with_context(|| format!("load corpus for {bench}"))?;

    let mut queries: HashMap<String, EvalQuery> = HashMap::new();
    for row in read_jsonl::<QueryRow>(&dir.join("queries.jsonl"))? {
        queries.insert(
            row.id.clone(),
            EvalQuery {
                id: row.id,
                text: row.text,
                relevant: Vec::new(),
                answer: None,
            },
        );
    }

    // Optional gold answers (answers.jsonl: {"_id", "answer"}). Present for the
    // multi-hop QA datasets, absent for pure retrieval dumps — load if there.
    let answers_path = dir.join("answers.jsonl");
    if answers_path.is_file() {
        for row in read_jsonl::<AnswerRow>(&answers_path)? {
            if let Some(q) = queries.get_mut(&row.id) {
                q.answer = Some(row.answer);
            }
        }
    }

    let qrels_path = dir.join("qrels").join("test.tsv");
    let qrels_file =
        File::open(&qrels_path).with_context(|| format!("open qrels {}", qrels_path.display()))?;
    for (lineno, line) in BufReader::new(qrels_file).lines().enumerate() {
        let line = line?;
        if lineno == 0 && line.starts_with("query-id") {
            // BEIR ships a TSV header; skip it.
            continue;
        }
        let mut parts = line.split('\t');
        let qid = parts.next().unwrap_or("").trim();
        let _zero = parts.next();
        let did = parts.next().unwrap_or("").trim();
        let rel = parts.next().unwrap_or("0").trim();
        if qid.is_empty() || did.is_empty() {
            continue;
        }
        if rel.parse::<i32>().unwrap_or(0) > 0 {
            if let Some(q) = queries.get_mut(qid) {
                q.relevant.push(did.to_string());
            }
        }
    }

    // Keep only queries with at least one relevant doc; the others are
    // unusable for retrieval metrics.
    let mut all_queries: Vec<EvalQuery> = queries
        .into_values()
        .filter(|q| !q.relevant.is_empty())
        .collect();
    all_queries.sort_by(|a, b| a.id.cmp(&b.id));
    if let Some(n) = limit_queries {
        all_queries.truncate(n);
    }

    // If `limit_corpus` clipped, make sure every relevant doc for the
    // sampled queries is also in the corpus (otherwise metrics collapse
    // to zero artificially). We append missing relevant docs from the
    // full corpus on a second pass.
    let mut passages = passages;
    if limit_corpus.is_some() {
        let known: std::collections::HashSet<String> =
            passages.iter().map(|p| p.id.clone()).collect();
        let mut need: std::collections::HashSet<String> = std::collections::HashSet::new();
        for q in &all_queries {
            for d in &q.relevant {
                if !known.contains(d) {
                    need.insert(d.clone());
                }
            }
        }
        if !need.is_empty() {
            for row in read_jsonl::<CorpusRow>(&dir.join("corpus.jsonl"))? {
                if need.contains(&row.id) {
                    passages.push(Passage {
                        id: row.id.clone(),
                        text: row.text,
                    });
                    need.remove(&row.id);
                    if need.is_empty() {
                        break;
                    }
                }
            }
        }
    }

    Ok(LoadedDataset {
        name: bench.to_string(),
        passages,
        queries: all_queries,
    })
}

fn load_corpus(path: &Path, limit: Option<usize>) -> Result<Vec<Passage>> {
    let file = File::open(path).with_context(|| format!("open corpus {}", path.display()))?;
    let mut out = Vec::with_capacity(limit.unwrap_or(10_000));
    for (i, line) in BufReader::new(file).lines().enumerate() {
        if let Some(n) = limit {
            if out.len() >= n {
                break;
            }
        }
        let line = line.with_context(|| format!("read corpus line {i}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: CorpusRow = serde_json::from_str(&line)
            .with_context(|| format!("parse corpus line {i}: {line}"))?;
        out.push(Passage {
            id: row.id,
            text: row.text,
        });
    }
    Ok(out)
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| format!("read line {i} of {}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(
            serde_json::from_str::<T>(&line)
                .with_context(|| format!("parse line {i} of {}", path.display()))?,
        );
    }
    Ok(out)
}

/// Convenience: resolve the canonical datasets root next to this crate.
pub fn default_datasets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("datasets")
}
