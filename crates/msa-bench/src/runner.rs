//! Eval loop: builds a Tantivy index over the loaded passages, sweeps
//! `alpha` and `topn_rerank`, computes the standard retrieval metrics
//! per cell, and reports per-query latency percentiles.
//!
//! The runner re-uses `msa-core::MsaIndex` and `msa-core::DenseEncoder`
//! so it exercises the exact code path the MCP server uses; if a
//! benchmark passes here it passes in production.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use chrono::Utc;
use msa_core::config::{MsaConfig, StorageConfig};
#[cfg(feature = "bench-ollama")]
use msa_core::embeddings::OllamaClient;
use msa_core::embeddings::{CandleModernBert, DenseEncoder, EmbedInputKind, SharedEncoder};
use msa_core::index::{CollectionRegistry, MsaIndex};
use msa_core::schema::{ChunkConfig, ChunkHit, Document, SearchFilter};
use serde::Serialize;
use tempfile::TempDir;

use crate::dataset::{EvalQuery, LoadedDataset, Passage};
use crate::metrics::{mrr_at_k, ndcg_at_k, percentile_ms, recall_at_k};

/// Encoder backend selection for dense scoring.
pub enum EncoderBackend {
    /// In-process Candle ModernBERT (production default, needs Granite bundle).
    CandleModernBert,
    /// Fastembed ONNX runtime (feature `bench-fastembed`).
    #[cfg(feature = "bench-fastembed")]
    Fastembed { model: String },
    /// Ollama HTTP embedding server (feature `bench-ollama`).
    #[cfg(feature = "bench-ollama")]
    Ollama {
        model: String,
        base_url: String,
        dim: usize,
        num_ctx: usize,
        num_thread: usize,
        query_prefix: Option<String>,
        doc_prefix: Option<String>,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct SweepConfig<'a> {
    pub alphas: &'a [f32],
    pub topn_reranks: &'a [usize],
    pub top_k_report: usize,
    /// Word-level chunk size. `0` = one chunk per passage (doc-level, BEIR
    /// default); non-zero exercises the real P=64 chunker.
    pub chunk_size: usize,
}

#[derive(Debug, Serialize, Clone)]
pub struct CellResult {
    pub alpha: f32,
    pub topn_rerank: usize,
    pub ndcg_at_10: f64,
    pub mrr_at_10: f64,
    pub recall_at_10: f64,
    pub recall_at_20: f64,
    pub latency_p50_ms: u128,
    pub latency_p95_ms: u128,
}

#[derive(Debug, Serialize)]
pub struct RunReport {
    pub run_id: String,
    pub dataset: String,
    pub corpus_size: usize,
    pub num_queries: usize,
    pub model_id: String,
    pub model_version: String,
    pub profile: String,
    pub index_build_ms: u128,
    pub encode_passages_ms: u128,
    pub cells: Vec<CellResult>,
}

pub fn run(
    dataset: &LoadedDataset,
    bundle_dir: Option<&std::path::Path>,
    sweep: SweepConfig,
    backend: EncoderBackend,
) -> Result<RunReport> {
    let storage = TempDir::new().context("create temp storage for tantivy")?;
    let config = MsaConfig {
        storage: StorageConfig {
            storage_dir: storage.path().to_path_buf(),
        },
        chunking: ChunkConfig {
            // `chunk_size = 0` (default) means "one chunk per passage" — BEIR
            // passages are short paragraphs and the multi-hop datasets expect
            // doc-level retrieval. A non-zero value exercises the real P=64
            // word chunker so the benchmark can measure chunked retrieval too.
            chunk_size: if sweep.chunk_size == 0 {
                100_000
            } else {
                sweep.chunk_size
            },
            overlap: 0,
        },
        ..Default::default()
    };

    // The dense encoder is only needed when some alpha < 1.0 (hybrid/dense).
    // A pure BM25 gate (alpha = 1.0 only) runs without the Granite bundle, so
    // it works on any machine with no model download.
    let needs_dense = sweep.alphas.iter().any(|&a| a < 1.0);
    let encoder: Option<SharedEncoder> = if needs_dense {
        match &backend {
            EncoderBackend::CandleModernBert => {
                let bundle = bundle_dir.ok_or_else(|| {
                    anyhow::anyhow!(
                        "alpha < 1.0 requested but no Granite bundle provided; pass \
                         --bundle / MCP_VL_MSA_GRANITE_BUNDLE, or sweep only alpha=1.0 (BM25)"
                    )
                })?;
                Some(Arc::new(
                    CandleModernBert::load_from_dir(
                        bundle,
                        "ibm-granite/granite-embedding-97m-multilingual-r2",
                        "msa-bench",
                        "granite-r2-97m",
                        384,
                        Some(8192),
                    )
                    .context("load Granite bundle")?,
                ))
            }
            #[cfg(feature = "bench-fastembed")]
            EncoderBackend::Fastembed { model } => {
                let embedding_model = crate::fastembed_adapter::parse_model(model)?;
                Some(Arc::new(
                    crate::fastembed_adapter::FastembedEncoder::try_new(embedding_model)
                        .context("init fastembed encoder")?,
                ))
            }
            #[cfg(feature = "bench-ollama")]
            EncoderBackend::Ollama {
                model,
                base_url,
                dim,
                num_ctx,
                num_thread,
                query_prefix,
                doc_prefix,
            } => {
                let (def_q, def_d) = crate::prefix_encoder::default_prefixes(model);
                let qp = query_prefix
                    .as_deref()
                    .map(crate::prefix_encoder::unescape_cli)
                    .unwrap_or(def_q);
                let dp = doc_prefix
                    .as_deref()
                    .map(crate::prefix_encoder::unescape_cli)
                    .unwrap_or(def_d);
                tracing::info!(
                    model = %model,
                    base_url = %base_url,
                    dim,
                    query_prefix = %qp,
                    doc_prefix = %dp,
                    "ollama: connecting to embedding server"
                );
                let client = OllamaClient::new(base_url, model, *dim)
                    .context("init ollama client")?
                    .with_num_ctx(*num_ctx)
                    .with_num_thread(*num_thread);
                Some(Arc::new(crate::prefix_encoder::PrefixedEncoder::new(
                    client, qp, dp,
                )))
            }
        }
    } else {
        None
    };

    let registry = CollectionRegistry::new();
    let idx_arc = registry
        .open_or_create(&dataset.name, &config.storage.storage_dir, &config.chunking)
        .context("open tantivy collection")?;
    let idx: &MsaIndex = idx_arc.as_ref();

    // ----- Index build (passages -> tantivy + optional cached embeddings) ---
    // Single batched commit via index_documents (A2 path) instead of a commit
    // per passage: far fewer segments, much faster build on weak hardware.
    let t_build = Instant::now();
    let encoder_for_index: Option<&dyn DenseEncoder> = encoder.as_deref().map(|e| e as _);
    let docs: Vec<Document> = dataset.passages.iter().map(doc_from_passage).collect();
    idx.index_documents(&docs, encoder_for_index)
        .context("bulk index passages")?;
    let index_build_ms = t_build.elapsed().as_millis();
    let encode_passages_ms = index_build_ms; // chunk-level encode happens inline

    // ----- Sweep -----
    let mut cells = Vec::with_capacity(sweep.alphas.len() * sweep.topn_reranks.len());
    for &topn in sweep.topn_reranks {
        // Query embeddings cache keyed on query.id — same query is
        // re-issued across alpha values within the inner loop.
        let mut q_emb_cache: HashMap<String, Vec<f32>> = HashMap::new();
        for &alpha in sweep.alphas {
            let cell = eval_cell(
                idx,
                encoder.as_deref(),
                &dataset.queries,
                alpha,
                topn,
                sweep.top_k_report,
                &mut q_emb_cache,
            )?;
            cells.push(cell);
        }
    }

    let report = RunReport {
        run_id: format!(
            "{}-{}",
            Utc::now().format("%Y-%m-%dT%H:%M:%SZ"),
            dataset.name
        ),
        dataset: dataset.name.clone(),
        corpus_size: dataset.passages.len(),
        num_queries: dataset.queries.len(),
        model_id: encoder
            .as_ref()
            .map(|e| e.model_id().to_string())
            .unwrap_or_else(|| "none(bm25-only)".to_string()),
        model_version: encoder
            .as_ref()
            .map(|e| e.model_version().to_string())
            .unwrap_or_default(),
        profile: encoder
            .as_ref()
            .map(|e| e.profile().to_string())
            .unwrap_or_else(|| "bm25".to_string()),
        index_build_ms,
        encode_passages_ms,
        cells,
    };
    drop(storage); // explicit; tempdir auto-cleans on drop anyway
    Ok(report)
}

fn doc_from_passage(p: &Passage) -> Document {
    Document {
        id: p.id.clone(),
        text: p.text.clone(),
        metadata: Default::default(),
        created_at: Utc::now(),
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_cell(
    idx: &msa_core::index::MsaIndex,
    encoder: Option<&dyn DenseEncoder>,
    queries: &[EvalQuery],
    alpha: f32,
    topn_rerank: usize,
    top_k_report: usize,
    q_emb_cache: &mut HashMap<String, Vec<f32>>,
) -> Result<CellResult> {
    if alpha < 1.0 && encoder.is_none() {
        anyhow::bail!("alpha {alpha} < 1.0 requires a dense encoder, but none was loaded");
    }
    let mut ndcg_sum = 0.0f64;
    let mut mrr_sum = 0.0f64;
    let mut r10_sum = 0.0f64;
    let mut r20_sum = 0.0f64;
    let mut latencies: Vec<u128> = Vec::with_capacity(queries.len());

    let empty_excludes: HashSet<String> = HashSet::new();
    let filter: Option<SearchFilter> = None;

    for q in queries {
        let t = Instant::now();

        // Tantivy's default QueryParser is boolean and chokes on `:` `.` `,`
        // `?` etc. Strip them: BM25 already does its own tokenization, so
        // we only need to feed alphanumeric tokens separated by whitespace.
        let bm25_query: String = q
            .text
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c.is_whitespace() {
                    c
                } else {
                    ' '
                }
            })
            .collect();
        let bm25_query = bm25_query.split_whitespace().collect::<Vec<_>>().join(" ");

        // Query embedding (cached across alpha sweep for this topn).
        let q_emb = if (alpha - 1.0).abs() < 1e-6 {
            // Pure BM25 path doesn't need the encoder.
            None
        } else {
            if !q_emb_cache.contains_key(&q.id) {
                // Dense encoder uses the original text (no sanitization).
                let enc = encoder.expect("encoder present when alpha < 1.0");
                let v = enc.encode_kind(EmbedInputKind::Query, &q.text)?;
                q_emb_cache.insert(q.id.clone(), v);
            }
            q_emb_cache.get(&q.id).cloned()
        };

        let hits: Vec<ChunkHit> = if let Some(q_emb_v) = q_emb {
            idx.search_hybrid(
                &bm25_query,
                topn_rerank,
                filter.as_ref(),
                &empty_excludes,
                alpha,
                Some(&q_emb_v),
            )?
        } else {
            // alpha == 1.0 -> BM25-only, ignore topn_rerank inflation, take top
            // top_k_report.
            idx.search_excluding(&bm25_query, topn_rerank, filter.as_ref(), &empty_excludes)?
        };

        latencies.push(t.elapsed().as_millis());

        // Project chunk hits -> doc ids, preserving rank order, dedup.
        let mut seen: HashSet<String> = HashSet::new();
        let mut ranked_docs: Vec<String> = Vec::with_capacity(top_k_report);
        for h in hits {
            if seen.insert(h.doc_id.clone()) {
                ranked_docs.push(h.doc_id);
                if ranked_docs.len() >= top_k_report {
                    break;
                }
            }
        }

        let relevant: HashSet<String> = q.relevant.iter().cloned().collect();
        ndcg_sum += ndcg_at_k(&ranked_docs, &relevant, 10);
        mrr_sum += mrr_at_k(&ranked_docs, &relevant, 10);
        r10_sum += recall_at_k(&ranked_docs, &relevant, 10);
        r20_sum += recall_at_k(&ranked_docs, &relevant, 20);
    }

    let n = queries.len() as f64;
    Ok(CellResult {
        alpha,
        topn_rerank,
        ndcg_at_10: ndcg_sum / n,
        mrr_at_10: mrr_sum / n,
        recall_at_10: r10_sum / n,
        recall_at_20: r20_sum / n,
        latency_p50_ms: percentile_ms(&latencies, 50),
        latency_p95_ms: percentile_ms(&latencies, 95),
    })
}
