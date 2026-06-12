//! `msa-rag-bench` — entry point.
//!
//! See `crates/msa-bench/README.md` for usage. The binary is gated on
//! `--features bench` because it pulls the Candle in-process encoder.

mod convbench;
mod dataset;
mod gate;
mod metrics;
mod runner;

#[cfg(feature = "bench-fastembed")]
mod fastembed_adapter;

#[cfg(any(feature = "bench-ollama", feature = "bench-fastembed"))]
mod prefix_encoder;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "msa-rag-bench",
    about = "MSA-RAG-BENCHMARKS evaluator for mcp-vl-msa-rs: BM25 vs BM25+Granite dense rerank"
)]
struct Cli {
    /// Benchmark name. Currently supported: `hotpotqa`, `2wikimultihopqa`.
    #[arg(long, default_value = "hotpotqa")]
    dataset: String,

    /// Override datasets root (default: `crates/msa-bench/datasets`).
    #[arg(long)]
    datasets_root: Option<PathBuf>,

    /// Limit number of queries (head of the file). Useful for smoke runs.
    #[arg(long)]
    limit_queries: Option<usize>,

    /// Limit corpus size (head of the file). Missing relevant docs for
    /// sampled queries are appended automatically to keep metrics fair.
    #[arg(long)]
    limit_corpus: Option<usize>,

    /// Comma-separated alpha values to sweep. `1.0` = BM25 only.
    #[arg(long, default_value = "0.0,0.3,0.5,0.7,1.0")]
    alpha_sweep: String,

    /// Comma-separated `topN_rerank` values to sweep.
    #[arg(long, default_value = "20,50,100")]
    topn_rerank_sweep: String,

    /// `topk` for report metrics (NDCG@10, MRR@10, Recall@10).
    #[arg(long, default_value_t = 10)]
    top_k_report: usize,

    /// Word-level chunk size. `0` (default) = one chunk per passage (BEIR
    /// doc-level); a non-zero value (e.g. 64) exercises the real P=64 chunker.
    #[arg(long, default_value_t = 0)]
    chunk_size: usize,

    /// Candle ModernBERT bundle directory (Granite-R2-97m prefixed).
    /// Falls back to `MCP_VL_MSA_GRANITE_BUNDLE` env var. Only required when the
    /// alpha sweep includes a value < 1.0 (hybrid/dense); a pure BM25 gate
    /// (`--alpha-sweep 1.0`) runs without any model.
    #[arg(long)]
    bundle: Option<PathBuf>,

    /// Output JSON file. Default: `crates/msa-bench/results/<ts>-<dataset>.json`.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Run a gate instead of the retrieval sweep. Supported: `injection-ablation`
    /// (A5 Strato 3: full-text injection vs snippet-only answer presence at an
    /// equal char budget). Requires answers.jsonl in the dataset dir.
    #[arg(long)]
    gate: Option<String>,

    /// conv-bench: path to the LongMemEval-S json.
    #[arg(
        long,
        default_value = "crates/msa-bench/datasets/longmemeval/longmemeval_s.json"
    )]
    convbench_json: PathBuf,

    /// conv-bench: comma-separated conditions
    /// (bm25-short,bm25-rich,recency,oracle-band,hybrid). Default: bm25 four.
    #[arg(long)]
    convbench_conditions: Option<String>,

    /// conv-bench: dump per-question top-K rich evidence (model-in-loop).
    #[arg(long)]
    convbench_emit_evidence: Option<PathBuf>,

    /// injection-ablation: character budget B both conditions fill. Default
    /// matches the tool's max_total_chars so injection can reach saturation
    /// (a too-tight budget starves full-text injection and favors snippet
    /// breadth — informative, but not the lever's regime).
    #[arg(long, default_value_t = 16000)]
    budget_chars: usize,

    /// injection-ablation: per-document injection cap for condition C.
    #[arg(long, default_value_t = 2000)]
    gate_max_chars_per_doc: usize,

    /// injection-ablation: candidate chunk hits retrieved per query.
    #[arg(long, default_value_t = 10)]
    gate_top_k: usize,

    /// injection-ablation: pass if injection beats snippet by >= this many points.
    #[arg(long, default_value_t = 5.0)]
    threshold_points: f64,

    /// injection-ablation: also dump per-query evidence (question, gold answer,
    /// ctx_c full-text, ctx_d snippet) as JSONL for the model-in-loop gate
    /// (M3 Strato 2) — the downstream harness scores answer F1 on the same
    /// evidence this gate scores for presence.
    #[arg(long)]
    gate_emit_evidence: Option<PathBuf>,

    /// Encoder backend for dense scoring: `candle-modernbert` (default),
    /// `fastembed` (requires feature `bench-fastembed`), or
    /// `ollama` (requires feature `bench-ollama`).
    #[arg(long, default_value = "candle-modernbert")]
    encoder_backend: String,

    /// Fastembed model identifier (only used with `--encoder-backend fastembed`).
    /// Supported: `bge-m3` (default), `bge-small-en-v1.5`, `all-minilm-l6-v2`,
    /// `nomic-embed-text-v1.5`, `gte-base-en-v1.5`, `multilingual-e5-base`, etc.
    #[arg(long, default_value = "bge-m3")]
    fastembed_model: String,

    /// Ollama embedding model name (only used with `--encoder-backend ollama`).
    #[arg(long, default_value = "qwen3-embedding:0.6b")]
    ollama_model: String,

    /// Ollama server base URL (only used with `--encoder-backend ollama`).
    #[arg(long, default_value = "http://localhost:11434")]
    ollama_url: String,

    /// Ollama embedding vector dimension (only used with `--encoder-backend ollama`).
    /// Required because the Ollama `/api/embeddings` response does not declare the
    /// dimension upfront; `OllamaClient::new` validates every response against it.
    #[arg(long, default_value_t = 1024)]
    ollama_dim: usize,

    /// Per-request context cap for the Ollama encoder (`options.num_ctx`).
    /// Needed because the server forces a large default context that OOM-kills
    /// local embedding models. `0` leaves the server default.
    #[arg(long, default_value_t = 2048)]
    ollama_num_ctx: usize,

    /// CPU threads for the Ollama encoder (`options.num_thread`). The server
    /// defaults to 1 thread for embeddings (~10x slower). `0` = server default.
    #[arg(long, default_value_t = 4)]
    ollama_num_thread: usize,

    /// Override the query-side prompt prefix for the Ollama encoder. When
    /// omitted, the model's known convention is used (e.g. nomic
    /// `search_query: `). Pass an empty string to disable; `\n` → newline.
    #[arg(long)]
    ollama_query_prefix: Option<String>,

    /// Override the passage-side prompt prefix for the Ollama encoder. When
    /// omitted, the model's known convention is used. Empty string disables.
    #[arg(long)]
    ollama_doc_prefix: Option<String>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "msa_bench=info,msa_core=warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    // Conversational bench (LongMemEval) has per-question haystacks and its
    // own loader: dispatch before the BEIR-style dataset load.
    if cli.gate.as_deref() == Some("conv-bench") {
        let conditions = cli
            .convbench_conditions
            .as_deref()
            .map(|s| s.split(',').map(|x| x.trim().to_string()).collect());
        let out = cli.output.clone().unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("results")
                .join("convbench_results.jsonl")
        });
        convbench::run(&convbench::ConvBenchArgs {
            dataset_json: &cli.convbench_json,
            output_jsonl: &out,
            limit_questions: cli.limit_queries,
            conditions,
            hybrid_alpha: 0.3,
            emit_evidence: cli.convbench_emit_evidence.clone(),
            #[cfg(feature = "bench-fastembed")]
            fastembed_model: Some(cli.fastembed_model.clone()),
            #[cfg(feature = "bench-ollama")]
            ollama: Some((
                cli.ollama_url.clone(),
                cli.ollama_model.clone(),
                cli.ollama_dim,
                cli.ollama_num_ctx,
                cli.ollama_num_thread,
            )),
        })?;
        return Ok(());
    }

    // Bundle is optional: only needed for hybrid/dense (alpha < 1.0). A pure
    // BM25 gate runs with no model at all.
    let bundle = cli.bundle.clone().or_else(|| {
        std::env::var("MCP_VL_MSA_GRANITE_BUNDLE")
            .ok()
            .map(PathBuf::from)
    });

    let datasets_root = cli
        .datasets_root
        .clone()
        .unwrap_or_else(dataset::default_datasets_root);

    let alphas: Vec<f32> = parse_csv_f32(&cli.alpha_sweep).context("--alpha-sweep")?;
    let topns: Vec<usize> =
        parse_csv_usize(&cli.topn_rerank_sweep).context("--topn-rerank-sweep")?;

    tracing::info!(
        dataset = %cli.dataset,
        bundle = bundle.as_deref().map(|p| p.display().to_string()).unwrap_or_else(|| "none(bm25)".into()),
        datasets_root = %datasets_root.display(),
        chunk_size = cli.chunk_size,
        encoder_backend = %cli.encoder_backend,
        ?alphas,
        ?topns,
        "msa-rag-bench start"
    );

    let ds = dataset::load(
        &datasets_root,
        &cli.dataset,
        cli.limit_queries,
        cli.limit_corpus,
    )?;
    tracing::info!(
        passages = ds.passages.len(),
        queries = ds.queries.len(),
        "dataset loaded"
    );

    // Gate mode: run a single gate and exit before the retrieval sweep.
    if let Some(gate_name) = cli.gate.as_deref() {
        match gate_name {
            "injection-ablation" => {
                let chunk_size = if cli.chunk_size == 0 {
                    64
                } else {
                    cli.chunk_size
                };
                let rep = gate::run(
                    &ds,
                    gate::GateConfig {
                        top_k: cli.gate_top_k,
                        budget_chars: cli.budget_chars,
                        max_chars_per_doc: cli.gate_max_chars_per_doc,
                        chunk_size,
                        threshold_points: cli.threshold_points,
                    },
                    cli.gate_emit_evidence.as_deref(),
                )?;
                let out_path = cli.output.clone().unwrap_or_else(|| {
                    let ts = chrono::Utc::now().format("%Y-%m-%dT%H%M%SZ");
                    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("results")
                        .join(format!("{ts}-{}-injection-ablation.json", ds.name))
                });
                if let Some(parent) = out_path.parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                std::fs::write(&out_path, serde_json::to_string_pretty(&rep)?)
                    .context("write gate output")?;
                eprintln!();
                eprintln!("=== injection-ablation gate: {} ===", rep.dataset);
                eprintln!(
                    "extractive={} (excluded: no_answer={}, yes_no={}, non_extractive={})",
                    rep.num_extractive,
                    rep.excluded_no_answer,
                    rep.excluded_yes_no,
                    rep.excluded_non_extractive
                );
                eprintln!(
                    "budget={}c per_doc_cap={}c top_k={} chunk_size={}",
                    rep.budget_chars, rep.max_chars_per_doc, rep.top_k, rep.chunk_size
                );
                eprintln!(
                    "answer presence:  injection(C)={:.3}  snippet(D)={:.3}  oracle={:.3}",
                    rep.rate_injection, rep.rate_snippet, rep.rate_oracle
                );
                eprintln!(
                    "oracle_capture={:.3}  |  delta vs snippet = {:+.1} pts",
                    rep.oracle_capture, rep.delta_points
                );
                eprintln!(
                    "verdict (oracle_capture>=0.95 AND injection>snippet) -> {}",
                    if rep.passed { "PASS" } else { "FAIL" }
                );
                eprintln!("report: {}", out_path.display());
                if !rep.passed {
                    std::process::exit(1);
                }
                return Ok(());
            }
            other => anyhow::bail!("unknown --gate '{other}' (supported: injection-ablation)"),
        }
    }

    let backend = match cli.encoder_backend.as_str() {
        "candle-modernbert" => runner::EncoderBackend::CandleModernBert,
        #[cfg(feature = "bench-fastembed")]
        "fastembed" => runner::EncoderBackend::Fastembed {
            model: cli.fastembed_model.clone(),
        },
        #[cfg(not(feature = "bench-fastembed"))]
        "fastembed" => {
            anyhow::bail!(
                "--encoder-backend fastembed requires feature bench-fastembed; \
                 rebuild with --features bench,bench-fastembed"
            );
        }
        #[cfg(feature = "bench-ollama")]
        "ollama" => runner::EncoderBackend::Ollama {
            model: cli.ollama_model.clone(),
            base_url: cli.ollama_url.clone(),
            dim: cli.ollama_dim,
            num_ctx: cli.ollama_num_ctx,
            num_thread: cli.ollama_num_thread,
            query_prefix: cli.ollama_query_prefix.clone(),
            doc_prefix: cli.ollama_doc_prefix.clone(),
        },
        #[cfg(not(feature = "bench-ollama"))]
        "ollama" => {
            anyhow::bail!(
                "--encoder-backend ollama requires feature bench-ollama; \
                 rebuild with --features bench,bench-ollama"
            );
        }
        other => {
            anyhow::bail!(
                "unknown --encoder-backend '{other}' \
                 (supported: candle-modernbert, fastembed, ollama)"
            );
        }
    };

    let report = runner::run(
        &ds,
        bundle.as_deref(),
        runner::SweepConfig {
            alphas: &alphas,
            topn_reranks: &topns,
            top_k_report: cli.top_k_report,
            chunk_size: cli.chunk_size,
        },
        backend,
    )?;

    let out_path = cli.output.clone().unwrap_or_else(|| {
        let ts = chrono::Utc::now().format("%Y-%m-%dT%H%M%SZ");
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("results")
            .join(format!("{ts}-{}.json", ds.name))
    });
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&out_path, &json).context("write output")?;
    tracing::info!(output = %out_path.display(), "report written");

    // Also dump a compact summary to stderr.
    eprintln!();
    eprintln!("=== {} ===", report.dataset);
    eprintln!(
        "corpus={} queries={} index_build_ms={}",
        report.corpus_size, report.num_queries, report.index_build_ms
    );
    eprintln!(
        "{:>5} {:>5} {:>8} {:>8} {:>8} {:>8} {:>6} {:>6}",
        "alpha", "topN", "NDCG@10", "MRR@10", "R@10", "R@20", "p50", "p95"
    );
    for c in &report.cells {
        eprintln!(
            "{:>5.2} {:>5} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>6} {:>6}",
            c.alpha,
            c.topn_rerank,
            c.ndcg_at_10,
            c.mrr_at_10,
            c.recall_at_10,
            c.recall_at_20,
            c.latency_p50_ms,
            c.latency_p95_ms
        );
    }
    Ok(())
}

fn parse_csv_f32(s: &str) -> Result<Vec<f32>> {
    s.split(',')
        .map(|p| p.trim().parse::<f32>().map_err(Into::into))
        .collect()
}
fn parse_csv_usize(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|p| p.trim().parse::<usize>().map_err(Into::into))
        .collect()
}
