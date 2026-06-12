//! Standalone smoke test for the Candle ModernBERT encoder on a target
//! device. Single-purpose binary used to validate that A1 actually loads
//! and runs a Granite-R2-97m bundle on a foreign architecture (primarily
//! aarch64-linux-android via Termux).
//!
//! Usage:
//!   granite-smoke                            (reads MCP_VL_MSA_GRANITE_BUNDLE)
//!   granite-smoke --model-dir <path>
//!
//! Prints one JSON object to stdout with the smoke metrics. Exit codes:
//!   0  OK + ranking expected
//!   1  generic error (missing bundle, load failure, etc.)
//!   2  embedding dim mismatch
//!   3  invalid norm produced
//!   4  ranking regression vs spike snapshot
//!
//! Built only with `--features embeddings-candle`.

use std::process::ExitCode;
use std::time::Instant;

use msa_core::embeddings::{CandleModernBert, DenseEncoder, EmbedInputKind};

const QUERIES: &[&str] = &[
    "Come funziona il dense rerank nel motore MSA?",
    "How does the encoder pool token embeddings?",
];

const PASSAGES: &[&str] = &[
    "L'utente ha aperto la sessione codex-vl alle 14:30 per implementare il refactor di chatwidget e PingPong.",
    "mcp-msa-rs v0.4 implementa il pattern macro MSA del paper EverMind-AI con BM25 sparso e rerank opzionale.",
    "The encoder applies mean pooling across token embeddings with attention mask, followed by L2 normalization to produce a unit vector.",
    "The trading bot opens a long position on BTCUSDT when the 5-minute RSI crosses below 30 and volume spikes.",
];

// Expected top-1 passage index per query (matches spike 2026-05-17):
// Q0 -> P1 ("mcp-msa-rs ... rerank opzionale")
// Q1 -> P2 ("mean pooling across token embeddings ...")
const EXPECTED_TOP1: [usize; 2] = [1, 2];

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("granite-smoke: {e}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<ExitCode, String> {
    // Resolve bundle dir: explicit --model-dir arg wins over env var.
    let args: Vec<String> = std::env::args().collect();
    let mut model_dir: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                model_dir = args.get(i + 1).cloned();
                i += 2;
            }
            "-h" | "--help" => {
                println!(
                    "granite-smoke — load Granite-R2-97m bundle and exercise CandleModernBert\n\
                     \n\
                     Usage:\n  granite-smoke [--model-dir PATH]\n\
                     Env:   MCP_VL_MSA_GRANITE_BUNDLE (fallback if --model-dir absent)"
                );
                return Ok(ExitCode::SUCCESS);
            }
            other => return Err(format!("unknown arg {other:?}")),
        }
    }
    let model_dir = model_dir
        .or_else(|| std::env::var("MCP_VL_MSA_GRANITE_BUNDLE").ok())
        .ok_or_else(|| {
            "missing bundle: pass --model-dir PATH or set MCP_VL_MSA_GRANITE_BUNDLE".to_string()
        })?;

    let target = host_target();

    let t_load = Instant::now();
    let encoder = CandleModernBert::load_from_dir(
        std::path::Path::new(&model_dir),
        "ibm-granite/granite-embedding-97m-multilingual-r2",
        "smoke",
        "granite-r2-97m",
        384,
        Some(8192),
    )
    .map_err(|e| format!("load_from_dir: {e}"))?;
    let load_ms = t_load.elapsed().as_millis();

    let dim = encoder.dim();

    let all_texts: Vec<&str> = QUERIES.iter().chain(PASSAGES.iter()).copied().collect();
    let t_enc = Instant::now();
    let mut emb = Vec::with_capacity(all_texts.len());
    for (idx, text) in all_texts.iter().enumerate() {
        let kind = if idx < QUERIES.len() {
            EmbedInputKind::Query
        } else {
            EmbedInputKind::Passage
        };
        let v = encoder
            .encode_kind(kind, text)
            .map_err(|e| format!("encode[{idx}]: {e}"))?;
        if v.len() != dim {
            return Ok(emit_failure(
                &model_dir,
                &target,
                load_ms,
                dim,
                None,
                None,
                "dim_mismatch",
                ExitCode::from(2),
            ));
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return Ok(emit_failure(
                &model_dir,
                &target,
                load_ms,
                dim,
                None,
                None,
                "invalid_norm",
                ExitCode::from(3),
            ));
        }
        emb.push(v);
    }
    let encode_ms_total = t_enc.elapsed().as_millis();
    let n = emb.len() as u128;
    let ms_per_text = encode_ms_total.checked_div(n).unwrap_or(0);

    // Ranking check: each query's top-1 passage must match EXPECTED_TOP1.
    let nq = QUERIES.len();
    let mut top1: Vec<usize> = Vec::with_capacity(nq);
    let mut top1_scores: Vec<f32> = Vec::with_capacity(nq);
    for qi in 0..nq {
        let q = &emb[qi];
        let mut best = (0usize, f32::NEG_INFINITY);
        for (pi, p) in emb[nq..].iter().enumerate() {
            let dot: f32 = q.iter().zip(p.iter()).map(|(a, b)| a * b).sum();
            if dot > best.1 {
                best = (pi, dot);
            }
        }
        top1.push(best.0);
        top1_scores.push(best.1);
    }
    let ranking_ok = top1
        .iter()
        .zip(EXPECTED_TOP1.iter())
        .all(|(got, want)| got == want);

    let max_rss_kb = read_max_rss_kb();

    let status = if ranking_ok {
        "ok"
    } else {
        "ranking_regression"
    };
    let exit = if ranking_ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(4)
    };

    println!(
        "{{\"status\":\"{status}\",\"target\":\"{target}\",\"model_dir\":\"{md}\",\
         \"dim\":{dim},\"load_ms\":{load_ms},\"encode_ms_total\":{encode_ms_total},\
         \"ms_per_text\":{ms_per_text},\"texts\":{n},\"max_rss_kb\":{rss},\
         \"top1\":[{t0},{t1}],\"expected_top1\":[{e0},{e1}],\
         \"top1_scores\":[{s0:.4},{s1:.4}]}}",
        md = json_escape(&model_dir),
        dim = dim,
        load_ms = load_ms,
        encode_ms_total = encode_ms_total,
        ms_per_text = ms_per_text,
        n = n,
        rss = max_rss_kb
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".to_string()),
        t0 = top1[0],
        t1 = top1[1],
        e0 = EXPECTED_TOP1[0],
        e1 = EXPECTED_TOP1[1],
        s0 = top1_scores[0],
        s1 = top1_scores[1],
    );

    Ok(exit)
}

#[allow(clippy::too_many_arguments)]
fn emit_failure(
    model_dir: &str,
    target: &str,
    load_ms: u128,
    dim: usize,
    _encode_ms_total: Option<u128>,
    _ms_per_text: Option<u128>,
    reason: &str,
    exit: ExitCode,
) -> ExitCode {
    let rss = read_max_rss_kb()
        .map(|v| v.to_string())
        .unwrap_or_else(|| "null".to_string());
    println!(
        "{{\"status\":\"{reason}\",\"target\":\"{target}\",\"model_dir\":\"{md}\",\
         \"dim\":{dim},\"load_ms\":{load_ms},\"max_rss_kb\":{rss}}}",
        md = json_escape(model_dir),
    );
    exit
}

fn host_target() -> String {
    // Format consistent with rustc TARGET triple.
    format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
}

/// Read VmRSS from /proc/self/status; works on Linux and Android.
/// Returns None if not available (e.g. macOS, where /proc is absent).
fn read_max_rss_kb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb_str = rest.split_whitespace().next()?;
            return kb_str.parse::<u64>().ok();
        }
    }
    None
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}
