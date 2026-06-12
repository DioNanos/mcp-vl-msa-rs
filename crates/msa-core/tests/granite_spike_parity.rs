//! Anti-regression parity test for the Candle ModernBERT encoder against
//! the Granite-R2-97m spike snapshot captured on 2026-05-17 on x86_64 CPU.
//!
//! The test is `#[ignore]` by default because it needs a local bundle
//! prepared via `scripts/prepare-granite-r2-97m.sh`. Run it with:
//!
//! ```sh
//! export MCP_VL_MSA_GRANITE_BUNDLE=/path/to/bundle
//! cargo test -p msa-core --features embeddings --test granite_spike_parity -- --ignored
//! ```
//!
//! Snapshot source: cosine similarity matrix produced by
//! `~/Dev/sandbox/candle-embedding-test/target/release/spike_granite`
//! against the same 6 fixture passages and 2 queries listed below.
//! Tolerance is ±0.005 (CPU forward pass is deterministic; bigger drifts
//! signal an architectural change worth investigating).

#![cfg(feature = "embeddings-candle")]

use std::path::PathBuf;

use msa_core::embeddings::{CandleModernBert, DenseEncoder, EmbedInputKind};

const QUERIES: &[&str] = &[
    "Come funziona il dense rerank nel motore MSA?",
    "How does the encoder pool token embeddings?",
];

const FIXTURES: &[&str] = &[
    "L'utente ha aperto la sessione codex-vl alle 14:30 per implementare il refactor di chatwidget e PingPong.",
    "mcp-msa-rs v0.4 implementa il pattern macro MSA del paper EverMind-AI con BM25 sparso e rerank opzionale.",
    "pub fn search(&self, query: &str, top_k: usize) -> Vec<ChunkHit> { let candidates = self.bm25.query(query); candidates.iter().take(top_k).collect() }",
    "The encoder applies mean pooling across token embeddings with attention mask, followed by L2 normalization to produce a unit vector.",
    "Oggi sono andato al mercato a comprare pomodori e basilico per la cena di stasera con gli amici.",
    "The trading bot opens a long position on BTCUSDT when the 5-minute RSI crosses below 30 and volume spikes.",
];

// Snapshot cosine values from the 2026-05-17 spike.
// Rows: queries. Columns: fixtures (F0..F5).
const SNAPSHOT: [[f32; 6]; 2] = [
    [0.747, 0.803, 0.779, 0.766, 0.782, 0.788],
    [0.813, 0.788, 0.794, 0.893, 0.764, 0.791],
];

const TOLERANCE: f32 = 0.005;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dim mismatch");
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[test]
#[ignore = "needs local granite bundle; run scripts/prepare-granite-r2-97m.sh and set MCP_VL_MSA_GRANITE_BUNDLE"]
fn granite_spike_parity() {
    let bundle = std::env::var("MCP_VL_MSA_GRANITE_BUNDLE")
        .expect("MCP_VL_MSA_GRANITE_BUNDLE env var must point at a prepared bundle dir");
    let bundle_dir = PathBuf::from(bundle);

    let encoder = CandleModernBert::load_from_dir(
        &bundle_dir,
        "ibm-granite/granite-embedding-97m-multilingual-r2",
        "spike-parity-anchor",
        "granite-r2-97m",
        384,
        Some(8192),
    )
    .expect("load granite bundle");

    assert_eq!(
        encoder.dim(),
        384,
        "Granite-R2-97m produces 384-dim vectors"
    );

    // Encode passages and queries. Granite ignores kind, but we still pass
    // it explicitly to exercise the trait surface.
    let passages = encoder
        .encode_batch_kind(EmbedInputKind::Passage, FIXTURES)
        .expect("encode passages");
    let queries = encoder
        .encode_batch_kind(EmbedInputKind::Query, QUERIES)
        .expect("encode queries");

    assert_eq!(passages.len(), FIXTURES.len());
    assert_eq!(queries.len(), QUERIES.len());

    for (qi, q_vec) in queries.iter().enumerate() {
        for (fi, p_vec) in passages.iter().enumerate() {
            let got = cosine(q_vec, p_vec);
            let want = SNAPSHOT[qi][fi];
            let delta = (got - want).abs();
            assert!(
                delta <= TOLERANCE,
                "Q{qi}xF{fi}: got {got:.4} expected {want:.4} (delta {delta:.4} > tolerance {TOLERANCE})"
            );
        }
    }

    // Sanity ranking checks (top-1 must remain stable even if individual
    // cosines drift within tolerance).
    let top1 = |query_idx: usize| -> usize {
        let q = &queries[query_idx];
        passages
            .iter()
            .enumerate()
            .map(|(i, p)| (i, cosine(q, p)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
    };
    assert_eq!(top1(0), 1, "Q0 top-1 must be F1 (mcp-msa-rs)");
    assert_eq!(top1(1), 3, "Q1 top-1 must be F3 (mean pooling encoder)");
}
