# msa-bench

End-to-end retrieval benchmark for `mcp-msa-rs`, evaluating the
hybrid retrieval stack (Tantivy BM25 + optional Candle ModernBERT
dense rerank) against open RAG benchmarks.

## Why this crate

A1 ships an in-process Candle backend defaulting to
`ibm-granite/granite-embedding-97m-multilingual-r2`. The spike
validation (`2026-05-17_dense_scorer_spike_report.md`) covered only
6 fixtures and 2 queries; before committing to the storage-sidecar
A2 we need real benchmark numbers on a public corpus to confirm the
encoder lift is statistically meaningful and to decide the default
fusion weight `dense_alpha`.

## Datasets

We use [EverMind-AI/MSA-RAG-BENCHMARKS](https://huggingface.co/datasets/EverMind-AI/MSA-RAG-BENCHMARKS),
the dataset bundle accompanying the MSA paper (arXiv:2603.23516). The
initial scope is:

- **HotpotQA** — multi-hop QA, 7.4M passages
- **2WikiMultiHopQA** — multi-hop QA, ~430k passages

Both target the multi-hop reasoning where Granite-R2-97m claims a
+8.7 MTEB ML lead over `multilingual-e5-small`.

Datasets are downloaded into `datasets/<benchmark>/` and **not
committed** (see `datasets/.gitignore`). Fetch them once with:

```sh
./scripts/download-bench-datasets.sh
```

## Granite bundle

The encoder needs a local Granite bundle prepared by
`scripts/prepare-granite-r2-97m.sh` (root of the repo). Set the
runtime env var:

```sh
export MCP_VL_MSA_GRANITE_BUNDLE=/path/to/granite-r2-97m
```

## Run

### BM25-only regression gate (no model required)

A pure BM25 sweep (`--alpha-sweep 1.0`) needs **no Granite bundle** — it runs
on any machine, including RPi3-class hardware, and is the cheap regression gate
used to confirm retrieval quality has not dropped:

```sh
cd <repo-root>
cargo run --release --features bench --bin msa-rag-bench -- \
  --dataset hotpotqa \
  --limit-queries 300 \
  --limit-corpus 3000 \
  --alpha-sweep 1.0 \
  --topn-rerank-sweep 20 \
  --output crates/msa-bench/results/<date>_bm25_hotpot.json
```

Committed baseline: `results/2026-05-30_bm25_hotpot_baseline.json`
(NDCG@10 ≈ 0.796, MRR@10 ≈ 0.875, R@10 ≈ 0.895 on the 3131-doc / 300-query
subset; index build ~0.7s, query p50 <1ms — x86_64 CPU).

To exercise the **real P=64 word chunker** instead of doc-level passages, add
`--chunk-size 64`.

### Full hybrid sweep (needs Granite bundle)

```sh
export MCP_VL_MSA_GRANITE_BUNDLE=/path/to/granite-r2-97m   # see prepare-granite-r2-97m.sh
cargo run --release --features bench --bin msa-rag-bench -- \
  --dataset hotpotqa \
  --limit-queries 500 \
  --limit-corpus 5000 \
  --alpha-sweep 0.0,0.3,0.5,0.7,1.0 \
  --topn-rerank-sweep 20,50,100 \
  --output crates/msa-bench/results/<date>_hotpotqa.json
```

## Metrics

Reported per `(dataset, alpha, topn_rerank)` cell:

- **NDCG@10** — log2-discount, ideal-DCG normalization
- **MRR@10** — reciprocal rank of the first relevant hit, truncated at 10
- **Recall@10** / **Recall@20** — fraction of relevant items retrieved
- **Latency p50 / p95** — query-time wall clock, excluding index build
- **Index build cost** — reported once per run separately

`alpha = 1.0` is BM25-only baseline. `alpha = 0.0` is dense-only.
Anything in between is hybrid:

```text
score = α · bm25_normalized + (1 − α) · ((cosine + 1) / 2)
```

## Layout

```
crates/msa-bench/
├── Cargo.toml
├── README.md
├── src/
│   ├── main.rs        bin entry + CLI
│   ├── dataset.rs     HotpotQA / 2WikiMultiHopQA loaders
│   ├── metrics.rs     NDCG@k, MRR@k, Recall@k, latency percentiles
│   └── runner.rs      alpha sweep, topN sweep, eval loop
├── datasets/          gitignored — populated by scripts/download-bench-datasets.sh
├── results/           checked-in small JSON outputs
└── scripts/
    └── download-bench-datasets.sh
```

## Output

Each run writes one JSON file under `results/`:

```json
{
  "run_id": "2026-05-18T08:00:00Z-hotpotqa",
  "dataset": "hotpotqa",
  "corpus_size": 5000,
  "num_queries": 500,
  "model_id": "ibm-granite/granite-embedding-97m-multilingual-r2",
  "model_version": "<bundle sha>",
  "index_build_ms": 12345,
  "cells": [
    {
      "alpha": 0.5,
      "topn_rerank": 50,
      "ndcg_at_10": 0.612,
      "mrr_at_10": 0.534,
      "recall_at_10": 0.78,
      "recall_at_20": 0.86,
      "latency_p50_ms": 41,
      "latency_p95_ms": 88
    }
  ]
}
```

Small enough to commit; trends over time are visible in `git log results/`.
