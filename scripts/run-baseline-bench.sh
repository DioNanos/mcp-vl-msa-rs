#!/usr/bin/env bash
# One-shot baseline measurement: BM25-only vs BM25+Granite-R2-97m dense hybrid,
# on hotpotqa + 2wikimultihopqa (bounded subset). Logs everything; writes JSON
# under crates/msa-bench/results/. Launched in background.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

BUNDLE="${GRANITE_BUNDLE:-$HOME/.local/state/mcp-vl-msa-rs/granite-r2-97m}"
TS="$(date -u +%Y-%m-%d)"
LOG="$REPO_ROOT/crates/msa-bench/results/${TS}_baseline_run.log"
mkdir -p "$REPO_ROOT/crates/msa-bench/results"

exec > >(tee -a "$LOG") 2>&1
echo "=== baseline bench run @ $(date -u +%FT%TZ) ==="

echo "--- step 1: prepare Granite-R2-97m bundle ($BUNDLE) ---"
scripts/prepare-granite-r2-97m.sh "$BUNDLE"

export MCP_VL_MSA_GRANITE_BUNDLE="$BUNDLE"

echo "--- step 2: build msa-rag-bench (--features bench, Candle) ---"
cargo build --release --features bench --bin msa-rag-bench

echo "--- step 3: sweep hotpotqa ---"
cargo run --release --features bench --bin msa-rag-bench -- \
  --dataset hotpotqa \
  --limit-queries 500 --limit-corpus 5000 \
  --alpha-sweep 0.0,0.3,0.5,0.7,1.0 \
  --topn-rerank-sweep 20,50,100 \
  --output "crates/msa-bench/results/${TS}_hotpotqa_baseline_hybrid.json"

echo "--- step 4: sweep 2wikimultihopqa ---"
cargo run --release --features bench --bin msa-rag-bench -- \
  --dataset 2wikimultihopqa \
  --limit-queries 500 --limit-corpus 5000 \
  --alpha-sweep 0.0,0.3,0.5,0.7,1.0 \
  --topn-rerank-sweep 20,50,100 \
  --output "crates/msa-bench/results/${TS}_2wikimultihopqa_baseline_hybrid.json"

echo "=== DONE @ $(date -u +%FT%TZ) ==="
