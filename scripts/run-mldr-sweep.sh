#!/usr/bin/env bash
# Quality sweep on mldr-it: BM25 vs dense/hybrid across the 5 Ollama tier
# models + the granite-97m candle baseline. Bounded corpus for tractable CPU
# encode. Each model = one JSON under results/. Logs everything. Background.
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"; cd "$ROOT"
BIN=./target/release/msa-rag-bench
DR=crates/msa-bench/datasets
OUT=crates/msa-bench/results
TS="$(date -u +%Y-%m-%d)"
LOG="$OUT/${TS}_mldr-it_sweep.log"
GRANITE=${GRANITE_BUNDLE:-$HOME/.local/state/mcp-vl-msa-rs/granite-r2-97m}
ALPHAS="0.0,0.3,0.5,0.7,1.0"; LC=500; LQ=100; NCTX=512; NTHR=4
exec > >(tee -a "$LOG") 2>&1
echo "=== mldr-it sweep @ $(date -u +%FT%TZ) | corpus=$LC queries=$LQ num_ctx=$NCTX ==="

run_ollama() { # name model dim
  echo "--- [$(date -u +%T)] ollama:$1 (dim $3) ---"
  MCP_VL_MSA_GRANITE_BUNDLE=/dev/null timeout 5400 "$BIN" \
    --dataset mldr-it --datasets-root "$DR" \
    --encoder-backend ollama --ollama-model "$2" --ollama-dim "$3" \
    --ollama-num-ctx "$NCTX" --ollama-num-thread "$NTHR" \
    --limit-corpus "$LC" --limit-queries "$LQ" \
    --alpha-sweep "$ALPHAS" --topn-rerank-sweep 50 \
    --output "$OUT/${TS}_mldr-it_${1}.json" || echo "!! $1 FAILED"
}

run_ollama qwen3-emb        qwen3-embedding:0.6b      1024
run_ollama nomic-v2-moe     nomic-embed-text-v2-moe   768
run_ollama embeddinggemma   embeddinggemma            768
run_ollama arctic2          snowflake-arctic-embed2   1024
run_ollama bge-m3           bge-m3                    1024

echo "--- [$(date -u +%T)] candle granite-97m baseline ---"
MCP_VL_MSA_GRANITE_BUNDLE="$GRANITE" timeout 3600 "$BIN" \
  --dataset mldr-it --datasets-root "$DR" \
  --encoder-backend candle-modernbert --bundle "$GRANITE" \
  --limit-corpus "$LC" --limit-queries "$LQ" \
  --alpha-sweep "$ALPHAS" --topn-rerank-sweep 50 \
  --output "$OUT/${TS}_mldr-it_granite97m.json" || echo "!! granite FAILED"

echo "=== SWEEP_DONE @ $(date -u +%FT%TZ) ==="
