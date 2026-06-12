#!/usr/bin/env bash
# Reader-step gate — SECOND READER replication (model-diversity check).
# Identical harness, evidence, prompts, temperature(0), scoring and sample
# (seed 11) as run_readerstep_gate_20260612.sh; the ONLY variable is the
# reader model/endpoint. First reader was GLM-5.1 (Z.AI). This run points the
# same script at an Ollama Cloud model over its Anthropic /v1/messages endpoint.
#
# Usage: MODEL=deepseek-v4-pro:cloud ./scripts/run_readerstep_gate_ollama.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # crates/msa-bench

MODEL="${MODEL:-deepseek-v4-pro:cloud}"
BASE_URL="${BASE_URL:-http://localhost:11434}"
export OLLAMA_BENCH_KEY="${OLLAMA_BENCH_KEY:-ollama}"
# Reasoning models (deepseek-v4-pro) emit a `thinking` block that counts against
# max_tokens. At 512 the answer text was truncated/empty on hard questions
# (22-46% empty preds). Use a generous cap, identical across all 3 conditions so
# the within-model extract-vs-direct gate stays controlled. GLM (non-reasoning)
# ran clean at 512. Override with MAXTOK=... if needed.
MAXTOK="${MAXTOK:-4096}"

EV=results/2026-06-07_consolidation_ev.jsonl
PY=scripts/convbench_model_gate.py
D=2026-06-12
SLUG="$(echo "$MODEL" | tr ':/.' '___')"
OUTDIR=results
COMMON=(--evidence "$EV" --base-url "$BASE_URL" --model "$MODEL"
        --api-key-env OLLAMA_BENCH_KEY --max-tokens "$MAXTOK")

echo "=== reader=$MODEL @ $BASE_URL (max_tokens=$MAXTOK) ==="

echo "=== R1 control: flat + direct @512 ($(date +%H:%M))"
python3 "$PY" "${COMMON[@]}" --renders flat --ask-mode direct \
  --output "${OUTDIR}/${D}_readerstep_${SLUG}_control_direct512.json"

echo "=== R2 primary: flat + extract @512 ($(date +%H:%M))"
python3 "$PY" "${COMMON[@]}" --renders flat --ask-mode extract \
  --output "${OUTDIR}/${D}_readerstep_${SLUG}_extract512.json"

echo "=== R3 exploratory: structured-now + extract @512 ($(date +%H:%M))"
python3 "$PY" "${COMMON[@]}" --renders structured-now --ask-mode extract \
  --output "${OUTDIR}/${D}_readerstep_${SLUG}_snow_extract512.json"

echo "=== DONE ($(date +%H:%M)) — summaries: ${OUTDIR}/${D}_readerstep_${SLUG}_*.json"
