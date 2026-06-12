#!/usr/bin/env bash
# Reader-step gate run — design 2026-06-12 (post-parere Codex).
# Single variable = answering protocol; token budget held at 512 everywhere.
set -euo pipefail
cd "$(dirname "$0")/.."   # crates/msa-bench

source ~/.config/ai-shell/providers.zsh 2>/dev/null || true
: "${ZAI_API_KEY_A:?ZAI_API_KEY_A not set}"

EV=results/2026-06-07_consolidation_ev.jsonl
PY=scripts/convbench_model_gate.py
D=2026-06-12

echo "=== R1 control: flat + direct @512 ($(date +%H:%M))"
python3 "$PY" --evidence "$EV" --renders flat --ask-mode direct --max-tokens 512 \
  --output "results/${D}_readerstep_control_direct512.json"

echo "=== R2 primary: flat + extract @512 ($(date +%H:%M))"
python3 "$PY" --evidence "$EV" --renders flat --ask-mode extract --max-tokens 512 \
  --output "results/${D}_readerstep_extract512.json"

echo "=== R3 exploratory: structured-now + extract @512 ($(date +%H:%M))"
python3 "$PY" --evidence "$EV" --renders structured-now --ask-mode extract --max-tokens 512 \
  --output "results/${D}_readerstep_explore_snow_extract512.json"

echo "=== DONE ($(date +%H:%M)) — summaries in results/${D}_readerstep_*.json"
