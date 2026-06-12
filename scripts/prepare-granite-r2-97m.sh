#!/usr/bin/env bash
#
# prepare-granite-r2-97m.sh
#
# Setup-time preparation of a local bundle for the
# `ibm-granite/granite-embedding-97m-multilingual-r2` ModernBERT encoder.
#
# Outputs a directory containing:
#   <bundle_dir>/config.json
#   <bundle_dir>/tokenizer.json
#   <bundle_dir>/model.safetensors    # weight keys rewritten with `model.` prefix
#   <bundle_dir>/MODEL_BUNDLE.txt     # provenance: source URLs + sha256s + date
#
# The runtime mcp-msa-rs server reads this directory directly when the
# `[embeddings]` section selects `backend = "candle-modernbert"` and
# `model_dir = "<bundle_dir>"`. The server never talks to HuggingFace at
# runtime; all download / rename happens here, once, at setup time.
#
# Why the rename: Granite-R2 distributes safetensors with weight keys
# starting at `embeddings.tok_embeddings.weight`, `layers.0.attn.Wo.weight`,
# etc. candle-transformers `ModernBert::load` expects the keys to live under
# `model.embeddings.tok_embeddings.weight` and so on. Prefixing once at
# setup time keeps the runtime fully pure-Rust and removes any need for a
# custom VarBuilder wrapper.
#
# Usage:
#   scripts/prepare-granite-r2-97m.sh <bundle_dir>
#
# Environment overrides:
#   GRANITE_REPO       default: ibm-granite/granite-embedding-97m-multilingual-r2
#   GRANITE_REVISION   default: main
#   HF_TOKEN           optional, for gated downloads (Granite is public so not required)

set -euo pipefail

REPO="${GRANITE_REPO:-ibm-granite/granite-embedding-97m-multilingual-r2}"
REV="${GRANITE_REVISION:-main}"

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <bundle_dir>" >&2
    exit 2
fi
BUNDLE_DIR="$1"

command -v python3 >/dev/null || { echo "python3 required" >&2; exit 1; }
command -v curl >/dev/null || { echo "curl required" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "sha256sum required" >&2; exit 1; }

mkdir -p "$BUNDLE_DIR"

HF_BASE="https://huggingface.co/${REPO}/resolve/${REV}"
CURL_AUTH=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    CURL_AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

fetch() {
    local fname="$1"
    local dest="${BUNDLE_DIR}/${fname}"
    if [[ -s "$dest" ]]; then
        echo "[skip] $dest already present"
        return 0
    fi
    echo "[get ] $HF_BASE/$fname  ->  $dest"
    curl -fSL --retry 3 "${CURL_AUTH[@]}" -o "$dest" "${HF_BASE}/${fname}"
}

fetch "config.json"
fetch "tokenizer.json"

# Download original safetensors to a staging filename so we can verify the
# digest before we overwrite the (potentially already-prepared) bundle file.
ORIG_STAGE="${BUNDLE_DIR}/.staging.model.original.safetensors"
PREPARED="${BUNDLE_DIR}/model.safetensors"
if [[ ! -s "$ORIG_STAGE" ]]; then
    echo "[get ] $HF_BASE/model.safetensors  ->  $ORIG_STAGE"
    curl -fSL --retry 3 "${CURL_AUTH[@]}" -o "$ORIG_STAGE" \
        "${HF_BASE}/model.safetensors"
fi
ORIG_SHA="$(sha256sum "$ORIG_STAGE" | awk '{print $1}')"
echo "[sha ] original safetensors: $ORIG_SHA"

# Atomically rewrite the safetensors header so every tensor key is prefixed
# with `model.`, matching candle ModernBert::load expectations. The tensor
# data payload is byte-identical; only the leading JSON header changes.
echo "[xfm ] rewriting safetensor keys with 'model.' prefix"
python3 - "$ORIG_STAGE" "${PREPARED}.tmp" <<'PY'
import json, struct, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src, "rb") as f:
    hdr_len = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(hdr_len))
    payload = f.read()
new = {}
for k, v in hdr.items():
    new[k if k == "__metadata__" else "model." + k] = v
new_hdr = json.dumps(new, separators=(",", ":")).encode("utf-8")
pad = (8 - (len(new_hdr) % 8)) % 8
new_hdr += b" " * pad
with open(dst, "wb") as f:
    f.write(struct.pack("<Q", len(new_hdr)))
    f.write(new_hdr)
    f.write(payload)
PY
mv "${PREPARED}.tmp" "$PREPARED"
PREPARED_SHA="$(sha256sum "$PREPARED" | awk '{print $1}')"
echo "[sha ] prepared safetensors: $PREPARED_SHA"

# Remove the staging copy to save space.
rm -f "$ORIG_STAGE"

# Provenance record so the runtime can log model_version and so audits can
# verify the bundle later.
cat > "${BUNDLE_DIR}/MODEL_BUNDLE.txt" <<EOF
repo: ${REPO}
revision: ${REV}
source: ${HF_BASE}
prepared_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
original_safetensors_sha256: ${ORIG_SHA}
prepared_safetensors_sha256: ${PREPARED_SHA}
notes: |
  Safetensor weight keys rewritten with 'model.' prefix to match
  candle-transformers ModernBert::load expectations. Tensor payload is
  byte-identical to the original; only the leading JSON header changed.
EOF

echo
echo "[ok  ] bundle ready: $BUNDLE_DIR"
echo
echo "Next: point mcp-msa-rs config at this directory, e.g.:"
echo
cat <<EOF
  [embeddings]
  backend = "candle-modernbert"
  dim = 384
  model_id = "${REPO}"
  model_version = "${PREPARED_SHA}"
  profile = "granite-r2-97m"
  model_dir = "${BUNDLE_DIR}"
  max_len = 8192
EOF
