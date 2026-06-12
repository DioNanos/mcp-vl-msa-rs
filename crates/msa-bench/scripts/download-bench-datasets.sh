#!/usr/bin/env bash
#
# download-bench-datasets.sh
#
# Pulls the MSA-RAG-BENCHMARKS dataset bundle from HuggingFace and
# converts the proprietary pickle format into the BEIR-compatible JSONL
# layout the Rust loader expects.
#
# Usage:
#   ./scripts/download-bench-datasets.sh                  # default: hotpotqa, 2wikimultihopqa
#   ./scripts/download-bench-datasets.sh hotpotqa
#   ./scripts/download-bench-datasets.sh hotpotqa 2wikimultihopqa musique

set -euo pipefail

REPO="EverMind-AI/MSA-RAG-BENCHMARKS"
BASE_URL="https://huggingface.co/datasets/${REPO}/resolve/main"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASETS_DIR="${ROOT}/datasets"

DEFAULT_BENCHES=(hotpotqa 2wikimultihopqa)
if [[ $# -eq 0 ]]; then
    BENCHES=("${DEFAULT_BENCHES[@]}")
else
    BENCHES=("$@")
fi

command -v python3 >/dev/null || { echo "python3 required" >&2; exit 1; }
command -v curl >/dev/null || { echo "curl required" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "sha256sum required" >&2; exit 1; }

for bench in "${BENCHES[@]}"; do
    out="${DATASETS_DIR}/${bench}"
    mkdir -p "${out}/qrels" "${out}/.staging"

    for kind in qdata mdata; do
        f="${kind}_${bench}.pkl"
        url="${BASE_URL}/${bench}/${f}"
        dst="${out}/.staging/${f}"
        if [[ -s "$dst" ]]; then
            echo "[skip] ${dst}"
        else
            echo "[get ] ${url} -> ${dst}"
            curl -fsSL --retry 3 -o "$dst" "$url"
        fi
    done

    QDATA="${out}/.staging/qdata_${bench}.pkl"
    MDATA="${out}/.staging/mdata_${bench}.pkl"
    QSHA=$(sha256sum "$QDATA" | awk '{print $1}')
    MSHA=$(sha256sum "$MDATA" | awk '{print $1}')

    echo "[conv] ${bench}: pickle -> BEIR JSONL"
    python3 - "$QDATA" "$MDATA" "$out" "$bench" <<'PYEOF'
import json, pickle, sys, os

qpath, mpath, outdir, bench = sys.argv[1:5]
with open(qpath, 'rb') as f:
    qdata = pickle.load(f)
with open(mpath, 'rb') as f:
    mdata = pickle.load(f)

assert isinstance(mdata, list), f"mdata is {type(mdata).__name__}"
assert isinstance(qdata, list), f"qdata is {type(qdata).__name__}"

text_to_id = {}
prefix_to_id = {}
corpus_path = os.path.join(outdir, "corpus.jsonl")
with open(corpus_path, 'w', encoding='utf-8') as cf:
    for i, t in enumerate(mdata):
        if not isinstance(t, str):
            t = str(t)
        doc_id = str(i)
        text_to_id[t] = doc_id
        prefix_to_id.setdefault(t[:80], doc_id)
        cf.write(json.dumps({"_id": doc_id, "text": t}, ensure_ascii=False) + "\n")

queries_path = os.path.join(outdir, "queries.jsonl")
qrels_path   = os.path.join(outdir, "qrels", "test.tsv")
answers_path = os.path.join(outdir, "answers.jsonl")
matched_exact = 0
matched_prefix = 0
unmatched = 0
num_answers = 0
with open(queries_path, 'w', encoding='utf-8') as qf, \
     open(qrels_path, 'w', encoding='utf-8') as rf, \
     open(answers_path, 'w', encoding='utf-8') as af:
    rf.write("query-id\titeration\tdoc-id\trelevance\n")
    for i, row in enumerate(qdata):
        if not isinstance(row, dict):
            continue
        qid = f"q-{i}"
        text = row.get("query") or row.get("question") or ""
        refs = row.get("reference_list") or row.get("references") or []
        qf.write(json.dumps({"_id": qid, "text": text}, ensure_ascii=False) + "\n")
        # Gold answer when the QA dataset ships one (used by the
        # injection-ablation gate; harmless to omit for pure retrieval dumps).
        ans = row.get("answer")
        if ans is not None:
            af.write(json.dumps({"_id": qid, "answer": str(ans)}, ensure_ascii=False) + "\n")
            num_answers += 1
        for ref in refs:
            if not isinstance(ref, str):
                continue
            doc_id = text_to_id.get(ref)
            if doc_id is not None:
                matched_exact += 1
            else:
                doc_id = prefix_to_id.get(ref[:80])
                if doc_id is not None:
                    matched_prefix += 1
                else:
                    unmatched += 1
                    continue
            rf.write(f"{qid}\t0\t{doc_id}\t1\n")

print(json.dumps({
    "corpus_size": len(mdata),
    "num_queries": len(qdata),
    "matched_exact": matched_exact,
    "matched_prefix": matched_prefix,
    "unmatched_refs": unmatched,
    "num_answers": num_answers,
}, indent=2))
PYEOF

    cat > "${out}/MSA_BENCH.txt" <<EOF
source_repo: ${REPO}
source_url: ${BASE_URL}/${bench}
qdata_sha256: ${QSHA}
mdata_sha256: ${MSHA}
converted_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
layout: BEIR-compatible (corpus.jsonl, queries.jsonl, qrels/test.tsv)
relevance_derivation: |
  binary qrels derived by exact-string match between qdata[i].reference_list
  and the mdata corpus; fallback on first 80-char prefix when exact fails.
EOF

    echo "[ok  ] ${out}"
    echo
done

echo "All done. Datasets ready under: ${DATASETS_DIR}"
