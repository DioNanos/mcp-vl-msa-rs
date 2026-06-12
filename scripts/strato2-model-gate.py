#!/usr/bin/env python3
"""Model-in-loop injection-ablation gate (M3 Strato 2).

Consumes the per-query evidence JSONL emitted by
`msa-rag-bench --gate injection-ablation --gate-emit-evidence <path>`
(rows: query_id, question, answer, ctx_c full-text, ctx_d snippet-only)
and asks a generator to answer each question twice — once from ctx_c
(condition C, original-text injection) and once from ctx_d (condition D,
snippet-only). Scores SQuAD-style token F1 and exact match against the
gold answer. The deterministic gate measured answer *presence*; this one
measures the downstream effect the paper's -37% lever includes: answering
worse from fragments even when the answer is present.

Generator: any Anthropic-Messages-compatible endpoint. Defaults target
Z.AI GLM (base URL + key from env). The key is read from the environment
and never logged.

Usage:
  export ZAI_API_KEY_A=...   # or pass --api-key-env
  python3 scripts/strato2-model-gate.py \
    --evidence /tmp/strato2_evidence_hotpotqa.jsonl \
    --output crates/msa-bench/results/2026-06-05_hotpotqa_strato2_glm51.json \
    --limit 200

Incremental per-query rows are checkpointed to <output>.rows.jsonl so an
interrupted run can resume with --resume.
"""

import argparse
import collections
import json
import os
import re
import string
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------- scoring
# SQuAD official normalization: lowercase, strip punctuation/articles,
# collapse whitespace.

ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = s.translate(PUNCT_TABLE)
    s = ARTICLES.sub(" ", s)
    return " ".join(s.split())


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


def presence(ctx: str, gold: str) -> bool:
    """Mirror of gate.rs normalize+contains (lowercase, ws-collapsed)."""
    norm = lambda t: " ".join(t.lower().split())
    return norm(gold) in norm(ctx)


# ---------------------------------------------------------------- generator

SYSTEM_PROMPT = (
    "You answer questions strictly from the provided evidence. "
    "Reply with ONLY the shortest exact answer span (a few words). "
    "No explanation, no preamble, no punctuation beyond what the answer itself requires. "
    "If the evidence does not state the answer explicitly, reply with your best "
    "guess based only on the evidence."
)


def ask(base_url: str, api_key: str, model: str, ctx: str, question: str,
        timeout: int, max_retries: int = 3) -> str:
    body = json.dumps({
        "model": model,
        "max_tokens": 64,
        "temperature": 0,
        "system": SYSTEM_PROMPT,
        "messages": [{
            "role": "user",
            "content": f"Evidence:\n{ctx}\n\nQuestion: {question}\nAnswer:",
        }],
    }).encode()
    url = f"{base_url.rstrip('/')}/v1/messages"
    last_err = None
    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": "2023-06-01",
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
            parts = data.get("content") or []
            text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
            return text.strip()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:200]
            last_err = f"HTTP {e.code}: {detail}"
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(last_err)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = repr(e)
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"giving up after {max_retries} attempts: {last_err}")


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=200,
                    help="first N evidence rows (deterministic subset)")
    ap.add_argument("--base-url", default="https://api.z.ai/api/anthropic")
    ap.add_argument("--model", default="GLM-5.1")
    ap.add_argument("--api-key-env", default="ZAI_API_KEY_A")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="pause between calls (rate-limit kindness)")
    ap.add_argument("--resume", action="store_true",
                    help="skip query_ids already in <output>.rows.jsonl")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        print(f"FATAL: env {args.api_key_env} empty/missing", file=sys.stderr)
        return 2

    rows = []
    with open(args.evidence, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if len(rows) >= args.limit:
                break
    if not rows:
        print("FATAL: no evidence rows", file=sys.stderr)
        return 2

    rows_path = args.output + ".rows.jsonl"
    done = {}
    if args.resume and os.path.exists(rows_path):
        with open(rows_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["query_id"]] = r
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"resume: {len(done)} rows already scored", file=sys.stderr)

    out_rows = open(rows_path, "a", encoding="utf-8")
    scored = list(done.values())
    t_start = time.time()
    consecutive_failures = 0

    for i, row in enumerate(rows):
        if row["query_id"] in done:
            continue
        try:
            pred_c = ask(args.base_url, api_key, args.model,
                         row["ctx_c"], row["question"], args.timeout)
            time.sleep(args.sleep)
            pred_d = ask(args.base_url, api_key, args.model,
                         row["ctx_d"], row["question"], args.timeout)
            time.sleep(args.sleep)
            consecutive_failures = 0
        except RuntimeError as e:
            consecutive_failures += 1
            print(f"[{i+1}/{len(rows)}] {row['query_id']} FAILED: {e}", file=sys.stderr)
            if consecutive_failures >= 3:
                print("FATAL: 3 consecutive failures — stopping (per runbook: "
                      "stop on problems, do not thrash)", file=sys.stderr)
                break
            continue

        rec = {
            "query_id": row["query_id"],
            "question": row["question"],
            "gold": row["answer"],
            "pred_c": pred_c,
            "pred_d": pred_d,
            "f1_c": f1_score(pred_c, row["answer"]),
            "f1_d": f1_score(pred_d, row["answer"]),
            "em_c": exact_match(pred_c, row["answer"]),
            "em_d": exact_match(pred_d, row["answer"]),
            "present_c": presence(row["ctx_c"], row["answer"]),
            "present_d": presence(row["ctx_d"], row["answer"]),
        }
        scored.append(rec)
        out_rows.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_rows.flush()
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            el = time.time() - t_start
            mc = sum(r["f1_c"] for r in scored) / len(scored)
            md = sum(r["f1_d"] for r in scored) / len(scored)
            print(f"[{i+1}/{len(rows)}] scored={len(scored)} "
                  f"F1_C={mc:.3f} F1_D={md:.3f} elapsed={el:.0f}s", file=sys.stderr)

    out_rows.close()
    if not scored:
        print("FATAL: nothing scored", file=sys.stderr)
        return 1

    n = len(scored)
    mean = lambda k: sum(r[k] for r in scored) / n
    f1_c, f1_d = mean("f1_c"), mean("f1_d")
    delta = (f1_c - f1_d) * 100

    both_present = [r for r in scored if r["present_c"] and r["present_d"]]
    strat = None
    if both_present:
        bp = len(both_present)
        strat = {
            "n": bp,
            "f1_c": sum(r["f1_c"] for r in both_present) / bp,
            "f1_d": sum(r["f1_d"] for r in both_present) / bp,
        }
        strat["delta_points"] = (strat["f1_c"] - strat["f1_d"]) * 100

    summary = {
        "gate": "strato2-model-in-loop",
        "evidence_file": args.evidence,
        "model": args.model,
        "base_url": args.base_url,
        "temperature": 0,
        "num_scored": n,
        "f1_c": f1_c,
        "f1_d": f1_d,
        "delta_points": delta,
        "em_c": mean("em_c"),
        "em_d": mean("em_d"),
        "acceptance_rule": "F1(C) > F1(D) + 5 points",
        "passed": delta > 5.0,
        "both_present_stratum": strat,
        "note": (
            "Strong generator (GLM-5.1) compresses the C-D gap via parametric "
            "knowledge; treat delta as a conservative floor of the paper's -37% "
            "lever, complementing the deterministic presence gate. The "
            "both_present_stratum isolates 'answers worse from fragments even "
            "when the answer is present'."
        ),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    verdict = "PASS" if summary["passed"] else "FAIL"
    print(f"msa-a5-recall-gate-strato2:{verdict}:F1_C={f1_c:.4f}:F1_D={f1_d:.4f}:delta={delta:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
