#!/usr/bin/env python3
"""Model-in-loop reading gate for the conversational bench (design v2).

Consumes the evidence JSONL emitted by `msa-rag-bench --gate conv-bench
--convbench-emit-evidence` (top-K rich capsule texts from the bm25-rich
condition) and asks a generator to answer each question from that evidence.
SQuAD token-F1 per question category. Labeled *proxy deterministic-ingest*:
this measures whether the READER aggregates the evidence the retrieval now
delivers (the live question after C1-C3b: multi-session coverage is ~88%,
does the answer survive the reading step?).

Usage:
  export ZAI_API_KEY_A=...
  python3 convbench_model_gate.py --evidence ev.jsonl --output out.json \
      --per-category 50
"""

import argparse
import collections
import json
import os
import random
import re
import string
import sys
import time
import urllib.error
import urllib.request

random.seed(11)

ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower().translate(PUNCT_TABLE)
    s = ARTICLES.sub(" ", s)
    return " ".join(s.split())


def f1_score(prediction: str, gold: str) -> float:
    p = normalize_answer(prediction).split()
    g = normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided memory excerpts from past "
    "chat sessions. Reply with the shortest exact answer (a few words or a "
    "short phrase). No explanation. If the excerpts do not contain enough "
    "information, give your best guess from them."
)

# Reader-step (famiglia B, esperimento 2026-06-07): scaffolding di lettura.
# Stessa evidenza, stesso render; cambia SOLO il protocollo di risposta: il
# modello prima enumera i fatti rilevanti da TUTTI gli estratti, poi risponde.
SYSTEM_PROMPT_EXTRACT = (
    "You answer questions using ONLY the provided memory excerpts from past "
    "chat sessions. Work in two steps. STEP 1 — go through EVERY excerpt and "
    "list each fact relevant to the question (one per line, include "
    "counts/dates/values; aggregate across excerpts when the question asks "
    "for totals, counts or comparisons). STEP 2 — on the last line write "
    "'FINAL:' followed by the shortest exact answer (a few words). If the "
    "excerpts are insufficient, give your best guess from them."
)

def parse_final(pred):
    """Estrae la risposta dopo l'ultimo 'FINAL:' (fallback: testo intero)."""
    if "FINAL:" in pred:
        return pred.rsplit("FINAL:", 1)[1].strip()
    return pred.strip()


def render_flat(evidence, question_date=None):
    """R1: elenco piatto, ordine rilevanza (baseline)."""
    return "\n\n".join(f"[memory {i+1}] {e['text']}" for i, e in enumerate(evidence)), None


def render_structured(evidence, question_date=None, with_now=False):
    """R2a/R2b: gruppi per sessione, data in intestazione, ordine cronologico.
    SOLO riordino/intestazioni: il set di testi e' identico a render_flat
    (freeze dell'evidenza — condizione del parere Codex)."""
    by_sess = {}
    for e in evidence:
        by_sess.setdefault((e.get("date") or "unknown", e.get("session")), []).append(e["text"])
    parts = []
    for (date, _sess), texts in sorted(by_sess.items(), key=lambda kv: kv[0][0]):
        body = "\n".join(f"- {t}" for t in texts)
        parts.append(f"[session of {date}]\n{body}")
    ctx = "\n\n".join(parts)
    now = f"Current date: {question_date}" if (with_now and question_date) else None
    return ctx, now


def render_inline_dates(evidence, question_date=None):
    """R3: ordine di RILEVANZA intatto (protegge multi-session) + data inline
    per capsula + Current date. Il render candidato per recall_section."""
    ctx = "\n\n".join(
        f"[memory {i+1} — {e.get('date') or 'unknown date'}] {e['text']}"
        for i, e in enumerate(evidence)
    )
    now = f"Current date: {question_date}" if question_date else None
    return ctx, now


RENDERS = {
    "flat": lambda ev, qd: render_flat(ev, qd),
    "structured": lambda ev, qd: render_structured(ev, qd, with_now=False),
    "structured-now": lambda ev, qd: render_structured(ev, qd, with_now=True),
    "inline-dates": lambda ev, qd: render_inline_dates(ev, qd),
}


def ask(base_url, api_key, model, ctx, now_line, question, timeout=120, retries=3,
        system_prompt=SYSTEM_PROMPT, max_tokens=None):
    now_part = f"{now_line}\n" if now_line else ""
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens or 96,
        "temperature": 0,
        "system": system_prompt,
        "messages": [{"role": "user",
                      "content": f"{now_part}Memory excerpts:\n{ctx}\n\nQuestion: {question}\nAnswer:"}],
    }).encode()
    url = f"{base_url.rstrip('/')}/v1/messages"
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": "2023-06-01",
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
            return "".join(p.get("text", "") for p in data.get("content", [])
                           if p.get("type") == "text").strip()
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode(errors='replace')[:160]}"
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(last)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = repr(e)
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"giving up: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--per-category", type=int, default=50,
                    help="sample size per question category")
    ap.add_argument("--categories", default="multi-session,temporal-reasoning,"
                    "knowledge-update,single-session-user")
    ap.add_argument("--base-url", default="https://api.z.ai/api/anthropic")
    ap.add_argument("--model", default="GLM-5.1")
    ap.add_argument("--api-key-env", default="ZAI_API_KEY_A")
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="max_tokens per la risposta; default: 96 direct, 1024 extract. "
                         "Per il gate reader-step usare 512 su ENTRAMBE le condizioni "
                         "(controllo flat-direct-512: unica variabile = protocollo)")
    ap.add_argument("--ask-mode", default="direct", choices=["direct", "extract"],
                    help="extract = scaffolding extract-then-answer (FINAL: parsing)")
    ap.add_argument("--evidence-condition", default=None,
                    help="filtra le righe evidence per campo 'condition' "
                         "(default: righe senza condition, i.e. bm25-rich)")
    ap.add_argument("--renders", default="flat",
                    help="comma-separated: flat,structured,structured-now — "
                    "tutte sulle STESSE domande/evidenza (paired)")
    args = ap.parse_args()

    renders = [r.strip() for r in args.renders.split(",") if r.strip()]
    for rn in renders:
        if rn not in RENDERS:
            print(f"FATAL: render sconosciuto {rn}", file=sys.stderr)
            return 2
    key = os.environ.get(args.api_key_env, "")
    if not key:
        print(f"FATAL: env {args.api_key_env} missing", file=sys.stderr)
        return 2

    wanted = set(args.categories.split(","))
    rows = [json.loads(l, strict=False) for l in open(args.evidence, encoding="utf-8")]
    rows = [r for r in rows if r.get("condition") == args.evidence_condition]
    rows = [r for r in rows if not r["question_id"].endswith("_abs")
            and r["question_type"] in wanted and r["evidence"]]
    by_cat = collections.defaultdict(list)
    for r in rows:
        by_cat[r["question_type"]].append(r)
    sample = []
    for cat, items in by_cat.items():
        random.shuffle(items)
        sample.extend(items[:args.per_category])
    print(f"sample: {len(sample)} domande "
          f"({ {c: min(len(v), args.per_category) for c, v in by_cat.items()} })",
          file=sys.stderr)

    rows_path = args.output + ".rows.jsonl"
    out_rows = open(rows_path, "w", encoding="utf-8")
    scored = []
    fails = 0
    t0 = time.time()
    for i, r in enumerate(sample):
        gold = r["answer"] if isinstance(r["answer"], str) else json.dumps(r["answer"])
        # Normalizza evidenza: v1 era lista di stringhe, v2 oggetti con date.
        ev = [{"text": e, "session": None, "date": None} if isinstance(e, str) else e
              for e in r["evidence"]]
        qd = r.get("question_date")
        # Freeze check (parere Codex): il set di testi e' lo stesso per OGNI
        # render per costruzione; lo si asserisce comunque.
        base_set = sorted(x["text"] for x in ev)
        stop = False
        for render_name in renders:
            ctx, now_line = RENDERS[render_name](ev, qd)
            for t in base_set:
                assert t in ctx, f"freeze violato: testo perso nel render {render_name}"
            try:
                final_miss = False
                if args.ask_mode == "extract":
                    raw = ask(args.base_url, key, args.model, ctx, now_line,
                              r["question"], system_prompt=SYSTEM_PROMPT_EXTRACT,
                              max_tokens=args.max_tokens or 1024)
                    # Policy pre-fissata (gate design 2026-06-12): se manca
                    # FINAL: si valuta l'intera predizione e si conta il miss.
                    final_miss = "FINAL:" not in raw
                    pred = parse_final(raw)
                else:
                    pred = ask(args.base_url, key, args.model, ctx, now_line,
                              r["question"], max_tokens=args.max_tokens)
                fails = 0
            except RuntimeError as e:
                fails += 1
                print(f"[{i+1}] {r['question_id']}/{render_name} FAILED: {e}", file=sys.stderr)
                if fails >= 3:
                    print("FATAL: 3 failure consecutivi, stop.", file=sys.stderr)
                    stop = True
                break
            rec = {"question_id": r["question_id"], "category": r["question_type"],
                   "render": render_name, "ask_mode": args.ask_mode,
                   "rendered_chars": len(ctx),
                   "final_miss": final_miss,
                   "gold": gold, "pred": pred, "f1": f1_score(pred, gold)}
            scored.append(rec)
            out_rows.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_rows.flush()
            time.sleep(args.sleep)
        if stop:
            break
        if (i + 1) % 20 == 0:
            mean = sum(x["f1"] for x in scored) / len(scored)
            print(f"[{i+1}/{len(sample)}] F1={mean:.3f} ({time.time()-t0:.0f}s)",
                  file=sys.stderr)
    out_rows.close()

    if not scored:
        return 1
    misses = sum(1 for x in scored if x.get("final_miss"))
    summary = {"harness": "convbench-model-in-loop (proxy deterministic-ingest)",
               "model": args.model, "n": len(scored),
               "ask_mode": args.ask_mode, "max_tokens": args.max_tokens,
               "final_parse_miss": misses,
               "final_parse_miss_rate": round(misses / len(scored), 4),
               "per_category": {}}
    for cat in sorted({x["category"] for x in scored}):
        summary["per_category"][cat] = {}
        for rn in sorted({x.get("render", "flat") for x in scored}):
            xs = [x["f1"] for x in scored
                  if x["category"] == cat and x.get("render", "flat") == rn]
            if xs:
                summary["per_category"][cat][rn] = {"n": len(xs),
                                                    "f1": round(sum(xs) / len(xs), 4)}
    json.dump(summary, open(args.output, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
