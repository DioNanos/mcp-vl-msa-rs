#!/usr/bin/env python3
"""Regenerate the multi-family reader-step table in docs/NEGATIVE_RESULTS.md.

Convention (the published numbers are defined by THIS script):
- pair direct/extract rows by question_id, category == multi-session
- delta = mean(extract.f1 - direct.f1)
- paired bootstrap, 1000 resamples, percentile CI at indices 25/975
- RNG: random.seed(11) reset PER READER, so each reader's row is
  independently reproducible regardless of which readers run before it.

Usage: python3 crates/msa-bench/scripts/readerstep_family_cis.py
(paths resolve relative to this script's location — any cwd works)
"""

import json
import random
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

READERS = [
    ("GLM-5.1", "2026-06-12_readerstep_"),
    ("DeepSeek-v4-pro", "2026-06-12_readerstep_deepseek-v4-pro_cloud_"),
    ("Kimi-k2.6", "2026-06-12_readerstep_kimi-k2_6_cloud_"),
    ("MiniMax-m3", "2026-06-12_readerstep_minimax-m3_cloud_"),
]
N_BOOT = 1000


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def boot_ci(deltas):
    n = len(deltas)
    mean = sum(deltas) / n
    samples = sorted(
        sum(deltas[random.randint(0, n - 1)] for _ in range(n)) / n
        for _ in range(N_BOOT)
    )
    return mean, samples[25], samples[975]


def main():
    print(f"{'reader':<17}{'n':>4}{'delta':>9}{'CI95':>18}{'  sens.(no-empty)':>20}")
    for name, base in READERS:
        d = {r["question_id"]: r
             for r in rows(RESULTS / (base + "control_direct512.json.rows.jsonl"))
             if r["category"] == "multi-session"}
        e = {r["question_id"]: r
             for r in rows(RESULTS / (base + "extract512.json.rows.jsonl"))
             if r["category"] == "multi-session"}
        keys = [k for k in d if k in e]

        random.seed(11)  # reset per reader: each row reproducible in isolation
        mean, lo, hi = boot_ci([e[k]["f1"] - d[k]["f1"] for k in keys])

        keys2 = [k for k in keys if d[k]["pred"].strip() and e[k]["pred"].strip()]
        random.seed(11)
        m2, lo2, hi2 = boot_ci([e[k]["f1"] - d[k]["f1"] for k in keys2])

        print(f"{name:<17}{len(keys):>4}{mean:>+9.3f}   [{lo:+.2f},{hi:+.2f}]"
              f"   {m2:>+7.3f} (n={len(keys2)}) [{lo2:+.2f},{hi2:+.2f}]")


if __name__ == "__main__":
    main()
