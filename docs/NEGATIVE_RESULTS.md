# Negative results

Most retrieval projects publish what worked. This page documents what we
measured, what failed its pre-registered gate, and what we shipped instead.
Every experiment below fixed its acceptance thresholds **before** the run;
deltas are paired, with bootstrap confidence intervals (1000 resamples).
Absolute scores are proxies — decisions were made on paired deltas only.

Benchmarks: HotpotQA (extractive QA), MLDR-it (long-doc retrieval, Italian),
LongMemEval-S (conversational memory, 500 questions). Reader model for
model-in-loop gates: GLM-5.1, temperature 0.

## Refuted: dense / hybrid retrieval (three times)

**Claim tested**: adding a dense encoder (hybrid BM25 + cosine rerank) improves
retrieval over plain BM25 on our workloads.

| Run | Workload | Outcome |
|---|---|---|
| Encoder sweep | HotpotQA + MLDR-it | no hybrid configuration beat the +5 gate |
| Model-in-loop | HotpotQA, 200 q | gains within noise |
| Conv-bench C4 | LongMemEval-S | ≤ +0.8 recall@5 on every gap category |

Hybrid scoring stays in the codebase (`dense_alpha`, optional, off by
default) because the result is workload-specific and encoders keep improving —
we expect to re-test periodically. But we ship no embedding model, and BM25 is
not a placeholder: on these workloads it is the engine.

## Refuted: recency prior / time-aware retrieval

**Claim tested**: boosting recent documents (or oracle timestamp banding)
improves temporal-reasoning and knowledge-update questions.

LongMemEval-S, recall@5: recency prior scored **−1.4** on knowledge-update —
its own target category. Even an *oracle* timestamp band (cheating: the true
question date window) gained only +2.4/+2.5 against a +5 gate. Time does not
belong in the retrieval score; it belongs in the serving layer (see below).

## Refuted: write-time consolidation

**Claim tested**: distilling cross-session digests at write time (topic-driven
consolidation, LLM-assisted) closes the multi-session aggregation gap.
Pre-registered blocker gate: **≥ +20 F1** on multi-session.

Result: **+2.9** [−1.4, +8.0] — an order of magnitude short — and the guard
was violated: knowledge-update regressed (digests aggregate stale values and
hide the latest update from the reader). Penetration diagnosis: digests
reached the top-10 in only 17% of multi-session questions; specific capsule
wording beats summaries under BM25 on pointed queries.

Lesson: *which* facts need aggregating is decided by the question, not by the
write. The consolidation module remains in the code (explicit use, no runtime
trigger) — measured as neutral when no digests exist.

## Partially confirmed: original-text injection

The MSA paper reports −37.1% without original-text injection (§4.3). On our
corpus the model-in-loop effect is real but concentrated: overall **+1.05 F1**
(below the +5 gate), yet **+14.6 F1** on the stratum where snippets alone
miss the content. Injection stays — `msa_fetch_doc` is cheap insurance for
exactly the cases where snippets fail — but we do not claim the paper's
headline number transfers.

## Confirmed: rich capsules at ingestion

Deterministic capsule composition (`enrich`: rich text assembly, sanitization,
low-signal gate — no LLM involved) is the single most profitable retrieval
move we measured on LongMemEval-S, recall@5:

| Category | short | rich | Δ [CI95] |
|---|---|---|---|
| temporal-reasoning | 84.3 | 93.7 | +9.4 [+3, +16] |
| multi-session | 82.6 | 93.4 | +10.7 [+4, +17] |
| knowledge-update | 93.1 | 100.0 | +6.9 [+1, +12] |
| single-session-user | 78.1 | 98.4 | +20.3 [+11, +31] |
| single-session-assistant | 89.3 | 100.0 | +10.7 [+4, +20] |

## Refuted: reader-step answering scaffolding

**Claim tested**: an extract-then-answer protocol (the reader first enumerates
every relevant fact across the excerpts, then answers) closes the
multi-session reading gap. Pre-registered gate: **≥ +10 F1** on multi-session,
with the token budget held constant across conditions (so the only variable is
the answering protocol, not the output length).

Result on the multi-session target: **−0.03 F1** [paired CI95 −8.1, +7.6] —
centered on zero, a full order of magnitude below the gate. The exploratory
date-serving variant confirmed the intrinsic trade-off below still stands.
Fifth pre-registered refutation of the multi-session line.

## The open problem: reading, not retrieval

With retrieval recall at 93–100% and evidence coverage at 88%, downstream
answer quality still splits: F1 ≈ 0.85 single-session vs ≈ 0.48 multi-session
and ≈ 0.35 temporal. Serving dates to the reader recovers +11..+17 on
temporal/knowledge-update but consistently hurts multi-session aggregation
(−4..−8): the trade-off is intrinsic to the render, not the order.

Five independent, pre-registered attempts have now failed to move
multi-session aggregation — retrieval, recency, hybrid dense, write-time
consolidation, and reader-step scaffolding. The gap is not *how we retrieve*
nor *how we ask the model to answer* over flat evidence; it is plausibly a
problem of how the cross-session evidence is *represented* before the reader.
That is the open question — it needs a new hypothesis, not another serving
variant.

If you are evaluating memory systems: ask vendors for their multi-session
*reading* numbers, not their retrieval recall.
