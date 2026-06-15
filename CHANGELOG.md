# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.4.2]

Packaging and docs. No functional or API changes.

### Added

- Status badges (CI, tests, benchmarks) and a reproducible **Benchmarks**
  section in the README, surfacing the pre-registered evaluation in
  `docs/NEGATIVE_RESULTS.md`.

## [0.4.1]

AI-first discoverability: make the server self-explanatory to MCP clients with
partial protocol support.

### Added

- `readOnlyHint` annotations on the non-mutating tools (`msa_search`,
  `msa_fetch_doc`, `msa_stats`, `msa_list_collections`, `msa_manifest`,
  `msa_search_iterative`, `msa_interleave_round`) so gating clients can
  auto-approve them. The two iterative tools also carry `idempotentHint = false`
  (they mutate only an internal TTL session cache, never user data).
- Teaching error on `msa_search`: an empty result against a collection that does
  not exist now carries an additive `hint` field naming the existing
  collections (capped at 20) and `msa_index` as the way to create one. An
  existing-but-empty collection keeps its silent `{"hits":[]}` response.
- Operative `description`s on request-struct fields (e.g. `collection` is
  "created on first msa_index"), emitted into the JSON tool schemas.
- README: `default_tools_approval_mode = "approve"` in the codex config snippet,
  an equivalent Claude Code (`~/.claude.json`) snippet, and an "AI client
  compatibility" section.

### Changed

- Server `instructions` rewritten to lead with the general agent-memory use.
- README reframed to lead with the general agent-memory use rather than the
  `codex-vl` tie.
- Replaced the non-canonical Apache-2.0 license text with the standard one.

## [0.4.0]

- Hybrid BM25 + dense rerank behind the `embeddings` feature, with an Ollama
  backend and a per-call `dense_alpha`.
- Agent-memory surface: `msa_remember` / `msa_forget` (sanitize, low-signal
  gate, dedup, standardized metadata).
- Filesystem source metadata (`created_at` / `source` / `ext` / `dir`) at index
  time; exact `num_documents` / `total_tokens` in `msa_stats`.
- `msa-bench` reproducible benchmark crate; prebuilt-binary packaging.
