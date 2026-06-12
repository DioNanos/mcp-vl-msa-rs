# mcp-vl-msa-rs

MSA-flavor retrieval/memory engine, exposed as an MCP stdio server. Pure Rust,
BM25 over [tantivy](https://github.com/quickwit-oss/tantivy), zero ML deps in the
default build; optional in-process dense rerank. Vivling-first (the long-term
memory for `codex-vl` Vivling) but a reusable, AI-agnostic memory/RAG server.

The name: `vl` ties it to the Vivling / codex-vl family (its first consumer);
`msa` is the retrieval pattern it borrows from the Memory Sparse Attention paper
(arXiv:2603.23516) — an *extrinsic* approximation, not the neural model. Distinct
from MiniMax's MSA-architecture LLMs, which are *intrinsic* (in-model) generators.

**Status**: v0.4 — hybrid sparse+dense optional.

## Why

The original [Memory Sparse Attention](https://arxiv.org/abs/2603.23516) paper (EverMind-AI) describes an end-to-end trainable sparse attention layer over chunk-pooled KV caches. That is a neural artifact and is not portable to a pure-Rust MCP server. What *is* portable, and what this repo aims to deliver, is the MSA macro pattern:

1. **Chunked storage** of long-form text with a small fixed pool size (`P=64` words by default, mirroring the paper).
2. **Top-k sparse routing** over chunks (BM25 surrogate; learned routing is out of scope).
3. **Original text injection** (paper §4.3, ablation -37.1% without): `msa_search` returns chunks, `msa_fetch_doc` returns the full document.
4. **Memory Interleave** as a *protocol* (planned v0.4): the AI client orchestrates multi-hop retrieval through repeated tool calls with a server-side cursor.

Design and rationale are documented in the project notes (negative results, gate methodology); see [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md).

## Tool surface

| Tool | Since | Description |
|---|---|---|
| `msa_index` | v0.1 | Index a document; existing chunks for `doc_id` are replaced. |
| `msa_search` | v0.1 | Top-k chunks, score normalized 0.0–1.0. |
| `msa_fetch_doc` | v0.1 | Full original text of a document. |
| `msa_delete` | v0.1 | Remove a document and all its chunks. |
| `msa_list_collections` | v0.1 | Collections open in the registry. |
| `msa_stats` | v0.1 | Per-collection statistics (exact `num_documents` / `total_tokens`). |
| `SearchFilter` | v0.2 | Metadata filter (`where_eq`/`where_in`/`created_*`), post-retrieval. |
| `msa_search_iterative` | v0.3 | Memory Interleave with server-side cursor; dedups across rounds. |
| `msa_drop_session` | v0.3 | Force-evict a Memory Interleave session before TTL. |
| `dense_alpha` on `msa_search` | v0.4 | Hybrid BM25 + cosine rerank. Requires `--features embeddings` + `[embeddings]` config. |
| `msa_remember` / `msa_forget` | v0.4 | Agent-memory surface: enrich + low-signal gate + content-hash dedup; standard metadata (`kind` / `source_id` / `created_at`). |
| `msa_sync_path` | v0.4 | Mirror a directory into a collection (filesystem source; blake3 delta sync). |

## Install

**Prebuilt binary** (recommended) — download the archive for your platform from
the [latest release](https://github.com/DioNanos/mcp-vl-msa-rs/releases/latest),
extract, and point your MCP client at the binary:

```bash
tar xzf mcp-vl-msa-rs-x86_64-unknown-linux-gnu.tar.gz
install -m755 mcp-vl-msa-rs-*/mcp-vl-msa-rs ~/.local/bin/
```

Prebuilt targets (Linux + Android): `x86_64-unknown-linux-gnu`,
`x86_64-unknown-linux-musl`, `aarch64-unknown-linux-gnu`,
`aarch64-unknown-linux-musl` (edge / ARM / Termux), `aarch64-linux-android`.

**macOS**: no prebuilt binary is shipped (it would need Apple code-signing).
Install from source instead — `cargo install` below compiles it on your Mac in
one command, no signing needed.

**From source** (Rust toolchain) — `--locked` is required (the workspace
`Cargo.lock` pins a working `time` / `tantivy-common` resolution; a fresh
resolve breaks the build), and `mcp-msa-server` is the package name (the
binary it installs is `mcp-vl-msa-rs`):

```bash
cargo install --git https://github.com/DioNanos/mcp-vl-msa-rs \
  --locked --features source-fs mcp-msa-server
```

**Nix** — the server is also packaged as a NixOS module in the
[Lararium](https://github.com/DioNanos) hosting stack (for user-owned
assistant nodes). _Coming with the Lararium flake._

## Build & test

```bash
cd mcp-vl-msa-rs

# Default: pure BM25, zero network deps
cargo build --release
cargo test

# Hybrid sparse + dense (Ollama-compatible embedding service required)
cargo build --release --features embeddings
cargo test  --features embeddings
```

### Hybrid mode config

Add `[embeddings]` to `MCP_MSA_CONFIG` to activate dense rerank. Without
this section the server stays in BM25-only mode even when the binary was
built with `--features embeddings`.

```toml
[storage]
storage_dir = "~/.local/state/mcp-vl-msa-rs"

[chunking]
chunk_size = 64
overlap = 0

[embeddings]
backend = "ollama"
url     = "http://127.0.0.1:11434"
model   = "nomic-embed-text"
dim     = 768
```

The AI client opts into hybrid scoring per-call by passing `dense_alpha`
to `msa_search` (or any future tool that supports it). `dense_alpha = 1.0`
(default) is BM25-only; `0.0` is dense-only; intermediate values are a
linear blend `α·bm25 + (1-α)·((cos+1)/2)`. Cosine is shifted to `[0,1]`
so it composes linearly with the already max-normalized BM25 score.

## Run as MCP stdio

```bash
# Default storage: ~/.local/state/mcp-vl-msa-rs/
./target/release/mcp-vl-msa-rs

# With explicit config
MCP_VL_MSA_CONFIG=~/.config/mcp-vl-msa-rs/config.toml \
MCP_DEVICE=my-node \
./target/release/mcp-vl-msa-rs
```

Example `~/.codex/config.toml` entry:

```toml
[mcp_servers.vl_msa]
command = "/path/to/mcp-vl-msa-rs/target/release/mcp-vl-msa-rs"
env = { MCP_DEVICE = "my-node" }
```

## Storage layout

```
~/.local/state/mcp-vl-msa-rs/
├── <collection_a>/        ← tantivy index directory
├── <collection_b>/
└── ...
```

Each collection is an independent tantivy index. Collection names are validated
(rejected if they contain path separators, `..`, etc.) so a collection cannot
escape the root.

## Roadmap

Shipped:

- **v0.2** — `SearchFilter` (where_eq / where_in / created range), post-retrieval.
- **v0.3** — `msa_search_iterative` Memory Interleave with server-side cursor + TTL'd `MsaSession` registry.
- **v0.4** — hybrid BM25 + dense rerank behind feature flag `embeddings`, Ollama backend, per-call `dense_alpha`; agent-memory surface (`msa_remember` / `msa_forget`); filesystem source metadata (`created_at` / `source` / `ext` / `dir`) at index time; exact `num_documents` / `total_tokens` in `msa_stats`; `msa-bench` reproducible benchmark crate; prebuilt-binary packaging.

Next (not yet built):

- Query-time tantivy filter (today `SearchFilter` runs post-retrieval; fine for
  normal corpora, but a pre-filter would help when selectivity is high on a very
  large index).
- ACL for multi-tenant collections.
- Tool-description tuning.

## Related work

- **MSA paper** ([arXiv:2603.23516](https://arxiv.org/abs/2603.23516)) — the
  architectural inspiration (neural, intrinsic); this repo is an extrinsic,
  pure-Rust approximation of the macro pattern.
- **Vivling** (in `codex-vl`) — the first downstream consumer: this server is
  its long-term memory.

## License

Apache-2.0. See [LICENSE](./LICENSE).
