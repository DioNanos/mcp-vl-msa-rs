# Releasing

Prebuilt binaries are built in CI for **Linux and Android only**. macOS is not
shipped as a prebuilt binary — macOS users install from source with
`cargo install` (it compiles on their machine, so no Apple code-signing is
involved). This keeps the release pipeline free of signing/notarization.

## 1. Tag → CI builds Linux + Android

Push a `v*` tag. `.github/workflows/release.yml` builds, with
`--locked --features source-fs`, and uploads the assets to the release:

| Target | How |
|---|---|
| `x86_64-unknown-linux-gnu` | native |
| `x86_64-unknown-linux-musl` | native + musl-tools |
| `aarch64-unknown-linux-gnu` | `cross` |
| `aarch64-unknown-linux-musl` | `cross` |
| `aarch64-linux-android` | `cross` (NDK in the cross image) |

Each artifact is `mcp-vl-msa-rs-<target>.tar.gz` + `.sha256`.

```bash
git tag -a v0.4.0 -m "mcp-vl-msa-rs 0.4.0"
git push origin v0.4.0
```

## 2. macOS — no release asset

macOS users run the `cargo install` command in the README. Nothing to build or
sign on our side.

## Notes

- `--locked` is mandatory everywhere: a fresh dependency resolve picks an
  incompatible `time` / `tantivy-common` combination and fails to compile.
- `source-fs` must be enabled or the binary ships without `msa_sync_path`.
- `embeddings` is intentionally **not** in the release binaries (heavy ML dep,
  against the zero-ML-by-default design); build from source with
  `--features embeddings` to use the in-process dense rerank.
