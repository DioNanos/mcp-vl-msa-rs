#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-$HOME/.local/bin/mcp-msa-rs}"

cargo build --release --manifest-path "$ROOT/crates/mcp-msa-server/Cargo.toml"
mkdir -p "$(dirname "$DEST")"
install -m 0755 "$ROOT/target/release/mcp-msa-rs" "$DEST"
echo "$DEST"
