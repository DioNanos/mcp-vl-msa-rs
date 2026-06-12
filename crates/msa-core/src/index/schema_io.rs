//! Tantivy field handle set, on-disk manifest, stored-doc accessors, and
//! post-retrieval filter logic. No index-level business logic lives here.

use std::collections::HashMap;
use std::path::PathBuf;

use tantivy::schema::{Field, OwnedValue, STORED, STRING, TEXT};
use tantivy::TantivyDocument;

use crate::error::{MsaError, Result};
use crate::schema::SearchFilter;

// ── on-disk schema version ────────────────────────────────────────────────────

/// On-disk schema version. Bump when the tantivy schema changes in a way that
/// is not transparently backward-compatible. Recorded in `manifest.json` per
/// collection; `open` refuses an index written by a *newer* version rather
/// than reading it with the wrong field layout.
pub(super) const SCHEMA_VERSION: u32 = 3;

/// Sentinel value in `chunk_idx` that marks the full-document record.
pub(super) const SENTINEL_CHUNK_IDX: u64 = u64::MAX;

// ── per-collection manifest ───────────────────────────────────────────────────

/// Per-collection manifest sitting next to the tantivy index. Small on purpose:
/// it records the schema version so a future, incompatible layout is detected
/// instead of silently misread.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct CollectionManifest {
    pub schema_version: u32,
}

impl CollectionManifest {
    pub fn path(dir: &std::path::Path) -> PathBuf {
        dir.join("manifest.json")
    }

    /// Read the manifest. `Ok(None)` means the file is absent — a legacy
    /// pre-A1 collection, which is fine. `Err` means the file exists but is
    /// unreadable or not valid JSON: that is corruption, not legacy, and must
    /// surface instead of being silently downgraded to "no manifest".
    pub fn read(dir: &std::path::Path) -> Result<Option<Self>> {
        let path = Self::path(dir);
        match std::fs::read_to_string(&path) {
            Ok(raw) => {
                let m = serde_json::from_str(&raw).map_err(|e| {
                    MsaError::Schema(format!("manifest {} is malformed: {e}", path.display()))
                })?;
                Ok(Some(m))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(MsaError::Io(e)),
        }
    }

    pub fn write(dir: &std::path::Path, version: u32) -> Result<()> {
        let m = CollectionManifest {
            schema_version: version,
        };
        std::fs::write(Self::path(dir), serde_json::to_string_pretty(&m)?)?;
        Ok(())
    }
}

// ── field handles ─────────────────────────────────────────────────────────────

/// Resolved field handles for one collection's tantivy schema. `token_count`,
/// `content_hash`, `index_profile_hash`, and `source_id` are `None` on older
/// on-disk schemas that pre-date their introduction.
#[derive(Clone)]
pub(super) struct Fields {
    pub doc_id: Field,
    pub chunk_idx: Field,
    pub is_sentinel: Field,
    pub chunk_text: Field,
    pub full_text: Field,
    pub char_start: Field,
    pub char_end: Field,
    pub created_at: Field,
    pub metadata: Field,
    /// Always present in the schema (forward-compat). Holds packed f32
    /// little-endian bytes when the dense scorer was active at index time;
    /// empty otherwise. Search-time hybrid scoring tolerates missing
    /// embeddings by falling back to BM25.
    pub embedding: Field,
    /// Per-chunk token count (word-level, matching the chunker). `None` for
    /// indices created before SCHEMA_VERSION 2, so `stats()` degrades to
    /// reporting 0 total tokens rather than misreading a missing field.
    pub token_count: Option<Field>,
    /// A6 fingerprint fields (SCHEMA_VERSION 3), written on the sentinel only.
    /// `None` for older on-disk schemas (2): the same values are always also
    /// written into the sentinel's `_msa` metadata block, which IS the
    /// compat/source-of-truth layer — these typed fields are an accelerator for
    /// new collections. Tantivy cannot add fields to an existing on-disk
    /// schema, so a legacy collection self-heals via `_msa` metadata on
    /// reindex without a physical rebuild (Codex design audit §8).
    pub content_hash: Option<Field>,
    pub index_profile_hash: Option<Field>,
    pub source_id: Option<Field>,
}

/// Build the canonical schema for a freshly-created collection.
pub(super) fn build_schema() -> tantivy::schema::Schema {
    let mut sb = tantivy::schema::Schema::builder();
    sb.add_text_field("doc_id", STRING | STORED);
    sb.add_u64_field("chunk_idx", tantivy::schema::FAST | STORED);
    sb.add_u64_field("is_sentinel", tantivy::schema::FAST | STORED);
    sb.add_text_field("chunk_text", TEXT | STORED);
    sb.add_text_field("full_text", STORED);
    sb.add_u64_field("char_start", tantivy::schema::FAST | STORED);
    sb.add_u64_field("char_end", tantivy::schema::FAST | STORED);
    sb.add_i64_field("created_at", tantivy::schema::FAST | STORED);
    sb.add_text_field("metadata", STORED);
    sb.add_bytes_field("embedding", STORED);
    sb.add_u64_field("token_count", tantivy::schema::FAST | STORED);
    // A6 (SCHEMA_VERSION 3): fingerprint fields on the sentinel. STRING so
    // `source_id` could be term-queried later; STORED so manifest reads them.
    sb.add_text_field("content_hash", STRING | STORED);
    sb.add_text_field("index_profile_hash", STRING | STORED);
    sb.add_text_field("source_id", STRING | STORED);
    sb.build()
}

// ── stored-doc field extractors ───────────────────────────────────────────────

pub(super) fn first_text(doc: &TantivyDocument, field: Field) -> Option<String> {
    match doc.get_first(field)? {
        OwnedValue::Str(s) => Some(s.clone()),
        _ => None,
    }
}

pub(super) fn first_u64(doc: &TantivyDocument, field: Field) -> Option<u64> {
    match doc.get_first(field)? {
        OwnedValue::U64(n) => Some(*n),
        _ => None,
    }
}

pub(super) fn first_i64(doc: &TantivyDocument, field: Field) -> Option<i64> {
    match doc.get_first(field)? {
        OwnedValue::I64(n) => Some(*n),
        _ => None,
    }
}

pub(super) fn first_bytes(doc: &TantivyDocument, field: Field) -> Option<Vec<u8>> {
    match doc.get_first(field)? {
        OwnedValue::Bytes(b) => Some(b.clone()),
        _ => None,
    }
}

// ── post-retrieval filter ─────────────────────────────────────────────────────

/// Apply a `SearchFilter` against a chunk's metadata + `created_at` (unix s).
pub(super) fn filter_matches(
    f: &SearchFilter,
    metadata: &HashMap<String, serde_json::Value>,
    created_unix: i64,
) -> bool {
    for (k, expected) in &f.where_eq {
        match metadata.get(k) {
            Some(actual) if actual == expected => {}
            _ => return false,
        }
    }
    for (k, allowed) in &f.where_in {
        match metadata.get(k) {
            Some(actual) if allowed.iter().any(|v| v == actual) => {}
            _ => return false,
        }
    }
    if let Some(after) = f.created_after {
        if created_unix < after.timestamp() {
            return false;
        }
    }
    if let Some(before) = f.created_before {
        if created_unix > before.timestamp() {
            return false;
        }
    }
    true
}

// ── filesystem helpers ────────────────────────────────────────────────────────

pub(super) fn dir_size(path: &std::path::Path) -> Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_file() {
            total += meta.len();
        } else if meta.is_dir() {
            total += dir_size(&entry.path())?;
        }
    }
    Ok(total)
}
