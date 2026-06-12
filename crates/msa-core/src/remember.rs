//! Agent-memory surface: `remember` / `forget` orchestration.
//!
//! This module is a thin, opinionated wrapper over the existing pipeline:
//! enrich sanitization + low-signal gate, generated doc_id (`mem-<uuid4>`),
//! content-hash dedup, standard server-side metadata injection, and finally
//! `index_document` or `delete_document` / `apply_delta`.
//!
//! Zero LLM in the ingestion path — consistent with the zero-ML thesis.
//! The enrichment and gate logic lives in [`crate::enrich`]; this module owns
//! the agent-memory *policy*: what standard metadata to inject, how to dedup,
//! when to honour `force`.

use std::collections::HashMap;

use chrono::Utc;
use serde_json::Value;
use uuid::Uuid;

use crate::enrich::{
    compose_rich_text, is_low_signal_detail, sanitize_detail, INDEX_TEXT_VERSION,
    METADATA_KEY_DETAIL_CHARS, METADATA_KEY_INDEX_TEXT_VERSION, METADATA_KEY_RICH,
    METADATA_KEY_SOURCE,
};
use crate::error::Result;
use crate::fingerprint::{content_hash_v1, reject_reserved_metadata};
use crate::index::MsaIndex;
use crate::schema::Document;
use crate::validate::{validate_doc_id, validate_text, DEFAULT_MAX_TEXT_BYTES};

// ── public constants ──────────────────────────────────────────────────────────

/// Standard metadata key for the memory kind (fact | episode | preference | task).
pub const META_KEY_KIND: &str = "kind";
/// Standard metadata key for the caller-supplied scope tag.
pub const META_KEY_SOURCE_ID: &str = "source_id";
/// Standard metadata key stamped server-side at index time.
pub const META_KEY_CREATED_AT: &str = "created_at";

// ── public enums and types ────────────────────────────────────────────────────

/// Allowed memory kind values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryKind {
    Fact,
    Episode,
    Preference,
    Task,
}

impl MemoryKind {
    /// Parse from a caller-supplied string. Unknown values are an error:
    /// silently coercing a typo (`"preferense"`) to `Fact` would mask caller
    /// bugs on an agent-facing surface.
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fact" => Ok(Self::Fact),
            "episode" => Ok(Self::Episode),
            "preference" => Ok(Self::Preference),
            "task" => Ok(Self::Task),
            other => Err(format!(
                "unknown kind '{other}': expected fact | episode | preference | task"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fact => "fact",
            Self::Episode => "episode",
            Self::Preference => "preference",
            Self::Task => "task",
        }
    }
}

/// Outcome returned by [`remember`].
#[derive(Debug, Clone)]
pub struct RememberOutcome {
    pub doc_id: String,
    /// The enriched text that was (or would have been) indexed.
    pub capsule_text: String,
    /// True when the input text already existed in the collection (content-hash dedup).
    pub deduplicated: bool,
    /// If deduplicated, the existing doc_id that holds the same content.
    pub existing_doc_id: Option<String>,
    /// True when the low-signal gate was bypassed via `force: true`.
    pub forced: bool,
    /// Set when the gate rejected the input (no indexing done).
    pub low_signal_reason: Option<String>,
}

/// Outcome returned by [`forget_by_source_id`].
#[derive(Debug, Clone)]
pub struct ForgetBatchOutcome {
    pub deleted: usize,
}

// ── public entry points ───────────────────────────────────────────────────────

/// Index a memory capsule.
///
/// # Parameters
/// - `idx`           — the open collection to write into
/// - `text`          — the raw fact/episode to remember
/// - `kind`          — memory kind (default `fact`)
/// - `source_id`     — optional scope tag for later batch forget
/// - `explicit_doc_id` — caller-supplied doc_id (update path); if `None`, a `mem-<uuid4>` id is generated
/// - `user_meta`     — free-form caller metadata; merged after standard keys (cannot overwrite them)
/// - `force`         — bypass the low-signal gate when `true`
pub fn remember(
    idx: &MsaIndex,
    text: &str,
    kind: MemoryKind,
    source_id: Option<&str>,
    explicit_doc_id: Option<&str>,
    user_meta: Option<HashMap<String, Value>>,
    force: bool,
) -> Result<RememberOutcome> {
    // ── 1. Basic input validation ─────────────────────────────────────────────
    validate_text(text, DEFAULT_MAX_TEXT_BYTES)?;

    let user_meta = user_meta.unwrap_or_default();
    reject_reserved_metadata(&user_meta)?;

    // ── 2. Sanitize + low-signal gate ─────────────────────────────────────────
    let sanitized = sanitize_detail(text);
    if is_low_signal_detail(&sanitized) && !force {
        let reason = low_signal_reason(text);
        return Ok(RememberOutcome {
            doc_id: String::new(),
            capsule_text: String::new(),
            deduplicated: false,
            existing_doc_id: None,
            forced: false,
            low_signal_reason: Some(reason),
        });
    }

    // ── 3. Compose rich capsule text ──────────────────────────────────────────
    let composed = compose_rich_text(text, &sanitized, 800);
    let capsule_text = composed.text.clone();

    // ── 4. Content-hash dedup (text-only; metadata is caller-supplied) ────────
    // The fingerprint covers only the original `text` (pre-sanitization) with an
    // empty metadata map — standard metadata (created_at, kind, …) is
    // server-assigned at each write, so including it would break dedup for
    // identical texts submitted at different times.
    let empty_meta: HashMap<String, Value> = HashMap::new();
    let content_hash = content_hash_v1(text, &empty_meta);

    if explicit_doc_id.is_none() {
        // Scan the manifest for an existing doc with the same content_hash.
        // Dedup is best-effort (O(n) manifest scan); not a hard guarantee
        // across concurrent writes.
        if let Some(existing_id) = find_by_content_hash(idx, &content_hash)? {
            return Ok(RememberOutcome {
                doc_id: existing_id.clone(),
                capsule_text,
                deduplicated: true,
                existing_doc_id: Some(existing_id),
                forced: false,
                low_signal_reason: None,
            });
        }
    }

    // ── 5. Assign doc_id (generated when not supplied) ───────────────────────
    let doc_id = match explicit_doc_id {
        Some(id) => {
            validate_doc_id(id)?;
            id.to_string()
        }
        None => generate_doc_id(),
    };

    // ── 6. Build standard metadata (standard keys win over user-supplied) ─────
    let created_at = Utc::now();
    let mut metadata: HashMap<String, Value> = user_meta;
    // Standard keys overwrite whatever the caller supplied.
    metadata.insert(
        META_KEY_KIND.to_string(),
        Value::String(kind.as_str().to_string()),
    );
    metadata.insert(
        META_KEY_CREATED_AT.to_string(),
        Value::String(created_at.to_rfc3339()),
    );
    metadata.insert(
        METADATA_KEY_SOURCE.to_string(),
        Value::String("agent".to_string()),
    );
    metadata.insert(
        METADATA_KEY_INDEX_TEXT_VERSION.to_string(),
        Value::from(INDEX_TEXT_VERSION),
    );
    metadata.insert(METADATA_KEY_RICH.to_string(), Value::Bool(true));
    metadata.insert(
        METADATA_KEY_DETAIL_CHARS.to_string(),
        Value::from(composed.detail_chars as u64),
    );
    // Store the raw-text content hash for dedup on future remember calls.
    metadata.insert(
        "_raw_content_hash".to_string(),
        Value::String(content_hash.clone()),
    );
    if let Some(sid) = source_id {
        metadata.insert(
            META_KEY_SOURCE_ID.to_string(),
            Value::String(sid.to_string()),
        );
    }

    // ── 7. Log forced bypass for audit ───────────────────────────────────────
    if force && is_low_signal_detail(&sanitized) {
        let reason = low_signal_reason(text);
        tracing::warn!(
            collection = idx.name(),
            doc_id = %doc_id,
            low_signal_reason = %reason,
            "msa_remember: low-signal gate bypassed via force=true"
        );
    }

    // ── 8. Index via apply_delta so the `source_id` tantivy field is set ─────
    // `index_document` always passes source_id=None to `stage_document`, which
    // stores an empty string in the tantivy field and breaks manifest filters.
    // `apply_delta` is the only public path that propagates source_id correctly.
    let doc = Document {
        id: doc_id.clone(),
        text: capsule_text.clone(),
        metadata,
        created_at,
    };
    let opts = crate::index::DeltaOptions {
        skip_unchanged: false, // always index (we already handled dedup above)
        source_id: source_id.map(str::to_string),
        if_manifest_digest: None,
        if_index_profile_hash: None,
    };
    let caps = crate::index::BatchCaps {
        max_docs: 2,
        max_text_bytes: usize::MAX,
    };
    idx.apply_delta(&[doc], &[], None, &opts, caps)?;

    Ok(RememberOutcome {
        doc_id,
        capsule_text,
        deduplicated: false,
        existing_doc_id: None,
        forced: force && is_low_signal_detail(&sanitized),
        low_signal_reason: None,
    })
}

/// Delete a batch of documents that share the given `source_id`.
///
/// Returns the number of documents deleted. Uses the manifest to enumerate
/// docs in the collection scoped to `source_id`, then issues an `apply_delta`
/// delete. Not atomic across the manifest read and the delete commit, but
/// safe under concurrent writes (a newly-indexed doc is left untouched).
pub fn forget_by_source_id(idx: &MsaIndex, source_id: &str) -> Result<ForgetBatchOutcome> {
    // Collect all doc_ids for this source via a full manifest scan.
    // We page through the whole scope in one pass (limit = MAX_MANIFEST_LIMIT).
    let profile_hash = idx.current_index_profile_hash(None);
    let manifest = idx.manifest(&profile_hash, Some(source_id), None, 5000, None)?;
    let doc_ids: Vec<String> = manifest.entries.into_iter().map(|e| e.doc_id).collect();
    let count = doc_ids.len();
    if count == 0 {
        return Ok(ForgetBatchOutcome { deleted: 0 });
    }

    let delete_refs: Vec<&str> = doc_ids.iter().map(|s| s.as_str()).collect();
    let opts = crate::index::DeltaOptions {
        skip_unchanged: false,
        source_id: Some(source_id.to_string()),
        if_manifest_digest: None,
        if_index_profile_hash: None,
    };
    let caps = crate::index::BatchCaps {
        max_docs: count + 1,
        max_text_bytes: usize::MAX,
    };
    let outcome = idx.apply_delta(&[], &delete_refs, None, &opts, caps)?;
    Ok(ForgetBatchOutcome {
        deleted: outcome.deleted,
    })
}

// ── private helpers ───────────────────────────────────────────────────────────

/// Generate a unique doc_id using `uuid` v4 (already in the workspace) with a
/// `mem-` prefix for human legibility, instead of adding a new `ulid`
/// dependency. Not time-sortable; sortability is not a requirement here —
/// `created_at` metadata covers temporal queries.
///
/// Format: `mem-<uuid_v4_hyphen_stripped>` — 4 + 1 + 32 = 37 chars, well
/// within the 512-byte `MAX_DOC_ID_LEN`.
fn generate_doc_id() -> String {
    format!("mem-{}", Uuid::new_v4().simple())
}

/// Return a human-readable reason string explaining why the gate rejected the input.
fn low_signal_reason(text: &str) -> String {
    let trimmed = text.trim();
    let len = trimmed.chars().count();
    if len < 80 {
        return format!("text too short ({len} chars; minimum 80)");
    }
    let alnum = trimmed.chars().filter(|c| c.is_alphanumeric()).count();
    if (alnum as f32) / (len as f32) < 0.4 {
        return "alphanumeric density below 40% (likely noise or special-character spam)"
            .to_string();
    }
    "repeated identical lines (no new information)".to_string()
}

/// O(n) linear scan of the manifest to find a doc with a matching `content_hash`.
///
/// The content_hash stored in the manifest was computed over the *enriched*
/// text (capsule) + standard metadata at index time — NOT the same hash we
/// compute here (which covers only the raw `text` with an empty metadata map).
/// Therefore we store the raw-text hash in a dedicated metadata field
/// `_raw_text_hash` and compare against that instead.
///
/// Actually: we store `_raw_content_hash` in user metadata at index time
/// (see `remember`), then scan for it here.  This avoids the mismatch.
///
/// Simplification: instead of a dedicated stored field (requires schema change),
/// we fetch each doc's stored metadata and check.  Acceptable for the dedup
/// path which only runs on the first `remember` of a given text.
fn find_by_content_hash(idx: &MsaIndex, content_hash: &str) -> Result<Option<String>> {
    let profile_hash = idx.current_index_profile_hash(None);
    // Page through all docs in the collection.
    let manifest = idx.manifest(&profile_hash, None, None, 5000, None)?;
    for entry in manifest.entries {
        // Attempt to fetch the doc and check its stored `_raw_content_hash`.
        if let Ok(doc) = idx.fetch_doc(&entry.doc_id) {
            if let Some(Value::String(stored_hash)) = doc.metadata.get("_raw_content_hash") {
                if stored_hash == content_hash {
                    return Ok(Some(entry.doc_id));
                }
            }
        }
    }
    Ok(None)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MsaConfig;
    use crate::index::CollectionRegistry;
    use crate::schema::ChunkConfig;

    fn fresh_index(name: &str) -> (tempfile::TempDir, std::sync::Arc<MsaIndex>) {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = MsaConfig {
            storage: crate::config::StorageConfig {
                storage_dir: dir.path().to_path_buf(),
            },
            chunking: ChunkConfig {
                chunk_size: 32,
                overlap: 0,
            },
            ..Default::default()
        };
        let registry = CollectionRegistry::new();
        let idx = registry
            .open_or_create(name, &config.storage.storage_dir, &config.chunking)
            .expect("open collection");
        (dir, idx)
    }

    fn long_text() -> &'static str {
        "The reconnect handler in retry_backoff.rs now respects the max_attempts \
         limit and logs a warning via tracing when the backoff ceiling is hit; \
         this prevents infinite retry loops under transient network partitions."
    }

    #[test]
    fn gate_rejects_short_text() {
        let (_d, idx) = fresh_index("grs");
        let out =
            remember(&idx, "ok", MemoryKind::Fact, None, None, None, false).expect("no error");
        assert!(
            out.low_signal_reason.is_some(),
            "short text must be rejected"
        );
        assert!(!out.deduplicated);
    }

    #[test]
    fn gate_force_bypass_indexes_low_signal() {
        let (_d, idx) = fresh_index("gfb");
        let out = remember(&idx, "ok", MemoryKind::Fact, None, None, None, true)
            .expect("force must succeed");
        assert!(out.low_signal_reason.is_none());
        assert!(!out.doc_id.is_empty());
    }

    #[test]
    fn dedup_returns_existing_doc_id() {
        let (_d, idx) = fresh_index("ded");
        let text = long_text();
        let out1 = remember(&idx, text, MemoryKind::Fact, None, None, None, false)
            .expect("first remember ok");
        assert!(!out1.deduplicated);
        let id1 = out1.doc_id.clone();

        // Second remember with the same text: must dedup.
        let out2 = remember(&idx, text, MemoryKind::Fact, None, None, None, false)
            .expect("second remember ok");
        assert!(out2.deduplicated, "identical text must dedup");
        assert_eq!(out2.existing_doc_id.as_deref(), Some(id1.as_str()));
    }

    #[test]
    fn standard_metadata_set_and_user_meta_merged() {
        let (_d, idx) = fresh_index("meta");
        let mut user = HashMap::new();
        user.insert("project".to_string(), Value::String("vl".to_string()));
        // Standard key in user_meta must be overwritten by server.
        user.insert(
            META_KEY_KIND.to_string(),
            Value::String("episode".to_string()),
        );

        let out = remember(
            &idx,
            long_text(),
            MemoryKind::Task,
            Some("sid-1"),
            None,
            Some(user),
            false,
        )
        .expect("remember ok");

        let doc = idx.fetch_doc(&out.doc_id).expect("fetch ok");
        // Standard keys correctly set.
        assert_eq!(
            doc.metadata.get(META_KEY_KIND),
            Some(&Value::String("task".to_string()))
        );
        assert_eq!(
            doc.metadata.get(META_KEY_SOURCE_ID),
            Some(&Value::String("sid-1".to_string()))
        );
        assert_eq!(
            doc.metadata.get(METADATA_KEY_SOURCE),
            Some(&Value::String("agent".to_string()))
        );
        // Custom user key preserved.
        assert_eq!(
            doc.metadata.get("project"),
            Some(&Value::String("vl".to_string()))
        );
        // created_at is a non-empty string.
        assert!(doc
            .metadata
            .get(META_KEY_CREATED_AT)
            .and_then(|v| v.as_str())
            .is_some());
    }

    #[test]
    fn forget_by_source_id_deletes_all_in_scope() {
        let (_d, idx) = fresh_index("fgt");
        for i in 0..3usize {
            let text = format!(
                "Session memory item {i}: the retry handler now respects the backoff ceiling \
                 and emits a tracing warning at level warn when max_attempts is reached."
            );
            remember(
                &idx,
                &text,
                MemoryKind::Fact,
                Some("sess-A"),
                None,
                None,
                false,
            )
            .expect("remember ok");
        }
        // Index one doc with a different source_id.
        remember(
            &idx,
            long_text(),
            MemoryKind::Fact,
            Some("sess-B"),
            None,
            None,
            false,
        )
        .expect("remember B ok");

        let out = forget_by_source_id(&idx, "sess-A").expect("forget ok");
        assert_eq!(out.deleted, 3, "all 3 sess-A docs must be deleted");

        // sess-B doc must still be present.
        let profile = idx.current_index_profile_hash(None);
        let m = idx
            .manifest(&profile, Some("sess-B"), None, 100, None)
            .expect("manifest ok");
        assert_eq!(m.total_count, 1, "sess-B doc must survive");
    }

    #[test]
    fn explicit_doc_id_skips_dedup_and_updates() {
        let (_d, idx) = fresh_index("upd");
        let text = long_text();
        let out1 = remember(
            &idx,
            text,
            MemoryKind::Fact,
            None,
            Some("custom-id"),
            None,
            false,
        )
        .expect("remember ok");
        assert_eq!(out1.doc_id, "custom-id");
        assert!(!out1.deduplicated);

        // Re-remember with same explicit id (update).
        let out2 = remember(
            &idx,
            text,
            MemoryKind::Episode,
            None,
            Some("custom-id"),
            None,
            false,
        )
        .expect("update ok");
        assert_eq!(out2.doc_id, "custom-id");
        assert!(!out2.deduplicated, "explicit doc_id path skips dedup scan");
    }

    #[test]
    fn ulid_format_is_valid_doc_id() {
        let id = generate_doc_id();
        crate::validate::validate_doc_id(&id).expect("generated id must be valid");
        assert!(id.starts_with("mem-"), "must have mem- prefix");
    }
}
