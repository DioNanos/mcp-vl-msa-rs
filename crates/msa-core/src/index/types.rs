//! Public types for the index layer: fingerprints, manifest pagination, delta
//! batch options and outcomes.

use serde::Serialize;

/// A6 fingerprint of one document, read from its sentinel. `content_hash`
/// is `None` only on a legacy doc never reindexed under A6 (`hash_state` =
/// legacy_missing, computed by the manifest layer). See [`super::MsaIndex::doc_fingerprint`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DocFingerprint {
    pub content_hash: Option<String>,
    pub index_profile_hash: Option<String>,
    pub source_id: Option<String>,
}

/// Freshness of a document relative to the collection's *current* index
/// profile, reported by [`super::MsaIndex::manifest`] so a client knows what to
/// resend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HashState {
    /// content_hash present and index_profile_hash matches the current profile.
    Ok,
    /// No content_hash — a legacy doc never reindexed under A6 (resend it).
    LegacyMissing,
    /// content_hash present but indexed under a different chunking/embedding
    /// profile — its chunks/embeddings are stale (resend it).
    ProfileStale,
}

/// Classify a document's freshness against the current index-profile hash.
pub(super) fn classify_hash_state(fp: &DocFingerprint, current_profile_hash: &str) -> HashState {
    if fp.content_hash.is_none() {
        HashState::LegacyMissing
    } else if fp.index_profile_hash.as_deref() != Some(current_profile_hash) {
        HashState::ProfileStale
    } else {
        HashState::Ok
    }
}

/// One manifest row: a document's id + fingerprint + freshness.
#[derive(Debug, Clone, Serialize)]
pub struct ManifestEntry {
    pub doc_id: String,
    pub content_hash: Option<String>,
    pub index_profile_hash: Option<String>,
    pub source_id: Option<String>,
    pub hash_state: HashState,
}

/// A page of [`super::MsaIndex::manifest`]. `total_count` and `manifest_digest` cover
/// the whole filtered scope (not just this page), so a client can diff cheaply
/// and pass `manifest_digest` back as an optimistic-concurrency precondition.
#[derive(Debug, Clone, Serialize)]
pub struct Manifest {
    pub total_count: usize,
    pub manifest_digest: String,
    pub next_cursor: Option<String>,
    pub entries: Vec<ManifestEntry>,
}

/// Hard ceiling for a single manifest page (Codex design audit §7).
pub const MAX_MANIFEST_LIMIT: usize = 5000;
/// Default manifest page size when the caller does not specify one.
pub const DEFAULT_MANIFEST_LIMIT: usize = 1000;

/// Per-batch ceilings for [`super::MsaIndex::apply_delta`] (sourced from config).
#[derive(Debug, Clone, Copy)]
pub struct BatchCaps {
    pub max_docs: usize,
    pub max_text_bytes: usize,
}

/// Options for [`super::MsaIndex::apply_delta`] (Codex design audit §6).
#[derive(Debug, Clone, Default)]
pub struct DeltaOptions {
    /// Skip an upsert whose content_hash AND index_profile_hash already match
    /// what is indexed (no re-chunk/re-embed).
    pub skip_unchanged: bool,
    /// Tag written into each upserted doc's `_msa` block; also the scope for
    /// the `if_manifest_digest` precondition.
    pub source_id: Option<String>,
    /// Optimistic-concurrency: if set, the scope's current manifest_digest must
    /// equal this or the call fails with `Conflict` and writes nothing.
    pub if_manifest_digest: Option<String>,
    /// If set, the collection's current index_profile_hash must equal this or
    /// the call fails with `Conflict` (chunking/model changed since the client
    /// computed its delta).
    pub if_index_profile_hash: Option<String>,
}

/// Outcome status for one doc in an [`super::MsaIndex::apply_delta`] batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeltaStatus {
    Indexed,
    Unchanged,
    Deleted,
    NotFound,
    /// A delete target whose stored `source_id` differs from the scoped
    /// `source_id`: not deleted, so a sync of one source cannot evict another
    /// source's docs from a shared collection (Codex design audit, delete scope).
    SkippedScopeMismatch,
}

/// Per-doc result of an [`super::MsaIndex::apply_delta`] batch.
#[derive(Debug, Clone, Serialize)]
pub struct DeltaResult {
    pub doc_id: String,
    pub status: DeltaStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_chunks: Option<usize>,
}

/// Outcome of an atomic [`super::MsaIndex::apply_delta`] (upsert + delete) batch.
#[derive(Debug, Clone, Serialize)]
pub struct DeltaOutcome {
    pub indexed: usize,
    pub unchanged: usize,
    pub deleted: usize,
    pub not_found: usize,
    /// Delete targets skipped because their stored `source_id` differs from the
    /// scoped `source_id` (only possible when `source_id` is set).
    pub skipped_scope_mismatch: usize,
    /// The scope's manifest_digest AFTER the commit — the client's next
    /// `if_manifest_digest`.
    pub new_manifest_digest: String,
    pub results: Vec<DeltaResult>,
}

/// Whether an upsert can be skipped: both content and index profile already
/// match what is stored.
pub(super) fn should_skip_upsert(
    stored: Option<&DocFingerprint>,
    content_hash: &str,
    current_profile_hash: &str,
) -> bool {
    match stored {
        Some(fp) => {
            fp.content_hash.as_deref() == Some(content_hash)
                && fp.index_profile_hash.as_deref() == Some(current_profile_hash)
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_hash_state_logic() {
        let legacy = DocFingerprint::default();
        assert_eq!(classify_hash_state(&legacy, "p1"), HashState::LegacyMissing);
        let stale = DocFingerprint {
            content_hash: Some("c".into()),
            index_profile_hash: Some("pOLD".into()),
            source_id: None,
        };
        assert_eq!(classify_hash_state(&stale, "p1"), HashState::ProfileStale);
        let fresh = DocFingerprint {
            content_hash: Some("c".into()),
            index_profile_hash: Some("p1".into()),
            source_id: None,
        };
        assert_eq!(classify_hash_state(&fresh, "p1"), HashState::Ok);
    }

    #[test]
    fn should_skip_upsert_logic() {
        assert!(!should_skip_upsert(None, "c", "p"));
        let fp = DocFingerprint {
            content_hash: Some("c".into()),
            index_profile_hash: Some("p".into()),
            source_id: None,
        };
        assert!(should_skip_upsert(Some(&fp), "c", "p"));
        assert!(!should_skip_upsert(Some(&fp), "c2", "p"));
        assert!(!should_skip_upsert(Some(&fp), "c", "p2"));
    }
}
