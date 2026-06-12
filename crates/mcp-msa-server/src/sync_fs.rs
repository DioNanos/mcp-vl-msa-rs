//! Filesystem source adapter for `msa_sync_path` (A6.5, feature `source-fs`).
//!
//! Mirrors a directory into a collection: walk → glob filter → read UTF-8 →
//! reconcile against the index via the atomic `apply_delta` core (skip
//! unchanged, index new/changed, prune in-scope docs no longer on disk). The
//! engine stays source-agnostic; this is the one place that touches the
//! filesystem, and only when the deployment opts in via config.
//!
//! Safety (Codex design audit §5): refused unless `[sync].enabled`; `path`
//! must canonicalize to under a configured `allowed_root`; symlinked dirs are
//! not followed by default; per-file and per-run size/count caps; `prune`
//! requires a `source_id` and never runs if any file errored during the scan.
//!
//! ## FS metadata (A6.5.1)
//!
//! Each indexed document receives filesystem-derived metadata populated by
//! [`fs_metadata`]:
//! - `created_at`: file mtime as ISO-8601 UTC, also stored on `Document::created_at`
//!   so `SearchFilter::created_after/created_before` works on FS-synced collections.
//! - `source`: always `"fs"`.
//! - `ext`: file extension without leading dot (`"md"`, `"txt"`, `""`).
//! - `dir`: first path component of the relative path (`"projects"` for
//!   `projects/x/y.md`), or `""` when the file sits at the sync root.
//!
//! **Hash note**: `content_hash_v1` covers both text and metadata. Adding these
//! fields changes the content hash of all existing FS-synced documents → a
//! one-time re-index on the next `sync_path` call after upgrading. Text content
//! that is actually unchanged will NOT re-chunk if the metadata is also identical
//! (i.e., mtime, ext, dir are the same). This is intentional and documented.

use std::collections::{HashMap, HashSet};
use std::time::SystemTime;

use chrono::{DateTime, Utc};
use globset::{Glob, GlobSet, GlobSetBuilder};
use rmcp::ErrorData as McpError;
use walkdir::WalkDir;

use msa_core::config::SyncConfig;
use msa_core::error::MsaError;
use msa_core::fingerprint::content_hash_v1;
use msa_core::index::{BatchCaps, DeltaOptions, MsaIndex};
use msa_core::schema::Document;

use crate::server::{user_or_internal, SyncPathParams};

fn invalid(msg: impl Into<String>) -> McpError {
    McpError::invalid_params(msg.into(), None)
}
fn internal(msg: impl Into<String>) -> McpError {
    McpError::internal_error(msg.into(), None)
}

fn build_globset(globs: &[String]) -> Result<GlobSet, McpError> {
    let mut b = GlobSetBuilder::new();
    for g in globs {
        b.add(Glob::new(g).map_err(|e| invalid(format!("bad glob {g:?}: {e}")))?);
    }
    b.build()
        .map_err(|e| internal(format!("globset build: {e}")))
}

// ── FS metadata helpers ───────────────────────────────────────────────────────

/// Extract the `ext` field from a relative path string (no leading dot).
/// Returns `""` when the file has no extension or the name starts with a dot.
fn extract_ext(rel_str: &str) -> &str {
    let filename = rel_str.rsplit('/').next().unwrap_or(rel_str);
    // A leading dot marks a hidden file — its "extension" is the whole name.
    // We treat that as no extension, consistent with Unix conventions.
    if filename.starts_with('.') {
        return "";
    }
    match filename.rfind('.') {
        Some(pos) if pos + 1 < filename.len() => &filename[pos + 1..],
        _ => "",
    }
}

/// Extract the `dir` field: first path component of a relative path, or `""`
/// when the file sits directly at the sync root.
fn extract_dir(rel_str: &str) -> &str {
    match rel_str.find('/') {
        Some(pos) => &rel_str[..pos],
        None => "",
    }
}

/// Derive filesystem metadata from a file's `mtime` and its relative path.
///
/// Returns a `HashMap` with keys `created_at` (RFC3339 UTC), `source` (`"fs"`),
/// `ext`, and `dir`. The `mtime` is also returned as a `DateTime<Utc>` so the
/// caller can set `Document::created_at` consistently.
///
/// When `mtime` is unavailable (e.g., the OS returns an error), `Utc::now()`
/// is used as a fallback.
fn fs_metadata(
    rel_str: &str,
    mtime: Option<SystemTime>,
) -> (HashMap<String, serde_json::Value>, DateTime<Utc>) {
    let created_at: DateTime<Utc> = mtime
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| {
            DateTime::<Utc>::from_timestamp(d.as_secs() as i64, d.subsec_nanos())
                .unwrap_or_else(Utc::now)
        })
        .unwrap_or_else(Utc::now);

    let created_at_str = created_at.to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let ext = extract_ext(rel_str).to_string();
    let dir = extract_dir(rel_str).to_string();

    let mut meta = HashMap::new();
    meta.insert(
        "created_at".to_string(),
        serde_json::Value::String(created_at_str),
    );
    meta.insert(
        "source".to_string(),
        serde_json::Value::String("fs".to_string()),
    );
    meta.insert("ext".to_string(), serde_json::Value::String(ext));
    meta.insert("dir".to_string(), serde_json::Value::String(dir));
    (meta, created_at)
}

// ── Scanned ───────────────────────────────────────────────────────────────────

/// A file picked up by the scan, ready to upsert.
struct Scanned {
    doc_id: String,
    text: String,
    /// Filesystem-derived metadata (`source`, `ext`, `dir`, `created_at` ISO-8601).
    metadata: HashMap<String, serde_json::Value>,
    /// File mtime as `DateTime<Utc>`, used to populate `Document::created_at`
    /// so that `SearchFilter::created_after/created_before` works correctly.
    mtime: DateTime<Utc>,
}

/// Execute `msa_sync_path`. Returns the response JSON value.
pub fn sync_path(
    idx: &MsaIndex,
    cfg: &SyncConfig,
    caps: BatchCaps,
    embedder: Option<&dyn msa_core::embeddings::DenseEncoder>,
    p: &SyncPathParams,
) -> Result<serde_json::Value, McpError> {
    if !cfg.enabled {
        return Err(invalid(
            "filesystem sync is disabled ([sync].enabled = false)",
        ));
    }
    if p.prune && p.source_id.is_none() {
        return Err(invalid("prune=true requires source_id (scoped prune only)"));
    }
    if let Some(sid) = &p.source_id {
        msa_core::validate::validate_source_id(sid).map_err(user_or_internal)?;
    }

    // ---- path safety: canonicalize and contain under an allowed_root -------
    let canon =
        std::fs::canonicalize(&p.path).map_err(|e| invalid(format!("path {:?}: {e}", p.path)))?;
    if !canon.is_dir() {
        return Err(invalid(format!("path {:?} is not a directory", p.path)));
    }
    let contained = cfg.allowed_roots.iter().any(|root| {
        std::fs::canonicalize(root)
            .map(|rc| canon.starts_with(&rc))
            .unwrap_or(false)
    });
    if !contained {
        return Err(invalid(
            "path is not under any configured [sync].allowed_root",
        ));
    }

    let globs = if p.globs.is_empty() {
        &cfg.default_globs
    } else {
        &p.globs
    };
    let include = build_globset(globs)?;
    let exclude = build_globset(&p.exclude_globs)?;
    let has_exclude = !p.exclude_globs.is_empty();
    // The request may only TIGHTEN the symlink policy, never widen it: if the
    // config forbids following symlinks, no request can enable it. When the
    // config allows it, a request can still turn it off.
    let follow = cfg.follow_symlinks && p.follow_symlinks.unwrap_or(true);
    let prefix = p.doc_id_prefix.as_deref().unwrap_or("");

    // ---- walk + read --------------------------------------------------------
    let mut scanned: Vec<Scanned> = Vec::new();
    let mut errors: Vec<serde_json::Value> = Vec::new();
    let mut scanned_files = 0usize;

    for entry in WalkDir::new(&canon).follow_links(follow) {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                errors.push(serde_json::json!({ "path": null, "reason": e.to_string() }));
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
        }
        scanned_files += 1;
        if scanned_files > cfg.max_files {
            return Err(invalid(format!(
                "scan exceeded [sync].max_files = {}",
                cfg.max_files
            )));
        }
        let rel = match entry.path().strip_prefix(&canon) {
            Ok(r) => r,
            Err(_) => continue,
        };
        if !include.is_match(rel) || (has_exclude && exclude.is_match(rel)) {
            continue;
        }
        let rel_str = match rel.to_str() {
            Some(s) => s.replace(std::path::MAIN_SEPARATOR, "/"),
            None => {
                errors.push(serde_json::json!({
                    "path": entry.path().to_string_lossy(),
                    "reason": "non-UTF-8 path"
                }));
                continue;
            }
        };
        // Symlink-following defense: WalkDir with follow_links resolves symlinks,
        // so a link inside the root could point OUTSIDE it. Verify the resolved
        // file still lives under the canonical sync root; skip + record escapes.
        // (With follow off, symlinked files are not `is_file()` and never reach
        // here, so this only runs in the dangerous case.)
        if follow {
            match std::fs::canonicalize(entry.path()) {
                Ok(real) if real.starts_with(&canon) => {}
                Ok(_) => {
                    errors.push(serde_json::json!({
                        "path": rel_str,
                        "reason": "resolved path escapes the sync root; skipped"
                    }));
                    continue;
                }
                Err(e) => {
                    errors.push(serde_json::json!({ "path": rel_str, "reason": e.to_string() }));
                    continue;
                }
            }
        }
        let file_meta = entry.metadata().ok();
        let size = file_meta.as_ref().map(|m| m.len()).unwrap_or(0);
        if size as usize > cfg.max_file_bytes {
            errors.push(serde_json::json!({
                "path": rel_str,
                "reason": format!("file too large: {size} bytes (max {})", cfg.max_file_bytes)
            }));
            continue;
        }
        let mtime_sys = file_meta.as_ref().and_then(|m| m.modified().ok());
        let (meta, mtime_dt) = fs_metadata(&rel_str, mtime_sys);
        match std::fs::read_to_string(entry.path()) {
            Ok(text) => scanned.push(Scanned {
                doc_id: format!("{prefix}{rel_str}"),
                text,
                metadata: meta,
                mtime: mtime_dt,
            }),
            Err(e) => errors.push(serde_json::json!({ "path": rel_str, "reason": e.to_string() })),
        }
    }
    let candidate_files = scanned.len();
    let scanned_ids: HashSet<&str> = scanned.iter().map(|s| s.doc_id.as_str()).collect();

    // ---- prune set: in-scope docs no longer on disk ------------------------
    // Never prune if any file errored (a transient IO error must not delete
    // valid memory), and only when scoped by source_id.
    let prune_blocked_by_errors = p.prune && !errors.is_empty();
    let mut prune_ids: Vec<String> = Vec::new();
    if p.prune && p.source_id.is_some() && errors.is_empty() {
        let profile = idx.current_index_profile_hash(embedder);
        let mut cursor: Option<String> = None;
        loop {
            let page = idx
                .manifest(
                    &profile,
                    p.source_id.as_deref(),
                    None,
                    msa_core::index::MAX_MANIFEST_LIMIT,
                    cursor.as_deref(),
                )
                .map_err(user_or_internal)?;
            for e in &page.entries {
                if !scanned_ids.contains(e.doc_id.as_str()) {
                    prune_ids.push(e.doc_id.clone());
                }
            }
            match page.next_cursor {
                Some(c) => cursor = Some(c),
                None => break,
            }
        }
    }

    // ---- dry-run: report the plan without writing --------------------------
    if p.dry_run {
        let (would_index, would_unchanged) =
            plan_counts(idx, embedder, p.source_id.as_deref(), &scanned)
                .map_err(user_or_internal)?;
        return Ok(serde_json::json!({
            "collection": p.collection,
            "source_id": p.source_id,
            "dry_run": true,
            "scanned_files": scanned_files,
            "candidate_files": candidate_files,
            "would_index": would_index,
            "would_unchanged": would_unchanged,
            "would_delete": prune_ids.len(),
            "prune_blocked_by_errors": prune_blocked_by_errors,
            "errors": errors,
        }));
    }

    // ---- apply: upserts in bounded batches, then prune ---------------------
    let mut indexed = 0usize;
    let mut unchanged = 0usize;
    let mut deleted = 0usize;
    let mut committed_batches = 0usize;
    let max_docs = caps.max_docs.max(1);

    for chunk in scanned.chunks(max_docs) {
        let docs: Vec<Document> = chunk
            .iter()
            .map(|s| Document {
                id: s.doc_id.clone(),
                text: s.text.clone(),
                // FS metadata: source, ext, dir, created_at (ISO-8601 string).
                // These keys are intentionally included in the content_hash so
                // that a metadata-only change (e.g. mtime shift) is detectable.
                // A one-time re-index of all FS-synced docs occurs when upgrading
                // from the previous version that stored no metadata.
                metadata: s.metadata.clone(),
                // Set Document::created_at from the file mtime so that
                // SearchFilter::created_after / created_before work correctly
                // on FS-synced collections.
                created_at: s.mtime,
            })
            .collect();
        let opts = DeltaOptions {
            skip_unchanged: true,
            source_id: p.source_id.clone(),
            if_manifest_digest: None,
            if_index_profile_hash: None,
        };
        let o = idx
            .apply_delta(&docs, &[], embedder, &opts, caps)
            .map_err(user_or_internal)?;
        indexed += o.indexed;
        unchanged += o.unchanged;
        committed_batches += 1;
    }

    let prune_executed = !prune_ids.is_empty();
    for chunk in prune_ids.chunks(max_docs) {
        let refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
        let opts = DeltaOptions {
            skip_unchanged: false,
            source_id: p.source_id.clone(),
            if_manifest_digest: None,
            if_index_profile_hash: None,
        };
        let o = idx
            .apply_delta(&[], &refs, embedder, &opts, caps)
            .map_err(user_or_internal)?;
        deleted += o.deleted;
        committed_batches += 1;
    }

    let manifest_digest = idx
        .manifest(
            &idx.current_index_profile_hash(embedder),
            p.source_id.as_deref(),
            None,
            1,
            None,
        )
        .map_err(user_or_internal)?
        .manifest_digest;

    Ok(serde_json::json!({
        "collection": p.collection,
        "source_id": p.source_id,
        "scanned_files": scanned_files,
        "candidate_files": candidate_files,
        "indexed": indexed,
        "unchanged": unchanged,
        "deleted": deleted,
        "prune_executed": prune_executed,
        "prune_blocked_by_errors": prune_blocked_by_errors,
        "errors": errors,
        // A single batch with no prune is one atomic commit; multiple batches
        // are individually atomic but not collectively.
        "atomic": committed_batches <= 1,
        "committed_batches": committed_batches,
        "manifest_digest": manifest_digest,
    }))
}

/// Plan helper: how many scanned files would index vs are already unchanged,
/// by diffing computed content hashes against the scope manifest.
///
/// Uses the same metadata that `sync_path` would pass to `apply_delta`, so
/// the dry-run count stays consistent with what the real run would do.
fn plan_counts(
    idx: &MsaIndex,
    embedder: Option<&dyn msa_core::embeddings::DenseEncoder>,
    source_id: Option<&str>,
    scanned: &[Scanned],
) -> Result<(usize, usize), MsaError> {
    let profile = idx.current_index_profile_hash(embedder);
    // Map doc_id -> (stored content_hash, profile-fresh?)
    let mut stored: HashMap<String, (Option<String>, bool)> = HashMap::new();
    let mut cursor: Option<String> = None;
    loop {
        let page = idx.manifest(
            &profile,
            source_id,
            None,
            msa_core::index::MAX_MANIFEST_LIMIT,
            cursor.as_deref(),
        )?;
        for e in page.entries {
            let fresh = e.hash_state == msa_core::index::HashState::Ok;
            stored.insert(e.doc_id, (e.content_hash, fresh));
        }
        match page.next_cursor {
            Some(c) => cursor = Some(c),
            None => break,
        }
    }
    let mut would_index = 0;
    let mut would_unchanged = 0;
    for s in scanned {
        // Use the real FS metadata, not empty_meta, so the hash matches what
        // apply_delta will compute and the dry-run count is accurate.
        let ch = content_hash_v1(&s.text, &s.metadata);
        match stored.get(&s.doc_id) {
            Some((Some(stored_ch), true)) if *stored_ch == ch => would_unchanged += 1,
            _ => would_index += 1,
        }
    }
    Ok((would_index, would_unchanged))
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    // ---- extract_ext --------------------------------------------------------

    #[test]
    fn ext_plain_extension() {
        assert_eq!(extract_ext("notes.md"), "md");
    }

    #[test]
    fn ext_nested_path() {
        assert_eq!(extract_ext("projects/x/y.txt"), "txt");
    }

    #[test]
    fn ext_no_extension() {
        assert_eq!(extract_ext("Makefile"), "");
    }

    #[test]
    fn ext_hidden_file() {
        // ".gitignore" is a dotfile — no extension by convention.
        assert_eq!(extract_ext(".gitignore"), "");
    }

    #[test]
    fn ext_double_extension() {
        // Only the last extension is returned.
        assert_eq!(extract_ext("archive.tar.gz"), "gz");
    }

    #[test]
    fn ext_trailing_dot() {
        // "foo." has an empty extension after the dot — treat as no extension.
        assert_eq!(extract_ext("foo."), "");
    }

    // ---- extract_dir --------------------------------------------------------

    #[test]
    fn dir_nested_path() {
        assert_eq!(extract_dir("projects/x/y.md"), "projects");
    }

    #[test]
    fn dir_single_level() {
        assert_eq!(extract_dir("docs/readme.md"), "docs");
    }

    #[test]
    fn dir_root_file() {
        assert_eq!(extract_dir("README.md"), "");
    }

    #[test]
    fn dir_root_no_extension() {
        assert_eq!(extract_dir("Makefile"), "");
    }

    // ---- fs_metadata --------------------------------------------------------

    #[test]
    fn fs_metadata_keys_present() {
        let (meta, _) = fs_metadata("projects/foo/bar.md", None);
        assert!(meta.contains_key("created_at"), "created_at missing");
        assert!(meta.contains_key("source"), "source missing");
        assert!(meta.contains_key("ext"), "ext missing");
        assert!(meta.contains_key("dir"), "dir missing");
    }

    #[test]
    fn fs_metadata_source_is_fs() {
        let (meta, _) = fs_metadata("any.txt", None);
        assert_eq!(meta["source"], serde_json::Value::String("fs".to_string()));
    }

    #[test]
    fn fs_metadata_ext_and_dir() {
        let (meta, _) = fs_metadata("projects/design/spec.md", None);
        assert_eq!(meta["ext"], serde_json::Value::String("md".to_string()));
        assert_eq!(
            meta["dir"],
            serde_json::Value::String("projects".to_string())
        );
    }

    #[test]
    fn fs_metadata_mtime_propagated() {
        // Known mtime: 2024-01-15T10:30:00Z = unix 1705312200
        let mtime = SystemTime::UNIX_EPOCH + Duration::from_secs(1_705_312_200);
        let (meta, dt) = fs_metadata("readme.md", Some(mtime));
        // DateTime<Utc> returned must match.
        assert_eq!(dt.timestamp(), 1_705_312_200);
        // Metadata created_at string must be ISO-8601 UTC.
        let ca = meta["created_at"]
            .as_str()
            .expect("created_at must be a string");
        assert!(ca.contains("2024-01-15"), "expected 2024-01-15 in {ca}");
        assert!(ca.ends_with('Z'), "expected UTC Z suffix in {ca}");
    }

    #[test]
    fn fs_metadata_fallback_when_no_mtime() {
        // With None mtime the function must still return a valid RFC3339 string.
        let (meta, _dt) = fs_metadata("readme.md", None);
        let ca = meta["created_at"]
            .as_str()
            .expect("created_at must be a string");
        assert!(!ca.is_empty());
        // Basic RFC3339 structure: contains 'T' separator and ends with 'Z'.
        assert!(
            ca.contains('T') && ca.ends_with('Z'),
            "not a valid RFC3339 UTC: {ca}"
        );
    }

    #[test]
    fn fs_metadata_root_file_has_empty_dir() {
        let (meta, _) = fs_metadata("README.md", None);
        assert_eq!(meta["dir"], serde_json::Value::String(String::new()));
    }
}
