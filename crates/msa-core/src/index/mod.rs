//! Per-collection tantivy index. One on-disk index per collection.
//!
//! This module is split into focused submodules:
//! - [`types`]     — public data types (fingerprints, manifests, delta options/outcomes)
//! - [`schema_io`] — tantivy field handles, manifest JSON, stored-doc accessors, filters
//! - [`write`]     — `MsaIndex` struct + construction + write path
//! - [`read`]      — search (BM25 + hybrid), fetch_doc, stats, manifest, doc_fingerprint
//! - [`delta`]     — `apply_delta` atomic upsert+delete batch
//! - [`registry`]  — process-wide `CollectionRegistry`
//!
//! All public items from the original `index.rs` are re-exported from this
//! module so dependent crates (`mcp-msa-server`, `msa-bench`) require no
//! `use`-path changes.

mod delta;
mod read;
mod registry;
mod schema_io;
mod types;
mod write;

// Re-export the full public surface that `lib.rs` and dependent crates expect.
pub use registry::CollectionRegistry;
pub use types::{
    BatchCaps, DeltaOptions, DeltaOutcome, DeltaResult, DeltaStatus, DocFingerprint, HashState,
    Manifest, ManifestEntry, DEFAULT_MANIFEST_LIMIT, MAX_MANIFEST_LIMIT,
};
pub use write::MsaIndex;

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::collections::HashSet;

    use chrono::Utc;
    use tantivy::schema::{Schema, STORED, STRING, TEXT};
    use tantivy::Index;
    use tempfile::TempDir;

    use crate::error::MsaError;
    use crate::schema::{ChunkConfig, Document, SearchFilter};

    use super::registry::CollectionRegistry;
    use super::schema_io::CollectionManifest;
    use super::types::{BatchCaps, DeltaOptions, HashState};
    use super::write::MsaIndex;

    fn fresh_index() -> (TempDir, MsaIndex) {
        let dir = tempfile::tempdir().unwrap();
        let idx = MsaIndex::open(
            "test".into(),
            dir.path().to_path_buf(),
            ChunkConfig::default(),
        )
        .unwrap();
        (dir, idx)
    }

    #[test]
    fn index_and_search_basic() {
        let (_d, idx) = fresh_index();
        idx.index_document(
            &Document {
                id: "d1".into(),
                text: "il gatto nero dorme sul divano rosso".into(),
                metadata: Default::default(),
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();
        idx.index_document(
            &Document {
                id: "d2".into(),
                text: "configurazione XML namespace schema validation parser".into(),
                metadata: Default::default(),
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();

        let hits = idx.search("gatto", 5, None).unwrap();
        assert!(!hits.is_empty(), "expected hits for 'gatto'");
        assert_eq!(hits[0].doc_id, "d1");
        assert!(hits[0].score > 0.0);
        assert!(hits[0].score <= 1.0 + f32::EPSILON);
    }

    #[test]
    fn fetch_doc_returns_full_text() {
        let (_d, idx) = fresh_index();
        let original = "alpha bravo charlie delta echo foxtrot golf hotel";
        idx.index_document(
            &Document {
                id: "d1".into(),
                text: original.into(),
                metadata: Default::default(),
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();
        let fetched = idx.fetch_doc("d1").unwrap();
        assert_eq!(fetched.text, original);
    }

    // ---- A6.1 fingerprint foundation --------------------------------------

    fn meta(pairs: &[(&str, serde_json::Value)]) -> HashMap<String, serde_json::Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn v3_index_writes_fingerprint_and_fetch_strips_msa() {
        let (_d, idx) = fresh_index();
        let m = meta(&[
            ("path", serde_json::json!("a.md")),
            ("kind", serde_json::json!("doc")),
        ]);
        idx.index_document(
            &Document {
                id: "d1".into(),
                text: "the cat sleeps".into(),
                metadata: m.clone(),
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();

        // fetch_doc must NOT expose the reserved _msa block, but keep user keys.
        let fetched = idx.fetch_doc("d1").unwrap();
        assert!(!fetched
            .metadata
            .contains_key(crate::fingerprint::RESERVED_MSA_KEY));
        assert_eq!(
            fetched.metadata.get("path"),
            Some(&serde_json::json!("a.md"))
        );

        // doc_fingerprint reads the typed v3 field and matches content_hash_v1.
        let fp = idx.doc_fingerprint("d1").unwrap().expect("fingerprint");
        assert_eq!(
            fp.content_hash,
            Some(crate::fingerprint::content_hash_v1("the cat sleeps", &m))
        );
        assert!(fp.index_profile_hash.is_some());
        assert_eq!(fp.source_id, None, "no source_id on the plain index path");
        assert!(idx.doc_fingerprint("missing").unwrap().is_none());
    }

    #[test]
    fn index_rejects_reserved_msa_metadata() {
        let (_d, idx) = fresh_index();
        let bad = meta(&[(crate::fingerprint::RESERVED_MSA_KEY, serde_json::json!("x"))]);
        let err = idx
            .index_document(
                &Document {
                    id: "d1".into(),
                    text: "x".into(),
                    metadata: bad,
                    created_at: Utc::now(),
                },
                None,
            )
            .unwrap_err();
        assert!(matches!(err, MsaError::InvalidInput(_)));
    }

    /// Build a legacy SCHEMA_VERSION-2 index on disk (no fingerprint fields),
    /// open it with the current binary, reindex a doc, and verify the
    /// fingerprint self-heals via the `_msa` metadata fallback (no rebuild).
    #[test]
    fn legacy_v2_collection_self_heals_via_msa_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // v2 schema = current minus the three A6 fields.
        let mut sb = Schema::builder();
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
        Index::create_in_dir(&path, sb.build()).unwrap();
        CollectionManifest::write(&path, 2).unwrap();

        // Open with the current binary: the 3 fingerprint fields resolve to None.
        let idx = MsaIndex::open("legacy".into(), path.clone(), ChunkConfig::default()).unwrap();
        assert!(
            idx.fields.content_hash.is_none(),
            "legacy schema has no typed field"
        );

        let m = meta(&[("k", serde_json::json!("v"))]);
        idx.index_document(
            &Document {
                id: "d1".into(),
                text: "legacy doc".into(),
                metadata: m.clone(),
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();

        // Self-heal: fingerprint readable from the _msa metadata fallback.
        let fp = idx
            .doc_fingerprint("d1")
            .unwrap()
            .expect("fingerprint via fallback");
        assert_eq!(
            fp.content_hash,
            Some(crate::fingerprint::content_hash_v1("legacy doc", &m))
        );
        assert!(fp.index_profile_hash.is_some());
        // And the user still never sees _msa.
        assert!(!idx
            .fetch_doc("d1")
            .unwrap()
            .metadata
            .contains_key(crate::fingerprint::RESERVED_MSA_KEY));
    }

    // ---- A6.2 manifest -----------------------------------------------------

    #[test]
    fn manifest_pagination_digest_filters() {
        let (_d, idx) = fresh_index();
        for id in ["p/a", "p/b", "q/c"] {
            idx.index_document(
                &Document {
                    id: id.into(),
                    text: format!("text of {id}"),
                    metadata: Default::default(),
                    created_at: Utc::now(),
                },
                None,
            )
            .unwrap();
        }
        let profile = idx.current_index_profile_hash(None);

        // All fresh, sorted by doc_id, total_count correct.
        let m = idx.manifest(&profile, None, None, 10, None).unwrap();
        assert_eq!(m.total_count, 3);
        assert_eq!(
            m.entries
                .iter()
                .map(|e| e.doc_id.as_str())
                .collect::<Vec<_>>(),
            ["p/a", "p/b", "q/c"]
        );
        assert!(m.entries.iter().all(|e| e.hash_state == HashState::Ok));
        assert!(m.next_cursor.is_none());

        // Digest is stable across calls.
        let m2 = idx.manifest(&profile, None, None, 10, None).unwrap();
        assert_eq!(m.manifest_digest, m2.manifest_digest);

        // A different "current" profile flags everything profile_stale.
        let stale = idx.manifest("bogus-profile", None, None, 10, None).unwrap();
        assert!(stale
            .entries
            .iter()
            .all(|e| e.hash_state == HashState::ProfileStale));

        // Pagination: limit 2 -> page1 (p/a,p/b) + cursor; page2 (q/c).
        let p1 = idx.manifest(&profile, None, None, 2, None).unwrap();
        assert_eq!(p1.entries.len(), 2);
        assert_eq!(p1.next_cursor.as_deref(), Some("p/b"));
        let p2 = idx
            .manifest(&profile, None, None, 2, p1.next_cursor.as_deref())
            .unwrap();
        assert_eq!(
            p2.entries
                .iter()
                .map(|e| e.doc_id.as_str())
                .collect::<Vec<_>>(),
            ["q/c"]
        );
        assert!(p2.next_cursor.is_none());
        // total_count and digest are scope-wide, identical across pages.
        assert_eq!(p1.total_count, 3);
        assert_eq!(p1.manifest_digest, m.manifest_digest);

        // doc_id_prefix filter.
        let pfx = idx.manifest(&profile, None, Some("p/"), 10, None).unwrap();
        assert_eq!(pfx.total_count, 2);
        assert!(pfx.entries.iter().all(|e| e.doc_id.starts_with("p/")));

        // source_id filter: these docs have no source_id, so a filter excludes all.
        let none = idx
            .manifest(&profile, Some("docs"), None, 10, None)
            .unwrap();
        assert_eq!(none.total_count, 0);
    }

    #[test]
    fn manifest_digest_changes_with_content() {
        let (_d, idx) = fresh_index();
        idx.index_document(
            &Document {
                id: "d1".into(),
                text: "first".into(),
                metadata: Default::default(),
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();
        let profile = idx.current_index_profile_hash(None);
        let before = idx
            .manifest(&profile, None, None, 10, None)
            .unwrap()
            .manifest_digest;
        // Reindex same doc_id with different content -> digest must change.
        idx.index_document(
            &Document {
                id: "d1".into(),
                text: "second different".into(),
                metadata: Default::default(),
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();
        let after = idx
            .manifest(&profile, None, None, 10, None)
            .unwrap()
            .manifest_digest;
        assert_ne!(before, after);
    }

    // ---- A6.3 apply_delta --------------------------------------------------

    fn caps() -> BatchCaps {
        BatchCaps {
            max_docs: 100,
            max_text_bytes: 32 * 1024 * 1024,
        }
    }
    fn docv(id: &str, text: &str) -> Document {
        Document {
            id: id.into(),
            text: text.into(),
            metadata: Default::default(),
            created_at: Utc::now(),
        }
    }

    #[test]
    fn apply_delta_skips_unchanged_reindexes_changed() {
        let (_d, idx) = fresh_index();
        let opts = DeltaOptions {
            skip_unchanged: true,
            ..Default::default()
        };
        let o1 = idx
            .apply_delta(&[docv("a", "hello world")], &[], None, &opts, caps())
            .unwrap();
        assert_eq!(
            (o1.indexed, o1.unchanged, o1.deleted, o1.not_found),
            (1, 0, 0, 0)
        );
        // same content -> unchanged (skipped, no reindex).
        let o2 = idx
            .apply_delta(&[docv("a", "hello world")], &[], None, &opts, caps())
            .unwrap();
        assert_eq!((o2.indexed, o2.unchanged), (0, 1));
        // changed content -> reindexed.
        let o3 = idx
            .apply_delta(&[docv("a", "hello there")], &[], None, &opts, caps())
            .unwrap();
        assert_eq!((o3.indexed, o3.unchanged), (1, 0));
        assert_eq!(idx.fetch_doc("a").unwrap().text, "hello there");
        assert_ne!(o2.new_manifest_digest, o3.new_manifest_digest);
        assert_eq!(o3.results[0].num_chunks, Some(1));
    }

    #[test]
    fn apply_delta_upsert_and_delete_atomic() {
        let (_d, idx) = fresh_index();
        idx.apply_delta(
            &[docv("old", "gone soon"), docv("keep", "stays")],
            &[],
            None,
            &DeltaOptions::default(),
            caps(),
        )
        .unwrap();
        let o = idx
            .apply_delta(
                &[docv("new", "fresh")],
                &["old"],
                None,
                &DeltaOptions::default(),
                caps(),
            )
            .unwrap();
        assert_eq!((o.indexed, o.deleted, o.not_found), (1, 1, 0));
        assert!(idx.fetch_doc("old").is_err());
        assert_eq!(idx.fetch_doc("new").unwrap().text, "fresh");
        assert_eq!(idx.fetch_doc("keep").unwrap().text, "stays");
    }

    #[test]
    fn apply_delta_rejects_doc_in_both_sides() {
        let (_d, idx) = fresh_index();
        let r = idx.apply_delta(
            &[docv("x", "t")],
            &["x"],
            None,
            &DeltaOptions::default(),
            caps(),
        );
        assert!(matches!(r, Err(MsaError::InvalidInput(_))));
    }

    #[test]
    fn apply_delta_profile_precondition_conflict_no_write() {
        let (_d, idx) = fresh_index();
        idx.apply_delta(
            &[docv("a", "one")],
            &[],
            None,
            &DeltaOptions::default(),
            caps(),
        )
        .unwrap();
        let opts = DeltaOptions {
            if_index_profile_hash: Some("WRONG".into()),
            ..Default::default()
        };
        let r = idx.apply_delta(&[docv("a", "two")], &[], None, &opts, caps());
        assert!(matches!(r, Err(MsaError::Conflict(_))));
        // Nothing written: still "one".
        assert_eq!(idx.fetch_doc("a").unwrap().text, "one");
    }

    #[test]
    fn apply_delta_manifest_precondition_gate() {
        let (_d, idx) = fresh_index();
        idx.apply_delta(
            &[docv("a", "one")],
            &[],
            None,
            &DeltaOptions::default(),
            caps(),
        )
        .unwrap();
        // Stale digest -> Conflict, no write.
        let stale = DeltaOptions {
            if_manifest_digest: Some("staledigest".into()),
            ..Default::default()
        };
        assert!(matches!(
            idx.apply_delta(&[docv("b", "two")], &[], None, &stale, caps()),
            Err(MsaError::Conflict(_))
        ));
        assert!(idx.fetch_doc("b").is_err());
        // Correct digest -> succeeds.
        let real = idx
            .manifest(&idx.current_index_profile_hash(None), None, None, 1, None)
            .unwrap()
            .manifest_digest;
        let ok = DeltaOptions {
            if_manifest_digest: Some(real),
            ..Default::default()
        };
        let o = idx
            .apply_delta(&[docv("b", "two")], &[], None, &ok, caps())
            .unwrap();
        assert_eq!(o.indexed, 1);
        assert_eq!(idx.fetch_doc("b").unwrap().text, "two");
    }

    #[test]
    fn apply_delta_enforces_batch_caps() {
        let (_d, idx) = fresh_index();
        let small = BatchCaps {
            max_docs: 1,
            max_text_bytes: 1 << 20,
        };
        let r = idx.apply_delta(
            &[docv("a", "x"), docv("b", "y")],
            &[],
            None,
            &DeltaOptions::default(),
            small,
        );
        assert!(matches!(r, Err(MsaError::InvalidInput(_))));
    }

    #[test]
    fn apply_delta_delete_missing_is_not_found() {
        let (_d, idx) = fresh_index();
        let o = idx
            .apply_delta(&[], &["ghost"], None, &DeltaOptions::default(), caps())
            .unwrap();
        assert_eq!((o.indexed, o.deleted, o.not_found), (0, 0, 1));
    }

    #[test]
    fn apply_delta_source_id_scopes_manifest() {
        let (_d, idx) = fresh_index();
        let opts = DeltaOptions {
            source_id: Some("docs".into()),
            ..Default::default()
        };
        idx.apply_delta(&[docv("a", "x")], &[], None, &opts, caps())
            .unwrap();
        let profile = idx.current_index_profile_hash(None);
        let m = idx
            .manifest(&profile, Some("docs"), None, 10, None)
            .unwrap();
        assert_eq!(m.total_count, 1);
        assert_eq!(m.entries[0].source_id.as_deref(), Some("docs"));
        // A different source scope sees nothing.
        assert_eq!(
            idx.manifest(&profile, Some("other"), None, 10, None)
                .unwrap()
                .total_count,
            0
        );
    }

    #[test]
    fn apply_delta_scoped_upsert_does_not_steal_other_source() {
        let (_d, idx) = fresh_index();
        let a = DeltaOptions {
            source_id: Some("A".into()),
            ..Default::default()
        };
        idx.apply_delta(&[docv("shared", "original")], &[], None, &a, caps())
            .unwrap();

        // Source B upserts the same doc_id with CHANGED content -> must not steal.
        let b = DeltaOptions {
            source_id: Some("B".into()),
            ..Default::default()
        };
        let o = idx
            .apply_delta(&[docv("shared", "hijacked")], &[], None, &b, caps())
            .unwrap();
        assert_eq!((o.indexed, o.skipped_scope_mismatch), (0, 1));
        assert_eq!(idx.fetch_doc("shared").unwrap().text, "original");
        assert_eq!(
            idx.doc_fingerprint("shared")
                .unwrap()
                .unwrap()
                .source_id
                .as_deref(),
            Some("A")
        );

        // Same content + skip_unchanged: the collision must NOT hide as unchanged.
        let b_skip = DeltaOptions {
            source_id: Some("B".into()),
            skip_unchanged: true,
            ..Default::default()
        };
        let o2 = idx
            .apply_delta(&[docv("shared", "original")], &[], None, &b_skip, caps())
            .unwrap();
        assert_eq!(o2.skipped_scope_mismatch, 1);
        assert_eq!(o2.unchanged, 0);
    }

    #[test]
    fn apply_delta_concurrent_precondition_only_one_commits() {
        use std::sync::Arc;
        let (_d, idx) = fresh_index();
        idx.apply_delta(
            &[docv("seed", "x")],
            &[],
            None,
            &DeltaOptions::default(),
            caps(),
        )
        .unwrap();
        let digest = idx
            .manifest(&idx.current_index_profile_hash(None), None, None, 1, None)
            .unwrap()
            .manifest_digest;

        // Two concurrent syncs share the SAME if_manifest_digest precondition.
        // The delta lock makes precondition+commit atomic, so exactly one wins;
        // the other sees the changed digest and gets Conflict.
        let idx = Arc::new(idx);
        let mk = |id: &'static str, txt: &'static str, dg: String| {
            let i = idx.clone();
            std::thread::spawn(move || {
                let opts = DeltaOptions {
                    if_manifest_digest: Some(dg),
                    ..Default::default()
                };
                i.apply_delta(&[docv(id, txt)], &[], None, &opts, caps())
            })
        };
        let h1 = mk("a", "one", digest.clone());
        let h2 = mk("b", "two", digest.clone());
        let r1 = h1.join().unwrap();
        let r2 = h2.join().unwrap();

        let oks = [&r1, &r2].iter().filter(|r| r.is_ok()).count();
        let conflicts = [&r1, &r2]
            .iter()
            .filter(|r| matches!(r, Err(MsaError::Conflict(_))))
            .count();
        assert_eq!(
            oks, 1,
            "exactly one concurrent sync may commit on the same precondition"
        );
        assert_eq!(
            conflicts, 1,
            "the loser must get Conflict, not a silent overwrite"
        );
    }

    #[test]
    fn apply_delta_scoped_delete_protects_other_sources() {
        let (_d, idx) = fresh_index();
        let src_a = DeltaOptions {
            source_id: Some("A".into()),
            ..Default::default()
        };
        idx.apply_delta(&[docv("shared", "x")], &[], None, &src_a, caps())
            .unwrap();

        // A delete scoped to "B" must NOT evict a doc owned by "A".
        let src_b = DeltaOptions {
            source_id: Some("B".into()),
            ..Default::default()
        };
        let o = idx
            .apply_delta(&[], &["shared"], None, &src_b, caps())
            .unwrap();
        assert_eq!(
            (o.deleted, o.not_found, o.skipped_scope_mismatch),
            (0, 0, 1)
        );
        assert!(
            idx.fetch_doc("shared").is_ok(),
            "scope mismatch must not delete"
        );

        // A delete scoped to "A" deletes it.
        let o2 = idx
            .apply_delta(&[], &["shared"], None, &src_a, caps())
            .unwrap();
        assert_eq!((o2.deleted, o2.skipped_scope_mismatch), (1, 0));
        assert!(idx.fetch_doc("shared").is_err());
    }

    #[test]
    fn search_filter_where_eq_drops_non_matching() {
        let (_d, idx) = fresh_index();
        let mut meta_a = HashMap::new();
        meta_a.insert("kind".to_string(), serde_json::json!("turn"));
        let mut meta_b = HashMap::new();
        meta_b.insert("kind".to_string(), serde_json::json!("loop_runtime"));

        idx.index_document(
            &Document {
                id: "a".into(),
                text: "alpha gatto bravo".into(),
                metadata: meta_a,
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();
        idx.index_document(
            &Document {
                id: "b".into(),
                text: "alpha gatto bravo".into(), // identical text
                metadata: meta_b,
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();

        let mut filter = SearchFilter::default();
        filter
            .where_eq
            .insert("kind".to_string(), serde_json::json!("turn"));

        let hits = idx.search("gatto", 5, Some(&filter)).unwrap();
        assert!(!hits.is_empty(), "expected at least one hit");
        assert!(
            hits.iter().all(|h| h.doc_id == "a"),
            "filter should keep only doc 'a', got: {:?}",
            hits.iter().map(|h| h.doc_id.as_str()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn search_filter_where_in_accepts_any() {
        let (_d, idx) = fresh_index();
        let kinds = ["turn", "loop_runtime", "loop_config"];
        for (i, kind) in kinds.iter().enumerate() {
            let mut meta = HashMap::new();
            meta.insert("kind".to_string(), serde_json::json!(kind));
            idx.index_document(
                &Document {
                    id: format!("d{i}"),
                    text: "alpha gatto bravo".into(),
                    metadata: meta,
                    created_at: Utc::now(),
                },
                None,
            )
            .unwrap();
        }

        let mut filter = SearchFilter::default();
        filter.where_in.insert(
            "kind".to_string(),
            vec![serde_json::json!("turn"), serde_json::json!("loop_config")],
        );

        let hits = idx.search("gatto", 5, Some(&filter)).unwrap();
        let ids: Vec<_> = hits.iter().map(|h| h.doc_id.clone()).collect();
        assert!(
            ids.contains(&"d0".to_string()),
            "expected d0 (turn) in {ids:?}"
        );
        assert!(
            ids.contains(&"d2".to_string()),
            "expected d2 (loop_config) in {ids:?}"
        );
        assert!(
            !ids.contains(&"d1".to_string()),
            "did not expect d1 (loop_runtime) in {ids:?}"
        );
    }

    #[test]
    fn search_excluding_drops_listed_doc_ids() {
        let (_d, idx) = fresh_index();
        for i in 0..5 {
            idx.index_document(
                &Document {
                    id: format!("d{i}"),
                    text: "alpha gatto bravo charlie".into(),
                    metadata: Default::default(),
                    created_at: Utc::now(),
                },
                None,
            )
            .unwrap();
        }

        let mut excludes = HashSet::new();
        excludes.insert("d0".to_string());
        excludes.insert("d2".to_string());

        let hits = idx.search_excluding("gatto", 5, None, &excludes).unwrap();
        let ids: Vec<_> = hits.iter().map(|h| h.doc_id.clone()).collect();
        assert!(
            !ids.contains(&"d0".into()),
            "d0 should be excluded, got {ids:?}"
        );
        assert!(
            !ids.contains(&"d2".into()),
            "d2 should be excluded, got {ids:?}"
        );
        // The remaining three docs should all be reachable.
        assert!(ids.iter().any(|id| id == "d1"));
        assert!(ids.iter().any(|id| id == "d3"));
        assert!(ids.iter().any(|id| id == "d4"));
    }

    #[test]
    fn search_hybrid_combines_bm25_and_dense() {
        use crate::embeddings::testing::MockEncoder as MockScorer;
        use crate::embeddings::{DenseEncoder as _, EmbedInputKind};
        let (_d, idx) = fresh_index();
        let scorer = MockScorer::new(8);

        // Both docs share the BM25 token "alpha", but only doc "first"
        // matches the dense embedding (mock keys on the first whitespace
        // word). With dense_alpha < 1 the rerank should prefer "first".
        idx.index_document(
            &Document {
                id: "first".into(),
                text: "first alpha bravo charlie".into(),
                metadata: Default::default(),
                created_at: Utc::now(),
            },
            Some(&scorer),
        )
        .unwrap();
        idx.index_document(
            &Document {
                id: "delta".into(),
                text: "delta alpha bravo charlie".into(),
                metadata: Default::default(),
                created_at: Utc::now(),
            },
            Some(&scorer),
        )
        .unwrap();

        let q_emb = scorer.encode_kind(EmbedInputKind::Query, "first").unwrap();
        let hits = idx
            .search_hybrid("alpha", 5, None, &HashSet::new(), 0.2, Some(&q_emb))
            .unwrap();
        assert!(!hits.is_empty());
        assert_eq!(
            hits[0].doc_id,
            "first",
            "dense rerank should rank 'first' above 'delta' (got {:?})",
            hits.iter()
                .map(|h| (h.doc_id.clone(), h.score))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn search_hybrid_rejects_alpha_lt_one_without_query_embedding() {
        let (_d, idx) = fresh_index();
        idx.index_document(
            &Document {
                id: "x".into(),
                text: "alpha".into(),
                metadata: Default::default(),
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();
        let err = idx
            .search_hybrid("alpha", 5, None, &HashSet::new(), 0.5, None)
            .unwrap_err();
        match err {
            MsaError::InvalidQuery(_) => {}
            other => panic!("expected InvalidQuery, got {other:?}"),
        }
    }

    #[test]
    fn search_hybrid_falls_back_to_neutral_cosine_when_chunk_has_no_embedding() {
        use crate::embeddings::testing::MockEncoder as MockScorer;
        use crate::embeddings::{DenseEncoder as _, EmbedInputKind};
        let (_d, idx) = fresh_index();
        let scorer = MockScorer::new(8);
        // Indexed without scorer → no chunk embedding stored.
        idx.index_document(
            &Document {
                id: "x".into(),
                text: "alpha bravo".into(),
                metadata: Default::default(),
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();
        let q_emb = scorer.encode_kind(EmbedInputKind::Query, "alpha").unwrap();
        // Should not panic and should still return a hit.
        let hits = idx
            .search_hybrid("alpha", 5, None, &HashSet::new(), 0.5, Some(&q_emb))
            .unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn stats_counts_documents_chunks_and_tokens() {
        let (_d, idx) = fresh_index();
        // chunk_size default = 64 words; craft a doc with a known word count.
        let words: Vec<String> = (0..130).map(|i| format!("w{i}")).collect();
        let text = words.join(" "); // 130 words -> 3 chunks (64,64,2)
        idx.index_document(
            &Document {
                id: "d1".into(),
                text: text.clone(),
                metadata: Default::default(),
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();
        idx.index_document(
            &Document {
                id: "d2".into(),
                text: "alpha bravo charlie".into(), // 1 chunk, 3 tokens
                metadata: Default::default(),
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();

        let s = idx.stats().unwrap();
        assert_eq!(s.num_documents, 2, "two documents -> two sentinels");
        assert_eq!(s.num_chunks, 4, "3 chunks (d1) + 1 chunk (d2)");
        assert_eq!(s.total_tokens, 130 + 3, "sum of per-chunk word counts");
        assert!(s.disk_bytes > 0);
    }

    #[test]
    fn stats_updates_after_delete() {
        let (_d, idx) = fresh_index();
        idx.index_document(
            &Document {
                id: "d1".into(),
                text: "uno due tre".into(),
                metadata: Default::default(),
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();
        assert_eq!(idx.stats().unwrap().num_documents, 1);
        idx.delete_document("d1").unwrap();
        let s = idx.stats().unwrap();
        assert_eq!(s.num_documents, 0, "deleted doc no longer counted");
        assert_eq!(s.num_chunks, 0);
        assert_eq!(s.total_tokens, 0);
    }

    #[test]
    fn drop_collection_removes_from_disk_and_listing() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().to_path_buf();
        let reg = CollectionRegistry::new();
        let idx = reg
            .open_or_create("gone", &base, &ChunkConfig::default())
            .expect("open");
        idx.index_document(
            &Document {
                id: "d1".into(),
                text: "contenuto da eliminare".into(),
                metadata: Default::default(),
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();
        assert!(base.join("gone").exists());
        assert!(reg.list_on_disk(&base).contains(&"gone".to_string()));

        // Drop it.
        assert!(reg.drop_collection("gone", &base).unwrap());
        assert!(!base.join("gone").exists(), "live dir must be gone");
        assert!(
            !reg.list_on_disk(&base).contains(&"gone".to_string()),
            "dropped collection must disappear from listing"
        );
        // No quarantine leftover visible as a collection.
        assert!(reg.list().is_empty(), "registry evicted the dropped index");

        // Dropping again returns false (nothing on disk).
        assert!(!reg.drop_collection("gone", &base).unwrap());

        // Can be recreated cleanly with the same name.
        let idx2 = reg
            .open_or_create("gone", &base, &ChunkConfig::default())
            .expect("recreate");
        assert_eq!(idx2.stats().unwrap().num_documents, 0, "fresh, empty");
    }

    #[test]
    fn drop_collection_rejects_invalid_name() {
        let dir = tempfile::tempdir().unwrap();
        let reg = CollectionRegistry::new();
        match reg.drop_collection("../escape", dir.path()) {
            Err(MsaError::InvalidInput(_)) => {}
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn concurrent_double_drop_is_idempotent() {
        use std::sync::Arc as StdArc;
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().to_path_buf();
        let reg = StdArc::new(CollectionRegistry::new());
        reg.open_or_create("c", &base, &ChunkConfig::default())
            .unwrap()
            .index_document(
                &Document {
                    id: "d".into(),
                    text: "x".into(),
                    metadata: Default::default(),
                    created_at: Utc::now(),
                },
                None,
            )
            .unwrap();

        // Two threads drop the same collection concurrently.
        let (r1, b1) = (reg.clone(), base.clone());
        let (r2, b2) = (reg.clone(), base.clone());
        let h1 = std::thread::spawn(move || r1.drop_collection("c", &b1));
        let h2 = std::thread::spawn(move || r2.drop_collection("c", &b2));
        let o1 = h1.join().unwrap().expect("drop 1 ok");
        let o2 = h2.join().unwrap().expect("drop 2 ok");

        // Exactly one wins (true), the other is a clean idempotent false — never
        // an IO error.
        assert!(
            o1 ^ o2,
            "exactly one drop should report true, got {o1}/{o2}"
        );
        assert!(!base.join("c").exists());
        assert!(!reg.list_on_disk(&base).contains(&"c".to_string()));
    }

    #[test]
    fn drop_then_reopen_is_fresh_under_registry() {
        // After a successful drop, re-opening must not resurrect old data and
        // the registry must not still hold the dropped collection.
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().to_path_buf();
        let reg = CollectionRegistry::new();
        reg.open_or_create("c", &base, &ChunkConfig::default())
            .unwrap()
            .index_document(
                &Document {
                    id: "old".into(),
                    text: "vecchio dato".into(),
                    metadata: Default::default(),
                    created_at: Utc::now(),
                },
                None,
            )
            .unwrap();
        assert!(reg.drop_collection("c", &base).unwrap());
        assert!(
            !reg.list().contains(&"c".to_string()),
            "evicted from registry"
        );

        let idx = reg
            .open_or_create("c", &base, &ChunkConfig::default())
            .expect("reopen");
        assert_eq!(idx.stats().unwrap().num_documents, 0, "fresh after drop");
        assert!(idx.fetch_doc("old").is_err(), "old data must be gone");
    }

    #[test]
    fn list_on_disk_discovers_persisted_collections() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().to_path_buf();
        let reg = CollectionRegistry::new();
        // Create two collections via the registry (validated names).
        let idx_a = reg
            .open_or_create("alpha", &base, &ChunkConfig::default())
            .expect("open alpha");
        idx_a
            .index_document(
                &Document {
                    id: "x".into(),
                    text: "hello world".into(),
                    metadata: Default::default(),
                    created_at: Utc::now(),
                },
                None,
            )
            .unwrap();
        reg.open_or_create("beta", &base, &ChunkConfig::default())
            .expect("open beta");
        // A stray directory that is NOT a tantivy index must be ignored.
        std::fs::create_dir_all(base.join("not_an_index")).unwrap();

        let found = reg.list_on_disk(&base);
        assert!(found.contains(&"alpha".to_string()), "got {found:?}");
        assert!(found.contains(&"beta".to_string()), "got {found:?}");
        assert!(
            !found.contains(&"not_an_index".to_string()),
            "stray dir without meta.json must be ignored, got {found:?}"
        );
    }

    #[test]
    fn open_rejects_newer_schema_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();
        // Create a normal index, then overwrite the manifest with a future version.
        MsaIndex::open("c".into(), path.clone(), ChunkConfig::default()).expect("create index");
        std::fs::write(
            path.join("manifest.json"),
            serde_json::json!({"schema_version": 9999}).to_string(),
        )
        .unwrap();
        match MsaIndex::open("c".into(), path, ChunkConfig::default()) {
            Err(MsaError::Schema(_)) => {}
            Err(other) => panic!("expected Schema error, got {other:?}"),
            Ok(_) => panic!("expected open to reject a newer schema version"),
        }
    }

    #[test]
    fn open_rejects_malformed_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();
        MsaIndex::open("c".into(), path.clone(), ChunkConfig::default()).expect("create index");
        // Corrupt the manifest: present but not valid JSON.
        std::fs::write(path.join("manifest.json"), b"{ this is not json").unwrap();
        match MsaIndex::open("c".into(), path, ChunkConfig::default()) {
            Err(MsaError::Schema(_)) => {}
            Err(other) => panic!("expected Schema error for malformed manifest, got {other:?}"),
            Ok(_) => panic!("expected open to reject a malformed manifest"),
        }
    }

    #[test]
    fn index_documents_bulk_single_commit() {
        let (_d, idx) = fresh_index();
        let docs: Vec<Document> = (0..10)
            .map(|i| Document {
                id: format!("d{i}"),
                text: format!("documento numero {i} con alcune parole comuni"),
                metadata: Default::default(),
                created_at: Utc::now(),
            })
            .collect();
        let total_chunks = idx.index_documents(&docs, None).unwrap();
        assert_eq!(total_chunks, 10, "1 chunk per short doc");
        let s = idx.stats().unwrap();
        assert_eq!(s.num_documents, 10);
        assert_eq!(s.num_chunks, 10);
        // Searchable after the single commit.
        assert!(!idx.search("comuni", 20, None).unwrap().is_empty());
    }

    #[test]
    fn index_documents_rolls_back_on_embedder_error() {
        use crate::embeddings::{DenseEncoder, EmbedInputKind};
        use crate::error::Result;
        // Encoder that always fails, to force a mid-batch error.
        struct FailEncoder;
        impl DenseEncoder for FailEncoder {
            fn encode_batch_kind(&self, _k: EmbedInputKind, _t: &[&str]) -> Result<Vec<Vec<f32>>> {
                Err(MsaError::Config("boom".into()))
            }
            fn dim(&self) -> usize {
                8
            }
            fn model_id(&self) -> &str {
                "fail"
            }
        }
        let (_d, idx) = fresh_index();
        // Seed one committed document so we can prove the failed batch did not
        // disturb the previously committed state.
        idx.index_document(
            &Document {
                id: "seed".into(),
                text: "seed alpha".into(),
                metadata: Default::default(),
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();

        let docs = vec![Document {
            id: "x".into(),
            text: "this will fail to embed".into(),
            metadata: Default::default(),
            created_at: Utc::now(),
        }];
        let err = idx.index_documents(&docs, Some(&FailEncoder)).unwrap_err();
        matches!(err, MsaError::Config(_));

        // The failed batch must have been rolled back: still exactly the seed.
        let s = idx.stats().unwrap();
        assert_eq!(s.num_documents, 1, "rolled back batch leaves only the seed");
        assert!(idx.fetch_doc("x").is_err(), "failed doc must not exist");
        assert!(idx.fetch_doc("seed").is_ok(), "committed seed survives");
    }

    #[test]
    fn single_doc_failure_rolls_back_and_preserves_prior_version() {
        use crate::embeddings::{DenseEncoder, EmbedInputKind};
        use crate::error::Result;
        struct FailEncoder;
        impl DenseEncoder for FailEncoder {
            fn encode_batch_kind(&self, _k: EmbedInputKind, _t: &[&str]) -> Result<Vec<Vec<f32>>> {
                Err(MsaError::Config("boom".into()))
            }
            fn dim(&self) -> usize {
                8
            }
            fn model_id(&self) -> &str {
                "fail"
            }
        }
        let (_d, idx) = fresh_index();
        // Commit a good version of doc "k".
        idx.index_document(
            &Document {
                id: "k".into(),
                text: "versione buona".into(),
                metadata: Default::default(),
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();

        // Reindex the SAME doc_id with a failing embedder.
        let err = idx
            .index_document(
                &Document {
                    id: "k".into(),
                    text: "versione che fallisce".into(),
                    metadata: Default::default(),
                    created_at: Utc::now(),
                },
                Some(&FailEncoder),
            )
            .unwrap_err();
        matches!(err, MsaError::Config(_));

        // A subsequent successful, unrelated write must commit cleanly without
        // dragging in the rolled-back delete/chunks.
        idx.index_document(
            &Document {
                id: "other".into(),
                text: "documento indipendente".into(),
                metadata: Default::default(),
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();

        // The good version of "k" must still be intact.
        let fetched = idx.fetch_doc("k").expect("good version of k survives");
        assert_eq!(fetched.text, "versione buona");
        let s = idx.stats().unwrap();
        assert_eq!(s.num_documents, 2, "k (good) + other; failed reindex gone");
    }

    #[test]
    fn release_writer_then_write_again() {
        let (_d, idx) = fresh_index();
        idx.index_document(
            &Document {
                id: "a".into(),
                text: "primo".into(),
                metadata: Default::default(),
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();
        assert!(idx.release_writer(), "writer existed and was dropped");
        assert!(!idx.release_writer(), "second release is a no-op");
        // Next write lazily recreates the writer and still works.
        idx.index_document(
            &Document {
                id: "b".into(),
                text: "secondo".into(),
                metadata: Default::default(),
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();
        assert_eq!(idx.stats().unwrap().num_documents, 2);
    }

    #[test]
    fn writer_is_reused_across_calls() {
        // Two sequential single-document indexes must both succeed against the
        // same long-lived writer (tantivy forbids two writers per index, so if
        // we leaked a writer this would error on the second call).
        let (_d, idx) = fresh_index();
        for i in 0..3 {
            idx.index_document(
                &Document {
                    id: format!("d{i}"),
                    text: "ripetuto contenuto".into(),
                    metadata: Default::default(),
                    created_at: Utc::now(),
                },
                None,
            )
            .unwrap();
        }
        assert_eq!(idx.stats().unwrap().num_documents, 3);
    }

    #[test]
    fn reindex_same_doc_id_keeps_one_document() {
        let (_d, idx) = fresh_index();
        idx.index_document(
            &Document {
                id: "d1".into(),
                text: "alpha bravo charlie".into(), // 3 tokens, 1 chunk
                metadata: Default::default(),
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();
        // Reindex the same doc_id with different, longer content.
        idx.index_document(
            &Document {
                id: "d1".into(),
                text: "one two three four five".into(), // 5 tokens, 1 chunk
                metadata: Default::default(),
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();

        let s = idx.stats().unwrap();
        assert_eq!(s.num_documents, 1, "replace must leave a single sentinel");
        assert_eq!(s.num_chunks, 1);
        assert_eq!(s.total_tokens, 5, "stats reflect the latest version");
        let fetched = idx.fetch_doc("d1").unwrap();
        assert_eq!(fetched.text, "one two three four five");
    }

    #[test]
    fn delete_removes_chunks() {
        let (_d, idx) = fresh_index();
        idx.index_document(
            &Document {
                id: "d1".into(),
                text: "termine specifico unico".into(),
                metadata: Default::default(),
                created_at: Utc::now(),
            },
            None,
        )
        .unwrap();
        assert!(!idx.search("specifico", 5, None).unwrap().is_empty());
        idx.delete_document("d1").unwrap();
        assert!(idx.search("specifico", 5, None).unwrap().is_empty());
    }
}

#[cfg(test)]
mod merge_tests {
    use chrono::Utc;

    use crate::schema::{ChunkConfig, Document};

    use super::write::MsaIndex;

    /// F2: a commit-per-document workload (the vivling capsule pattern) must
    /// not accumulate unbounded segments — past the threshold the write path
    /// merges explicitly. Without this, 30 single-doc commits = 30 segments.
    #[test]
    fn segment_count_stays_bounded_under_commit_per_doc() {
        let dir = tempfile::tempdir().unwrap();
        let idx = MsaIndex::open(
            "merge-test".into(),
            dir.path().to_path_buf(),
            ChunkConfig::default(),
        )
        .unwrap();
        for i in 0..(MsaIndex::MAX_SEGMENTS_BEFORE_MERGE + 8) {
            idx.index_document(
                &Document {
                    id: format!("doc-{i}"),
                    text: format!("capsule number {i} about segment growth"),
                    metadata: Default::default(),
                    created_at: Utc::now(),
                },
                None,
            )
            .unwrap();
        }
        let segments = idx.index.searchable_segment_ids().unwrap().len();
        assert!(
            segments <= MsaIndex::MAX_SEGMENTS_BEFORE_MERGE,
            "explicit merge must bound segments, got {segments}"
        );
        // Content intact after merges.
        let hits = idx.search("segment growth", 40, None).unwrap();
        assert!(!hits.is_empty());
    }
}
