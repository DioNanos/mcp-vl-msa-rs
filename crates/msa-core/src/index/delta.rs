//! `MsaIndex::apply_delta` — atomic upsert + delete batch with optimistic-
//! concurrency preconditions. This is the autonomy core that `msa_sync_delta`,
//! `msa_index_batch`, and `msa_delete_batch` build on.

use tantivy::Term;

use crate::embeddings::DenseEncoder;
use crate::error::{MsaError, Result};
use crate::fingerprint::content_hash_v1;
use crate::fingerprint::reject_reserved_metadata;
use crate::schema::Document;
use crate::validate::{validate_doc_id, validate_source_id, validate_text, DEFAULT_MAX_TEXT_BYTES};

use super::types::{
    should_skip_upsert, BatchCaps, DeltaOptions, DeltaOutcome, DeltaResult, DeltaStatus,
};
use super::write::MsaIndex;

impl MsaIndex {
    /// Apply an upsert + delete batch atomically: a single writer commit, with
    /// rollback on any error (Codex design audit §6). This is the autonomy core
    /// the `msa_sync_delta` / `msa_index_batch` / `msa_delete_batch` tools build
    /// on — a client crash can never leave the collection half-synced.
    ///
    /// Order: validate everything → reject a doc_id present in both sides →
    /// check the optimistic preconditions (`if_index_profile_hash`, then
    /// `if_manifest_digest` over `source_id` scope) **before** any write →
    /// classify each upsert (skip-unchanged vs reindex) and each delete
    /// (deleted vs not_found) against the committed state → stage deletes +
    /// upserts → one commit → rollback on failure. `new_manifest_digest` is the
    /// scope digest after the commit (the client's next precondition value).
    pub fn apply_delta(
        &self,
        upserts: &[Document],
        deletes: &[&str],
        embedder: Option<&dyn DenseEncoder>,
        opts: &DeltaOptions,
        caps: BatchCaps,
    ) -> Result<DeltaOutcome> {
        // ---- 1. caps -------------------------------------------------------
        let n = upserts.len() + deletes.len();
        if n > caps.max_docs {
            return Err(MsaError::InvalidInput(format!(
                "batch too large: {n} docs (max {})",
                caps.max_docs
            )));
        }
        let total_text: usize = upserts.iter().map(|d| d.text.len()).sum();
        if total_text > caps.max_text_bytes {
            return Err(MsaError::InvalidInput(format!(
                "batch text too large: {total_text} bytes (max {})",
                caps.max_text_bytes
            )));
        }

        // ---- 2. validate inputs -------------------------------------------
        if let Some(sid) = &opts.source_id {
            validate_source_id(sid)?;
        }
        for d in upserts {
            validate_doc_id(&d.id)?;
            validate_text(&d.text, DEFAULT_MAX_TEXT_BYTES)?;
            reject_reserved_metadata(&d.metadata)?;
        }
        for id in deletes {
            validate_doc_id(id)?;
        }

        // ---- 3. no doc_id in both upsert and delete -----------------------
        let del_set: std::collections::HashSet<&str> = deletes.iter().copied().collect();
        if let Some(d) = upserts.iter().find(|d| del_set.contains(d.id.as_str())) {
            return Err(MsaError::InvalidInput(format!(
                "doc_id {:?} appears in both upserts and deletes",
                d.id
            )));
        }

        // Acquire the delta lock for the whole read-modify-write: preconditions,
        // classification (which read committed state), staging and commit. No
        // other write can commit in this window, so the optimistic-concurrency
        // preconditions below are atomic w.r.t. commits (Codex final audit §1/§2).
        let _delta = self.delta_lock.lock().expect("delta lock poisoned");

        // ---- 4. preconditions (before any write) --------------------------
        let current_profile = self.current_index_profile_hash(embedder);
        if let Some(want) = &opts.if_index_profile_hash {
            if want != &current_profile {
                return Err(MsaError::Conflict(format!(
                    "if_index_profile_hash mismatch: client {want}, current {current_profile}"
                )));
            }
        }
        if let Some(want) = &opts.if_manifest_digest {
            let cur = self
                .manifest(&current_profile, opts.source_id.as_deref(), None, 1, None)?
                .manifest_digest;
            if want != &cur {
                return Err(MsaError::Conflict(format!(
                    "if_manifest_digest mismatch: client {want}, current {cur}"
                )));
            }
        }

        // ---- 5. classify upserts (skip-unchanged) against committed state --
        let mut to_index: Vec<(&Document, String)> = Vec::new();
        let mut results: Vec<DeltaResult> = Vec::new();
        let mut unchanged = 0usize;
        let mut skipped_scope_mismatch = 0usize;
        for d in upserts {
            let ch = content_hash_v1(&d.text, &d.metadata);
            let stored = self.doc_fingerprint(&d.id)?;
            // Scoped upsert never overwrites a doc_id owned by another source.
            // Checked BEFORE skip-unchanged so a cross-source collision cannot
            // hide as "unchanged" either (Codex final audit §3).
            if let (Some(scope), Some(fp)) = (&opts.source_id, &stored) {
                if fp.source_id.as_deref() != Some(scope.as_str()) {
                    skipped_scope_mismatch += 1;
                    results.push(DeltaResult {
                        doc_id: d.id.clone(),
                        status: DeltaStatus::SkippedScopeMismatch,
                        content_hash: Some(ch),
                        num_chunks: None,
                    });
                    continue;
                }
            }
            if opts.skip_unchanged && should_skip_upsert(stored.as_ref(), &ch, &current_profile) {
                unchanged += 1;
                results.push(DeltaResult {
                    doc_id: d.id.clone(),
                    status: DeltaStatus::Unchanged,
                    content_hash: Some(ch),
                    num_chunks: None,
                });
            } else {
                to_index.push((d, ch));
            }
        }

        // ---- 6. classify deletes (deleted | not_found | scope_mismatch) ---
        // `skipped_scope_mismatch` is shared with the upsert pass above.
        let mut deleted = 0usize;
        let mut not_found = 0usize;
        let mut effective_deletes: Vec<&str> = Vec::new();
        let mut delete_results: Vec<DeltaResult> = Vec::with_capacity(deletes.len());
        for id in deletes {
            let status = match self.doc_fingerprint(id)? {
                None => {
                    not_found += 1;
                    DeltaStatus::NotFound
                }
                Some(fp) => {
                    // Scoped delete never evicts a doc owned by another source.
                    let scope_mismatch = opts
                        .source_id
                        .as_deref()
                        .is_some_and(|scope| fp.source_id.as_deref() != Some(scope));
                    if scope_mismatch {
                        skipped_scope_mismatch += 1;
                        DeltaStatus::SkippedScopeMismatch
                    } else {
                        deleted += 1;
                        effective_deletes.push(id);
                        DeltaStatus::Deleted
                    }
                }
            };
            delete_results.push(DeltaResult {
                doc_id: (*id).to_string(),
                status,
                content_hash: None,
                num_chunks: None,
            });
        }

        // ---- 7. single transaction: stage deletes + upserts, one commit ---
        let chunk_counts = self.with_writer(|writer| {
            let staged = (|| {
                for id in &effective_deletes {
                    writer.delete_term(Term::from_field_text(self.fields.doc_id, id));
                }
                let mut counts = Vec::with_capacity(to_index.len());
                for (d, _) in &to_index {
                    let c = Self::stage_document(
                        &self.fields,
                        &self.chunking,
                        writer,
                        d,
                        embedder,
                        &current_profile,
                        opts.source_id.as_deref(),
                    )?;
                    counts.push(c);
                }
                Ok::<Vec<usize>, MsaError>(counts)
            })();
            match staged {
                Ok(counts) => {
                    if let Err(e) = writer.commit() {
                        let _ = writer.rollback();
                        return Err(e.into());
                    }
                    Ok(counts)
                }
                Err(e) => {
                    let _ = writer.rollback();
                    Err(e)
                }
            }
        })?;

        // ---- 8. assemble results ------------------------------------------
        let indexed = to_index.len();
        for ((d, ch), c) in to_index.iter().zip(chunk_counts.iter()) {
            results.push(DeltaResult {
                doc_id: d.id.clone(),
                status: DeltaStatus::Indexed,
                content_hash: Some(ch.clone()),
                num_chunks: Some(*c),
            });
        }
        results.extend(delete_results);

        // ---- 9. fresh scope digest for the client's next precondition -----
        let new_manifest_digest = self
            .manifest(&current_profile, opts.source_id.as_deref(), None, 1, None)?
            .manifest_digest;

        Ok(DeltaOutcome {
            indexed,
            unchanged,
            deleted,
            not_found,
            skipped_scope_mismatch,
            new_manifest_digest,
            results,
        })
    }
}
