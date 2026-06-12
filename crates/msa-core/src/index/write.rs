//! `MsaIndex` construction and write path: open, index_document(s),
//! delete_document, release_writer. The shared staging helper (`stage_document`)
//! lives here too — it is the single code path that turns a `Document` into
//! tantivy records, so the chunks+sentinel invariant has exactly one owner.

use std::path::PathBuf;
use std::sync::Mutex;

use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, Term};

use crate::chunker::chunk_text;
use crate::embeddings::{pack_f32, DenseEncoder, EmbedInputKind};
use crate::error::{MsaError, Result};
use crate::fingerprint::{
    content_hash_v1, index_profile_hash_v1, reject_reserved_metadata, IndexProfile,
    CONTENT_HASH_ALGO, RESERVED_MSA_KEY,
};
use crate::schema::{ChunkConfig, Document};

use super::schema_io::{
    build_schema, CollectionManifest, Fields, SCHEMA_VERSION, SENTINEL_CHUNK_IDX,
};

// ── MsaIndex struct ───────────────────────────────────────────────────────────

/// Per-collection tantivy index — one on-disk index per collection.
///
/// Schema (tantivy fields):
/// - `doc_id`     STRING | STORED  — sanitized document id
/// - `chunk_idx`  FAST   | STORED  — `u64`. Chunks use 0..N. The full-doc
///   sentinel uses `u64::MAX`.
/// - `is_sentinel` FAST            — 0 for chunk records, 1 for the full-doc
///   sentinel. Search excludes sentinels.
/// - `chunk_text` TEXT   | STORED  — chunk content, BM25-searchable
/// - `full_text`  STORED           — only set on the sentinel
/// - `char_start`, `char_end` FAST | STORED — `u64`
/// - `created_at` FAST   | STORED  — unix seconds
/// - `metadata`   STORED           — serialized JSON object
///
/// Persistence is handled by tantivy itself (mmap on disk). No WAL needed
/// for v0.2; tantivy commits are atomic.
pub struct MsaIndex {
    pub(super) name: String,
    pub(super) path: PathBuf,
    pub(super) chunking: ChunkConfig,
    /// Writer heap arena in bytes, clamped to tantivy's supported range.
    pub(super) writer_heap_bytes: usize,
    pub(super) fields: Fields,
    pub(super) index: Index,
    pub(super) reader: IndexReader,
    /// Long-lived writer, created lazily on the first write and reused across
    /// calls. Read-only paths (search/fetch/stats) never allocate it, so they
    /// cost no arena RAM. The `Mutex` serializes writes on a single collection
    /// (no two 50 MB arenas opened in parallel), and tantivy itself disallows
    /// more than one writer per index.
    pub(super) writer: Mutex<Option<IndexWriter>>,
    /// Serializes the full read-modify-write of every write path. The `writer`
    /// mutex only serializes commits; this is held across
    /// precondition-read → classify → stage → commit so `apply_delta`'s
    /// optimistic-concurrency preconditions (`if_manifest_digest` /
    /// `if_index_profile_hash`) are atomic w.r.t. any concurrent commit
    /// (Codex final audit §1/§2). Acquired before the `writer` mutex; never the
    /// reverse, so no deadlock.
    pub(super) delta_lock: Mutex<()>,
}

impl std::fmt::Debug for MsaIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MsaIndex")
            .field("name", &self.name)
            .field("path", &self.path)
            .field("chunking", &self.chunking)
            .field("writer_heap_bytes", &self.writer_heap_bytes)
            .finish_non_exhaustive()
    }
}

// ── construction ──────────────────────────────────────────────────────────────

impl MsaIndex {
    /// Open or create the on-disk tantivy index for `name` rooted at `path`.
    ///
    /// Field handles are resolved **from the schema actually on disk** (by
    /// name), not from a freshly-built schema, so an older collection that
    /// lacks a field (e.g. `token_count`) is read correctly instead of having
    /// its field handles silently point at the wrong columns. A small
    /// `manifest.json` records the schema version; opening an index written by
    /// a newer version is refused rather than misread.
    ///
    /// **Path-safety note**: this takes an arbitrary `path` and does NOT
    /// validate `name` as a collection identifier. The safe public entry point
    /// is [`super::registry::CollectionRegistry::open_or_create`], which validates
    /// the name (rejecting traversal/separators) before deriving the path.
    /// Callers that build paths directly are responsible for their own containment.
    ///
    /// Uses the default writer heap arena
    /// ([`crate::config::default_writer_heap_bytes`]). Use
    /// [`Self::open_with_heap`] to override on big machines.
    pub fn open(name: String, path: PathBuf, chunking: ChunkConfig) -> Result<Self> {
        Self::open_with_heap(
            name,
            path,
            chunking,
            crate::config::default_writer_heap_bytes(),
        )
    }

    /// Like [`Self::open`] but with an explicit writer heap arena in bytes,
    /// clamped to tantivy's supported range. A small arena (default 15 MiB)
    /// bounds peak indexing RAM on weak hardware.
    pub fn open_with_heap(
        name: String,
        path: PathBuf,
        chunking: ChunkConfig,
        writer_heap_bytes: usize,
    ) -> Result<Self> {
        std::fs::create_dir_all(&path)?;

        let existing = CollectionManifest::read(&path)?;
        if let Some(m) = &existing {
            if m.schema_version > SCHEMA_VERSION {
                return Err(MsaError::Schema(format!(
                    "collection '{name}' was written by schema version {} but this \
                     binary supports up to {SCHEMA_VERSION}; upgrade the binary",
                    m.schema_version
                )));
            }
        }

        let index = match Index::open_in_dir(&path) {
            Ok(idx) => idx,
            Err(_) => {
                let idx = Index::create_in_dir(&path, build_schema())?;
                CollectionManifest::write(&path, SCHEMA_VERSION)?;
                idx
            }
        };

        // Resolve fields from the real on-disk schema. Core fields are
        // required; `token_count` is optional for forward/backward compat.
        let schema = index.schema();
        let req = |n: &str| -> Result<tantivy::schema::Field> {
            schema
                .get_field(n)
                .map_err(|_| MsaError::Schema(format!("collection '{name}' missing field '{n}'")))
        };
        let fields = Fields {
            doc_id: req("doc_id")?,
            chunk_idx: req("chunk_idx")?,
            is_sentinel: req("is_sentinel")?,
            chunk_text: req("chunk_text")?,
            full_text: req("full_text")?,
            char_start: req("char_start")?,
            char_end: req("char_end")?,
            created_at: req("created_at")?,
            metadata: req("metadata")?,
            embedding: req("embedding")?,
            token_count: schema.get_field("token_count").ok(),
            content_hash: schema.get_field("content_hash").ok(),
            index_profile_hash: schema.get_field("index_profile_hash").ok(),
            source_id: schema.get_field("source_id").ok(),
        };

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        // Clamp to tantivy's supported writer budget. The minimum is enforced
        // by tantivy; we clamp here too (to the same constant the default uses)
        // so a tiny config value degrades gracefully instead of erroring at
        // first write.
        const TANTIVY_MAX_HEAP: usize = u32::MAX as usize - 1_000_000;
        let writer_heap_bytes =
            writer_heap_bytes.clamp(crate::config::MIN_WRITER_HEAP_BYTES, TANTIVY_MAX_HEAP);

        Ok(Self {
            name,
            path,
            chunking,
            writer_heap_bytes,
            fields,
            index,
            reader,
            writer: Mutex::new(None),
            delta_lock: Mutex::new(()),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Obtain the long-lived writer, creating it on first use. The returned
    /// guard holds the writer mutex, serializing all writes on this index.
    pub(super) fn with_writer<R>(
        &self,
        f: impl FnOnce(&mut IndexWriter) -> Result<R>,
    ) -> Result<R> {
        let mut guard = self.writer.lock().expect("writer mutex poisoned");
        if guard.is_none() {
            *guard = Some(self.index.writer(self.writer_heap_bytes)?);
        }
        let writer = guard.as_mut().expect("writer just initialized");
        f(writer)
    }

    /// Past this many searchable segments a successful write triggers an
    /// explicit merge (F2, vivling live audit 2026-06-07: a commit-per-capsule
    /// workload accumulated 4027 segments for 4233 docs, opening ~4k segment
    /// readers per recall). EXPLICIT and synchronous by design — the native
    /// LogMergePolicy merges asynchronously and contends with the writer at
    /// uncontrolled moments (DeepSeek co-review); here the merge runs only
    /// inside our own write path, while we already hold the writer mutex.
    /// Readers keep searching the pre-merge segments meanwhile.
    pub(super) const MAX_SEGMENTS_BEFORE_MERGE: usize = 24;

    /// Merge all searchable segments when the count exceeds the threshold.
    /// Best-effort: a merge failure leaves the index correct (just
    /// fragmented), so it logs and moves on.
    pub(super) fn maybe_merge_segments(&self, writer: &mut IndexWriter) {
        let ids = match self.index.searchable_segment_ids() {
            Ok(ids) => ids,
            Err(e) => {
                tracing::warn!("segment id listing failed, skipping merge: {e}");
                return;
            }
        };
        if ids.len() <= Self::MAX_SEGMENTS_BEFORE_MERGE {
            return;
        }
        tracing::info!(
            "merging {} segments (threshold {})",
            ids.len(),
            Self::MAX_SEGMENTS_BEFORE_MERGE
        );
        match writer.merge(&ids).wait() {
            Ok(_) => {
                if let Err(e) = self.reader.reload() {
                    tracing::warn!("reader reload after merge failed: {e}");
                }
            }
            Err(e) => tracing::warn!("segment merge failed (index stays fragmented): {e}"),
        }
    }

    /// The chunking + embedding identity at index time, for the index-profile
    /// fingerprint (skip-unchanged must reindex when chunking or the dense model
    /// changes). `embedding_backend` is left `None` — the encoder trait does not
    /// expose it; model_id/version/profile/dim identify the model.
    pub(super) fn index_profile(&self, embedder: Option<&dyn DenseEncoder>) -> IndexProfile {
        match embedder {
            Some(e) => IndexProfile {
                chunk_size: self.chunking.chunk_size,
                overlap: self.chunking.overlap,
                embedding_active: true,
                embedding_backend: None,
                embedding_model_id: Some(e.model_id().to_string()),
                embedding_model_version: Some(e.model_version().to_string()),
                embedding_profile: Some(e.profile().to_string()),
                embedding_dim: Some(e.dim()),
            },
            None => IndexProfile {
                chunk_size: self.chunking.chunk_size,
                overlap: self.chunking.overlap,
                embedding_active: false,
                ..Default::default()
            },
        }
    }

    /// Stage one document's records (chunks + full-text sentinel) into `writer`
    /// without committing. Existing records for the same `doc_id` are deleted
    /// first so a reindex replaces rather than duplicates. Shared by the single
    /// and bulk index paths so the chunks+sentinel invariant lives in one place.
    ///
    /// The A6 fingerprint (`content_hash` + `index_profile_hash` + `source_id`)
    /// is written into the sentinel's `_msa` metadata block (compat layer,
    /// always present) and, when the on-disk schema is v3, into the typed
    /// sentinel fields too. Chunk records keep the *user* metadata only.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn stage_document(
        fields: &Fields,
        chunking: &ChunkConfig,
        writer: &IndexWriter,
        doc: &Document,
        embedder: Option<&dyn DenseEncoder>,
        profile_hash: &str,
        source_id: Option<&str>,
    ) -> Result<usize> {
        // Client metadata must not spoof the reserved server block.
        reject_reserved_metadata(&doc.metadata)?;

        writer.delete_term(Term::from_field_text(fields.doc_id, &doc.id));

        let chunks = chunk_text(&doc.text, chunking);
        let user_metadata_json = serde_json::to_string(&doc.metadata)?;
        let created_unix = doc.created_at.timestamp();
        let content_hash = content_hash_v1(&doc.text, &doc.metadata);

        for (idx, c) in chunks.iter().enumerate() {
            let embedding_bytes: Vec<u8> = match embedder {
                Some(e) => pack_f32(&e.encode_kind(EmbedInputKind::Passage, &c.text)?),
                None => Vec::new(),
            };
            let mut d = doc!(
                fields.doc_id => doc.id.clone(),
                fields.chunk_idx => idx as u64,
                fields.is_sentinel => 0u64,
                fields.chunk_text => c.text.clone(),
                fields.char_start => c.char_offset.0 as u64,
                fields.char_end => c.char_offset.1 as u64,
                fields.created_at => created_unix,
                fields.metadata => user_metadata_json.clone(),
                fields.embedding => embedding_bytes,
            );
            if let Some(tc) = fields.token_count {
                d.add_u64(tc, c.token_count as u64);
            }
            writer.add_document(d)?;
        }

        // Sentinel metadata = user metadata + the reserved `_msa` block. This
        // block is the cross-version source of truth for the fingerprint;
        // `fetch_doc`/`search` strip it so the user never sees it.
        let mut sentinel_meta = doc.metadata.clone();
        sentinel_meta.insert(
            RESERVED_MSA_KEY.to_string(),
            serde_json::json!({
                "content_hash": content_hash,
                "content_hash_algo": CONTENT_HASH_ALGO,
                "index_profile_hash": profile_hash,
                "source_id": source_id,
            }),
        );
        let sentinel_meta_json = serde_json::to_string(&sentinel_meta)?;

        // Sentinel record holding the full text for `fetch_doc`. No embedding.
        let mut s = doc!(
            fields.doc_id => doc.id.clone(),
            fields.chunk_idx => SENTINEL_CHUNK_IDX,
            fields.is_sentinel => 1u64,
            fields.full_text => doc.text.clone(),
            fields.created_at => created_unix,
            fields.metadata => sentinel_meta_json,
            fields.embedding => Vec::<u8>::new(),
        );
        // Typed v3 fields (accelerator) when the on-disk schema has them.
        if let Some(f) = fields.content_hash {
            s.add_text(f, &content_hash);
        }
        if let Some(f) = fields.index_profile_hash {
            s.add_text(f, profile_hash);
        }
        if let Some(f) = fields.source_id {
            s.add_text(f, source_id.unwrap_or(""));
        }
        writer.add_document(s)?;

        Ok(chunks.len())
    }

    // ── public write API ──────────────────────────────────────────────────────

    /// Index (or reindex) a single document. Existing records for this
    /// `doc_id` are replaced. Thin wrapper over [`Self::index_documents`] so
    /// there is exactly **one** staging+commit path with rollback-on-error.
    /// This matters because the writer is now long-lived: an error during
    /// staging must roll back the uncommitted records, otherwise a later
    /// write's `commit()` would publish them. When `embedder` is `Some`, each
    /// chunk's dense embedding is computed and stored for hybrid rerank.
    pub fn index_document(
        &self,
        doc: &Document,
        embedder: Option<&dyn DenseEncoder>,
    ) -> Result<usize> {
        self.index_documents(std::slice::from_ref(doc), embedder)
    }

    /// Index a batch of documents under a **single** commit. This is the
    /// throughput path: one writer, one fsync at the end, instead of a writer
    /// arena + commit per document. The chunks+sentinel of each document are
    /// staged atomically; on any error the in-progress (uncommitted) batch is
    /// rolled back so nothing partial is persisted (and the long-lived writer
    /// is left clean for the next call). Returns total chunks.
    pub fn index_documents(
        &self,
        docs: &[Document],
        embedder: Option<&dyn DenseEncoder>,
    ) -> Result<usize> {
        // Serialize the whole write against other writes (so apply_delta's
        // preconditions stay atomic vs commits). Held until this returns.
        let _delta = self.delta_lock.lock().expect("delta lock poisoned");
        // The index profile (chunking + embedding identity) is identical for
        // every doc in the batch — compute its hash once. A profile change
        // (chunk size, dense model) changes this, so skip-unchanged downstream
        // correctly forces a reindex even on identical text.
        let profile_hash = index_profile_hash_v1(&self.index_profile(embedder));
        self.with_writer(|writer| {
            let mut total = 0usize;
            let staged = (|| {
                for doc in docs {
                    total += Self::stage_document(
                        &self.fields,
                        &self.chunking,
                        writer,
                        doc,
                        embedder,
                        &profile_hash,
                        None,
                    )?;
                }
                Ok::<(), MsaError>(())
            })();
            match staged {
                Ok(()) => {
                    if let Err(e) = writer.commit() {
                        // A failed commit leaves the writer state uncertain;
                        // roll back so the next call starts clean.
                        let _ = writer.rollback();
                        return Err(e.into());
                    }
                    self.maybe_merge_segments(writer);
                    Ok(total)
                }
                Err(e) => {
                    // Drop the uncommitted batch; the index stays at the last
                    // committed state and the writer is reusable.
                    let _ = writer.rollback();
                    Err(e)
                }
            }
        })
    }

    pub fn delete_document(&self, doc_id: &str) -> Result<()> {
        let _delta = self.delta_lock.lock().expect("delta lock poisoned");
        self.with_writer(|writer| {
            writer.delete_term(Term::from_field_text(self.fields.doc_id, doc_id));
            if let Err(e) = writer.commit() {
                let _ = writer.rollback();
                return Err(e.into());
            }
            Ok(())
        })
    }

    /// Release the long-lived writer, freeing its heap arena (~15 MiB).
    ///
    /// Multi-tenant deployments (one collection per Vivling) accumulate one
    /// writer arena per collection that has ever been written. On RPi3-class
    /// hardware that adds up, so an idle collection can drop its writer here;
    /// the next write lazily recreates it. Returns `true` if a writer was
    /// actually dropped. Read paths are unaffected (they never hold a writer).
    pub fn release_writer(&self) -> bool {
        let mut guard = self.writer.lock().expect("writer mutex poisoned");
        guard.take().is_some()
    }
}
