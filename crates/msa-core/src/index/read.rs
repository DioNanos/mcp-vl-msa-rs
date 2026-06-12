//! `MsaIndex` read path: search (BM25 + hybrid), fetch_doc, stats, manifest,
//! doc_fingerprint, current_index_profile_hash.

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::{TantivyDocument, Term};

use crate::embeddings::{cosine_unit, unpack_f32, DenseEncoder};
use crate::error::{MsaError, Result};
use crate::fingerprint::{index_profile_hash_v1, RESERVED_MSA_KEY};
use crate::schema::{ChunkHit, Document, MsaStats, SearchFilter};
use crate::validate::{validate_doc_id_filter, validate_source_id};

use super::schema_io::{
    dir_size, filter_matches, first_bytes, first_i64, first_text, first_u64, SENTINEL_CHUNK_IDX,
};
use super::types::{
    classify_hash_state, DocFingerprint, Manifest, ManifestEntry, MAX_MANIFEST_LIMIT,
};
use super::write::MsaIndex;

impl MsaIndex {
    /// The hash of the index profile this collection would write *now* (given
    /// `embedder`). Used by [`Self::manifest`] to flag `profile_stale` docs and
    /// by the sync path to decide skip-unchanged.
    pub fn current_index_profile_hash(&self, embedder: Option<&dyn DenseEncoder>) -> String {
        index_profile_hash_v1(&self.index_profile(embedder))
    }

    /// Top-k chunk hits for `query`. BM25 score from tantivy, normalized to
    /// 0.0–1.0 via per-batch max-scaling. Sentinels are excluded.
    ///
    /// `filter` is applied **post-retrieval** in v0.2: tantivy returns up to
    /// `top_k * FILTER_OVERFETCH` candidates, then `metadata` and `created_at`
    /// are checked in Rust. This avoids per-key tantivy field declarations
    /// while staying correct for low-selectivity filters. Highly selective
    /// filters on large corpora may want query-time filtering — TODO(v0.4).
    pub fn search(
        &self,
        query: &str,
        top_k: usize,
        filter: Option<&SearchFilter>,
    ) -> Result<Vec<ChunkHit>> {
        self.search_excluding(query, top_k, filter, &HashSet::new())
    }

    /// Like [`Self::search`] but also drops chunks whose `doc_id` is in
    /// `exclude_doc_ids`. Used by `msa_search_iterative` to dedup across
    /// rounds of a Memory Interleave session.
    pub fn search_excluding(
        &self,
        query: &str,
        top_k: usize,
        filter: Option<&SearchFilter>,
        exclude_doc_ids: &HashSet<String>,
    ) -> Result<Vec<ChunkHit>> {
        self.search_hybrid(query, top_k, filter, exclude_doc_ids, 1.0, None)
    }

    /// BM25 + optional dense rerank. `dense_alpha` is the BM25 weight in
    /// `[0.0, 1.0]`: `1.0` is BM25-only, `0.0` is dense-only. `query_emb`
    /// must be `Some(&[f32])` whenever `dense_alpha < 1.0`; otherwise the
    /// call is rejected with `MsaError::InvalidQuery`. Chunks indexed
    /// without a stored embedding contribute neutral cosine `0.5`.
    pub fn search_hybrid(
        &self,
        query: &str,
        top_k: usize,
        filter: Option<&SearchFilter>,
        exclude_doc_ids: &HashSet<String>,
        dense_alpha: f32,
        query_emb: Option<&[f32]>,
    ) -> Result<Vec<ChunkHit>> {
        if query.trim().is_empty() {
            return Err(MsaError::InvalidQuery("query is empty".into()));
        }
        if !(0.0..=1.0).contains(&dense_alpha) {
            return Err(MsaError::InvalidQuery(format!(
                "dense_alpha must be in [0.0, 1.0], got {dense_alpha}"
            )));
        }
        let dense_active = dense_alpha < 1.0;
        if dense_active && query_emb.is_none() {
            return Err(MsaError::InvalidQuery(
                "dense_alpha < 1.0 requires a query embedding".into(),
            ));
        }

        self.reader.reload()?;
        let searcher = self.reader.searcher();
        let qp = QueryParser::for_index(&self.index, vec![self.fields.chunk_text]);
        let parsed = qp
            .parse_query(query)
            .map_err(|e| MsaError::InvalidQuery(e.to_string()))?;

        // Overfetch when a filter, exclude list, or dense rerank is active
        // so post-processing still has a decent chance of returning top_k.
        const POSTPROCESS_OVERFETCH: usize = 4;
        const MAX_OVERFETCH: usize = 1024;
        let needs_overfetch = filter.is_some() || !exclude_doc_ids.is_empty() || dense_active;
        let fetch_limit = if needs_overfetch {
            (top_k.saturating_mul(POSTPROCESS_OVERFETCH)).min(MAX_OVERFETCH)
        } else {
            top_k
        }
        .max(1);

        let top = searcher.search(&parsed, &TopDocs::with_limit(fetch_limit))?;
        let max_raw = top
            .iter()
            .map(|(s, _)| *s)
            .fold(f32::NEG_INFINITY, f32::max);

        // First pass: collect candidates after filter + exclude. Score is
        // BM25-normalized; dense rerank happens in a second pass below.
        let mut candidates: Vec<ChunkHit> = Vec::with_capacity(top.len());
        let mut chunk_embeddings: Vec<Vec<f32>> = Vec::with_capacity(top.len());
        for (raw_score, addr) in top {
            let stored: TantivyDocument = searcher.doc(addr)?;
            let chunk_idx = first_u64(&stored, self.fields.chunk_idx).unwrap_or(0) as usize;
            if chunk_idx as u64 == SENTINEL_CHUNK_IDX as usize as u64 {
                continue;
            }
            let doc_id = first_text(&stored, self.fields.doc_id).unwrap_or_default();
            if exclude_doc_ids.contains(&doc_id) {
                continue;
            }
            let snippet = first_text(&stored, self.fields.chunk_text).unwrap_or_default();
            let cs = first_u64(&stored, self.fields.char_start).unwrap_or(0) as usize;
            let ce = first_u64(&stored, self.fields.char_end).unwrap_or(0) as usize;
            let metadata = first_text(&stored, self.fields.metadata)
                .and_then(|s| serde_json::from_str::<HashMap<String, serde_json::Value>>(&s).ok())
                .unwrap_or_default();
            let created_unix = first_i64(&stored, self.fields.created_at).unwrap_or(0);
            if let Some(f) = filter {
                if !filter_matches(f, &metadata, created_unix) {
                    continue;
                }
            }

            let normalized = if max_raw > 0.0 {
                raw_score / max_raw
            } else {
                0.0
            };
            let cached_emb = if dense_active {
                first_bytes(&stored, self.fields.embedding)
                    .map(|b| unpack_f32(&b))
                    .unwrap_or_default()
            } else {
                Vec::new()
            };

            candidates.push(ChunkHit {
                doc_id,
                chunk_idx,
                score: normalized,
                raw_score,
                snippet,
                char_offset: (cs, ce),
                metadata,
            });
            chunk_embeddings.push(cached_emb);
        }

        if dense_active {
            let q = query_emb.expect("query_emb required when dense_active");
            for (hit, emb) in candidates.iter_mut().zip(chunk_embeddings.iter()) {
                let cos = if emb.is_empty() {
                    0.5
                } else {
                    cosine_unit(q, emb)
                };
                hit.score = dense_alpha * hit.score + (1.0 - dense_alpha) * cos;
            }
            candidates.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        candidates.truncate(top_k);
        Ok(candidates)
    }

    /// Fetch the full original text of a document via the sentinel record.
    pub fn fetch_doc(&self, doc_id: &str) -> Result<Document> {
        self.reader.reload()?;
        let searcher = self.reader.searcher();

        // Use a term query on doc_id; iterate stored docs to find the sentinel.
        let term = Term::from_field_text(self.fields.doc_id, doc_id);
        let q = tantivy::query::TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic);
        let top = searcher.search(&q, &TopDocs::with_limit(1024))?;

        for (_, addr) in top {
            let stored: TantivyDocument = searcher.doc(addr)?;
            let is_sentinel = first_u64(&stored, self.fields.is_sentinel).unwrap_or(0);
            if is_sentinel == 1 {
                let full_text = first_text(&stored, self.fields.full_text).unwrap_or_default();
                let mut metadata: std::collections::HashMap<String, serde_json::Value> =
                    first_text(&stored, self.fields.metadata)
                        .and_then(|s| serde_json::from_str(&s).ok())
                        .unwrap_or_default();
                // The reserved server fingerprint block is never exposed.
                metadata.remove(RESERVED_MSA_KEY);
                let created_unix = first_i64(&stored, self.fields.created_at).unwrap_or(0);
                let created_at =
                    DateTime::<Utc>::from_timestamp(created_unix, 0).unwrap_or_else(Utc::now);
                return Ok(Document {
                    id: doc_id.to_string(),
                    text: full_text,
                    metadata,
                    created_at,
                });
            }
        }
        Err(MsaError::UnknownDocument {
            collection: self.name.clone(),
            doc_id: doc_id.to_string(),
        })
    }

    /// Read the A6 fingerprint of one document from its sentinel. Typed v3
    /// fields take precedence; on a legacy (v2) collection those fields are
    /// absent, so it falls back to the `_msa` metadata block written on the
    /// last reindex (self-heal). `Ok(None)` if the document has no sentinel.
    /// This is the per-doc primitive the paginated manifest (A6.2) builds on.
    pub fn doc_fingerprint(&self, doc_id: &str) -> Result<Option<DocFingerprint>> {
        self.reader.reload()?;
        let searcher = self.reader.searcher();
        let term = Term::from_field_text(self.fields.doc_id, doc_id);
        let q = tantivy::query::TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic);
        let top = searcher.search(&q, &TopDocs::with_limit(1024))?;
        for (_, addr) in top {
            let stored: TantivyDocument = searcher.doc(addr)?;
            if first_u64(&stored, self.fields.is_sentinel).unwrap_or(0) == 1 {
                return Ok(Some(self.read_fingerprint(&stored)));
            }
        }
        Ok(None)
    }

    /// Extract the fingerprint from a sentinel doc: typed v3 fields first, else
    /// the `_msa` metadata block (legacy compat).
    pub(super) fn read_fingerprint(&self, stored: &TantivyDocument) -> DocFingerprint {
        let nonempty = |f: Option<tantivy::schema::Field>| -> Option<String> {
            f.and_then(|fld| first_text(stored, fld))
                .filter(|s| !s.is_empty())
        };
        let v3_content = nonempty(self.fields.content_hash);
        if v3_content.is_some() {
            return DocFingerprint {
                content_hash: v3_content,
                index_profile_hash: nonempty(self.fields.index_profile_hash),
                source_id: nonempty(self.fields.source_id),
            };
        }
        // Fallback to the `_msa` block in the sentinel metadata.
        let block = first_text(stored, self.fields.metadata)
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| v.get(RESERVED_MSA_KEY).cloned());
        let s = |b: &serde_json::Value, k: &str| -> Option<String> {
            b.get(k).and_then(|x| x.as_str()).map(String::from)
        };
        match block {
            Some(b) => DocFingerprint {
                content_hash: s(&b, "content_hash"),
                index_profile_hash: s(&b, "index_profile_hash"),
                source_id: s(&b, "source_id"),
            },
            None => DocFingerprint::default(),
        }
    }

    /// Exact per-collection statistics, computed from fast-field columns only
    /// (no stored-document loads), so it stays cheap on weak hardware.
    ///
    /// Invariant: `index_document` writes exactly one sentinel record per
    /// document (`is_sentinel == 1`) plus one record per chunk. Therefore
    /// `num_documents` == number of live sentinels and `num_chunks` == number
    /// of live non-sentinel records. `total_tokens` sums the `token_count`
    /// fast field over chunk records; it reports 0 for legacy indices created
    /// before that field existed (`token_count: None`).
    pub fn stats(&self) -> Result<MsaStats> {
        self.reader.reload()?;
        let searcher = self.reader.searcher();

        let mut num_documents: u64 = 0;
        let mut num_chunks: u64 = 0;
        let mut total_tokens: u64 = 0;

        for segment_reader in searcher.segment_readers() {
            let ff = segment_reader.fast_fields();
            let sentinel_col = ff.u64("is_sentinel")?;
            let token_col = self
                .fields
                .token_count
                .map(|_| ff.u64("token_count"))
                .transpose()
                .ok()
                .flatten();
            let alive = segment_reader.alive_bitset();

            for doc in 0..segment_reader.max_doc() {
                if let Some(bs) = alive {
                    if !bs.is_alive(doc) {
                        continue;
                    }
                }
                let is_sentinel = sentinel_col.first(doc).unwrap_or(0);
                if is_sentinel == 1 {
                    num_documents += 1;
                } else {
                    num_chunks += 1;
                    if let Some(col) = &token_col {
                        total_tokens += col.first(doc).unwrap_or(0);
                    }
                }
            }
        }

        let disk_bytes = dir_size(&self.path).unwrap_or(0);
        Ok(MsaStats {
            collection: self.name.clone(),
            num_documents,
            num_chunks,
            total_tokens,
            disk_bytes,
        })
    }

    /// A paginated manifest of the collection: one row per document with its
    /// fingerprint and freshness (`hash_state`) versus `current_profile_hash`.
    /// Sorted by `doc_id` for stable pagination and a stable `manifest_digest`.
    ///
    /// `total_count` and `manifest_digest` cover the whole filtered scope, not
    /// just the returned page, so a client can diff its corpus against the
    /// index in O(δ) and reuse `manifest_digest` as a sync precondition.
    /// `cursor` is the last `doc_id` of the previous page (exclusive lower
    /// bound). `limit` is clamped to `[1, MAX_MANIFEST_LIMIT]`.
    pub fn manifest(
        &self,
        current_profile_hash: &str,
        source_id_filter: Option<&str>,
        doc_id_prefix: Option<&str>,
        limit: usize,
        cursor: Option<&str>,
    ) -> Result<Manifest> {
        if let Some(sid) = source_id_filter {
            validate_source_id(sid)?;
        }
        if let Some(pfx) = doc_id_prefix {
            validate_doc_id_filter(pfx)?;
        }
        if let Some(c) = cursor {
            validate_doc_id_filter(c)?;
        }
        self.reader.reload()?;
        let searcher = self.reader.searcher();

        // Collect every matching sentinel's fingerprint. Sentinels are found via
        // the is_sentinel fast field (like `stats`), then their stored fields
        // are loaded to read doc_id + fingerprint.
        let mut all: Vec<ManifestEntry> = Vec::new();
        for (ord, segment_reader) in searcher.segment_readers().iter().enumerate() {
            let sentinel_col = segment_reader.fast_fields().u64("is_sentinel")?;
            let alive = segment_reader.alive_bitset();
            for doc in 0..segment_reader.max_doc() {
                if let Some(bs) = alive {
                    if !bs.is_alive(doc) {
                        continue;
                    }
                }
                if sentinel_col.first(doc).unwrap_or(0) != 1 {
                    continue;
                }
                let stored: TantivyDocument =
                    searcher.doc(tantivy::DocAddress::new(ord as u32, doc))?;
                let doc_id = first_text(&stored, self.fields.doc_id).unwrap_or_default();
                if let Some(pfx) = doc_id_prefix {
                    if !doc_id.starts_with(pfx) {
                        continue;
                    }
                }
                let fp = self.read_fingerprint(&stored);
                if let Some(sid) = source_id_filter {
                    if fp.source_id.as_deref() != Some(sid) {
                        continue;
                    }
                }
                let hash_state = classify_hash_state(&fp, current_profile_hash);
                all.push(ManifestEntry {
                    doc_id,
                    content_hash: fp.content_hash,
                    index_profile_hash: fp.index_profile_hash,
                    source_id: fp.source_id,
                    hash_state,
                });
            }
        }

        // Canonical order for stable pagination + digest.
        all.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
        let total_count = all.len();

        // Digest over the WHOLE filtered scope (doc_id + content_hash), length-
        // prefixed and domain-separated, stable across pages.
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"msa-manifest-digest-v1\0");
        for e in &all {
            hasher.update(&(e.doc_id.len() as u64).to_le_bytes());
            hasher.update(e.doc_id.as_bytes());
            let ch = e.content_hash.as_deref().unwrap_or("");
            hasher.update(&(ch.len() as u64).to_le_bytes());
            hasher.update(ch.as_bytes());
        }
        let manifest_digest = hasher.finalize().to_hex().to_string();

        // Page: entries with doc_id strictly greater than `cursor`, up to limit.
        let limit = limit.clamp(1, MAX_MANIFEST_LIMIT);
        let start = match cursor {
            Some(c) => all.partition_point(|e| e.doc_id.as_str() <= c),
            None => 0,
        };
        let end = (start + limit).min(all.len());
        let next_cursor = if end < all.len() {
            all.get(end - 1).map(|e| e.doc_id.clone())
        } else {
            None
        };
        let entries = all[start..end].to_vec();

        Ok(Manifest {
            total_count,
            manifest_digest,
            next_cursor,
            entries,
        })
    }
}
