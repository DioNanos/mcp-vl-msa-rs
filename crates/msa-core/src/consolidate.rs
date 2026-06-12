//! Cross-session consolidation: distill groups of related capsules into
//! versioned "digest" documents that pre-aggregate facts scattered across
//! sessions (design: see docs/NEGATIVE_RESULTS.md — write-time consolidation was refuted).
//!
//! Everything here is deterministic and sync except the distillation itself,
//! which is an injectable [`DistillBackend`]. Async callers (e.g. an LLM
//! adapter) use the two-phase flow instead of the trait: [`prepare`] (sync)
//! -> model call (their runtime) -> [`commit`] (sync). `msa-core` stays free
//! of tokio (same rule as `DenseEncoder` in `embeddings.rs`).
//!
//! Member capsules are never destroyed: a digest is a NEW document that
//! supersedes its members only at serving time (interleave round assembly).

use std::collections::HashMap;
use std::collections::HashSet;

use serde_json::json;

use crate::enrich::sanitize_detail;
use crate::error::{MsaError, Result};
use crate::index::MsaIndex;
use crate::schema::Document;

/// Metadata value identifying a digest document.
pub const DIGEST_KIND: &str = "digest";
/// Metadata keys written on digest documents.
pub const METADATA_KEY_KIND: &str = "kind";
pub const METADATA_KEY_DIGEST_KEY: &str = "digest_key";
pub const METADATA_KEY_MEMBER_IDS: &str = "member_ids";
pub const METADATA_KEY_MEMBER_COUNT: &str = "member_count";
pub const METADATA_KEY_VERSION: &str = "version";

/// Doc-id prefix for digest documents: `digest::{fnv1a(key)}`. Deterministic
/// id => re-indexing the same key replaces the previous version (the index
/// deletes by doc_id before adding), which keeps the "one active digest per
/// key" invariant for free.
pub const DIGEST_ID_PREFIX: &str = "digest::";

/// How to attribute a document to a "session" for the >= min_sessions gate.
#[derive(Debug, Clone)]
pub enum SessionKeySource {
    /// Read a metadata key (e.g. vivling capsules carry `day`).
    MetadataKey(String),
    /// Take the doc-id prefix up to the first `separator` (e.g. the bench
    /// uses ids of form `s3::t12`).
    DocIdPrefix { separator: String },
}

#[derive(Debug, Clone)]
pub struct ConsolidateOptions {
    pub session_key: SessionKeySource,
    /// Top TF-IDF terms kept per document for grouping.
    pub top_terms_per_doc: usize,
    /// Terms present in more than this many documents are too generic to key
    /// a group (also bounds pair-counting work).
    pub max_term_df: usize,
    /// Two docs are connected when they share at least this many top terms.
    pub min_shared_terms: usize,
    /// A group is a candidate only with at least this many member docs...
    pub min_members: usize,
    /// ...spanning at least this many distinct sessions.
    pub min_sessions: usize,
    /// An existing digest is STALE when current members exceed its recorded
    /// `member_count` by more than this margin.
    pub stale_margin: usize,
    /// Hard cap on candidates returned per call (stale-first).
    pub max_candidates: usize,
    /// Hard cap on total member text chars handed to the backend.
    pub max_member_chars: usize,
    /// Max chars of the produced digest text.
    pub digest_max_chars: usize,
}

impl Default for ConsolidateOptions {
    fn default() -> Self {
        Self {
            session_key: SessionKeySource::MetadataKey("day".to_string()),
            top_terms_per_doc: 8,
            max_term_df: 64,
            min_shared_terms: 3,
            min_members: 3,
            min_sessions: 2,
            stale_margin: 2,
            max_candidates: 3,
            max_member_chars: 12_000,
            digest_max_chars: 800,
        }
    }
}

/// A group of related capsules worth distilling.
#[derive(Debug, Clone)]
pub struct ConsolidationCandidate {
    /// Stable group key (top shared terms, sorted, joined with `+`).
    pub key: String,
    /// Member doc ids, ordered by `created_at` ascending.
    pub doc_ids: Vec<String>,
    pub total_chars: usize,
    /// True when an existing digest for this key fell behind the members.
    pub stale: bool,
}

/// Sync half #1 of the two-phase flow: everything the backend needs.
#[derive(Debug, Clone)]
pub struct DistillInput {
    pub key: String,
    /// Member texts, redacted, ordered by `created_at` ascending.
    pub member_texts: Vec<String>,
    pub member_doc_ids: Vec<String>,
    /// Version the digest will get on commit (existing + 1, or 1).
    pub next_version: u64,
    pub digest_max_chars: usize,
}

/// The committed digest document, as written to the index.
#[derive(Debug, Clone)]
pub struct ConsolidatedDigest {
    pub doc_id: String,
    pub key: String,
    pub text: String,
    pub member_doc_ids: Vec<String>,
    pub version: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum DistillError {
    #[error("distill backend failed: {0}")]
    Backend(String),
}

/// Distillation strategy. SYNC by design: msa-core must not depend on tokio.
/// Async callers skip this trait and drive [`prepare`]/[`commit`] themselves.
pub trait DistillBackend: Send + Sync {
    fn distill(
        &self,
        key: &str,
        member_texts: &[String],
        max_chars: usize,
    ) -> std::result::Result<String, DistillError>;
}

/// Date-ordered concat with sentence-level dedup. Serves as (a) the bench
/// baseline, (b) an explicit no-model mode. Never an implicit fallback.
pub struct DeterministicConcatBackend;

impl DistillBackend for DeterministicConcatBackend {
    fn distill(
        &self,
        _key: &str,
        member_texts: &[String],
        max_chars: usize,
    ) -> std::result::Result<String, DistillError> {
        let mut seen: HashSet<String> = HashSet::new();
        let mut out = String::new();
        'outer: for text in member_texts {
            for sentence in split_sentences(text) {
                let norm = normalize_for_dedup(sentence);
                if norm.is_empty() || !seen.insert(norm) {
                    continue;
                }
                let needed = sentence.chars().count() + usize::from(!out.is_empty());
                if out.chars().count() + needed > max_chars {
                    break 'outer;
                }
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(sentence.trim());
            }
        }
        Ok(out)
    }
}

/// Deterministic digest doc id for a group key.
pub fn digest_doc_id(key: &str) -> String {
    format!("{DIGEST_ID_PREFIX}{:016x}", fnv1a64(key.as_bytes()))
}

/// Scan the collection and return groups of related non-digest documents
/// that meet the membership/session gates and either have no digest yet or
/// have a stale one. Stale candidates sort first, then larger groups.
pub fn select_candidates(
    index: &MsaIndex,
    opts: &ConsolidateOptions,
) -> Result<Vec<ConsolidationCandidate>> {
    let (docs, digests) = fetch_corpus(index)?;
    if docs.len() < opts.min_members {
        return Ok(Vec::new());
    }

    // Per-doc top TF-IDF terms (document frequency computed on this corpus).
    let tokenized: Vec<Vec<String>> = docs.iter().map(|d| tokenize(&d.text)).collect();
    let mut df: HashMap<&str, usize> = HashMap::new();
    for terms in &tokenized {
        let unique: HashSet<&str> = terms.iter().map(String::as_str).collect();
        for t in unique {
            *df.entry(t).or_insert(0) += 1;
        }
    }
    let n_docs = docs.len() as f64;
    let top_terms: Vec<HashSet<&str>> = tokenized
        .iter()
        .map(|terms| {
            let mut tf: HashMap<&str, usize> = HashMap::new();
            for t in terms {
                *tf.entry(t.as_str()).or_insert(0) += 1;
            }
            let mut scored: Vec<(&str, f64)> = tf
                .into_iter()
                .filter(|(t, _)| {
                    let d = df[*t];
                    d >= 2 && d <= opts.max_term_df
                })
                .map(|(t, c)| (t, c as f64 * (n_docs / df[t] as f64).ln()))
                .collect();
            // Deterministic order: score desc, then term asc.
            scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
            scored
                .into_iter()
                .take(opts.top_terms_per_doc)
                .map(|(t, _)| t)
                .collect()
        })
        .collect();

    // Connect docs sharing >= min_shared_terms top terms (union-find over
    // a term -> docs inverted map; max_term_df bounds the pair work).
    let mut term_docs: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, terms) in top_terms.iter().enumerate() {
        for t in terms {
            term_docs.entry(t).or_default().push(i);
        }
    }
    let mut shared: HashMap<(usize, usize), usize> = HashMap::new();
    for docs_with_term in term_docs.values() {
        for (a_pos, &a) in docs_with_term.iter().enumerate() {
            for &b in &docs_with_term[a_pos + 1..] {
                *shared.entry((a, b)).or_insert(0) += 1;
            }
        }
    }
    let mut uf = UnionFind::new(docs.len());
    for (&(a, b), &count) in &shared {
        if count >= opts.min_shared_terms {
            uf.union(a, b);
        }
    }

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..docs.len() {
        groups.entry(uf.find(i)).or_default().push(i);
    }

    let mut candidates = Vec::new();
    for members in groups.into_values() {
        if members.len() < opts.min_members {
            continue;
        }
        let sessions: HashSet<String> = members
            .iter()
            .map(|&i| session_of(&docs[i], &opts.session_key))
            .collect();
        if sessions.len() < opts.min_sessions {
            continue;
        }
        // Key inheritance first: term-derived keys drift as members grow
        // (small-corpus TF-IDF is unstable), so a group that covers >= 50%
        // of an existing digest's members KEEPS that digest's key. The first
        // key wins for the lineage; versioning stays monotonic.
        let member_id_set: HashSet<&str> = members.iter().map(|&i| docs[i].id.as_str()).collect();
        let inherited_key = digests
            .iter()
            .filter_map(|d| {
                let mids = digest_member_ids(&d.metadata);
                if mids.is_empty() {
                    return None;
                }
                let overlap = mids
                    .iter()
                    .filter(|m| member_id_set.contains(m.as_str()))
                    .count();
                if overlap * 2 < mids.len() {
                    return None;
                }
                d.metadata
                    .get(METADATA_KEY_DIGEST_KEY)
                    .and_then(|v| v.as_str())
                    .map(|k| (overlap, k.to_string()))
            })
            .max_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)))
            .map(|(_, k)| k);

        let key = match inherited_key {
            Some(k) => k,
            None => {
                // Fresh key: terms shared by the most members, top 4, sorted.
                let mut term_count: HashMap<&str, usize> = HashMap::new();
                for &i in &members {
                    for t in &top_terms[i] {
                        *term_count.entry(t).or_insert(0) += 1;
                    }
                }
                let mut ranked: Vec<(&str, usize)> = term_count.into_iter().collect();
                ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
                let mut key_terms: Vec<&str> = ranked.into_iter().take(4).map(|(t, _)| t).collect();
                key_terms.sort_unstable();
                key_terms.join("+")
            }
        };
        if key.is_empty() {
            continue;
        }

        // Members ordered by created_at for stable digests.
        let mut ordered: Vec<usize> = members;
        ordered.sort_by_key(|&i| (docs[i].created_at, docs[i].id.clone()));
        let doc_ids: Vec<String> = ordered.iter().map(|&i| docs[i].id.clone()).collect();
        let total_chars = ordered.iter().map(|&i| docs[i].text.chars().count()).sum();

        // Skip keys whose digest is still fresh; flag stale ones.
        let stale = match index.fetch_doc(&digest_doc_id(&key)) {
            Ok(existing) => {
                let recorded = existing
                    .metadata
                    .get(METADATA_KEY_MEMBER_COUNT)
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                if doc_ids.len() <= recorded + opts.stale_margin {
                    continue; // fresh enough
                }
                true
            }
            Err(_) => false, // no digest yet
        };

        candidates.push(ConsolidationCandidate {
            key,
            doc_ids,
            total_chars,
            stale,
        });
    }

    // Stale first, then larger groups; key as the final tiebreaker.
    candidates.sort_by(|a, b| {
        b.stale
            .cmp(&a.stale)
            .then_with(|| b.doc_ids.len().cmp(&a.doc_ids.len()))
            .then_with(|| a.key.cmp(&b.key))
    });
    candidates.truncate(opts.max_candidates);
    Ok(candidates)
}

/// Sync phase 1: fetch + redact member texts and compute the next version.
/// Input redaction happens here; output redaction happens in [`commit`]
/// (defense in depth: a model can regenerate secrets it saw).
pub fn prepare(
    index: &MsaIndex,
    candidate: &ConsolidationCandidate,
    opts: &ConsolidateOptions,
) -> Result<DistillInput> {
    let mut member_texts = Vec::with_capacity(candidate.doc_ids.len());
    let mut budget = opts.max_member_chars;
    for id in &candidate.doc_ids {
        let doc = index.fetch_doc(id)?;
        let redacted = sanitize_detail(&doc.text);
        let take: String = redacted.chars().take(budget).collect();
        budget = budget.saturating_sub(take.chars().count());
        member_texts.push(take);
        if budget == 0 {
            break;
        }
    }
    let next_version = match index.fetch_doc(&digest_doc_id(&candidate.key)) {
        Ok(existing) => existing
            .metadata
            .get(METADATA_KEY_VERSION)
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
            .saturating_add(1),
        Err(_) => 1,
    };
    Ok(DistillInput {
        key: candidate.key.clone(),
        member_texts,
        member_doc_ids: candidate.doc_ids.clone(),
        next_version,
        digest_max_chars: opts.digest_max_chars,
    })
}

/// Sync phase 2: redact the produced text and upsert the digest document.
pub fn commit(
    index: &MsaIndex,
    input: &DistillInput,
    distilled_text: &str,
) -> Result<ConsolidatedDigest> {
    let text = sanitize_detail(distilled_text);
    let text: String = text.chars().take(input.digest_max_chars).collect();
    if text.trim().is_empty() {
        return Err(MsaError::InvalidInput("empty digest text".to_string()));
    }
    let doc_id = digest_doc_id(&input.key);
    // Collision guard (audit finding): two UNRELATED groups can in principle
    // produce the same term key. Overwriting a digest whose member set is
    // disjoint from ours would silently destroy that lineage — refuse instead
    // (the caller logs and skips; both groups keep retrieving via members).
    if let Ok(existing) = index.fetch_doc(&doc_id) {
        let prev_members = digest_member_ids(&existing.metadata);
        let disjoint = !prev_members.is_empty()
            && !input
                .member_doc_ids
                .iter()
                .any(|m| prev_members.contains(m));
        if disjoint {
            return Err(MsaError::InvalidInput(format!(
                "digest key collision for {doc_id}: member sets are disjoint"
            )));
        }
    }
    let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
    metadata.insert(METADATA_KEY_KIND.into(), json!(DIGEST_KIND));
    metadata.insert(METADATA_KEY_DIGEST_KEY.into(), json!(input.key));
    metadata.insert(METADATA_KEY_MEMBER_IDS.into(), json!(input.member_doc_ids));
    metadata.insert(
        METADATA_KEY_MEMBER_COUNT.into(),
        json!(input.member_doc_ids.len()),
    );
    metadata.insert(METADATA_KEY_VERSION.into(), json!(input.next_version));
    let doc = Document {
        id: doc_id.clone(),
        text: text.clone(),
        metadata,
        created_at: chrono::Utc::now(),
    };
    index.index_document(&doc, None)?;
    Ok(ConsolidatedDigest {
        doc_id,
        key: input.key.clone(),
        text,
        member_doc_ids: input.member_doc_ids.clone(),
        version: input.next_version,
    })
}

/// Convenience for sync callers: prepare -> backend -> commit.
pub fn consolidate(
    index: &MsaIndex,
    candidate: &ConsolidationCandidate,
    backend: &dyn DistillBackend,
    opts: &ConsolidateOptions,
) -> Result<ConsolidatedDigest> {
    let input = prepare(index, candidate, opts)?;
    let text = backend
        .distill(&input.key, &input.member_texts, input.digest_max_chars)
        .map_err(|e| MsaError::InvalidInput(e.to_string()))?;
    commit(index, &input, &text)
}

/// True when a document is a digest (used by serving-side supersede).
pub fn is_digest(metadata: &HashMap<String, serde_json::Value>) -> bool {
    metadata
        .get(METADATA_KEY_KIND)
        .and_then(|v| v.as_str())
        .is_some_and(|k| k == DIGEST_KIND)
}

/// Member ids of a digest hit (empty for non-digests).
pub fn digest_member_ids(metadata: &HashMap<String, serde_json::Value>) -> Vec<String> {
    metadata
        .get(METADATA_KEY_MEMBER_IDS)
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// internals

fn fetch_corpus(index: &MsaIndex) -> Result<(Vec<Document>, Vec<Document>)> {
    let mut docs = Vec::new();
    let mut digests = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let profile_hash = index.current_index_profile_hash(None);
        let page = index.manifest(
            &profile_hash,
            None,
            None,
            crate::index::DEFAULT_MANIFEST_LIMIT,
            cursor.as_deref(),
        )?;
        for entry in &page.entries {
            let Ok(doc) = index.fetch_doc(&entry.doc_id) else {
                continue;
            };
            if entry.doc_id.starts_with(DIGEST_ID_PREFIX) || is_digest(&doc.metadata) {
                digests.push(doc);
            } else {
                docs.push(doc);
            }
        }
        match page.next_cursor {
            Some(next) => cursor = Some(next),
            None => break,
        }
    }
    Ok((docs, digests))
}

fn session_of(doc: &Document, source: &SessionKeySource) -> String {
    match source {
        SessionKeySource::MetadataKey(key) => doc
            .metadata
            .get(key)
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_else(|| doc.id.clone()),
        SessionKeySource::DocIdPrefix { separator } => doc
            .id
            .split(separator.as_str())
            .next()
            .unwrap_or(&doc.id)
            .to_string(),
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.chars().count() >= 3)
        .map(str::to_lowercase)
        .collect()
}

fn split_sentences(text: &str) -> impl Iterator<Item = &str> {
    text.split(['\n', '.'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

fn normalize_for_dedup(sentence: &str) -> String {
    sentence
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};
    use tempfile::TempDir;

    use super::*;
    use crate::schema::ChunkConfig;

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

    fn capsule(id: &str, day: &str, ts: i64, text: &str) -> Document {
        let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
        metadata.insert("day".into(), json!(day));
        Document {
            id: id.into(),
            text: text.into(),
            metadata,
            created_at: Utc.timestamp_opt(ts, 0).unwrap(),
        }
    }

    /// Three sessions talk about the marathon training plan; one stray doc
    /// talks about something else. Expect exactly one candidate spanning the
    /// marathon docs and at least 2 sessions.
    fn seed_marathon(idx: &MsaIndex) {
        let docs = [
            capsule(
                "c1",
                "2026-06-01",
                100,
                "started marathon training plan with intervals and tempo runs near the river",
            ),
            capsule(
                "c2",
                "2026-06-02",
                200,
                "marathon training week two: intervals improved, tempo runs felt easier",
            ),
            capsule(
                "c3",
                "2026-06-03",
                300,
                "skipped marathon intervals, long tempo runs instead, training plan adjusted",
            ),
            capsule(
                "c4",
                "2026-06-03",
                400,
                "bought a blue ceramic teapot at the flea market",
            ),
        ];
        for d in &docs {
            idx.index_document(d, None).unwrap();
        }
    }

    fn opts() -> ConsolidateOptions {
        ConsolidateOptions {
            min_shared_terms: 2,
            max_term_df: 10,
            ..ConsolidateOptions::default()
        }
    }

    #[test]
    fn select_candidates_groups_cross_session_topic() {
        let (_d, idx) = fresh_index();
        seed_marathon(&idx);
        let cands = select_candidates(&idx, &opts()).unwrap();
        assert_eq!(cands.len(), 1, "expected exactly one group: {cands:?}");
        let c = &cands[0];
        assert_eq!(c.doc_ids, vec!["c1", "c2", "c3"], "ordered by created_at");
        assert!(!c.stale);
        assert!(!c.key.is_empty());
    }

    #[test]
    fn consolidate_writes_versioned_digest_and_fresh_key_is_skipped() {
        let (_d, idx) = fresh_index();
        seed_marathon(&idx);
        let o = opts();
        let cands = select_candidates(&idx, &o).unwrap();
        let digest = consolidate(&idx, &cands[0], &DeterministicConcatBackend, &o).unwrap();
        assert_eq!(digest.version, 1);
        assert!(digest.doc_id.starts_with(DIGEST_ID_PREFIX));

        // The digest is fetchable, marked, and carries its members.
        let stored = idx.fetch_doc(&digest.doc_id).unwrap();
        assert!(is_digest(&stored.metadata));
        assert_eq!(digest_member_ids(&stored.metadata), vec!["c1", "c2", "c3"]);

        // Same corpus again: the digest is fresh -> no candidates, and the
        // digest doc itself never becomes a member.
        let again = select_candidates(&idx, &o).unwrap();
        assert!(again.is_empty(), "fresh digest must be skipped: {again:?}");
    }

    #[test]
    fn stale_digest_is_reconsolidated_with_bumped_version() {
        let (_d, idx) = fresh_index();
        seed_marathon(&idx);
        let o = ConsolidateOptions {
            stale_margin: 0,
            ..opts()
        };
        let cands = select_candidates(&idx, &o).unwrap();
        consolidate(&idx, &cands[0], &DeterministicConcatBackend, &o).unwrap();

        // New capsules on the same topic from new sessions -> stale digest.
        idx.index_document(
            &capsule(
                "c5",
                "2026-06-04",
                500,
                "marathon training plan: race week, final tempo runs and easy intervals",
            ),
            None,
        )
        .unwrap();
        idx.index_document(
            &capsule(
                "c6",
                "2026-06-05",
                600,
                "completed the marathon, training plan and intervals paid off",
            ),
            None,
        )
        .unwrap();

        let cands = select_candidates(&idx, &o).unwrap();
        assert_eq!(cands.len(), 1, "{cands:?}");
        assert!(cands[0].stale, "digest behind members must be stale");
        let digest = consolidate(&idx, &cands[0], &DeterministicConcatBackend, &o).unwrap();
        assert_eq!(digest.version, 2, "version must bump on re-consolidation");
        assert!(digest.member_doc_ids.len() >= 5);
    }

    #[test]
    fn concat_backend_dedups_and_respects_max_chars() {
        let texts = vec![
            "alpha beta gamma. shared sentence here".to_string(),
            "Shared sentence HERE. delta epsilon".to_string(),
        ];
        let out = DeterministicConcatBackend
            .distill("k", &texts, 1000)
            .unwrap();
        assert_eq!(
            out.matches("hared sentence").count(),
            1,
            "dedup failed: {out}"
        );
        let tiny = DeterministicConcatBackend.distill("k", &texts, 20).unwrap();
        assert!(tiny.chars().count() <= 20);
    }

    #[test]
    fn commit_redacts_secrets_in_digest_text() {
        let (_d, idx) = fresh_index();
        seed_marathon(&idx);
        let o = opts();
        let cands = select_candidates(&idx, &o).unwrap();
        let input = prepare(&idx, &cands[0], &o).unwrap();
        let leaked = "training notes with password = hunter2hunter2 inside";
        let digest = commit(&idx, &input, leaked).unwrap();
        assert!(
            !digest.text.contains("hunter2hunter2"),
            "secret survived redaction: {}",
            digest.text
        );
    }

    #[test]
    fn session_gate_rejects_single_session_groups() {
        let (_d, idx) = fresh_index();
        // 3 related docs, all in the SAME session/day.
        for (i, ts) in [(1, 100), (2, 200), (3, 300)] {
            idx.index_document(
                &capsule(
                    &format!("s{i}"),
                    "2026-06-01",
                    ts,
                    "marathon training plan with intervals and tempo runs",
                ),
                None,
            )
            .unwrap();
        }
        let cands = select_candidates(&idx, &opts()).unwrap();
        assert!(
            cands.is_empty(),
            "single-session group must not consolidate"
        );
    }

    #[test]
    fn commit_refuses_disjoint_member_collision() {
        let (_d, idx) = fresh_index();
        seed_marathon(&idx);
        let o = opts();
        let cands = select_candidates(&idx, &o).unwrap();
        let input = prepare(&idx, &cands[0], &o).unwrap();
        commit(&idx, &input, "digest one").unwrap();

        // Same key, completely different members: must refuse, not overwrite.
        let forged = DistillInput {
            member_doc_ids: vec!["zz1".into(), "zz2".into()],
            ..input.clone()
        };
        let err = commit(&idx, &forged, "digest two").unwrap_err();
        assert!(err.to_string().contains("collision"), "{err}");
        // Original digest untouched.
        let stored = idx.fetch_doc(&digest_doc_id(&input.key)).unwrap();
        assert!(stored.text.contains("digest one"));
    }

    #[test]
    fn default_stale_margin_requires_three_new_members() {
        let (_d, idx) = fresh_index();
        seed_marathon(&idx);
        let o = opts(); // stale_margin: 2 (default)
        let cands = select_candidates(&idx, &o).unwrap();
        consolidate(&idx, &cands[0], &DeterministicConcatBackend, &o).unwrap();

        // +2 new members: within margin -> still fresh, no candidates.
        idx.index_document(
            &capsule(
                "c5",
                "2026-06-04",
                500,
                "marathon training plan: race week, final tempo runs and easy intervals",
            ),
            None,
        )
        .unwrap();
        idx.index_document(
            &capsule(
                "c6",
                "2026-06-05",
                600,
                "completed the marathon, training plan and intervals paid off",
            ),
            None,
        )
        .unwrap();
        assert!(select_candidates(&idx, &o).unwrap().is_empty());

        // +3rd new member: beyond margin -> stale.
        idx.index_document(
            &capsule(
                "c7",
                "2026-06-06",
                700,
                "post marathon recovery, reviewing the training plan and intervals log",
            ),
            None,
        )
        .unwrap();
        let cands = select_candidates(&idx, &o).unwrap();
        assert_eq!(cands.len(), 1, "{cands:?}");
        assert!(cands[0].stale);
    }

    #[test]
    fn doc_id_prefix_session_source_works() {
        let (_d, idx) = fresh_index();
        for (id, ts) in [("s1::t1", 100), ("s2::t1", 200), ("s3::t1", 300)] {
            idx.index_document(
                &Document {
                    id: id.into(),
                    text: "marathon training plan with intervals and tempo runs".into(),
                    metadata: Default::default(),
                    created_at: Utc.timestamp_opt(ts, 0).unwrap(),
                },
                None,
            )
            .unwrap();
        }
        let o = ConsolidateOptions {
            session_key: SessionKeySource::DocIdPrefix {
                separator: "::".into(),
            },
            ..opts()
        };
        let cands = select_candidates(&idx, &o).unwrap();
        assert_eq!(cands.len(), 1);
    }
}
