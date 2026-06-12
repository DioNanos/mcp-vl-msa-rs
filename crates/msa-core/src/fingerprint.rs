//! Document & index fingerprinting (A6 — autonomy foundation).
//!
//! Two deterministic hashes drive change-detection so a client (or the FS
//! adapter) can sync a corpus into a collection without re-indexing what
//! hasn't changed:
//!
//! - [`content_hash_v1`] — over the document `text` **and** its canonical
//!   metadata. A metadata-only change still changes the hash, because metadata
//!   feeds filters/fetch/routing (Codex design audit §2).
//! - [`index_profile_hash_v1`] — over the chunking + dense-embedding profile.
//!   A document with identical content must be re-indexed when the chunk size
//!   or the embedding model/profile changes, or its chunks/embeddings go stale
//!   (Codex design audit §2). Skip-unchanged requires **both** hashes to match.
//!
//! These are **change-detection** fingerprints, not signatures: blake3 is
//! chosen for speed + determinism, not for cryptographic resistance.
//!
//! The byte layouts are fixed and documented so a client can reproduce the
//! same `content_hash` and diff against [`crate::index`]'s manifest without
//! re-sending unchanged documents.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::error::{MsaError, Result};

/// Algorithm/version tag stored alongside a content hash. Bump when the byte
/// layout of [`content_hash_v1`] changes.
pub const CONTENT_HASH_ALGO: &str = "blake3-doc-v1";

/// Reserved top-level metadata key owned by the server (holds the `_msa`
/// fingerprint block on the sentinel record). Client-supplied metadata
/// containing this key is rejected so a client cannot spoof server fields.
pub const RESERVED_MSA_KEY: &str = "_msa";

/// Reject client metadata that uses the reserved [`RESERVED_MSA_KEY`].
///
/// Called at the index boundary before any work, mapping to `invalid_params`.
pub fn reject_reserved_metadata<S: std::hash::BuildHasher>(
    metadata: &std::collections::HashMap<String, Value, S>,
) -> Result<()> {
    if metadata.contains_key(RESERVED_MSA_KEY) {
        return Err(MsaError::InvalidInput(format!(
            "metadata key {RESERVED_MSA_KEY:?} is reserved by the server"
        )));
    }
    Ok(())
}

/// Content fingerprint over `text` + canonical metadata (hex, lowercase).
///
/// Byte layout (domain-separated, length-prefixed so fields cannot collide):
/// ```text
/// blake3(
///   b"msa-doc-hash-v1\0"
///   || u64_le(text.len())            || text_utf8
///   || u64_le(canon_meta.len())      || canon_meta_json_utf8
/// )
/// ```
/// `canon_meta_json` is [`canonical_metadata_json`] of `metadata` (`{}` when
/// empty). `created_at` is intentionally NOT included — `msa_index` assigns it
/// server-side, so including it would make the same document non-deterministic.
pub fn content_hash_v1<S: std::hash::BuildHasher>(
    text: &str,
    metadata: &std::collections::HashMap<String, Value, S>,
) -> String {
    let canon_meta = canonical_metadata_json(metadata);
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"msa-doc-hash-v1\0");
    hasher.update(&(text.len() as u64).to_le_bytes());
    hasher.update(text.as_bytes());
    hasher.update(&(canon_meta.len() as u64).to_le_bytes());
    hasher.update(canon_meta.as_bytes());
    hasher.finalize().to_hex().to_string()
}

/// The chunking + embedding profile that, when changed, must force a reindex
/// even if the document content is identical (stale chunks/embeddings).
///
/// Invariant (Codex review §2): the tuple `(embedding_model_id,
/// embedding_model_version, embedding_profile, embedding_dim)` must uniquely
/// identify the embedding *backend* too. The `DenseEncoder` trait does not
/// expose a backend id, so `embedding_backend` is currently `None`; this is
/// safe only while no two backends share that tuple. If that ever changes, add
/// `backend_id()` to the trait and populate `embedding_backend` here.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexProfile {
    pub chunk_size: usize,
    pub overlap: usize,
    /// Whether a dense scorer was active at index time.
    pub embedding_active: bool,
    pub embedding_backend: Option<String>,
    pub embedding_model_id: Option<String>,
    pub embedding_model_version: Option<String>,
    pub embedding_profile: Option<String>,
    pub embedding_dim: Option<usize>,
}

/// Index-profile fingerprint (hex, lowercase). Canonical-JSON of the profile,
/// then blake3. See [`IndexProfile`].
pub fn index_profile_hash_v1(p: &IndexProfile) -> String {
    fn opt_str(s: &Option<String>) -> Value {
        match s {
            Some(v) => Value::String(v.clone()),
            None => Value::Null,
        }
    }
    fn opt_usize(n: Option<usize>) -> Value {
        match n {
            Some(v) => Value::from(v as u64),
            None => Value::Null,
        }
    }
    let profile = serde_json::json!({
        "schema": "msa-index-profile-v1",
        "chunk_size": p.chunk_size as u64,
        "overlap": p.overlap as u64,
        "embedding": {
            "active": p.embedding_active,
            "backend": opt_str(&p.embedding_backend),
            "model_id": opt_str(&p.embedding_model_id),
            "model_version": opt_str(&p.embedding_model_version),
            "profile": opt_str(&p.embedding_profile),
            "dim": opt_usize(p.embedding_dim),
        }
    });
    let canon = canonical_json(&profile);
    blake3::hash(canon.as_bytes()).to_hex().to_string()
}

/// Canonical JSON of a metadata map: an object with keys sorted
/// lexicographically (recursively), no whitespace. Empty map → `"{}"`.
pub fn canonical_metadata_json<S: std::hash::BuildHasher>(
    metadata: &std::collections::HashMap<String, Value, S>,
) -> String {
    // Build a serde Map then canonicalize as a Value::Object.
    let obj: serde_json::Map<String, Value> = metadata
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    canonical_json(&Value::Object(obj))
}

/// Deterministic, whitespace-free serialization of a JSON value with object
/// keys sorted lexicographically at every level. Arrays keep their order.
/// Scalars use serde_json's canonical formatting.
pub fn canonical_json(value: &Value) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Object(map) => {
            // Sort keys lexicographically by their UTF-8 bytes.
            let sorted: BTreeMap<&String, &Value> = map.iter().collect();
            out.push('{');
            let mut first = true;
            for (k, v) in sorted {
                if !first {
                    out.push(',');
                }
                first = false;
                // Key is a JSON string: reuse serde for correct escaping.
                out.push_str(&Value::String(k.clone()).to_string());
                out.push(':');
                write_canonical(v, out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(v, out);
            }
            out.push(']');
        }
        // Scalars (string/number/bool/null): serde_json's compact form is
        // already whitespace-free and deterministic for our inputs.
        other => out.push_str(&other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn meta(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn canonical_json_sorts_keys_recursively_no_whitespace() {
        let v = serde_json::json!({
            "b": 1,
            "a": {"z": [3, 2, 1], "y": "x"},
        });
        // keys sorted at every level; arrays preserve order; no spaces.
        assert_eq!(canonical_json(&v), r#"{"a":{"y":"x","z":[3,2,1]},"b":1}"#);
    }

    #[test]
    fn empty_metadata_canonicalizes_to_braces() {
        let m: HashMap<String, Value> = HashMap::new();
        assert_eq!(canonical_metadata_json(&m), "{}");
    }

    #[test]
    fn content_hash_is_order_independent_for_metadata() {
        let m1 = meta(&[("path", Value::from("a.md")), ("kind", Value::from("doc"))]);
        let m2 = meta(&[("kind", Value::from("doc")), ("path", Value::from("a.md"))]);
        assert_eq!(content_hash_v1("hello", &m1), content_hash_v1("hello", &m2));
    }

    #[test]
    fn content_hash_changes_with_text_and_metadata() {
        let m = meta(&[("k", Value::from("v"))]);
        let m2 = meta(&[("k", Value::from("w"))]);
        let base = content_hash_v1("hello", &m);
        assert_ne!(
            base,
            content_hash_v1("hello world", &m),
            "text change must change hash"
        );
        assert_ne!(
            base,
            content_hash_v1("hello", &m2),
            "metadata change must change hash"
        );
        assert_ne!(
            base,
            content_hash_v1("hello", &HashMap::new()),
            "metadata presence must change hash"
        );
    }

    #[test]
    fn content_hash_length_prefix_prevents_field_collision() {
        // Without length-prefixing, ("ab","") and ("a","b") could collide.
        let empty: HashMap<String, Value> = HashMap::new();
        let h_ab = content_hash_v1("ab", &empty);
        let h_a_meta = content_hash_v1("a", &meta(&[("", Value::from("b"))]));
        assert_ne!(h_ab, h_a_meta);
    }

    #[test]
    fn content_hash_is_stable_test_vector() {
        // Fixed vector so a cross-client reimplementation can verify itself.
        let empty: HashMap<String, Value> = HashMap::new();
        let h = content_hash_v1("hello", &empty);
        assert_eq!(h.len(), 64, "blake3 hex is 64 chars");
        // Recompute the documented layout independently and compare.
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"msa-doc-hash-v1\0");
        hasher.update(&(5u64).to_le_bytes());
        hasher.update(b"hello");
        hasher.update(&(2u64).to_le_bytes()); // "{}"
        hasher.update(b"{}");
        assert_eq!(h, hasher.finalize().to_hex().to_string());
    }

    #[test]
    fn index_profile_hash_distinguishes_chunking_and_embedding() {
        let bm25 = IndexProfile {
            chunk_size: 64,
            overlap: 0,
            ..Default::default()
        };
        let bm25_other_chunk = IndexProfile {
            chunk_size: 128,
            ..bm25.clone()
        };
        let dense = IndexProfile {
            embedding_active: true,
            embedding_model_id: Some("granite-r2-97m".into()),
            embedding_dim: Some(384),
            ..bm25.clone()
        };
        let a = index_profile_hash_v1(&bm25);
        assert_eq!(a, index_profile_hash_v1(&bm25.clone()), "deterministic");
        assert_ne!(
            a,
            index_profile_hash_v1(&bm25_other_chunk),
            "chunk_size matters"
        );
        assert_ne!(
            a,
            index_profile_hash_v1(&dense),
            "embedding profile matters"
        );
        assert_eq!(a.len(), 64);
    }

    #[test]
    fn reject_reserved_metadata_blocks_msa_key() {
        let bad = meta(&[(RESERVED_MSA_KEY, Value::from("x"))]);
        assert!(reject_reserved_metadata(&bad).is_err());
        let ok = meta(&[("path", Value::from("a.md"))]);
        assert!(reject_reserved_metadata(&ok).is_ok());
    }
}
