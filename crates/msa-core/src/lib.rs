//! mcp-vl-msa-rs (msa-core) — MSA-flavor retrieval/memory engine over tantivy.
//!
//! Public surface:
//! - [`schema`] — `Document`, `ChunkHit`, `SearchFilter`, `MsaStats`
//! - [`chunker`] — word-based chunking with overlap (MSA `P=64` default)
//! - [`index`] — `MsaIndex` wrapping tantivy per collection
//! - [`config`] — runtime configuration loader
//!
//! See `README.md` and `~/.docs/projects/mcp-msa-rs/` for design.

pub mod chunker;
pub mod config;
pub mod consolidate;
pub mod embeddings;
pub mod enrich;
pub mod error;
pub mod fingerprint;
pub mod index;
pub mod interleave;
pub mod remember;
pub mod schema;
pub mod session;
pub mod validate;

pub use config::{AgentMemoryConfig, MsaConfig};
pub use error::{MsaError, Result};
pub use fingerprint::{
    content_hash_v1, index_profile_hash_v1, IndexProfile, CONTENT_HASH_ALGO, RESERVED_MSA_KEY,
};
pub use index::{
    BatchCaps, CollectionRegistry, DeltaOptions, DeltaOutcome, DeltaResult, DeltaStatus,
    DocFingerprint, HashState, Manifest, ManifestEntry, MsaIndex, DEFAULT_MANIFEST_LIMIT,
    MAX_MANIFEST_LIMIT,
};
pub use interleave::{
    run_round, DedupMode, EvidenceUnitId, FetchError, InjectedDoc, InterleaveParams,
    InterleaveStateV1, RoundBudget, RoundResponse, RoundStats, StopHints,
};
pub use remember::{
    forget_by_source_id, remember, ForgetBatchOutcome, MemoryKind, RememberOutcome,
    META_KEY_CREATED_AT, META_KEY_KIND, META_KEY_SOURCE_ID,
};
pub use schema::{ChunkHit, Document, MsaStats, SearchFilter};
