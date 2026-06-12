use thiserror::Error;

#[derive(Debug, Error)]
pub enum MsaError {
    #[error("tantivy error: {0}")]
    Tantivy(#[from] tantivy::TantivyError),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("collection not found: {0}")]
    UnknownCollection(String),

    #[error("document not found: {collection}/{doc_id}")]
    UnknownDocument { collection: String, doc_id: String },

    #[error("invalid query: {0}")]
    InvalidQuery(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// An optimistic-concurrency precondition failed (e.g. a stale
    /// `if_manifest_digest`/`if_index_profile_hash` in `apply_delta`). The
    /// caller's view is out of date; nothing was written. Caller-caused, so the
    /// server maps it to `invalid_params`.
    #[error("precondition failed: {0}")]
    Conflict(String),

    #[error("schema mismatch: {0}")]
    Schema(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, MsaError>;
