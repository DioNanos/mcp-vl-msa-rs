//! Optional dense scorer for hybrid retrieval (feature `embeddings`).
//!
//! When the AI client requests `dense_alpha > 0` on `msa_search` or
//! `msa_search_iterative`, the server fetches up to `top_k * 4` BM25
//! candidates from tantivy, computes the query embedding via the
//! configured [`DenseEncoder`], reads each candidate's cached chunk
//! embedding, and rescores:
//!
//! ```text
//! score = α · bm25_normalized + (1 - α) · ((cosine + 1) / 2)
//! ```
//!
//! Cosine similarity is shifted to `[0, 1]` so it composes linearly with
//! the already-normalized BM25 score. Chunks without a cached embedding
//! fall back to BM25 only (`cosine = 0` shifted = `0.5`); the alpha can be
//! pushed to `1.0` to keep BM25 behavior unchanged.
//!
//! Embeddings are computed at index time and stored in tantivy as a
//! little-endian f32 byte vector (A1; sidecar generation storage planned in
//! A2). A mismatched embedding dimension on a re-opened index returns an
//! error at search time rather than crashing.
//!
//! ## Backends (v0.5 / A1)
//!
//! - `candle-modernbert` (feature `embeddings-candle`, **production
//!   default**): in-process ModernBERT encoder loaded from a local bundle,
//!   targets Granite-R2-97m-multilingual. No network at runtime.
//! - `ollama` (feature `embeddings-ollama`, **deprecated**): HTTP client
//!   against an Ollama-compatible service. Kept for one release; new
//!   deployments should use `candle-modernbert`.

use std::sync::Arc;

#[allow(unused_imports)]
use crate::error::{MsaError, Result};

/// Whether a text is being embedded as a search **query** or a stored
/// **passage** / document chunk. Some encoders (notably the E5 family) use a
/// different prompt prefix per kind; Granite-R2 ignores this hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbedInputKind {
    Query,
    Passage,
}

/// Sync, kind-aware dense embedding encoder.
///
/// Sync because `index_document` and the search hot path are not async and
/// we do not want to drag tokio into every call site. Backends that need
/// background work (e.g. blocking HTTP) handle it internally.
pub trait DenseEncoder: Send + Sync {
    /// Encode a batch of texts of the given kind. Vectors are L2-normalized
    /// and validated against [`Self::dim`] before being returned.
    fn encode_batch_kind(&self, kind: EmbedInputKind, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Reported embedding dimension. Stored in the index manifest and used
    /// to validate cached embeddings at search time.
    fn dim(&self) -> usize;

    /// Stable identifier of the model, e.g.
    /// `"ibm-granite/granite-embedding-97m-multilingual-r2"`. Used as the
    /// primary key for shadow-generation namespacing.
    fn model_id(&self) -> &str;

    /// Version pin: HF revision, content sha256, or bundle digest. Two
    /// configurations with the same `model_id` but different
    /// `model_version` are treated as different generations.
    fn model_version(&self) -> &str {
        ""
    }

    /// Optional human-readable profile tag, e.g. `"granite-r2-97m"` or
    /// `"e5-base"`. Used in logs and storage layout.
    fn profile(&self) -> &str {
        "default"
    }

    /// Convenience single-text encode.
    fn encode_kind(&self, kind: EmbedInputKind, text: &str) -> Result<Vec<f32>> {
        self.encode_batch_kind(kind, &[text])?
            .into_iter()
            .next()
            .ok_or_else(|| MsaError::Config("encoder returned no vector".into()))
    }

    /// Backward-compatibility helper. Defaults to encoding as a `Query`.
    /// Prefer [`Self::encode_kind`] in new call sites.
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.encode_kind(EmbedInputKind::Query, text)
    }
}

/// Shared, thread-safe encoder handle used across the MCP server.
pub type SharedEncoder = Arc<dyn DenseEncoder>;

/// Deprecated alias preserved for one release while existing call sites are
/// migrated to [`SharedEncoder`].
#[deprecated(
    since = "0.5.0",
    note = "use SharedEncoder; the EmbeddingScorer trait is gone in v0.6"
)]
pub type SharedScorer = SharedEncoder;

// -------------------------------------------------------------------------
// Cosine + pack helpers (storage-layer utilities, used by index.rs).
// -------------------------------------------------------------------------

/// Cosine similarity shifted to `[0.0, 1.0]`. Vectors must have equal
/// length and non-zero norm; otherwise returns `0.5` (the neutral value).
pub fn cosine_unit(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.5;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.5;
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    ((cos + 1.0) / 2.0).clamp(0.0, 1.0)
}

/// Pack an `&[f32]` as little-endian bytes for tantivy storage.
pub fn pack_f32(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Unpack `&[u8]` to `Vec<f32>`. Returns empty on length mismatch.
pub fn unpack_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// -------------------------------------------------------------------------
// Backend: Candle ModernBERT (production default, feature
// `embeddings-candle`).
// -------------------------------------------------------------------------

#[cfg(feature = "embeddings-candle")]
mod candle_modernbert {
    use super::*;
    use std::path::Path;
    use std::sync::Mutex;

    use candle_core::{DType, Device, IndexOp, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::modernbert::{Config as ModernBertConfig, ModernBert};
    use tokenizers::Tokenizer;

    /// Default conservative max input length. Granite-R2 supports 32K
    /// context but smartphone Termux targets need headroom; bump only after
    /// on-device memory profiling.
    pub const DEFAULT_MAX_LEN: usize = 8192;

    /// In-process Candle ModernBERT encoder. Loaded from a local bundle
    /// prepared by `scripts/prepare-granite-r2-97m.sh`. The bundle layout
    /// is:
    ///
    /// ```text
    /// <model_dir>/
    ///   config.json           # ModernBERT config
    ///   tokenizer.json        # HuggingFace tokenizer
    ///   model.safetensors     # weights with `model.` key prefix
    /// ```
    ///
    /// The setup script handles the Granite-specific safetensor key rename
    /// (Granite distributes weights without the `model.` prefix that
    /// candle ModernBERT expects). The runtime therefore never touches the
    /// network or `/tmp` and is fully deterministic offline.
    pub struct CandleModernBert {
        model_id: String,
        model_version: String,
        profile: String,
        dim: usize,
        max_len: usize,
        // Serialize forward passes. The MCP stdio server is effectively
        // single-threaded so the Mutex cost is negligible.
        inner: Mutex<Inner>,
    }

    struct Inner {
        tokenizer: Tokenizer,
        model: ModernBert,
        device: Device,
    }

    impl CandleModernBert {
        /// Load from a local bundle directory.
        pub fn load_from_dir(
            model_dir: &Path,
            model_id: impl Into<String>,
            model_version: impl Into<String>,
            profile: impl Into<String>,
            dim: usize,
            max_len: Option<usize>,
        ) -> Result<Self> {
            if !model_dir.is_dir() {
                return Err(MsaError::Config(format!(
                    "candle-modernbert: model_dir {} is not a directory",
                    model_dir.display()
                )));
            }
            let config_path = model_dir.join("config.json");
            let tokenizer_path = model_dir.join("tokenizer.json");
            let weights_path = model_dir.join("model.safetensors");

            for (label, p) in [
                ("config.json", &config_path),
                ("tokenizer.json", &tokenizer_path),
                ("model.safetensors", &weights_path),
            ] {
                if !p.exists() {
                    return Err(MsaError::Config(format!(
                        "candle-modernbert: missing bundle file {} at {}",
                        label,
                        p.display()
                    )));
                }
            }

            let config: ModernBertConfig = serde_json::from_slice(
                &std::fs::read(&config_path)
                    .map_err(|e| MsaError::Config(format!("read config.json: {e}")))?,
            )
            .map_err(|e| MsaError::Config(format!("parse config.json: {e}")))?;

            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| MsaError::Config(format!("load tokenizer.json: {e}")))?;

            let device = Device::Cpu;
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[&weights_path], DType::F32, &device)
                    .map_err(|e| MsaError::Config(format!("mmap safetensors: {e}")))?
            };
            let model = ModernBert::load(vb, &config)
                .map_err(|e| MsaError::Config(format!("ModernBert::load: {e}")))?;

            let model_id_s = model_id.into();
            let model_version_s = model_version.into();
            let profile_s = profile.into();

            tracing::info!(
                model_id = %model_id_s,
                model_version = %model_version_s,
                profile = %profile_s,
                dim,
                "candle-modernbert: model loaded"
            );

            Ok(Self {
                model_id: model_id_s,
                model_version: model_version_s,
                profile: profile_s,
                dim,
                max_len: max_len.unwrap_or(DEFAULT_MAX_LEN),
                inner: Mutex::new(Inner {
                    tokenizer,
                    model,
                    device,
                }),
            })
        }

        fn encode_one(&self, text: &str) -> Result<Vec<f32>> {
            let guard = self
                .inner
                .lock()
                .map_err(|_| MsaError::Config("candle inner poisoned".into()))?;

            let enc = guard
                .tokenizer
                .encode(text, true)
                .map_err(|e| MsaError::Config(format!("tokenize: {e}")))?;

            // Truncate to max_len if the tokenizer was not configured with
            // truncation (defensive; most HF tokenizer.json embed it).
            let mut ids: Vec<u32> = enc.get_ids().to_vec();
            let mut mask: Vec<u32> = enc.get_attention_mask().to_vec();
            if ids.len() > self.max_len {
                ids.truncate(self.max_len);
                mask.truncate(self.max_len);
            }

            let token_ids = Tensor::new(&ids[..], &guard.device)
                .map_err(|e| MsaError::Config(format!("token tensor: {e}")))?
                .unsqueeze(0)
                .map_err(|e| MsaError::Config(format!("token unsqueeze: {e}")))?;
            let attention_mask = Tensor::new(&mask[..], &guard.device)
                .map_err(|e| MsaError::Config(format!("mask tensor: {e}")))?
                .unsqueeze(0)
                .map_err(|e| MsaError::Config(format!("mask unsqueeze: {e}")))?;

            let hidden = guard
                .model
                .forward(&token_ids, &attention_mask)
                .map_err(|e| MsaError::Config(format!("forward: {e}")))?;

            // Granite R2 uses CLS pooling (first token), per its model card.
            let pooled = hidden
                .i((.., 0, ..))
                .map_err(|e| MsaError::Config(format!("cls pool: {e}")))?;
            let norm = pooled
                .sqr()
                .and_then(|t| t.sum_keepdim(1))
                .and_then(|t| t.sqrt())
                .map_err(|e| MsaError::Config(format!("norm calc: {e}")))?;
            let normalized = pooled
                .broadcast_div(&norm)
                .and_then(|t| t.squeeze(0))
                .map_err(|e| MsaError::Config(format!("normalize: {e}")))?;

            let vec: Vec<f32> = normalized
                .to_vec1()
                .map_err(|e| MsaError::Config(format!("to_vec1: {e}")))?;

            if vec.len() != self.dim {
                return Err(MsaError::Config(format!(
                    "candle-modernbert: produced dim {} but configured dim is {}",
                    vec.len(),
                    self.dim
                )));
            }
            // Reject NaN / non-finite norms.
            let n2: f32 = vec.iter().map(|x| x * x).sum();
            if !n2.is_finite() || n2 == 0.0 {
                return Err(MsaError::Config(format!(
                    "candle-modernbert: invalid norm {n2}"
                )));
            }

            Ok(vec)
        }
    }

    impl DenseEncoder for CandleModernBert {
        fn encode_batch_kind(
            &self,
            _kind: EmbedInputKind,
            texts: &[&str],
        ) -> Result<Vec<Vec<f32>>> {
            // Granite-R2 ignores the query/passage distinction (no prompt
            // template). Encode sequentially; batching is a fase-B
            // optimization once we measure end-to-end throughput.
            let mut out = Vec::with_capacity(texts.len());
            for t in texts {
                out.push(self.encode_one(t)?);
            }
            Ok(out)
        }

        fn dim(&self) -> usize {
            self.dim
        }
        fn model_id(&self) -> &str {
            &self.model_id
        }
        fn model_version(&self) -> &str {
            &self.model_version
        }
        fn profile(&self) -> &str {
            &self.profile
        }
    }
}

#[cfg(feature = "embeddings-candle")]
pub use candle_modernbert::{CandleModernBert, DEFAULT_MAX_LEN};

// -------------------------------------------------------------------------
// Backend: Ollama HTTP (deprecated, feature `embeddings-ollama`).
// -------------------------------------------------------------------------

#[cfg(feature = "embeddings-ollama")]
mod ollama {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::time::Duration;

    /// Embedding client for Ollama (`POST /api/embeddings`). Other backends
    /// that expose a compatible JSON shape (`{model, prompt}` → `{embedding}`)
    /// will work as well.
    ///
    /// **Deprecated**: prefer [`super::CandleModernBert`] for new
    /// deployments. Will be removed in v0.6.
    pub struct OllamaClient {
        base_url: String,
        model: String,
        dim: usize,
        num_ctx: Option<usize>,
        num_thread: Option<usize>,
        client: reqwest::blocking::Client,
    }

    #[derive(Serialize)]
    struct EmbedReq<'a> {
        model: &'a str,
        prompt: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        options: Option<EmbedOpts>,
    }

    /// Per-request options. `num_ctx` caps the context window (so a model does
    /// not inherit a huge server default and OOM); `num_thread` sets CPU threads
    /// (the server otherwise picks 1, which is ~10x slower for embeddings).
    #[derive(Serialize)]
    struct EmbedOpts {
        #[serde(skip_serializing_if = "Option::is_none")]
        num_ctx: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        num_thread: Option<usize>,
    }

    /// Truncate to the first `max_chars` characters on a UTF-8 boundary.
    fn truncate_to_chars(s: &str, max_chars: usize) -> &str {
        match s.char_indices().nth(max_chars) {
            Some((idx, _)) => &s[..idx],
            None => s,
        }
    }

    #[derive(Deserialize)]
    struct EmbedResp {
        embedding: Vec<f32>,
    }

    impl OllamaClient {
        pub fn new(
            base_url: impl Into<String>,
            model: impl Into<String>,
            dim: usize,
        ) -> Result<Self> {
            let client = reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .map_err(|e| MsaError::Config(format!("reqwest build: {e}")))?;
            Ok(Self {
                base_url: base_url.into(),
                model: model.into(),
                dim,
                num_ctx: None,
                num_thread: None,
                client,
            })
        }

        /// Cap the per-request context window (`options.num_ctx`). Needed when
        /// the server forces a large default context (`OLLAMA_CONTEXT_LENGTH`)
        /// that would make a local embedding model allocate an oversized KV
        /// cache and get OOM-killed. `0` leaves the server default.
        pub fn with_num_ctx(mut self, num_ctx: usize) -> Self {
            self.num_ctx = if num_ctx > 0 { Some(num_ctx) } else { None };
            self
        }

        /// Set CPU threads (`options.num_thread`). The server defaults to 1
        /// thread for embeddings, which is ~10x slower than using all cores.
        /// `0` leaves the server default.
        pub fn with_num_thread(mut self, num_thread: usize) -> Self {
            self.num_thread = if num_thread > 0 {
                Some(num_thread)
            } else {
                None
            };
            self
        }

        fn embed_inner(&self, text: &str) -> Result<Vec<f32>> {
            let url = format!("{}/api/embeddings", self.base_url.trim_end_matches('/'));
            // The /api/embeddings endpoint errors (rather than truncating) when
            // the input exceeds the context, so cap the prompt to a conservative
            // char budget derived from num_ctx (~2 chars/token worst case).
            let prompt = match self.num_ctx {
                Some(nc) => truncate_to_chars(text, nc.saturating_mul(2).saturating_sub(64)),
                None => text,
            };
            let resp: EmbedResp = self
                .client
                .post(&url)
                .json(&EmbedReq {
                    model: &self.model,
                    prompt,
                    options: if self.num_ctx.is_some() || self.num_thread.is_some() {
                        Some(EmbedOpts {
                            num_ctx: self.num_ctx,
                            num_thread: self.num_thread,
                        })
                    } else {
                        None
                    },
                })
                .send()
                .and_then(|r| r.error_for_status())
                .and_then(|r| r.json::<EmbedResp>())
                .map_err(|e| MsaError::Config(format!("ollama embed: {e}")))?;
            if resp.embedding.len() != self.dim {
                return Err(MsaError::Config(format!(
                    "ollama returned dim {} but configured dim is {}",
                    resp.embedding.len(),
                    self.dim
                )));
            }
            Ok(resp.embedding)
        }
    }

    impl DenseEncoder for OllamaClient {
        fn encode_batch_kind(
            &self,
            _kind: EmbedInputKind,
            texts: &[&str],
        ) -> Result<Vec<Vec<f32>>> {
            // Ollama API embeds one prompt per request; sequential is fine
            // for the deprecated path.
            let mut out = Vec::with_capacity(texts.len());
            for t in texts {
                out.push(self.embed_inner(t)?);
            }
            Ok(out)
        }
        fn dim(&self) -> usize {
            self.dim
        }
        fn model_id(&self) -> &str {
            &self.model
        }
        fn profile(&self) -> &str {
            "ollama"
        }
    }
}

#[cfg(feature = "embeddings-ollama")]
pub use ollama::OllamaClient;

// -------------------------------------------------------------------------
// Test scaffolding.
// -------------------------------------------------------------------------

/// Deterministic in-process encoder used by unit tests of other modules.
/// Public-in-crate so `mod tests` in `index.rs` can reach it without a
/// duplicate definition. Gated to test builds.
#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    /// One-hot vector keyed on the hash of the first whitespace word.
    /// Two inputs that share their first word produce identical embeddings,
    /// which lets tests construct deterministic dense-rerank scenarios.
    pub struct MockEncoder {
        dim: usize,
    }

    impl MockEncoder {
        pub fn new(dim: usize) -> Self {
            Self { dim }
        }
    }

    impl DenseEncoder for MockEncoder {
        fn encode_batch_kind(
            &self,
            _kind: EmbedInputKind,
            texts: &[&str],
        ) -> Result<Vec<Vec<f32>>> {
            let mut out = Vec::with_capacity(texts.len());
            for t in texts {
                let mut v = vec![0.0; self.dim];
                if let Some(w) = t.split_whitespace().next() {
                    let h = w
                        .bytes()
                        .fold(0u64, |a, b| a.wrapping_mul(33).wrapping_add(b as u64));
                    v[(h as usize) % self.dim] = 1.0;
                }
                out.push(v);
            }
            Ok(out)
        }
        fn dim(&self) -> usize {
            self.dim
        }
        fn model_id(&self) -> &str {
            "mock"
        }
        fn profile(&self) -> &str {
            "mock"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::MockEncoder;
    use super::*;

    #[test]
    fn cosine_identity_is_one() {
        let v = vec![0.1, 0.2, 0.3, 0.4];
        assert!((cosine_unit(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_orthogonal_is_half() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_unit(&a, &b) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn cosine_opposite_is_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!(cosine_unit(&a, &b) < 1e-5);
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let v = vec![0.1, -0.2, 1.5, -42.7, 0.0];
        assert_eq!(unpack_f32(&pack_f32(&v)), v);
    }

    #[test]
    fn mock_encoder_deterministic() {
        let s = MockEncoder::new(8);
        let a = s.encode_kind(EmbedInputKind::Query, "hello world").unwrap();
        let b = s
            .encode_kind(EmbedInputKind::Passage, "hello there")
            .unwrap();
        assert_eq!(a, b, "first word identical → same embedding");
        let c = s.encode_kind(EmbedInputKind::Query, "alpha bravo").unwrap();
        assert_ne!(a, c, "different first word → different embedding");
    }

    #[test]
    fn encode_batch_returns_per_input_vector() {
        let s = MockEncoder::new(4);
        let out = s
            .encode_batch_kind(EmbedInputKind::Query, &["one two", "three four"])
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 4);
        assert_eq!(out[1].len(), 4);
    }
}
