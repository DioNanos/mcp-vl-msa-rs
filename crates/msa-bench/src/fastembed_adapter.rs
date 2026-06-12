//! Fastembed ONNX encoder adapter (feature `bench-fastembed`).
//!
//! Wraps [`fastembed::TextEmbedding`] to implement the [`DenseEncoder`] trait,
//! enabling benchmark comparison against the Candle-ModernBERT production path.
//!
//! Fastembed downloads ONNX models from HuggingFace on first use and caches
//! them locally. Output vectors are already L2-normalized by fastembed.

use std::sync::Mutex;

use msa_core::embeddings::{DenseEncoder, EmbedInputKind};
use msa_core::error::{MsaError, Result};

// ModelTrait provides get_model_info() for EmbeddingModel.
use fastembed::ModelTrait;

/// Adapter wrapping a [`fastembed::TextEmbedding`] as a [`DenseEncoder`].
pub struct FastembedEncoder {
    model_id: String,
    dim: usize,
    profile: String,
    // fastembed::TextEmbedding::embed() takes &mut self, so we need a Mutex
    // (same pattern as CandleModernBert).
    inner: Mutex<fastembed::TextEmbedding>,
}

impl FastembedEncoder {
    /// Create a new encoder from a fastembed [`EmbeddingModel`](fastembed::EmbeddingModel) variant.
    pub fn try_new(model: fastembed::EmbeddingModel) -> Result<Self> {
        let info = fastembed::EmbeddingModel::get_model_info(&model).ok_or_else(|| {
            MsaError::Config("fastembed: model not found in internal registry".into())
        })?;
        let model_code = info.model_code.clone();
        let dim = info.dim;
        let short_name = model_code.split('/').next_back().unwrap_or("unknown");
        let profile = format!("fastembed-{short_name}");

        tracing::info!(
            model_code = %model_code,
            dim,
            "fastembed: loading ONNX model (first run downloads from HuggingFace)"
        );

        let te = fastembed::TextEmbedding::try_new(fastembed::TextInitOptions::new(model))
            .map_err(|e| MsaError::Config(format!("fastembed init ({model_code}): {e}")))?;

        Ok(Self {
            model_id: model_code,
            dim,
            profile,
            inner: Mutex::new(te),
        })
    }
}

impl DenseEncoder for FastembedEncoder {
    fn encode_batch_kind(&self, _kind: EmbedInputKind, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // fastembed does not distinguish query/passage at the API level.
        // Models that support it (e.g. BGE-M3) handle it internally.
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| MsaError::Config("fastembed mutex poisoned".into()))?;

        let embeddings = guard
            .embed(texts, None)
            .map_err(|e| MsaError::Config(format!("fastembed embed: {e}")))?;

        // Validate dimensions on every call (defensive, same as CandleModernBert).
        for (i, emb) in embeddings.iter().enumerate() {
            if emb.len() != self.dim {
                return Err(MsaError::Config(format!(
                    "fastembed: embedding[{}] dim {} != expected {}",
                    i,
                    emb.len(),
                    self.dim
                )));
            }
        }

        Ok(embeddings)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn model_version(&self) -> &str {
        "onnx"
    }

    fn profile(&self) -> &str {
        &self.profile
    }
}

/// Parse a CLI model name string to a [`fastembed::EmbeddingModel`].
///
/// Delegates to fastembed's built-in `FromStr` impl which matches on the
/// enum Debug name case-insensitively (e.g. `"BGEM3"`, `"bgem3"`, `"BgeM3"`
/// all resolve to `EmbeddingModel::BGEM3`). Also accepts common aliases
/// like `"bge-m3"`.
pub fn parse_model(name: &str) -> anyhow::Result<fastembed::EmbeddingModel> {
    // Try fastembed's built-in FromStr first (matches Debug names like
    // "BGEM3", "AllMiniLML6V2", etc.).
    if let Ok(model) = name.parse::<fastembed::EmbeddingModel>() {
        return Ok(model);
    }
    // Fallback: common aliases that don't match the Debug name.
    match name.to_lowercase().as_str() {
        "bge-m3" => Ok(fastembed::EmbeddingModel::BGEM3),
        "all-minilm-l6-v2" => Ok(fastembed::EmbeddingModel::AllMiniLML6V2),
        "nomic-embed-text-v1.5" => Ok(fastembed::EmbeddingModel::NomicEmbedTextV15),
        "gte-base-en-v1.5" => Ok(fastembed::EmbeddingModel::GTEBaseENV15),
        "gte-large-en-v1.5" => Ok(fastembed::EmbeddingModel::GTELargeENV15),
        "multilingual-e5-small" => Ok(fastembed::EmbeddingModel::MultilingualE5Small),
        "multilingual-e5-base" => Ok(fastembed::EmbeddingModel::MultilingualE5Base),
        "multilingual-e5-large" => Ok(fastembed::EmbeddingModel::MultilingualE5Large),
        "snowflake-arctic-embed-xs" => Ok(fastembed::EmbeddingModel::SnowflakeArcticEmbedXS),
        "snowflake-arctic-embed-s" => Ok(fastembed::EmbeddingModel::SnowflakeArcticEmbedS),
        "snowflake-arctic-embed-m" => Ok(fastembed::EmbeddingModel::SnowflakeArcticEmbedM),
        "snowflake-arctic-embed-l" => Ok(fastembed::EmbeddingModel::SnowflakeArcticEmbedL),
        "modernbert-embed-large" => Ok(fastembed::EmbeddingModel::ModernBertEmbedLarge),
        other => anyhow::bail!(
            "unknown fastembed model '{other}'. \
             Use the enum variant name (e.g. BGEM3, AllMiniLML6V2, SnowflakeArcticEmbedS) \
             or a common alias (e.g. bge-m3, all-minilm-l6-v2). \
             See fastembed::EmbeddingModel docs for the full list."
        ),
    }
}
