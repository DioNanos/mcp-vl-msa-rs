use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{MsaError, Result};
use crate::schema::ChunkConfig;

/// Runtime configuration. Loaded from `MCP_VL_MSA_CONFIG` (TOML) or filled with
/// defaults rooted at `~/.local/state/mcp-vl-msa-rs/`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MsaConfig {
    pub storage: StorageConfig,
    #[serde(default)]
    pub chunking: ChunkConfig,
    /// Indexing knobs (writer heap budget). Defaults are tuned for low-RAM
    /// edge targets (Raspberry Pi 3 class).
    #[serde(default)]
    pub indexing: IndexingConfig,
    /// Default per-round knobs for `msa_interleave_round` (A5 tier defaults).
    #[serde(default)]
    pub interleave: InterleaveConfig,
    /// Filesystem source adapter (`msa_sync_path`, feature `source-fs`).
    /// Disabled by default; ignored entirely on builds without the feature.
    #[serde(default)]
    pub sync: SyncConfig,
    /// Agent-memory surface (`msa_remember` / `msa_forget`).
    #[serde(default)]
    pub agent_memory: AgentMemoryConfig,
    /// Optional dense scorer. Has effect only when the binary is built
    /// with `--features embeddings`. With the feature off the section is
    /// accepted but ignored, with a warning logged at startup.
    #[serde(default)]
    pub embeddings: Option<EmbeddingsConfig>,
}

/// Configuration for the agent-memory surface (`msa_remember` / `msa_forget`).
///
/// The default collection is used when `msa_remember` is called without an
/// explicit `collection` parameter.  If neither the parameter nor the config
/// supplies a collection, the tool returns `invalid_params`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMemoryConfig {
    /// Default collection for `msa_remember` / `msa_forget` when the caller
    /// omits the `collection` field.  Must be a valid collection name
    /// (ASCII alphanumerics + `_`, `-`, `:`).  Default: `"agent-memory"`.
    #[serde(default = "default_agent_memory_collection")]
    pub default_collection: String,
    /// Max chars of `text` that pass the low-signal gate without `force: true`.
    /// Setting this to `0` makes the gate a no-op (always passes).  Default: 80.
    #[serde(default = "default_agent_memory_min_chars")]
    pub min_signal_chars: usize,
}

fn default_agent_memory_collection() -> String {
    "agent-memory".to_string()
}

fn default_agent_memory_min_chars() -> usize {
    80
}

impl Default for AgentMemoryConfig {
    fn default() -> Self {
        Self {
            default_collection: default_agent_memory_collection(),
            min_signal_chars: default_agent_memory_min_chars(),
        }
    }
}

/// Indexing-time configuration. The single knob that matters on weak hardware
/// is the per-writer heap arena: tantivy keeps an in-RAM arena per index
/// writer and flushes a segment when it fills. A long-lived writer with a
/// small arena (default 15 MiB, tantivy's minimum) keeps peak RAM bounded on
/// a Raspberry Pi 3 while still amortizing well under batch indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingConfig {
    /// Writer heap arena in bytes. Clamped at open time to tantivy's supported
    /// range (>= ~15 MiB). Bump on big machines for higher indexing throughput.
    #[serde(default = "default_writer_heap_bytes")]
    pub writer_heap_bytes: usize,
    /// Max documents (upserts + deletes) accepted in one `apply_delta` batch.
    /// Beyond this the call is `invalid_params`; a sync client chunks its batch.
    /// Floor-friendly default (Codex design audit §6); raise on big machines.
    #[serde(default = "default_max_batch_docs")]
    pub max_batch_docs: usize,
    /// Max total `text` bytes accepted in one `apply_delta` batch.
    #[serde(default = "default_max_batch_text_bytes")]
    pub max_batch_text_bytes: usize,
    /// Max raw request JSON bytes a batch tool will accept (enforced at the
    /// handler before deep work; the parse itself already happened).
    #[serde(default = "default_max_batch_json_bytes")]
    pub max_batch_json_bytes: usize,
}

/// Default batch ceilings — floor-friendly (Codex §6).
pub fn default_max_batch_docs() -> usize {
    100
}
pub fn default_max_batch_text_bytes() -> usize {
    32 * 1024 * 1024
}
pub fn default_max_batch_json_bytes() -> usize {
    48 * 1024 * 1024
}

/// tantivy's documented minimum writer budget (`MEMORY_BUDGET_NUM_BYTES_MIN`
/// = 15 × 1_000_000). Used as both the default arena and the clamp floor, so
/// there is a single source of truth for the smallest legal heap.
pub const MIN_WRITER_HEAP_BYTES: usize = 15_000_000;

/// Default writer heap arena: the tantivy minimum, a sane RPi3 default.
pub fn default_writer_heap_bytes() -> usize {
    MIN_WRITER_HEAP_BYTES
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            writer_heap_bytes: default_writer_heap_bytes(),
            max_batch_docs: default_max_batch_docs(),
            max_batch_text_bytes: default_max_batch_text_bytes(),
            max_batch_json_bytes: default_max_batch_json_bytes(),
        }
    }
}

/// Default per-round knobs for `msa_interleave_round` (A5), applied when the
/// client omits a field. These are the **tier** knobs (design §6): a Raspberry
/// Pi 3 floor deployment lowers them here; a top-tier box raises them. The
/// server still clamps every resolved value into its hard ceilings
/// ([`crate::interleave::InterleaveParams::clamped`]) regardless of config, so
/// config can tune *down* from the ceiling but never past it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterleaveConfig {
    #[serde(default = "default_il_top_k")]
    pub default_top_k: usize,
    #[serde(default = "default_il_fetch_top_n")]
    pub default_fetch_top_n: usize,
    #[serde(default = "default_il_max_chars_per_doc")]
    pub default_max_chars_per_doc: usize,
    #[serde(default = "default_il_max_total_chars")]
    pub default_max_total_chars: usize,
    #[serde(default = "default_il_max_rounds")]
    pub default_max_rounds: u32,
    #[serde(default = "default_il_low_gain_threshold")]
    pub default_low_gain_threshold: f32,
}

fn default_il_top_k() -> usize {
    crate::interleave::DEFAULT_TOP_K
}
fn default_il_fetch_top_n() -> usize {
    crate::interleave::DEFAULT_FETCH_TOP_N
}
fn default_il_max_chars_per_doc() -> usize {
    crate::interleave::DEFAULT_MAX_CHARS_PER_DOC
}
fn default_il_max_total_chars() -> usize {
    crate::interleave::DEFAULT_MAX_TOTAL_CHARS
}
fn default_il_max_rounds() -> u32 {
    crate::interleave::DEFAULT_MAX_ROUNDS
}
fn default_il_low_gain_threshold() -> f32 {
    crate::interleave::DEFAULT_LOW_GAIN_THRESHOLD
}

impl Default for InterleaveConfig {
    fn default() -> Self {
        Self {
            default_top_k: default_il_top_k(),
            default_fetch_top_n: default_il_fetch_top_n(),
            default_max_chars_per_doc: default_il_max_chars_per_doc(),
            default_max_total_chars: default_il_max_total_chars(),
            default_max_rounds: default_il_max_rounds(),
            default_low_gain_threshold: default_il_low_gain_threshold(),
        }
    }
}

/// Backend selector for the dense embedding scorer.
///
/// Production default is `candle-modernbert` (in-process, no daemon).
/// `ollama` is a deprecated transitional backend kept for one release.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsConfig {
    /// Backend identifier: `"candle-modernbert"` or `"ollama"` (deprecated).
    pub backend: String,

    /// Embedding dimension produced by the model. Used to validate cached
    /// embeddings at search time and to namespace shadow generations.
    pub dim: usize,

    /// Model identity used in shadow-generation namespacing. The triple
    /// `(model_id, model_version, profile)` must change together when
    /// upgrading; the storage layer keys vectors by this triple.
    #[serde(default)]
    pub model_id: String,
    #[serde(default)]
    pub model_version: String,
    #[serde(default)]
    pub profile: String,

    /// Local model bundle directory (candle-modernbert only). The bundle is
    /// expected to contain `config.json`, `tokenizer.json`, and a safetensors
    /// file with weight keys already prefixed with `model.` (see
    /// `scripts/prepare-granite-r2-97m.sh`).
    ///
    /// No automatic HF download happens at runtime — the server must run
    /// offline-deterministic. If `model_dir` is absent the backend fails fast.
    #[serde(default)]
    pub model_dir: Option<PathBuf>,

    /// Maximum input length in tokens. Conservative default (8192) leaves
    /// headroom on smartphone Termux targets even though Granite R2 supports
    /// 32K context. Increase only after on-device memory profiling.
    #[serde(default)]
    pub max_len: Option<usize>,

    /// Base URL of the deprecated Ollama-compatible service (e.g.
    /// `http://127.0.0.1:11434`). Only honored when `backend == "ollama"`.
    #[serde(default)]
    pub url: Option<String>,

    /// Model name passed to the deprecated Ollama backend.
    #[serde(default)]
    pub model: Option<String>,
}

/// Filesystem source adapter config for `msa_sync_path` (server feature
/// `source-fs`). Pure data — the FS-walk logic lives in the server. Disabled
/// by default and `allowed_roots` empty, so nothing can be read until a
/// deployment opts in (Codex design audit §5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    /// `msa_sync_path` is refused unless this is true.
    #[serde(default)]
    pub enabled: bool,
    /// Allowed root directories; a sync `path` must canonicalize to under one
    /// of these. Empty = nothing allowed.
    #[serde(default)]
    pub allowed_roots: Vec<PathBuf>,
    /// Max files scanned per run.
    #[serde(default = "default_sync_max_files")]
    pub max_files: usize,
    /// Max bytes read from a single file (larger files are skipped as errors).
    #[serde(default = "default_sync_max_file_bytes")]
    pub max_file_bytes: usize,
    /// Follow symlinked directories while walking (default false: do not).
    #[serde(default)]
    pub follow_symlinks: bool,
    /// Default include globs when the request omits them.
    #[serde(default = "default_sync_globs")]
    pub default_globs: Vec<String>,
}

pub fn default_sync_max_files() -> usize {
    20_000
}
pub fn default_sync_max_file_bytes() -> usize {
    10 * 1024 * 1024
}
pub fn default_sync_globs() -> Vec<String> {
    vec!["**/*.md".to_string(), "**/*.txt".to_string()]
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            allowed_roots: Vec::new(),
            max_files: default_sync_max_files(),
            max_file_bytes: default_sync_max_file_bytes(),
            follow_symlinks: false,
            default_globs: default_sync_globs(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Base directory: each collection lives under `<storage_dir>/<name>/`.
    pub storage_dir: PathBuf,
}

impl Default for MsaConfig {
    fn default() -> Self {
        Self {
            storage: StorageConfig {
                storage_dir: default_storage_dir(),
            },
            chunking: ChunkConfig::default(),
            indexing: IndexingConfig::default(),
            interleave: InterleaveConfig::default(),
            sync: SyncConfig::default(),
            agent_memory: AgentMemoryConfig::default(),
            embeddings: None,
        }
    }
}

fn default_storage_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".local/state/mcp-vl-msa-rs")
    } else {
        PathBuf::from(".mcp-vl-msa-rs")
    }
}

impl MsaConfig {
    /// Load config from `MCP_VL_MSA_CONFIG` env path if set, otherwise defaults.
    pub fn load() -> Result<Self> {
        match std::env::var("MCP_VL_MSA_CONFIG") {
            Ok(path) => {
                let raw = std::fs::read_to_string(&path)
                    .map_err(|e| MsaError::Config(format!("read {path}: {e}")))?;
                toml::from_str(&raw).map_err(|e| MsaError::Config(format!("parse {path}: {e}")))
            }
            Err(_) => Ok(Self::default()),
        }
    }

    /// Resolve a collection's on-disk path, rejecting unsafe names. Returns an
    /// error for any name that fails [`crate::validate::validate_collection_name`]
    /// (separators, `..`, absolute, non-ASCII, etc.) so a bad name can never be
    /// turned into a path that escapes the storage root.
    pub fn collection_path(&self, name: &str) -> Result<PathBuf> {
        crate::validate::validate_collection_name(name)?;
        Ok(self.storage.storage_dir.join(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MsaConfig {
        MsaConfig {
            storage: StorageConfig {
                storage_dir: PathBuf::from("/tmp/msa-test-root"),
            },
            ..Default::default()
        }
    }

    #[test]
    fn collection_path_rejects_traversal() {
        let c = cfg();
        assert!(c.collection_path("../etc").is_err());
        assert!(c.collection_path("a/b").is_err());
        assert!(c.collection_path("").is_err());
    }

    #[test]
    fn collection_path_accepts_valid_and_stays_under_root() {
        let c = cfg();
        let p = c.collection_path("vivling::abc-1").expect("valid name");
        assert!(p.starts_with("/tmp/msa-test-root"));
        assert!(p.ends_with("vivling::abc-1"));
    }
}
