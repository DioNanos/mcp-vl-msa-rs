//! Per-model prompt-prefix decorator for the Ollama benchmark backend.
//!
//! Ollama serves these embedding models with a passthrough template
//! (`{{ .Prompt }}`), so the asymmetric query/passage prompt prefixes several
//! models were trained with are **not** applied by the server. This decorator
//! restores them at the bench level, keyed by [`EmbedInputKind`], so encoders
//! like nomic-v2 (`search_query:` / `search_document:`), arctic-embed
//! (`query:`), qwen3-embedding (instruct) and EmbeddingGemma are evaluated the
//! way they were trained instead of being handicapped in the comparison.

use msa_core::embeddings::{DenseEncoder, EmbedInputKind};
use msa_core::error::Result;

/// Wraps a [`DenseEncoder`], prepending a kind-dependent prefix to every text
/// before delegating. An empty prefix is a passthrough for that kind.
pub struct PrefixedEncoder<E> {
    inner: E,
    query_prefix: String,
    doc_prefix: String,
}

impl<E: DenseEncoder> PrefixedEncoder<E> {
    pub fn new(inner: E, query_prefix: String, doc_prefix: String) -> Self {
        Self {
            inner,
            query_prefix,
            doc_prefix,
        }
    }
}

impl<E: DenseEncoder> DenseEncoder for PrefixedEncoder<E> {
    fn encode_batch_kind(&self, kind: EmbedInputKind, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let prefix = match kind {
            EmbedInputKind::Query => &self.query_prefix,
            EmbedInputKind::Passage => &self.doc_prefix,
        };
        if prefix.is_empty() {
            return self.inner.encode_batch_kind(kind, texts);
        }
        let owned: Vec<String> = texts.iter().map(|t| format!("{prefix}{t}")).collect();
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
        self.inner.encode_batch_kind(kind, &refs)
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn model_version(&self) -> &str {
        self.inner.model_version()
    }
    fn profile(&self) -> &str {
        self.inner.profile()
    }
}

/// Built-in query/passage prefixes for known Ollama embedding models, matched
/// on the tag-stripped model name. Returns `("", "")` for models that need no
/// prefix (e.g. bge-m3) or that are unknown. Sources: official model cards
/// (nomic-embed v2, Snowflake Arctic, Qwen3-Embedding, Google EmbeddingGemma).
pub fn default_prefixes(model: &str) -> (String, String) {
    let base = model.split(':').next().unwrap_or(model);
    let (q, d): (&str, &str) = match base {
        "nomic-embed-text-v2-moe" | "nomic-embed-text" => {
            ("search_query: ", "search_document: ")
        }
        "snowflake-arctic-embed2" | "snowflake-arctic-embed" => ("query: ", ""),
        "qwen3-embedding" => (
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
            "",
        ),
        "embeddinggemma" => ("task: search result | query: ", "title: none | text: "),
        _ => ("", ""),
    };
    (q.to_string(), d.to_string())
}

/// Interpret a CLI-provided prefix, turning the literal escapes `\n`/`\t` into
/// real control characters so multi-line instruct prefixes fit on one CLI arg.
pub fn unescape_cli(s: &str) -> String {
    s.replace("\\n", "\n").replace("\\t", "\t")
}
