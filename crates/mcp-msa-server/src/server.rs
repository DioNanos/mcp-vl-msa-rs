//! MCP stdio server. The tool handlers wrap [`msa_core::index::MsaIndex`] with
//! a [`CollectionRegistry`] so each call resolves (or lazily creates) the
//! requested collection.
//!
//! Tools exposed:
//! - `msa_index` — index a document
//! - `msa_search` — top-k chunk retrieval
//! - `msa_fetch_doc` — original text injection (paper §4.3)
//! - `msa_delete` — remove a document
//! - `msa_list_collections` — collections on disk (not only those open)
//! - `msa_drop_collection` — delete a whole collection
//! - `msa_stats` — per-collection statistics
//! - `msa_manifest` — paginated fingerprint manifest for O(delta) sync diffing (A6)
//! - `msa_sync_delta` — atomic upsert+delete batch (primary A6 sync primitive)
//! - `msa_index_batch` — atomic batch upsert (upsert-only convenience)
//! - `msa_delete_batch` — atomic batch delete (delete-only, source-scoped)
//! - `msa_sync_path` — mirror a filesystem directory into a collection (feature `source-fs`)
//! - `msa_interleave_round` — one bounded, stateless Memory Interleave round: route → dedup → fetch → inject (paper §3.5)
//! - `msa_search_iterative` — legacy server-side-cursor multi-hop helper (no injection); superseded by `msa_interleave_round`
//! - `msa_drop_session` — drop a `msa_search_iterative` session early

use std::borrow::Cow;
use std::sync::Arc;

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::*;
use rmcp::{tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler};
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::Deserialize;
use uuid::Uuid;

use msa_core::config::MsaConfig;
use msa_core::embeddings::{EmbedInputKind, SharedEncoder};
use msa_core::index::CollectionRegistry;
use msa_core::interleave::{run_round, DedupMode, InterleaveParams, InterleaveStateV1};
use msa_core::remember::{forget_by_source_id, remember, MemoryKind};
use msa_core::schema::{Document, SearchFilter};
use msa_core::session::{MsaSession, SessionRegistry};

#[derive(Clone)]
pub struct MsaServer {
    config: Arc<MsaConfig>,
    registry: Arc<CollectionRegistry>,
    sessions: Arc<SessionRegistry>,
    /// Optional dense encoder. `None` keeps the server in pure BM25 mode.
    scorer: Option<SharedEncoder>,
    /// Held to keep the rmcp `#[tool_router]` macro state alive; reads are
    /// internal to rmcp and not visible to dead-code analysis.
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct IndexParams {
    pub collection: String,
    pub doc_id: String,
    pub text: String,
    #[serde(default)]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SearchParams {
    pub collection: String,
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Optional pre-retrieval filter on document metadata and created_at.
    #[serde(default)]
    pub filter: Option<SearchFilter>,
    /// BM25 weight for hybrid scoring in `[0.0, 1.0]`. `1.0` (default) is
    /// BM25-only; values below 1.0 require the server to be configured
    /// with a dense scorer (feature `embeddings` + config). When the
    /// dense scorer is absent the server returns an error.
    #[serde(default)]
    pub dense_alpha: Option<f32>,
}

fn default_top_k() -> usize {
    16
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FetchDocParams {
    pub collection: String,
    pub doc_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DeleteParams {
    pub collection: String,
    pub doc_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StatsParams {
    pub collection: String,
}

/// Parameters for `msa_manifest` (A6 — paginated fingerprint manifest).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ManifestParams {
    pub collection: String,
    /// Only list docs whose stored `source_id` equals this (scope a sync).
    #[serde(default)]
    pub source_id: Option<String>,
    /// Only list docs whose `doc_id` starts with this prefix.
    #[serde(default)]
    pub doc_id_prefix: Option<String>,
    /// Page size; clamped to `[1, 5000]`, default 1000.
    #[serde(default)]
    pub limit: Option<usize>,
    /// Last `doc_id` of the previous page (exclusive lower bound). `null` = first page.
    #[serde(default)]
    pub cursor: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DropCollectionParams {
    pub collection: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SearchIterativeParams {
    pub collection: String,
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// `None` starts a new session; pass the returned `session_id` on
    /// subsequent rounds to dedup against the chunks already shown.
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub filter: Option<SearchFilter>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DropSessionParams {
    pub session_id: String,
}

/// One document for a batch upsert (`msa_sync_delta` / `msa_index_batch`).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DocInput {
    pub doc_id: String,
    pub text: String,
    #[serde(default)]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

/// Parameters for `msa_sync_delta` (A6 — atomic upsert + delete).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SyncDeltaParams {
    pub collection: String,
    /// Scope tag stored on each upserted doc; also scopes deletes
    /// (a scoped delete never evicts another source's docs) and the
    /// `if_manifest_digest` precondition.
    #[serde(default)]
    pub source_id: Option<String>,
    /// Skip an upsert already indexed with the same content + index profile.
    #[serde(default = "default_skip_unchanged")]
    pub skip_unchanged: bool,
    /// Optimistic concurrency: the scope's current manifest_digest must equal
    /// this, else the call fails (nothing written). Get it from `msa_manifest`.
    #[serde(default)]
    pub if_manifest_digest: Option<String>,
    /// Optimistic concurrency on the index profile (chunking/model identity).
    #[serde(default)]
    pub if_index_profile_hash: Option<String>,
    #[serde(default)]
    pub upsert_docs: Vec<DocInput>,
    #[serde(default)]
    pub delete_doc_ids: Vec<String>,
}

/// Parameters for `msa_index_batch` (upsert-only convenience over sync_delta).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct IndexBatchParams {
    pub collection: String,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default = "default_skip_unchanged")]
    pub skip_unchanged: bool,
    #[serde(default)]
    pub docs: Vec<DocInput>,
}

/// Parameters for `msa_delete_batch` (delete-only convenience over sync_delta).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DeleteBatchParams {
    pub collection: String,
    /// If set, a doc whose stored `source_id` differs is NOT deleted.
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub doc_ids: Vec<String>,
}

fn default_skip_unchanged() -> bool {
    true
}

/// Parameters for `msa_sync_path` (A6 — filesystem source adapter, feature
/// `source-fs`). Mirrors a directory into a collection in one call.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SyncPathParams {
    pub collection: String,
    /// Directory to mirror; must canonicalize to under a configured allowed_root.
    pub path: String,
    /// Sync scope (required when `prune` is true). Tags upserts; bounds prune.
    #[serde(default)]
    pub source_id: Option<String>,
    /// Include globs (relative to `path`); empty = the configured defaults.
    #[serde(default)]
    pub globs: Vec<String>,
    /// Exclude globs, applied after include.
    #[serde(default)]
    pub exclude_globs: Vec<String>,
    /// Prefix prepended to each file's relative path to form its doc_id.
    #[serde(default)]
    pub doc_id_prefix: Option<String>,
    /// Delete in-scope docs no longer present on disk (needs `source_id`; never
    /// runs if any file errored during the scan).
    #[serde(default)]
    pub prune: bool,
    /// Override the configured `follow_symlinks`.
    #[serde(default)]
    pub follow_symlinks: Option<bool>,
    /// Plan only: scan + diff, report counts, write nothing.
    #[serde(default)]
    pub dry_run: bool,
}

/// Parameters for `msa_interleave_round` (A5, stateless Memory Interleave).
///
/// Every numeric knob is optional: an omitted knob falls back to the server's
/// configured tier default ([`msa_core::config::InterleaveConfig`]), then every
/// value is clamped into the hard server ceiling. The client owns the stop
/// decision by inspecting `stop_hints` + `stats` and choosing whether to call
/// again.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InterleaveRoundParams {
    pub collection: String,
    /// The query for THIS round. The client reformulates it each round; the
    /// server does not rewrite it.
    pub query: String,
    /// `null`/omitted starts round 1. Otherwise pass back the `state` object
    /// returned by the previous round (stateless round-trip, design D1).
    #[serde(default)]
    pub state: Option<InterleaveStateV1>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub fetch_top_n: Option<usize>,
    #[serde(default)]
    pub max_chars_per_doc: Option<usize>,
    #[serde(default)]
    pub max_total_chars: Option<usize>,
    #[serde(default)]
    pub max_rounds: Option<u32>,
    #[serde(default)]
    pub low_gain_threshold: Option<f32>,
    /// BM25 weight in `[0.0, 1.0]`. `1.0`/omitted is BM25-only; `< 1.0`
    /// requires the server to be started with a dense scorer (else error,
    /// parity with `msa_search`).
    #[serde(default)]
    pub dense_alpha: Option<f32>,
    /// Dedup strategy. Only `"id"` is supported in A5; an unknown mode is
    /// rejected with the list of supported modes.
    #[serde(default = "default_dedup_mode")]
    pub dedup_mode: String,
    #[serde(default)]
    pub filter: Option<SearchFilter>,
}

fn default_dedup_mode() -> String {
    "id".to_string()
}

// ── agent-memory: msa_remember / msa_forget ──────────────────────────────────

/// Parameters for `msa_remember`: index a memory capsule via the enrich pipeline.
///
/// The collection defaults to `[agent_memory] default_collection` in the server
/// config (usually `"agent-memory"`).  If neither the parameter nor the config
/// provides one, the tool returns `invalid_params`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RememberParams {
    /// Target collection.  Omit to use the server-configured default
    /// (`[agent_memory] default_collection`, usually `"agent-memory"`).
    #[serde(default)]
    pub collection: Option<String>,
    /// The fact, episode, or preference to store.  Passed through the enrich
    /// sanitization pipeline before indexing; secrets are redacted.
    pub text: String,
    /// Memory kind for filtering: `fact` (default), `episode`, `preference`, `task`.
    #[serde(default)]
    pub kind: Option<String>,
    /// Optional scope tag (e.g. `"session-2026-06-12"`).  Used by `msa_forget`
    /// with `source_id` to batch-delete all memories from a session.
    #[serde(default)]
    pub source_id: Option<String>,
    /// Explicit doc_id for updates.  Omit to let the server generate a ULID.
    /// Supplying the same `doc_id` twice replaces the previous memory.
    #[serde(default)]
    pub doc_id: Option<String>,
    /// Free-form metadata merged into the document.  Standard server keys
    /// (`kind`, `source_id`, `created_at`, `source`) always win if they
    /// conflict with caller-supplied keys.
    #[serde(default)]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
    /// Bypass the low-signal gate.  Use when the text is intentionally short or
    /// repetitive (e.g. a single-word preference).  The bypass is logged at WARN.
    #[serde(default)]
    pub force: bool,
}

/// Parameters for `msa_forget`: delete one or more memory capsules.
///
/// Exactly one of `doc_id` (single delete) or `source_id` (batch delete by
/// scope tag) must be supplied — not both.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ForgetParams {
    /// Target collection.  Defaults to the server-configured default.
    #[serde(default)]
    pub collection: Option<String>,
    /// Delete a single document by id.  Mutually exclusive with `source_id`.
    #[serde(default)]
    pub doc_id: Option<String>,
    /// Batch-delete all documents whose `source_id` matches this value.
    /// Mutually exclusive with `doc_id`.  Returns the count of deleted docs.
    #[serde(default)]
    pub source_id: Option<String>,
}

// schemars 1.2 emits empty-property schemas without `properties: {}`, which
// Anthropic's tool-schema validator rejects (drops the entire tools/list).
// Hand-write the schema for the no-arg tools to keep the catalog visible.
#[derive(Debug, Default, Deserialize)]
pub struct EmptyParams {}

impl JsonSchema for EmptyParams {
    fn schema_name() -> Cow<'static, str> {
        Cow::Borrowed("EmptyParams")
    }
    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        schemars::json_schema!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        })
    }
}

#[tool_router]
impl MsaServer {
    pub fn new(config: MsaConfig) -> Self {
        Self::with_scorer(config, None)
    }

    pub fn with_scorer(config: MsaConfig, scorer: Option<SharedEncoder>) -> Self {
        Self {
            config: Arc::new(config),
            registry: Arc::new(CollectionRegistry::new()),
            sessions: Arc::new(SessionRegistry::with_default_ttl()),
            scorer,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Index a document into a collection. Existing chunks for the same doc_id are replaced."
    )]
    pub async fn msa_index(
        &self,
        Parameters(p): Parameters<IndexParams>,
    ) -> std::result::Result<CallToolResult, McpError> {
        // Bound caller inputs before touching the filesystem or building the
        // chunk stream. Collection name is validated inside open_or_create.
        msa_core::validate::validate_doc_id(&p.doc_id).map_err(bad_input)?;
        msa_core::validate::validate_text(&p.text, msa_core::validate::DEFAULT_MAX_TEXT_BYTES)
            .map_err(bad_input)?;
        let idx = self
            .registry
            .open_or_create_with_heap(
                &p.collection,
                &self.config.storage.storage_dir,
                &self.config.chunking,
                self.config.indexing.writer_heap_bytes,
            )
            .map_err(bad_input)?;
        let doc = Document {
            id: p.doc_id,
            text: p.text,
            metadata: p
                .metadata
                .map(|m| m.into_iter().collect())
                .unwrap_or_default(),
            created_at: chrono::Utc::now(),
        };
        // index_document can return InvalidInput (e.g. reserved `_msa` metadata,
        // A6) which is a caller fault -> invalid_params, not internal_error.
        let n_chunks = idx
            .index_document(&doc, self.scorer.as_deref().map(|s| s as _))
            .map_err(user_or_internal)?;
        Ok(ok_json(serde_json::json!({
            "collection": p.collection,
            "doc_id": doc.id,
            "num_chunks": n_chunks
        })))
    }

    #[tool(
        description = "Top-k chunk retrieval over a collection (BM25, score normalized 0.0-1.0)."
    )]
    pub async fn msa_search(
        &self,
        Parameters(p): Parameters<SearchParams>,
    ) -> std::result::Result<CallToolResult, McpError> {
        msa_core::validate::validate_query(&p.query).map_err(bad_input)?;
        let idx = self
            .registry
            .open_or_create_with_heap(
                &p.collection,
                &self.config.storage.storage_dir,
                &self.config.chunking,
                self.config.indexing.writer_heap_bytes,
            )
            .map_err(open_err)?;
        let hits = self
            .run_search(
                &idx,
                &p.query,
                msa_core::validate::clamp_top_k(p.top_k),
                p.filter.as_ref(),
                p.dense_alpha,
                &Default::default(),
            )
            .map_err(internal)?;
        Ok(ok_json(serde_json::json!({ "hits": hits })))
    }

    #[tool(
        description = "Fetch the full original text of a document (MSA-style original text injection, paper §4.3)."
    )]
    pub async fn msa_fetch_doc(
        &self,
        Parameters(p): Parameters<FetchDocParams>,
    ) -> std::result::Result<CallToolResult, McpError> {
        msa_core::validate::validate_doc_id(&p.doc_id).map_err(bad_input)?;
        let idx = self
            .registry
            .open_or_create_with_heap(
                &p.collection,
                &self.config.storage.storage_dir,
                &self.config.chunking,
                self.config.indexing.writer_heap_bytes,
            )
            .map_err(open_err)?;
        let doc = idx.fetch_doc(&p.doc_id).map_err(internal)?;
        Ok(ok_json(serde_json::json!(doc)))
    }

    #[tool(description = "Remove a document and all its chunks from a collection.")]
    pub async fn msa_delete(
        &self,
        Parameters(p): Parameters<DeleteParams>,
    ) -> std::result::Result<CallToolResult, McpError> {
        msa_core::validate::validate_doc_id(&p.doc_id).map_err(bad_input)?;
        let idx = self
            .registry
            .open_or_create_with_heap(
                &p.collection,
                &self.config.storage.storage_dir,
                &self.config.chunking,
                self.config.indexing.writer_heap_bytes,
            )
            .map_err(open_err)?;
        idx.delete_document(&p.doc_id).map_err(internal)?;
        Ok(ok_json(serde_json::json!({ "deleted": p.doc_id })))
    }

    #[tool(
        description = "List all collections, including ones persisted on disk from previous runs \
                       (not only those already open in this process)."
    )]
    pub async fn msa_list_collections(
        &self,
        Parameters(_): Parameters<EmptyParams>,
    ) -> std::result::Result<CallToolResult, McpError> {
        let collections = self.registry.list_on_disk(&self.config.storage.storage_dir);
        Ok(ok_json(serde_json::json!({ "collections": collections })))
    }

    #[tool(
        description = "Delete an entire collection and all its documents from disk. \
                       Irreversible. Returns dropped=false if the collection did not exist."
    )]
    pub async fn msa_drop_collection(
        &self,
        Parameters(p): Parameters<DropCollectionParams>,
    ) -> std::result::Result<CallToolResult, McpError> {
        let dropped = self
            .registry
            .drop_collection(&p.collection, &self.config.storage.storage_dir)
            .map_err(open_err)?;
        Ok(ok_json(
            serde_json::json!({ "collection": p.collection, "dropped": dropped }),
        ))
    }

    /// **Deprecated** since A5 — superseded by [`Self::msa_interleave_round`],
    /// which is stateless (no server session to lose) and runs the full Memory
    /// Interleave round (route + dedup + fetch + original-text injection). This
    /// dedup-only cursor helper is kept for backward compatibility and will be
    /// removed in a future major. New callers should use `msa_interleave_round`.
    #[tool(
        description = "[DEPRECATED — use msa_interleave_round] Dedup-only iterative top-k retrieval with a \
                       server-side cursor (no original-text injection). Superseded by msa_interleave_round, \
                       which is stateless and does the full Memory Interleave round (route+dedup+fetch+inject). \
                       Kept for backward compatibility; will be removed in a future major. Pass session_id from \
                       a previous response to exclude chunks already shown; supply the next round's query \
                       yourself. Stop when exhausted=true. Sessions expire after ~10 minutes of idle."
    )]
    pub async fn msa_search_iterative(
        &self,
        Parameters(p): Parameters<SearchIterativeParams>,
    ) -> std::result::Result<CallToolResult, McpError> {
        msa_core::validate::validate_query(&p.query).map_err(bad_input)?;
        let idx = self
            .registry
            .open_or_create_with_heap(
                &p.collection,
                &self.config.storage.storage_dir,
                &self.config.chunking,
                self.config.indexing.writer_heap_bytes,
            )
            .map_err(open_err)?;

        let mut session = match p.session_id.as_deref() {
            Some(sid) => match self.sessions.take(sid) {
                Some(existing) => {
                    if existing.collection != p.collection {
                        return Err(McpError::invalid_params(
                            format!(
                                "session {sid} is bound to collection '{}', not '{}'",
                                existing.collection, p.collection
                            ),
                            None,
                        ));
                    }
                    existing
                }
                None => {
                    return Err(McpError::invalid_params(
                        format!("session {sid} not found or expired"),
                        None,
                    ))
                }
            },
            None => MsaSession::new(
                Uuid::new_v4().to_string(),
                p.collection.clone(),
                p.query.clone(),
            ),
        };

        let hits = self
            .run_search(
                &idx,
                &p.query,
                msa_core::validate::clamp_top_k(p.top_k),
                p.filter.as_ref(),
                None,
                &session.seen_doc_ids,
            )
            .map_err(internal)?;

        for h in &hits {
            session.seen_doc_ids.insert(h.doc_id.clone());
        }
        session.round += 1;
        let exhausted = hits.is_empty();
        let response = serde_json::json!({
            "session_id": session.id,
            "round": session.round,
            "hits": hits,
            "exhausted": exhausted,
            "seen_doc_count": session.seen_doc_ids.len(),
        });
        // Keep the session warm even when exhausted, so the AI can request
        // a different facet via `msa_search_iterative` with a new query in
        // the same session. The TTL will eventually clean it up.
        self.sessions.store(session);
        Ok(ok_json(response))
    }

    #[tool(
        description = "One bounded Memory Interleave round (MSA pattern, paper §3.5): route -> dedup -> \
                       fetch -> inject. Returns hits, injected original text (bounded by a char budget), \
                       the updated `state`, telemetry (stats/budget), and stop_hints. STATELESS: pass the \
                       returned `state` back on the next round (no server session to lose on reconnect). \
                       YOU reformulate the query each round and decide when to stop — read stop_hints + \
                       stats and stop when you have enough to answer, or on hard_stop. Stop on COVERAGE, \
                       not speed: marginal_gain_chars falls by construction and a rising dedup_ratio only \
                       signals a collapsing candidate pool — on a multi-hop task do not stop at low_gain / \
                       dedup hints until you have seen every entity the question needs. This is the full \
                       Memory Interleave loop that supersedes msa_search_iterative (kept for now)."
    )]
    pub async fn msa_interleave_round(
        &self,
        Parameters(p): Parameters<InterleaveRoundParams>,
    ) -> std::result::Result<CallToolResult, McpError> {
        msa_core::validate::validate_query(&p.query).map_err(bad_input)?;
        let dedup_mode = DedupMode::parse(&p.dedup_mode).map_err(bad_input)?;
        let state = InterleaveStateV1::resolve(p.state).map_err(bad_input)?;

        // Merge client knobs over the configured tier defaults, then clamp into
        // the hard server ceilings (config can tune down, never past the cap).
        let il = &self.config.interleave;
        let params = InterleaveParams::clamped(
            p.top_k.unwrap_or(il.default_top_k),
            p.fetch_top_n.unwrap_or(il.default_fetch_top_n),
            p.max_chars_per_doc.unwrap_or(il.default_max_chars_per_doc),
            p.max_total_chars.unwrap_or(il.default_max_total_chars),
            p.max_rounds.unwrap_or(il.default_max_rounds),
            p.low_gain_threshold
                .unwrap_or(il.default_low_gain_threshold),
            dedup_mode,
        );

        let idx = self
            .registry
            .open_or_create_with_heap(
                &p.collection,
                &self.config.storage.storage_dir,
                &self.config.chunking,
                self.config.indexing.writer_heap_bytes,
            )
            .map_err(open_err)?;

        // Compute the query embedding only when a dense rerank is requested and
        // a scorer is configured (else a clean error, parity with msa_search).
        let dense_alpha = p.dense_alpha.unwrap_or(1.0);
        let q_emb = self
            .query_embedding_for(&p.query, p.dense_alpha)
            .map_err(user_or_internal)?;

        let response = run_round(
            &idx,
            &p.query,
            &state,
            &params,
            p.filter.as_ref(),
            dense_alpha,
            q_emb.as_deref(),
        )
        .map_err(user_or_internal)?;

        Ok(ok_json(serde_json::json!(response)))
    }

    #[tool(description = "Drop a Memory Interleave session immediately, without waiting for TTL.")]
    pub async fn msa_drop_session(
        &self,
        Parameters(p): Parameters<DropSessionParams>,
    ) -> std::result::Result<CallToolResult, McpError> {
        let dropped = self.sessions.drop_session(&p.session_id);
        Ok(ok_json(
            serde_json::json!({ "session_id": p.session_id, "dropped": dropped }),
        ))
    }

    #[tool(description = "Per-collection index statistics (chunk count, disk size).")]
    pub async fn msa_stats(
        &self,
        Parameters(p): Parameters<StatsParams>,
    ) -> std::result::Result<CallToolResult, McpError> {
        let idx = self
            .registry
            .open_or_create_with_heap(
                &p.collection,
                &self.config.storage.storage_dir,
                &self.config.chunking,
                self.config.indexing.writer_heap_bytes,
            )
            .map_err(open_err)?;
        let stats = idx.stats().map_err(internal)?;
        Ok(ok_json(serde_json::json!(stats)))
    }

    #[tool(
        description = "Paginated fingerprint manifest of a collection: one row per document with its \
                       content_hash, index_profile_hash, source_id, and hash_state (ok | legacy_missing \
                       | profile_stale vs the collection's CURRENT index profile). Use it to diff your \
                       corpus against the index in O(delta): fetch the manifest, compare hashes locally, \
                       then resend only changed/new docs (msa_sync_delta) and delete the missing ones. \
                       total_count and manifest_digest cover the whole filtered scope; pass manifest_digest \
                       back as msa_sync_delta's if_manifest_digest for optimistic concurrency. Page via \
                       cursor (last doc_id of the previous page); limit clamped to [1,5000], default 1000."
    )]
    pub async fn msa_manifest(
        &self,
        Parameters(p): Parameters<ManifestParams>,
    ) -> std::result::Result<CallToolResult, McpError> {
        let idx = self
            .registry
            .open_or_create_with_heap(
                &p.collection,
                &self.config.storage.storage_dir,
                &self.config.chunking,
                self.config.indexing.writer_heap_bytes,
            )
            .map_err(open_err)?;
        // The "current" profile = what an index write would use now (chunking +
        // configured scorer), so hash_state flags docs that need a reindex.
        let profile_hash = idx.current_index_profile_hash(self.scorer.as_deref().map(|s| s as _));
        let manifest = idx
            .manifest(
                &profile_hash,
                p.source_id.as_deref(),
                p.doc_id_prefix.as_deref(),
                p.limit.unwrap_or(msa_core::index::DEFAULT_MANIFEST_LIMIT),
                p.cursor.as_deref(),
            )
            .map_err(internal)?;
        Ok(ok_json(serde_json::json!({
            "collection": p.collection,
            "manifest_format_version": 3,
            "content_hash_algo": msa_core::fingerprint::CONTENT_HASH_ALGO,
            "index_profile_hash": profile_hash,
            "source_id": p.source_id,
            "total_count": manifest.total_count,
            "manifest_digest": manifest.manifest_digest,
            "next_cursor": manifest.next_cursor,
            "entries": manifest.entries,
        })))
    }

    #[tool(
        description = "Atomically apply an upsert + delete batch to a collection (the primary A6 sync \
                       primitive). One commit, rolled back on any error — a client crash never leaves \
                       the collection half-synced. skip_unchanged skips docs already indexed with the \
                       same content + index profile (no re-chunk/embed). source_id scopes the sync: it \
                       tags upserted docs and a scoped delete never evicts another source's docs. Pass \
                       if_manifest_digest (from msa_manifest) and/or if_index_profile_hash for optimistic \
                       concurrency: on mismatch the call fails with invalid_params and writes nothing. \
                       Returns per-doc results + new_manifest_digest (use as the next if_manifest_digest)."
    )]
    pub async fn msa_sync_delta(
        &self,
        Parameters(p): Parameters<SyncDeltaParams>,
    ) -> std::result::Result<CallToolResult, McpError> {
        self.run_delta(
            &p.collection,
            p.source_id,
            p.skip_unchanged,
            p.if_manifest_digest,
            p.if_index_profile_hash,
            p.upsert_docs,
            p.delete_doc_ids,
        )
    }

    #[tool(
        description = "Index/replace a batch of documents in one atomic commit (upsert-only convenience \
                       over msa_sync_delta). skip_unchanged (default true) skips docs already indexed \
                       with the same content + index profile. source_id tags the docs for later scoped \
                       sync. Far cheaper than calling msa_index per document."
    )]
    pub async fn msa_index_batch(
        &self,
        Parameters(p): Parameters<IndexBatchParams>,
    ) -> std::result::Result<CallToolResult, McpError> {
        self.run_delta(
            &p.collection,
            p.source_id,
            p.skip_unchanged,
            None,
            None,
            p.docs,
            Vec::new(),
        )
    }

    #[tool(
        description = "Delete a batch of documents in one atomic commit (delete-only convenience over \
                       msa_sync_delta). If source_id is set, a doc whose stored source_id differs is NOT \
                       deleted (counted in skipped_scope_mismatch), so one source cannot evict another's \
                       docs from a shared collection. Missing doc_ids are counted in not_found, not an error."
    )]
    pub async fn msa_delete_batch(
        &self,
        Parameters(p): Parameters<DeleteBatchParams>,
    ) -> std::result::Result<CallToolResult, McpError> {
        self.run_delta(
            &p.collection,
            p.source_id,
            true,
            None,
            None,
            Vec::new(),
            p.doc_ids,
        )
    }

    #[tool(
        description = "Store a memory capsule for an agent: sanitizes the text (secrets redacted), \
                       applies a low-signal gate (short/noisy input rejected unless force=true), \
                       deduplicates identical content, injects standard metadata (kind, source_id, \
                       created_at, source=agent), and indexes the enriched capsule. \
                       Returns doc_id (ULID), deduplicated flag (if same content already indexed), \
                       and low_signal_reason (if gate rejected without force). \
                       Use msa_forget to delete. Use msa_search with filter kind=episode etc to retrieve. \
                       Complement to msa_index: where msa_index is doc-centric, msa_remember is \
                       agent-memory-centric (opinionated pipeline, standardized metadata)."
    )]
    pub async fn msa_remember(
        &self,
        Parameters(p): Parameters<RememberParams>,
    ) -> std::result::Result<CallToolResult, McpError> {
        let collection =
            resolve_collection(p.collection, &self.config.agent_memory.default_collection)?;
        msa_core::validate::validate_collection_name(&collection).map_err(bad_input)?;
        let idx = self
            .registry
            .open_or_create_with_heap(
                &collection,
                &self.config.storage.storage_dir,
                &self.config.chunking,
                self.config.indexing.writer_heap_bytes,
            )
            .map_err(open_err)?;

        let kind = MemoryKind::parse(p.kind.as_deref().unwrap_or("fact"))
            .map_err(|e| McpError::invalid_params(e, None))?;
        let user_meta: Option<std::collections::HashMap<String, serde_json::Value>> =
            p.metadata.map(|m| m.into_iter().collect());

        let outcome = remember(
            &idx,
            &p.text,
            kind,
            p.source_id.as_deref(),
            p.doc_id.as_deref(),
            user_meta,
            p.force,
        )
        .map_err(user_or_internal)?;

        Ok(ok_json(serde_json::json!({
            "collection": collection,
            "doc_id": outcome.doc_id,
            "deduplicated": outcome.deduplicated,
            "existing_doc_id": outcome.existing_doc_id,
            "forced": outcome.forced,
            "low_signal_reason": outcome.low_signal_reason,
        })))
    }

    #[tool(description = "Delete one or more agent memory capsules. \
                       Two mutually exclusive forms: \
                       (1) {collection?, doc_id} — delete a single capsule by doc_id; \
                       (2) {collection?, source_id} — batch-delete all capsules whose source_id matches \
                       (e.g. all memories from a session). Returns {deleted: N}. \
                       Exactly one of doc_id or source_id is required; supplying both is an error.")]
    pub async fn msa_forget(
        &self,
        Parameters(p): Parameters<ForgetParams>,
    ) -> std::result::Result<CallToolResult, McpError> {
        let collection =
            resolve_collection(p.collection, &self.config.agent_memory.default_collection)?;
        msa_core::validate::validate_collection_name(&collection).map_err(bad_input)?;

        match (&p.doc_id, &p.source_id) {
            (Some(_), Some(_)) => {
                return Err(McpError::invalid_params(
                    "msa_forget: doc_id and source_id are mutually exclusive; supply exactly one",
                    None,
                ));
            }
            (None, None) => {
                return Err(McpError::invalid_params(
                    "msa_forget: exactly one of doc_id or source_id is required",
                    None,
                ));
            }
            _ => {}
        }

        let idx = self
            .registry
            .open_or_create_with_heap(
                &collection,
                &self.config.storage.storage_dir,
                &self.config.chunking,
                self.config.indexing.writer_heap_bytes,
            )
            .map_err(open_err)?;

        if let Some(doc_id) = &p.doc_id {
            // Single delete.
            msa_core::validate::validate_doc_id(doc_id).map_err(bad_input)?;
            idx.delete_document(doc_id).map_err(internal)?;
            Ok(ok_json(serde_json::json!({
                "collection": collection,
                "deleted": 1,
                "doc_id": doc_id,
            })))
        } else {
            // Batch delete by source_id.
            let source_id = p.source_id.as_deref().unwrap();
            msa_core::validate::validate_source_id(source_id).map_err(bad_input)?;
            let outcome = forget_by_source_id(&idx, source_id).map_err(user_or_internal)?;
            Ok(ok_json(serde_json::json!({
                "collection": collection,
                "deleted": outcome.deleted,
                "source_id": source_id,
            })))
        }
    }

    #[tool(
        description = "Mirror a filesystem directory into a collection in one call (turnkey sync; \
                       requires the server built with --features source-fs and [sync].enabled + \
                       allowed_roots configured). Walks `path`, matches include/exclude globs, reads \
                       UTF-8 files, and reconciles against the index: skips unchanged, indexes new/changed, \
                       and (with prune + source_id) deletes in-scope docs no longer on disk. prune never \
                       runs if any file errored. doc_id = doc_id_prefix + the file's relative path. \
                       Use dry_run to preview. Large dirs are committed in bounded batches (atomic=false)."
    )]
    pub async fn msa_sync_path(
        &self,
        Parameters(p): Parameters<SyncPathParams>,
    ) -> std::result::Result<CallToolResult, McpError> {
        #[cfg(feature = "source-fs")]
        {
            let idx = self
                .registry
                .open_or_create_with_heap(
                    &p.collection,
                    &self.config.storage.storage_dir,
                    &self.config.chunking,
                    self.config.indexing.writer_heap_bytes,
                )
                .map_err(open_err)?;
            let caps = msa_core::index::BatchCaps {
                max_docs: self.config.indexing.max_batch_docs,
                max_text_bytes: self.config.indexing.max_batch_text_bytes,
            };
            let embedder = self.scorer.as_deref().map(|s| s as _);
            let value =
                crate::sync_fs::sync_path(idx.as_ref(), &self.config.sync, caps, embedder, &p)?;
            Ok(ok_json(value))
        }
        #[cfg(not(feature = "source-fs"))]
        {
            let _ = &p;
            Err(McpError::invalid_params(
                "msa_sync_path requires the server built with --features source-fs".to_string(),
                None,
            ))
        }
    }
}

/// Resolve the collection: use the caller-supplied value when present, fall
/// back to the server default.  Returns `invalid_params` if both are absent.
fn resolve_collection(
    param: Option<String>,
    default: &str,
) -> std::result::Result<String, McpError> {
    match param {
        Some(c) if !c.is_empty() => Ok(c),
        _ => {
            if default.is_empty() {
                Err(McpError::invalid_params(
                    "collection is required: neither the parameter nor [agent_memory] \
                     default_collection in config is set",
                    None,
                ))
            } else {
                Ok(default.to_string())
            }
        }
    }
}

impl MsaServer {
    /// Internal: dispatch to the right MsaIndex method based on whether a
    /// dense rerank was requested *and* the server has a scorer configured.
    fn run_search(
        &self,
        idx: &msa_core::index::MsaIndex,
        query: &str,
        top_k: usize,
        filter: Option<&SearchFilter>,
        dense_alpha: Option<f32>,
        excludes: &std::collections::HashSet<String>,
    ) -> msa_core::error::Result<Vec<msa_core::schema::ChunkHit>> {
        match dense_alpha {
            Some(alpha) if alpha < 1.0 => {
                let scorer = self.scorer.as_ref().ok_or_else(|| {
                    msa_core::error::MsaError::Config(
                        "dense_alpha < 1.0 requires the server to be started with a dense scorer; \
                         build with --features embeddings and set [embeddings] in the config"
                            .into(),
                    )
                })?;
                let q_emb = scorer.encode_kind(EmbedInputKind::Query, query)?;
                idx.search_hybrid(query, top_k, filter, excludes, alpha, Some(&q_emb))
            }
            _ => idx.search_excluding(query, top_k, filter, excludes),
        }
    }

    /// Encode the round query into a dense vector, but only when a dense rerank
    /// is actually requested (`dense_alpha < 1.0`). Returns `None` for BM25-only
    /// rounds. A request for dense rerank without a configured scorer is a
    /// `Config` error (parity with `run_search`), surfaced to the client.
    fn query_embedding_for(
        &self,
        query: &str,
        dense_alpha: Option<f32>,
    ) -> msa_core::error::Result<Option<Vec<f32>>> {
        match dense_alpha {
            Some(alpha) if alpha < 1.0 => {
                let scorer = self.scorer.as_ref().ok_or_else(|| {
                    msa_core::error::MsaError::Config(
                        "dense_alpha < 1.0 requires the server to be started with a dense scorer; \
                         build with --features embeddings and set [embeddings] in the config"
                            .into(),
                    )
                })?;
                Ok(Some(scorer.encode_kind(EmbedInputKind::Query, query)?))
            }
            _ => Ok(None),
        }
    }

    /// Shared core for the A6 write tools (`msa_sync_delta` / `msa_index_batch`
    /// / `msa_delete_batch`): build docs/options/caps, run the atomic
    /// `apply_delta`, serialize the outcome. One commit, rollback on error.
    #[allow(clippy::too_many_arguments)]
    fn run_delta(
        &self,
        collection: &str,
        source_id: Option<String>,
        skip_unchanged: bool,
        if_manifest_digest: Option<String>,
        if_index_profile_hash: Option<String>,
        upserts: Vec<DocInput>,
        delete_doc_ids: Vec<String>,
    ) -> std::result::Result<CallToolResult, McpError> {
        let idx = self
            .registry
            .open_or_create_with_heap(
                collection,
                &self.config.storage.storage_dir,
                &self.config.chunking,
                self.config.indexing.writer_heap_bytes,
            )
            .map_err(open_err)?;

        let docs: Vec<Document> = upserts
            .into_iter()
            .map(|d| Document {
                id: d.doc_id,
                text: d.text,
                metadata: d
                    .metadata
                    .map(|m| m.into_iter().collect())
                    .unwrap_or_default(),
                created_at: chrono::Utc::now(),
            })
            .collect();
        let delete_refs: Vec<&str> = delete_doc_ids.iter().map(|s| s.as_str()).collect();

        let embedder = self.scorer.as_deref().map(|s| s as _);
        let opts = msa_core::index::DeltaOptions {
            skip_unchanged,
            source_id: source_id.clone(),
            if_manifest_digest,
            if_index_profile_hash,
        };
        let caps = msa_core::index::BatchCaps {
            max_docs: self.config.indexing.max_batch_docs,
            max_text_bytes: self.config.indexing.max_batch_text_bytes,
        };
        let outcome = idx
            .apply_delta(&docs, &delete_refs, embedder, &opts, caps)
            .map_err(user_or_internal)?;
        let index_profile_hash = idx.current_index_profile_hash(embedder);

        Ok(ok_json(serde_json::json!({
            "collection": collection,
            "source_id": source_id,
            "atomic": true,
            "committed": true,
            "content_hash_algo": msa_core::fingerprint::CONTENT_HASH_ALGO,
            "index_profile_hash": index_profile_hash,
            "indexed": outcome.indexed,
            "unchanged": outcome.unchanged,
            "deleted": outcome.deleted,
            "not_found": outcome.not_found,
            "skipped_scope_mismatch": outcome.skipped_scope_mismatch,
            "new_manifest_digest": outcome.new_manifest_digest,
            "results": outcome.results,
        })))
    }
}

#[tool_handler]
impl ServerHandler for MsaServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(
                "mcp-vl-msa-rs",
                env!("CARGO_PKG_VERSION"),
            ))
            .with_instructions(
                "Durable, searchable long-term memory for an AI agent, over MCP. \
                 Index documents, notes or past conversations into named collections \
                 (msa_index / msa_index_batch); retrieve the top-k relevant chunks for \
                 a query with msa_search, then call msa_fetch_doc to inject the full \
                 original text back to the model — the chunks locate, the document \
                 grounds. Use msa_remember / msa_forget to manage standalone agent \
                 memories (dedup + low-signal gate). For multi-hop questions, \
                 msa_interleave_round runs one bounded route->dedup->read step you can \
                 repeat. BM25 by default, no embeddings; pass dense_alpha on msa_search \
                 only when the server is built and configured for hybrid rerank. \
                 Collections persist on disk across sessions.",
            )
    }
}

fn ok_json(value: serde_json::Value) -> CallToolResult {
    CallToolResult::success(vec![Content::text(value.to_string())])
}

fn internal<E: std::fmt::Display>(e: E) -> McpError {
    McpError::internal_error(e.to_string(), None)
}

/// Map a validation/bad-request failure to `invalid_params` rather than
/// `internal_error`, so the client sees it as its own fault (bad collection
/// name, oversize input) instead of a server bug.
fn bad_input<E: std::fmt::Display>(e: E) -> McpError {
    McpError::invalid_params(e.to_string(), None)
}

/// Map a core error from a collection-opening path to the right MCP class:
/// a caller-caused bad input (invalid collection name) is `invalid_params`,
/// everything else (IO, tantivy, schema-version mismatch on disk) is a real
/// server-side `internal_error`.
fn open_err(e: msa_core::error::MsaError) -> McpError {
    match e {
        msa_core::error::MsaError::InvalidInput(_) => bad_input(e),
        other => internal(other),
    }
}

/// Map a core error from the interleave round to the right MCP class. A
/// caller-caused fault — bad input or an unparsable/invalid query (tantivy
/// syntax, dense_alpha out of range) — is `invalid_params`; everything else
/// (missing scorer config, IO, tantivy internals) is `internal_error`.
pub(crate) fn user_or_internal(e: msa_core::error::MsaError) -> McpError {
    use msa_core::error::MsaError::{Conflict, InvalidInput, InvalidQuery};
    match e {
        // Conflict = a stale optimistic-concurrency precondition (the client's
        // manifest/profile view is out of date); caller-caused, nothing written.
        InvalidInput(_) | InvalidQuery(_) | Conflict(_) => bad_input(e),
        other => internal(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::ErrorCode;

    fn server() -> MsaServer {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = MsaConfig {
            storage: msa_core::config::StorageConfig {
                storage_dir: dir.path().to_path_buf(),
            },
            ..Default::default()
        };
        // Leak the tempdir so the storage path stays valid for the test body.
        std::mem::forget(dir);
        MsaServer::new(config)
    }

    fn code(e: &McpError) -> ErrorCode {
        e.code
    }

    /// Parse the first text content of a successful tool result as JSON.
    fn result_json(res: &CallToolResult) -> serde_json::Value {
        let text = res
            .content
            .first()
            .and_then(|c| c.as_text())
            .map(|t| t.text.clone())
            .expect("tool result has text content");
        serde_json::from_str(&text).expect("tool result text is JSON")
    }

    #[tokio::test]
    async fn invalid_collection_maps_to_invalid_params() {
        let s = server();
        // search
        let err = s
            .msa_search(Parameters(SearchParams {
                collection: "../escape".into(),
                query: "x".into(),
                top_k: 5,
                filter: None,
                dense_alpha: None,
            }))
            .await
            .expect_err("invalid collection must error");
        assert_eq!(code(&err), ErrorCode::INVALID_PARAMS);

        // fetch
        let err = s
            .msa_fetch_doc(Parameters(FetchDocParams {
                collection: "a/b".into(),
                doc_id: "d1".into(),
            }))
            .await
            .expect_err("invalid collection must error");
        assert_eq!(code(&err), ErrorCode::INVALID_PARAMS);

        // delete
        let err = s
            .msa_delete(Parameters(DeleteParams {
                collection: "a/b".into(),
                doc_id: "d1".into(),
            }))
            .await
            .expect_err("invalid collection must error");
        assert_eq!(code(&err), ErrorCode::INVALID_PARAMS);

        // stats
        let err = s
            .msa_stats(Parameters(StatsParams {
                collection: "a/b".into(),
            }))
            .await
            .expect_err("invalid collection must error");
        assert_eq!(code(&err), ErrorCode::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn invalid_doc_id_maps_to_invalid_params() {
        let s = server();
        let err = s
            .msa_fetch_doc(Parameters(FetchDocParams {
                collection: "ok".into(),
                doc_id: "".into(),
            }))
            .await
            .expect_err("empty doc_id must error");
        assert_eq!(code(&err), ErrorCode::INVALID_PARAMS);

        let err = s
            .msa_delete(Parameters(DeleteParams {
                collection: "ok".into(),
                doc_id: "a\0b".into(),
            }))
            .await
            .expect_err("NUL doc_id must error");
        assert_eq!(code(&err), ErrorCode::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn index_reserved_msa_metadata_maps_to_invalid_params() {
        // A6: the reserved `_msa` metadata key is a caller fault. The core
        // rejects it as InvalidInput; the handler must surface invalid_params,
        // not internal_error.
        let s = server();
        let mut m = serde_json::Map::new();
        m.insert(
            "_msa".to_string(),
            serde_json::json!({"content_hash": "spoof"}),
        );
        let err = s
            .msa_index(Parameters(IndexParams {
                collection: "c".into(),
                doc_id: "d1".into(),
                text: "hello".into(),
                metadata: Some(m),
            }))
            .await
            .expect_err("reserved _msa metadata must error");
        assert_eq!(code(&err), ErrorCode::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn drop_collection_invalid_name_maps_to_invalid_params() {
        let s = server();
        let err = s
            .msa_drop_collection(Parameters(DropCollectionParams {
                collection: "../escape".into(),
            }))
            .await
            .expect_err("invalid collection must error");
        assert_eq!(code(&err), ErrorCode::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn drop_collection_reports_dropped_flag() {
        let s = server();
        // Index into a fresh collection, then drop it: dropped=true.
        s.msa_index(Parameters(IndexParams {
            collection: "tmp".into(),
            doc_id: "d1".into(),
            text: "contenuto".into(),
            metadata: None,
        }))
        .await
        .expect("index ok");
        let res = s
            .msa_drop_collection(Parameters(DropCollectionParams {
                collection: "tmp".into(),
            }))
            .await
            .expect("drop ok");
        let payload = result_json(&res);
        assert_eq!(payload["dropped"], serde_json::json!(true));

        // Dropping a never-created collection: dropped=false (not an error).
        let res = s
            .msa_drop_collection(Parameters(DropCollectionParams {
                collection: "ghost".into(),
            }))
            .await
            .expect("drop ghost ok");
        assert_eq!(result_json(&res)["dropped"], serde_json::json!(false));
    }

    #[tokio::test]
    async fn empty_query_maps_to_invalid_params() {
        let s = server();
        let err = s
            .msa_search(Parameters(SearchParams {
                collection: "ok".into(),
                query: "   ".into(),
                top_k: 5,
                filter: None,
                dense_alpha: None,
            }))
            .await
            .expect_err("empty query must error");
        assert_eq!(code(&err), ErrorCode::INVALID_PARAMS);
    }

    // ---- A5.3 msa_interleave_round ----------------------------------------

    fn il_params(
        collection: &str,
        query: &str,
        state: Option<InterleaveStateV1>,
    ) -> InterleaveRoundParams {
        InterleaveRoundParams {
            collection: collection.into(),
            query: query.into(),
            state,
            top_k: None,
            fetch_top_n: None,
            max_chars_per_doc: None,
            max_total_chars: None,
            max_rounds: None,
            low_gain_threshold: None,
            dense_alpha: None,
            dedup_mode: "id".into(),
            filter: None,
        }
    }

    async fn index_doc(s: &MsaServer, collection: &str, doc_id: &str, text: &str) {
        s.msa_index(Parameters(IndexParams {
            collection: collection.into(),
            doc_id: doc_id.into(),
            text: text.into(),
            metadata: None,
        }))
        .await
        .expect("index ok");
    }

    #[tokio::test]
    async fn manifest_lists_fingerprints_and_hash_state() {
        let s = server();
        for (id, txt) in [
            ("p/a", "alpha text"),
            ("p/b", "bravo text"),
            ("q/c", "charlie text"),
        ] {
            s.msa_index(Parameters(IndexParams {
                collection: "c".into(),
                doc_id: id.into(),
                text: txt.into(),
                metadata: None,
            }))
            .await
            .expect("index ok");
        }
        let v = result_json(
            &s.msa_manifest(Parameters(ManifestParams {
                collection: "c".into(),
                source_id: None,
                doc_id_prefix: None,
                limit: None,
                cursor: None,
            }))
            .await
            .expect("manifest ok"),
        );
        assert_eq!(v["total_count"], serde_json::json!(3));
        assert_eq!(v["content_hash_algo"], serde_json::json!("blake3-doc-v1"));
        assert!(v["manifest_digest"].as_str().unwrap().len() == 64);
        let entries = v["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 3);
        // Freshly indexed under the current profile -> all "ok" with a hash.
        for e in entries {
            assert_eq!(e["hash_state"], serde_json::json!("ok"));
            assert!(e["content_hash"].as_str().is_some());
        }
        // doc_id_prefix filter.
        let pfx = result_json(
            &s.msa_manifest(Parameters(ManifestParams {
                collection: "c".into(),
                source_id: None,
                doc_id_prefix: Some("p/".into()),
                limit: None,
                cursor: None,
            }))
            .await
            .expect("manifest prefix ok"),
        );
        assert_eq!(pfx["total_count"], serde_json::json!(2));
    }

    #[tokio::test]
    async fn manifest_invalid_collection_maps_to_invalid_params() {
        let s = server();
        let err = s
            .msa_manifest(Parameters(ManifestParams {
                collection: "../escape".into(),
                source_id: None,
                doc_id_prefix: None,
                limit: None,
                cursor: None,
            }))
            .await
            .expect_err("invalid collection must error");
        assert_eq!(code(&err), ErrorCode::INVALID_PARAMS);
    }

    // ---- A6.4 write tools --------------------------------------------------

    fn di(id: &str, text: &str) -> DocInput {
        DocInput {
            doc_id: id.into(),
            text: text.into(),
            metadata: None,
        }
    }

    #[tokio::test]
    async fn sync_delta_upsert_delete_skip_and_digest() {
        let s = server();
        let v = result_json(
            &s.msa_sync_delta(Parameters(SyncDeltaParams {
                collection: "c".into(),
                source_id: Some("docs".into()),
                skip_unchanged: true,
                if_manifest_digest: None,
                if_index_profile_hash: None,
                upsert_docs: vec![di("a", "alpha"), di("b", "bravo")],
                delete_doc_ids: vec![],
            }))
            .await
            .expect("sync ok"),
        );
        assert_eq!(v["indexed"], serde_json::json!(2));
        assert_eq!(v["committed"], serde_json::json!(true));
        let digest = v["new_manifest_digest"].as_str().unwrap().to_string();

        // Re-upsert a (same content) + delete b, with the correct precondition.
        let v2 = result_json(
            &s.msa_sync_delta(Parameters(SyncDeltaParams {
                collection: "c".into(),
                source_id: Some("docs".into()),
                skip_unchanged: true,
                if_manifest_digest: Some(digest),
                if_index_profile_hash: None,
                upsert_docs: vec![di("a", "alpha")],
                delete_doc_ids: vec!["b".into()],
            }))
            .await
            .expect("sync2 ok"),
        );
        assert_eq!(v2["unchanged"], serde_json::json!(1));
        assert_eq!(v2["deleted"], serde_json::json!(1));
    }

    #[tokio::test]
    async fn sync_delta_stale_digest_maps_to_invalid_params() {
        let s = server();
        s.msa_index_batch(Parameters(IndexBatchParams {
            collection: "c".into(),
            source_id: None,
            skip_unchanged: true,
            docs: vec![di("a", "x")],
        }))
        .await
        .expect("batch ok");
        let err = s
            .msa_sync_delta(Parameters(SyncDeltaParams {
                collection: "c".into(),
                source_id: None,
                skip_unchanged: true,
                if_manifest_digest: Some("staledigest".into()),
                if_index_profile_hash: None,
                upsert_docs: vec![di("b", "y")],
                delete_doc_ids: vec![],
            }))
            .await
            .expect_err("stale precondition must error");
        assert_eq!(code(&err), ErrorCode::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn delete_batch_counts_deleted_and_not_found() {
        let s = server();
        s.msa_index_batch(Parameters(IndexBatchParams {
            collection: "c".into(),
            source_id: Some("A".into()),
            skip_unchanged: true,
            docs: vec![di("a", "x")],
        }))
        .await
        .expect("ok");
        let v = result_json(
            &s.msa_delete_batch(Parameters(DeleteBatchParams {
                collection: "c".into(),
                source_id: Some("A".into()),
                doc_ids: vec!["a".into(), "ghost".into()],
            }))
            .await
            .expect("del ok"),
        );
        assert_eq!(v["deleted"], serde_json::json!(1));
        assert_eq!(v["not_found"], serde_json::json!(1));
    }

    #[tokio::test]
    async fn index_batch_reserved_msa_maps_to_invalid_params() {
        let s = server();
        let mut m = serde_json::Map::new();
        m.insert("_msa".to_string(), serde_json::json!({"x": 1}));
        let err = s
            .msa_index_batch(Parameters(IndexBatchParams {
                collection: "c".into(),
                source_id: None,
                skip_unchanged: true,
                docs: vec![DocInput {
                    doc_id: "a".into(),
                    text: "x".into(),
                    metadata: Some(m),
                }],
            }))
            .await
            .expect_err("reserved _msa must error");
        assert_eq!(code(&err), ErrorCode::INVALID_PARAMS);
    }

    // ---- deny_unknown_fields regression (M3 backlog) ---------------------
    //
    // The tool input structs derive `Deserialize` without `deny_unknown_fields`,
    // so a typo in a field name silently produced an empty result (e.g. `docs`
    // -> nothing, `doc_id` -> nothing). These tests exercise the wire path:
    // we build the same `serde_json::Value` the MCP framework would pass to
    // `Parameters<P>::from_context_part`, run it through `serde_json::from_value`
    // the same way the framework does, and assert that
    //   1) a typo on the top-level payload is rejected with the bad field name
    //      in the error message, and
    //   2) a correct payload still deserializes and the tool returns success.
    #[tokio::test]
    async fn index_batch_rejects_unknown_field_doc_id_at_top_level() {
        // Caller typo: `doc_id` (singular) instead of `docs` (plural). The
        // previous behavior was to silently deserialize `docs = []` and
        // return indexed=0. With `deny_unknown_fields` we want a deserialization
        // error that names the offending field.
        let bad = serde_json::json!({
            "collection": "c",
            "skip_unchanged": true,
            "doc_id": "d1",           // <-- the typo
        });
        let err = serde_json::from_value::<IndexBatchParams>(bad)
            .expect_err("typo `doc_id` at top level must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("doc_id"),
            "error message must name the offending field, got: {msg}",
        );
    }

    #[tokio::test]
    async fn index_batch_rejects_unknown_field_documents_inside_docs() {
        // Caller typo: a single DocInput has `documents` instead of `text`.
        // The previous behavior was to deserialize `text = ""` and either
        // skip the doc or fail downstream with a confusing error. With
        // `deny_unknown_fields` we want a deserialization error that names
        // the offending field.
        // Include the required `text` field too: with `deny_unknown_fields`
        // applied to DocInput, an extra `documents` key is reported as the
        // *only* problem (not the missing required field, which is satisfied).
        let bad = serde_json::json!({
            "collection": "c",
            "skip_unchanged": true,
            "docs": [
                { "doc_id": "d1", "text": "alpha", "documents": "alpha" }
            ],
        });
        let err = serde_json::from_value::<IndexBatchParams>(bad)
            .expect_err("typo `documents` inside docs[] must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("documents"),
            "error message must name the offending field, got: {msg}",
        );
    }

    #[tokio::test]
    async fn index_batch_correct_payload_still_passes() {
        // The happy path: a well-formed payload deserializes, the tool runs,
        // and the tool reports indexed > 0. This guards against a too-aggressive
        // `deny_unknown_fields` that would reject legitimate fields.
        let s = server();
        let res = s
            .msa_index_batch(Parameters(IndexBatchParams {
                collection: "c".into(),
                source_id: Some("smoke".into()),
                skip_unchanged: true,
                docs: vec![DocInput {
                    doc_id: "d1".into(),
                    text: "alpha".into(),
                    metadata: None,
                }],
            }))
            .await
            .expect("well-formed payload must succeed");
        let v = result_json(&res);
        assert_eq!(v["indexed"], serde_json::json!(1));
        assert_eq!(v["committed"], serde_json::json!(true));
    }

    #[tokio::test]
    async fn interleave_round_one_injects_and_returns_state() {
        let s = server();
        index_doc(&s, "c", "a", "interleave memory alpha").await;
        index_doc(&s, "c", "b", "interleave memory bravo").await;

        let res = s
            .msa_interleave_round(Parameters(il_params("c", "interleave", None)))
            .await
            .expect("round ok");
        let v = result_json(&res);
        assert_eq!(v["round"], serde_json::json!(1));
        assert!(
            !v["injected_docs"].as_array().unwrap().is_empty(),
            "round 1 must inject fresh docs"
        );
        assert_eq!(v["state"]["schema_version"], serde_json::json!(1));
        assert!(!v["state"]["seen_doc_ids"].as_array().unwrap().is_empty());
        // Injected docs carry the bounded original text.
        assert!(v["injected_docs"][0]["text"]
            .as_str()
            .unwrap()
            .contains("interleave"));
    }

    #[tokio::test]
    async fn interleave_round_two_dedups_via_returned_state() {
        let s = server();
        index_doc(&s, "c", "a", "interleave memory alpha").await;
        index_doc(&s, "c", "b", "interleave memory bravo").await;
        index_doc(&s, "c", "d", "interleave memory delta").await;

        let r1 = result_json(
            &s.msa_interleave_round(Parameters(il_params("c", "interleave", None)))
                .await
                .expect("round1 ok"),
        );
        // Feed the returned state back — exactly what a client does.
        let state: InterleaveStateV1 =
            serde_json::from_value(r1["state"].clone()).expect("state round-trips");
        let seen_round1 = state.seen_doc_ids.clone();

        let r2 = result_json(
            &s.msa_interleave_round(Parameters(il_params("c", "interleave", Some(state))))
                .await
                .expect("round2 ok"),
        );
        assert_eq!(r2["round"], serde_json::json!(2));
        assert!(
            r2["stats"]["dedup_skipped_count"].as_u64().unwrap() >= 1,
            "round-1 docs must dedup out in round 2"
        );
        for inj in r2["injected_docs"].as_array().unwrap() {
            let id = inj["doc_id"].as_str().unwrap();
            assert!(
                !seen_round1.contains(&id.to_string()),
                "must not re-inject {id}"
            );
        }
    }

    #[tokio::test]
    async fn interleave_unsupported_dedup_mode_maps_to_invalid_params() {
        let s = server();
        index_doc(&s, "c", "a", "interleave memory alpha").await;
        let mut p = il_params("c", "interleave", None);
        p.dedup_mode = "semantic".into();
        let err = s
            .msa_interleave_round(Parameters(p))
            .await
            .expect_err("unsupported dedup_mode must error");
        assert_eq!(code(&err), ErrorCode::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn interleave_invalid_collection_maps_to_invalid_params() {
        let s = server();
        let err = s
            .msa_interleave_round(Parameters(il_params("../escape", "interleave", None)))
            .await
            .expect_err("invalid collection must error");
        assert_eq!(code(&err), ErrorCode::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn interleave_empty_query_maps_to_invalid_params() {
        let s = server();
        let err = s
            .msa_interleave_round(Parameters(il_params("c", "   ", None)))
            .await
            .expect_err("empty query must error");
        assert_eq!(code(&err), ErrorCode::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn interleave_dense_without_scorer_errors() {
        // Parity with msa_search: dense_alpha < 1.0 with no configured scorer
        // is an error, not a silent BM25 fallback.
        let s = server();
        index_doc(&s, "c", "a", "interleave memory alpha").await;
        let mut p = il_params("c", "interleave", None);
        p.dense_alpha = Some(0.5);
        let err = s
            .msa_interleave_round(Parameters(p))
            .await
            .expect_err("dense without scorer must error");
        assert_eq!(code(&err), ErrorCode::INTERNAL_ERROR);
    }

    // ---- agent-memory: msa_remember / msa_forget --------------------------

    fn remember_params(
        text: &str,
        kind: Option<&str>,
        source_id: Option<&str>,
        force: bool,
    ) -> RememberParams {
        RememberParams {
            collection: Some("test-memories".into()),
            text: text.into(),
            kind: kind.map(String::from),
            source_id: source_id.map(String::from),
            doc_id: None,
            metadata: None,
            force,
        }
    }

    const LONG_TEXT: &str = "The reconnect handler in retry_backoff.rs now respects the \
        max_attempts limit and logs a warning via tracing when the backoff ceiling is \
        hit; this prevents infinite retry loops under transient network partitions.";

    #[tokio::test]
    async fn remember_low_signal_gate_rejects_short_text() {
        let s = server();
        let res = s
            .msa_remember(Parameters(remember_params("short", None, None, false)))
            .await
            .expect("call ok (gate result, not error)");
        let v = result_json(&res);
        assert!(
            v["low_signal_reason"].as_str().is_some(),
            "short text must be rejected by gate: {v}"
        );
        assert_eq!(v["doc_id"], serde_json::json!(""));
    }

    #[tokio::test]
    async fn remember_force_bypasses_gate() {
        let s = server();
        let res = s
            .msa_remember(Parameters(remember_params("ok", None, None, true)))
            .await
            .expect("force must succeed");
        let v = result_json(&res);
        assert!(
            v["low_signal_reason"].is_null(),
            "force must suppress gate: {v}"
        );
        assert!(!v["doc_id"].as_str().unwrap_or("").is_empty());
    }

    #[tokio::test]
    async fn remember_dedup_returns_existing_doc_id() {
        let s = server();
        let v1 = result_json(
            &s.msa_remember(Parameters(remember_params(LONG_TEXT, None, None, false)))
                .await
                .expect("first remember ok"),
        );
        assert!(!v1["deduplicated"].as_bool().unwrap_or(true));
        let id1 = v1["doc_id"].as_str().unwrap().to_string();

        let v2 = result_json(
            &s.msa_remember(Parameters(remember_params(LONG_TEXT, None, None, false)))
                .await
                .expect("second remember ok"),
        );
        assert!(
            v2["deduplicated"].as_bool().unwrap_or(false),
            "same text must dedup: {v2}"
        );
        assert_eq!(v2["existing_doc_id"].as_str(), Some(id1.as_str()));
    }

    #[tokio::test]
    async fn remember_stores_kind_metadata_and_searchable() {
        let s = server();
        let v = result_json(
            &s.msa_remember(Parameters(RememberParams {
                collection: Some("tm".into()),
                text: LONG_TEXT.into(),
                kind: Some("episode".into()),
                source_id: Some("sess-1".into()),
                doc_id: None,
                metadata: None,
                force: false,
            }))
            .await
            .expect("remember ok"),
        );
        let doc_id = v["doc_id"].as_str().unwrap().to_string();
        assert!(!doc_id.is_empty());

        // Search with kind filter.
        let search_v = result_json(
            &s.msa_search(Parameters(SearchParams {
                collection: "tm".into(),
                query: "reconnect handler".into(),
                top_k: 5,
                filter: Some(msa_core::schema::SearchFilter {
                    where_eq: {
                        let mut m = std::collections::HashMap::new();
                        m.insert("kind".to_string(), serde_json::json!("episode"));
                        m
                    },
                    ..Default::default()
                }),
                dense_alpha: None,
            }))
            .await
            .expect("search ok"),
        );
        let hits = search_v["hits"].as_array().unwrap();
        assert!(
            !hits.is_empty(),
            "episode must be retrievable by kind filter"
        );
        assert_eq!(hits[0]["metadata"]["kind"], serde_json::json!("episode"));
        assert_eq!(hits[0]["metadata"]["source"], serde_json::json!("agent"));
    }

    #[tokio::test]
    async fn forget_single_by_doc_id() {
        let s = server();
        let v = result_json(
            &s.msa_remember(Parameters(remember_params(LONG_TEXT, None, None, false)))
                .await
                .expect("remember ok"),
        );
        let doc_id = v["doc_id"].as_str().unwrap().to_string();

        let fv = result_json(
            &s.msa_forget(Parameters(ForgetParams {
                collection: Some("test-memories".into()),
                doc_id: Some(doc_id.clone()),
                source_id: None,
            }))
            .await
            .expect("forget ok"),
        );
        assert_eq!(fv["deleted"], serde_json::json!(1));
        assert_eq!(fv["doc_id"].as_str(), Some(doc_id.as_str()));
    }

    #[tokio::test]
    async fn forget_batch_by_source_id() {
        let s = server();
        // Index 3 capsules under source_id=sess-A.
        for i in 0..3usize {
            let text = format!(
                "{LONG_TEXT} iteration {i} adds unique words to pass dedup check uniquely",
            );
            s.msa_remember(Parameters(RememberParams {
                collection: Some("test-memories".into()),
                text,
                kind: None,
                source_id: Some("sess-A".into()),
                doc_id: None,
                metadata: None,
                force: false,
            }))
            .await
            .expect("remember ok");
        }
        let fv = result_json(
            &s.msa_forget(Parameters(ForgetParams {
                collection: Some("test-memories".into()),
                doc_id: None,
                source_id: Some("sess-A".into()),
            }))
            .await
            .expect("batch forget ok"),
        );
        assert_eq!(fv["deleted"], serde_json::json!(3));
        assert_eq!(fv["source_id"].as_str(), Some("sess-A"));
    }

    #[tokio::test]
    async fn forget_both_doc_id_and_source_id_is_error() {
        let s = server();
        let err = s
            .msa_forget(Parameters(ForgetParams {
                collection: Some("c".into()),
                doc_id: Some("d1".into()),
                source_id: Some("sess".into()),
            }))
            .await
            .expect_err("both must error");
        assert_eq!(code(&err), ErrorCode::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn forget_neither_doc_id_nor_source_id_is_error() {
        let s = server();
        let err = s
            .msa_forget(Parameters(ForgetParams {
                collection: Some("c".into()),
                doc_id: None,
                source_id: None,
            }))
            .await
            .expect_err("neither must error");
        assert_eq!(code(&err), ErrorCode::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn remember_unknown_field_rejected() {
        let bad = serde_json::json!({
            "collection": "c",
            "text": "hello",
            "unknown_key": true,
        });
        let err = serde_json::from_value::<RememberParams>(bad)
            .expect_err("unknown field must be rejected");
        assert!(err.to_string().contains("unknown_key"), "{err}");
    }

    #[tokio::test]
    async fn forget_unknown_field_rejected() {
        let bad = serde_json::json!({
            "collection": "c",
            "doc_id": "d1",
            "typo_field": "x",
        });
        let err = serde_json::from_value::<ForgetParams>(bad)
            .expect_err("unknown field must be rejected");
        assert!(err.to_string().contains("typo_field"), "{err}");
    }

    #[tokio::test]
    async fn remember_default_collection_used_when_omitted() {
        // The default server config uses "agent-memory" as the default collection.
        let s = server();
        let res = s
            .msa_remember(Parameters(RememberParams {
                collection: None, // omitted → uses config default
                text: LONG_TEXT.into(),
                kind: None,
                source_id: None,
                doc_id: None,
                metadata: None,
                force: false,
            }))
            .await
            .expect("default collection remember ok");
        let v = result_json(&res);
        assert_eq!(v["collection"].as_str(), Some("agent-memory"));
        assert!(!v["doc_id"].as_str().unwrap_or("").is_empty());
    }

    // ---- A6.5 msa_sync_path (feature source-fs) ---------------------------

    #[cfg(feature = "source-fs")]
    mod sync_fs_tests {
        use super::*;

        fn write_file(dir: &std::path::Path, rel: &str, content: &str) {
            let p = dir.join(rel);
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(p, content).unwrap();
        }

        fn server_with_sync(content_root: &std::path::Path) -> MsaServer {
            let dir = tempfile::tempdir().expect("tempdir");
            let mut config = MsaConfig {
                storage: msa_core::config::StorageConfig {
                    storage_dir: dir.path().to_path_buf(),
                },
                ..Default::default()
            };
            config.sync.enabled = true;
            config.sync.allowed_roots = vec![content_root.to_path_buf()];
            std::mem::forget(dir);
            MsaServer::new(config)
        }

        fn spp(
            collection: &str,
            path: &std::path::Path,
            source_id: Option<&str>,
            prune: bool,
            dry_run: bool,
        ) -> SyncPathParams {
            SyncPathParams {
                collection: collection.into(),
                path: path.to_string_lossy().into_owned(),
                source_id: source_id.map(String::from),
                globs: vec!["**/*.md".into()],
                exclude_globs: vec![],
                doc_id_prefix: None,
                prune,
                follow_symlinks: None,
                dry_run,
            }
        }

        #[tokio::test]
        async fn sync_path_indexes_skips_and_prunes() {
            let content = tempfile::tempdir().unwrap();
            write_file(content.path(), "a.md", "alpha");
            write_file(content.path(), "sub/b.md", "bravo");
            write_file(content.path(), "ignore.log", "nope"); // not a *.md
            let s = server_with_sync(content.path());

            // Initial sync: 2 markdown files indexed, one atomic batch.
            let v = result_json(
                &s.msa_sync_path(Parameters(spp(
                    "c",
                    content.path(),
                    Some("docs"),
                    true,
                    false,
                )))
                .await
                .expect("sync ok"),
            );
            assert_eq!(v["candidate_files"], serde_json::json!(2));
            assert_eq!(v["indexed"], serde_json::json!(2));
            assert_eq!(v["atomic"], serde_json::json!(true));

            // Re-sync unchanged content -> all skipped.
            let v2 = result_json(
                &s.msa_sync_path(Parameters(spp(
                    "c",
                    content.path(),
                    Some("docs"),
                    true,
                    false,
                )))
                .await
                .expect("resync ok"),
            );
            assert_eq!(v2["unchanged"], serde_json::json!(2));
            assert_eq!(v2["indexed"], serde_json::json!(0));

            // Remove a file, sync with prune -> the missing in-scope doc is deleted.
            std::fs::remove_file(content.path().join("a.md")).unwrap();
            let v3 = result_json(
                &s.msa_sync_path(Parameters(spp(
                    "c",
                    content.path(),
                    Some("docs"),
                    true,
                    false,
                )))
                .await
                .expect("prune sync ok"),
            );
            assert_eq!(v3["deleted"], serde_json::json!(1));
            assert_eq!(v3["prune_executed"], serde_json::json!(true));
        }

        #[tokio::test]
        async fn sync_path_dry_run_writes_nothing() {
            let content = tempfile::tempdir().unwrap();
            write_file(content.path(), "a.md", "alpha");
            let s = server_with_sync(content.path());
            let v = result_json(
                &s.msa_sync_path(Parameters(spp(
                    "c",
                    content.path(),
                    Some("docs"),
                    false,
                    true,
                )))
                .await
                .expect("dry run ok"),
            );
            assert_eq!(v["dry_run"], serde_json::json!(true));
            assert_eq!(v["would_index"], serde_json::json!(1));
            // Nothing was written: a real sync now still reports it as new.
            let real = result_json(
                &s.msa_sync_path(Parameters(spp(
                    "c",
                    content.path(),
                    Some("docs"),
                    false,
                    false,
                )))
                .await
                .expect("real sync ok"),
            );
            assert_eq!(real["indexed"], serde_json::json!(1));
        }

        #[tokio::test]
        async fn sync_path_rejects_outside_allowed_roots() {
            let content = tempfile::tempdir().unwrap();
            let other = tempfile::tempdir().unwrap();
            let s = server_with_sync(content.path());
            let err = s
                .msa_sync_path(Parameters(spp("c", other.path(), None, false, false)))
                .await
                .expect_err("path outside allowed_roots must error");
            assert_eq!(code(&err), ErrorCode::INVALID_PARAMS);
        }

        #[tokio::test]
        async fn sync_path_prune_requires_source_id() {
            let content = tempfile::tempdir().unwrap();
            let s = server_with_sync(content.path());
            let err = s
                .msa_sync_path(Parameters(spp("c", content.path(), None, true, false)))
                .await
                .expect_err("prune without source_id must error");
            assert_eq!(code(&err), ErrorCode::INVALID_PARAMS);
        }

        #[tokio::test]
        async fn sync_path_disabled_errors() {
            // The default server (sync disabled) refuses sync_path.
            let s = server();
            let content = tempfile::tempdir().unwrap();
            let err = s
                .msa_sync_path(Parameters(spp("c", content.path(), None, false, false)))
                .await
                .expect_err("disabled sync must error");
            assert_eq!(code(&err), ErrorCode::INVALID_PARAMS);
        }

        #[cfg(unix)]
        #[tokio::test]
        async fn sync_path_symlink_escape_is_skipped() {
            use std::os::unix::fs::symlink;
            let content = tempfile::tempdir().unwrap();
            let outside = tempfile::tempdir().unwrap();
            write_file(content.path(), "inside.md", "internal content");
            write_file(outside.path(), "secret.md", "external secret content");
            // A symlink inside the sync root pointing OUTSIDE it.
            symlink(
                outside.path().join("secret.md"),
                content.path().join("link.md"),
            )
            .unwrap();

            // Server with follow_symlinks ENABLED in config (the dangerous case).
            let dir = tempfile::tempdir().unwrap();
            let mut config = MsaConfig {
                storage: msa_core::config::StorageConfig {
                    storage_dir: dir.path().to_path_buf(),
                },
                ..Default::default()
            };
            config.sync.enabled = true;
            config.sync.allowed_roots = vec![content.path().to_path_buf()];
            config.sync.follow_symlinks = true;
            std::mem::forget(dir);
            let s = MsaServer::new(config);

            let v = result_json(
                &s.msa_sync_path(Parameters(spp(
                    "c",
                    content.path(),
                    Some("docs"),
                    false,
                    false,
                )))
                .await
                .expect("sync ok"),
            );
            // Only inside.md is indexed; the escaping symlink is skipped + recorded.
            assert_eq!(
                v["indexed"],
                serde_json::json!(1),
                "escaping symlink must not be indexed"
            );
            assert!(
                !v["errors"].as_array().unwrap().is_empty(),
                "the escape must be recorded as an error"
            );
        }

        /// After a sync_path the hits must carry fs metadata (`source`, `ext`,
        /// `dir`, `created_at`) and those keys must be filterable via where_eq.
        #[tokio::test]
        async fn sync_path_hits_include_fs_metadata() {
            let content = tempfile::tempdir().unwrap();
            write_file(
                content.path(),
                "docs/guide.md",
                "filesystem metadata test content",
            );
            let s = server_with_sync(content.path());

            // Index the file.
            result_json(
                &s.msa_sync_path(Parameters(spp(
                    "c",
                    content.path(),
                    Some("fs"),
                    false,
                    false,
                )))
                .await
                .expect("sync ok"),
            );

            // Search and verify metadata fields appear in hits.
            let v = result_json(
                &s.msa_search(Parameters(SearchParams {
                    collection: "c".into(),
                    query: "filesystem metadata".into(),
                    top_k: 5,
                    filter: None,
                    dense_alpha: None,
                }))
                .await
                .expect("search ok"),
            );
            let hits = v["hits"].as_array().expect("hits must be an array");
            assert!(!hits.is_empty(), "must have at least one hit");
            let meta = &hits[0]["metadata"];
            assert_eq!(
                meta["source"],
                serde_json::json!("fs"),
                "source must be 'fs'"
            );
            assert_eq!(meta["ext"], serde_json::json!("md"), "ext must be 'md'");
            assert_eq!(meta["dir"], serde_json::json!("docs"), "dir must be 'docs'");
            // created_at must be a non-empty ISO-8601 UTC string.
            let ca = meta["created_at"]
                .as_str()
                .expect("created_at must be a string");
            assert!(
                ca.contains('T') && ca.ends_with('Z'),
                "created_at must be UTC RFC3339: {ca}"
            );
        }

        /// `created_after` filter must exclude documents whose mtime is before
        /// the threshold, and include those whose mtime is on or after it.
        ///
        /// Because `filetime::set_file_mtime` is not in scope here we use a
        /// future timestamp as the threshold: the file's actual mtime (now) is
        /// before a threshold set to the year 2100, so the file is excluded.
        /// Conversely, a threshold set before epoch excludes nothing.
        #[tokio::test]
        async fn sync_path_created_after_filter_excludes_old_files() {
            let content = tempfile::tempdir().unwrap();
            write_file(content.path(), "note.md", "filter by date content here");
            let s = server_with_sync(content.path());

            // Sync: file mtime is approximately now.
            result_json(
                &s.msa_sync_path(Parameters(spp(
                    "c",
                    content.path(),
                    Some("fs"),
                    false,
                    false,
                )))
                .await
                .expect("sync ok"),
            );

            // Filter: created_after = year 2100 → no file can pass (mtime is now).
            let far_future: chrono::DateTime<chrono::Utc> =
                "2100-01-01T00:00:00Z".parse().expect("parse 2100");
            let v_future = result_json(
                &s.msa_search(Parameters(SearchParams {
                    collection: "c".into(),
                    query: "filter by date".into(),
                    top_k: 5,
                    filter: Some(msa_core::schema::SearchFilter {
                        created_after: Some(far_future),
                        ..Default::default()
                    }),
                    dense_alpha: None,
                }))
                .await
                .expect("search ok"),
            );
            let hits_future = v_future["hits"].as_array().expect("hits array");
            assert!(
                hits_future.is_empty(),
                "created_after=2100 must exclude all current files; got {} hits",
                hits_future.len()
            );

            // Filter: created_after = epoch → all files pass.
            let epoch: chrono::DateTime<chrono::Utc> =
                "1970-01-01T00:00:00Z".parse().expect("parse epoch");
            let v_epoch = result_json(
                &s.msa_search(Parameters(SearchParams {
                    collection: "c".into(),
                    query: "filter by date".into(),
                    top_k: 5,
                    filter: Some(msa_core::schema::SearchFilter {
                        created_after: Some(epoch),
                        ..Default::default()
                    }),
                    dense_alpha: None,
                }))
                .await
                .expect("search ok"),
            );
            let hits_epoch = v_epoch["hits"].as_array().expect("hits array");
            assert!(
                !hits_epoch.is_empty(),
                "created_after=epoch must include all current files"
            );
        }
    }
}
