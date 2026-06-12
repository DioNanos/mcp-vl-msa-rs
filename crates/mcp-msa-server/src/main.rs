use anyhow::Result;
use mcp_msa_server::MsaServer;
use msa_core::embeddings::SharedEncoder;
use msa_core::MsaConfig;
use rmcp::ServiceExt;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "mcp_msa_server=info,msa_core=info".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let args: Vec<String> = std::env::args().collect();
    if matches!(args.get(1).map(String::as_str), Some("--help" | "-h")) {
        eprintln!("mcp-vl-msa-rs — MSA-flavor retrieval/memory MCP server (stdio)");
        eprintln!();
        eprintln!("Usage:");
        eprintln!("  mcp-vl-msa-rs                Start MCP server on stdio");
        eprintln!("  mcp-vl-msa-rs --version      Print version");
        eprintln!();
        eprintln!("Environment:");
        eprintln!("  MCP_VL_MSA_CONFIG    Optional TOML config path");
        eprintln!("  MCP_DEVICE           Device identity (logged at startup)");
        eprintln!();
        eprintln!("Default storage: ~/.local/state/mcp-vl-msa-rs/<collection>/");
        return Ok(());
    }
    if matches!(args.get(1).map(String::as_str), Some("--version" | "-V")) {
        println!("mcp-vl-msa-rs {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    let config = MsaConfig::load()?;
    std::fs::create_dir_all(&config.storage.storage_dir)?;
    let scorer = build_encoder(&config)?;
    tracing::info!(
        device = std::env::var("MCP_DEVICE").as_deref().unwrap_or("unknown"),
        storage = %config.storage.storage_dir.display(),
        dense = scorer.is_some(),
        "starting mcp-vl-msa-rs (stdio)"
    );

    let server = MsaServer::with_scorer(config, scorer);
    let service = server
        .serve(rmcp::transport::stdio())
        .await
        .map_err(|e| anyhow::anyhow!("stdio service failed: {e}"))?;
    service.waiting().await?;
    Ok(())
}

/// Build the dense encoder selected by the `[embeddings]` config section.
/// The production default is `candle-modernbert` (in-process, no daemon).
/// `ollama` is kept as a deprecated transitional backend.
#[cfg(any(feature = "embeddings-candle", feature = "embeddings-ollama"))]
fn build_encoder(config: &MsaConfig) -> Result<Option<SharedEncoder>> {
    use std::sync::Arc;

    let Some(emb) = &config.embeddings else {
        return Ok(None);
    };

    match emb.backend.as_str() {
        #[cfg(feature = "embeddings-candle")]
        "candle-modernbert" => {
            use msa_core::embeddings::CandleModernBert;
            let model_dir = emb.model_dir.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "candle-modernbert: [embeddings.model_dir] is required (see \
                     scripts/prepare-granite-r2-97m.sh)"
                )
            })?;
            let encoder = CandleModernBert::load_from_dir(
                model_dir,
                emb.model_id.clone(),
                emb.model_version.clone(),
                emb.profile.clone(),
                emb.dim,
                emb.max_len,
            )
            .map_err(|e| anyhow::anyhow!("init candle-modernbert encoder: {e}"))?;
            Ok(Some(Arc::new(encoder) as SharedEncoder))
        }
        #[cfg(feature = "embeddings-ollama")]
        "ollama" => {
            use msa_core::embeddings::OllamaClient;
            tracing::warn!(
                "[embeddings] backend=ollama is deprecated; migrate to \
                 candle-modernbert (in-process, no daemon)"
            );
            let url = emb
                .url
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("ollama backend: [embeddings.url] is required"))?;
            let model = emb
                .model
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("ollama backend: [embeddings.model] is required"))?;
            let client = OllamaClient::new(url, model, emb.dim)
                .map_err(|e| anyhow::anyhow!("init ollama encoder: {e}"))?;
            Ok(Some(Arc::new(client) as SharedEncoder))
        }
        other => Err(anyhow::anyhow!(
            "unsupported embeddings backend: {other:?} (built-in options: \
             'candle-modernbert', deprecated 'ollama')"
        )),
    }
}

#[cfg(not(any(feature = "embeddings-candle", feature = "embeddings-ollama")))]
fn build_encoder(config: &MsaConfig) -> Result<Option<SharedEncoder>> {
    if config.embeddings.is_some() {
        tracing::warn!(
            "[embeddings] section present in config but binary built without \
             --features embeddings; falling back to BM25-only mode"
        );
    }
    Ok(None)
}
