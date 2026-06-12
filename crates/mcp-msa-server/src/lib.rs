mod hint;
pub mod server;
/// Filesystem source adapter for `msa_sync_path` (compiled only with `source-fs`).
#[cfg(feature = "source-fs")]
mod sync_fs;

pub use server::MsaServer;
