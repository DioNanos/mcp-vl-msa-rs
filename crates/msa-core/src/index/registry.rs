//! Process-wide registry of open `MsaIndex` instances, keyed by collection
//! name. Single authority for path-safety (name validation happens here before
//! any filesystem path is constructed).

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{MsaError, Result};
use crate::schema::ChunkConfig;

use super::write::MsaIndex;

/// Process-wide registry of open indices, keyed by collection name.
#[derive(Default)]
pub struct CollectionRegistry {
    inner: std::sync::Mutex<HashMap<String, Arc<MsaIndex>>>,
    /// Monotonic counter for collision-resistant quarantine directory names.
    drop_seq: std::sync::atomic::AtomicU64,
}

impl CollectionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Open or create a collection using the default writer heap. See
    /// [`Self::open_or_create_with_heap`] to pass a configured arena size.
    pub fn open_or_create(
        &self,
        name: &str,
        base_dir: &std::path::Path,
        chunking: &ChunkConfig,
    ) -> Result<Arc<MsaIndex>> {
        self.open_or_create_with_heap(
            name,
            base_dir,
            chunking,
            crate::config::default_writer_heap_bytes(),
        )
    }

    /// Open or create a collection with an explicit writer heap arena.
    ///
    /// Single authority for path safety: a name that fails validation never
    /// becomes a filesystem path. This is the only place collection dirs are
    /// constructed, so `..`, separators and absolute paths are impossible.
    pub fn open_or_create_with_heap(
        &self,
        name: &str,
        base_dir: &std::path::Path,
        chunking: &ChunkConfig,
        writer_heap_bytes: usize,
    ) -> Result<Arc<MsaIndex>> {
        crate::validate::validate_collection_name(name)?;
        let mut guard = self.inner.lock().expect("registry poisoned");
        if let Some(idx) = guard.get(name) {
            return Ok(idx.clone());
        }
        let path = base_dir.join(name);
        let idx = Arc::new(MsaIndex::open_with_heap(
            name.to_string(),
            path,
            chunking.clone(),
            writer_heap_bytes,
        )?);
        guard.insert(name.to_string(), idx.clone());
        Ok(idx)
    }

    /// Drop a collection: evict it from the in-memory registry and delete its
    /// on-disk index. Returns `true` if the collection existed on disk.
    ///
    /// Safe-deletion protocol (audit S6 + concurrency fix): the eviction and
    /// the rename-into-quarantine happen **while holding the registry mutex**,
    /// so a concurrent `open_or_create` cannot re-open the live directory and
    /// re-insert the collection in the gap between evict and rename. The live
    /// directory is renamed to a hidden, collision-resistant sibling
    /// (`.trash-<name>-<pid>-<seq>`) — atomic on a single filesystem and
    /// instantly invisible to `list_on_disk`/`open`. The recursive removal of
    /// the quarantined directory runs **after** the lock is released
    /// (best-effort), so a slow delete does not block other collections.
    ///
    /// Idempotent under concurrency: if the live directory is already gone
    /// (e.g. a racing drop won), this returns `Ok(false)` instead of erroring.
    /// We never `remove_dir_all` the original path while it may still be open.
    pub fn drop_collection(&self, name: &str, base_dir: &std::path::Path) -> Result<bool> {
        crate::validate::validate_collection_name(name)?;
        let path = base_dir.join(name);

        // --- critical section: evict + quarantine rename, atomic vs open ---
        let trash = {
            let mut guard = self.inner.lock().expect("registry poisoned");
            // Drop our Arc first so the writer/reader is released before we
            // move the directory (if no other Arc clone is still held).
            if let Some(idx) = guard.remove(name) {
                idx.release_writer();
                drop(idx);
            }
            if !path.exists() {
                // Nothing on disk (never created, or a racing drop already
                // renamed it). Idempotent no-op.
                return Ok(false);
            }
            let seq = self
                .drop_seq
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let pid = std::process::id();
            let trash = base_dir.join(format!(".trash-{name}-{pid}-{seq}"));
            match std::fs::rename(&path, &trash) {
                Ok(()) => trash,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    // A concurrent drop renamed it between exists() and rename.
                    return Ok(false);
                }
                Err(e) => return Err(MsaError::Io(e)),
            }
        };
        // --- end critical section; lock released before slow delete ---

        if let Err(e) = std::fs::remove_dir_all(&trash) {
            // Logically dropped (rename succeeded); the leftover is hidden and
            // can be reaped later. Surface as a warning, not a hard error.
            tracing::warn!(
                target: "msa::drop",
                "collection '{name}' renamed to quarantine but cleanup failed: {e}"
            );
        }
        Ok(true)
    }

    /// Collections currently open in memory (this process's registry).
    pub fn list(&self) -> Vec<String> {
        self.inner
            .lock()
            .expect("registry poisoned")
            .keys()
            .cloned()
            .collect()
    }

    /// All persisted collections discovered by scanning `base_dir`, unioned
    /// with those open in memory. A subdirectory counts as a collection only
    /// if its name passes validation (so stray/manual dirs that could not have
    /// been created by us are ignored) and it looks like a tantivy index
    /// (has a `meta.json`). Sorted and deduplicated.
    pub fn list_on_disk(&self, base_dir: &std::path::Path) -> Vec<String> {
        let mut names: std::collections::BTreeSet<String> = self.list().into_iter().collect();
        if let Ok(entries) = std::fs::read_dir(base_dir) {
            for entry in entries.flatten() {
                if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    continue;
                }
                let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                    continue;
                };
                if crate::validate::validate_collection_name(&name).is_err() {
                    continue;
                }
                if entry.path().join("meta.json").exists() {
                    names.insert(name);
                }
            }
        }
        names.into_iter().collect()
    }
}
