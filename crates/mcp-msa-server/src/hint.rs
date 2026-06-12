//! Teaching hints for weak MCP clients.
//!
//! Some AI clients with partial MCP support silently treat an empty search
//! result as "no data" without realizing the collection they queried was never
//! created. These helpers turn that dead end into an additive `hint` field on
//! the response that names the existing collections and the tool to create one.

/// Cap the number of collection names listed in a hint, so the message stays
/// small even on servers hosting many collections.
const MAX_LISTED_COLLECTIONS: usize = 20;

/// Build a teaching hint for a search that returned no hits against a
/// collection that does not exist (neither open in memory nor persisted on
/// disk). Returns `None` when the collection exists — callers must only add the
/// hint when the collection is genuinely absent, so an existing-but-empty
/// collection keeps its silent `{"hits":[]}` response unchanged.
///
/// `existing` is the set of collection names known before this call (from
/// `list_on_disk`), since opening a collection lazily creates its directory.
pub fn missing_collection_hint(collection: &str, existing: &[String]) -> String {
    let mut listed: Vec<&str> = existing
        .iter()
        .take(MAX_LISTED_COLLECTIONS)
        .map(String::as_str)
        .collect();
    listed.sort_unstable();
    let joined = listed.join(", ");
    let more = existing.len().saturating_sub(MAX_LISTED_COLLECTIONS);
    let suffix = if more > 0 {
        format!(", … (+{more} more)")
    } else {
        String::new()
    };
    format!(
        "collection '{collection}' does not exist; existing collections: [{joined}{suffix}]; \
         index documents with msa_index to create it"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_existing_collections_and_the_create_tool() {
        let existing = vec!["beta".to_string(), "alpha".to_string()];
        let h = missing_collection_hint("ghost", &existing);
        assert!(h.contains("'ghost' does not exist"));
        // Listed sorted.
        assert!(h.contains("[alpha, beta]"));
        assert!(h.contains("msa_index"));
    }

    #[test]
    fn empty_existing_yields_empty_brackets() {
        let h = missing_collection_hint("ghost", &[]);
        assert!(h.contains("existing collections: []"));
    }

    #[test]
    fn caps_the_listed_collections_at_twenty() {
        let existing: Vec<String> = (0..30).map(|i| format!("c{i:02}")).collect();
        let h = missing_collection_hint("ghost", &existing);
        // 30 - 20 = 10 elided.
        assert!(h.contains("(+10 more)"), "hint was: {h}");
        // First 20 sorted alphabetically: c00..c19 present, c29 elided.
        assert!(h.contains("c00"));
        assert!(!h.contains("c29"));
    }
}
