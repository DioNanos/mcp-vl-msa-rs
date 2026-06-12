//! Retrieval metrics. All functions take a ranked list of doc ids (best
//! first) and the set of relevant doc ids for the query, and assume
//! binary relevance (relevant / not relevant). Extension to graded
//! relevance is straightforward but unnecessary for MSA-RAG-BENCHMARKS.

use std::collections::HashSet;

/// NDCG@k with binary relevance and log2(i+2) discount.
pub fn ndcg_at_k(ranked: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    let k = k.min(ranked.len());
    if k == 0 || relevant.is_empty() {
        return 0.0;
    }
    let dcg: f64 = ranked
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, doc)| {
            if relevant.contains(doc) {
                1.0 / ((i as f64 + 2.0).log2())
            } else {
                0.0
            }
        })
        .sum();
    let n_rel_in_top_k = ranked
        .iter()
        .take(k)
        .filter(|d| relevant.contains(*d))
        .count();
    if n_rel_in_top_k == 0 {
        return 0.0;
    }
    let ideal_rels = relevant.len().min(k);
    let idcg: f64 = (0..ideal_rels)
        .map(|i| 1.0 / ((i as f64 + 2.0).log2()))
        .sum();
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

/// MRR@k: reciprocal rank of the first relevant doc, truncated at k.
pub fn mrr_at_k(ranked: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    let k = k.min(ranked.len());
    for (i, doc) in ranked.iter().take(k).enumerate() {
        if relevant.contains(doc) {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

/// Recall@k: |relevant ∩ top-k| / |relevant|.
pub fn recall_at_k(ranked: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let k = k.min(ranked.len());
    let hits = ranked
        .iter()
        .take(k)
        .filter(|d| relevant.contains(*d))
        .count();
    hits as f64 / relevant.len() as f64
}

/// Compute the p-th percentile of a slice in milliseconds. `p` in 0..=100.
/// Non-mutating: sorts a copy.
pub fn percentile_ms(samples: &[u128], p: u8) -> u128 {
    if samples.is_empty() {
        return 0;
    }
    let mut sorted: Vec<u128> = samples.to_vec();
    sorted.sort_unstable();
    // Nearest-rank method: idx = ceil(p/100 * n) - 1. The previous
    // round(p/100 * (n-1)) gave p50 of [10..100] = 60 instead of 50.
    let idx = ((p as f64 / 100.0) * (sorted.len() as f64)).ceil() as usize;
    sorted[idx.saturating_sub(1).min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rels(items: &[&str]) -> HashSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }
    fn rank(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn ndcg_perfect_ranking() {
        let r = rank(&["a", "b", "c", "d"]);
        let rel = rels(&["a", "b"]);
        // Both at rank 1 and 2 = ideal.
        assert!((ndcg_at_k(&r, &rel, 10) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ndcg_worst_case() {
        let r = rank(&["x", "y", "z"]);
        let rel = rels(&["a"]);
        assert_eq!(ndcg_at_k(&r, &rel, 10), 0.0);
    }

    #[test]
    fn mrr_first_position() {
        assert!((mrr_at_k(&rank(&["a", "b"]), &rels(&["a"]), 10) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn mrr_third_position() {
        assert!((mrr_at_k(&rank(&["x", "y", "a"]), &rels(&["a"]), 10) - (1.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn mrr_outside_topk() {
        assert_eq!(mrr_at_k(&rank(&["x", "y", "a"]), &rels(&["a"]), 2), 0.0);
    }

    #[test]
    fn recall_two_of_three() {
        let r = rank(&["a", "x", "b"]);
        let rel = rels(&["a", "b", "c"]);
        assert!((recall_at_k(&r, &rel, 10) - (2.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn percentile_basic() {
        let s = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        assert_eq!(percentile_ms(&s, 50), 50);
        assert_eq!(percentile_ms(&s, 95), 100);
    }
}
