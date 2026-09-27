//! Scoring helpers for diarization: optimal label mapping and a duration-weighted
//! speaker error rate. [`crate::recluster`] uses the matching to carry online labels over;
//! the eval harness (`examples/diarize_eval.rs`) and the tests use the scoring.

use std::collections::HashMap;

/// Maximum-weight one-to-one matching of rows to columns (Hungarian algorithm, O(n³)).
///
/// `weights[r][c]` is the gain of matching row `r` to column `c`; rows may have different
/// lengths (missing entries count as 0). Returns the matched column for every row, or `None`
/// when the row is left over (more rows than columns). A row can be matched to a column with
/// zero weight; callers that care filter those out.
pub fn max_weight_matching(weights: &[Vec<f64>]) -> Vec<Option<usize>> {
    let rows = weights.len();
    let cols = weights.iter().map(Vec::len).max().unwrap_or(0);
    if rows == 0 || cols == 0 {
        return vec![None; rows];
    }
    let n = rows.max(cols);
    let w = |r: usize, c: usize| -> f64 {
        weights
            .get(r)
            .and_then(|row| row.get(c))
            .copied()
            .filter(|x| x.is_finite())
            .unwrap_or(0.0)
    };
    let max_w = (0..rows)
        .flat_map(|r| (0..cols).map(move |c| (r, c)))
        .map(|(r, c)| w(r, c))
        .fold(0.0f64, f64::max);
    // Square cost matrix (1-based below), padding costs `max_w` (= weight 0).
    let cost = |r: usize, c: usize| -> f64 { max_w - w(r, c) };

    let inf = f64::INFINITY;
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1]; // p[col] = row matched to col (1-based, 0 = none)
    let mut way = vec![0usize; n + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=n {
                if !used[j] {
                    let cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut out = vec![None; rows];
    for (j, &pj) in p.iter().enumerate().skip(1) {
        let (r, c) = (pj - 1, j - 1);
        if r < rows && c < cols {
            out[r] = Some(c);
        }
    }
    out
}

/// Result of [`score`].
#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    /// Duration-weighted fraction of speech whose hypothesis label is wrong after the best
    /// one-to-one mapping of hypothesis labels to true speakers. Unlabeled (`None`) speech and
    /// speech of extra (unmapped) hypothesis speakers count as errors.
    pub error_rate: f64,
    /// Duration-weighted fraction of speech with no label at all (part of `error_rate`).
    pub unlabeled_rate: f64,
    /// Distinct hypothesis labels (excluding `None`).
    pub hyp_speakers: usize,
    /// Distinct true speakers.
    pub ref_speakers: usize,
}

/// Score hypothesis labels against the truth, one entry per segment.
pub fn score<R: AsRef<str>, H: AsRef<str>>(
    reference: &[R],
    hypothesis: &[Option<H>],
    durations_ms: &[u64],
) -> Score {
    assert_eq!(reference.len(), hypothesis.len());
    assert_eq!(reference.len(), durations_ms.len());
    let mut ref_ix: HashMap<&str, usize> = HashMap::new();
    let mut hyp_ix: HashMap<&str, usize> = HashMap::new();
    for r in reference {
        let k = ref_ix.len();
        ref_ix.entry(r.as_ref()).or_insert(k);
    }
    for h in hypothesis.iter().flatten() {
        let k = hyp_ix.len();
        hyp_ix.entry(h.as_ref()).or_insert(k);
    }
    let mut conf = vec![vec![0.0f64; ref_ix.len()]; hyp_ix.len()];
    let mut total = 0.0;
    let mut unlabeled = 0.0;
    for ((r, h), &d) in reference.iter().zip(hypothesis).zip(durations_ms) {
        let d = d as f64;
        total += d;
        match h {
            Some(h) => conf[hyp_ix[h.as_ref()]][ref_ix[r.as_ref()]] += d,
            None => unlabeled += d,
        }
    }
    let mapping = max_weight_matching(&conf);
    let matched: f64 = mapping
        .iter()
        .enumerate()
        .filter_map(|(h, c)| c.map(|c| conf[h][c]))
        .sum();
    let (error_rate, unlabeled_rate) = if total > 0.0 {
        (1.0 - matched / total, unlabeled / total)
    } else {
        (0.0, 0.0)
    };
    Score {
        error_rate,
        unlabeled_rate,
        hyp_speakers: hyp_ix.len(),
        ref_speakers: ref_ix.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_picks_the_global_optimum() {
        // Greedy would take (0,0)=9 and then (1,1)=1; optimal is (0,1)+(1,0) = 8+8.
        let w = vec![vec![9.0, 8.0], vec![8.0, 1.0]];
        assert_eq!(max_weight_matching(&w), vec![Some(1), Some(0)]);
    }

    #[test]
    fn matching_handles_rectangles() {
        let w = vec![vec![1.0, 5.0, 2.0]];
        assert_eq!(max_weight_matching(&w), vec![Some(1)]);
        let w = vec![vec![1.0], vec![5.0], vec![2.0]];
        assert_eq!(max_weight_matching(&w), vec![None, Some(0), None]);
        assert!(max_weight_matching(&[]).is_empty());
    }

    #[test]
    fn score_is_permutation_invariant() {
        let r = ["a", "a", "b", "c"];
        let h = [Some("x"), Some("x"), Some("y"), Some("z")];
        let s = score(&r, &h, &[1000, 1000, 500, 500]);
        assert_eq!(s.error_rate, 0.0);
        assert_eq!((s.hyp_speakers, s.ref_speakers), (3, 3));
    }

    #[test]
    fn score_counts_splits_merges_and_unlabeled() {
        let r = ["a", "a", "b", "b"];
        // "a" split into x/y, "b" half unlabeled.
        let h = [Some("x"), Some("y"), Some("z"), None];
        let s = score(&r, &h, &[3000, 1000, 1000, 1000]);
        assert!((s.error_rate - 2.0 / 6.0).abs() < 1e-9);
        assert!((s.unlabeled_rate - 1.0 / 6.0).abs() < 1e-9);
        // Everything merged into one label: only the bigger speaker maps.
        let h = [Some("x"); 4];
        let s = score(&r, &h, &[3000, 1000, 1000, 1000]);
        assert!((s.error_rate - 2.0 / 6.0).abs() < 1e-9);
    }
}
