//! Speaker change detection inside one utterance.
//!
//! The VAD closes an utterance only after a pause, so when one person answers another
//! without a pause (or interrupts), both end up in one final. [`change_points`] finds where
//! the voice changes so the caller can cut the utterance into single-speaker pieces:
//!
//! 1. Embed windows of [`CHANGE_WINDOW_MS`] every [`CHANGE_HOP_MS`].
//! 2. At every hop boundary, compare the mean embedding of the windows just before it with
//!    the mean of the windows just after it ([`CHANGE_CONTEXT_MS`] each side).
//! 3. Local minima below [`ClusterConfig::change_threshold`] become candidates, strongest
//!    first, at least [`ClusterConfig::min_piece_ms`] apart and from the ends.
//! 4. Verification: while two neighbouring pieces (mean of the windows inside each) are at
//!    least `change_threshold` similar, the boundary between them is dropped.
//! 5. Each cut moves to the quietest 20 ms within ±[`SNAP_MS`] so words aren't cut.

use crate::cluster::{dot, normalized, ClusterConfig};
use crate::embedder::Embedder;
use kenes_types::SAMPLE_RATE;

/// Window length for change detection.
pub const CHANGE_WINDOW_MS: u64 = 1500;
/// Hop between windows (and resolution of candidate change points before snapping).
pub const CHANGE_HOP_MS: u64 = 500;
/// How much audio on each side of a candidate point is compared.
pub const CHANGE_CONTEXT_MS: u64 = 2000;
/// A cut moves to the quietest spot within this distance.
pub const SNAP_MS: u64 = 300;

const SR_MS: usize = SAMPLE_RATE as usize / 1000;

/// Embeddings of fixed-length windows over one utterance.
#[derive(Clone, Debug, Default)]
pub struct Windows {
    /// Window length in samples.
    pub win: usize,
    /// Utterance length in samples.
    pub len: usize,
    /// Start sample of each window (ascending).
    pub starts: Vec<usize>,
    /// Normalized embedding of each window.
    pub embs: Vec<Vec<f32>>,
}

/// Embed windows of `win_ms` every `hop_ms` (the last window is aligned to the end so the
/// whole utterance is covered). Utterances shorter than one window give one window over
/// everything. Cost: about `win_ms / hop_ms` times the cost of embedding the utterance once.
pub fn window_embeddings(
    embedder: &mut Embedder,
    samples: &[f32],
    win_ms: u64,
    hop_ms: u64,
) -> anyhow::Result<Windows> {
    let win = (win_ms as usize * SR_MS).min(samples.len());
    let hop = (hop_ms as usize * SR_MS).max(1);
    let mut starts: Vec<usize> = (0..)
        .map(|k| k * hop)
        .take_while(|&s| s + win <= samples.len())
        .collect();
    let last = samples.len() - win;
    if starts.last() != Some(&last) {
        starts.push(last);
    }
    let mut embs = Vec::with_capacity(starts.len());
    for &s in &starts {
        embs.push(embedder.embed(&samples[s..s + win])?);
    }
    Ok(Windows {
        win,
        len: samples.len(),
        starts,
        embs,
    })
}

fn mean(embs: &[&[f32]]) -> Option<Vec<f32>> {
    let first = embs.first()?;
    let mut m = vec![0.0f32; first.len()];
    for e in embs {
        m.iter_mut().zip(e.iter()).for_each(|(a, x)| *a += x);
    }
    normalized(&m)
}

/// Candidate change points (sample offsets, ascending) from window embeddings; no snapping.
/// Candidates are tested every `hop_ms`.
pub fn pick_changes(w: &Windows, hop_ms: u64, cfg: &ClusterConfig) -> Vec<usize> {
    let min_piece = cfg.min_piece_ms as usize * SR_MS;
    if w.embs.len() < 2 || w.len < 2 * min_piece.max(1) {
        return Vec::new();
    }
    let hop = (hop_ms as usize * SR_MS).max(1);
    let ctx = CHANGE_CONTEXT_MS as usize * SR_MS;
    let win = w.win;
    // Windows before `b`: end ≤ b (or the first window, if none fits), start ≥ b - ctx.
    let left = |b: usize| -> Option<Vec<f32>> {
        let mut v: Vec<&[f32]> = w
            .starts
            .iter()
            .zip(&w.embs)
            .filter(|(&s, _)| s + win <= b && s + ctx >= b)
            .map(|(_, e)| e.as_slice())
            .collect();
        if v.is_empty() && b < win {
            v.push(&w.embs[0]);
        }
        mean(&v)
    };
    let right = |b: usize| -> Option<Vec<f32>> {
        let mut v: Vec<&[f32]> = w
            .starts
            .iter()
            .zip(&w.embs)
            .filter(|(&s, _)| s >= b && s + win <= b + ctx.max(win))
            .map(|(_, e)| e.as_slice())
            .collect();
        if v.is_empty() && b + win > w.len {
            v.push(w.embs.last().expect("non-empty"));
        }
        mean(&v)
    };
    let points: Vec<usize> = (1..)
        .map(|k| k * hop)
        .take_while(|&b| b + min_piece <= w.len)
        .filter(|&b| b >= min_piece)
        .collect();
    let sims: Vec<f32> = points
        .iter()
        .map(|&b| match (left(b), right(b)) {
            (Some(l), Some(r)) => dot(&l, &r),
            _ => 1.0,
        })
        .collect();
    // Local minima under the threshold, most dissimilar first.
    let mut cand: Vec<(usize, f32)> = (0..points.len())
        .filter(|&i| sims[i] < cfg.change_threshold)
        .filter(|&i| {
            (i == 0 || sims[i] <= sims[i - 1]) && (i + 1 == sims.len() || sims[i] <= sims[i + 1])
        })
        .map(|i| (points[i], sims[i]))
        .collect();
    cand.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let mut cuts: Vec<usize> = Vec::new();
    for (b, _) in cand {
        if cuts.iter().all(|&c| c.abs_diff(b) >= min_piece) {
            cuts.push(b);
        }
    }
    cuts.sort_unstable();

    // Verification on whole pieces: drop the weakest boundary while two neighbours match.
    let piece = |a: usize, b: usize| -> Option<Vec<f32>> {
        let inside: Vec<&[f32]> = w
            .starts
            .iter()
            .zip(&w.embs)
            .filter(|(&s, _)| s + win / 2 >= a && s + win / 2 < b)
            .map(|(_, e)| e.as_slice())
            .collect();
        mean(&inside)
    };
    while !cuts.is_empty() {
        let mut bounds = vec![0usize];
        bounds.extend(&cuts);
        bounds.push(w.len);
        let pieces: Vec<Option<Vec<f32>>> = bounds.windows(2).map(|p| piece(p[0], p[1])).collect();
        let most_similar = (0..cuts.len())
            .map(|i| {
                let s = match (&pieces[i], &pieces[i + 1]) {
                    (Some(a), Some(b)) => dot(a, b),
                    _ => 1.0, // a piece without its own window can't be told apart
                };
                (i, s)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1));
        match most_similar {
            Some((i, s)) if s >= cfg.change_threshold => {
                cuts.remove(i);
            }
            _ => break,
        }
    }
    cuts
}

/// Move each cut to the quietest 20 ms frame within ±`snap_ms`, keeping pieces at least
/// `min_piece` samples long and cuts in order.
pub fn snap_to_quiet(samples: &[f32], cuts: &mut [usize], snap_ms: u64, min_piece: usize) {
    let frame = 20 * SR_MS;
    let snap = snap_ms as usize * SR_MS;
    let mut prev = 0usize;
    for i in 0..cuts.len() {
        let next = cuts.get(i + 1).copied().unwrap_or(samples.len());
        let lo = cuts[i].saturating_sub(snap).max(prev + min_piece);
        let hi = (cuts[i] + snap).min(next.saturating_sub(min_piece));
        if lo + frame <= hi {
            let mut best = (f32::MAX, cuts[i]);
            let mut s = lo;
            while s + frame <= hi {
                let e: f32 = samples[s..s + frame].iter().map(|x| x * x).sum();
                if e < best.0 {
                    best = (e, s + frame / 2);
                }
                s += frame / 2;
            }
            cuts[i] = best.1;
        }
        prev = cuts[i];
    }
}

/// Where the speaker changes inside one utterance (16 kHz mono): sample offsets, ascending,
/// each piece at least [`ClusterConfig::min_piece_ms`] long. Empty for utterances shorter
/// than two pieces, for one speaker, or if embedding fails.
///
/// Tuned so that single-speaker utterances are almost never split (see `EVAL.md`); cost is
/// about 3 × the cost of embedding the utterance once (1.5 s windows every 0.5 s), roughly
/// 4 % of real time on one P-core.
pub fn change_points(embedder: &mut Embedder, samples: &[f32], cfg: &ClusterConfig) -> Vec<usize> {
    let min_piece = cfg.min_piece_ms as usize * SR_MS;
    if samples.len() < 2 * min_piece.max(SR_MS * 500) {
        return Vec::new();
    }
    let w = match window_embeddings(embedder, samples, CHANGE_WINDOW_MS, CHANGE_HOP_MS) {
        Ok(w) => w,
        Err(e) => {
            log::debug!("change detection skipped: {e}");
            return Vec::new();
        }
    };
    let mut cuts = pick_changes(&w, CHANGE_HOP_MS, cfg);
    snap_to_quiet(samples, &mut cuts, SNAP_MS, min_piece);
    cuts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::l2_normalize;

    fn unit(k: usize) -> Vec<f32> {
        let mut v = vec![0.0; 8];
        v[k] = 1.0;
        v
    }

    /// Windows every 0.5 s of 1.5 s over `secs`, voice `a` before `change_s`, `b` after
    /// (windows straddling the change get a mix).
    fn windows(secs: f32, change_s: Option<f32>, a: usize, b: usize) -> Windows {
        let win = 1500 * SR_MS;
        let len = (secs * 16000.0) as usize;
        let hop = 500 * SR_MS;
        let mut starts: Vec<usize> = (0..)
            .map(|k| k * hop)
            .take_while(|&s| s + win <= len)
            .collect();
        if starts.last() != Some(&(len - win)) {
            starts.push(len - win);
        }
        let embs = starts
            .iter()
            .map(|&s| {
                let c = change_s.map_or(usize::MAX, |c| (c * 16000.0) as usize);
                let before = c.saturating_sub(s).min(win) as f32 / win as f32;
                let mut v: Vec<f32> = unit(a)
                    .iter()
                    .zip(unit(b))
                    .map(|(x, y)| before * x + (1.0 - before) * y)
                    .collect();
                l2_normalize(&mut v);
                v
            })
            .collect();
        Windows {
            win,
            len,
            starts,
            embs,
        }
    }

    fn cfg() -> ClusterConfig {
        ClusterConfig {
            change_threshold: 0.5,
            min_piece_ms: 1000,
            ..ClusterConfig::default()
        }
    }

    #[test]
    fn one_voice_is_never_split() {
        for secs in [2.5f32, 5.0, 12.0, 20.0] {
            assert!(pick_changes(&windows(secs, None, 0, 0), 500, &cfg()).is_empty());
        }
    }

    #[test]
    fn finds_a_change_near_the_truth() {
        for (secs, at) in [(6.0f32, 3.0f32), (10.0, 2.2), (10.0, 7.5), (4.0, 1.6)] {
            let cuts = pick_changes(&windows(secs, Some(at), 0, 1), 500, &cfg());
            assert_eq!(cuts.len(), 1, "{secs} s, change at {at}: {cuts:?}");
            let got = cuts[0] as f32 / 16000.0;
            assert!(
                (got - at).abs() <= 0.5,
                "{secs} s, change at {at}: got {got}"
            );
        }
    }

    #[test]
    fn respects_min_piece() {
        // Change 0.6 s before the end: the second piece would be too short.
        assert!(pick_changes(&windows(6.0, Some(5.4), 0, 1), 500, &cfg()).is_empty());
        assert!(pick_changes(&windows(1.8, Some(0.9), 0, 1), 500, &cfg()).is_empty());
    }

    #[test]
    fn snapping_prefers_silence_and_keeps_order() {
        let mut x = vec![0.5f32; 16000 * 4];
        // Quiet gap at 2.2–2.3 s.
        x[16000 * 22 / 10..16000 * 23 / 10]
            .iter_mut()
            .for_each(|v| *v = 0.0);
        let mut cuts = vec![32000];
        snap_to_quiet(&x, &mut cuts, 300, 16000);
        assert!((35200..=36800).contains(&cuts[0]), "{cuts:?}");
        // Min piece wins over silence.
        let mut cuts = vec![20000];
        snap_to_quiet(&x, &mut cuts, 300, 16000);
        assert!(cuts[0] >= 16000 && cuts[0] <= 24800);
    }
}
