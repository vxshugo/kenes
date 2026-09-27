//! Streaming speaker assignment ([`OnlineClusterer`]) and the knobs it shares with
//! [`crate::recluster`] ([`ClusterConfig`]).

use std::collections::HashMap;

/// Label of the enrolled user (voiceprint match).
pub const ME: &str = "me";

/// Tuning for [`OnlineClusterer`], [`crate::recluster`] and [`crate::change_points`].
///
/// Similarities are cosine similarities between L2-normalized embeddings. The defaults are
/// tuned for [`crate::SPEAKER_MODEL`] on synthetic 10–15-speaker ru/kk meetings, segmented
/// by the real kenes-stt VAD and cut with [`crate::change_points`] (see `EVAL.md`); a
/// different embedding model needs different values.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterConfig {
    /// Online: similarity to a speaker's centroid needed to join that speaker. Below it the
    /// segment opens a new speaker.
    pub threshold: f32,
    /// Segments shorter than this are not embedded; callers pass `embedding: None`.
    pub min_embed_ms: u64,
    /// Online: two speakers whose centroids reach this similarity are merged into one
    /// (guards against one person being split in two). `> 1.0` disables merging.
    pub merge_threshold: f32,
    /// Similarity to the enrolled voiceprint needed for `"me"`, online (per segment) and in
    /// [`crate::recluster`] (per cluster).
    pub voiceprint_threshold: f32,
    /// A segment shorter than this never opens a new speaker: its embedding is too noisy to
    /// tell a new voice from a known one (or it's a cough). If it matches nobody it is treated
    /// like a segment without embedding.
    pub min_new_speaker_ms: u64,
    /// Segments shorter than this have noisier embeddings: online they join a known speaker
    /// (or the user) at a similarity [`ClusterConfig::short_segment_relax`] below the usual
    /// threshold, and offline they don't seed clusters but join the nearest one afterwards.
    pub short_segment_ms: u64,
    /// How much lower the online join thresholds are for short segments.
    pub short_segment_relax: f32,
    /// A segment without a usable embedding inherits the label of the previous segment of the
    /// same source if that one ended at most this long before it started; otherwise `None`.
    pub context_ms: u64,
    /// Upper bound on distinct speakers per source (online and offline). Once reached, new
    /// voices go to the nearest existing speaker.
    pub max_speakers: usize,
    /// Offline: average-linkage similarity down to which clusters keep merging.
    pub recluster_threshold: f32,
    /// Offline: a cluster with less speech than this is taken for a stray piece of a known
    /// voice and joins its nearest cluster if that one is at least
    /// `recluster_threshold - 0.1` similar. `0` disables this.
    pub min_cluster_ms: u64,
    /// [`crate::change_points`]: the voice changes where the audio just before and just
    /// after a point are less similar than this (and the resulting pieces too).
    pub change_threshold: f32,
    /// [`crate::change_points`]: shortest piece an utterance is cut into.
    pub min_piece_ms: u64,
}

impl Default for ClusterConfig {
    /// Values tuned in `EVAL.md` for [`crate::SPEAKER_MODEL`].
    fn default() -> Self {
        ClusterConfig {
            threshold: 0.60,
            min_embed_ms: 500,
            merge_threshold: 0.70,
            voiceprint_threshold: 0.60,
            min_new_speaker_ms: 1500,
            short_segment_ms: 3000,
            short_segment_relax: 0.1,
            context_ms: 2000,
            max_speakers: 40,
            recluster_threshold: 0.625,
            min_cluster_ms: 5000,
            change_threshold: 0.55,
            min_piece_ms: 1000,
        }
    }
}

/// Cosine similarity of two vectors (0 when either is zero or the lengths differ).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        ab += x as f64 * y as f64;
        aa += x as f64 * x as f64;
        bb += y as f64 * y as f64;
    }
    if aa <= 0.0 || bb <= 0.0 {
        return 0.0;
    }
    (ab / (aa.sqrt() * bb.sqrt())) as f32
}

/// Scale `v` to unit length. Returns `false` (leaving `v` untouched) when it is empty, zero
/// or not finite.
pub fn l2_normalize(v: &mut [f32]) -> bool {
    let norm = v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
    if v.is_empty() || !norm.is_finite() || norm <= 1e-12 {
        return false;
    }
    let inv = (1.0 / norm) as f32;
    v.iter_mut().for_each(|x| *x *= inv);
    true
}

/// Normalized copy of `v`, or `None` if it can't be normalized.
pub(crate) fn normalized(v: &[f32]) -> Option<Vec<f32>> {
    let mut v = v.to_vec();
    l2_normalize(&mut v).then_some(v)
}

/// Dot product; equals cosine similarity for unit vectors.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// How much a segment counts in a centroid: its duration in seconds, clamped to
/// `0.5..=10` so long monologues don't swamp a speaker and blips still count a little.
pub(crate) fn duration_weight(duration_ms: u64) -> f32 {
    (duration_ms as f32 / 1000.0).clamp(0.5, 10.0)
}

/// `"<prefix>:<n>"` → `n`.
pub(crate) fn label_number(prefix: &str, label: &str) -> Option<u32> {
    label.strip_prefix(prefix)?.strip_prefix(':')?.parse().ok()
}

#[derive(Clone, Debug)]
struct Speaker {
    label: String,
    /// Order of creation; the user (voiceprint) is 0.
    number: u32,
    is_me: bool,
    /// Σ weight·embedding over assigned segments.
    sum: Vec<f32>,
    /// Σ weight.
    weight: f32,
    /// normalize(sum), or the voiceprint for the user.
    centroid: Vec<f32>,
}

#[derive(Clone, Debug)]
struct Last {
    end_ms: u64,
    label: Option<String>,
}

/// Streaming speaker assignment within one source (`"mic"` or `"sys"`).
///
/// Every speaker is a centroid: the duration-weighted mean of its segments' normalized
/// embeddings, re-normalized. A segment joins the most similar speaker if the cosine
/// similarity reaches [`ClusterConfig::threshold`], otherwise it opens `"<prefix>:<N>"`.
/// With a voiceprint set, a segment whose similarity to it reaches
/// [`ClusterConfig::voiceprint_threshold`] (and beats every speaker by margin) is `"me"`.
///
/// After each assignment, speakers whose centroids reach [`ClusterConfig::merge_threshold`]
/// are merged. Labels already returned are never changed: the surviving label (the user, else
/// the speaker with more speech, else the older one) is used from then on, and
/// [`OnlineClusterer::resolve`] maps the retired label to it. Numbers are never reused.
#[derive(Clone, Debug)]
pub struct OnlineClusterer {
    prefix: String,
    cfg: ClusterConfig,
    dim: Option<usize>,
    speakers: Vec<Speaker>,
    /// Retired (merged-away) label → surviving label. Kept flat: values are never retired.
    aliases: HashMap<String, String>,
    next_number: u32,
    last: Option<Last>,
}

impl OnlineClusterer {
    /// A clusterer that labels speakers `"<prefix>:1"`, `"<prefix>:2"`, …
    pub fn new(prefix: &str, cfg: ClusterConfig) -> Self {
        OnlineClusterer {
            prefix: prefix.to_string(),
            cfg,
            dim: None,
            speakers: Vec::new(),
            aliases: HashMap::new(),
            next_number: 1,
            last: None,
        }
    }

    /// Enable the `"me"` label: segments matching this embedding (the enrolled user's voice,
    /// from [`crate::Embedder::embed`]) are labeled `"me"`. Replaces an earlier voiceprint.
    /// A zero or non-finite embedding is ignored.
    pub fn set_voiceprint(&mut self, embedding: Vec<f32>) {
        let Some(vp) = normalized(&embedding) else {
            log::warn!("ignoring an empty or invalid voiceprint");
            return;
        };
        if self.dim.is_some_and(|d| d != vp.len()) {
            log::warn!(
                "voiceprint has {} dims but embeddings have {:?}; ignoring it",
                vp.len(),
                self.dim
            );
            return;
        }
        self.dim = Some(vp.len());
        if let Some(me) = self.speakers.iter_mut().find(|s| s.is_me) {
            me.centroid = vp;
        } else {
            self.speakers.insert(
                0,
                Speaker {
                    label: ME.to_string(),
                    number: 0,
                    is_me: true,
                    sum: vec![0.0; vp.len()],
                    weight: 0.0,
                    centroid: vp,
                },
            );
        }
    }

    /// Whether a voiceprint is set.
    pub fn has_voiceprint(&self) -> bool {
        self.speakers.iter().any(|s| s.is_me)
    }

    /// Label a finished segment. `embedding: None` means the segment was too short to embed
    /// (or embedding failed); it then inherits the label of the previous segment if that one
    /// ended within [`ClusterConfig::context_ms`] before `start_ms`, else gets `None`.
    /// Call in chronological order.
    pub fn assign(
        &mut self,
        embedding: Option<&[f32]>,
        start_ms: u64,
        end_ms: u64,
    ) -> Option<String> {
        let duration_ms = end_ms.saturating_sub(start_ms);
        let emb = embedding.and_then(|e| self.accept_embedding(e));
        let label = emb
            .and_then(|e| self.assign_embedding(&e, duration_ms))
            .or_else(|| self.context_label(start_ms));
        self.last = Some(Last {
            end_ms: end_ms.max(start_ms),
            label: label.clone(),
        });
        label
    }

    /// The label that `label` currently stands for: itself, or the label of the speaker it
    /// was merged into.
    pub fn resolve<'a>(&'a self, label: &'a str) -> &'a str {
        self.aliases.get(label).map(String::as_str).unwrap_or(label)
    }

    /// Retired labels and the labels they were merged into.
    pub fn merges(&self) -> impl Iterator<Item = (&str, &str)> {
        self.aliases.iter().map(|(a, b)| (a.as_str(), b.as_str()))
    }

    /// Number of distinct speakers currently known (excluding the user).
    pub fn num_speakers(&self) -> usize {
        self.speakers.iter().filter(|s| !s.is_me).count()
    }

    /// The source prefix this clusterer labels with.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// The configuration in use.
    pub fn config(&self) -> &ClusterConfig {
        &self.cfg
    }

    fn accept_embedding(&mut self, e: &[f32]) -> Option<Vec<f32>> {
        if let Some(d) = self.dim {
            if e.len() != d {
                log::warn!("embedding has {} dims, expected {d}; ignoring it", e.len());
                return None;
            }
        }
        let e = normalized(e)?;
        self.dim = Some(e.len());
        Some(e)
    }

    fn context_label(&self, start_ms: u64) -> Option<String> {
        let last = self.last.as_ref()?;
        if start_ms > last.end_ms.saturating_add(self.cfg.context_ms) {
            return None;
        }
        last.label.as_deref().map(|l| self.resolve(l).to_string())
    }

    fn assign_embedding(&mut self, e: &[f32], duration_ms: u64) -> Option<String> {
        let w = duration_weight(duration_ms);
        // Short segments give noisier embeddings that sit further from their speaker's
        // centroid; they join at a lower similarity.
        let relax = if duration_ms < self.cfg.short_segment_ms {
            self.cfg.short_segment_relax
        } else {
            0.0
        };
        // Best candidate by margin over its own threshold (the user has a separate one).
        let best = self
            .speakers
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let thr = if s.is_me {
                    self.cfg.voiceprint_threshold
                } else {
                    self.cfg.threshold
                };
                (i, dot(e, &s.centroid) - (thr - relax))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((i, margin)) = best {
            if margin >= 0.0 {
                return Some(self.add_to(i, e, w));
            }
        }
        if duration_ms < self.cfg.min_new_speaker_ms {
            return None;
        }
        if self.num_speakers() >= self.cfg.max_speakers.max(1) {
            // At the cap: the nearest known voice (never the user without a voiceprint match).
            let i = self
                .speakers
                .iter()
                .enumerate()
                .filter(|(_, s)| !s.is_me)
                .map(|(i, s)| (i, dot(e, &s.centroid)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(i, _)| i)?;
            return Some(self.add_to(i, e, w));
        }
        let number = self.next_number;
        self.next_number += 1;
        let label = format!("{}:{number}", self.prefix);
        self.speakers.push(Speaker {
            label: label.clone(),
            number,
            is_me: false,
            sum: e.iter().map(|x| x * w).collect(),
            weight: w,
            centroid: e.to_vec(),
        });
        Some(label)
    }

    /// Add a segment to speaker `i`, then merge speakers that became too similar.
    /// Returns the label to emit (the survivor's, if `i` got merged away).
    fn add_to(&mut self, i: usize, e: &[f32], w: f32) -> String {
        let s = &mut self.speakers[i];
        s.sum.iter_mut().zip(e).for_each(|(a, x)| *a += w * x);
        s.weight += w;
        if !s.is_me {
            let mut c = s.sum.clone();
            if l2_normalize(&mut c) {
                s.centroid = c;
            }
        }
        let label = s.label.clone();
        self.merge_similar(i);
        self.resolve(&label).to_string()
    }

    /// Merge speaker `i` with others while any pair involving it is similar enough.
    fn merge_similar(&mut self, mut i: usize) {
        loop {
            let si = &self.speakers[i];
            let candidate = self
                .speakers
                .iter()
                .enumerate()
                .filter(|&(j, sj)| j != i && !(sj.is_me && si.is_me))
                .map(|(j, sj)| {
                    // Joining the user is judged against the voiceprint threshold, since
                    // the user's centroid is the voiceprint itself.
                    let thr = if sj.is_me || si.is_me {
                        self.cfg.voiceprint_threshold.max(self.cfg.threshold)
                    } else {
                        self.cfg.merge_threshold
                    };
                    (j, dot(&si.centroid, &sj.centroid) - thr)
                })
                .filter(|&(_, m)| m >= 0.0)
                .max_by(|a, b| a.1.total_cmp(&b.1));
            let Some((j, _)) = candidate else { return };
            i = self.merge(i, j);
        }
    }

    /// Merge speakers `a` and `b`; returns the survivor's new index.
    fn merge(&mut self, a: usize, b: usize) -> usize {
        let (sa, sb) = (&self.speakers[a], &self.speakers[b]);
        let keep_a = if sa.is_me != sb.is_me {
            sa.is_me
        } else if sa.weight != sb.weight {
            sa.weight > sb.weight
        } else {
            sa.number < sb.number
        };
        let (keep, gone) = if keep_a { (a, b) } else { (b, a) };
        let removed = self.speakers.remove(gone);
        let keep = if gone < keep { keep - 1 } else { keep };
        let k = &mut self.speakers[keep];
        k.sum
            .iter_mut()
            .zip(&removed.sum)
            .for_each(|(x, y)| *x += y);
        k.weight += removed.weight;
        if !k.is_me {
            let mut c = k.sum.clone();
            if l2_normalize(&mut c) {
                k.centroid = c;
            }
        }
        let survivor = k.label.clone();
        log::debug!("speaker {} merged into {survivor}", removed.label);
        for v in self.aliases.values_mut() {
            if *v == removed.label {
                *v = survivor.clone();
            }
        }
        self.aliases.insert(removed.label, survivor);
        keep
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 16;

    /// Unit vector along axis `k`, slightly tilted towards axis `k+1` by `tilt`.
    fn voice(k: usize, tilt: f32) -> Vec<f32> {
        let mut v = vec![0.0; DIM];
        v[k % DIM] = 1.0;
        v[(k + 1) % DIM] = tilt;
        l2_normalize(&mut v);
        v
    }

    /// A vector with cosine `sim` to unit vector `a` (orthogonal part along axis `k`).
    fn blend(a: &[f32], k: usize, sim: f32) -> Vec<f32> {
        let mut o = vec![0.0; DIM];
        o[k] = 1.0;
        let d = dot(&o, a);
        o.iter_mut().zip(a).for_each(|(x, y)| *x -= d * y);
        l2_normalize(&mut o);
        let s = (1.0 - sim * sim).sqrt();
        a.iter().zip(&o).map(|(x, y)| sim * x + s * y).collect()
    }

    fn cfg() -> ClusterConfig {
        ClusterConfig {
            threshold: 0.5,
            min_embed_ms: 1000,
            merge_threshold: 0.7,
            voiceprint_threshold: 0.6,
            min_new_speaker_ms: 1500,
            short_segment_ms: 0,
            short_segment_relax: 0.0,
            context_ms: 2000,
            max_speakers: 40,
            recluster_threshold: 0.4,
            min_cluster_ms: 0,
            change_threshold: 0.4,
            min_piece_ms: 1000,
        }
    }

    #[test]
    fn same_voice_same_label_new_voice_new_label() {
        let mut c = OnlineClusterer::new("mic", cfg());
        assert_eq!(
            c.assign(Some(&voice(0, 0.1)), 0, 3000).as_deref(),
            Some("mic:1")
        );
        assert_eq!(
            c.assign(Some(&voice(0, 0.2)), 4000, 7000).as_deref(),
            Some("mic:1")
        );
        assert_eq!(
            c.assign(Some(&voice(2, 0.1)), 8000, 11000).as_deref(),
            Some("mic:2")
        );
        assert_eq!(
            c.assign(Some(&voice(0, 0.0)), 12000, 15000).as_deref(),
            Some("mic:1")
        );
        assert_eq!(
            c.assign(Some(&voice(4, 0.0)), 16000, 19000).as_deref(),
            Some("mic:3")
        );
        assert_eq!(
            c.assign(Some(&voice(2, 0.3)), 20000, 23000).as_deref(),
            Some("mic:2")
        );
        assert_eq!(c.num_speakers(), 3);
    }

    #[test]
    fn embedding_scale_does_not_matter() {
        let mut c = OnlineClusterer::new("sys", cfg());
        let a: Vec<f32> = voice(0, 0.1).iter().map(|x| x * 37.0).collect();
        assert_eq!(c.assign(Some(&a), 0, 3000).as_deref(), Some("sys:1"));
        assert_eq!(
            c.assign(Some(&voice(0, 0.1)), 3000, 6000).as_deref(),
            Some("sys:1")
        );
    }

    #[test]
    fn voiceprint_gives_me() {
        let mut c = OnlineClusterer::new("mic", cfg());
        c.set_voiceprint(voice(5, 0.0));
        assert!(c.has_voiceprint());
        assert_eq!(
            c.assign(Some(&voice(0, 0.0)), 0, 3000).as_deref(),
            Some("mic:1")
        );
        assert_eq!(
            c.assign(Some(&blend(&voice(5, 0.0), 9, 0.8)), 3000, 6000)
                .as_deref(),
            Some("me")
        );
        // Below the voiceprint threshold but above the cluster threshold: not "me".
        assert_eq!(
            c.assign(Some(&blend(&voice(5, 0.0), 9, 0.55)), 6000, 9000)
                .as_deref(),
            Some("mic:2")
        );
        // The user never counts as a speaker and never gets a number.
        assert_eq!(c.num_speakers(), 2);
    }

    #[test]
    fn voiceprint_and_cluster_compete_by_margin() {
        let mut c = OnlineClusterer::new("mic", cfg());
        let vp = voice(5, 0.0);
        c.set_voiceprint(vp.clone());
        // Someone who sounds a bit like the user (0.55 < 0.6): their own speaker.
        assert_eq!(
            c.assign(Some(&blend(&vp, 7, 0.55)), 0, 3000).as_deref(),
            Some("mic:1")
        );
        // Barely passes the voiceprint (margin 0.02) but is ~1.0 to mic:1 (margin ~0.5).
        let x = blend(&vp, 7, 0.62);
        assert!(dot(&x, &vp) >= 0.6);
        assert_eq!(c.assign(Some(&x), 3000, 6000).as_deref(), Some("mic:1"));
        // A clear match of the voiceprint is still the user.
        assert_eq!(
            c.assign(Some(&blend(&vp, 9, 0.9)), 6000, 9000).as_deref(),
            Some("me")
        );
    }

    #[test]
    fn short_segments_use_recent_context() {
        let mut c = OnlineClusterer::new("mic", cfg());
        assert_eq!(c.assign(None, 0, 500), None, "nothing before");
        assert_eq!(
            c.assign(Some(&voice(0, 0.0)), 1000, 4000).as_deref(),
            Some("mic:1")
        );
        assert_eq!(
            c.assign(None, 5500, 6000).as_deref(),
            Some("mic:1"),
            "1.5 s gap"
        );
        // Chains from the short segment itself (ended at 6000).
        assert_eq!(c.assign(None, 7900, 8300).as_deref(), Some("mic:1"));
        assert_eq!(c.assign(None, 10_400, 10_800), None, "2.1 s gap");
        assert_eq!(
            c.assign(None, 11_000, 11_500),
            None,
            "previous was unlabeled"
        );
        // Overlapping start counts as recent.
        assert_eq!(
            c.assign(Some(&voice(3, 0.0)), 12_000, 15_000).as_deref(),
            Some("mic:2")
        );
        assert_eq!(c.assign(None, 14_000, 14_500).as_deref(), Some("mic:2"));
    }

    #[test]
    fn short_segments_never_open_speakers() {
        let mut c = OnlineClusterer::new("sys", cfg());
        assert_eq!(
            c.assign(Some(&voice(0, 0.0)), 0, 3000).as_deref(),
            Some("sys:1")
        );
        // 1.2 s of an unknown voice right after: no new speaker, falls back to context.
        assert_eq!(
            c.assign(Some(&voice(6, 0.0)), 3500, 4700).as_deref(),
            Some("sys:1")
        );
        // Same, far from anything: unlabeled.
        assert_eq!(c.assign(Some(&voice(6, 0.0)), 20_000, 21_200), None);
        // But a short segment of a known voice is recognized.
        assert_eq!(
            c.assign(Some(&voice(0, 0.1)), 40_000, 41_000).as_deref(),
            Some("sys:1")
        );
        // A long one of the unknown voice opens a speaker.
        assert_eq!(
            c.assign(Some(&voice(6, 0.0)), 50_000, 52_000).as_deref(),
            Some("sys:2")
        );
        assert_eq!(c.num_speakers(), 2);
    }

    #[test]
    fn merge_keeps_emitted_labels_and_redirects_later_ones() {
        let mut c = OnlineClusterer::new("mic", cfg());
        let a = voice(0, 0.0);
        // b is similar to a (0.45 < threshold) so it opens its own speaker.
        let b = blend(&a, 3, 0.45);
        assert_eq!(c.assign(Some(&a), 0, 5000).as_deref(), Some("mic:1"));
        assert_eq!(c.assign(Some(&b), 6000, 8000).as_deref(), Some("mic:2"));
        // Midpoint segments pull mic:2 towards mic:1 until their centroids pass 0.7.
        let mid = {
            let mut v: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
            l2_normalize(&mut v);
            v
        };
        let mut labels = Vec::new();
        for k in 0..6u64 {
            labels.push(
                c.assign(Some(&mid), 10_000 + k * 4000, 13_000 + k * 4000)
                    .unwrap(),
            );
        }
        assert_eq!(c.num_speakers(), 1, "labels: {labels:?}");
        // mic:1 had more speech, so it survives; mic:2 now resolves to it.
        assert_eq!(c.resolve("mic:2"), "mic:1");
        assert_eq!(c.merges().collect::<Vec<_>>(), vec![("mic:2", "mic:1")]);
        assert_eq!(labels.last().map(String::as_str), Some("mic:1"));
        // Later segments of either voice get the survivor, and context resolves too.
        assert_eq!(c.assign(Some(&b), 40_000, 43_000).as_deref(), Some("mic:1"));
        assert_eq!(c.assign(None, 43_500, 44_000).as_deref(), Some("mic:1"));
        // Numbers are never reused.
        assert_eq!(
            c.assign(Some(&voice(8, 0.0)), 50_000, 53_000).as_deref(),
            Some("mic:3")
        );
    }

    #[test]
    fn cluster_close_to_voiceprint_merges_into_me() {
        let mut c = OnlineClusterer::new("mic", cfg());
        let vp = voice(5, 0.0);
        c.set_voiceprint(vp.clone());
        // Enrolled in a quiet room; the first meeting segments are a bit off (0.55 < 0.6).
        let off = blend(&vp, 9, 0.55);
        assert_eq!(c.assign(Some(&off), 0, 3000).as_deref(), Some("mic:1"));
        // A clearer one matches the voiceprint directly.
        let near = blend(&vp, 11, 0.75);
        assert_eq!(c.assign(Some(&near), 4000, 7000).as_deref(), Some("me"));
        // Segments between the two drag mic:1 towards the voiceprint; once its centroid
        // passes the voiceprint threshold it merges into "me".
        let between = {
            let mut v: Vec<f32> = off.iter().zip(&near).map(|(x, y)| x + y).collect();
            l2_normalize(&mut v);
            v
        };
        let mut last = None;
        for k in 0..4u64 {
            last = c.assign(Some(&between), 10_000 + k * 4000, 13_000 + k * 4000);
        }
        assert_eq!(last.as_deref(), Some("me"));
        assert_eq!(c.resolve("mic:1"), "me");
        assert_eq!(c.num_speakers(), 0);
    }

    #[test]
    fn speaker_cap_reuses_nearest() {
        let mut c = OnlineClusterer::new(
            "sys",
            ClusterConfig {
                max_speakers: 3,
                ..cfg()
            },
        );
        for k in 0..3 {
            let s = k as u64 * 5000;
            assert_eq!(
                c.assign(Some(&voice(k * 3, 0.0)), s, s + 3000),
                Some(format!("sys:{}", k + 1))
            );
        }
        let near_2 = blend(&voice(3, 0.0), 14, 0.3);
        assert_eq!(
            c.assign(Some(&near_2), 20_000, 23_000).as_deref(),
            Some("sys:2")
        );
        assert_eq!(c.num_speakers(), 3);
    }

    #[test]
    fn bad_embeddings_fall_back_to_context() {
        let mut c = OnlineClusterer::new("mic", cfg());
        assert_eq!(
            c.assign(Some(&voice(0, 0.0)), 0, 3000).as_deref(),
            Some("mic:1")
        );
        assert_eq!(
            c.assign(Some(&[0.0; DIM]), 3500, 6000).as_deref(),
            Some("mic:1")
        );
        assert_eq!(
            c.assign(Some(&[f32::NAN; DIM]), 6500, 9000).as_deref(),
            Some("mic:1")
        );
        assert_eq!(
            c.assign(Some(&[1.0; 3]), 9500, 12000).as_deref(),
            Some("mic:1"),
            "wrong dim"
        );
        assert_eq!(c.num_speakers(), 1);
    }

    #[test]
    fn helpers() {
        assert!((cosine(&[1.0, 0.0], &[2.0, 0.0]) - 1.0).abs() < 1e-6);
        assert_eq!(cosine(&[1.0, 0.0], &[0.0, 0.0]), 0.0);
        assert_eq!(cosine(&[1.0], &[1.0, 0.0]), 0.0);
        let mut v = vec![3.0, 4.0];
        assert!(l2_normalize(&mut v));
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!(!l2_normalize(&mut [0.0, 0.0]));
        assert_eq!(label_number("mic", "mic:12"), Some(12));
        assert_eq!(label_number("mic", "sys:12"), None);
        assert_eq!(label_number("mic", "me"), None);
    }
}
