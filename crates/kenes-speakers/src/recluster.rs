//! End-of-meeting re-clustering.

use std::collections::{BTreeMap, HashMap};

use crate::cluster::{dot, duration_weight, label_number, normalized, ClusterConfig, ME};
use crate::metrics::max_weight_matching;

/// Most segments per cluster that go through the O(n²) split check (the longest are kept).
/// 4000 segments ≈ 32 MB of similarities.
const MAX_AHC_ITEMS: usize = 4000;

/// A cluster smaller than [`ClusterConfig::min_cluster_ms`] is absorbed by its nearest
/// neighbour if their similarity is at least `recluster_threshold - ABSORB_MARGIN`.
const ABSORB_MARGIN: f32 = 0.1;

/// A cluster is split only where its parts are below `recluster_threshold - SPLIT_MARGIN`:
/// splitting one real person in two is worse than leaving two people merged.
const SPLIT_MARGIN: f32 = 0.05;

/// A segment moves to another cluster only if it is this much more similar to it.
const MOVE_MARGIN: f32 = 0.1;

/// One final segment as stored by the caller.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterItem {
    pub segment_id: String,
    /// `"mic"` or `"sys"`: items are only clustered with items of the same prefix.
    pub prefix: String,
    /// Speaker embedding from [`crate::Embedder::embed`]; empty when the segment was too
    /// short to embed.
    pub embedding: Vec<f32>,
    pub duration_ms: u64,
    /// The label the segment got online (what the user saw and may have renamed).
    pub online_label: Option<String>,
}

/// Re-cluster a whole meeting offline, per prefix, and return `(segment_id, label)` for every
/// item in input order (pass items in chronological order).
///
/// Conservative by design: it starts from the online speakers and only changes them where
/// the whole meeting's evidence is clear, so it is not worse than the online labels on
/// average (see `EVAL.md`).
///
/// 1. One group per online label. Groups whose average-linkage similarity (cosine,
///    duration-weighted) is at least [`ClusterConfig::recluster_threshold`] are merged (and
///    beyond that while there are more than [`ClusterConfig::max_speakers`]).
/// 2. A cluster whose segments of at least [`ClusterConfig::short_segment_ms`] fall apart
///    into voices less than `recluster_threshold - 0.05` similar, each with at least
///    [`ClusterConfig::min_cluster_ms`] of speech, is split.
/// 3. A segment moves to another cluster if it is at least 0.1 more similar to it than to
///    the rest of its own (the online label of a short segment is often a guess).
/// 4. Clusters with less than [`ClusterConfig::min_cluster_ms`] of speech join their nearest
///    cluster if it is at least `recluster_threshold - 0.1` similar. Embedded segments
///    without an online label join the nearest cluster.
/// 5. With a `voiceprint`, the cluster most similar to it becomes `"me"` if its centroid
///    reaches [`ClusterConfig::voiceprint_threshold`].
/// 6. The other clusters take over online labels by maximum total overlap (duration of
///    segments that had that label online), one label per cluster, so names the user gave
///    stay with the person and as little as possible changes. Clusters left without a label
///    get fresh numbers above every number used online. Without a voiceprint, an online
///    `"me"` is treated like any label.
/// 7. Segments without an embedding follow their online label: to the cluster that got most
///    of that label's speech, else they keep it.
///
/// Deterministic: the same input gives the same output.
pub fn recluster(
    items: &[ClusterItem],
    voiceprint: Option<&[f32]>,
    cfg: &ClusterConfig,
) -> Vec<(String, Option<String>)> {
    let mut out: Vec<Option<String>> = items.iter().map(|it| it.online_label.clone()).collect();
    let mut prefixes: Vec<&str> = Vec::new();
    for it in items {
        if !prefixes.contains(&it.prefix.as_str()) {
            prefixes.push(&it.prefix);
        }
    }
    let vp = voiceprint.and_then(normalized);
    for prefix in prefixes {
        let idx: Vec<usize> = (0..items.len())
            .filter(|&i| items[i].prefix == prefix)
            .collect();
        recluster_prefix(prefix, items, &idx, vp.as_deref(), cfg, &mut out);
    }
    items
        .iter()
        .zip(out)
        .map(|(it, l)| (it.segment_id.clone(), l))
        .collect()
}

fn recluster_prefix(
    prefix: &str,
    items: &[ClusterItem],
    idx: &[usize],
    vp: Option<&[f32]>,
    cfg: &ClusterConfig,
    out: &mut [Option<String>],
) {
    // Embedded items (the most common dimension wins; odd ones are treated as unembedded).
    let mut dims: BTreeMap<usize, usize> = BTreeMap::new();
    for &i in idx {
        if !items[i].embedding.is_empty() {
            *dims.entry(items[i].embedding.len()).or_default() += 1;
        }
    }
    let Some((&dim, _)) = dims
        .iter()
        .max_by_key(|&(d, n)| (*n, std::cmp::Reverse(*d)))
    else {
        return;
    };
    let vp = vp.filter(|v| v.len() == dim);
    let mut emb_items: Vec<usize> = Vec::new(); // item index
    let mut vecs: Vec<Vec<f32>> = Vec::new();
    for &i in idx {
        if items[i].embedding.len() == dim {
            if let Some(v) = normalized(&items[i].embedding) {
                emb_items.push(i);
                vecs.push(v);
            }
        }
    }
    if vecs.is_empty() {
        return;
    }
    let weights: Vec<f32> = emb_items
        .iter()
        .map(|&i| duration_weight(items[i].duration_ms))
        .collect();

    // Long segments are trusted to shape clusters; short ones just follow.
    let anchor_ms = cfg.short_segment_ms.max(cfg.min_new_speaker_ms);
    let is_anchor = |k: usize| items[emb_items[k]].duration_ms >= anchor_ms;
    const NONE: usize = usize::MAX;

    // 1. Start from the online speakers: one group per online label. (If nothing was
    //    labeled online, every long segment starts alone, which makes this a plain AHC.)
    let mut cluster_of = vec![NONE; vecs.len()];
    let mut group_ix: HashMap<&str, usize> = HashMap::new();
    for (k, c) in cluster_of.iter_mut().enumerate() {
        if let Some(l) = items[emb_items[k]].online_label.as_deref() {
            let n = group_ix.len();
            *c = *group_ix.entry(l).or_insert(n);
        }
    }
    let mut n_ids = group_ix.len();
    // Long segments without an online label start alone (all of them, if nothing was
    // labeled online: then this is a plain AHC).
    let mut singles: Vec<usize> = (0..vecs.len())
        .filter(|&k| cluster_of[k] == NONE && (is_anchor(k) || n_ids == 0))
        .collect();
    singles.sort_by_key(|&k| (std::cmp::Reverse(items[emb_items[k]].duration_ms), k));
    singles.truncate(MAX_AHC_ITEMS);
    singles.sort_unstable();
    for k in singles {
        cluster_of[k] = n_ids;
        n_ids += 1;
    }

    // 2. Merge groups that are one voice: average linkage between two groups is the dot
    //    product of their duration-weighted mean embeddings.
    {
        let mut means = vec![vec![0.0f32; dim]; n_ids];
        let mut gw = vec![0.0f32; n_ids];
        for (k, &c) in cluster_of.iter().enumerate() {
            if c != NONE {
                means[c]
                    .iter_mut()
                    .zip(&vecs[k])
                    .for_each(|(a, x)| *a += weights[k] * x);
                gw[c] += weights[k];
            }
        }
        for (m, &w) in means.iter_mut().zip(&gw) {
            m.iter_mut().for_each(|a| *a /= w.max(1e-6));
        }
        let refs: Vec<&[f32]> = means.iter().map(Vec::as_slice).collect();
        let merged = ahc(&refs, &gw, cfg.recluster_threshold, cfg.max_speakers.max(1));
        for c in cluster_of.iter_mut().filter(|c| **c != NONE) {
            *c = merged[*c];
        }
        n_ids = merged.iter().max().map_or(0, |m| m + 1);
    }

    // 3. Split a cluster whose long segments fall apart into clearly different voices
    //    (below `recluster_threshold - SPLIT_MARGIN`), each with at least `min_cluster_ms`
    //    of speech: two people who shared one online label.
    let merged_ids = n_ids;
    for c in 0..merged_ids {
        let members: Vec<usize> = (0..vecs.len()).filter(|&k| cluster_of[k] == c).collect();
        let total: u64 = members
            .iter()
            .map(|&k| items[emb_items[k]].duration_ms)
            .sum();
        let mut anchors: Vec<usize> = members.iter().copied().filter(|&k| is_anchor(k)).collect();
        if anchors.len() < 2 || total < 2 * cfg.min_cluster_ms {
            continue;
        }
        anchors.sort_by_key(|&k| (std::cmp::Reverse(items[emb_items[k]].duration_ms), k));
        anchors.truncate(MAX_AHC_ITEMS);
        let refs: Vec<&[f32]> = anchors.iter().map(|&k| vecs[k].as_slice()).collect();
        let ws: Vec<f32> = anchors.iter().map(|&k| weights[k]).collect();
        let sub = ahc(
            &refs,
            &ws,
            cfg.recluster_threshold - SPLIT_MARGIN,
            cfg.max_speakers.max(1),
        );
        let ns = sub.iter().max().map_or(0, |m| m + 1);
        let mut dur = vec![0u64; ns];
        let mut sums = vec![vec![0.0f32; dim]; ns];
        for (&k, &j) in anchors.iter().zip(&sub) {
            dur[j] += items[emb_items[k]].duration_ms;
            sums[j]
                .iter_mut()
                .zip(&vecs[k])
                .for_each(|(a, x)| *a += weights[k] * x);
        }
        let parts: Vec<usize> = (0..ns)
            .filter(|&j| dur[j] >= cfg.min_cluster_ms.max(1))
            .collect();
        if parts.len() < 2 {
            continue;
        }
        let cents: Vec<Vec<f32>> = parts
            .iter()
            .map(|&j| normalized(&sums[j]).unwrap_or_else(|| vec![0.0; dim]))
            .collect();
        let biggest = (0..parts.len())
            .max_by_key(|&i| (dur[parts[i]], std::cmp::Reverse(i)))
            .unwrap_or(0);
        let ids: Vec<usize> = (0..parts.len())
            .map(|i| {
                if i == biggest {
                    c
                } else {
                    n_ids += 1;
                    n_ids - 1
                }
            })
            .collect();
        for &k in &members {
            let best = (0..parts.len())
                .max_by(|&a, &b| {
                    dot(&vecs[k], &cents[a])
                        .total_cmp(&dot(&vecs[k], &cents[b]))
                        .then(b.cmp(&a))
                })
                .unwrap_or(0);
            cluster_of[k] = ids[best];
        }
    }

    // Duration-weighted, normalized centroid per id (zero vector for unused ids).
    let sums_of = |cluster_of: &[usize]| -> Vec<Vec<f32>> {
        let mut sums = vec![vec![0.0f32; dim]; n_ids];
        for (k, &c) in cluster_of.iter().enumerate() {
            if c != NONE {
                sums[c]
                    .iter_mut()
                    .zip(&vecs[k])
                    .for_each(|(s, x)| *s += weights[k] * x);
            }
        }
        sums
    };
    let centroids = |cluster_of: &[usize]| -> Vec<Vec<f32>> {
        sums_of(cluster_of)
            .into_iter()
            .map(|s| normalized(&s).unwrap_or_else(|| vec![0.0; dim]))
            .collect()
    };
    let live_ids = |cluster_of: &[usize]| -> Vec<usize> {
        let mut v: Vec<usize> = cluster_of.iter().copied().filter(|&c| c != NONE).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let nearest = |v: &[f32], cents: &[Vec<f32>], live: &[usize], skip: usize| {
        live.iter()
            .filter(|&&c| c != skip)
            .map(|&c| (c, dot(v, &cents[c])))
            .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
    };

    // 4. A segment moves to another cluster only if it is clearly closer to it than to the
    //    rest of its own (a segment alone counts as `recluster_threshold` to itself).
    for _ in 0..3 {
        let sums = sums_of(&cluster_of);
        let cents = centroids(&cluster_of);
        let live = live_ids(&cluster_of);
        let mut moved = false;
        for k in 0..vecs.len() {
            let own = cluster_of[k];
            if own == NONE {
                continue;
            }
            let rest: Vec<f32> = sums[own]
                .iter()
                .zip(&vecs[k])
                .map(|(a, x)| a - weights[k] * x)
                .collect();
            let s_own = normalized(&rest).map_or(cfg.recluster_threshold, |r| dot(&vecs[k], &r));
            if let Some((c, s)) = nearest(&vecs[k], &cents, &live, own) {
                if s - s_own >= MOVE_MARGIN {
                    cluster_of[k] = c;
                    moved = true;
                }
            }
        }
        if !moved {
            break;
        }
    }

    // 5. A cluster with less than `min_cluster_ms` of speech is more likely a stray piece of
    //    a known voice (overlap, laughter, a far-field blip) than a new person: it joins its
    //    nearest cluster if that one is within ABSORB_MARGIN of the threshold.
    loop {
        let mut size = vec![0u64; n_ids];
        for (k, &c) in cluster_of.iter().enumerate() {
            if c != NONE {
                size[c] += items[emb_items[k]].duration_ms;
            }
        }
        let live = live_ids(&cluster_of);
        let mut small: Vec<usize> = live
            .iter()
            .copied()
            .filter(|&c| size[c] < cfg.min_cluster_ms)
            .collect();
        small.sort_by_key(|&c| (size[c], c));
        let cents = centroids(&cluster_of);
        let absorbed = small.into_iter().find_map(|sc| {
            nearest(&cents[sc], &cents, &live, sc)
                .filter(|&(_, sim)| sim >= cfg.recluster_threshold - ABSORB_MARGIN)
                .map(|(t, _)| (sc, t))
        });
        let Some((from, to)) = absorbed else { break };
        cluster_of
            .iter_mut()
            .filter(|c| **c == from)
            .for_each(|c| *c = to);
    }

    // 6. Embedded segments without an online label join the nearest cluster.
    let cents = centroids(&cluster_of);
    let live = live_ids(&cluster_of);
    for k in 0..vecs.len() {
        if cluster_of[k] == NONE {
            cluster_of[k] = nearest(&vecs[k], &cents, &live, NONE).map_or(0, |(c, _)| c);
        }
    }

    // Compact ids to 0..n_clusters.
    let mut remap: HashMap<usize, usize> = HashMap::new();
    for c in cluster_of.iter_mut() {
        let k = remap.len();
        *c = *remap.entry(*c).or_insert(k);
    }
    let n_clusters = remap.len();
    let final_centroids: Vec<Vec<f32>> = {
        let mut sums = vec![vec![0.0f32; dim]; n_clusters];
        for (k, &c) in cluster_of.iter().enumerate() {
            sums[c]
                .iter_mut()
                .zip(&vecs[k])
                .for_each(|(s, x)| *s += weights[k] * x);
        }
        sums.into_iter()
            .map(|s| normalized(&s).unwrap_or_else(|| vec![0.0; dim]))
            .collect()
    };

    // Cluster order: by first appearance, for stable fresh numbering.
    let mut first_seen = vec![usize::MAX; n_clusters];
    for (k, &c) in cluster_of.iter().enumerate() {
        first_seen[c] = first_seen[c].min(emb_items[k]);
    }

    // 2. The user.
    let me_cluster = vp.and_then(|vp| {
        (0..n_clusters)
            .map(|c| (c, dot(&final_centroids[c], vp)))
            .filter(|&(_, s)| s >= cfg.voiceprint_threshold)
            .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
            .map(|(c, _)| c)
    });

    // 3. Online labels → clusters by maximum total overlap.
    let mut label_names: Vec<String> = Vec::new();
    let mut label_ix: HashMap<String, usize> = HashMap::new();
    for &i in idx {
        if let Some(l) = &items[i].online_label {
            if vp.is_some() && l == ME {
                continue;
            }
            if !label_ix.contains_key(l) {
                label_ix.insert(l.clone(), label_names.len());
                label_names.push(l.clone());
            }
        }
    }
    let mut overlap = vec![vec![0.0f64; label_names.len()]; n_clusters];
    for (k, &c) in cluster_of.iter().enumerate() {
        if let Some(&l) = items[emb_items[k]]
            .online_label
            .as_ref()
            .and_then(|l| label_ix.get(l))
        {
            overlap[c][l] += items[emb_items[k]].duration_ms as f64;
        }
    }
    // Ties go to the cluster that appeared first: a bonus far below 1 ms in total.
    let mut rank: Vec<usize> = (0..n_clusters).collect();
    rank.sort_by_key(|&c| first_seen[c]);
    for (r, &c) in rank.iter().enumerate() {
        let bonus = 0.5 * (n_clusters - r) as f64 / (n_clusters * n_clusters) as f64;
        overlap[c]
            .iter_mut()
            .filter(|x| **x > 0.0)
            .for_each(|x| *x += bonus);
    }
    let mut cluster_label: Vec<Option<String>> = vec![None; n_clusters];
    if let Some(c) = me_cluster {
        cluster_label[c] = Some(ME.to_string());
        overlap[c].iter_mut().for_each(|x| *x = 0.0);
    }
    let rows: Vec<usize> = (0..n_clusters).filter(|&c| Some(c) != me_cluster).collect();
    let matching =
        max_weight_matching(&rows.iter().map(|&c| overlap[c].clone()).collect::<Vec<_>>());
    for (r, m) in rows.iter().zip(matching) {
        if let Some(l) = m.filter(|&l| overlap[*r][l] > 0.0) {
            cluster_label[*r] = Some(label_names[l].clone());
        }
    }
    let next = idx
        .iter()
        .filter_map(|&i| items[i].online_label.as_deref())
        .filter_map(|l| label_number(prefix, l))
        .max()
        .unwrap_or(0)
        + 1;
    let mut fresh: Vec<usize> = (0..n_clusters)
        .filter(|&c| cluster_label[c].is_none())
        .collect();
    fresh.sort_by_key(|&c| first_seen[c]);
    for (i, c) in fresh.into_iter().enumerate() {
        cluster_label[c] = Some(format!("{prefix}:{}", next + i as u32));
    }

    // 4. Unembedded items follow their online label to where most of its speech went.
    let mut label_votes: HashMap<&str, HashMap<usize, u64>> = HashMap::new();
    for (k, &c) in cluster_of.iter().enumerate() {
        let it = &items[emb_items[k]];
        if let Some(l) = it.online_label.as_deref() {
            *label_votes.entry(l).or_default().entry(c).or_default() += it.duration_ms.max(1);
        }
    }
    let follow: HashMap<&str, usize> = label_votes
        .iter()
        .map(|(&l, votes)| {
            let (&c, _) = votes
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .expect("non-empty");
            (l, c)
        })
        .collect();

    let mut is_embedded = vec![false; items.len()];
    for (k, &i) in emb_items.iter().enumerate() {
        out[i] = cluster_label[cluster_of[k]].clone();
        is_embedded[i] = true;
    }
    for &i in idx {
        if !is_embedded[i] {
            if let Some(&c) = items[i].online_label.as_deref().and_then(|l| follow.get(l)) {
                out[i] = cluster_label[c].clone();
            }
        }
    }
}

/// Weighted average-linkage agglomerative clustering of unit vectors on cosine similarity.
/// Merges while the best pair's average similarity is at least `threshold`, and beyond that
/// while there are more than `max_clusters` clusters. Returns a cluster id (`0..k`, numbered
/// by first member) per point.
///
/// Uses the nearest-neighbor-chain algorithm (O(n²) time, n²/2 floats), which builds the
/// full dendrogram; average linkage is reducible, so replaying its merges in order of
/// decreasing similarity gives the same clusters as the greedy algorithm.
pub(crate) fn ahc(
    points: &[&[f32]],
    weights: &[f32],
    threshold: f32,
    max_clusters: usize,
) -> Vec<usize> {
    let n = points.len();
    if n <= 1 {
        return vec![0; n];
    }
    // Condensed similarity matrix, s(i, j) for i < j.
    let at = |i: usize, j: usize| -> usize {
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        i * (2 * n - i - 1) / 2 + (j - i - 1)
    };
    let mut sim = vec![0.0f32; n * (n - 1) / 2];
    for i in 0..n {
        for j in i + 1..n {
            sim[at(i, j)] = dot(points[i], points[j]);
        }
    }
    let mut w: Vec<f64> = weights.iter().map(|&x| x.max(1e-3) as f64).collect();
    let mut active = vec![true; n];
    let mut merges: Vec<(f32, usize, usize)> = Vec::with_capacity(n - 1);
    let mut chain: Vec<usize> = Vec::new();
    let mut remaining = n;
    while remaining > 1 {
        if chain.is_empty() {
            chain.push(active.iter().position(|&a| a).expect("active cluster"));
        }
        loop {
            let a = *chain.last().expect("chain");
            let prev = chain.len().checked_sub(2).map(|k| chain[k]);
            // Most similar active cluster; prefer the previous chain element on ties so the
            // chain terminates.
            let mut best = usize::MAX;
            let mut best_s = f32::NEG_INFINITY;
            if let Some(p) = prev {
                best = p;
                best_s = sim[at(a, p)];
            }
            for c in 0..n {
                if c != a && active[c] {
                    let s = sim[at(a, c)];
                    if s > best_s {
                        best_s = s;
                        best = c;
                    }
                }
            }
            if Some(best) == prev {
                chain.pop();
                chain.pop();
                // Merge `best` into `a` (Lance–Williams for weighted average linkage).
                let (wa, wb) = (w[a], w[best]);
                for c in 0..n {
                    if active[c] && c != a && c != best {
                        let s =
                            (wa * sim[at(a, c)] as f64 + wb * sim[at(best, c)] as f64) / (wa + wb);
                        sim[at(a, c)] = s as f32;
                    }
                }
                w[a] = wa + wb;
                active[best] = false;
                remaining -= 1;
                merges.push((best_s, a, best));
                break;
            }
            chain.push(best);
        }
    }

    // Replay merges from most to least similar with union-find.
    merges.sort_by(|x, y| y.0.total_cmp(&x.0));
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    let mut clusters = n;
    for &(s, a, b) in &merges {
        if s < threshold && clusters <= max_clusters {
            break;
        }
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            parent[ra.max(rb)] = ra.min(rb);
            clusters -= 1;
        }
    }
    let mut id: HashMap<usize, usize> = HashMap::new();
    (0..n)
        .map(|i| {
            let r = find(&mut parent, i);
            let k = id.len();
            *id.entry(r).or_insert(k)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::l2_normalize;

    const DIM: usize = 16;

    fn voice(k: usize, jitter: f32, seed: u32) -> Vec<f32> {
        // Deterministic pseudo-noise so tests don't need a RNG crate.
        let mut v = vec![0.0; DIM];
        v[k % DIM] = 1.0;
        let mut x = seed.wrapping_mul(2654435761).wrapping_add(k as u32 * 97);
        for e in v.iter_mut() {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *e += jitter * ((x % 1000) as f32 / 1000.0 - 0.5);
        }
        l2_normalize(&mut v);
        v
    }

    fn item(id: usize, prefix: &str, v: Vec<f32>, ms: u64, online: Option<&str>) -> ClusterItem {
        ClusterItem {
            segment_id: format!("{prefix}-{id}"),
            prefix: prefix.to_string(),
            embedding: v,
            duration_ms: ms,
            online_label: online.map(str::to_string),
        }
    }

    fn cfg() -> ClusterConfig {
        ClusterConfig {
            recluster_threshold: 0.5,
            voiceprint_threshold: 0.6,
            min_new_speaker_ms: 1500,
            min_cluster_ms: 0,
            ..ClusterConfig::default()
        }
    }

    /// A unit vector with cosine `sim` to `a`, rotated towards axis `k`.
    fn towards(a: &[f32], k: usize, sim: f32) -> Vec<f32> {
        let mut o = vec![0.0; DIM];
        o[k] = 1.0;
        let d = dot(&o, a);
        o.iter_mut().zip(a).for_each(|(x, y)| *x -= d * y);
        l2_normalize(&mut o);
        let s = (1.0 - sim * sim).sqrt();
        a.iter().zip(&o).map(|(x, y)| sim * x + s * y).collect()
    }

    #[test]
    fn tiny_clusters_join_a_close_neighbour() {
        let a = voice(0, 0.0, 0);
        // A stray 2 s segment at 0.45 to A (below the 0.5 threshold, above 0.5 - 0.1),
        // and a real 2 s newcomer far from everyone.
        let items = vec![
            item(0, "mic", a.clone(), 6000, Some("mic:1")),
            item(1, "mic", towards(&a, 5, 0.45), 2000, Some("mic:2")),
            item(2, "mic", a.clone(), 6000, Some("mic:1")),
            item(3, "mic", voice(9, 0.0, 0), 2000, Some("mic:3")),
        ];
        let off = recluster(&items, None, &cfg());
        assert_eq!(
            labels(&off),
            vec![Some("mic:1"), Some("mic:2"), Some("mic:1"), Some("mic:3")]
        );
        let on = recluster(
            &items,
            None,
            &ClusterConfig {
                min_cluster_ms: 5000,
                ..cfg()
            },
        );
        assert_eq!(
            labels(&on),
            vec![Some("mic:1"), Some("mic:1"), Some("mic:1"), Some("mic:3")]
        );
    }

    fn labels(r: &[(String, Option<String>)]) -> Vec<Option<&str>> {
        r.iter().map(|(_, l)| l.as_deref()).collect()
    }

    #[test]
    fn ahc_separates_and_respects_threshold() {
        let pts: Vec<Vec<f32>> = (0..12).map(|i| voice(i % 3, 0.3, i as u32)).collect();
        let refs: Vec<&[f32]> = pts.iter().map(|v| v.as_slice()).collect();
        let w = vec![1.0; 12];
        let c = ahc(&refs, &w, 0.5, 40);
        for i in 0..12 {
            assert_eq!(c[i], c[i % 3], "point {i}: {c:?}");
        }
        assert_eq!(c.iter().max(), Some(&2));
        // Threshold above every similarity: nothing merges.
        let c = ahc(&refs, &w, 1.01, 40);
        assert_eq!(c.iter().max(), Some(&11));
        // ...unless the cap forces it.
        let c = ahc(&refs, &w, 1.01, 3);
        assert_eq!(c.iter().max(), Some(&2));
    }

    #[test]
    fn online_labels_carry_over_and_errors_get_fixed() {
        // Truth: A A B A B C. Online got segment 3 wrong (said mic:2) and split C off late.
        let v = [
            voice(0, 0.2, 1),
            voice(0, 0.2, 2),
            voice(4, 0.2, 3),
            voice(0, 0.2, 4),
            voice(4, 0.2, 5),
            voice(8, 0.2, 6),
        ];
        let online = [
            Some("mic:1"),
            Some("mic:1"),
            Some("mic:2"),
            Some("mic:2"),
            Some("mic:2"),
            Some("mic:3"),
        ];
        let items: Vec<ClusterItem> = (0..6)
            .map(|i| item(i, "mic", v[i].clone(), 3000, online[i]))
            .collect();
        let r = recluster(&items, None, &cfg());
        assert_eq!(r[0].0, "mic-0");
        assert_eq!(
            labels(&r),
            vec![
                Some("mic:1"),
                Some("mic:1"),
                Some("mic:2"),
                Some("mic:1"),
                Some("mic:2"),
                Some("mic:3")
            ]
        );
    }

    #[test]
    fn user_names_stay_with_the_majority_person() {
        // Online labels numbered in a different order than first appearance: B is "sys:7".
        let items = vec![
            item(0, "sys", voice(1, 0.2, 1), 4000, Some("sys:7")),
            item(1, "sys", voice(6, 0.2, 2), 4000, Some("sys:3")),
            item(2, "sys", voice(1, 0.2, 3), 4000, Some("sys:7")),
            item(3, "sys", voice(6, 0.2, 4), 2000, Some("sys:7")), // online mistake
            item(4, "sys", voice(6, 0.2, 5), 4000, Some("sys:3")),
        ];
        let r = recluster(&items, None, &cfg());
        assert_eq!(
            labels(&r),
            vec![
                Some("sys:7"),
                Some("sys:3"),
                Some("sys:7"),
                Some("sys:3"),
                Some("sys:3")
            ]
        );
    }

    #[test]
    fn matching_keeps_the_most_speech_unchanged() {
        // Online lumped two people into mic:2; person 0 also had 4 s as mic:5.
        let items = vec![
            item(0, "mic", voice(0, 0.2, 1), 4000, Some("mic:2")),
            item(1, "mic", voice(9, 0.2, 2), 3000, Some("mic:2")),
            item(2, "mic", voice(0, 0.2, 3), 4000, Some("mic:5")),
            item(3, "mic", voice(9, 0.2, 4), 3000, Some("mic:2")),
            item(4, "mic", voice(0, 0.2, 5), 4000, Some("mic:2")),
        ];
        let r = recluster(&items, None, &cfg());
        // person 0 → mic:5 (4 s kept) + person 9 → mic:2 (6 s kept) beats
        // person 0 → mic:2 (8 s) + person 9 → new label (0 s).
        assert_eq!(
            labels(&r),
            vec![
                Some("mic:5"),
                Some("mic:2"),
                Some("mic:5"),
                Some("mic:2"),
                Some("mic:5")
            ]
        );
    }

    #[test]
    fn new_clusters_get_fresh_numbers_above_all_used() {
        // Online lumped two people into mic:2; mic:5 appeared only on an unembedded segment.
        let items = vec![
            item(0, "mic", voice(0, 0.2, 1), 4000, Some("mic:2")),
            item(1, "mic", voice(9, 0.2, 2), 3000, Some("mic:2")),
            item(2, "mic", Vec::new(), 600, Some("mic:5")),
            item(3, "mic", voice(9, 0.2, 4), 3000, Some("mic:2")),
            item(4, "mic", voice(0, 0.2, 5), 4000, Some("mic:2")),
            item(5, "mic", voice(13, 0.2, 6), 4000, None),
        ];
        let r = recluster(&items, None, &cfg());
        assert_eq!(
            labels(&r),
            vec![
                Some("mic:2"),
                Some("mic:6"),
                Some("mic:5"),
                Some("mic:6"),
                Some("mic:2"),
                Some("mic:7")
            ]
        );
    }

    #[test]
    fn ties_go_to_the_earlier_cluster() {
        let items = vec![
            item(0, "sys", voice(0, 0.2, 1), 3000, Some("sys:1")),
            item(1, "sys", voice(5, 0.2, 2), 3000, Some("sys:1")),
        ];
        let r = recluster(&items, None, &cfg());
        assert_eq!(labels(&r), vec![Some("sys:1"), Some("sys:2")]);
    }

    #[test]
    fn voiceprint_makes_me_and_short_items_follow() {
        let vp = voice(3, 0.0, 0);
        let items = vec![
            item(0, "mic", voice(3, 0.2, 1), 5000, Some("mic:1")),
            item(1, "mic", voice(7, 0.2, 2), 5000, Some("mic:2")),
            item(2, "mic", Vec::new(), 600, Some("mic:1")), // too short to embed
            item(3, "mic", voice(3, 0.2, 3), 5000, Some("me")),
            item(4, "mic", voice(3, 0.2, 4), 1000, Some("mic:2")), // short, embedded, wrong online
            item(5, "mic", Vec::new(), 700, None),
            item(6, "mic", Vec::new(), 700, Some("mic:9")), // label with no embedded segments
        ];
        let r = recluster(&items, Some(&vp), &cfg());
        assert_eq!(
            labels(&r),
            vec![
                Some("me"),
                Some("mic:2"),
                Some("me"),
                Some("me"),
                Some("me"),
                None,
                Some("mic:9")
            ]
        );
        // Without a voiceprint the same person keeps an ordinary label.
        let r = recluster(&items, None, &cfg());
        assert_eq!(labels(&r)[0], labels(&r)[3]);
        assert_eq!(labels(&r)[1], Some("mic:2"));
    }

    #[test]
    fn voiceprint_below_threshold_is_not_me() {
        let vp = voice(3, 0.0, 0);
        let items = vec![
            item(0, "mic", voice(5, 0.2, 1), 5000, Some("mic:1")),
            item(1, "mic", voice(5, 0.2, 2), 5000, Some("me")),
        ];
        let r = recluster(&items, Some(&vp), &cfg());
        assert_eq!(labels(&r), vec![Some("mic:1"), Some("mic:1")]);
    }

    #[test]
    fn prefixes_are_clustered_separately() {
        let items = vec![
            item(0, "mic", voice(0, 0.2, 1), 3000, Some("mic:1")),
            item(1, "sys", voice(0, 0.2, 2), 3000, Some("sys:1")),
            item(2, "mic", voice(0, 0.2, 3), 3000, Some("mic:1")),
            item(3, "sys", voice(5, 0.2, 4), 3000, Some("sys:1")),
        ];
        let r = recluster(&items, None, &cfg());
        assert_eq!(
            labels(&r),
            vec![Some("mic:1"), Some("sys:1"), Some("mic:1"), Some("sys:2")]
        );
    }

    #[test]
    fn recluster_is_stable() {
        let mut items: Vec<ClusterItem> = (0..40)
            .map(|i| {
                let spk = [0, 4, 8, 12][i % 4];
                let ms = if i % 5 == 0 {
                    800
                } else {
                    2500 + (i as u64 * 131) % 5000
                };
                item(
                    i,
                    "sys",
                    voice(spk, 0.35, i as u32),
                    ms,
                    Some(["sys:1", "sys:2", "sys:3", "sys:4"][i % 4]),
                )
            })
            .collect();
        items[7].online_label = Some("sys:1".into()); // an online error
        let a = recluster(&items, None, &cfg());
        let b = recluster(&items, None, &cfg());
        assert_eq!(a, b, "deterministic");
        // Feeding the result back as online labels changes nothing.
        let again: Vec<ClusterItem> = items
            .iter()
            .zip(&a)
            .map(|(it, (_, l))| ClusterItem {
                online_label: l.clone(),
                ..it.clone()
            })
            .collect();
        assert_eq!(recluster(&again, None, &cfg()), a, "idempotent");
        assert_eq!(a[7].1.as_deref(), Some("sys:4"), "online error fixed");
    }

    #[test]
    fn nothing_embedded_keeps_online_labels() {
        let items = vec![
            item(0, "mic", Vec::new(), 500, Some("mic:1")),
            item(1, "mic", Vec::new(), 500, None),
        ];
        let r = recluster(&items, None, &cfg());
        assert_eq!(labels(&r), vec![Some("mic:1"), None]);
        assert!(recluster(&[], None, &cfg()).is_empty());
    }
}
