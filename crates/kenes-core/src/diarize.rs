//! Speaker labels for final segments.
//!
//! The session feeds every audio chunk into per-source ring buffers. When a
//! final segment arrives, its audio is cut from the ring, embedded and
//! assigned to a speaker online. When the meeting ends, all embeddings are
//! re-clustered once and only the labels that changed are reported.

use std::collections::VecDeque;
use std::path::Path;

use anyhow::{Context, Result};
use kenes_speakers::{recluster, ClusterConfig, ClusterItem, Embedder, OnlineClusterer};
use kenes_types::{AudioChunk, Segment, Source, SAMPLE_RATE};
use serde::{Deserialize, Serialize};

use crate::settings::MicMode;
use crate::store::{now_iso, Store};

pub const ME: &str = "me";
const VOICEPRINT_KEY: &str = "voiceprint";
/// Enough history for the longest segment (20 s) plus recognition delay.
const RING_MS: u64 = 60_000;
/// Gaps up to this long are filled with silence; longer ones restart the ring.
const MAX_FILL_MS: u64 = 5_000;
const SAMPLES_PER_MS: u64 = SAMPLE_RATE as u64 / 1000;

/// Sliding window of recent audio for one source, addressed by session time.
#[derive(Default)]
struct AudioRing {
    /// Session sample index of `samples[0]`.
    start: u64,
    samples: VecDeque<f32>,
}

impl AudioRing {
    fn end(&self) -> u64 {
        self.start + self.samples.len() as u64
    }

    fn push(&mut self, chunk: &AudioChunk) {
        let at = chunk.start_ms * SAMPLES_PER_MS;
        if self.samples.is_empty() || at > self.end() + MAX_FILL_MS * SAMPLES_PER_MS {
            self.samples.clear();
            self.start = at;
        } else if at > self.end() {
            // start_ms is rounded to whole ms, so tiny gaps are rounding, not lost audio.
            let gap = at - self.end();
            if gap > SAMPLES_PER_MS {
                self.samples.extend(std::iter::repeat_n(0.0, gap as usize));
            }
        }
        self.samples.extend(&chunk.samples);
        let cap = (RING_MS * SAMPLES_PER_MS) as usize;
        if self.samples.len() > cap {
            let excess = self.samples.len() - cap;
            self.samples.drain(..excess);
            self.start += excess as u64;
        }
    }

    fn slice(&self, start_ms: u64, end_ms: u64) -> Vec<f32> {
        let from = (start_ms * SAMPLES_PER_MS).clamp(self.start, self.end());
        let to = (end_ms * SAMPLES_PER_MS).clamp(from, self.end());
        self.samples.range((from - self.start) as usize..(to - self.start) as usize).copied().collect()
    }
}

pub struct Diarizer {
    embedder: Option<Embedder>,
    mic_mode: MicMode,
    cfg: ClusterConfig,
    voiceprint: Option<Vec<f32>>,
    mic: OnlineClusterer,
    sys: OnlineClusterer,
    mic_ring: AudioRing,
    sys_ring: AudioRing,
    items: Vec<ClusterItem>,
}

impl Diarizer {
    /// `embedder: None` still labels headset-mode mic segments as "me"; everything else stays unlabeled.
    pub fn new(embedder: Option<Embedder>, mic_mode: MicMode, voiceprint: Option<Vec<f32>>) -> Self {
        let cfg = ClusterConfig::default();
        let mut mic = OnlineClusterer::new("mic", cfg.clone());
        if let Some(vp) = &voiceprint {
            mic.set_voiceprint(vp.clone());
        }
        Self {
            embedder,
            mic_mode,
            sys: OnlineClusterer::new("sys", cfg.clone()),
            cfg,
            voiceprint,
            mic,
            mic_ring: AudioRing::default(),
            sys_ring: AudioRing::default(),
            items: Vec::new(),
        }
    }

    fn needs_audio(&self, source: Source) -> bool {
        self.embedder.is_some() && !(source == Source::Mic && self.mic_mode == MicMode::Me)
    }

    pub fn push_audio(&mut self, chunk: &AudioChunk) {
        if !self.needs_audio(chunk.source) {
            return;
        }
        match chunk.source {
            Source::Mic => self.mic_ring.push(chunk),
            Source::System => self.sys_ring.push(chunk),
        }
    }

    /// Sets `seg.speaker` for a final segment. Returns the embedding, if one was computed, for storage.
    pub fn label(&mut self, seg: &mut Segment) -> Option<Vec<f32>> {
        if seg.source == Source::Mic && self.mic_mode == MicMode::Me {
            seg.speaker = Some(ME.into());
            return None;
        }
        let embedder = self.embedder.as_mut()?;
        let ring = match seg.source {
            Source::Mic => &self.mic_ring,
            Source::System => &self.sys_ring,
        };
        let duration_ms = seg.end_ms.saturating_sub(seg.start_ms);
        let embedding = if duration_ms >= self.cfg.min_embed_ms {
            let audio = ring.slice(seg.start_ms, seg.end_ms);
            match embedder.embed(&audio) {
                Ok(e) => Some(e),
                Err(e) => {
                    log::warn!("speaker embedding for {} failed: {e:#}", seg.id);
                    None
                }
            }
        } else {
            None
        };
        self.assign(seg, embedding.as_deref());
        embedding
    }

    /// Labels `seg` online and remembers it for [`Diarizer::finish`].
    fn assign(&mut self, seg: &mut Segment, embedding: Option<&[f32]>) {
        let (clusterer, prefix) = match seg.source {
            Source::Mic => (&mut self.mic, "mic"),
            Source::System => (&mut self.sys, "sys"),
        };
        seg.speaker = clusterer.assign(embedding, seg.start_ms, seg.end_ms);
        // Unembedded segments too: re-clustering moves them along with their online label.
        self.items.push(ClusterItem {
            segment_id: seg.id.clone(),
            prefix: prefix.into(),
            embedding: embedding.map(<[f32]>::to_vec).unwrap_or_default(),
            duration_ms: seg.end_ms.saturating_sub(seg.start_ms),
            online_label: seg.speaker.clone(),
        });
    }

    /// Re-clusters the whole meeting; returns only segments whose label changed.
    pub fn finish(&self) -> Vec<(String, Option<String>)> {
        let mut changed: Vec<(usize, Option<String>)> = Vec::new();
        // The voiceprint is the user at the mic; a similar voice in the call is someone else.
        for (prefix, voiceprint) in [("mic", self.voiceprint.as_deref()), ("sys", None)] {
            let idx: Vec<usize> = (0..self.items.len()).filter(|&i| self.items[i].prefix == prefix).collect();
            if idx.is_empty() {
                continue;
            }
            let items: Vec<ClusterItem> = idx.iter().map(|&i| self.items[i].clone()).collect();
            for (&i, (_, label)) in idx.iter().zip(recluster(&items, voiceprint, &self.cfg)) {
                if label != self.items[i].online_label {
                    changed.push((i, label));
                }
            }
        }
        changed.sort_by_key(|&(i, _)| i);
        changed.into_iter().map(|(i, label)| (self.items[i].segment_id.clone(), label)).collect()
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StoredVoiceprint {
    embedding: Vec<f32>,
    created_at: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceprintStatus {
    pub enrolled: bool,
    pub created_at: Option<String>,
}

pub fn load_voiceprint(store: &Store) -> Result<Option<Vec<f32>>> {
    Ok(stored_voiceprint(store)?.map(|v| v.embedding))
}

fn stored_voiceprint(store: &Store) -> Result<Option<StoredVoiceprint>> {
    let Some(raw) = store.get_kv(VOICEPRINT_KEY)? else { return Ok(None) };
    Ok(serde_json::from_str(&raw).ok())
}

pub fn voiceprint_status(store: &Store) -> Result<VoiceprintStatus> {
    let vp = stored_voiceprint(store)?;
    Ok(VoiceprintStatus { enrolled: vp.is_some(), created_at: vp.map(|v| v.created_at) })
}

pub fn clear_voiceprint(store: &Store) -> Result<()> {
    store.set_kv(VOICEPRINT_KEY, "null")
}

/// Minimum voiced audio for a usable voiceprint.
const MIN_ENROLL_SPEECH_MS: u64 = 4_000;
const FRAME: usize = 480; // 30 ms

/// Keeps the voiced 30 ms frames: louder than 3× the quietest-fifth noise floor.
fn voiced(samples: &[f32]) -> Vec<f32> {
    let frames = samples.as_chunks::<FRAME>().0;
    if frames.is_empty() {
        return Vec::new();
    }
    let rms = |f: &[f32; FRAME]| (f.iter().map(|s| s * s).sum::<f32>() / FRAME as f32).sqrt();
    let mut levels: Vec<f32> = frames.iter().map(rms).collect();
    levels.sort_by(f32::total_cmp);
    let floor = levels[levels.len() / 5];
    let threshold = (floor * 3.0).max(0.004);
    frames.iter().filter(|f| rms(f) > threshold).flatten().copied().collect()
}

/// Records the user's voice from the mic and stores it as the "me" voiceprint.
/// Returns the amount of speech used, in ms.
pub fn enroll_voice(
    store: &Store,
    models_dir: &Path,
    mic_device: Option<String>,
    seconds: u64,
    num_threads: i32,
) -> Result<u64> {
    let seconds = seconds.clamp(5, 60);
    let model = kenes_speakers::ensure_speaker_model(models_dir, &mut |_| {})
        .context("загрузка модели голосов")?;
    let mut embedder = Embedder::new(&model, num_threads).context("инициализация модели голосов")?;

    let (tx, rx) = crossbeam_channel::unbounded();
    let capture = kenes_audio::start_capture(
        kenes_audio::CaptureConfig { mic: Some(kenes_audio::DeviceSel::from_option(mic_device)), system: None },
        tx,
    )
    .context("запуск микрофона")?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(seconds);
    let mut samples = Vec::with_capacity(seconds as usize * SAMPLE_RATE as usize);
    while let Some(left) = deadline.checked_duration_since(std::time::Instant::now()) {
        match rx.recv_timeout(left) {
            Ok(chunk) => samples.extend(chunk.samples),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => break,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => match capture.take_error() {
                Some(e) => anyhow::bail!("микрофон отключился: {}", e.message),
                None => anyhow::bail!("микрофон отключился"),
            },
        }
    }
    capture.stop();

    let speech = voiced(&samples);
    let speech_ms = speech.len() as u64 / SAMPLES_PER_MS;
    anyhow::ensure!(
        speech_ms >= MIN_ENROLL_SPEECH_MS,
        "слишком мало речи ({:.1} с из {seconds} с): говорите громче или ближе к микрофону",
        speech_ms as f64 / 1000.0
    );
    let embedding = embedder.embed(&speech)?;
    let stored = StoredVoiceprint { embedding, created_at: now_iso() };
    store.set_kv(VOICEPRINT_KEY, &serde_json::to_string(&stored)?)?;
    Ok(speech_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(source: Source, start_ms: u64, len: usize, value: f32) -> AudioChunk {
        AudioChunk { source, start_ms, samples: vec![value; len] }
    }

    #[test]
    fn ring_slices_by_session_time() {
        let mut r = AudioRing::default();
        r.push(&chunk(Source::System, 1000, 16_000, 0.1)); // 1.0–2.0 s
        r.push(&chunk(Source::System, 2000, 16_000, 0.2)); // 2.0–3.0 s
        let s = r.slice(1500, 2500);
        assert_eq!(s.len(), 16_000);
        assert_eq!(s[0], 0.1);
        assert_eq!(s[15_999], 0.2);
        // Out-of-range requests are clamped, not panics.
        assert_eq!(r.slice(0, 1100).len(), 1600);
        assert!(r.slice(5000, 6000).is_empty());
    }

    #[test]
    fn ring_fills_small_gaps_and_resets_on_large_ones() {
        let mut r = AudioRing::default();
        r.push(&chunk(Source::Mic, 0, 1600, 0.5)); // 0–100 ms
        r.push(&chunk(Source::Mic, 300, 1600, 0.5)); // 200 ms gap
        assert_eq!(r.end(), 400 * SAMPLES_PER_MS);
        assert_eq!(r.slice(150, 160), vec![0.0; 160]);
        r.push(&chunk(Source::Mic, 20_000, 1600, 0.5));
        assert_eq!(r.start, 20_000 * SAMPLES_PER_MS);
    }

    #[test]
    fn ring_is_bounded() {
        let mut r = AudioRing::default();
        for i in 0..70 {
            r.push(&chunk(Source::Mic, i * 1000, 16_000, 0.0));
        }
        assert_eq!(r.samples.len() as u64, RING_MS * SAMPLES_PER_MS);
        assert_eq!(r.start, 10_000 * SAMPLES_PER_MS);
    }

    #[test]
    fn headset_mode_labels_mic_as_me_without_a_model() {
        let mut d = Diarizer::new(None, MicMode::Me, None);
        let mut seg = Segment {
            id: "mic-1".into(),
            source: Source::Mic,
            speaker: None,
            start_ms: 0,
            end_ms: 2000,
            text: "привет".into(),
            is_final: true,
        };
        assert!(d.label(&mut seg).is_none());
        assert_eq!(seg.speaker.as_deref(), Some(ME));
        let mut sys = Segment { id: "system-1".into(), source: Source::System, ..seg };
        sys.speaker = None;
        d.label(&mut sys);
        assert_eq!(sys.speaker, None);
        assert!(d.finish().is_empty());
    }

    fn axis(k: usize) -> Vec<f32> {
        let mut v = vec![0.0; 16];
        v[k] = 1.0;
        v
    }

    fn item(id: &str, prefix: &str, embedding: Vec<f32>, duration_ms: u64, online: &str) -> ClusterItem {
        ClusterItem {
            segment_id: id.into(),
            prefix: prefix.into(),
            embedding,
            duration_ms,
            online_label: Some(online.into()),
        }
    }

    #[test]
    fn voiceprint_never_turns_call_audio_into_me() {
        // Someone in the call sounds like the user; the user is on the mic, never in the call.
        let vp = axis(0);
        for mode in [MicMode::Me, MicMode::Room] {
            let mut d = Diarizer::new(None, mode, Some(vp.clone()));
            d.items = vec![
                item("system-1", "sys", vp.clone(), 4000, "sys:1"),
                item("system-2", "sys", axis(5), 4000, "sys:2"),
                item("system-3", "sys", vp.clone(), 4000, "sys:1"),
            ];
            assert_eq!(d.finish(), Vec::new(), "{mode:?}");
        }
        // The mic in room mode still gets "me" from the voiceprint.
        let mut d = Diarizer::new(None, MicMode::Room, Some(vp.clone()));
        d.items = vec![
            item("mic-1", "mic", vp.clone(), 4000, "mic:1"),
            item("system-1", "sys", vp.clone(), 4000, "sys:1"),
        ];
        assert_eq!(d.finish(), vec![("mic-1".to_string(), Some(ME.to_string()))]);
    }

    fn final_seg(id: &str, start_ms: u64, end_ms: u64) -> Segment {
        Segment {
            id: id.into(),
            source: Source::Mic,
            speaker: None,
            start_ms,
            end_ms,
            text: "сөз".into(),
            is_final: true,
        }
    }

    #[test]
    fn short_segments_follow_their_label_when_reclustered() {
        let cfg = ClusterConfig {
            threshold: 0.5,
            recluster_threshold: 0.4,
            short_segment_ms: 0,
            short_segment_relax: 0.0,
            min_cluster_ms: 0,
            ..ClusterConfig::default()
        };
        let mut d = Diarizer::new(None, MicMode::Room, None);
        d.mic = OnlineClusterer::new("mic", cfg.clone());
        d.cfg = cfg;
        // One person heard as two voices online (0.45 < 0.5), one voice offline (≥ 0.4).
        let a = axis(0);
        let mut b = vec![0.0; 16];
        b[0] = 0.45;
        b[1] = (1.0f32 - 0.45 * 0.45).sqrt();
        let mut s1 = final_seg("mic-1", 0, 6000);
        let mut s2 = final_seg("mic-2", 7000, 11_000);
        let mut s3 = final_seg("mic-3", 11_500, 11_900); // too short to embed
        d.assign(&mut s1, Some(&a));
        d.assign(&mut s2, Some(&b));
        d.assign(&mut s3, None);
        let online: Vec<_> = [&s1, &s2, &s3].iter().map(|s| s.speaker.clone().unwrap()).collect();
        assert_eq!(online, ["mic:1", "mic:2", "mic:2"]);
        // "mic:2" is retired; the short segment must not keep it.
        assert_eq!(
            d.finish(),
            vec![("mic-2".to_string(), Some("mic:1".to_string())), ("mic-3".to_string(), Some("mic:1".to_string()))]
        );
    }

    #[test]
    fn voiced_drops_silence() {
        let mut s = vec![0.001f32; FRAME * 20];
        s.extend(vec![0.2f32; FRAME * 5]);
        assert_eq!(voiced(&s).len(), FRAME * 5);
    }

    #[test]
    fn voiceprint_round_trip() {
        let store = Store::open_in_memory().unwrap();
        assert!(!voiceprint_status(&store).unwrap().enrolled);
        store
            .set_kv(VOICEPRINT_KEY, &serde_json::to_string(&StoredVoiceprint { embedding: vec![1.0, 0.0], created_at: "t".into() }).unwrap())
            .unwrap();
        assert_eq!(load_voiceprint(&store).unwrap(), Some(vec![1.0, 0.0]));
        assert_eq!(voiceprint_status(&store).unwrap().created_at.as_deref(), Some("t"));
        clear_voiceprint(&store).unwrap();
        assert_eq!(load_voiceprint(&store).unwrap(), None);
    }
}
