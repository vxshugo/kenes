//! Per-source utterance segmentation: VAD state machine, audio buffer,
//! timestamps and id allocation. No recognizer in here, so it's testable with
//! a fake VAD.
//!
//! The VAD is sherpa-onnx's `VoiceActivityDetector` (Silero). We only use its
//! "currently in speech" flag, fed one 512-sample window at a time, and keep
//! our own copy of the audio. That lets us:
//! - decode the still-open utterance for partial results,
//! - add pre-roll before the VAD's start so first syllables aren't clipped,
//! - force-close long utterances at a quiet spot (`max_segment_ms`),
//! - map sample positions back to the chunk timeline (`AudioChunk::start_ms`).
//!
//! Boundary arithmetic mirrors `voice-activity-detector.cc` of sherpa-onnx
//! v1.13.8 (pinned in Cargo.toml): the detector flips to "speech" once
//! `min_speech` of speech has been seen, and its own segment then starts
//! `2 * window + min_speech` samples before the current position. It flips back
//! after `min_silence` of silence, or after only 0.1 s of silence once the
//! utterance is longer than `max_speech_duration` (its "soft split").

use std::collections::VecDeque;

use kenes_types::{Segment, Source, SAMPLE_RATE};

/// Silero window at 16 kHz. We always feed the VAD exactly this many samples.
pub(crate) const WINDOW: usize = 512;

const SR: u64 = SAMPLE_RATE as u64;

/// Minimum silence sherpa-onnx switches to after `max_speech_duration`
/// (`new_min_silence_duration_s_`).
const SOFT_MIN_SILENCE_S: f32 = 0.1;

/// Gain control for the VAD's copy of the audio (the recognizer gets the
/// original). Silero misses quiet speech (peaks around 0.01, e.g. a laptop mic
/// across the room) but ignores stationary noise even when it is amplified,
/// so we lift quiet input towards this window RMS…
const AGC_TARGET_RMS: f32 = 0.08;
/// …by at most this much (+20 dB), never attenuating.
const AGC_MAX_GAIN: f32 = 10.0;
/// Level envelope: instant attack, release with a ~1.5 s half-life
/// (0.5^(1 / (1.5 s * 31.25 windows/s))).
const AGC_RELEASE: f32 = 0.985;

/// A voice activity detector fed in fixed windows.
pub(crate) trait Vad: Send {
    /// Feed exactly [`WINDOW`] samples.
    fn accept_window(&mut self, window: &[f32]);
    /// Whether the detector is inside a speech region right now.
    fn is_speech(&self) -> bool;
    /// Forget all state (model state and buffered audio).
    fn reset(&mut self);
}

impl<V: Vad + ?Sized> Vad for Box<V> {
    fn accept_window(&mut self, window: &[f32]) {
        (**self).accept_window(window)
    }
    fn is_speech(&self) -> bool {
        (**self).is_speech()
    }
    fn reset(&mut self) {
        (**self).reset()
    }
}

/// VAD parameters, also passed to sherpa-onnx.
#[derive(Clone, Debug)]
pub(crate) struct VadParams {
    pub threshold: f32,
    pub min_silence_s: f32,
    pub min_speech_s: f32,
    /// sherpa-onnx `max_speech_duration`: after this the VAD splits at 0.1 s pauses.
    pub soft_max_s: f32,
}

impl VadParams {
    fn min_speech(&self) -> u64 {
        (SR as f32 * self.min_speech_s) as u64
    }
    fn min_silence(&self) -> u64 {
        (SR as f32 * self.min_silence_s) as u64
    }
    fn soft_max(&self) -> u64 {
        (SR as f32 * self.soft_max_s) as u64
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SegmenterConfig {
    pub vad: VadParams,
    /// Extra audio kept before the VAD's own segment start.
    pub preroll: u64,
    /// Audio kept after the estimated end of speech.
    pub tail: u64,
    /// Hard limit on utterance length (samples).
    pub max_segment: u64,
    /// A forced cut looks back this far for the quietest 100 ms.
    pub cut_search: u64,
    /// If the VAD has been in "speech" this long (music, noise), reset it at
    /// the next forced cut so its internal buffer stays bounded.
    pub vad_reset: u64,
    /// After such a reset, keep the utterance open this long while the VAD
    /// re-detects ongoing speech.
    pub grace: u64,
}

impl SegmenterConfig {
    pub fn new(max_segment_ms: u64) -> Self {
        let max_segment_ms = max_segment_ms.max(2_000);
        SegmenterConfig {
            // Tuned on FLEURS kk/ru, Common Voice kk and kk↔ru clips (README):
            // 0.15 s min speech catches short words whose syllables Silero
            // scores as separate ~200 ms bursts; longer pre-roll/tail or a
            // 0.4 s min silence measured slightly worse.
            vad: VadParams {
                threshold: 0.5,
                min_silence_s: 0.5,
                min_speech_s: 0.15,
                // Prefer splitting at a breath once an utterance is half the
                // hard limit; the forced cut is the fallback.
                soft_max_s: (max_segment_ms as f32 / 2_000.0).max(1.5),
            },
            preroll: ms_to_samples(250),
            tail: ms_to_samples(150),
            max_segment: ms_to_samples(max_segment_ms),
            cut_search: ms_to_samples(3_000),
            vad_reset: ms_to_samples(max_segment_ms.max(20_000) * 3),
            grace: ms_to_samples(800),
        }
    }
}

pub(crate) fn ms_to_samples(ms: u64) -> u64 {
    ms * SR / 1000
}

/// A closed utterance waiting to be decoded.
#[derive(Debug)]
pub(crate) struct Closed {
    /// Set when partials were emitted, so the final must reuse it (and must be
    /// emitted even if the text turns out empty).
    pub id: Option<String>,
    pub start_ms: u64,
    pub end_ms: u64,
    pub samples: Vec<f32>,
    /// `(offset into samples, ms)` timeline anchors; the first is at offset 0.
    anchors: Vec<(usize, u64)>,
}

impl Closed {
    /// Session time of sample `offset` of this utterance.
    pub fn ms_at(&self, offset: usize) -> u64 {
        if offset >= self.samples.len() {
            return self.end_ms;
        }
        let i = self.anchors.partition_point(|&(a, _)| a <= offset);
        let (a, ms) = self.anchors[i.saturating_sub(1)];
        (ms + (offset - a) as u64 * 1000 / SR).clamp(self.start_ms, self.end_ms)
    }
}

#[derive(Debug)]
struct Open {
    id: Option<String>,
    /// Absolute sample index of the first sample of the utterance.
    start: u64,
    /// Our estimate of the VAD's own `start_` (absolute), for its soft split.
    vad_start: u64,
    /// Audio end (absolute) covered by the last partial decode.
    partial_end: u64,
    partial_text: String,
    /// While `Some`, the VAD was reset by a forced cut and hasn't re-detected
    /// speech yet; the utterance stays open until this position.
    grace_until: Option<u64>,
}

pub(crate) struct Segmenter<V: Vad> {
    source: Source,
    cfg: SegmenterConfig,
    vad: V,
    /// Audio from absolute index `base` to `base + buf.len()` (= total received).
    buf: Vec<f32>,
    base: u64,
    /// Absolute index up to which audio has been fed to the VAD.
    fed: u64,
    /// VAD state after the last window.
    vad_speech: bool,
    /// `(absolute sample index, ms)` of each chunk start still in `buf`.
    anchors: VecDeque<(u64, u64)>,
    /// RMS envelope for the VAD's gain control.
    agc_env: f32,
    /// Scratch copy of the window with gain applied.
    vad_window: Vec<f32>,
    /// The next utterance may not start before this (end of the previous one).
    floor: u64,
    open: Option<Open>,
    next_id: u64,
}

impl<V: Vad> Segmenter<V> {
    pub fn new(source: Source, cfg: SegmenterConfig, vad: V) -> Self {
        Segmenter {
            source,
            cfg,
            vad,
            buf: Vec::new(),
            base: 0,
            fed: 0,
            vad_speech: false,
            anchors: VecDeque::new(),
            agc_env: 0.0,
            vad_window: vec![0.0; WINDOW],
            floor: 0,
            open: None,
            next_id: 1,
        }
    }

    pub fn source(&self) -> Source {
        self.source
    }

    /// Forget everything, including the id counter.
    pub fn reset(&mut self) {
        self.vad.reset();
        self.buf.clear();
        self.base = 0;
        self.fed = 0;
        self.vad_speech = false;
        self.anchors.clear();
        self.agc_env = 0.0;
        self.floor = 0;
        self.open = None;
        self.next_id = 1;
    }

    fn total(&self) -> u64 {
        self.base + self.buf.len() as u64
    }

    /// Append a chunk; utterances that closed are pushed to `out`.
    pub fn push(&mut self, start_ms: u64, samples: &[f32], out: &mut Vec<Closed>) {
        if samples.is_empty() {
            return;
        }
        let total = self.total();
        if self.anchors.back().is_none_or(|&(idx, ms)| {
            // Only store an anchor when the timeline isn't just continuing.
            let expected = ms + (total - idx) * 1000 / SR;
            start_ms.abs_diff(expected) > 1 || total - idx > 30 * SR
        }) {
            self.anchors.push_back((total, start_ms));
        }
        self.buf.extend_from_slice(samples);

        while self.fed + WINDOW as u64 <= self.total() {
            let from = (self.fed - self.base) as usize;
            // Sherpa decides its soft split before looking at the new window.
            let soft = self.soft_split_possible();
            let window = &self.buf[from..from + WINDOW];
            let rms = (window.iter().map(|x| x * x).sum::<f32>() / WINDOW as f32).sqrt();
            self.agc_env = rms.max(self.agc_env * AGC_RELEASE);
            let gain = (AGC_TARGET_RMS / self.agc_env.max(1e-9)).clamp(1.0, AGC_MAX_GAIN);
            for (dst, src) in self.vad_window.iter_mut().zip(window) {
                *dst = (src * gain).clamp(-1.0, 1.0);
            }
            self.vad.accept_window(&self.vad_window);
            self.fed += WINDOW as u64;
            let speech = self.vad.is_speech();
            self.step(speech, soft, out);
        }
        self.trim();
    }

    /// Whether the VAD may be using its short 0.1 s silence right now. Errs on
    /// the side of "yes": that only makes us keep a little more trailing audio.
    fn soft_split_possible(&self) -> bool {
        let Some(u) = &self.open else { return false };
        if !self.vad_speech {
            return false;
        }
        let vad_buffered = (self.fed + WINDOW as u64).saturating_sub(u.vad_start);
        vad_buffered + 2 * WINDOW as u64 > self.cfg.vad.soft_max()
    }

    fn step(&mut self, speech: bool, soft: bool, out: &mut Vec<Closed>) {
        let was = std::mem::replace(&mut self.vad_speech, speech);
        let w = WINDOW as u64;
        match (was, speech) {
            (false, true) => {
                log::trace!("{:?} speech at {} ms", self.source, self.ms_at(self.fed));
                let vad_start = self.fed.saturating_sub(2 * w + self.cfg.vad.min_speech());
                match &mut self.open {
                    // Speech resumed after a forced cut: keep the open utterance.
                    Some(u) => {
                        u.grace_until = None;
                        u.vad_start = vad_start;
                    }
                    None => {
                        let start = vad_start
                            .saturating_sub(self.cfg.preroll)
                            .max(self.floor)
                            .max(self.base);
                        self.open = Some(Open {
                            id: None,
                            start,
                            vad_start,
                            partial_end: start,
                            partial_text: String::new(),
                            grace_until: None,
                        });
                    }
                }
            }
            (true, false) => {
                log::trace!(
                    "{:?} silence at {} ms (soft {soft})",
                    self.source,
                    self.ms_at(self.fed)
                );
                let silence = if soft {
                    (SR as f32 * SOFT_MIN_SILENCE_S) as u64
                } else {
                    self.cfg.vad.min_silence()
                };
                // Keep a little of the silence, but never more than half of it:
                // after a short soft-split pause the next word starts right away.
                let end = self.fed.saturating_sub(silence) + self.cfg.tail.min(silence / 2);
                out.push(self.close(end));
            }
            (true, true) => {
                if let Some(u) = &self.open {
                    if self.fed - u.start >= self.cfg.max_segment {
                        self.forced_cut(out);
                    }
                }
            }
            (false, false) => {
                if let Some(u) = &self.open {
                    if u.grace_until.is_some_and(|g| self.fed >= g) {
                        log::trace!(
                            "{:?} grace expired at {} ms",
                            self.source,
                            self.ms_at(self.fed)
                        );
                        // Speech didn't resume after the cut: close what's there.
                        out.push(self.close(self.fed));
                    }
                }
            }
        }
    }

    /// Close the open utterance at the quietest spot of the last
    /// `cut_search` samples, and continue with a new one from there. The VAD
    /// keeps running: resetting Silero mid-speech makes it miss the ongoing
    /// speech for seconds, so we only do that after `vad_reset`.
    fn forced_cut(&mut self, out: &mut Vec<Closed>) {
        let Some(u) = &self.open else { return };
        let lo =
            (u.start + self.cfg.max_segment / 2).max(self.fed.saturating_sub(self.cfg.cut_search));
        let cut = self.quietest(lo, self.fed);
        let mut vad_start = u.vad_start;
        log::trace!(
            "{:?} forced cut at {} ms (now {} ms)",
            self.source,
            self.ms_at(cut),
            self.ms_at(self.fed)
        );
        out.push(self.close(cut));
        let mut grace_until = None;
        if self.fed - vad_start >= self.cfg.vad_reset {
            log::debug!(
                "{:?}: VAD in speech for {} s, resetting",
                self.source,
                (self.fed - vad_start) / SR
            );
            self.vad.reset();
            self.vad_speech = false;
            vad_start = self.fed;
            grace_until = Some(self.fed + self.cfg.grace);
        }
        self.open = Some(Open {
            id: None,
            start: cut,
            vad_start,
            partial_end: cut,
            partial_text: String::new(),
            grace_until,
        });
    }

    /// Centre of the lowest-energy 100 ms stretch in `[lo, hi)` (the latest
    /// one on ties); `hi` if the range is too short. 100 ms rather than a
    /// single frame so we land in a pause, not in a stop consonant's closure.
    fn quietest(&self, lo: u64, hi: u64) -> u64 {
        const HOP: usize = 160;
        const FRAMES: usize = 10;
        let lo = lo.max(self.base);
        if hi < lo + (HOP * FRAMES) as u64 {
            return hi;
        }
        let a = (lo - self.base) as usize;
        let b = (hi - self.base) as usize;
        let energies: Vec<f32> = self.buf[a..b]
            .as_chunks::<HOP>()
            .0
            .iter()
            .map(|f| f.iter().map(|x| x * x).sum())
            .collect();
        let mut best = (f32::INFINITY, hi);
        let mut sum: f32 = energies[..FRAMES].iter().sum();
        for i in FRAMES..=energies.len() {
            if i > FRAMES {
                sum += energies[i - 1] - energies[i - 1 - FRAMES];
            }
            if sum <= best.0 {
                best = (sum, lo + ((i - FRAMES / 2) * HOP) as u64);
            }
        }
        best.1
    }

    fn close(&mut self, end: u64) -> Closed {
        let u = self.open.take().expect("close without open utterance");
        let end = end.clamp(u.start, self.total());
        self.floor = end;
        let (a, b) = ((u.start - self.base) as usize, (end - self.base) as usize);
        let start_ms = self.ms_at(u.start);
        let mut anchors = vec![(0, start_ms)];
        anchors.extend(
            self.anchors
                .iter()
                .filter(|&&(idx, _)| idx > u.start && idx < end)
                .map(|&(idx, ms)| ((idx - u.start) as usize, ms)),
        );
        Closed {
            id: u.id,
            start_ms,
            end_ms: self.ms_at(end).max(start_ms),
            samples: self.buf[a..b].to_vec(),
            anchors,
        }
    }

    /// Close any open utterance at the end of the audio received so far
    /// (input ended). The VAD state is left alone; call `reset` to reuse.
    pub fn flush(&mut self, out: &mut Vec<Closed>) {
        if self.open.is_some() {
            let end = self.total();
            out.push(self.close(end));
        }
    }

    /// Drop audio nobody can need anymore.
    fn trim(&mut self) {
        let idle_keep = 3 * WINDOW as u64 + self.cfg.vad.min_speech() + self.cfg.preroll + SR / 4;
        let keep_from = match &self.open {
            Some(u) => u.start,
            None => self.total().saturating_sub(idle_keep),
        };
        if keep_from >= self.base + SR {
            self.buf.drain(..(keep_from - self.base) as usize);
            self.base = keep_from;
            while self.anchors.len() >= 2 && self.anchors[1].0 <= self.base {
                self.anchors.pop_front();
            }
        }
    }

    /// Session time of absolute sample `idx`, from the chunk timeline.
    pub fn ms_at(&self, idx: u64) -> u64 {
        let i = self.anchors.partition_point(|&(a, _)| a <= idx);
        match i.checked_sub(1).and_then(|i| self.anchors.get(i)) {
            Some(&(a, ms)) => ms + (idx - a) * 1000 / SR,
            None => self.anchors.front().map_or(idx * 1000 / SR, |&(_, ms)| ms),
        }
    }

    /// Samples in the open utterance, if any.
    pub fn open_len(&self) -> Option<u64> {
        self.open.as_ref().map(|u| self.total() - u.start)
    }

    /// Samples received since the last partial decode of the open utterance.
    pub fn new_since_partial(&self) -> Option<u64> {
        self.open.as_ref().map(|u| self.total() - u.partial_end)
    }

    /// Audio of the open utterance and the absolute end it covers.
    pub fn partial_audio(&self) -> Option<(&[f32], u64)> {
        let u = self.open.as_ref()?;
        Some((&self.buf[(u.start - self.base) as usize..], self.total()))
    }

    /// Record the result of decoding the open utterance up to `end`. Returns
    /// the partial segment to emit, if the text is new and non-empty.
    pub fn apply_partial(&mut self, end: u64, text: &str) -> Option<Segment> {
        let text = normalize(text);
        let start = self.open.as_ref()?.start;
        let (start_ms, end_ms) = (self.ms_at(start), self.ms_at(end));
        let id_num = &mut self.next_id;
        let source = self.source;
        let u = self.open.as_mut()?;
        u.partial_end = end;
        if text.is_empty() || text == u.partial_text {
            return None;
        }
        u.partial_text = text.clone();
        let id = u.id.get_or_insert_with(|| alloc_id(source, id_num)).clone();
        Some(Segment {
            id,
            source,
            speaker: None,
            start_ms,
            end_ms: end_ms.max(start_ms),
            text,
            is_final: false,
        })
    }

    /// Turn a decoded closed utterance into the final segment, or `None` if it
    /// is empty and nothing was shown for it.
    #[cfg(test)]
    pub fn final_segment(&mut self, c: Closed, text: &str) -> Option<Segment> {
        let len = c.samples.len();
        self.final_segments(c, &[(0, len, text.to_string())]).pop()
    }

    /// Final segments for a closed utterance decoded as consecutive `pieces`
    /// `(from, to, text)` (sample offsets; one piece when it wasn't split).
    ///
    /// The first piece keeps the utterance's id, and is emitted even with
    /// empty text if partials were shown under that id (the UI then drops
    /// them). Later pieces get fresh ids and are skipped when empty.
    pub fn final_segments(&mut self, c: Closed, pieces: &[(usize, usize, String)]) -> Vec<Segment> {
        let mut out = Vec::with_capacity(pieces.len());
        let mut original = c.id.clone();
        for (k, (from, to, text)) in pieces.iter().enumerate() {
            let text = normalize(text);
            let id = match (k, original.take()) {
                (0, Some(id)) => id,
                _ if text.is_empty() => continue,
                _ => alloc_id(self.source, &mut self.next_id),
            };
            let (start_ms, end_ms) = (c.ms_at(*from), c.ms_at(*to));
            out.push(Segment {
                id,
                source: self.source,
                speaker: None,
                start_ms: if k == 0 { c.start_ms } else { start_ms },
                end_ms: if k + 1 == pieces.len() {
                    c.end_ms
                } else {
                    end_ms.max(start_ms)
                },
                text,
                is_final: true,
            });
        }
        out
    }
}

fn alloc_id(source: Source, next: &mut u64) -> String {
    let id = format!("{}-{}", source.as_str(), *next);
    *next += 1;
    id
}

/// Trim and collapse whitespace. The models emit lowercase text without
/// punctuation; we leave that alone.
pub(crate) fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Energy VAD with sherpa-onnx's hysteresis: a window is "loud" if its
    /// mean |x| > 0.05; speech starts after `min_speech` of consecutive loud
    /// windows and ends after `min_silence` of quiet ones.
    pub(crate) struct FakeVad {
        min_speech: u64,
        min_silence: u64,
        loud_run: u64,
        quiet_run: u64,
        speech: bool,
        pub resets: usize,
    }

    impl FakeVad {
        pub fn new(p: &VadParams) -> Self {
            FakeVad {
                min_speech: p.min_speech(),
                min_silence: p.min_silence(),
                loud_run: 0,
                quiet_run: 0,
                speech: false,
                resets: 0,
            }
        }
    }

    impl Vad for FakeVad {
        fn accept_window(&mut self, window: &[f32]) {
            assert_eq!(window.len(), WINDOW);
            let loud = window.iter().map(|x| x.abs()).sum::<f32>() / WINDOW as f32 > 0.05;
            let w = WINDOW as u64;
            if loud {
                self.loud_run += w;
                self.quiet_run = 0;
                if self.loud_run > self.min_speech {
                    self.speech = true;
                }
            } else {
                self.quiet_run += w;
                self.loud_run = 0;
                if self.quiet_run >= self.min_silence {
                    self.speech = false;
                }
            }
        }
        fn is_speech(&self) -> bool {
            self.speech
        }
        fn reset(&mut self) {
            self.loud_run = 0;
            self.quiet_run = 0;
            self.speech = false;
            self.resets += 1;
        }
    }

    pub(crate) fn tone(ms: u64) -> Vec<f32> {
        (0..ms_to_samples(ms))
            .map(|i| 0.3 * ((i as f32) * 0.3).sin().signum())
            .collect()
    }

    pub(crate) fn silence(ms: u64) -> Vec<f32> {
        vec![0.0; ms_to_samples(ms) as usize]
    }

    fn cfg() -> SegmenterConfig {
        SegmenterConfig::new(20_000)
    }

    fn seg(cfg: SegmenterConfig) -> Segmenter<FakeVad> {
        let vad = FakeVad::new(&cfg.vad);
        Segmenter::new(Source::Mic, cfg, vad)
    }

    /// Feed `audio` in 100 ms chunks on a contiguous timeline from `t0_ms`.
    fn feed(s: &mut Segmenter<FakeVad>, t0_ms: u64, audio: &[f32]) -> Vec<Closed> {
        let mut out = Vec::new();
        for (i, c) in audio.chunks(1600).enumerate() {
            s.push(t0_ms + i as u64 * 100, c, &mut out);
        }
        out
    }

    /// Duration of the "speech" (|x| > 0.1) contained in `samples`.
    fn loud_ms(samples: &[f32]) -> u64 {
        samples.iter().filter(|x| x.abs() > 0.1).count() as u64 * 1000 / SR
    }

    #[test]
    fn one_utterance_keeps_all_speech_and_padding() {
        let mut s = seg(cfg());
        let audio = [silence(1_000), tone(2_000), silence(1_500)].concat();
        let out = feed(&mut s, 0, &audio);
        assert_eq!(out.len(), 1);
        let c = &out[0];
        // Starts before the speech (pre-roll) and ends after it (tail).
        assert!(c.start_ms <= 1_000 - 250, "start {}", c.start_ms);
        assert!(c.start_ms >= 1_000 - 700, "start {}", c.start_ms);
        assert!(c.end_ms >= 3_000 + 100, "end {}", c.end_ms);
        assert!(c.end_ms <= 3_000 + 600, "end {}", c.end_ms);
        assert_eq!(loud_ms(&c.samples), 2_000, "no speech clipped");
        assert_eq!(c.samples.len() as u64, ms_to_samples(c.end_ms - c.start_ms));
    }

    #[test]
    fn timestamps_follow_chunk_timeline() {
        let mut s = seg(cfg());
        let audio = [silence(500), tone(1_000), silence(1_000)].concat();
        // Session started 10 s before this source's first chunk.
        let out = feed(&mut s, 10_000, &audio);
        assert_eq!(out.len(), 1);
        assert!(out[0].start_ms >= 10_000 && out[0].start_ms <= 10_500);

        // A gap in the timeline (capture hiccup): later audio maps past it.
        let mut out = Vec::new();
        for (i, c) in [silence(500), tone(1_000), silence(1_000)]
            .concat()
            .chunks(1600)
            .enumerate()
        {
            s.push(60_000 + i as u64 * 100, c, &mut out);
        }
        assert_eq!(out.len(), 1);
        assert!(
            out[0].start_ms >= 60_000 && out[0].start_ms <= 60_500,
            "{}",
            out[0].start_ms
        );
        assert!(
            out[0].end_ms > 61_500 && out[0].end_ms < 62_200,
            "{}",
            out[0].end_ms
        );
    }

    #[test]
    fn two_utterances_do_not_overlap() {
        let mut s = seg(cfg());
        let audio = [tone(1_000), silence(700), tone(1_000), silence(1_000)].concat();
        let out = feed(&mut s, 0, &audio);
        assert_eq!(out.len(), 2);
        assert!(out[0].end_ms <= out[1].start_ms);
        assert_eq!(loud_ms(&out[0].samples) + loud_ms(&out[1].samples), 2_000);
    }

    #[test]
    fn short_blip_is_ignored() {
        let mut s = seg(cfg());
        let out = feed(
            &mut s,
            0,
            &[silence(500), tone(100), silence(1_000)].concat(),
        );
        assert!(out.is_empty());
        assert!(s.open_len().is_none());
    }

    #[test]
    fn long_speech_is_force_cut_without_losing_audio() {
        let mut c = cfg();
        c.max_segment = ms_to_samples(5_000);
        c.vad.soft_max_s = 100.0;
        let mut s = seg(c);
        // 12 s of speech with a quieter dip every 1.7 s (not a pause).
        let mut audio = Vec::new();
        for _ in 0..7 {
            audio.extend(tone(1_650));
            audio.extend(vec![0.08f32; ms_to_samples(50) as usize]);
        }
        audio.truncate(ms_to_samples(12_000) as usize);
        audio.extend(silence(1_500));
        let out = feed(&mut s, 0, &audio);
        assert!(out.len() >= 3, "got {} pieces", out.len());
        for w in out.windows(2) {
            // Contiguous: the next piece starts exactly where the last ended.
            assert_eq!(w[0].end_ms, w[1].start_ms);
        }
        for c in &out {
            assert!(
                c.end_ms - c.start_ms <= 5_000 + 32,
                "piece of {} ms",
                c.end_ms - c.start_ms
            );
        }
        let total: u64 = out.iter().map(|c| loud_ms(&c.samples)).sum();
        assert_eq!(total, loud_ms(&audio));
        assert_eq!(s.vad.resets, 0, "the VAD keeps running across forced cuts");
    }

    #[test]
    fn forced_cut_prefers_a_pause_over_a_short_dip() {
        let mut c = cfg();
        c.max_segment = ms_to_samples(4_000);
        c.vad.soft_max_s = 100.0;
        let mut s = seg(c);
        // A 30 ms stop-like dip at 2.5 s and a 120 ms quiet pause at 3.2 s,
        // both too short to end the utterance.
        let audio = [
            tone(2_500),
            silence(30),
            tone(670),
            vec![0.02; ms_to_samples(120) as usize],
            tone(1_000),
            silence(1_000),
        ]
        .concat();
        let out = feed(&mut s, 0, &audio);
        assert_eq!(out.len(), 2, "{out:?}");
        assert!(
            out[0].end_ms >= 3_200 && out[0].end_ms <= 3_320,
            "cut at {}",
            out[0].end_ms
        );
    }

    #[test]
    fn vad_reset_after_very_long_speech_keeps_the_tail() {
        let mut c = cfg();
        c.max_segment = ms_to_samples(3_000);
        c.vad_reset = ms_to_samples(2_000);
        c.vad.soft_max_s = 100.0;
        let mut s = seg(c);
        let audio = [tone(3_200), silence(2_000)].concat();
        let out = feed(&mut s, 0, &audio);
        assert_eq!(out.len(), 2, "{out:?}");
        let total: u64 = out.iter().map(|c| loud_ms(&c.samples)).sum();
        assert_eq!(total, 3_200);
        assert!(s.open_len().is_none());
        assert_eq!(s.vad.resets, 1);
    }

    #[test]
    fn partial_then_final_share_id_and_ids_are_dense() {
        let mut s = seg(cfg());
        let mut out = feed(&mut s, 0, &[silence(300), tone(1_500)].concat());
        assert!(out.is_empty());
        let (audio, end) = s.partial_audio().unwrap();
        assert!(!audio.is_empty());
        let p = s.apply_partial(end, "  сәлем   ").unwrap();
        assert_eq!(p.id, "mic-1");
        assert_eq!(p.text, "сәлем");
        assert!(!p.is_final);
        // Same text again: nothing new to show.
        assert!(s.apply_partial(end, "сәлем").is_none());
        assert_eq!(s.new_since_partial(), Some(0));

        out.extend(feed(&mut s, 1_800, &silence(1_000)));
        assert_eq!(out.len(), 1);
        let f = s.final_segment(out.pop().unwrap(), "сәлем қалай").unwrap();
        assert_eq!((f.id.as_str(), f.is_final), ("mic-1", true));
        assert_eq!(f.start_ms, p.start_ms);

        // An utterance that decodes to nothing and had no partials: no id used.
        let out = feed(&mut s, 2_800, &[tone(1_000), silence(1_000)].concat());
        assert!(s
            .final_segment(out.into_iter().next().unwrap(), " ")
            .is_none());
        let out = feed(&mut s, 4_800, &[tone(1_000), silence(1_000)].concat());
        let f = s
            .final_segment(out.into_iter().next().unwrap(), "иә")
            .unwrap();
        assert_eq!(f.id, "mic-2");
    }

    #[test]
    fn empty_final_after_partial_is_still_emitted() {
        let mut s = seg(cfg());
        feed(&mut s, 0, &tone(1_000));
        let (_, end) = s.partial_audio().unwrap();
        s.apply_partial(end, "м").unwrap();
        let out = feed(&mut s, 1_000, &silence(1_000));
        let f = s
            .final_segment(out.into_iter().next().unwrap(), "")
            .unwrap();
        assert_eq!(f.id, "mic-1");
        assert!(f.is_final && f.text.is_empty());
    }

    #[test]
    fn flush_closes_open_utterance() {
        let mut s = seg(cfg());
        feed(&mut s, 0, &tone(1_000));
        let mut out = Vec::new();
        s.flush(&mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(loud_ms(&out[0].samples), 1_000);
    }

    #[test]
    fn buffer_is_trimmed_while_idle() {
        let mut s = seg(cfg());
        feed(&mut s, 0, &silence(60_000));
        assert!(s.buf.len() < 2 * SR as usize, "{}", s.buf.len());
        assert!(s.anchors.len() <= 3);
    }

    #[test]
    fn odd_chunk_sizes_are_handled() {
        let mut s = seg(cfg());
        let audio = [silence(700), tone(1_300), silence(900)].concat();
        let mut out = Vec::new();
        let mut t = 0usize;
        for (i, len) in [7usize, 333, 1600, 320, 1, 4000].iter().cycle().enumerate() {
            if t >= audio.len() {
                break;
            }
            let end = (t + len).min(audio.len());
            s.push((t as u64) * 1000 / SR, &audio[t..end], &mut out);
            t = end;
            let _ = i;
        }
        assert_eq!(out.len(), 1);
        assert_eq!(loud_ms(&out[0].samples), 1_300);
    }

    #[test]
    fn quiet_speech_is_lifted_for_the_vad() {
        let mut s = seg(cfg());
        // Peak 0.01: below the fake VAD's 0.05 threshold without gain.
        let quiet: Vec<f32> = tone(1_500).iter().map(|x| x / 30.0).collect();
        let out = feed(
            &mut s,
            0,
            &[silence(500), quiet.clone(), silence(1_000)].concat(),
        );
        assert_eq!(out.len(), 1);
        // The recognizer gets the original, unamplified audio.
        let peak = out[0].samples.iter().fold(0f32, |m, x| m.max(x.abs()));
        assert!((peak - 0.01).abs() < 1e-6, "peak {peak}");
        assert_eq!(
            out[0].samples.iter().filter(|x| x.abs() > 0.005).count(),
            quiet.len()
        );
    }

    #[test]
    fn normalize_collapses_whitespace() {
        assert_eq!(normalize("  a \t b\n\nc  "), "a b c");
        assert_eq!(normalize("   "), "");
    }
}
