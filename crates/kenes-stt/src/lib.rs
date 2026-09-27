//! Local speech recognition for kenes: model registry and download, Silero
//! VAD, an offline GigaAM CTC recognizer (on ONNX Runtime with GigaAM's own
//! front-end, or through sherpa-onnx), and the streaming [`Transcriber`] that
//! turns [`AudioChunk`]s into partial and final [`Segment`]s. See
//! `docs/CONTRACT.md` and this crate's README.
//!
//! ```no_run
//! use kenes_stt::{ensure_model, SttConfig, Transcriber};
//! # fn main() -> anyhow::Result<()> {
//! let cfg = SttConfig::default();
//! ensure_model(&cfg.model_id, &cfg.models_dir, &mut |p| eprintln!("{:.0}%", p * 100.0))?;
//! let (audio_tx, audio_rx) = crossbeam_channel::unbounded();
//! let (seg_tx, seg_rx) = crossbeam_channel::unbounded();
//! let worker = Transcriber::new(cfg)?.spawn(audio_rx, seg_tx);
//! // … send AudioChunks on audio_tx, read Segments from seg_rx …
//! drop(audio_tx); // flushes the open utterances as finals, then the thread exits
//! worker.join().unwrap();
//! # drop(seg_rx);
//! # Ok(()) }
//! ```

mod engine;
mod gigaam;
mod registry;
mod segmenter;
mod splitter;

use std::path::PathBuf;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError, TrySendError};
use kenes_types::{AudioChunk, Segment, Source, SAMPLE_RATE};

use engine::{load_recognizer, Recognizer, SileroVad};
use registry::ModelKind;
use segmenter::{ms_to_samples, Closed, Segmenter, SegmenterConfig, Vad};
use splitter::{plan_pieces, SplitResult, SplitterHandle};

pub use engine::{SttBackend, BACKEND_ENV};
pub use gigaam::{LogMel, HOP, N_FFT, N_MELS};
pub use splitter::{SplitOptions, Splitter};

pub use registry::{
    available_models, available_models_in, download_verified, ensure_model, is_downloaded,
    model_path, models_dir, sha256_file, verify_model, ModelInfo, DEFAULT_MODEL, VAD_MODEL,
};

/// Recognizer settings. `SttConfig::default()` gives the documented defaults.
#[derive(Clone, Debug)]
pub struct SttConfig {
    /// Registry id, e.g. `"gigaam-multilingual-ctc"`.
    pub model_id: String,
    pub models_dir: PathBuf,
    /// ONNX Runtime intra-op threads for the recognizer.
    pub num_threads: i32,
    /// Re-decode the open utterance this often (ms of new audio). The worker
    /// stretches the interval when decoding gets expensive; `0` disables partials.
    pub partial_interval_ms: u64,
    /// Force-close utterances longer than this (min 2000).
    pub max_segment_ms: u64,
}

impl Default for SttConfig {
    fn default() -> Self {
        SttConfig {
            model_id: DEFAULT_MODEL.to_string(),
            models_dir: models_dir(),
            num_threads: 4,
            partial_interval_ms: 700,
            max_segment_ms: 20_000,
        }
    }
}

/// Streaming transcriber: one recognizer shared by both sources, one VAD and
/// audio buffer per source. Create it with [`Transcriber::new`] (loads the
/// model, ~1 s), then either [`spawn`](Transcriber::spawn) it on live audio or
/// use [`transcribe_buffer`](Transcriber::transcribe_buffer).
pub struct Transcriber {
    recognizer: Box<dyn Recognizer>,
    /// The backend actually in use (never `Auto`).
    backend: SttBackend,
    /// Index 0: mic, 1: system.
    segs: [Segmenter<Box<dyn Vad>>; 2],
    policy: PartialPolicy,
    /// CPU budget: no partial decode before this.
    partial_ready_at: Instant,
    rtf: RtfEstimate,
    stats: Stats,
    splitter: Option<SplitterHandle>,
    split_opts: SplitOptions,
}

fn slot(source: Source) -> usize {
    match source {
        Source::Mic => 0,
        Source::System => 1,
    }
}

impl Transcriber {
    /// Load the recognizer and two VADs. The model must already be on disk
    /// (see [`ensure_model`]). Same as [`with_backend`](Self::with_backend)
    /// with [`SttBackend::Auto`].
    pub fn new(cfg: SttConfig) -> Result<Self> {
        Self::with_backend(cfg, SttBackend::Auto)
    }

    /// Like [`new`](Self::new), choosing the recognizer engine. `Auto` means
    /// `$KENES_STT_BACKEND` if set, else the model's preferred backend (ONNX
    /// Runtime for every GigaAM model). If the ONNX Runtime backend can't load
    /// the model, sherpa-onnx is used instead and a warning is logged; see
    /// [`backend`](Self::backend) for what was picked.
    pub fn with_backend(cfg: SttConfig, backend: SttBackend) -> Result<Self> {
        let Some(spec) = registry::spec(&cfg.model_id) else {
            bail!("unknown model {:?}", cfg.model_id);
        };
        if spec.kind == ModelKind::Vad {
            bail!("{} is not a speech recognition model", cfg.model_id);
        }
        let t0 = Instant::now();
        let (recognizer, backend) = load_recognizer(
            spec,
            &model_path(spec.id, &cfg.models_dir),
            cfg.num_threads,
            backend,
        )?;
        let seg_cfg = SegmenterConfig::new(cfg.max_segment_ms);
        let vad_path = model_path(VAD_MODEL, &cfg.models_dir).join("silero_vad.onnx");
        let mic_vad = SileroVad::new(&vad_path, &seg_cfg.vad)?;
        let sys_vad = SileroVad::new(&vad_path, &seg_cfg.vad)?;
        log::info!(
            "loaded {} ({backend} backend) in {:.2} s",
            cfg.model_id,
            t0.elapsed().as_secs_f32()
        );
        Ok(Self::from_parts(
            &cfg,
            seg_cfg,
            recognizer,
            backend,
            [Box::new(mic_vad), Box::new(sys_vad)],
        ))
    }

    fn from_parts(
        cfg: &SttConfig,
        seg_cfg: SegmenterConfig,
        recognizer: Box<dyn Recognizer>,
        backend: SttBackend,
        [mic_vad, sys_vad]: [Box<dyn Vad>; 2],
    ) -> Self {
        Transcriber {
            recognizer,
            backend,
            segs: [
                Segmenter::new(Source::Mic, seg_cfg.clone(), mic_vad),
                Segmenter::new(Source::System, seg_cfg, sys_vad),
            ],
            policy: PartialPolicy::new(cfg.partial_interval_ms),
            partial_ready_at: Instant::now(),
            rtf: RtfEstimate::default(),
            stats: Stats::default(),
            splitter: None,
            split_opts: SplitOptions::default(),
        }
    }

    /// Install a speaker-change detector for final utterances.
    ///
    /// For every final of at least [`SplitOptions::min_utterance_ms`], `f`
    /// gets exactly the samples that would be decoded and returns the sample
    /// offsets where the speaker changes (empty: no split). Each piece is
    /// then decoded on its own. The first piece is the final for the
    /// utterance's existing id; the others become finals with fresh ids from
    /// the same per-source counter, with their own `start_ms`/`end_ms`.
    /// Later pieces that decode to nothing are skipped. Pieces shorter than
    /// [`SplitOptions::min_piece_ms`] are merged into a neighbour. Partials
    /// are never split.
    ///
    /// `f` runs on a helper thread (`kenes-stt-split`) owned by the
    /// transcriber. The worker waits for it up to
    /// `timeout_ms + timeout_ratio × length` and otherwise emits the
    /// utterance unsplit. A panic in `f` (caught with `catch_unwind`) also
    /// means "no split", and `f` keeps being used. Replaces any previous
    /// splitter. Call before [`spawn`](Self::spawn); also applies to
    /// [`transcribe_buffer`](Self::transcribe_buffer).
    pub fn set_splitter(&mut self, f: Splitter) {
        self.splitter = Some(SplitterHandle::new(f));
    }

    /// Change when the splitter is used (see [`SplitOptions`]).
    pub fn set_split_options(&mut self, opts: SplitOptions) {
        self.split_opts = opts;
    }

    /// Consume chunks until `rx` disconnects; emit partial and final segments.
    ///
    /// Finals are sent with a blocking `send` and are never dropped. Partials
    /// are only decoded when the input queue is empty, and are sent with
    /// `try_send`, so they are dropped if `tx` is a full bounded channel. When
    /// `rx` disconnects, open utterances are flushed as finals before the
    /// thread exits. The thread also exits if `tx`'s receiver is dropped.
    pub fn spawn(self, rx: Receiver<AudioChunk>, tx: Sender<Segment>) -> JoinHandle<()> {
        std::thread::Builder::new()
            .name("kenes-stt".into())
            .spawn(move || self.run(rx, tx))
            .expect("failed to spawn the STT thread")
    }

    /// Offline helper for tests/CLI: VAD-split and transcribe a whole buffer
    /// (16 kHz mono). Uses the same segmentation as the live path; timestamps
    /// start at 0 and ids at `<source>-1`.
    pub fn transcribe_buffer(&mut self, source: Source, samples: &[f32]) -> Result<Vec<Segment>> {
        let i = slot(source);
        self.segs[i].reset();
        let mut out = Vec::new();
        let mut closed = Vec::new();
        let chunk = ms_to_samples(100) as usize;
        let mut chunks = samples.chunks(chunk).enumerate().peekable();
        while let Some((n, c)) = chunks.next() {
            self.segs[i].push(n as u64 * 100, c, &mut closed);
            if chunks.peek().is_none() {
                self.segs[i].flush(&mut closed);
            }
            for c in closed.drain(..) {
                out.extend(self.finalize(i, c, true)?);
            }
        }
        self.segs[i].reset();
        Ok(out)
    }

    /// Decode one utterance (16 kHz mono) as-is, without VAD. Raw recognizer
    /// output, for benchmarks and tests.
    pub fn recognize(&mut self, samples: &[f32]) -> Result<String> {
        self.recognizer.recognize(samples)
    }

    /// Like [`recognize`](Self::recognize) but without the 0.3 s of trailing
    /// silence every decode normally gets. Only for comparing with reference
    /// implementations that don't pad.
    #[doc(hidden)]
    pub fn recognize_unpadded(&mut self, samples: &[f32]) -> Result<String> {
        self.recognizer.decode(samples)
    }

    /// The recognizer engine in use: [`SttBackend::Ort`] or
    /// [`SttBackend::Sherpa`], never `Auto`.
    pub fn backend(&self) -> SttBackend {
        self.backend
    }

    fn run(mut self, rx: Receiver<AudioChunk>, tx: Sender<Segment>) {
        log::debug!("STT worker started");
        let mut closed = Vec::new();
        'outer: loop {
            let due = self.pick_partial(Instant::now());
            let first = match due {
                PartialDue::Now(_) => match rx.try_recv() {
                    Ok(c) => Some(c),
                    Err(TryRecvError::Empty) => None,
                    Err(TryRecvError::Disconnected) => break 'outer,
                },
                // A partial is waiting only for the CPU budget: sleep on the
                // queue until then.
                PartialDue::After(wait) => match rx.recv_timeout(wait) {
                    Ok(c) => Some(c),
                    Err(RecvTimeoutError::Timeout) => continue 'outer,
                    Err(RecvTimeoutError::Disconnected) => break 'outer,
                },
                PartialDue::No => match rx.recv() {
                    Ok(c) => Some(c),
                    Err(_) => break 'outer,
                },
            };

            if let Some(mut chunk) = first {
                // Drain everything queued before spending time on partials;
                // finals are decoded as soon as their utterance closes.
                loop {
                    let seg = &mut self.segs[slot(chunk.source)];
                    seg.push(chunk.start_ms, &chunk.samples, &mut closed);
                    if !self.emit_finals(chunk.source, &mut closed, &tx) {
                        return;
                    }
                    chunk = match rx.try_recv() {
                        Ok(c) => c,
                        Err(TryRecvError::Empty) => continue 'outer,
                        Err(TryRecvError::Disconnected) => break 'outer,
                    };
                }
            }

            if let PartialDue::Now(i) = due {
                if !self.emit_partial(i, &tx) {
                    return;
                }
            }
        }

        // Input ended: flush open utterances.
        for source in [Source::Mic, Source::System] {
            self.segs[slot(source)].flush(&mut closed);
            if !self.emit_finals(source, &mut closed, &tx) {
                return;
            }
        }
        log::info!("STT worker done: {:?}, rtf≈{:.3}", self.stats, self.rtf.rtf);
    }

    /// Decode and send the finals in `closed`. Returns false if `tx` is gone.
    fn emit_finals(
        &mut self,
        source: Source,
        closed: &mut Vec<Closed>,
        tx: &Sender<Segment>,
    ) -> bool {
        for c in closed.drain(..) {
            let segs = self
                .finalize(slot(source), c, false)
                .expect("non-strict finalize doesn't fail");
            for seg in segs {
                self.stats.finals += 1;
                if tx.send(seg).is_err() {
                    log::warn!("segment receiver dropped; stopping STT worker");
                    return false;
                }
            }
        }
        true
    }

    /// Split (if a splitter is set), decode and number a closed utterance.
    /// With `strict`, a decode error is returned; otherwise it is logged and
    /// the piece counts as empty.
    fn finalize(&mut self, i: usize, c: Closed, strict: bool) -> Result<Vec<Segment>> {
        let len = c.samples.len();
        let cuts = self.split_points(&c.samples);
        let min_piece = ms_to_samples(self.split_opts.min_piece_ms) as usize;
        let plan = plan_pieces(len, cuts, min_piece);
        if plan.len() > 1 {
            self.stats.split_utterances += 1;
            self.stats.split_pieces += plan.len() as u64;
            log::debug!(
                "{:?} final {}–{} ms split into {} pieces at {:?} ms",
                self.segs[i].source(),
                c.start_ms,
                c.end_ms,
                plan.len(),
                plan[1..]
                    .iter()
                    .map(|&(from, _)| c.ms_at(from))
                    .collect::<Vec<_>>()
            );
        }
        let mut pieces = Vec::with_capacity(plan.len());
        for (from, to) in plan {
            let text = if strict {
                self.recognizer.recognize(&c.samples[from..to])?
            } else {
                self.decode(&c.samples[from..to], true)
            };
            pieces.push((from, to, text));
        }
        Ok(self.segs[i].final_segments(c, &pieces))
    }

    /// Ask the splitter (if any) where the speaker changes in `samples`.
    fn split_points(&mut self, samples: &[f32]) -> Vec<usize> {
        let Some(splitter) = &mut self.splitter else {
            return Vec::new();
        };
        if (samples.len() as u64) < ms_to_samples(self.split_opts.min_utterance_ms) {
            return Vec::new();
        }
        let t0 = Instant::now();
        let result = splitter.split(samples, self.split_opts.timeout(samples.len()));
        let waited = t0.elapsed();
        // Splitting competes with partials for the CPU like a decode does.
        self.stats.split_calls += 1;
        self.stats.split_secs += waited.as_secs_f64();
        self.partial_ready_at = self
            .partial_ready_at
            .max(Instant::now() + self.policy.cooldown(waited));
        let audio_s = samples.len() as f64 / SAMPLE_RATE as f64;
        match result {
            SplitResult::Cuts(cuts, took) => {
                log::debug!(
                    "splitter: {:.2} s audio in {:.3} s → {} cuts",
                    audio_s,
                    took.as_secs_f64(),
                    cuts.len()
                );
                cuts
            }
            SplitResult::Panicked => {
                self.stats.split_failures += 1;
                log::warn!("splitter panicked on {audio_s:.2} s of audio; not splitting");
                Vec::new()
            }
            SplitResult::TimedOut => {
                self.stats.split_failures += 1;
                log::warn!(
                    "splitter took longer than {:.2} s on {audio_s:.2} s of audio; not splitting",
                    waited.as_secs_f64()
                );
                Vec::new()
            }
            SplitResult::Busy => {
                self.stats.split_failures += 1;
                log::warn!("splitter still busy with an earlier utterance; not splitting");
                Vec::new()
            }
            SplitResult::Dead => {
                log::error!("splitter thread is gone; disabling speaker splitting");
                self.splitter = None;
                Vec::new()
            }
        }
    }

    /// Decode the open utterance of source slot `i` and send a partial.
    fn emit_partial(&mut self, i: usize, tx: &Sender<Segment>) -> bool {
        let Transcriber {
            recognizer,
            segs,
            rtf,
            stats,
            policy,
            partial_ready_at,
            ..
        } = self;
        let Some((audio, end)) = segs[i].partial_audio() else {
            return true;
        };
        let t0 = Instant::now();
        let text = recognizer.recognize(audio).unwrap_or_else(|e| {
            log::warn!("partial decode failed: {e:#}");
            String::new()
        });
        let took = t0.elapsed();
        rtf.update(audio.len(), took.as_secs_f64());
        *partial_ready_at = Instant::now() + policy.cooldown(took);
        stats.partial_decodes += 1;
        let Some(seg) = segs[i].apply_partial(end, &text) else {
            return true;
        };
        match tx.try_send(seg) {
            Ok(()) => stats.partials += 1,
            Err(TrySendError::Full(_)) => stats.partials_dropped += 1,
            Err(TrySendError::Disconnected(_)) => {
                log::warn!("segment receiver dropped; stopping STT worker");
                return false;
            }
        }
        true
    }

    fn decode(&mut self, samples: &[f32], is_final: bool) -> String {
        let t0 = Instant::now();
        let text = self.recognizer.recognize(samples).unwrap_or_else(|e| {
            log::error!("decode failed: {e:#}");
            String::new()
        });
        let secs = t0.elapsed().as_secs_f64();
        self.rtf.update(samples.len(), secs);
        log::debug!(
            "{} decode of {:.2} s audio took {:.3} s",
            if is_final { "final" } else { "partial" },
            samples.len() as f64 / SAMPLE_RATE as f64,
            secs
        );
        text
    }

    /// Which open utterance most needs a partial decode, and whether the CPU
    /// budget allows it now.
    fn pick_partial(&self, now: Instant) -> PartialDue {
        let mut best: Option<(usize, u64)> = None;
        for (i, seg) in self.segs.iter().enumerate() {
            let (Some(len), Some(new)) = (seg.open_len(), seg.new_since_partial()) else {
                continue;
            };
            if self.policy.wanted(len, new, &self.rtf) && best.is_none_or(|(_, b)| new > b) {
                best = Some((i, new));
            }
        }
        match best {
            None => PartialDue::No,
            Some((i, _)) => match self.partial_ready_at.checked_duration_since(now) {
                Some(wait) if !wait.is_zero() => PartialDue::After(wait),
                _ => PartialDue::Now(i),
            },
        }
    }
}

#[derive(Debug, PartialEq)]
enum PartialDue {
    /// Decode a partial for source slot `i` (if no input is queued).
    Now(usize),
    /// A partial is wanted but the CPU budget allows it only after this.
    After(Duration),
    No,
}

/// When to re-decode an open utterance. Each partial re-decodes the whole
/// utterance (CTC, not streaming), so cost grows with its length.
#[derive(Clone, Debug)]
struct PartialPolicy {
    /// Minimum new audio between partials of one utterance, in samples
    /// (0 = no partials).
    interval: u64,
    /// Fraction of wall time the worker may spend on partials (both sources
    /// together): after a partial that took `d`, wait `d * (1 - duty) / duty`.
    duty: f64,
    /// No partials for an utterance whose decode would take longer than this.
    max_cost_s: f64,
}

impl PartialPolicy {
    fn new(interval_ms: u64) -> Self {
        PartialPolicy {
            interval: ms_to_samples(interval_ms),
            duty: 0.3,
            max_cost_s: 1.5,
        }
    }

    /// Whether an utterance of `len` samples, `new` of them not yet covered
    /// by a partial, deserves a partial decode.
    fn wanted(&self, len: u64, new: u64, rtf: &RtfEstimate) -> bool {
        self.interval > 0 && new >= self.interval && rtf.predict(len as usize) <= self.max_cost_s
    }

    fn cooldown(&self, took: Duration) -> Duration {
        took.mul_f64((1.0 - self.duty) / self.duty)
    }
}

/// Running estimate of decode seconds per audio second.
#[derive(Clone, Debug)]
struct RtfEstimate {
    rtf: f64,
}

impl Default for RtfEstimate {
    fn default() -> Self {
        // Conservative start; converges after a couple of decodes.
        RtfEstimate { rtf: 0.1 }
    }
}

impl RtfEstimate {
    fn update(&mut self, samples: usize, secs: f64) {
        let audio = samples as f64 / SAMPLE_RATE as f64;
        if audio >= 1.0 {
            self.rtf = 0.7 * self.rtf + 0.3 * (secs / audio);
        }
    }

    fn predict(&self, samples: usize) -> f64 {
        self.rtf * samples as f64 / SAMPLE_RATE as f64
    }
}

#[derive(Debug, Default)]
struct Stats {
    finals: u64,
    partial_decodes: u64,
    partials: u64,
    partials_dropped: u64,
    split_calls: u64,
    split_secs: f64,
    split_failures: u64,
    split_utterances: u64,
    split_pieces: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segmenter::tests::{silence, tone, FakeVad};
    use std::sync::{Arc, Mutex};

    /// "Recognizes" one word per 100 ms of loud audio, so text grows with the
    /// utterance and clipping shows up as missing words. Optional sleep per
    /// call simulates decode cost.
    struct FakeRecognizer {
        delay: Duration,
        calls: Arc<Mutex<Vec<usize>>>,
    }

    impl Recognizer for FakeRecognizer {
        fn decode(&mut self, samples: &[f32]) -> Result<String> {
            let loud = samples.iter().filter(|x| x.abs() > 0.1).count();
            Ok(vec!["сөз"; loud / 1600].join(" "))
        }

        // Records the unpadded length of every utterance it is asked for.
        fn recognize(&mut self, samples: &[f32]) -> Result<String> {
            self.calls.lock().unwrap().push(samples.len());
            std::thread::sleep(self.delay);
            self.decode(samples)
        }
    }

    fn fake(partial_interval_ms: u64, delay_ms: u64) -> (Transcriber, Arc<Mutex<Vec<usize>>>) {
        let cfg = SttConfig {
            partial_interval_ms,
            max_segment_ms: 20_000,
            ..SttConfig::default()
        };
        let seg_cfg = SegmenterConfig::new(cfg.max_segment_ms);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let rec = FakeRecognizer {
            delay: Duration::from_millis(delay_ms),
            calls: calls.clone(),
        };
        let vads: [Box<dyn Vad>; 2] = [
            Box::new(FakeVad::new(&seg_cfg.vad)),
            Box::new(FakeVad::new(&seg_cfg.vad)),
        ];
        (
            Transcriber::from_parts(&cfg, seg_cfg, Box::new(rec), SttBackend::Sherpa, vads),
            calls,
        )
    }

    fn words(s: &Segment) -> usize {
        s.text.split_whitespace().count()
    }

    #[test]
    fn transcribe_buffer_splits_and_numbers() {
        let (mut t, _) = fake(700, 0);
        let audio = [
            silence(500),
            tone(1_000),
            silence(800),
            tone(2_000),
            silence(300),
        ]
        .concat();
        let segs = t.transcribe_buffer(Source::System, &audio).unwrap();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].id, "system-1");
        assert_eq!(segs[1].id, "system-2");
        assert_eq!(words(&segs[0]), 10);
        // The last utterance was still open at the end of the buffer: flushed.
        assert_eq!(words(&segs[1]), 20);
        assert!(segs.iter().all(|s| s.is_final && s.speaker.is_none()));
        assert!(segs[0].end_ms <= segs[1].start_ms);
        // Ids restart per call.
        let again = t.transcribe_buffer(Source::System, &audio).unwrap();
        assert_eq!(again[0].id, "system-1");
    }

    /// Feed `audio` in 100 ms chunks, optionally paced, and collect output.
    fn run_live(
        t: Transcriber,
        feeds: Vec<(Source, Vec<f32>)>,
        pace: Option<Duration>,
    ) -> Vec<Segment> {
        let (atx, arx) = crossbeam_channel::unbounded();
        let (stx, srx) = crossbeam_channel::unbounded();
        let h = t.spawn(arx, stx);
        let mut chunks: Vec<Vec<AudioChunk>> = feeds
            .into_iter()
            .map(|(source, audio)| {
                audio
                    .chunks(1600)
                    .enumerate()
                    .map(|(i, c)| AudioChunk {
                        source,
                        start_ms: i as u64 * 100,
                        samples: c.to_vec(),
                    })
                    .collect()
            })
            .collect();
        let n = chunks.iter().map(|c| c.len()).max().unwrap_or(0);
        for i in 0..n {
            for c in chunks.iter_mut() {
                if let Some(chunk) = c.get(i) {
                    atx.send(chunk.clone()).unwrap();
                }
            }
            if let Some(p) = pace {
                std::thread::sleep(p);
            }
        }
        drop(atx);
        h.join().unwrap();
        srx.try_iter().collect()
    }

    fn check_invariants(segs: &[Segment]) {
        use std::collections::HashMap;
        let mut finals: HashMap<&str, usize> = HashMap::new();
        let mut seen_final = std::collections::HashSet::new();
        for s in segs {
            assert!(
                !seen_final.contains(&s.id),
                "segment after final for {}",
                s.id
            );
            if s.is_final {
                *finals.entry(&s.id).or_default() += 1;
                seen_final.insert(s.id.clone());
            }
            let prefix = format!("{}-", s.source.as_str());
            assert!(s.id.starts_with(&prefix), "{} vs {:?}", s.id, s.source);
            assert!(s.end_ms >= s.start_ms);
        }
        for s in segs {
            assert_eq!(
                finals.get(s.id.as_str()),
                Some(&1),
                "id {} must have exactly one final",
                s.id
            );
        }
    }

    #[test]
    fn live_emits_partials_then_one_final_per_id() {
        let (t, _) = fake(300, 5);
        let mic = [
            silence(500),
            tone(3_000),
            silence(1_000),
            tone(1_500),
            silence(1_000),
        ]
        .concat();
        let sys = [silence(1_500), tone(2_000), silence(1_000)].concat();
        let segs = run_live(
            t,
            vec![(Source::Mic, mic), (Source::System, sys)],
            Some(Duration::from_millis(10)),
        );
        check_invariants(&segs);

        let finals: Vec<_> = segs.iter().filter(|s| s.is_final).collect();
        let mic_finals: Vec<_> = finals.iter().filter(|s| s.source == Source::Mic).collect();
        let sys_finals: Vec<_> = finals
            .iter()
            .filter(|s| s.source == Source::System)
            .collect();
        assert_eq!(mic_finals.len(), 2);
        assert_eq!(sys_finals.len(), 1);
        assert_eq!(
            (mic_finals[0].id.as_str(), mic_finals[1].id.as_str()),
            ("mic-1", "mic-2")
        );
        assert_eq!(sys_finals[0].id, "system-1");
        assert_eq!(words(mic_finals[0]), 30);
        assert_eq!(words(mic_finals[1]), 15);
        assert_eq!(words(sys_finals[0]), 20);

        // Partials came before the final, grew, and never exceeded it.
        let partials: Vec<_> = segs
            .iter()
            .filter(|s| !s.is_final && s.id == "mic-1")
            .collect();
        assert!(partials.len() >= 3, "only {} partials", partials.len());
        for w in partials.windows(2) {
            assert!(words(w[0]) <= words(w[1]));
        }
        assert!(partials
            .iter()
            .all(|p| p.start_ms == mic_finals[0].start_ms));
    }

    #[test]
    fn slow_decoder_backs_off_partials_but_keeps_finals() {
        // Each decode takes 150 ms; audio arrives at 5× real time.
        let (t, calls) = fake(200, 150);
        let mut mic = Vec::new();
        for _ in 0..6 {
            mic.extend(tone(2_500));
            mic.extend(silence(1_000));
        }
        let segs = run_live(t, vec![(Source::Mic, mic)], Some(Duration::from_millis(20)));
        check_invariants(&segs);
        let finals: Vec<_> = segs.iter().filter(|s| s.is_final).collect();
        assert_eq!(finals.len(), 6);
        assert!(finals.iter().all(|f| words(f) == 25));
        // 21 s of audio at 200 ms would be ~100 partial decodes if we didn't back off.
        let n = calls.lock().unwrap().len();
        assert!(n < 60, "{n} decodes");
    }

    #[test]
    fn no_partials_when_disabled() {
        let (t, calls) = fake(0, 0);
        let mic = [tone(2_000), silence(1_000)].concat();
        let segs = run_live(t, vec![(Source::Mic, mic)], None);
        assert_eq!(segs.len(), 1);
        assert!(segs[0].is_final);
        assert_eq!(calls.lock().unwrap().len(), 1);
    }

    #[test]
    fn disconnect_flushes_open_utterance() {
        let (t, _) = fake(700, 0);
        // Input stops mid-sentence.
        let segs = run_live(
            t,
            vec![(Source::System, [silence(300), tone(1_200)].concat())],
            None,
        );
        let f: Vec<_> = segs.iter().filter(|s| s.is_final).collect();
        assert_eq!(f.len(), 1);
        assert_eq!(words(f[0]), 12);
    }

    #[test]
    fn worker_stops_when_output_dropped() {
        let (t, _) = fake(0, 0);
        let (atx, arx) = crossbeam_channel::unbounded();
        let (stx, srx) = crossbeam_channel::bounded(1);
        drop(srx);
        let h = t.spawn(arx, stx);
        for (i, c) in [tone(1_000), silence(1_000)]
            .concat()
            .chunks(1600)
            .enumerate()
        {
            let _ = atx.send(AudioChunk {
                source: Source::Mic,
                start_ms: i as u64 * 100,
                samples: c.to_vec(),
            });
        }
        // Thread exits on the first final even though input is still open.
        h.join().unwrap();
        drop(atx);
    }

    #[test]
    fn partial_policy_respects_interval_cost_and_budget() {
        let p = PartialPolicy::new(700);
        let rtf = RtfEstimate { rtf: 0.05 };
        let sec = SAMPLE_RATE as u64;
        assert!(!p.wanted(2 * sec, sec / 2, &rtf));
        assert!(p.wanted(2 * sec, sec, &rtf));
        assert!(p.wanted(20 * sec, sec, &rtf));
        // Too expensive (40 s × 0.05 = 2 s per decode): no partials at all.
        assert!(!p.wanted(40 * sec, 10 * sec, &rtf));
        assert!(!PartialPolicy::new(0).wanted(2 * sec, 2 * sec, &rtf));
        // 30 % duty: a 300 ms decode buys 700 ms of rest.
        assert_eq!(
            p.cooldown(Duration::from_millis(300)),
            Duration::from_millis(700)
        );
    }

    #[test]
    fn pick_partial_waits_for_budget() {
        let (mut t, _) = fake(300, 0);
        let mut closed = Vec::new();
        t.segs[0].push(0, &tone(1_500), &mut closed);
        let now = Instant::now();
        assert_eq!(t.pick_partial(now), PartialDue::Now(0));
        t.partial_ready_at = now + Duration::from_millis(200);
        assert_eq!(
            t.pick_partial(now),
            PartialDue::After(Duration::from_millis(200))
        );
        assert_eq!(
            t.pick_partial(now + Duration::from_millis(250)),
            PartialDue::Now(0)
        );
        // The source with more undecoded audio goes first.
        t.segs[1].push(0, &tone(2_500), &mut closed);
        assert_eq!(
            t.pick_partial(now + Duration::from_millis(250)),
            PartialDue::Now(1)
        );
    }

    // ---- speaker-change splitting ----

    /// "Speaker B" talks louder than the 0.3 test tone.
    fn loud_tone(ms: u64) -> Vec<f32> {
        tone(ms).iter().map(|x| x * 5.0 / 3.0).collect()
    }

    /// Voice-like for the fake VAD (mean |x| > 0.05) but no words for the
    /// fake recognizer (|x| <= 0.1): a murmur that decodes to nothing.
    fn murmur(ms: u64) -> Vec<f32> {
        tone(ms).iter().map(|x| x * 0.08 / 0.3).collect()
    }

    /// Fake change-point detector: cuts where the level jumps across 0.4 or
    /// 0.1, i.e. between murmur, speaker A (0.3) and speaker B (0.5).
    fn level_splitter(calls: Arc<Mutex<Vec<usize>>>) -> Splitter {
        Box::new(move |s: &[f32]| {
            calls.lock().unwrap().push(s.len());
            let class = |x: f32| (x.abs() > 0.1) as u8 + (x.abs() > 0.4) as u8;
            let mut cuts = Vec::new();
            let mut prev: Option<u8> = None;
            for (i, &x) in s.iter().enumerate() {
                if x == 0.0 {
                    continue; // silence doesn't change the speaker
                }
                let c = class(x);
                if prev.is_some_and(|p| p != c) {
                    cuts.push(i);
                }
                prev = Some(c);
            }
            cuts
        })
    }

    #[test]
    fn split_final_gets_new_ids_and_exact_timestamps() {
        let (mut t, _) = fake(700, 0);
        let calls = Arc::new(Mutex::new(Vec::new()));
        t.set_splitter(level_splitter(calls.clone()));
        // A speaks 2 s, B answers with no pause for 1.5 s; later A again alone.
        let audio = [
            silence(500),
            tone(2_000),
            loud_tone(1_500),
            silence(1_000),
            tone(1_000),
            silence(1_000),
        ]
        .concat();
        let segs = t.transcribe_buffer(Source::Mic, &audio).unwrap();
        let summary: Vec<_> = segs.iter().map(|s| (s.id.as_str(), words(s))).collect();
        assert_eq!(summary, [("mic-1", 20), ("mic-2", 15), ("mic-3", 10)]);
        // The cut lands exactly where B starts, on the chunk timeline.
        assert_eq!(segs[0].end_ms, 2_500);
        assert_eq!(segs[1].start_ms, 2_500);
        assert!(segs[0].start_ms < 500 && segs[1].end_ms > 4_000);
        assert!(segs[1].end_ms <= segs[2].start_ms);
        // The 1 s utterance was too short to ask the splitter about.
        assert_eq!(calls.lock().unwrap().len(), 1);
    }

    #[test]
    fn split_pieces_follow_timeline_gaps() {
        let (mut t, _) = fake(700, 0);
        t.set_splitter(level_splitter(Arc::default()));
        // B's chunks arrive after a 5 s hole in the capture timeline.
        let (atx, arx) = crossbeam_channel::unbounded();
        let (stx, srx) = crossbeam_channel::unbounded();
        let h = t.spawn(arx, stx);
        let mut ms = 0;
        for part in [
            [silence(500), tone(2_000)].concat(),
            [loud_tone(1_500), silence(1_000)].concat(),
        ] {
            for c in part.chunks(1600) {
                atx.send(AudioChunk {
                    source: Source::System,
                    start_ms: ms,
                    samples: c.to_vec(),
                })
                .unwrap();
                ms += 100;
            }
            ms += 5_000;
        }
        drop(atx);
        h.join().unwrap();
        let segs: Vec<Segment> = srx.try_iter().filter(|s| s.is_final).collect();
        assert_eq!(segs.len(), 2, "{segs:?}");
        assert_eq!(
            (segs[0].id.as_str(), segs[1].id.as_str()),
            ("system-1", "system-2")
        );
        assert_eq!(segs[1].start_ms, 7_500);
        assert!(
            segs[1].end_ms > 9_000 && segs[1].end_ms < 9_700,
            "{}",
            segs[1].end_ms
        );
    }

    #[test]
    fn tiny_split_pieces_are_merged() {
        let (mut t, calls) = fake(700, 0);
        let seen = Arc::new(Mutex::new(0usize));
        let seen2 = seen.clone();
        t.set_splitter(Box::new(move |s: &[f32]| {
            *seen2.lock().unwrap() = s.len();
            let half = s.len() / 2;
            // 3 ms at the start, a 10 ms sliver in the middle, a real cut,
            // and an out-of-range offset.
            vec![50, half, half + 160, s.len() + 10]
        }));
        let audio = [silence(300), tone(3_000), silence(1_000)].concat();
        let segs = t.transcribe_buffer(Source::Mic, &audio).unwrap();
        assert_eq!(segs.len(), 2, "{segs:?}");
        // Two decodes covering the whole utterance, split at `half`.
        let calls = calls.lock().unwrap().clone();
        let len = *seen.lock().unwrap();
        assert_eq!(calls, [len / 2, len - len / 2]);
        assert_eq!(segs[0].end_ms, segs[1].start_ms);
        // (The fake counts whole 100 ms runs, so a mid-word cut can lose one.)
        assert!((29..=30).contains(&(words(&segs[0]) + words(&segs[1]))));
    }

    #[test]
    fn empty_first_piece_keeps_the_original_id() {
        let (mut t, _) = fake(300, 0);
        t.set_splitter(level_splitter(Arc::default()));
        let mic = [silence(300), murmur(1_500), tone(1_500), silence(1_000)].concat();
        let segs = run_live(
            t,
            vec![(Source::Mic, mic.clone())],
            Some(Duration::from_millis(20)),
        );
        check_invariants(&segs);
        // A partial with words was shown for mic-1 before the utterance closed.
        assert!(segs
            .iter()
            .any(|s| !s.is_final && s.id == "mic-1" && words(s) > 0));
        let finals: Vec<_> = segs.iter().filter(|s| s.is_final).collect();
        assert_eq!(finals.len(), 2, "{finals:?}");
        // mic-1 was only murmur: it's closed with empty text so the UI drops it…
        assert_eq!(
            (finals[0].id.as_str(), finals[0].text.as_str()),
            ("mic-1", "")
        );
        // …and the words move to a new id.
        assert_eq!((finals[1].id.as_str(), words(finals[1])), ("mic-2", 15));
        assert_eq!(finals[0].end_ms, finals[1].start_ms);

        // Offline (no partials, so no id yet): the empty piece is just skipped.
        let (mut t, _) = fake(300, 0);
        t.set_splitter(level_splitter(Arc::default()));
        let segs = t.transcribe_buffer(Source::Mic, &mic).unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!((segs[0].id.as_str(), words(&segs[0])), ("mic-1", 15));
        assert_eq!(segs[0].start_ms, 1_800);
    }

    #[test]
    fn panicking_splitter_falls_back_to_no_split() {
        let (mut t, _) = fake(700, 0);
        let mut n = 0;
        t.set_splitter(Box::new(move |s: &[f32]| {
            n += 1;
            if n == 1 {
                panic!("change-point detector bug");
            }
            vec![s.len() / 2]
        }));
        let utt = [tone(3_000), silence(1_000)].concat();
        let audio = [utt.clone(), utt].concat();
        let (atx, arx) = crossbeam_channel::unbounded();
        let (stx, srx) = crossbeam_channel::unbounded();
        let h = t.spawn(arx, stx);
        for (i, c) in audio.chunks(1600).enumerate() {
            let chunk = AudioChunk {
                source: Source::Mic,
                start_ms: i as u64 * 100,
                samples: c.to_vec(),
            };
            atx.send(chunk).unwrap();
        }
        drop(atx);
        h.join().expect("worker survives a panicking splitter");
        let segs: Vec<Segment> = srx.try_iter().filter(|s| s.is_final).collect();
        let summary: Vec<_> = segs.iter().map(|s| (s.id.as_str(), words(s))).collect();
        // First utterance unsplit; the splitter keeps working afterwards.
        assert_eq!(summary.len(), 3, "{summary:?}");
        assert_eq!(summary[0], ("mic-1", 30));
        assert!((29..=30).contains(&(words(&segs[1]) + words(&segs[2]))));
        assert_eq!(segs[1].end_ms, segs[2].start_ms);
    }

    #[test]
    fn slow_splitter_times_out_without_stalling() {
        let (mut t, _) = fake(700, 0);
        t.set_split_options(SplitOptions {
            timeout_ms: 50,
            timeout_ratio: 0.0,
            ..SplitOptions::default()
        });
        t.set_splitter(Box::new(|s: &[f32]| {
            std::thread::sleep(Duration::from_millis(400));
            vec![s.len() / 2]
        }));
        let utt = [tone(3_000), silence(1_000)].concat();
        let t0 = Instant::now();
        let segs = t
            .transcribe_buffer(Source::Mic, &[utt.clone(), utt].concat())
            .unwrap();
        // Both unsplit: the first timed out, the second found it still busy.
        assert_eq!(segs.len(), 2, "{segs:?}");
        assert!(segs.iter().all(|s| words(s) == 30));
        assert!(
            t0.elapsed() < Duration::from_millis(350),
            "{:?}",
            t0.elapsed()
        );
    }

    #[test]
    fn split_options_threshold() {
        let (mut t, _) = fake(700, 0);
        let calls = Arc::new(Mutex::new(Vec::new()));
        t.set_splitter(level_splitter(calls.clone()));
        t.set_split_options(SplitOptions {
            min_utterance_ms: 5_000,
            ..SplitOptions::default()
        });
        let audio = [silence(300), tone(2_000), loud_tone(1_500), silence(1_000)].concat();
        let segs = t.transcribe_buffer(Source::Mic, &audio).unwrap();
        assert_eq!(segs.len(), 1);
        assert!(calls.lock().unwrap().is_empty());
    }
}
