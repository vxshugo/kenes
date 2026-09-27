//! One live meeting: capture → speech recognition → events + storage.
//!
//! Threads per session:
//! - the session worker (this module): loads models, then pumps audio chunks to
//!   the transcriber and segments to the event sink and the store;
//! - capture threads (owned by `kenes-audio`);
//! - the transcriber thread (owned by `kenes-stt`).
//!
//! Final segments get a speaker label (see [`crate::diarize`]) before they are
//! stored and emitted; when the session ends, labels are re-clustered once.
//!
//! Stopping drops the audio side first, so the transcriber flushes the open
//! utterance as a final segment before the worker exits.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use crossbeam_channel::{after, bounded, select, unbounded, Receiver};
use kenes_audio::{CaptureConfig, DeviceSel};
use kenes_stt::{SttConfig, Transcriber};
use kenes_types::{AudioChunk, PipelineEvent, Segment, SessionState, Source, SpeakerChange, SAMPLE_RATE};

use crate::diarize::{self, Diarizer};
use crate::settings::{CoreSettings, MicMode};
use crate::store::Store;

pub type EventSink = Arc<dyn Fn(PipelineEvent) + Send + Sync>;

/// Where a session's audio comes from.
#[derive(Clone, Debug, Default)]
pub enum AudioInput {
    /// Live microphone and/or system audio, per the settings.
    #[default]
    Capture,
    /// Recorded WAV files (e.g. from `kenes-rec`) replayed through the same pipeline,
    /// `speed` times faster than real time. The session ends when the files do.
    Files { mic: Option<PathBuf>, system: Option<PathBuf>, speed: f32 },
}

pub const VAD_MODEL: &str = "silero-vad";
const SPEAKER_MODEL: &str = "speaker-embedding";

/// Level meter update period, in samples.
const LEVEL_WINDOW: usize = SAMPLE_RATE as usize / 10;
/// How long `stop` waits for a clean flush before detaching the worker.
const STOP_TIMEOUT: Duration = Duration::from_secs(15);

struct Running {
    meeting_id: String,
    stop: Arc<AtomicBool>,
    done: Receiver<()>,
}

pub struct SessionManager {
    store: Arc<Store>,
    sink: EventSink,
    models_dir: PathBuf,
    current: Mutex<Option<Running>>,
    /// Last status the pipeline reported, so a reloaded UI can catch up.
    state: Arc<Mutex<SessionState>>,
}

/// The running session, for a UI that (re)attaches mid-meeting.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveSession {
    pub meeting_id: String,
    pub state: SessionState,
}

impl SessionManager {
    pub fn new(store: Arc<Store>, models_dir: PathBuf, sink: EventSink) -> Self {
        let state = Arc::new(Mutex::new(SessionState::Idle));
        let tracked = state.clone();
        let sink: EventSink = Arc::new(move |ev| {
            if let PipelineEvent::Status { state, .. } = &ev {
                *tracked.lock().unwrap_or_else(|e| e.into_inner()) = *state;
            }
            sink(ev)
        });
        Self { store, sink, models_dir, current: Mutex::new(None), state }
    }

    pub fn live(&self) -> Option<LiveSession> {
        let meeting_id = self.current_meeting()?;
        let state = *self.state.lock().unwrap_or_else(|e| e.into_inner());
        Some(LiveSession { meeting_id, state })
    }

    pub fn current_meeting(&self) -> Option<String> {
        let mut cur = self.current.lock().unwrap();
        // A worker that died on its own (e.g. model load failed) no longer counts.
        if cur.as_ref().is_some_and(|r| r.done.try_recv().is_ok()) {
            if let Some(r) = cur.take() {
                let _ = self.store.end_meeting(&r.meeting_id);
            }
        }
        cur.as_ref().map(|r| r.meeting_id.clone())
    }

    /// Creates the meeting and starts the pipeline in the background.
    /// Returns the meeting id immediately; progress arrives as events.
    pub fn start(&self, title: &str, context: &str, settings: CoreSettings) -> Result<String> {
        self.start_with_input(title, context, settings, AudioInput::Capture)
    }

    pub fn start_with_input(
        &self,
        title: &str,
        context: &str,
        mut settings: CoreSettings,
        input: AudioInput,
    ) -> Result<String> {
        if self.current_meeting().is_some() {
            bail!("a session is already running");
        }
        if let AudioInput::Files { mic, system, .. } = &input {
            // The pipeline's source switches follow the files given.
            settings.capture_mic = mic.is_some();
            settings.capture_system = system.is_some();
        }
        if !settings.capture_mic && !settings.capture_system {
            bail!("both microphone and system audio capture are disabled");
        }
        let meeting = self.store.create_meeting(title, context)?;
        let stop = Arc::new(AtomicBool::new(false));
        let (done_tx, done_rx) = bounded(1);

        let worker = Worker {
            meeting_id: meeting.id.clone(),
            settings,
            input,
            models_dir: self.models_dir.clone(),
            store: self.store.clone(),
            sink: self.sink.clone(),
            stop: stop.clone(),
        };
        std::thread::Builder::new()
            .name("kenes-session".into())
            .spawn(move || {
                let sink = worker.sink.clone();
                match worker.run() {
                    Ok(()) => sink(status(SessionState::Idle, None)),
                    Err(e) => {
                        log::error!("session failed: {e:#}");
                        sink(PipelineEvent::Error { message: format!("{e:#}") });
                        sink(status(SessionState::Error, Some(format!("{e:#}"))));
                    }
                }
                let _ = done_tx.send(());
            })
            .context("spawning session thread")?;

        *self.current.lock().unwrap() = Some(Running { meeting_id: meeting.id.clone(), stop, done: done_rx });
        Ok(meeting.id)
    }

    /// Stops the running session, waiting for the last utterance to be transcribed.
    pub fn stop(&self) -> Result<()> {
        let Some(running) = self.current.lock().unwrap().take() else { return Ok(()) };
        running.stop.store(true, Ordering::SeqCst);
        if running.done.recv_timeout(STOP_TIMEOUT).is_err() {
            // Most likely stuck in a model download; it checks the flag when done.
            log::warn!("session worker did not stop within {STOP_TIMEOUT:?}; detaching");
        }
        self.store.end_meeting(&running.meeting_id)
    }
}

fn status(state: SessionState, message: Option<String>) -> PipelineEvent {
    PipelineEvent::Status { state, message }
}

struct Worker {
    meeting_id: String,
    settings: CoreSettings,
    input: AudioInput,
    models_dir: PathBuf,
    store: Arc<Store>,
    sink: EventSink,
    stop: Arc<AtomicBool>,
}

impl Worker {
    fn stopped(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    fn progress_sink(&self, model: &str) -> impl FnMut(f32) {
        let sink = self.sink.clone();
        let name = model.to_owned();
        let mut last = -1.0f32;
        move |p| {
            // Throttle to whole percents; downloads report far more often.
            if p >= 1.0 || p - last >= 0.01 {
                last = p;
                sink(PipelineEvent::ModelProgress { model: name.clone(), progress: p });
            }
        }
    }

    /// Speaker labels are a bonus: if the model can't load, the session runs without them.
    /// Also returns the speaker model path when it loaded, for the change-point splitter.
    fn load_diarizer(&self) -> (Diarizer, Option<PathBuf>) {
        let needs_model = self.settings.mic_mode == MicMode::Room || self.settings.capture_system;
        let embedder = if needs_model {
            let loaded = kenes_speakers::ensure_speaker_model(&self.models_dir, &mut self.progress_sink(SPEAKER_MODEL))
                .and_then(|path| Ok((kenes_speakers::Embedder::new(&path, 1)?, path)));
            match loaded {
                Ok(e) => Some(e),
                Err(e) => {
                    log::error!("speaker model unavailable: {e:#}");
                    (self.sink)(PipelineEvent::Error {
                        message: format!("разделение по спикерам недоступно: {e:#}"),
                    });
                    None
                }
            }
        } else {
            None
        };
        let voiceprint = match diarize::load_voiceprint(&self.store) {
            Ok(vp) => vp,
            Err(e) => {
                log::warn!("reading voiceprint: {e:#}");
                None
            }
        };
        let (embedder, path) = embedder.unzip();
        (Diarizer::new(embedder, self.settings.mic_mode, voiceprint), path)
    }

    fn run(self) -> Result<()> {
        (self.sink)(status(SessionState::Loading, Some("Загрузка моделей распознавания".into())));
        for model in [VAD_MODEL, self.settings.stt_model.as_str()] {
            if self.stopped() {
                return Ok(());
            }
            kenes_stt::ensure_model(model, &self.models_dir, &mut self.progress_sink(model))
                .with_context(|| format!("загрузка модели {model}"))?;
        }

        let mut transcriber = Transcriber::new(SttConfig {
            model_id: self.settings.stt_model.clone(),
            models_dir: self.models_dir.clone(),
            num_threads: self.settings.num_threads,
            partial_interval_ms: 700,
            max_segment_ms: 20_000,
        })
        .context("инициализация распознавания")?;
        let (mut diarizer, speaker_model) = self.load_diarizer();
        if let Some(path) = speaker_model {
            // People often answer without a pause, so the VAD glues their turns together;
            // cut each long utterance where the voice changes before it is decoded.
            match kenes_speakers::Embedder::new(&path, 1) {
                Ok(mut embedder) => {
                    let cfg = kenes_speakers::ClusterConfig::default();
                    transcriber.set_splitter(Box::new(move |samples| {
                        kenes_speakers::change_points(&mut embedder, samples, &cfg)
                    }));
                }
                Err(e) => log::warn!("speaker change detection disabled: {e:#}"),
            }
        }
        if self.stopped() {
            return Ok(());
        }

        let (stt_tx, stt_rx) = unbounded::<AudioChunk>();
        let (seg_tx, seg_rx) = unbounded::<Segment>();
        let stt_thread = transcriber.spawn(stt_rx, seg_tx);

        let (audio_tx, audio_rx) = unbounded::<AudioChunk>();
        let capture = match &self.input {
            AudioInput::Capture => {
                let sel = |on: bool, dev: &Option<String>| on.then(|| DeviceSel::from_option(dev.clone()));
                let handle = kenes_audio::start_capture(
                    CaptureConfig {
                        mic: sel(self.settings.capture_mic, &self.settings.mic_device),
                        system: sel(self.settings.capture_system, &self.settings.system_device),
                    },
                    audio_tx,
                )
                .context("запуск захвата звука")?;
                Audio::Live(handle)
            }
            AudioInput::Files { mic, system, speed } => {
                Audio::Replay(replay::start(mic.as_deref(), system.as_deref(), *speed, audio_tx, self.stop.clone())?)
            }
        };
        let capture_errors = match &capture {
            Audio::Live(h) => h.errors(),
            Audio::Replay(_) => crossbeam_channel::never(),
        };
        (self.sink)(status(SessionState::Running, None));

        let mut levels = Levels::default();
        let mut forward = Some(stt_tx);
        loop {
            if self.stopped() {
                break;
            }
            select! {
                recv(audio_rx) -> msg => match msg {
                    Ok(chunk) => {
                        for ev in levels.push(&chunk) {
                            (self.sink)(ev);
                        }
                        diarizer.push_audio(&chunk);
                        if let Some(tx) = &forward {
                            if tx.send(chunk).is_err() {
                                // The transcriber died; keep draining audio so capture isn't blocked.
                                forward = None;
                                (self.sink)(PipelineEvent::Error { message: "распознавание остановилось".into() });
                            }
                        }
                    }
                    Err(_) => {
                        if matches!(capture, Audio::Live(_)) {
                            (self.sink)(PipelineEvent::Error { message: "захват звука прервался".into() });
                        }
                        break;
                    }
                },
                recv(capture_errors) -> msg => {
                    // One source died (e.g. the recorder crashed); the other keeps running.
                    if let Ok(e) = msg {
                        (self.sink)(PipelineEvent::Error { message: format!("захват ({}): {}", e.source.as_str(), e.message) });
                    }
                },
                recv(seg_rx) -> msg => {
                    if let Ok(seg) = msg {
                        self.emit_segment(&mut diarizer, seg);
                    }
                },
                recv(after(Duration::from_millis(200))) -> _ => {},
            }
        }

        match capture {
            Audio::Live(h) => h.stop(),
            Audio::Replay(t) => {
                let _ = t.join();
            }
        }
        drop(forward);
        // Remaining audio already sent is transcribed; the final flush arrives here.
        for seg in seg_rx.iter() {
            self.emit_segment(&mut diarizer, seg);
        }
        let _ = stt_thread.join();

        let changes = diarizer.finish();
        if !changes.is_empty() {
            log::info!("re-clustering changed {} speaker labels", changes.len());
            if let Err(e) = self.store.set_segment_speakers(&self.meeting_id, &changes) {
                log::error!("saving re-clustered speakers: {e:#}");
            }
            let changes = changes
                .into_iter()
                .map(|(segment_id, speaker)| SpeakerChange { segment_id, speaker })
                .collect();
            (self.sink)(PipelineEvent::SpeakersRelabeled { changes });
        }
        Ok(())
    }

    fn emit_segment(&self, diarizer: &mut Diarizer, mut seg: Segment) {
        if seg.is_final && is_noise(&seg) {
            // An empty final tells the UI to drop the partial (docs/CONTRACT.md).
            seg.text.clear();
        }
        if seg.is_final && !seg.text.is_empty() {
            let embedding = diarizer.label(&mut seg);
            if let Err(e) = self.store.save_segment(&self.meeting_id, &seg) {
                log::error!("saving segment {}: {e:#}", seg.id);
            }
            if let Some(e) = embedding {
                if let Err(err) = self.store.save_embedding(&self.meeting_id, &seg.id, &e) {
                    log::error!("saving embedding {}: {err:#}", seg.id);
                }
            }
        }
        (self.sink)(PipelineEvent::Segment(seg));
    }
}

enum Audio {
    Live(kenes_audio::CaptureHandle),
    Replay(std::thread::JoinHandle<()>),
}

mod replay {
    //! Feeds WAV files into the pipeline as if they were being captured.

    use std::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use anyhow::{Context, Result};
    use crossbeam_channel::Sender;
    use kenes_audio::CHUNK_SAMPLES;
    use kenes_types::{AudioChunk, Source, SAMPLE_RATE};

    pub fn start(
        mic: Option<&Path>,
        system: Option<&Path>,
        speed: f32,
        tx: Sender<AudioChunk>,
        stop: Arc<AtomicBool>,
    ) -> Result<std::thread::JoinHandle<()>> {
        let load = |p: Option<&Path>| -> Result<Vec<f32>> {
            p.map_or(Ok(Vec::new()), |p| kenes_audio::wav::read(p).with_context(|| format!("чтение {}", p.display())))
        };
        let (mic, system) = (load(mic)?, load(system)?);
        let speed = if speed > 0.0 { speed } else { 1.0 };
        let thread = std::thread::Builder::new().name("kenes-replay".into()).spawn(move || {
            let started = Instant::now();
            let len = mic.len().max(system.len());
            for at in (0..len).step_by(CHUNK_SAMPLES) {
                if stop.load(Ordering::SeqCst) {
                    return;
                }
                let start_ms = at as u64 * 1000 / SAMPLE_RATE as u64;
                for (source, samples) in [(Source::Mic, &mic), (Source::System, &system)] {
                    if at < samples.len() {
                        let end = (at + CHUNK_SAMPLES).min(samples.len());
                        let chunk = AudioChunk { source, start_ms, samples: samples[at..end].to_vec() };
                        if tx.send(chunk).is_err() {
                            return;
                        }
                    }
                }
                // Pace like a live stream so the speaker ring buffer still holds each final's audio.
                let due = Duration::from_secs_f64((at + CHUNK_SAMPLES) as f64 / SAMPLE_RATE as f64 / speed as f64);
                if let Some(wait) = due.checked_sub(started.elapsed()) {
                    std::thread::sleep(wait);
                }
            }
        })?;
        Ok(thread)
    }
}

/// Short noises (a cough, a click) sometimes decode to a lone letter.
fn is_noise(seg: &Segment) -> bool {
    let text = seg.text.trim();
    text.chars().count() <= 1 && seg.end_ms.saturating_sub(seg.start_ms) < 800
}

/// RMS per source over fixed windows, for the UI level meters.
#[derive(Default)]
struct Levels {
    mic: (f64, usize),
    system: (f64, usize),
}

impl Levels {
    fn push(&mut self, chunk: &AudioChunk) -> Vec<PipelineEvent> {
        let acc = match chunk.source {
            Source::Mic => &mut self.mic,
            Source::System => &mut self.system,
        };
        let mut out = Vec::new();
        for &s in &chunk.samples {
            acc.0 += (s as f64) * (s as f64);
            acc.1 += 1;
            if acc.1 == LEVEL_WINDOW {
                let rms = (acc.0 / acc.1 as f64).sqrt() as f32;
                out.push(PipelineEvent::Level { source: chunk.source, rms });
                *acc = (0.0, 0);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lone_letters_are_noise() {
        let seg = |text: &str, ms: u64| Segment {
            id: "mic-1".into(),
            source: Source::Mic,
            speaker: None,
            start_ms: 0,
            end_ms: ms,
            text: text.into(),
            is_final: true,
        };
        assert!(is_noise(&seg(" а ", 400)));
        assert!(is_noise(&seg("", 400)));
        assert!(!is_noise(&seg("а", 1500)));
        assert!(!is_noise(&seg("да", 300)));
    }

    #[test]
    fn levels_emit_per_window() {
        let mut l = Levels::default();
        let chunk = AudioChunk { source: Source::Mic, start_ms: 0, samples: vec![0.5; LEVEL_WINDOW * 2 + 10] };
        let evs = l.push(&chunk);
        assert_eq!(evs.len(), 2);
        match &evs[0] {
            PipelineEvent::Level { source, rms } => {
                assert_eq!(*source, Source::Mic);
                assert!((rms - 0.5).abs() < 1e-6);
            }
            other => panic!("unexpected {other:?}"),
        }
        // The 10 leftover samples carry over into the next window.
        let evs = l.push(&AudioChunk { source: Source::Mic, start_ms: 0, samples: vec![0.0; LEVEL_WINDOW - 10] });
        assert_eq!(evs.len(), 1);
    }
}
