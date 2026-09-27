//! Capture of the microphone and system audio as two separate 16 kHz mono streams.
//!
//! [`start_capture`] opens one stream per enabled [`Source`] and sends
//! [`AudioChunk`]s (512 samples = 32 ms, `f32` in `[-1, 1]`) into the channel you
//! pass. Both streams are stamped on one session clock that starts inside
//! [`start_capture`]. Runtime failures (a device disappearing, the audio server
//! going away) end that stream and are reported through [`CaptureHandle::errors`] /
//! [`CaptureHandle::take_error`]; the other stream keeps running.
//!
//! Backends: Linux uses `parec` (or `pw-record`) subprocesses and `pactl`; macOS
//! uses cpal, including its Core Audio process-tap loopback for system audio.

use std::fmt;
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use crossbeam_channel::{Receiver, Sender};

pub use kenes_types::{AudioChunk, DeviceInfo, DeviceKind, Source, SAMPLE_RATE};

mod chunker;
pub mod pcm;
pub use chunker::CHUNK_SAMPLES;
pub mod wav;

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
use linux as backend;

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
use macos as backend;

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod unsupported;
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
use unsupported as backend;

/// How long [`start_capture`] waits for every stream to deliver its first audio.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(3);

/// Which device to capture from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum DeviceSel {
    /// The system default, resolved when capture starts. A later change of the
    /// default device does not move a running stream; restart capture for that.
    #[default]
    Default,
    /// A [`DeviceInfo::id`] from [`list_devices`].
    Id(String),
}

impl DeviceSel {
    /// `None` → `Default`, `Some(id)` → `Id(id)`; matches the `micDevice`/`systemDevice` settings.
    pub fn from_option(id: Option<String>) -> Self {
        id.map_or(Self::Default, Self::Id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureConfig {
    /// `None`: don't capture the mic.
    pub mic: Option<DeviceSel>,
    /// `None`: don't capture system audio.
    pub system: Option<DeviceSel>,
}

impl Default for CaptureConfig {
    /// Both sources from their default devices.
    fn default() -> Self {
        Self {
            mic: Some(DeviceSel::Default),
            system: Some(DeviceSel::Default),
        }
    }
}

/// A stream that stopped on its own. It will not deliver more chunks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureError {
    pub source: Source,
    pub message: String,
}

impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} capture stopped: {}",
            self.source.as_str(),
            self.message
        )
    }
}

impl std::error::Error for CaptureError {}

/// What was actually opened for one source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamInfo {
    pub source: Source,
    /// Resolved device id/name (for display and logs).
    pub device: String,
    /// `"parec"`, `"pw-record"` or `"cpal"`.
    pub backend: &'static str,
}

/// Lists capture devices: microphones ([`DeviceKind::Input`]) and loopbacks of
/// outputs ([`DeviceKind::Monitor`]).
pub fn list_devices() -> Result<Vec<DeviceInfo>> {
    backend::list_devices()
}

/// Starts capturing and returns once every enabled stream is delivering audio
/// (or has had 3 s to do so; typically it takes 100-200 ms). A stream that fails during startup
/// makes the whole call fail, and nothing keeps running.
///
/// `tx` should be unbounded (or generously bounded) and drained promptly: a
/// blocked send stalls the reader, which makes the audio server drop audio.
/// A stream ends when the handle is stopped/dropped, when it fails (reported on
/// [`CaptureHandle::errors`]), or when the receiving side of `tx` is dropped. The
/// streams hold the only clones of `tx`, so its receiver disconnects once every
/// stream has ended.
pub fn start_capture(cfg: CaptureConfig, tx: Sender<AudioChunk>) -> Result<CaptureHandle> {
    let wanted: Vec<(Source, DeviceSel)> = [(Source::Mic, cfg.mic), (Source::System, cfg.system)]
        .into_iter()
        .filter_map(|(source, sel)| sel.map(|sel| (source, sel)))
        .collect();
    if wanted.is_empty() {
        bail!("capture config enables neither the mic nor system audio");
    }

    let session_start = Instant::now();
    let (err_tx, err_rx) = crossbeam_channel::unbounded();
    let (started_tx, started_rx) = crossbeam_channel::unbounded();
    let mut handle = CaptureHandle {
        session_start,
        streams: Vec::new(),
        workers: Vec::new(),
        errors: err_rx,
        _errors_tx: err_tx.clone(),
    };
    for (source, sel) in wanted {
        let ctx = StreamCtx {
            source,
            session_start,
            tx: tx.clone(),
            errors: err_tx.clone(),
            started: started_tx.clone(),
        };
        // On error `handle` is dropped here, stopping the streams opened so far.
        let (worker, info) = backend::open_stream(&sel, ctx)?;
        log::info!(
            "capturing {} from {} via {}",
            source.as_str(),
            info.device,
            info.backend
        );
        handle.workers.push(worker);
        handle.streams.push(info);
    }
    drop(started_tx);
    handle.wait_started(&started_rx)?;
    Ok(handle)
}

/// Everything a backend stream needs to report back.
pub(crate) struct StreamCtx {
    pub source: Source,
    pub session_start: Instant,
    pub tx: Sender<AudioChunk>,
    pub errors: Sender<CaptureError>,
    /// Signalled once, with the session time (ms) of the first sample.
    pub started: Sender<(Source, u64)>,
}

impl StreamCtx {
    pub(crate) fn fail(&self, message: impl Into<String>) {
        let message = message.into();
        log::error!("{} capture stopped: {message}", self.source.as_str());
        let _ = self.errors.send(CaptureError {
            source: self.source,
            message,
        });
    }
}

/// Running capture. Dropping it (or calling [`CaptureHandle::stop`]) stops every
/// stream and joins its threads; any partially filled chunk is flushed into the
/// channel before that returns.
pub struct CaptureHandle {
    session_start: Instant,
    streams: Vec<StreamInfo>,
    workers: Vec<backend::Worker>,
    errors: Receiver<CaptureError>,
    /// Keeps `errors` connected (never "ready" in `select!`) after all streams end.
    _errors_tx: Sender<CaptureError>,
}

impl CaptureHandle {
    pub fn stop(self) {}

    /// The instant all `start_ms` values are measured from.
    pub fn session_start(&self) -> Instant {
        self.session_start
    }

    /// The streams that were opened.
    pub fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    /// Receiver of runtime stream failures (at most one per source), usable in
    /// `crossbeam_channel::select!`; it stays connected while the handle lives.
    /// Clones share one queue: each error is delivered to only one receiver.
    pub fn errors(&self) -> Receiver<CaptureError> {
        self.errors.clone()
    }

    /// Non-blocking poll for a stream failure.
    pub fn take_error(&self) -> Option<CaptureError> {
        self.errors.try_recv().ok()
    }

    fn wait_started(&self, started: &Receiver<(Source, u64)>) -> Result<()> {
        let deadline = Instant::now() + STARTUP_TIMEOUT;
        let mut pending: Vec<Source> = self.streams.iter().map(|s| s.source).collect();
        let mut offsets = Vec::new();
        while !pending.is_empty() {
            crossbeam_channel::select! {
                recv(started) -> msg => match msg {
                    Ok((source, first_ms)) => {
                        pending.retain(|&s| s != source);
                        offsets.push((source, first_ms));
                    }
                    Err(_) => {
                        // Every stream has already exited; say why if one reported it.
                        if let Ok(err) = self.errors.try_recv() {
                            bail!(err);
                        }
                        break;
                    }
                },
                recv(self.errors) -> msg => {
                    if let Ok(err) = msg {
                        bail!(err);
                    }
                },
                default(deadline.saturating_duration_since(Instant::now())) => {
                    for s in &pending {
                        log::warn!(
                            "{} stream delivered no audio within {STARTUP_TIMEOUT:?}; continuing",
                            s.as_str()
                        );
                    }
                    break;
                }
            }
        }
        for (source, first_ms) in offsets {
            log::debug!("{} stream starts at {first_ms} ms", source.as_str());
        }
        Ok(())
    }
}

impl Drop for CaptureHandle {
    fn drop(&mut self) {
        // Signal every stream first so they shut down in parallel, then join.
        for w in &mut self.workers {
            w.signal_stop();
        }
        for w in &mut self.workers {
            w.join();
        }
    }
}
