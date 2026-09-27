// UNTESTED on macOS
//! macOS backend (cpal / Core Audio).
//!
//! **UNTESTED on macOS**: this module was written on Linux. It is type-checked with
//! `cargo check --target aarch64-apple-darwin` but has never been run.
//!
//! - Mic: a cpal input stream on the default (or chosen) input device.
//! - System audio: cpal 0.18 turns an input stream opened on an *output* device into
//!   a loopback recording. Internally it creates a Core Audio process tap
//!   (`CATapDescription` over all processes, unmuted) on that device plus a private
//!   aggregate device, and destroys both when the stream is dropped. This needs
//!   macOS 14.6+ according to cpal (the tap API itself appeared in 14.2).
//!   Limitation: cpal only does this for devices *without* inputs, so a duplex output
//!   (e.g. some USB headsets that are one Core Audio device) cannot be loopback-
//!   captured and is reported as an error rather than silently recording its mic.
//!
//! Both streams are requested as `f32` (the Audio Unit converts from the hardware
//! format), downmixed to mono in the callback, then resampled to 16 kHz with
//! [`crate::pcm::Resampler`] (windowed-sinc) on a worker thread, so the real-time
//! callback only averages channels and does a non-blocking channel send.
//!
//! Permissions (the app bundle's Info.plist / entitlements):
//! - `NSMicrophoneUsageDescription`: mic access prompt.
//! - `NSAudioCaptureUsageDescription`: system-audio (process tap) prompt.
//! - Entitlement `com.apple.security.device.audio-input` for hardened-runtime /
//!   sandboxed builds.
//!
//! Without permission Core Audio typically delivers silence rather than an error.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender};
use kenes_types::{DeviceInfo, DeviceKind, Source, SAMPLE_RATE};

use crate::chunker::{session_samples, Chunker};
use crate::pcm::Resampler;
use crate::{DeviceSel, StreamCtx, StreamInfo};

/// Callback buffers that may queue up before the worker thread (≈ 2.5 s at 48 kHz / 512 frames).
const RAW_QUEUE: usize = 256;
const STREAM_TIMEOUT: Duration = Duration::from_secs(3);

fn device_name(device: &cpal::Device) -> String {
    device
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| "unknown device".to_string())
}

fn device_id(device: &cpal::Device) -> Option<String> {
    device.id().ok().map(|id| id.to_string())
}

pub(crate) fn list_devices() -> Result<Vec<DeviceInfo>> {
    let host = cpal::default_host();
    let default_in = host.default_input_device().and_then(|d| device_id(&d));
    let default_out = host.default_output_device().and_then(|d| device_id(&d));
    let mut devices = Vec::new();

    for d in host.input_devices().context("listing input devices")? {
        let Some(id) = device_id(&d) else { continue };
        devices.push(DeviceInfo {
            is_default: default_in.as_deref() == Some(id.as_str()),
            name: device_name(&d),
            kind: DeviceKind::Input,
            id,
        });
    }
    for d in host.output_devices().context("listing output devices")? {
        if d.supports_input() {
            log::debug!(
                "{} has inputs; cpal cannot loopback-record it",
                device_name(&d)
            );
            continue;
        }
        let Some(id) = device_id(&d) else { continue };
        devices.push(DeviceInfo {
            is_default: default_out.as_deref() == Some(id.as_str()),
            name: device_name(&d),
            kind: DeviceKind::Monitor,
            id,
        });
    }
    Ok(devices)
}

fn pick_device(host: &cpal::Host, sel: &DeviceSel, source: Source) -> Result<cpal::Device> {
    match (sel, source) {
        (DeviceSel::Default, Source::Mic) => host
            .default_input_device()
            .context("no default input device"),
        (DeviceSel::Default, Source::System) => host
            .default_output_device()
            .context("no default output device"),
        (DeviceSel::Id(id), _) => {
            let parsed: cpal::DeviceId = id
                .parse()
                .map_err(|e| anyhow!("invalid device id {id:?}: {e}"))?;
            host.device_by_id(&parsed)
                .with_context(|| format!("audio device {id:?} not found"))
        }
    }
}

pub(crate) fn open_stream(sel: &DeviceSel, ctx: StreamCtx) -> Result<(Worker, StreamInfo)> {
    let host = cpal::default_host();
    let device = pick_device(&host, sel, ctx.source)?;
    let name = device_name(&device);
    let supported = match ctx.source {
        Source::Mic => device
            .default_input_config()
            .with_context(|| format!("querying input format of {name}"))?,
        Source::System => {
            if device.supports_input() {
                bail!(
                    "{name} also has inputs, and cpal can only loopback-record output-only \
                     devices; pick another output device for system audio"
                );
            }
            device
                .default_output_config()
                .with_context(|| format!("querying output format of {name}"))?
        }
    };
    let mut config = supported.config();
    config.buffer_size = cpal::BufferSize::Default;

    let source = ctx.source;
    let (stop_tx, stop_rx) = crossbeam_channel::bounded::<()>(1);
    let (ready_tx, ready_rx) = crossbeam_channel::bounded::<Result<()>>(1);
    let thread = thread::Builder::new()
        .name(format!("kenes-audio-{}", source.as_str()))
        .spawn(move || run_stream(device, config, ctx, stop_rx, ready_tx))
        .context("spawning audio thread")?;
    let mut worker = Worker {
        stop: Some(stop_tx),
        thread: Some(thread),
    };
    match ready_rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(e.context(format!("opening {name}"))),
        Err(_) => {
            worker.join();
            bail!("audio thread for {name} exited during startup");
        }
    }
    Ok((
        worker,
        StreamInfo {
            source,
            device: name,
            backend: "cpal",
        },
    ))
}

/// Builds and runs one cpal stream; owns it for its whole life so the `Stream`
/// never has to cross threads.
fn run_stream(
    device: cpal::Device,
    config: cpal::StreamConfig,
    ctx: StreamCtx,
    stop_rx: Receiver<()>,
    ready_tx: Sender<Result<()>>,
) {
    let channels = usize::from(config.channels.max(1));
    let rate = config.sample_rate;
    let (raw_tx, raw_rx) = crossbeam_channel::bounded::<Vec<f32>>(RAW_QUEUE);
    let (fatal_tx, fatal_rx) = crossbeam_channel::bounded::<String>(1);
    let dropped = Arc::new(AtomicU64::new(0));
    let dropped_cb = dropped.clone();
    let source = ctx.source;

    let on_data = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        let mut mono = Vec::with_capacity(data.len() / channels);
        crate::pcm::downmix_into(data, channels, &mut mono);
        if raw_tx.try_send(mono).is_err() {
            dropped_cb.fetch_add(1, Ordering::Relaxed);
        }
    };
    let on_error = move |err: cpal::Error| match err.kind() {
        cpal::ErrorKind::Xrun
        | cpal::ErrorKind::DeviceChanged
        | cpal::ErrorKind::RealtimeDenied => {
            log::warn!("{} stream: {err}", source.as_str());
        }
        _ => {
            let _ = fatal_tx.try_send(err.to_string());
        }
    };

    let stream = match device.build_input_stream::<f32, _, _>(
        config,
        on_data,
        on_error,
        Some(STREAM_TIMEOUT),
    ) {
        Ok(s) => s,
        Err(e) => {
            let _ = ready_tx.send(Err(anyhow!("building input stream: {e}")));
            return;
        }
    };
    if let Err(e) = stream.play() {
        let _ = ready_tx.send(Err(anyhow!("starting input stream: {e}")));
        return;
    }
    let _ = ready_tx.send(Ok(()));

    let mut pipe = Pipeline {
        resampler: Resampler::new(rate, SAMPLE_RATE),
        chunker: Chunker::new(source),
        out: Vec::new(),
        started: false,
        receiver_gone: false,
    };
    let mut reported_drops = 0;
    loop {
        crossbeam_channel::select! {
            recv(raw_rx) -> msg => match msg {
                Ok(mono) => pipe.feed(&mono, &ctx),
                Err(_) => break,
            },
            recv(fatal_rx) -> msg => {
                let message = msg.unwrap_or_else(|_| "audio stream closed".to_string());
                drop(stream);
                ctx.fail(message);
                return;
            },
            recv(stop_rx) -> _ => break,
        }
        let d = dropped.load(Ordering::Relaxed);
        if d != reported_drops {
            log::warn!(
                "{} capture: worker fell behind, dropped {} buffers",
                source.as_str(),
                d - reported_drops
            );
            reported_drops = d;
        }
        if pipe.receiver_gone {
            log::debug!("{} chunk receiver dropped; stopping", source.as_str());
            return;
        }
    }

    drop(stream);
    for mono in raw_rx.try_iter() {
        pipe.feed(&mono, &ctx);
    }
    pipe.finish(&ctx);
}

struct Pipeline {
    resampler: Resampler,
    chunker: Chunker,
    out: Vec<f32>,
    started: bool,
    receiver_gone: bool,
}

impl Pipeline {
    fn feed(&mut self, mono: &[f32], ctx: &StreamCtx) {
        self.out.clear();
        self.resampler.process(mono, &mut self.out);
        let now = session_samples(ctx.session_start);
        let gone = &mut self.receiver_gone;
        self.chunker.push(&self.out, now, &mut |chunk| {
            if !*gone && ctx.tx.send(chunk).is_err() {
                *gone = true;
            }
        });
        if !self.started {
            if let Some(first_ms) = self.chunker.origin_ms() {
                self.started = true;
                let _ = ctx.started.send((ctx.source, first_ms));
            }
        }
    }

    fn finish(mut self, ctx: &StreamCtx) {
        if self.receiver_gone {
            return;
        }
        self.out.clear();
        self.resampler.flush(&mut self.out);
        let now = session_samples(ctx.session_start);
        let mut emit = |chunk| {
            let _ = ctx.tx.send(chunk);
        };
        self.chunker.push(&self.out, now, &mut emit);
        self.chunker.flush(&mut emit);
    }
}

/// One cpal stream owned by its thread.
pub(crate) struct Worker {
    stop: Option<Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    pub(crate) fn signal_stop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.try_send(());
        }
    }

    pub(crate) fn join(&mut self) {
        self.signal_stop();
        if let Some(t) = self.thread.take() {
            if t.join().is_err() {
                log::error!("audio thread panicked");
            }
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.join();
    }
}
