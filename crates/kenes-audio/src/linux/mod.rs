//! Linux backend: one `parec` (or `pw-record`) subprocess per source, printing raw
//! s16le 16 kHz mono PCM to a pipe that a reader thread turns into chunks. The
//! audio server does the resampling and downmixing. Devices come from `pactl`.
//!
//! This avoids linking libpulse/libpipewire/ALSA, which keeps the build free of
//! `-dev` packages.

mod pactl;

use std::io::{BufRead, BufReader, ErrorKind, Read};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, ChildStderr, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use kenes_types::{DeviceInfo, Source, SAMPLE_RATE};

use crate::chunker::{session_samples, Chunker};
use crate::{DeviceSel, StreamCtx, StreamInfo};

/// Requested server-side latency. Small enough for live captions, large enough
/// that the reader wakes only ~50 times a second.
const LATENCY_MS: u32 = 20;
/// Env var to force a recorder: `parec` or `pw-record`.
const BACKEND_ENV: &str = "KENES_AUDIO_BACKEND";
const STDERR_TAIL_LINES: usize = 20;

pub(crate) fn list_devices() -> Result<Vec<DeviceInfo>> {
    let sources = pactl::list_sources()?;
    let default_source = pactl::default_source().ok();
    let default_sink = pactl::default_sink().ok();
    Ok(pactl::to_device_infos(
        &sources,
        default_source.as_deref(),
        default_sink.as_deref(),
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tool {
    Parec,
    PwRecord,
}

impl Tool {
    fn program(self) -> &'static str {
        match self {
            Tool::Parec => "parec",
            Tool::PwRecord => "pw-record",
        }
    }

    fn detect() -> Result<Self> {
        if let Ok(forced) = std::env::var(BACKEND_ENV) {
            return match forced.as_str() {
                "parec" => Ok(Tool::Parec),
                "pw-record" => Ok(Tool::PwRecord),
                other => bail!("{BACKEND_ENV}={other:?}: expected \"parec\" or \"pw-record\""),
            };
        }
        [Tool::Parec, Tool::PwRecord]
            .into_iter()
            .find(|t| find_in_path(t.program()).is_some())
            .context(
                "neither `parec` nor `pw-record` found in PATH \
                 (install pulseaudio-utils or pipewire-bin)",
            )
    }
}

fn find_in_path(program: &str) -> Option<PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(program))
        .find(|p| {
            p.metadata()
                .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        })
}

/// What the recorder should connect to.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Target {
    /// The server's default at connect time (`@DEFAULT_SOURCE@` / `@DEFAULT_MONITOR@`;
    /// pipewire-pulse pins the stream to the device it resolves to).
    Default,
    /// A specific source by name. For monitors this is `<sink>.monitor`.
    Named { name: String, monitor: bool },
}

/// Resolves the selection to a target plus a human-readable device name.
fn resolve(sel: &DeviceSel, source: Source) -> Result<(Target, String)> {
    match sel {
        DeviceSel::Default => {
            let name = match source {
                Source::Mic => pactl::default_source(),
                Source::System => pactl::default_sink().map(|s| format!("{s}.monitor")),
            };
            let display = name.unwrap_or_else(|e| {
                log::debug!("could not resolve the default device name: {e:#}");
                "default".to_string()
            });
            Ok((Target::Default, display))
        }
        DeviceSel::Id(id) => {
            let sources = match pactl::list_sources() {
                Ok(s) => s,
                Err(e) => {
                    log::warn!("cannot verify device {id:?} ({e:#}); trying it anyway");
                    let target = Target::Named {
                        name: id.clone(),
                        monitor: id.ends_with(".monitor"),
                    };
                    return Ok((target, id.clone()));
                }
            };
            if let Some(s) = sources.iter().find(|s| &s.name == id) {
                return Ok((
                    Target::Named {
                        name: s.name.clone(),
                        monitor: s.is_monitor,
                    },
                    id.clone(),
                ));
            }
            // Accept a sink name for system audio and capture its monitor.
            let monitor = format!("{id}.monitor");
            if source == Source::System {
                if let Some(s) = sources.iter().find(|s| s.name == monitor) {
                    return Ok((
                        Target::Named {
                            name: s.name.clone(),
                            monitor: true,
                        },
                        monitor,
                    ));
                }
            }
            let known: Vec<&str> = sources.iter().map(|s| s.name.as_str()).collect();
            bail!(
                "unknown audio device {id:?}; available sources: {}",
                known.join(", ")
            )
        }
    }
}

fn recorder_args(tool: Tool, target: &Target, source: Source) -> Vec<String> {
    let tag = format!("kenes-{}", source.as_str());
    match tool {
        Tool::Parec => {
            let device = match (target, source) {
                (Target::Named { name, .. }, _) => name.clone(),
                (Target::Default, Source::Mic) => "@DEFAULT_SOURCE@".into(),
                (Target::Default, Source::System) => "@DEFAULT_MONITOR@".into(),
            };
            vec![
                format!("--device={device}"),
                "--format=s16le".into(),
                format!("--rate={SAMPLE_RATE}"),
                "--channels=1".into(),
                format!("--latency-msec={LATENCY_MS}"),
                "--raw".into(),
                "--client-name=kenes".into(),
                format!("--stream-name={tag}"),
            ]
        }
        Tool::PwRecord => {
            // pw-record targets nodes: a monitor is captured by targeting its sink with
            // stream.capture.sink; targeting "<sink>.monitor" would silently fall back
            // to the default microphone.
            let (node, capture_sink) = match (target, source) {
                (
                    Target::Named {
                        name,
                        monitor: true,
                    },
                    _,
                ) => (
                    Some(name.strip_suffix(".monitor").unwrap_or(name).to_string()),
                    true,
                ),
                (
                    Target::Named {
                        name,
                        monitor: false,
                    },
                    _,
                ) => (Some(name.clone()), false),
                (Target::Default, Source::Mic) => (None, false),
                (Target::Default, Source::System) => (None, true),
            };
            let mut props = format!("{{ application.name = kenes node.name = {tag}");
            if capture_sink {
                props.push_str(" stream.capture.sink = true");
            }
            props.push_str(" }");
            let mut args = Vec::new();
            if let Some(node) = node {
                args.extend(["--target".to_string(), node]);
            }
            args.extend([
                "-P".to_string(),
                props,
                "--format".into(),
                "s16".into(),
                "--rate".into(),
                SAMPLE_RATE.to_string(),
                "--channels".into(),
                "1".into(),
                "--latency".into(),
                format!("{LATENCY_MS}ms"),
                "--raw".into(),
                "-".into(),
            ]);
            args
        }
    }
}

pub(crate) fn open_stream(sel: &DeviceSel, ctx: StreamCtx) -> Result<(Worker, StreamInfo)> {
    let tool = Tool::detect()?;
    let (target, device) = resolve(sel, ctx.source)?;
    let args = recorder_args(tool, &target, ctx.source);
    log::debug!("spawning {} {}", tool.program(), args.join(" "));

    let mut child = Command::new(tool.program())
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Own process group: a terminal Ctrl-C must not kill the recorder behind our back;
        // we stop it ourselves. If we die, it gets EPIPE on its next write and exits.
        .process_group(0)
        .spawn()
        .with_context(|| format!("starting {}", tool.program()))?;
    let stdout = child
        .stdout
        .take()
        .context("recorder stdout not captured")?;
    let stderr = child
        .stderr
        .take()
        .context("recorder stderr not captured")?;

    let mut worker = Worker {
        stop: Arc::new(AtomicBool::new(false)),
        child: Arc::new(Mutex::new(child)),
        thread: None,
    };
    let reader = Reader {
        tool,
        stop: worker.stop.clone(),
        child: worker.child.clone(),
        ctx,
    };
    let source = reader.ctx.source;
    worker.thread = Some(
        thread::Builder::new()
            .name(format!("kenes-audio-{}", source.as_str()))
            .spawn(move || reader.run(stdout, stderr))
            .context("spawning audio reader thread")?,
    );
    let info = StreamInfo {
        source,
        device,
        backend: tool.program(),
    };
    Ok((worker, info))
}

/// One recorder subprocess plus the thread that reads it.
pub(crate) struct Worker {
    stop: Arc<AtomicBool>,
    child: Arc<Mutex<Child>>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    pub(crate) fn signal_stop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // Killing the child closes its stdout, which unblocks the reader.
        let _ = lock(&self.child).kill();
    }

    pub(crate) fn join(&mut self) {
        if let Some(t) = self.thread.take() {
            if t.join().is_err() {
                log::error!("audio reader thread panicked");
            }
        }
        // Reap the child so it doesn't linger as a zombie.
        let _ = lock(&self.child).wait();
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.signal_stop();
        self.join();
    }
}

fn lock(child: &Mutex<Child>) -> MutexGuard<'_, Child> {
    child.lock().unwrap_or_else(|e| e.into_inner())
}

struct Reader {
    tool: Tool,
    stop: Arc<AtomicBool>,
    child: Arc<Mutex<Child>>,
    ctx: StreamCtx,
}

impl Reader {
    fn run(self, mut stdout: ChildStdout, stderr: ChildStderr) {
        let source = self.ctx.source;
        let stderr_tail = Arc::new(Mutex::new(Vec::<String>::new()));
        let stderr_thread = spawn_stderr_drain(self.tool, source, stderr, stderr_tail.clone());

        let mut decoder = crate::pcm::S16leDecoder::new();
        let mut chunker = Chunker::new(source);
        // Up to 0.5 s per read, so a briefly stalled reader catches up in one go.
        let mut buf = vec![0u8; 16 * 1024];
        let mut samples = Vec::with_capacity(buf.len() / 2);
        let mut started = false;
        let mut receiver_gone = false;

        let read_result = loop {
            match stdout.read(&mut buf) {
                Ok(0) => break Ok(()),
                Ok(n) => {
                    samples.clear();
                    decoder.decode(&buf[..n], &mut samples);
                    let now = session_samples(self.ctx.session_start);
                    chunker.push(&samples, now, &mut |chunk| {
                        if !receiver_gone && self.ctx.tx.send(chunk).is_err() {
                            receiver_gone = true;
                        }
                    });
                    if !started {
                        if let Some(first_ms) = chunker.origin_ms() {
                            started = true;
                            let _ = self.ctx.started.send((source, first_ms));
                        }
                    }
                    if receiver_gone {
                        log::debug!("{} chunk receiver dropped; stopping", source.as_str());
                        break Ok(());
                    }
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => break Err(e),
            }
        };

        if !receiver_gone {
            chunker.flush(&mut |chunk| {
                let _ = self.ctx.tx.send(chunk);
            });
        }
        drop(stdout);

        let stopping = self.stop.load(Ordering::SeqCst);
        if stopping || receiver_gone {
            let _ = lock(&self.child).kill();
            join_quietly(stderr_thread);
            return;
        }

        // The recorder went away on its own: find out why and report it.
        let status = self.wait_for_exit(Duration::from_secs(1));
        join_quietly(stderr_thread);
        let tail = lock_tail(&stderr_tail).join(" | ");
        let mut message = match (read_result, status) {
            (Err(e), _) => format!("reading from {} failed: {e}", self.tool.program()),
            (Ok(()), Some(status)) => format!("{} exited ({status})", self.tool.program()),
            (Ok(()), None) => format!("{} closed its output", self.tool.program()),
        };
        if !tail.is_empty() {
            message.push_str(": ");
            message.push_str(&tail);
        }
        self.ctx.fail(message);
    }

    /// Polls for the child's exit without holding the lock (so `stop` can still kill it).
    fn wait_for_exit(&self, timeout: Duration) -> Option<std::process::ExitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Ok(Some(status)) = lock(&self.child).try_wait() {
                return Some(status);
            }
            if Instant::now() >= deadline {
                let _ = lock(&self.child).kill();
                return None;
            }
            thread::sleep(Duration::from_millis(10));
        }
    }
}

fn join_quietly(thread: Option<JoinHandle<()>>) {
    if let Some(t) = thread {
        let _ = t.join();
    }
}

fn lock_tail(tail: &Mutex<Vec<String>>) -> MutexGuard<'_, Vec<String>> {
    tail.lock().unwrap_or_else(|e| e.into_inner())
}

/// Logs the recorder's stderr and keeps its last lines for error messages.
fn spawn_stderr_drain(
    tool: Tool,
    source: Source,
    stderr: ChildStderr,
    tail: Arc<Mutex<Vec<String>>>,
) -> Option<JoinHandle<()>> {
    let body = move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }
            log::warn!("[{} {}] {line}", tool.program(), source.as_str());
            let mut tail = lock_tail(&tail);
            if tail.len() == STDERR_TAIL_LINES {
                tail.remove(0);
            }
            tail.push(line);
        }
    };
    thread::Builder::new()
        .name(format!("kenes-audio-{}-stderr", source.as_str()))
        .spawn(body)
        .map_err(|e| log::error!("cannot spawn stderr reader: {e}"))
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named(name: &str, monitor: bool) -> Target {
        Target::Named {
            name: name.into(),
            monitor,
        }
    }

    #[test]
    fn parec_args() {
        let args = recorder_args(Tool::Parec, &named("mic_src", false), Source::Mic);
        assert_eq!(
            args,
            [
                "--device=mic_src",
                "--format=s16le",
                "--rate=16000",
                "--channels=1",
                "--latency-msec=20",
                "--raw",
                "--client-name=kenes",
                "--stream-name=kenes-mic"
            ]
        );
        let args = recorder_args(Tool::Parec, &Target::Default, Source::System);
        assert_eq!(args[0], "--device=@DEFAULT_MONITOR@");
        let args = recorder_args(Tool::Parec, &Target::Default, Source::Mic);
        assert_eq!(args[0], "--device=@DEFAULT_SOURCE@");
    }

    #[test]
    fn pw_record_args_target_sink_for_monitors() {
        let args = recorder_args(Tool::PwRecord, &named("spk.monitor", true), Source::System);
        assert_eq!(&args[..2], ["--target", "spk"]);
        assert!(args[3].contains("stream.capture.sink = true"));
        assert_eq!(args.last().unwrap(), "-");
        assert!(args.windows(2).any(|w| w == ["--format", "s16"]));
        assert!(args.windows(2).any(|w| w == ["--rate", "16000"]));
        assert!(args.windows(2).any(|w| w == ["--channels", "1"]));

        let args = recorder_args(Tool::PwRecord, &named("mic_src", false), Source::Mic);
        assert_eq!(&args[..2], ["--target", "mic_src"]);
        assert!(!args[3].contains("capture.sink"));

        let args = recorder_args(Tool::PwRecord, &Target::Default, Source::System);
        assert!(!args.contains(&"--target".to_string()));
        assert!(args[1].contains("stream.capture.sink = true"));
        let args = recorder_args(Tool::PwRecord, &Target::Default, Source::Mic);
        assert!(!args.contains(&"--target".to_string()));
        assert!(!args[1].contains("capture.sink"));
    }

    #[test]
    fn finds_programs_in_path() {
        assert!(find_in_path("sh").is_some());
        assert!(find_in_path("definitely-not-a-program-kenes").is_none());
    }
}
