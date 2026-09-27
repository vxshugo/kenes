//! Real capture tests against the running PulseAudio/PipeWire server.
//!
//! They are `#[ignore]`d because they need an audio server; run them with
//! `cargo test -p kenes-audio -- --ignored`. Nothing is played on the speakers:
//! test tones go to a temporary null sink that is always unloaded again.
#![cfg(target_os = "linux")]

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;
use kenes_audio::pcm::rms;
use kenes_audio::{start_capture, wav, AudioChunk, CaptureConfig, DeviceKind, DeviceSel, Source};

/// These tests count threads/processes and flip an env var, so run them one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn pactl(args: &[&str]) -> String {
    let out = Command::new("pactl")
        .args(args)
        .output()
        .expect("running pactl");
    assert!(
        out.status.success(),
        "pactl {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

/// A `module-null-sink`, unloaded on drop even if the test panics.
struct NullSink {
    module: Option<String>,
    name: String,
}

impl NullSink {
    fn load(name: &str) -> Self {
        // Clean up leftovers from an earlier run that was killed before its guard ran.
        let arg = format!("sink_name={name}");
        for line in pactl(&["list", "short", "modules"]).lines() {
            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() > 2
                && cols[1] == "module-null-sink"
                && cols[2].split_whitespace().any(|a| a == arg)
            {
                let _ = Command::new("pactl")
                    .args(["unload-module", cols[0]])
                    .status();
            }
        }
        let module = pactl(&[
            "load-module",
            "module-null-sink",
            &format!("sink_name={name}"),
            &format!("sink_properties=device.description={name}"),
        ])
        .trim()
        .to_string();
        assert!(!module.is_empty(), "load-module printed no module id");
        Self {
            module: Some(module),
            name: name.to_string(),
        }
    }

    fn monitor(&self) -> String {
        format!("{}.monitor", self.name)
    }

    fn unload(&mut self) {
        if let Some(m) = self.module.take() {
            let _ = Command::new("pactl").args(["unload-module", &m]).status();
        }
    }
}

impl Drop for NullSink {
    fn drop(&mut self) {
        self.unload();
    }
}

/// A background `paplay`, killed on drop.
struct Player(Child);

impl Player {
    fn play(sink: &str, file: &std::path::Path) -> Self {
        let child = Command::new("paplay")
            .arg(format!("--device={sink}"))
            .arg(file)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawning paplay");
        Self(child)
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// A per-process file in the temp dir, removed on drop.
struct TmpFile(PathBuf);

impl TmpFile {
    fn new(prefix: &str, name: &str) -> Self {
        let file = format!("{prefix}-{}-{name}", std::process::id());
        Self(std::env::temp_dir().join(file))
    }
}

impl std::ops::Deref for TmpFile {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for TmpFile {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl Drop for TmpFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn tone_file(name: &str, freq: f32, secs: f32) -> TmpFile {
    let path = TmpFile::new("kenes-audio-it", name);
    let n = (16_000.0 * secs) as usize;
    let samples: Vec<f32> = (0..n)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * freq * i as f32 / 16_000.0).sin())
        .collect();
    wav::write(&path, &samples).unwrap();
    path
}

fn collect_for(rx: &Receiver<AudioChunk>, dur: Duration) -> Vec<AudioChunk> {
    let deadline = Instant::now() + dur;
    let mut chunks = Vec::new();
    while let Ok(c) = rx.recv_deadline(deadline) {
        chunks.push(c);
    }
    chunks
}

/// Checks that chunks are 32 ms, contiguous and in order; returns the samples.
fn check_timeline(chunks: &[AudioChunk], source: Source) -> Vec<f32> {
    assert!(!chunks.is_empty(), "no chunks received");
    assert!(
        chunks[0].start_ms < 1500,
        "first chunk at {} ms",
        chunks[0].start_ms
    );
    for (i, c) in chunks.iter().enumerate() {
        assert_eq!(c.source, source);
        assert!(c.samples.iter().all(|s| (-1.0..=1.0).contains(s)));
        let last = i + 1 == chunks.len();
        if !last {
            assert_eq!(c.samples.len(), 512, "chunk {i}");
        }
        if i > 0 {
            assert_eq!(
                c.start_ms,
                chunks[i - 1].start_ms + 32,
                "chunk {i} not contiguous"
            );
        }
    }
    chunks
        .iter()
        .flat_map(|c| c.samples.iter().copied())
        .collect()
}

fn zero_crossing_freq(samples: &[f32]) -> f32 {
    let crossings = samples
        .windows(2)
        .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
        .count();
    crossings as f32 / 2.0 / (samples.len() as f32 / 16_000.0)
}

/// The loudest 0.5 s window, to skip silence before playback starts.
fn loudest_window(samples: &[f32]) -> &[f32] {
    let win = 8000.min(samples.len());
    (0..=samples.len() - win)
        .step_by(800)
        .map(|i| &samples[i..i + win])
        .max_by(|a, b| rms(a).total_cmp(&rms(b)))
        .unwrap()
}

/// Live threads spawned by kenes-audio (they are all named `kenes-audio-*`).
fn audio_threads() -> usize {
    std::fs::read_dir("/proc/self/task")
        .unwrap()
        .flatten()
        .filter(|t| {
            std::fs::read_to_string(t.path().join("comm"))
                .is_ok_and(|c| c.starts_with("kenes-audio"))
        })
        .count()
}

/// Our direct children that are still around (running or zombie), as `(pid, comm)`.
fn child_processes() -> Vec<(String, String)> {
    let me = std::process::id().to_string();
    let mut out = Vec::new();
    for entry in std::fs::read_dir("/proc").unwrap().flatten() {
        let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else {
            continue;
        };
        // "pid (comm) state ppid ..."
        let (Some(open), Some(close)) = (stat.find('('), stat.rfind(')')) else {
            continue;
        };
        let fields: Vec<&str> = stat[close + 2..].split(' ').collect();
        if fields.get(1) == Some(&me.as_str()) {
            out.push((
                stat[..open].trim().to_string(),
                stat[open + 1..close].to_string(),
            ));
        }
    }
    out
}

fn recorder_children() -> Vec<(String, String)> {
    child_processes()
        .into_iter()
        .filter(|(_, comm)| comm != "paplay")
        .collect()
}

fn capture_tone(backend: &str, sink_name: &str) {
    let mut sink = NullSink::load(sink_name);
    let tone = tone_file(&format!("tone-{backend}.wav"), 1000.0, 3.0);
    std::env::set_var("KENES_AUDIO_BACKEND", backend);

    assert_eq!(audio_threads(), 0);
    let (tx, rx) = crossbeam_channel::unbounded();
    let cfg = CaptureConfig {
        mic: None,
        system: Some(DeviceSel::Id(sink.monitor())),
    };
    let handle = start_capture(cfg, tx).expect("start_capture");
    std::env::remove_var("KENES_AUDIO_BACKEND");
    assert_eq!(handle.streams().len(), 1);
    assert_eq!(handle.streams()[0].backend, backend);
    assert_eq!(handle.streams()[0].device, sink.monitor());

    let _player = Player::play(&sink.name, &tone);
    let chunks = collect_for(&rx, Duration::from_millis(2500));
    assert!(handle.take_error().is_none());

    let t = Instant::now();
    handle.stop();
    let stop_time = t.elapsed();
    let mut chunks = chunks;
    chunks.extend(rx.try_iter());
    sink.unload();

    assert!(
        stop_time < Duration::from_millis(200),
        "stop took {stop_time:?}"
    );
    assert_eq!(audio_threads(), 0, "threads leaked");
    assert!(
        recorder_children().is_empty(),
        "left child processes: {:?}",
        recorder_children()
    );

    let samples = check_timeline(&chunks, Source::System);
    let seconds = samples.len() as f32 / 16_000.0;
    assert!((2.0..3.2).contains(&seconds), "captured {seconds} s");
    let window = loudest_window(&samples);
    let level = rms(window);
    let freq = zero_crossing_freq(window);
    eprintln!("{backend}: {seconds:.2} s captured, tone rms {level:.3}, ~{freq:.0} Hz, stop {stop_time:?}");
    assert!(level > 0.1, "tone rms {level} (expected ~0.35)");
    assert!((freq - 1000.0).abs() < 30.0, "dominant frequency {freq} Hz");
}

#[test]
#[ignore = "needs a PulseAudio/PipeWire server"]
fn captures_tone_from_null_sink_monitor_with_parec() {
    let _g = serial();
    capture_tone("parec", "kenes_test_sink");
}

#[test]
#[ignore = "needs a PulseAudio/PipeWire server"]
fn captures_tone_from_null_sink_monitor_with_pw_record() {
    let _g = serial();
    capture_tone("pw-record", "kenes_test_sink_pw");
}

#[test]
#[ignore = "needs a PulseAudio/PipeWire server"]
fn captures_default_mic_and_system() {
    let _g = serial();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = start_capture(CaptureConfig::default(), tx).expect("start_capture");
    let chunks = collect_for(&rx, Duration::from_secs(2));
    let t = Instant::now();
    handle.stop();
    assert!(
        t.elapsed() < Duration::from_millis(200),
        "stop took {:?}",
        t.elapsed()
    );
    let mut chunks = chunks;
    chunks.extend(rx.try_iter());

    for source in [Source::Mic, Source::System] {
        let mine: Vec<AudioChunk> = chunks
            .iter()
            .filter(|c| c.source == source)
            .cloned()
            .collect();
        let samples = check_timeline(&mine, source);
        let seconds = samples.len() as f32 / 16_000.0;
        eprintln!(
            "{}: {} chunks, first at {} ms, {seconds:.2} s, rms {:.4}",
            source.as_str(),
            mine.len(),
            mine[0].start_ms,
            rms(&samples)
        );
        assert!(
            (1.6..2.3).contains(&seconds),
            "{} captured {seconds} s",
            source.as_str()
        );
    }
}

#[test]
#[ignore = "needs a PulseAudio/PipeWire server"]
fn reports_error_when_recorder_dies() {
    let _g = serial();
    let sink = NullSink::load("kenes_test_sink_crash");
    let (tx, rx) = crossbeam_channel::unbounded();
    let cfg = CaptureConfig {
        mic: None,
        system: Some(DeviceSel::Id(sink.monitor())),
    };
    let handle = start_capture(cfg, tx).expect("start_capture");
    assert!(!collect_for(&rx, Duration::from_millis(300)).is_empty());

    // Simulate a crash (or the audio server going away) by killing the recorder.
    let children = recorder_children();
    assert_eq!(children.len(), 1, "{children:?}");
    assert_eq!(children[0].1, "parec");
    Command::new("kill")
        .args(["-9", &children[0].0])
        .status()
        .unwrap();

    let err = handle
        .errors()
        .recv_timeout(Duration::from_secs(3))
        .expect("no error reported after the recorder died");
    eprintln!("reported: {err}");
    assert_eq!(err.source, Source::System);
    assert!(err.message.contains("parec exited"), "{}", err.message);
    // The stream is over: the channel goes quiet instead of spinning.
    let _ = collect_for(&rx, Duration::from_millis(100));
    assert!(collect_for(&rx, Duration::from_millis(300)).is_empty());
    handle.stop();
    assert!(
        recorder_children().is_empty(),
        "left child processes: {:?}",
        recorder_children()
    );
}

#[test]
#[ignore = "needs a PulseAudio/PipeWire server"]
fn rejects_unknown_device_up_front() {
    let _g = serial();
    let (tx, _rx) = crossbeam_channel::unbounded();
    let cfg = CaptureConfig {
        mic: Some(DeviceSel::Id("no_such_source_kenes".into())),
        system: None,
    };
    let t = Instant::now();
    let err = start_capture(cfg, tx)
        .err()
        .expect("unknown device must fail");
    let msg = format!("{err:#}");
    assert!(msg.contains("no_such_source_kenes"), "{msg}");
    assert!(t.elapsed() < Duration::from_secs(2));
    assert!(recorder_children().is_empty());
}

#[test]
#[ignore = "needs a PulseAudio/PipeWire server"]
fn stops_when_receiver_is_dropped() {
    let _g = serial();
    assert_eq!(audio_threads(), 0);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = start_capture(
        CaptureConfig {
            mic: Some(DeviceSel::Default),
            system: None,
        },
        tx,
    )
    .unwrap();
    assert!(!collect_for(&rx, Duration::from_millis(200)).is_empty());
    drop(rx);
    std::thread::sleep(Duration::from_millis(200));
    // The reader noticed and exited on its own; stop() only reaps.
    assert!(handle.take_error().is_none());
    handle.stop();
    assert_eq!(audio_threads(), 0, "threads leaked");
    assert!(recorder_children().is_empty());
}

#[test]
#[ignore = "needs a PulseAudio/PipeWire server"]
fn lists_devices_from_pactl() {
    let _g = serial();
    let devices = kenes_audio::list_devices().unwrap();
    for d in &devices {
        eprintln!(
            "{:?} default={} {} ({})",
            d.kind, d.is_default, d.name, d.id
        );
    }
    assert!(devices
        .iter()
        .any(|d| d.kind == DeviceKind::Monitor && d.is_default));
    assert!(devices.iter().filter(|d| d.is_default).count() <= 2);
}

#[test]
#[ignore = "needs parec"]
fn startup_failure_is_returned_synchronously() {
    let _g = serial();
    // Point parec (and pactl) at a server that doesn't exist: parec exits right away.
    std::env::set_var("PULSE_SERVER", "unix:/nonexistent/kenes-test");
    std::env::set_var("KENES_AUDIO_BACKEND", "parec");
    let (tx, _rx) = crossbeam_channel::unbounded();
    let t = Instant::now();
    let result = start_capture(
        CaptureConfig {
            mic: Some(DeviceSel::Default),
            system: None,
        },
        tx,
    );
    std::env::remove_var("PULSE_SERVER");
    std::env::remove_var("KENES_AUDIO_BACKEND");
    let msg = format!(
        "{:#}",
        result
            .err()
            .expect("capture against a dead server must fail")
    );
    eprintln!("startup error after {:?}: {msg}", t.elapsed());
    assert!(msg.contains("parec exited"), "{msg}");
    assert!(t.elapsed() < Duration::from_secs(2));
    assert_eq!(audio_threads(), 0, "threads leaked");
    assert!(recorder_children().is_empty());
}
