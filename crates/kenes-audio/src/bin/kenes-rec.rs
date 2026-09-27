//! Records the microphone and system audio into two 16 kHz mono WAV files.
//!
//! ```text
//! kenes-rec --out rec/standup --seconds 1800
//! kenes-rec --list
//! ```

use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use crossbeam_channel::{select, tick};
use kenes_audio::pcm::to_dbfs;
use kenes_audio::wav::WavWriter;
use kenes_audio::{
    list_devices, start_capture, AudioChunk, CaptureConfig, DeviceKind, DeviceSel, Source,
    SAMPLE_RATE,
};

const USAGE: &str = "\
kenes-rec: record mic and system audio to <out>/mic.wav and <out>/system.wav (16 kHz mono)

USAGE:
    kenes-rec [--out DIR] [--seconds N] [--no-mic] [--no-system]
              [--mic ID] [--system ID] [--force]
    kenes-rec --list

OPTIONS:
    --out DIR       Output directory (default: ./kenes-rec-<unix time>)
    --seconds N     Stop after N seconds (default: run until Ctrl-C)
    --no-mic        Don't record the microphone
    --no-system     Don't record system audio
    --mic ID        Microphone device id (see --list; default: system default)
    --system ID     System-audio device id, e.g. a sink monitor (default: default output)
    --force         Overwrite existing mic.wav/system.wav in DIR
    --list          List capture devices and exit
    -h, --help      Show this help

Both files share one timeline: each starts at the session start (padded with
silence), so sample N of mic.wav and sample N of system.wav happened together.
Set RUST_LOG=debug for backend details.";

struct Args {
    out: Option<PathBuf>,
    seconds: Option<f64>,
    mic: Option<DeviceSel>,
    system: Option<DeviceSel>,
    force: bool,
    list: bool,
}

fn parse_args() -> Result<Option<Args>> {
    let mut args = Args {
        out: None,
        seconds: None,
        mic: Some(DeviceSel::Default),
        system: Some(DeviceSel::Default),
        force: false,
        list: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        let mut value = |name: &str| -> Result<String> {
            inline
                .clone()
                .or_else(|| it.next())
                .with_context(|| format!("{name} needs a value"))
        };
        match flag.as_str() {
            "-h" | "--help" => return Ok(None),
            "--out" | "-o" => args.out = Some(PathBuf::from(value("--out")?)),
            "--seconds" | "-s" => {
                let v = value("--seconds")?;
                let secs: f64 = v.parse().with_context(|| format!("bad --seconds {v:?}"))?;
                if !(secs > 0.0 && secs.is_finite()) {
                    bail!("--seconds must be positive");
                }
                args.seconds = Some(secs);
            }
            "--no-mic" => args.mic = None,
            "--no-system" => args.system = None,
            "--mic" => args.mic = Some(DeviceSel::Id(value("--mic")?)),
            "--system" => args.system = Some(DeviceSel::Id(value("--system")?)),
            "--force" | "-f" => args.force = true,
            "--list" | "-l" => args.list = true,
            other => bail!("unknown argument {other:?} (see --help)"),
        }
    }
    Ok(Some(args))
}

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let args = match parse_args() {
        Ok(Some(args)) => args,
        Ok(None) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("error: {e:#}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let result = if args.list {
        print_devices()
    } else {
        record(args)
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn print_devices() -> Result<bool> {
    let devices = list_devices()?;
    for (kind, title) in [
        (DeviceKind::Input, "Microphones (--mic)"),
        (DeviceKind::Monitor, "System audio (--system)"),
    ] {
        println!("{title}:");
        for d in devices.iter().filter(|d| d.kind == kind) {
            let mark = if d.is_default { "*" } else { " " };
            println!("  {mark} {}\n      id: {}", d.name, d.id);
        }
    }
    println!("(* = default)");
    Ok(true)
}

/// One output file plus meter state for one source.
struct Track {
    source: Source,
    writer: WavWriter,
    sum_sq: f64,
    count: usize,
    peak: f32,
    loudest: f32,
    failed: bool,
}

impl Track {
    fn write(&mut self, chunk: &AudioChunk) -> Result<()> {
        // Keep the file on the session timeline: fill gaps (and the start offset) with silence.
        let expected = chunk.start_ms * u64::from(SAMPLE_RATE) / 1000;
        if expected > self.writer.len() + 16 {
            self.writer
                .write_silence((expected - self.writer.len()) as usize)?;
        }
        self.writer.write(&chunk.samples)?;
        for &s in &chunk.samples {
            self.sum_sq += f64::from(s) * f64::from(s);
            self.peak = self.peak.max(s.abs());
        }
        self.count += chunk.samples.len();
        Ok(())
    }

    /// Returns the meter text for the samples since the last call.
    fn take_meter(&mut self, color: bool) -> String {
        const WIDTH: usize = 20;
        let rms = if self.count == 0 {
            0.0
        } else {
            (self.sum_sq / self.count as f64).sqrt() as f32
        };
        self.loudest = self.loudest.max(rms);
        let db = to_dbfs(rms);
        let filled = (((db + 60.0) / 60.0).clamp(0.0, 1.0) * WIDTH as f32).round() as usize;
        let bar = format!("{}{}", "█".repeat(filled), "·".repeat(WIDTH - filled));
        let clip = self.peak >= 0.99;
        let level = if self.failed {
            "FAILED".to_string()
        } else if self.count == 0 {
            "no data".to_string()
        } else if db <= -99.0 {
            "silent".to_string()
        } else {
            format!("{db:>4.0} dB")
        };
        let clip_mark = match (clip, color) {
            (true, true) => " \x1b[31mCLIP\x1b[0m",
            (true, false) => " CLIP",
            _ => "",
        };
        self.sum_sq = 0.0;
        self.count = 0;
        self.peak = 0.0;
        format!("{:<6} {bar} {level:>7}{clip_mark}", self.source.as_str())
    }
}

fn record(args: Args) -> Result<bool> {
    if args.mic.is_none() && args.system.is_none() {
        bail!("nothing to record: both --no-mic and --no-system given");
    }
    let out_dir = args.out.clone().unwrap_or_else(|| {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        PathBuf::from(format!("kenes-rec-{secs}"))
    });
    std::fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;

    let mut tracks = Vec::new();
    for (source, sel) in [(Source::Mic, &args.mic), (Source::System, &args.system)] {
        if sel.is_none() {
            continue;
        }
        let path = out_dir.join(format!("{}.wav", source.as_str()));
        if path.exists() && !args.force {
            bail!(
                "{} already exists (use --force to overwrite)",
                path.display()
            );
        }
        tracks.push(Track {
            source,
            writer: WavWriter::create(&path)?,
            sum_sq: 0.0,
            count: 0,
            peak: 0.0,
            loudest: 0.0,
            failed: false,
        });
    }

    let (stop_tx, stop_rx) = crossbeam_channel::bounded::<()>(1);
    static INTERRUPTS: AtomicUsize = AtomicUsize::new(0);
    ctrlc::set_handler(move || {
        if INTERRUPTS.fetch_add(1, Ordering::SeqCst) >= 1 {
            eprintln!("\nkenes-rec: interrupted twice, exiting without finalizing");
            std::process::exit(130);
        }
        let _ = stop_tx.try_send(());
    })
    .context("installing Ctrl-C handler")?;

    let (tx, rx) = crossbeam_channel::unbounded();
    let cfg = CaptureConfig {
        mic: args.mic.clone(),
        system: args.system.clone(),
    };
    let handle = start_capture(cfg, tx)?;
    let errors = handle.errors();

    let tty = std::io::stderr().is_terminal();
    let duration = args.seconds.map(Duration::from_secs_f64);
    eprintln!(
        "kenes-rec: recording to {}/ {}",
        out_dir.display(),
        match args.seconds {
            Some(s) => format!("for {s} s (Ctrl-C stops early)"),
            None => "until Ctrl-C".to_string(),
        }
    );
    for s in handle.streams() {
        eprintln!("  {:<6} <- {} ({})", s.source.as_str(), s.device, s.backend);
    }

    // Measure from the session start so the displayed time matches the file length.
    let started = handle.session_start();
    let deadline = duration.map(|d| started + d);
    let meter = tick(Duration::from_millis(500));
    let mut last_flush = Instant::now();
    let mut last_plain_line = Instant::now();
    let mut ok = true;

    loop {
        let timeout = deadline.map_or(Duration::from_secs(3600), |d| {
            d.saturating_duration_since(Instant::now())
        });
        select! {
            recv(rx) -> msg => match msg {
                Ok(chunk) => write_chunk(&mut tracks, &chunk)?,
                Err(_) => break,
            },
            recv(errors) -> msg => {
                if let Ok(err) = msg {
                    if tty { eprintln!(); }
                    eprintln!("kenes-rec: {err}");
                    ok = false;
                    if let Some(t) = tracks.iter_mut().find(|t| t.source == err.source) {
                        t.failed = true;
                    }
                    if tracks.iter().all(|t| t.failed) {
                        break;
                    }
                }
            },
            recv(meter) -> _ => {
                let elapsed = started.elapsed().as_secs_f64();
                let line = format!(
                    "{}  {}",
                    fmt_time(elapsed),
                    tracks.iter_mut().map(|t| t.take_meter(tty)).collect::<Vec<_>>().join("  |  ")
                );
                if tty {
                    eprint!("\r\x1b[2K{line}");
                    let _ = std::io::stderr().flush();
                } else if last_plain_line.elapsed() >= Duration::from_secs(5) {
                    eprintln!("{line}");
                    last_plain_line = Instant::now();
                }
                if last_flush.elapsed() >= Duration::from_secs(2) {
                    for t in &mut tracks {
                        t.writer.flush()?;
                    }
                    last_flush = Instant::now();
                }
            },
            recv(stop_rx) -> _ => break,
            default(timeout) => {
                if deadline.is_some_and(|d| Instant::now() >= d) {
                    break;
                }
            },
        }
    }
    if tty {
        eprintln!();
    }

    // Stopping flushes each stream's last partial chunk into the channel; write those too.
    handle.stop();
    for chunk in rx.try_iter() {
        write_chunk(&mut tracks, &chunk)?;
    }

    eprintln!(
        "kenes-rec: stopped after {}",
        fmt_time(started.elapsed().as_secs_f64())
    );
    for t in tracks {
        let secs = t.writer.len() as f64 / f64::from(SAMPLE_RATE);
        let path = t.writer.path().display().to_string();
        let loudest = to_dbfs(t.loudest);
        t.writer.finalize()?;
        let note = if loudest < -70.0 {
            "  (silent the whole time — was anything playing?)"
        } else {
            ""
        };
        eprintln!("  {path}: {}{note}", fmt_time(secs));
    }
    Ok(ok)
}

fn write_chunk(tracks: &mut [Track], chunk: &AudioChunk) -> Result<()> {
    if let Some(t) = tracks.iter_mut().find(|t| t.source == chunk.source) {
        t.write(chunk)?;
    }
    Ok(())
}

fn fmt_time(secs: f64) -> String {
    let tenths = (secs.max(0.0) * 10.0).round() as u64;
    format!(
        "{:02}:{:02}.{}",
        tenths / 600,
        tenths / 10 % 60,
        tenths % 10
    )
}
