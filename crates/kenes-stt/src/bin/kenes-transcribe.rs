//! kenes-transcribe: run the kenes STT pipeline on a WAV file.
//!
//! ```text
//! kenes-transcribe <file.wav> [--model <id>] [--threads 4] [--source mic|system]
//!                  [--simulate-live [--speed 1.0] [--partial-ms 700] [--other <file2.wav>]]
//!                  [--max-segment-ms 20000]
//!                  [--bench] [--models-dir <dir>]
//! kenes-transcribe --list-models
//! kenes-transcribe --download <id> [--verify]
//! ```

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use kenes_stt::{
    available_models_in, ensure_model, is_downloaded, model_path, verify_model, SttConfig,
    Transcriber,
};
use kenes_types::{AudioChunk, Segment, Source, SAMPLE_RATE};

const USAGE: &str = "\
usage: kenes-transcribe <file.wav> [options]
       kenes-transcribe --list-models
       kenes-transcribe --download <id> [--verify]

options:
  --model <id>          model id (default: gigaam-multilingual-ctc)
  --threads <n>         recognizer threads (default: 4)
  --source mic|system   which source to label the audio as (default: mic)
  --simulate-live       stream 100 ms chunks through the live Transcriber
  --speed <x>           with --simulate-live: feed at x times real time (default: 1)
  --other <file.wav>    with --simulate-live: feed this file as the other source at the same time
  --partial-ms <ms>     partial re-decode interval (default: 700)
  --max-segment-ms <ms> force-close utterances longer than this (default: 20000)
  --bench               measure decode time for 5/10/20 s of this audio
  --models-dir <dir>    override $KENES_MODELS_DIR / the default models dir
  --verify              with --download: re-hash every file";

struct Args {
    file: Option<PathBuf>,
    model: String,
    threads: i32,
    source: Source,
    live: bool,
    other: Option<PathBuf>,
    speed: f64,
    partial_ms: u64,
    max_segment_ms: u64,
    bench: bool,
    list: bool,
    download: Option<String>,
    verify: bool,
    models_dir: PathBuf,
}

fn parse_args() -> Result<Args> {
    let defaults = SttConfig::default();
    let mut a = Args {
        file: None,
        model: defaults.model_id,
        threads: defaults.num_threads,
        source: Source::Mic,
        live: false,
        other: None,
        speed: 1.0,
        partial_ms: defaults.partial_interval_ms,
        max_segment_ms: defaults.max_segment_ms,
        bench: false,
        list: false,
        download: None,
        verify: false,
        models_dir: defaults.models_dir,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().with_context(|| format!("{name} needs a value"));
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "--model" => a.model = value("--model")?,
            "--threads" => a.threads = value("--threads")?.parse()?,
            "--source" => {
                a.source = match value("--source")?.as_str() {
                    "mic" => Source::Mic,
                    "system" => Source::System,
                    s => bail!("unknown source {s:?}"),
                }
            }
            "--simulate-live" => a.live = true,
            "--other" => a.other = Some(PathBuf::from(value("--other")?)),
            "--speed" => a.speed = value("--speed")?.parse()?,
            "--partial-ms" => a.partial_ms = value("--partial-ms")?.parse()?,
            "--max-segment-ms" => a.max_segment_ms = value("--max-segment-ms")?.parse()?,
            "--bench" => a.bench = true,
            "--list-models" => a.list = true,
            "--download" => a.download = Some(value("--download")?),
            "--verify" => a.verify = true,
            "--models-dir" => a.models_dir = PathBuf::from(value("--models-dir")?),
            s if s.starts_with('-') => bail!("unknown option {s}\n\n{USAGE}"),
            _ if a.file.is_none() => a.file = Some(PathBuf::from(arg)),
            _ => bail!("only one input file is supported"),
        }
    }
    if a.speed <= 0.0 {
        bail!("--speed must be positive");
    }
    Ok(a)
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = parse_args()?;
    if args.list {
        println!("models dir: {}", args.models_dir.display());
        for m in available_models_in(&args.models_dir) {
            println!(
                "{:<32} {:>5} MB  {:<14} {:<11} {}",
                m.id,
                m.size_mb,
                m.languages.join(","),
                if m.downloaded { "downloaded" } else { "-" },
                m.name
            );
        }
        return Ok(());
    }
    if let Some(id) = &args.download {
        let dir = download(id, &args.models_dir)?;
        println!("{id}: {}", dir.display());
        if args.verify {
            let bad = verify_model(id, &args.models_dir)?;
            if !bad.is_empty() {
                bail!("corrupt files: {bad:?}");
            }
            println!("sha256 ok");
        }
        return Ok(());
    }
    let Some(file) = &args.file else {
        bail!("{USAGE}")
    };

    let audio = read_wav_16k_mono(file)?;
    let audio_s = audio.len() as f64 / SAMPLE_RATE as f64;
    eprintln!("{}: {:.1} s of audio", file.display(), audio_s);

    download(&args.model, &args.models_dir)?;
    let cfg = SttConfig {
        model_id: args.model.clone(),
        models_dir: args.models_dir.clone(),
        num_threads: args.threads,
        partial_interval_ms: args.partial_ms,
        max_segment_ms: args.max_segment_ms,
    };
    let t0 = Instant::now();
    let mut transcriber = Transcriber::new(cfg)?;
    let load_s = t0.elapsed().as_secs_f64();
    eprintln!(
        "loaded {} in {:.2} s ({} threads)",
        args.model, load_s, args.threads
    );

    if args.bench {
        return bench(&mut transcriber, &audio, load_s);
    }
    if args.live {
        let mut feeds = vec![(args.source, audio)];
        if let Some(other) = &args.other {
            let other_source = match args.source {
                Source::Mic => Source::System,
                Source::System => Source::Mic,
            };
            feeds.push((other_source, read_wav_16k_mono(other)?));
        }
        return simulate_live(transcriber, feeds, args.speed);
    }

    let t0 = Instant::now();
    let segs = transcriber.transcribe_buffer(args.source, &audio)?;
    let took = t0.elapsed().as_secs_f64();
    for s in &segs {
        println!(
            "[{} → {}] {:<9} {}",
            fmt_ms(s.start_ms),
            fmt_ms(s.end_ms),
            s.id,
            s.text
        );
    }
    eprintln!(
        "{} segments, {:.2} s for {:.1} s of audio, RTF {:.3}",
        segs.len(),
        took,
        audio_s,
        took / audio_s
    );
    Ok(())
}

fn download(id: &str, dir: &Path) -> Result<PathBuf> {
    if is_downloaded(id, dir) {
        return Ok(model_path(id, dir));
    }
    let mut last = -1i32;
    let path = ensure_model(id, dir, &mut |p| {
        let pct = (p * 100.0).floor() as i32;
        if pct != last {
            let filled = (pct / 4) as usize;
            eprint!(
                "\rdownloading {id} [{}{}] {pct:>3}%",
                "#".repeat(filled),
                " ".repeat(25 - filled)
            );
            let _ = std::io::stderr().flush();
            last = pct;
        }
    })?;
    eprintln!();
    Ok(path)
}

fn fmt_ms(ms: u64) -> String {
    format!("{:02}:{:02}.{:03}", ms / 60_000, ms / 1000 % 60, ms % 1000)
}

/// Read any PCM/float WAV, downmix to mono and resample to 16 kHz.
fn read_wav_16k_mono(path: &Path) -> Result<Vec<f32>> {
    let mut r =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = r.spec();
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    let ch = spec.channels.max(1) as usize;
    let mono: Vec<f32> = if ch == 1 {
        interleaved
    } else {
        interleaved
            .chunks(ch)
            .map(|f| f.iter().sum::<f32>() / ch as f32)
            .collect()
    };
    if spec.sample_rate == SAMPLE_RATE {
        return Ok(mono);
    }
    let rs = sherpa_onnx::LinearResampler::create(spec.sample_rate as i32, SAMPLE_RATE as i32)
        .context("creating resampler")?;
    Ok(rs.resample(&mono, true))
}

fn simulate_live(t: Transcriber, feeds: Vec<(Source, Vec<f32>)>, speed: f64) -> Result<()> {
    let (atx, arx) = crossbeam_channel::unbounded::<AudioChunk>();
    let (stx, srx) = crossbeam_channel::unbounded::<Segment>();
    let worker = t.spawn(arx, stx);
    let chunk = SAMPLE_RATE as usize / 10;
    // Interleave both sources' 100 ms chunks on one timeline.
    let mut chunks: Vec<AudioChunk> = feeds
        .iter()
        .flat_map(|(source, audio)| {
            audio.chunks(chunk).enumerate().map(|(i, c)| AudioChunk {
                source: *source,
                start_ms: i as u64 * 100,
                samples: c.to_vec(),
            })
        })
        .collect();
    chunks.sort_by_key(|c| c.start_ms);
    let audio_ms =
        feeds.iter().map(|(_, a)| a.len()).max().unwrap_or(0) as u64 * 1000 / SAMPLE_RATE as u64;
    let start = Instant::now();
    // When the audio at session time `ms` was handed to the transcriber.
    let fed_at = move |ms: u64| start + Duration::from_secs_f64(ms as f64 / 1000.0 / speed);

    let feeder = std::thread::spawn(move || {
        for c in chunks {
            // A chunk can only be sent once all of its audio "exists".
            let ready = fed_at(c.start_ms + c.duration_ms());
            if let Some(wait) = ready.checked_duration_since(Instant::now()) {
                std::thread::sleep(wait);
            }
            if atx.send(c).is_err() {
                break;
            }
        }
    });

    let mut partial_lat = Vec::new();
    let mut final_lat = Vec::new();
    let mut partial_line = false;
    for s in srx.iter() {
        let now = Instant::now();
        let lat = now
            .saturating_duration_since(fed_at(s.end_ms.min(audio_ms)))
            .as_secs_f64()
            * speed;
        let mut err = std::io::stderr();
        if s.is_final {
            final_lat.push(lat);
            if partial_line {
                eprint!("\r\x1b[2K");
            }
            partial_line = false;
            println!(
                "[{} → {}] {:<9} {}",
                fmt_ms(s.start_ms),
                fmt_ms(s.end_ms),
                s.id,
                s.text
            );
        } else {
            partial_lat.push(lat);
            let tail: String = {
                let chars: Vec<char> = s.text.chars().collect();
                chars[chars.len().saturating_sub(90)..].iter().collect()
            };
            eprint!("\r\x1b[2K… {:<9} {}", s.id, tail);
            partial_line = true;
        }
        let _ = err.flush();
    }
    feeder.join().ok();
    worker.join().ok();
    if partial_line {
        eprintln!();
    }
    let wall = start.elapsed().as_secs_f64();
    eprintln!(
        "{:.1} s of audio at {speed}x in {:.1} s; {} partials, latency (audio time) median {:.2} s, p90 {:.2} s; \
         {} finals, latency from utterance end median {:.2} s, p90 {:.2} s",
        audio_ms as f64 / 1000.0,
        wall,
        partial_lat.len(),
        percentile(&mut partial_lat, 0.5),
        percentile(&mut partial_lat, 0.9),
        final_lat.len(),
        percentile(&mut final_lat, 0.5),
        percentile(&mut final_lat, 0.9),
    );
    Ok(())
}

fn percentile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn bench(t: &mut Transcriber, audio: &[f32], load_s: f64) -> Result<()> {
    // Loop the file if it is shorter than the longest slice.
    let need = 20 * SAMPLE_RATE as usize;
    let mut long = Vec::with_capacity(need);
    while long.len() < need {
        long.extend_from_slice(&audio[..audio.len().min(need - long.len())]);
    }
    let t0 = Instant::now();
    t.recognize(&long[..SAMPLE_RATE as usize])?;
    println!(
        "load {:.2} s, first decode (1 s) {:.3} s",
        load_s,
        t0.elapsed().as_secs_f64()
    );
    for secs in [1usize, 2, 5, 10, 15, 20] {
        let slice = &long[..secs * SAMPLE_RATE as usize];
        let mut times: Vec<f64> = (0..3)
            .map(|_| {
                let t0 = Instant::now();
                t.recognize(slice).map(|_| t0.elapsed().as_secs_f64())
            })
            .collect::<Result<_>>()?;
        let med = percentile(&mut times, 0.5);
        println!(
            "{secs:>3} s audio: decode {med:.3} s  RTF {:.3}",
            med / secs as f64
        );
    }
    if let Some((rss, hwm)) = memory_kb() {
        println!("RSS {} MB, peak {} MB", rss / 1024, hwm / 1024);
    }
    Ok(())
}

/// Current and peak resident memory (Linux only).
fn memory_kb() -> Option<(u64, u64)> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let field = |name: &str| {
        status
            .lines()
            .find(|l| l.starts_with(name))?
            .split_whitespace()
            .nth(1)?
            .parse::<u64>()
            .ok()
    };
    Some((field("VmRSS:")?, field("VmHWM:")?))
}
