//! WER / CER / RTF of kenes-stt on the benchmark sets in `bench/data` (see `bench/README.md`),
//! with the same text normalisation as `bench/run_bench.py`.
//!
//! ```text
//! cargo run --release -p kenes-stt --example bench_sets -- \
//!     [--model gigaam-multilingual-ctc] [--backend auto|ort|sherpa] [--threads 4] \
//!     [--sets fleurs_ru,fleurs_kk,cv_kk,codeswitch] [--mode whole|unpadded|vad] \
//!     [--limit N] [--repeat K] [--data bench/data] [--out results.json]
//! ```
//!
//! Modes: `whole` decodes each clip in one call (`Transcriber::recognize`, which appends 0.3 s
//! of silence), `unpadded` does the same without the padding (what `run_bench.py` does), and
//! `vad` runs `transcribe_buffer`, the live worker's segmentation, and joins the finals.
//! `--out` writes the per-utterance results in `run_bench.py`'s JSON layout, so its `score()`
//! can recompute the metrics (including the code-switch per-language split).

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use kenes_stt::{ensure_model, models_dir, SttBackend, SttConfig, Transcriber};
use kenes_types::{Source, SAMPLE_RATE};
use serde_json::{json, Value};

struct Args {
    model: String,
    backend: SttBackend,
    threads: i32,
    sets: Vec<String>,
    mode: String,
    limit: usize,
    repeat: usize,
    data: PathBuf,
    out: Option<PathBuf>,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        model: SttConfig::default().model_id,
        backend: SttBackend::Auto,
        threads: 4,
        sets: ["fleurs_ru", "fleurs_kk", "cv_kk", "codeswitch"]
            .map(String::from)
            .to_vec(),
        mode: "whole".into(),
        limit: usize::MAX,
        repeat: 1,
        data: Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/data"),
        out: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().with_context(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--model" => a.model = value()?,
            "--backend" => a.backend = value()?.parse()?,
            "--threads" => a.threads = value()?.parse()?,
            "--sets" => a.sets = value()?.split(',').map(String::from).collect(),
            "--mode" => a.mode = value()?,
            "--limit" => a.limit = value()?.parse()?,
            "--repeat" => a.repeat = value()?.parse::<usize>()?.max(1),
            "--data" => a.data = value()?.into(),
            "--out" => a.out = Some(value()?.into()),
            other => bail!("unknown argument {other}"),
        }
    }
    if !["whole", "unpadded", "vad"].contains(&a.mode.as_str()) {
        bail!("--mode must be whole, unpadded or vad");
    }
    Ok(a)
}

/// `run_bench.normalize`: lowercase, ё→е, anything but letters/digits/whitespace → space,
/// collapse spaces.
fn normalize(s: &str) -> String {
    let s: String = s
        .to_lowercase()
        .replace('ё', "е")
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn edit_distance<T: PartialEq>(r: &[T], h: &[T]) -> usize {
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for (i, rc) in r.iter().enumerate() {
        let mut cur = vec![i + 1; h.len() + 1];
        for (j, hc) in h.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(rc != hc))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[h.len()]
}

fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut r = hound::WavReader::open(path).with_context(|| format!("{}", path.display()))?;
    let spec = r.spec();
    if spec.sample_rate != SAMPLE_RATE || spec.channels != 1 {
        bail!("{}: need 16 kHz mono", path.display());
    }
    Ok(match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    })
}

/// User + system CPU seconds of this process (Linux), for CPU-s per audio-s.
fn cpu_seconds() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let rest = &stat[stat.rfind(')')? + 2..];
    let f: Vec<&str> = rest.split_whitespace().collect();
    let ticks = f.get(11)?.parse::<f64>().ok()? + f.get(12)?.parse::<f64>().ok()?;
    Some(ticks / 100.0)
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let args = parse_args()?;
    let dir = models_dir();
    ensure_model(&args.model, &dir, &mut |_| {})?;
    let mut t = Transcriber::with_backend(
        SttConfig {
            model_id: args.model.clone(),
            models_dir: dir,
            num_threads: args.threads,
            ..SttConfig::default()
        },
        args.backend,
    )?;
    eprintln!(
        "{} on {} ({} threads, mode {})",
        args.model,
        t.backend(),
        args.threads,
        args.mode
    );

    let run = |t: &mut Transcriber, audio: &[f32]| -> Result<String> {
        match args.mode.as_str() {
            "whole" => t.recognize(audio),
            "unpadded" => t.recognize_unpadded(audio),
            _ => Ok(t
                .transcribe_buffer(Source::Mic, audio)?
                .iter()
                .filter(|s| s.is_final && !s.text.is_empty())
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" ")),
        }
    };

    let mut warmed = false;
    let mut sets_json = serde_json::Map::new();
    let mut table = Vec::new();
    for set in &args.sets {
        let manifest = args.data.join(set).join("manifest.jsonl");
        let text = std::fs::read_to_string(&manifest)
            .with_context(|| format!("reading {}", manifest.display()))?;
        let (mut word_err, mut words, mut char_err, mut chars) = (0, 0, 0, 0);
        let (mut audio_s, mut proc_s, mut cpu_s) = (0.0, 0.0, 0.0);
        let mut items = Vec::new();
        for line in text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .take(args.limit)
        {
            let item: Value = serde_json::from_str(line)?;
            let wav = args
                .data
                .join(set)
                .join(item["wav"].as_str().unwrap_or_default());
            let reference = item["text"].as_str().unwrap_or_default().to_string();
            let audio = read_wav(&wav)?;
            if !warmed {
                run(&mut t, &audio)?;
                warmed = true;
            }
            let mut best: Option<(f64, f64, String)> = None;
            for _ in 0..args.repeat {
                let c0 = cpu_seconds().unwrap_or(0.0);
                let t0 = Instant::now();
                let hyp = run(&mut t, &audio)?;
                let dt = t0.elapsed().as_secs_f64();
                let dc = cpu_seconds().unwrap_or(0.0) - c0;
                if best.as_ref().is_none_or(|b| dt < b.0) {
                    best = Some((dt, dc, hyp));
                }
            }
            let (dt, dc, hyp) = best.expect("repeat >= 1");
            let (r, h) = (normalize(&reference), normalize(&hyp));
            let (rw, hw): (Vec<&str>, Vec<&str>) =
                (r.split(' ').collect(), h.split_whitespace().collect());
            let (rc, hc): (Vec<char>, Vec<char>) = (r.chars().collect(), h.chars().collect());
            word_err += edit_distance(&rw, &hw);
            words += rw.len();
            char_err += edit_distance(&rc, &hc);
            chars += rc.len();
            let duration = audio.len() as f64 / SAMPLE_RATE as f64;
            audio_s += duration;
            proc_s += dt;
            cpu_s += dc;
            eprintln!(
                "  [{set}] {}: rtf={:.3} | {}",
                item["id"].as_str().unwrap_or("?"),
                dt / duration,
                hyp.chars().take(90).collect::<String>()
            );
            let mut out = json!({
                "id": item["id"], "ref": reference, "hyp": hyp,
                "duration": duration, "proc_s": dt, "cpu_s": dc,
            });
            for k in ["parts", "part_langs", "order"] {
                if let Some(v) = item.get(k) {
                    out[k] = v.clone();
                }
            }
            items.push(out);
        }
        let wer = 100.0 * word_err as f64 / words.max(1) as f64;
        let cer = 100.0 * char_err as f64 / chars.max(1) as f64;
        let rtf = proc_s / audio_s.max(1e-9);
        eprintln!("== {set}: WER {wer:.2}  CER {cer:.2}  RTF {rtf:.4}");
        table.push(format!(
            "| {set} | {} | {wer:.2} | {cer:.2} | {rtf:.4} | {:.3} |",
            items.len(),
            cpu_s / audio_s.max(1e-9)
        ));
        sets_json.insert(
            set.clone(),
            json!({
                "metrics": {"n": items.len(), "wer": wer, "cer": cer, "rtf": rtf,
                            "audio_s": audio_s, "proc_s": proc_s, "cpu_s": cpu_s},
                "items": items,
            }),
        );
    }
    println!(
        "{} / {} / {} threads / mode {}",
        args.model,
        t.backend(),
        args.threads,
        args.mode
    );
    println!("| set | n | WER % | CER % | RTF | CPU-s/s |\n|---|---|---|---|---|---|");
    for row in &table {
        println!("{row}");
    }
    if let Some(out) = &args.out {
        let doc = json!({
            "model": format!("{}-{}-{}", args.model, t.backend(), args.mode),
            "desc": format!("kenes-stt {} backend, mode {}", t.backend(), args.mode),
            "kind": "kenes_stt", "threads": args.threads, "repeat": args.repeat,
            "sets": Value::Object(sets_json),
        });
        std::fs::write(out, serde_json::to_string_pretty(&doc)?)?;
        eprintln!("wrote {}", out.display());
    }
    Ok(())
}
