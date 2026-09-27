//! Echo cancellation evaluation on synthetic echo scenes; results in `crates/kenes-aec/EVAL.md`.
//!
//! ```text
//! cargo run --release -p kenes-core --example aec_eval -- [--scenes 24] [--no-asr] [--sweep]
//!     [--export DIR] [--data bench/data]
//! ```
//!
//! Each scene is a short call built from FLEURS/Common Voice utterances: the far end talks
//! alone, the user talks alone, both talk at once, the far end talks alone again. The mic
//! hears the user plus the far end through a simulated laptop speaker and room (see
//! `sim.rs`). The mic goes through `kenes_aec::StreamCanceller` exactly as in a session
//! (32 ms chunks, both sources interleaved on one timeline), then through the recognizer
//! (`Transcriber::transcribe_buffer`) and the transcript-level echo guard.

mod sim;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use kenes_aec::{AecConfig, StreamCanceller};
use kenes_core::echo_guard::{self, EchoGuard};
use kenes_stt::{SttConfig, Transcriber};
use kenes_types::{AudioChunk, Segment, Source};
use sim::*;

struct Args {
    scenes: usize,
    asr: bool,
    sweep: bool,
    export: Option<PathBuf>,
    data: PathBuf,
}

fn parse_args() -> Result<Args> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/data");
    let mut a = Args {
        scenes: 24,
        asr: true,
        sweep: false,
        export: None,
        data: root,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--scenes" => a.scenes = it.next().context("--scenes N")?.parse()?,
            "--no-asr" => a.asr = false,
            "--sweep" => a.sweep = true,
            "--export" => a.export = Some(it.next().context("--export DIR")?.into()),
            "--data" => a.data = it.next().context("--data DIR")?.into(),
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    Ok(a)
}

fn load(dir: &Path) -> Result<Vec<Utt>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "wav"))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .map(|p| {
            let samples = kenes_audio::wav::read(&p)?;
            let text = std::fs::read_to_string(p.with_extension("txt"))
                .unwrap_or_default()
                .trim()
                .to_owned();
            Ok(Utt {
                name: p.file_stem().unwrap_or_default().to_string_lossy().into(),
                samples,
                text,
            })
        })
        .collect()
}

struct Pools {
    ru: Vec<Utt>,
    kk: Vec<Utt>,
    cv: Vec<Utt>,
}

/// The base set: near end alternates ru/kk (FLEURS), far end is the other FLEURS language
/// and Common Voice kk; utterances never repeat between the two sides.
fn base_scenes(
    p: &Pools,
    n: usize,
    tweak: impl Fn(usize, &mut EchoPath),
) -> Vec<(EchoPath, Scene)> {
    (0..n)
        .map(|i| {
            let mut rng = Rng::new(1000 + i as u64);
            let (nears, fars) = if i % 2 == 0 {
                (&p.ru, &p.kk)
            } else {
                (&p.kk, &p.ru)
            };
            let ne = &nears[(i * 2) % nears.len()];
            let dtn = &nears[(i * 2 + 1) % nears.len()];
            let fe1 = &fars[(30 + i * 2) % fars.len()];
            let fe2 = &fars[(31 + i * 2) % fars.len()];
            let dtf = &p.cv[i % p.cv.len()];
            let mut path = EchoPath::random(&mut rng);
            tweak(i, &mut path);
            let scene = build_scene(fe1, ne, dtn, dtf, fe2, &path, &mut rng);
            (path, scene)
        })
        .collect()
}

/// Runs the mic through the canceller as a session does: 32 ms chunks, mic then system at
/// each position. Returns the processed mic (same length and timeline), the CPU time and
/// the canceller's stats.
fn cancel(mic: &[f32], far: &[f32]) -> (Vec<f32>, Duration, kenes_aec::AecStats) {
    let mut sc = StreamCanceller::new(AecConfig::default());
    let t = Instant::now();
    let mut out: Vec<AudioChunk> = Vec::new();
    for (k, m) in mic.chunks(512).enumerate() {
        let start_ms = k as u64 * 32;
        out.extend(sc.push(&AudioChunk {
            source: Source::Mic,
            start_ms,
            samples: m.to_vec(),
        }));
        if let Some(f) = far.chunks(512).nth(k) {
            out.extend(sc.push(&AudioChunk {
                source: Source::System,
                start_ms,
                samples: f.to_vec(),
            }));
        }
    }
    out.extend(sc.finish());
    let took = t.elapsed();
    let stats = sc.stats();
    let mut y = vec![0.0f32; mic.len()];
    for c in out {
        let at = c.start_ms as usize * SR / 1000;
        for (d, s) in y[at..].iter_mut().zip(&c.samples) {
            *d = *s;
        }
    }
    (y, took, stats)
}

#[derive(Default, Clone, Copy)]
struct SignalScore {
    erle1: f64,
    erle2: f64,
    ne_in: f64,
    ne_out: f64,
    ne_gain: f64,
    dt_in: f64,
    dt_out: f64,
}

fn signal_score(s: &Scene, out: &[f32], mic: &[f32]) -> SignalScore {
    let fe: Vec<&Region> = s
        .regions
        .iter()
        .filter(|r| r.kind == Kind::FarOnly)
        .collect();
    let ne = s
        .regions
        .iter()
        .find(|r| r.kind == Kind::NearOnly)
        .expect("near-only region");
    let dt = s
        .regions
        .iter()
        .find(|r| r.kind == Kind::Double)
        .expect("double-talk region");
    SignalScore {
        erle1: erle(mic, out, fe[0].start, fe[0].end),
        erle2: erle(mic, out, fe[1].start, fe[1].end),
        ne_in: si_sdr(&mic[ne.start..ne.end], &s.near[ne.start..ne.end]),
        ne_out: si_sdr(&out[ne.start..ne.end], &s.near[ne.start..ne.end]),
        ne_gain: db(energy(&out[ne.start..ne.end]) / energy(&mic[ne.start..ne.end])),
        dt_in: si_sdr(&mic[dt.start..dt.end], &s.near[dt.start..dt.end]),
        dt_out: si_sdr(&out[dt.start..dt.end], &s.near[dt.start..dt.end]),
    }
}

fn mean(v: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = v.collect();
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

// ---- recognition ----

#[derive(Default, Clone, Copy)]
struct AsrScore {
    /// Words recognized in far-end-only regions (all of them are echo) / far-end words there.
    leaked: usize,
    far_words: usize,
    ne_err: usize,
    ne_ref: usize,
    dt_err: usize,
    dt_ref: usize,
}

impl AsrScore {
    fn add(&mut self, o: &AsrScore) {
        self.leaked += o.leaked;
        self.far_words += o.far_words;
        self.ne_err += o.ne_err;
        self.ne_ref += o.ne_ref;
        self.dt_err += o.dt_err;
        self.dt_ref += o.dt_ref;
    }
    fn row(&self, name: &str) -> String {
        let pct = |a: usize, b: usize| 100.0 * a as f64 / b.max(1) as f64;
        format!(
            "| {name} | {} / {} ({:.1}%) | {:.1}% | {:.1}% |",
            self.leaked,
            self.far_words,
            pct(self.leaked, self.far_words),
            pct(self.ne_err, self.ne_ref),
            pct(self.dt_err, self.dt_ref)
        )
    }
}

/// The region a segment belongs to: the one its midpoint falls in (±300 ms).
fn region_of<'a>(s: &'a Scene, seg: &Segment) -> Option<&'a Region> {
    let mid = (seg.start_ms + seg.end_ms) as usize / 2 * SR / 1000;
    let slack = SR * 3 / 10;
    s.regions
        .iter()
        .find(|r| mid + slack >= r.start && mid < r.end + slack)
}

fn asr_score(s: &Scene, segs: &[Segment]) -> AsrScore {
    let mut sc = AsrScore::default();
    for r in &s.regions {
        let hyp: Vec<String> = segs
            .iter()
            .filter(|g| g.is_final && region_of(s, g).is_some_and(|x| std::ptr::eq(x, r)))
            .flat_map(|g| normalize(&g.text))
            .collect();
        match r.kind {
            Kind::FarOnly => {
                sc.leaked += hyp.len();
                sc.far_words += normalize(&r.far_text).len();
            }
            Kind::NearOnly => {
                let reference = normalize(&r.near_text);
                sc.ne_err += edit_distance(&reference, &hyp);
                sc.ne_ref += reference.len();
            }
            Kind::Double => {
                let reference = normalize(&r.near_text);
                sc.dt_err += edit_distance(&reference, &hyp);
                sc.dt_ref += reference.len();
            }
        }
    }
    sc
}

/// Replays finals through the echo guard as a session would see them: each final
/// arrives 300 ms after its audio ends, preceded by a partial 700 ms into longer
/// utterances; the guard is polled every 100 ms. Returns the mic finals it let through.
fn guard(mic: &[Segment], sys: &[Segment], far: &[f32]) -> (Vec<Segment>, echo_guard::GuardStats) {
    let mut g = EchoGuard::new();
    for (k, c) in far.chunks(512).enumerate() {
        let start = k as u64 * 32;
        g.system_audio(start, start + 32, rms(c));
    }
    let mut events: Vec<(u64, Segment)> = Vec::new();
    for s in mic.iter().chain(sys) {
        if s.end_ms > s.start_ms + 700 {
            let mut p = s.clone();
            p.is_final = false;
            events.push((s.start_ms + 700, p));
        }
        events.push((s.end_ms + 300, s.clone()));
    }
    events.sort_by_key(|e| e.0);
    let t0 = Instant::now();
    let at = |ms: u64| t0 + Duration::from_millis(ms);
    let mut kept = Vec::new();
    let mut take = |v: Vec<Segment>| {
        kept.extend(
            v.into_iter()
                .filter(|s| s.is_final && s.source == Source::Mic && !s.text.is_empty()),
        )
    };
    let end = events.last().map_or(0, |e| e.0) + 5_000;
    let mut next = events.into_iter().peekable();
    for tick in (0..end).step_by(100) {
        while let Some((_, seg)) = next.next_if(|e| e.0 <= tick) {
            take(g.push(seg, at(tick)));
        }
        take(g.poll(at(tick)));
    }
    take(g.finish(at(end)));
    (kept, g.stats().clone())
}

/// Transcribes every job on `threads` recognizers; also returns the recognizer backend used.
fn transcribe_all(
    jobs: Vec<(Source, Vec<f32>)>,
    threads: usize,
) -> Result<(Vec<Vec<Segment>>, String)> {
    let models = kenes_stt::models_dir();
    let cfg = SttConfig {
        models_dir: models,
        num_threads: 2,
        ..SttConfig::default()
    };
    let jobs: Vec<(usize, Source, Vec<f32>)> = jobs
        .into_iter()
        .enumerate()
        .map(|(i, (s, a))| (i, s, a))
        .collect();
    let queue = std::sync::Mutex::new(jobs);
    let results = std::sync::Mutex::new(Vec::new());
    let backend = std::sync::Mutex::new(String::new());
    std::thread::scope(|scope| -> Result<()> {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| -> Result<()> {
                    let mut t = Transcriber::new(cfg.clone())?;
                    *backend.lock().unwrap() = format!("{:?}", t.backend());
                    loop {
                        let Some((i, source, audio)) = queue.lock().unwrap().pop() else {
                            return Ok(());
                        };
                        let segs = t.transcribe_buffer(source, &audio)?;
                        results.lock().unwrap().push((i, segs));
                    }
                })
            })
            .collect();
        for w in workers {
            w.join().expect("transcriber thread panicked")?;
        }
        Ok(())
    })?;
    let mut r = results.into_inner().unwrap();
    r.sort_by_key(|x| x.0);
    Ok((
        r.into_iter().map(|x| x.1).collect(),
        backend.into_inner().unwrap(),
    ))
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let args = parse_args()?;
    let pools = Pools {
        ru: load(&args.data.join("fleurs_ru"))?,
        kk: load(&args.data.join("fleurs_kk"))?,
        cv: load(&args.data.join("cv_kk"))?,
    };
    println!(
        "utterances: {} ru, {} kk (FLEURS), {} kk (Common Voice)\n",
        pools.ru.len(),
        pools.kk.len(),
        pools.cv.len()
    );

    if let Some(dir) = &args.export {
        return export(&pools, dir);
    }
    if args.sweep {
        return sweep(&pools);
    }

    let scenes = base_scenes(&pools, args.scenes, |_, _| {});
    let audio_s: f64 = scenes
        .iter()
        .map(|(_, s)| s.mic.len() as f64 / SR as f64)
        .sum();
    println!(
        "## Signal level: {} scenes, {:.1} min of audio\n",
        scenes.len(),
        audio_s / 60.0
    );
    println!("| # | delay ms | echo dB | clip | RT60 s | ERLE fe1 | ERLE fe2 | NE SI-SDR in→out | NE gain dB | DT SI-SDR in→out |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    let mut cpu = Duration::ZERO;
    let mut processed = Vec::new();
    let mut scores = Vec::new();
    for (i, (p, s)) in scenes.iter().enumerate() {
        let (out, took, _) = cancel(&s.mic, &s.far);
        cpu += took;
        let sc = signal_score(s, &out, &s.mic);
        println!(
            "| {i} | {:.0} | {:.0} | {} | {:.2} | {:.1} | {:.1} | {:.1} → {:.1} | {:+.2} | {:.1} → {:.1} |",
            p.delay_ms, p.echo_db, if p.clip { "yes" } else { "no" }, p.rt60, sc.erle1, sc.erle2, sc.ne_in, sc.ne_out, sc.ne_gain, sc.dt_in, sc.dt_out
        );
        processed.push(out);
        scores.push(sc);
    }
    println!(
        "| **mean** | | | | | **{:.1}** | **{:.1}** | **{:.1} → {:.1}** | **{:+.2}** | **{:.1} → {:.1}** |\n",
        mean(scores.iter().map(|s| s.erle1)),
        mean(scores.iter().map(|s| s.erle2)),
        mean(scores.iter().map(|s| s.ne_in)),
        mean(scores.iter().map(|s| s.ne_out)),
        mean(scores.iter().map(|s| s.ne_gain)),
        mean(scores.iter().map(|s| s.dt_in)),
        mean(scores.iter().map(|s| s.dt_out)),
    );
    println!(
        "AEC CPU: {:.2} s for {:.0} s of audio, RTF {:.4} (one thread)\n",
        cpu.as_secs_f64(),
        audio_s,
        cpu.as_secs_f64() / audio_s
    );

    // Headphones: the same scenes without any echo; the canceller must not hurt the user.
    let mut hs = Vec::new();
    for (_, s) in &scenes {
        let mic: Vec<f32> = s.mic.iter().zip(&s.echo).map(|(m, e)| m - e).collect();
        let (out, _, _) = cancel(&mic, &s.far);
        hs.push((mic, out));
    }
    println!("Headphones (no echo, far end active): NE SI-SDR {:.1} → {:.1} dB, DT SI-SDR {:.1} → {:.1} dB\n",
        mean(scenes.iter().zip(&hs).map(|((_, s), (m, o))| signal_score(s, o, m).ne_in)),
        mean(scenes.iter().zip(&hs).map(|((_, s), (m, o))| signal_score(s, o, m).ne_out)),
        mean(scenes.iter().zip(&hs).map(|((_, s), (m, o))| signal_score(s, o, m).dt_in)),
        mean(scenes.iter().zip(&hs).map(|((_, s), (m, o))| signal_score(s, o, m).dt_out)),
    );

    if !args.asr {
        return Ok(());
    }
    let mut jobs = Vec::new();
    for (i, (_, s)) in scenes.iter().enumerate() {
        jobs.push((Source::Mic, s.mic.clone()));
        jobs.push((Source::Mic, processed[i].clone()));
        jobs.push((Source::System, s.far.clone()));
        jobs.push((Source::Mic, hs[i].0.clone()));
        jobs.push((Source::Mic, hs[i].1.clone()));
    }
    let t = Instant::now();
    let (segs, backend) = transcribe_all(jobs, 6)?;
    eprintln!("transcribed in {:.0} s", t.elapsed().as_secs_f64());
    let mut rows = [AsrScore::default(); 6];
    let mut gstats = [
        echo_guard::GuardStats::default(),
        echo_guard::GuardStats::default(),
    ];
    // (containment, is echo) for every mic final that overlaps system finals.
    let mut pairs: Vec<(f32, bool, usize)> = Vec::new();
    let mut leaks: Vec<String> = Vec::new();
    for (i, (_, s)) in scenes.iter().enumerate() {
        let (raw, aec, sys, hraw, haec) = (
            &segs[i * 5],
            &segs[i * 5 + 1],
            &segs[i * 5 + 2],
            &segs[i * 5 + 3],
            &segs[i * 5 + 4],
        );
        rows[0].add(&asr_score(s, raw));
        rows[1].add(&asr_score(s, aec));
        let (graw, st0) = guard(raw, sys, &s.far);
        let (gaec, st1) = guard(aec, sys, &s.far);
        rows[2].add(&asr_score(s, &graw));
        rows[3].add(&asr_score(s, &gaec));
        for g in gaec
            .iter()
            .filter(|g| region_of(s, g).is_some_and(|r| r.kind == Kind::FarOnly))
        {
            leaks.push(format!(
                "scene {i}, {:.1}–{:.1} s: \"{}\"",
                g.start_ms as f64 / 1000.0,
                g.end_ms as f64 / 1000.0,
                g.text
            ));
        }
        rows[4].add(&asr_score(s, hraw));
        rows[5].add(&asr_score(s, haec));
        for (k, st) in [st0, st1].into_iter().enumerate() {
            gstats[k].held += st.held;
            gstats[k].dropped += st.dropped;
            gstats[k].held_for += st.held_for;
        }
        for m in raw.iter().chain(aec) {
            let Some(r) = region_of(s, m) else { continue };
            let lo = m.start_ms.saturating_sub(1_500);
            let hi = m.end_ms + 1_500;
            let call: Vec<String> = sys
                .iter()
                .filter(|f| f.start_ms <= hi && f.end_ms >= lo)
                .map(|f| echo_guard::normalize(&f.text))
                .collect();
            let text = echo_guard::normalize(&m.text);
            if call.is_empty() || text.is_empty() {
                continue;
            }
            pairs.push((
                echo_guard::containment(&text, &call.join(" ")),
                r.kind == Kind::FarOnly,
                text.split(' ').count(),
            ));
        }
    }
    println!(
        "## Recognition (`Transcriber::transcribe_buffer`, {}, backend {backend})\n",
        SttConfig::default().model_id
    );
    println!("| mic transcript | far-end words leaked (far-only regions) | near-end WER, near-only | near-end WER, double talk |");
    println!("|---|---|---|---|");
    println!("{}", rows[0].row("no AEC"));
    println!("{}", rows[2].row("no AEC + text guard"));
    println!("{}", rows[1].row("AEC"));
    println!("{}", rows[3].row("AEC + text guard"));
    println!("{}", rows[4].row("headphones, no AEC"));
    println!("{}", rows[5].row("headphones, AEC"));
    for (k, name) in ["no AEC", "AEC"].iter().enumerate() {
        let st = &gstats[k];
        println!(
            "\ntext guard after {name}: {} mic finals held, {} dropped, mean hold {:.0} ms",
            st.held,
            st.dropped,
            st.held_for.as_secs_f64() * 1000.0 / st.held.max(1) as f64
        );
    }
    println!("\nEcho left after AEC + text guard (far-only turns):\n");
    for l in &leaks {
        println!("- {l}");
    }
    println!("\n### Guard threshold (mic finals overlapping system finals; one-word finals need an exact word match)\n");
    println!("| threshold | echo finals caught | user finals dropped |");
    println!("|---|---|---|");
    let echo_n = pairs.iter().filter(|p| p.1 && p.2 > 1).count();
    let user_n = pairs.iter().filter(|p| !p.1 && p.2 > 1).count();
    for th in [0.3f32, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
        let caught = pairs.iter().filter(|p| p.1 && p.2 > 1 && p.0 >= th).count();
        let false_drop = pairs
            .iter()
            .filter(|p| !p.1 && p.2 > 1 && p.0 >= th)
            .count();
        println!("| {th:.1} | {caught} / {echo_n} | {false_drop} / {user_n} |");
    }
    Ok(())
}

/// Delay sweep beyond AEC3's own window, and clock drift over a long call.
fn sweep(p: &Pools) -> Result<()> {
    let delays = [
        0.0f32, 50.0, 150.0, 250.0, 350.0, 450.0, 550.0, 700.0, 850.0, 1000.0,
    ];
    let scenes = base_scenes(p, delays.len(), |i, path| path.delay_ms = delays[i]);
    println!("## Delay sweep\n\n| delay ms | ERLE fe1 | ERLE fe2 | reference pre-delay ms |\n|---|---|---|---|");
    for (path, s) in &scenes {
        let (out, _, st) = cancel(&s.mic, &s.far);
        let sc = signal_score(s, &out, &s.mic);
        println!(
            "| {:.0} | {:.1} | {:.1} | {} |",
            path.delay_ms, sc.erle1, sc.erle2, st.reference_delay_ms
        );
    }
    for (ppm, d0) in [(300.0f32, 150.0f32), (-300.0, 400.0), (600.0, 150.0)] {
        let secs = 1200;
        let mut rng = Rng::new(77);
        let mut path = EchoPath::random(&mut rng);
        path.delay_ms = d0;
        path.drift_ppm = ppm;
        path.echo_db = -8.0;
        let s = build_long_scene(&p.ru, &p.kk, &path, secs, &mut rng);
        let (out, _, st) = cancel(&s.mic, &s.far);
        let end = d0 + ppm * 1e-6 * secs as f32 * 1000.0;
        println!("\n### {secs} s call, drift {ppm:+} ppm: echo delay {d0:.0} → {end:.0} ms ({} re-alignments)\n", st.realignments);
        println!("| minute | ERLE (far-only turns) |\n|---|---|");
        let fes: Vec<&Region> = s
            .regions
            .iter()
            .filter(|r| r.kind == Kind::FarOnly)
            .collect();
        for minute in 0..secs / 60 {
            let (a, b) = (minute * 60 * SR, (minute + 1) * 60 * SR);
            let (mut ein, mut eout) = (0.0, 0.0);
            for r in fes.iter().filter(|r| r.start >= a && r.start < b) {
                ein += energy(&s.mic[r.start..r.end]);
                eout += energy(&out[r.start..r.end]);
            }
            if ein > 0.0 {
                println!("| {minute} | {:.1} |", db(ein / eout.max(1e-20)));
            }
        }
    }
    Ok(())
}

/// Writes a ~3 min call for an end-to-end `kenes-cli --replay-mic/--replay-system` run.
fn export(p: &Pools, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut rng = Rng::new(4242);
    let mut path = EchoPath::random(&mut rng);
    path.delay_ms = 120.0;
    path.echo_db = -6.0;
    path.clip = true;
    // Different utterances from the base set's first scenes.
    let nears: Vec<&Utt> = p.ru.iter().skip(40).take(8).collect();
    let fars: Vec<&Utt> = p.kk.iter().skip(10).take(8).collect();
    let own = |v: Vec<&Utt>| {
        v.into_iter()
            .map(|u| Utt {
                name: u.name.clone(),
                samples: u.samples.clone(),
                text: u.text.clone(),
            })
            .collect::<Vec<_>>()
    };
    let s = build_long_scene(&own(nears), &own(fars), &path, 180, &mut rng);
    kenes_audio::wav::write(dir.join("mic_with_echo.wav"), &s.mic)?;
    kenes_audio::wav::write(dir.join("far.wav"), &s.far)?;
    let mut script = String::new();
    for r in &s.regions {
        let t = |x: usize| format!("{:02}:{:02}", x / SR / 60, x / SR % 60);
        let who = match r.kind {
            Kind::FarOnly => format!("far: {}", r.far_text),
            Kind::NearOnly => format!("user: {}", r.near_text),
            Kind::Double => format!("user: {} || far: {}", r.near_text, r.far_text),
        };
        script += &format!("[{}–{}] {:?} {}\n", t(r.start), t(r.end), r.kind, who);
    }
    std::fs::write(dir.join("script.txt"), script)?;
    println!(
        "wrote {} ({:.0} s): echo {} dB, delay {} ms, RT60 {:.2} s",
        dir.display(),
        s.mic.len() as f64 / SR as f64,
        path.echo_db,
        path.delay_ms,
        path.rt60
    );
    Ok(())
}
