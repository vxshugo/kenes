//! Evaluation harness for kenes-speakers; the numbers in `EVAL.md` come from here.
//!
//! Data: `testdata-cache/meetings/mNN/` from `eval/build_meetings.py` (synthetic ru/kk
//! meetings, conditions `clean`, `call`, `room`); models in `testdata-cache/models/`
//! (`testdata-cache/fetch.sh`).
//!
//! ```text
//! diarize_eval embed  [--models a.onnx,b.onnx] [--jobs 6]   # cache embeddings per model
//! diarize_eval pairs  [--models ...]                         # threshold-free: pair EER
//! diarize_eval tune   --model M [--quick]                    # grid search on the tune split
//! diarize_eval report --model M [--set k=v,...]              # one config, tune + test splits
//! diarize_eval curve  --model M --key K --from a --to b --step s [--set ...]  # sensitivity
//! diarize_eval bench  --model M [--threads 1] [--reps 30]    # CPU time, init, memory
//! diarize_eval bench-recluster [--dim 192]                  # recluster time for long meetings
//!
//! Pipeline-like segmentation (kenes-stt VAD finals instead of oracle turns):
//! diarize_eval segment [--jobs 4]                   # run kenes-stt over call/room, cache finals
//! diarize_eval embed-real --model M                 # embed finals + change-detection windows
//! diarize_eval change-eval --model M [--hop 500]    # change detection vs threshold
//! diarize_eval split --model M --set change_threshold=X [--hop 500]  # cut finals, embed pieces
//! diarize_eval tune-real|report-real --model M [--variant raw|split_...] [--set ...]
//! diarize_eval embed-pieces --model X --from-model M --variant V   # same pieces, other model
//! ```
//! Run with `cargo run --release -p kenes-speakers --example diarize_eval -- <cmd> ...`.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use kenes_speakers::{
    metrics, recluster, ClusterConfig, ClusterItem, Embedder, OnlineClusterer, ME,
};
use serde::Deserialize;

const CONDITIONS: [&str; 3] = ["clean", "call", "room"];
/// Segments shorter than this are never embedded (the cache stores NaN for them).
const MIN_CACHE_MS: u64 = 300;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata-cache")
}

#[derive(Deserialize, Clone)]
struct Seg {
    speaker: String,
    start_ms: u64,
    end_ms: u64,
}

#[derive(Deserialize, Clone)]
struct Meeting {
    id: String,
    split: String,
    me: String,
    segments: Vec<Seg>,
}

fn load_meetings() -> Result<Vec<Meeting>> {
    let dir = root().join("meetings");
    let mut out = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .with_context(|| format!("{} (run eval/build_meetings.py)", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("manifest.json").exists())
        .collect();
    entries.sort();
    for p in entries {
        let m: Meeting = serde_json::from_str(&std::fs::read_to_string(p.join("manifest.json"))?)?;
        out.push(m);
    }
    if out.is_empty() {
        bail!("no meetings in {}", dir.display());
    }
    Ok(out)
}

fn read_wav(p: &Path) -> Result<Vec<f32>> {
    let mut r = hound::WavReader::open(p).with_context(|| p.display().to_string())?;
    let spec = r.spec();
    if spec.sample_rate != 16_000 || spec.channels != 1 {
        bail!("{}: expected 16 kHz mono", p.display());
    }
    Ok(r.samples::<i16>()
        .map(|s| s.unwrap_or(0) as f32 / 32768.0)
        .collect())
}

fn model_stem(m: &str) -> String {
    Path::new(m)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| m.to_string())
}

fn model_path(m: &str) -> PathBuf {
    let p = PathBuf::from(m);
    if p.exists() {
        p
    } else {
        let name = if m.ends_with(".onnx") {
            m.to_string()
        } else {
            format!("{m}.onnx")
        };
        root().join("models").join(name)
    }
}

fn all_models() -> Result<Vec<String>> {
    let mut v: Vec<String> = std::fs::read_dir(root().join("models"))?
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".onnx"))
        .collect();
    v.sort();
    Ok(v)
}

// ------------------------------------------------------------------ embedding cache

type Emb = Option<Vec<f32>>;

fn cache_path(model: &str, meeting: &str, what: &str) -> PathBuf {
    root()
        .join("emb")
        .join(model_stem(model))
        .join(format!("{meeting}_{what}.bin"))
}

fn write_cache(p: &Path, embs: &[Emb], dim: usize) -> Result<()> {
    std::fs::create_dir_all(p.parent().unwrap())?;
    let mut buf = Vec::with_capacity(8 + embs.len() * dim * 4);
    buf.extend_from_slice(&(embs.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(dim as u32).to_le_bytes());
    for e in embs {
        for k in 0..dim {
            let x = e.as_ref().map_or(f32::NAN, |v| v[k]);
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(p, buf)?;
    Ok(())
}

fn read_cache(p: &Path) -> Result<Vec<Emb>> {
    let b = std::fs::read(p).with_context(|| format!("{} (run `embed` first)", p.display()))?;
    let n = u32::from_le_bytes(b[0..4].try_into()?) as usize;
    let dim = u32::from_le_bytes(b[4..8].try_into()?) as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = 8 + i * dim * 4;
        let v: Vec<f32> = (0..dim)
            .map(|k| f32::from_le_bytes(b[off + 4 * k..off + 4 * k + 4].try_into().unwrap()))
            .collect();
        out.push(if v[0].is_nan() { None } else { Some(v) });
    }
    Ok(out)
}

struct Cache {
    /// [meeting][condition] → embedding per segment
    segs: Vec<Vec<Vec<Emb>>>,
    /// [meeting][condition] → enrollment embedding (clean, room; None for call)
    enroll: Vec<Vec<Emb>>,
}

fn load_cache(model: &str, meetings: &[Meeting]) -> Result<Cache> {
    let mut segs = Vec::new();
    let mut enroll = Vec::new();
    for m in meetings {
        let mut s = Vec::new();
        let mut e = Vec::new();
        for c in CONDITIONS {
            s.push(read_cache(&cache_path(model, &m.id, c))?);
            e.push(if c == "call" {
                None
            } else {
                read_cache(&cache_path(model, &m.id, &format!("enroll_{c}")))?
                    .pop()
                    .flatten()
            });
        }
        segs.push(s);
        enroll.push(e);
    }
    Ok(Cache { segs, enroll })
}

fn cmd_embed(models: Vec<String>, jobs: usize) -> Result<()> {
    let meetings = load_meetings()?;
    let mut work: Vec<(String, usize, &str)> = Vec::new();
    for model in &models {
        for (mi, m) in meetings.iter().enumerate() {
            for c in CONDITIONS {
                if !cache_path(model, &m.id, c).exists() {
                    work.push((model.clone(), mi, c));
                }
            }
        }
    }
    println!(
        "{} (model, meeting, condition) jobs on {jobs} threads",
        work.len()
    );
    let work = std::sync::Mutex::new(work);
    let stats: std::sync::Mutex<BTreeMap<String, (f64, f64)>> = Default::default();
    std::thread::scope(|s| {
        for _ in 0..jobs.max(1) {
            s.spawn(|| -> Result<()> {
                let mut loaded: Option<(String, Embedder)> = None;
                loop {
                    let Some((model, mi, cond)) = work.lock().unwrap().pop() else {
                        return Ok(());
                    };
                    if loaded.as_ref().map(|(n, _)| n != &model).unwrap_or(true) {
                        loaded = Some((model.clone(), Embedder::new(&model_path(&model), 1)?));
                    }
                    let emb = &mut loaded.as_mut().unwrap().1;
                    let m = &meetings[mi];
                    let dir = root().join("meetings").join(&m.id);
                    let audio = read_wav(&dir.join(format!("{cond}.wav")))?;
                    let t = Instant::now();
                    let mut audio_s = 0.0;
                    let mut out = Vec::new();
                    for seg in &m.segments {
                        let a = (seg.start_ms * 16) as usize;
                        let b = ((seg.end_ms * 16) as usize).min(audio.len());
                        if seg.end_ms - seg.start_ms < MIN_CACHE_MS || b <= a {
                            out.push(None);
                            continue;
                        }
                        audio_s += (b - a) as f64 / 16000.0;
                        out.push(emb.embed(&audio[a..b]).ok());
                    }
                    let dt = t.elapsed().as_secs_f64();
                    write_cache(&cache_path(&model, &m.id, cond), &out, emb.dim())?;
                    if cond != "call" {
                        let e = read_wav(&dir.join(format!("enroll_{cond}.wav")))?;
                        let v = emb.embed(&e).ok();
                        write_cache(
                            &cache_path(&model, &m.id, &format!("enroll_{cond}")),
                            &[v],
                            emb.dim(),
                        )?;
                    }
                    let mut st = stats.lock().unwrap();
                    let e = st.entry(model_stem(&model)).or_default();
                    e.0 += audio_s;
                    e.1 += dt;
                    println!(
                        "  {} {} {cond}: {:.1} s audio in {:.2} s",
                        model_stem(&model),
                        m.id,
                        audio_s,
                        dt
                    );
                }
            });
        }
    });
    for (m, (a, t)) in stats.into_inner().unwrap() {
        println!("{m}: RTF {:.4} (1 thread, {} parallel jobs)", t / a, jobs);
    }
    Ok(())
}

// ------------------------------------------------------------------ realistic segmentation

/// A final utterance from the kenes-stt pipeline (VAD segmentation of the meeting audio).
#[derive(serde::Serialize, Deserialize, Clone, Debug)]
struct RealSeg {
    start_ms: u64,
    end_ms: u64,
    text: String,
}

const REAL_CONDITIONS: [&str; 2] = ["call", "room"];

fn realseg_path(meeting: &str, cond: &str) -> PathBuf {
    root()
        .join("realseg")
        .join(format!("{meeting}_{cond}.json"))
}

/// Run kenes-stt's Transcriber (VAD + GigaAM) over call.wav / room.wav and cache the finals.
fn cmd_segment(jobs: usize) -> Result<()> {
    let meetings = load_meetings()?;
    let mut work: Vec<(usize, &str)> = Vec::new();
    for (mi, m) in meetings.iter().enumerate() {
        for c in REAL_CONDITIONS {
            if !realseg_path(&m.id, c).exists() {
                work.push((mi, c));
            }
        }
    }
    println!("{} (meeting, condition) jobs", work.len());
    let work = std::sync::Mutex::new(work);
    std::thread::scope(|s| {
        for _ in 0..jobs.max(1) {
            s.spawn(|| -> Result<()> {
                let mut stt = kenes_stt::Transcriber::new(kenes_stt::SttConfig {
                    num_threads: 2,
                    ..Default::default()
                })?;
                loop {
                    let Some((mi, cond)) = work.lock().unwrap().pop() else {
                        return Ok(());
                    };
                    let m = &meetings[mi];
                    let audio = read_wav(
                        &root()
                            .join("meetings")
                            .join(&m.id)
                            .join(format!("{cond}.wav")),
                    )?;
                    let t = Instant::now();
                    let source = if cond == "call" {
                        kenes_types::Source::System
                    } else {
                        kenes_types::Source::Mic
                    };
                    let segs: Vec<RealSeg> = stt
                        .transcribe_buffer(source, &audio)?
                        .into_iter()
                        .filter(|s| s.is_final && !s.text.trim().is_empty())
                        .map(|s| RealSeg {
                            start_ms: s.start_ms,
                            end_ms: s.end_ms,
                            text: s.text,
                        })
                        .collect();
                    let p = realseg_path(&m.id, cond);
                    std::fs::create_dir_all(p.parent().unwrap())?;
                    std::fs::write(&p, serde_json::to_string(&segs)?)?;
                    println!(
                        "  {} {cond}: {} finals in {:.0} s",
                        m.id,
                        segs.len(),
                        t.elapsed().as_secs_f64()
                    );
                }
            });
        }
    });
    Ok(())
}

// ------------------------------------------------------------------ realistic: caches

fn real_emb_path(model: &str, variant: &str, meeting: &str, cond: &str) -> PathBuf {
    root()
        .join("emb-real")
        .join(model_stem(model))
        .join(variant)
        .join(format!("{meeting}_{cond}.bin"))
}

fn win_path(model: &str, meeting: &str, cond: &str) -> PathBuf {
    root()
        .join("win")
        .join(model_stem(model))
        .join(format!("{meeting}_{cond}.bin"))
}

fn pieces_path(model: &str, variant: &str, meeting: &str, cond: &str) -> PathBuf {
    root()
        .join("emb-real")
        .join(model_stem(model))
        .join(variant)
        .join(format!("{meeting}_{cond}.json"))
}

/// Finals shorter than this are never split (the kenes-stt hook threshold).
const SPLIT_MIN_MS: u64 = 2500;
/// Window hop of the cached change-detection windows.
const CACHE_HOP_MS: u64 = 250;

fn load_realsegs(meeting: &str, cond: &str) -> Result<Vec<RealSeg>> {
    let p = realseg_path(meeting, cond);
    Ok(serde_json::from_str(
        &std::fs::read_to_string(&p).with_context(|| format!("{} (run `segment`)", p.display()))?,
    )?)
}

fn write_windows(p: &Path, all: &[kenes_speakers::change::Windows], dim: usize) -> Result<()> {
    std::fs::create_dir_all(p.parent().unwrap())?;
    let mut b: Vec<u8> = Vec::new();
    let mut put = |x: u32| b.extend_from_slice(&x.to_le_bytes());
    put(all.len() as u32);
    put(dim as u32);
    for w in all {
        put(w.starts.len() as u32);
        put(w.win as u32);
        put(w.len as u32);
        for &s in &w.starts {
            put(s as u32);
        }
    }
    for w in all {
        for e in &w.embs {
            for x in e {
                b.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    std::fs::write(p, b)?;
    Ok(())
}

fn read_windows(p: &Path) -> Result<Vec<kenes_speakers::change::Windows>> {
    let b = std::fs::read(p).with_context(|| format!("{} (run `embed-real`)", p.display()))?;
    let mut off = 0usize;
    let mut get = || {
        let x = u32::from_le_bytes(b[off..off + 4].try_into().unwrap());
        off += 4;
        x as usize
    };
    let n = get();
    let dim = get();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let nw = get();
        let win = get();
        let len = get();
        let starts: Vec<usize> = (0..nw).map(|_| get()).collect();
        out.push(kenes_speakers::change::Windows {
            win,
            len,
            starts,
            embs: Vec::new(),
        });
    }
    for w in out.iter_mut() {
        for _ in 0..w.starts.len() {
            let v: Vec<f32> = (0..dim)
                .map(|k| f32::from_le_bytes(b[off + 4 * k..off + 4 * k + 4].try_into().unwrap()))
                .collect();
            off += 4 * dim;
            w.embs.push(v);
        }
    }
    Ok(out)
}

/// Embed the pipeline's finals (variant "raw") and cache change-detection windows.
fn cmd_embed_real(model: &str, jobs: usize) -> Result<()> {
    let meetings = load_meetings()?;
    let mut work: Vec<(usize, &str)> = Vec::new();
    for (mi, m) in meetings.iter().enumerate() {
        for c in REAL_CONDITIONS {
            if !real_emb_path(model, "raw", &m.id, c).exists()
                || !win_path(model, &m.id, c).exists()
            {
                work.push((mi, c));
            }
        }
    }
    let work = std::sync::Mutex::new(work);
    std::thread::scope(|s| {
        for _ in 0..jobs.max(1) {
            s.spawn(|| -> Result<()> {
                let mut emb = Embedder::new(&model_path(model), 1)?;
                loop {
                    let Some((mi, cond)) = work.lock().unwrap().pop() else {
                        return Ok(());
                    };
                    let m = &meetings[mi];
                    let audio = read_wav(
                        &root()
                            .join("meetings")
                            .join(&m.id)
                            .join(format!("{cond}.wav")),
                    )?;
                    let segs = load_realsegs(&m.id, cond)?;
                    let mut embs = Vec::new();
                    let mut wins = Vec::new();
                    for sg in &segs {
                        let a = (sg.start_ms * 16) as usize;
                        let b = ((sg.end_ms * 16) as usize).min(audio.len());
                        let x = &audio[a.min(b)..b];
                        embs.push(if sg.end_ms - sg.start_ms >= MIN_CACHE_MS {
                            emb.embed(x).ok()
                        } else {
                            None
                        });
                        wins.push(if sg.end_ms - sg.start_ms >= SPLIT_MIN_MS {
                            kenes_speakers::change::window_embeddings(
                                &mut emb,
                                x,
                                kenes_speakers::change::CHANGE_WINDOW_MS,
                                CACHE_HOP_MS,
                            )?
                        } else {
                            Default::default()
                        });
                    }
                    write_cache(&real_emb_path(model, "raw", &m.id, cond), &embs, emb.dim())?;
                    write_windows(&win_path(model, &m.id, cond), &wins, emb.dim())?;
                    println!("  {} {cond}: {} finals", m.id, segs.len());
                }
            });
        }
    });
    Ok(())
}

/// Windows as if they had been computed with `hop_ms` (a multiple of the cache hop).
fn subsample(w: &kenes_speakers::change::Windows, hop_ms: u64) -> kenes_speakers::change::Windows {
    let hop = (hop_ms * 16) as usize;
    let last = w.starts.last().copied();
    let keep: Vec<usize> = (0..w.starts.len())
        .filter(|&i| w.starts[i].is_multiple_of(hop) || Some(w.starts[i]) == last)
        .collect();
    kenes_speakers::change::Windows {
        win: w.win,
        len: w.len,
        starts: keep.iter().map(|&i| w.starts[i]).collect(),
        embs: keep.iter().map(|&i| w.embs[i].clone()).collect(),
    }
}

// ------------------------------------------------------------------ realistic: truth & scoring

const FRAME_MS: u64 = 10;

/// Who speaks in every 10 ms frame (bit per speaker index).
struct Truth {
    speakers: Vec<String>,
    frames: Vec<u32>,
    me: Option<usize>,
    /// (start_ms, end_ms, speaker index), sorted by start.
    turns: Vec<(u64, u64, usize)>,
}

fn truth(m: &Meeting) -> Truth {
    let mut speakers: Vec<String> = Vec::new();
    let mut turns = Vec::new();
    for s in &m.segments {
        let k = match speakers.iter().position(|x| x == &s.speaker) {
            Some(k) => k,
            None => {
                speakers.push(s.speaker.clone());
                speakers.len() - 1
            }
        };
        turns.push((s.start_ms, s.end_ms, k));
    }
    turns.sort();
    let end = turns.iter().map(|t| t.1).max().unwrap_or(0);
    let mut frames = vec![0u32; (end / FRAME_MS + 1) as usize];
    for &(a, b, k) in &turns {
        for f in (a / FRAME_MS)..(b / FRAME_MS) {
            frames[f as usize] |= 1 << k;
        }
    }
    let me = speakers.iter().position(|x| x == &m.me);
    Truth {
        speakers,
        frames,
        me,
        turns,
    }
}

impl Truth {
    fn span(&self, a_ms: u64, b_ms: u64) -> &[u32] {
        let n = self.frames.len();
        let a = ((a_ms / FRAME_MS) as usize).min(n);
        let b = ((b_ms / FRAME_MS) as usize).min(n);
        &self.frames[a..b.max(a)]
    }
    /// Frames per speaker inside [a, b).
    fn counts(&self, a_ms: u64, b_ms: u64) -> Vec<u32> {
        let mut c = vec![0u32; self.speakers.len()];
        for &f in self.span(a_ms, b_ms) {
            for (k, ck) in c.iter_mut().enumerate() {
                if f >> k & 1 == 1 {
                    *ck += 1;
                }
            }
        }
        c
    }
    fn is_mixed(&self, a_ms: u64, b_ms: u64) -> bool {
        self.counts(a_ms, b_ms)
            .iter()
            .filter(|&&c| c as u64 * FRAME_MS >= 300)
            .count()
            >= 2
    }
    /// True change points inside [a, b): starts of a different speaker's turn.
    fn changes(&self, a_ms: u64, b_ms: u64) -> Vec<u64> {
        let inside: Vec<&(u64, u64, usize)> = self
            .turns
            .iter()
            .filter(|t| t.1.min(b_ms) > t.0.max(a_ms) + 300)
            .collect();
        inside
            .windows(2)
            .filter(|w| w[0].2 != w[1].2)
            .map(|w| {
                if w[1].0 >= w[0].1 {
                    (w[0].1 + w[1].0) / 2
                } else {
                    w[1].0
                }
            })
            .filter(|&c| c > a_ms && c < b_ms)
            .collect()
    }
}

#[derive(Default, Clone, Debug)]
struct RealAgg {
    speech: f64,
    correct: f64,
    unlabeled: f64,
    mixed: f64,
    pure: f64,
    labels: f64,
    true_spk: f64,
    n: f64,
    me_tp: f64,
    me_fp: f64,
    me_fn: f64,
}

impl RealAgg {
    fn acc(&self) -> f64 {
        self.correct / self.speech.max(1.0)
    }
    fn merge(&mut self, o: &RealAgg) {
        self.speech += o.speech;
        self.correct += o.correct;
        self.unlabeled += o.unlabeled;
        self.mixed += o.mixed;
        self.pure += o.pure;
        self.labels += o.labels;
        self.true_spk += o.true_spk;
        self.n += o.n;
        self.me_tp += o.me_tp;
        self.me_fp += o.me_fp;
        self.me_fn += o.me_fn;
    }
}

/// Frame-level scoring of labeled segments against the truth, with the optimal one-to-one
/// mapping of labels to speakers. Only frames inside segments and with speech count.
fn score_real(t: &Truth, segs: &[(u64, u64)], labels: &[Option<String>]) -> RealAgg {
    let mut ix: HashMap<&str, usize> = HashMap::new();
    for l in labels.iter().flatten() {
        let k = ix.len();
        ix.entry(l.as_str()).or_insert(k);
    }
    let ns = t.speakers.len();
    let mut conf = vec![vec![0.0f64; ns]; ix.len()];
    let mut a = RealAgg {
        n: 1.0,
        true_spk: ns as f64,
        labels: ix.len() as f64,
        ..Default::default()
    };
    for ((s, e), l) in segs.iter().zip(labels) {
        let c = t.counts(*s, *e);
        let speech = t.span(*s, *e).iter().filter(|&&f| f != 0).count() as f64;
        a.speech += speech;
        a.pure += *c.iter().max().unwrap_or(&0) as f64;
        if t.is_mixed(*s, *e) {
            a.mixed += speech;
        }
        match l {
            Some(l) => {
                for k in 0..ns {
                    conf[ix[l.as_str()]][k] += c[k] as f64;
                }
            }
            None => a.unlabeled += speech,
        }
    }
    let map = metrics::max_weight_matching(&conf);
    for ((s, e), l) in segs.iter().zip(labels) {
        let Some(l) = l else { continue };
        let Some(k) = map[ix[l.as_str()]] else {
            continue;
        };
        a.correct += t.span(*s, *e).iter().filter(|&&f| f >> k & 1 == 1).count() as f64;
    }
    if let Some(me) = t.me {
        for ((s, e), l) in segs.iter().zip(labels) {
            let is_me = l.as_deref() == Some(ME);
            for &f in t.span(*s, *e) {
                if f == 0 {
                    continue;
                }
                match (f >> me & 1 == 1, is_me) {
                    (true, true) => a.me_tp += 1.0,
                    (false, true) => a.me_fp += 1.0,
                    (true, false) => a.me_fn += 1.0,
                    _ => {}
                }
            }
        }
    }
    a
}

struct RealSet {
    /// [meeting][cond] → segments (ms) and embeddings
    segs: Vec<Vec<Vec<(u64, u64)>>>,
    embs: Vec<Vec<Vec<Emb>>>,
    /// room enrollment embedding per meeting
    enroll: Vec<Emb>,
    truths: Vec<Truth>,
}

fn load_real(model: &str, variant: &str, meetings: &[Meeting]) -> Result<RealSet> {
    let mut segs = Vec::new();
    let mut embs = Vec::new();
    let mut enroll = Vec::new();
    for m in meetings {
        let mut ss = Vec::new();
        let mut ee = Vec::new();
        for c in REAL_CONDITIONS {
            let sg: Vec<RealSeg> = if variant == "raw" {
                load_realsegs(&m.id, c)?
            } else {
                serde_json::from_str(&std::fs::read_to_string(pieces_path(
                    model, variant, &m.id, c,
                ))?)?
            };
            ss.push(sg.iter().map(|s| (s.start_ms, s.end_ms)).collect());
            ee.push(read_cache(&real_emb_path(model, variant, &m.id, c))?);
        }
        segs.push(ss);
        embs.push(ee);
        enroll.push(
            read_cache(&cache_path(model, &m.id, "enroll_room"))?
                .pop()
                .flatten(),
        );
    }
    Ok(RealSet {
        segs,
        embs,
        enroll,
        truths: meetings.iter().map(truth).collect(),
    })
}

#[derive(Default, Clone, Debug)]
struct RealRun {
    online: RealAgg,
    recl: RealAgg,
}

fn run_real(
    set: &RealSet,
    mi: usize,
    ci: usize,
    vp: Option<&[f32]>,
    cfg: &ClusterConfig,
) -> RealRun {
    let prefix = if REAL_CONDITIONS[ci] == "call" {
        "sys"
    } else {
        "mic"
    };
    let segs = &set.segs[mi][ci];
    let embs = &set.embs[mi][ci];
    let mut order: Vec<usize> = (0..segs.len()).collect();
    order.sort_by_key(|&i| (segs[i].1, i));
    let mut c = OnlineClusterer::new(prefix, cfg.clone());
    if let Some(vp) = vp {
        c.set_voiceprint(vp.to_vec());
    }
    let mut online = vec![None; segs.len()];
    let use_emb = |i: usize| {
        embs[i]
            .clone()
            .filter(|_| segs[i].1 - segs[i].0 >= cfg.min_embed_ms)
    };
    for &i in &order {
        online[i] = c.assign(use_emb(i).as_deref(), segs[i].0, segs[i].1);
    }
    let items: Vec<ClusterItem> = order
        .iter()
        .map(|&i| ClusterItem {
            segment_id: i.to_string(),
            prefix: prefix.to_string(),
            embedding: use_emb(i).unwrap_or_default(),
            duration_ms: segs[i].1 - segs[i].0,
            online_label: online[i].clone(),
        })
        .collect();
    let mut recl = vec![None; segs.len()];
    for (id, l) in recluster(&items, vp, cfg) {
        recl[id.parse::<usize>().unwrap()] = l;
    }
    RealRun {
        online: score_real(&set.truths[mi], segs, &online),
        recl: score_real(&set.truths[mi], segs, &recl),
    }
}

fn eval_real(
    meetings: &[Meeting],
    set: &RealSet,
    split: &str,
    ci: usize,
    with_vp: bool,
    cfg: &ClusterConfig,
) -> RealRun {
    let mut r = RealRun::default();
    for (mi, m) in meetings.iter().enumerate() {
        if split != "all" && m.split != split {
            continue;
        }
        let vp = if with_vp {
            set.enroll[mi].as_deref()
        } else {
            None
        };
        let x = run_real(set, mi, ci, vp, cfg);
        r.online.merge(&x.online);
        r.recl.merge(&x.recl);
    }
    r
}

/// Mean error (1 - accuracy) over call and room.
fn real_objective(r: &[RealRun], recl: bool) -> f64 {
    r.iter()
        .map(|x| 1.0 - if recl { x.recl.acc() } else { x.online.acc() })
        .sum::<f64>()
        / r.len() as f64
}

// ------------------------------------------------------------------ realistic: change detection

fn cmd_change_eval(model: &str, hop_ms: u64, base: ClusterConfig) -> Result<()> {
    let meetings = load_meetings()?;
    let mut data = Vec::new(); // (split, truth idx, seg, windows, audio-less)
    let truths: Vec<Truth> = meetings.iter().map(truth).collect();
    for (mi, m) in meetings.iter().enumerate() {
        for c in REAL_CONDITIONS {
            let segs = load_realsegs(&m.id, c)?;
            let wins = read_windows(&win_path(model, &m.id, c))?;
            for (sg, w) in segs.into_iter().zip(wins) {
                if sg.end_ms - sg.start_ms >= SPLIT_MIN_MS {
                    data.push((m.split.clone(), mi, c, sg, subsample(&w, hop_ms)));
                }
            }
        }
    }
    println!(
        "### change detection, {} (hop {hop_ms} ms, min piece {} ms)\n",
        model_stem(model),
        base.min_piece_ms
    );
    println!("| split | change_threshold | finals ≥2.5 s | single-speaker finals split | changes found (±0.5 s) | cuts that are real (±0.5 s) | mixed speech before → after |");
    println!("|---|---|---|---|---|---|---|");
    for split in ["tune", "test"] {
        for thr in range(0.0, 0.6, 0.05) {
            let cfg = ClusterConfig {
                change_threshold: thr,
                ..base.clone()
            };
            let (
                mut n,
                mut single,
                mut single_split,
                mut truth_n,
                mut found,
                mut cuts_n,
                mut cuts_ok,
            ) = (0, 0, 0, 0, 0, 0, 0);
            let (mut mixed_before, mut mixed_after, mut speech) = (0.0f64, 0.0f64, 0.0f64);
            for (sp, mi, _c, sg, w) in &data {
                if sp != split {
                    continue;
                }
                let t = &truths[*mi];
                n += 1;
                let truth_c = t.changes(sg.start_ms, sg.end_ms);
                let cuts: Vec<u64> = kenes_speakers::change::pick_changes(w, hop_ms, &cfg)
                    .into_iter()
                    .map(|c| sg.start_ms + c as u64 / 16)
                    .collect();
                if truth_c.is_empty() {
                    single += 1;
                    if !cuts.is_empty() {
                        single_split += 1;
                    }
                }
                truth_n += truth_c.len();
                found += truth_c
                    .iter()
                    .filter(|&&tc| cuts.iter().any(|&c| c.abs_diff(tc) <= 500))
                    .count();
                cuts_n += cuts.len();
                cuts_ok += cuts
                    .iter()
                    .filter(|&&c| truth_c.iter().any(|&tc| c.abs_diff(tc) <= 500))
                    .count();
                let sp_frames = t
                    .span(sg.start_ms, sg.end_ms)
                    .iter()
                    .filter(|&&f| f != 0)
                    .count() as f64;
                speech += sp_frames;
                if t.is_mixed(sg.start_ms, sg.end_ms) {
                    mixed_before += sp_frames;
                }
                let mut bounds = vec![sg.start_ms];
                bounds.extend(&cuts);
                bounds.push(sg.end_ms);
                for p in bounds.windows(2) {
                    if t.is_mixed(p[0], p[1]) {
                        mixed_after +=
                            t.span(p[0], p[1]).iter().filter(|&&f| f != 0).count() as f64;
                    }
                }
            }
            println!(
                "| {split} | {thr:.2} | {n} | {single_split}/{single} ({:.1}%) | {found}/{truth_n} ({:.0}%) | {cuts_ok}/{cuts_n} ({:.0}%) | {:.0}% → {:.0}% |",
                single_split as f64 / single.max(1) as f64 * 100.0,
                found as f64 / truth_n.max(1) as f64 * 100.0,
                cuts_ok as f64 / cuts_n.max(1) as f64 * 100.0,
                mixed_before / speech.max(1.0) * 100.0,
                mixed_after / speech.max(1.0) * 100.0
            );
        }
    }
    Ok(())
}

/// Split the finals at change points (from cached windows) and embed the pieces.
fn cmd_split(model: &str, hop_ms: u64, cfg: &ClusterConfig, jobs: usize) -> Result<String> {
    let tag = format!(
        "split_t{:.2}_p{}_h{hop_ms}",
        cfg.change_threshold, cfg.min_piece_ms
    );
    let meetings = load_meetings()?;
    let mut work: Vec<(usize, &str)> = Vec::new();
    for (mi, m) in meetings.iter().enumerate() {
        for c in REAL_CONDITIONS {
            if !real_emb_path(model, &tag, &m.id, c).exists() {
                work.push((mi, c));
            }
        }
    }
    let work = std::sync::Mutex::new(work);
    std::thread::scope(|s| {
        for _ in 0..jobs.max(1) {
            s.spawn(|| -> Result<()> {
                let mut emb = Embedder::new(&model_path(model), 1)?;
                loop {
                    let Some((mi, cond)) = work.lock().unwrap().pop() else {
                        return Ok(());
                    };
                    let m = &meetings[mi];
                    let audio = read_wav(
                        &root()
                            .join("meetings")
                            .join(&m.id)
                            .join(format!("{cond}.wav")),
                    )?;
                    let segs = load_realsegs(&m.id, cond)?;
                    let wins = read_windows(&win_path(model, &m.id, cond))?;
                    let mut pieces = Vec::new();
                    let mut embs = Vec::new();
                    for (sg, w) in segs.iter().zip(&wins) {
                        let a = (sg.start_ms * 16) as usize;
                        let b = ((sg.end_ms * 16) as usize).min(audio.len());
                        let x = &audio[a.min(b)..b];
                        let mut cuts = if sg.end_ms - sg.start_ms >= SPLIT_MIN_MS {
                            kenes_speakers::change::pick_changes(&subsample(w, hop_ms), hop_ms, cfg)
                        } else {
                            Vec::new()
                        };
                        kenes_speakers::change::snap_to_quiet(
                            x,
                            &mut cuts,
                            kenes_speakers::change::SNAP_MS,
                            cfg.min_piece_ms as usize * 16,
                        );
                        let mut bounds = vec![0usize];
                        bounds.extend(&cuts);
                        bounds.push(x.len());
                        for p in bounds.windows(2) {
                            let (ps, pe) = (
                                sg.start_ms + p[0] as u64 / 16,
                                sg.start_ms + p[1] as u64 / 16,
                            );
                            pieces.push(RealSeg {
                                start_ms: ps,
                                end_ms: pe,
                                text: String::new(),
                            });
                            embs.push(if pe - ps >= MIN_CACHE_MS {
                                emb.embed(&x[p[0]..p[1]]).ok()
                            } else {
                                None
                            });
                        }
                    }
                    write_cache(&real_emb_path(model, &tag, &m.id, cond), &embs, emb.dim())?;
                    std::fs::write(
                        pieces_path(model, &tag, &m.id, cond),
                        serde_json::to_string(&pieces)?,
                    )?;
                }
            });
        }
    });
    println!("{tag}");
    Ok(tag)
}

/// Embed another model's pieces (same segmentation) with `model`, for a fair model comparison.
fn cmd_embed_pieces(model: &str, from_model: &str, variant: &str, jobs: usize) -> Result<()> {
    let meetings = load_meetings()?;
    let mut work: Vec<(usize, &str)> = Vec::new();
    for (mi, m) in meetings.iter().enumerate() {
        for c in REAL_CONDITIONS {
            if !real_emb_path(model, variant, &m.id, c).exists() {
                work.push((mi, c));
            }
        }
    }
    let work = std::sync::Mutex::new(work);
    std::thread::scope(|s| {
        for _ in 0..jobs.max(1) {
            s.spawn(|| -> Result<()> {
                let mut emb = Embedder::new(&model_path(model), 1)?;
                loop {
                    let Some((mi, cond)) = work.lock().unwrap().pop() else {
                        return Ok(());
                    };
                    let m = &meetings[mi];
                    let audio = read_wav(
                        &root()
                            .join("meetings")
                            .join(&m.id)
                            .join(format!("{cond}.wav")),
                    )?;
                    let pieces: Vec<RealSeg> = serde_json::from_str(&std::fs::read_to_string(
                        pieces_path(from_model, variant, &m.id, cond),
                    )?)?;
                    let embs: Vec<Emb> = pieces
                        .iter()
                        .map(|p| {
                            let a = (p.start_ms * 16) as usize;
                            let b = ((p.end_ms * 16) as usize).min(audio.len());
                            if p.end_ms - p.start_ms >= MIN_CACHE_MS {
                                emb.embed(&audio[a.min(b)..b]).ok()
                            } else {
                                None
                            }
                        })
                        .collect();
                    write_cache(
                        &real_emb_path(model, variant, &m.id, cond),
                        &embs,
                        emb.dim(),
                    )?;
                    std::fs::write(
                        pieces_path(model, variant, &m.id, cond),
                        serde_json::to_string(&pieces)?,
                    )?;
                }
            });
        }
    });
    Ok(())
}

fn cmd_report_real(model: &str, variant: &str, cfg: &ClusterConfig) -> Result<()> {
    let meetings = load_meetings()?;
    let set = load_real(model, variant, &meetings)?;
    println!(
        "### {} on pipeline segments ({variant})\n",
        model_stem(model)
    );
    println!("| split | cond | segments | mixed speech | purity bound | online acc | online labels | recluster acc | recluster labels | true speakers |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for split in ["tune", "test"] {
        for (ci, cond) in REAL_CONDITIONS.iter().enumerate() {
            let r = eval_real(&meetings, &set, split, ci, false, cfg);
            let nseg: usize = meetings
                .iter()
                .enumerate()
                .filter(|(_, m)| m.split == split)
                .map(|(mi, _)| set.segs[mi][ci].len())
                .sum();
            let n = r.online.n.max(1.0);
            println!(
                "| {split} | {cond} | {nseg} | {:.1}% | {:.1}% | {:.1}% | {:.1} | {:.1}% | {:.1} | {:.1} |",
                r.online.mixed / r.online.speech * 100.0,
                r.online.pure / r.online.speech * 100.0,
                r.online.acc() * 100.0,
                r.online.labels / n,
                r.recl.acc() * 100.0,
                r.recl.labels / n,
                r.online.true_spk / n
            );
        }
    }
    println!("\nPer meeting:\n");
    println!("| meeting | split | cond | segments | mixed speech | online acc | labels | recluster acc | labels | true |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for (mi, m) in meetings.iter().enumerate() {
        for (ci, cond) in REAL_CONDITIONS.iter().enumerate() {
            let r = run_real(&set, mi, ci, None, cfg);
            println!(
                "| {} | {} | {cond} | {} | {:.0}% | {:.1}% | {} | {:.1}% | {} | {} |",
                m.id,
                m.split,
                set.segs[mi][ci].len(),
                r.online.mixed / r.online.speech.max(1.0) * 100.0,
                r.online.acc() * 100.0,
                r.online.labels,
                r.recl.acc() * 100.0,
                r.recl.labels,
                r.online.true_spk
            );
        }
    }
    println!("\nRoom with voiceprint:\n");
    println!("| split | online acc | online me P / R | recluster acc | recluster me P / R |");
    println!("|---|---|---|---|---|");
    for split in ["tune", "test"] {
        let r = eval_real(&meetings, &set, split, 1, true, cfg);
        let pr = |a: &RealAgg| {
            (
                a.me_tp / (a.me_tp + a.me_fp).max(1.0) * 100.0,
                a.me_tp / (a.me_tp + a.me_fn).max(1.0) * 100.0,
            )
        };
        let (p1, r1) = pr(&r.online);
        let (p2, r2) = pr(&r.recl);
        println!(
            "| {split} | {:.1}% | {p1:.0}% / {r1:.0}% | {:.1}% | {p2:.0}% / {r2:.0}% |",
            r.online.acc() * 100.0,
            r.recl.acc() * 100.0
        );
    }
    Ok(())
}

fn cmd_tune_real(model: &str, variant: &str, base: ClusterConfig) -> Result<()> {
    let meetings = load_meetings()?;
    let set = load_real(model, variant, &meetings)?;
    let ev = |cfg: &ClusterConfig, split: &str| -> Vec<RealRun> {
        (0..REAL_CONDITIONS.len())
            .map(|c| eval_real(&meetings, &set, split, c, false, cfg))
            .collect()
    };
    println!(
        "### tuning on pipeline segments ({variant}), {}\n",
        model_stem(model)
    );
    // Stage 1: online threshold × merge margin.
    let thrs = range(0.5, 0.8, 0.025);
    let margins = [0.05f32, 0.1, 0.15, 9.0];
    let grid: Vec<(f32, f32)> = thrs
        .iter()
        .flat_map(|&t| margins.iter().map(move |&m| (t, m)))
        .collect();
    let res = par_map(&grid, |&(t, mg)| {
        let c = ClusterConfig {
            threshold: t,
            merge_threshold: t + mg,
            ..base.clone()
        };
        let (a, b) = (ev(&c, "tune"), ev(&c, "test"));
        (t, mg, real_objective(&a, false), real_objective(&b, false))
    });
    println!(
        "| threshold | {} |",
        margins
            .iter()
            .map(|m| if *m > 1.0 {
                "no merge".into()
            } else {
                format!("merge +{m}")
            })
            .collect::<Vec<_>>()
            .join(" | ")
    );
    println!("|---|{}", "---|".repeat(margins.len()));
    for &t in &thrs {
        let row: Vec<String> = margins
            .iter()
            .map(|&mg| {
                let r = res.iter().find(|r| r.0 == t && r.1 == mg).unwrap();
                format!("{:.1}% ({:.1}%)", r.2 * 100.0, r.3 * 100.0)
            })
            .collect();
        println!("| {t:.3} | {} |", row.join(" | "));
    }
    println!("\n(online error, tune split; test split in parentheses)\n");
    let b1 = res
        .iter()
        .cloned()
        .min_by(|a, b| a.2.total_cmp(&b.2))
        .unwrap();
    let mut cfg = ClusterConfig {
        threshold: b1.0,
        merge_threshold: b1.0 + b1.1,
        ..base.clone()
    };
    println!("best online: threshold {} merge +{}\n", b1.0, b1.1);
    // Stage 2: short-segment knobs.
    let mut s2 = Vec::new();
    for me in [500u64, 1000] {
        for mn in [500u64, 1000, 1500] {
            for sm in [0u64, 2000, 3000] {
                for rl in [0.1f32, 0.2] {
                    if mn >= me && !(sm == 0 && rl != 0.1) {
                        s2.push((me, mn, sm, rl));
                    }
                }
            }
        }
    }
    let res2 = par_map(&s2, |&(me, mn, sm, rl)| {
        let c = ClusterConfig {
            min_embed_ms: me,
            min_new_speaker_ms: mn,
            short_segment_ms: sm,
            short_segment_relax: if sm == 0 { 0.0 } else { rl },
            ..cfg.clone()
        };
        (
            me,
            mn,
            sm,
            rl,
            real_objective(&ev(&c, "tune"), false),
            real_objective(&ev(&c, "test"), false),
        )
    });
    let mut s2s = res2.clone();
    s2s.sort_by(|a, b| a.4.total_cmp(&b.4));
    println!(
        "| min_embed_ms | min_new_speaker_ms | short_segment_ms | relax | online err tune | test |"
    );
    println!("|---|---|---|---|---|---|");
    for r in s2s.iter().take(12) {
        println!(
            "| {} | {} | {} | {} | {:.2}% | {:.2}% |",
            r.0,
            r.1,
            r.2,
            if r.2 == 0 { 0.0 } else { r.3 },
            r.4 * 100.0,
            r.5 * 100.0
        );
    }
    let b2 = s2s[0];
    cfg.min_embed_ms = b2.0;
    cfg.min_new_speaker_ms = b2.1;
    cfg.short_segment_ms = b2.2;
    cfg.short_segment_relax = if b2.2 == 0 { 0.0 } else { b2.3 };
    // Stage 3: recluster.
    let rthrs = range(0.5, 0.85, 0.025);
    let mins = [0u64, 3000, 5000, 8000];
    let g3: Vec<(f32, u64)> = rthrs
        .iter()
        .flat_map(|&t| mins.iter().map(move |&m| (t, m)))
        .collect();
    // Worst per-meeting change (recluster − online accuracy) on the tune split.
    let worst_tune = |c: &ClusterConfig| -> f64 {
        let mut w = f64::MAX;
        for (mi, m) in meetings.iter().enumerate() {
            if m.split == "tune" {
                for ci in 0..REAL_CONDITIONS.len() {
                    let r = run_real(&set, mi, ci, None, c);
                    w = w.min(r.recl.acc() - r.online.acc());
                }
            }
        }
        w
    };
    let res3 = par_map(&g3, |&(t, mc)| {
        let c = ClusterConfig {
            recluster_threshold: t,
            min_cluster_ms: mc,
            ..cfg.clone()
        };
        let (a, b) = (ev(&c, "tune"), ev(&c, "test"));
        (
            t,
            mc,
            real_objective(&a, true),
            real_objective(&b, true),
            real_objective(&a, false),
            worst_tune(&c),
        )
    });
    println!(
        "\n| recluster_threshold | {} |",
        mins.map(|m| format!("min_cluster {m}")).join(" | ")
    );
    println!("|---|{}", "---|".repeat(mins.len()));
    for &t in &rthrs {
        let row: Vec<String> = mins
            .iter()
            .map(|&m| {
                let r = res3.iter().find(|r| r.0 == t && r.1 == m).unwrap();
                format!("{:.1}% ({:.1}%)", r.2 * 100.0, r.3 * 100.0)
            })
            .collect();
        println!("| {t:.3} | {} |", row.join(" | "));
    }
    // Best tune error among settings where no tune meeting gets more than 1 point worse
    // than online (recluster must be safe, not just good on average).
    let b3 = res3
        .iter()
        .filter(|r| r.5 >= -0.01)
        .cloned()
        .min_by(|a, b| a.2.total_cmp(&b.2).then(a.0.total_cmp(&b.0)))
        .unwrap_or_else(|| {
            res3.iter()
                .cloned()
                .min_by(|a, b| a.2.total_cmp(&b.2))
                .unwrap()
        });
    cfg.recluster_threshold = b3.0;
    cfg.min_cluster_ms = b3.1;
    println!(
        "\n(recluster error, tune split; test in parentheses; online error at these settings: {:.1}%)\n\nchosen: recluster_threshold {} min_cluster_ms {} (best tune error with no tune meeting > 1 point worse than online; worst {:+.1})\n",
        b3.4 * 100.0,
        b3.0,
        b3.1,
        b3.5 * 100.0
    );
    println!("tuned: {cfg:?}\n");
    let (a, b) = (ev(&cfg, "tune"), ev(&cfg, "test"));
    let acc = |r: &[RealRun], recl: bool| (1.0 - real_objective(r, recl)) * 100.0;
    println!(
        "SUMMARY-REAL {} accuracy online tune {:.1}% test {:.1}% | recluster tune {:.1}% test {:.1}% | call/room test recluster {:.1}% / {:.1}%",
        model_stem(model),
        acc(&a, false),
        acc(&b, false),
        acc(&a, true),
        acc(&b, true),
        b[0].recl.acc() * 100.0,
        b[1].recl.acc() * 100.0
    );
    Ok(())
}

// ------------------------------------------------------------------ pair EER

fn eer(same: &mut [f32], diff: &mut [f32]) -> (f64, f32) {
    same.sort_by(f32::total_cmp);
    diff.sort_by(f32::total_cmp);
    if same.is_empty() || diff.is_empty() {
        return (f64::NAN, f32::NAN);
    }
    // threshold t: FRR = P(same < t), FAR = P(diff >= t)
    let mut best = (f64::MAX, 0.0f32, 0.0f64);
    for i in 0..=200 {
        let t = -0.2 + i as f32 * 0.006;
        let frr = same.partition_point(|&x| x < t) as f64 / same.len() as f64;
        let far = 1.0 - diff.partition_point(|&x| x < t) as f64 / diff.len() as f64;
        let gap = (frr - far).abs();
        if gap < best.0 {
            best = (gap, t, (frr + far) / 2.0);
        }
    }
    (best.2, best.1)
}

fn cmd_pairs(models: Vec<String>) -> Result<()> {
    let meetings = load_meetings()?;
    println!("| model | cond | EER ≥1 s | thr | EER ≥3 s | thr | EER 1–2 s vs ≥3 s |");
    println!("|---|---|---|---|---|---|---|");
    for model in &models {
        let cache = load_cache(model, &meetings)?;
        for (ci, c) in CONDITIONS.iter().enumerate() {
            let mut buckets: [(Vec<f32>, Vec<f32>); 3] = Default::default();
            for (mi, m) in meetings.iter().enumerate() {
                let e = &cache.segs[mi][ci];
                let n = m.segments.len();
                for i in 0..n {
                    for j in i + 1..n {
                        let (Some(a), Some(b)) = (&e[i], &e[j]) else {
                            continue;
                        };
                        let da = m.segments[i].end_ms - m.segments[i].start_ms;
                        let db = m.segments[j].end_ms - m.segments[j].start_ms;
                        let s = kenes_speakers::cosine(a, b);
                        let same = m.segments[i].speaker == m.segments[j].speaker;
                        let mut push = |k: usize| {
                            if same {
                                buckets[k].0.push(s)
                            } else {
                                buckets[k].1.push(s)
                            }
                        };
                        if da >= 1000 && db >= 1000 {
                            push(0);
                        }
                        if da >= 3000 && db >= 3000 {
                            push(1);
                        }
                        let short = |d: u64| (1000..2000).contains(&d);
                        if (short(da) && db >= 3000) || (short(db) && da >= 3000) {
                            push(2);
                        }
                    }
                }
            }
            let r: Vec<(f64, f32)> = buckets.iter_mut().map(|(s, d)| eer(s, d)).collect();
            println!(
                "| {} | {c} | {:.1}% | {:.2} | {:.1}% | {:.2} | {:.1}% |",
                model_stem(model),
                r[0].0 * 100.0,
                r[0].1,
                r[1].0 * 100.0,
                r[1].1,
                r[2].0 * 100.0
            );
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ diarization runs

#[derive(Default, Clone, Debug)]
struct Agg {
    total_ms: f64,
    err_ms: f64,
    unlabeled_ms: f64,
    count_abs_err: f64,
    hyp_count: f64,
    ref_count: f64,
    n: f64,
    me_tp: f64,
    me_fp: f64,
    me_fn: f64,
}

impl Agg {
    fn add(&mut self, s: &metrics::Score, total_ms: f64) {
        self.total_ms += total_ms;
        self.err_ms += s.error_rate * total_ms;
        self.unlabeled_ms += s.unlabeled_rate * total_ms;
        self.count_abs_err += (s.hyp_speakers as f64 - s.ref_speakers as f64).abs();
        self.hyp_count += s.hyp_speakers as f64;
        self.ref_count += s.ref_speakers as f64;
        self.n += 1.0;
    }
    fn merge(&mut self, o: &Agg) {
        self.total_ms += o.total_ms;
        self.err_ms += o.err_ms;
        self.unlabeled_ms += o.unlabeled_ms;
        self.count_abs_err += o.count_abs_err;
        self.hyp_count += o.hyp_count;
        self.ref_count += o.ref_count;
        self.n += o.n;
        self.me_tp += o.me_tp;
        self.me_fp += o.me_fp;
        self.me_fn += o.me_fn;
    }
    fn ser(&self) -> f64 {
        self.err_ms / self.total_ms.max(1.0)
    }
    fn me_p(&self) -> f64 {
        self.me_tp / (self.me_tp + self.me_fp).max(1.0)
    }
    fn me_r(&self) -> f64 {
        self.me_tp / (self.me_tp + self.me_fn).max(1.0)
    }
    fn fmt(&self) -> String {
        format!(
            "{:.1}% | {:.1}% | {:.1} / {:.1} | {:.1}",
            self.ser() * 100.0,
            self.unlabeled_ms / self.total_ms.max(1.0) * 100.0,
            self.hyp_count / self.n.max(1.0),
            self.ref_count / self.n.max(1.0),
            self.count_abs_err / self.n.max(1.0)
        )
    }
}

#[derive(Default, Clone, Debug)]
struct RunResult {
    online: Agg,
    /// distinct labels emitted online (incl. ones merged away later)
    online_emitted: f64,
    recl: Agg,
}

/// Online + recluster on one meeting/condition.
fn run_one(
    m: &Meeting,
    embs: &[Emb],
    vp: Option<&[f32]>,
    prefix: &str,
    cfg: &ClusterConfig,
) -> (
    metrics::Score,
    usize,
    metrics::Score,
    Vec<Option<String>>,
    Vec<Option<String>>,
) {
    let mut order: Vec<usize> = (0..m.segments.len()).collect();
    order.sort_by_key(|&i| (m.segments[i].end_ms, i)); // finalized in order of end time
    let mut c = OnlineClusterer::new(prefix, cfg.clone());
    if let Some(vp) = vp {
        c.set_voiceprint(vp.to_vec());
    }
    let mut online = vec![None; m.segments.len()];
    for &i in &order {
        let s = &m.segments[i];
        let d = s.end_ms - s.start_ms;
        let e = embs[i].as_deref().filter(|_| d >= cfg.min_embed_ms);
        online[i] = c.assign(e, s.start_ms, s.end_ms);
    }
    let emitted: std::collections::HashSet<&String> = online.iter().flatten().collect();
    let items: Vec<ClusterItem> = order
        .iter()
        .map(|&i| {
            let s = &m.segments[i];
            let d = s.end_ms - s.start_ms;
            ClusterItem {
                segment_id: i.to_string(),
                prefix: prefix.to_string(),
                embedding: embs[i]
                    .clone()
                    .filter(|_| d >= cfg.min_embed_ms)
                    .unwrap_or_default(),
                duration_ms: d,
                online_label: online[i].clone(),
            }
        })
        .collect();
    let mut recl = vec![None; m.segments.len()];
    for (id, l) in recluster(&items, vp, cfg) {
        recl[id.parse::<usize>().unwrap()] = l;
    }
    let refs: Vec<&str> = m.segments.iter().map(|s| s.speaker.as_str()).collect();
    let durs: Vec<u64> = m.segments.iter().map(|s| s.end_ms - s.start_ms).collect();
    let so = metrics::score(&refs, &online, &durs);
    let sr = metrics::score(&refs, &recl, &durs);
    (so, emitted.len(), sr, online, recl)
}

fn me_counts(m: &Meeting, hyp: &[Option<String>], agg: &mut Agg) {
    for (s, h) in m.segments.iter().zip(hyp) {
        let d = (s.end_ms - s.start_ms) as f64;
        let is_me = h.as_deref() == Some(ME);
        match (s.speaker == m.me, is_me) {
            (true, true) => agg.me_tp += d,
            (false, true) => agg.me_fp += d,
            (true, false) => agg.me_fn += d,
            _ => {}
        }
    }
}

/// Evaluate `cfg` on meetings of `split` ("tune", "test" or "all") for one condition.
fn evaluate(
    meetings: &[Meeting],
    cache: &Cache,
    split: &str,
    cond: usize,
    with_vp: bool,
    cfg: &ClusterConfig,
) -> RunResult {
    let prefix = if CONDITIONS[cond] == "call" {
        "sys"
    } else {
        "mic"
    };
    let mut r = RunResult::default();
    for (mi, m) in meetings.iter().enumerate() {
        if split != "all" && m.split != split {
            continue;
        }
        let vp = if with_vp {
            cache.enroll[mi][cond].as_deref()
        } else {
            None
        };
        let total: f64 = m
            .segments
            .iter()
            .map(|s| (s.end_ms - s.start_ms) as f64)
            .sum();
        let (so, emitted, sr, online, recl) = run_one(m, &cache.segs[mi][cond], vp, prefix, cfg);
        r.online.add(&so, total);
        r.recl.add(&sr, total);
        r.online_emitted += emitted as f64;
        if with_vp {
            me_counts(m, &online, &mut r.online);
            me_counts(m, &recl, &mut r.recl);
        }
    }
    r
}

fn parse_set(cfg: &mut ClusterConfig, s: &str) -> Result<()> {
    for kv in s.split(',').filter(|x| !x.is_empty()) {
        let (k, v) = kv.split_once('=').context("expected k=v")?;
        match k {
            "threshold" => cfg.threshold = v.parse()?,
            "merge_threshold" => cfg.merge_threshold = v.parse()?,
            "voiceprint_threshold" => cfg.voiceprint_threshold = v.parse()?,
            "recluster_threshold" => cfg.recluster_threshold = v.parse()?,
            "min_embed_ms" => cfg.min_embed_ms = v.parse()?,
            "min_new_speaker_ms" => cfg.min_new_speaker_ms = v.parse()?,
            "context_ms" => cfg.context_ms = v.parse()?,
            "max_speakers" => cfg.max_speakers = v.parse()?,
            "min_cluster_ms" => cfg.min_cluster_ms = v.parse()?,
            "short_segment_ms" => cfg.short_segment_ms = v.parse()?,
            "short_segment_relax" => cfg.short_segment_relax = v.parse()?,
            "change_threshold" => cfg.change_threshold = v.parse()?,
            "min_piece_ms" => cfg.min_piece_ms = v.parse()?,
            _ => bail!("unknown key {k}"),
        }
    }
    Ok(())
}

fn range(a: f32, b: f32, step: f32) -> Vec<f32> {
    let n = ((b - a) / step).round() as usize;
    (0..=n)
        .map(|i| ((a + i as f32 * step) * 1000.0).round() / 1000.0)
        .collect()
}

/// Mean SER over the conditions that matter (call and room), plus clean at half weight.
fn objective(r: &[RunResult], recl: bool) -> f64 {
    let s = |k: usize| {
        if recl {
            r[k].recl.ser()
        } else {
            r[k].online.ser()
        }
    };
    (0.5 * s(0) + s(1) + s(2)) / 2.5
}

fn par_map<T: Sync, R: Send>(xs: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let jobs = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(12);
    let chunk = xs.len().div_ceil(jobs).max(1);
    std::thread::scope(|s| {
        let hs: Vec<_> = xs
            .chunks(chunk)
            .map(|c| s.spawn(|| c.iter().map(&f).collect::<Vec<_>>()))
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

fn cmd_tune(model: &str, quick: bool, base: ClusterConfig) -> Result<()> {
    let meetings = load_meetings()?;
    let cache = load_cache(model, &meetings)?;
    let eval3 = |cfg: &ClusterConfig, split: &str| -> Vec<RunResult> {
        (0..3)
            .map(|c| evaluate(&meetings, &cache, split, c, false, cfg))
            .collect()
    };

    // Stage 1: online threshold × merge margin (other knobs at base).
    let thrs = if quick {
        range(0.2, 0.8, 0.05)
    } else {
        range(0.15, 0.85, 0.025)
    };
    let margins = [0.05f32, 0.1, 0.15, 0.2, 0.3, 9.0];
    let grid: Vec<(f32, f32)> = thrs
        .iter()
        .flat_map(|&t| margins.iter().map(move |&m| (t, m)))
        .collect();
    let res = par_map(&grid, |&(t, mg)| {
        let cfg = ClusterConfig {
            threshold: t,
            merge_threshold: t + mg,
            ..base.clone()
        };
        (t, mg, objective(&eval3(&cfg, "tune"), false))
    });
    let best1 = res
        .iter()
        .cloned()
        .min_by(|a, b| a.2.total_cmp(&b.2))
        .unwrap();
    println!(
        "## {}: online, threshold × merge margin (tune split, objective = SER)\n",
        model_stem(model)
    );
    println!(
        "| threshold | {} |",
        margins
            .iter()
            .map(|m| if *m > 1.0 {
                "no merge".to_string()
            } else {
                format!("+{m}")
            })
            .collect::<Vec<_>>()
            .join(" | ")
    );
    println!("|---|{}", "---|".repeat(margins.len()));
    for &t in &thrs {
        let row: Vec<String> = margins
            .iter()
            .map(|&mg| {
                let v = res.iter().find(|r| r.0 == t && r.1 == mg).unwrap().2;
                format!("{:.1}%", v * 100.0)
            })
            .collect();
        println!("| {t:.3} | {} |", row.join(" | "));
    }
    println!(
        "\nbest: threshold {} merge +{} → {:.2}%\n",
        best1.0,
        best1.1,
        best1.2 * 100.0
    );

    // Stage 2: short-segment knobs.
    let mut cfg = ClusterConfig {
        threshold: best1.0,
        merge_threshold: best1.0 + best1.1,
        ..base.clone()
    };
    let mut s2: Vec<(u64, u64, u64, f32)> = Vec::new();
    for me in [500u64, 800, 1000, 1500] {
        for mn in [500u64, 1000, 1500, 2000] {
            for sm in [0u64, 1500, 2000, 3000] {
                for rl in [0.1f32, 0.2, 0.3] {
                    if mn >= me && !(sm == 0 && rl != 0.1) {
                        s2.push((me, mn, sm, rl));
                    }
                }
            }
        }
    }
    let res2 = par_map(&s2, |&(me, mn, sm, rl)| {
        let c = ClusterConfig {
            min_embed_ms: me,
            min_new_speaker_ms: mn,
            short_segment_ms: sm,
            short_segment_relax: if sm == 0 { 0.0 } else { rl },
            ..cfg.clone()
        };
        (me, mn, sm, rl, objective(&eval3(&c, "tune"), false))
    });
    println!("| min_embed_ms | min_new_speaker_ms | short_segment_ms | short_segment_relax | online SER (tune) |");
    println!("|---|---|---|---|---|");
    let mut sorted2 = res2.clone();
    sorted2.sort_by(|a, b| a.4.total_cmp(&b.4));
    for r in sorted2.iter().take(if quick { 10 } else { 25 }) {
        println!(
            "| {} | {} | {} | {} | {:.2}% |",
            r.0,
            r.1,
            r.2,
            if r.2 == 0 { 0.0 } else { r.3 },
            r.4 * 100.0
        );
    }
    let best2 = sorted2[0];
    cfg.min_embed_ms = best2.0;
    cfg.min_new_speaker_ms = best2.1;
    cfg.short_segment_ms = best2.2;
    cfg.short_segment_relax = if best2.2 == 0 { 0.0 } else { best2.3 };
    println!(
        "\nbest: min_embed {} min_new {} short_segment {} relax {}\n",
        cfg.min_embed_ms, cfg.min_new_speaker_ms, cfg.short_segment_ms, cfg.short_segment_relax
    );
    // Thresholds again, now with the short-segment settings.
    let res1b = par_map(&grid, |&(t, mg)| {
        let c = ClusterConfig {
            threshold: t,
            merge_threshold: t + mg,
            ..cfg.clone()
        };
        (t, mg, objective(&eval3(&c, "tune"), false))
    });
    let best1b = res1b
        .iter()
        .cloned()
        .min_by(|a, b| a.2.total_cmp(&b.2))
        .unwrap();
    cfg.threshold = best1b.0;
    cfg.merge_threshold = best1b.0 + best1b.1;
    println!(
        "re-tuned: threshold {} merge +{} → {:.2}%\n",
        best1b.0,
        best1b.1,
        best1b.2 * 100.0
    );

    // Stage 3: recluster threshold × smallest cluster that stands on its own.
    let rthrs = if quick {
        range(0.3, 0.85, 0.05)
    } else {
        range(0.3, 0.85, 0.025)
    };
    let mins = [0u64, 3000, 5000, 8000];
    let grid3: Vec<(f32, u64)> = rthrs
        .iter()
        .flat_map(|&t| mins.iter().map(move |&m| (t, m)))
        .collect();
    let res3 = par_map(&grid3, |&(t, mc)| {
        let c = ClusterConfig {
            recluster_threshold: t,
            min_cluster_ms: mc,
            ..cfg.clone()
        };
        (t, mc, objective(&eval3(&c, "tune"), true))
    });
    println!(
        "| recluster_threshold | {} |",
        mins.map(|m| format!("min_cluster {m} ms")).join(" | ")
    );
    println!("|---|{}", "---|".repeat(mins.len()));
    for &t in &rthrs {
        let row: Vec<String> = mins
            .iter()
            .map(|&m| {
                format!(
                    "{:.2}%",
                    res3.iter().find(|r| r.0 == t && r.1 == m).unwrap().2 * 100.0
                )
            })
            .collect();
        println!("| {t:.3} | {} |", row.join(" | "));
    }
    let best3 = res3
        .iter()
        .cloned()
        .min_by(|a, b| a.2.total_cmp(&b.2))
        .unwrap();
    cfg.recluster_threshold = best3.0;
    cfg.min_cluster_ms = best3.1;
    println!(
        "\nbest recluster_threshold {} min_cluster_ms {}\n",
        best3.0, best3.1
    );

    // Stage 4: voiceprint threshold (room and clean, with enrollment).
    let vthrs = range(0.2, 0.8, 0.025);
    let res4 = par_map(&vthrs, |&t| {
        let c = ClusterConfig {
            voiceprint_threshold: t,
            ..cfg.clone()
        };
        let mut a = Agg::default();
        let mut b = Agg::default();
        for cond in [0usize, 2] {
            let r = evaluate(&meetings, &cache, "tune", cond, true, &c);
            a.merge(&r.online);
            b.merge(&r.recl);
        }
        (t, a, b)
    });
    println!("| voiceprint_threshold | online SER | online me P / R | recluster SER | recluster me P / R |");
    println!("|---|---|---|---|---|");
    for (t, a, b) in &res4 {
        println!(
            "| {t:.3} | {:.1}% | {:.0}% / {:.0}% | {:.1}% | {:.0}% / {:.0}% |",
            a.ser() * 100.0,
            a.me_p() * 100.0,
            a.me_r() * 100.0,
            b.ser() * 100.0,
            b.me_p() * 100.0,
            b.me_r() * 100.0
        );
    }
    // Voiceprint threshold: best online + recluster SER with enrollment (tune split).
    let bestv = res4
        .iter()
        .min_by(|a, b| (a.1.ser() + a.2.ser()).total_cmp(&(b.1.ser() + b.2.ser())))
        .unwrap();
    cfg.voiceprint_threshold = bestv.0;
    println!("\ntuned config: {cfg:?}\n");

    // Held-out numbers for the model comparison.
    println!("| model | cond | online SER tune | online SER test | recluster SER tune | recluster SER test | speakers est/true (test, recluster) |");
    println!("|---|---|---|---|---|---|---|");
    let tune = eval3(&cfg, "tune");
    let test = eval3(&cfg, "test");
    for c in 0..3 {
        println!(
            "| {} | {} | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1} / {:.1} |",
            model_stem(model),
            CONDITIONS[c],
            tune[c].online.ser() * 100.0,
            test[c].online.ser() * 100.0,
            tune[c].recl.ser() * 100.0,
            test[c].recl.ser() * 100.0,
            test[c].recl.hyp_count / test[c].recl.n.max(1.0),
            test[c].recl.ref_count / test[c].recl.n.max(1.0)
        );
    }
    println!(
        "SUMMARY {} objective online tune {:.2}% test {:.2}% | recluster tune {:.2}% test {:.2}%",
        model_stem(model),
        objective(&tune, false) * 100.0,
        objective(&test, false) * 100.0,
        objective(&tune, true) * 100.0,
        objective(&test, true) * 100.0
    );
    Ok(())
}

fn print_report(model: &str, cfg: &ClusterConfig) -> Result<()> {
    let meetings = load_meetings()?;
    let cache = load_cache(model, &meetings)?;
    println!("## {}\n\n`{cfg:?}`\n", model_stem(model));
    println!("| split | cond | stage | SER | unlabeled | speakers est / true | abs count err | labels emitted |");
    println!("|---|---|---|---|---|---|---|---|");
    for split in ["tune", "test"] {
        for (ci, c) in CONDITIONS.iter().enumerate() {
            let r = evaluate(&meetings, &cache, split, ci, false, cfg);
            println!(
                "| {split} | {c} | online | {} | {:.1} |",
                r.online.fmt(),
                r.online_emitted / r.online.n.max(1.0)
            );
            println!("| {split} | {c} | recluster | {} | |", r.recl.fmt());
        }
    }
    println!("\nWith voiceprint (\"me\" = the enrolled speaker):\n");
    println!("| split | cond | stage | SER | me precision | me recall |");
    println!("|---|---|---|---|---|---|");
    for split in ["tune", "test"] {
        for ci in [0usize, 2] {
            let r = evaluate(&meetings, &cache, split, ci, true, cfg);
            for (stage, a) in [("online", &r.online), ("recluster", &r.recl)] {
                println!(
                    "| {split} | {} | {stage} | {:.1}% | {:.1}% | {:.1}% |",
                    CONDITIONS[ci],
                    a.ser() * 100.0,
                    a.me_p() * 100.0,
                    a.me_r() * 100.0
                );
            }
        }
    }
    // Error by segment length (test split, all conditions, no voiceprint).
    println!("\nPer segment length (test split, online / recluster, no voiceprint):\n");
    println!("| cond | length | segments | share of speech | online wrong | recluster wrong |");
    println!("|---|---|---|---|---|---|");
    for (ci, c) in CONDITIONS.iter().enumerate() {
        let prefix = if *c == "call" { "sys" } else { "mic" };
        let buckets = [
            (0u64, 1000u64, "<1 s"),
            (1000, 2000, "1–2 s"),
            (2000, 7000, "2–7 s"),
            (7000, 60_000, "≥7 s"),
        ];
        let mut acc: Vec<(usize, f64, f64, f64)> = vec![(0, 0.0, 0.0, 0.0); buckets.len()];
        let mut total = 0.0;
        for (mi, m) in meetings.iter().enumerate() {
            if m.split != "test" {
                continue;
            }
            let (_, _, _, online, recl) = run_one(m, &cache.segs[mi][ci], None, prefix, cfg);
            let ok_o = correct_after_mapping(m, &online);
            let ok_r = correct_after_mapping(m, &recl);
            for (i, s) in m.segments.iter().enumerate() {
                let d = (s.end_ms - s.start_ms) as f64;
                total += d;
                let b = buckets
                    .iter()
                    .position(|b| (b.0..b.1).contains(&(s.end_ms - s.start_ms)))
                    .unwrap();
                acc[b].0 += 1;
                acc[b].1 += d;
                if !ok_o[i] {
                    acc[b].2 += d;
                }
                if !ok_r[i] {
                    acc[b].3 += d;
                }
            }
        }
        for (b, a) in buckets.iter().zip(acc) {
            println!(
                "| {c} | {} | {} | {:.1}% | {:.1}% | {:.1}% |",
                b.2,
                a.0,
                a.1 / total * 100.0,
                a.2 / a.1.max(1.0) * 100.0,
                a.3 / a.1.max(1.0) * 100.0
            );
        }
    }
    Ok(())
}

/// Per segment: is its label right under the optimal hyp → ref mapping?
fn correct_after_mapping(m: &Meeting, hyp: &[Option<String>]) -> Vec<bool> {
    let mut r_ix: HashMap<&str, usize> = HashMap::new();
    let mut h_ix: HashMap<&str, usize> = HashMap::new();
    for s in &m.segments {
        let k = r_ix.len();
        r_ix.entry(&s.speaker).or_insert(k);
    }
    for h in hyp.iter().flatten() {
        let k = h_ix.len();
        h_ix.entry(h).or_insert(k);
    }
    let mut conf = vec![vec![0.0; r_ix.len()]; h_ix.len()];
    for (s, h) in m.segments.iter().zip(hyp) {
        if let Some(h) = h {
            conf[h_ix[h.as_str()]][r_ix[s.speaker.as_str()]] += (s.end_ms - s.start_ms) as f64;
        }
    }
    let map = metrics::max_weight_matching(&conf);
    m.segments
        .iter()
        .zip(hyp)
        .map(|(s, h)| {
            h.as_ref()
                .is_some_and(|h| map[h_ix[h.as_str()]] == Some(r_ix[s.speaker.as_str()]))
        })
        .collect()
}

// ------------------------------------------------------------------ bench

fn rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|x| x.parse().ok())
        })
        .unwrap_or(0)
}

fn cmd_bench(model: &str, threads: i32, reps: usize) -> Result<()> {
    let meetings = load_meetings()?;
    let audio = read_wav(
        &root()
            .join("meetings")
            .join(&meetings[0].id)
            .join("room.wav"),
    )?;
    let rss0 = rss_kb();
    let t = Instant::now();
    let mut e = Embedder::new(&model_path(model), threads)?;
    let init_ms = t.elapsed().as_secs_f64() * 1000.0;
    let rss1 = rss_kb();
    let mut row = vec![
        model_stem(model),
        format!(
            "{:.1}",
            std::fs::metadata(model_path(model))?.len() as f64 / 1e6
        ),
        threads.to_string(),
        format!("{init_ms:.0}"),
    ];
    for secs in [1.0f64, 2.0, 5.0, 10.0, 20.0] {
        let n = (secs * 16000.0) as usize;
        let start = 16000 * 30;
        let x = &audio[start..start + n];
        for _ in 0..3 {
            e.embed(x)?;
        }
        let mut ts: Vec<f64> = (0..reps)
            .map(|_| {
                let t = Instant::now();
                e.embed(x).unwrap();
                t.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        ts.sort_by(f64::total_cmp);
        row.push(format!("{:.1}", ts[ts.len() / 2]));
    }
    // change_points on a 20 s utterance
    let x = &audio[16000 * 30..16000 * 50];
    let cfg = ClusterConfig::default();
    kenes_speakers::change_points(&mut e, x, &cfg);
    let mut ts: Vec<f64> = (0..5)
        .map(|_| {
            let t = Instant::now();
            kenes_speakers::change_points(&mut e, x, &cfg);
            t.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    ts.sort_by(f64::total_cmp);
    let rss2 = rss_kb();
    row.push(format!("{}", (rss1.saturating_sub(rss0)) / 1024));
    row.push(format!("{}", (rss2.saturating_sub(rss0)) / 1024));
    row.push(format!("{:.0}", ts[2]));
    println!("| {} |", row.join(" | "));
    Ok(())
}

/// Objective on both splits while one knob varies (others from `base`).
fn cmd_curve(
    model: &str,
    key: &str,
    from: f32,
    to: f32,
    step: f32,
    base: ClusterConfig,
) -> Result<()> {
    let meetings = load_meetings()?;
    let cache = load_cache(model, &meetings)?;
    println!("| {key} | online tune | online test | recluster tune | recluster test | room online test | room recluster test |");
    println!("|---|---|---|---|---|---|---|");
    for v in range(from, to, step) {
        let mut c = base.clone();
        parse_set(
            &mut c,
            &format!(
                "{key}={}",
                if key.ends_with("_ms") {
                    format!("{}", v as u64)
                } else {
                    format!("{v}")
                }
            ),
        )?;
        if key == "threshold" {
            c.merge_threshold = v + (base.merge_threshold - base.threshold);
        }
        let tune: Vec<RunResult> = (0..3)
            .map(|k| evaluate(&meetings, &cache, "tune", k, false, &c))
            .collect();
        let test: Vec<RunResult> = (0..3)
            .map(|k| evaluate(&meetings, &cache, "test", k, false, &c))
            .collect();
        println!(
            "| {v} | {:.2}% | {:.2}% | {:.2}% | {:.2}% | {:.2}% | {:.2}% |",
            objective(&tune, false) * 100.0,
            objective(&test, false) * 100.0,
            objective(&tune, true) * 100.0,
            objective(&test, true) * 100.0,
            test[2].online.ser() * 100.0,
            test[2].recl.ser() * 100.0
        );
    }
    Ok(())
}

/// Like `curve`, on pipeline segments: per split and condition, online / recluster accuracy.
fn cmd_curve_real(
    model: &str,
    variant: &str,
    key: &str,
    from: f32,
    to: f32,
    step: f32,
    base: ClusterConfig,
) -> Result<()> {
    let meetings = load_meetings()?;
    let set = load_real(model, variant, &meetings)?;
    println!("| {key} | tune online | tune recluster | test online | test recluster | worst meeting Δ recluster − online: tune / test |");
    println!("|---|---|---|---|---|---|");
    for v in range(from, to, step) {
        let mut c = base.clone();
        parse_set(
            &mut c,
            &format!(
                "{key}={}",
                if key.ends_with("_ms") {
                    format!("{}", v as u64)
                } else {
                    format!("{v}")
                }
            ),
        )?;
        let mut cells = Vec::new();
        for split in ["tune", "test"] {
            let r: Vec<RealRun> = (0..REAL_CONDITIONS.len())
                .map(|ci| eval_real(&meetings, &set, split, ci, false, &c))
                .collect();
            let acc =
                |f: &dyn Fn(&RealRun) -> f64| r.iter().map(f).sum::<f64>() / r.len() as f64 * 100.0;
            cells.push(format!("{:.1}%", acc(&|x| x.online.acc())));
            cells.push(format!("{:.1}%", acc(&|x| x.recl.acc())));
        }
        let mut worst = [(f64::MAX, String::new()), (f64::MAX, String::new())];
        for (mi, m) in meetings.iter().enumerate() {
            let w = &mut worst[usize::from(m.split == "test")];
            for (ci, cond) in REAL_CONDITIONS.iter().enumerate() {
                let r = run_real(&set, mi, ci, None, &c);
                let d = (r.recl.acc() - r.online.acc()) * 100.0;
                if d < w.0 {
                    *w = (d, format!("{} {cond}", m.id));
                }
            }
        }
        println!(
            "| {v} | {} | {:+.1} ({}) / {:+.1} ({}) |",
            cells.join(" | "),
            worst[0].0,
            worst[0].1,
            worst[1].0,
            worst[1].1
        );
    }
    Ok(())
}

/// Label × true-speaker seconds for one meeting/condition, online and after recluster.
fn cmd_debug_real(
    model: &str,
    variant: &str,
    meeting: &str,
    cond: &str,
    cfg: &ClusterConfig,
) -> Result<()> {
    let meetings = load_meetings()?;
    let set = load_real(model, variant, &meetings)?;
    let mi = meetings
        .iter()
        .position(|m| m.id == meeting)
        .context("meeting")?;
    let ci = REAL_CONDITIONS
        .iter()
        .position(|c| *c == cond)
        .context("cond")?;
    let prefix = if cond == "call" { "sys" } else { "mic" };
    let segs = &set.segs[mi][ci];
    let embs = &set.embs[mi][ci];
    let t = &set.truths[mi];
    let mut order: Vec<usize> = (0..segs.len()).collect();
    order.sort_by_key(|&i| (segs[i].1, i));
    let mut c = OnlineClusterer::new(prefix, cfg.clone());
    let mut online = vec![None; segs.len()];
    let use_emb = |i: usize| {
        embs[i]
            .clone()
            .filter(|_| segs[i].1 - segs[i].0 >= cfg.min_embed_ms)
    };
    for &i in &order {
        online[i] = c.assign(use_emb(i).as_deref(), segs[i].0, segs[i].1);
    }
    let items: Vec<ClusterItem> = order
        .iter()
        .map(|&i| ClusterItem {
            segment_id: i.to_string(),
            prefix: prefix.to_string(),
            embedding: use_emb(i).unwrap_or_default(),
            duration_ms: segs[i].1 - segs[i].0,
            online_label: online[i].clone(),
        })
        .collect();
    let mut recl = vec![None; segs.len()];
    for (id, l) in recluster(&items, None, cfg) {
        recl[id.parse::<usize>().unwrap()] = l;
    }
    for (name, labels) in [("online", &online), ("recluster", &recl)] {
        let mut tab: BTreeMap<String, Vec<u32>> = BTreeMap::new();
        for ((s, e), l) in segs.iter().zip(labels.iter()) {
            let row = tab
                .entry(l.clone().unwrap_or_else(|| "-".into()))
                .or_insert_with(|| vec![0; t.speakers.len()]);
            for (k, n) in t.counts(*s, *e).iter().enumerate() {
                row[k] += n;
            }
        }
        println!(
            "{name}: label → seconds per true speaker (S0..S{})",
            t.speakers.len() - 1
        );
        for (l, row) in tab {
            let cells: Vec<String> = row
                .iter()
                .map(|&n| {
                    if n == 0 {
                        ".".into()
                    } else {
                        format!("{:.0}", n as f64 * FRAME_MS as f64 / 1000.0)
                    }
                })
                .collect();
            println!("  {l:>7}: {}", cells.join(" "));
        }
    }
    Ok(())
}

/// Time `recluster` on a long synthetic meeting: `n` segments of 15 voices.
fn cmd_bench_recluster(n: usize, dim: usize) -> Result<()> {
    let mut x: u64 = 88172645463325252;
    let mut rnd = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % 10_000) as f32 / 10_000.0 - 0.5
    };
    let voices: Vec<Vec<f32>> = (0..15).map(|_| (0..dim).map(|_| rnd()).collect()).collect();
    let items: Vec<ClusterItem> = (0..n)
        .map(|i| {
            let v = &voices[(i * 7 + i / 3) % 15];
            ClusterItem {
                segment_id: i.to_string(),
                prefix: "sys".into(),
                embedding: v.iter().map(|a| a + 0.6 * rnd()).collect(),
                duration_ms: 800 + (i as u64 * 7919) % 12_000,
                online_label: Some(format!("sys:{}", 1 + i % 15)),
            }
        })
        .collect();
    let t = Instant::now();
    let r = recluster(&items, None, &ClusterConfig::default());
    let labels: std::collections::HashSet<_> = r.iter().filter_map(|(_, l)| l.clone()).collect();
    println!(
        "recluster: {n} segments × {dim} dims → {} speakers in {:.0} ms",
        labels.len(),
        t.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

// ------------------------------------------------------------------ main

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let models = match get("--models") {
        Some(m) => m.split(',').map(str::to_string).collect(),
        None => all_models()?,
    };
    let mut cfg = ClusterConfig::default();
    if let Some(s) = get("--set") {
        parse_set(&mut cfg, &s)?;
    }
    match args.first().map(String::as_str) {
        Some("embed") => cmd_embed(models, get("--jobs").map_or(Ok(6), |j| j.parse())?),
        Some("pairs") => cmd_pairs(models),
        Some("tune") => cmd_tune(
            &get("--model").context("--model")?,
            args.iter().any(|a| a == "--quick"),
            cfg,
        ),
        Some("report") => print_report(&get("--model").context("--model")?, &cfg),
        Some("segment") => cmd_segment(get("--jobs").map_or(Ok(4), |j| j.parse())?),
        Some("embed-real") => cmd_embed_real(
            &get("--model").context("--model")?,
            get("--jobs").map_or(Ok(8), |j| j.parse())?,
        ),
        Some("change-eval") => cmd_change_eval(
            &get("--model").context("--model")?,
            get("--hop").map_or(Ok(500), |j| j.parse())?,
            cfg,
        ),
        Some("split") => cmd_split(
            &get("--model").context("--model")?,
            get("--hop").map_or(Ok(500), |j| j.parse())?,
            &cfg,
            get("--jobs").map_or(Ok(8), |j| j.parse())?,
        )
        .map(|_| ()),
        Some("embed-pieces") => cmd_embed_pieces(
            &get("--model").context("--model")?,
            &get("--from-model").context("--from-model")?,
            &get("--variant").context("--variant")?,
            get("--jobs").map_or(Ok(8), |j| j.parse())?,
        ),
        Some("report-real") => cmd_report_real(
            &get("--model").context("--model")?,
            &get("--variant").unwrap_or_else(|| "raw".into()),
            &cfg,
        ),
        Some("tune-real") => cmd_tune_real(
            &get("--model").context("--model")?,
            &get("--variant").unwrap_or_else(|| "raw".into()),
            cfg,
        ),
        Some("curve") => {
            let f = |k: &str| -> Result<f32> { Ok(get(k).context(k.to_string())?.parse()?) };
            cmd_curve(
                &get("--model").context("--model")?,
                &get("--key").context("--key")?,
                f("--from")?,
                f("--to")?,
                f("--step")?,
                cfg,
            )
        }
        Some("curve-real") => {
            let f = |k: &str| -> Result<f32> { Ok(get(k).context(k.to_string())?.parse()?) };
            cmd_curve_real(
                &get("--model").context("--model")?,
                &get("--variant").unwrap_or_else(|| "raw".into()),
                &get("--key").context("--key")?,
                f("--from")?,
                f("--to")?,
                f("--step")?,
                cfg,
            )
        }
        Some("debug-real") => cmd_debug_real(
            &get("--model").context("--model")?,
            &get("--variant").unwrap_or_else(|| "raw".into()),
            &get("--meeting").context("--meeting")?,
            &get("--cond").context("--cond")?,
            &cfg,
        ),
        Some("bench-recluster") => {
            for n in [500, 1000, 2000, 4000, 6000] {
                cmd_bench_recluster(n, get("--dim").map_or(Ok(192), |t| t.parse())?)?;
            }
            Ok(())
        }
        Some("bench") => cmd_bench(
            &get("--model").context("--model")?,
            get("--threads").map_or(Ok(1), |t| t.parse())?,
            get("--reps").map_or(Ok(30), |t| t.parse())?,
        ),
        _ => {
            eprintln!("usage: diarize_eval embed|pairs|tune|report|bench (see source)");
            Ok(())
        }
    }
}
