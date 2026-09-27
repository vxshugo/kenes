//! Tests with the real models and real speech. They download models on first
//! run (~230 MB) and need test audio, so they are `#[ignore]`d:
//!
//! ```text
//! cargo test -p kenes-stt --release -- --ignored --nocapture
//! ```
//!
//! Audio comes from `$KENES_STT_TESTDATA`, else `testdata-cache/`, else the
//! benchmark's `bench/data/`: directories `fleurs_kk/`, `fleurs_ru/`,
//! `codeswitch/` with `<clip>.wav` (16 kHz mono) + `<clip>.txt` (reference).

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use kenes_stt::{ensure_model, models_dir, SttBackend, SttConfig, Transcriber, DEFAULT_MODEL};
use kenes_types::{AudioChunk, Segment, Source, SAMPLE_RATE};

const KAZAKH_LETTERS: &str = "әғқңөұүһі";

fn data_root() -> Option<PathBuf> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        std::env::var_os("KENES_STT_TESTDATA").map(PathBuf::from),
        Some(manifest.join("testdata-cache")),
        Some(manifest.join("../../bench/data")),
    ];
    candidates
        .into_iter()
        .flatten()
        .find(|p| p.join("fleurs_kk").is_dir())
}

/// Up to `n` `(wav, reference)` pairs from `set`, sorted by name.
fn clips(set: &str, n: usize) -> Vec<(PathBuf, String)> {
    let Some(root) = data_root() else {
        panic!("no test audio: set KENES_STT_TESTDATA or fill testdata-cache/ (see README)");
    };
    let mut wavs: Vec<PathBuf> = std::fs::read_dir(root.join(set))
        .unwrap_or_else(|e| panic!("{set}: {e}"))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "wav") && p.with_extension("txt").is_file())
        .collect();
    wavs.sort();
    wavs.truncate(n);
    wavs.into_iter()
        .map(|w| {
            let text = std::fs::read_to_string(w.with_extension("txt")).unwrap();
            (w, text.trim().to_string())
        })
        .collect()
}

fn read_wav(path: &Path) -> Vec<f32> {
    let mut r = hound::WavReader::open(path).unwrap();
    let spec = r.spec();
    assert_eq!(
        (spec.sample_rate, spec.channels),
        (SAMPLE_RATE, 1),
        "{}",
        path.display()
    );
    match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().map(Result::unwrap).collect(),
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.unwrap() as f32 * scale)
                .collect()
        }
    }
}

/// Lowercase, ё→е, letters/digits only, single spaces.
fn norm(s: &str) -> String {
    let s: String = s
        .to_lowercase()
        .replace('ё', "е")
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Character edit distance and reference length (after `norm`).
fn char_errors(hyp: &str, reference: &str) -> (usize, usize) {
    let h: Vec<char> = norm(hyp).chars().collect();
    let r: Vec<char> = norm(reference).chars().collect();
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
    (prev[h.len()], r.len())
}

fn transcriber(model: &str, partial_interval_ms: u64) -> Transcriber {
    transcriber_with(
        model,
        partial_interval_ms,
        SttConfig::default().max_segment_ms,
    )
}

fn transcriber_with(model: &str, partial_interval_ms: u64, max_segment_ms: u64) -> Transcriber {
    transcriber_full(model, partial_interval_ms, max_segment_ms, SttBackend::Auto)
}

fn transcriber_on(model: &str, backend: SttBackend) -> Transcriber {
    let t = transcriber_full(model, 700, SttConfig::default().max_segment_ms, backend);
    if backend != SttBackend::Auto {
        assert_eq!(t.backend(), backend, "{model} fell back from {backend}");
    }
    t
}

fn transcriber_full(
    model: &str,
    partial_interval_ms: u64,
    max_segment_ms: u64,
    backend: SttBackend,
) -> Transcriber {
    let dir = models_dir();
    ensure_model(model, &dir, &mut |_| {}).expect("download model");
    let t0 = Instant::now();
    let t = Transcriber::with_backend(
        SttConfig {
            model_id: model.into(),
            models_dir: dir,
            partial_interval_ms,
            max_segment_ms,
            ..SttConfig::default()
        },
        backend,
    )
    .unwrap();
    eprintln!(
        "loaded {model} ({} backend) in {:.2} s",
        t.backend(),
        t0.elapsed().as_secs_f64()
    );
    t
}

fn join_finals(segs: &[Segment]) -> String {
    segs.iter()
        .filter(|s| s.is_final)
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
#[ignore = "needs models and test audio"]
fn kazakh_and_russian_come_out_in_their_own_scripts() {
    let mut t = transcriber(DEFAULT_MODEL, 700);
    for (set, n) in [("fleurs_kk", 3), ("fleurs_ru", 3), ("codeswitch", 2)] {
        for (wav, reference) in clips(set, n) {
            let segs = t.transcribe_buffer(Source::Mic, &read_wav(&wav)).unwrap();
            let text = join_finals(&segs);
            let (e, len) = char_errors(&text, &reference);
            eprintln!(
                "[{set}] {}\n  ref: {reference}\n  hyp: {text}\n  CER {:.3}",
                wav.display(),
                e as f64 / len as f64
            );
            let kk_letters = text.chars().filter(|c| KAZAKH_LETTERS.contains(*c)).count();
            match set {
                "fleurs_kk" => assert!(kk_letters > 0, "no Kazakh letters in {text:?}"),
                "fleurs_ru" => assert!(
                    kk_letters * 100 <= text.len(),
                    "Kazakh letters in Russian: {text:?}"
                ),
                _ => {}
            }
            assert!(
                (e as f64) < 0.35 * len as f64,
                "CER too high for {}",
                wav.display()
            );
            // Cyrillic, not transliterated.
            assert!(
                text.chars()
                    .filter(|c| c.is_alphabetic())
                    .all(|c| !c.is_ascii()),
                "{text:?}"
            );
        }
    }
}

/// VAD splitting must not cost accuracy compared with decoding the whole
/// clip at once (i.e. no clipped syllables).
#[test]
#[ignore = "needs models and test audio"]
fn vad_segmentation_keeps_whole_clip_accuracy() {
    let mut t = transcriber(DEFAULT_MODEL, 700);
    let n: usize = std::env::var("KENES_STT_CLIPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    for set in ["fleurs_kk", "fleurs_ru", "codeswitch", "cv_kk"] {
        let (mut whole, mut split, mut total, mut audio_s, mut split_s) = (0, 0, 0, 0.0, 0.0);
        for (wav, reference) in clips(set, n) {
            let audio = read_wav(&wav);
            let whole_text = if std::env::var_os("KENES_STT_SKIP_WHOLE").is_some() {
                reference.clone()
            } else {
                t.recognize(&audio).unwrap()
            };
            let t0 = Instant::now();
            let segs = t.transcribe_buffer(Source::System, &audio).unwrap();
            split_s += t0.elapsed().as_secs_f64();
            audio_s += audio.len() as f64 / SAMPLE_RATE as f64;
            let split_text = join_finals(&segs);
            let (ew, len) = char_errors(&whole_text, &reference);
            let (es, _) = char_errors(&split_text, &reference);
            if es > ew + 2 && std::env::var_os("KENES_STT_VERBOSE").is_some() {
                eprintln!("[{set}] {}: split worse ({es} vs {ew} errors)\n  whole: {whole_text}\n  split: {split_text}", wav.display());
            }
            whole += ew;
            split += es;
            total += len;
        }
        let (cw, cs) = (whole as f64 / total as f64, split as f64 / total as f64);
        eprintln!(
            "{set}: CER whole-clip {cw:.4}, VAD-split {cs:.4}, split RTF {:.3}",
            split_s / audio_s
        );
        if std::env::var_os("KENES_STT_SKIP_WHOLE").is_none() {
            assert!(
                cs <= cw + 0.01,
                "{set}: VAD split CER {cs:.4} vs whole {cw:.4}"
            );
        }
    }
}

fn chunks_of(source: Source, audio: &[f32]) -> Vec<AudioChunk> {
    audio
        .chunks(SAMPLE_RATE as usize / 10)
        .enumerate()
        .map(|(i, c)| AudioChunk {
            source,
            start_ms: i as u64 * 100,
            samples: c.to_vec(),
        })
        .collect()
}

fn check_invariants(segs: &[Segment]) {
    use std::collections::{HashMap, HashSet};
    let mut finals: HashMap<&str, usize> = HashMap::new();
    let mut closed = HashSet::new();
    for s in segs {
        assert!(
            !closed.contains(&s.id),
            "segment for {} after its final",
            s.id
        );
        assert!(s.id.starts_with(&format!("{}-", s.source.as_str())));
        if s.is_final {
            *finals.entry(&s.id).or_default() += 1;
            closed.insert(s.id.clone());
        }
    }
    for s in segs {
        assert_eq!(
            finals.get(s.id.as_str()),
            Some(&1),
            "{} needs exactly one final",
            s.id
        );
    }
}

/// Both sources at once through the live worker, paced at `speed`× real
/// time (`None`: as fast as possible, so partials get skipped).
fn run_live(
    t: Transcriber,
    mic: &[f32],
    sys: &[f32],
    speed: Option<f64>,
) -> Vec<(Instant, Segment)> {
    let (atx, arx) = crossbeam_channel::unbounded();
    let (stx, srx) = crossbeam_channel::unbounded();
    let worker = t.spawn(arx, stx);
    let (mic, sys) = (chunks_of(Source::Mic, mic), chunks_of(Source::System, sys));
    let start = Instant::now();
    let feeder = std::thread::spawn(move || {
        for i in 0..mic.len().max(sys.len()) {
            if let Some(speed) = speed {
                let due = start + Duration::from_secs_f64((i + 1) as f64 * 0.1 / speed);
                std::thread::sleep(due.saturating_duration_since(Instant::now()));
            }
            for c in [mic.get(i), sys.get(i)].into_iter().flatten() {
                atx.send(c.clone()).unwrap();
            }
        }
    });
    let out: Vec<_> = srx.iter().map(|s| (Instant::now(), s)).collect();
    feeder.join().unwrap();
    worker.join().unwrap();
    out
}

fn concat_clips(set: &str, n: usize) -> (Vec<f32>, String) {
    let mut audio = Vec::new();
    let mut text = Vec::new();
    for (wav, reference) in clips(set, n) {
        audio.extend(read_wav(&wav));
        text.push(reference);
    }
    (audio, text.join(" "))
}

#[test]
#[ignore = "needs models and test audio"]
fn live_two_sources_match_offline_accuracy() {
    let (kk, kk_ref) = concat_clips("fleurs_kk", 3);
    let (ru, ru_ref) = concat_clips("fleurs_ru", 3);
    for speed in [None, Some(2.0)] {
        let out = run_live(transcriber(DEFAULT_MODEL, 700), &kk, &ru, speed);
        let segs: Vec<Segment> = out.iter().map(|(_, s)| s.clone()).collect();
        check_invariants(&segs);
        let by_source = |src| {
            segs.iter()
                .filter(|s| s.source == src)
                .cloned()
                .collect::<Vec<_>>()
        };
        let (mic, sys) = (by_source(Source::Mic), by_source(Source::System));
        let (e_kk, n_kk) = char_errors(&join_finals(&mic), &kk_ref);
        let (e_ru, n_ru) = char_errors(&join_finals(&sys), &ru_ref);
        let partials = segs.iter().filter(|s| !s.is_final).count();
        eprintln!(
            "speed {speed:?}: {} finals, {partials} partials; CER kk {:.3} ru {:.3}",
            segs.len() - partials,
            e_kk as f64 / n_kk as f64,
            e_ru as f64 / n_ru as f64
        );
        for s in segs.iter().filter(|s| s.is_final) {
            eprintln!(
                "  [{:>6} → {:>6}] {:<9} {}",
                s.start_ms, s.end_ms, s.id, s.text
            );
        }
        assert!((e_kk as f64) < 0.3 * n_kk as f64 && (e_ru as f64) < 0.3 * n_ru as f64);
        if speed.is_some() {
            // Partials get ~30 % of wall time, so at 2x there are fewer than live.
            assert!(partials >= 3, "expected partials at 2x, got {partials}");
        }
    }
}

/// Real download through GitHub's redirect, then a resumed one from a
/// half-finished `.part` file.
#[test]
#[ignore = "needs network"]
fn download_verified_fetches_and_resumes() {
    let url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";
    let sha = "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6";
    let size = 643_854;
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("vad/silero_vad.onnx");
    let mut calls = 0;
    kenes_stt::download_verified(url, sha, size, &dest, &mut |_| calls += 1).unwrap();
    assert_eq!(std::fs::metadata(&dest).unwrap().len(), size);
    assert!(calls >= 2);

    // Leave half of it as a .part and download again: resumes, same result.
    let data = std::fs::read(&dest).unwrap();
    std::fs::remove_file(&dest).unwrap();
    let part = dest.with_file_name("silero_vad.onnx.part");
    std::fs::write(&part, &data[..data.len() / 2]).unwrap();
    let mut first = None;
    kenes_stt::download_verified(url, sha, size, &dest, &mut |p| {
        if p > 0.0 && first.is_none() {
            first = Some(p);
        }
    })
    .unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), data);
    assert!(!part.exists());
    // Progress started at ~50 %, i.e. the first half wasn't downloaded again.
    assert!(
        first.unwrap() >= 0.49,
        "resume progress started at {first:?}"
    );
}

/// Two speakers with a 0.2 s gap (the VAD glues them into one utterance);
/// a splitter that finds the gap separates them again.
#[test]
#[ignore = "needs models and test audio"]
fn splitter_separates_glued_turns() {
    // 40 s limit: the VAD's soft split (at half the limit) mustn't kick in.
    let mut t = transcriber_with(DEFAULT_MODEL, 700, 40_000);
    t.set_splitter(Box::new(|s: &[f32]| {
        // Middle of the longest run of exact zeros (the synthetic gap).
        let (mut best, mut run_start, mut cur) = ((0, 0), 0, 0);
        for (i, &x) in s.iter().enumerate() {
            if x == 0.0 {
                if cur == 0 {
                    run_start = i;
                }
                cur += 1;
                if cur > best.1 {
                    best = (run_start, cur);
                }
            } else {
                cur = 0;
            }
        }
        if best.1 > 1600 {
            vec![best.0 + best.1 / 2]
        } else {
            vec![]
        }
    }));
    let kk = clips("fleurs_kk", 1).remove(0);
    let ru = clips("fleurs_ru", 1).remove(0);
    // Trim the clips' leading/trailing silence so the 0.2 s gap is all there is.
    let trim = |v: Vec<f32>| {
        let rms: Vec<f32> = v
            .chunks(160)
            .map(|c| (c.iter().map(|x| x * x).sum::<f32>() / c.len() as f32).sqrt())
            .collect();
        let peak = rms.iter().cloned().fold(0.0, f32::max);
        let first = rms.iter().position(|&r| r > 0.1 * peak).unwrap_or(0);
        let last = rms
            .iter()
            .rposition(|&r| r > 0.1 * peak)
            .unwrap_or(rms.len() - 1);
        v[first.saturating_sub(5) * 160..((last + 6) * 160).min(v.len())].to_vec()
    };
    let (a, b) = (trim(read_wav(&kk.0)), trim(read_wav(&ru.0)));
    let gap = vec![0.0; SAMPLE_RATE as usize / 5];
    let audio = [a.clone(), gap, b].concat();
    let mut unsplit = transcriber_with(DEFAULT_MODEL, 700, 40_000);
    let glued = unsplit.transcribe_buffer(Source::Mic, &audio).unwrap();
    assert_eq!(glued.len(), 1, "the VAD should glue the turns: {glued:?}");
    eprintln!("without splitter: {}", glued[0].text);
    let segs = t.transcribe_buffer(Source::Mic, &audio).unwrap();
    for s in &segs {
        eprintln!(
            "[{:>6} → {:>6}] {:<6} {}",
            s.start_ms, s.end_ms, s.id, s.text
        );
    }
    assert!(segs.len() >= 2, "{segs:?}");
    let a_ms = a.len() as u64 * 1000 / SAMPLE_RATE as u64;
    let first: Vec<_> = segs.iter().filter(|s| s.end_ms <= a_ms + 200).collect();
    let rest: Vec<_> = segs.iter().filter(|s| s.start_ms >= a_ms).collect();
    assert_eq!(
        first.len() + rest.len(),
        segs.len(),
        "a segment straddles the turn change"
    );
    let join = |v: &[&Segment]| {
        v.iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    };
    let (e1, n1) = char_errors(&join(&first), &kk.1);
    let (e2, n2) = char_errors(&join(&rest), &ru.1);
    eprintln!(
        "CER kk {:.3} ru {:.3}",
        e1 as f64 / n1 as f64,
        e2 as f64 / n2 as f64
    );
    assert!(e1 * 10 < n1 && e2 * 10 < n2);
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

/// The ONNX Runtime backend uses the ONNX Runtime that sherpa-onnx links statically, so the
/// Silero VAD, the speaker embedder (kenes-speakers) and both recognizer backends all run on
/// one runtime in one process. Load them all, use them concurrently from several threads, drop
/// everything and load again: results must not change.
#[test]
#[ignore = "needs models and test audio"]
fn one_onnx_runtime_for_vad_speakers_and_both_backends() {
    let dir = models_dir();
    let spk_model = kenes_speakers::ensure_speaker_model(&dir, &mut |_| {}).unwrap();
    // sherpa-onnx first (it creates the ORT environment), then `ort` joins it.
    let mut emb = kenes_speakers::Embedder::new(&spk_model, 1).unwrap();
    let mut ort_t = transcriber_on(DEFAULT_MODEL, SttBackend::Ort);
    let mut sherpa_t = transcriber_on(DEFAULT_MODEL, SttBackend::Sherpa);

    let (wav, reference) = clips("codeswitch", 1).remove(0);
    let audio = read_wav(&wav);
    let ort_text = ort_t.recognize(&audio).unwrap();
    let sherpa_text = sherpa_t.recognize(&audio).unwrap();
    let vad_text = join_finals(&ort_t.transcribe_buffer(Source::Mic, &audio).unwrap());
    let voice = emb.embed(&audio).unwrap();
    for (what, text) in [
        ("ort", &ort_text),
        ("sherpa", &sherpa_text),
        ("ort+vad", &vad_text),
    ] {
        let (e, n) = char_errors(text, &reference);
        eprintln!("{what}: CER {:.3}  {text}", e as f64 / n as f64);
        assert!(e * 10 < n, "{what} CER too high: {text}");
    }

    // All four models busy at once, each from its own thread.
    std::thread::scope(|s| {
        s.spawn(|| {
            for _ in 0..3 {
                assert_eq!(ort_t.recognize(&audio).unwrap(), ort_text);
            }
        });
        s.spawn(|| {
            for _ in 0..3 {
                assert_eq!(sherpa_t.recognize(&audio).unwrap(), sherpa_text);
            }
        });
        s.spawn(|| {
            for _ in 0..3 {
                let v = emb.embed(&audio).unwrap();
                assert!(cosine(&v, &voice) > 0.999);
            }
        });
    });

    // Tear everything down (the last sherpa-onnx object releases its ORT env reference)
    // and start again, as a new meeting would.
    drop((ort_t, sherpa_t, emb));
    let mut again = transcriber_on(DEFAULT_MODEL, SttBackend::Ort);
    assert_eq!(again.recognize(&audio).unwrap(), ort_text);
    assert_eq!(
        join_finals(&again.transcribe_buffer(Source::Mic, &audio).unwrap()),
        vad_text
    );
    let mut emb = kenes_speakers::Embedder::new(&spk_model, 1).unwrap();
    assert!(cosine(&emb.embed(&audio).unwrap(), &voice) > 0.999);
}

/// Every registry model loads on the ONNX Runtime backend (no silent fallback) and gets
/// its language right; `gigaam-v3-ru-ctc` has a different vocabulary (34 symbols) and no
/// `encoded_lengths` output.
#[test]
#[ignore = "needs models and test audio"]
fn every_model_runs_on_onnx_runtime() {
    for (model, set) in [
        ("gigaam-multilingual-ctc", "codeswitch"),
        ("gigaam-multilingual-large-ctc", "codeswitch"),
        ("gigaam-v3-ru-ctc", "fleurs_ru"),
    ] {
        let mut t = transcriber_on(model, SttBackend::Ort);
        let (mut errors, mut total) = (0, 0);
        for (wav, reference) in clips(set, 2) {
            let text = t.recognize(&read_wav(&wav)).unwrap();
            let (e, n) = char_errors(&text, &reference);
            eprintln!(
                "[{model}] {}: CER {:.3}  {text}",
                wav.display(),
                e as f64 / n as f64
            );
            errors += e;
            total += n;
        }
        assert!(errors * 20 < total, "{model}: CER {errors}/{total}");
        // Short and silent inputs decode to nothing rather than failing.
        assert_eq!(t.recognize(&[]).unwrap(), "");
        assert_eq!(t.recognize(&vec![0.0; 800]).unwrap(), "");
    }
}
