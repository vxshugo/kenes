//! Tests with the real speaker model. They download ~28 MB (plus a few small WAVs) into
//! `testdata-cache/`, so they are ignored by default:
//!
//! ```text
//! cargo test -p kenes-speakers --release -- --ignored
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use kenes_speakers::{
    change_points, cosine, ensure_speaker_model, is_speaker_model_downloaded, recluster,
    speaker_model_path, ClusterConfig, ClusterItem, Embedder, OnlineClusterer, SPEAKER_MODEL,
};

fn cache() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata-cache")
}

fn models_dir() -> PathBuf {
    cache().join("models-dir")
}

fn embedder() -> Embedder {
    let path = ensure_speaker_model(&models_dir(), &mut |_| {}).expect("download speaker model");
    Embedder::new(&path, 1).expect("load speaker model")
}

/// Sample recordings of three speakers published next to the sherpa-onnx speaker models.
fn sample(name: &str) -> Vec<f32> {
    const FILES: &[(&str, &str, u64)] = &[
        (
            "fangjun-sr-1.wav",
            "33c24061180224d2350143ee19e3af031446995c676bd25996325d34bb20a4d5",
            73_606,
        ),
        (
            "fangjun-sr-2.wav",
            "c9209ff0cc83cb3c30b3efb376cec6df93f33ceb0f3c1a8eb28fced7a8d08409",
            165_370,
        ),
        (
            "leijun-sr-1.wav",
            "160a3d9bf5dd5038da8191b4430e1f3f751461613ae9489401b6b35a61b488ad",
            134_474,
        ),
        (
            "leijun-sr-2.wav",
            "37a759c036f2520d143708dfe46d45901b38a0a6e621b52d6f1aef2c000d7fe0",
            152_368,
        ),
        (
            "liudehua-sr-1.wav",
            "f39ebb7357d537009a9b7e59a6c9c0199f2204fcffb3c2d9d829a6961813eca1",
            95_390,
        ),
    ];
    let &(file, sha, size) = FILES.iter().find(|f| f.0 == name).expect("known sample");
    let dest = cache().join("sr").join(file);
    kenes_stt::download_verified(
        &format!("https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/{file}"),
        sha,
        size,
        &dest,
        &mut |_| {},
    )
    .expect("download sample");
    read_wav(&dest)
}

fn read_wav(p: &Path) -> Vec<f32> {
    let mut r = hound::WavReader::open(p).unwrap();
    assert_eq!(r.spec().sample_rate, 16_000);
    r.samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

#[test]
#[ignore = "downloads the speaker model"]
fn ensure_model_downloads_once_and_verifies() {
    let dir = models_dir();
    let p = ensure_speaker_model(&dir, &mut |_| {}).unwrap();
    assert_eq!(p, speaker_model_path(&dir));
    assert!(is_speaker_model_downloaded(&dir));
    assert_eq!(std::fs::metadata(&p).unwrap().len(), SPEAKER_MODEL.size);
    // Second call is a stat, not a download.
    let t = Instant::now();
    let mut last = 0.0;
    ensure_speaker_model(&dir, &mut |x| last = x).unwrap();
    assert_eq!(last, 1.0);
    assert!(t.elapsed().as_millis() < 50);
}

#[test]
#[ignore = "downloads the speaker model"]
fn embeddings_separate_speakers() {
    let mut e = embedder();
    let names = [
        "fangjun-sr-1.wav",
        "fangjun-sr-2.wav",
        "leijun-sr-1.wav",
        "leijun-sr-2.wav",
        "liudehua-sr-1.wav",
    ];
    let v: Vec<Vec<f32>> = names.iter().map(|n| e.embed(&sample(n)).unwrap()).collect();
    for x in &v {
        assert_eq!(x.len(), e.dim());
        let norm: f32 = x.iter().map(|a| a * a).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }
    let same = [cosine(&v[0], &v[1]), cosine(&v[2], &v[3])];
    let diff = [
        cosine(&v[0], &v[2]),
        cosine(&v[0], &v[4]),
        cosine(&v[1], &v[3]),
        cosine(&v[2], &v[4]),
        cosine(&v[3], &v[4]),
    ];
    eprintln!("same {same:?} diff {diff:?}");
    let cfg = ClusterConfig::default();
    for s in same {
        assert!(
            s > cfg.threshold,
            "same-speaker similarity {s} below threshold"
        );
    }
    for d in diff {
        assert!(
            d < cfg.threshold,
            "different-speaker similarity {d} above threshold"
        );
    }
}

#[test]
#[ignore = "downloads the speaker model"]
fn embed_rejects_too_short_or_silent_audio() {
    let mut e = embedder();
    assert!(e.embed(&[]).is_err());
    assert!(e.embed(&[0.1; 800]).is_err(), "50 ms");
    // Silence gives a valid (if meaningless) embedding or an error, but never a crash.
    let _ = e.embed(&vec![0.0; 16_000]);
    let x = sample("leijun-sr-1.wav");
    assert!(e.embed(&x[..8000]).is_ok(), "0.5 s");
    let mut noisy = x.clone();
    noisy[100] = f32::NAN;
    assert!(e.embed(&noisy).is_ok(), "NaN samples are zeroed");
}

#[test]
#[ignore = "downloads the speaker model"]
fn pipeline_on_sample_speakers() {
    let mut e = embedder();
    // A tiny "call": three people take turns; the clusterer should find three voices.
    let clips = [
        ("fangjun-sr-2.wav", 0),
        ("leijun-sr-1.wav", 1),
        ("liudehua-sr-1.wav", 2),
        ("leijun-sr-2.wav", 1),
        ("fangjun-sr-1.wav", 0),
    ];
    let mut c = OnlineClusterer::new("sys", ClusterConfig::default());
    let mut t = 0u64;
    let mut items = Vec::new();
    let mut online = Vec::new();
    for (i, (f, _)) in clips.iter().enumerate() {
        let x = sample(f);
        let ms = x.len() as u64 / 16;
        let v = e.embed(&x).unwrap();
        let l = c.assign(Some(&v), t, t + ms);
        items.push(ClusterItem {
            segment_id: format!("system-{i}"),
            prefix: "sys".into(),
            embedding: v,
            duration_ms: ms,
            online_label: l.clone(),
        });
        online.push(l);
        t += ms + 500;
    }
    let online: Vec<&str> = online.iter().map(|l| l.as_deref().unwrap()).collect();
    assert_eq!(online, ["sys:1", "sys:2", "sys:3", "sys:2", "sys:1"]);
    let re = recluster(&items, None, &ClusterConfig::default());
    let re: Vec<&str> = re.iter().map(|(_, l)| l.as_deref().unwrap()).collect();
    assert_eq!(re, online, "recluster keeps correct online labels");
}

#[test]
#[ignore = "downloads the speaker model"]
fn embedder_is_fast_enough() {
    let mut e = embedder();
    let x = sample("fangjun-sr-2.wav"); // 5.2 s
    for _ in 0..3 {
        e.embed(&x).unwrap();
    }
    let mut ts: Vec<f64> = (0..10)
        .map(|_| {
            let t = Instant::now();
            e.embed(&x).unwrap();
            t.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    ts.sort_by(f64::total_cmp);
    eprintln!("5.2 s segment: median {:.1} ms (1 thread)", ts[5]);
    // Smoke check only (shared, busy machines); EVAL.md has the real numbers (~60 ms).
    assert!(ts[5] < 500.0);
}

#[test]
#[ignore = "downloads the speaker model"]
fn change_points_split_two_voices_not_one() {
    let mut e = embedder();
    let cfg = ClusterConfig::default();
    // Two people answering each other without a pause: one cut near the join.
    let a = sample("fangjun-sr-2.wav");
    let b = sample("leijun-sr-1.wav");
    let two: Vec<f32> = a.iter().chain(&b).copied().collect();
    let cuts = change_points(&mut e, &two, &cfg);
    eprintln!("two voices, join at {}: {cuts:?}", a.len());
    assert_eq!(cuts.len(), 1, "{cuts:?}");
    assert!(
        cuts[0].abs_diff(a.len()) <= 16_000 * 6 / 10,
        "cut at {} vs join at {}",
        cuts[0],
        a.len()
    );
    // One person, two recordings back to back: no cut.
    let c = sample("leijun-sr-2.wav");
    let one: Vec<f32> = b.iter().chain(&c).copied().collect();
    assert!(change_points(&mut e, &one, &cfg).is_empty());
    // Too short to hold two pieces: nothing.
    assert!(change_points(&mut e, &two[..16_000 * 3 / 2], &cfg).is_empty());
}

#[test]
fn embedder_is_send_and_sync() {
    fn check<T: Send + Sync>() {}
    check::<Embedder>();
    check::<OnlineClusterer>();
}
