//! GigaAM CTC models on ONNX Runtime, with GigaAM's own log-mel front-end.
//!
//! **Why.** sherpa-onnx 1.13.8 feeds every GigaAM model the v1/v2 front-end: a 25 ms kaldi
//! fbank with n_fft 400. GigaAM-v3 and GigaAM-Multilingual were trained on torchaudio's
//! `MelSpectrogram` with a 20 ms window (n_fft = win = 320, hop 160, no centre padding), and
//! on code-switched ru/kk audio the mismatch costs 30–50 % more errors (`bench/RESULTS.md`).
//! [`LogMel`] computes the training front-end exactly; [`GigaamOrt`] runs the same int8 ONNX
//! file on it and decodes greedily.
//!
//! **Which ONNX Runtime.** sherpa-onnx links its own ONNX Runtime statically (1.28.2 in
//! sherpa-onnx 1.13.8) and we still need sherpa for the Silero VAD and speaker embeddings.
//! Linking a second runtime (the `ort` crate's default) would clash with it, and loading one
//! at run time would mean shipping and downloading a 20 MB library per platform. Instead
//! `ort` is built with `alternative-backend`, which makes it link nothing, and [`init_ort`]
//! hands it the C API table of the runtime that is already in the binary: `OrtGetApiBase`
//! resolves at link time against sherpa-onnx-sys's `libonnxruntime.a`. So there is exactly one
//! ONNX Runtime in the process, shared by the VAD, the speaker model and this recognizer, and
//! nothing extra to link, bundle or download on Linux or macOS.

use std::ffi::CStr;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, bail, Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{TensorElementType, TensorRef, ValueType};
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

use crate::engine::{Recognizer, TAIL_PADDING};

/// FFT size and window length: 20 ms at 16 kHz.
pub const N_FFT: usize = 320;
/// Frame shift: 10 ms.
pub const HOP: usize = 160;
/// Mel bins (the models' `features` input is `[1, N_MELS, T]`).
pub const N_MELS: usize = 64;
/// Floor applied before the log, as in GigaAM's `SpecScaler`.
const LOG_FLOOR: f64 = 1e-9;
const LOG_CEIL: f64 = 1e9;
/// Fewer frames than this (0.1 s) decode to nothing; the reference implementation
/// (`bench/run_bench.py`) skips them too.
const MIN_FRAMES: usize = 10;
/// Level of the silence appended to every utterance: white noise at about −70 dBFS.
///
/// GigaAM drops the last characters when the audio stops right after the last phoneme, so
/// every decode gets 0.3 s of trailing silence. Digital zeros don't work for this front-end:
/// they hit the `log(1e-9)` floor (−20.7, where this noise gives about −11) and cost accuracy.
/// On Common Voice kk (short, tightly trimmed clips), zeros gave 12.1 % WER where no padding
/// gave 10.7 % and this noise 8.2 % (whole clips, 220M model). −60 and −80 dBFS were within
/// noise of −70; dithering the whole utterance instead made code-switched audio worse.
const PAD_NOISE_STD: f32 = 3e-4;

/// GigaAM-v3 / Multilingual log-mel features: an exact port of `gigaam.preprocess.FeatureExtractor`
/// as configured for those models (torchaudio `MelSpectrogram(sample_rate=16000, n_fft=320,
/// win_length=320, hop_length=160, n_mels=64, center=False)` followed by `log(clamp(x, 1e-9, 1e9))`).
///
/// - frames of 320 samples every 160, no padding at either end;
/// - periodic Hann window, `|rfft(320)|²` (power spectrum, 161 bins);
/// - 64 triangular HTK-scale mel filters over 0–8000 Hz, no normalisation;
/// - natural log of the mel energies clamped to `[1e-9, 1e9]`.
///
/// Computed in `f64` and returned as `f32`. Exported for benchmarks and tests.
pub struct LogMel {
    fft: Arc<dyn RealToComplex<f64>>,
    window: Vec<f64>,
    filters: Vec<MelFilter>,
    frame: Vec<f64>,
    spectrum: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    power: Vec<f64>,
}

/// One triangular filter: weights for FFT bins `start..start + weights.len()`.
struct MelFilter {
    start: usize,
    weights: Vec<f64>,
}

impl Default for LogMel {
    fn default() -> Self {
        Self::new()
    }
}

impl LogMel {
    pub fn new() -> Self {
        let fft = RealFftPlanner::<f64>::new().plan_fft_forward(N_FFT);
        let window = (0..N_FFT)
            .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos())
            .collect();
        LogMel {
            frame: fft.make_input_vec(),
            spectrum: fft.make_output_vec(),
            scratch: fft.make_scratch_vec(),
            power: vec![0.0; N_FFT / 2 + 1],
            filters: mel_filters(),
            window,
            fft,
        }
    }

    /// Number of feature frames for `len` samples: `1 + (len - 320) / 160`, or 0 below 320.
    pub fn frames(len: usize) -> usize {
        if len < N_FFT {
            0
        } else {
            1 + (len - N_FFT) / HOP
        }
    }

    /// Features for 16 kHz mono `samples`, mel-major: value `(m, t)` is at `m * T + t`, where
    /// `T = LogMel::frames(samples.len())`, the layout of the models' `[1, 64, T]` input.
    /// Non-finite samples count as 0.
    pub fn compute(&mut self, samples: &[f32]) -> Vec<f32> {
        let t = Self::frames(samples.len());
        let mut out = vec![0.0f32; N_MELS * t];
        for f in 0..t {
            let src = &samples[f * HOP..f * HOP + N_FFT];
            for ((x, &s), &w) in self.frame.iter_mut().zip(src).zip(&self.window) {
                *x = if s.is_finite() { s as f64 * w } else { 0.0 };
            }
            self.fft
                .process_with_scratch(&mut self.frame, &mut self.spectrum, &mut self.scratch)
                .expect("FFT buffers are sized by the plan");
            for (p, c) in self.power.iter_mut().zip(&self.spectrum) {
                *p = c.norm_sqr();
            }
            for (m, filter) in self.filters.iter().enumerate() {
                let energy: f64 = filter
                    .weights
                    .iter()
                    .zip(&self.power[filter.start..])
                    .map(|(w, p)| w * p)
                    .sum();
                out[m * t + f] = energy.clamp(LOG_FLOOR, LOG_CEIL).ln() as f32;
            }
        }
        out
    }
}

/// torchaudio's `melscale_fbanks(n_freqs=161, f_min=0, f_max=8000, n_mels=64,
/// sample_rate=16000, norm=None, mel_scale="htk")`, stored sparsely.
fn mel_filters() -> Vec<MelFilter> {
    let n_freqs = N_FFT / 2 + 1;
    let f_max = 8000.0f64;
    let hz_to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let mel_to_hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
    // linspace(0, f_max, n_freqs) and linspace(mel(0), mel(f_max), n_mels + 2).
    let freq = |i: usize| f_max * i as f64 / (n_freqs - 1) as f64;
    let m_max = hz_to_mel(f_max);
    let f_pts: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz(m_max * i as f64 / (N_MELS + 1) as f64))
        .collect();
    (0..N_MELS)
        .map(|m| {
            let (lo, mid, hi) = (f_pts[m], f_pts[m + 1], f_pts[m + 2]);
            let weight = |i: usize| {
                let f = freq(i);
                let down = (f - lo) / (mid - lo);
                let up = (hi - f) / (hi - mid);
                down.min(up).max(0.0)
            };
            let bins: Vec<usize> = (0..n_freqs).filter(|&i| weight(i) > 0.0).collect();
            match (bins.first(), bins.last()) {
                (Some(&a), Some(&b)) => MelFilter {
                    start: a,
                    weights: (a..=b).map(weight).collect(),
                },
                // Can't happen at 16 kHz / 64 bins, but an all-zero filter is harmless.
                _ => MelFilter {
                    start: 0,
                    weights: Vec::new(),
                },
            }
        })
        .collect()
}

/// Version of the ONNX Runtime `ort` was pointed at, or why that failed.
static ORT: OnceLock<Result<String, String>> = OnceLock::new();

/// Point `ort` at the ONNX Runtime linked into this binary by sherpa-onnx (see the module
/// docs). Idempotent; returns the runtime's version.
pub(crate) fn init_ort() -> Result<&'static str> {
    let r = ORT.get_or_init(|| {
        // SAFETY: `OrtGetApiBase` is ONNX Runtime's C entry point. With `alternative-backend`,
        // ort-sys only declares it; the definition comes from sherpa-onnx-sys's static
        // `libonnxruntime.a` (or its shared build, with `SHERPA_ONNX_LIB_DIR`). The returned
        // struct and strings are static data of the runtime.
        let (version, api) = unsafe {
            let base = ort::sys::OrtGetApiBase();
            if base.is_null() {
                return Err("OrtGetApiBase returned null".to_string());
            }
            let version = CStr::from_ptr(((*base).GetVersionString)())
                .to_string_lossy()
                .into_owned();
            let api = ((*base).GetApi)(ort::sys::ORT_API_VERSION);
            if api.is_null() {
                return Err(format!(
                    "ONNX Runtime {version} does not provide C API version {}",
                    ort::sys::ORT_API_VERSION
                ));
            }
            // Only the fields up to ORT_API_VERSION exist in `OrtApi` (api-17 feature), and the
            // runtime's table is at least that long, so this copy stays in bounds.
            (version, (*api).clone())
        };
        ort::set_api(api);
        ort::init()
            .with_name("kenes-stt")
            .with_telemetry(false)
            .commit();
        // `ort` releases its environment from a `.fini_array` hook at exit, which for a
        // statically linked runtime runs after the runtime's own C++ static destructors.
        // Keep one reference forever so that release never happens (sherpa-onnx's objects
        // also just stay alive at exit).
        let env = ort::environment::Environment::current().map_err(|e| e.to_string())?;
        std::mem::forget(env);
        log::info!("using the ONNX Runtime {version} linked by sherpa-onnx");
        Ok(version)
    });
    r.as_deref()
        .map_err(|e| anyhow!("ONNX Runtime unavailable: {e}"))
}

/// Deterministic white noise, uniform with standard deviation `std` (xorshift32, fixed seed,
/// so the same utterance always decodes the same way).
fn quiet_noise(n: usize, std: f32) -> impl Iterator<Item = f32> {
    let amp = std * 3f32.sqrt();
    let mut state = 0x9E37_79B9u32;
    (0..n).map(move |_| {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32 / u32::MAX as f32 * 2.0 - 1.0) * amp
    })
}

/// A GigaAM CTC model (`model.int8.onnx` + `tokens.txt`, the sherpa-onnx export) run
/// directly on ONNX Runtime with [`LogMel`] features and greedy CTC decoding.
pub(crate) struct GigaamOrt {
    session: Session,
    mel: LogMel,
    /// Symbol per token id; id 0 is the word separator `" "`.
    tokens: Vec<String>,
    blank: usize,
}

impl GigaamOrt {
    pub fn new(model: &Path, tokens: &Path, num_threads: usize) -> Result<Self> {
        let version = init_ort()?;
        let (tokens, blank) = read_tokens(tokens)?;
        let session = build_session(model, num_threads.max(1)).with_context(|| {
            format!("ONNX Runtime {version} failed to load {}", model.display())
        })?;
        check_io(&session, tokens.len())
            .with_context(|| format!("{} is not a GigaAM CTC model", model.display()))?;
        Ok(GigaamOrt {
            session,
            mel: LogMel::new(),
            tokens,
            blank,
        })
    }
}

fn build_session(model: &Path, num_threads: usize) -> ort::Result<Session> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::All)?
        .with_intra_threads(num_threads)?
        .with_inter_threads(1)?
        .with_parallel_execution(false)?
        // Spinning worker threads tripled CPU time for the same decodes (see the sherpa
        // config in engine.rs); a meeting app is idle most of the time.
        .with_intra_op_spinning(false)?
        .with_inter_op_spinning(false)?
        .commit_from_file(model)
}

/// The model must take `features` f32 `[N, 64, T]` + `feature_lengths` i64 `[N]` and give
/// `log_probs` f32 `[N, T', vocab]`. `encoded_lengths` is optional (the v3 export lacks it).
fn check_io(session: &Session, vocab: usize) -> Result<()> {
    let tensor = |t: &ValueType| match t {
        ValueType::Tensor { ty, shape, .. } => Some((*ty, shape.to_vec())),
        _ => None,
    };
    let input = |name: &str| {
        session
            .inputs()
            .iter()
            .find(|i| i.name() == name)
            .and_then(|i| tensor(i.dtype()))
            .ok_or_else(|| anyhow!("no tensor input {name:?}"))
    };
    let (ty, shape) = input("features")?;
    if ty != TensorElementType::Float32 || shape.len() != 3 || shape[1] != N_MELS as i64 {
        bail!("input `features` is {ty:?} {shape:?}, expected f32 [N, {N_MELS}, T]");
    }
    let (ty, _) = input("feature_lengths")?;
    if ty != TensorElementType::Int64 {
        bail!("input `feature_lengths` is {ty:?}, expected i64");
    }
    let (ty, shape) = session
        .outputs()
        .iter()
        .find(|o| o.name() == "log_probs")
        .and_then(|o| tensor(o.dtype()))
        .ok_or_else(|| anyhow!("no tensor output \"log_probs\""))?;
    if ty != TensorElementType::Float32 || shape.len() != 3 {
        bail!("output `log_probs` is {ty:?} {shape:?}, expected f32 [N, T, vocab]");
    }
    if shape[2] >= 0 && shape[2] as usize != vocab {
        bail!(
            "output `log_probs` has {} classes but tokens.txt has {vocab}",
            shape[2]
        );
    }
    Ok(())
}

/// Parse sherpa-onnx `tokens.txt` (`<symbol> <id>` per line; the space token's line is
/// `"  0"`). Returns the symbols by id and the blank id (`<blk>`, else the last id).
fn read_tokens(path: &Path) -> Result<(Vec<String>, usize)> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut by_id: Vec<Option<String>> = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let (sym, id) = line
            .rsplit_once(' ')
            .ok_or_else(|| anyhow!("{}:{}: expected `<symbol> <id>`", path.display(), n + 1))?;
        let id: usize = id
            .parse()
            .with_context(|| format!("{}:{}: bad id", path.display(), n + 1))?;
        if id >= 4096 {
            bail!("{}:{}: id {id} is out of range", path.display(), n + 1);
        }
        if by_id.len() <= id {
            by_id.resize(id + 1, None);
        }
        // `"  0".rsplit_once(' ')` gives `(" ", "0")`; SentencePiece-style `▁` is a space too.
        let sym = if sym.is_empty() { " " } else { sym };
        by_id[id] = Some(sym.replace('\u{2581}', " "));
    }
    if by_id.len() < 2 || by_id.iter().any(Option::is_none) {
        bail!("{}: token ids are not 0..n", path.display());
    }
    let tokens: Vec<String> = by_id.into_iter().map(Option::unwrap).collect();
    let blank = tokens
        .iter()
        .position(|t| t == "<blk>")
        .unwrap_or(tokens.len() - 1);
    Ok((tokens, blank))
}

/// Greedy CTC: best class per frame, merge repeats, drop blanks, then trim and collapse
/// whitespace. `log_probs` is `[frames][vocab]` row-major.
fn ctc_greedy(log_probs: &[f32], vocab: usize, tokens: &[String], blank: usize) -> String {
    let mut text = String::new();
    let mut prev = usize::MAX;
    for row in log_probs.chunks_exact(vocab) {
        let best = row
            .iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            })
            .0;
        if best != prev && best != blank {
            if let Some(t) = tokens.get(best) {
                text.push_str(t);
            }
        }
        prev = best;
    }
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

impl Recognizer for GigaamOrt {
    /// Like the default, but the 0.3 s of trailing silence is quiet noise
    /// ([`PAD_NOISE_STD`]), not zeros.
    fn recognize(&mut self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        let mut padded = Vec::with_capacity(samples.len() + TAIL_PADDING);
        padded.extend_from_slice(samples);
        padded.extend(quiet_noise(TAIL_PADDING, PAD_NOISE_STD));
        self.decode(&padded)
    }

    fn decode(&mut self, samples: &[f32]) -> Result<String> {
        let t = LogMel::frames(samples.len());
        if t < MIN_FRAMES {
            return Ok(String::new());
        }
        let features = self.mel.compute(samples);
        let lengths = [t as i64];
        let outputs = self.session.run(ort::inputs![
            "features" => TensorRef::from_array_view(([1usize, N_MELS, t], features.as_slice()))?,
            "feature_lengths" => TensorRef::from_array_view(([1usize], &lengths[..]))?,
        ])?;
        let (shape, log_probs) = outputs["log_probs"].try_extract_tensor::<f32>()?;
        if shape.len() != 3 || shape[0] != 1 || shape[2] as usize != self.tokens.len() {
            bail!("unexpected log_probs shape {shape:?}");
        }
        let frames = shape[1] as usize;
        // Batch of one without padding: every output frame is valid, but honour
        // `encoded_lengths` (i32 in the Multilingual export) when the model has it.
        let valid = match outputs.get("encoded_lengths") {
            Some(v) => v
                .try_extract_tensor::<i64>()
                .map(|(_, l)| l.first().copied().unwrap_or(0))
                .or_else(|_| {
                    v.try_extract_tensor::<i32>()
                        .map(|(_, l)| l.first().copied().unwrap_or(0) as i64)
                })?
                .clamp(0, frames as i64) as usize,
            None => frames,
        };
        let vocab = self.tokens.len();
        Ok(ctc_greedy(
            &log_probs[..valid * vocab],
            vocab,
            &self.tokens,
            self.blank,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_matches_torchaudio_without_centering() {
        assert_eq!(LogMel::frames(0), 0);
        assert_eq!(LogMel::frames(319), 0);
        assert_eq!(LogMel::frames(320), 1);
        assert_eq!(LogMel::frames(479), 1);
        assert_eq!(LogMel::frames(480), 2);
        assert_eq!(LogMel::frames(16_000), 99);
    }

    #[test]
    fn filterbank_is_htk_triangles() {
        let f = mel_filters();
        assert_eq!(f.len(), N_MELS);
        // Filters are ordered, overlap by one neighbour and stay inside 0..=160.
        for w in f.windows(2) {
            assert!(w[0].start <= w[1].start);
        }
        let last = f.last().unwrap();
        assert!(last.start + last.weights.len() <= N_FFT / 2 + 1);
        // Peak value < 1 unless a bin hits the centre exactly; never above 1.
        for m in &f {
            assert!(!m.weights.is_empty());
            assert!(m.weights.iter().all(|&w| (0.0..=1.0).contains(&w)));
        }
        // The first filter starts at 0 Hz; its centre (~57 Hz) is just above bin 1 (50 Hz).
        assert_eq!(f[0].start, 1);
    }

    /// Reference values from `bench/run_bench.py`'s `GigaAMLogMel` formulas (numpy, float64)
    /// for `x[n] = 0.5 sin(2π·440 n/16000) + 0.1 sin(2π·3000 n/16000)`, n < 1600.
    #[test]
    fn features_match_the_python_reference() {
        let x: Vec<f32> = (0..1600)
            .map(|n| {
                let t = n as f64 / 16_000.0;
                (0.5 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                    + 0.1 * (2.0 * std::f64::consts::PI * 3000.0 * t).sin()) as f32
            })
            .collect();
        let t = LogMel::frames(x.len());
        assert_eq!(t, 9);
        let f = LogMel::new().compute(&x);
        // (mel, frame, value), computed in float64 with the same formulas.
        let expected = [
            (0, 0, -9.823_614_f32),
            (7, 0, -3.986_469),
            (7, 4, -4.014_165),
            (10, 3, 5.460_522),
            (63, 8, -20.723_266), // log(1e-9): nothing above 7.4 kHz
        ];
        for (m, fr, v) in expected {
            let got = f[m * t + fr];
            assert!(
                (got - v).abs() < 1e-4,
                "mel {m} frame {fr}: got {got}, expected {v}"
            );
        }
    }

    #[test]
    fn silence_hits_the_log_floor() {
        let f = LogMel::new().compute(&[0.0; 800]);
        assert_eq!(f.len(), N_MELS * 4);
        let floor = (1e-9f64).ln() as f32;
        assert!(f.iter().all(|&v| v == floor));
        // NaN/inf samples are treated as silence rather than poisoning the frame.
        let mut x = vec![0.0f32; 800];
        x[10] = f32::NAN;
        x[20] = f32::INFINITY;
        assert!(LogMel::new().compute(&x).iter().all(|&v| v == floor));
    }

    #[test]
    fn padding_noise_is_quiet_and_deterministic() {
        let a: Vec<f32> = quiet_noise(4800, PAD_NOISE_STD).collect();
        let b: Vec<f32> = quiet_noise(4800, PAD_NOISE_STD).collect();
        assert_eq!(a, b);
        let rms = (a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32).sqrt();
        assert!((rms / PAD_NOISE_STD - 1.0).abs() < 0.05, "rms {rms}");
        let mean = a.iter().sum::<f32>() / a.len() as f32;
        assert!(mean.abs() < PAD_NOISE_STD / 10.0);
        // About −11 in log-mel units: far above the −20.7 floor that zeros give.
        let f = LogMel::new().compute(&a);
        let avg = f.iter().sum::<f32>() / f.len() as f32;
        assert!((-13.0..-9.0).contains(&avg), "mean log-mel {avg}");
    }

    #[test]
    fn tokens_parse_space_and_blank() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("tokens.txt");
        std::fs::write(&p, "  0\n' 1\nа 2\nб 3\n<blk> 4\n").unwrap();
        let (tokens, blank) = read_tokens(&p).unwrap();
        assert_eq!(tokens, [" ", "'", "а", "б", "<blk>"]);
        assert_eq!(blank, 4);
        // A gap in the ids is an error.
        std::fs::write(&p, "  0\nа 2\n").unwrap();
        assert!(read_tokens(&p).is_err());
    }

    #[test]
    fn greedy_ctc_merges_repeats_and_drops_blanks() {
        let tokens: Vec<String> = [" ", "а", "б", "<blk>"].map(String::from).to_vec();
        let one_hot = |ids: &[usize]| -> Vec<f32> {
            ids.iter()
                .flat_map(|&i| (0..4).map(move |k| if k == i { 0.0 } else { -5.0 }))
                .collect()
        };
        // " " а а <blk> а б б " " " " б <blk> " "  →  " ааб б ", trimmed.
        let lp = one_hot(&[0, 1, 1, 3, 1, 2, 2, 0, 0, 2, 3, 0]);
        assert_eq!(ctc_greedy(&lp, 4, &tokens, 3), "ааб б");
        assert_eq!(ctc_greedy(&one_hot(&[3, 3, 0, 3]), 4, &tokens, 3), "");
    }
}
