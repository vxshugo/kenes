//! Recognizer backends (sherpa-onnx, and GigaAM on ONNX Runtime in `gigaam.rs`), backend
//! selection, and the sherpa-onnx Silero VAD.

use std::path::Path;
use std::str::FromStr;

use anyhow::{anyhow, bail, Result};
use kenes_types::SAMPLE_RATE;
use serde::{Deserialize, Serialize};
use sherpa_onnx::{
    OfflineNemoEncDecCtcModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};

use crate::gigaam::GigaamOrt;
use crate::registry::{ModelKind, ModelSpec};
use crate::segmenter::{Vad, VadParams, WINDOW};

/// Silence appended to every utterance before decoding (0.3 s).
pub(crate) const TAIL_PADDING: usize = SAMPLE_RATE as usize * 3 / 10;

/// Environment variable that overrides [`SttBackend::Auto`]: `ort` or `sherpa`.
pub const BACKEND_ENV: &str = "KENES_STT_BACKEND";

/// Turns a complete utterance (16 kHz mono) into text.
pub(crate) trait Recognizer: Send {
    /// Decode exactly `samples`.
    fn decode(&mut self, samples: &[f32]) -> Result<String>;

    /// Decode one utterance. GigaAM drops the last characters when audio ends right after
    /// the last phoneme ("марокко" → "маро"), so 0.3 s of silence is appended first.
    fn recognize(&mut self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        let mut padded = Vec::with_capacity(samples.len() + TAIL_PADDING);
        padded.extend_from_slice(samples);
        padded.resize(samples.len() + TAIL_PADDING, 0.0);
        self.decode(&padded)
    }
}

/// Which engine runs the speech recognition model.
///
/// Serialized in lowercase (`"auto"`, `"ort"`, `"sherpa"`) for the `sttBackend` setting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SttBackend {
    /// `$KENES_STT_BACKEND` if set, else the model's preferred backend (ONNX Runtime for all
    /// GigaAM models). Falls back to sherpa-onnx with a warning if that fails to load.
    #[default]
    Auto,
    /// ONNX Runtime with GigaAM's own 20 ms log-mel front-end: more accurate on
    /// GigaAM-v3/Multilingual and about twice as fast. Falls back to sherpa-onnx with a
    /// warning if the model can't be loaded this way.
    #[serde(alias = "onnxruntime")]
    Ort,
    /// sherpa-onnx's `OfflineRecognizer`, with its 25 ms kaldi fbank front-end.
    #[serde(alias = "sherpa-onnx")]
    Sherpa,
}

impl SttBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            SttBackend::Auto => "auto",
            SttBackend::Ort => "ort",
            SttBackend::Sherpa => "sherpa",
        }
    }
}

impl std::fmt::Display for SttBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for SttBackend {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(SttBackend::Auto),
            "ort" | "onnxruntime" => Ok(SttBackend::Ort),
            "sherpa" | "sherpa-onnx" => Ok(SttBackend::Sherpa),
            other => bail!("unknown STT backend {other:?}; expected auto, ort or sherpa"),
        }
    }
}

/// Resolve `Auto` to a concrete backend: explicit choices pass through, then the value of
/// `$KENES_STT_BACKEND` (`env`), then the model's preferred backend.
fn resolve_backend(spec: &ModelSpec, requested: SttBackend, env: Option<&str>) -> SttBackend {
    if requested != SttBackend::Auto {
        return requested;
    }
    let from_env = env.and_then(|v| {
        v.parse::<SttBackend>()
            .map_err(|e| log::warn!("ignoring {BACKEND_ENV}: {e}"))
            .ok()
    });
    match from_env {
        Some(b) if b != SttBackend::Auto => b,
        _ => spec.backend,
    }
}

type Loaded = Result<Box<dyn Recognizer>>;

/// Load the recognizer for `spec` from `dir` with the requested backend. An ONNX Runtime
/// backend that fails to load falls back to sherpa-onnx with a warning. Returns the backend
/// actually in use (never `Auto`).
pub(crate) fn load_recognizer(
    spec: &ModelSpec,
    dir: &Path,
    num_threads: i32,
    requested: SttBackend,
) -> Result<(Box<dyn Recognizer>, SttBackend)> {
    if spec.kind == ModelKind::Vad {
        bail!("{} is not a speech recognition model", spec.id);
    }
    check_files(spec, dir)?;
    let env = std::env::var(BACKEND_ENV).ok();
    load_with(
        spec,
        resolve_backend(spec, requested, env.as_deref()),
        || {
            let (model, tokens) = (dir.join("model.int8.onnx"), dir.join("tokens.txt"));
            Ok(Box::new(GigaamOrt::new(
                &model,
                &tokens,
                num_threads.max(1) as usize,
            )?))
        },
        || Ok(Box::new(SherpaRecognizer::new(spec, dir, num_threads)?)),
    )
}

fn load_with(
    spec: &ModelSpec,
    backend: SttBackend,
    ort: impl FnOnce() -> Loaded,
    sherpa: impl FnOnce() -> Loaded,
) -> Result<(Box<dyn Recognizer>, SttBackend)> {
    if backend == SttBackend::Ort {
        match ort() {
            Ok(r) => return Ok((r, SttBackend::Ort)),
            Err(e) => log::warn!(
                "can't run {} on ONNX Runtime ({e:#}); falling back to sherpa-onnx",
                spec.id
            ),
        }
    }
    Ok((sherpa()?, SttBackend::Sherpa))
}

/// sherpa-onnx exits the process on some bad inputs instead of returning an error, so
/// check the files ourselves first.
fn check_files(spec: &ModelSpec, dir: &Path) -> Result<()> {
    for f in spec.files {
        let p = dir.join(f.name);
        let len = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
        if len != f.size {
            bail!(
                "model {} is not downloaded ({} missing or incomplete); call ensure_model first",
                spec.id,
                p.display()
            );
        }
    }
    Ok(())
}

pub(crate) struct SherpaRecognizer {
    inner: OfflineRecognizer,
}

impl SherpaRecognizer {
    pub fn new(spec: &ModelSpec, dir: &Path, num_threads: i32) -> Result<Self> {
        check_files(spec, dir)?;
        let path = |name: &str| Some(dir.join(name).to_string_lossy().into_owned());
        let mut config = OfflineRecognizerConfig::default();
        match spec.kind {
            ModelKind::NemoCtc => {
                config.model_config.nemo_ctc = OfflineNemoEncDecCtcModelConfig {
                    model: path("model.int8.onnx"),
                };
                config.model_config.tokens = path("tokens.txt");
            }
            ModelKind::Vad => bail!("{} is not a speech recognition model", spec.id),
        }
        config.model_config.num_threads = num_threads.max(1);
        config.model_config.provider = Some(cpu_provider(dir.parent().unwrap_or(dir)));
        config.decoding_method = Some("greedy_search".into());
        let inner = OfflineRecognizer::create(&config).ok_or_else(|| {
            anyhow!(
                "sherpa-onnx failed to load model {} from {}",
                spec.id,
                dir.display()
            )
        })?;
        Ok(SherpaRecognizer { inner })
    }
}

/// ONNX Runtime settings passed through sherpa-onnx's provider config file
/// (`cpu:<path>`). By default ORT's worker threads spin after each op, which
/// here tripled CPU time for the same decodes; a meeting app idles most of
/// the time, so turn spinning off.
const ORT_CONFIG: &str = "\
# Written by kenes-stt; ONNX Runtime session options for sherpa-onnx.
SessionConfig.session.intra_op.allow_spinning=0
SessionConfig.session.inter_op.allow_spinning=0
";

/// `cpu:<config>` with the config above written next to the models, or
/// plain `cpu` if that isn't possible (sherpa-onnx splits the provider
/// string at the first `:`, so the path must not contain one).
fn cpu_provider(models_dir: &Path) -> String {
    let path = models_dir.join("ort-cpu.cfg");
    let write = || -> std::io::Result<()> {
        if std::fs::read_to_string(&path).ok().as_deref() != Some(ORT_CONFIG) {
            let tmp = path.with_extension("cfg.tmp");
            std::fs::write(&tmp, ORT_CONFIG)?;
            std::fs::rename(&tmp, &path)?;
        }
        Ok(())
    };
    let path_str = path.to_string_lossy();
    match write() {
        Ok(()) if !path_str.contains(':') => format!("cpu:{path_str}"),
        Ok(()) => "cpu".into(),
        Err(e) => {
            log::warn!("can't write {}: {e}; ORT threads will spin", path.display());
            "cpu".into()
        }
    }
}

impl Recognizer for SherpaRecognizer {
    fn decode(&mut self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        let stream = self.inner.create_stream();
        stream.accept_waveform(SAMPLE_RATE as i32, samples);
        self.inner.decode(&stream);
        let result = stream
            .get_result()
            .ok_or_else(|| anyhow!("sherpa-onnx returned no result"))?;
        Ok(result.text)
    }
}

pub(crate) struct SileroVad {
    inner: VoiceActivityDetector,
}

impl SileroVad {
    pub fn new(model: &Path, p: &VadParams) -> Result<Self> {
        if !model.is_file() {
            bail!(
                "VAD model {} is missing; call ensure_model first",
                model.display()
            );
        }
        let config = VadModelConfig {
            silero_vad: SileroVadModelConfig {
                model: Some(model.to_string_lossy().into_owned()),
                threshold: p.threshold,
                min_silence_duration: p.min_silence_s,
                min_speech_duration: p.min_speech_s,
                window_size: WINDOW as i32,
                max_speech_duration: p.soft_max_s,
            },
            sample_rate: SAMPLE_RATE as i32,
            num_threads: 1,
            provider: Some("cpu".into()),
            debug: false,
            ..Default::default()
        };
        // Buffer size only sets the initial capacity; it grows if needed.
        let inner = VoiceActivityDetector::create(&config, 30.0)
            .ok_or_else(|| anyhow!("sherpa-onnx failed to load VAD {}", model.display()))?;
        Ok(SileroVad { inner })
    }
}

impl Vad for SileroVad {
    fn accept_window(&mut self, window: &[f32]) {
        self.inner.accept_waveform(window);
        // We cut utterances ourselves; drop the VAD's queued copies.
        self.inner.clear();
    }

    fn is_speech(&self) -> bool {
        self.inner.detected()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{spec, DEFAULT_MODEL};

    struct Named(&'static str);
    impl Recognizer for Named {
        fn decode(&mut self, _: &[f32]) -> Result<String> {
            Ok(self.0.to_string())
        }
    }
    fn ok(name: &'static str) -> Loaded {
        Ok(Box::new(Named(name)))
    }

    #[test]
    fn backend_names_parse_and_serialize() {
        for (s, b) in [
            ("auto", SttBackend::Auto),
            ("", SttBackend::Auto),
            ("ort", SttBackend::Ort),
            ("ONNXRuntime", SttBackend::Ort),
            ("sherpa", SttBackend::Sherpa),
            ("sherpa-onnx", SttBackend::Sherpa),
        ] {
            assert_eq!(s.parse::<SttBackend>().unwrap(), b, "{s}");
        }
        assert!("tensorflow".parse::<SttBackend>().is_err());
        assert_eq!(SttBackend::default(), SttBackend::Auto);
        for b in [SttBackend::Auto, SttBackend::Ort, SttBackend::Sherpa] {
            let json = serde_json::to_string(&b).unwrap();
            assert_eq!(json, format!("\"{}\"", b.as_str()));
            assert_eq!(serde_json::from_str::<SttBackend>(&json).unwrap(), b);
            assert_eq!(b.as_str().parse::<SttBackend>().unwrap(), b);
        }
        assert_eq!(
            serde_json::from_str::<SttBackend>("\"sherpa-onnx\"").unwrap(),
            SttBackend::Sherpa
        );
    }

    #[test]
    fn auto_resolves_to_env_then_model_preference() {
        let m = spec(DEFAULT_MODEL).unwrap();
        assert_eq!(m.backend, SttBackend::Ort);
        assert_eq!(resolve_backend(m, SttBackend::Auto, None), SttBackend::Ort);
        assert_eq!(
            resolve_backend(m, SttBackend::Auto, Some("sherpa")),
            SttBackend::Sherpa
        );
        assert_eq!(
            resolve_backend(m, SttBackend::Auto, Some("auto")),
            SttBackend::Ort
        );
        // A typo in the variable is ignored, not fatal.
        assert_eq!(
            resolve_backend(m, SttBackend::Auto, Some("sherpaa")),
            SttBackend::Ort
        );
        // An explicit choice beats the variable.
        assert_eq!(
            resolve_backend(m, SttBackend::Ort, Some("sherpa")),
            SttBackend::Ort
        );
        assert_eq!(
            resolve_backend(m, SttBackend::Sherpa, Some("ort")),
            SttBackend::Sherpa
        );
    }

    #[test]
    fn failed_ort_load_falls_back_to_sherpa() {
        let m = spec(DEFAULT_MODEL).unwrap();
        let decode = |(mut r, b): (Box<dyn Recognizer>, SttBackend)| (r.decode(&[]).unwrap(), b);

        let got = load_with(m, SttBackend::Ort, || ok("ort"), || ok("sherpa")).unwrap();
        assert_eq!(decode(got), ("ort".into(), SttBackend::Ort));

        let got = load_with(
            m,
            SttBackend::Ort,
            || Err(anyhow!("ONNX Runtime unavailable")),
            || ok("sherpa"),
        )
        .unwrap();
        assert_eq!(decode(got), ("sherpa".into(), SttBackend::Sherpa));

        // Sherpa requested: ORT is never tried.
        let got = load_with(
            m,
            SttBackend::Sherpa,
            || panic!("ORT must not be loaded"),
            || ok("sherpa"),
        )
        .unwrap();
        assert_eq!(decode(got), ("sherpa".into(), SttBackend::Sherpa));

        // Both failing is an error.
        assert!(load_with(
            m,
            SttBackend::Ort,
            || Err(anyhow!("no ORT")),
            || Err(anyhow!("no sherpa"))
        )
        .is_err());
    }

    #[test]
    fn recognize_pads_and_skips_empty_input() {
        struct Len;
        impl Recognizer for Len {
            fn decode(&mut self, s: &[f32]) -> Result<String> {
                Ok(s.len().to_string())
            }
        }
        assert_eq!(Len.recognize(&[]).unwrap(), "");
        assert_eq!(
            Len.recognize(&[0.1; 100]).unwrap(),
            (100 + TAIL_PADDING).to_string()
        );
        assert_eq!(Len.decode(&[0.1; 100]).unwrap(), "100");
    }
}
