//! sherpa-onnx backed implementations of the recognizer and VAD traits.

use std::path::Path;

use anyhow::{anyhow, bail, Result};
use kenes_types::SAMPLE_RATE;
use sherpa_onnx::{
    OfflineNemoEncDecCtcModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};

use crate::registry::{ModelKind, ModelSpec};
use crate::segmenter::{Vad, VadParams, WINDOW};

/// Zeros appended to every utterance before decoding (0.3 s).
const TAIL_PADDING: usize = SAMPLE_RATE as usize * 3 / 10;

/// Turns a complete utterance (16 kHz mono) into text.
pub(crate) trait Recognizer: Send {
    fn recognize(&mut self, samples: &[f32]) -> Result<String>;
}

pub(crate) struct SherpaRecognizer {
    inner: OfflineRecognizer,
}

impl SherpaRecognizer {
    pub fn new(spec: &ModelSpec, dir: &Path, num_threads: i32) -> Result<Self> {
        // sherpa-onnx exits the process on some bad inputs instead of
        // returning an error, so check the files ourselves first.
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
    fn recognize(&mut self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        let stream = self.inner.create_stream();
        // GigaAM drops the last characters when audio ends right after the
        // last phoneme ("марокко" → "маро"); a little trailing silence fixes it.
        let mut padded = Vec::with_capacity(samples.len() + TAIL_PADDING);
        padded.extend_from_slice(samples);
        padded.resize(samples.len() + TAIL_PADDING, 0.0);
        stream.accept_waveform(SAMPLE_RATE as i32, &padded);
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
