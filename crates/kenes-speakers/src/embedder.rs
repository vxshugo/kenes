//! Speaker embeddings with sherpa-onnx.

use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use kenes_types::SAMPLE_RATE;
use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig};

use crate::cluster::l2_normalize;

/// Shortest input [`Embedder::embed`] accepts (100 ms). Anything this short gives a
/// meaningless embedding anyway; callers should stay above [`crate::ClusterConfig::min_embed_ms`].
pub const MIN_EMBED_SAMPLES: usize = SAMPLE_RATE as usize / 10;

/// 16 kHz mono audio → L2-normalized speaker embedding.
///
/// `Send + Sync`: one instance can live on a worker thread or be shared behind a lock.
/// `embed` takes `&mut self` only to keep callers from running it concurrently on one
/// instance, which would just contend for the same ONNX Runtime thread pool.
pub struct Embedder {
    inner: SpeakerEmbeddingExtractor,
    dim: usize,
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder").field("dim", &self.dim).finish()
    }
}

impl Embedder {
    /// Load a sherpa-onnx speaker embedding model (e.g. the file returned by
    /// [`crate::ensure_speaker_model`]). `num_threads` is ONNX Runtime's intra-op thread
    /// count; 1–2 is plenty (see `EVAL.md`). Takes ~0.1–0.3 s including a warm-up run.
    pub fn new(model_path: &Path, num_threads: i32) -> Result<Self> {
        // sherpa-onnx aborts the process on some unreadable models instead of returning an
        // error, so check what we can first.
        let len = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);
        if len < 1024 {
            bail!(
                "speaker model {} is missing or truncated; call ensure_speaker_model first",
                model_path.display()
            );
        }
        let config = SpeakerEmbeddingExtractorConfig {
            model: Some(model_path.to_string_lossy().into_owned()),
            num_threads: num_threads.max(1),
            debug: false,
            provider: Some(cpu_provider(model_path.parent().unwrap_or(Path::new(".")))),
        };
        let t = Instant::now();
        let inner = SpeakerEmbeddingExtractor::create(&config).ok_or_else(|| {
            anyhow!(
                "sherpa-onnx failed to load speaker model {}",
                model_path.display()
            )
        })?;
        let dim = usize::try_from(inner.dim()).unwrap_or(0);
        if dim == 0 {
            bail!(
                "speaker model {} reports no embedding dimension",
                model_path.display()
            );
        }
        let mut e = Embedder { inner, dim };
        // The first ONNX Runtime run allocates and plans; pay for it now, not on the first
        // meeting segment.
        let warm: Vec<f32> = (0..SAMPLE_RATE as usize)
            .map(|i| 0.01 * ((i as f32) * 0.07).sin())
            .collect();
        e.embed(&warm)?;
        log::info!(
            "speaker model {} loaded in {} ms (dim {dim}, {} threads)",
            model_path.display(),
            t.elapsed().as_millis(),
            num_threads.max(1)
        );
        Ok(e)
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// 16 kHz mono → L2-normalized embedding. Fails on input shorter than
    /// [`MIN_EMBED_SAMPLES`] or silence; callers skip audio shorter than
    /// [`crate::ClusterConfig::min_embed_ms`] anyway. Cost is linear in length
    /// (see `EVAL.md` for numbers).
    pub fn embed(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        if samples.len() < MIN_EMBED_SAMPLES {
            bail!(
                "{} ms of audio is too short to embed",
                samples.len() * 1000 / SAMPLE_RATE as usize
            );
        }
        let stream = self
            .inner
            .create_stream()
            .ok_or_else(|| anyhow!("sherpa-onnx could not create a stream"))?;
        if samples.iter().all(|x| x.is_finite()) {
            stream.accept_waveform(SAMPLE_RATE as i32, samples);
        } else {
            let clean: Vec<f32> = samples
                .iter()
                .map(|&x| if x.is_finite() { x } else { 0.0 })
                .collect();
            stream.accept_waveform(SAMPLE_RATE as i32, &clean);
        }
        stream.input_finished();
        // compute() on a stream with no frames makes the C API hand back an empty buffer
        // that the Rust binding then reads `dim` floats from; never get there.
        if !self.inner.is_ready(&stream) {
            bail!("not enough audio for a speaker embedding");
        }
        let mut v = self
            .inner
            .compute(&stream)
            .ok_or_else(|| anyhow!("sherpa-onnx returned no embedding"))?;
        if v.len() != self.dim {
            bail!("embedding has {} values, expected {}", v.len(), self.dim);
        }
        if !l2_normalize(&mut v) {
            bail!("embedding is zero or not finite (silent input?)");
        }
        Ok(v)
    }
}

/// ONNX Runtime options passed via sherpa-onnx's `cpu:<config file>` provider string (same
/// trick as kenes-stt): without it ORT worker threads spin after every run, burning CPU
/// in an app that is mostly idle between segments. Without the memory arena, RSS after
/// 20 s inputs is ~90 MB instead of ~140 MB, at the same speed.
const ORT_CONFIG: &str = "\
# Written by kenes-speakers; ONNX Runtime session options for sherpa-onnx.
SessionConfig.session.intra_op.allow_spinning=0
SessionConfig.session.inter_op.allow_spinning=0
EnableCpuMemArena=0
";

fn cpu_provider(dir: &Path) -> String {
    let path = dir.join("ort-cpu.cfg");
    let write = || -> std::io::Result<()> {
        if std::fs::read_to_string(&path).ok().as_deref() != Some(ORT_CONFIG) {
            static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let tmp = path.with_extension(format!("cfg.{}.{n}.tmp", std::process::id()));
            std::fs::write(&tmp, ORT_CONFIG)?;
            std::fs::rename(&tmp, &path)?;
        }
        Ok(())
    };
    let path_str = path.to_string_lossy();
    // sherpa-onnx splits the provider string at the first ':'.
    match write() {
        Ok(()) if !path_str.contains(':') => format!("cpu:{path_str}"),
        Ok(()) => "cpu".into(),
        Err(e) => {
            log::warn!("can't write {}: {e}; ORT threads will spin", path.display());
            "cpu".into()
        }
    }
}
