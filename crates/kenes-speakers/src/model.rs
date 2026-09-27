//! The pinned speaker embedding model and its download.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Sub-directory of the models dir that holds speaker models.
pub const SPEAKER_MODEL_DIR: &str = "speaker-embedding";

/// A downloadable model file pinned by URL, size and SHA-256.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeakerModel {
    /// File name inside `<models_dir>/speaker-embedding/`.
    pub file: &'static str,
    pub url: &'static str,
    /// Lowercase hex.
    pub sha256: &'static str,
    /// Bytes.
    pub size: u64,
}

/// The model [`ensure_speaker_model`] fetches and [`crate::ClusterConfig::default`] is tuned
/// for: 3D-Speaker CAM++ trained on ~200k Chinese speakers ("zh-cn common"), 192-dim
/// embeddings, 28 MB. Chosen by measurement on synthetic ru/kk meetings; see `EVAL.md`.
pub const SPEAKER_MODEL: SpeakerModel = SpeakerModel {
    file: "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
    sha256: "f682b514c05d947ee3fa91cd6ec6c5c7543479a128373fa29b1faedccd21fd11",
    size: 28_281_138,
};

/// Where [`SPEAKER_MODEL`] lives: `<models_dir>/speaker-embedding/<file>`.
pub fn speaker_model_path(models_dir: &Path) -> PathBuf {
    models_dir.join(SPEAKER_MODEL_DIR).join(SPEAKER_MODEL.file)
}

/// Whether [`SPEAKER_MODEL`] is on disk (by size; files only get their final name after the
/// hash was checked).
pub fn is_speaker_model_downloaded(models_dir: &Path) -> bool {
    std::fs::metadata(speaker_model_path(models_dir))
        .is_ok_and(|m| m.is_file() && m.len() == SPEAKER_MODEL.size)
}

/// Make sure the speaker model is in `<models_dir>/speaker-embedding/`, downloading it
/// (~28 MB, SHA-256 checked, resumable) if needed. Returns the model file's path.
/// `progress` gets `0.0..=1.0`. Blocking; call it off the UI thread. Cheap when the file is
/// already there (one `stat`).
pub fn ensure_speaker_model(models_dir: &Path, progress: &mut dyn FnMut(f32)) -> Result<PathBuf> {
    let dest = speaker_model_path(models_dir);
    if is_speaker_model_downloaded(models_dir) {
        progress(1.0);
        return Ok(dest);
    }
    log::info!(
        "downloading speaker model {} ({} bytes)",
        SPEAKER_MODEL.file,
        SPEAKER_MODEL.size
    );
    kenes_stt::download_verified(
        SPEAKER_MODEL.url,
        SPEAKER_MODEL.sha256,
        SPEAKER_MODEL.size,
        &dest,
        progress,
    )
    .with_context(|| format!("downloading speaker model {}", SPEAKER_MODEL.file))?;
    Ok(dest)
}
