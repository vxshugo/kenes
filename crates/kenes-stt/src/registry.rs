//! Model registry and on-demand download.
//!
//! Every file is pinned by URL, size and SHA-256. A file is downloaded to
//! `<name>.part`, hashed while streaming, and renamed into place only after the
//! hash matches, so a file that exists under its final name with the right size
//! is trusted without re-hashing (hashing 600 MB on every launch costs seconds).

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::engine::SttBackend;

/// Id of the model used when the settings don't name one.
pub const DEFAULT_MODEL: &str = "gigaam-multilingual-ctc";
/// Id of the voice activity detector every ASR model needs.
pub const VAD_MODEL: &str = "silero-vad";

/// A model as the UI sees it (`list_models` in `docs/CONTRACT.md`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    /// ISO 639-1 codes the model was trained on.
    pub languages: Vec<String>,
    /// Download size in MB, including the VAD it depends on.
    pub size_mb: u64,
    /// All files (including the VAD) are present in [`models_dir()`].
    pub downloaded: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModelKind {
    /// Silero VAD, `files[0]` is the ONNX model.
    Vad,
    /// NeMo-style CTC model for sherpa-onnx: `model.int8.onnx` + `tokens.txt`.
    NemoCtc,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ModelFile {
    pub name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ModelSpec {
    pub id: &'static str,
    pub name: &'static str,
    pub languages: &'static [&'static str],
    pub kind: ModelKind,
    pub files: &'static [ModelFile],
    /// Other registry ids that must be present for this model to be usable.
    pub requires: &'static [&'static str],
    /// What [`SttBackend::Auto`] means for this model (never `Auto` itself).
    pub backend: SttBackend,
}

/// Community sherpa-onnx builds of GigaAM Multilingual (no official build yet).
macro_rules! bayaya {
    ($path:literal) => {
        concat!(
            "https://github.com/fgeeer77/bayaya-models/releases/download/",
            $path
        )
    };
}

/// All known models. ASR models first, in the order the UI should list them.
pub(crate) const REGISTRY: &[ModelSpec] = &[
    ModelSpec {
        id: "gigaam-multilingual-ctc",
        name: "GigaAM Multilingual CTC (220M, ru+kk)",
        languages: &["ru", "kk", "ky", "uz", "en"],
        kind: ModelKind::NemoCtc,
        files: &[
            ModelFile {
                name: "model.int8.onnx",
                url: bayaya!("gigaam-multilingual-ctc/model.int8.onnx"),
                sha256: "f66bff0186d649a2300da895e9f81d4ef8764519db2dc46429a44b883e90d105",
                size: 224_762_518,
            },
            ModelFile {
                name: "tokens.txt",
                url: bayaya!("gigaam-multilingual-ctc/tokens.txt"),
                sha256: "9b5df7987cb4ca52c1a468649ce897fab1cd182067416e29fef49dfaa7a856c2",
                size: 391,
            },
        ],
        requires: &[VAD_MODEL],
        // Trained on a 20 ms log-mel that sherpa-onnx 1.13.8 doesn't compute.
        backend: SttBackend::Ort,
    },
    ModelSpec {
        id: "gigaam-multilingual-large-ctc",
        name: "GigaAM Multilingual Large CTC (600M, ru+kk)",
        languages: &["ru", "kk", "ky", "uz", "en"],
        kind: ModelKind::NemoCtc,
        files: &[
            ModelFile {
                name: "model.int8.onnx",
                url: bayaya!("gigaam-multilingual-large-ctc/model.int8.onnx"),
                sha256: "7fdb9427c1c871407ecbde741fd7bb0479924981c89aa9f4241587bbcb085ae3",
                size: 591_645_636,
            },
            ModelFile {
                name: "tokens.txt",
                url: bayaya!("gigaam-multilingual-large-ctc/tokens.txt"),
                sha256: "9b5df7987cb4ca52c1a468649ce897fab1cd182067416e29fef49dfaa7a856c2",
                size: 391,
            },
        ],
        requires: &[VAD_MODEL],
        // Trained on a 20 ms log-mel that sherpa-onnx 1.13.8 doesn't compute.
        backend: SttBackend::Ort,
    },
    ModelSpec {
        id: "gigaam-v3-ru-ctc",
        name: "GigaAM v3 CTC (Russian only)",
        languages: &["ru"],
        kind: ModelKind::NemoCtc,
        files: &[
            ModelFile {
                name: "model.int8.onnx",
                url: "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16/resolve/32a4c7cc81809bd132e2d935ab99e9e6ab47fbec/model.int8.onnx",
                sha256: "f86ebfa0429ced91be6054fc344827e9c6c2572f3c318416cd974b06f66437ec",
                size: 224_721_476,
            },
            ModelFile {
                name: "tokens.txt",
                url: "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16/resolve/32a4c7cc81809bd132e2d935ab99e9e6ab47fbec/tokens.txt",
                sha256: "17cc514451bcceac9c280068c71502f8448f99e9fb1456b8d0761651fd0392f2",
                size: 196,
            },
        ],
        requires: &[VAD_MODEL],
        // Trained on a 20 ms log-mel that sherpa-onnx 1.13.8 doesn't compute.
        backend: SttBackend::Ort,
    },
    ModelSpec {
        id: VAD_MODEL,
        name: "Silero VAD (sherpa-onnx export)",
        languages: &[],
        kind: ModelKind::Vad,
        files: &[ModelFile {
            name: "silero_vad.onnx",
            url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
            sha256: "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
            size: 643_854,
        }],
        requires: &[],
        backend: SttBackend::Sherpa,
    },
];

pub(crate) fn spec(id: &str) -> Option<&'static ModelSpec> {
    REGISTRY.iter().find(|m| m.id == id)
}

/// Where models live: `$KENES_MODELS_DIR`, else `<data_dir>/kenes/models`
/// (`~/.local/share/kenes/models` on Linux,
/// `~/Library/Application Support/kenes/models` on macOS).
pub fn models_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("KENES_MODELS_DIR").filter(|d| !d.is_empty()) {
        return PathBuf::from(dir);
    }
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("kenes")
        .join("models")
}

/// ASR models the user can pick, with download status for [`models_dir()`].
pub fn available_models() -> Vec<ModelInfo> {
    available_models_in(&models_dir())
}

/// Like [`available_models`] for an explicit directory.
pub fn available_models_in(dir: &Path) -> Vec<ModelInfo> {
    REGISTRY
        .iter()
        .filter(|m| m.kind != ModelKind::Vad)
        .map(|m| {
            let closure = with_requirements(m);
            let bytes: u64 = closure.iter().flat_map(|s| s.files).map(|f| f.size).sum();
            ModelInfo {
                id: m.id.to_string(),
                name: m.name.to_string(),
                languages: m.languages.iter().map(|l| l.to_string()).collect(),
                size_mb: bytes.div_ceil(1_000_000),
                downloaded: closure.iter().all(|s| is_present(s, dir)),
            }
        })
        .collect()
}

/// Directory holding the files of model `id` inside `models_dir`.
pub fn model_path(id: &str, models_dir: &Path) -> PathBuf {
    models_dir.join(id)
}

/// Whether all files of `id` and of the models it requires are present.
pub fn is_downloaded(id: &str, models_dir: &Path) -> bool {
    spec(id).is_some_and(|m| {
        with_requirements(m)
            .iter()
            .all(|s| is_present(s, models_dir))
    })
}

/// Make sure model `id` (and the VAD it needs) is on disk, downloading what's
/// missing. Returns the model's directory. `progress` gets `0.0..=1.0` over
/// all bytes of the model, including files that were already present.
///
/// Blocking; run it off the UI thread. Safe to call when everything is
/// already downloaded (it only stats the files).
pub fn ensure_model(id: &str, models_dir: &Path, progress: &mut dyn FnMut(f32)) -> Result<PathBuf> {
    let m = spec(id).ok_or_else(|| {
        let known: Vec<_> = REGISTRY.iter().map(|m| m.id).collect();
        anyhow!("unknown model {id:?}; known models: {}", known.join(", "))
    })?;
    let closure = with_requirements(m);
    let total: u64 = closure.iter().flat_map(|s| s.files).map(|f| f.size).sum();
    let mut done: u64 = 0;
    progress(0.0);
    for s in &closure {
        let dir = model_path(s.id, models_dir);
        for f in s.files {
            let dest = dir.join(f.name);
            if file_ok(&dest, f) {
                done += f.size;
                continue;
            }
            fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
            log::info!("downloading {} {} ({} bytes)", s.id, f.name, f.size);
            let base = done;
            download_file(&f.remote(), &dest, &mut |bytes| {
                progress(((base + bytes) as f64 / total as f64) as f32)
            })
            .with_context(|| format!("downloading {} for model {}", f.url, s.id))?;
            done += f.size;
        }
    }
    progress(1.0);
    Ok(model_path(id, models_dir))
}

fn with_requirements(m: &'static ModelSpec) -> Vec<&'static ModelSpec> {
    let mut out = vec![m];
    for r in m.requires {
        if let Some(s) = spec(r) {
            out.push(s);
        }
    }
    out
}

fn is_present(s: &ModelSpec, models_dir: &Path) -> bool {
    let dir = model_path(s.id, models_dir);
    s.files.iter().all(|f| file_ok(&dir.join(f.name), f))
}

fn file_ok(path: &Path, f: &ModelFile) -> bool {
    fs::metadata(path).is_ok_and(|md| md.is_file() && md.len() == f.size)
}

fn part_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    dest.with_file_name(name)
}

/// Download `f` to `dest` via `dest.part`, resuming a previous partial
/// download when the server honours `Range`. `on_bytes` gets the number of
/// bytes of this file that are on disk so far.
/// Download `url` to `dest` unless `dest` already exists with this SHA-256.
///
/// Same machinery as the model registry: streams to `<dest>.part` (resuming a
/// previous partial download when the server supports `Range`), verifies size
/// and SHA-256, then renames into place, so `dest` never holds a partial or
/// corrupt file. `progress` gets `0.0..=1.0` for this file. Blocking. For
/// crates with their own models (e.g. speaker embeddings); `sha256` is
/// lowercase hex.
pub fn download_verified(
    url: &str,
    sha256: &str,
    size: u64,
    dest: &Path,
    progress: &mut dyn FnMut(f32),
) -> Result<()> {
    let sha256 = sha256.to_ascii_lowercase();
    if fs::metadata(dest).is_ok_and(|m| m.is_file() && m.len() == size)
        && sha256_file(dest)? == sha256
    {
        progress(1.0);
        return Ok(());
    }
    if let Some(dir) = dest.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let remote = Remote {
        url,
        sha256: &sha256,
        size,
    };
    progress(0.0);
    download_file(&remote, dest, &mut |bytes| {
        progress((bytes as f64 / size.max(1) as f64) as f32)
    })
    .with_context(|| format!("downloading {url}"))?;
    progress(1.0);
    Ok(())
}

/// What to fetch and how to check it.
#[derive(Clone, Copy, Debug)]
struct Remote<'a> {
    url: &'a str,
    sha256: &'a str,
    size: u64,
}

impl ModelFile {
    fn remote(&self) -> Remote<'static> {
        Remote {
            url: self.url,
            sha256: self.sha256,
            size: self.size,
        }
    }
}

fn download_file(f: &Remote, dest: &Path, on_bytes: &mut dyn FnMut(u64)) -> Result<()> {
    let part = part_path(dest);
    let mut last_err = None;
    // A couple of attempts: flaky Wi-Fi shouldn't cost a 600 MB restart.
    for attempt in 0..3 {
        if attempt > 0 {
            log::warn!("retrying download of {} (attempt {})", f.url, attempt + 1);
            std::thread::sleep(Duration::from_secs(2));
        }
        match download_attempt(f, &part, on_bytes) {
            Ok(()) => {
                fs::rename(&part, dest).with_context(|| {
                    format!("renaming {} to {}", part.display(), dest.display())
                })?;
                return Ok(());
            }
            Err(e) if e.downcast_ref::<HashMismatch>().is_some() => {
                let _ = fs::remove_file(&part);
                return Err(e);
            }
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow!("download failed")))
}

#[derive(Debug)]
struct HashMismatch {
    expected: String,
    actual: String,
}

impl std::fmt::Display for HashMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sha256 mismatch: expected {}, got {}",
            self.expected, self.actual
        )
    }
}

impl std::error::Error for HashMismatch {}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(20))
        .timeout_read(Duration::from_secs(60))
        .user_agent(concat!("kenes-stt/", env!("CARGO_PKG_VERSION")))
        .try_proxy_from_env(true)
        .build()
}

fn download_attempt(f: &Remote, part: &Path, on_bytes: &mut dyn FnMut(u64)) -> Result<()> {
    let mut hasher = Sha256::new();
    let mut have = fs::metadata(part).map(|m| m.len()).unwrap_or(0);
    if have > f.size {
        fs::remove_file(part)?;
        have = 0;
    }

    let mut req = agent().get(f.url);
    if have > 0 {
        req = req.set("Range", &format!("bytes={have}-"));
    }
    let resp = match req.call() {
        Ok(r) => r,
        // Range past the end: the .part is complete (or garbage); re-check below.
        Err(ureq::Error::Status(416, _)) if have == f.size => {
            return finish_existing(f, part, on_bytes);
        }
        Err(e) => return Err(e.into()),
    };

    let mut file = if resp.status() == 206 && have > 0 {
        // Resume: hash what we already have first.
        let mut existing = File::open(part)?;
        hash_reader(&mut existing, &mut hasher)?;
        let mut file = OpenOptions::new().append(true).open(part)?;
        file.seek(SeekFrom::End(0))?;
        file
    } else {
        have = 0;
        File::create(part)?
    };
    on_bytes(have);

    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 256 * 1024];
    let mut since_report = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        file.write_all(&buf[..n])?;
        have += n as u64;
        since_report += n as u64;
        if have > f.size {
            bail!("server sent more than the expected {} bytes", f.size);
        }
        if since_report >= 1 << 20 {
            since_report = 0;
            on_bytes(have);
        }
    }
    file.sync_all()?;
    drop(file);
    on_bytes(have);
    if have != f.size {
        bail!("incomplete download: got {have} of {} bytes", f.size);
    }
    check_hash(f, hasher)
}

fn finish_existing(f: &Remote, part: &Path, on_bytes: &mut dyn FnMut(u64)) -> Result<()> {
    let mut hasher = Sha256::new();
    hash_reader(&mut File::open(part)?, &mut hasher)?;
    on_bytes(f.size);
    check_hash(f, hasher)
}

fn hash_reader(r: &mut dyn Read, hasher: &mut Sha256) -> Result<()> {
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = r.read(&mut buf)?;
        if n == 0 {
            return Ok(());
        }
        hasher.update(&buf[..n]);
    }
}

fn check_hash(f: &Remote, hasher: Sha256) -> Result<()> {
    let actual = hex(&hasher.finalize());
    if actual != f.sha256 {
        return Err(HashMismatch {
            expected: f.sha256.to_string(),
            actual,
        }
        .into());
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// SHA-256 of a file on disk, hex encoded. Used by tests and `--verify`.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    hash_reader(&mut File::open(path)?, &mut hasher)?;
    Ok(hex(&hasher.finalize()))
}

/// Re-hash every file of `id` (and its requirements). Returns the files that
/// are missing or corrupt.
pub fn verify_model(id: &str, models_dir: &Path) -> Result<Vec<PathBuf>> {
    let m = spec(id).ok_or_else(|| anyhow!("unknown model {id:?}"))?;
    let mut bad = Vec::new();
    for s in with_requirements(m) {
        for f in s.files {
            let p = model_path(s.id, models_dir).join(f.name);
            if !file_ok(&p, f) || sha256_file(&p)? != f.sha256 {
                bad.push(p);
            }
        }
    }
    Ok(bad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn registry_is_consistent() {
        let mut ids = HashSet::new();
        for m in REGISTRY {
            assert!(ids.insert(m.id), "duplicate id {}", m.id);
            assert!(!m.files.is_empty(), "{} has no files", m.id);
            let mut names = HashSet::new();
            for f in m.files {
                assert!(names.insert(f.name), "{}: duplicate file {}", m.id, f.name);
                assert_eq!(f.sha256.len(), 64, "{}/{}: sha256 length", m.id, f.name);
                assert!(
                    f.sha256
                        .chars()
                        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                    "{}/{}: sha256 must be lowercase hex",
                    m.id,
                    f.name
                );
                assert!(f.size > 0);
                assert!(f.url.starts_with("https://"), "{}: {}", m.id, f.url);
                assert!(
                    f.url.ends_with(f.name),
                    "{}: url {} vs name {}",
                    m.id,
                    f.url,
                    f.name
                );
            }
            for r in m.requires {
                assert!(spec(r).is_some(), "{} requires unknown {}", m.id, r);
            }
            if m.kind == ModelKind::NemoCtc {
                assert!(names.contains("model.int8.onnx") && names.contains("tokens.txt"));
                assert!(m.requires.contains(&VAD_MODEL));
                assert!(!m.languages.is_empty());
            }
            assert_ne!(
                m.backend,
                SttBackend::Auto,
                "{}: backend must be concrete",
                m.id
            );
        }
        assert!(spec(DEFAULT_MODEL).is_some());
        assert!(spec(VAD_MODEL).is_some_and(|m| m.kind == ModelKind::Vad));
    }

    #[test]
    fn bayaya_macro_builds_urls() {
        assert_eq!(
            spec(DEFAULT_MODEL).unwrap().files[0].url,
            "https://github.com/fgeeer77/bayaya-models/releases/download/gigaam-multilingual-ctc/model.int8.onnx"
        );
    }

    #[test]
    fn available_models_hides_vad_and_reports_status() {
        let dir = tempfile::tempdir().unwrap();
        let list = available_models_in(dir.path());
        assert!(list.iter().all(|m| m.id != VAD_MODEL));
        assert_eq!(list[0].id, DEFAULT_MODEL);
        assert!(list.iter().all(|m| !m.downloaded));
        assert_eq!(list[0].size_mb, 226); // 224.8 MB model + VAD, rounded up

        // Fake the files with the right sizes: counts as downloaded.
        for id in [DEFAULT_MODEL, VAD_MODEL] {
            let s = spec(id).unwrap();
            let d = model_path(id, dir.path());
            fs::create_dir_all(&d).unwrap();
            for f in s.files {
                File::create(d.join(f.name))
                    .unwrap()
                    .set_len(f.size)
                    .unwrap();
            }
        }
        assert!(is_downloaded(DEFAULT_MODEL, dir.path()));
        let list = available_models_in(dir.path());
        assert!(
            list.iter()
                .find(|m| m.id == DEFAULT_MODEL)
                .unwrap()
                .downloaded
        );
        assert!(
            !list
                .iter()
                .find(|m| m.id == "gigaam-v3-ru-ctc")
                .unwrap()
                .downloaded
        );
        // …but the hashes are wrong.
        assert!(!verify_model(DEFAULT_MODEL, dir.path()).unwrap().is_empty());
    }

    #[test]
    fn model_info_wire_shape() {
        let info = &available_models_in(Path::new("/nonexistent"))[0];
        let json = serde_json::to_value(info).unwrap();
        assert_eq!(json["id"], DEFAULT_MODEL);
        assert!(json["sizeMb"].is_u64());
        assert_eq!(json["downloaded"], false);
        assert!(json["languages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|l| l == "kk"));
    }

    #[test]
    fn unknown_model_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = ensure_model("nope", dir.path(), &mut |_| {}).unwrap_err();
        assert!(err.to_string().contains("gigaam-multilingual-ctc"));
    }

    #[test]
    fn download_verified_is_a_noop_for_a_good_file() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("sub/x.onnx");
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        fs::write(&dest, b"hello").unwrap();
        let sha = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
        let mut last = 0.0;
        // Unreachable URL: must not be contacted.
        download_verified("https://invalid.invalid/x", sha, 5, &dest, &mut |p| {
            last = p
        })
        .unwrap();
        assert_eq!(last, 1.0);
        // Wrong hash: it tries to re-download (and fails on the bad URL).
        let bad = "0".repeat(64);
        assert!(
            download_verified("https://invalid.invalid/x", &bad, 5, &dest, &mut |_| {}).is_err()
        );
        assert_eq!(
            fs::read(&dest).unwrap(),
            b"hello",
            "existing file untouched on failure"
        );
    }

    #[test]
    fn models_dir_honours_env() {
        // Only assert the fallback shape; mutating env in parallel tests is racy.
        let d = models_dir();
        assert!(d.ends_with("kenes/models") || std::env::var_os("KENES_MODELS_DIR").is_some());
    }
}
