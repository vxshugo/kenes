//! Dump GigaAM log-mel features ([`kenes_stt::LogMel`]) of 16 kHz mono WAV files as `.npy`
//! (`float32`, shape `(64, T)`), to compare with the Python reference:
//!
//! ```text
//! cargo run --release -p kenes-stt --example gigaam_features -- OUT_DIR a.wav b.wav …
//! uv run --project bench python crates/kenes-stt/scripts/compare_features.py OUT_DIR a.wav b.wav …
//! ```

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use kenes_stt::{LogMel, N_MELS};
use kenes_types::SAMPLE_RATE;

fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut r = hound::WavReader::open(path).with_context(|| format!("{}", path.display()))?;
    let spec = r.spec();
    if spec.sample_rate != SAMPLE_RATE || spec.channels != 1 {
        bail!("{}: need 16 kHz mono", path.display());
    }
    Ok(match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    })
}

/// Minimal NPY v1.0 writer for a C-order little-endian f32 matrix.
fn write_npy(path: &Path, rows: usize, cols: usize, data: &[f32]) -> Result<()> {
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    // Magic (6) + version (2) + length (2) + header + '\n' must be a multiple of 64.
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(args.next().context("usage: gigaam_features OUT_DIR WAV…")?);
    std::fs::create_dir_all(&out)?;
    let mut mel = LogMel::new();
    for wav in args.map(PathBuf::from) {
        let audio = read_wav(&wav)?;
        let feats = mel.compute(&audio);
        let t = LogMel::frames(audio.len());
        let dest = out
            .join(wav.file_stem().unwrap_or_default())
            .with_extension("npy");
        write_npy(&dest, N_MELS, t, &feats)?;
        println!("{} → {} ({N_MELS}×{t})", wav.display(), dest.display());
    }
    Ok(())
}
