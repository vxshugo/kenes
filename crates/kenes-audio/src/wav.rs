//! WAV helpers. Everything written is 16 kHz mono 16-bit PCM; anything hound can
//! read (8/16/24/32-bit int or 32-bit float, any rate, any channel count) is read
//! back as 16 kHz mono `f32`.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use kenes_types::SAMPLE_RATE;

use crate::pcm;

fn spec_16k_mono() -> hound::WavSpec {
    hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    }
}

/// Writes `samples` (16 kHz mono) to `path` as 16-bit PCM.
pub fn write(path: impl AsRef<Path>, samples: &[f32]) -> Result<()> {
    let mut w = WavWriter::create(path)?;
    w.write(samples)?;
    w.finalize()
}

/// Reads a WAV file and converts it to 16 kHz mono `f32`.
pub fn read(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let (samples, rate) = read_mono(path)?;
    Ok(pcm::resample(&samples, rate, SAMPLE_RATE))
}

/// Reads a WAV file as mono `f32` (channels averaged) at its native sample rate.
pub fn read_mono(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
    let path = path.as_ref();
    let reader =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels == 0 || spec.sample_rate == 0 {
        bail!("{}: invalid WAV header ({spec:?})", path.display());
    }
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<_, _>>()
            .with_context(|| format!("decoding {}", path.display()))?,
        hound::SampleFormat::Int => {
            if !(1..=32).contains(&spec.bits_per_sample) {
                bail!(
                    "{}: unsupported bit depth {}",
                    path.display(),
                    spec.bits_per_sample
                );
            }
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|s| s as f32 * scale))
                .collect::<Result<_, _>>()
                .with_context(|| format!("decoding {}", path.display()))?
        }
    };
    let mut mono = Vec::with_capacity(interleaved.len() / usize::from(spec.channels));
    pcm::downmix_into(&interleaved, usize::from(spec.channels), &mut mono);
    Ok((mono, spec.sample_rate))
}

/// Incremental 16 kHz mono 16-bit WAV writer.
///
/// Call [`WavWriter::flush`] now and then while recording: it rewrites the header,
/// so the file stays playable even if the process dies before [`WavWriter::finalize`].
pub struct WavWriter {
    inner: hound::WavWriter<BufWriter<File>>,
    path: PathBuf,
    len: u64,
}

impl WavWriter {
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let inner = hound::WavWriter::create(&path, spec_16k_mono())
            .with_context(|| format!("creating {}", path.display()))?;
        Ok(Self {
            inner,
            path,
            len: 0,
        })
    }

    pub fn write(&mut self, samples: &[f32]) -> Result<()> {
        let mut w = self.inner.get_i16_writer(samples.len() as u32);
        for &s in samples {
            w.write_sample(pcm::f32_to_i16(s));
        }
        w.flush()
            .with_context(|| format!("writing {}", self.path.display()))?;
        self.len += samples.len() as u64;
        Ok(())
    }

    /// Appends `n` samples of silence.
    pub fn write_silence(&mut self, n: usize) -> Result<()> {
        const ZEROS: [f32; 1024] = [0.0; 1024];
        let mut left = n;
        while left > 0 {
            let k = left.min(ZEROS.len());
            self.write(&ZEROS[..k])?;
            left -= k;
        }
        Ok(())
    }

    /// Samples written so far.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Flushes buffered data and updates the header.
    pub fn flush(&mut self) -> Result<()> {
        self.inner
            .flush()
            .with_context(|| format!("flushing {}", self.path.display()))
    }

    pub fn finalize(self) -> Result<()> {
        let path = self.path;
        self.inner
            .finalize()
            .with_context(|| format!("finalizing {}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcm::rms;
    use crate::pcm::tests::{sine, zero_crossing_freq};

    /// A per-process file in the temp dir, removed on drop.
    struct TmpFile(PathBuf);

    impl TmpFile {
        fn new(prefix: &str, name: &str) -> Self {
            let file = format!("{prefix}-{}-{name}", std::process::id());
            Self(std::env::temp_dir().join(file))
        }
    }

    impl std::ops::Deref for TmpFile {
        type Target = Path;
        fn deref(&self) -> &Path {
            &self.0
        }
    }

    impl AsRef<Path> for TmpFile {
        fn as_ref(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TmpFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn tmp(name: &str) -> TmpFile {
        TmpFile::new("kenes-audio-wav", name)
    }

    #[test]
    fn round_trip_16k_mono() {
        let path = tmp("rt.wav");
        let input = sine(1000.0, SAMPLE_RATE, 0.5, 0.5);
        write(&path, &input).unwrap();

        let spec = hound::WavReader::open(&path).unwrap().spec();
        assert_eq!(spec, spec_16k_mono());
        let back = read(&path).unwrap();
        assert_eq!(back.len(), input.len());
        for (a, b) in input.iter().zip(&back) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn writer_flushes_valid_header_and_writes_silence() {
        let path = tmp("incremental.wav");
        let mut w = WavWriter::create(&path).unwrap();
        w.write(&[0.25; 100]).unwrap();
        w.write_silence(3000).unwrap();
        w.flush().unwrap();
        assert_eq!(w.len(), 3100);
        // Readable before finalize: the header was updated by flush.
        assert_eq!(hound::WavReader::open(&path).unwrap().len(), 3100);
        w.write(&[0.5; 10]).unwrap();
        w.finalize().unwrap();
        let back = read(&path).unwrap();
        assert_eq!(back.len(), 3110);
        assert!((back[0] - 0.25).abs() < 1e-3);
        assert_eq!(back[100..3100].iter().filter(|&&s| s != 0.0).count(), 0);
    }

    #[test]
    fn reads_other_rates_and_channels() {
        // 44.1 kHz stereo, 24-bit: left = 1 kHz tone, right = the same tone.
        let path = tmp("stereo44.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 24,
            sample_format: hound::SampleFormat::Int,
        };
        let tone = sine(1000.0, 44_100, 1.0, 0.5);
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for &s in &tone {
            let v = (s * 8_388_607.0) as i32;
            w.write_sample(v).unwrap();
            w.write_sample(v).unwrap();
        }
        w.finalize().unwrap();

        let out = read(&path).unwrap();
        assert_eq!(out.len(), 16_000);
        let body = &out[100..out.len() - 100];
        assert!((zero_crossing_freq(body, 16_000) - 1000.0).abs() < 5.0);
        assert!((rms(body) - 0.5 / 2f32.sqrt()).abs() < 0.01);

        // 48 kHz float mono.
        let path = tmp("float48.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for s in sine(500.0, 48_000, 0.5, 0.25) {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();
        let (native, rate) = read_mono(&path).unwrap();
        assert_eq!((native.len(), rate), (24_000, 48_000));
        let out = read(&path).unwrap();
        assert_eq!(out.len(), 8_000);
        assert!((zero_crossing_freq(&out[100..7900], 16_000) - 500.0).abs() < 5.0);
    }

    #[test]
    fn missing_file_has_context() {
        let err = read("/nonexistent/kenes.wav").unwrap_err();
        assert!(format!("{err:#}").contains("/nonexistent/kenes.wav"));
    }
}
