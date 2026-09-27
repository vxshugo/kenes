//! Sample-level helpers: s16le decoding, levels, downmixing and resampling.

use std::f64::consts::PI;

/// Converts one signed 16-bit sample to `f32` in `[-1.0, 1.0)`.
#[inline]
pub fn i16_to_f32(s: i16) -> f32 {
    f32::from(s) / 32768.0
}

/// Converts an `f32` sample to `i16`, clamping out-of-range values.
#[inline]
pub fn f32_to_i16(s: f32) -> i16 {
    (s.clamp(-1.0, 1.0) * 32767.0).round() as i16
}

/// Streaming decoder for raw little-endian 16-bit PCM.
///
/// Reads from a pipe can end in the middle of a sample; the odd byte is kept
/// and joined with the first byte of the next call.
#[derive(Debug, Default)]
pub struct S16leDecoder {
    carry: Option<u8>,
}

impl S16leDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Decodes `bytes`, appending the samples to `out`.
    pub fn decode(&mut self, mut bytes: &[u8], out: &mut Vec<f32>) {
        if let Some(lo) = self.carry.take() {
            match bytes.split_first() {
                Some((&hi, rest)) => {
                    out.push(i16_to_f32(i16::from_le_bytes([lo, hi])));
                    bytes = rest;
                }
                None => {
                    self.carry = Some(lo);
                    return;
                }
            }
        }
        let (pairs, rest) = bytes.as_chunks::<2>();
        if let [odd] = rest {
            self.carry = Some(*odd);
        }
        out.extend(pairs.iter().map(|&p| i16_to_f32(i16::from_le_bytes(p))));
    }

    /// True if half a sample is waiting for its second byte.
    pub fn has_carry(&self) -> bool {
        self.carry.is_some()
    }
}

/// Root mean square of `samples` (0.0 for an empty slice).
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum / samples.len() as f64).sqrt() as f32
}

/// Converts a linear level to dBFS, floored at -100 dB.
pub fn to_dbfs(level: f32) -> f32 {
    if level <= 1e-5 {
        -100.0
    } else {
        20.0 * level.log10()
    }
}

/// Averages interleaved frames of `channels` samples into mono, appending to `out`.
pub fn downmix_into(interleaved: &[f32], channels: usize, out: &mut Vec<f32>) {
    if channels <= 1 {
        out.extend_from_slice(interleaved);
        return;
    }
    let scale = 1.0 / channels as f32;
    out.extend(
        interleaved
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() * scale),
    );
}

/// Resamples a whole buffer (mono) from `in_rate` to `out_rate`.
///
/// The result has exactly `ceil(len * out_rate / in_rate)` samples.
pub fn resample(input: &[f32], in_rate: u32, out_rate: u32) -> Vec<f32> {
    let mut rs = Resampler::new(in_rate, out_rate);
    let mut out = Vec::with_capacity(input.len() * out_rate as usize / in_rate as usize + 1);
    rs.process(input, &mut out);
    rs.flush(&mut out);
    out
}

/// Zero crossings of the sinc on each side of the kernel center.
const KERNEL_ZEROS: f64 = 16.0;
/// Cutoff as a fraction of the lower Nyquist frequency.
const ROLLOFF: f64 = 0.9;
/// Above this many distinct phases the filter taps are computed per output sample
/// instead of being precomputed.
const MAX_PHASES: u64 = 4096;

/// Streaming mono resampler: a Blackman-windowed sinc low-pass, evaluated at the
/// exact rational position of every output sample (polyphase).
///
/// Positions are tracked with integer arithmetic, so there is no drift over long
/// streams. When downsampling to 16 kHz the response is flat to 6 kHz, -3 dB at
/// 7 kHz, and aliases are attenuated by 55-65 dB, which is plenty for speech
/// recognition. Latency is `half_width` input samples (about 1 ms at 48 kHz).
#[derive(Debug, Clone)]
pub struct Resampler {
    /// Rates reduced by their GCD; `out_rate` is also the number of filter phases.
    in_rate: u64,
    out_rate: u64,
    cutoff: f64,
    half_width: i64,
    /// `out_rate` rows of `2 * half_width` normalized taps, or empty when too many phases.
    bank: Vec<f32>,
    scratch: Vec<f32>,
    buf: Vec<f32>,
    /// Absolute input index of `buf[0]`.
    buf_start: u64,
    /// Input samples received so far.
    n_in: u64,
    /// Index of the next output sample.
    next_out: u64,
}

impl Resampler {
    pub fn new(in_rate: u32, out_rate: u32) -> Self {
        assert!(in_rate > 0 && out_rate > 0, "sample rates must be positive");
        let g = gcd(in_rate, out_rate);
        let (in_rate, out_rate) = (u64::from(in_rate / g), u64::from(out_rate / g));
        let cutoff = in_rate.min(out_rate) as f64 / in_rate as f64 * ROLLOFF;
        let half_width = (KERNEL_ZEROS / cutoff).ceil() as i64;

        let mut rs = Self {
            in_rate,
            out_rate,
            cutoff,
            half_width,
            bank: Vec::new(),
            scratch: Vec::new(),
            buf: Vec::new(),
            buf_start: 0,
            n_in: 0,
            next_out: 0,
        };
        if !rs.is_passthrough() && out_rate <= MAX_PHASES {
            let mut bank = Vec::with_capacity(out_rate as usize * rs.num_taps());
            for phase in 0..out_rate {
                rs.compute_taps(phase);
                bank.extend_from_slice(&rs.scratch);
            }
            rs.bank = bank;
        }
        rs
    }

    fn is_passthrough(&self) -> bool {
        self.in_rate == self.out_rate
    }

    fn num_taps(&self) -> usize {
        2 * self.half_width as usize
    }

    /// Fills `scratch` with the taps for input offsets `1 - half_width ..= half_width`
    /// around the output position `base + phase / out_rate`, normalized to unit DC gain.
    fn compute_taps(&mut self, phase: u64) {
        let frac = phase as f64 / self.out_rate as f64;
        let hw = self.half_width;
        self.scratch.clear();
        let mut sum = 0.0;
        for j in (1 - hw)..=hw {
            let tau = (j as f64 - frac).abs();
            let u = tau / hw as f64;
            let w = if u >= 1.0 {
                0.0
            } else {
                let window = 0.42 + 0.5 * (PI * u).cos() + 0.08 * (2.0 * PI * u).cos();
                self.cutoff * sinc(self.cutoff * tau) * window
            };
            sum += w;
            self.scratch.push(w as f32);
        }
        let norm = (1.0 / sum) as f32;
        self.scratch.iter_mut().for_each(|t| *t *= norm);
    }

    /// Feeds `input` and appends every output sample that can be computed so far.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        self.n_in += input.len() as u64;
        if self.is_passthrough() {
            out.extend_from_slice(input);
            return;
        }
        self.buf.extend_from_slice(input);
        self.produce(self.n_in, u64::MAX, out);
        self.trim();
    }

    /// Emits the tail (treating the future as silence) and resets the resampler.
    pub fn flush(&mut self, out: &mut Vec<f32>) {
        if !self.is_passthrough() {
            let pad = self.half_width as usize + 1;
            self.buf.resize(self.buf.len() + pad, 0.0);
            let total = (self.n_in * self.out_rate).div_ceil(self.in_rate);
            self.produce(self.n_in + pad as u64, total, out);
        }
        self.buf.clear();
        self.buf_start = 0;
        self.n_in = 0;
        self.next_out = 0;
    }

    /// Computes outputs while their kernel window fits inside the first `available`
    /// input samples, stopping at output index `limit`.
    fn produce(&mut self, available: u64, limit: u64, out: &mut Vec<f32>) {
        let hw = self.half_width;
        let ntaps = self.num_taps();
        while self.next_out < limit {
            let pos = self.next_out * self.in_rate;
            let base = pos / self.out_rate;
            if base + hw as u64 >= available {
                break;
            }
            let phase = pos % self.out_rate;
            if self.bank.is_empty() {
                self.compute_taps(phase);
            }
            let taps = if self.bank.is_empty() {
                &self.scratch[..]
            } else {
                &self.bank[phase as usize * ntaps..][..ntaps]
            };
            // Input index of the first tap; samples before the stream start are silence.
            let first = base as i64 + 1 - hw;
            let skip = (-first).max(0) as usize;
            let start = (first + skip as i64) as u64 - self.buf_start;
            let input = &self.buf[start as usize..][..ntaps - skip];
            out.push(dot(input, &taps[skip..]));
            self.next_out += 1;
        }
    }

    /// Drops input that no future output sample needs.
    fn trim(&mut self) {
        let base = self.next_out * self.in_rate / self.out_rate;
        let keep_from = (base as i64 - self.half_width + 1).max(0) as u64;
        if keep_from > self.buf_start {
            let drop = ((keep_from - self.buf_start) as usize).min(self.buf.len());
            self.buf.drain(..drop);
            self.buf_start += drop as u64;
        }
    }
}

/// Dot product with 8 independent accumulators so the compiler can vectorize it.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let ((a8, a_rest), (b8, b_rest)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    let tail: f32 = a_rest.iter().zip(b_rest).map(|(x, y)| x * y).sum();
    for (x, y) in a8.iter().zip(b8) {
        for k in 0..8 {
            lanes[k] += x[k] * y[k];
        }
    }
    lanes.iter().sum::<f32>() + tail
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn sine(freq: f32, rate: u32, secs: f32, amp: f32) -> Vec<f32> {
        let n = (rate as f32 * secs) as usize;
        (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin())
            .collect()
    }

    /// Crude frequency estimate from zero crossings.
    pub(crate) fn zero_crossing_freq(samples: &[f32], rate: u32) -> f32 {
        let crossings = samples
            .windows(2)
            .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
            .count();
        crossings as f32 / 2.0 / (samples.len() as f32 / rate as f32)
    }

    #[test]
    fn s16le_basic_values() {
        let mut d = S16leDecoder::new();
        let mut out = Vec::new();
        let bytes: Vec<u8> = [0i16, 16384, -16384, i16::MAX, i16::MIN]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        d.decode(&bytes, &mut out);
        assert_eq!(out, vec![0.0, 0.5, -0.5, 32767.0 / 32768.0, -1.0]);
        assert!(!d.has_carry());
    }

    #[test]
    fn s16le_odd_byte_carry_over() {
        let samples: Vec<i16> = (0..257).map(|i| (i * 97 - 12000) as i16).collect();
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let expected: Vec<f32> = samples.iter().map(|&s| i16_to_f32(s)).collect();

        // Split the stream at every awkward size, including single bytes and empty reads.
        for sizes in [vec![1], vec![3], vec![1, 0, 2, 5], vec![7, 1, 1]] {
            let mut d = S16leDecoder::new();
            let mut out = Vec::new();
            let mut pos = 0;
            let mut k = 0;
            while pos < bytes.len() {
                let n = sizes[k % sizes.len()].min(bytes.len() - pos);
                d.decode(&bytes[pos..pos + n], &mut out);
                pos += n;
                k += 1;
            }
            assert_eq!(out, expected, "split sizes {sizes:?}");
            assert!(!d.has_carry());
        }

        let mut d = S16leDecoder::new();
        let mut out = Vec::new();
        d.decode(&[0x34], &mut out);
        assert!(out.is_empty() && d.has_carry());
        d.decode(&[], &mut out);
        assert!(out.is_empty() && d.has_carry());
        d.decode(&[0x12], &mut out);
        assert_eq!(out, vec![i16_to_f32(0x1234)]);
    }

    #[test]
    fn f32_i16_round_trip() {
        for s in [-32767i16, -1000, 0, 1, 1000, 32767] {
            assert_eq!(f32_to_i16(i16_to_f32(s) * 32768.0 / 32767.0), s);
        }
        assert_eq!(f32_to_i16(2.0), i16::MAX);
        assert_eq!(f32_to_i16(-2.0), -i16::MAX);
    }

    #[test]
    fn rms_and_dbfs() {
        assert_eq!(rms(&[]), 0.0);
        let s = sine(1000.0, 16_000, 1.0, 0.5);
        assert!((rms(&s) - 0.5 / 2f32.sqrt()).abs() < 1e-3);
        assert!((to_dbfs(1.0)).abs() < 1e-6);
        assert_eq!(to_dbfs(0.0), -100.0);
    }

    #[test]
    fn downmix_averages_channels() {
        let mut out = Vec::new();
        downmix_into(&[1.0, 0.0, 0.5, 0.5, -1.0, 1.0], 2, &mut out);
        assert_eq!(out, vec![0.5, 0.5, 0.0]);
        out.clear();
        downmix_into(&[0.1, 0.2], 1, &mut out);
        assert_eq!(out, vec![0.1, 0.2]);
    }

    #[test]
    fn resample_48k_to_16k_keeps_tone() {
        let input = sine(1000.0, 48_000, 1.0, 0.5);
        let out = resample(&input, 48_000, 16_000);
        assert_eq!(out.len(), 16_000);
        let body = &out[200..out.len() - 200];
        let f = zero_crossing_freq(body, 16_000);
        assert!((f - 1000.0).abs() < 5.0, "freq {f}");
        let r = rms(body);
        assert!((r - rms(&input)).abs() < 0.005, "rms {r}");
    }

    #[test]
    fn resample_rejects_aliases() {
        // 12 kHz is above the 8 kHz output Nyquist; without filtering it would fold to 4 kHz.
        let input = sine(12_000.0, 48_000, 0.5, 0.5);
        let out = resample(&input, 48_000, 16_000);
        let r = rms(&out[200..out.len() - 200]);
        assert!(r < 0.002, "alias rms {r}");
    }

    #[test]
    fn resample_44100_and_upsampling() {
        let out = resample(&sine(440.0, 44_100, 2.0, 0.3), 44_100, 16_000);
        assert_eq!(out.len(), 32_000);
        let f = zero_crossing_freq(&out[100..out.len() - 100], 16_000);
        assert!((f - 440.0).abs() < 3.0, "freq {f}");

        let out = resample(&sine(440.0, 8_000, 1.0, 0.3), 8_000, 16_000);
        assert_eq!(out.len(), 16_000);
        let body = &out[100..out.len() - 100];
        assert!((zero_crossing_freq(body, 16_000) - 440.0).abs() < 3.0);
        assert!((rms(body) - 0.3 / 2f32.sqrt()).abs() < 0.005);
    }

    #[test]
    fn resample_dc_gain_is_one() {
        let out = resample(&vec![0.5; 4800], 48_000, 16_000);
        for &s in &out[100..out.len() - 100] {
            assert!((s - 0.5).abs() < 1e-4, "{s}");
        }
    }

    #[test]
    fn streaming_matches_batch() {
        let input = sine(700.0, 44_100, 1.0, 0.4);
        let batch = resample(&input, 44_100, 16_000);

        let mut rs = Resampler::new(44_100, 16_000);
        let mut out = Vec::new();
        let mut pos = 0;
        for (i, n) in [1usize, 7, 441, 1000, 3, 4096].iter().cycle().enumerate() {
            if pos >= input.len() {
                break;
            }
            let n = (*n + i).min(input.len() - pos);
            rs.process(&input[pos..pos + n], &mut out);
            pos += n;
        }
        rs.flush(&mut out);
        assert_eq!(out.len(), batch.len());
        for (a, b) in out.iter().zip(&batch) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn odd_rates_without_phase_bank() {
        // 44 101 Hz reduces to 44101:16000, more phases than the precomputed bank allows.
        let rs = Resampler::new(44_101, 16_000);
        assert!(rs.bank.is_empty());
        let out = resample(&sine(1000.0, 44_101, 0.5, 0.5), 44_101, 16_000);
        assert_eq!(out.len(), 8_000);
        let body = &out[100..out.len() - 100];
        assert!((zero_crossing_freq(body, 16_000) - 1000.0).abs() < 5.0);
        assert!((rms(body) - 0.5 / 2f32.sqrt()).abs() < 0.005);
    }

    #[test]
    fn same_rate_is_passthrough() {
        let input = sine(300.0, 16_000, 0.1, 0.2);
        assert_eq!(resample(&input, 16_000, 16_000), input);
    }
}
