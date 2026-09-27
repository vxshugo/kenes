//! Synthetic echo scenes for evaluating echo cancellation (16 kHz mono).
//!
//! A scene is a short "call": the far end talks alone, the user talks alone, both talk at
//! once, and the far end talks alone again. The mic hears the user plus the far end played
//! through a laptop speaker: a speaker high-pass, optional soft clipping, a synthetic room
//! impulse response, a pure delay, optional clock drift, and background noise.

#![allow(dead_code)]

pub const SR: usize = 16_000;

/// splitmix64: small, deterministic, good enough for test signals.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    pub fn gauss(&mut self) -> f32 {
        let u1 = self.unit().max(1e-7);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len() as f64).sqrt() as f32
}

pub fn energy(x: &[f32]) -> f64 {
    x.iter().map(|&v| v as f64 * v as f64).sum()
}

pub fn db(x: f64) -> f64 {
    10.0 * x.max(1e-20).log10()
}

/// RMS over the 20 ms frames that are within 40 dB of the loudest one (speech, not pauses).
pub fn active_rms(x: &[f32]) -> f32 {
    let frames: Vec<f32> = x.chunks(320).map(rms).collect();
    let peak = frames.iter().cloned().fold(0.0f32, f32::max);
    let gate = peak * 0.01;
    let (mut e, mut n) = (0.0f64, 0usize);
    for f in x.chunks(320) {
        if rms(f) > gate {
            e += energy(f);
            n += f.len();
        }
    }
    if n == 0 {
        0.0
    } else {
        (e / n as f64).sqrt() as f32
    }
}

pub fn scale_to(x: &[f32], target_rms: f32) -> Vec<f32> {
    let r = active_rms(x).max(1e-9);
    x.iter().map(|&v| v * target_rms / r).collect()
}

/// Direct-form biquad (RBJ cookbook).
#[derive(Clone, Copy)]
pub struct Biquad {
    b: [f32; 3],
    a: [f32; 2],
    z: [f32; 2],
}

impl Biquad {
    pub fn highpass(f0: f32, q: f32) -> Self {
        let w = 2.0 * std::f32::consts::PI * f0 / SR as f32;
        let (s, c) = w.sin_cos();
        let alpha = s / (2.0 * q);
        let a0 = 1.0 + alpha;
        Biquad {
            b: [(1.0 + c) / 2.0 / a0, -(1.0 + c) / a0, (1.0 + c) / 2.0 / a0],
            a: [-2.0 * c / a0, (1.0 - alpha) / a0],
            z: [0.0; 2],
        }
    }
    pub fn peak(f0: f32, q: f32, gain_db: f32) -> Self {
        let a = 10f32.powf(gain_db / 40.0);
        let w = 2.0 * std::f32::consts::PI * f0 / SR as f32;
        let (s, c) = w.sin_cos();
        let alpha = s / (2.0 * q);
        let a0 = 1.0 + alpha / a;
        Biquad {
            b: [
                (1.0 + alpha * a) / a0,
                -2.0 * c / a0,
                (1.0 - alpha * a) / a0,
            ],
            a: [-2.0 * c / a0, (1.0 - alpha / a) / a0],
            z: [0.0; 2],
        }
    }
    pub fn lowpass(f0: f32, q: f32) -> Self {
        let w = 2.0 * std::f32::consts::PI * f0 / SR as f32;
        let (s, c) = w.sin_cos();
        let alpha = s / (2.0 * q);
        let a0 = 1.0 + alpha;
        Biquad {
            b: [(1.0 - c) / 2.0 / a0, (1.0 - c) / a0, (1.0 - c) / 2.0 / a0],
            a: [-2.0 * c / a0, (1.0 - alpha) / a0],
            z: [0.0; 2],
        }
    }
    pub fn run(&mut self, x: f32) -> f32 {
        // Transposed direct form II.
        let y = self.b[0] * x + self.z[0];
        self.z[0] = self.b[1] * x - self.a[0] * y + self.z[1];
        self.z[1] = self.b[2] * x - self.a[1] * y;
        y
    }
    pub fn apply(mut self, x: &mut [f32]) {
        for v in x {
            *v = self.run(*v);
        }
    }
}

/// Speaker → mic impulse response of a laptop in a room with the given RT60 (seconds):
/// a direct path, a few strong early reflections (desk, screen), an exponentially decaying
/// diffuse tail, and a small-speaker response (no bass, a resonance around 1–3 kHz).
pub fn laptop_rir(rng: &mut Rng, rt60: f32) -> Vec<f32> {
    let len = (rt60 * SR as f32) as usize;
    let mut h = vec![0.0f32; len];
    h[0] = 1.0;
    for _ in 0..6 {
        let t = (rng.range(0.001, 0.015) * SR as f32) as usize;
        h[t] += rng.range(-0.6, 0.6);
    }
    // Diffuse tail from ~5 ms: amplitude decays by 60 dB over rt60.
    let onset = SR / 200;
    let decay = 6.908 / (rt60 * SR as f32);
    let mut tail = vec![0.0f32; len];
    for (n, t) in tail.iter_mut().enumerate().skip(onset) {
        *t = rng.gauss() * (-decay * (n - onset) as f32).exp();
    }
    // Direct-to-reverberant ratio 0–8 dB: laptop speakers sit ~20 cm from the mic.
    let drr_db = rng.range(0.0, 8.0);
    let direct_e = energy(&h);
    let tail_e = energy(&tail);
    let g = ((direct_e / tail_e) as f32 * 10f32.powf(-drr_db / 10.0)).sqrt();
    for (a, b) in h.iter_mut().zip(&tail) {
        *a += g * b;
    }
    Biquad::highpass(rng.range(250.0, 450.0), 0.8).apply(&mut h);
    Biquad::peak(rng.range(1000.0, 3000.0), 1.5, rng.range(2.0, 6.0)).apply(&mut h);
    let n = energy(&h).sqrt() as f32;
    h.iter_mut().for_each(|v| *v /= n);
    h
}

/// Plain direct convolution, output as long as `x`.
pub fn convolve(x: &[f32], h: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    // Work over output blocks so the inner loop vectorizes.
    for (k, &hk) in h.iter().enumerate() {
        if hk == 0.0 {
            continue;
        }
        let (src, dst) = (&x[..x.len().saturating_sub(k)], &mut y[k..]);
        for (d, s) in dst.iter_mut().zip(src) {
            *d += hk * s;
        }
    }
    y
}

/// Laptop speakers compress loud peaks: `tanh`, driven so peaks lose about 3 dB.
pub fn soft_clip(x: &[f32]) -> Vec<f32> {
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    let a = 1.2 / peak;
    x.iter().map(|&v| (a * v).tanh() / a).collect()
}

/// Resamples by linear interpolation so that `ppm` > 0 means the mic clock runs fast
/// (the echo drifts later over time).
pub fn drift(x: &[f32], ppm: f32) -> Vec<f32> {
    if ppm == 0.0 {
        return x.to_vec();
    }
    let step = 1.0 / (1.0 + ppm as f64 * 1e-6);
    let n = x.len();
    (0..n)
        .map(|i| {
            let t = i as f64 * step;
            let j = t.floor() as usize;
            let f = (t - j as f64) as f32;
            if j + 1 < n {
                x[j] * (1.0 - f) + x[j + 1] * f
            } else {
                0.0
            }
        })
        .collect()
}

/// Background noise: pinkish (low-passed) Gaussian noise with a little hum.
pub fn noise(rng: &mut Rng, n: usize, level: f32) -> Vec<f32> {
    let mut v: Vec<f32> = (0..n).map(|_| rng.gauss()).collect();
    Biquad::lowpass(2500.0, 0.7).apply(&mut v);
    let r = rms(&v).max(1e-9);
    v.iter_mut().for_each(|s| *s *= level / r);
    v
}

pub struct EchoPath {
    pub rt60: f32,
    pub delay_ms: f32,
    /// Echo level relative to the near-end speech level, dB (negative = quieter).
    pub echo_db: f32,
    pub noise_db: f32,
    pub clip: bool,
    pub drift_ppm: f32,
    pub rir: Vec<f32>,
}

impl EchoPath {
    pub fn random(rng: &mut Rng) -> Self {
        let rt60 = rng.range(0.2, 0.5);
        let rir = laptop_rir(rng, rt60);
        EchoPath {
            rt60,
            delay_ms: rng.range(20.0, 250.0).round(),
            echo_db: rng.range(-20.0, -3.0).round(),
            noise_db: rng.range(-45.0, -35.0).round(),
            clip: rng.unit() < 0.5,
            drift_ppm: 0.0,
            rir,
        }
    }

    /// What the mic hears of `far` (already at the loopback level), before gain.
    pub fn render(&self, far: &[f32]) -> Vec<f32> {
        let driven = if self.clip {
            soft_clip(far)
        } else {
            far.to_vec()
        };
        let e = convolve(&driven, &self.rir);
        let e = drift(&e, self.drift_ppm);
        let d = (self.delay_ms * SR as f32 / 1000.0) as usize;
        let mut out = vec![0.0f32; far.len()];
        if d < far.len() {
            out[d..].copy_from_slice(&e[..far.len() - d]);
        }
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    FarOnly,
    NearOnly,
    Double,
}

#[derive(Clone, Debug)]
pub struct Region {
    pub kind: Kind,
    /// Sample span of the talk (for far-end regions extended by the echo tail).
    pub start: usize,
    pub end: usize,
    pub near_text: String,
    pub far_text: String,
}

pub struct Scene {
    pub mic: Vec<f32>,
    pub far: Vec<f32>,
    pub near: Vec<f32>,
    pub echo: Vec<f32>,
    pub regions: Vec<Region>,
}

pub struct Utt {
    pub name: String,
    pub samples: Vec<f32>,
    pub text: String,
}

/// Level of both the user's speech at the mic and the far end at the loopback: −26 dBFS.
pub const SPEECH_RMS: f32 = 0.05;

/// far-only (fe1), near-only (ne), double talk (dt_near + dt_far), far-only again (fe2).
pub fn build_scene(
    fe1: &Utt,
    ne: &Utt,
    dt_near: &Utt,
    dt_far: &Utt,
    fe2: &Utt,
    path: &EchoPath,
    rng: &mut Rng,
) -> Scene {
    let gap = SR; // 1 s between turns
    let tail = (path.delay_ms as usize * SR / 1000) + (path.rt60 * SR as f32) as usize;
    let lead = SR / 2;
    let dt_len = dt_near.samples.len().max(dt_far.samples.len() + SR / 2);
    let total = lead
        + fe1.samples.len()
        + gap
        + ne.samples.len()
        + gap
        + dt_len
        + gap
        + fe2.samples.len()
        + tail
        + lead;
    let mut far = vec![0.0f32; total];
    let mut near = vec![0.0f32; total];
    let put = |dst: &mut Vec<f32>, at: usize, u: &Utt| {
        let s = scale_to(&u.samples, SPEECH_RMS);
        for (d, v) in dst[at..].iter_mut().zip(s) {
            *d += v;
        }
    };
    let mut regions = Vec::new();
    let mut at = lead;
    put(&mut far, at, fe1);
    regions.push(Region {
        kind: Kind::FarOnly,
        start: at,
        end: at + fe1.samples.len() + tail,
        near_text: String::new(),
        far_text: fe1.text.clone(),
    });
    at += fe1.samples.len() + gap;
    put(&mut near, at, ne);
    regions.push(Region {
        kind: Kind::NearOnly,
        start: at,
        end: at + ne.samples.len(),
        near_text: ne.text.clone(),
        far_text: String::new(),
    });
    at += ne.samples.len() + gap;
    // The far end starts talking 0.5 s into the user's turn.
    put(&mut near, at, dt_near);
    put(&mut far, at + SR / 2, dt_far);
    regions.push(Region {
        kind: Kind::Double,
        start: at,
        end: at + dt_len,
        near_text: dt_near.text.clone(),
        far_text: dt_far.text.clone(),
    });
    at += dt_len + gap;
    put(&mut far, at, fe2);
    regions.push(Region {
        kind: Kind::FarOnly,
        start: at,
        end: (at + fe2.samples.len() + tail).min(total),
        near_text: String::new(),
        far_text: fe2.text.clone(),
    });

    let mut echo = path.render(&far);
    let far_active: Vec<f32> = echo.clone();
    let g = SPEECH_RMS * 10f32.powf(path.echo_db / 20.0) / active_rms(&far_active).max(1e-9);
    echo.iter_mut().for_each(|v| *v *= g);
    let n = noise(rng, total, SPEECH_RMS * 10f32.powf(path.noise_db / 20.0));
    let mic: Vec<f32> = (0..total)
        .map(|i| (near[i] + echo[i] + n[i]).clamp(-1.0, 1.0))
        .collect();
    Scene {
        mic,
        far,
        near,
        echo,
        regions,
    }
}

/// Scale-invariant SDR of `est` against `reference`, dB.
pub fn si_sdr(est: &[f32], reference: &[f32]) -> f64 {
    let n = est.len().min(reference.len());
    let (e, r) = (&est[..n], &reference[..n]);
    let me = e.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let mr = r.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let dot: f64 = e
        .iter()
        .zip(r)
        .map(|(&a, &b)| (a as f64 - me) * (b as f64 - mr))
        .sum();
    let rr: f64 = r.iter().map(|&b| (b as f64 - mr).powi(2)).sum();
    let alpha = dot / rr.max(1e-20);
    let (mut s, mut d) = (0.0f64, 0.0f64);
    for (&a, &b) in e.iter().zip(r) {
        let t = alpha * (b as f64 - mr);
        s += t * t;
        d += (a as f64 - me - t).powi(2);
    }
    db(s / d.max(1e-20))
}

/// Echo return loss enhancement over `[a, b)`: input power over output power, dB.
pub fn erle(mic: &[f32], out: &[f32], a: usize, b: usize) -> f64 {
    db(energy(&mic[a..b]) / energy(&out[a..b]).max(1e-20))
}

// ---- text ----

pub fn normalize(text: &str) -> Vec<String> {
    let lower = text.to_lowercase().replace('ё', "е");
    lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Word-level edit distance.
pub fn edit_distance(a: &[String], b: &[String]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let sub = prev[j - 1] + usize::from(a[i - 1] != b[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// A long call: [far-only, near-only, double talk] turns repeated until `seconds`.
pub fn build_long_scene(
    nears: &[Utt],
    fars: &[Utt],
    path: &EchoPath,
    seconds: usize,
    rng: &mut Rng,
) -> Scene {
    let total = seconds * SR;
    let mut far = vec![0.0f32; total];
    let mut near = vec![0.0f32; total];
    let put = |dst: &mut Vec<f32>, at: usize, u: &Utt| {
        let s = scale_to(&u.samples, SPEECH_RMS);
        for (d, v) in dst[at..].iter_mut().zip(s) {
            *d += v;
        }
    };
    let mut regions = Vec::new();
    let tail = SR / 2;
    let mut at = SR / 2;
    let (mut ni, mut fi) = (0, 0);
    loop {
        let fe = &fars[fi % fars.len()];
        let ne = &nears[ni % nears.len()];
        let dn = &nears[(ni + 1) % nears.len()];
        let df = &fars[(fi + 1) % fars.len()];
        let dt_len = dn.samples.len().max(df.samples.len() + SR / 2);
        let need = fe.samples.len() + ne.samples.len() + dt_len + 3 * SR + tail;
        if at + need >= total {
            break;
        }
        put(&mut far, at, fe);
        regions.push(Region {
            kind: Kind::FarOnly,
            start: at,
            end: at + fe.samples.len() + tail,
            near_text: String::new(),
            far_text: fe.text.clone(),
        });
        at += fe.samples.len() + SR;
        put(&mut near, at, ne);
        regions.push(Region {
            kind: Kind::NearOnly,
            start: at,
            end: at + ne.samples.len(),
            near_text: ne.text.clone(),
            far_text: String::new(),
        });
        at += ne.samples.len() + SR;
        put(&mut near, at, dn);
        put(&mut far, at + SR / 2, df);
        regions.push(Region {
            kind: Kind::Double,
            start: at,
            end: at + dt_len,
            near_text: dn.text.clone(),
            far_text: df.text.clone(),
        });
        at += dt_len + SR;
        ni += 2;
        fi += 2;
    }
    let mut echo = path.render(&far);
    let g = SPEECH_RMS * 10f32.powf(path.echo_db / 20.0) / active_rms(&echo).max(1e-9);
    echo.iter_mut().for_each(|v| *v *= g);
    let n = noise(rng, total, SPEECH_RMS * 10f32.powf(path.noise_db / 20.0));
    let mic: Vec<f32> = (0..total)
        .map(|i| (near[i] + echo[i] + n[i]).clamp(-1.0, 1.0))
        .collect();
    Scene {
        mic,
        far,
        near,
        echo,
        regions,
    }
}
