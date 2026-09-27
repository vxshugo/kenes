//! Acoustic echo cancellation for the microphone stream.
//!
//! Without headphones the laptop mic also hears the call through the speakers. The
//! system-audio stream is exactly what the speakers play, so it serves as the far-end
//! reference: [`EchoCanceller`] removes its echo from the mic with WebRTC's AEC3 (the
//! pure-Rust [`sonora`] port), and [`StreamCanceller`] pairs the two streams by their
//! position on the shared session timeline.
//!
//! On top of AEC3:
//! - a coarse delay tracker ([`AecConfig::track_delay`]) pre-delays the reference when the
//!   echo arrives later than AEC3's own ~450 ms search window (Bluetooth speakers, clock
//!   drift over long meetings);
//! - while the far end has been silent for [`AecConfig::hangover_ms`], the mic passes
//!   through untouched (only delayed by the canceller's fixed 18 ms, so the output
//!   stays continuous);
//! - when the far end plays but never reaches the mic (headphones), the mic also passes
//!   through: AEC3 would otherwise dent the user's speech whenever both talk. Echo
//!   turning up again switches cancelling back on within about a second of it.
//!
//! Everything here works on 16 kHz mono `f32` audio in 10 ms frames ([`FRAME`]).

mod delay;
mod stream;

use std::collections::VecDeque;

pub use stream::StreamCanceller;

/// Sample rate of everything in this crate (the pipeline's rate).
pub const SAMPLE_RATE: u32 = kenes_types::SAMPLE_RATE;
/// Processing frame: 10 ms at 16 kHz.
pub const FRAME: usize = 160;
/// The mic is delayed this much before AEC3, so the reference always leads its echo a
/// little: AEC3 copes badly with echo that is exactly simultaneous with the reference
/// (a zero delay on the timeline) and not at all with echo that comes first.
const NEAR_DELAY: usize = 160;
/// AEC3's own delay (its block framing).
const AEC3_LATENCY: usize = 128;
/// Output lags input by this much: [`NEAR_DELAY`] plus AEC3's own block framing (18 ms).
pub const LATENCY_SAMPLES: usize = NEAR_DELAY + AEC3_LATENCY;

const SAMPLES_PER_MS: usize = SAMPLE_RATE as usize / 1000;
/// AEC3 is left alone while the echo sits between these delays behind the (pre-delayed)
/// reference: its own search covers about 0–450 ms at 16 kHz.
const NATIVE_DELAY_MS: std::ops::Range<usize> = 30..400;
/// Where a re-alignment puts the echo when it had drifted late (room to drift further)…
const TARGET_LATE_MS: usize = 100;
/// …and when it had drifted early.
const TARGET_EARLY_MS: usize = 250;
/// At most one reference re-alignment per this many frames (5 s).
const REALIGN_COOLDOWN: u64 = 500;

#[derive(Clone, Debug, PartialEq)]
pub struct AecConfig {
    /// Far-end frames quieter than this (RMS, dBFS) count as silence.
    pub far_silence_dbfs: f32,
    /// Keep cancelling this long after the far end falls silent: covers the echo delay
    /// plus the room's reverb tail. After that the mic passes through untouched.
    pub hangover_ms: u32,
    /// Track echo delays beyond AEC3's own window and pre-delay the reference.
    pub track_delay: bool,
    /// Longest echo delay the tracker searches.
    pub max_delay_ms: u32,
    /// [`StreamCanceller`]: how far (in mic audio) a mic frame may wait for the system
    /// audio of the same moment before it is processed without it.
    pub max_wait_ms: u32,
}

impl Default for AecConfig {
    fn default() -> Self {
        Self {
            far_silence_dbfs: -60.0,
            hangover_ms: 1000,
            track_delay: true,
            max_delay_ms: 1000,
            max_wait_ms: 150,
        }
    }
}

/// What the canceller has been doing, for logs and the evaluation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AecStats {
    /// 10 ms frames processed.
    pub frames: u64,
    /// Frames whose output came from AEC3 (the rest passed through).
    pub active_frames: u64,
    /// Pre-delay currently applied to the reference, ms.
    pub reference_delay_ms: u32,
    /// How many times the reference was re-aligned.
    pub realignments: u32,
    /// Echo delay found by the coarse tracker, ms (if it found one).
    pub tracked_delay_ms: Option<u32>,
    /// AEC3's own delay estimate, ms, on top of `reference_delay_ms`.
    pub aec_delay_ms: Option<i32>,
    /// AEC3's echo return loss enhancement estimate, dB.
    pub erle_db: Option<f64>,
    /// Whether the far end currently reaches the mic (false: headphones, pass-through).
    pub echo_path: bool,
}

/// Echo canceller for one mic stream, fed one 10 ms frame pair at a time.
pub struct EchoCanceller {
    cfg: AecConfig,
    apm: sonora::AudioProcessing,
    /// Samples of output still owed by a replaced AEC3 instance (taken from the pass-through).
    fresh: usize,
    /// Reference pre-delay line; its length is the pre-delay.
    far_line: VecDeque<f32>,
    /// The mic on its way into AEC3 ([`NEAR_DELAY`]).
    near_line: VecDeque<f32>,
    /// The mic delayed like AEC3's output, for pass-through.
    raw_line: VecDeque<f32>,
    silence_power: f32,
    hangover_frames: u64,
    /// Frames since the (pre-delayed) reference last had sound; starts "long ago".
    since_far: u64,
    /// Output mix: 0 = pass-through, 1 = AEC3.
    mix: f32,
    tracker: Option<delay::DelayTracker>,
    last_realign: Option<u64>,
    coupling: Coupling,
    stats: AecStats,
}

/// Is the far end actually reaching the mic? With headphones it isn't, and AEC3 would
/// only dent the user's speech whenever both talk. Two kinds of frames tell:
/// - the far end plays loudly and the mic stays at its noise floor: no echo path;
/// - the mic goes in well above its floor and AEC3 brings it back down to it: echo
///   was there and got removed.
///
/// Frames with the user talking prove nothing either way and are ignored.
struct Coupling {
    /// Recent evidence, newest last: `true` = echo removed, `false` = mic silent.
    recent: VecDeque<bool>,
    removed: usize,
    coupled: bool,
    /// Mic noise floor (frame power), measured while the far end is quiet: follows
    /// drops at once, rises 1 dB per second. Unknown until the far end first pauses.
    floor: Option<f32>,
    smoothed: Option<f32>,
}

impl Coupling {
    /// Evidence frames kept (10 s).
    const WINDOW: usize = 1000;
    /// Evidence needed before deciding there is no echo path (3 s).
    const MIN_EVIDENCE: usize = 300;
    /// Below this share of echo frames in the window there is no echo path…
    const OFF_BELOW: f32 = 0.1;
    /// …and from this share among the newest `RECENT` there is one again.
    const ON_ABOVE: f32 = 0.5;
    const RECENT: usize = 200;
    /// Far-end frames this loud (−40 dBFS) would be audible on the mic through speakers.
    const LOUD_FAR: f32 = 1e-4;
    /// Mic frames quieter than this (−80 dBFS, digital silence) say nothing.
    const MIN_POWER: f32 = 1e-8;
    const FLOOR_RISE: f32 = 1.0023;

    fn new() -> Self {
        // Start by assuming speakers: cancelling for nothing costs less than leaking echo.
        Self {
            recent: VecDeque::with_capacity(Self::WINDOW + 1),
            removed: 0,
            coupled: true,
            floor: None,
            smoothed: None,
        }
    }

    fn coupled(&self) -> bool {
        self.coupled
    }

    fn power(x: &[f32]) -> f32 {
        x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32
    }

    /// Mic frames while the far end (and its echo) is quiet, for the noise floor.
    /// Digital silence (the start of the stream, a muted mic) is not a noise floor.
    fn track_floor(&mut self, raw: &[f32]) {
        let zeros = raw.iter().filter(|&&v| v == 0.0).count();
        let p = Self::power(raw);
        if p < Self::MIN_POWER || zeros * 4 > raw.len() {
            return;
        }
        // Smoothed over a few frames, so the minimum sits near the noise's mean.
        let p = self.smoothed.map_or(p, |s| 0.7 * s + 0.3 * p);
        self.smoothed = Some(p);
        self.floor = Some(match self.floor {
            Some(f) if p >= f => f * Self::FLOOR_RISE,
            _ => p,
        });
    }

    /// A frame while the far end plays (`far_power`): the mic before and after AEC3.
    fn observe(&mut self, far_power: f32, raw: &[f32], out: &[f32]) {
        let e_in = Self::power(raw);
        let Some(floor) = self.floor else { return };
        if e_in < Self::MIN_POWER {
            return;
        }
        let e_out = Self::power(out);
        let evidence = if e_in > 16.0 * floor && e_out < 2.0 * floor {
            true // 12 dB above the floor in, within 3 dB of it out
        } else if far_power > Self::LOUD_FAR && e_in < 4.0 * floor {
            false // within 6 dB of the floor: noise only
        } else {
            return;
        };
        self.recent.push_back(evidence);
        self.removed += usize::from(evidence);
        if self.recent.len() > Self::WINDOW && self.recent.pop_front() == Some(true) {
            self.removed -= 1;
        }
        let was = self.coupled;
        let n = self.recent.len();
        if self.coupled {
            self.coupled =
                !(n >= Self::MIN_EVIDENCE && (self.removed as f32) < Self::OFF_BELOW * n as f32);
        } else {
            let newest = self
                .recent
                .iter()
                .rev()
                .take(Self::RECENT)
                .filter(|&&e| e)
                .count();
            self.coupled = newest as f32 >= Self::ON_ABOVE * Self::RECENT as f32;
        }
        if was != self.coupled {
            log::info!(
                "echo path {}",
                if self.coupled {
                    "found: cancelling"
                } else {
                    "not found (headphones?): mic passes through"
                }
            );
        }
    }
}

impl EchoCanceller {
    pub fn new(cfg: AecConfig) -> Self {
        let silence = 10f32.powf(cfg.far_silence_dbfs / 10.0);
        let hangover_frames = u64::from(cfg.hangover_ms) * SAMPLES_PER_MS as u64 / FRAME as u64;
        let tracker = cfg
            .track_delay
            .then(|| delay::DelayTracker::new(cfg.max_delay_ms as usize * SAMPLES_PER_MS / FRAME));
        Self {
            cfg,
            apm: new_aec3(),
            fresh: 0,
            far_line: VecDeque::new(),
            near_line: std::iter::repeat_n(0.0, NEAR_DELAY).collect(),
            raw_line: std::iter::repeat_n(0.0, LATENCY_SAMPLES).collect(),
            silence_power: silence,
            hangover_frames,
            since_far: u64::MAX / 2,
            mix: 0.0,
            tracker,
            last_realign: None,
            coupling: Coupling::new(),
            stats: AecStats::default(),
        }
    }

    pub fn config(&self) -> &AecConfig {
        &self.cfg
    }

    /// Cancels the echo of `far` from `near`, in place. Both are one [`FRAME`] from the
    /// same moment on the shared timeline. The output lags the input by [`LATENCY_SAMPLES`].
    pub fn process_frame(&mut self, far: &[f32], near: &mut [f32]) {
        assert_eq!(far.len(), FRAME, "far frame must be {FRAME} samples");
        assert_eq!(near.len(), FRAME, "near frame must be {FRAME} samples");
        let frame = self.stats.frames;
        self.stats.frames += 1;

        if let Some(t) = &mut self.tracker {
            if t.push(far, near) {
                self.realign(frame);
            }
        }

        let mut far_now = [0.0f32; FRAME];
        if self.far_line.is_empty() {
            far_now.copy_from_slice(far);
        } else {
            self.far_line.extend(far);
            for v in far_now.iter_mut() {
                *v = self.far_line.pop_front().unwrap_or(0.0);
            }
        }

        let mut unused = [0.0f32; FRAME];
        let _ = self.apm.process_render_f32(&[&far_now], &mut [&mut unused]);
        self.near_line.extend(near.iter());
        let mut delayed = [0.0f32; FRAME];
        for v in delayed.iter_mut() {
            *v = self.near_line.pop_front().unwrap_or(0.0);
        }
        let mut out = [0.0f32; FRAME];
        let _ = self.apm.process_capture_f32(&[&delayed], &mut [&mut out]);

        self.raw_line.extend(near.iter());
        let mut raw = [0.0f32; FRAME];
        for v in raw.iter_mut() {
            *v = self.raw_line.pop_front().unwrap_or(0.0);
        }
        if self.fresh > 0 {
            // A new AEC3 instance starts with its latency buffer empty.
            let n = self.fresh.min(FRAME);
            out[..n].copy_from_slice(&raw[..n]);
            self.fresh -= n;
        }

        let power = far_now.iter().map(|v| v * v).sum::<f32>() / FRAME as f32;
        let far_active = power > self.silence_power;
        self.since_far = if far_active {
            0
        } else {
            self.since_far.saturating_add(1)
        };
        if self.since_far > self.hangover_frames {
            self.coupling.track_floor(&raw);
        }
        if far_active {
            self.coupling.observe(power, &raw, &out);
        }
        let target = if self.since_far <= self.hangover_frames && self.coupling.coupled() {
            1.0
        } else {
            0.0
        };
        if target > 0.0 || self.mix > 0.0 {
            self.stats.active_frames += 1;
        }
        let (from, step) = (self.mix, (target - self.mix) / FRAME as f32);
        for (i, (dst, (aec, raw))) in near.iter_mut().zip(out.iter().zip(raw)).enumerate() {
            let m = from + step * (i + 1) as f32;
            *dst = m * aec + (1.0 - m) * raw;
        }
        self.mix = target;
        self.stats.echo_path = self.coupling.coupled();
    }

    /// [`Self::process_frame`] over whole frames; a trailing partial frame is left as is.
    pub fn process(&mut self, far: &[f32], near: &mut [f32]) {
        for (f, n) in far
            .as_chunks::<FRAME>()
            .0
            .iter()
            .zip(near.as_chunks_mut::<FRAME>().0)
        {
            self.process_frame(f, n);
        }
    }

    pub fn stats(&self) -> AecStats {
        let s = self.apm.statistics();
        AecStats {
            aec_delay_ms: s.delay_ms,
            erle_db: s.echo_return_loss_enhancement,
            ..self.stats.clone()
        }
    }

    /// The coarse tracker confirmed a delay: move the reference if the echo sits outside
    /// the range AEC3 handles well.
    fn realign(&mut self, frame: u64) {
        let Some(delay) = self.tracker.as_ref().and_then(|t| t.estimate()) else {
            return;
        };
        let delay_ms = delay * FRAME / SAMPLES_PER_MS;
        self.stats.tracked_delay_ms = Some(delay_ms as u32);
        let current_ms = self.far_line.len() / SAMPLES_PER_MS;
        let want_ms = if delay_ms >= current_ms + NATIVE_DELAY_MS.end {
            delay_ms - TARGET_LATE_MS
        } else if current_ms > 0 && delay_ms < current_ms + NATIVE_DELAY_MS.start {
            // Pre-delayed too much (the delay shrank, e.g. clock drift): the echo is about
            // to arrive before its reference, which no canceller can follow.
            delay_ms.saturating_sub(TARGET_EARLY_MS)
        } else {
            return;
        };
        if self
            .last_realign
            .is_some_and(|at| frame < at + REALIGN_COOLDOWN)
        {
            return;
        }
        let want = want_ms * SAMPLES_PER_MS / FRAME * FRAME;
        if want == self.far_line.len() {
            return;
        }
        log::info!(
            "echo delay {delay_ms} ms: reference pre-delay {current_ms} → {} ms",
            want / SAMPLES_PER_MS
        );
        if want > self.far_line.len() {
            // The reference repeats nothing: it pauses for the difference.
            for _ in self.far_line.len()..want {
                self.far_line.push_front(0.0);
            }
        } else {
            self.far_line.drain(..self.far_line.len() - want);
        }
        // AEC3 would take many seconds to re-find a delay that jumped; a new instance
        // converges within a second or two of far-end speech.
        self.apm = new_aec3();
        self.fresh = AEC3_LATENCY;
        self.last_realign = Some(frame);
        self.stats.realignments += 1;
        self.stats.reference_delay_ms = (want / SAMPLES_PER_MS) as u32;
    }
}

fn new_aec3() -> sonora::AudioProcessing {
    use sonora::config::EchoCanceller as Aec3;
    let config = sonora::Config {
        // No high-pass filter: with the far end silent the output must equal the input.
        echo_canceller: Some(Aec3 {
            enforce_high_pass_filtering: false,
            ..Default::default()
        }),
        ..Default::default()
    };
    let stream = sonora::StreamConfig::new(SAMPLE_RATE, 1);
    let mut apm = sonora::AudioProcessing::builder()
        .config(config)
        .capture_config(stream)
        .render_config(stream)
        .build();
    // Render and capture frames are fed pairwise from one timeline, so the only delay is
    // the acoustic one inside the signals; AEC3 estimates that itself.
    let _ = apm.set_stream_delay_ms(0);
    apm
}

#[cfg(test)]
pub(crate) mod testutil {
    /// Deterministic white-ish noise.
    pub fn noise(seed: u64, n: usize, level: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * level
            })
            .collect()
    }

    /// A simple echo path: delay, a few decaying taps, gain.
    pub fn echo(far: &[f32], delay: usize, gain: f32) -> Vec<f32> {
        let taps = [1.0f32, 0.5, -0.3, 0.2, 0.1, -0.05];
        let mut out = vec![0.0; far.len()];
        for (i, o) in out.iter_mut().enumerate() {
            for (k, &t) in taps.iter().enumerate() {
                let spread = k * 7;
                if i >= delay + spread {
                    *o += t * far[i - delay - spread];
                }
            }
            *o *= gain;
        }
        out
    }

    pub fn energy(x: &[f32]) -> f64 {
        x.iter().map(|&v| v as f64 * v as f64).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::*;
    use super::*;

    #[test]
    fn silent_far_end_passes_mic_through_delayed() {
        let near = noise(1, 20 * FRAME, 0.1);
        let mut out = near.clone();
        let mut aec = EchoCanceller::new(AecConfig::default());
        aec.process(&vec![0.0; near.len()], &mut out);
        assert_eq!(
            &out[LATENCY_SAMPLES..],
            &near[..near.len() - LATENCY_SAMPLES]
        );
        assert!(out[..LATENCY_SAMPLES].iter().all(|&v| v == 0.0));
        assert_eq!(aec.stats().active_frames, 0);
    }

    #[test]
    fn aec3_latency_matches_constant() {
        // Without pass-through (a far end that is never "silent"), AEC3 itself must
        // delay the mic by exactly LATENCY_SAMPLES; the pass-through path relies on it.
        let cfg = AecConfig {
            far_silence_dbfs: -200.0,
            hangover_ms: 0,
            track_delay: false,
            ..Default::default()
        };
        let mut aec = EchoCanceller::new(cfg);
        let near = noise(2, 100 * FRAME, 0.1);
        let far = vec![1e-9; near.len()];
        let mut out = near.clone();
        aec.process(&far, &mut out);
        let (a, b) = (20 * FRAME, 90 * FRAME);
        let best = (0..400)
            .max_by(|&x, &y| {
                let c = |lag: usize| {
                    (a..b)
                        .map(|i| near[i] as f64 * out[i + lag] as f64)
                        .sum::<f64>()
                };
                c(x).total_cmp(&c(y))
            })
            .unwrap();
        assert_eq!(best, LATENCY_SAMPLES);
    }

    #[test]
    fn cancels_echo_of_far_end() {
        let n = 16_000 * 8;
        let far = noise(3, n, 0.1);
        let echo = echo(&far, 1_600, 0.5); // 100 ms
        let mut out = echo.clone();
        let mut aec = EchoCanceller::new(AecConfig::default());
        aec.process(&far, &mut out);
        let tail = n - 3 * 16_000..n; // after convergence
        let erle = 10.0 * (energy(&echo[tail.clone()]) / energy(&out[tail]).max(1e-12)).log10();
        assert!(erle > 20.0, "ERLE {erle:.1} dB");
        assert!(aec.stats().active_frames > 0);
    }

    /// Far-end turns: 1.5 s of talk, 1.5 s of silence.
    fn spurts(seed: u64, n: usize) -> Vec<f32> {
        let mut far = noise(seed, n, 0.1);
        for (i, v) in far.iter_mut().enumerate() {
            if i % 48_000 >= 24_000 {
                *v = 0.0;
            }
        }
        far
    }

    #[test]
    fn headphones_pass_the_mic_through_until_echo_appears() {
        let n = 16_000 * 15;
        let mut aec = EchoCanceller::new(AecConfig::default());
        // Headphones: the far end talks, the mic only hears its own faint noise.
        let far = spurts(5, n);
        let mic = noise(6, n, 0.002);
        let mut out = mic.clone();
        aec.process(&far, &mut out);
        assert!(!aec.stats().echo_path);
        let tail = n - 2 * 16_000..n;
        assert!(
            tail.clone().all(|i| out[i] == mic[i - LATENCY_SAMPLES]),
            "not passed through untouched"
        );
        // Speakers again: the far end now comes back through the mic.
        let far = spurts(7, n);
        let mic: Vec<f32> = echo(&far, 800, 0.5)
            .iter()
            .zip(noise(8, n, 0.002))
            .map(|(e, v)| e + v)
            .collect();
        let mut out = mic.clone();
        aec.process(&far, &mut out);
        assert!(aec.stats().echo_path);
        let erle = 10.0 * (energy(&mic[tail.clone()]) / energy(&out[tail]).max(1e-12)).log10();
        assert!(erle > 15.0, "ERLE {erle:.1} dB");
    }

    #[test]
    fn realigns_reference_for_long_delays() {
        // 700 ms: outside AEC3's own window; speech-like on/off bursts so the tracker locks.
        let n = 16_000 * 30;
        let mut far = noise(4, n, 0.1);
        for (i, v) in far.iter_mut().enumerate() {
            if (i / 4_000) % 3 == 2 {
                *v = 0.0;
            }
        }
        let echo = echo(&far, 11_200, 0.5);
        let mut out = echo.clone();
        let mut aec = EchoCanceller::new(AecConfig::default());
        aec.process(&far, &mut out);
        let s = aec.stats();
        assert!(
            s.tracked_delay_ms.is_some_and(|d| d.abs_diff(700) <= 20),
            "{s:?}"
        );
        assert_eq!(s.realignments, 1, "{s:?}");
        let tail = n - 5 * 16_000..n;
        let erle = 10.0 * (energy(&echo[tail.clone()]) / energy(&out[tail]).max(1e-12)).log10();
        assert!(erle > 15.0, "ERLE {erle:.1} dB, {s:?}");
    }
}
