//! Coarse echo-delay tracking over a wider range than AEC3 searches by itself.
//!
//! AEC3 finds the echo delay on its own, but only within about 450 ms. Bluetooth
//! speakers, large output buffers or an hour of clock drift between two devices can
//! push the echo further behind its reference. This tracker estimates the delay up
//! to [`crate::AecConfig::max_delay_ms`] from 10 ms log-energy envelopes, which is
//! cheap (a few thousand multiply-adds per frame) and robust to the echo path's
//! coloration, reverb and clipping. The canceller uses it only to pre-delay the
//! reference so that the echo lands back inside AEC3's window.

use std::collections::VecDeque;

/// Log-energy floor, dB relative to full scale: digital silence and quiet noise look alike.
const FLOOR_DB: f32 = -75.0;
/// A far-end frame counts as sound above this level.
const ACTIVE_DB: f32 = -55.0;
/// Shortest stretch correlated, in frames (2 s).
const MIN_WINDOW: usize = 200;

pub(crate) struct DelayTracker {
    far: VecDeque<f32>,
    near: VecDeque<f32>,
    /// Longest delay searched, in frames.
    max_lag: usize,
    /// Frames correlated per estimate.
    window: usize,
    /// Frames between estimates.
    step: usize,
    since: usize,
    /// Last raw estimate (frames) and how many estimates in a row agreed with it.
    last: Option<usize>,
    streak: u32,
    confirmed: Option<usize>,
}

impl DelayTracker {
    pub(crate) fn new(max_lag: usize) -> Self {
        let window = 500; // 5 s
        Self {
            far: VecDeque::with_capacity(window + max_lag + 1),
            near: VecDeque::with_capacity(window + 1),
            max_lag,
            window,
            step: 50,
            since: 0,
            last: None,
            streak: 0,
            confirmed: None,
        }
    }

    /// The delay, in frames, once three estimates in a row agreed on it.
    pub(crate) fn estimate(&self) -> Option<usize> {
        self.confirmed
    }

    /// Adds one frame of each signal. Returns true when a new estimate was confirmed.
    pub(crate) fn push(&mut self, far: &[f32], near: &[f32]) -> bool {
        self.far.push_back(log_energy(far));
        self.near.push_back(log_energy(near));
        if self.far.len() > self.window + self.max_lag {
            self.far.pop_front();
        }
        if self.near.len() > self.window {
            self.near.pop_front();
        }
        self.since += 1;
        if self.since < self.step {
            return false;
        }
        self.since = 0;
        let Some(lag) = self.correlate() else {
            return false;
        };
        let agrees = self.last.is_some_and(|l| l.abs_diff(lag) <= 3);
        self.streak = if agrees { self.streak + 1 } else { 1 };
        self.last = Some(lag);
        if self.streak >= 3 && self.confirmed.is_none_or(|c| c.abs_diff(lag) > 3) {
            self.confirmed = Some(lag);
            return true;
        }
        false
    }

    /// The lag with the highest envelope correlation, if it clearly stands out.
    fn correlate(&self) -> Option<usize> {
        // Correlate the newest `n` frames: the whole window once there is enough
        // history for every lag, a shorter one (at least MIN_WINDOW) early on.
        let n = self.near.len().min(self.far.len().saturating_sub(self.max_lag));
        if n < MIN_WINDOW {
            return None;
        }
        let far: Vec<f32> = self.far.iter().copied().collect();
        let near: Vec<f32> = self.near.iter().skip(self.near.len() - n).copied().collect();
        // Enough far-end sound in the searched history to say anything?
        let active = far.iter().filter(|&&v| v > ACTIVE_DB).count();
        if active * 4 < n {
            return None;
        }
        let (near_mean, near_var) = mean_var(&near);
        if near_var < 1.0 {
            return None;
        }
        let end = far.len();
        let mut scores = Vec::with_capacity(self.max_lag + 1);
        for lag in 0..=self.max_lag {
            // near[i] (the last n frames) against far delayed by `lag` frames.
            let seg = &far[end - n - lag..end - lag];
            let (m, v) = mean_var(seg);
            if v < 1.0 {
                scores.push(f32::MIN);
                continue;
            }
            let cov: f64 = seg.iter().zip(&near).map(|(&a, &b)| (a - m) as f64 * (b - near_mean) as f64).sum::<f64>() / n as f64;
            scores.push((cov / (v as f64 * near_var as f64).sqrt()) as f32);
        }
        let (best, &peak) = scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;
        let runner_up = scores
            .iter()
            .enumerate()
            .filter(|(l, _)| l.abs_diff(best) > 8)
            .map(|(_, &s)| s)
            .fold(f32::MIN, f32::max);
        (peak >= 0.4 && peak - runner_up >= 0.08).then_some(best)
    }
}

fn log_energy(x: &[f32]) -> f32 {
    let e = x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32;
    (10.0 * (e + 1e-12).log10()).max(FLOOR_DB)
}

fn mean_var(x: &[f32]) -> (f32, f32) {
    let n = x.len().max(1) as f32;
    let m = x.iter().sum::<f32>() / n;
    let v = x.iter().map(|a| (a - m) * (a - m)).sum::<f32>() / n;
    (m, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FRAME;

    /// Speech-like test signal: noise bursts of random length and level.
    fn bursts(seed: u64, frames: usize) -> Vec<f32> {
        let mut s = seed;
        let mut rnd = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let mut out = Vec::with_capacity(frames * FRAME);
        while out.len() < frames * FRAME {
            let len = ((rnd() + 0.5) * 30.0) as usize + 5;
            let level = if rnd() > 0.0 { (rnd() + 0.6) * 0.2 } else { 0.0 };
            for _ in 0..len * FRAME {
                out.push(rnd() * level);
            }
        }
        out.truncate(frames * FRAME);
        out
    }

    fn run(delay_frames: usize, echo_gain: f32, near_level: f32) -> Option<usize> {
        let frames = 1500;
        let far = bursts(1, frames);
        let talk = bursts(2, frames);
        let mut t = DelayTracker::new(100);
        for k in 0..frames {
            let f = &far[k * FRAME..(k + 1) * FRAME];
            let near: Vec<f32> = (0..FRAME)
                .map(|i| {
                    let n = k * FRAME + i;
                    let e = if n >= delay_frames * FRAME { far[n - delay_frames * FRAME] * echo_gain } else { 0.0 };
                    e + talk[n] * near_level
                })
                .collect();
            t.push(f, &near);
        }
        t.estimate()
    }

    #[test]
    fn finds_delays_beyond_aec3_window() {
        for d in [0, 12, 45, 70, 95] {
            let est = run(d, 0.3, 0.0).expect("no estimate");
            assert!(est.abs_diff(d) <= 1, "delay {d}: estimated {est}");
        }
    }

    #[test]
    fn survives_moderate_near_end_talk() {
        let est = run(60, 0.3, 0.02).expect("no estimate");
        assert!(est.abs_diff(60) <= 1, "estimated {est}");
    }

    #[test]
    fn no_estimate_without_echo() {
        assert_eq!(run(30, 0.0, 0.2), None);
    }
}
