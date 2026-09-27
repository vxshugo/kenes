//! Pairs the mic with the system audio played at the same moment of the session timeline.
//!
//! Chunks of both sources arrive interleaved and slightly out of step. Each 10 ms mic
//! frame waits until the system audio covering the same span has arrived, then goes
//! through the [`EchoCanceller`] together with it. If the system audio is late by more
//! than [`AecConfig::max_wait_ms`] of mic audio (or stopped altogether), the frame is
//! processed without it (as silence), so the added latency stays bounded.
//!
//! The output is the processed mic, stamped with the timeline positions of the input:
//! the canceller's fixed 18 ms delay is taken out of the timestamps, so transcript times
//! and the speaker ring buffer line up exactly as without echo cancellation.

use std::collections::VecDeque;

use kenes_types::{AudioChunk, Source};

use crate::{AecConfig, AecStats, EchoCanceller, FRAME, LATENCY_SAMPLES, SAMPLES_PER_MS};

/// A chunk that starts within this many samples of where its stream left off continues
/// it: `start_ms` is rounded down to whole milliseconds.
const JITTER: u64 = SAMPLES_PER_MS as u64;
/// Reference audio kept ahead of the mic at most (the mic stalled): 5 s.
const MAX_FAR_AHEAD: usize = 5 * crate::SAMPLE_RATE as usize;

pub struct StreamCanceller {
    aec: EchoCanceller,
    max_wait: u64,
    /// Reference samples not used yet; `far[0]` is at session sample `far_start`.
    far: VecDeque<f32>,
    far_start: u64,
    /// Session sample just past the newest reference sample (the reference frontier).
    far_end: Option<u64>,
    /// Mic samples not processed yet; `near[0]` is at session sample `near_start`.
    near: VecDeque<f32>,
    near_start: u64,
    near_started: bool,
    /// Where the samples inside the canceller came from, in feed order: `Some(position)`
    /// for mic audio, `None` for padding. Output sample k belongs to fed sample k.
    fed: VecDeque<(Option<u64>, usize)>,
}

impl StreamCanceller {
    pub fn new(cfg: AecConfig) -> Self {
        let max_wait = u64::from(cfg.max_wait_ms) * SAMPLES_PER_MS as u64;
        Self {
            aec: EchoCanceller::new(cfg),
            max_wait,
            far: VecDeque::new(),
            far_start: 0,
            far_end: None,
            near: VecDeque::new(),
            near_start: 0,
            near_started: false,
            // AEC3's latency buffer starts out holding silence.
            fed: VecDeque::from([(None, LATENCY_SAMPLES)]),
        }
    }

    pub fn stats(&self) -> AecStats {
        self.aec.stats()
    }

    /// Mic audio waiting for its reference, ms.
    pub fn pending_ms(&self) -> u64 {
        self.near.len() as u64 / SAMPLES_PER_MS as u64
    }

    /// Takes a chunk of either source and returns the mic audio that became ready.
    /// System chunks are only kept as the reference: the caller forwards them itself.
    pub fn push(&mut self, chunk: &AudioChunk) -> Vec<AudioChunk> {
        let at = chunk.start_ms * SAMPLES_PER_MS as u64;
        let mut out = Vec::new();
        match chunk.source {
            Source::System => self.push_far(at, &chunk.samples),
            Source::Mic => {
                let next = self.near_start + self.near.len() as u64;
                if !self.near_started {
                    self.near_started = true;
                    self.near_start = at;
                } else if at.abs_diff(next) > JITTER {
                    // The mic lost audio (or the timeline jumped): finish what we have and
                    // restart at the new position. The canceller's state carries over.
                    self.drain(true, &mut out);
                    self.near_start = at;
                }
                self.near.extend(&chunk.samples);
            }
        }
        self.drain(false, &mut out);
        out
    }

    /// Processes everything still held back (e.g. when the session stops).
    pub fn finish(&mut self) -> Vec<AudioChunk> {
        let mut out = Vec::new();
        self.drain(true, &mut out);
        // Push the canceller's last LATENCY_SAMPLES out with padding.
        for _ in 0..LATENCY_SAMPLES.div_ceil(FRAME) {
            let mut frame = [0.0f32; FRAME];
            self.aec.process_frame(&[0.0; FRAME], &mut frame);
            self.fed.push_back((None, FRAME));
            self.emit(&frame, &mut out);
        }
        out
    }

    fn push_far(&mut self, at: u64, samples: &[f32]) {
        let mut samples = samples;
        match self.far_end {
            None => {
                self.far_start = at;
                self.far_end = Some(at);
            }
            Some(end) if at > end + JITTER => {
                let gap = at - end;
                if gap as usize > MAX_FAR_AHEAD {
                    self.far.clear();
                    self.far_start = at;
                } else {
                    // Lost reference audio: it was played, but we can't know what it was.
                    self.far.extend(std::iter::repeat_n(0.0, gap as usize));
                }
                self.far_end = Some(at);
            }
            Some(end) if at + JITTER < end => {
                // Overlaps what we already have: keep only the new part.
                let skip = ((end - at) as usize).min(samples.len());
                samples = &samples[skip..];
            }
            Some(_) => {}
        }
        self.far.extend(samples);
        self.far_end = Some(self.far_end.unwrap_or(at) + samples.len() as u64);
        // Drop reference audio from before the mic's current position.
        if self.near_started {
            let used = self
                .near_start
                .saturating_sub(self.far_start)
                .min(self.far.len() as u64) as usize;
            self.far.drain(..used);
            self.far_start += used as u64;
        }
        if self.far.len() > MAX_FAR_AHEAD {
            let excess = self.far.len() - MAX_FAR_AHEAD;
            self.far.drain(..excess);
            self.far_start += excess as u64;
        }
    }

    /// Processes every mic frame whose reference is in (or waited long enough for);
    /// with `force`, also the last partial frame, padded.
    fn drain(&mut self, force: bool, out: &mut Vec<AudioChunk>) {
        let newest = self.near_start + self.near.len() as u64;
        while !self.near.is_empty() {
            let n = self.near_start;
            let real = self.near.len().min(FRAME);
            if real < FRAME && !force {
                break;
            }
            let covered = self.far_end.is_some_and(|end| end >= n + FRAME as u64);
            let waited = n + FRAME as u64 + self.max_wait <= newest;
            if !(covered || waited || force) {
                break;
            }
            let mut far = [0.0f32; FRAME];
            for (i, v) in far.iter_mut().enumerate() {
                let pos = n + i as u64;
                if pos >= self.far_start {
                    *v = self
                        .far
                        .get((pos - self.far_start) as usize)
                        .copied()
                        .unwrap_or(0.0);
                }
            }
            let mut frame = [0.0f32; FRAME];
            for (dst, src) in frame.iter_mut().zip(self.near.drain(..real)) {
                *dst = src;
            }
            self.aec.process_frame(&far, &mut frame);
            self.fed.push_back((Some(n), real));
            if real < FRAME {
                self.fed.push_back((None, FRAME - real));
            }
            self.near_start += real as u64;
            // The reference up to here is used up.
            let used = self
                .near_start
                .saturating_sub(self.far_start)
                .min(self.far.len() as u64) as usize;
            self.far.drain(..used);
            self.far_start += used as u64;
            self.emit(&frame, out);
        }
    }

    /// Maps one processed frame back to the timeline positions of the samples it holds.
    fn emit(&mut self, frame: &[f32], out: &mut Vec<AudioChunk>) {
        let mut rest = frame;
        while !rest.is_empty() {
            let Some((pos, len)) = self.fed.pop_front() else {
                break;
            };
            let take = len.min(rest.len());
            if let Some(pos) = pos {
                match out.last_mut() {
                    Some(last)
                        if last.start_ms * SAMPLES_PER_MS as u64 + last.samples.len() as u64
                            == pos =>
                    {
                        last.samples.extend_from_slice(&rest[..take]);
                    }
                    _ => out.push(AudioChunk {
                        source: Source::Mic,
                        start_ms: pos / SAMPLES_PER_MS as u64,
                        samples: rest[..take].to_vec(),
                    }),
                }
            }
            if take < len {
                let pos = pos.map(|p| p + take as u64);
                self.fed.push_front((pos, len - take));
            }
            rest = &rest[take..];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::*;

    fn chunk(source: Source, start_ms: u64, samples: &[f32]) -> AudioChunk {
        AudioChunk {
            source,
            start_ms,
            samples: samples.to_vec(),
        }
    }

    /// Feeds two whole signals as 32 ms chunks, the mic first at each position.
    fn feed(sc: &mut StreamCanceller, mic: &[f32], far: Option<&[f32]>) -> Vec<AudioChunk> {
        let mut out = Vec::new();
        for (k, m) in mic.chunks(512).enumerate() {
            out.extend(sc.push(&chunk(Source::Mic, k as u64 * 32, m)));
            if let Some(far) = far {
                if let Some(f) = far.chunks(512).nth(k) {
                    out.extend(sc.push(&chunk(Source::System, k as u64 * 32, f)));
                }
            }
        }
        out.extend(sc.finish());
        out
    }

    fn concat(chunks: &[AudioChunk]) -> Vec<f32> {
        chunks
            .iter()
            .flat_map(|c| c.samples.iter().copied())
            .collect()
    }

    fn assert_contiguous(chunks: &[AudioChunk], from_ms: u64) {
        let mut next = from_ms * 16;
        for c in chunks {
            assert_eq!(c.source, Source::Mic);
            assert_eq!(c.start_ms * 16, next, "chunk at {} ms", c.start_ms);
            next += c.samples.len() as u64;
        }
    }

    #[test]
    fn without_reference_mic_passes_through_on_the_same_timeline() {
        let mic = noise(1, 16_000 + 100, 0.1);
        let mut sc = StreamCanceller::new(AecConfig::default());
        let out = feed(&mut sc, &mic, None);
        assert_contiguous(&out, 0);
        // Pass-through, exactly, and the latency is taken out of the timestamps.
        assert_eq!(concat(&out), mic);
    }

    #[test]
    fn silent_reference_passes_through_exactly() {
        let mic = noise(2, 16_000 * 2, 0.1);
        let mut sc = StreamCanceller::new(AecConfig::default());
        let out = feed(&mut sc, &mic, Some(&vec![0.0; mic.len()]));
        assert_contiguous(&out, 0);
        assert_eq!(concat(&out), mic);
    }

    #[test]
    fn mic_waits_for_its_reference_but_not_forever() {
        let cfg = AecConfig {
            max_wait_ms: 100,
            ..Default::default()
        };
        let mut sc = StreamCanceller::new(cfg);
        let m = noise(3, 512, 0.1);
        // Reference for 0–64 ms arrived; mic chunks keep coming.
        sc.push(&chunk(Source::System, 0, &noise(4, 1024, 0.1)));
        let first = sc.push(&chunk(Source::Mic, 0, &m));
        // 0–32 ms mic: 3 whole frames covered; output lags by 18 ms, timestamps don't.
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].start_ms, 0);
        assert_eq!(first[0].samples.len(), 3 * FRAME - LATENCY_SAMPLES);
        let second = sc.push(&chunk(Source::Mic, 32, &m));
        assert_eq!(concat(&second).len(), 3 * FRAME, "covered up to 64 ms");
        // No reference beyond 64 ms: frames wait until the mic is 100 ms past them.
        assert!(sc.push(&chunk(Source::Mic, 64, &m)).is_empty());
        assert!(sc.push(&chunk(Source::Mic, 96, &m)).is_empty());
        assert!(sc.push(&chunk(Source::Mic, 128, &m)).is_empty());
        let late = sc.push(&chunk(Source::Mic, 160, &m));
        assert!(!late.is_empty());
        assert!(
            sc.pending_ms() <= 100 + 10,
            "pending {} ms",
            sc.pending_ms()
        );
        // The reference arriving now releases the rest.
        let rest = sc.push(&chunk(Source::System, 64, &noise(5, 512 * 4, 0.1)));
        assert!(!rest.is_empty());
        assert!(sc.pending_ms() < 10);
    }

    #[test]
    fn reference_arriving_first_or_second_gives_the_same_output() {
        let far = noise(6, 16_000 * 3, 0.1);
        let mic: Vec<f32> = echo(&far, 800, 0.4)
            .iter()
            .zip(noise(7, far.len(), 0.01))
            .map(|(a, b)| a + b)
            .collect();
        let mut a = StreamCanceller::new(AecConfig::default());
        let out_a = feed(&mut a, &mic, Some(&far));
        // Same data, reference chunk before the mic chunk at each position.
        let mut b = StreamCanceller::new(AecConfig::default());
        let mut out_b = Vec::new();
        for (k, (m, f)) in mic.chunks(512).zip(far.chunks(512)).enumerate() {
            out_b.extend(b.push(&chunk(Source::System, k as u64 * 32, f)));
            out_b.extend(b.push(&chunk(Source::Mic, k as u64 * 32, m)));
        }
        out_b.extend(b.finish());
        assert_contiguous(&out_a, 0);
        assert_contiguous(&out_b, 0);
        assert_eq!(concat(&out_a), concat(&out_b));
        assert_eq!(concat(&out_a).len(), mic.len());
        // And the echo is gone by the end.
        let n = mic.len();
        let e_in = energy(&mic[n - 16_000..]);
        let e_out = energy(&concat(&out_a)[n - 16_000..]);
        assert!(e_in / e_out > 30.0, "suppression {:.1}x", e_in / e_out);
    }

    #[test]
    fn mic_gap_restarts_timeline() {
        let mut sc = StreamCanceller::new(AecConfig::default());
        let m = noise(8, 512, 0.1);
        let mut out = Vec::new();
        out.extend(sc.push(&chunk(Source::Mic, 0, &m)));
        out.extend(sc.push(&chunk(Source::Mic, 32, &m)));
        // 300 ms lost.
        out.extend(sc.push(&chunk(Source::Mic, 364, &m)));
        out.extend(sc.finish());
        let total: usize = out.iter().map(|c| c.samples.len()).sum();
        assert_eq!(total, 3 * 512);
        let starts: Vec<u64> = out.iter().map(|c| c.start_ms).collect();
        assert_eq!(starts.first(), Some(&0));
        assert!(out.iter().any(|c| c.start_ms == 364), "{starts:?}");
        // Pieces before the gap are contiguous and end at 64 ms.
        let before: usize = out
            .iter()
            .filter(|c| c.start_ms < 364)
            .map(|c| c.samples.len())
            .sum();
        assert_eq!(before, 1024);
    }

    #[test]
    fn reference_gap_is_filled_with_silence() {
        let mut sc = StreamCanceller::new(AecConfig::default());
        sc.push(&chunk(Source::System, 0, &[0.1; 512]));
        sc.push(&chunk(Source::System, 100, &[0.1; 512]));
        assert_eq!(sc.far_end, Some((100 + 32) * 16));
        assert_eq!(sc.far.len(), (100 + 32) * 16);
        // A repeated chunk adds nothing.
        sc.push(&chunk(Source::System, 100, &[0.1; 512]));
        assert_eq!(sc.far.len(), (100 + 32) * 16);
    }
}
