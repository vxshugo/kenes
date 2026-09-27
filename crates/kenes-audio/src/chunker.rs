//! Cuts a stream of 16 kHz samples into [`AudioChunk`]s stamped on the shared session clock.
//!
//! Timestamps come from the sample count, not from when a read happened, so they
//! are jitter-free and chunks of one stream are exactly contiguous. The stream is
//! anchored to the session clock by its first read (the data that just arrived is
//! assumed to end "now"). If the stream later loses audio (the server dropped
//! samples, or the device was suspended and resumed), wall-clock time runs ahead of
//! the sample count; once that lag has persisted for [`GAP_CONFIRM`] the timeline
//! jumps forward so both sources stay comparable.

use std::time::Instant;

use kenes_types::{AudioChunk, Source, SAMPLE_RATE};

const RATE: u64 = SAMPLE_RATE as u64;
/// Samples per [`AudioChunk`] (32 ms, one Silero VAD window at 16 kHz). Only the
/// last chunk of a stream, flushed on stop, can be shorter.
pub const CHUNK_SAMPLES: usize = 512;
/// A lag below this is treated as ordinary scheduling/buffering jitter.
const GAP_THRESHOLD: u64 = RATE / 4;
/// How long a lag must persist before the timeline jumps.
const GAP_CONFIRM: u64 = RATE;

/// Samples (at 16 kHz) elapsed since `start` on the monotonic clock.
pub(crate) fn session_samples(start: Instant) -> u64 {
    (start.elapsed().as_micros() * u128::from(RATE) / 1_000_000) as u64
}

pub(crate) fn samples_to_ms(samples: u64) -> u64 {
    samples * 1000 / RATE
}

pub(crate) struct Chunker {
    source: Source,
    chunk_len: usize,
    pending: Vec<f32>,
    /// Session sample index of `pending[0]`.
    pending_start: u64,
    /// Session sample index just past the last sample received; `None` until anchored.
    next_index: Option<u64>,
    /// Session sample index of the stream's first sample.
    origin: Option<u64>,
    /// `(first seen at, smallest lag since)` while the lag stays above the threshold.
    gap: Option<(u64, u64)>,
}

impl Chunker {
    pub(crate) fn new(source: Source) -> Self {
        Self::with_chunk_len(source, CHUNK_SAMPLES)
    }

    pub(crate) fn with_chunk_len(source: Source, chunk_len: usize) -> Self {
        assert!(chunk_len > 0);
        Self {
            source,
            chunk_len,
            pending: Vec::with_capacity(chunk_len * 2),
            pending_start: 0,
            next_index: None,
            origin: None,
            gap: None,
        }
    }

    /// Session time (ms) of the stream's first sample, once anchored.
    pub(crate) fn origin_ms(&self) -> Option<u64> {
        self.origin.map(samples_to_ms)
    }

    /// Adds `samples` that became available at session time `now` (in samples) and
    /// emits every complete chunk.
    pub(crate) fn push(&mut self, samples: &[f32], now: u64, emit: &mut impl FnMut(AudioChunk)) {
        if samples.is_empty() {
            return;
        }
        let n = samples.len() as u64;
        let next = match self.next_index {
            None => {
                let start = now.saturating_sub(n);
                self.pending_start = start;
                self.origin = Some(start);
                start
            }
            Some(next) => self.check_gap(next, n, now, emit),
        };
        self.pending.extend_from_slice(samples);
        self.next_index = Some(next + n);

        let mut offset = 0;
        while self.pending.len() - offset >= self.chunk_len {
            emit(AudioChunk {
                source: self.source,
                start_ms: samples_to_ms(self.pending_start),
                samples: self.pending[offset..offset + self.chunk_len].to_vec(),
            });
            offset += self.chunk_len;
            self.pending_start += self.chunk_len as u64;
        }
        self.pending.drain(..offset);
    }

    /// Returns where the new samples start on the timeline, jumping over lost audio.
    fn check_gap(&mut self, next: u64, n: u64, now: u64, emit: &mut impl FnMut(AudioChunk)) -> u64 {
        let lag = now.saturating_sub(next + n);
        if lag <= GAP_THRESHOLD {
            self.gap = None;
            return next;
        }
        let (since, min_lag) = self.gap.get_or_insert((now, lag));
        *min_lag = (*min_lag).min(lag);
        if now.saturating_sub(*since) < GAP_CONFIRM {
            return next;
        }
        let jump = *min_lag;
        self.gap = None;
        // Complete the partial chunk with silence from the lost span (always longer than a
        // chunk) so only the last chunk of a stream is ever short.
        let pad = (self.chunk_len - self.pending.len()) % self.chunk_len;
        if pad as u64 <= jump {
            self.pending.resize(self.pending.len() + pad, 0.0);
        }
        self.flush(emit);
        log::warn!(
            "{} stream lost about {} ms of audio; re-syncing timestamps",
            self.source.as_str(),
            samples_to_ms(jump)
        );
        let next = next + jump;
        self.pending_start = next;
        next
    }

    /// Emits whatever is buffered as a final, shorter chunk.
    pub(crate) fn flush(&mut self, emit: &mut impl FnMut(AudioChunk)) {
        if self.pending.is_empty() {
            return;
        }
        let samples = std::mem::take(&mut self.pending);
        let len = samples.len() as u64;
        emit(AudioChunk {
            source: self.source,
            start_ms: samples_to_ms(self.pending_start),
            samples,
        });
        self.pending_start += len;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(chunker: &mut Chunker, samples: &[f32], now: u64, out: &mut Vec<AudioChunk>) {
        chunker.push(samples, now, &mut |c| out.push(c));
    }

    #[test]
    fn anchors_first_read_and_counts_samples() {
        let mut c = Chunker::with_chunk_len(Source::Mic, 160);
        let mut out = Vec::new();
        // 100 samples arrive 100 ms (1600 samples) into the session: they started at 1500.
        collect(&mut c, &[0.1; 100], 1600, &mut out);
        assert!(out.is_empty());
        assert_eq!(c.origin_ms(), Some(93));
        collect(&mut c, &[0.2; 300], 1900, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].start_ms, 1500 / 16);
        assert_eq!(out[1].start_ms, (1500 + 160) / 16);
        assert!(out
            .iter()
            .all(|c| c.samples.len() == 160 && c.source == Source::Mic));
        assert_eq!(out[0].samples[99], 0.1);
        assert_eq!(out[0].samples[100], 0.2);

        // Timestamps follow the sample count even when reads arrive with jitter.
        collect(&mut c, &[0.3; 80], 2400, &mut out);
        collect(&mut c, &[0.3; 80], 2000, &mut out);
        assert_eq!(out.len(), 3);
        assert_eq!(out[2].start_ms, (1500 + 320) / 16);

        c.flush(&mut |ch| out.push(ch));
        assert_eq!(out.len(), 4);
        assert_eq!(out[3].samples.len(), 80);
        assert_eq!(out[3].start_ms, (1500 + 480) / 16);
    }

    #[test]
    fn contiguous_chunks_have_contiguous_timestamps() {
        let mut c = Chunker::new(Source::System);
        let mut out = Vec::new();
        let mut now = 5000;
        for _ in 0..100 {
            now += 320; // 20 ms reads, real time
            collect(&mut c, &[0.0; 320], now, &mut out);
        }
        assert_eq!(out.len(), 100 * 320 / CHUNK_SAMPLES);
        for w in out.windows(2) {
            assert_eq!(w[1].start_ms - w[0].start_ms, 32);
            assert_eq!(w[0].duration_ms(), 32);
        }
    }

    #[test]
    fn short_stall_does_not_shift_timeline() {
        let mut c = Chunker::with_chunk_len(Source::Mic, 320);
        let mut out = Vec::new();
        let mut now = 320;
        collect(&mut c, &[0.0; 320], now, &mut out);
        // Reader stalls 400 ms, then drains the backlog in a few reads at the same instant.
        now += 320 * 21;
        for _ in 0..21 {
            collect(&mut c, &[0.0; 320], now, &mut out);
        }
        for _ in 0..100 {
            now += 320;
            collect(&mut c, &[0.0; 320], now, &mut out);
        }
        for (i, ch) in out.iter().enumerate() {
            assert_eq!(ch.start_ms, i as u64 * 20);
        }
    }

    #[test]
    fn persistent_gap_jumps_forward() {
        let mut c = Chunker::with_chunk_len(Source::System, 320);
        let mut out = Vec::new();
        let mut now = 0;
        for _ in 0..10 {
            now += 320;
            collect(&mut c, &[0.0; 320], now, &mut out);
        }
        assert_eq!(out.last().unwrap().start_ms, 180);
        // Three seconds of audio never arrive, then the stream continues in real time.
        now += 3 * RATE;
        for _ in 0..100 {
            now += 320;
            collect(&mut c, &[0.0; 320], now, &mut out);
        }
        let last = out.last().unwrap();
        let expected_ms = samples_to_ms(now - 320);
        assert!(
            last.start_ms.abs_diff(expected_ms) <= 1,
            "last chunk at {} ms, wall clock says {} ms",
            last.start_ms,
            expected_ms
        );
        // Monotonic, never overlapping.
        for w in out.windows(2) {
            assert!(w[1].start_ms >= w[0].start_ms + w[0].duration_ms());
        }
    }

    #[test]
    fn gap_resync_keeps_every_chunk_full_length() {
        // CHUNK_SAMPLES promises 32 ms chunks except the last one flushed on stop.
        let mut c = Chunker::new(Source::Mic);
        let mut out = Vec::new();
        let mut now = 0;
        // 10 × 20 ms reads: six full chunks plus 128 samples waiting when the gap hits.
        for _ in 0..10 {
            now += 320;
            collect(&mut c, &[0.25; 320], now, &mut out);
        }
        now += 3 * RATE;
        for _ in 0..100 {
            now += 320;
            collect(&mut c, &[0.25; 320], now, &mut out);
        }
        c.flush(&mut |ch| out.push(ch));
        let (_, all_but_last) = out.split_last().unwrap();
        for (i, ch) in all_but_last.iter().enumerate() {
            assert_eq!(
                ch.samples.len(),
                CHUNK_SAMPLES,
                "chunk {i} at {} ms",
                ch.start_ms
            );
        }
        // Every captured sample is delivered exactly once, and chunks never overlap.
        let real = out
            .iter()
            .flat_map(|c| &c.samples)
            .filter(|&&s| s == 0.25)
            .count();
        assert_eq!(real, 110 * 320);
        for w in out.windows(2) {
            assert!(w[1].start_ms >= w[0].start_ms + w[0].duration_ms());
        }
        let last = out.last().unwrap();
        assert!(samples_to_ms(now).abs_diff(last.start_ms + last.duration_ms()) <= 1);
    }

    #[test]
    fn session_samples_counts_at_16k() {
        let start = Instant::now() - std::time::Duration::from_millis(250);
        let s = session_samples(start);
        assert!((4000..4400).contains(&s), "{s}");
    }
}
