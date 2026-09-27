//! Optional speaker-change splitting of final utterances.
//!
//! The VAD glues turns together when people answer within 0.5 s or talk over
//! each other. A splitter (in practice `kenes_speakers::change_points`,
//! wrapped by kenes-core) looks at a final utterance's audio and returns the
//! sample offsets where the speaker changes; each piece is then decoded and
//! emitted as its own final.
//!
//! The closure runs on a helper thread owned by the `Transcriber`, and the
//! worker waits for it with a timeout, so a slow or stuck splitter delays a
//! final by at most the timeout and a panicking one only costs that split.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError};
use kenes_types::SAMPLE_RATE;

/// Speaker-change detector: given a final utterance (16 kHz mono, exactly
/// the samples that get decoded), return the sample offsets where the
/// speaker changes. Empty means "one speaker".
pub type Splitter = Box<dyn FnMut(&[f32]) -> Vec<usize> + Send>;

/// When and how the splitter is used. See [`crate::Transcriber::set_splitter`].
#[derive(Clone, Debug, PartialEq)]
pub struct SplitOptions {
    /// Only finals at least this long are offered to the splitter (default 2500).
    pub min_utterance_ms: u64,
    /// Pieces shorter than this are merged into a neighbour instead of being
    /// decoded alone (default 400).
    pub min_piece_ms: u64,
    /// Wait at most `timeout_ms + timeout_ratio × utterance length` for the
    /// splitter, then emit the utterance unsplit (default 2000 ms + 0.25).
    pub timeout_ms: u64,
    pub timeout_ratio: f32,
}

impl Default for SplitOptions {
    fn default() -> Self {
        SplitOptions {
            min_utterance_ms: 2_500,
            min_piece_ms: 400,
            timeout_ms: 2_000,
            timeout_ratio: 0.25,
        }
    }
}

impl SplitOptions {
    pub(crate) fn timeout(&self, samples: usize) -> Duration {
        let audio_s = samples as f64 / SAMPLE_RATE as f64;
        Duration::from_millis(self.timeout_ms)
            + Duration::from_secs_f64(audio_s * self.timeout_ratio.max(0.0) as f64)
    }
}

/// What happened when we asked the splitter.
#[derive(Debug, PartialEq)]
pub(crate) enum SplitResult {
    /// It answered (possibly with no cuts) after `took` of its own time.
    Cuts(Vec<usize>, Duration),
    /// It panicked on this input.
    Panicked,
    /// It didn't answer within the timeout (the answer will be discarded).
    TimedOut,
    /// It is still busy with an earlier, timed-out request.
    Busy,
    /// Its thread is gone.
    Dead,
}

type Response = (u64, Option<Vec<usize>>, Duration);

pub(crate) struct SplitterHandle {
    req_tx: Option<Sender<(u64, Vec<f32>)>>,
    resp_rx: Receiver<Response>,
    next_seq: u64,
    /// A request whose response hasn't been consumed yet.
    in_flight: Option<u64>,
    _thread: JoinHandle<()>,
}

impl SplitterHandle {
    pub fn new(mut f: Splitter) -> Self {
        let (req_tx, req_rx) = crossbeam_channel::unbounded::<(u64, Vec<f32>)>();
        let (resp_tx, resp_rx) = crossbeam_channel::unbounded::<Response>();
        let thread = std::thread::Builder::new()
            .name("kenes-stt-split".into())
            .spawn(move || {
                for (seq, samples) in req_rx {
                    let t0 = Instant::now();
                    let cuts = catch_unwind(AssertUnwindSafe(|| f(&samples))).ok();
                    if resp_tx.send((seq, cuts, t0.elapsed())).is_err() {
                        break;
                    }
                }
            })
            .expect("failed to spawn the splitter thread");
        SplitterHandle {
            req_tx: Some(req_tx),
            resp_rx,
            next_seq: 0,
            in_flight: None,
            _thread: thread,
        }
    }

    /// Ask for the change points of `samples`, waiting at most `timeout`.
    pub fn split(&mut self, samples: &[f32], timeout: Duration) -> SplitResult {
        // Collect the answer to an earlier timed-out request, if it's there.
        while let Some(seq) = self.in_flight {
            match self.resp_rx.try_recv() {
                Ok((s, _, _)) if s == seq => self.in_flight = None,
                Ok(_) => {}
                Err(TryRecvError::Empty) => return SplitResult::Busy,
                Err(TryRecvError::Disconnected) => return SplitResult::Dead,
            }
        }
        let Some(tx) = &self.req_tx else {
            return SplitResult::Dead;
        };
        let seq = self.next_seq;
        self.next_seq += 1;
        if tx.send((seq, samples.to_vec())).is_err() {
            return SplitResult::Dead;
        }
        self.in_flight = Some(seq);
        let deadline = Instant::now() + timeout;
        loop {
            match self.resp_rx.recv_deadline(deadline) {
                Ok((s, result, took)) if s == seq => {
                    self.in_flight = None;
                    return match result {
                        Some(cuts) => SplitResult::Cuts(cuts, took),
                        None => SplitResult::Panicked,
                    };
                }
                Ok(_) => {}
                Err(RecvTimeoutError::Timeout) => return SplitResult::TimedOut,
                Err(RecvTimeoutError::Disconnected) => return SplitResult::Dead,
            }
        }
    }
}

impl Drop for SplitterHandle {
    fn drop(&mut self) {
        // Closing the request channel ends the thread once it finishes its
        // current call. Don't join: a stuck splitter must not block shutdown.
        self.req_tx.take();
    }
}

/// Turn raw cut offsets into consecutive `(from, to)` pieces covering
/// `0..len`, dropping out-of-range and duplicate cuts and merging pieces
/// shorter than `min_piece` into their shorter neighbour.
pub(crate) fn plan_pieces(
    len: usize,
    mut cuts: Vec<usize>,
    min_piece: usize,
) -> Vec<(usize, usize)> {
    cuts.retain(|&c| c > 0 && c < len);
    cuts.sort_unstable();
    cuts.dedup();
    let mut bounds = Vec::with_capacity(cuts.len() + 2);
    bounds.push(0);
    bounds.extend(cuts);
    bounds.push(len);
    while bounds.len() > 2 {
        let piece = |k: usize, b: &[usize]| b[k + 1] - b[k];
        let n = bounds.len() - 1;
        let (k, shortest) = (0..n)
            .map(|k| (k, piece(k, &bounds)))
            .min_by_key(|&(_, l)| l)
            .unwrap();
        if shortest >= min_piece {
            break;
        }
        // Remove the boundary on the side of the shorter neighbour (bound k
        // joins piece k to the left, bound k + 1 to the right).
        let merge_left = k == n - 1 || (k > 0 && piece(k - 1, &bounds) <= piece(k + 1, &bounds));
        bounds.remove(if merge_left { k } else { k + 1 });
    }
    bounds.windows(2).map(|w| (w[0], w[1])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_pieces_sanitizes_and_merges() {
        assert_eq!(plan_pieces(100, vec![], 10), vec![(0, 100)]);
        assert_eq!(
            plan_pieces(100, vec![0, 100, 250, 50, 50], 10),
            vec![(0, 50), (50, 100)]
        );
        // A 5-sample piece at the start joins the next one.
        assert_eq!(plan_pieces(100, vec![5, 60], 10), vec![(0, 60), (60, 100)]);
        // …at the end joins the previous one.
        assert_eq!(plan_pieces(100, vec![40, 97], 10), vec![(0, 40), (40, 100)]);
        // …in the middle joins its shorter neighbour.
        assert_eq!(
            plan_pieces(100, vec![30, 35, 60], 10),
            vec![(0, 30), (30, 60), (60, 100)]
        );
        assert_eq!(
            plan_pieces(100, vec![20, 25, 60], 10),
            vec![(0, 25), (25, 60), (60, 100)]
        );
        assert_eq!(
            plan_pieces(100, vec![40, 45, 80], 10),
            vec![(0, 40), (40, 80), (80, 100)]
        );
        // Everything tiny: one piece.
        assert_eq!(
            plan_pieces(20, vec![5, 10, 15], 10),
            vec![(0, 10), (10, 20)]
        );
        assert_eq!(plan_pieces(12, vec![4, 8], 10), vec![(0, 12)]);
    }

    #[test]
    fn handle_reports_cuts_panics_and_timeouts() {
        let mut calls = 0;
        let mut h = SplitterHandle::new(Box::new(move |s: &[f32]| {
            calls += 1;
            match calls {
                1 => vec![s.len() / 2],
                2 => panic!("splitter bug"),
                3 => {
                    std::thread::sleep(Duration::from_millis(300));
                    vec![1]
                }
                _ => vec![],
            }
        }));
        let audio = vec![0.0; 1000];
        let t = Duration::from_secs(5);
        assert!(matches!(h.split(&audio, t), SplitResult::Cuts(c, _) if c == vec![500]));
        assert_eq!(h.split(&audio, t), SplitResult::Panicked);
        // Survives the panic; the slow third call times out…
        assert_eq!(
            h.split(&audio, Duration::from_millis(20)),
            SplitResult::TimedOut
        );
        // …and while it is still running, we don't queue behind it.
        assert_eq!(h.split(&audio, t), SplitResult::Busy);
        std::thread::sleep(Duration::from_millis(400));
        // Its late answer is discarded; the next request gets its own.
        assert!(matches!(h.split(&audio, t), SplitResult::Cuts(c, _) if c.is_empty()));
    }
}
