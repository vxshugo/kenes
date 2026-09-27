//! Transcript-level echo guard: a safety net for echo the canceller left in the mic.
//!
//! If the call's audio still leaks into the mic, the same words are recognized twice:
//! once from system audio and again from the mic. The guard holds each mic final
//! briefly (at most [`MAX_HOLD`]) while the call around the same time is still being
//! transcribed, and drops it if its text repeats the overlapping system finals. A
//! dropped final becomes an empty final when the UI has seen partials for its id
//! (the contract's "this was noise"), and disappears otherwise; it is never stored or
//! labeled.
//!
//! Mic finals are only held while the call had sound around them, so the user's own
//! lines are not delayed while the other side is quiet. System segments and mic
//! partials are never held. Mic finals leave in the order they arrived.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use kenes_types::{Segment, Source};

/// Longest a mic final waits for the call's transcript.
pub const MAX_HOLD: Duration = Duration::from_millis(2_500);
/// System finals within this distance of a mic final (either side) are compared with it.
const WINDOW_MS: u64 = 1_500;
/// A mic final whose character trigrams are at least this much contained in the
/// overlapping system text is echo. Tuned on the synthetic echo set, see
/// `crates/kenes-aec/EVAL.md`.
pub const THRESHOLD: f32 = 0.6;
/// System audio louder than this (chunk RMS) is sound that may come back as echo.
const ACTIVE_RMS: f32 = 0.003; // about −50 dBFS
/// Sound this close to a system final counts as transcribed by it.
const COVER_SLACK_MS: u64 = 500;
/// Untranscribed system sound shorter than this doesn't make a mic final wait.
const MIN_UNCOVERED_MS: u64 = 150;
/// System history kept for matching.
const HISTORY_MS: u64 = 60_000;

struct Held {
    seg: Segment,
    deadline: Instant,
}

struct SysFinal {
    start_ms: u64,
    end_ms: u64,
    text: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GuardStats {
    pub held: u64,
    pub dropped: u64,
    /// Sum of hold times, for the average.
    pub held_for: Duration,
}

#[derive(Default)]
pub struct EchoGuard {
    held: VecDeque<Held>,
    /// Mic ids the UI has partials for.
    mic_partials: HashSet<String>,
    /// System utterances with partials but no final yet: id → start.
    sys_open: HashMap<String, u64>,
    sys_finals: VecDeque<SysFinal>,
    /// Merged spans of system audio with sound, ms.
    activity: VecDeque<(u64, u64)>,
    stats: GuardStats,
}

enum Verdict {
    Keep,
    Drop,
    Wait,
}

impl EchoGuard {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stats(&self) -> &GuardStats {
        &self.stats
    }

    /// Records one chunk of system audio (the call), so the guard knows when echo was possible.
    pub fn system_audio(&mut self, start_ms: u64, end_ms: u64, rms: f32) {
        if rms < ACTIVE_RMS {
            return;
        }
        match self.activity.back_mut() {
            Some(last) if start_ms <= last.1 + 1 => last.1 = last.1.max(end_ms),
            _ => self.activity.push_back((start_ms, end_ms)),
        }
        while self.activity.front().is_some_and(|a| a.1 + HISTORY_MS < end_ms) {
            self.activity.pop_front();
        }
    }

    /// Takes a segment from the transcriber and returns what to emit now, in order.
    pub fn push(&mut self, seg: Segment, now: Instant) -> Vec<Segment> {
        let mut out = Vec::new();
        match (seg.source, seg.is_final) {
            (Source::System, false) => {
                let start = self.sys_open.entry(seg.id.clone()).or_insert(seg.start_ms);
                *start = (*start).min(seg.start_ms);
                out.push(seg);
            }
            (Source::System, true) => {
                self.sys_open.remove(&seg.id);
                let text = normalize(&seg.text);
                if !text.is_empty() {
                    self.sys_finals.push_back(SysFinal { start_ms: seg.start_ms, end_ms: seg.end_ms, text });
                    let newest = seg.end_ms;
                    while self.sys_finals.front().is_some_and(|f| f.end_ms + HISTORY_MS < newest) {
                        self.sys_finals.pop_front();
                    }
                }
                out.push(seg);
            }
            (Source::Mic, false) => {
                self.mic_partials.insert(seg.id.clone());
                out.push(seg);
            }
            (Source::Mic, true) if normalize(&seg.text).is_empty() => {
                // Nothing to compare: an empty (noise) final goes straight out.
                self.mic_partials.remove(&seg.id);
                out.push(seg);
            }
            (Source::Mic, true) => {
                self.stats.held += 1;
                self.held.push_back(Held { seg, deadline: now + MAX_HOLD });
            }
        }
        self.release(now, false, &mut out);
        out
    }

    /// Releases mic finals whose wait is over. Call regularly (e.g. on every loop turn).
    pub fn poll(&mut self, now: Instant) -> Vec<Segment> {
        let mut out = Vec::new();
        self.release(now, false, &mut out);
        out
    }

    /// Decides everything still held (the session is ending; all system finals are in).
    pub fn finish(&mut self, now: Instant) -> Vec<Segment> {
        let mut out = Vec::new();
        self.release(now, true, &mut out);
        out
    }

    fn release(&mut self, now: Instant, all: bool, out: &mut Vec<Segment>) {
        while let Some(front) = self.held.front() {
            let verdict = match self.judge(&front.seg, now >= front.deadline || all) {
                Verdict::Wait => break,
                v => v,
            };
            let Held { mut seg, deadline } = self.held.pop_front().expect("front exists");
            self.stats.held_for += now.saturating_duration_since(deadline - MAX_HOLD);
            let had_partials = self.mic_partials.remove(&seg.id);
            match verdict {
                Verdict::Drop => {
                    self.stats.dropped += 1;
                    log::info!("echo guard: dropped {} ({:?})", seg.id, seg.text);
                    if had_partials {
                        // Tells the UI to drop the partial (docs/CONTRACT.md).
                        seg.text.clear();
                        out.push(seg);
                    }
                }
                _ => out.push(seg),
            }
        }
    }

    fn judge(&self, seg: &Segment, expired: bool) -> Verdict {
        let lo = seg.start_ms.saturating_sub(WINDOW_MS);
        let hi = seg.end_ms + WINDOW_MS;
        // Echo comes after the call's audio: the call had to have sound before the mic final ended.
        if !self.active_between(lo, seg.end_ms) {
            return Verdict::Keep;
        }
        let mut overlapping: Vec<&SysFinal> =
            self.sys_finals.iter().filter(|f| f.start_ms <= hi && f.end_ms >= lo).collect();
        overlapping.sort_by_key(|f| f.start_ms);
        if !overlapping.is_empty() {
            let call: String = overlapping.iter().map(|f| f.text.as_str()).collect::<Vec<_>>().join(" ");
            if is_echo(&normalize(&seg.text), &call) {
                return Verdict::Drop;
            }
        }
        if expired {
            return Verdict::Keep;
        }
        // Could a matching system final still come?
        let open = self.sys_open.values().any(|&start| start <= hi);
        if open || self.untranscribed_sound(lo, seg.end_ms) {
            Verdict::Wait
        } else {
            Verdict::Keep
        }
    }

    fn active_between(&self, a: u64, b: u64) -> bool {
        self.activity.iter().any(|&(s, e)| s < b && e > a)
    }

    /// System sound in `[a, b)` that no system final covers yet.
    fn untranscribed_sound(&self, a: u64, b: u64) -> bool {
        for &(s, e) in &self.activity {
            let (mut s, e) = (s.max(a), e.min(b));
            if s >= e {
                continue;
            }
            // Walk the finals in time order and skip over covered stretches.
            let mut finals: Vec<(u64, u64)> = self
                .sys_finals
                .iter()
                .map(|f| (f.start_ms.saturating_sub(COVER_SLACK_MS), f.end_ms + COVER_SLACK_MS))
                .filter(|&(fs, fe)| fs < e && fe > s)
                .collect();
            finals.sort_unstable();
            for (fs, fe) in finals {
                if fs > s + MIN_UNCOVERED_MS {
                    return true;
                }
                s = s.max(fe);
            }
            if e > s + MIN_UNCOVERED_MS {
                return true;
            }
        }
        false
    }
}

/// Lowercase letters and digits, single spaces; `ё` folded into `е`.
pub fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars().flat_map(char::to_lowercase) {
        let c = if c == 'ё' { 'е' } else { c };
        if c.is_alphanumeric() {
            out.push(c);
        } else if !out.is_empty() && !out.ends_with(' ') {
            out.push(' ');
        }
    }
    out.truncate(out.trim_end().len());
    out
}

fn trigrams(text: &str) -> HashMap<[char; 3], u32> {
    let chars: Vec<char> = std::iter::once(' ').chain(text.chars()).chain(std::iter::once(' ')).collect();
    let mut grams = HashMap::new();
    for w in chars.windows(3) {
        *grams.entry([w[0], w[1], w[2]]).or_insert(0) += 1;
    }
    grams
}

/// Share of `mic`'s character trigrams that also occur in `call` (both normalized).
/// Containment rather than a symmetric score: a mic final often covers only part of
/// a longer system utterance.
pub fn containment(mic: &str, call: &str) -> f32 {
    let m = trigrams(mic);
    let c = trigrams(call);
    let total: u32 = m.values().sum();
    if total == 0 {
        return 0.0;
    }
    let shared: u32 = m.iter().map(|(g, &n)| n.min(c.get(g).copied().unwrap_or(0))).sum();
    shared as f32 / total as f32
}

fn is_echo(mic: &str, call: &str) -> bool {
    if mic.is_empty() || call.is_empty() {
        return false;
    }
    if !mic.contains(' ') {
        // One word: trigram overlap between unrelated words is common, so require the word itself.
        return call.split(' ').any(|w| w == mic);
    }
    containment(mic, call) >= THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(id: &str, source: Source, start_ms: u64, end_ms: u64, text: &str, is_final: bool) -> Segment {
        Segment { id: id.into(), source, speaker: None, start_ms, end_ms, text: text.into(), is_final }
    }

    fn texts(v: &[Segment]) -> Vec<(String, String, bool)> {
        v.iter().map(|s| (s.id.clone(), s.text.clone(), s.is_final)).collect()
    }

    const CALL: &str = "мы перенесли релиз на следующую среду из за тестов";

    /// Guard with the call audible from 1 s to 5 s.
    fn guard_with_call() -> EchoGuard {
        let mut g = EchoGuard::new();
        for t in (1_000..5_000).step_by(32) {
            g.system_audio(t, t + 32, 0.05);
        }
        g
    }

    #[test]
    fn normalizes_text() {
        assert_eq!(normalize("  Ещё, раз!  Всё  "), "еще раз все");
        assert_eq!(normalize("?!"), "");
    }

    #[test]
    fn containment_scores() {
        assert!((containment("релиз на среду", "релиз на среду") - 1.0).abs() < 1e-6);
        assert!(containment("перенесли релиз", CALL) > 0.9);
        assert!(containment("я думаю это хорошая идея", CALL) < 0.4);
        assert_eq!(containment("", CALL), 0.0);
    }

    #[test]
    fn echo_of_the_call_is_dropped_as_empty_final_after_partials() {
        let t0 = Instant::now();
        let mut g = guard_with_call();
        // The mic transcribes the echo; its partial shows up in the UI.
        assert_eq!(g.push(seg("mic-3", Source::Mic, 1_200, 4_900, "перенесли релиз", false), t0).len(), 1);
        // System partial: the call's utterance is still open.
        assert_eq!(g.push(seg("system-7", Source::System, 1_000, 3_000, "мы перенесли", false), t0).len(), 1);
        // The mic final arrives first and is held.
        assert!(g.push(seg("mic-3", Source::Mic, 1_200, 4_900, "перенесли релиз на следующую среду", true), t0).is_empty());
        assert!(g.poll(t0 + Duration::from_millis(500)).is_empty());
        // The system final is emitted at once, followed by the mic final's verdict.
        let out = g.push(seg("system-7", Source::System, 1_000, 4_800, CALL, true), t0 + Duration::from_millis(600));
        assert_eq!(
            texts(&out),
            vec![("system-7".into(), CALL.into(), true), ("mic-3".into(), String::new(), true)]
        );
        assert_eq!(g.stats().dropped, 1);
    }

    #[test]
    fn dropped_final_without_partials_disappears() {
        let t0 = Instant::now();
        let mut g = guard_with_call();
        g.push(seg("system-1", Source::System, 1_000, 4_800, CALL, true), t0);
        // A split piece: fresh id, never had partials.
        let out = g.push(seg("mic-9", Source::Mic, 2_000, 3_500, "релиз на следующую среду", true), t0);
        assert!(out.is_empty(), "{out:?}");
        assert!(g.poll(t0 + MAX_HOLD * 2).is_empty());
        assert!(g.finish(t0 + MAX_HOLD * 2).is_empty());
    }

    #[test]
    fn users_own_words_are_kept() {
        let t0 = Instant::now();
        let mut g = guard_with_call();
        g.push(seg("system-1", Source::System, 1_000, 4_800, CALL, true), t0);
        g.push(seg("mic-1", Source::Mic, 2_000, 4_000, "а что с", false), t0);
        let out = g.push(seg("mic-1", Source::Mic, 2_000, 4_000, "а что с документацией для клиента", true), t0);
        assert_eq!(texts(&out), vec![("mic-1".into(), "а что с документацией для клиента".into(), true)]);
    }

    #[test]
    fn no_hold_while_the_call_is_quiet() {
        let t0 = Instant::now();
        let mut g = guard_with_call();
        // 10 s: the call has been silent for 5 s, even with a system utterance open elsewhere.
        g.push(seg("system-2", Source::System, 30_000, 31_000, "потом", false), t0);
        let out = g.push(seg("mic-5", Source::Mic, 10_000, 12_000, "перенесли релиз на среду", true), t0);
        assert_eq!(out.len(), 1);
        assert_eq!(g.stats().dropped, 0);
    }

    #[test]
    fn hold_is_bounded_when_no_system_final_comes() {
        let t0 = Instant::now();
        let mut g = guard_with_call();
        // The call had sound (say, music) but nothing was transcribed.
        assert!(g.push(seg("mic-2", Source::Mic, 2_000, 3_000, "какой то текст", true), t0).is_empty());
        assert!(g.poll(t0 + Duration::from_millis(2_000)).is_empty());
        let out = g.poll(t0 + MAX_HOLD);
        assert_eq!(texts(&out), vec![("mic-2".into(), "какой то текст".into(), true)]);
    }

    #[test]
    fn mic_finals_leave_in_order_and_system_is_never_held() {
        let t0 = Instant::now();
        let mut g = guard_with_call();
        assert!(g.push(seg("mic-1", Source::Mic, 1_500, 2_500, "первая фраза пользователя", true), t0).is_empty());
        // A later mic final that alone could go out waits behind the first one.
        g.system_audio(20_000, 20_032, 0.0);
        assert!(g.push(seg("mic-2", Source::Mic, 9_000, 9_500, "вторая", true), t0).is_empty());
        // Partials and system segments pass immediately.
        assert_eq!(g.push(seg("mic-3", Source::Mic, 10_000, 10_500, "тре", false), t0).len(), 1);
        let sys = g.push(seg("system-4", Source::System, 1_000, 4_900, "совсем другие слова звонка", true), t0);
        assert_eq!(sys.len(), 3);
        assert_eq!(sys[0].id, "system-4");
        assert_eq!((sys[1].id.as_str(), sys[2].id.as_str()), ("mic-1", "mic-2"));
    }

    #[test]
    fn one_word_needs_an_exact_match() {
        let t0 = Instant::now();
        let mut g = guard_with_call();
        g.push(seg("system-1", Source::System, 1_000, 4_800, "да конечно давайте так", true), t0);
        g.push(seg("mic-1", Source::Mic, 2_000, 2_400, "да", false), t0);
        let out = g.push(seg("mic-1", Source::Mic, 2_000, 2_400, "да", true), t0);
        assert_eq!(texts(&out), vec![("mic-1".into(), String::new(), true)]);
        let out = g.push(seg("mic-2", Source::Mic, 3_000, 3_400, "дак", true), t0);
        assert_eq!(texts(&out), vec![("mic-2".into(), "дак".into(), true)]);
    }

    #[test]
    fn empty_mic_final_passes_and_finish_flushes() {
        let t0 = Instant::now();
        let mut g = guard_with_call();
        g.push(seg("mic-1", Source::Mic, 1_000, 1_300, "м", false), t0);
        let out = g.push(seg("mic-1", Source::Mic, 1_000, 1_300, "", true), t0);
        assert_eq!(texts(&out), vec![("mic-1".into(), String::new(), true)]);
        assert!(g.push(seg("mic-2", Source::Mic, 2_000, 3_000, "что то свое", true), t0).is_empty());
        let out = g.finish(t0);
        assert_eq!(texts(&out), vec![("mic-2".into(), "что то свое".into(), true)]);
    }

    #[test]
    fn far_away_system_finals_are_not_compared() {
        let t0 = Instant::now();
        let mut g = guard_with_call();
        // Same words, but said by the call 20 s later (outside ±1.5 s).
        g.push(seg("system-1", Source::System, 22_000, 24_000, CALL, true), t0);
        let out = g.finish(t0);
        assert!(out.is_empty());
        let out = g.push(seg("mic-1", Source::Mic, 1_500, 3_000, "перенесли релиз на следующую среду", true), t0);
        assert!(out.is_empty(), "held: the call had untranscribed sound");
        let out = g.poll(t0 + MAX_HOLD);
        assert_eq!(out.len(), 1);
        assert!(!out[0].text.is_empty());
    }
}
