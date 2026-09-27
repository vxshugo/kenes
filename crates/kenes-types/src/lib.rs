//! Shared types that flow between the kenes crates and the UI.
//!
//! Audio moves between threads as [`AudioChunk`]s (16 kHz mono f32). Speech
//! recognition turns them into [`Segment`]s, which the app forwards to the UI
//! as [`PipelineEvent`]s. The serde shapes here are the wire contract with the
//! TypeScript side (see `docs/CONTRACT.md`), so rename carefully.

use serde::{Deserialize, Serialize};

/// Sample rate of every [`AudioChunk`] in the pipeline.
pub const SAMPLE_RATE: u32 = 16_000;

/// Where audio came from. On a call, `Mic` is the user and `System` is
/// everyone else, which gives us two-speaker diarization for free.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    Mic,
    System,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Mic => "mic",
            Source::System => "system",
        }
    }
}

/// A block of 16 kHz mono PCM samples in `[-1.0, 1.0]`.
#[derive(Clone, Debug)]
pub struct AudioChunk {
    pub source: Source,
    /// Milliseconds since the capture session started, for the first sample.
    pub start_ms: u64,
    pub samples: Vec<f32>,
}

impl AudioChunk {
    pub fn duration_ms(&self) -> u64 {
        self.samples.len() as u64 * 1000 / SAMPLE_RATE as u64
    }
}

/// One utterance of recognized speech.
///
/// While someone is still talking the recognizer emits partial segments
/// (`is_final == false`) that share one `id`; each replaces the previous one.
/// The final segment with that `id` is emitted exactly once.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    pub id: String,
    pub source: Source,
    /// Speaker label, set on final segments once diarization has run:
    /// `"me"` (the user's enrolled voice, or the mic in headset mode),
    /// `"mic:N"` (N-th voice heard by the mic), `"sys:N"` (N-th voice in the call).
    pub speaker: Option<String>,
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    pub is_final: bool,
}

/// Kind of audio device, as the capture layer sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceKind {
    /// A microphone or other input.
    Input,
    /// A loopback/monitor of an output device (what the speakers play).
    Monitor,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeviceInfo {
    /// Backend-specific identifier to pass back into capture config.
    pub id: String,
    pub name: String,
    pub kind: DeviceKind,
    pub is_default: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionState {
    Idle,
    /// Loading or downloading models.
    Loading,
    Running,
    Error,
}

/// A segment whose speaker label changed after the fact (end-of-meeting re-clustering).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerChange {
    pub segment_id: String,
    pub speaker: Option<String>,
}

/// Everything the Rust side pushes to the UI during a session.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum PipelineEvent {
    Segment(Segment),
    /// Input level per source, roughly 10 times a second, for the VU meter.
    Level { source: Source, rms: f32 },
    Status { state: SessionState, message: Option<String> },
    /// Model download progress, `0.0..=1.0`.
    ModelProgress { model: String, progress: f32 },
    Error { message: String },
    /// Speaker labels were revised for already-emitted final segments.
    SpeakersRelabeled { changes: Vec<SpeakerChange> },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_event_wire_shape() {
        let ev = PipelineEvent::Segment(Segment {
            id: "s1".into(),
            source: Source::System,
            speaker: None,
            start_ms: 10,
            end_ms: 20,
            text: "сәлем".into(),
            is_final: true,
        });
        let json = serde_json::to_value(&ev).unwrap();
        assert_eq!(json["type"], "segment");
        assert_eq!(json["source"], "system");
        assert_eq!(json["startMs"], 10);
        assert_eq!(json["isFinal"], true);
    }

    #[test]
    fn relabel_event_wire_shape() {
        let ev = PipelineEvent::SpeakersRelabeled {
            changes: vec![SpeakerChange { segment_id: "system-3".into(), speaker: Some("sys:2".into()) }],
        };
        assert_eq!(
            serde_json::to_value(ev).unwrap(),
            serde_json::json!({"type": "speakersRelabeled", "changes": [{"segmentId": "system-3", "speaker": "sys:2"}]})
        );
    }

    #[test]
    fn level_event_wire_shape() {
        let json = serde_json::to_value(PipelineEvent::Level { source: Source::Mic, rms: 0.5 }).unwrap();
        assert_eq!(json, serde_json::json!({"type": "level", "source": "mic", "rms": 0.5}));
    }
}
