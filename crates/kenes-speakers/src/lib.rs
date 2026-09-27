//! Speaker diarization for kenes meetings of up to ~15 people (see `docs/CONTRACT.md`,
//! section `kenes-speakers`, and `README.md` / `EVAL.md` in this crate).
//!
//! - [`Embedder`]: 16 kHz mono → L2-normalized speaker embedding (sherpa-onnx, the model in
//!   [`SPEAKER_MODEL`], fetched by [`ensure_speaker_model`]).
//! - [`OnlineClusterer`]: live labels per source (`"mic:N"`, `"sys:N"`, `"me"`, or `None`).
//! - [`recluster`]: better labels for the whole meeting once it ends, keeping online labels
//!   (and so user-given names) attached to the same people.
//! - [`change_points`]: where the speaker changes inside one VAD utterance, so it can be cut
//!   into single-speaker pieces before recognition and labeling.

pub mod change;
mod cluster;
mod embedder;
pub mod metrics;
mod model;
mod recluster;

pub use change::change_points;
pub use cluster::{cosine, l2_normalize, ClusterConfig, OnlineClusterer, ME};
pub use embedder::{Embedder, MIN_EMBED_SAMPLES};
pub use model::{
    ensure_speaker_model, is_speaker_model_downloaded, speaker_model_path, SpeakerModel,
    SPEAKER_MODEL, SPEAKER_MODEL_DIR,
};
pub use recluster::{recluster, ClusterItem};
