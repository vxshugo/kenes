//! Kenes core: runs a meeting session (capture → speech recognition) and
//! persists transcripts, notes and settings. The Tauri app and `kenes-cli`
//! are thin shells over this crate.

pub mod diarize;
pub mod echo_guard;
pub mod session;
pub mod settings;
pub mod store;

use std::path::PathBuf;

pub use session::{AudioInput, EventSink, LiveSession, SessionManager};
pub use store::{Meeting, MeetingSummary, Note, Speaker, Store};

/// `$KENES_DATA_DIR`, else the platform data dir (e.g. `~/.local/share/kenes`).
pub fn default_data_dir() -> PathBuf {
    std::env::var_os("KENES_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::data_dir()
                .unwrap_or_else(std::env::temp_dir)
                .join("kenes")
        })
}
