//! App settings: one JSON object shared with the UI (see `docs/CONTRACT.md`).
//!
//! Rust only interprets the audio/STT keys; everything else is kept verbatim so
//! the UI can add fields without a Rust change. `autoHintMode` is deliberately
//! absent: the UI derives its default from `micMode`.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::store::Store;

const KEY: &str = "settings";

pub fn defaults() -> Value {
    json!({
        "sttModel": "gigaam-multilingual-ctc",
        "numThreads": 4,
        "captureMic": true,
        "captureSystem": true,
        "micMode": "me",
        "micDevice": null,
        "systemDevice": null,
        "claudeModel": "claude-opus-5",
        "hintEffort": "low",
        "summaryEffort": "high",
        "myNames": [],
        "rollingSummaryMinutes": 4,
        "answerLanguage": "auto",
        "profile": ""
    })
}

/// Who the microphone hears.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MicMode {
    /// Only the user (headset / online call): every mic segment is "me".
    Me,
    /// Several people (in-person or hybrid): mic voices are clustered.
    Room,
}

/// The part of the settings the Rust pipeline acts on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CoreSettings {
    pub stt_model: String,
    pub num_threads: i32,
    pub capture_mic: bool,
    pub capture_system: bool,
    pub mic_mode: MicMode,
    pub mic_device: Option<String>,
    pub system_device: Option<String>,
}

/// Stored settings layered over the defaults, so new keys get sane values.
pub fn load(store: &Store) -> anyhow::Result<Value> {
    let mut merged = defaults();
    if let Some(raw) = store.get_kv(KEY)? {
        match serde_json::from_str::<Value>(&raw) {
            Ok(Value::Object(saved)) => merge_into(&mut merged, saved),
            Ok(_) | Err(_) => log::warn!("ignoring malformed stored settings"),
        }
    }
    Ok(merged)
}

pub fn save(store: &Store, settings: &Value) -> anyhow::Result<()> {
    anyhow::ensure!(settings.is_object(), "settings must be a JSON object");
    // Validate the keys Rust depends on before persisting anything.
    let mut merged = defaults();
    merge_into(&mut merged, settings.as_object().cloned().unwrap_or_default());
    core(&merged)?;
    store.set_kv(KEY, &serde_json::to_string(&merged)?)
}

pub fn core(settings: &Value) -> anyhow::Result<CoreSettings> {
    let mut c: CoreSettings = serde_json::from_value(settings.clone())
        .map_err(|e| anyhow::anyhow!("invalid settings: {e}"))?;
    c.num_threads = c.num_threads.clamp(1, 16);
    Ok(c)
}

fn merge_into(base: &mut Value, overlay: Map<String, Value>) {
    if let Value::Object(b) = base {
        b.extend(overlay);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_parse_as_core() {
        let c = core(&defaults()).unwrap();
        assert_eq!(c.stt_model, "gigaam-multilingual-ctc");
        assert!(c.capture_mic && c.capture_system);
        assert_eq!(c.mic_device, None);
        assert_eq!(c.mic_mode, MicMode::Me);
    }

    #[test]
    fn save_keeps_ui_keys_and_fills_defaults() {
        let store = Store::open_in_memory().unwrap();
        save(&store, &json!({"numThreads": 64, "uiOnly": {"x": 1}})).unwrap();
        let s = load(&store).unwrap();
        assert_eq!(s["uiOnly"]["x"], 1);
        assert_eq!(s["claudeModel"], "claude-opus-5");
        assert_eq!(core(&s).unwrap().num_threads, 16);
    }

    #[test]
    fn save_rejects_wrong_types() {
        let store = Store::open_in_memory().unwrap();
        assert!(save(&store, &json!({"captureMic": "yes"})).is_err());
        assert!(save(&store, &json!({"micMode": "crowd"})).is_err());
        assert!(save(&store, &json!([1, 2])).is_err());
    }
}
