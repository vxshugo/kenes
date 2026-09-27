//! Tauri shell over `kenes-core`. Commands and the `kenes://event` / `kenes://hotkey`
//! payloads are specified in `docs/CONTRACT.md`.

mod hotkeys;
mod platform;
mod secrets;

use std::sync::Arc;

use kenes_core::diarize::{self, VoiceprintStatus};
use kenes_core::{default_data_dir, settings, LiveSession, Meeting, MeetingSummary, SessionManager, Store};
use kenes_types::DeviceInfo;
use serde::Serialize;
use serde_json::Value;
use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};

use hotkeys::{HotkeyConfig, HotkeyStatus, Hotkeys};
use platform::PlatformInfo;
use secrets::Secrets;

const EVENT: &str = "kenes://event";

struct AppState {
    store: Arc<Store>,
    sessions: SessionManager,
    secrets: Secrets,
    /// The window runs under XWayland (`gnomeAlwaysOnTop`); fixed for the process.
    x11_forced: bool,
}

type CmdResult<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    format!("{e:#}")
}

/// Runs blocking work (SQLite, subprocesses, thread joins) off the main thread.
async fn blocking<T: Send + 'static>(f: impl FnOnce() -> anyhow::Result<T> + Send + 'static) -> CmdResult<T> {
    tauri::async_runtime::spawn_blocking(f).await.map_err(err)?.map_err(err)
}

#[tauri::command]
async fn list_devices() -> CmdResult<Vec<DeviceInfo>> {
    blocking(kenes_audio::list_devices).await
}

#[tauri::command]
async fn list_models() -> CmdResult<Vec<kenes_stt::ModelInfo>> {
    blocking(|| Ok(kenes_stt::available_models())).await
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Started {
    meeting_id: String,
}

#[tauri::command]
async fn start_session(state: State<'_, AppState>, title: String, context: String) -> CmdResult<Started> {
    let core = settings::load(&state.store).and_then(|s| settings::core(&s)).map_err(err)?;
    let title = if title.trim().is_empty() { "Встреча".to_owned() } else { title };
    let meeting_id = state.sessions.start(&title, &context, core).map_err(err)?;
    Ok(Started { meeting_id })
}

/// The running session, if any, for a UI that (re)loads mid-meeting.
#[tauri::command]
async fn session_status(app: AppHandle) -> CmdResult<Option<LiveSession>> {
    blocking(move || Ok(app.state::<AppState>().sessions.live())).await
}

#[tauri::command]
async fn stop_session(app: AppHandle) -> CmdResult<()> {
    blocking(move || app.state::<AppState>().sessions.stop()).await
}

#[tauri::command]
async fn get_settings(state: State<'_, AppState>) -> CmdResult<Value> {
    settings::load(&state.store).map_err(err)
}

#[tauri::command]
async fn save_settings(state: State<'_, AppState>, settings: Value) -> CmdResult<()> {
    settings::save(&state.store, &settings).map_err(err)
}

#[tauri::command]
async fn get_api_key(app: AppHandle) -> CmdResult<Option<String>> {
    blocking(move || Ok(app.state::<AppState>().secrets.get())).await
}

#[tauri::command]
async fn set_api_key(app: AppHandle, key: String) -> CmdResult<()> {
    blocking(move || app.state::<AppState>().secrets.set(&key)).await
}

#[tauri::command]
async fn list_meetings(state: State<'_, AppState>) -> CmdResult<Vec<MeetingSummary>> {
    state.store.list_meetings().map_err(err)
}

#[tauri::command]
async fn get_meeting(state: State<'_, AppState>, id: String) -> CmdResult<Meeting> {
    state.store.get_meeting(&id).map_err(err)?.ok_or_else(|| format!("встреча {id} не найдена"))
}

#[tauri::command]
async fn save_note(
    state: State<'_, AppState>,
    meeting_id: String,
    kind: String,
    content: String,
    trigger: Option<String>,
) -> CmdResult<String> {
    if !matches!(kind.as_str(), "hint" | "summary" | "final") {
        return Err(format!("unknown note kind {kind}"));
    }
    state.store.save_note(&meeting_id, &kind, &content, trigger.as_deref()).map_err(err)
}

#[tauri::command]
async fn delete_meeting(state: State<'_, AppState>, id: String) -> CmdResult<()> {
    if state.sessions.current_meeting().as_deref() == Some(id.as_str()) {
        return Err("нельзя удалить идущую встречу".into());
    }
    state.store.delete_meeting(&id).map_err(err)
}

#[tauri::command]
async fn rename_speaker(state: State<'_, AppState>, meeting_id: String, label: String, name: String) -> CmdResult<()> {
    state.store.rename_speaker(&meeting_id, &label, &name).map_err(err)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Enrolled {
    speech_ms: u64,
}

#[tauri::command]
async fn enroll_voice(app: AppHandle, seconds: u64) -> CmdResult<Enrolled> {
    blocking(move || {
        let state = app.state::<AppState>();
        anyhow::ensure!(state.sessions.current_meeting().is_none(), "остановите встречу перед записью голоса");
        let core = settings::core(&settings::load(&state.store)?)?;
        let speech_ms =
            diarize::enroll_voice(&state.store, &kenes_stt::models_dir(), core.mic_device, seconds, core.num_threads)?;
        Ok(Enrolled { speech_ms })
    })
    .await
}

#[tauri::command]
async fn voiceprint_status(state: State<'_, AppState>) -> CmdResult<VoiceprintStatus> {
    diarize::voiceprint_status(&state.store).map_err(err)
}

#[tauri::command]
async fn clear_voiceprint(state: State<'_, AppState>) -> CmdResult<()> {
    diarize::clear_voiceprint(&state.store).map_err(err)
}

/// Applies the global-shortcut settings; resolves once the system has answered (the portal
/// may first show its own dialog).
#[tauri::command]
async fn configure_hotkeys(app: AppHandle, config: HotkeyConfig) -> CmdResult<HotkeyStatus> {
    let hotkeys = app.state::<Hotkeys>();
    Ok(hotkeys.configure(&app, config).await)
}

#[tauri::command]
async fn hotkey_status(hotkeys: State<'_, Hotkeys>) -> CmdResult<HotkeyStatus> {
    Ok(hotkeys.status())
}

#[tauri::command]
async fn platform_info(state: State<'_, AppState>, hotkeys: State<'_, Hotkeys>) -> CmdResult<PlatformInfo> {
    Ok(platform::info(state.x11_forced, hotkeys.backend))
}

fn setup_tray(app: &tauri::App) -> tauri::Result<()> {
    let toggle = MenuItem::with_id(app, "toggle", "Показать / скрыть", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Выйти", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&toggle, &quit])?;
    let mut tray = TrayIconBuilder::with_id("kenes").tooltip("Kenes").menu(&menu).on_menu_event(
        |app, event| match event.id.as_ref() {
            "toggle" => {
                if let Some(w) = app.get_webview_window("main") {
                    if w.is_visible().unwrap_or(false) {
                        let _ = w.hide();
                    } else {
                        let _ = w.show();
                        let _ = w.set_focus();
                    }
                }
            }
            "quit" => app.exit(0),
            _ => {}
        },
    );
    if let Some(icon) = app.default_window_icon() {
        tray = tray.icon(icon.clone());
    }
    tray.build(app)?;
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Settings that must be read before GTK starts (the display backend).
    let data_dir = default_data_dir();
    let store = Store::open(&data_dir.join("kenes.db")).map(Arc::new);
    let stored = store.as_ref().ok().and_then(|s| settings::load(s).ok()).unwrap_or(serde_json::Value::Null);
    let x11_forced = platform::apply(&stored);
    if x11_forced {
        log::info!("GNOME on Wayland: running under XWayland so the window can stay on top (gnomeAlwaysOnTop)");
    }

    let hotkeys = Hotkeys::new(hotkeys::detect_backend());
    let mut builder = tauri::Builder::default().plugin(tauri_plugin_opener::init());
    #[cfg(desktop)]
    if hotkeys.backend == hotkeys::BackendKind::Plugin {
        builder = builder.plugin(hotkeys.plugin());
    }

    let app = builder
        .manage(hotkeys)
        .setup(move |app| {
            let store = store?;
            let handle = app.handle().clone();
            let sink: kenes_core::EventSink = Arc::new(move |ev| {
                if let Err(e) = handle.emit(EVENT, &ev) {
                    log::warn!("emit failed: {e}");
                }
            });
            let sessions = SessionManager::new(store.clone(), kenes_stt::models_dir(), sink);
            app.manage(AppState { store, sessions, secrets: Secrets::new(&data_dir), x11_forced });
            if let Err(e) = setup_tray(app) {
                // Some Linux desktops have no tray; the window still works.
                log::warn!("tray unavailable: {e}");
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            list_devices,
            list_models,
            start_session,
            stop_session,
            session_status,
            get_settings,
            save_settings,
            get_api_key,
            set_api_key,
            list_meetings,
            get_meeting,
            save_note,
            delete_meeting,
            rename_speaker,
            enroll_voice,
            voiceprint_status,
            clear_voiceprint,
            configure_hotkeys,
            hotkey_status,
            platform_info,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app, event| {
        if let RunEvent::Exit = event {
            // Finalize the transcript and end the meeting before the process goes away.
            if let Err(e) = app.state::<AppState>().sessions.stop() {
                log::error!("stopping session on exit: {e:#}");
            }
        }
    });
}
