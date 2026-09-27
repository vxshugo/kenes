//! System-wide shortcuts («Что ответить?», «Кратко: 5 мин», show/hide the panel) that work
//! while another app (the call) has focus.
//!
//! Backends, picked once at start-up:
//! - Wayland session (Linux): the XDG GlobalShortcuts portal via `ashpd`. The compositor owns
//!   the bindings: it shows its own dialog the first time and may assign other keys than the
//!   ones we suggest. GNOME needs an app id for that, which a host (non-Flatpak) app gets by
//!   registering with `org.freedesktop.host.portal.Registry`, and that needs a desktop entry
//!   named after the id (see [`portal::ensure_desktop_entry`]).
//! - macOS and X11: `tauri-plugin-global-shortcut` (a key grab with exactly the configured keys).
//! - Anything else: none. The in-app shortcuts keep working either way.
//!
//! Every activation is emitted as `kenes://hotkey` `{ action }`; the UI runs hint/recap.
//! "toggle" is handled here (the webview of a hidden window can't show itself).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, Runtime};

pub const HOTKEY_EVENT: &str = "kenes://hotkey";
pub const HOTKEY_STATUS_EVENT: &str = "kenes://hotkey-status";

/// The desktop entry / portal app id. Matches `identifier` in `tauri.conf.json`.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub const APP_ID: &str = "kz.kenes.app";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Action {
    Hint,
    Recap,
    Toggle,
}

impl Action {
    pub const ALL: [Action; 3] = [Action::Hint, Action::Recap, Action::Toggle];

    pub fn id(self) -> &'static str {
        match self {
            Action::Hint => "hint",
            Action::Recap => "recap",
            Action::Toggle => "toggle",
        }
    }

    pub fn from_id(id: &str) -> Option<Action> {
        Action::ALL.into_iter().find(|a| a.id() == id)
    }

    /// Shown by the system in its shortcut dialog and settings.
    pub fn description(self) -> &'static str {
        match self {
            Action::Hint => "Kenes: что ответить?",
            Action::Recap => "Kenes: кратко за 5 минут",
            Action::Toggle => "Kenes: показать или скрыть окно",
        }
    }
}

/// What the UI asks for (`configure_hotkeys`). Accelerators look like
/// `CommandOrControl+Alt+Enter`: modifiers, then one key in `KeyboardEvent.code` style.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HotkeyConfig {
    pub enabled: bool,
    pub hint: String,
    pub recap: String,
    pub toggle: String,
}

impl HotkeyConfig {
    pub fn accelerator(&self, action: Action) -> &str {
        match action {
            Action::Hint => &self.hint,
            Action::Recap => &self.recap,
            Action::Toggle => &self.toggle,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    Portal,
    Plugin,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))] // Pending/Cancelled come from the portal
pub enum HotkeyState {
    /// Disabled in the settings.
    Off,
    /// Waiting for the user in the system's shortcut dialog.
    Pending,
    Active,
    /// The user closed the system dialog without confirming.
    Cancelled,
    /// No usable backend (no portal, not a desktop session, …).
    Unavailable,
    Error,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BoundHotkey {
    pub action: Action,
    /// How to trigger it: the system's description (portal) or our accelerator (plugin).
    /// `None` = not bound.
    pub trigger: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HotkeyStatus {
    pub backend: BackendKind,
    pub state: HotkeyState,
    pub bindings: Vec<BoundHotkey>,
    /// Russian, for the settings screen.
    pub message: Option<String>,
}

impl HotkeyStatus {
    fn new(backend: BackendKind, state: HotkeyState, message: Option<String>) -> Self {
        let bindings = Action::ALL.into_iter().map(|action| BoundHotkey { action, trigger: None }).collect();
        Self { backend, state, bindings, message }
    }
}

#[derive(Clone, Serialize)]
struct HotkeyPayload {
    action: Action,
}

/// Which backend this process uses. Wayland wins even when the window itself runs under
/// XWayland (`GDK_BACKEND=x11`): an X11 key grab only sees keys while an X11 window has focus.
pub fn detect_backend() -> BackendKind {
    if cfg!(target_os = "macos") || cfg!(target_os = "windows") {
        return BackendKind::Plugin;
    }
    if cfg!(target_os = "linux") {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let wayland = env("XDG_SESSION_TYPE").as_deref() == Some("wayland") || env("WAYLAND_DISPLAY").is_some();
        if wayland {
            return BackendKind::Portal;
        }
        if env("DISPLAY").is_some() {
            return BackendKind::Plugin;
        }
    }
    BackendKind::None
}

/// Runs an action: shows/hides the window for "toggle", makes sure the panel is visible for the
/// others (without taking focus from the call), and tells the UI.
pub fn fire<R: Runtime>(app: &AppHandle<R>, action: Action) {
    log::info!("global shortcut: {}", action.id());
    if let Some(w) = app.get_webview_window("main") {
        let visible = w.is_visible().unwrap_or(true);
        match action {
            Action::Toggle if visible => {
                let _ = w.hide();
            }
            Action::Toggle => {
                let _ = w.show();
                let _ = w.set_focus();
            }
            _ if !visible => {
                let _ = w.show();
            }
            _ => {}
        }
    }
    if let Err(e) = app.emit(HOTKEY_EVENT, HotkeyPayload { action }) {
        log::warn!("emit {HOTKEY_EVENT} failed: {e}");
    }
}

/// Shared state behind the `configure_hotkeys` / `hotkey_status` commands.
pub struct Hotkeys {
    pub backend: BackendKind,
    status: Mutex<HotkeyStatus>,
    applied: Mutex<Option<HotkeyConfig>>,
    /// Plugin backend: registered hotkey id → action.
    plugin_actions: Arc<Mutex<HashMap<u32, Action>>>,
    /// Serializes reconfiguration (a portal bind can wait for the user for a while).
    apply_lock: tokio::sync::Mutex<()>,
    /// Bumped when new settings arrive: a portal bind still waiting for the user gives up
    /// (its system dialog is closed) so the new settings apply right away.
    cancel_bind: tokio::sync::watch::Sender<u64>,
    #[cfg(target_os = "linux")]
    portal: tokio::sync::Mutex<portal::PortalState>,
}

impl Hotkeys {
    pub fn new(backend: BackendKind) -> Self {
        let state = if backend == BackendKind::None { HotkeyState::Unavailable } else { HotkeyState::Off };
        let message = (backend == BackendKind::None)
            .then(|| "Глобальные сочетания недоступны в этой системе. Сочетания внутри окна работают.".to_owned());
        Self {
            backend,
            status: Mutex::new(HotkeyStatus::new(backend, state, message)),
            applied: Mutex::new(None),
            plugin_actions: Arc::new(Mutex::new(HashMap::new())),
            apply_lock: tokio::sync::Mutex::new(()),
            cancel_bind: tokio::sync::watch::Sender::new(0),
            #[cfg(target_os = "linux")]
            portal: tokio::sync::Mutex::new(portal::PortalState::default()),
        }
    }

    pub fn status(&self) -> HotkeyStatus {
        self.status.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn set_status<R: Runtime>(&self, app: &AppHandle<R>, status: HotkeyStatus) -> HotkeyStatus {
        *self.status.lock().unwrap_or_else(|e| e.into_inner()) = status.clone();
        if let Err(e) = app.emit(HOTKEY_STATUS_EVENT, &status) {
            log::warn!("emit {HOTKEY_STATUS_EVENT} failed: {e}");
        }
        status
    }

    /// The plugin backend's global handler; install it with the plugin at start-up.
    #[cfg(desktop)]
    pub fn plugin<R: Runtime>(&self) -> tauri::plugin::TauriPlugin<R> {
        use tauri_plugin_global_shortcut::ShortcutState;
        let actions = self.plugin_actions.clone();
        tauri_plugin_global_shortcut::Builder::new()
            .with_handler(move |app, shortcut, event| {
                if event.state != ShortcutState::Pressed {
                    return;
                }
                let action = actions.lock().unwrap_or_else(|e| e.into_inner()).get(&shortcut.id()).copied();
                if let Some(action) = action {
                    fire(app, action);
                }
            })
            .build()
    }

    /// Applies the settings. Re-applying the same settings is a no-op (e.g. after a webview
    /// reload); while the system dialog for them is open, it waits for that answer instead.
    pub async fn configure<R: Runtime>(&self, app: &AppHandle<R>, cfg: HotkeyConfig) -> HotkeyStatus {
        if self.applied.lock().unwrap_or_else(|e| e.into_inner()).as_ref() != Some(&cfg) {
            self.cancel_bind.send_modify(|g| *g = g.wrapping_add(1));
        }
        let _guard = self.apply_lock.lock().await;
        let same = self.applied.lock().unwrap_or_else(|e| e.into_inner()).as_ref() == Some(&cfg);
        let current = self.status();
        if same && matches!(current.state, HotkeyState::Active | HotkeyState::Off | HotkeyState::Unavailable) {
            return current;
        }
        *self.applied.lock().unwrap_or_else(|e| e.into_inner()) = Some(cfg.clone());
        match self.backend {
            BackendKind::None => current,
            BackendKind::Plugin => {
                let status = self.apply_plugin(app, &cfg);
                self.set_status(app, status)
            }
            BackendKind::Portal => self.apply_portal(app, &cfg).await,
        }
    }

    #[cfg(desktop)]
    fn apply_plugin<R: Runtime>(&self, app: &AppHandle<R>, cfg: &HotkeyConfig) -> HotkeyStatus {
        use std::str::FromStr;
        use tauri_plugin_global_shortcut::{GlobalShortcut, Shortcut};

        let Some(gs) = app.try_state::<GlobalShortcut<R>>() else {
            return HotkeyStatus::new(BackendKind::Plugin, HotkeyState::Unavailable, Some("Модуль сочетаний не загружен.".into()));
        };
        // `unregister_all`/`register` block until the thread that also runs the shortcut handler
        // (the main thread on macOS, the X11 event thread on Linux) has done the work, and that
        // handler locks `plugin_actions`. Holding the lock across these calls would deadlock the
        // app if a shortcut is pressed meanwhile, so it is only taken for the map updates.
        if let Err(e) = gs.unregister_all() {
            log::warn!("unregistering global shortcuts: {e}");
        }
        self.plugin_actions.lock().unwrap_or_else(|e| e.into_inner()).clear();
        if !cfg.enabled {
            return HotkeyStatus::new(BackendKind::Plugin, HotkeyState::Off, None);
        }
        let mut status = HotkeyStatus::new(BackendKind::Plugin, HotkeyState::Active, None);
        let mut problems = Vec::new();
        let mut bound = std::collections::HashSet::new();
        for (i, action) in Action::ALL.into_iter().enumerate() {
            let accel = cfg.accelerator(action).trim();
            if accel.is_empty() {
                continue;
            }
            let shortcut = match Shortcut::from_str(accel) {
                Ok(s) => s,
                Err(e) => {
                    problems.push(format!("«{accel}»: {e}"));
                    continue;
                }
            };
            if bound.contains(&shortcut.id()) {
                problems.push(format!("«{accel}» назначено дважды"));
                continue;
            }
            match gs.register(shortcut) {
                Ok(()) => {
                    bound.insert(shortcut.id());
                    self.plugin_actions.lock().unwrap_or_else(|e| e.into_inner()).insert(shortcut.id(), action);
                    status.bindings[i].trigger = Some(accel.to_owned());
                }
                Err(e) => problems.push(format!("«{accel}»: {e}")),
            }
        }
        if !problems.is_empty() {
            status.message = Some(format!("Не удалось назначить: {}. Возможно, сочетание занято другим приложением.", problems.join("; ")));
            if bound.is_empty() {
                status.state = HotkeyState::Error;
            }
        }
        status
    }

    #[cfg(not(desktop))]
    fn apply_plugin<R: Runtime>(&self, _app: &AppHandle<R>, _cfg: &HotkeyConfig) -> HotkeyStatus {
        HotkeyStatus::new(BackendKind::Plugin, HotkeyState::Unavailable, None)
    }

    #[cfg(target_os = "linux")]
    async fn apply_portal<R: Runtime>(&self, app: &AppHandle<R>, cfg: &HotkeyConfig) -> HotkeyStatus {
        let mut portal = self.portal.lock().await;
        portal.close().await;
        if !cfg.enabled {
            return self.set_status(app, HotkeyStatus::new(BackendKind::Portal, HotkeyState::Off, None));
        }
        self.set_status(
            app,
            HotkeyStatus::new(
                BackendKind::Portal,
                HotkeyState::Pending,
                Some("Подтвердите сочетания в системном окне.".into()),
            ),
        );
        let status = portal.bind(app, cfg, self.cancel_bind.subscribe()).await;
        self.set_status(app, status)
    }

    #[cfg(not(target_os = "linux"))]
    async fn apply_portal<R: Runtime>(&self, _app: &AppHandle<R>, _cfg: &HotkeyConfig) -> HotkeyStatus {
        HotkeyStatus::new(BackendKind::Portal, HotkeyState::Unavailable, None)
    }

    /// Portal: the system changed the bindings (e.g. in GNOME Settings).
    #[cfg(target_os = "linux")]
    fn update_triggers<R: Runtime>(&self, app: &AppHandle<R>, triggers: &[(String, String)]) {
        let mut status = self.status();
        for (id, trigger) in triggers {
            if let Some(b) = status.bindings.iter_mut().find(|b| b.action.id() == id) {
                b.trigger = (!trigger.is_empty()).then(|| trigger.clone());
            }
        }
        self.set_status(app, status);
    }
}

/// `CommandOrControl+Alt+Enter` → `CTRL+ALT+Return`: the trigger format of the XDG shortcuts
/// spec (modifiers, then an xkb keysym name), for the portal's `preferred_trigger`.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub fn to_xdg_trigger(accelerator: &str) -> Option<String> {
    let mut mods: Vec<&str> = Vec::new();
    let mut key: Option<String> = None;
    for token in accelerator.split('+').map(str::trim) {
        if token.is_empty() || key.is_some() {
            return None;
        }
        let m = match token.to_ascii_uppercase().as_str() {
            "COMMANDORCONTROL" | "COMMANDORCTRL" | "CMDORCTRL" | "CMDORCONTROL" | "CONTROL" | "CTRL" => Some("CTRL"),
            "ALT" | "OPTION" => Some("ALT"),
            "SHIFT" => Some("SHIFT"),
            "SUPER" | "META" | "COMMAND" | "CMD" => Some("LOGO"),
            _ => None,
        };
        match m {
            Some(m) if !mods.contains(&m) => mods.push(m),
            Some(_) => {}
            None => key = Some(xkb_keysym(token)?),
        }
    }
    let key = key?;
    mods.sort_by_key(|m| ["CTRL", "ALT", "SHIFT", "LOGO"].iter().position(|x| x == m));
    mods.push(&key);
    Some(mods.join("+"))
}

/// `KeyboardEvent.code`-style key name → xkb keysym name.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn xkb_keysym(code: &str) -> Option<String> {
    let upper = code.to_ascii_uppercase();
    if let Some(letter) = upper.strip_prefix("KEY").filter(|l| l.len() == 1 && l.as_bytes()[0].is_ascii_uppercase()) {
        return Some(letter.to_ascii_lowercase());
    }
    if upper.len() == 1 && upper.as_bytes()[0].is_ascii_alphabetic() {
        return Some(upper.to_ascii_lowercase());
    }
    if let Some(d) = upper.strip_prefix("DIGIT").filter(|d| d.len() == 1 && d.as_bytes()[0].is_ascii_digit()) {
        return Some(d.to_owned());
    }
    if upper.len() == 1 && upper.as_bytes()[0].is_ascii_digit() {
        return Some(upper);
    }
    if let Some(n) = upper.strip_prefix('F').and_then(|n| n.parse::<u8>().ok()).filter(|n| (1..=24).contains(n)) {
        return Some(format!("F{n}"));
    }
    let name = match upper.as_str() {
        "ENTER" => "Return",
        "SPACE" => "space",
        "TAB" => "Tab",
        "ESCAPE" | "ESC" => "Escape",
        "BACKSPACE" => "BackSpace",
        "DELETE" => "Delete",
        "INSERT" => "Insert",
        "HOME" => "Home",
        "END" => "End",
        "PAGEUP" => "Prior",
        "PAGEDOWN" => "Next",
        "ARROWUP" | "UP" => "Up",
        "ARROWDOWN" | "DOWN" => "Down",
        "ARROWLEFT" | "LEFT" => "Left",
        "ARROWRIGHT" | "RIGHT" => "Right",
        "MINUS" => "minus",
        "EQUAL" => "equal",
        "COMMA" => "comma",
        "PERIOD" => "period",
        "SLASH" => "slash",
        "BACKSLASH" => "backslash",
        "SEMICOLON" => "semicolon",
        "QUOTE" => "apostrophe",
        "BACKQUOTE" => "grave",
        "BRACKETLEFT" => "bracketleft",
        "BRACKETRIGHT" => "bracketright",
        _ => return None,
    };
    Some(name.to_owned())
}

#[cfg(target_os = "linux")]
pub mod portal {
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use ashpd::desktop::global_shortcuts::{GlobalShortcuts, NewShortcut};
    use ashpd::desktop::{CreateSessionOptions, ResponseError, Session};
    use ashpd::zbus;
    use futures_util::StreamExt;
    use tauri::{AppHandle, Manager, Runtime};

    use super::{fire, to_xdg_trigger, Action, BackendKind, HotkeyConfig, HotkeyState, HotkeyStatus, Hotkeys, APP_ID};

    /// How long the system dialog may stay open before we close it and give up on this attempt.
    const BIND_TIMEOUT: Duration = Duration::from_secs(600);

    #[derive(Default)]
    pub struct PortalState {
        /// A connection registered as `APP_ID`; kept for the life of the process.
        conn: Option<zbus::Connection>,
        proxy: Option<GlobalShortcuts>,
        session: Option<Session<GlobalShortcuts>>,
        listeners: Vec<tauri::async_runtime::JoinHandle<()>>,
    }

    fn status(state: HotkeyState, message: impl Into<Option<String>>) -> HotkeyStatus {
        HotkeyStatus::new(BackendKind::Portal, state, message.into())
    }

    impl PortalState {
        pub async fn close(&mut self) {
            for l in self.listeners.drain(..) {
                l.abort();
            }
            if let Some(session) = self.session.take() {
                if let Err(e) = session.close().await {
                    log::debug!("closing shortcuts session: {e}");
                }
            }
        }

        /// A portal proxy on a connection that is registered under our app id. GNOME refuses
        /// global shortcuts to a host app without one ("An app id is required").
        async fn ensure_proxy(&mut self) -> Result<(), HotkeyStatus> {
            if self.proxy.is_some() {
                return Ok(());
            }
            if let Err(e) = ensure_desktop_entry() {
                log::warn!("desktop entry for {APP_ID}: {e:#}");
            }
            let app_id = ashpd::AppID::try_from(APP_ID).map_err(|e| status(HotkeyState::Error, format!("{e}")))?;
            let no_bus = |e: zbus::Error| status(HotkeyState::Unavailable, format!("Нет сессионной шины D-Bus: {e}"));
            let mut last_err = None;
            // A desktop entry written just now can take a moment to be noticed by the portal.
            // Each attempt uses a fresh connection: a failed registration taints the old one.
            for attempt in 0..4 {
                if attempt > 0 {
                    tokio::time::sleep(Duration::from_millis(400)).await;
                }
                let conn = zbus::Connection::session().await.map_err(no_bus)?;
                match ashpd::register_host_app_with_connection(conn.clone(), app_id.clone()).await {
                    Ok(()) => {
                        last_err = None;
                        self.conn = Some(conn);
                        break;
                    }
                    Err(e) => last_err = Some(e),
                }
            }
            let conn = match (last_err, self.conn.clone()) {
                (None, Some(conn)) => conn,
                (err, _) => {
                    // Older portals have no registry; try anyway on a plain connection.
                    if let Some(e) = err {
                        log::warn!("registering {APP_ID} with the portal: {e}");
                    }
                    let conn = zbus::Connection::session().await.map_err(no_bus)?;
                    self.conn = Some(conn.clone());
                    conn
                }
            };
            let proxy = GlobalShortcuts::with_connection(conn).await.map_err(|e| {
                log::warn!("GlobalShortcuts portal unavailable: {e}");
                status(
                    HotkeyState::Unavailable,
                    "В системе нет портала глобальных сочетаний (xdg-desktop-portal GlobalShortcuts). Сочетания внутри окна Kenes работают."
                        .to_owned(),
                )
            })?;
            log::info!("GlobalShortcuts portal version {}", proxy.version());
            self.proxy = Some(proxy);
            Ok(())
        }

        pub async fn bind<R: Runtime>(
            &mut self,
            app: &AppHandle<R>,
            cfg: &HotkeyConfig,
            cancel: tokio::sync::watch::Receiver<u64>,
        ) -> HotkeyStatus {
            if let Err(s) = self.ensure_proxy().await {
                return s;
            }
            let proxy = self.proxy.take().expect("ensured above");
            let conn = self.conn.clone().expect("set with the proxy");
            let bound = bind_with(&proxy, &conn, app, cfg, cancel).await;
            self.proxy = Some(proxy);
            self.session = bound.session;
            self.listeners = bound.listeners;
            match bound.status {
                Ok(s) => s,
                Err(s) => {
                    self.close().await;
                    s
                }
            }
        }
    }

    struct Bound {
        status: Result<HotkeyStatus, HotkeyStatus>,
        session: Option<Session<GlobalShortcuts>>,
        listeners: Vec<tauri::async_runtime::JoinHandle<()>>,
    }

    /// One session: subscribe, bind (the system may show its dialog), report what got bound.
    async fn bind_with<R: Runtime>(
        proxy: &GlobalShortcuts,
        conn: &zbus::Connection,
        app: &AppHandle<R>,
        cfg: &HotkeyConfig,
        mut cancel: tokio::sync::watch::Receiver<u64>,
    ) -> Bound {
        let fail = |state, msg: String| Bound { status: Err(status(state, msg)), session: None, listeners: Vec::new() };
        let session = match proxy.create_session(CreateSessionOptions::default()).await {
            Ok(s) => s,
            Err(e) => {
                log::warn!("GlobalShortcuts.CreateSession: {e}");
                return fail(HotkeyState::Error, format!("Система отказала в глобальных сочетаниях: {e}"));
            }
        };
        // Listen before binding so no activation is missed.
        let streams = futures_util::try_join!(proxy.receive_activated(), proxy.receive_shortcuts_changed());
        let (mut activated, mut changed) = match streams {
            Ok(s) => s,
            Err(e) => {
                return Bound {
                    status: Err(status(HotkeyState::Error, format!("Не удалось подписаться на сочетания: {e}"))),
                    session: Some(session),
                    listeners: Vec::new(),
                }
            }
        };
        let handle = app.clone();
        let on_activate = tauri::async_runtime::spawn(async move {
            while let Some(a) = activated.next().await {
                if let Some(action) = Action::from_id(a.shortcut_id()) {
                    fire(&handle, action);
                }
            }
        });
        let handle = app.clone();
        let on_change = tauri::async_runtime::spawn(async move {
            while let Some(c) = changed.next().await {
                let triggers: Vec<(String, String)> =
                    c.shortcuts().iter().map(|s| (s.id().to_owned(), s.trigger_description().to_owned())).collect();
                handle.state::<Hotkeys>().update_triggers(&handle, &triggers);
            }
        });
        let listeners = vec![on_activate, on_change];

        let shortcuts: Vec<NewShortcut> = Action::ALL
            .into_iter()
            .map(|a| {
                let preferred = to_xdg_trigger(cfg.accelerator(a));
                NewShortcut::new(a.id(), a.description()).preferred_trigger(preferred.as_deref())
            })
            .collect();
        let request = proxy.bind_shortcuts(&session, &shortcuts, None, Default::default());
        enum Outcome<T> {
            Answered(T),
            Superseded,
            TimedOut,
        }
        let outcome = tokio::select! {
            r = request => Outcome::Answered(r),
            _ = cancel.changed() => Outcome::Superseded,
            _ = tokio::time::sleep(BIND_TIMEOUT) => Outcome::TimedOut,
        };
        let status = match outcome {
            Outcome::Superseded | Outcome::TimedOut => {
                // Closing the session is not enough: the portal keeps the request and its dialog.
                close_requests(conn).await;
                let message = if matches!(outcome, Outcome::TimedOut) {
                    "Системное окно сочетаний осталось без ответа и закрыто. Нажмите «Назначить сочетания»."
                } else {
                    "Отменено: настройки изменились."
                };
                Err(status(HotkeyState::Error, message.to_owned()))
            }
            Outcome::Answered(request) => match request.and_then(|r| r.response()) {
                Ok(bound) => {
                    let mut st = status(HotkeyState::Active, None);
                    for b in st.bindings.iter_mut() {
                        b.trigger = bound
                            .shortcuts()
                            .iter()
                            .find(|s| s.id() == b.action.id())
                            .map(|s| s.trigger_description().to_owned())
                            .filter(|t| !t.is_empty());
                    }
                    if st.bindings.iter().all(|b| b.trigger.is_none()) {
                        st.message = Some("Система не назначила ни одного сочетания. Назначьте их в настройках GNOME: «Приложения» → Kenes.".into());
                    }
                    Ok(st)
                }
                Err(ashpd::Error::Response(ResponseError::Cancelled)) => Err(status(
                    HotkeyState::Cancelled,
                    "Сочетания не назначены: системное окно закрыли без подтверждения.".to_owned(),
                )),
                Err(e) => {
                    log::warn!("GlobalShortcuts.BindShortcuts: {e}");
                    Err(status(HotkeyState::Error, format!("Не удалось назначить сочетания: {e}")))
                }
            },
        };
        Bound { status, session: Some(session), listeners }
    }

    /// Closes this connection's pending portal requests (`org.freedesktop.portal.Request.Close`),
    /// which also dismisses their dialogs. Requests live under `…/request/<sender>/<token>`.
    async fn close_requests(conn: &zbus::Connection) {
        let Some(name) = conn.unique_name() else { return };
        let base = format!("/org/freedesktop/portal/desktop/request/{}", name.trim_start_matches(':').replace('.', "_"));
        let introspect = async {
            zbus::fdo::IntrospectableProxy::builder(conn)
                .destination("org.freedesktop.portal.Desktop")?
                .path(base.as_str())?
                .build()
                .await?
                .introspect()
                .await
                .map_err(zbus::Error::from)
        };
        let xml = match introspect.await {
            Ok(xml) => xml,
            Err(e) => {
                log::debug!("listing portal requests: {e}");
                return;
            }
        };
        for node in child_nodes(&xml) {
            let path = format!("{base}/{node}");
            let closed = conn
                .call_method(Some("org.freedesktop.portal.Desktop"), path.as_str(), Some("org.freedesktop.portal.Request"), "Close", &())
                .await;
            match closed {
                Ok(_) => log::info!("closed pending portal request {path}"),
                Err(e) => log::debug!("closing portal request {path}: {e}"),
            }
        }
    }

    /// Child node names from D-Bus introspection XML.
    fn child_nodes(xml: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut rest = xml;
        while let Some(i) = rest.find("<node name=\"") {
            rest = &rest[i + 12..];
            if let Some(end) = rest.find('"') {
                let name = &rest[..end];
                if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                    out.push(name.to_owned());
                }
                rest = &rest[end..];
            }
        }
        out
    }

    /// The portal only registers host apps that have a desktop entry named after their id.
    /// Packaged builds may install theirs under another name, so write a small one into
    /// `$XDG_DATA_HOME/applications` when no `kz.kenes.app.desktop` exists anywhere on the
    /// XDG data path. A file we wrote (marked `X-Kenes-Generated`) is refreshed when the
    /// executable moved; one we didn't write is left alone.
    pub fn ensure_desktop_entry() -> anyhow::Result<()> {
        let file_name = format!("{APP_ID}.desktop");
        let home_dir = data_home()?.join("applications");
        let ours = home_dir.join(&file_name);
        for dir in data_dirs() {
            let candidate = dir.join("applications").join(&file_name);
            if candidate != ours && candidate.is_file() {
                return Ok(());
            }
        }
        let exe = std::env::current_exe()?;
        let wanted = desktop_entry(&exe, cfg!(debug_assertions));
        if !should_write_entry(std::fs::read(&ours), &wanted)? {
            return Ok(());
        }
        std::fs::create_dir_all(&home_dir)?;
        std::fs::write(&ours, wanted)?;
        log::info!("wrote {} for the global shortcuts portal", ours.display());
        Ok(())
    }

    /// Whether to (re)write our entry, given what is at its path now.
    /// Only a missing file or an outdated one of ours is written; ours are always UTF-8.
    fn should_write_entry(existing: std::io::Result<Vec<u8>>, wanted: &str) -> anyhow::Result<bool> {
        match existing {
            Ok(bytes) => Ok(std::str::from_utf8(&bytes).is_ok_and(|s| s != wanted && s.contains("X-Kenes-Generated=true"))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(true),
            // Unreadable: whose it is can't be told, so it is left alone.
            Err(e) => Err(anyhow::Error::new(e).context("reading the existing desktop entry")),
        }
    }

    fn data_home() -> anyhow::Result<PathBuf> {
        if let Some(d) = std::env::var_os("XDG_DATA_HOME").filter(|d| !d.is_empty()) {
            return Ok(PathBuf::from(d));
        }
        let home = std::env::var_os("HOME").ok_or_else(|| anyhow::anyhow!("HOME is not set"))?;
        Ok(Path::new(&home).join(".local/share"))
    }

    fn data_dirs() -> Vec<PathBuf> {
        let raw = std::env::var("XDG_DATA_DIRS").ok().filter(|d| !d.is_empty());
        raw.as_deref().unwrap_or("/usr/local/share:/usr/share").split(':').filter(|d| !d.is_empty()).map(PathBuf::from).collect()
    }

    pub fn desktop_entry(exe: &Path, dev: bool) -> String {
        let mut s = String::from("[Desktop Entry]\nType=Application\n");
        s.push_str(if dev { "Name=Kenes (dev)\n" } else { "Name=Kenes\n" });
        s.push_str("Comment=Ассистент для встреч на русском и казахском\n");
        s.push_str(&format!("Exec={}\n", quote_exec(&exe.to_string_lossy())));
        s.push_str("Terminal=false\nCategories=Office;\nStartupWMClass=kenes-app\n");
        if dev {
            // A dev binary needs the Vite server; keep it out of the app grid.
            s.push_str("NoDisplay=true\n");
        }
        s.push_str("X-Kenes-Generated=true\n");
        s
    }

    /// Quoting per the Desktop Entry spec: the `Exec` rule puts `\` before `"`, `` ` ``, `$` and
    /// `\`, and the string escapes apply on top of that, so each such backslash is written `\\`
    /// (the spec: a literal `$` in a quoted argument is `\\$`, a literal `\` is `\\\\`). GLib
    /// rejects the whole value for a bare `\$`. Control characters get string escapes.
    fn quote_exec(path: &str) -> String {
        let plain = path.chars().all(|c| c.is_ascii_alphanumeric() || "/._-+".contains(c));
        if plain {
            return path.to_owned();
        }
        let mut out = String::from("\"");
        for c in path.chars() {
            match c {
                '"' | '`' | '$' => {
                    out.push_str("\\\\");
                    out.push(c);
                }
                '\\' => out.push_str("\\\\\\\\"),
                '%' => out.push_str("%%"),
                '\n' => out.push_str("\\n"),
                '\t' => out.push_str("\\t"),
                '\r' => out.push_str("\\r"),
                _ => out.push(c),
            }
        }
        out.push('"');
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn child_nodes_come_from_introspection_xml() {
            let xml = r#"<node><interface name="org.freedesktop.DBus.Introspectable"/><node name="ashpd_MkN4wRYaLM"/><node name="t2"></node></node>"#;
            assert_eq!(child_nodes(xml), vec!["ashpd_MkN4wRYaLM", "t2"]);
            assert!(child_nodes("<node/>").is_empty());
        }

        #[test]
        fn desktop_entry_quotes_paths_and_hides_dev_builds() {
            let e = desktop_entry(Path::new("/opt/kenes/kenes-app"), false);
            assert!(e.contains("Exec=/opt/kenes/kenes-app\n"));
            assert!(e.contains("Name=Kenes\n"));
            assert!(!e.contains("NoDisplay"));
            let d = desktop_entry(Path::new("/home/a b/$x/kenes-app"), true);
            assert!(d.contains("Exec=\"/home/a b/\\\\$x/kenes-app\"\n"), "{d}");
            assert!(d.contains("NoDisplay=true"));
            assert!(d.contains("X-Kenes-Generated=true"));
        }

        /// An `Exec` value read as the Desktop Entry spec says: string escapes first (anything but
        /// `\s \n \t \r \\` is invalid, and GLib's key file parser rejects the value), then quoting.
        fn exec_argv(value: &str) -> Option<Vec<String>> {
            let mut s = String::new();
            let mut chars = value.chars();
            while let Some(c) = chars.next() {
                if c != '\\' {
                    s.push(c);
                    continue;
                }
                s.push(match chars.next()? {
                    's' => ' ',
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    '\\' => '\\',
                    _ => return None,
                });
            }
            let mut args = Vec::new();
            let mut chars = s.chars().peekable();
            while let Some(&c) = chars.peek() {
                if c == ' ' {
                    chars.next();
                    continue;
                }
                let mut arg = String::new();
                if c == '"' {
                    chars.next();
                    loop {
                        match chars.next()? {
                            '"' => break,
                            '\\' => {
                                let e = chars.next()?;
                                if !"\"`$\\".contains(e) {
                                    return None;
                                }
                                arg.push(e);
                            }
                            ch => arg.push(ch),
                        }
                    }
                } else {
                    while let Some(&ch) = chars.peek() {
                        if ch == ' ' {
                            break;
                        }
                        arg.push(ch);
                        chars.next();
                    }
                }
                args.push(arg.replace("%%", "%"));
            }
            Some(args)
        }

        #[test]
        fn exec_line_reads_back_as_the_executable_path() {
            let paths = [
                "/opt/kenes/kenes-app",
                "/home/a b/kenes-app",
                "/home/хуго/kenes-app",
                "/home/a b/$x/kenes-app",
                "/opt/we\"ird/kenes-app",
                "/opt/tick`/kenes-app",
                "/opt/back\\slash/kenes-app",
                "/opt/100%/kenes-app",
                "/tmp/new\nline/kenes-app",
            ];
            for p in paths {
                let entry = desktop_entry(Path::new(p), false);
                let exec = entry.lines().find_map(|l| l.strip_prefix("Exec=")).expect("an Exec line");
                assert_eq!(exec_argv(exec), Some(vec![p.to_owned()]), "Exec={exec}");
                assert!(entry.lines().all(|l| l.is_empty() || l.starts_with('[') || l.contains('=')), "{entry}");
            }
        }

        #[test]
        fn never_writes_over_an_entry_it_did_not_write() {
            use std::io::{Error, ErrorKind};
            let wanted = desktop_entry(Path::new("/opt/kenes/kenes-app"), false);
            let outdated = desktop_entry(Path::new("/old/place/kenes-app"), false);
            assert!(should_write_entry(Err(Error::from(ErrorKind::NotFound)), &wanted).unwrap());
            assert!(should_write_entry(Ok(outdated.into_bytes()), &wanted).unwrap());
            assert!(!should_write_entry(Ok(wanted.clone().into_bytes()), &wanted).unwrap());
            let users = "[Desktop Entry]\nType=Application\nName=Kenes\nExec=/usr/local/bin/kenes\n";
            assert!(!should_write_entry(Ok(users.into()), &wanted).unwrap());
            // Not UTF-8 (a legacy-encoded localized name) or unreadable: whose it is can't be told,
            // so it is left alone.
            let mut legacy = users.as_bytes().to_vec();
            legacy.extend_from_slice(b"Name[ru]=\xca\xe5\xed\xe5\xf1\n");
            assert!(!should_write_entry(Ok(legacy), &wanted).unwrap_or(false));
            assert!(!should_write_entry(Err(Error::from(ErrorKind::PermissionDenied)), &wanted).unwrap_or(false));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accelerators_convert_to_xdg_triggers() {
        assert_eq!(to_xdg_trigger("CommandOrControl+Alt+Enter").as_deref(), Some("CTRL+ALT+Return"));
        assert_eq!(to_xdg_trigger("Alt+CommandOrControl+KeyK").as_deref(), Some("CTRL+ALT+k"));
        assert_eq!(to_xdg_trigger("Super+Shift+Digit1").as_deref(), Some("SHIFT+LOGO+1"));
        assert_eq!(to_xdg_trigger("Ctrl+Alt+Space").as_deref(), Some("CTRL+ALT+space"));
        assert_eq!(to_xdg_trigger("Ctrl+F5").as_deref(), Some("CTRL+F5"));
        assert_eq!(to_xdg_trigger("Ctrl+ArrowUp").as_deref(), Some("CTRL+Up"));
        assert_eq!(to_xdg_trigger("Ctrl+Alt+P").as_deref(), Some("CTRL+ALT+p"));
        assert_eq!(to_xdg_trigger("Ctrl+Alt+Quote").as_deref(), Some("CTRL+ALT+apostrophe"));
        assert_eq!(to_xdg_trigger("Ctrl+Alt"), None);
        assert_eq!(to_xdg_trigger("Ctrl+KeyA+KeyB"), None);
        assert_eq!(to_xdg_trigger("Ctrl+Nonsense"), None);
        assert_eq!(to_xdg_trigger(""), None);
    }

    #[test]
    fn actions_round_trip_through_ids() {
        for a in Action::ALL {
            assert_eq!(Action::from_id(a.id()), Some(a));
        }
        assert_eq!(Action::from_id("nope"), None);
        assert_eq!(serde_json::to_string(&Action::Recap).unwrap(), "\"recap\"");
    }

    #[test]
    fn config_parses_from_the_ui_shape() {
        let cfg: HotkeyConfig = serde_json::from_value(serde_json::json!({
            "enabled": true, "hint": "CommandOrControl+Alt+Enter", "recap": "CommandOrControl+Alt+KeyK", "toggle": "CommandOrControl+Alt+KeyP"
        }))
        .unwrap();
        assert_eq!(cfg.accelerator(Action::Recap), "CommandOrControl+Alt+KeyK");
        let st = serde_json::to_value(HotkeyStatus::new(BackendKind::Portal, HotkeyState::Pending, None)).unwrap();
        assert_eq!(st["backend"], "portal");
        assert_eq!(st["state"], "pending");
        assert_eq!(st["bindings"][0], serde_json::json!({"action": "hint", "trigger": null}));
    }
}
