//! Display-server choices that must be made before GTK starts.
//!
//! GNOME on Wayland ignores "keep above" from ordinary clients: xdg-shell has no such request,
//! so the `alwaysOnTop` window option does nothing there. Mutter does honour
//! `_NET_WM_STATE_ABOVE` for X11 clients, so with the `gnomeAlwaysOnTop` setting (default on)
//! the app runs under XWayland (`GDK_BACKEND=x11`) on GNOME Wayland sessions. Trade-offs are in
//! `docs/UI_NOTES.md` (window size at fractional scaling, applies after a restart).

use serde::Serialize;
use serde_json::Value;

use crate::hotkeys::BackendKind;

/// The environment variables the decision depends on.
#[derive(Clone, Debug, Default)]
pub struct DisplayEnv {
    pub session_type: Option<String>,
    pub wayland_display: Option<String>,
    pub display: Option<String>,
    pub current_desktop: Option<String>,
    pub gdk_backend: Option<String>,
}

impl DisplayEnv {
    pub fn from_process() -> Self {
        let get = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        Self {
            session_type: get("XDG_SESSION_TYPE"),
            wayland_display: get("WAYLAND_DISPLAY"),
            display: get("DISPLAY"),
            current_desktop: get("XDG_CURRENT_DESKTOP"),
            gdk_backend: get("GDK_BACKEND"),
        }
    }

    pub fn wayland(&self) -> bool {
        self.session_type.as_deref() == Some("wayland") || self.wayland_display.is_some()
    }

    /// `XDG_CURRENT_DESKTOP` is a colon-separated list, e.g. `ubuntu:GNOME`.
    pub fn gnome(&self) -> bool {
        self.current_desktop.as_deref().is_some_and(|d| d.split(':').any(|p| p.eq_ignore_ascii_case("gnome")))
    }
}

/// Whether to run the window under XWayland: GNOME on Wayland, the setting on, XWayland
/// available, and the user didn't pick a GDK backend themselves.
pub fn should_force_x11(env: &DisplayEnv, want: bool) -> bool {
    cfg!(target_os = "linux") && want && env.wayland() && env.gnome() && env.display.is_some() && env.gdk_backend.is_none()
}

/// `gnomeAlwaysOnTop` from the stored settings; missing means on.
pub fn wants_gnome_on_top(settings: &Value) -> bool {
    settings.get("gnomeAlwaysOnTop").and_then(Value::as_bool).unwrap_or(true)
}

/// Call first thing in `run()`, before anything touches GTK. Returns whether XWayland was chosen.
pub fn apply(settings: &Value) -> bool {
    let env = DisplayEnv::from_process();
    let force = should_force_x11(&env, wants_gnome_on_top(settings));
    if force {
        // No other threads exist yet, so changing the environment is safe here.
        std::env::set_var("GDK_BACKEND", "x11");
    }
    force
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlatformInfo {
    pub os: &'static str,
    /// "wayland" / "x11" on Linux, else null.
    pub session_type: Option<String>,
    pub desktop: Option<String>,
    pub gnome: bool,
    /// The window runs under XWayland because of `gnomeAlwaysOnTop`.
    pub x11_forced: bool,
    pub hotkey_backend: BackendKind,
}

pub fn info(x11_forced: bool, hotkey_backend: BackendKind) -> PlatformInfo {
    let env = DisplayEnv::from_process();
    let session_type = if cfg!(target_os = "linux") {
        Some(if env.wayland() { "wayland".to_owned() } else { env.session_type.clone().unwrap_or_else(|| "x11".into()) })
    } else {
        None
    };
    PlatformInfo {
        os: std::env::consts::OS,
        session_type,
        gnome: env.gnome(),
        desktop: env.current_desktop,
        x11_forced,
        hotkey_backend,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gnome_wayland() -> DisplayEnv {
        DisplayEnv {
            session_type: Some("wayland".into()),
            wayland_display: Some("wayland-0".into()),
            display: Some(":0".into()),
            current_desktop: Some("ubuntu:GNOME".into()),
            gdk_backend: None,
        }
    }

    #[test]
    fn forces_x11_only_on_gnome_wayland_with_xwayland_and_no_user_choice() {
        let env = gnome_wayland();
        assert_eq!(should_force_x11(&env, true), cfg!(target_os = "linux"));
        assert!(!should_force_x11(&env, false));
        assert!(!should_force_x11(&DisplayEnv { current_desktop: Some("KDE".into()), ..env.clone() }, true));
        assert!(!should_force_x11(&DisplayEnv { display: None, ..env.clone() }, true));
        assert!(!should_force_x11(&DisplayEnv { gdk_backend: Some("wayland".into()), ..env.clone() }, true));
        let x11 = DisplayEnv { session_type: Some("x11".into()), wayland_display: None, ..env };
        assert!(!should_force_x11(&x11, true));
    }

    #[test]
    fn setting_defaults_to_on() {
        assert!(wants_gnome_on_top(&serde_json::json!({})));
        assert!(!wants_gnome_on_top(&serde_json::json!({"gnomeAlwaysOnTop": false})));
        assert!(wants_gnome_on_top(&serde_json::json!({"gnomeAlwaysOnTop": "x"})));
    }
}
