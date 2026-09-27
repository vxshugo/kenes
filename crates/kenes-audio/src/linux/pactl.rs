//! Device discovery through `pactl` (works against PulseAudio and pipewire-pulse).

use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use kenes_types::{DeviceInfo, DeviceKind};
use serde_json::Value;

/// A PulseAudio source as `pactl` reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PaSource {
    pub name: String,
    pub description: Option<String>,
    pub is_monitor: bool,
}

/// Runs `pactl` with a C/UTF-8 locale: localized decimal commas break its JSON
/// output, and localized labels would break `pactl info` parsing.
fn run(args: &[&str]) -> Result<String> {
    let out = Command::new("pactl")
        .args(args)
        .env("LC_ALL", "C.UTF-8")
        .stdin(Stdio::null())
        .output()
        .context("running pactl (is pulseaudio-utils installed?)")?;
    if !out.status.success() {
        bail!(
            "`pactl {}` failed ({}): {}",
            args.join(" "),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    String::from_utf8(out.stdout).context("pactl printed non-UTF-8 output")
}

/// All sources, preferring the JSON output of newer `pactl` (16+).
pub(crate) fn list_sources() -> Result<Vec<PaSource>> {
    match run(&["-f", "json", "list", "sources"]).and_then(|s| parse_json_sources(&s)) {
        Ok(sources) => Ok(sources),
        Err(e) => {
            log::debug!("pactl JSON listing unavailable ({e:#}); using `pactl list short sources`");
            let text = run(&["list", "short", "sources"])?;
            Ok(parse_short_sources(&text))
        }
    }
}

/// Name of the default source (microphone).
pub(crate) fn default_source() -> Result<String> {
    default_via("get-default-source", "Default Source:")
}

/// Name of the default sink (speakers/headphones); its monitor is `<name>.monitor`.
pub(crate) fn default_sink() -> Result<String> {
    default_via("get-default-sink", "Default Sink:")
}

fn default_via(subcommand: &str, info_label: &str) -> Result<String> {
    // `get-default-*` exists since PulseAudio 15; older versions only have `pactl info`.
    let name = match run(&[subcommand]) {
        Ok(out) => out.trim().to_string(),
        Err(_) => parse_info_field(&run(&["info"])?, info_label).unwrap_or_default(),
    };
    if name.is_empty() || name == "@DEFAULT_SINK@" || name == "@DEFAULT_SOURCE@" {
        bail!("the audio server reports no default device for `{subcommand}`");
    }
    Ok(name)
}

pub(crate) fn parse_info_field(info: &str, label: &str) -> Option<String> {
    info.lines()
        .find_map(|l| l.trim().strip_prefix(label))
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

pub(crate) fn parse_json_sources(json: &str) -> Result<Vec<PaSource>> {
    let value: Value = serde_json::from_str(json).context("parsing `pactl -f json` output")?;
    let Value::Array(items) = value else {
        bail!("`pactl -f json list sources` did not print an array");
    };
    let sources = items
        .iter()
        .filter_map(|item| {
            let name = item.get("name")?.as_str()?.to_string();
            let description = item
                .get("description")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|d| !d.is_empty())
                .map(String::from);
            // pactl calls the monitored sink `monitor_source` in JSON (sic); "monitor_of_sink"
            // is the text-mode label. Either being non-empty means this is a monitor.
            let monitors_sink = ["monitor_source", "monitor_of_sink"].iter().any(|k| {
                item.get(k)
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.is_empty() && s != "n/a")
            });
            let class_monitor = item
                .pointer("/properties/device.class")
                .and_then(Value::as_str)
                .is_some_and(|c| c == "monitor");
            let is_monitor = monitors_sink || class_monitor || name.ends_with(".monitor");
            Some(PaSource {
                name,
                description,
                is_monitor,
            })
        })
        .collect();
    Ok(sources)
}

/// Parses `pactl list short sources`: `index<TAB>name<TAB>driver<TAB>spec<TAB>state`.
pub(crate) fn parse_short_sources(text: &str) -> Vec<PaSource> {
    text.lines()
        .filter_map(|line| {
            let name = line.split('\t').nth(1)?.trim();
            (!name.is_empty()).then(|| PaSource {
                name: name.to_string(),
                description: None,
                is_monitor: name.ends_with(".monitor"),
            })
        })
        .collect()
}

pub(crate) fn to_device_infos(
    sources: &[PaSource],
    default_source: Option<&str>,
    default_sink: Option<&str>,
) -> Vec<DeviceInfo> {
    let default_monitor = default_sink.map(|s| format!("{s}.monitor"));
    let mut devices: Vec<DeviceInfo> = sources
        .iter()
        .map(|s| {
            let (kind, is_default) = if s.is_monitor {
                (
                    DeviceKind::Monitor,
                    default_monitor.as_deref() == Some(s.name.as_str()),
                )
            } else {
                (DeviceKind::Input, default_source == Some(s.name.as_str()))
            };
            DeviceInfo {
                id: s.name.clone(),
                name: s.description.clone().unwrap_or_else(|| s.name.clone()),
                kind,
                is_default,
            }
        })
        .collect();
    // Inputs first, defaults first within each kind; otherwise keep pactl's order.
    devices.sort_by_key(|d| (d.kind == DeviceKind::Monitor, !d.is_default));
    devices
}

#[cfg(test)]
mod tests {
    use super::*;

    const JSON: &str = include_str!("../../testdata/pactl-sources.json");
    const SHORT: &str = include_str!("../../testdata/pactl-sources-short.txt");
    const MIC: &str = "alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Mic1__source";
    const SINK: &str =
        "alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Speaker__sink";

    #[test]
    fn parses_json_fixture() {
        let sources = parse_json_sources(JSON).unwrap();
        assert_eq!(sources.len(), 5);
        let monitors: Vec<_> = sources.iter().filter(|s| s.is_monitor).collect();
        assert_eq!(monitors.len(), 4);
        assert!(monitors.iter().all(|s| s.name.ends_with(".monitor")));
        let mic = sources.iter().find(|s| !s.is_monitor).unwrap();
        assert_eq!(mic.name, MIC);
        assert_eq!(
            mic.description.as_deref(),
            Some("Meteor Lake-P HD Audio Controller Digital Microphone")
        );
    }

    #[test]
    fn json_detects_monitor_without_suffix() {
        let json = r#"[{"name":"weird","description":"W","monitor_source":"some_sink","properties":{}},
                       {"name":"mic","description":"","monitor_source":"","properties":{"device.class":"sound"}},
                       {"description":"no name"}]"#;
        let sources = parse_json_sources(json).unwrap();
        assert_eq!(
            sources,
            vec![
                PaSource {
                    name: "weird".into(),
                    description: Some("W".into()),
                    is_monitor: true
                },
                PaSource {
                    name: "mic".into(),
                    description: None,
                    is_monitor: false
                },
            ]
        );
        assert!(parse_json_sources("{}").is_err());
        assert!(parse_json_sources("[{\"balance\":0,00}]").is_err());
    }

    #[test]
    fn parses_short_fixture() {
        let sources = parse_short_sources(SHORT);
        assert_eq!(sources.len(), 5);
        assert_eq!(sources.iter().filter(|s| s.is_monitor).count(), 4);
        assert_eq!(sources[4].name, MIC);
        assert!(sources.iter().all(|s| s.description.is_none()));
        assert!(parse_short_sources("\n\ngarbage\n").is_empty());
    }

    #[test]
    fn json_and_short_agree_on_names() {
        let a: Vec<_> = parse_json_sources(JSON)
            .unwrap()
            .into_iter()
            .map(|s| (s.name, s.is_monitor))
            .collect();
        let b: Vec<_> = parse_short_sources(SHORT)
            .into_iter()
            .map(|s| (s.name, s.is_monitor))
            .collect();
        assert_eq!(a, b);
    }

    #[test]
    fn builds_device_list_with_defaults() {
        let devices = to_device_infos(&parse_json_sources(JSON).unwrap(), Some(MIC), Some(SINK));
        assert_eq!(devices.len(), 5);
        assert_eq!(devices[0].kind, DeviceKind::Input);
        assert_eq!(devices[0].id, MIC);
        assert!(devices[0].is_default);
        assert_eq!(devices[1].kind, DeviceKind::Monitor);
        assert_eq!(devices[1].id, format!("{SINK}.monitor"));
        assert_eq!(
            devices[1].name,
            "Monitor of Meteor Lake-P HD Audio Controller Speaker"
        );
        assert!(devices[1].is_default);
        assert_eq!(devices.iter().filter(|d| d.is_default).count(), 2);

        let devices = to_device_infos(&parse_short_sources(SHORT), None, None);
        assert!(devices.iter().all(|d| !d.is_default && d.name == d.id));
    }

    #[test]
    fn parses_pactl_info() {
        let info = "Server Name: PulseAudio (on PipeWire 1.6.2)\nDefault Sink: sink_a\nDefault Source: src_b\n";
        assert_eq!(
            parse_info_field(info, "Default Sink:").as_deref(),
            Some("sink_a")
        );
        assert_eq!(
            parse_info_field(info, "Default Source:").as_deref(),
            Some("src_b")
        );
        assert_eq!(parse_info_field("Default Sink: \n", "Default Sink:"), None);
    }
}
