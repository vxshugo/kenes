//! Terminal front end for the live pipeline: `kenes-cli [--no-mic] [--no-system] [--room] [--no-echo-cancel] [--mic <id>] [--system <id>]`.
//! Prints partial and final transcript lines; Ctrl-C stops and saves the meeting.

use std::io::Write;
use std::sync::Arc;

use anyhow::Result;
use kenes_core::settings::MicMode;
use kenes_core::{default_data_dir, settings, AudioInput, SessionManager, Store};
use kenes_types::{PipelineEvent, SessionState, Source};

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let data_dir = default_data_dir();
    let store = Arc::new(Store::open(&data_dir.join("kenes.db"))?);
    let mut core = settings::core(&settings::load(&store)?)?;
    let mut title = "Встреча из терминала".to_owned();
    let (mut replay_mic, mut replay_system, mut speed) = (None, None, 1.0f32);

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--no-mic" => core.capture_mic = false,
            "--no-system" => core.capture_system = false,
            "--room" => core.mic_mode = MicMode::Room,
            "--no-echo-cancel" => core.echo_cancellation = false,
            "--model" => core.stt_model = args.next().expect("--model <id>"),
            "--threads" => core.num_threads = args.next().expect("--threads N").parse()?,
            "--title" => title = args.next().expect("--title <text>"),
            "--mic" => core.mic_device = Some(args.next().expect("--mic <device id>")),
            "--system" => core.system_device = Some(args.next().expect("--system <device id>")),
            "--replay-mic" => {
                replay_mic = Some(args.next().expect("--replay-mic <file.wav>").into())
            }
            "--replay-system" => {
                replay_system = Some(args.next().expect("--replay-system <file.wav>").into())
            }
            "--speed" => speed = args.next().expect("--speed N").parse()?,
            "-h" | "--help" => {
                println!("kenes-cli [--no-mic] [--no-system] [--room] [--no-echo-cancel] [--mic <id>] [--system <id>] [--model <id>] [--threads N]\n          [--title <text>] [--replay-mic a.wav] [--replay-system b.wav] [--speed N]");
                return Ok(());
            }
            other => anyhow::bail!("unknown argument {other}"),
        }
    }

    let (quit_tx, quit_rx) = crossbeam_channel::bounded::<()>(1);
    let sink_quit = quit_tx.clone();
    let sink: kenes_core::EventSink = Arc::new(move |ev| render(&ev, &sink_quit));
    let manager = SessionManager::new(store.clone(), kenes_stt::models_dir(), sink);

    let input = if replay_mic.is_some() || replay_system.is_some() {
        AudioInput::Files {
            mic: replay_mic,
            system: replay_system,
            speed,
        }
    } else {
        AudioInput::Capture
    };
    let meeting_id = manager.start_with_input(&title, "", core, input)?;
    ctrlc::set_handler(move || {
        let _ = quit_tx.try_send(());
    })?;
    let _ = quit_rx.recv();
    eprintln!("\nОстанавливаю…");
    manager.stop()?;
    eprintln!(
        "Сохранено: встреча {meeting_id} в {}",
        data_dir.join("kenes.db").display()
    );
    Ok(())
}

fn label(source: Source, speaker: Option<&str>) -> String {
    match speaker {
        Some("me") => "Я".into(),
        Some(l) if l.starts_with("sys:") => format!("Участник {}", &l[4..]),
        Some(l) if l.starts_with("mic:") => format!("Зал {}", &l[4..]),
        Some(l) => l.into(),
        None if source == Source::Mic => "Микрофон".into(),
        None => "Звонок".into(),
    }
}

fn render(ev: &PipelineEvent, quit: &crossbeam_channel::Sender<()>) {
    let mut out = std::io::stdout().lock();
    match ev {
        PipelineEvent::Segment(s) if s.is_final => {
            let secs = s.start_ms / 1000;
            let _ = writeln!(
                out,
                "\r\x1b[2K[{:02}:{:02}] {}: {}",
                secs / 60,
                secs % 60,
                label(s.source, s.speaker.as_deref()),
                s.text
            );
        }
        PipelineEvent::Segment(s) => {
            let _ = write!(out, "\r\x1b[2K\x1b[2m… {}\x1b[0m", s.text);
        }
        PipelineEvent::ModelProgress { model, progress } => {
            let _ = write!(out, "\r\x1b[2KМодель {model}: {:.0}%", progress * 100.0);
        }
        PipelineEvent::Status { state, message } => {
            let _ = writeln!(
                out,
                "\r\x1b[2K[{state:?}] {}",
                message.as_deref().unwrap_or("")
            );
            // Idle means the session ended on its own (replayed files ran out) or failed.
            if matches!(state, SessionState::Error | SessionState::Idle) {
                let _ = quit.try_send(());
            }
        }
        PipelineEvent::Error { message } => {
            let _ = writeln!(out, "\r\x1b[2KОшибка: {message}");
        }
        PipelineEvent::SpeakersRelabeled { changes } => {
            let _ = writeln!(
                out,
                "\r\x1b[2KУточнены спикеры для {} реплик",
                changes.len()
            );
        }
        PipelineEvent::Level { .. } => {}
    }
    let _ = out.flush();
}
