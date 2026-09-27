# Kenes: module contract

Kenes is a desktop meeting copilot for macOS and Linux. It captures the microphone and system audio,
transcribes Russian and Kazakh locally, and uses the Claude API for live hints and meeting summaries.
This file fixes the interfaces between modules so they can be built in parallel. If you need to change
an interface, change it here in the same edit.

## Repository layout and ownership

| Path | What | Depends on |
|---|---|---|
| `crates/kenes-types` | Shared types: `AudioChunk`, `Segment`, `PipelineEvent`, `DeviceInfo`, `Source` | serde |
| `crates/kenes-audio` | Capture of mic and system audio as two separate 16 kHz mono streams | kenes-types |
| `crates/kenes-stt` | Model registry and download, Silero VAD, offline recognizer, streaming `Transcriber` | kenes-types |
| `crates/kenes-speakers` | Speaker embeddings, online clustering, end-of-meeting re-clustering, voiceprint matching | kenes-types, kenes-stt (download helper) |
| `crates/kenes-aec` | Acoustic echo cancellation of the mic against system audio (WebRTC AEC3 via `sonora`) | kenes-types, sonora |
| `crates/kenes-core` | Session orchestration (audio → echo cancellation → STT → speakers → events), SQLite storage, `kenes-cli` binary | all of the above |
| `app/src-tauri` | Tauri shell: commands, events, tray, window, API key storage | kenes-core |
| `app/src` | React + TS UI and the Claude orchestrator (`app/src/llm`) | Tauri commands/events |
| `bench/` | Python ASR benchmark (WER/RTF) | – |

Rust toolchain: `export PATH="$HOME/.cargo/bin:$PATH"` (rustup is installed but not on the default PATH).
All crates are members of the root Cargo workspace; shared dependency versions live in the root
`Cargo.toml` under `[workspace.dependencies]`. Add a crate-specific dependency to your own crate's
`Cargo.toml` only.

## Audio format

Everything between crates is `kenes_types::AudioChunk`: 16 kHz, mono, `f32` in `[-1, 1]`, chunks of
roughly 20–100 ms, `start_ms` measured from session start on a monotonic clock. The two sources
(`Source::Mic`, `Source::System`) are never mixed.

## `kenes-audio`

```rust
pub enum DeviceSel { Default, Id(String) }

pub struct CaptureConfig {
    pub mic: Option<DeviceSel>,     // None: don't capture the mic
    pub system: Option<DeviceSel>,  // None: don't capture system audio
}

pub fn list_devices() -> anyhow::Result<Vec<kenes_types::DeviceInfo>>;
pub fn start_capture(cfg: CaptureConfig, tx: crossbeam_channel::Sender<AudioChunk>) -> anyhow::Result<CaptureHandle>;

pub struct CaptureHandle { /* … */ }
impl CaptureHandle { pub fn stop(self); }   // Drop also stops capture and joins threads

pub mod wav { /* read/write 16 kHz mono WAV helpers */ }

// Additions (implemented):
impl DeviceSel { pub fn from_option(id: Option<String>) -> Self; }  // settings micDevice/systemDevice → DeviceSel
impl Default for CaptureConfig { /* mic + system, both DeviceSel::Default */ }
pub const CHUNK_SAMPLES: usize = 512;        // every chunk is 32 ms, except a shorter final one flushed on stop
pub struct CaptureError { pub source: Source, pub message: String }  // a stream died mid-session
pub struct StreamInfo { pub source: Source, pub device: String, pub backend: &'static str }
impl CaptureHandle {
    pub fn errors(&self) -> crossbeam_channel::Receiver<CaptureError>;  // for select!; ≤ 1 error per source
    pub fn take_error(&self) -> Option<CaptureError>;                   // non-blocking poll
    pub fn streams(&self) -> &[StreamInfo];                             // what was actually opened
    pub fn session_start(&self) -> std::time::Instant;                  // origin of start_ms
}
pub mod pcm { pub fn rms(&[f32]) -> f32; pub fn resample(&[f32], u32, u32) -> Vec<f32>; /* … */ }
// wav::read(path) -> Vec<f32> (any rate/channels → 16 kHz mono), wav::write(path, &[f32]), wav::WavWriter
```

Behavior:
- `start_capture` returns after every stream has delivered audio. This usually takes 100–200 ms.
  An error while starting fails the call, and nothing is left running. After that, a failed
  stream sends one `CaptureError` and stops. The other stream keeps going, and nothing restarts.
- Stopping, or dropping the receiver, ends capture. A stopped stream sends its last short chunk
  and then joins in a few ms.
- Use an unbounded `tx`, or drain it promptly. A blocked send stalls the reader, and the audio
  server then drops samples.
- `start_ms` counts samples from the moment the stream's first audio arrives. Chunks of one
  source are exactly contiguous. If a stream loses more than 250 ms of audio, its timeline
  jumps forward to catch up.

Backends:
- Linux: `parec` / `pw-record` subprocesses reading raw PCM (s16le, 16 kHz, mono) from stdout. The mic is
  the default source. System audio is `<default sink>.monitor`. Devices come from `pactl`. This needs
  no `-dev` packages, which this machine doesn't have.
- macOS: `cpal` for the mic. System audio uses cpal 0.18's loopback, a Core Audio process tap on
  the output device, which cpal says needs macOS 14.6+. The macOS code can't run on the Linux dev
  box; it sits behind `#[cfg(target_os = "macos")]` and is marked untested. It does type-check with
  `cargo check --target aarch64-apple-darwin`.

## `kenes-stt`

```rust
pub fn models_dir() -> std::path::PathBuf;   // $KENES_MODELS_DIR, else <data_dir>/kenes/models
pub fn available_models() -> Vec<ModelInfo>; // id, display name, languages, size, downloaded?
pub fn ensure_model(id: &str, models_dir: &Path, progress: &mut dyn FnMut(f32)) -> anyhow::Result<PathBuf>;

pub struct SttConfig {
    pub model_id: String,          // e.g. "gigaam-multilingual-ctc"
    pub models_dir: PathBuf,
    pub num_threads: i32,          // default 4
    pub partial_interval_ms: u64,  // default 700: re-decode the open segment this often
    pub max_segment_ms: u64,       // default 20_000: force-close long segments
}

pub struct Transcriber { /* recognizer + one VAD per source */ }
impl Transcriber {
    pub fn new(cfg: SttConfig) -> anyhow::Result<Self>;               // = with_backend(cfg, SttBackend::Auto)
    /// Choose the recognizer engine. `Ort` falls back to `Sherpa` (with a logged warning)
    /// if ONNX Runtime can't load the model; `backend()` tells which one is in use.
    pub fn with_backend(cfg: SttConfig, backend: SttBackend) -> anyhow::Result<Self>;
    pub fn backend(&self) -> SttBackend;                               // Ort or Sherpa, never Auto
    /// Consume chunks until `rx` disconnects; emit partial and final segments.
    pub fn spawn(self, rx: Receiver<AudioChunk>, tx: Sender<Segment>) -> std::thread::JoinHandle<()>;
    /// Offline helper for tests/CLI: VAD-split and transcribe a whole buffer.
    pub fn transcribe_buffer(&mut self, source: Source, samples: &[f32]) -> anyhow::Result<Vec<Segment>>;
    /// Optional speaker-change splitting of finals (call before `spawn`).
    pub fn set_splitter(&mut self, f: Splitter);
    pub fn set_split_options(&mut self, opts: SplitOptions);
}

/// serde/FromStr: "auto" | "ort" | "sherpa" (also "onnxruntime", "sherpa-onnx").
pub enum SttBackend {
    Auto,   // default: $KENES_STT_BACKEND if set, else the model's preference (Ort for every GigaAM model)
    Ort,    // ONNX Runtime + GigaAM's own 20 ms log-mel front-end (the one the models were trained on)
    Sherpa, // sherpa-onnx OfflineRecognizer (25 ms kaldi fbank front-end); the fallback
}

/// Gets exactly the samples a final decodes (16 kHz mono, no padding) and returns the
/// sample offsets where the speaker changes; empty = no split.
pub type Splitter = Box<dyn FnMut(&[f32]) -> Vec<usize> + Send>;
pub struct SplitOptions {
    pub min_utterance_ms: u64, // default 2500: shorter finals aren't offered to the splitter
    pub min_piece_ms: u64,     // default 400: shorter pieces merge into their shorter neighbour
    pub timeout_ms: u64,       // default 2000 …
    pub timeout_ratio: f32,    // … + 0.25 × utterance length: then give up and don't split
}
```

Splitting: each piece is decoded on its own. The first piece is the final for the utterance's
existing id; the other pieces become finals with fresh ids from the same per-source counter
(`mic-5` → `mic-5`, `mic-6`, …) and their own `start_ms`/`end_ms` on the chunk timeline. Later
pieces that decode to nothing are skipped; the first piece follows the empty-final rule below.
Partials are never split. The closure runs on a helper thread (`kenes-stt-split`) owned by the
transcriber, and the worker waits at most the timeout. A panic in it is caught (`catch_unwind`,
so not with `panic = "abort"`) and means "no split" for that utterance; the splitter stays in use.

Segment rules: ids look like `mic-17` / `system-4`; partials for an open utterance reuse the id;
exactly one `is_final: true` per id; text is the raw recognizer output (no punctuation is fine —
Claude restores it downstream). A final with empty `text` happens only for an id that already had
partials and means "this was noise": drop the partial from the UI and don't store it.

Backends: both run the same int8 ONNX file on the same ONNX Runtime. That runtime is the one
sherpa-onnx links statically; `ort` is built without a runtime of its own and is pointed at it, so
nothing extra is linked, bundled or downloaded. `SttConfig` has no backend field (adding one would
break struct literals in other crates): pass the choice to `with_backend`. The `sttBackend` setting
below is where the choice lives; `KENES_STT_BACKEND=ort|sherpa` overrides `Auto` for debugging.

Also exported (additions, see `crates/kenes-stt/README.md`): `SttConfig::default()`, `DEFAULT_MODEL`,
`ModelInfo` (serde camelCase: `{ id, name, languages, sizeMb, downloaded }`), `available_models_in(dir)`,
`is_downloaded(id, dir)`, `model_path(id, dir)`, `verify_model(id, dir)`, `Transcriber::recognize(samples)`,
`BACKEND_ENV`, and `LogMel` (the GigaAM front-end, for benchmarks).

For other crates' models (e.g. `kenes-speakers`): `kenes_stt::download_verified(url, sha256, size, dest, progress)
-> anyhow::Result<()>` downloads one file with the same machinery (`.part` + resume, size and SHA-256 check,
atomic rename, `progress` 0..=1). It is a no-op if `dest` already exists with that hash, and it blocks.

## `kenes-speakers` (diarization for meetings of up to ~15 people)

Speaker labels (`Segment.speaker`, final segments only):
- `"me"`: the user. In `micMode: "me"` every mic segment is the user. In `micMode: "room"` a mic
  segment is `"me"` only when it matches the enrolled voiceprint.
- `"mic:N"`: the N-th distinct voice heard by the microphone (`micMode: "room"`: in-person or hybrid).
- `"sys:N"`: the N-th distinct voice in the call (system audio).
- `null`: too short or too uncertain to tell.
Numbers start at 1 per prefix in order of first appearance and never get reused within a meeting.

```rust
pub fn ensure_speaker_model(models_dir: &Path, progress: &mut dyn FnMut(f32)) -> anyhow::Result<PathBuf>;

pub struct Embedder { /* sherpa-onnx speaker embedding extractor */ }
impl Embedder {
    pub fn new(model_path: &Path, num_threads: i32) -> anyhow::Result<Self>;
    /// 16 kHz mono → L2-normalized embedding. Callers skip audio shorter than `ClusterConfig::min_embed_ms`.
    pub fn embed(&mut self, samples: &[f32]) -> anyhow::Result<Vec<f32>>;
}

pub struct ClusterConfig { pub threshold: f32, pub min_embed_ms: u64, /* tuned defaults via Default */ }

/// Streaming assignment within one source ("mic" or "sys").
pub struct OnlineClusterer { /* … */ }
impl OnlineClusterer {
    pub fn new(prefix: &str, cfg: ClusterConfig) -> Self;
    pub fn set_voiceprint(&mut self, embedding: Vec<f32>);   // enables the "me" label
    /// `embedding: None` = segment too short to embed; uses recent context instead (or returns None).
    pub fn assign(&mut self, embedding: Option<&[f32]>, start_ms: u64, end_ms: u64) -> Option<String>;
}

pub struct ClusterItem { pub segment_id: String, pub prefix: String, pub embedding: Vec<f32>,
                         pub duration_ms: u64, pub online_label: Option<String> }
/// Offline re-clustering of the whole meeting. Returned labels reuse the online labels by majority overlap,
/// so names the user already gave to speakers stay attached to the right people.
pub fn recluster(items: &[ClusterItem], voiceprint: Option<&[f32]>, cfg: &ClusterConfig) -> Vec<(String, Option<String>)>;
```

Also exported: `change_points(&mut Embedder, samples, &ClusterConfig) -> Vec<usize>` (sample offsets
where the voice changes inside one utterance; `kenes-core` installs it into `Transcriber::set_splitter`),
`speaker_model_path`, `is_speaker_model_downloaded`, `SPEAKER_MODEL` (3D-Speaker CAM++, 192-dim),
`MIN_EMBED_SAMPLES`, `Embedder::dim`, `OnlineClusterer::{resolve, merges, num_speakers, has_voiceprint}`.
`ClusterConfig::default()` is tuned for `SPEAKER_MODEL`: changing the model means retuning (see
`crates/kenes-speakers/EVAL.md`). `Embedder` is `Send + Sync`, `embed` takes `&mut self`.

`kenes-core` keeps a ring buffer of recent audio per source. For each final segment it cuts the
segment's audio, embeds it, assigns a label, and only then emits the segment. It stores embeddings
per segment. When a session stops, it runs `recluster`, updates the stored labels, and emits
`speakersRelabeled` with only the segments whose label changed.

## `kenes-aec` (echo of the call in the mic)

The system stream is exactly what the speakers play, so it is the far-end reference for cancelling
the call's echo from the mic. WebRTC AEC3 via the pure-Rust `sonora` port, 16 kHz mono, 10 ms frames.

```rust
pub const FRAME: usize = 160;              // 10 ms
pub const LATENCY_SAMPLES: usize = 288;    // fixed 18 ms: 10 ms mic delay (the reference must lead) + AEC3's 8 ms

pub struct AecConfig {
    pub far_silence_dbfs: f32, // -60: quieter far-end frames are silence
    pub hangover_ms: u32,      // 1000: keep cancelling this long after the far end falls silent
    pub track_delay: bool,     // true: coarse tracker pre-delays the reference past AEC3's ~450 ms window
    pub max_delay_ms: u32,     // 1000: longest delay the tracker searches
    pub max_wait_ms: u32,      // 150: StreamCanceller, how long a mic frame waits for its reference
}
pub struct EchoCanceller { /* … */ }       // frame level
impl EchoCanceller {
    pub fn new(cfg: AecConfig) -> Self;
    /// One frame each, same moment of the timeline; `near` is overwritten, LATENCY_SAMPLES late.
    pub fn process_frame(&mut self, far: &[f32], near: &mut [f32]);
    pub fn process(&mut self, far: &[f32], near: &mut [f32]);   // whole frames
    pub fn stats(&self) -> AecStats;
}
pub struct StreamCanceller { /* … */ }     // what the session uses
impl StreamCanceller {
    pub fn new(cfg: AecConfig) -> Self;
    /// Either source. Returns processed mic chunks that became ready; system chunks are only
    /// kept as the reference (the caller forwards them itself).
    pub fn push(&mut self, chunk: &AudioChunk) -> Vec<AudioChunk>;
    pub fn finish(&mut self) -> Vec<AudioChunk>;   // flush on stop
    pub fn stats(&self) -> AecStats;
    pub fn pending_ms(&self) -> u64;
}
```

Behavior:
- Mic and system are paired by `start_ms`. A mic frame waits until the system audio for the same
  span has arrived, or until the mic is `max_wait_ms` past it (then that reference counts as silence).
  Without a system stream the mic passes through with at most `max_wait_ms` delay.
- Output chunks keep the input's timeline: the fixed 18 ms delay is taken out of `start_ms`, so
  processed chunks are contiguous and cover exactly the input samples (chunk sizes vary).
- The mic comes out bit-exact (only delayed) while the far end has been silent for `hangover_ms`, and
  while the far end plays but no echo path is found (headphones: the mic stays at its noise floor
  while the far end is loud). Echo turning up switches cancelling back on (`AecStats::echo_path`).
- Mic gaps restart the mic timeline (the canceller state carries over); reference gaps are filled
  with silence.

`kenes-core` runs the mic through a `StreamCanceller` before the level meter, the speaker ring buffer
and the transcriber when `echoCancellation` is on and both sources are captured (live and replay).
With the same condition, `kenes_core::echo_guard::EchoGuard` holds each non-empty mic final while the
call had sound in the 1.5 s before it and its transcript may still come (at most 2.5 s), and drops it
if its character trigrams are ≥ 60% contained in the system finals within ±1.5 s (one-word finals:
the word must occur there). A dropped final is sent as an empty final if its id had partials,
otherwise not at all, and is never stored or labeled. System segments and mic partials are never
held; mic finals keep their order. Numbers: `crates/kenes-aec/EVAL.md`.

## Tauri commands (`app/src-tauri` ⇄ `app/src`)

All names are snake_case in Rust and invoked as `invoke("name", { camelCaseArgs })` from TS.

| Command | Args | Returns |
|---|---|---|
| `list_devices` | – | `DeviceInfo[]` |
| `list_models` | – | `ModelInfo[]` (`{ id, name, languages: string[], sizeMb, downloaded }`) |
| `start_session` | `{ title: string, context: string }` | `{ meetingId: string }` |
| `stop_session` | – | `void` |
| `session_status` | – | `LiveSession \| null` (`{ meetingId, state: "idle" \| "loading" \| "running" \| "error" }`, from `SessionManager::live()`; a reloaded UI reattaches with it) |
| `get_settings` | – | `Settings` (JSON object; see below) |
| `save_settings` | `{ settings: Settings }` | `void` |
| `get_api_key` | – | `string \| null` |
| `set_api_key` | `{ key: string }` | `void` |
| `list_meetings` | – | `MeetingSummary[]` (`{ id, title, startedAt, endedAt \| null }`, ISO strings) |
| `get_meeting` | `{ id: string }` | `Meeting` (`MeetingSummary` + `{ context, segments: Segment[], notes: Note[], speakers: Speaker[] }`) |
| `save_note` | `{ meetingId, kind: "hint" \| "summary" \| "final", content: string, trigger: string \| null }` | `string` (note id) |
| `delete_meeting` | `{ id: string }` | `void` |
| `rename_speaker` | `{ meetingId, label: string, name: string }` (empty name = reset) | `void` |
| `enroll_voice` | `{ seconds: number }` (records the mic now; no session may be running) | `{ speechMs: number }` |
| `voiceprint_status` | – | `{ enrolled: boolean, createdAt: string \| null }` |
| `clear_voiceprint` | – | `void` |
| `configure_hotkeys` | `{ config: { enabled: boolean, hint: string, recap: string, toggle: string } }` (accelerators, see below) | `HotkeyStatus`; resolves once the system answered (the portal may first show its own dialog) |
| `hotkey_status` | – | `HotkeyStatus` |
| `platform_info` | – | `{ os, sessionType: "wayland" \| "x11" \| null, desktop: string \| null, gnome: boolean, x11Forced: boolean, hotkeyBackend: "portal" \| "plugin" \| "none" }` |

`Note` = `{ id, meetingId, kind, content, trigger, createdAt }`. Only final segments with non-empty text are stored.
`Speaker` = `{ label, name: string | null, segmentCount, talkMs }` (every label seen in the meeting, named or not).

Global shortcuts (`app/src-tauri/src/hotkeys.rs`):
- Accelerators look like `CommandOrControl+Alt+Enter`: modifiers (`CommandOrControl`, `Control`, `Alt`,
  `Shift`, `Super`), then one key named like `KeyboardEvent.code` (`KeyK`, `Digit1`, `Enter`, `F5`, `ArrowUp`, …).
- Backend, fixed per process: Wayland sessions use the XDG GlobalShortcuts portal (`ashpd`), where the
  compositor owns the keys (the accelerators are only `preferred_trigger` suggestions, converted to
  `CTRL+ALT+Return`); macOS and X11 use `tauri-plugin-global-shortcut` with exactly these keys.
- `HotkeyStatus` = `{ backend: "portal" | "plugin" | "none", state: "off" | "pending" | "active" | "cancelled" |
  "unavailable" | "error", bindings: { action: "hint" | "recap" | "toggle", trigger: string | null }[],
  message: string | null }`. `trigger` is what the system reports (portal) or the accelerator (plugin).
- `configure_hotkeys` with the settings already applied is a no-op, so a reloaded UI can call it again.
  New settings while the portal dialog is open close that dialog and apply the new ones.
- GNOME only lets a host app use the portal under an app id with a desktop entry, so on Wayland the shell
  registers as `kz.kenes.app` and writes `~/.local/share/applications/kz.kenes.app.desktop` if no entry
  with that name exists on the XDG data path.

`Settings` is one JSON object persisted by Rust. Rust reads the audio/STT keys; the rest belongs to the UI:

```ts
type Settings = {
  // read by Rust
  sttModel: string;            // default "gigaam-multilingual-ctc"
  sttBackend: "auto" | "ort" | "sherpa"; // default "auto" (= "ort" for every model today). kenes-core passes it to
                               // Transcriber::with_backend; save_settings rejects other values. Settings tab:
                               // «Движок распознавания», applies to the next session
  numThreads: number;          // default 4
  captureMic: boolean;         // default true
  captureSystem: boolean;      // default true
  micMode: "me" | "room";      // default "me": mic = only the user (headset/online). "room": several people on the mic
  micDevice: string | null;    // null = default
  systemDevice: string | null; // null = default
  echoCancellation: boolean;   // default true: remove the call's echo from the mic (speakers, no headphones):
                               // kenes-aec + the transcript echo guard; acts only when both sources are captured
  // read by the app shell at start-up (app/src-tauri), before the window exists
  gnomeAlwaysOnTop: boolean;   // default true: on GNOME Wayland run under XWayland (GDK_BACKEND=x11) so the
                               // window can stay on top; takes effect after a restart
  // UI only
  claudeModel: string;         // default "claude-opus-5"
  hintEffort: "low" | "medium" | "high";
  summaryEffort: "low" | "medium" | "high" | "xhigh";
  autoHintMode: "addressed" | "any" | "off"; // replaces autoHints (false → "off"); default "addressed" if micMode "room", else "any"
  myNames: string[];           // how people address the user, e.g. ["Хуго", "Hugo"]; default []
  rollingSummaryMinutes: number; // default 4, 0 = off
  answerLanguage: "auto" | "ru" | "kk";
  profile: string;             // "about me": role, company, what I usually discuss
  globalHotkeys: boolean;      // default true; the UI passes these two to configure_hotkeys
  hotkeys: { hint: string; recap: string; toggle: string };
                               // default CommandOrControl+Alt+Enter / +Alt+KeyK / +Alt+KeyP
};
```

## Tauri events

Global shortcuts (from `app/src-tauri`, not the pipeline):
- `kenes://hotkey` with `{ action: "hint" | "recap" | "toggle" }` when a system-wide shortcut fires. The shell
  itself shows/hides the window for "toggle" and shows a hidden window (without focusing it) for the others;
  the UI runs «Что ответить?» / «Кратко: 5 мин».
- `kenes://hotkey-status` with a `HotkeyStatus` whenever it changes (e.g. rebound in the system settings).

The pipeline uses one event name, `kenes://event`, whose payload is a serialized `kenes_types::PipelineEvent`:

```ts
type PipelineEvent =
  | ({ type: "segment" } & Segment)
  | { type: "level"; source: "mic" | "system"; rms: number }
  | { type: "status"; state: "idle" | "loading" | "running" | "error"; message: string | null }
  | { type: "modelProgress"; model: string; progress: number }
  | { type: "error"; message: string }
  | { type: "speakersRelabeled"; changes: { segmentId: string; speaker: string | null }[] };

type Segment = {
  id: string; source: "mic" | "system"; speaker: string | null;
  startMs: number; endMs: number; text: string; isFinal: boolean;
};
```

## Claude

The UI calls Claude directly with `@anthropic-ai/sdk` (the key comes from `get_api_key`). Rust never talks
to Claude. Only text leaves the machine; audio never does.
