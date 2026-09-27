# Known gaps and next steps

Status as of 2026-09-27. Each item says what is wrong or missing, how we know, and the likely fix.

## Accuracy

- **Speaker labels are measured on synthetic data only.** A live run of a synthetic 11-speaker
  ru/kk call (`crates/kenes-speakers/testdata-cache/meetings/m12`, played through a PipeWire null
  sink) found 11 of 11 speakers. It attributed 90.9% of speech frames to the right person, and 0.9% of
  speech ended up in segments shared with another speaker. Replaying the same file with
  `kenes-cli --replay-system` gives the same score. Offline, over 12 meetings: call 96–99%,
  simulated room 92–95% (`crates/kenes-speakers/EVAL.md`). What's left:
  - real rooms (far-field laptop mic, crosstalk);
  - segments under 1 s, which are 33–41% wrong online;
  - similar voices, which add 1–3 extra labels in a room.
- **GigaAM features: done, with a smaller payoff than the Python benchmark suggested.** kenes-stt now
  runs GigaAM on ONNX Runtime with the model's own 20 ms log-mel (`crates/kenes-stt/src/gigaam.rs`).
  That is the default for every model, and sherpa-onnx's recognizer remains the fallback (`sttBackend`
  setting, `KENES_STT_BACKEND`). There is still only one ONNX Runtime in the process: `ort` is built
  with `alternative-backend` and pointed at the runtime sherpa-onnx links statically. Nothing is
  downloaded, and Linux and macOS work the same way (macOS type-checks; it has not been linked on a Mac).
  - **Features** match `run_bench.py` to 8.7e-4 at most (float64 math to 1e-6). The int8 model turns
    differences that small into different words on ~5% of clips. The two Python references
    themselves (numpy port vs torchaudio) disagree on 11 of 210 clips.
  - **Accuracy.** When one decode spans a pause, as in the synthetic code-switch clips decoded
    whole, the new backend fixes sherpa's garbled second halves: 220M CER 2.3 → 1.0, large 0.8 →
    0.7. In the live path (VAD splits at 0.5 s pauses) the two backends are at parity: 187 vs 182
    word errors out of 3,269 for the 220M model, 142 vs 142 for the large one.
  - **Speed**: parity. It is the same runtime, and the "1.8× faster" from the Python benchmark was
    not reproducible. Official ONNX Runtime 1.28.2 and 1.30.0 builds, loaded dynamically, were
    within ~15% of the bundled one.
  - **Zero padding hurts with exact features.** Decodes are therefore padded with −70 dBFS noise
    instead of zeros (cv_kk 12.1 → 8.2 % WER).
  - Numbers: `crates/kenes-stt/README.md`. What's left: `gigaam-v3-ru-ctc` is 0.1–0.4 WER points
    (1–5 words) behind sherpa on FLEURS ru, which is within noise but not better; and kenes-core
    still has to pass `sttBackend` to `Transcriber::with_backend`.
- **Only synthetic and read speech has been measured.** Record real meetings with `kenes-rec` and run
  `bench/run_bench.py --custom DIR` before tuning further.
- **Model hosting.** The GigaAM-Multilingual ONNX files come from a personal GitHub release
  (`fgeeer77/bayaya-models`). They are pinned by sha256, but should be mirrored somewhere we control.

## Audio

- **Echo cancellation is measured on synthetic echo only.** Without headphones the mic hears the call,
  and the remote side's words used to show up twice, as the user or a room speaker. Now two layers,
  both behind `echoCancellation` (default on, active when both sources are captured):
  `crates/kenes-aec` runs the mic through WebRTC AEC3 (pure-Rust `sonora`) with the system stream as
  the reference, plus a delay tracker for echo later than 450 ms and a headphone detector that passes
  the mic through untouched; `kenes_core::echo_guard` drops mic finals that repeat the call's
  transcript. On 24 synthetic laptop-speaker scenes (`crates/kenes-aec/EVAL.md`): far-end words
  recognized from the mic 100.5% → 0.9% (AEC) and 0.5% (text guard alone); ERLE 24 dB on the first
  far-end turn, 31 dB converged; user-only speech bit-exact; with headphones the mic passes through
  once the detector has seen a few seconds of far-end speech (double-talk WER 4.2% either way). Costs:
  about 2–3% of one core, 18 ms fixed latency (not visible in timestamps). What's left:
  - a real call with laptop speakers (and with headphones) recorded by `kenes-rec` and replayed;
  - double talk: AEC3 dents the user's voice while both talk (WER 6.0% → 6.7% in those turns);
    `sonora` doesn't expose AEC3's suppressor tuning (the `aec3` crate does);
  - echo that arrives before its reference on the timeline (monitor stream delivered late) cannot be
    cancelled; unmeasured on real devices, macOS in particular.
- **macOS has never run.** The capture code (cpal loopback, which needs macOS 14.6+) type-checks for
  `aarch64-apple-darwin`, but nothing has been built or run on a Mac. First steps: a signed debug
  `.app` build, the permission prompts, and `kenes-rec` on a real call.

## App

- **The real Claude API has not been called yet.** Hints, summaries, name suggestions (structured
  output), server-side fallbacks and caching were only checked with a stubbed `fetch`. The first real run should
  confirm that `fallbacks: "default"` is accepted, that `output_config.format` works together with adaptive
  thinking, and that `cache_read_input_tokens` grows turn over turn (Summary tab → «Расход Claude» shows
  the cache share and an approximate cost per meeting).
- **Always-on-top on GNOME Wayland** works by running under XWayland (`gnomeAlwaysOnTop`, default on):
  Mutter then accepts `_NET_WM_STATE_ABOVE` (checked with `xprop` on GNOME 50). Limits: at fractional
  scaling the window renders at the next whole scale (about 20% larger at 167%), and the setting applies
  after a restart. Off, the window is native Wayland and has to be pinned via Alt+Space. Other Wayland
  compositors keep native Wayland; whether they honour "above" is untested.
- **Global hotkeys** («Что ответить?», «Кратко: 5 мин», show/hide) work while the call has focus: the XDG
  GlobalShortcuts portal on Wayland, `tauri-plugin-global-shortcut` on macOS/X11. On GNOME the portal
  needs an app id, so the app registers as `kz.kenes.app` and writes a desktop entry for it if none
  exists. Checked here up to GNOME's binding dialog; a key press from another app still needs a manual
  test on GNOME, and the macOS/X11 path has not run yet. Dev builds write a hidden (`NoDisplay`) entry,
  which GNOME Settings → Apps may not list for rebinding.
- **Reattach after a webview reload**: `session_status` + `get_meeting` restore the transcript, names,
  hints, rolling summary, timer and usage, and rebuild the Claude conversation from the stored
  transcript. Streaming cards, name suggestions and partials at reload time are lost.
- **Long meetings** roll the Claude conversation over to a new one seeded with the rolling summary
  before it passes 70% of the model's context window (matters for 200K models such as Haiku 4.5). Not
  yet tried against the real API.
- **Packaging.** macOS notarization and Linux AppImage/deb have not been built yet.
