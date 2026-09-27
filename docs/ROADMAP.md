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
- **GigaAM features in sherpa-onnx.** sherpa-onnx 1.13.8 computes GigaAM v1/v2-style features: a 25 ms
  kaldi fbank with n_fft 400. GigaAM-v3/Multilingual were trained on a 20 ms, n_fft 320 torchaudio
  log-mel. On synthetic code-switched audio this costs 30–50% more errors, and it runs about 1.8× slower
  than plain onnxruntime with exact features. The difference on pure ru/kk is within noise.
  Fix: a GigaAM backend on the `ort` crate behind the existing recognizer trait in `kenes-stt`.
  First solve how the ONNX Runtime used by `ort` and the one inside sherpa-onnx can live in one
  process (sherpa `shared` feature + `ort` `load-dynamic` against the same library?).
  Details and a reference implementation are in `bench/RESULTS.md` and `bench/run_bench.py` (`GigaAMLogMel`).
- **Only synthetic and read speech has been measured.** Record real meetings with `kenes-rec` and run
  `bench/run_bench.py --custom DIR` before tuning further.
- **Model hosting.** The GigaAM-Multilingual ONNX files come from a personal GitHub release
  (`fgeeer77/bayaya-models`). They are pinned by sha256, but should be mirrored somewhere we control.

## Audio

- **No echo cancellation.** Without headphones, the mic also picks up the call from the speakers, and the
  remote side's words show up twice. Candidates: `sonora` (pure-Rust WebRTC AEC3) or anarlog's ONNX AEC,
  using the system stream as the far-end reference.
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
