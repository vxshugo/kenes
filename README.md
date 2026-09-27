# Kenes

A meeting copilot for macOS and Linux. It listens to your microphone and to the call (system audio)
as two separate streams, transcribes Russian and Kazakh locally, and uses Claude for live hints,
rolling summaries and a final meeting summary. Audio never leaves the machine; only text goes to
the Claude API.

```
mic ─┐                                   ┌─► UI: live transcript, hints, summaries
     ├─► 16 kHz ─► VAD ─► local ASR ─► segments ─┤
call ┘   (kenes-audio)   (kenes-stt)     └─► SQLite (kenes-core)
                                                 UI ─► Claude API (text only)
```

## Layout

| Path | What |
|---|---|
| `crates/kenes-types` | Shared types and the UI event wire format |
| `crates/kenes-audio` | Mic + system audio capture (Linux: PipeWire/Pulse via `parec`; macOS: Core Audio) and `kenes-rec` |
| `crates/kenes-stt` | Model download, Silero VAD, local recognizer (sherpa-onnx), `kenes-transcribe` |
| `crates/kenes-aec` | Removes the call's echo from the mic when you use speakers (WebRTC AEC3, pure Rust) |
| `crates/kenes-core` | Session pipeline, SQLite storage, settings, `kenes-cli` |
| `app/` | Tauri 2 app: `src-tauri/` (Rust shell), `src/` (React UI + Claude layer in `src/llm`) |
| `bench/` | ASR benchmark for Russian/Kazakh models |
| `docs/CONTRACT.md` | Interfaces between the modules above |

## Development

Requirements: Rust (rustup), Node 22 + pnpm. On Linux also `parec`/`pactl` (package
`pulseaudio-utils`; works on PipeWire systems too) and the Tauri prerequisites (WebKitGTK 4.1).

```bash
export PATH="$HOME/.cargo/bin:$PATH"

# Live transcript in the terminal (downloads models on first run)
cargo run --release -p kenes-core --bin kenes-cli

# Record mic.wav + system.wav, e.g. for the ASR benchmark
cargo run --release -p kenes-audio --bin kenes-rec -- --seconds 600 --out ~/kenes-rec

# The desktop app
cd app && pnpm install && pnpm tauri dev

# UI only, in a browser, with a scripted mock meeting
cd app && pnpm dev
```

Data lives in `~/.local/share/kenes` (Linux) or `~/Library/Application Support/kenes` (macOS):
`kenes.db` (meetings) and `models/`. Override with `KENES_DATA_DIR` / `KENES_MODELS_DIR`.
The Anthropic API key is stored in the OS keychain (Settings tab) or read from `ANTHROPIC_API_KEY`.

## Meetings with many people

Kenes labels every final line with a speaker: `Я`, `Участник N` (voices in the call) or `Зал N`
(voices on the mic in a room). Labels come from voice embeddings clustered live, and are refined
once more when the meeting ends. Rename a speaker in the participants panel and the name is used
everywhere, including in Claude's hints and summaries.

Pick the meeting format before starting:
- **Онлайн, я в наушниках**: the mic is you, the call is everyone else.
- **В зале**: one mic hears the whole room. Record your voice once in Settings → «Мой голос» so
  Kenes can tell you apart from the audience.
- **Гибрид**: a room plus remote participants.

For a room of 10–15 people, a USB speakerphone or conference mic in the middle of the table
works far better than the laptop's built-in mic, for both recognition and speaker separation.
Overlapping speech and very short remarks (under about a second) are the main sources of errors.

## macOS notes

System audio capture uses Core Audio process taps via cpal loopback (macOS 14.6+) and needs the "System Audio
Recording" permission. macOS only asks for it on a signed `.app`, so test capture with a bundled
build signed by a stable identity (a free self-signed "Code Signing" certificate works):
`pnpm tauri build --debug --bundles app`.

## Models

Nothing is bundled; models are downloaded on first use into the models directory and verified by
SHA-256. Check each model's license before redistributing it.

| Model | Use | Source | License |
|---|---|---|---|
| GigaAM-Multilingual CTC 220M / 600M (int8 ONNX) | Russian + Kazakh ASR | [ai-sage/GigaAM-Multilingual](https://huggingface.co/ai-sage/GigaAM-Multilingual); ONNX build from [fgeeer77/bayaya-models](https://github.com/fgeeer77/bayaya-models) | MIT (upstream) |
| GigaAM-v3 CTC (int8 ONNX) | Russian-only ASR | [csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16) | MIT (upstream) |
| Silero VAD | Voice activity detection | [sherpa-onnx release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) | MIT |
| 3D-Speaker CAM++ | Speaker embeddings | [sherpa-onnx release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models) | Apache-2.0 |

Accuracy and speed measurements: `bench/RESULTS.md` (ASR), `crates/kenes-speakers/EVAL.md` (speakers).
Known gaps: `docs/ROADMAP.md`.

## License

MIT, see [LICENSE](LICENSE).
