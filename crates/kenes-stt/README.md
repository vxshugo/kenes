# kenes-stt

Local speech recognition for kenes. It turns 16 kHz mono `AudioChunk`s from the mic and system
audio into partial and final `Segment`s, entirely on the CPU. Russian and Kazakh work in the same
utterance. The interface is fixed in `docs/CONTRACT.md`. This file covers what's behind it and
how it performs.

## Engine

Two recognizer backends run the same int8 GigaAM ONNX files. `SttBackend::Auto` (the default)
picks ONNX Runtime for every model in the registry:

- **`ort`: ONNX Runtime with GigaAM's own front-end** (`src/gigaam.rs`). GigaAM-v3 and
  GigaAM-Multilingual were trained on torchaudio's log-mel with a 20 ms window (n_fft = win =
  320, hop 160, no centre padding, periodic Hann, power spectrum, 64 HTK mel bins over 0–8000 Hz,
  `ln(clamp(x, 1e-9, 1e9))`). `LogMel` computes exactly that (`realfft`, in `f64`), the model runs
  through the [`ort`](https://crates.io/crates/ort) crate, and the output is decoded with greedy
  CTC (blank = `<blk>`, id 0 = space). The inputs and outputs are checked when the model loads.
  `encoded_lengths` is used if present; the v3 export doesn't have it, and the Multilingual export
  gives it as `int32`.
- **`sherpa`: sherpa-onnx's `OfflineRecognizer`**, the official
  [`sherpa-onnx`](https://crates.io/crates/sherpa-onnx) Rust crate. sherpa-onnx 1.13.8 gives every
  GigaAM model the v1/v2 front-end (a 25 ms kaldi fbank with n_fft 400), which costs accuracy on
  code-switched audio (below). It stays as the fallback.

**One ONNX Runtime per process.** sherpa-onnx is still needed for the Silero VAD (here) and the
speaker embeddings (`kenes-speakers`), and it links its own ONNX Runtime statically (1.28.2 in
sherpa-onnx 1.13.8, the same version in the Linux and macOS archives). The `ort` crate normally links
or downloads a second one, which gives duplicate symbols, or two runtimes if it is loaded as a shared
library. Instead `ort` is built with `default-features = false` and `alternative-backend`, so it
links nothing. On first use `gigaam::init_ort` calls `OrtGetApiBase()`, which the linker resolves
against sherpa-onnx-sys's `libonnxruntime.a`, and hands that API table to `ort::set_api`. The
VAD, the speaker model and both recognizers then share one runtime and one ORT environment. Nothing
is added to the build, the bundle or the downloads, and it works the same way on Linux and macOS.
`ort` requests C API version 17, and the runtime provides up to 28. Alternatives that were rejected:

- **`ort` with `load-dynamic` and a pinned `libonnxruntime` downloaded at first run**: a second
  runtime in the process, next to the static one. It needs a sha256-pinned ~20 MB library per
  platform (and codesigning on macOS), and a download failure path.
- **sherpa-onnx's `shared` feature, with `ort` loading the same `libonnxruntime.so`**: one runtime,
  but `libsherpa-onnx-c-api` and `libonnxruntime` then have to ship next to the binary with an
  rpath, which means Tauri bundling work. Cargo also unifies the feature for `kenes-speakers`, which
  keeps the default `static`, and sherpa-onnx-sys refuses `static` and `shared` together.

Guards: if the runtime doesn't provide API 17, or the model's inputs and outputs don't match
(`features` f32 `[N, 64, T]`, `feature_lengths` i64, `log_probs` f32 `[N, T, vocab]` with the same
vocabulary size as `tokens.txt`), the ORT backend fails to load. The transcriber then logs a warning
and uses sherpa-onnx. `ort` releases its environment in a `.fini_array` hook at exit, which for a
statically linked runtime would run after the runtime's own C++ static destructors. We keep one
reference to the environment for the life of the process so that release never happens.

Choosing the backend: `Transcriber::with_backend(cfg, backend)`, the `sttBackend` setting (see
`docs/CONTRACT.md`), or `KENES_STT_BACKEND=ort|sherpa` to override `Auto`. `Transcriber::backend()`
reports which one is in use. `kenes-transcribe --backend` does the same.

**Build.** The sherpa-onnx build script downloads a prebuilt static library (22 MB `.tar.bz2`, 135 MB
unpacked, cached in `target/sherpa-onnx-prebuilt/`) for linux x64/aarch64 and macOS arm64/x64. You
don't need cmake or clang, only the system C++ runtime (libstdc++ on Linux, libc++ on macOS). Two
environment variables change where the library comes from:

- `SHERPA_ONNX_LIB_DIR` points at libraries you already have. With a shared build, `OrtGetApiBase`
  then comes from its `libonnxruntime`, which works too.
- `SHERPA_ONNX_ARCHIVE_DIR` points at a directory holding the archive, for offline builds.

The community `sherpa-rs` crate wasn't needed. The release `kenes-transcribe` binary is 33 MB.

## Models

Models download on demand to `models_dir()`: `$KENES_MODELS_DIR` if set, otherwise
`~/.local/share/kenes/models` on Linux and `~/Library/Application Support/kenes/models` on macOS.
Each model gets `<models_dir>/<id>/<file>`. `ensure_model(id, …)` also fetches `silero-vad`, since
every ASR model needs it. Every file is pinned by URL, size and SHA-256.

A download streams to `<file>.part`. It resumes with HTTP `Range` if interrupted, and retries twice.
The hash is computed while streaming, and the file is renamed into place only if it matches. On
later starts, a file with the right size under its final name is trusted without hashing it again.
`verify_model` re-hashes everything.

| id | what | languages | download | license |
|---|---|---|---|---|
| `gigaam-multilingual-ctc` (**default**) | Sber GigaAM Multilingual CTC, 220M params, int8 | ru, kk, ky, uz, en (one character vocabulary) | 225 MB | MIT |
| `gigaam-multilingual-large-ctc` | GigaAM Multilingual Large CTC, 600M, int8 | same | 592 MB | MIT |
| `gigaam-v3-ru-ctc` | GigaAM v3 CTC, official sherpa-onnx export | ru only | 225 MB | MIT |
| `silero-vad` | Silero VAD, the sherpa-onnx (k2-fsa) export of v4 | – | 0.6 MB | MIT |

Sources:

- GigaAM Multilingual is a community sherpa-onnx conversion in
  [fgeeer77/bayaya-models](https://github.com/fgeeer77/bayaya-models/releases). It was converted by
  CI with GigaAM's own `to_onnx`, and its output was checked against the original model at a CER
  of 0–0.6 %. sherpa-onnx loads it as a NeMo CTC model (`is_giga_am` metadata).
- GigaAM v3 comes from
  [csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16),
  pinned to revision `32a4c7cc`.
- Silero VAD is `asr-models/silero_vad.onnx` from the sherpa-onnx releases.

The pinned URLs and hashes are in `src/registry.rs`. `available_models()` lists only the three ASR
models. Their `sizeMb` includes the VAD.

## API

This is the contract API. Details are in `docs/CONTRACT.md`.

```rust
let cfg = SttConfig::default();                    // default model, 4 threads, 700 ms partials, 20 s max
ensure_model(&cfg.model_id, &cfg.models_dir, &mut |p| ui_progress(p))?;  // blocking; 0.0..=1.0
let t = Transcriber::new(cfg)?;                    // loads the model: ~1–2 s (backend: Auto)
// or Transcriber::with_backend(cfg, SttBackend::Sherpa)?; t.backend() says which one runs
let handle = t.spawn(audio_rx, segment_tx);        // one worker thread, both sources
// …
drop(audio_tx);                                    // worker flushes open utterances, then exits
handle.join().unwrap();
```

These were added on top of the contract:

- `SttConfig::default()`, `DEFAULT_MODEL`, `VAD_MODEL`.
- `ModelInfo` serializes as `{ id, name, languages, sizeMb, downloaded }`.
- Registry helpers: `available_models_in(dir)`, `is_downloaded(id, dir)`, `model_path(id, dir)`,
  `verify_model(id, dir)`, `sha256_file(path)`.
- `download_verified(url, sha256, size, dest, progress)` downloads one file with the same
  `.part`/resume/hash/rename logic, for crates that ship their own models (`kenes-speakers`). It
  does nothing if `dest` already has that hash.
- `Transcriber::recognize(samples)` decodes one utterance without the VAD, for benchmarks.
  (`recognize_unpadded`, hidden from the docs, skips the tail padding, to compare with
  `bench/run_bench.py`.)
- `SttBackend` (`Auto`/`Ort`/`Sherpa`, serde `"auto"`/`"ort"`/`"sherpa"`, also `FromStr`),
  `Transcriber::with_backend(cfg, backend)`, `Transcriber::backend()`, and `BACKEND_ENV`
  (`"KENES_STT_BACKEND"`). They are in the contract too.
- `LogMel` with `N_FFT`, `HOP`, `N_MELS`: the GigaAM front-end, for benchmarks and feature checks.
- `Transcriber::set_splitter(f)` and `set_split_options(SplitOptions)` split finals at speaker
  changes. See below.

## How streaming works

The `spawn` worker is a single thread. It owns one recognizer and, for each source, a Silero VAD,
an audio buffer and a timeline.

- **VAD.** The detector gets exactly one 512-sample window at a time, and we only read its "in
  speech" flag. The audio is our own copy, so we can decode the open utterance for partials and cut
  where we like. Settings: threshold 0.5, minimum speech 0.15 s, minimum silence 0.5 s.
  `max_speech_duration` is `max_segment_ms / 2`, and past that length sherpa splits at 0.1 s pauses.
- **Gain control, for the VAD only.** Silero misses quiet speech. On FLEURS clips peaking at 0.01
  it dropped whole phrases, and CER went from 1.6 % to 4.9 %. The window fed to the VAD is therefore
  scaled towards 0.08 RMS, at most ×10, never attenuated, with a 1.5 s release. Silero stays at or
  below p = 0.29 on white, pink or brown noise at any level, so this doesn't create false triggers
  on stationary noise. The recognizer still gets the original audio.
- **Utterance start.** The VAD's own start is `2 × window + min_speech` before the trigger. We add
  250 ms of pre-roll, but never overlap the previous utterance.
- **Utterance end.** The end is the VAD's end of speech plus 150 ms (or 50 ms after a soft 0.1 s
  split). Every decode also gets 0.3 s of trailing zeros, because GigaAM drops the last characters
  when the audio stops right after the last phoneme ("марокко" became "маро").
- **Forced cut** at `max_segment_ms`. The cut goes at the quietest 100 ms of the last 3 s, and the
  next utterance continues from that exact sample, so no audio is lost. The VAD is not reset:
  resetting Silero mid-speech makes it miss ongoing speech for seconds (p ≈ 0.02 on clear speech).
  The only exception is 3 × `max_segment_ms` of uninterrupted "speech" (music, for example). Then
  it resets to bound its internal buffer, and a 0.8 s grace period keeps the utterance open.
- **Partials.** When the input queue is empty and an open utterance has at least
  `partial_interval_ms` of new audio, the whole utterance is decoded again and a partial is sent
  with the same id. Every decode starts from scratch, because the model isn't streaming. The cost
  is capped in two ways:
  - Partials may use at most 30 % of wall time, for both sources together. After a partial that
    took `d`, the next one waits `2.3 × d`.
  - A partial is skipped if its predicted decode time, from a running RTF estimate, is over 1.5 s.

  If the input queue backs up, the worker drains it first (decoding finals as they close), so
  partials are the first thing to go.
- **Finals.** A final is decoded as soon as its utterance closes, sent with a blocking `send`, and
  never dropped. It has the same id as its partials. Empty text is skipped, unless partials were
  already shown for that id. Then a final with empty text is sent, meaning "drop it".
- **Ids and time.** Ids are `mic-N` and `system-N`, counted per source from 1. A number is taken
  only when something is emitted, so ids stay consecutive. `start_ms`/`end_ms` map sample positions
  through the `AudioChunk.start_ms` anchors, so gaps in the timeline are respected. Segments are in
  order within a source. Across sources they come in processing order.
- **Speaker-change splits (optional).** In real meetings people answer within 0.5 s or talk over
  each other, so the VAD glues different speakers' turns into one final. kenes-core can install a
  splitter, a closure wrapping `kenes_speakers::change_points`:
  - It is offered every final of at least 2.5 s (`SplitOptions::min_utterance_ms`). It gets
    exactly the samples that would be decoded and returns sample offsets of speaker changes.
  - Offsets are sanitized: out-of-range and duplicate cuts are dropped. Pieces under 0.4 s
    (`min_piece_ms`) are merged into their shorter neighbour.
  - Each piece is decoded on its own, with the usual padding. The first piece is the final for
    the existing id. The others get fresh ids from the same counter, with `start_ms`/`end_ms`
    mapped through the utterance's own copy of the timeline anchors, so gaps still count. Later
    pieces that decode to nothing are skipped. The first piece is emitted even if empty when
    partials were shown for its id, which retracts them.
  - The closure runs on a helper thread, `kenes-stt-split`. The worker waits up to
    `2 s + 0.25 × utterance length` and otherwise emits the utterance unsplit, logs a warning,
    and discards the late answer. While the splitter is still busy with a timed-out request,
    new finals skip it rather than queue behind it.
  - A panic is caught and means "no split" for that utterance; the splitter keeps being used. If
    its thread dies, splitting is turned off.
  - Time spent waiting for the splitter counts against the partial budget, like a partial
    decode. It is logged at debug level and summed in the worker's exit stats.
  - Partials are never split. `transcribe_buffer` applies the splitter too.
- **Text.** Trimmed, with whitespace collapsed. Otherwise it is the raw model output: lowercase,
  no punctuation, numbers spelled out as words.

## What the integrator (kenes-core) needs to know

- **Threads.**
  - `Transcriber::new` loads the model in 1–2 s and takes ~450 MB. Run it and `ensure_model` off
    the UI thread.
  - `spawn` starts one thread named `kenes-stt`. Each decode uses `num_threads` ONNX Runtime
    threads. ORT's thread spinning is turned off through `<models_dir>/ort-cpu.cfg`, which we write
    and pass as `cpu:<path>`.
  - The VAD runs single-threaded on the worker.
  - With a splitter installed there is one more thread, `kenes-stt-split`, where the closure runs.
    It is not joined on shutdown, so a stuck closure can't block it.
- **Channels.**
  - Audio → STT: use an unbounded channel, or a bounded one that holds 30 s or more (two sources
    at 100 ms is 20 chunks/s). The worker never drops audio, but a blocked capture thread will.
  - STT → you: unbounded is best. If it is bounded and full, partials are dropped (`try_send`) and
    finals block.
- **Back-pressure.** When decoding can't keep up (a loaded CPU), the worker stops producing
  partials and keeps producing finals, which then arrive later. Nothing is dropped.
- **Stopping.** Drop every audio `Sender`. The worker decodes the open utterances as finals (up to
  ~1–2 s) and exits. It also exits right away if the segment receiver is dropped.
- **One `Transcriber` per session.** `spawn` consumes it, and ids restart at 1 with each new one.
  Switching models means creating a new one.
- **Storage.** Store finals only, and skip finals with empty text.

## Measured on this machine

Ubuntu 26.04, Intel Meteor Lake (18 threads). Other agents were building and benchmarking at the
same time (load average 6–13), so treat these as upper bounds. The first number in each cell is
from the quietest run.

**Decode time** for one utterance, with `kenes-transcribe --bench`:

| audio | 2 threads | **4 threads** | 8 threads |
|---|---|---|---|
| 1 s | 0.27 s | 0.09–0.13 s | 0.08 s |
| 2 s | 0.42 s | 0.11–0.21 s | 0.13 s |
| 5 s | 0.55 s | **0.19–0.49 s** (RTF 0.04–0.10) | 0.31 s |
| 10 s | 1.06 s | **0.51–1.01 s** (RTF 0.05–0.10) | 0.63 s |
| 20 s | 3.05 s | **1.21–1.85 s** (RTF 0.06–0.09) | 1.48 s |

**Model load and memory:**

| model | load | RSS |
|---|---|---|
| default, 220M | 0.9–1.9 s | 400–470 MB, including 2 VADs |
| `gigaam-v3-ru-ctc` | 1.0 s | 400 MB |
| `-large-ctc`, 600M | 2.8 s | 980 MB, and 5 s decodes in 0.64 s (RTF 0.13–0.16) |

**Live** (`--simulate-live`, 1×, 100 ms chunks, 108 s of kk/ru speech with few pauses):

| scenario | CPU | partials | partial latency | final latency |
|---|---|---|---|---|
| mic only | 0.9 core on average | 84 | median 0.30 s, p90 0.54 s | median 1.09 s, p90 1.27 s |
| mic + system at once (108 s + 86 s) | 1.4 cores | 74 | median 0.39 s, p90 0.80 s | median 1.14 s, p90 1.92 s |

- Partial latency is how old the newest audio in a partial is when the partial arrives.
- Final latency is measured from the segment's `end_ms`. It includes the 0.5 s of silence the VAD
  waits for.
- Before the partial budget and the no-spinning setting, the two-source run used 3.2 cores.
- Offline `transcribe_buffer` runs at RTF 0.05–0.08.

**Accuracy** (CER, default model, text lowercased and stripped of punctuation). These clips are
the benchmark's 60 FLEURS kk, 60 FLEURS ru, 60 Common Voice kk and 30 kk↔ru pairs of FLEURS
sentences. "Whole clip" decodes each file in one pass. "VAD split" is `transcribe_buffer`, the
same path the live worker uses.

| set | whole clip | VAD split |
|---|---|---|
| FLEURS kk | 1.64 % | 1.76 % |
| FLEURS ru | 0.72 % | 0.74 % |
| Common Voice kk | 2.09 % | 2.35 % |
| kk↔ru pairs | 2.34 % | **1.19 %** |

Before the gain control and the VAD tuning, FLEURS kk was at 4.94 %. The pairs do better when
split because each language gets its own decode.

### Sample output (default model, `transcribe_buffer`)

Kazakh (FLEURS):
```
тауарларды тасымалдау үшін кемелерді пайдалану әзірше адамдардың және тауарлардың үлкен мөлшерлерін мұхиттарда жылжытудың ең тиімді жолы болып табылады
```
Russian (FLEURS):
```
ученые надеются понять как образуются планеты особенно то как образовалась земля поскольку кометы сталкивались с землей очень давно
```
Russian then Kazakh (`bench/data/codeswitch/cs01_ru-kk.wav`), in `--simulate-live`:
```
… mic-1     одна из бомб взорвалась возле офиса генерал г
[00:01.580 → 00:06.626] mic-1     одна из бомб взорвалась возле офиса генерал губернатора
… mic-2     атап айтқанда адамның бетіндегі микро
[00:08.812 → 00:16.834] mic-2     атап айтқанда адамның бетіндегі микроөзгерістерді түсіндіру арқылы оның өтірік айтуын анықтауға болады
```
Kazakh and Russian come out in their own scripts: ә ғ қ ң ө ұ ү һ і for Kazakh, and no Kazakh
letters in Russian.

## CLI

```
cargo build --release -p kenes-stt
target/release/kenes-transcribe --list-models
target/release/kenes-transcribe --download gigaam-multilingual-ctc [--verify]
target/release/kenes-transcribe file.wav [--model <id>] [--threads 4]    # any rate/channels
target/release/kenes-transcribe file.wav --simulate-live [--speed 2] [--other system.wav]
target/release/kenes-transcribe file.wav --bench                          # decode time for 1–20 s
```

With `RUST_LOG=kenes_stt=trace` it prints every VAD transition and cut.

## Tests

- `cargo test -p kenes-stt` runs the unit tests, with no models needed. They cover the segmenter
  state machine with a fake VAD (padding, no clipping, forced cuts without losing audio, VAD reset
  and grace, the timeline and gaps, id rules, empty-final retraction, gain control, trimming, odd
  chunk sizes). They also cover the worker with a fake recognizer (partials, then exactly one final
  per id, two sources, back-off with a slow decoder, flush on disconnect, exit when the output is
  dropped, the partial budget) and registry integrity and wire shape. Splitter tests use a fake
  splitter: ids and exact cut timestamps, timeline gaps inside a split utterance, merging of tiny
  and out-of-range cuts, an empty first piece after partials (and offline), a panicking
  splitter, a slow splitter timing out without stalling, and the length threshold. The GigaAM
  backend's tests need no model either: frame count, filterbank shape, feature values against
  the Python formulas (to 1e-4), the log floor for silence and NaN input, `tokens.txt` parsing,
  greedy CTC, and backend selection (names, `Auto` → env → registry, fallback to sherpa when the
  ORT load fails, padding).
- `cargo test -p kenes-stt --release -- --ignored --nocapture` runs the tests with real models and
  audio: script checks for kk and ru, VAD split against whole-clip CER, two sources through the
  live worker at 2× and at full speed, a real download with resume, and a real kk+ru pair with a
  0.2 s gap. The VAD glues that pair into one final, and a gap-finding splitter separates it
  again at exactly 12.47 s, with 0 % CER on both pieces. These run on the default backend (`ort`).
  Two more cover the backend itself:
  - `one_onnx_runtime_for_vad_speakers_and_both_backends` loads the `kenes-speakers` embedder,
    the ORT recognizer (with its two Silero VADs) and the sherpa recognizer in one process. It
    uses all of them from three threads at once and checks the outputs don't change, then drops
    everything and loads again. `kenes-speakers` is a dev-dependency only for this test.
  - `every_model_runs_on_onnx_runtime` loads all three ASR models on ORT (no fallback allowed) and
    checks CER plus empty and too-short input.

  Audio comes from `$KENES_STT_TESTDATA`, then `testdata-cache/` (gitignored: a few FLEURS/CV
  clips, copied from `bench/data/`), then `bench/data/`. Set `KENES_STT_CLIPS=60` for the full
  comparison above.
- `examples/bench_sets.rs` runs the benchmark sets (`bench/data`) through either backend and
  prints WER/CER/RTF. Modes: `whole` (one decode per clip), `unpadded` (same, without the tail
  padding, like `run_bench.py`) and `vad` (the live segmentation). `--out` writes
  `run_bench.py`-style JSON. `examples/gigaam_features.rs` together with
  `scripts/compare_features.py` checks `LogMel` against the Python references.

## Known limits

- **Text format.** No punctuation, casing or digits. Claude restores them downstream.
- **Code-switching.** Across utterances it is at its best, because each utterance is decoded on its
  own. Within an utterance the model copes, since it has one character vocabulary for all its
  languages, but decoding a kk+ru pair as one utterance gave 2.3 % CER against 1.2 % when split.
  The test data had no truly intra-sentential switching (Russian words inside a Kazakh sentence),
  and English or Latin-script terms weren't evaluated.
- **Short words.** A word shorter than about 0.15 s on its own ("да", "иә") can be missed by the
  VAD. A short noise such as a cough may decode to a one- or two-letter final. We don't filter
  those.
- **Pauses in a word.** A pause of 0.5 s or more splits the utterance, and a hesitation or restart
  can then show up twice. For example, one FLEURS clip gave "психология" + "психологияға" where the
  whole-clip decode had "психологияға" once.
- **Session start.** At the start of a stream, and after the rare music reset, Silero's LSTM state
  is empty. Speech already under way at that moment may not be detected until the next pause.
- **Session length.** sherpa-onnx's VAD counts samples in `int32`, which overflows after ~37 h of
  continuous audio per source. This hasn't been tested.
- **Not verified.**
  - macOS. The static osx libraries exist, but nothing was built or run on a Mac.
  - The large model and `gigaam-v3-ru-ctc` in live mode, beyond `--bench`.
  - Noisy real-world microphones. The gain control was only checked against synthetic noise.
