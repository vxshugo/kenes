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

The community `sherpa-rs` crate wasn't needed. The release `kenes-transcribe` binary is 34 MB
(33 MB before `ort`).

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

The pinned URLs and hashes are in `src/registry.rs`. Each entry also names the backend `Auto`
resolves to (`backend: SttBackend::Ort` for all three ASR models). Both backends read the same
`model.int8.onnx` and `tokens.txt`, so switching backends downloads nothing. `available_models()` lists only the three ASR
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
  split). Every decode also gets 0.3 s of trailing silence, because GigaAM drops the last
  characters when the audio stops right after the last phoneme ("марокко" became "маро"). On the
  `sherpa` backend that silence is zeros. On `ort` it is deterministic white noise at −70 dBFS:
  zeros hit the exact front-end's `log(1e-9)` floor (−20.7, against about −11 for that noise),
  far below any recorded silence, and that costs accuracy. On Common Voice kk (220M, whole clips)
  WER was 12.1 % with zeros, 10.7 % unpadded and 8.2 % with the noise.
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
    threads (intra-op; inter-op 1) on either backend. ORT's thread spinning is turned off: for
    `ort` through session options, for sherpa-onnx through `<models_dir>/ort-cpu.cfg`, which we
    write and pass as `cpu:<path>`.
  - **Backend.** `Transcriber::new` uses `SttBackend::Auto`, which today means `ort` for every
    model. To honour the `sttBackend` setting (`"auto" | "ort" | "sherpa"`), read it as a string
    and call `Transcriber::with_backend(cfg, s.parse().unwrap_or_default())`, so that an unknown
    value means `Auto`. If `ort` can't load the model it falls back to sherpa by itself and logs a
    warning. `backend()` reports which one ended up running, for example for the status line.
  - The ORT backend shares sherpa-onnx's ONNX Runtime and its environment with the VADs and
    `kenes-speakers`' embedder, so there is nothing to initialise and no order to respect.
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
from the quietest run. The decode-time and load tables were measured on the `sherpa` backend,
before `ort` existed. The next section compares the two backends.

### `ort` vs `sherpa` backend

**Features.** `examples/gigaam_features.rs` and `scripts/compare_features.py` compared `LogMel`
with the references on 6 clips (2 kk↔ru pairs, FLEURS ru, FLEURS kk, 2 Common Voice kk; 427–1991
frames each). Max |Δ| of the log-mel values:

| against | max \|Δ\| | mean \|Δ\| |
|---|---|---|
| the same formulas in float64 | 9.5e-7 (f32 rounding) | 1–3e-7 |
| `bench/run_bench.py` `GigaAMLogMel` (numpy) | 3.0e-5 to 8.7e-4 | 3e-7 to 2.3e-6 |
| `gigaam.preprocess.FeatureExtractor` (torchaudio, what the model was trained with) | 1.6e-4 to 2.4e-3 | 2–7e-6 |
| (numpy vs torchaudio, for scale) | 1.6e-4 to 2.5e-3 | 2–8e-6 |

The int8 model amplifies differences that small. With the same onnxruntime, numpy and
torchaudio features give different transcripts on 11 of the 210 benchmark clips (WER moves up to
±0.35 per set), and noise realisations in the padding move cv_kk by ±0.8. Rust (ORT 1.28.2) and
`run_bench.py` (ORT 1.23.2) differ on 12 of 210 clips, all single-letter flips at uncertain
spots. "Match" below means within that noise.

**Accuracy** (WER / CER %, `examples/bench_sets.rs`, 4 threads, scored like `run_bench.py`; the
scores agree with its jiwer-based `score()` to the digit). Modes: *unpadded* is one decode per clip
with nothing appended, which is what `run_bench.py` does. *whole* is `Transcriber::recognize`,
one decode per clip plus the tail padding. *VAD* is `transcribe_buffer`, the live segmentation.

| model | mode | backend | fleurs_ru | fleurs_kk | cv_kk | codeswitch |
|---|---|---|---|---|---|---|
| 220M | unpadded | Python ORT (`RESULTS.md` ortfeat) | 4.2 / 0.7 | 5.8 / 1.6 | 10.1 / 2.4 | 5.1 / 1.0 |
| | | **ort** | 3.81 / 0.67 | 5.80 / 1.62 | 10.68 / 2.39 | 5.44 / 1.03 |
| | | Python sherpa (`RESULTS.md`) | 4.0 / 0.7 | 6.1 / 1.6 | 8.8 / 2.0 | 7.5 / 2.3 |
| | | sherpa | 3.98 / 0.72 | 6.12 / 1.65 | 8.77 / 2.00 | 7.46 / 2.27 |
| | whole | **ort** | 3.89 / 0.68 | 5.59 / 1.62 | 8.22 / 1.83 | 5.56 / 1.04 |
| | | sherpa | 4.06 / 0.72 | 5.80 / 1.64 | 8.49 / 2.09 | 7.46 / 2.34 |
| | VAD | **ort** | 4.31 / 0.83 | 6.87 / 1.78 | 8.49 / 2.09 | 5.18 / 1.11 |
| | | sherpa | 3.81 / 0.74 | 6.55 / 1.76 | 9.32 / 2.35 | 5.31 / 1.19 |
| large 600M | unpadded | Python ORT (`RESULTS.md`) | 2.2 / 0.4 | 4.5 / 1.4 | 8.8 / 2.0 | 4.4 / 0.7 |
| | | **ort** | 2.20 / 0.42 | 4.51 / 1.44 | 9.04 / 2.04 | 3.92 / 0.69 |
| | | Python sherpa (`RESULTS.md`) | 2.3 / 0.4 | 4.7 / 1.5 | 7.1 / 1.5 | 5.7 / 0.8 |
| | | sherpa | 2.37 / 0.44 | 4.62 / 1.47 | 7.12 / 1.44 | 5.69 / 0.79 |
| | whole | **ort** | 2.20 / 0.42 | 4.51 / 1.44 | 8.49 / 1.87 | 4.42 / 0.74 |
| | | sherpa | 2.28 / 0.43 | 4.83 / 1.47 | 7.40 / 1.57 | 5.82 / 0.82 |
| | VAD | **ort** | 2.45 / 0.48 | 5.26 / 1.55 | 7.40 / 1.96 | 4.68 / 1.04 |
| | | sherpa | 2.54 / 0.48 | 5.80 / 1.58 | 6.30 / 1.30 | 4.42 / 0.96 |
| v3 ru | unpadded | **ort** / sherpa | 3.13 / 0.58 · 2.71 / 0.52 | – | – | ru half WER 7.3 · 12.3 |
| | whole | **ort** / sherpa | 3.05 / 0.54 · 2.79 / 0.54 | – | – | ru half WER 7.8 · 14.2 |
| | VAD | **ort** / sherpa | 3.21 / 0.60 · 3.13 / 0.61 | – | – | ru half WER 2.6 · 2.8 |

Sizes: fleurs_ru 1182 words, fleurs_kk 931, cv_kk 365, codeswitch 791, so one word is 0.1, 0.1,
0.3 and 0.13 points. Reading the table:

- The Rust backends reproduce the Python benchmark rows within noise, sherpa almost to the
  digit.
- **One decode that spans a pause** is where `ort` wins. The code-switch clips are two FLEURS
  sentences with their own leading and trailing silence plus 0.3 s of zeros, decoded whole.
  sherpa's front-end garbles the part after the pause (220M: ru half WER 9.0 vs 5.9, CER 2.3 vs
  1.0; large: 5.8 vs 4.4 WER). It isn't the zeros: filling the gap with noise gave sherpa the same
  7.46 %. The live worker only decodes across a pause of 0.5 s or more when the VAD doesn't hear
  it as silence (background noise, music). Offline decodes of longer audio hit it more often.
- **The live path (VAD)** splits at 0.5 s pauses, and there the two backends are at parity. Total
  word errors over the four sets are 187 (`ort`) vs 182 (sherpa) of 3269 for the 220M model, and
  142 vs 142 for the large one. cv_kk swings either way by 3–4 words.
- A kk↔ru switch *without* a pause is also at parity: 30 trimmed FLEURS kk+ru pairs joined with
  50 ms of room noise, so the VAD keeps each pair in one utterance. VAD mode, 220M: `ort` 5.56 /
  1.45, sherpa 5.37 / 1.62. Large: 3.83 / 1.19 vs 3.64 / 1.24.
- `gigaam-v3-ru-ctc` on pure Russian: sherpa is 1–5 words better, which is within noise but
  not a win for `ort`. On Russian after Kazakh, `ort` is much better.

**Speed.** Same runtime, same speed. `kenes-transcribe --bench` (median of 3, 4 threads) at
load average 1.3–2.7, backends alternating:

| model | backend | 5 s | 10 s | 20 s | model load | RSS |
|---|---|---|---|---|---|---|
| 220M | **ort** | 0.17–0.18 s | 0.33–0.38 s (RTF 0.033–0.038) | 0.74–0.83 s | 0.72 s | 459 MB |
| 220M | sherpa | 0.17–0.26 s | 0.34–0.48 s (RTF 0.034–0.048) | 0.74–1.04 s | 0.72 s | 455 MB |
| large 600M | **ort** | 0.40 s | 0.77 s (RTF 0.077) | 1.62 s | 1.5 s | 955 MB |
| large 600M | sherpa | 0.42 s | 0.78 s (RTF 0.078) | 1.89 s | 1.6 s | 997 MB |
| v3 ru | **ort** | 0.16 s | 0.34 s (RTF 0.034) | 0.71 s | 0.85 s | 424 MB |
| v3 ru | sherpa | 0.20 s | 0.35 s (RTF 0.035) | 0.70 s | 0.70 s | 414 MB |

RSS includes the two VADs. `bench/RESULTS.md` has pip onnxruntime 1.8× faster than the
sherpa-onnx wheel, and that doesn't reproduce here. Back to back, the sherpa-onnx wheel, pip
onnxruntime 1.23 and both Rust backends all ran the 220M model at RTF 0.031–0.04, and the large
model on pip onnxruntime at 0.076–0.08. Official ONNX Runtime 1.28.2 and 1.30.0 builds, loaded
with `dlopen` in place of the bundled one (a quick experiment, not in the code), were within ~15% of
it on the large model. So a separate runtime wouldn't buy speed either.

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

| scenario | backend | CPU | partials | partial latency | final latency |
|---|---|---|---|---|---|
| mic only | **ort** | 0.58–0.63 core | 103–106 | median 0.15–0.16 s, p90 0.29–0.33 s | median 0.64–0.70 s, p90 0.74–0.80 s |
| mic only | sherpa | 0.62–0.64 core | 102–103 | median 0.16 s, p90 0.30–0.35 s | median 0.67–0.73 s, p90 0.88–0.90 s |
| mic + system at once (108 s + 86 s) | **ort** | 0.93–0.98 core | 149–154 | median 0.19 s, p90 0.35–0.37 s | median 0.67–0.69 s, p90 0.83–0.86 s |
| mic + system at once | sherpa | 0.89–1.02 core | 139–164 | median 0.16–0.20 s, p90 0.30–0.39 s | median 0.64–0.74 s, p90 0.82–0.91 s |
| mic only, earlier (load 6–13) | sherpa | 0.9 core | 84 | median 0.30 s, p90 0.54 s | median 1.09 s, p90 1.27 s |
| mic + system, earlier (load 6–13) | sherpa | 1.4 cores | 74 | median 0.39 s, p90 0.80 s | median 1.14 s, p90 1.92 s |

The `ort`/`sherpa` rows are two alternating rounds each at load average 1.8–4.1. The finals in
both runs are the same text except for two words, and `ort` gets "архипелагтар" right where
sherpa writes "архипелакгтар".

- Partial latency is how old the newest audio in a partial is when the partial arrives.
- Final latency is measured from the segment's `end_ms`. It includes the 0.5 s of silence the VAD
  waits for.
- Before the partial budget and the no-spinning setting, the two-source run used 3.2 cores.
- Offline `transcribe_buffer` runs at RTF 0.05–0.08.

**Accuracy** (CER, default model, text lowercased and stripped of punctuation). These clips are
the benchmark's 60 FLEURS kk, 60 FLEURS ru, 60 Common Voice kk and 30 kk↔ru pairs of FLEURS
sentences. "Whole clip" decodes each file in one pass. "VAD split" is `transcribe_buffer`, the
same path the live worker uses.

| set | whole clip, sherpa | whole clip, **ort** | VAD split, sherpa | VAD split, **ort** |
|---|---|---|---|---|
| FLEURS kk | 1.64 % | 1.62 % | 1.76 % | 1.78 % |
| FLEURS ru | 0.72 % | 0.68 % | 0.74 % | 0.83 % |
| Common Voice kk | 2.09 % | 1.83 % | 2.35 % | 2.09 % |
| kk↔ru pairs | 2.34 % | **1.04 %** | 1.19 % | 1.11 % |

Before the gain control and the VAD tuning, FLEURS kk was at 4.94 % (sherpa). With sherpa, the
pairs do better when split because each part gets its own decode. With `ort`, a whole-clip decode
is already as good.

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
- **The ONNX Runtime comes with sherpa-onnx.** The `ort` backend uses whatever runtime the pinned
  sherpa-onnx links. A sherpa-onnx upgrade therefore changes the runtime under both backends. A
  runtime older than C API 17 (ONNX Runtime 1.17) makes `ort` fall back to sherpa.
- **`ort` is a release candidate** (`=2.0.0-rc.13`, pinned exactly). Only its session and tensor
  API is used, with `alternative-backend`.
- **Not verified.**
  - macOS. `cargo check -p kenes-stt --all-targets` passes for `aarch64-apple-darwin` and
    `x86_64-apple-darwin`; `ring` needs a C cross compiler, and `zig cc` was used as `CC`. Both osx
    archives define `_OrtGetApiBase` and carry ONNX Runtime 1.28.2. Nothing was linked or run on
    a Mac.
  - The large model and `gigaam-v3-ru-ctc` in live mode, beyond `--bench`.
  - Noisy real-world microphones. The gain control was only checked against synthetic noise.
