# kenes-audio

Captures the microphone and system audio (what the other people on a call say) as two
separate 16 kHz mono `f32` streams. Chunks are 512 samples (32 ms) and carry `start_ms` on
one session clock that both sources share. The two sources are never mixed.

```rust
let (tx, rx) = crossbeam_channel::unbounded();
let handle = kenes_audio::start_capture(CaptureConfig::default(), tx)?; // mic + system, default devices
for chunk in rx { /* chunk.source, chunk.start_ms, chunk.samples */ }
// handle.errors(): a Receiver<CaptureError> for streams that die mid-session
// drop(handle) or handle.stop(): stops and joins everything (takes a few ms)
```

`list_devices()` returns microphones (`DeviceKind::Input`) and output loopbacks
(`DeviceKind::Monitor`). Pass a device's `id` back as `DeviceSel::Id`. `wav::read` loads any
WAV file as 16 kHz mono, and `wav::write` / `wav::WavWriter` write 16 kHz mono 16-bit files.

## Linux

Linux has no library bindings, so building needs no `-dev` packages. Each source is a
`parec` subprocess:

```
parec --device=<source> --format=s16le --rate=16000 --channels=1 --latency-msec=20 --raw \
      --client-name=kenes --stream-name=kenes-mic|kenes-system
```

The sound server (PipeWire or PulseAudio) resamples and downmixes. A reader thread per
stream decodes stdout, handling reads that end mid-sample, and cuts it into chunks.

- **Devices.** They come from `pactl -f json list sources`. If JSON output isn't available,
  we fall back to `pactl list short sources`. Sources named `*.monitor` are system-audio
  devices, and the default one is `<default sink>.monitor`.
- **Default device.** `DeviceSel::Default` maps to `@DEFAULT_SOURCE@` and `@DEFAULT_MONITOR@`.
  The server fixes the actual device when the stream connects. If the user changes the
  default device later, the running stream stays where it is, so restart capture to follow
  the new default.
- **Device disappears.** If the selected device goes away (for example, it is unplugged),
  PipeWire moves the stream to the current default of the same kind and no error is
  reported. We saw this happen with a monitor.
- **Unknown device IDs.** `start_capture` rejects them up front. It also accepts a sink name
  for `system` and records that sink's monitor.
- **`pw-record` fallback.** If `parec` isn't installed, we use `pw-record` with the same raw
  s16 output. For a monitor, we target the sink with `stream.capture.sink=true`, because
  targeting `<sink>.monitor` quietly records the default mic instead. Set
  `KENES_AUDIO_BACKEND=parec|pw-record` to force one of them.
- **Recorder dies.** If the recorder exits by itself (it crashed or the server went away),
  we send one `CaptureError` that includes the tail of its stderr. That stream then ends and
  the other one keeps running. There is no automatic restart.
- **Stopping.** Stop kills the child, joins the reader, and reaps the child. Recorders run
  in their own process group, so a terminal Ctrl-C doesn't reach them. If our process dies,
  they get `EPIPE` on their next write and exit.

## macOS (untested)

This code was written on Linux. It type-checks with
`cargo check --target aarch64-apple-darwin`, but it has never been run.

- **Mic.** Uses a cpal input stream on the default or chosen input device.
- **System audio.** Uses cpal 0.18's loopback. Opening an input stream on an *output*
  device makes cpal create a Core Audio process tap over all processes, plus a private
  aggregate device. cpal requires macOS 14.6 or later for this. cpal can only do this for
  output-only devices. A duplex device, such as some USB headsets, gets an error instead of
  a recording of its mic.
- **Format conversion.** Streams are opened as `f32` and downmixed to mono in the callback.
  They are resampled to 16 kHz with the crate's own windowed-sinc resampler (`pcm::Resampler`)
  on a worker thread.
- **Permissions.** The app bundle needs:
  - `NSMicrophoneUsageDescription` in Info.plist, for the mic prompt.
  - `NSAudioCaptureUsageDescription` in Info.plist, for the system-audio or process-tap prompt.
  - The `com.apple.security.device.audio-input` entitlement, for hardened-runtime or sandboxed
    builds.

  If permission is missing, macOS usually delivers silence and no error. When you run
  `kenes-rec` from a terminal, the prompts are for the terminal app.

## Timing

`start_ms` counts from the session start (`CaptureHandle::session_start()`, taken inside
`start_capture`) on a monotonic clock.

- **Anchoring.** Each stream is placed on that clock when its first audio arrives. This is
  typically 140–200 ms after the session starts, and the two sources are within about 20 ms
  of each other. After that, time advances by sample count, so chunks of one stream are
  exactly contiguous and have no jitter.
- **Lost audio.** If a stream loses audio, wall-clock time runs ahead of the sample count.
  Once that lag is over 250 ms and has lasted 1 s, the stream's timeline jumps forward. The
  jump shows up as a gap in `start_ms`.
- **Clock drift.** Different device clocks can drift apart by a fraction of a second per hour.

## Known limits

- **No echo cancellation yet.** Without headphones, the mic also picks up the other side
  from the speakers. Their speech then shows up in both streams, and the mic copy is
  labeled as the user. Use headphones.
- **`start_capture` needs about 150 ms.** It waits for the first audio from every stream,
  so startup errors come back synchronously.
- **Keep the consumer fast.** Use an unbounded channel or drain it promptly. A blocked
  consumer stalls the reader, and then the server drops audio.

## kenes-rec

Use this to record real meetings for the ASR benchmark:

```
cargo run -p kenes-audio --bin kenes-rec -- --list
cargo run -p kenes-audio --bin kenes-rec -- --out rec/standup --seconds 1800
cargo run -p kenes-audio --bin kenes-rec -- --out rec/call --system alsa_output.usb-headset.monitor --no-mic
```

- **Output.** It writes `<out>/mic.wav` and `<out>/system.wav` as 16 kHz mono 16-bit files.
  Both are padded with silence to the session timeline, so sample N in one file lines up
  with sample N in the other.
- **Meter.** While recording, it prints a live RMS meter for each source every 0.5 s.
- **Stopping.** It runs until `--seconds` or Ctrl-C, and both finalize the files. The WAV
  headers are rewritten every 2 s, so a crash still leaves playable files. A second Ctrl-C
  exits immediately.
- **Overwriting.** It won't overwrite existing files unless you pass `--force`.

## Tests

```
cargo test -p kenes-audio                 # unit tests: PCM decoding, chunking, resampling, pactl parsing, WAV
cargo test -p kenes-audio -- --ignored    # real capture against the running sound server
```

The ignored tests create temporary `module-null-sink`s, play a 1 kHz tone into them with
`paplay`, and capture the sink's monitor with `parec` and `pw-record`. They check the level,
frequency, timeline, how fast stop is, and that no threads or processes are leaked. They
also kill a recorder to check error reporting, and record 2 s from the default mic and
system audio. Nothing plays through the speakers, and the sinks are always unloaded.
