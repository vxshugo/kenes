# Echo cancellation: evaluation

Status 2026-09-27. Everything here is synthetic: FLEURS / Common Voice speech played through a
simulated laptop speaker and room. No real laptop, room or call has been measured yet.

Reproduce:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo run --release -p kenes-core --example aec_eval                 # signal level + recognition, ~8 min
cargo run --release -p kenes-core --example aec_eval -- --sweep      # delay sweep, clock drift
cargo run --release -p kenes-core --example aec_eval -- --export DIR # a 3 min call for kenes-cli --replay-*
cargo run --release -p kenes-aec --example aec_bench                 # CPU
```

## What was built

- **Signal level, `kenes-aec`.** WebRTC AEC3 through [`sonora`](https://crates.io/crates/sonora) 0.2
  (pure-Rust port of WebRTC's audio processing module, BSD-3; only its echo canceller is enabled,
  without AEC3's high-pass filter). The system stream is the far-end reference. Around it:
  - `StreamCanceller` pairs mic and system audio by `start_ms` in 10 ms frames. A mic frame waits for
    the system audio of the same moment, at most 150 ms of mic audio, then goes without it.
  - The mic is delayed 10 ms before AEC3, so the reference always leads its echo. With the echo
    exactly simultaneous with the reference (0 ms on the timeline), AEC3 alone reached only 5 dB ERLE
    in one scene of the sweep; with the lead, 24 dB.
  - A coarse delay tracker (10 ms log-energy envelopes, normalized cross-correlation, 0–1000 ms, a
    new estimate every 0.5 s, confirmed after 3 agreeing ones) pre-delays the reference when the echo
    is more than 400 ms late, which is where AEC3's own search (about 450 ms at 16 kHz) ends, or when a
    pre-delayed echo drifts to under 30 ms. Each re-alignment starts a fresh AEC3 instance.
  - Pass-through: the output is the input sample for sample, only delayed, while the far end has been
    silent for 1 s, and while the far end plays but never reaches the mic (headphones). An echo-path
    detector watches frames with a loud far end: a mic that stays at its noise floor means no echo
    path; a mic that goes in 12 dB above the floor and comes out of AEC3 at the floor means echo was
    removed. It starts assuming speakers, switches to pass-through after 3 s of evidence with under
    10% "echo removed", and back after half of the latest 200 evidence frames show echo.
  - Latency: fixed 18 ms (10 ms lead + AEC3's 8 ms framing), taken out of the timestamps, so
    transcript times and the speaker ring buffer are unchanged.
- **Transcript level, `kenes_core::echo_guard`.** Holds a mic final while the call had sound in the
  1.5 s before it and a matching system final may still come (at most 2.5 s). Drops it if at least
  60% of its character trigrams occur in the system finals within ±1.5 s (one-word finals: the word
  itself must occur there). A dropped final becomes an empty final if the UI had partials for it,
  otherwise disappears; it is never stored or labeled. System segments are never held.

Both are on when `echoCancellation` is true (default) and both sources are captured.

## Choosing the canceller

All candidates run through the same harness on 12 scenes of the test set below (lag-compensated;
recognition with an earlier `kenes-transcribe` build). AEC3 rows with its high-pass filter off, except the
recognition columns, which were measured with it on.

| Candidate | ERLE first far turn | ERLE later far turn | Near-only SI-SDR (in 40.2) | Double-talk SI-SDR (in 14.2) | Far words leaked (452) | Near WER | Double-talk WER | Delay range |
|---|---|---|---|---|---|---|---|---|
| raw mic | 0 | 0 | 40.2 | 14.2 | 453 (100%) | 6.6% | 5.8% | – |
| `sonora` 0.2 (AEC3) | 23.1 dB | 31.2 dB | 40.2 | 14.4 | 2 (0.4%) | 5.6% | 5.8% | ≤ 450 ms |
| `aec3` 0.4 (another AEC3 port) | 22.2 dB | 30.9 dB | 40.2 | 14.6 | 2 (0.4%) | 5.6% | 5.3% | ≤ 450 ms, ≤ 800 ms with 10 matched filters |
| `decibri-aec` 0.2 (FDAF + conservative suppression) | 8.8 dB | 20.9 dB | 40.2 | 32.2 | 82 (18%) | 5.1% | 4.3% | ≤ 800 ms, slow to lock |

Not measured: `aec-rs` (speexdsp; its build needs cmake and bindgen/clang), `webrtc-audio-processing`
(C++ build), `fdaf-aec` (a single-block frequency-domain filter: the filter length is the frame
length, so a 250 ms echo tail means 250 ms of latency, and no delay estimation).

AEC3 is the only design that stops far-end-only echo from reaching the recognizer (0.4% of far words
leak vs 18% with decibri's gentler suppression). The two AEC3 ports are equivalent in quality and
CPU. `sonora` was picked for its maturity (about 10× the downloads, the plain WebRTC APM API; `aec3`
describes its graph API as a work in progress). `aec3` would give a configurable delay range; the
delay tracker makes that unnecessary. AEC3's high-pass filter is off because it alters the mic even
when the far end is silent (white noise in, correlation 0.93 with the output; 1.0000 without it).

## Test set

`sim.rs` in `crates/kenes-core/examples/aec_eval/`, seeded, 24 scenes (21 min):

- Near end (the user): FLEURS ru and kk, alternating per scene. Far end: FLEURS in the other language
  and Common Voice kk; no utterance is used on both sides. Speech at −26 dBFS (active RMS) for both
  the user at the mic and the far end in the system stream.
- Each scene: far end alone, 1 s gap, user alone, 1 s gap, double talk (the far end starts 0.5 s into
  the user's turn), 1 s gap, far end alone again (the "converged" measurement).
- Echo path: soft clipping (`tanh`, loud peaks lose about 3 dB) in half the scenes; a laptop speaker
  (high-pass at 250–450 Hz, a 2–6 dB resonance at 1–3 kHz); a synthetic impulse response with 6 early
  reflections within 15 ms, direct-to-reverberant ratio 0–8 dB and an exponential tail, RT60
  0.2–0.5 s; a pure delay of 20–250 ms; echo level −3 to −20 dB relative to the user's speech; pinkish
  noise at −35 to −45 dB relative to the user.
- "Headphones": the same scenes with the echo removed from the mic.

Metrics: ERLE = mic power / output power over far-only turns (including the echo tail); SI-SDR of
the output against the user's clean speech over the user-only turn and the double-talk turn;
recognition with `Transcriber::transcribe_buffer` (gigaam-multilingual-ctc, kenes-stt's default
backend), each segment assigned to the turn containing its midpoint. "Leaked" counts every word
recognized in far-only turns (all of it is echo) against the far end's word count. WER is against
the user's reference text in user-only and double-talk turns (lowercased, punctuation dropped).

## Signal level (24 scenes)

ERLE in the two far-only turns (the first includes AEC3's convergence), SI-SDR of the user's
speech in and out:

| # | delay ms | echo dB | clip | RT60 s | ERLE fe1 | ERLE fe2 | NE SI-SDR in→out | NE gain dB | DT SI-SDR in→out |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 192 | -15 | no | 0.48 | 21.3 | 23.4 | 41.5 → 41.5 | +0.00 | 19.1 → 15.0 |
| 1 | 35 | -9 | no | 0.38 | 15.5 | 35.2 | 40.7 → 40.7 | +0.00 | 21.5 → 22.7 |
| 2 | 189 | -7 | yes | 0.32 | 14.5 | 33.4 | 39.6 → 39.6 | +0.00 | 11.0 → 13.2 |
| 3 | 100 | -11 | yes | 0.39 | 27.5 | 33.7 | 43.4 → 43.4 | +0.00 | 18.7 → 16.9 |
| 4 | 240 | -8 | yes | 0.34 | 29.0 | 35.7 | 40.5 → 40.5 | +0.00 | 15.9 → 14.2 |
| 5 | 37 | -14 | yes | 0.44 | 14.3 | 29.9 | 42.9 → 42.9 | +0.00 | 18.7 → 14.7 |
| 6 | 214 | -4 | no | 0.21 | 22.0 | 41.0 | 41.8 → 41.8 | +0.00 | 12.9 → 13.0 |
| 7 | 32 | -12 | yes | 0.25 | 27.4 | 26.1 | 36.9 → 36.9 | +0.00 | 13.9 → 10.3 |
| 8 | 169 | -4 | no | 0.35 | 27.5 | 39.4 | 44.0 → 44.0 | +0.00 | 9.0 → 9.9 |
| 9 | 115 | -8 | no | 0.27 | 23.1 | 27.4 | 34.8 → 34.8 | +0.00 | 5.9 → 9.5 |
| 10 | 140 | -7 | no | 0.49 | 29.5 | 27.5 | 37.0 → 37.0 | +0.00 | 10.1 → 8.1 |
| 11 | 110 | -12 | no | 0.49 | 21.1 | 27.7 | 39.7 → 39.7 | +0.00 | 15.0 → 14.6 |
| 12 | 119 | -10 | yes | 0.27 | 16.2 | 35.6 | 42.1 → 42.1 | +0.00 | 13.4 → 6.8 |
| 13 | 239 | -10 | yes | 0.27 | 15.5 | 35.7 | 44.4 → 44.4 | +0.00 | 17.5 → 13.9 |
| 14 | 157 | -3 | yes | 0.40 | 30.5 | 31.8 | 39.0 → 39.0 | +0.00 | 6.7 → 9.1 |
| 15 | 83 | -7 | yes | 0.34 | 31.5 | 32.3 | 35.2 → 35.2 | +0.00 | 7.8 → 8.2 |
| 16 | 65 | -14 | no | 0.44 | 23.1 | 27.6 | 41.3 → 41.3 | +0.00 | 21.1 → 14.2 |
| 17 | 197 | -7 | yes | 0.27 | 28.7 | 37.0 | 41.2 → 41.2 | +0.00 | 13.3 → 13.8 |
| 18 | 245 | -4 | yes | 0.42 | 36.1 | 32.8 | 39.9 → 39.9 | +0.00 | 12.1 → 12.2 |
| 19 | 83 | -7 | no | 0.27 | 14.2 | 29.0 | 34.5 → 34.5 | +0.00 | 8.7 → 10.4 |
| 20 | 196 | -19 | yes | 0.38 | 22.5 | 22.2 | 41.2 → 41.2 | +0.00 | 25.3 → 19.2 |
| 21 | 43 | -12 | no | 0.23 | 31.2 | 33.1 | 43.2 → 43.2 | +0.00 | 20.3 → 18.2 |
| 22 | 92 | -16 | yes | 0.42 | 18.4 | 20.7 | 36.0 → 36.0 | +0.00 | 21.3 → 13.7 |
| 23 | 42 | -4 | no | 0.26 | 35.0 | 34.1 | 37.3 → 37.3 | +0.00 | 8.5 → 9.4 |
| **mean** | | | | | **24.0** | **31.3** | **39.9 → 39.9** | **+0.00** | **14.5 → 13.0** |

- Far end alone: 24.0 dB of echo removed on the first far turn, 31.3 dB once converged (20.7–41.0
  per scene). The noise floor limits what ERLE can show (echo-to-noise 15–42 dB).
- User alone: bit-exact (39.9 → 39.9 dB, gain ±0.00 dB): the far end is silent, so the mic passes
  through.
- Double talk: 14.5 → 13.0 dB SI-SDR on average. AEC3 removes the echo but also attenuates the
  user while both talk; per scene from −6.6 dB (scene 12) to +3.6 dB (scene 9).
- Headphones (the far end plays, no echo reaches the mic): user alone 39.9 → 39.9 dB, double talk
  39.7 → 39.7 dB: the echo-path detector switches to pass-through during the first far turn, before
  anyone talks over each other. Before the detector existed, AEC3 alone cost 39.7 → 24.3 dB in the
  double-talk turns and 4.2% → 7.2% WER.

## Recognition (24 scenes)

`Transcriber::transcribe_buffer` with gigaam-multilingual-ctc on kenes-stt's default (ONNX Runtime)
backend. The text guard is replayed on the finals as a session would see them (each final 300 ms
after its audio ends, a partial 700 ms into longer utterances, polled every 100 ms).

| mic transcript | far-end words leaked (far-only regions) | near-end WER, near-only | near-end WER, double talk |
|---|---|---|---|
| no AEC | 805 / 801 (100.5%) | 6.8% | 6.0% |
| no AEC + text guard | 4 / 801 (0.5%) | 6.8% | 6.0% |
| AEC | 7 / 801 (0.9%) | 7.1% | 6.7% |
| AEC + text guard | 7 / 801 (0.9%) | 7.1% | 6.7% |
| headphones, no AEC | 0 / 801 (0.0%) | 6.1% | 4.2% |
| headphones, AEC | 0 / 801 (0.0%) | 6.6% | 4.2% |

- Echo in the mic transcript is a near-copy of the call: without any cancellation 805 words were
  recognized in far-only turns, as many as the far end said.
- Either layer alone removes almost all of it: AEC 0.9%, text guard 0.5% (it held 113 mic finals,
  45 ms on average, and dropped 55 of them, all echo). With the AEC on, the guard had nothing left
  to drop (61 finals held, 230 ms on average): what the AEC leaves are garbled fragments from the
  first seconds of the first far turn, while AEC3 converges, which no longer match the call's text:

  - scene 5, 1.1–4.8 s: «была»
  - scene 12, 2.6–3.8 s: «сааттан кейн»
  - scene 13, 0.8–4.4 s: «не похож н»
  - scene 19, 3.2–5.2 s: «к»
- The user's words: 6.8% → 7.1% when the user talks alone, where the audio is bit-exact; the
  difference is the recognizer's VAD segmenting differently after a quieter (echo-free) mic.
  +0.7 points in double talk, where AEC3 attenuates the user.
- Headphones: double talk 4.2% either way (before the echo-path detector: 7.2%).

Text guard threshold, from every mic final that overlaps system finals (both the raw and the
processed mic; one-word finals are excluded here because they need the exact word). 57 of the 58
echo finals score 0.9 or more; all of the user's own finals score below 0.4:

| threshold | echo finals caught | user finals dropped |
|---|---|---|
| 0.3 | 58 / 58 | 1 / 51 |
| 0.4 | 57 / 58 | 0 / 51 |
| 0.5 | 57 / 58 | 0 / 51 |
| 0.6 | 57 / 58 | 0 / 51 |
| 0.7 | 57 / 58 | 0 / 51 |
| 0.8 | 57 / 58 | 0 / 51 |
| 0.9 | 57 / 58 | 0 / 51 |

0.6 sits in the middle of the gap. The one echo final that scores low is a garbled fragment. The
guard dropped no user words: WER with and without it is identical.

## Delay and clock drift

The mic's timeline is aligned with the system audio's by `start_ms`, so what remains is the acoustic
path plus output latency. Scenes 0–9 of the base set with a fixed delay, first and later far-only
turn; "AEC3 alone" is plain `sonora` without the lead or the tracker:

| Echo delay | AEC3 alone | kenes-aec | Reference pre-delay |
|---|---|---|---|
| 0 ms | 10.8 / 5.0 dB | 21.8 / 23.6 dB | 0 ms |
| 50 ms | 18.6 / 35.2 dB | 16.1 / 35.3 dB | 0 ms |
| 150 ms | 11.9 / 33.4 dB | 20.5 / 33.5 dB | 0 ms |
| 250 ms | 28.0 / 33.7 dB | 28.2 / 33.7 dB | 0 ms |
| 350 ms | 33.2 / 35.6 dB | 31.7 / 35.7 dB | 0 ms |
| 450 ms | 10.8 / 29.9 dB | 9.2 / 29.8 dB | 350 ms |
| 550 ms | 9.7 / 3.5 dB | 7.5 / 41.0 dB | 450 ms |
| 700 ms | 2.1 / 0.0 dB | 11.3 / 25.7 dB | 600 ms |
| 850 ms | 1.6 / 0.0 dB | 13.7 / 39.1 dB | 750 ms |
| 1000 ms | 1.1 / 1.8 dB | 7.6 / 26.8 dB | 900 ms |

Clock drift between a mic and speakers on different clocks moves the echo continuously. 20-minute
calls, ERLE over far-only turns, averaged per 4 minutes:

| Drift, echo delay | 0–4 min | 4–8 | 8–12 | 12–16 | 16–20 | Re-alignments |
|---|---|---|---|---|---|---|
| +300 ppm, 150 → 510 ms | 22.9 | 24.2 | 21.6 | 20.2 | 21.1 | 1 |
| -300 ppm, 400 → 40 ms | 22.6 | 26.2 | 22.9 | 14.8 | 20.0 | 3 |
| +600 ppm, 150 → 870 ms | 19.5 | 19.6 | 16.3 | 15.8 | 13.5 | 2 |

Without the tracker, the +600 ppm call falls to 0–4 dB ERLE once the delay passes about 570 ms
(minute 12 on). Each re-alignment costs a few seconds of weaker cancellation while the new AEC3
instance converges; the text guard covers that gap.

## CPU

| Measurement | RTF (one thread) |
|---|---|
| `aec_bench`: AEC3 alone, far-end bursts with echo, 60 s, best of 5 | 0.018 |
| `aec_bench`: kenes-aec default (AEC3 + delay tracker + detectors) | 0.019 |
| `aec_eval`: the whole `StreamCanceller` path over the 24 scenes (21 min) | 0.025 |

Intel Core Ultra 5 125H, measured while other jobs used part of the machine; numbers rose to
0.03–0.06 when it was fully loaded. About 2–3% of one core, on the session thread. The slowest single
10 ms frame took 5–6 ms (scheduler noise; the budget is 10 ms). The delay tracker adds about 5%.

Latency added to the mic path: the fixed 18 ms (taken out of the timestamps), plus up to 8 ms
because 32 ms capture chunks don't divide into 10 ms frames, plus however long the system chunk of
the same moment arrives after the mic chunk (usually within the same 32 ms period in live capture;
never more than 150 ms of mic audio). In replay both arrive together.

## End to end: `kenes-cli` replay

A 3-minute call exported with `--export` (echo −6 dB, 120 ms delay, clipping, RT60 0.35 s): four
turns of each kind, Kazakh far end, Russian user. Replayed at real-time speed through the full
session (capture replay → canceller → transcriber → speakers → guard), with echo cancellation off
(`--no-echo-cancel`) and on:

```bash
KENES_DATA_DIR=/tmp/k kenes-cli --replay-mic mic_with_echo.wav --replay-system far.wav [--no-echo-cancel]
```

Final lines (`[mm:ss] speaker: text`) in arrival order.

Echo cancellation **off** (the far end is `Участник N`, the user `Я`):

```text
[00:02] Я: йога көмегімен
[00:02] Участник 1: кундалиний йога көмегімен кундалиний энергиясы ағарту энергиясы йога позалары тыныс алу жаттығулары мантра және визуализация арқылы оянады
[00:04] Я: далини энергиясы ағарту энергиясы йога позалары тыныс алу жаттығулары мантра және визуализация арқылы оянады
[00:15] Я: территория гонконга была названа в честь острова гонконг место которое многие туристы считают главным объектом своего внимания
[00:28] Участник 1: жанкарло фисичелло көлігін басқаруды жоғалтып жарысты старттан кейін көп ұзамай аяқтады
[00:26] Я: не подвергайте ткань воздействию слишком высокой температуры что может привести к усадке или в очень редких случаях к ожогу
[00:39] Участник 1: валидегі күн тәртібіндегі басқа тақырыптарға әлемнің қалған ормандарын сақтау және дамушы елдерге азырақ ластайтын жолмен өсуіне көмектесу үшін технологияларды бөлісу кіреді
[00:39] Я: валидегі күн тәртібіндегі басқа тақырыптарға әлемнің қалған ормандарын сақтау және дамушы елдерге азырақ ластайтын жолмен өсуіне көмектесу үшін технологияларды бөлісу кіреді
[00:54] Я: поскольку световое загрязнение в период их расцвета не было такой проблемой как сегодня они обычно расположены в городах или в кампусах до которых легче добраться чем до тех которые построены в наше время
[01:09] Участник 1: философия ғұламасы аристотель әрбір заттың төрт элементтің біреуінен немесе бірнешесінен жасалған деген теория құрған
[01:09] Я: во время сравнения при помощи инфракрасной фурье спектроскопии выяснилось что состав этих кристаллов соответствует кристаллам найденным в моче пораженных домашних животных
[01:23] Участник 1: ми патологиялары мен іс қимыл арасындағы байланыс ғалымдарды олардың зерттеулерінде қолдайды
[01:23] Я: би патологиялары мен іс қимыл арасындағы байланыс ғалымдарды олардың зерттеулерінде қолдайды
[01:33] Я: во время войны за независимость сша тринадцать штатов в соответствии с статьями конфедерации впервые сформировали слабое центральное правительство единственным органом которого был конгресс
[01:49] Участник 1: гонконг аралы гонконгтың аумағына өз атауын береді және көптеген туристер басты назар аударатын орын болып табылады
[01:48] Я: с другой стороны погодные условия со снегом и дом являются нормальными во многих странах и дорожное движение продолжается круглый год почти без перерывов
[02:01] Участник 1: құрылыс ағаштары бұл оқыту әдісі емес жаңа компьютерлік бағдарламаны пайдалану немесе жаңа жобаны бастау секілді жаңа оқу тәжірибесінен өтіп жатқан жеке тұлғаларға қолдау көрсететін көмек түрі
[02:01] Я: құрылыс ағаштары бұл оқыту әдісі емес жаңа компьютерлік бағдарламаны пайдалану немесе жаңа жобаны бастау секілді жаңа оқу тәжірибесінен өтіп жатқан жеке тұлғаларға қолдау көрсететін көмек түрі
[02:17] Я: дети живущие на улицах возможно пережили тяжелые формы насилия или травмирования до того как убежать или быть
[02:26] Я: прошенными
[02:29] Я: помимо соревнований в среду карпендо принимал участие в двух индивидуальных соревнованиях начемпо
[02:35] Я: чемионата барлығы
[02:31] Участник 2: олар қауіпсіз жүзуге арналған құмды жағажайлардың барлығына дерлік және олардың көпшілігінде пахутукава ағаштарымен
[02:42] Участник 2: қамтамасыз етілген
[02:37] Я: және олардың көпшілігінде ахуту каава ағаштарымен қамтамасыз етілген
```

Echo cancellation **on**:

```text
[00:02] Участник 1: кундалиний йога көмегімен кундалиний энергиясы ағарту энергиясы йога позалары тыныс алу жаттығулары мантра және визуализация арқылы оянады
[00:15] Я: территория гонконга была названа в честь острова гонконг место которое многие туристы считают главным объектом своего внимания
[00:28] Участник 1: жанкарло фисичелло көлігін басқаруды жоғалтып жарысты старттан кейін көп ұзамай аяқтады
[00:26] Я: не подвергайте ткань воздействию слишком высокой температуры что может привести к усадке или в очень редких случаях к ожогу
[00:39] Участник 1: валидегі күн тәртібіндегі басқа тақырыптарға әлемнің қалған ормандарын сақтау және дамушы елдерге азырақ ластайтын жолмен өсуіне көмектесу үшін технологияларды бөлісу кіреді
[00:54] Я: поскольку световое загрязнение в период их расцвета не было такой проблемой как сегодня они обычно расположены в городах или в кампусах до которых легче добраться чем до тех которые построены в наше время
[01:09] Я: во время сравнения при помощи инпрокрасной
[01:09] Участник 1: философия ғұламасы аристотель әрбір заттың төрт элементтің біреуінен немесе бірнешесінен жасалған деген теория құрған
[01:12] Я: спектроскопии выяснилось что состав этих кристаллов соответствует кристаллам найденным в моче пораженных домашних животных
[01:23] Участник 1: ми патологиялары мен іс қимыл арасындағы байланыс ғалымдарды олардың зерттеулерінде қолдайды
[01:33] Я: во время войны за независимость сша
[01:36] Я: тринадцать штатов в соответствии с статьями конфедерации впервые сформировали слабое
[01:42] Я: центральное правительство единственным органом которого был конгресс
[01:49] Участник 1: гонконг аралы гонконгтың аумағына өз атауын береді және көптеген туристер басты назар аударатын орын болып табылады
[01:48] Я: с другой стороны погодные условия со снеными едом являются нормальными во многих странах а дорожное движение продолжается круглый год почти без перерывов
[02:01] Участник 1: құрылыс ағаштары бұл оқыту әдісі емес жаңа компьютерлік бағдарламаны пайдалану немесе жаңа жобаны бастау секілді жаңа оқу тәжірибесінен өтіп жатқан жеке тұлғаларға қолдау көрсететін көмек түрі
[02:17] Я: дети живущие на улицах возможно пережили тяжелые формы насилия или травмирования до того как убежать или быть прошенными
[02:29] Я: помимо соревнований в среду карпендо принимал участие в двух индивидуальных соревнованиях на чемпионатах
[02:31] Участник 2: олар қауіпсіз жүзуге арналған құмды жағажайлардың барлығына дерлік және олардың көпшілігінде пахутукава ағаштарымен
[02:42] Участник 2: қамтамасыз етілген
```

- Off: the far end's four solo turns came back as five `Я` lines (three complete repeats and one
  sentence in two pieces). In the last double-talk turn the echo added one more `Я` line and garbled
  the user's (`… начемпо` / `чемионата барлығы`).
- On: no echo line. `AecStats`: tracked delay 130 ms (true 120 ms; the tracker works in 10 ms
  frames), AEC3's own estimate 128 ms (120 ms plus the 10 ms lead, in 4 ms blocks), no
  re-alignment. The text guard held 11 mic finals for 0.5 s on average and had nothing left to drop.
- Costs on: in double talk AEC3 dropped «фурье» and changed «со снегом и льдом» to «со снеными
  едом» (off: «со снегом и дом»), and two user turns came out as several finals (all words right).
  The 01:33 turn is user-only with the far end silent, so the audio there is bit-identical; the
  split comes from the recognizer's VAD state after a quiet (echo-free) mic instead of a noisy one.

## Remaining limits

- **Synthetic only.** Real laptop speakers are more nonlinear than a `tanh`, rooms have noise
  sources and movement, and USB/Bluetooth devices add latency and drift. The next step is a real
  call recorded with `kenes-rec` (speakers on, then headphones) and replayed through `kenes-cli`.
- **Double talk.** AEC3 attenuates the user's voice while the far end talks over them: about +0.7
  WER points in double-talk turns here, and sometimes a split final. The raw mic transcribes double
  talk surprisingly well because the echo is quieter than the user. `sonora` does not expose AEC3's
  suppressor tuning; the `aec3` crate does, which would allow an ASR-oriented (less aggressive)
  near-end tuning.
- **Headphone detection needs the far end loud and the mic quiet.** It decides after about 3 s of
  far-end speech during which the user is silent, and learns the mic's noise floor only while the
  far end has been silent for over 1 s (the start of the session counts). Until then AEC3 runs, which only matters if the user talks over the
  far end in that time.
- **Delays beyond 400 ms** are handled by the tracker only after it has seen a few seconds of far-end
  speech (ERLE 7–14 dB in the first far turn instead of 16–30), and each re-alignment resets AEC3.
- **Echo that leads its reference** on the timeline by more than 10 ms (a monitor stream that is
  delivered later than the mic) cannot be cancelled; the capture layer anchors both streams to
  arrival time, which makes that unlikely but it is unmeasured on real devices.
- **The text guard needs the call's own transcript.** It cannot drop a mic final that mixes the
  user's words with echo, and it holds mic finals up to 2.5 s while the call has untranscribed
  sound (music, noise).
