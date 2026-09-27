# kenes ASR benchmark: results

Measured 2026-09-27 on the dev machine: Intel Core Ultra 5 125H (Meteor Lake, 4P+8E+2LPE cores,
18 threads), 30 GB RAM, Ubuntu 26.04, CPU only. Every model ran with **4 threads** (what the app will
use). Raw per-utterance outputs are in `results/<model>.json`; the generated tables are in
`results/summary.md` (accuracy) and `results/timing/summary.md` (speed).

## TL;DR

- **Default: GigaAM-Multilingual large CTC (600M), int8 ONNX.** It is the most accurate candidate
  that can actually ship: Kazakh FLEURS 4.5 %, Kazakh Common Voice 8.8 %, ru/kk code-switch 4.4 %,
  Russian 2.2 %. The PyTorch original is within ±0.5 of each of these. It keeps both languages
  correctly inside one utterance. It runs at
  RTF ≈ 0.05 (about 20× faster than real time) with 4 threads, uses about 1 GB RSS,
  and is a 564 MB download.
- **Fallback / low-end mode: GigaAM-Multilingual CTC (220M), int8 ONNX.** It is about 2× faster
  (RTF ≈ 0.026), needs half the RAM (~0.5 GB) and is a 214 MB download. It costs 1.3–2 WER
  points (ru 4.2 %, kk 5.8 %, code-switch 5.1 %). Switch to it when the device has < 8 GB RAM or when
  the large model's measured RTF on the device is above ~0.5.
- **Don't run these models through sherpa-onnx's feature extractor.** sherpa-onnx 1.13.8 computes
  GigaAM features with a 25 ms kaldi fbank, but the model was trained on a 20 ms torchaudio log-mel.
  The same ONNX file gets 30–50 % more errors on code-switched audio through sherpa, and the
  failures are occasionally catastrophic (example below). This is not an int8 problem: the fp32 export
  shows the same degradation in sherpa. With the GigaAM-exact log-mel front-end (40 lines of numpy,
  easy to port to Rust) plus onnxruntime, the int8 file is within ~0.5 WER of the official PyTorch
  fp32 model on every set, and about 1.8× faster than sherpa-onnx.
- Whisper candidates lose. Stock large-v3-turbo is decent on Russian (1.5 % vs 0.5 % for GigaAM
  large on the same 20 utterances) and is the only candidate with punctuation. But it
  misidentifies short Kazakh clips as Turkish or Icelandic (cv_kk WER 53 %). The kk/ru fine-tune
  fixes Kazakh (8.2 %) but hurts Russian (7.1 %) and code-switching (25 %: it often drops the
  second language entirely). Both need 10–17 s of compute per call on 4 threads (RTF ≈ 0.9–1.6 on
  10 s chunks, 3–5 on short ones), which is too slow for live audio on this laptop.

## Evaluation data (`prepare_data.py`)

| set | source | utts | words | audio | notes |
|---|---|---|---|---|---|
| fleurs_ru | FLEURS ru_ru test | 60 | 1182 | 11.1 min | read Wikipedia sentences, 6–19 s |
| fleurs_kk | FLEURS kk_kz test | 60 | 931 | 13.3 min | read Wikipedia sentences, 8–20 s |
| cv_kk | Common Voice 11 kk test (ungated HF mirror `Shirali/common_voice_11_0_kk`) | 60 | 365 | 5.0 min | crowdsourced, varied mics, short proverbs 3–9 s |
| codeswitch | synthetic: 1 FLEURS kk + 1 FLEURS ru utterance + 300 ms silence, alternating kk→ru / ru→kk, loudness-matched | 30 | 791 | 9.4 min | 13–23 s, disjoint from the sentences above |

- One recording per FLEURS sentence id, fixed seed.
- Utterances whose reference contains digits are excluded. GigaAM is char-level and has no digits;
  the GigaAM paper uses the same protocol.
- WER/CER use jiwer on normalised text: lowercase, ё→е, punctuation→space, collapsed spaces.
- The sets are small. 1 word is ≈0.1 WER point on fleurs_ru, ≈0.3 on cv_kk. Treat differences
  below ~1 point as noise; cv_kk is the noisiest.
- Common Voice via Mozilla now needs a login, and the KSC2 corpus on HF is 80 GB, so the ungated
  CV11 mirror was the cheap "in-the-wild-ish" option.

## Candidates

| key | what | runtime |
|---|---|---|
| gigaam-ml-ctc-pt / gigaam-ml-large-ctc-pt | GigaAM-Multilingual `multilingual_ctc` 220M / `multilingual_large_ctc` 600M, official checkpoints (GigaAM repo @7447938) | PyTorch 2.10 CPU fp32, `gigaam` package |
| gigaam-ml-ctc-onnx-int8 / gigaam-ml-large-ctc-onnx-int8 | community sherpa-onnx int8 builds (fgeeer77/bayaya-models) | sherpa-onnx 1.13.8 `OfflineRecognizer.from_nemo_ctc` |
| gigaam-ml-ctc-onnx-int8-ortfeat / gigaam-ml-large-ctc-onnx-int8-ortfeat | **the same int8 ONNX files** | onnxruntime 1.23 + GigaAM-exact log-mel in numpy (`run_bench.GigaAMLogMel`) + greedy CTC |
| gigaam-ml-ctc-onnx-fp32-self | own fp32 ONNX export (`export_onnx.py`) | sherpa-onnx (isolates quantization vs. front-end) |
| gigaam-v3-ru-ctc-onnx-int8 | GigaAM-v3 CTC, Russian only (csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16) | sherpa-onnx; fleurs_ru only |
| whisper-kaz-rus-ct2-int8 | abilmansplus/whisper-turbo-kaz-rus-v1 (LoRA on whisper-turbo-ksc2, merged by `convert_whisper.py`) | CTranslate2 int8 via faster-whisper, greedy, language auto-detect; first 20 utts/set |
| whisper-large-v3-turbo-ct2-int8 | openai whisper-large-v3-turbo (mobiuslabsgmbh CT2 build) | faster-whisper int8, greedy, language auto-detect; first 20 utts/set |
| vosk-small-kz-0.42 | Vosk small Kazakh (Kaldi) | vosk 0.3.45; kk sets only |

Not skipped: every requested candidate ran. The whisper models were limited to 20 utterances per set
because of RTF > 1 (as allowed). The "matched subset" table below compares every model on exactly
those 20.

## Accuracy (WER / CER %, full sets)

| model | fleurs_ru | fleurs_kk | cv_kk | codeswitch | peak RSS MB | size on disk MB |
|---|---|---|---|---|---|---|
| gigaam-ml-ctc-pt (PyTorch fp32) | 3.7 / 0.7 | 5.2 / 1.6 | 10.7 / 2.3 | 5.3 / 1.0 | 2067 | 842 (.ckpt) |
| gigaam-ml-large-ctc-pt (PyTorch fp32) | 2.4 / 0.4 | **4.4** / 1.4 | 9.3 / 2.1 | **4.3** / 0.7 | 4882 | 2233 (.ckpt) |
| gigaam-ml-ctc-onnx-int8 (sherpa-onnx) | 4.0 / 0.7 | 6.1 / 1.6 | 8.8 / 2.0 | 7.5 / 2.3 | 503 | 214 |
| gigaam-ml-large-ctc-onnx-int8 (sherpa-onnx) | 2.3 / 0.4 | 4.7 / 1.5 | **7.1** / 1.5 | 5.7 / 0.8 | 1031 | 564 |
| **gigaam-ml-ctc-onnx-int8-ortfeat** (ORT + exact features) | 4.2 / 0.7 | 5.8 / 1.6 | 10.1 / 2.4 | 5.1 / 1.0 | 544 | 214 |
| **gigaam-ml-large-ctc-onnx-int8-ortfeat** (ORT + exact features) | **2.2** / 0.4 | 4.5 / 1.4 | 8.8 / 2.0 | 4.4 / 0.7 | 1051 | 564 |
| gigaam-ml-ctc-onnx-fp32-self (sherpa-onnx, fp32) | 4.0 / 0.7 | 5.7 / 1.6 | 8.8 / 2.0 | 7.0 / 2.0 | 1478 | 844 |
| gigaam-v3-ru-ctc-onnx-int8 (sherpa-onnx) | 2.7 / 0.5 | – | – | – | 419 | 214 |
| whisper-kaz-rus-ct2-int8 (n=20/set) | 7.1 / 1.3 | 8.2 / 3.2 | 14.7 / 3.0 | 25.2 / 15.7 | 1628 | 786 |
| whisper-large-v3-turbo-ct2-int8 (n=20/set) | 1.5 / 0.6 | 24.2 / 6.8 | 53.5 / 25.9 | 17.3 / 4.5 | 2113 | 1547 (fp16, int8 at load) |
| vosk-small-kz-0.42 | – | 20.8 / 7.0 | 26.3 / 8.6 | – | 301 | 102 |

Peak RSS is the whole Python worker process: interpreter, runtime libraries and model. The PyTorch
peak includes loading the checkpoint.

### Matched subset: first 20 utterances of every set (the ones the whisper models ran)

| model | fleurs_ru | fleurs_kk | cv_kk | codeswitch |
|---|---|---|---|---|
| gigaam-ml-ctc-pt | 1.7 / 0.3 | 7.5 / 2.9 | 11.2 / 2.2 | 5.2 / 1.0 |
| gigaam-ml-large-ctc-pt | 0.5 / 0.1 | 5.7 / 2.5 | 10.3 / 2.5 | 3.9 / 0.8 |
| gigaam-ml-ctc-onnx-int8 (sherpa) | 2.2 / 0.3 | 9.1 / 3.0 | 11.2 / 2.0 | 8.3 / 3.0 |
| gigaam-ml-large-ctc-onnx-int8 (sherpa) | 0.5 / 0.1 | 6.0 / 2.5 | 8.6 / 1.5 | 5.2 / 0.8 |
| gigaam-ml-ctc-onnx-int8-ortfeat | 1.9 / 0.3 | 8.8 / 3.0 | 11.2 / 2.2 | 5.0 / 1.0 |
| gigaam-ml-large-ctc-onnx-int8-ortfeat | 0.5 / 0.1 | 5.7 / 2.5 | 10.3 / 2.3 | 4.5 / 0.8 |
| gigaam-v3-ru-ctc-onnx-int8 | 0.7 / 0.1 | – | – | – |
| whisper-kaz-rus-ct2-int8 | 7.1 / 1.3 | 8.2 / 3.2 | 14.7 / 3.0 | 25.2 / 15.7 |
| whisper-large-v3-turbo-ct2-int8 | 1.5 / 0.6 | 24.2 / 6.8 | 53.5 / 25.9 | 17.3 / 4.5 |
| vosk-small-kz-0.42 | – | 24.8 / 10.1 | 27.6 / 8.8 | – |

## Speed (RTF @ 4 threads) and memory

Separate timing pass (`results/timing/`): 10 FLEURS utterances per language (6–19 s each). Each
utterance was timed 3 times and the fastest run kept; whisper got 5 utterances timed 2 times. The
pass waited until the 1-minute load average was below 4 before each model. RTF = processing time
/ audio duration. CPU-s/s = CPU seconds spent per second of audio, summed over threads.

| model | runtime | RTF ru | RTF kk | ≈ time for a 10 s chunk | CPU-s/s | load time | RSS after load | peak RSS | on disk |
|---|---|---|---|---|---|---|---|---|---|
| **GigaAM-ML CTC 220M int8** | onnxruntime + exact log-mel | **0.027** | **0.024** | 0.26 s | 0.12–0.13 | 1.0 s | 381 MB | 459 MB | 214 MB |
| GigaAM-ML CTC 220M int8 | sherpa-onnx | 0.055 | 0.045 | 0.5 s | 0.18–0.22 | 0.8 s | 380 MB | 436 MB | 214 MB |
| GigaAM-ML CTC 220M fp32 | PyTorch | 0.060 | 0.046 | 0.5 s | 0.18–0.24 | 3.9 s | 1925 MB | 2004 MB | 842 MB |
| **GigaAM-ML large CTC 600M int8** | onnxruntime + exact log-mel | **0.051** | **0.056** | 0.55 s | 0.23–0.25 | 1.5 s | 884 MB | 958 MB | 564 MB |
| GigaAM-ML large CTC 600M int8 | sherpa-onnx | 0.095 | 0.091 | 0.9 s | 0.36–0.38 | 1.7 s | 887 MB | 950 MB | 564 MB |
| GigaAM-ML large CTC 600M fp32 | PyTorch | 0.093 | 0.099 | 1.0 s | 0.37–0.40 | 7.9 s | 4708 MB | 4806 MB | 2233 MB |
| GigaAM-v3 ru CTC int8 | sherpa-onnx | 0.033 | – | 0.33 s | 0.13 | 0.9 s | 384 MB | 417 MB | 214 MB |
| vosk-small-kz-0.42 | vosk (1 thread) | – | 0.071 | 0.7 s | 0.07 | 0.5 s | 228 MB | 284 MB | 102 MB |
| whisper-large-v3-turbo int8 | faster-whisper | 0.90 | (2.3)* | ~10 s per call, any length | 3.5 | 2.8 s | 1720 MB | 2027 MB | 1547 MB |
| whisper-turbo-kaz-rus int8 | faster-whisper | 1.58 | 1.45 | ~17 s per call | 5.6–6.0 | 1.2 s | 960 MB | 1630 MB | 786 MB |

\* The load average jumped to 20 while stock whisper ran the kk utterances, so that number is not
reliable. Its ru run and the other rows ran at load 3.4–6.

Takeaways:
- Both multilingual GigaAM models are far inside the real-time budget on 4 threads. The large one
  is about 20× faster than real time and the 220M about 40×. A 10 s VAD chunk transcribes in about
  0.5 s (large) or 0.25 s (220M).
- Plain onnxruntime is about 1.8× faster than sherpa-onnx on the same ONNX file (the cause was not
  investigated; session options or the bundled ORT build), in addition to being more accurate.
  int8 through onnxruntime is also about 2× faster than PyTorch fp32 and uses 4–5× less RAM.
- Whisper pays for a full 30 s window on every call: about 10 s of compute on 4 threads, even
  for a 3 s chunk. That is why its RTF on short cv_kk clips reached 3–5 in the accuracy runs.
  Whisper cannot keep up with live meeting audio on this CPU.

## Code-switching

The multilingual GigaAM models handle intra-file kk↔ru switches well. They have a single character
vocabulary (Latin + Russian Cyrillic + the 9 Kazakh letters), so there is no language decision and
each word comes out in the right script. Errors are ordinary acoustic ones: "стрикоза" for
"стрекоза", or a Latin `i` inside a Kazakh word.

Code-switch WER split by the reference half (errors attributed through the jiwer alignment):

| model | kk half | ru half | 1st half | 2nd half |
|---|---|---|---|---|
| gigaam-ml-ctc-pt | 4.9 | 5.7 | 3.8 | 6.9 |
| gigaam-ml-large-ctc-pt | 5.5 | 3.3 | 3.8 | 4.9 |
| gigaam-ml-ctc-onnx-int8 (sherpa) | 5.7 | 9.0 | 4.3 | **10.7** |
| gigaam-ml-large-ctc-onnx-int8 (sherpa) | 6.8 | 4.7 | 5.3 | 6.1 |
| gigaam-ml-ctc-onnx-int8-ortfeat | 4.9 | 5.2 | 3.8 | 6.4 |
| gigaam-ml-large-ctc-onnx-int8-ortfeat | 5.5 | 3.5 | 3.8 | 5.1 |
| gigaam-ml-ctc-onnx-fp32-self (sherpa) | 5.5 | 8.3 | 4.0 | **10.0** |
| whisper-kaz-rus-ct2-int8 | 19.1 | 30.2 | 9.8 | **39.1** |
| whisper-large-v3-turbo-ct2-int8 | 17.0 | 17.5 | 16.8 | 17.7 |

Whisper decides one language per 30 s window. The kk/ru fine-tune reports `kk` with probability
0.1–0.3 and often stops decoding after the first language. Stock turbo transcribes both halves but
misspells a lot.

Examples. REF is the reference; hypotheses are shown raw.

```
cs04_kk-ru  REF   Әрекетке жақын болғыңыз келсе, музыкаға жақын кемпинг учаскесін алу үшін ерте тұруыңыз керек. Он добавил, что "их, однако, не следует просить брать на себя обязательства, которые выходят за рамки их стадии разработки, ответственности и возможностей".
  large ortfeat  әрекетке жақын болғыңыз келсе музыкаға жақын кемпинг учаскесін алу үшін ерте тұруыңыз керек он добавил что их однако не следует просить брать на себя обязательства которые выходят за рамки их стадий разработки ответственности и возможностей
  small ortfeat  ... ерте тұруыңыз керек он добавил что их однако не следует просить брать на себя обязательства которые выходят за рамки их стадии разработки ответственности и возможнотей
  small sherpa   ... ерте тұруыңыз керекон ононе летпроситбрть на себя обязательства которые выходят за рамки ихт        <- same int8 file, sherpa front-end
  whisper-kaz-rus ... ерте тұруыңыз керек онда барлық что их однако не следует просить брать на себе обязательства которые выходит на замки их стади разработки ответственности и возможности
  whisper-turbo  хәрекетке жақын ... ерте тұруңыз керек он добавил что их однако не следует просить брать на себя обязательства которые выходят за рамки их стадии разработки ответственности и возможностей

cs05_ru-kk  REF   Все во вселенной состоит из материи. Вся материя состоит из мельчайших частиц, называемых атомами. Билеуші партия Оңтүстік батыс Африка халық ұйымы (SWAPO) парламенттік сайлауларда көптеген орынға ие болды.
  large ortfeat  все во вселенной состоит из материи вся материя состоит из мельчайших частиц называемых атомами билеуші партия оңтүстік батыс африка халық ұйымы сваппо парламенттік сайлауларда көптеген орынға ие болды
  whisper-kaz-rus все во вселенной состоит из материей вся материя состоит из мельчайших чистительц называемых атомами      <- Kazakh half dropped
  whisper-turbo  все во вселенной состоит из материи ... атомами белеуші партия оңтүстік ватыс африка халық ұйымы сваппо парламенттік сайлауларды көптеген орынға ие болды

cs06_kk-ru  REF   Жолбарыстың ақыруы арыстанның бар күшімен ақыруына ұқсамайды, ол ақырып қатты айтқан сөздерге көбірек ұқсайды. Представьте, что лыжный маршрут — это пешеходный маршрут.
  large ortfeat  жолбарыстың ақыруы арыстанның бар күшімен ақыруына ұқсамайды ол ақырып қатты айтқан сөздерге көбірек ұқсайды представьте что лыжный маршрут это пешеходный маршрут
  whisper-kaz-rus жолбарыстың ақыруы арыстанның бар күші мен ақыруына ұқсамайды ол ақырып қатты айтқан сөздер де көбірек ұқсайды      <- Russian half dropped
  whisper-turbo  жолбарыстың ақыруы ... сөздерде көмірек ұқсайды      <- Russian half dropped

cs12_kk-ru  REF   Пирамида дыбыс және жарық шоуы балаларға арналған аумақтағы ең қызық нәрселердің бірі болып табылады. Стрекозы и поденки являются единственными насекомыми, которые не могут складывать крылья назад.
  large ortfeat  пирамида дыбыс және жарық шоуы балаларға арналған аумақтағы ең қызық нәрселердің бірі болып табылады стрикозы и паденки являются единственными насекомыми которые не могут складывать крылья назад
  whisper-turbo  ... стрекозы и поденки являются единственными насекомыми которые не могут складывать крылья назад

cs07_ru-kk  REF   Важно различать глаголы и дополнения. Прайдтар бірден үшке дейін туыс ересек еркек арыстаннан, сондай-ақ отыз аналық пен абданнан тұрады.
  small ortfeat  важно различать глаголы и дополнения прайдтар бірден үшке дейін туыс ересек еркек арыстаннан сондай ақ отыз аналық пен абданнан тұрады
  whisper-turbo  важна различать глаголы и дополнения прайктор бірден-үшке дейін туыс ересек еркек арыстаннан сондай-ақ отыз аналық пен абдамдан тұрады
```

Stock whisper on short Kazakh clips (cv_kk), language auto-detected:

```
tr:0.31  Мәуесі бардың әуесі бар.       ->  Ve o iyisi vardır, o iyisi vardır.
is:0.96  Он үште отау иесі.             ->  Og hann stjóð og þá í þessu.
```

## Punctuation, casing, numbers

| model family | punctuation | casing | digits |
|---|---|---|---|
| GigaAM CTC (all variants) | none | lowercase only | never (numbers come out as words) |
| whisper-kaz-rus | none | lowercase | none seen |
| stock whisper-large-v3-turbo | yes (100 % of ru hyps, ~40 % of kk) | yes | yes (5 % of ru hyps) |
| vosk kz | none | lowercase | none |

GigaAM output is a lowercase stream without punctuation, with numbers spelled out (and inflected
for Russian). For kenes that is fine: the transcript goes to Claude, which copes with unpunctuated
text. The prompt should say so ("ASR transcript, lowercase, no punctuation, may mix Kazakh and
Russian"). If a readable transcript is needed in the UI, punctuation and casing can be restored
later, for example by Claude at summary time.

## sherpa-onnx vs. the model's own front-end (important for the Rust app)

Facts:

1. The community int8 ONNX for the 220M model is **bit-identical** to what `export_onnx.py` produces
   from the official checkpoint: `gigaam.to_onnx()`, sherpa metadata, then
   `onnxruntime.quantization.quantize_dynamic(weight_type=QUInt8)`. The sha256 is the same. The 600M
   build has byte-identical weights (all 1431 initializers), graph and metadata compared with the
   local export; the file differs by 534 bytes of protobuf framing. The artifacts are trustworthy
   and reproducible.
2. With identical ONNX weights, only the front-end changes the result. On the code-switch set, the
   220M int8 model scores:

   | front-end | WER | CER | mean \|Δ log-mel\| vs GigaAM |
   |---|---|---|---|
   | GigaAM log-mel (torchaudio-exact numpy port, 20 ms hann, n_fft 320, HTK mel, log(clamp 1e-9)) | **5.06** | 0.96 | 0 |
   | kaldi-native-fbank as sherpa configures it (25 ms window, n_fft 400) | 6.19 | 1.36 | 0.90 |
   | kaldi-native-fbank with 20 ms window | 5.94 | 1.62 | 0.82 |
   | sherpa-onnx end-to-end | 7.46 | 2.27 | – |

   Reproduce with `uv run diagnose_features.py --set codeswitch`.
3. The sherpa degradation is the same with the fp32 export (7.0 % on codeswitch vs. 5.1–5.3 % for
   ORT-int8 and PyTorch), so quantization is not the cause. The int8 file with exact features is
   close to PyTorch fp32. For the large model: 4.4 vs 4.3 % on codeswitch, 2.2 vs 2.4 % ru, 4.5 vs
   4.4 % kk. For the 220M: 5.1 vs 5.3 % on codeswitch; quantization costs about +0.5 on FLEURS
   ru/kk.
4. Root cause: sherpa-onnx 1.13.8 (`offline-recognizer-ctc-impl.h`) hard-codes the GigaAM v1/v2
   front-end ("GigaAM uses n_fft 400", default 25 ms frames, kaldi mel banks and log floor).
   GigaAM-v3 and Multilingual use `win_length = n_fft = 320`. `frame_length_ms` is not exposed in
   the Python or C APIs, so the app cannot configure around it.

**Recommendation for the Rust side:** run the ONNX with the `ort` crate directly instead of
sherpa-onnx's recognizer. The whole pipeline is small:

- **Front-end.** Port `GigaAMLogMel` from `run_bench.py`: frames of 320 samples every 160 with no
  centre padding, periodic hann window, `|rfft(320)|²`, a 161×64 HTK mel filterbank (0–8000 Hz, no
  norm), then `ln(clamp(x, 1e-9, 1e9))`. Input is `f32` 16 kHz mono in [-1, 1]. The numpy port matches
  torchaudio to 2e-4.
- **Model I/O.** Inputs are `features: f32 [1, 64, T]` and `feature_lengths: i64 [1]`. Outputs are
  `log_probs: f32 [1, T/4, 71]` and `encoded_lengths: i64 [1]`.
- **Decoding.** Greedy CTC: argmax per frame, collapse repeats, drop blank. Blank is id 70 (the last
  line of tokens.txt). Id 0 is the space character, so tokens.txt line 0 is `"  0"`: parse it with
  `rsplit(' ', 1)`.
- **Segmentation.** Feed VAD segments of ≤ 20–25 s. The bench tested up to 23 s per call; the
  official `transcribe()` refuses > 25 s. `run_bench.py` uses silero VAD for longer custom files.
- **Silence.** Digital-zero audio is handled correctly by the exact front-end, since GigaAM itself
  floors at log(1e-9). No dither is needed.

If the team prefers sherpa-onnx anyway, it works: 1–2.5 WER points worse and occasional garbled
spans after a pause or a language switch. That is acceptable for hints but not ideal. Fixing it
upstream means making sherpa use 20 ms / n_fft 320 frames and torchaudio-style mel banks for
`is_giga_am` models that declare it.

## Recommendation

**Default model: GigaAM-Multilingual large CTC (600M), int8 ONNX, run with onnxruntime and the
GigaAM-exact log-mel front-end.**
- Best, or within noise of the best, on every set: Kazakh read speech 4.5 %, Kazakh crowdsourced 8.8 %, Russian
  2.2 %, code-switch 4.4 %.
- One model for kk, ru and mixed speech, with no language selection.
- RTF ≈ 0.05 on 4 threads, ~1 GB RSS.

**Fallback: GigaAM-Multilingual CTC (220M), int8 ONNX, same pipeline.**
- RTF ≈ 0.026, ~0.5 GB RSS, 214 MB download.
- About 2 WER points worse on Russian, 1.3 on Kazakh, 0.7 on code-switch.
- Use it on machines with < 8 GB RAM or slow CPUs (auto-select with a first-run RTF probe), or as
  a "battery saver" option.

Not recommended:
- **GigaAM-v3 ru** (2.7 % on fleurs_ru, no better than the multilingual large model on this set)
  only makes sense for a Russian-only mode, and it cannot do Kazakh.
- **Whisper variants** are too slow on CPU (RTF > 1 at 4 threads; 30 s padding makes short VAD
  chunks even worse: RTF 3–5 on cv_kk) and unreliable on Kazakh or code-switching.
- **Vosk kz** is fast and tiny but 3–5× the WER.

### ONNX artifacts for the app

Source: <https://github.com/fgeeer77/bayaya-models/releases>. Community releases converted from
GigaAM @7447938 (MIT). They were verified here to match the local export. Consider mirroring them
to a kenes-controlled location, since this is a personal repo.

| model | file | URL | bytes | sha256 |
|---|---|---|---|---|
| default (600M) | model.int8.onnx | https://github.com/fgeeer77/bayaya-models/releases/download/gigaam-multilingual-large-ctc/model.int8.onnx | 591645636 | `7fdb9427c1c871407ecbde741fd7bb0479924981c89aa9f4241587bbcb085ae3` |
| default (600M) | tokens.txt | https://github.com/fgeeer77/bayaya-models/releases/download/gigaam-multilingual-large-ctc/tokens.txt | 391 | `9b5df7987cb4ca52c1a468649ce897fab1cd182067416e29fef49dfaa7a856c2` |
| fallback (220M) | model.int8.onnx | https://github.com/fgeeer77/bayaya-models/releases/download/gigaam-multilingual-ctc/model.int8.onnx | 224762518 | `f66bff0186d649a2300da895e9f81d4ef8764519db2dc46429a44b883e90d105` |
| fallback (220M) | tokens.txt | https://github.com/fgeeer77/bayaya-models/releases/download/gigaam-multilingual-ctc/tokens.txt | 391 | `9b5df7987cb4ca52c1a468649ce897fab1cd182067416e29fef49dfaa7a856c2` |
| (ru-only, optional) | model.int8.onnx | https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16/resolve/main/model.int8.onnx | 224721476 | `f86ebfa0429ced91be6054fc344827e9c6c2572f3c318416cd974b06f66437ec` |
| (ru-only, optional) | tokens.txt | https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16/resolve/main/tokens.txt | 196 | `17cc514451bcceac9c280068c71502f8448f99e9fb1456b8d0761651fd0392f2` |

Both multilingual models share the same tokens.txt (71 symbols: space, `'`, a–z, Russian а–я + ё,
Kazakh і ғ қ ң ү ұ һ ә ө, `<blk>`). Every sha256 above was computed on the downloaded files and
matches the release's SHA256SUMS. `fetch_models.sh` downloads and verifies them.

## Caveats

- **Timings.** Other agents were compiling Rust crates and running a diarization eval on the same
  machine during the whole session (load average 5–20). The accuracy runs' wall-clock RTFs in
  `results/summary.md` are inflated and noisy. The speed table above comes from a separate pass that
  waited for load < 4 where possible and kept the fastest of 3 runs per utterance. On an idle
  machine, expect those numbers or slightly better.
- **Read speech.** FLEURS and CV are read speech. Meeting audio (far-field, overlap, spontaneous
  speech, heavier code-switching inside sentences) will be harder for every model. Run your own
  recordings with `run_bench.py --custom DIR` before locking this in.
- **Synthetic code-switching.** The code-switch set switches at a sentence boundary with a pause.
  Real intra-sentential switching ("мен бүгін встречаға барамын") is not covered.
- **Short CV clips.** On cv_kk, sherpa's front-end did better than the exact front-end (7.1 vs 8.8
  for the large model). With 365 words that is 6 words, within noise, but worth rechecking on real
  recordings.
