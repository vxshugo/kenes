# kenes-speakers evaluation

How the speaker model, the change detector and the `ClusterConfig` defaults were chosen, and
how well they work. All numbers come from `examples/diarize_eval.rs` on synthetic ru/kk
meetings (see [Reproduce](#reproduce)).

## Summary

- **Model: 3D-Speaker CAM++ "zh-cn 16k common"** (`3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx`,
  28.3 MB, 192-dim, trained on ~200k Chinese speakers). It had the best accuracy for its CPU cost
  among 11 candidates, and was tied for the fastest. ERes2Net-base-200k scores about 1 point better after re-clustering but costs 3.5×
  the CPU. The WeSpeaker `_LM` exports and the 3D-Speaker VoxCeleb CAM++ don't work in this setup
  (EER 18–48%).
- **Cost:** 59 ms per 5 s segment on one P-core (Intel Core Ultra 5 125H), 49 ms on two, 138 ms on
  an E-core. That is RTF ≈ 0.012, or about 1.2% of one core per source while people talk. This is
  above the ~30 ms target. None of the candidates reaches 30 ms per 5 s on this CPU: the fastest,
  CAM++ and TitaNet-small, are both ~60 ms. `change_points` costs 0.8 s per 20 s utterance
  (4% of real time) on a P-core and 1.8 s (9%) on an E-core.
- **On pipeline segmentation** (the kenes-stt VAD finals, then `change_points`, 12 meetings × 10–15
  speakers), the share of speech correctly attributed is:

  | condition | split | online | after `recluster` | labels (recluster / true) |
  |---|---|---|---|---|
  | call (system audio) | tune | 97.7% | 98.6% | 13.2 / 12.8 |
  | call | test | 95.4% | 96.4% | 12.0 / 12.0 |
  | room (laptop mic, far field) | tune | 93.6% | 94.5% | 14.0 / 12.8 |
  | room | test | 92.1% | 94.1% | 14.0 / 12.0 |

  Without `change_points` the room condition drops to 73–79% online, because the VAD glues
  45–50% of the speech into multi-speaker utterances.
- **Recluster is never meaningfully worse than online.** Over all 24 meeting × condition runs,
  the worst single meeting loses 0.1 points (on raw, unsplit finals the worst loses 2.0).
- **"me" (voiceprint)** reaches 98–99% precision and 98–99% recall in the room condition.

## Data

Speakers come from **Common Voice 17** (the `fsicoli/common_voice_17_0` mirror on Hugging Face,
which keeps `client_id`). Kazakh speakers come from test/dev/other/train; Russian ones from test.
Neither FLEURS nor the bench's `cv_kk` parquet has speaker ids. After trimming silence, 65 kk
and 89 ru speakers have at least 6 clips and 25 s of speech. `eval/build_meetings.py` builds
12 meetings from them:

| meeting | split | speakers | minutes | segments | overlapping turn changes | room RT60 / SNR |
|---|---|---|---|---|---|---|
| m01 | tune | 12 ru | 8.8 | 87 | 7 | 0.40 s / 16.7 dB |
| m02 | tune | 13 kk | 7.7 | 74 | 6 | 0.36 / 16.5 |
| m03 | tune | 7 kk + 7 ru | 7.9 | 76 | 7 | 0.43 / 18.7 |
| m04 | tune | 6 kk + 7 ru | 6.5 | 70 | 11 | 0.43 / 18.6 |
| m05 | tune | 10 ru | 7.8 | 86 | 8 | 0.43 / 15.7 |
| m06 | tune | 15 kk | 6.5 | 68 | 6 | 0.53 / 19.8 |
| m07 | test | 5 kk + 6 ru | 7.0 | 65 | 6 | 0.59 / 15.7 |
| m08 | test | 5 kk + 5 ru | 7.5 | 67 | 7 | 0.32 / 15.7 |
| m09 | test | 13 ru | 7.7 | 73 | 6 | 0.35 / 15.4 |
| m10 | test | 14 kk | 6.7 | 78 | 6 | 0.52 / 15.5 |
| m11 | test | 6 kk + 7 ru | 7.9 | 69 | 10 | 0.32 / 17.0 |
| m12 | test | 5 kk + 6 ru | 6.2 | 66 | 8 | 0.51 / 15.9 |

- **Turns:** participation is Zipf-like: a few people talk a lot, and everyone speaks at least twice.
  A turn has 1–3 segments separated by 0.4–1.2 s pauses. Segment lengths are about 11%
  0.5–1 s (back-channels), 32% 1–2 s, 34% 2–7 s and 23% 7–20 s. Pauses between turns are 0.2–4 s, and
  15% of turn changes overlap by 0.1–0.6 s (interruptions).
- **"me":** in each meeting one speaker is "me". About 20 s of their clips are held out for an
  enrollment recording and never appear in the meeting.
- **Conditions:**
  - `clean`: the Common Voice audio as is, each speaker on their own device.
  - `call`: clean, then an Opus 24 kbit/s round trip.
  - `room`: each speaker sits at their own spot (0.5 m for "me", 1–3.5 m for the others) in
    one room. Each gets a synthetic RIR (RT60 0.3–0.6 s, direct-to-reverberant ratio falling
    with distance, critical distance ~1 m). Then a laptop-mic band-pass (150–6500 Hz) and pink
    noise at 15–20 dB SNR are applied.
- **Splits:** m01–m06 are for tuning; every number marked *test* is on m07–m12, which were never
  used to pick a value, with the exceptions noted below.

### Segmentation

There are two views of the same meetings:

1. **Pipeline (primary).** `kenes_stt::Transcriber::transcribe_buffer` (Silero VAD with 0.5 s min
   silence, GigaAM decoding, exactly the live segmentation) turns `call.wav` and `room.wav` into
   finals. Finals of at least 2.5 s are then cut by `change_points`, as the kenes-stt hook does.
   The truth is the manifest's speaker timeline, and scoring is per 10 ms frame.
2. **Oracle turns (secondary).** The manifest's segments are used as they are. This isolates the
   embeddings and clustering from segmentation. It was used for the first model screening.

The VAD glues speakers together. In raw finals, **19–23% of the call speech and 45–50% of the
room speech** sits in finals containing two or more speakers (≥ 0.3 s each). Room reverb fills
the short gaps between turns. That caps what any per-segment labeler can reach at 94–95% (call)
and 84–87% (room). This is what the coordinator saw live on m12 (36% mixed).

## Metrics

- **Accuracy (pipeline):** the share of speech frames (10 ms, inside VAD segments) whose segment
  label maps to a speaker who is actually talking in that frame. Labels are mapped to true
  speakers by the optimal one-to-one assignment (Hungarian). Unlabeled speech and extra labels
  count as wrong. Frames with overlapping speech are right if either speaker is mapped. The
  "purity bound" is the accuracy a perfect labeler would get with this segmentation.
- **SER (oracle):** the duration-weighted speaker error rate of segment labels after the same
  optimal mapping (`kenes_speakers::metrics::score`).
- **Pair EER:** a threshold-free check of the embedding alone. It is the equal error rate of
  "same speaker?" over all pairs of oracle segments within a meeting, with both segments ≥ 1 s,
  both ≥ 3 s, or one 1–2 s against one ≥ 3 s.
- **Labels:** the distinct labels a meeting ends up with, compared with the true speaker count.

## Model choice

The candidates are all from the sherpa-onnx `speaker-recongition-models` release, run through the
same `Embedder` (sherpa-onnx 1.13.8, CPU). The CPU columns are the median of 7 runs, pinned with
`taskset`: one P-core (1 thread) or two physical P-cores (2 threads). The pipeline columns are the
mean of call and room on identical pieces (the zh-cn model's split), with each model's
`ClusterConfig` tuned the same way on the tune split.

| model | MB | dim | ms / 5 s, 1 thr | 2 thr | pair EER ≥1 s call / room | oracle SER test, online / recl. | pipeline acc. test, online / recl. |
|---|---|---|---|---|---|---|---|
| **3D-Speaker CAM++ zh-cn common** | 28.3 | 192 | **59** | **49** | 3.6% / 7.8% | **3.7% / 2.4%** | **93.7% / 95.3%** |
| 3D-Speaker CAM++ zh+en advanced | 28.3 | 192 | 66 | 56 | 4.3% / 9.2% | 6.0% / 2.9% | 91.2% / 96.3% |
| NeMo TitaNet-small | 40.3 | 192 | 59 | 45 | 4.3% / 9.1% | 8.6% / 3.1% | 88.6% / 94.2% |
| 3D-Speaker ERes2Net base 200k | 39.6 | 512 | 209 | 154 | **2.6% / 6.3%** | 4.3% / 2.7% | 93.9% / 96.2% |
| 3D-Speaker ERes2NetV2 | 71.4 | 192 | 512 | 382 | 3.1% / 7.9% | 4.4% / 3.6% | 93.6% / 95.4% |
| 3D-Speaker ERes2Net VoxCeleb | 26.5 | 192 | 207 | 149 | 5.6% / 10.8% | 6.2% / 3.9% | – |
| NeMo TitaNet-large | 101.4 | 192 | 226 | 102 | 5.1% / 10.5% | 6.5% / 5.6% | – |
| WeSpeaker ResNet34-LM | 26.5 | 256 | 199 | 153 | 18.3% / 27.0% | 42% / 41% | – |
| WeSpeaker ResNet152-LM | 79.2 | 256 | 608 | 325 | 18.5% / 22.7% | 49% / 51% | – |
| 3D-Speaker CAM++ VoxCeleb | 29.6 | 512 | – | – | 36.1% / 39.1% | – | – |
| WeSpeaker CAM++-LM | 29.3 | 512 | – | – | 46.8% / 48.5% | – | – |

Notes:

- The oracle SER column is the mean of (½·clean + call + room). Each model had its own
  thresholds tuned on the tune split.
- Choosing between the first two is a judgment call. zh+en advanced reaches a slightly higher
  pipeline accuracy after recluster on test (96.3% vs 95.3%). But zh-cn common is 2.5 points
  better online, where the user watches the labels live, has lower pair EER in call and room, and
  is better on every oracle number. The change detector's thresholds are also tuned on its
  embeddings. ERes2Net-base-200k is the most accurate embedding, but at 209 ms per 5 s and
  2.2 s per 20 s `change_points` it is over budget.
- The WeSpeaker `_LM` models and the 3D-Speaker VoxCeleb CAM++ produce near-random or badly
  scaled similarities through sherpa-onnx here. That is probably a front-end/normalization
  mismatch in these exports. I did not investigate, since they are also over the CPU budget.
- Speaker identity is largely language-independent, and the Chinese-trained 3D-Speaker models
  transfer well to Russian and Kazakh.

Full pair-EER table (12 meetings; the "1–2 s vs ≥3 s" column shows how much short segments hurt):

| model | cond | EER ≥1 s | EER ≥3 s | EER 1–2 s vs ≥3 s |
|---|---|---|---|---|
| CAM++ zh-cn common | clean / call / room | 3.7 / 3.6 / 7.8% | 0.9 / 0.5 / 1.2% | 2.4 / 2.1 / 6.4% |
| CAM++ zh+en advanced | clean / call / room | 3.8 / 4.3 / 9.2% | 0.5 / 0.5 / 2.6% | 2.3 / 3.4 / 7.4% |
| ERes2Net base 200k | clean / call / room | 3.6 / 2.6 / 6.3% | 0.5 / 0.4 / 1.6% | 2.1 / 1.8 / 4.4% |
| ERes2NetV2 | clean / call / room | 4.3 / 3.1 / 7.9% | 0.4 / 0.4 / 1.8% | 3.0 / 1.7 / 5.3% |
| TitaNet-small | clean / call / room | 4.1 / 4.3 / 9.1% | 0.9 / 1.0 / 2.8% | 2.6 / 2.8 / 6.7% |
| TitaNet-large | clean / call / room | 4.6 / 5.1 / 10.5% | 1.8 / 1.8 / 3.7% | 3.2 / 3.2 / 8.4% |
| ERes2Net VoxCeleb | clean / call / room | 4.8 / 5.6 / 10.8% | 0.8 / 0.8 / 3.3% | 2.8 / 3.4 / 8.4% |

## Change detection (`change_points`)

This runs on finals of at least 2.5 s from the pipeline, with 1.5 s windows, a 0.5 s hop, 2 s of
context on each side, and pieces of at least 1 s. "Changes found" and "cuts that are real" use a
±0.5 s tolerance.

| split | `change_threshold` | single-speaker finals split | changes found | cuts that are real | mixed speech before → after |
|---|---|---|---|---|---|
| tune | 0.40 | 0.7% | 46% | 82% | 39% → 21% |
| tune | 0.45 | 1.0% | 56% | 81% | 39% → 16% |
| tune | 0.50 | 1.7% | 62% | 77% | 39% → 13% |
| **tune** | **0.55** | **4.2%** | **69%** | **71%** | **39% → 11%** |
| tune | 0.60 | 9.6% | 72% | 60% | 39% → 10% |
| test | 0.45 | 0.7% | 56% | 77% | 35% → 14% |
| test | 0.50 | 1.9% | 61% | 72% | 35% → 12% |
| **test** | **0.55** | **4.8%** | **66%** | **64%** | **35% → 9%** |
| test | 0.60 | 10.1% | 69% | 50% | 35% → 7% |

Downstream online error (tune / test) with pieces from each setting, `ClusterConfig` re-tuned
each time:

| unsplit | hop 500 @0.40 | @0.45 | @0.50 | **@0.55** | @0.60 | @0.65 | hop 250 @0.45 | hop 250 @0.50 |
|---|---|---|---|---|---|---|---|---|
| 15.4% / 17.1% | 7.3 / 10.0 | 5.9 / 9.4 | 4.9 / 8.1 | **4.4 / 6.3** | 4.7 / 6.8 | 5.9 / 7.9 | 6.1 / 9.6 | 4.5 / 8.2 |

**Default: `change_threshold = 0.55`, `min_piece_ms = 1000`, hop 500 ms.** That split rate is more
sensitive than "almost never": about 1 in 22 single-speaker finals of 2.5 s or more gets one cut,
but the cut moves to the quietest 20 ms within ±0.3 s. It halves the error of the next setting down.
For fewer false cuts, use 0.50 (1.8% of single-speaker finals) and accept about 2 points more
online error. Hop 250 ms is no better at matched false-cut rates and costs twice as much (1.5 s
per 20 s on a P-core, 4.3 s on an E-core).

Remaining misses are mostly changes with no acoustic break, less than 1 s from the edge of the
final, or between similar voices. After splitting, 8–14% of the speech is still in mixed pieces.

## `ClusterConfig` tuning (pipeline, split pieces)

The online threshold and merge margin are swept below. Cells show online error on tune, with
test in parentheses:

| threshold | merge +0.05 | **merge +0.10** | merge +0.15 | no merge |
|---|---|---|---|---|
| 0.550 | 10.6% (10.7%) | 10.0% (9.2%) | 9.8% (9.3%) | 9.0% (9.3%) |
| 0.575 | 7.7% (8.0%) | 7.2% (7.7%) | 7.0% (6.8%) | 6.3% (7.0%) |
| **0.600** | 6.1% (8.0%) | **4.4% (6.3%)** | 4.5% (5.9%) | 4.5% (6.2%) |
| 0.625 | 6.2% (8.0%) | 5.2% (7.0%) | 4.5% (7.4%) | 4.6% (7.6%) |
| 0.650 | 6.1% (6.5%) | 6.2% (7.1%) | 5.6% (8.0%) | 5.6% (8.1%) |
| 0.700 | 8.0% (9.9%) | 7.4% (11.1%) | 7.4% (11.5%) | 7.5% (11.5%) |

The short-segment settings, as online error on tune (test):

| min_embed_ms | min_new_speaker_ms | short_segment_ms | relax | online error |
|---|---|---|---|---|
| **500** | **1500** | **3000** | **0.1** | **4.4% (6.3%)** |
| 500 | 500 | 3000 | 0.1 | 4.5% (6.3%) |
| 500 | 1500 | 2000 | 0.1 | 5.5% (6.3%) |
| 500 | 1500 | 0 (off) | – | 6.1% (6.9%) |

On oracle turns the relaxation mattered even more: it cut the online SER objective (½·clean +
call + room) from 6.5% to 3.8% on test.
`min_new_speaker_ms = 500` scores the same on tune, but 1500 is kept. The synthetic data has no
coughs or door slams, and in a real room those shouldn't open new speakers.

`context_ms = 2000` is the spec's default and is **not tuned**: no segment in this data is
shorter than `min_embed_ms`, so the fallback never fires.

Here is `recluster_threshold` × `min_cluster_ms`, as recluster error on tune (test). The rule
was the lowest tune error such that **no tune meeting gets more than 1 point worse than
online**:

| recluster_threshold | 0 | 3000 | **5000** | 8000 |
|---|---|---|---|---|
| 0.575 | 4.0% (5.7%) | 3.7% (5.4%) | 3.6% (5.0%) | 4.0% (5.8%) |
| 0.600 | 4.4% (5.8%) | 4.0% (5.5%) | 3.5% (5.2%) | 3.7% (5.7%) |
| **0.625** | 4.5% (5.4%) | 4.2% (5.1%) | **3.5% (4.7%)** | 3.5% (5.0%) |
| 0.650 | 4.7% (4.5%) | 4.3% (4.2%) | 3.5% (3.9%) | 3.4% (4.1%) |
| 0.675 | 4.5% (4.8%) | 4.3% (4.5%) | 3.1% (4.2%) | 3.1% (4.0%) |
| 0.700 | 3.8% (7.5%) | 3.6% (7.4%) | 2.5% (7.0%) | 2.5% (6.7%) |

The same data shown as the worst single-meeting change from online to recluster (all tuned
online settings fixed):

| recluster_threshold | tune: mean online → recluster | worst tune meeting | test: mean online → recluster | worst test meeting |
|---|---|---|---|---|
| 0.600 | 95.6 → 96.5% | −0.5 | 93.7 → 94.8% | −1.9 |
| **0.625** | 95.6 → 96.5% | **+0.0** | 93.7 → 95.3% | **−0.0** |
| 0.650 | 95.6 → 96.5% | +0.0 | 93.7 → 96.1% | −0.1 |
| 0.675 | 95.6 → 96.9% | −1.8 | 93.7 → 95.8% | −5.9 |
| 0.700 | 95.6 → 97.5% | −1.8 | 93.7 → 93.0% | −12.4 |

At 0.675 and above, the split step starts cutting a dominant speaker in two, whose far and near
segments differ in the room. The argmin on tune (0.70) would have been a bad choice, which is why
the rule is constrained.

For `voiceprint_threshold` (room, with enrollment), "me" precision / recall:

| threshold | online (tune) | online (test) | recluster (test) |
|---|---|---|---|
| 0.45 | 78% / 99% | 79% / 99% | 95% / 99% |
| 0.50 | 98% / 99% | 92% / 99% | 97% / 99% |
| 0.575 | 99% / 98% | 97% / 99% | 98% / 99% |
| **0.60** | **99% / 98%** | **98% / 99%** | **99% / 99%** |
| 0.65 | 99% / 98% | 99% / 98% | 99% / 99% |
| 0.70 | 99% / 97% | 99% / 98% | 99% / 99% |

The tune split is flat from 0.575 to 0.65, so the default is 0.60, in the middle and on the
precise side.

### Final defaults

```rust
ClusterConfig {
    threshold: 0.60, merge_threshold: 0.70, voiceprint_threshold: 0.60,
    min_embed_ms: 500, min_new_speaker_ms: 1500,
    short_segment_ms: 3000, short_segment_relax: 0.1,
    context_ms: 2000, max_speakers: 40,
    recluster_threshold: 0.625, min_cluster_ms: 5000,
    change_threshold: 0.55, min_piece_ms: 1000,
}
```

## Results with the defaults

### Pipeline, before and after `change_points`

| split | cond | segments | mixed speech | purity bound | online acc. | online labels | recluster acc. | recluster labels | true |
|---|---|---|---|---|---|---|---|---|---|
| tune | call, raw finals | 513 | 23.0% | 94.0% | 92.1% | 12.8 | 93.2% | 12.0 | 12.8 |
| tune | call, split | 588 | 8.1% | 99.1% | 97.7% | 14.0 | 98.6% | 13.2 | 12.8 |
| tune | room, raw finals | 288 | 50.5% | 84.4% | 72.8% | 11.0 | 78.0% | 11.2 | 12.8 |
| tune | room, split | 444 | 13.9% | 98.2% | 93.6% | 15.3 | 94.5% | 14.0 | 12.8 |
| test | call, raw finals | 522 | 18.9% | 95.1% | 91.2% | 13.7 | 92.0% | 12.2 | 12.0 |
| test | call, split | 580 | 8.7% | 99.3% | 95.4% | 13.8 | 96.4% | 12.0 | 12.0 |
| test | room, raw finals | 301 | 45.2% | 87.0% | 78.5% | 11.3 | 80.3% | 11.8 | 12.0 |
| test | room, split | 449 | 10.5% | 98.8% | 92.1% | 15.5 | 94.1% | 14.0 | 12.0 |

### Per meeting (split pieces)

| meeting | split | cond | segments | mixed | online acc. | labels | recluster acc. | labels | true |
|---|---|---|---|---|---|---|---|---|---|
| m01 | tune | call | 109 | 13% | 97.8% | 14 | 99.2% | 12 | 12 |
| m01 | tune | room | 83 | 11% | 90.3% | 11 | 90.3% | 11 | 12 |
| m02 | tune | call | 113 | 4% | 95.6% | 15 | 97.1% | 14 | 13 |
| m02 | tune | room | 77 | 12% | 97.2% | 16 | 98.2% | 15 | 13 |
| m03 | tune | call | 93 | 7% | 98.1% | 15 | 98.7% | 14 | 14 |
| m03 | tune | room | 73 | 14% | 96.8% | 17 | 97.5% | 15 | 14 |
| m04 | tune | call | 83 | 8% | 98.0% | 13 | 98.5% | 13 | 13 |
| m04 | tune | room | 68 | 12% | 97.0% | 15 | 97.6% | 13 | 13 |
| m05 | tune | call | 103 | 9% | 98.7% | 11 | 99.4% | 10 | 10 |
| m05 | tune | room | 76 | 18% | 95.5% | 14 | 96.4% | 13 | 10 |
| m06 | tune | call | 87 | 8% | 97.6% | 16 | 98.2% | 16 | 15 |
| m06 | tune | room | 67 | 16% | 83.7% | 19 | 86.5% | 17 | 15 |
| m07 | test | call | 101 | 10% | 96.7% | 14 | 98.2% | 12 | 11 |
| m07 | test | room | 73 | 7% | 93.7% | 21 | 97.1% | 18 | 11 |
| m08 | test | call | 103 | 7% | 98.1% | 12 | 99.7% | 10 | 10 |
| m08 | test | room | 70 | 7% | 92.4% | 10 | 92.6% | 10 | 10 |
| m09 | test | call | 94 | 8% | 97.7% | 13 | 98.8% | 13 | 13 |
| m09 | test | room | 78 | 11% | 97.0% | 19 | 97.6% | 18 | 13 |
| m10 | test | call | 93 | 7% | 91.5% | 16 | 91.5% | 14 | 14 |
| m10 | test | room | 85 | 9% | 91.2% | 15 | 93.7% | 14 | 14 |
| m11 | test | call | 100 | 9% | 96.7% | 14 | 97.8% | 12 | 13 |
| m11 | test | room | 71 | 17% | 86.5% | 11 | 87.5% | 11 | 13 |
| m12 | test | call | 89 | 11% | 89.9% | 14 | 90.9% | 11 | 11 |
| m12 | test | room | 72 | 11% | 92.2% | 17 | 97.2% | 13 | 11 |

### Per meeting, raw finals (no `change_points`)

| meeting | cond | mixed | online acc. | labels | recluster acc. | labels | true |
|---|---|---|---|---|---|---|---|
| m01 tune | call / room | 24% / 50% | 93.3% / 73.9% | 13 / 10 | 94.0% / 77.9% | 12 / 11 | 12 |
| m02 tune | call / room | 20% / 45% | 93.5% / 83.4% | 15 / 14 | 94.9% / 83.9% | 13 / 13 | 13 |
| m03 tune | call / room | 19% / 60% | 87.9% / 80.7% | 13 / 11 | 91.3% / 80.7% | 13 / 11 | 14 |
| m04 tune | call / room | 30% / 55% | 93.6% / 74.3% | 11 / 10 | 93.7% / 73.8% | 11 / 10 | 13 |
| m05 tune | call / room | 23% / 47% | 92.1% / 76.9% | 10 / 10 | 92.3% / 78.6% | 10 / 9 | 10 |
| m06 tune | call / room | 23% / 45% | 92.9% / 41.6% | 15 / 11 | 92.7% / 71.6% | 13 / 13 | 15 |
| m07 test | call / room | 22% / 39% | 93.6% / 88.6% | 13 / 12 | 94.9% / 86.6% | 11 / 13 | 11 |
| m08 test | call / room | 16% / 42% | 94.7% / 81.1% | 10 / 8 | 95.5% / 82.2% | 10 / 9 | 10 |
| m09 test | call / room | 10% / 46% | 94.9% / 79.7% | 14 / 13 | 97.0% / 80.7% | 13 / 14 | 13 |
| m10 test | call / room | 18% / 39% | 86.4% / 76.9% | 16 / 15 | 86.0% / 79.6% | 15 / 14 | 14 |
| m11 test | call / room | 20% / 46% | 92.1% / 70.1% | 14 / 9 | 93.0% / 77.3% | 12 / 10 | 13 |
| m12 test | call / room | 28% / 60% | 83.3% / 74.8% | 15 / 11 | 83.6% / 74.8% | 12 / 11 | 11 |

m12 call is the coordinator's live test case. Here it reaches 83% on raw finals and 90–91% with
`change_points`. The live run reported 68% online and 66% after recluster, but that run used the
provisional CAM++ zh+en model and untuned placeholder thresholds: online 0.45, recluster 0.40
over-merges, and change 0.40 on a different embedding space.

### Oracle turns (reference)

Same defaults, oracle segments. The table shows SER, then estimated / true speakers.

| split | cond | online SER | recluster SER | speakers after recluster |
|---|---|---|---|---|
| tune | clean / call / room | 2.6% / 2.8% / 7.6% | 0.9% / 1.1% / 2.2% | 12.8 / 12.8 / 12.8 of 12.8 |
| test | clean / call / room | 4.3% / 3.8% / 8.3% | 2.3% / 2.3% / 6.6% | 11.7 / 11.8 / 11.3 of 12.0 |

Error by segment length (test, oracle, room): < 1 s 41% → 22%, 1–2 s 30% → 18%, 2–7 s 4.9% →
7.4%, ≥ 7 s 5.5% → 4.0% (online → recluster). Short segments carry most of the online error.

## How `recluster` became conservative

The first version clustered every long segment from scratch (average-linkage AHC) and then
mapped the clusters back to online labels. On pipeline segments its average was good, but single
meetings broke badly. At a threshold of 0.625 the worst meeting lost 8.7 points: m05 call went
from 98.7% to 90%, with two people merged. At 0.65 the worst lost 12.7: m10 room went from
91% to 78.5%, with a dominant speaker cut in two. The online labels, built from running
centroids, were more robust than one global threshold.

The shipped version starts from the online speakers and changes them only on clear evidence:

1. Merge online groups whose average-linkage similarity is ≥ `recluster_threshold`.
2. Split a cluster only where its segments fall apart below `recluster_threshold − 0.05` into
   parts with ≥ `min_cluster_ms` of speech each. This fixes two people who shared one online
   label; in m11 room, 3 people had been lumped into one online label.
3. Move a segment only if it is ≥ 0.1 more similar to another cluster than to the rest of its
   own. Letting short segments move too (their online label is often a guess) was worth
   +0.3 points on tune and +0.5 on test.
4. Absorb clusters under `min_cluster_ms` into a neighbour within `recluster_threshold − 0.1`.

Same data, same online settings:

| recluster | tune mean | test mean | worst meeting (either split), recluster − online |
|---|---|---|---|
| from scratch, 0.625 | 95.6% | 96.4% | −8.7 (m05 call, tune) |
| from scratch, 0.65 | 96.4% | 93.4% | −12.7 (m10 room, test) |
| **online-seeded (shipped), 0.625** | **96.5%** | **95.3%** | **−0.0** |

(Online alone: 95.6% tune, 93.7% test.)

## Performance

These are measured on an Intel Core Ultra 5 125H (Meteor Lake) with the powersave governor. Other
jobs were running on the machine, and timings are medians of pinned runs.

| | P-core, 1 thread | 2 P-cores | E-core, 1 thread | 2 E-cores |
|---|---|---|---|---|
| `embed` 1 s | 15 ms | 13 ms | 33 ms | 22 ms |
| `embed` 2 s | 25 ms | 23 ms | 58 ms | 37 ms |
| `embed` 5 s | 59 ms | 49 ms | 138 ms | 80 ms |
| `embed` 10 s | 113 ms | 93 ms | 271 ms | 156 ms |
| `embed` 20 s | 255 ms | 190 ms | 561 ms | 320 ms |
| `change_points` 20 s | 0.80 s (4.0% RT) | 0.73 s | 1.79 s (8.9% RT) | 1.14 s |

- **`Embedder::new`:** 0.25–0.4 s including one warm-up run. The model file is 28.3 MB.
- **Memory:** RSS grows by about 52 MB after load and about 90 MB after 20 s inputs. The ONNX
  Runtime arena is disabled via the provider config; with it, RSS was ~140 MB. Each `Embedder` is
  its own ONNX Runtime session.
- **`recluster`** (192 dims, 15 voices): 500 segments in 3 ms, 2000 in 17 ms, 6000 in 86 ms.
- **`OnlineClusterer::assign`:** microseconds (at most 40 centroids × 192 dims).
- Two threads help only on separate physical cores, and not much: CAM++ is small and
  memory-bound. On a P-core, `num_threads = 1` is the better use of the machine.

## Limits

- **The data is synthetic.** It is read speech from Common Voice, not spontaneous meeting talk.
  Each Common Voice speaker used their own device, which gives the embedding a channel cue that a
  real shared microphone doesn't. The `room` condition only partly removes it, since the source
  recordings still differ. Expect real rooms to be worse than these room numbers.
- **Overlapping speech** is simulated only as additive 0.1–0.6 s overlaps at turn changes.
  Crosstalk, laughter and people talking over each other for seconds are not modeled. An
  overlapped piece gets one label.
- **Very short turns** are weak. Segments under 1 s are 33–41% wrong online and 11–22% after
  recluster; 1–2 s segments are 17–30% wrong online. They are 11% of the speech but most of the
  errors. Back-channels ("да", "иә", "угу") are often attributed to the previous speaker.
- **Similar voices** are a problem, as are same-gender speakers at similar distances from a
  laptop mic. Room meetings end with 1–3 extra labels (14 vs 12), and m06 room stays at 84–87%.
- **Far-field laptop mic:** the room condition is 2–4 points worse than the call condition even
  after splitting, and 14–21 points worse than that without splitting. A speakerphone or USB mic in the middle of the
  table helps more than any threshold (see README).
- **Change detection** finds about 2/3 of the speaker changes inside utterances. Changes without
  a pause between similar voices are missed, and about 4–5% of single-speaker finals get one
  unnecessary cut.
- **The tuning set is small** (6 meetings × 2 conditions, 1000 pieces). Differences under
  about 0.5 points are noise. Several choices were made at the center of flat regions rather than
  at the exact tune optimum; the notes above say where.
- `context_ms` is untested: the data has no segments shorter than 0.5 s.

## Reproduce

```sh
cd crates/kenes-speakers
testdata-cache/fetch.sh                        # models (sha256-checked) + Common Voice kk/ru audio
uv run eval/build_meetings.py --meetings 12    # meetings into testdata-cache/meetings/
cd ../.. && export PATH="$HOME/.cargo/bin:$PATH"
E="cargo run --release -p kenes-speakers --example diarize_eval --"
$E embed --jobs 8                              # oracle segments, all models
$E pairs                                       # pair EER
$E tune --model 3dspeaker_speech_campplus_sv_zh-cn_16k-common        # oracle tuning
$E segment --jobs 4                            # kenes-stt finals for call/room (needs the STT model)
M=3dspeaker_speech_campplus_sv_zh-cn_16k-common
$E embed-real --model $M                       # finals + change-detection windows
$E change-eval --model $M --hop 500
$E split --model $M --set change_threshold=0.55   # → variant split_t0.55_p1000_h500
$E tune-real --model $M --variant split_t0.55_p1000_h500
$E report-real --model $M --variant split_t0.55_p1000_h500   # the tables above
$E curve-real --model $M --variant split_t0.55_p1000_h500 --key recluster_threshold --from 0.575 --to 0.7 --step 0.025
$E bench --model $M --threads 1                # use taskset -c <cpu> for stable numbers
$E bench-recluster
```

`testdata-cache/` (~1.6 GB) is gitignored. `fetch.sh` lists the candidate models, and their
sha256 values come from the release's `checksum.txt`.
