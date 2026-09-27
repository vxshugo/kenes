# kenes-speakers

This crate tells who is speaking in a kenes meeting (10–15 people, Russian and Kazakh). It gives
each voice a stable label for the whole meeting, so the transcript and the summaries can say who
said what. Everything runs locally on the CPU, with sherpa-onnx `=1.13.8`, the same version
kenes-stt uses.

The API contract is in [`docs/CONTRACT.md`](../../docs/CONTRACT.md) (section `kenes-speakers`).
Measurements are in [`EVAL.md`](EVAL.md).

## Model

The model is **3D-Speaker CAM++ "zh-cn 16k common"**: `3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx`,
from the sherpa-onnx `speaker-recongition-models` release. It is 28.3 MB (28,281,138 bytes),
has sha256 `f682b514c05d947ee3fa91cd6ec6c5c7543479a128373fa29b1faedccd21fd11`, and produces
192-dim embeddings. `ensure_speaker_model(models_dir, progress)` downloads it to
`<models_dir>/speaker-embedding/` through `kenes_stt::download_verified` (sha256-checked,
resumable).

It was chosen among 11 candidates on synthetic ru/kk meetings. It has the best accuracy for its
cost, and together with TitaNet-small it is the fastest.

| | |
|---|---|
| embedding a 5 s segment | 59 ms on one P-core, 49 ms on two, 138 ms on an E-core (Core Ultra 5 125H) |
| `change_points` on a 20 s utterance | 0.8 s on a P-core (4% of real time), 1.8 s on an E-core |
| `Embedder::new` | 0.25–0.4 s, including a warm-up run |
| memory per `Embedder` | about 50 MB after load, about 90 MB after 20 s inputs |
| `recluster` | 17 ms for 2000 segments, 86 ms for 6000 |

On meetings segmented by the real kenes-stt VAD and cut with `change_points`, these are the
shares of speech attributed to the right person (held-out test meetings):

| | online (live) | after `recluster` |
|---|---|---|
| call (system audio) | 95.4% | 96.4% |
| room (laptop mic across the table) | 92.1% | 94.1% |

The enrolled user ("me") gets 98–99% precision and recall. Without `change_points`, the room
figure drops to about 78%, because the VAD glues speakers together.

## API

```rust
pub fn ensure_speaker_model(models_dir: &Path, progress: &mut dyn FnMut(f32)) -> anyhow::Result<PathBuf>;
pub fn speaker_model_path(models_dir: &Path) -> PathBuf;          // addition
pub fn is_speaker_model_downloaded(models_dir: &Path) -> bool;     // addition
pub const SPEAKER_MODEL: SpeakerModel;                             // addition: file, url, sha256, size

pub struct Embedder;                                               // Send + Sync
impl Embedder {
    pub fn new(model_path: &Path, num_threads: i32) -> anyhow::Result<Self>;
    pub fn embed(&mut self, samples: &[f32]) -> anyhow::Result<Vec<f32>>;   // 16 kHz mono → unit vector
    pub fn dim(&self) -> usize;                                    // addition (192)
}

pub struct ClusterConfig { threshold, min_embed_ms, merge_threshold, voiceprint_threshold,
    min_new_speaker_ms, short_segment_ms, short_segment_relax, context_ms, max_speakers,
    recluster_threshold, min_cluster_ms, change_threshold, min_piece_ms }   // Default = tuned

pub struct OnlineClusterer;
impl OnlineClusterer {
    pub fn new(prefix: &str, cfg: ClusterConfig) -> Self;
    pub fn set_voiceprint(&mut self, embedding: Vec<f32>);
    pub fn assign(&mut self, embedding: Option<&[f32]>, start_ms: u64, end_ms: u64) -> Option<String>;
    pub fn resolve<'a>(&'a self, label: &'a str) -> &'a str;       // addition: merged label → survivor
    pub fn merges(&self) -> impl Iterator<Item = (&str, &str)>;    // addition
    pub fn num_speakers(&self) -> usize;                           // addition
}

pub struct ClusterItem { segment_id, prefix, embedding, duration_ms, online_label }
pub fn recluster(items: &[ClusterItem], voiceprint: Option<&[f32]>, cfg: &ClusterConfig)
    -> Vec<(String, Option<String>)>;

pub fn change_points(embedder: &mut Embedder, samples: &[f32], cfg: &ClusterConfig) -> Vec<usize>;
```

The crate also has `cosine`, `l2_normalize`, the `ME` constant, `metrics` (Hungarian matching and
speaker error rate) and `change::{window_embeddings, pick_changes, snap_to_quiet}`, which are the
building blocks the eval uses.

## How labels work

The labels follow the contract: `"me"`, `"mic:N"`, `"sys:N"`, or `None`. Numbers start at 1
for each prefix, in order of first appearance, and are never reused.

**Online** (`OnlineClusterer`, one per source). Each speaker is a centroid: the duration-weighted
mean of their segments' unit embeddings, re-normalized.

- A segment joins the most similar speaker if the cosine similarity is at least `threshold`
  (0.60). Segments shorter than `short_segment_ms` (3 s) join at 0.1 less, because short
  embeddings are noisier.
- Otherwise it opens a new speaker, but only if it is at least `min_new_speaker_ms` (1.5 s) long.
  Shorter ones inherit the label of the previous segment of that source if it ended within
  `context_ms` (2 s) before; otherwise they get `None`.
- `embedding: None` (a segment under `min_embed_ms`, 0.5 s) takes the same context path.
- With a voiceprint set, a segment that matches it at `voiceprint_threshold` (0.60) or more is
  `"me"`. The voiceprint and the speakers compete by margin over their own thresholds.
- After each assignment, two speakers whose centroids reach `merge_threshold` (0.70) are merged.
  A speaker that comes close to the voiceprint merges into `"me"`.
- **Merging never renames labels already emitted.** The survivor's label (the user, else the
  speaker with more speech, else the older one) is used from then on. `resolve("mic:5")` tells you
  that `mic:5` now means `mic:2`.
- The cap is `max_speakers` (40) per source. Beyond that, new voices go to the nearest speaker.

**End of meeting** (`recluster`). This step is conservative. It starts from the online speakers
and changes things only where the whole meeting is clear:

1. It merges online speakers whose average-linkage similarity is at least `recluster_threshold`
   (0.625).
2. It splits a speaker whose long segments fall apart into clearly different voices, each with
   at least `min_cluster_ms` (5 s) of speech.
3. It moves single segments that are at least 0.1 closer to another speaker.
4. It folds clusters with less than 5 s of speech into a close neighbour.
5. The voiceprint picks `"me"`.
6. Clusters take over the online labels that maximize the total unchanged speech, so names the
   user gave (`rename_speaker`) stay with the right person. Clusters with no online label get
   numbers above any number used online.
7. Segments without an embedding follow their online label.

On the eval set, `recluster` is at least as good as online in every meeting, with a worst case of
−0.1 points, and on average 1–2 points better. The function is deterministic.

**Change points** (`change_points`). The VAD only ends an utterance after a pause, so a quick
reply or an interruption lands in the same final. `change_points` embeds 1.5 s windows every
0.5 s. It compares 2 s before each point with 2 s after, cuts where they are less similar than
`change_threshold` (0.55), and checks the cut on the whole pieces. It then moves each cut to the
quietest 20 ms within ±0.3 s. Pieces are at least `min_piece_ms` (1 s) long. It finds about 2/3
of in-utterance speaker changes. About 1 in 20 single-speaker utterances of 2.5 s or more gets one
unneeded cut. For fewer cuts, use 0.50 (1 in 50), at about 2 points of accuracy.

## Integration notes (kenes-core, kenes-stt)

- **Threads.** `Embedder` is `Send + Sync`, and `embed` takes `&mut self`, so give each thread its
  own instance or put one behind a `Mutex`. Two instances (kenes-stt's splitter and kenes-core's
  labeler) cost about 2 × 90 MB. A shared `Arc<Mutex<Embedder>>` halves that; the calls are short.
  `num_threads = 1` is the right default. A second thread saves about 17% on P-cores (40% on
  E-cores), and only on a separate physical core.
- **Minimum length.** Don't embed segments shorter than `ClusterConfig::min_embed_ms` (500 ms);
  pass `None` instead. `embed` refuses input under 100 ms (`MIN_EMBED_SAMPLES`) and returns an
  error rather than crashing. Skip finals with empty text (noise) entirely: don't call `assign` for
  them.
- **Order.** Call `assign` per source in the order segments are finalized, with their real
  `start_ms`/`end_ms`.
- **Change points.** Call `change_points` on finals of at least 2.5 s. It returns sample offsets
  within the utterance, and each piece becomes its own segment to decode, embed and label.
- **Voiceprint.** Embed the enrollment speech with the same `Embedder`: 15–30 s, silence removed,
  in one `embed` call, with the mic the meeting will use. Store the 192 floats. Pass them to
  `set_voiceprint` on the `"mic"` clusterer in `micMode: "room"`, and to `recluster`. In
  `micMode: "me"`, label every mic segment `"me"` and don't cluster the mic.
- **Storage.** Keep the embedding of every final segment (192 × f32 = 768 bytes) for `recluster`,
  with an empty `Vec` for segments that weren't embedded.
- **Model swap.** Changing `SPEAKER_MODEL` requires re-tuning every threshold in `ClusterConfig`.
  Similarities are not comparable across models; the eval harness does this.
- ONNX Runtime options (no thread spinning, no memory arena) are written to
  `<models_dir>/speaker-embedding/ort-cpu.cfg`. The path must not contain `:`; if it does, plain
  `cpu` is used.

## Getting good results in a room

- **Use a speakerphone or a USB conference mic in the middle of the table.** This matters more
  than any setting. A laptop's built-in mic at the end of the table hears the nearby person
  clearly and everyone else as reverb. In the eval, a simulated far-field laptop mic costs 2–4
  points even with change detection, and 14–21 points without it. Real rooms are usually worse
  than the simulation.
- Keep the laptop fan and keyboard noise away from the mic. Close windows near the table.
- Enroll the voiceprint in the same room with the same mic, speaking normally for 20–30 s.
- For online calls, system audio works well (96% here). Headsets on the user's side keep the mic
  clean for `"me"`.
- Expect trouble with people talking over each other, very short interjections ("да", "иә",
  "угу"; segments under 1 s are wrong a third of the time online), and two similar voices at
  similar distances from the mic. Renaming speakers after the meeting is the practical fix, and
  `recluster` keeps those names attached.

## Tests

```sh
cargo test -p kenes-speakers                                  # unit tests, no model needed
cargo test -p kenes-speakers --release -- --ignored            # real model (downloads ~28 MB)
cargo clippy -p kenes-speakers --all-targets
```

The evaluation harness is `examples/diarize_eval.rs`; see [`EVAL.md`](EVAL.md#reproduce).
