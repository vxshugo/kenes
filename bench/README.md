# kenes ASR bench

CPU benchmark of local speech-to-text candidates for kenes (Russian, Kazakh, ru/kk code-switching).
Results and the recommendation are in [RESULTS.md](RESULTS.md).

## Run

Requires `uv` and `ffmpeg`. Everything (venv, data, models) stays inside `bench/`.

```bash
cd bench
uv sync                          # Python 3.12 venv, CPU-only torch
uv run prepare_data.py           # FLEURS kk/ru + Common Voice kk + synthetic code-switch -> data/
uv run convert_whisper.py        # whisper models (LoRA merge + CTranslate2 conversion) -> models/
uv run export_onnx.py            # optional: own ONNX export of GigaAM (fp32), see RESULTS.md
./fetch_models.sh                # GigaAM ONNX (sherpa format), GigaAM-v3 ru, vosk kz -> models/
uv run run_bench.py              # every model x every set, 4 threads -> results/
```

The official GigaAM PyTorch checkpoints are downloaded on first use into `models/gigaam/`.

Useful flags:

```bash
uv run run_bench.py --list                                  # models and whether their files exist
uv run run_bench.py --models gigaam-ml-ctc-onnx-int8-ortfeat --sets fleurs_ru,codeswitch
uv run run_bench.py --threads 2 --limit 20                  # other thread count / quick run
uv run run_bench.py --table-only                            # rebuild results/summary.{md,json}
# speed pass used for RESULTS.md (fastest of 3 runs per utterance, waits for an idle machine):
uv run run_bench.py --results-dir results/timing --sets fleurs_ru,fleurs_kk --limit 10 --repeat 3 --wait-quiet 4
uv run diagnose_features.py --set codeswitch                # sherpa-onnx vs GigaAM front-end
```

## Your own recordings

Put pairs `name.wav` + `name.txt` (reference transcript, UTF-8) in a folder; any format ffmpeg reads
works (it is resampled to 16 kHz mono). Files longer than 28 s are split with silero VAD, as the app
does, so whole-meeting recordings with a whole-meeting transcript are fine.

```bash
uv run run_bench.py --custom ~/kenes-recordings \
    --models gigaam-ml-ctc-onnx-int8-ortfeat,gigaam-ml-large-ctc-onnx-int8-ortfeat,whisper-kaz-rus-ct2-int8
# -> results/custom_kenes-recordings/summary.md (per-utterance hypotheses in <model>.json)
```

## Files

- `prepare_data.py`: evaluation sets (`data/<set>/*.wav|txt|manifest.jsonl`)
- `run_bench.py`: model registry, backends (gigaam PyTorch, sherpa-onnx, onnxruntime + numpy log-mel,
  faster-whisper, vosk), metrics, tables
- `convert_whisper.py`: merges the `abilmansplus/whisper-turbo-kaz-rus-v1` LoRA and converts it to CTranslate2
- `export_onnx.py`: exports GigaAM to sherpa-onnx format from the official repo
- `diagnose_features.py`: compares feature front-ends on the same ONNX model
- `results/`: raw per-utterance JSON per model, `summary.md`, `summary.json`
