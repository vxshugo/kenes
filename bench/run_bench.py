#!/usr/bin/env python3
"""kenes ASR benchmark: WER / CER / RTF / peak RSS for local ASR candidates on CPU.

Every model runs in its own subprocess (so peak RSS is per model and thread settings are
isolated), with a fixed thread count (default 4 = what the app uses). Audio is decoded to
float32 16 kHz mono before timing; RTF = sum(processing time) / sum(audio duration), measured
after one warm-up utterance.

  uv run run_bench.py                               # all models x default sets
  uv run run_bench.py --models gigaam-ml-ctc-onnx-int8 --sets fleurs_ru
  uv run run_bench.py --list                        # show models / sets
  uv run run_bench.py --table-only                  # rebuild results/summary.{json,md}
  uv run run_bench.py --custom path/to/dir          # your recordings: x.wav + x.txt pairs

Normalisation for WER/CER: lowercase, ё->е, punctuation -> space, collapse spaces.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS = ROOT / "results"
SR = 16000
DEFAULT_SETS = ["fleurs_ru", "fleurs_kk", "cv_kk", "codeswitch"]
LONG_AUDIO_S = 28.0  # longer inputs are VAD-segmented (custom recordings)


# ============================================================ model registry
@dataclass
class Spec:
    kind: str
    desc: str
    files: list[str]                      # relative to models/, for size-on-disk
    sets: list[str] | None = None         # restrict to these sets (None = all)
    limit: int | None = None              # max utterances per set (slow models)
    opts: dict = field(default_factory=dict)


MODELS: dict[str, Spec] = {
    "gigaam-ml-ctc-pt": Spec(
        "gigaam_pt", "GigaAM-Multilingual CTC 220M, official PyTorch fp32 (gigaam pkg)",
        ["gigaam/multilingual_ctc.ckpt"], opts={"name": "multilingual_ctc"}),
    "gigaam-ml-large-ctc-pt": Spec(
        "gigaam_pt", "GigaAM-Multilingual large CTC 600M, official PyTorch fp32 (gigaam pkg)",
        ["gigaam/multilingual_large_ctc.ckpt"], opts={"name": "multilingual_large_ctc"}),
    "gigaam-ml-ctc-onnx-int8": Spec(
        "sherpa_ctc", "GigaAM-Multilingual CTC 220M, community sherpa-onnx int8 (fgeeer77/bayaya-models)",
        ["sherpa-gigaam-ml-ctc/model.int8.onnx", "sherpa-gigaam-ml-ctc/tokens.txt"],
        opts={"dir": "sherpa-gigaam-ml-ctc"}),
    "gigaam-ml-large-ctc-onnx-int8": Spec(
        "sherpa_ctc", "GigaAM-Multilingual large CTC 600M, community sherpa-onnx int8 (fgeeer77/bayaya-models)",
        ["sherpa-gigaam-ml-large-ctc/model.int8.onnx", "sherpa-gigaam-ml-large-ctc/tokens.txt"],
        opts={"dir": "sherpa-gigaam-ml-large-ctc"}),
    "gigaam-ml-ctc-onnx-int8-ortfeat": Spec(
        "ort_ctc", "same community int8 ONNX as above, run with plain onnxruntime + GigaAM-exact log-mel "
        "(numpy, 20 ms window) instead of sherpa-onnx's fbank",
        ["sherpa-gigaam-ml-ctc/model.int8.onnx", "sherpa-gigaam-ml-ctc/tokens.txt"],
        opts={"dir": "sherpa-gigaam-ml-ctc"}),
    "gigaam-ml-large-ctc-onnx-int8-ortfeat": Spec(
        "ort_ctc", "same community large int8 ONNX, plain onnxruntime + GigaAM-exact log-mel (numpy)",
        ["sherpa-gigaam-ml-large-ctc/model.int8.onnx", "sherpa-gigaam-ml-large-ctc/tokens.txt"],
        opts={"dir": "sherpa-gigaam-ml-large-ctc"}),
    "gigaam-v3-ru-ctc-onnx-int8": Spec(
        "sherpa_ctc", "GigaAM-v3 CTC Russian-only, sherpa-onnx int8 (csukuangfj 2025-12-16)",
        ["sherpa-gigaam-v3-ru-ctc/model.int8.onnx", "sherpa-gigaam-v3-ru-ctc/tokens.txt"],
        sets=["fleurs_ru", "custom"], opts={"dir": "sherpa-gigaam-v3-ru-ctc"}),
    "whisper-kaz-rus-ct2-int8": Spec(
        "faster_whisper", "abilmansplus/whisper-turbo-kaz-rus-v1 (LoRA merged) -> CTranslate2 int8, greedy, auto-lang",
        ["ct2/whisper-turbo-kaz-rus-v1"], limit=20, opts={"path": "ct2/whisper-turbo-kaz-rus-v1"}),
    "whisper-large-v3-turbo-ct2-int8": Spec(
        "faster_whisper", "openai whisper-large-v3-turbo via faster-whisper int8, greedy, auto-lang",
        ["hf/faster-whisper-large-v3-turbo"], limit=20, opts={"path": "hf/faster-whisper-large-v3-turbo"}),
    "vosk-small-kz-0.42": Spec(
        "vosk", "vosk-model-small-kz-0.42 (Kaldi, single-threaded)",
        ["vosk-model-small-kz-0.42"], sets=["fleurs_kk", "cv_kk", "custom"],
        opts={"path": "vosk-model-small-kz-0.42"}),
}


def register_extra_models() -> None:
    """Optional self-exported ONNX builds (see export_onnx.py), registered only if present."""
    for name, sub, desc in [
        ("gigaam-ml-ctc-onnx-fp32-self", "self-onnx/multilingual_ctc/fp32",
         "GigaAM-Multilingual CTC 220M, own export fp32 (export_onnx.py)"),
        ("gigaam-ml-ctc-onnx-int8-self", "self-onnx/multilingual_ctc/int8",
         "GigaAM-Multilingual CTC 220M, own export int8 (export_onnx.py)"),
        ("gigaam-ml-large-ctc-onnx-fp32-self", "self-onnx/multilingual_large_ctc/fp32",
         "GigaAM-Multilingual large CTC 600M, own export fp32 (export_onnx.py)"),
        ("gigaam-ml-large-ctc-onnx-int8-self", "self-onnx/multilingual_large_ctc/int8",
         "GigaAM-Multilingual large CTC 600M, own export int8 (export_onnx.py)"),
    ]:
        d = MODELS_DIR / sub
        if d.is_dir() and any(d.glob("*.onnx")):
            onnx = sorted(p.name for p in d.glob("*.onnx") if not p.name.endswith(".data"))[0]
            files = [f"{sub}/{p.name}" for p in d.iterdir()]
            MODELS[name] = Spec("sherpa_ctc", desc, files, opts={"dir": sub, "model": onnx})


register_extra_models()


# ============================================================ text normalisation / metrics
def normalize(s: str) -> str:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[^\w\s]|_", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def split_errors(ref_words: list[str], hyp: str, boundary: int) -> tuple[list[int], list[int]]:
    """Attribute word errors of one utterance to [0,boundary) vs [boundary,end) of the ref.

    Returns ([errors_part1, words_part1], [errors_part2, words_part2]).
    Insertions are attributed to the part where they occur (by ref position)."""
    import jiwer

    ref = " ".join(ref_words)
    p1 = [0, boundary]
    p2 = [0, len(ref_words) - boundary]
    if not hyp.strip():
        return [boundary, boundary], [p2[1], p2[1]]
    out = jiwer.process_words(ref, hyp)
    for ch in out.alignments[0]:
        if ch.type == "equal":
            continue
        if ch.type == "insert":
            n = ch.hyp_end_idx - ch.hyp_start_idx
            (p1 if ch.ref_start_idx < boundary else p2)[0] += n
            continue
        for ri in range(ch.ref_start_idx, ch.ref_end_idx):
            (p1 if ri < boundary else p2)[0] += 1
    return p1, p2


def score(items: list[dict]) -> dict:
    import jiwer

    refs = [normalize(i["ref"]) for i in items]
    hyps = [normalize(i["hyp"]) for i in items]
    # jiwer rejects empty references; they don't occur in our sets but guard anyway
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    refs, hyps = [p[0] for p in pairs], [p[1] for p in pairs]
    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)
    raw_hyps = [i["hyp"] for i in items]
    res = {
        "n": len(items),
        "audio_s": round(sum(i["duration"] for i in items), 2),
        "proc_s": round(sum(i["proc_s"] for i in items), 3),
        "cpu_s": round(sum(i.get("cpu_s", 0.0) for i in items), 3),
        "wer": round(wer * 100, 2),
        "cer": round(cer * 100, 2),
        "frac_hyp_with_punct": round(np.mean([bool(re.search(r"[.,!?;:«»\"]", h)) for h in raw_hyps]), 2),
        "frac_hyp_with_upper": round(np.mean([any(c.isupper() for c in h) for h in raw_hyps]), 2),
        "frac_hyp_with_digits": round(np.mean([bool(re.search(r"\d", h)) for h in raw_hyps]), 2),
        "empty_hyps": sum(1 for h in hyps if not h),
    }
    res["rtf"] = round(res["proc_s"] / res["audio_s"], 4) if res["audio_s"] else None
    # CPU-seconds per audio-second: less sensitive to other load on the machine than wall RTF
    res["cpu_per_audio_s"] = round(res["cpu_s"] / res["audio_s"], 4) if res["audio_s"] else None
    # code-switch: per-language-part WER
    if items and "parts" in items[0]:
        acc: dict[str, list[int]] = {}
        for it in items:
            p1_words = normalize(it["parts"][0]).split()
            ref_words = normalize(it["ref"]).split()
            a, b = split_errors(ref_words, normalize(it["hyp"]), len(p1_words))
            for lang, (e, n) in zip(it["part_langs"], (a, b)):
                acc.setdefault(lang, [0, 0])
                acc[lang][0] += e
                acc[lang][1] += n
            for pos, (e, n) in zip(("first", "second"), (a, b)):
                acc.setdefault(pos, [0, 0])
                acc[pos][0] += e
                acc[pos][1] += n
        res["part_wer"] = {k: round(100 * e / max(n, 1), 2) for k, (e, n) in acc.items()}
    return res


# ============================================================ audio / sets
def load_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    try:
        a, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if sr == SR:
            return a.mean(axis=1) if a.shape[1] > 1 else a[:, 0]
    except Exception:  # noqa: BLE001 - fall back to ffmpeg for anything exotic
        pass
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout,
                         dtype=np.float32).copy()


def load_set(name: str, custom_dir: Path | None = None) -> list[dict]:
    if name == "custom":
        assert custom_dir is not None
        items = []
        for wav in sorted(p for p in custom_dir.iterdir()
                          if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}):
            txt = wav.with_suffix(".txt")
            if not txt.exists():
                print(f"  [custom] skip {wav.name}: no {txt.name}", file=sys.stderr)
                continue
            items.append({"id": wav.stem, "path": wav, "text": txt.read_text(encoding="utf-8").strip()})
        return items
    set_dir = DATA / name
    man = set_dir / "manifest.jsonl"
    if not man.exists():
        raise SystemExit(f"missing {man}; run `uv run prepare_data.py` first")
    items = []
    for line in man.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        d["path"] = set_dir / d["wav"]
        items.append(d)
    return items


# ============================================================ backends
class Backend:
    def transcribe(self, audio: np.ndarray) -> str:  # noqa: D401
        raise NotImplementedError

    extra: dict = {}


class GigaAMPT(Backend):
    def __init__(self, spec: Spec, threads: int):
        import torch
        import gigaam

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        self.torch = torch
        self.model = gigaam.load_model(spec.opts["name"], fp16_encoder=False, device="cpu",
                                       download_root=str(MODELS_DIR / "gigaam"))
        self.model.eval()

    def transcribe(self, audio: np.ndarray) -> str:
        torch = self.torch
        with torch.inference_mode():
            wav = torch.from_numpy(audio).unsqueeze(0)
            length = torch.tensor([wav.shape[-1]])
            enc, enc_len = self.model.forward(wav, length)
            return self.model._decode(enc, enc_len, length)[0][0]


class SherpaCTC(Backend):
    def __init__(self, spec: Spec, threads: int):
        import sherpa_onnx

        d = MODELS_DIR / spec.opts["dir"]
        self.rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(d / spec.opts.get("model", "model.int8.onnx")), tokens=str(d / "tokens.txt"),
            num_threads=threads, feature_dim=64, decoding_method="greedy_search")

    def transcribe(self, audio: np.ndarray) -> str:
        s = self.rec.create_stream()
        s.accept_waveform(SR, audio)
        self.rec.decode_stream(s)
        return s.result.text


class GigaAMLogMel:
    """Exact numpy port of gigaam.preprocess.FeatureExtractor (torchaudio MelSpectrogram:
    periodic hann, n_fft = win = 320, hop 160, center=False, power 2, 64 HTK mel bins with no
    norm, f 0..8000 Hz; then log(clamp(x, 1e-9, 1e9))). This is what the Rust app should port."""

    def __init__(self, n_fft: int = 320, hop: int = 160, n_mels: int = 64, sr: int = SR):
        self.n_fft, self.hop = n_fft, hop
        n = np.arange(n_fft)
        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * n / n_fft)).astype(np.float32)  # periodic
        n_freqs = n_fft // 2 + 1
        hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)  # noqa: E731
        mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)  # noqa: E731
        all_freqs = np.linspace(0, sr // 2, n_freqs)
        f_pts = mel2hz(np.linspace(hz2mel(0.0), hz2mel(sr / 2), n_mels + 2))
        f_diff = f_pts[1:] - f_pts[:-1]
        slopes = f_pts[None, :] - all_freqs[:, None]
        down = -slopes[:, :-2] / f_diff[:-1]
        up = slopes[:, 2:] / f_diff[1:]
        self.fb = np.maximum(0.0, np.minimum(down, up)).astype(np.float32)  # (n_freqs, n_mels)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop
        idx = np.arange(self.n_fft)[None, :] + self.hop * np.arange(n_frames)[:, None]
        frames = audio[idx] * self.window
        spec = np.abs(np.fft.rfft(frames, n=self.n_fft, axis=-1)) ** 2
        mel = spec.astype(np.float32) @ self.fb
        return np.log(np.clip(mel, 1e-9, 1e9)).T[None].astype(np.float32)  # (1, 64, T)


class OrtCTC(Backend):
    def __init__(self, spec: Spec, threads: int):
        import onnxruntime as ort

        d = MODELS_DIR / spec.opts["dir"]
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(d / spec.opts.get("model", "model.int8.onnx")), so,
                                         providers=["CPUExecutionProvider"])
        self.vocab = {}
        for line in (d / "tokens.txt").read_text(encoding="utf-8").splitlines():
            sym, i = line.rsplit(" ", 1)
            self.vocab[int(i)] = sym if sym else " "
        self.blank = max(self.vocab)
        self.feat = GigaAMLogMel()

    def transcribe(self, audio: np.ndarray) -> str:
        if len(audio) < 0.1 * SR:  # < 10 feature frames: nothing the model can decode
            return ""
        f = self.feat(audio)
        lp, ln = self.sess.run(None, {"features": f, "feature_lengths": np.array([f.shape[-1]], dtype=np.int64)})
        ids = lp[0, : int(ln[0])].argmax(-1)
        out, prev = [], -1
        for i in ids:
            if i != prev and i != self.blank:
                out.append(self.vocab[int(i)])
            prev = i
        return re.sub(r"\s+", " ", "".join(out)).strip()


class FasterWhisper(Backend):
    def __init__(self, spec: Spec, threads: int):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(str(MODELS_DIR / spec.opts["path"]), device="cpu",
                                  compute_type="int8", cpu_threads=threads, num_workers=1)
        self.beam = int(spec.opts.get("beam_size", 1))
        self.language = spec.opts.get("language")
        self.langs: list[str] = []

    def transcribe(self, audio: np.ndarray) -> str:
        segs, info = self.model.transcribe(
            audio, language=self.language, task="transcribe", beam_size=self.beam,
            condition_on_previous_text=False, vad_filter=False, without_timestamps=True)
        text = " ".join(s.text.strip() for s in segs)
        self.langs.append(f"{info.language}:{info.language_probability:.2f}")
        return text


class Vosk(Backend):
    def __init__(self, spec: Spec, threads: int):
        import vosk

        vosk.SetLogLevel(-1)
        self.vosk = vosk
        self.model = vosk.Model(str(MODELS_DIR / spec.opts["path"]))

    def transcribe(self, audio: np.ndarray) -> str:
        rec = self.vosk.KaldiRecognizer(self.model, SR)
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
        parts = []
        for i in range(0, len(pcm), 8000):
            if rec.AcceptWaveform(pcm[i:i + 8000]):
                parts.append(json.loads(rec.Result()).get("text", ""))
        parts.append(json.loads(rec.FinalResult()).get("text", ""))
        return " ".join(p for p in parts if p)


BACKENDS = {"gigaam_pt": GigaAMPT, "sherpa_ctc": SherpaCTC, "ort_ctc": OrtCTC, "faster_whisper": FasterWhisper, "vosk": Vosk}


class LongAudio:
    """VAD-segment inputs longer than LONG_AUDIO_S (custom recordings) with silero VAD."""

    def __init__(self):
        self.vad = None

    def segments(self, audio: np.ndarray) -> list[np.ndarray]:
        if len(audio) <= LONG_AUDIO_S * SR:
            return [audio]
        import sherpa_onnx

        if self.vad is None:
            model = MODELS_DIR / "silero_vad.onnx"
            if not model.exists():
                import requests

                url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
                model.write_bytes(requests.get(url, timeout=60).content)
            cfg = sherpa_onnx.VadModelConfig()
            cfg.silero_vad.model = str(model)
            cfg.silero_vad.min_silence_duration = 0.4
            cfg.silero_vad.max_speech_duration = 20.0
            cfg.sample_rate = SR
            self.cfg = cfg
        vad = sherpa_onnx.VoiceActivityDetector(self.cfg, buffer_size_in_seconds=len(audio) / SR + 5)
        win = self.cfg.silero_vad.window_size
        out = []
        for i in range(0, len(audio) - win + 1, win):
            vad.accept_waveform(audio[i:i + win])
            while not vad.empty():
                out.append(np.array(vad.front.samples, dtype=np.float32))
                vad.pop()
        vad.flush()
        while not vad.empty():
            out.append(np.array(vad.front.samples, dtype=np.float32))
            vad.pop()
        return out or [audio]


def rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 2**20


def worker(args) -> None:
    spec = MODELS[args.model]
    sets = args.sets.split(",")
    t0 = time.perf_counter()
    backend = BACKENDS[spec.kind](spec, args.threads)
    load_s = time.perf_counter() - t0
    rss_loaded = rss_mb()
    long = LongAudio()
    custom_dir = Path(args.custom) if args.custom else None

    def run(audio: np.ndarray) -> str:
        return " ".join(t for t in (backend.transcribe(seg) for seg in long.segments(audio)) if t.strip())

    loadavg_start = os.getloadavg()
    warmed = False
    out_sets = {}
    for set_name in sets:
        items = load_set(set_name, custom_dir)
        limit = args.limit or spec.limit
        if limit:
            items = items[:limit]
        results = []
        for it in items:
            audio = load_audio(it["path"])
            if not warmed:
                run(audio)
                warmed = True
            # --repeat K: time each utterance K times and keep the fastest run (robust to other
            # load on the machine); CPU time is taken from the same run
            best = None
            for _ in range(max(1, args.repeat)):
                c0 = os.times()
                t = time.perf_counter()
                hyp = run(audio)
                dt = time.perf_counter() - t
                c1 = os.times()
                if best is None or dt < best[0]:
                    best = (dt, c0, c1)
            dt, c0, c1 = best
            r = {"id": it["id"], "ref": it["text"], "hyp": hyp, "duration": len(audio) / SR,
                 "proc_s": dt, "cpu_s": (c1.user - c0.user) + (c1.system - c0.system)}
            for k in ("parts", "part_langs", "order"):
                if k in it:
                    r[k] = it[k]
            if isinstance(backend, FasterWhisper):
                r["lang_detected"] = backend.langs[-1]
            results.append(r)
            print(f"  [{args.model}/{set_name}] {it['id']}: rtf={dt / r['duration']:.3f} | {hyp[:90]}",
                  file=sys.stderr, flush=True)
        out_sets[set_name] = {"metrics": score(results), "items": results}
        out_sets[set_name]["metrics"]["loadavg_1m_end"] = round(os.getloadavg()[0], 2)
        m = out_sets[set_name]["metrics"]
        print(f"== {args.model} / {set_name}: WER {m['wer']}  CER {m['cer']}  RTF {m['rtf']}",
              file=sys.stderr, flush=True)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB -> MiB on Linux
    out = {
        "model": args.model, "desc": spec.desc, "kind": spec.kind, "threads": args.threads,
        "load_s": round(load_s, 2), "rss_after_load_mb": round(rss_loaded, 1),
        "peak_rss_mb": round(peak, 1), "size_mb": model_size_mb(spec),
        "repeat": args.repeat, "loadavg_start": [round(x, 2) for x in loadavg_start],
        "loadavg_end": [round(x, 2) for x in os.getloadavg()],
        "host": host_info(), "sets": out_sets,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def model_size_mb(spec: Spec) -> float | None:
    total = 0
    for f in spec.files:
        p = MODELS_DIR / f
        if p.is_dir():
            total += sum(q.stat().st_size for q in p.rglob("*") if q.is_file())
        elif p.exists():
            total += p.stat().st_size
        else:
            return None
    return round(total / 2**20, 1)


def host_info() -> dict:
    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"cpu": cpu, "nproc": os.cpu_count(), "python": platform.python_version()}


# ============================================================ orchestration / tables
def build_tables(res_dir: Path) -> None:
    runs = []
    for p in sorted(res_dir.glob("*.json")):
        if p.name == "summary.json":
            continue
        try:
            runs.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    order = {m: i for i, m in enumerate(MODELS)}
    runs.sort(key=lambda r: order.get(r["model"], 999))
    set_names = [s for s in DEFAULT_SETS if any(s in r["sets"] for r in runs)]
    set_names += sorted({s for r in runs for s in r["sets"]} - set(set_names))

    summary = []
    for r in runs:
        row = {k: r[k] for k in ("model", "desc", "threads", "load_s", "peak_rss_mb",
                                 "rss_after_load_mb", "size_mb")}
        tot_a = sum(s["metrics"]["audio_s"] for s in r["sets"].values())
        tot_p = sum(s["metrics"]["proc_s"] for s in r["sets"].values())
        row["rtf_all"] = round(tot_p / tot_a, 4) if tot_a else None
        row["sets"] = {k: v["metrics"] for k, v in r["sets"].items()}
        summary.append(row)
    (res_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [f"Threads: {summary[0]['threads'] if summary else '?'}. WER/CER in %, normalised "
             "(lowercase, ё->е, no punctuation). RTF = processing time / audio duration "
             "(lower is better; 0.1 = 10x faster than real time).", ""]
    lines.append("| model | " + " | ".join(f"{s} WER / CER" for s in set_names)
                 + " | RTF (all sets) | peak RSS MB | size MB |")
    lines.append("|---|" + "---|" * len(set_names) + "---|---|---|")
    for row in summary:
        cells = []
        for s in set_names:
            m = row["sets"].get(s)
            cells.append(f"{m['wer']:.1f} / {m['cer']:.1f}" + (f" (n={m['n']})" if m["n"] < 50 and not s.startswith("custom") and s != "codeswitch" else "") if m else "–")
        lines.append(f"| {row['model']} | " + " | ".join(cells)
                     + f" | {row['rtf_all']} | {row['peak_rss_mb']:.0f} | {row['size_mb']} |")
    lines += ["", "| model | set | n | audio min | WER | CER | RTF | punct | upper | digits |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for row in summary:
        for s in set_names:
            m = row["sets"].get(s)
            if not m:
                continue
            lines.append(f"| {row['model']} | {s} | {m['n']} | {m['audio_s'] / 60:.1f} | {m['wer']} | "
                         f"{m['cer']} | {m['rtf']} | {m['frac_hyp_with_punct']} | "
                         f"{m['frac_hyp_with_upper']} | {m['frac_hyp_with_digits']} |")
    # matched subset: slow models (whisper) only ran the first N utterances of each set, so
    # re-score every model on exactly those utterances for an apples-to-apples comparison
    matched = {}
    for s in set_names:
        ns = [len(r["sets"][s]["items"]) for r in runs if s in r["sets"]]
        if ns and min(ns) < max(ns):
            matched[s] = min(ns)
    if matched:
        lines += ["", "Matched subset (first N utterances of each set, the ones every model ran): WER / CER", "",
                  "| model | " + " | ".join(f"{s} (n={n})" for s, n in matched.items()) + " |",
                  "|---|" + "---|" * len(matched)]
        for r in runs:
            cells = []
            for s, n in matched.items():
                if s not in r["sets"]:
                    cells.append("–")
                    continue
                m = score(r["sets"][s]["items"][:n])
                cells.append(f"{m['wer']:.1f} / {m['cer']:.1f}")
            lines.append(f"| {r['model']} | " + " | ".join(cells) + " |")
    cs = [(row["model"], row["sets"]["codeswitch"]["part_wer"]) for row in summary
          if "codeswitch" in row["sets"] and "part_wer" in row["sets"]["codeswitch"]]
    if cs:
        lines += ["", "Code-switch WER split by the language of the reference half:", "",
                  "| model | kk half WER | ru half WER | 1st half WER | 2nd half WER |", "|---|---|---|---|---|"]
        for m, pw in cs:
            lines.append(f"| {m} | {pw.get('kk')} | {pw.get('ru')} | {pw.get('first')} | {pw.get('second')} |")
    (res_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="comma-separated model keys (default: all)")
    ap.add_argument("--sets", help=f"comma-separated sets (default: {','.join(DEFAULT_SETS)})")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, help="max utterances per set (overrides per-model limit)")
    ap.add_argument("--repeat", type=int, default=1, help="time each utterance N times, keep the fastest")
    ap.add_argument("--wait-quiet", type=float, metavar="LOAD",
                    help="before each model, wait (max 20 min) until the 1-min load average < LOAD")
    ap.add_argument("--custom", help="dir with x.wav + x.txt pairs; runs set 'custom' only")
    ap.add_argument("--results-dir", help="default results/ (or results/custom_<dir> for --custom)")
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--model", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        worker(args)
        return
    if args.list:
        for k, s in MODELS.items():
            print(f"{k:38s} {'OK ' if model_size_mb(s) else 'MISSING'} sets={s.sets or 'all'}  {s.desc}")
        print("sets:", ", ".join(DEFAULT_SETS), "+ custom (--custom DIR)")
        return

    if args.custom:
        res_dir = Path(args.results_dir or RESULTS / f"custom_{Path(args.custom).resolve().name}")
        sets = ["custom"]
    else:
        res_dir = Path(args.results_dir or RESULTS)
        sets = args.sets.split(",") if args.sets else DEFAULT_SETS
    res_dir.mkdir(parents=True, exist_ok=True)
    if args.table_only:
        build_tables(res_dir)
        return

    models = args.models.split(",") if args.models else list(MODELS)
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = str(args.threads)
    env.setdefault("HF_HUB_OFFLINE", "0")
    for m in models:
        if m not in MODELS:
            raise SystemExit(f"unknown model {m}; see --list")
        spec = MODELS[m]
        msets = [s for s in sets if spec.sets is None or s in spec.sets]
        if not msets:
            continue
        if model_size_mb(spec) is None and spec.kind != "gigaam_pt":
            print(f"!! {m}: model files missing under models/ ({spec.files}); skipped", file=sys.stderr)
            continue
        out = res_dir / f"{m}.json"
        cmd = [sys.executable, __file__, "--worker", "--model", m, "--sets", ",".join(msets),
               "--threads", str(args.threads), "--out", str(out)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.repeat > 1:
            cmd += ["--repeat", str(args.repeat)]
        if args.custom:
            cmd += ["--custom", str(Path(args.custom).resolve())]
        if args.wait_quiet:
            t_end = time.time() + 1200
            while os.getloadavg()[0] >= args.wait_quiet and time.time() < t_end:
                time.sleep(10)
            print(f"   load average before {m}: {os.getloadavg()[0]:.2f}", file=sys.stderr, flush=True)
        print(f">> {m} on {msets}", file=sys.stderr, flush=True)
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0:
            print(f"!! {m} failed with exit code {rc}", file=sys.stderr)
    build_tables(res_dir)


if __name__ == "__main__":
    main()
