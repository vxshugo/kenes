#!/usr/bin/env python3
"""Fetch a small ru / kk / code-switch evaluation set for the kenes ASR benchmark.

Sets written to bench/data/<set>/ as <id>.wav (16 kHz mono s16) + <id>.txt, plus manifest.jsonl:

  fleurs_kk   ~60 utts, FLEURS kk_kz test split (read speech, Wikipedia sentences)
  fleurs_ru   ~60 utts, FLEURS ru_ru test split
  cv_kk       ~60 utts, Common Voice 11 kk test split (crowdsourced, varied mics; from the
              ungated HF mirror Shirali/common_voice_11_0_kk)
  codeswitch  30 synthetic files: one FLEURS kk + one FLEURS ru utterance joined with 300 ms
              silence (alternating kk->ru / ru->kk), reference = both references joined

FLEURS is fetched as raw test.tsv + audio/test.tar.gz straight from the HF dataset repo (the
`datasets` loader for google/fleurs is script-based). Utterances whose reference contains
digits are excluded (GigaAM is char-level without digits; this mirrors the GigaAM paper's
protocol and keeps whisper's digit formatting from dominating the error rate).

Usage:  uv run prepare_data.py [--n 60] [--n-cs 30] [--seed 1234]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
DATA = ROOT / "data"
SR = 16000

FLEURS_BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
CV_KK_URL = (
    "https://huggingface.co/datasets/Shirali/common_voice_11_0_kk/resolve/main/"
    "data/test-00000-of-00001-a73dd91d0b289cf8.parquet"
)

csv.field_size_limit(sys.maxsize)


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    return dest


def to_16k_mono(src: bytes | Path) -> np.ndarray:
    """Decode anything ffmpeg understands into float32 16 kHz mono."""
    inp = "pipe:0" if isinstance(src, (bytes, bytearray)) else str(src)
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", inp,
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"]
    out = subprocess.run(cmd, input=src if isinstance(src, (bytes, bytearray)) else None,
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def write_item(set_dir: Path, uid: str, audio: np.ndarray, text: str, **meta) -> dict:
    set_dir.mkdir(parents=True, exist_ok=True)
    sf.write(set_dir / f"{uid}.wav", audio, SR, subtype="PCM_16")
    (set_dir / f"{uid}.txt").write_text(text.strip() + "\n", encoding="utf-8")
    return {"id": uid, "wav": f"{uid}.wav", "text": text.strip(),
            "duration": round(len(audio) / SR, 3), **meta}


def write_manifest(set_dir: Path, items: list[dict]) -> None:
    with open(set_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    dur = sum(i["duration"] for i in items)
    print(f"  -> {set_dir.relative_to(ROOT)}: {len(items)} utts, {dur/60:.1f} min")


# ---------------------------------------------------------------- FLEURS
def load_fleurs_rows(lang: str) -> list[dict]:
    tsv = download(f"{FLEURS_BASE}/{lang}/test.tsv", CACHE / "fleurs" / lang / "test.tsv")
    rows = []
    with open(tsv, encoding="utf-8") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            rows.append({"sid": r[0], "file": r[1], "raw": r[2], "norm": r[3],
                         "dur": int(r[5]) / SR, "gender": r[6]})
    return rows


def pick_fleurs(rows: list[dict], n: int, rng: random.Random, min_d: float, max_d: float,
                exclude_sids: set[str] = frozenset()) -> list[dict]:
    """One recording per sentence id, no digits, duration window."""
    by_sid: dict[str, list[dict]] = {}
    for r in rows:
        if re.search(r"\d", r["raw"]) or not (min_d <= r["dur"] <= max_d):
            continue
        if r["sid"] in exclude_sids:
            continue
        by_sid.setdefault(r["sid"], []).append(r)
    sids = sorted(by_sid)
    rng.shuffle(sids)
    return [rng.choice(by_sid[s]) for s in sids[:n]]


def extract_fleurs_audio(lang: str, files: set[str]) -> dict[str, np.ndarray]:
    tar_path = download(f"{FLEURS_BASE}/{lang}/audio/test.tar.gz",
                        CACHE / "fleurs" / lang / "test.tar.gz")
    out: dict[str, np.ndarray] = {}
    with tarfile.open(tar_path, "r|gz") as tar:
        for m in tar:
            name = Path(m.name).name
            if m.isfile() and name in files:
                out[name] = to_16k_mono(tar.extractfile(m).read())
                if len(out) == len(files):
                    break
    missing = files - out.keys()
    if missing:
        raise RuntimeError(f"{lang}: {len(missing)} files missing from tar")
    return out


# ---------------------------------------------------------------- Common Voice kk
def prepare_cv_kk(n: int, rng: random.Random) -> None:
    import pyarrow.parquet as pq

    print("[cv_kk] Common Voice 11 kk test (HF mirror Shirali/common_voice_11_0_kk)")
    try:
        pq_path = download(CV_KK_URL, CACHE / "cv_kk" / "test.parquet")
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP cv_kk: {e}")
        return
    rows = pq.read_table(pq_path).to_pylist()
    rows = [r for r in rows if not re.search(r"\d", r["sentence"])]
    # dedupe by sentence
    seen, uniq = set(), []
    for r in rows:
        if r["sentence"] not in seen:
            seen.add(r["sentence"])
            uniq.append(r)
    rng.shuffle(uniq)
    items = []
    set_dir = DATA / "cv_kk"
    for r in uniq:
        audio = to_16k_mono(r["audio"]["bytes"])
        d = len(audio) / SR
        if d < 1.5 or d > 20:
            continue
        uid = Path(r["audio"]["path"]).stem
        items.append(write_item(set_dir, uid, audio, r["sentence"], lang="kk",
                                source="common_voice_11_kk_test"))
        if len(items) >= n:
            break
    write_manifest(set_dir, items)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60, help="utterances per monolingual set")
    ap.add_argument("--n-cs", type=int, default=30, help="code-switch files")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    DATA.mkdir(exist_ok=True)

    picks: dict[str, list[dict]] = {}
    cs_picks: dict[str, list[dict]] = {}
    for lang in ("kk_kz", "ru_ru"):
        rows = load_fleurs_rows(lang)
        picks[lang] = pick_fleurs(rows, args.n, rng, 2.0, 20.0)
        used = {r["sid"] for r in picks[lang]}
        # short utterances for code-switch so kk + ru + 0.3 s stays < ~24 s
        cs_picks[lang] = pick_fleurs(rows, args.n_cs, rng, 3.0, 11.5, exclude_sids=used)

    for lang, set_name, code in (("kk_kz", "fleurs_kk", "kk"), ("ru_ru", "fleurs_ru", "ru")):
        print(f"[{set_name}] FLEURS {lang} test")
        need = {r["file"] for r in picks[lang]} | {r["file"] for r in cs_picks[lang]}
        audio = extract_fleurs_audio(lang, need)
        items = [write_item(DATA / set_name, Path(r["file"]).stem, audio[r["file"]], r["raw"],
                            lang=code, source=f"fleurs_{lang}_test", gender=r["gender"])
                 for r in picks[lang]]
        write_manifest(DATA / set_name, items)
        for r in cs_picks[lang]:
            r["audio"] = audio[r["file"]]

    print("[codeswitch] synthetic kk+ru concatenations")
    sil = np.zeros(int(0.3 * SR), dtype=np.float32)
    items = []
    for i, (kk, ru) in enumerate(zip(cs_picks["kk_kz"], cs_picks["ru_ru"])):
        first, second = (kk, ru) if i % 2 == 0 else (ru, kk)
        order = "kk-ru" if i % 2 == 0 else "ru-kk"
        a1, a2 = first["audio"], second["audio"]
        # loudness-match the two halves (different FLEURS speakers/mics)
        rms = lambda x: float(np.sqrt(np.mean(x ** 2)) + 1e-9)  # noqa: E731
        a2 = a2 * (rms(a1) / rms(a2))
        audio = np.concatenate([a1, sil, a2]).astype(np.float32)
        audio = audio / max(1.0, float(np.abs(audio).max()) / 0.99)
        text = f"{first['raw'].strip()} {second['raw'].strip()}"
        items.append(write_item(
            DATA / "codeswitch", f"cs{i:02d}_{order}", audio, text, lang="kk+ru", order=order,
            parts=[first["raw"].strip(), second["raw"].strip()],
            part_langs=order.split("-"),
            source=f"fleurs:{Path(first['file']).stem}+{Path(second['file']).stem}"))
    write_manifest(DATA / "codeswitch", items)

    if not args.skip_cv:
        prepare_cv_kk(args.n, rng)
    print("done")


if __name__ == "__main__":
    main()
