# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "scipy>=1.11", "soundfile>=0.12"]
# ///
"""Build synthetic ru/kk multi-speaker "meetings" for the kenes-speakers eval.

Speakers come from Common Voice 17 (fsicoli/common_voice_17_0 mirror on Hugging Face, which
keeps `client_id`): Kazakh test/dev/other/train and Russian test. Run ../testdata-cache/fetch.sh
first (or let this script tell you what's missing).

Each meeting has 10-15 speakers (ru-only, kk-only or mixed), 6-9 minutes of VAD-like segments:
Zipf-like participation (a few people talk a lot, some only 2-3 times), turns of 1-3 segments,
segment lengths 0.5-20 s with many 1-2 s ones, pauses, and ~15% of turn changes overlapping by
up to 0.6 s (interruptions). One speaker is "me" and has a separate enrollment recording made
from clips that never appear in the meeting.

Conditions rendered per meeting (16 kHz mono, int16):
  clean  the Common Voice audio as is (each speaker on their own device)
  call   clean -> Opus 24 kbit/s round trip (online call, system audio)
  room   every speaker at a position in one room: synthetic RIR (RT60 0.3-0.6 s, direct-to-
         reverberant ratio falling with distance 0.5 m (me) .. 3.5 m), laptop-mic band-pass,
         pink noise at 15-20 dB SNR (far-field laptop microphone)

Output: testdata-cache/meetings/mNN/{clean,call,room}.wav, enroll_{clean,room}.wav,
manifest.json (segments with true speakers). Meetings m01-m04 are the tuning split,
m05-m08 the held-out split.

Usage: uv run eval/build_meetings.py [--meetings 8] [--seed 7]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR = 16000
HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "testdata-cache"
CV = CACHE / "cv"
OUT = CACHE / "meetings"

csv.field_size_limit(sys.maxsize)


# ------------------------------------------------------------------ clips
def load_tsv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def decode_mp3(data: bytes) -> np.ndarray:
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SR),
         "-f", "f32le", "pipe:1"],
        input=data, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def trim_speech(x: np.ndarray) -> np.ndarray:
    """Energy VAD: drop leading/trailing silence (keep 100 ms), cut inner pauses to 160 ms."""
    hop = SR // 50  # 20 ms
    n = len(x) // hop
    if n < 25:
        return x[:0]
    fr = x[: n * hop].reshape(n, hop)
    db = 10 * np.log10(np.mean(fr**2, axis=1) + 1e-10)
    thr = max(np.percentile(db, 95) - 30, np.percentile(db, 10) + 8, -55)
    voiced = db > thr
    idx = np.flatnonzero(voiced)
    if len(idx) == 0:
        return x[:0]
    first, last = max(idx[0] - 5, 0), min(idx[-1] + 5, n - 1)
    keep = np.zeros(n, bool)
    run = 0
    for i in range(first, last + 1):
        if voiced[i]:
            run = 0
            keep[i] = True
        else:
            run += 1
            keep[i] = run <= 8
    return fr[keep].reshape(-1).copy()


def active_rms(x: np.ndarray) -> float:
    hop = SR // 50
    n = len(x) // hop
    if n == 0:
        return 1e-4
    e = np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1)
    top = e[e >= np.percentile(e, 50)]
    return float(np.sqrt(np.mean(top)) + 1e-9)


def load_speakers(min_clips_kk: int, min_clips_ru: int, rng: np.random.Generator):
    need = [CV / f for f in ["kk_test.tsv", "kk_dev.tsv", "kk_other.tsv", "kk_train.tsv",
                             "ru_test.tsv", "kk_test.tar", "kk_dev.tar", "kk_other.tar",
                             "kk_train.tar", "ru_test.tar"]]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        sys.exit(f"missing {missing}; run testdata-cache/fetch.sh first")

    by_spk: dict[str, dict] = {}
    tar_of: dict[str, Path] = {}
    for lang, splits in (("kk", ["test", "dev", "other", "train"]), ("ru", ["test"])):
        for s in splits:
            for r in load_tsv(CV / f"{lang}_{s}.tsv"):
                sid = f"{lang}-{hashlib.sha1(r['client_id'].encode()).hexdigest()[:10]}"
                d = by_spk.setdefault(sid, {"id": sid, "lang": lang, "gender": "", "paths": []})
                if r["path"] not in d["paths"]:
                    d["paths"].append(r["path"])
                d["gender"] = d["gender"] or (r.get("gender") or "")
                tar_of[r["path"]] = CV / f"{lang}_{s}.tar"

    chosen = {}
    for sid, d in by_spk.items():
        mn = min_clips_kk if d["lang"] == "kk" else min_clips_ru
        if len(d["paths"]) >= mn:
            chosen[sid] = d
    # ru has ~230 eligible speakers; we need far fewer. Keep the ones with most clips.
    ru = sorted((d for d in chosen.values() if d["lang"] == "ru"), key=lambda d: -len(d["paths"]))
    ru = ru[:90]
    kk = [d for d in chosen.values() if d["lang"] == "kk"]
    chosen = {d["id"]: d for d in ru + kk}
    wanted = {p: sid for sid, d in chosen.items() for p in d["paths"][:40]}
    print(f"decoding {len(wanted)} clips of {len(chosen)} speakers "
          f"({len(kk)} kk, {len(ru)} ru)")

    raw: dict[str, bytes] = {}
    for tar_path in sorted({tar_of[p] for p in wanted}):
        with tarfile.open(tar_path) as tar:
            for m in tar:
                name = Path(m.name).name
                if m.isfile() and name in wanted:
                    raw[name] = tar.extractfile(m).read()

    def work(item):
        name, data = item
        try:
            return name, trim_speech(decode_mp3(data))
        except subprocess.CalledProcessError:
            return name, np.zeros(0, np.float32)

    with ThreadPoolExecutor(max_workers=12) as ex:
        decoded = dict(ex.map(work, raw.items()))

    speakers = []
    for sid, d in chosen.items():
        clips = [decoded[p] for p in d["paths"][:40] if p in decoded and len(decoded[p]) >= SR]
        # level-normalize each clip to -26 dBFS active speech
        clips = [(c * (0.05 / active_rms(c))).astype(np.float32) for c in clips]
        total = sum(len(c) for c in clips) / SR
        if len(clips) >= 6 and total >= 25:
            speakers.append({"id": sid, "lang": d["lang"], "gender": d["gender"],
                             "clips": clips, "speech_s": total})
    print(f"usable: {sum(s['lang']=='kk' for s in speakers)} kk, "
          f"{sum(s['lang']=='ru' for s in speakers)} ru speakers")
    return speakers


# ------------------------------------------------------------------ meeting script
class ClipSource:
    """Hands out a speaker's speech, preferring audio not used yet in this meeting."""

    def __init__(self, clips: list[np.ndarray], rng: np.random.Generator):
        self.clips = clips
        self.rng = rng
        self.order = list(rng.permutation(len(clips)))
        self.pos = 0  # sample offset in current clip
        self.reused = 0

    def _clip(self) -> np.ndarray:
        if not self.order:
            self.order = list(self.rng.permutation(len(self.clips)))
            self.reused += 1
        return self.clips[self.order[0]]

    def take(self, n: int) -> np.ndarray:
        if n < 2 * SR:  # short segment: a random window of one clip
            c = self._clip()
            self.order.pop(0)
            self.pos = 0
            if len(c) <= n:
                return c.copy()
            s = int(self.rng.integers(0, len(c) - n))
            return c[s : s + n].copy()
        parts, have = [], 0
        while have < n:
            c = self._clip()
            piece = c[self.pos : self.pos + (n - have)]
            parts.append(piece)
            have += len(piece)
            self.pos += len(piece)
            if self.pos >= len(c):
                self.order.pop(0)
                self.pos = 0
            if have < n:
                gap = np.zeros(int(SR * self.rng.uniform(0.08, 0.2)), np.float32)
                parts.append(gap)
                have += len(gap)
        return np.concatenate(parts)[:n]


def seg_duration(rng: np.random.Generator) -> float:
    u = rng.random()
    if u < 0.10:
        return rng.uniform(0.5, 1.0)  # back-channel: "да", "иә", "угу"
    if u < 0.42:
        return rng.uniform(1.0, 2.0)
    if u < 0.78:
        return rng.uniform(2.0, 7.0)
    return rng.uniform(7.0, 20.0)


def make_script(spk: list[dict], me: int, rng: np.random.Generator, target_s: float):
    """List of (speaker_index, start_s, duration_s)."""
    n = len(spk)
    ranks = rng.permutation(n) + 1
    # "me" talks a fair amount in a room meeting
    ranks[ranks == 2], ranks[me] = ranks[me], 2
    w = 1.0 / ranks.astype(float) ** 0.9
    w /= w.sum()
    # everyone speaks at least twice (spread over the meeting), the rest is Zipf-like
    pending = {k: 2 for k in range(n)}
    t, prev, script = 0.0, -1, []
    while t < target_s or pending:
        cands = [k for k in pending if k != prev]
        if cands and (t >= target_s or rng.random() < 0.3):
            k = int(rng.choice(cands))
        else:
            k = prev
            while k == prev:
                k = int(rng.choice(n, p=w))
        if k in pending:
            pending[k] -= 1
            if pending[k] == 0:
                del pending[k]
        # pause before the turn
        u = rng.random()
        if not script:
            gap = 0.5
        elif u < 0.15:
            gap = -rng.uniform(0.1, 0.6)  # interruption
        elif u < 0.85:
            gap = rng.uniform(0.2, 1.5)
        else:
            gap = rng.uniform(1.5, 4.0)
        t = max(0.0, t + gap)
        nseg = int(rng.choice([1, 2, 3], p=[0.55, 0.3, 0.15]))
        for j in range(nseg):
            d = seg_duration(rng)
            script.append((k, t, d))
            t += d
            if j + 1 < nseg:
                t += rng.uniform(0.4, 1.2)  # pause inside a turn: VAD splits here
        prev = k
    return script


# ------------------------------------------------------------------ acoustics
def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(X))
    f[0] = 1
    y = np.fft.irfft(X / np.sqrt(f), n)
    return (y / (np.std(y) + 1e-9)).astype(np.float32)


def make_rir(dist: float, rt60: float, rng: np.random.Generator) -> np.ndarray:
    n = int(SR * min(rt60 * 1.2, 0.8))
    t = np.arange(n) / SR
    h = np.zeros(n, np.float32)
    d0 = int(SR * dist / 343.0)
    h[d0] = 1.0 / dist
    # a few early reflections
    for _ in range(6):
        k = d0 + int(rng.uniform(0.002, 0.02) * SR)
        if k < n:
            h[k] += rng.uniform(-0.6, 0.6) / (dist + rng.uniform(0.5, 2.0))
    # diffuse tail; energy set so the critical distance is ~1 m (DRR 0 dB at 1 m)
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    tail[: d0 + int(0.003 * SR)] = 0
    tail *= np.sqrt(1.0 / (np.sum(tail**2) + 1e-9))
    return h + tail.astype(np.float32)


def laptop_mic(x: np.ndarray) -> np.ndarray:
    sos = signal.butter(4, [150, 6500], btype="bandpass", fs=SR, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def opus_roundtrip(x: np.ndarray, kbps: int = 24) -> np.ndarray:
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV", subtype="FLOAT")
    enc = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "wav", "-i", "pipe:0",
                          "-c:a", "libopus", "-b:a", f"{kbps}k", "-application", "voip",
                          "-f", "ogg", "pipe:1"], input=buf.getvalue(), capture_output=True,
                         check=True).stdout
    dec = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0", "-ac", "1",
                          "-ar", str(SR), "-f", "f32le", "pipe:1"], input=enc,
                         capture_output=True, check=True).stdout
    y = np.frombuffer(dec, np.float32)
    out = np.zeros_like(x)
    out[: min(len(x), len(y))] = y[: len(x)]
    return out


def to_int16(x: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(x)) + 1e-9
    return (np.clip(x * (0.9 / peak), -1, 1) * 32767).astype(np.int16)


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meetings", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    speakers = load_speakers(8, 10, rng)
    kk = [s for s in speakers if s["lang"] == "kk"]
    ru = [s for s in speakers if s["lang"] == "ru"]
    rng.shuffle(kk)
    rng.shuffle(ru)
    langs = ["ru", "kk", "mixed", "mixed", "ru", "kk", "mixed", "mixed"]
    kk_use: dict[str, int] = defaultdict(int)
    OUT.mkdir(parents=True, exist_ok=True)

    for m in range(args.meetings):
        mid = f"m{m + 1:02d}"
        lang = langs[m % len(langs)]
        n = int(rng.integers(10, 16))
        n_kk = {"ru": 0, "kk": n, "mixed": n // 2}[lang]
        # kk speakers: least used first (the kk pool is small, so some repeat across meetings)
        kk_pool = sorted(kk, key=lambda s: (kk_use[s["id"]], rng.random()))
        pick = kk_pool[:n_kk]
        for s in pick:
            kk_use[s["id"]] += 1
        pick += [ru.pop() for _ in range(n - n_kk)]
        order = rng.permutation(len(pick))
        pick = [pick[i] for i in order]
        # "me": the speaker with the most audio, so ~20 s can be held out for enrollment
        me = int(np.argmax([s["speech_s"] for s in pick]))
        sources = []
        enroll = None
        for i, s in enumerate(pick):
            clips = list(s["clips"])
            if i == me:
                perm = rng.permutation(len(clips))
                held, acc = [], 0.0
                for j in perm:
                    if acc >= 20.0:
                        break
                    held.append(int(j))
                    acc += len(clips[j]) / SR
                enroll = np.concatenate([np.concatenate([clips[j], np.zeros(SR // 5, np.float32)])
                                         for j in held])
                clips = [c for j, c in enumerate(clips) if j not in set(held)]
            sources.append(ClipSource(clips, rng))

        target = rng.uniform(6 * 60, 9 * 60)
        script = make_script(pick, me, rng, target)
        total = int((max(st + d for _, st, d in script) + 1.0) * SR)

        # per-speaker level (people speak at different volumes / gain settings)
        gains = 10 ** (rng.uniform(-4, 4, len(pick)) / 20)
        tracks = [np.zeros(total, np.float32) for _ in pick]
        segs = []
        for k, st, d in script:
            a = sources[k].take(int(d * SR)) * gains[k]
            s0 = int(st * SR)
            tracks[k][s0 : s0 + len(a)] += a
            segs.append({"speaker": pick[k]["id"], "start_ms": int(st * 1000),
                         "end_ms": int((st * SR + len(a)) * 1000 / SR)})

        mdir = OUT / mid
        mdir.mkdir(parents=True, exist_ok=True)
        clean = np.sum(tracks, axis=0)
        sf.write(mdir / "clean.wav", to_int16(clean + 1e-4 * rng.standard_normal(total)), SR,
                 subtype="PCM_16")
        sf.write(mdir / "call.wav", to_int16(opus_roundtrip(clean)), SR, subtype="PCM_16")

        # room: one RIR per seat, same room, laptop mic, pink noise
        rt60 = rng.uniform(0.3, 0.6)
        dists = rng.uniform(1.0, 3.5, len(pick))
        dists[me] = 0.5
        room = np.zeros(total, np.float32)
        rirs = [make_rir(d, rt60, rng) for d in dists]
        for k in range(len(pick)):
            room += signal.fftconvolve(tracks[k], rirs[k])[:total].astype(np.float32)
        room = laptop_mic(room)
        snr = rng.uniform(15, 20)
        speech_level = active_rms(room[np.abs(clean) > 1e-3]) if np.any(np.abs(clean) > 1e-3) else 0.05
        noise = pink_noise(total, rng) * speech_level * 10 ** (-snr / 20)
        sf.write(mdir / "room.wav", to_int16(room + noise), SR, subtype="PCM_16")

        # enrollment: clean, and at the user's seat in the same room (a bit quieter)
        sf.write(mdir / "enroll_clean.wav", to_int16(enroll), SR, subtype="PCM_16")
        er = laptop_mic(signal.fftconvolve(enroll, rirs[me])[: len(enroll)].astype(np.float32))
        er += pink_noise(len(er), rng) * active_rms(er) * 10 ** (-(snr + 5) / 20)
        sf.write(mdir / "enroll_room.wav", to_int16(er), SR, subtype="PCM_16")

        manifest = {
            "id": mid,
            "split": "tune" if m < args.meetings // 2 else "test",
            "lang": lang,
            "duration_s": round(total / SR, 1),
            "rt60_s": round(float(rt60), 2),
            "snr_db": round(float(snr), 1),
            "me": pick[me]["id"],
            "speakers": [{"id": s["id"], "lang": s["lang"], "gender": s["gender"],
                          "distance_m": round(float(dists[i]), 2),
                          "reused_clips": sources[i].reused}
                         for i, s in enumerate(pick)],
            "segments": segs,
        }
        (mdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        talk = defaultdict(float)
        for s in segs:
            talk[s["speaker"]] += (s["end_ms"] - s["start_ms"]) / 1000
        print(f"{mid} [{manifest['split']}] {lang:5s} {n} spk, {total / SR / 60:.1f} min, "
              f"{len(segs)} segs, talk s: {sorted((round(v) for v in talk.values()), reverse=True)}")


if __name__ == "__main__":
    main()
