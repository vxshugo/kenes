"""Compare kenes-stt's Rust GigaAM log-mel (examples/gigaam_features.rs) with the references.

    cargo run --release -p kenes-stt --example gigaam_features -- OUT_DIR a.wav b.wav ...
    uv run --project bench python crates/kenes-stt/scripts/compare_features.py OUT_DIR a.wav b.wav ...

References, per clip:
  bench     bench/run_bench.py GigaAMLogMel (numpy port; float32 window/filterbank/matmul)
  gigaam    gigaam.preprocess.FeatureExtractor as configured for GigaAM-Multilingual
            (torchaudio MelSpectrogram, float32): what the model saw in training
  float64   the same formulas evaluated in float64 (isolates the references' rounding)
Prints max / mean / p99.9 absolute differences of the log-mel values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench"))
from run_bench import GigaAMLogMel, load_audio  # noqa: E402


def float64_logmel(x: np.ndarray) -> np.ndarray:
    n_fft, hop = 320, 160
    x = x.astype(np.float64)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    n = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * w, axis=-1)) ** 2
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)  # noqa: E731
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)  # noqa: E731
    freqs = np.linspace(0, 8000, n_fft // 2 + 1)
    f_pts = mel2hz(np.linspace(0, hz2mel(8000.0), 66))
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - freqs[:, None]
    fb = np.maximum(0.0, np.minimum(-slopes[:, :-2] / f_diff[:-1], slopes[:, 2:] / f_diff[1:]))
    return np.log(np.clip(spec @ fb, 1e-9, 1e9)).T


def gigaam_logmel(x: np.ndarray):
    try:
        import torch
        from gigaam.preprocess import FeatureExtractor
    except ImportError:
        return None
    fe = FeatureExtractor(16000, 64, win_length=320, hop_length=160, n_fft=320, center=False)
    with torch.inference_mode():
        f, _ = fe(torch.from_numpy(x)[None], torch.tensor([len(x)]))
    return f[0].numpy()


def stats(a: np.ndarray, b: np.ndarray) -> str:
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return f"max {d.max():.2e}  mean {d.mean():.2e}  p99.9 {np.quantile(d, 0.999):.2e}"


def main() -> None:
    out, wavs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    bench = GigaAMLogMel()
    worst: dict[str, float] = {}
    for wav in wavs:
        x = load_audio(wav)
        rust = np.load(out / f"{wav.stem}.npy")
        refs = {"bench": bench(x)[0], "gigaam": gigaam_logmel(x), "float64": float64_logmel(x)}
        print(f"{wav.name}: {rust.shape[1]} frames")
        for name, ref in refs.items():
            if ref is None:
                continue
            assert ref.shape == rust.shape, (name, ref.shape, rust.shape)
            print(f"  rust vs {name:8s} {stats(rust, ref)}")
            worst[name] = max(worst.get(name, 0.0), float(np.abs(rust - ref).max()))
        print(f"  bench vs gigaam   {stats(refs['bench'], refs['gigaam'])}" if refs["gigaam"] is not None else "")
    print("worst max |diff| over all clips:", {k: f"{v:.2e}" for k, v in worst.items()})


if __name__ == "__main__":
    main()
