#!/usr/bin/env python3
"""Why does GigaAM-Multilingual int8 degrade under sherpa-onnx but not under plain onnxruntime?

Runs the same int8 ONNX model with different front-ends on a set and reports WER/CER and the
feature distance to GigaAM's own log-mel (torchaudio, 20 ms window, n_fft 320):

  gigaam      exact numpy port of gigaam.preprocess.FeatureExtractor (run_bench.GigaAMLogMel)
  knf-25ms    kaldi-native-fbank configured the way sherpa-onnx 1.13.8 configures GigaAM
              (offline-recognizer-ctc-impl.h: hann, no preemph/dc, 0-8000 Hz, 64 bins,
              round_to_power_of_two=false, default frame_length_ms=25 -> n_fft 400)
  knf-20ms    same but frame_length_ms=20 (what GigaAM v3 / multilingual actually use)
  sherpa      sherpa_onnx.OfflineRecognizer.from_nemo_ctc (end-to-end, for reference)

Usage: uv run diagnose_features.py [--set codeswitch] [--model-dir sherpa-gigaam-ml-ctc]
"""
from __future__ import annotations

import argparse
import json

import jiwer
import kaldi_native_fbank as knf
import numpy as np

from run_bench import MODELS, MODELS_DIR, GigaAMLogMel, OrtCTC, SherpaCTC, Spec, load_audio, load_set, normalize


def knf_feats(audio: np.ndarray, frame_ms: float) -> np.ndarray:
    o = knf.FbankOptions()
    o.frame_opts.dither = 0
    o.frame_opts.remove_dc_offset = False
    o.frame_opts.preemph_coeff = 0
    o.frame_opts.window_type = "hann"
    o.frame_opts.round_to_power_of_two = False
    o.frame_opts.snip_edges = False  # sherpa-onnx default
    o.frame_opts.frame_length_ms = frame_ms
    o.mel_opts.low_freq = 0
    o.mel_opts.high_freq = 8000
    o.mel_opts.num_bins = 64
    fb = knf.OnlineFbank(o)
    fb.accept_waveform(16000, audio.tolist())
    fb.input_finished()
    f = np.stack([np.array(fb.get_frame(i)) for i in range(fb.num_frames_ready)])
    return f.T[None].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="codeswitch")
    ap.add_argument("--model-dir", default="sherpa-gigaam-ml-ctc")
    a = ap.parse_args()

    spec = Spec("ort_ctc", "", [], opts={"dir": a.model_dir})
    ort_b = OrtCTC(spec, 4)
    sherpa_b = SherpaCTC(Spec("sherpa_ctc", "", [], opts={"dir": a.model_dir}), 4)
    ref_feat = GigaAMLogMel()

    def decode(f: np.ndarray) -> str:
        lp, ln = ort_b.sess.run(None, {"features": f, "feature_lengths": np.array([f.shape[-1]], dtype=np.int64)})
        ids = lp[0, : int(ln[0])].argmax(-1)
        out, prev = [], -1
        for i in ids:
            if i != prev and i != ort_b.blank:
                out.append(ort_b.vocab[int(i)])
            prev = i
        return "".join(out)

    fronts = {
        "gigaam": ref_feat,
        "knf-25ms": lambda x: knf_feats(x, 25.0),
        "knf-20ms": lambda x: knf_feats(x, 20.0),
    }
    refs, hyps, dist = [], {k: [] for k in [*fronts, "sherpa"]}, {k: [] for k in fronts}
    for it in load_set(a.set):
        audio = load_audio(it["path"])
        refs.append(normalize(it["text"]))
        g = ref_feat(audio)
        for k, fn in fronts.items():
            f = fn(audio)
            t = min(f.shape[-1], g.shape[-1])
            dist[k].append(float(np.abs(f[..., :t] - g[..., :t]).mean()))
            hyps[k].append(normalize(decode(f)))
        hyps["sherpa"].append(normalize(sherpa_b.transcribe(audio)))
    res = {}
    for k, h in hyps.items():
        res[k] = {"wer": round(100 * jiwer.wer(refs, h), 2), "cer": round(100 * jiwer.cer(refs, h), 2),
                  "mean_abs_feat_diff_vs_gigaam": round(float(np.mean(dist[k])), 3) if k in dist else None}
    print(json.dumps({"set": a.set, "model": a.model_dir, "results": res}, indent=1))


if __name__ == "__main__":
    main()
