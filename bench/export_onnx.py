#!/usr/bin/env python3
"""Export GigaAM-Multilingual CTC models to sherpa-onnx format ourselves (fp32 + int8).

Mirrors csukuangfj's export-onnx-ctc-v3.py (the recipe the community bayaya-models builds use):
  1. gigaam.load_model(<name>).to_onnx()             -> <name>.onnx (fp32, opset 17)
  2. add sherpa-onnx metadata (model_type=EncDecCTCModel, is_giga_am=1, vocab_size, ...)
  3. tokens.txt = model vocabulary, one "<sym> <id>" per line, then "<blk> <N>"
  4. onnxruntime.quantization.quantize_dynamic(weight_type=QUInt8)  -> int8 variant
Outputs: models/self-onnx/<name>/{fp32,int8}/{model.onnx|model.int8.onnx,tokens.txt}

run_bench.py registers these automatically as gigaam-ml-*-onnx-{fp32,int8}-self.

Usage: uv run export_onnx.py [multilingual_ctc] [multilingual_large_ctc] [--no-int8]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
M = ROOT / "models"


def export(name: str, int8: bool) -> None:
    import gigaam
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    base = M / "self-onnx" / name
    fp32 = base / "fp32"
    fp32.mkdir(parents=True, exist_ok=True)
    model = gigaam.load_model(name, fp16_encoder=False, device="cpu", download_root=str(M / "gigaam"))
    vocab = list(model.cfg["decoding"]["vocabulary"])
    tokens = "".join(f"{s} {i}\n" for i, s in enumerate(vocab)) + f"<blk> {len(vocab)}\n"

    onnx_path = fp32 / "model.onnx"
    if not onnx_path.exists():
        tmp = base / "tmp_export"
        model.to_onnx(dir_path=str(tmp), dtype=torch.float32)
        src = tmp / f"{name}.onnx"
        meta = {
            "vocab_size": len(vocab) + 1, "normalize_type": "", "subsampling_factor": 4,
            "model_type": "EncDecCTCModel", "version": "1",
            "model_author": "https://github.com/salute-developers/GigaAM",
            "license": "https://github.com/salute-developers/GigaAM/blob/main/LICENSE",
            "language": "Multilingual", "comment": name, "is_giga_am": 1,
        }
        import onnx

        m = onnx.load(str(src))
        del m.metadata_props[:]
        for k, v in meta.items():
            p = m.metadata_props.add()
            p.key, p.value = k, str(v)
        # the 600M model is > 2 GB in fp32 -> protobuf limit -> keep weights as external data
        if sum(len(t.raw_data) for t in m.graph.initializer) > 1.8 * 2**30:
            onnx.save(m, str(onnx_path), save_as_external_data=True, all_tensors_to_one_file=True,
                      location="model.onnx.data")
        else:
            onnx.save(m, str(onnx_path))
        shutil.rmtree(tmp, ignore_errors=True)
    (fp32 / "tokens.txt").write_text(tokens, encoding="utf-8")
    print("fp32:", onnx_path)

    if int8:
        q = base / "int8"
        q.mkdir(exist_ok=True)
        qpath = q / "model.int8.onnx"
        if not qpath.exists():
            quantize_dynamic(model_input=str(onnx_path), model_output=str(qpath),
                             weight_type=QuantType.QUInt8)
        (q / "tokens.txt").write_text(tokens, encoding="utf-8")
        print("int8:", qpath)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=["multilingual_ctc"])
    ap.add_argument("--no-int8", action="store_true")
    a = ap.parse_args()
    for n in a.names:
        export(n, not a.no_int8)


if __name__ == "__main__":
    main()
