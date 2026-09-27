#!/usr/bin/env python3
"""Build the CTranslate2 model for abilmansplus/whisper-turbo-kaz-rus-v1 (used by run_bench.py).

The HF repo only holds a LoRA adapter (peft) on top of abilmansplus/whisper-turbo-ksc2, so:
  1. download base + adapter (+ tokenizer/preprocessor files of openai/whisper-large-v3-turbo)
  2. merge the adapter into the base weights (peft merge_and_unload)
  3. convert to CTranslate2 (int8 weights) -> models/ct2/whisper-turbo-kaz-rus-v1

Also downloads mobiuslabsgmbh/faster-whisper-large-v3-turbo (stock turbo, already CT2).

Usage: uv run convert_whisper.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
M = ROOT / "models"


def main() -> None:
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration

    hf = M / "hf"
    base_dir = snapshot_download("abilmansplus/whisper-turbo-ksc2", local_dir=hf / "whisper-turbo-ksc2",
                                 allow_patterns=["*.json", "*.txt", "*.safetensors"])
    ad_dir = snapshot_download("abilmansplus/whisper-turbo-kaz-rus-v1", local_dir=hf / "whisper-turbo-kaz-rus-v1",
                               allow_patterns=["adapter_*", "*.json", "*.txt"])
    proc_dir = snapshot_download("openai/whisper-large-v3-turbo", local_dir=hf / "whisper-large-v3-turbo",
                                 allow_patterns=["*.json", "*.txt"])
    snapshot_download("mobiuslabsgmbh/faster-whisper-large-v3-turbo",
                      local_dir=hf / "faster-whisper-large-v3-turbo")

    merged = hf / "whisper-turbo-kaz-rus-v1-merged"
    if not (merged / "model.safetensors").exists():
        print("merging LoRA adapter into base ...")
        base = WhisperForConditionalGeneration.from_pretrained(base_dir, dtype=torch.float32)
        model = PeftModel.from_pretrained(base, ad_dir).merge_and_unload()
        model.save_pretrained(merged)
        # tokenizer / feature extractor: the model card uses openai/whisper-large-v3-turbo's processor
        for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "normalizer.json",
                  "added_tokens.json", "special_tokens_map.json", "preprocessor_config.json"):
            src = Path(proc_dir) / f
            if src.exists():
                shutil.copy(src, merged / f)
        gen = Path(base_dir) / "generation_config.json"
        if gen.exists():
            shutil.copy(gen, merged / "generation_config.json")

    out = M / "ct2" / "whisper-turbo-kaz-rus-v1"
    if not (out / "model.bin").exists():
        import ctranslate2

        print("converting to CTranslate2 int8 ...")
        conv = ctranslate2.converters.TransformersConverter(
            str(merged), copy_files=["tokenizer.json", "preprocessor_config.json"])
        conv.convert(str(out), quantization="int8", force=True)
    print("ok:", out)


if __name__ == "__main__":
    main()
