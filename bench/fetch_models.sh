#!/usr/bin/env bash
# Download the ONNX / vosk models used by run_bench.py into bench/models/ and verify sha256.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models

fetch() { # url dest sha256
  local url=$1 dest=$2 sha=$3
  if [[ ! -f $dest ]]; then
    echo "-> $dest"
    mkdir -p "$(dirname "$dest")"
    curl -fL --retry 3 -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
  fi
  echo "$sha  $dest" | sha256sum -c --quiet
}

GH=https://github.com/fgeeer77/bayaya-models/releases/download
# GigaAM-Multilingual CTC 220M, sherpa-onnx int8 (bit-identical to export_onnx.py + quantize_dynamic)
fetch $GH/gigaam-multilingual-ctc/model.int8.onnx models/sherpa-gigaam-ml-ctc/model.int8.onnx \
  f66bff0186d649a2300da895e9f81d4ef8764519db2dc46429a44b883e90d105
fetch $GH/gigaam-multilingual-ctc/tokens.txt models/sherpa-gigaam-ml-ctc/tokens.txt \
  9b5df7987cb4ca52c1a468649ce897fab1cd182067416e29fef49dfaa7a856c2
# GigaAM-Multilingual large CTC 600M, sherpa-onnx int8
fetch $GH/gigaam-multilingual-large-ctc/model.int8.onnx models/sherpa-gigaam-ml-large-ctc/model.int8.onnx \
  7fdb9427c1c871407ecbde741fd7bb0479924981c89aa9f4241587bbcb085ae3
fetch $GH/gigaam-multilingual-large-ctc/tokens.txt models/sherpa-gigaam-ml-large-ctc/tokens.txt \
  9b5df7987cb4ca52c1a468649ce897fab1cd182067416e29fef49dfaa7a856c2

HF=https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16/resolve/main
fetch $HF/model.int8.onnx models/sherpa-gigaam-v3-ru-ctc/model.int8.onnx \
  f86ebfa0429ced91be6054fc344827e9c6c2572f3c318416cd974b06f66437ec
fetch $HF/tokens.txt models/sherpa-gigaam-v3-ru-ctc/tokens.txt \
  17cc514451bcceac9c280068c71502f8448f99e9fb1456b8d0761651fd0392f2

if [[ ! -d models/vosk-model-small-kz-0.42 ]]; then
  curl -fL -o models/vosk.zip https://alphacephei.com/vosk/models/vosk-model-small-kz-0.42.zip
  (cd models && unzip -q vosk.zip && rm vosk.zip)
fi
echo "models ok"
