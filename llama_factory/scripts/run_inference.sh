#!/usr/bin/env bash
set -euo pipefail

# Simple inference helper.
# Set MODEL_PATH to your HF checkpoint (base or fine-tuned).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/images/example.png}"
OUT_PATH="${OUT_PATH:-${ROOT_DIR}/outputs/predictions.jsonl}"

python "${ROOT_DIR}/llama_factory/inference_octa_CoT.py" \
  --model_path "${MODEL_PATH}" \
  --image "${IMAGE_PATH}" \
  --template qwen3_vl \
  --output "${OUT_PATH}"
