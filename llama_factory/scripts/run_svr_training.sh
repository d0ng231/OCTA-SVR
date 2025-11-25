#!/usr/bin/env bash
set -euo pipefail

# Minimal training launcher for SVR pretraining.
# Assumes LLaMA-Factory is installed (pip install llamafactory) and
# that OCTA-100K-SVR pairs live under data/OCTA-100K-SVR/.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${CONFIG:-${ROOT_DIR}/llama_factory/configs/svr_qwen3vl_full.yaml}"
DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/llama_factory/data}"

echo "Using config:      ${CONFIG}"
echo "Using dataset dir: ${DATASET_DIR}"

llamafactory-cli train "${CONFIG}" \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${ROOT_DIR}/outputs/qwen3vl_svr_full"
