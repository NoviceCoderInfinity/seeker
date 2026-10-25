#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: scripts/serve_vllm_qwen.sh <model-id-or-path> <port>" >&2
  exit 2
fi

model="$1"
port="$2"

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

python -m vllm.entrypoints.openai.api_server \
  --model "$model" \
  --port "$port" \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 32768
