#!/bin/bash
#SBATCH --job-name=vidcurator-zs-qwen3
#SBATCH --output=logs/zero_shot_qwen3_%j.out
#SBATCH --error=logs/zero_shot_qwen3_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs/zero_shot

if [[ -d "$HOME/miniconda3/envs/vidcurator" ]]; then
    export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
elif [[ -f .conda-env/bin/activate ]]; then
    source .conda-env/bin/activate
elif [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
else
    echo "ERROR: no Python environment found" >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL_PATH="${MODEL_PATH:-models/qwen3_vl_8b_instruct}"
PORT="${PORT:-8001}"
MATRIX_PATH="${MATRIX_PATH:-configs/zero_shot_matrix.json}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

echo "=== [$SLURM_JOB_ID] zero-shot Qwen3-VL started at $(date) ==="
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: model directory not found: $MODEL_PATH" >&2
    exit 1
fi

if [[ ! -f "$(ls "$MODEL_PATH"/*.safetensors 2>/dev/null | head -1)" ]]; then
    echo "ERROR: no safetensors weights in $MODEL_PATH" >&2
    exit 1
fi

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "killing vLLM server (pid=$SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "--- starting vLLM server ---"
CUDA_VISIBLE_DEVICES=0 \
    python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    > "logs/vllm_qwen3_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
echo "vLLM server pid: $SERVER_PID"

echo "--- waiting for server to be ready ---"
ready=0
for attempt in $(seq 1 120); do
    if python -c "
import json, urllib.request, sys
url = 'http://127.0.0.1:${PORT}/v1/models'
with urllib.request.urlopen(url, timeout=5) as r:
    data = json.loads(r.read().decode())
    print(data.get('data', [{}])[0].get('id', 'unknown'))
" 2>/dev/null; then
        ready=1
        echo "vLLM server ready after $attempt attempts"
        break
    fi
    sleep 5
done

if [[ "$ready" -ne 1 ]]; then
    echo "ERROR: vLLM server did not start within timeout" >&2
    tail -50 "logs/vllm_qwen3_${SLURM_JOB_ID}.log" >&2
    exit 1
fi

echo "--- running zero-shot matrix (Qwen3-VL only) ---"
python scripts/run_zero_shot_matrix.py \
    --matrix "$MATRIX_PATH" \
    --only-answerer qwen3_vl_8b_instruct \
    --allow-missing

echo "--- exporting paper artifacts ---"
for run_dir in runs/zero_shot/*__qwen3_vl_8b_instruct; do
    if [[ -d "$run_dir" ]]; then
        python scripts/export_paper_artifacts.py --run-dir "$run_dir" || true
    fi
done

echo "=== [$SLURM_JOB_ID] zero-shot Qwen3-VL finished at $(date) ==="
