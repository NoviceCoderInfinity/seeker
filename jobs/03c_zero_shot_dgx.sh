#!/bin/bash
#SBATCH --job-name=vidcurator-zs-dgx
#SBATCH --output=logs/zero_shot_dgx_%j.out
#SBATCH --error=logs/zero_shot_dgx_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

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

QWEN2_MODEL="${QWEN2_MODEL:-models/qwen2_5_vl_7b}"
QWEN3_MODEL="${QWEN3_MODEL:-models/qwen3_vl_8b_instruct}"
QWEN2_PORT="${QWEN2_PORT:-8000}"
QWEN3_PORT="${QWEN3_PORT:-8001}"
MATRIX_PATH="${MATRIX_PATH:-configs/zero_shot_matrix.json}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

echo "=== [$SLURM_JOB_ID] zero-shot DGX dual-model started at $(date) ==="
echo "Qwen2.5-VL: $QWEN2_MODEL on port $QWEN2_PORT (GPU 0)"
echo "Qwen3-VL:   $QWEN3_MODEL on port $QWEN3_PORT (GPU 1)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"

for model_path in "$QWEN2_MODEL" "$QWEN3_MODEL"; do
    if [[ ! -d "$model_path" ]]; then
        echo "ERROR: model directory not found: $model_path" >&2
        exit 1
    fi
    if [[ ! -f "$(ls "$model_path"/*.safetensors 2>/dev/null | head -1)" ]]; then
        echo "ERROR: no safetensors weights in $model_path" >&2
        exit 1
    fi
done

server_pids=()
cleanup() {
    echo "cleaning up vLLM servers"
    for pid in "${server_pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

start_server() {
    local gpu_id="$1"
    local port="$2"
    local model_path="$3"
    local log_name="$4"

    CUDA_VISIBLE_DEVICES="$gpu_id" \
        python -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --port "$port" \
        --dtype bfloat16 \
        --trust-remote-code \
        --max-model-len "$MAX_MODEL_LEN" \
        > "logs/vllm_${log_name}_${SLURM_JOB_ID}.log" 2>&1 &
    local pid=$!
    echo "started $log_name on GPU $gpu_id port $port (pid=$pid)"
    echo "$pid"
}

wait_for_server() {
    local port="$1"
    local label="$2"
    local attempt
    for attempt in $(seq 1 120); do
        if python -c "
import json, urllib.request
url = 'http://127.0.0.1:${port}/v1/models'
with urllib.request.urlopen(url, timeout=5) as r:
    data = json.loads(r.read().decode())
    print(data.get('data', [{}])[0].get('id', 'unknown'))
" 2>/dev/null; then
            echo "$label ready on port $port after $attempt attempts"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: $label did not start within timeout" >&2
    return 1
}

echo "--- starting both vLLM servers ---"
server_pids+=("$(start_server 0 "$QWEN2_PORT" "$QWEN2_MODEL" "qwen25")")
server_pids+=("$(start_server 1 "$QWEN3_PORT" "$QWEN3_MODEL" "qwen3")")

wait_for_server "$QWEN2_PORT" "Qwen2.5-VL" || exit 1
wait_for_server "$QWEN3_PORT" "Qwen3-VL" || exit 1

echo "--- running full zero-shot matrix (both models) ---"
python scripts/run_zero_shot_matrix.py \
    --matrix "$MATRIX_PATH" \
    --allow-missing

echo "--- exporting paper artifacts for all runs ---"
for run_dir in runs/zero_shot/*__qwen2_5_vl_7b runs/zero_shot/*__qwen3_vl_8b_instruct; do
    if [[ -d "$run_dir" ]]; then
        echo "  exporting $run_dir"
        python scripts/export_paper_artifacts.py --run-dir "$run_dir" || true
    fi
done

echo "--- exporting qualitative traces ---"
for ds in fixture nextgqa_timestamp_150; do
    run_dir="runs/zero_shot/${ds}__qwen2_5_vl_7b"
    if [[ -d "$run_dir" ]]; then
        python scripts/export_qualitative_traces.py \
            --run-dir "$run_dir" \
            --qa-path "data/processed/${ds/_150/}_150.jsonl" \
            --frames-path "data/indices/${ds}_frames.jsonl" \
            --policy scripted_curator \
            --max-examples 5 || true
    fi
done

echo "=== [$SLURM_JOB_ID] zero-shot DGX dual-model finished at $(date) ==="
