#!/bin/bash
#SBATCH --job-name=vidcurator-zs-multi
#SBATCH --output=logs/zs_multi_%j.out
#SBATCH --error=logs/zs_multi_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/frames data/indices runs

export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

PY="${VIDCURATOR_PY:-$HOME/miniconda3/envs/vidcurator/bin/python}"

echo "=== [$SLURM_JOB_ID] multi-dataset zero-shot started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"

FFMPEG="$($PY -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null || echo '')"
echo "ffmpeg: ${FFMPEG:-not found}"

# Function: decode + index + SigLIP for a dataset
prepare_dataset() {
    local qa_path="$1"
    local video_root="$2"
    local frames_root="$3"
    local dataset_name="$4"
    local fps="$5"
    local frames_index="data/indices/${dataset_name}_siglip_frames.jsonl"

    if [[ -f "$frames_index" && $(wc -l < "$frames_index") -gt 100 ]]; then
        echo "  $dataset_name: frame index ready ($(wc -l < "$frames_index") rows)"
        return 0
    fi

    echo "  $dataset_name: decoding at ${fps} fps..."
    $PY scripts/decode_split_frames.py \
        --qa-path "$qa_path" \
        --video-root "$video_root" \
        --out-root "$frames_root" \
        --fps "$fps" \
        --ffmpeg "$FFMPEG" \
        --manifest "runs/decode_${dataset_name}_manifest.json"

    $PY scripts/build_frame_index.py \
        --frames-root "$frames_root" \
        --output "data/indices/${dataset_name}_decoded_frames.jsonl" \
        --fps "$fps" --dataset "$dataset_name"

    echo "  $dataset_name: SigLIP embeddings..."
    $PY scripts/enrich_frame_index.py \
        --input "data/indices/${dataset_name}_decoded_frames.jsonl" \
        --output "$frames_index" \
        --model models/siglip_base \
        --batch-size 64 --device cuda

    echo "  $dataset_name: $(wc -l < "$frames_index") frames indexed"
}

# Function: serve VLM + eval + export
run_eval() {
    local config_path="$1"
    local run_dir="$2"
    local qa_path="$3"
    local frames_path="$4"
    local model_path="$5"
    local port="$6"

    echo "--- Serving $model_path on port $port ---"
    cleanup() {
        if [[ -n "${SERVER_PID:-}" ]]; then
            kill "$SERVER_PID" 2>/dev/null || true
            wait "$SERVER_PID" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT

    CUDA_VISIBLE_DEVICES=0 \
        $PY scripts/serve_transformers_vlm.py \
        --model "$model_path" --port "$port" --dtype bfloat16 \
        > "logs/vlm_${SLURM_JOB_ID}_${port}.log" 2>&1 &
    SERVER_PID=$!

    echo "Waiting for VLM..."
    ready=0
    for attempt in $(seq 1 240); do
        if $PY -c "
import json, urllib.request
url = 'http://127.0.0.1:${port}/v1/models'
with urllib.request.urlopen(url, timeout=5) as r:
    data = json.loads(r.read().decode())
    print(data.get('data', [{}])[0].get('id', 'unknown'))
" 2>/dev/null; then
            ready=1; echo "VLM ready after $attempt attempts"; break
        fi
        sleep 5
    done
    if [[ "$ready" -ne 1 ]]; then
        echo "ERROR: VLM failed"; tail -20 "logs/vlm_${SLURM_JOB_ID}_${port}.log" >&2; return 1
    fi

    echo "--- Evaluating $config_path ---"
    $PY -m vidcurator.run evaluate --config "$config_path" \
        2>&1 | tee "logs/eval_$(basename $config_path .json)_${SLURM_JOB_ID}.log"

    echo "--- Exporting artifacts ---"
    $PY scripts/export_paper_artifacts.py --run-dir "$run_dir" || true
    $PY scripts/export_qualitative_traces.py \
        --run-dir "$run_dir" \
        --qa-path "$qa_path" \
        --frames-path "$frames_path" \
        --policy scripted_curator --max-examples 5 || true

    echo "--- Killing VLM ---"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    trap - EXIT
}

# ─── Prepare all datasets ────────────────────────────────────────────────────
echo "=== Preparing datasets ==="
prepare_dataset data/processed/videomme_long_150.jsonl data/videos/videomme data/frames/videomme_long_150 videomme_long 0.1
prepare_dataset data/processed/nextqa_temporal_val_150.jsonl data/videos/nextqa/NExTVideo data/frames/nextqa_temporal_150 nextqa_temporal 1.0
prepare_dataset data/processed/egoschema_val_150.jsonl data/videos/egoschema data/frames/egoschema_val_150 egoschema 0.5

# ─── Run Qwen2.5-VL on all datasets ──────────────────────────────────────────
echo "=== Qwen2.5-VL on Video-MME ==="
run_eval configs/gpu_videomme_qwen2_5_zero_shot.json runs/gpu_videomme_qwen2_5_zero_shot \
    data/processed/videomme_long_150.jsonl data/indices/videomme_long_siglip_frames.jsonl \
    models/qwen2_5_vl_7b 8000

echo "=== Qwen2.5-VL on NExTQA ==="
run_eval configs/gpu_nextqa_qwen2_5_zero_shot.json runs/gpu_nextqa_qwen2_5_zero_shot \
    data/processed/nextqa_temporal_val_150.jsonl data/indices/nextqa_temporal_siglip_frames.jsonl \
    models/qwen2_5_vl_7b 8000

echo "=== Qwen2.5-VL on EgoSchema ==="
run_eval configs/gpu_egoschema_qwen2_5_zero_shot.json runs/gpu_egoschema_qwen2_5_zero_shot \
    data/processed/egoschema_val_150.jsonl data/indices/egoschema_siglip_frames.jsonl \
    models/qwen2_5_vl_7b 8000

echo "=== [$SLURM_JOB_ID] multi-dataset finished at $(date) ==="
