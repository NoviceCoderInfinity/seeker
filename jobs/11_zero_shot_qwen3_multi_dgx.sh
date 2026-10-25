#!/bin/bash
#SBATCH --job-name=vidcurator-zs-qwen3-multi
#SBATCH --output=/home/vis-comp/22b3929/vidcurator/logs/zs_qwen3_multi_%j.out
#SBATCH --error=/home/vis-comp/22b3929/vidcurator/logs/zs_qwen3_multi_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16

set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator || { echo "ERROR: cannot cd to vidcurator"; exit 1; }
mkdir -p logs data/frames data/indices runs

export HF_HOME="/home/vis-comp/22b3929/vidcurator/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

PY="${VIDCURATOR_PY:-/home/vis-comp/22b3929/miniconda3/envs/vidcurator/bin/python}"
if [[ ! -f "$PY" ]]; then
    echo "ERROR: python not found at $PY" >&2
    ls -la /home/vis-comp/22b3929/miniconda3/envs/vidcurator/bin/ >&2 || true
    exit 1
fi

echo "=== [$SLURM_JOB_ID] Qwen3-VL multi-dataset zero-shot started at $(date) ==="
echo "Node: $(hostname)"
echo "Python: $PY"
$PY --version
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"

FFMPEG="$($PY -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null || echo '')"
echo "ffmpeg: ${FFMPEG:-not found}"

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
        --batch-size 128 --device cuda

    echo "  $dataset_name: $(wc -l < "$frames_index") frames indexed"
}

run_eval() {
    local config_path="$1"
    local run_dir="$2"
    local qa_path="$3"
    local frames_path="$4"
    local model_path="$5"
    local port="$6"
    local config_name=$(basename "$config_path" .json)

    # Skip if already done
    if [[ -f "$run_dir/summary.json" ]]; then
        echo "--- $config_name: already done, skipping ---"
        return 0
    fi

    echo "--- Serving $model_path on port $port for $config_name ---"
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
        > "logs/vlm_${SLURM_JOB_ID}_${config_name}.log" 2>&1 &
    SERVER_PID=$!

    echo "Waiting for VLM..."
    ready=0
    for attempt in $(seq 1 300); do
        if $PY -c "
import json, urllib.request
url = 'http://127.0.0.1:${port}/v1/models'
with urllib.request.urlopen(url, timeout=5) as r:
    data = json.loads(r.read().decode())
    print(data.get('data', [{}])[0].get('id', 'unknown'))
" 2>/dev/null; then
            ready=1; echo "VLM ready after $attempt attempts ($(($attempt * 5))s)"; break
        fi
        sleep 5
    done
    if [[ "$ready" -ne 1 ]]; then
        echo "ERROR: VLM failed for $config_name"; tail -30 "logs/vlm_${SLURM_JOB_ID}_${config_name}.log" >&2
        return 1
    fi

    echo "--- Evaluating $config_name ---"
    $PY -m vidcurator.run evaluate --config "$config_path" \
        2>&1 | tee "logs/eval_${config_name}_${SLURM_JOB_ID}.log"

    echo "--- Exporting artifacts for $config_name ---"
    $PY scripts/export_paper_artifacts.py --run-dir "$run_dir" || true
    $PY scripts/export_qualitative_traces.py \
        --run-dir "$run_dir" \
        --qa-path "$qa_path" \
        --frames-path "$frames_path" \
        --policy scripted_curator --max-examples 5 || true

    echo "--- Results for $config_name ---"
    $PY -c "
import json
s = json.load(open('$run_dir/summary.json'))
for key in sorted(s['groups']):
    g = s['groups'][key]
    print(f'  {key}: acc={g[\"accuracy\"]:.3f} frames={g[\"avg_frames\"]:.1f} fep={g.get(\"fep\",\"n/a\")}')
print('CE-AUC:', {k: round(v,3) if v else None for k,v in sorted(s.get('context_efficiency',{}).items())})
print('SF-delta:', {k: round(v,3) for k,v in sorted(s.get('sf_delta',{}).items())})
" || true

    echo "--- Killing VLM ---"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    trap - EXIT
    sleep 5
}

# ─── Prepare all datasets (reuse if already done by L40 jobs) ────────────────
echo "=== Preparing datasets ==="
prepare_dataset data/processed/videomme_long_150.jsonl data/videos/videomme data/frames/videomme_long_150 videomme_long 0.1
prepare_dataset data/processed/nextqa_temporal_val_150.jsonl data/videos/nextqa/NExTVideo data/frames/nextqa_temporal_150 nextqa_temporal 1.0
prepare_dataset data/processed/egoschema_val_150.jsonl data/videos/egoschema data/frames/egoschema_val_150 egoschema 0.5

# ─── Run Qwen3-VL-8B on all datasets ─────────────────────────────────────────
echo "=== Qwen3-VL-8B on Video-MME ==="
run_eval configs/gpu_videomme_qwen3_zero_shot.json runs/gpu_videomme_qwen3_zero_shot \
    data/processed/videomme_long_150.jsonl data/indices/videomme_long_siglip_frames.jsonl \
    models/qwen3_vl_8b_instruct 8001

echo "=== Qwen3-VL-8B on NExTQA ==="
run_eval configs/gpu_nextqa_qwen3_zero_shot.json runs/gpu_nextqa_qwen3_zero_shot \
    data/processed/nextqa_temporal_val_150.jsonl data/indices/nextqa_temporal_siglip_frames.jsonl \
    models/qwen3_vl_8b_instruct 8001

echo "=== Qwen3-VL-8B on EgoSchema ==="
run_eval configs/gpu_egoschema_qwen3_zero_shot.json runs/gpu_egoschema_qwen3_zero_shot \
    data/processed/egoschema_val_150.jsonl data/indices/egoschema_siglip_frames.jsonl \
    models/qwen3_vl_8b_instruct 8001

echo "=== [$SLURM_JOB_ID] Qwen3-VL multi-dataset finished at $(date) ==="
