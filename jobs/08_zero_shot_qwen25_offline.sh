#!/bin/bash
#SBATCH --job-name=vidcurator-zs-qwen25
#SBATCH --output=logs/zs_qwen25_%j.out
#SBATCH --error=logs/zs_qwen25_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/frames/nextgqa_val_150 data/indices runs/gpu_nextgqa_qwen2_5_zero_shot

export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "=== [$SLURM_JOB_ID] zero-shot Qwen2.5-VL on NExT-GQA started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"

PY="${VIDCURATOR_PY:-$HOME/miniconda3/envs/vidcurator/bin/python}"

echo "--- verifying environment ---"
$PY -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

FFMPEG="$($PY -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null || echo '')"
echo "ffmpeg: ${FFMPEG:-not found}"

# Step 1: Decode frames + build index + SigLIP embeddings
echo "--- Step 1: Decode frames + SigLIP embeddings ---"
FRAMES_INDEX="data/indices/nextgqa_siglip_frames.jsonl"
if [[ ! -f "$FRAMES_INDEX" || $(wc -l < "$FRAMES_INDEX") -lt 100 ]]; then
    echo "  decoding frames at 1 fps..."
    $PY scripts/decode_split_frames.py \
        --qa-path data/processed/nextgqa_val_150.jsonl \
        --video-root data/videos/nextgqa \
        --out-root data/frames/nextgqa_val_150 \
        --fps 1.0 \
        --ffmpeg "$FFMPEG" \
        --manifest runs/decode_nextgqa_val_150_manifest.json

    echo "  building frame index..."
    $PY scripts/build_frame_index.py \
        --frames-root data/frames/nextgqa_val_150 \
        --output data/indices/nextgqa_val_150_decoded_frames.jsonl \
        --fps 1.0 \
        --dataset nextgqa

    echo "  enriching with SigLIP embeddings..."
    $PY scripts/enrich_frame_index.py \
        --input data/indices/nextgqa_val_150_decoded_frames.jsonl \
        --output "$FRAMES_INDEX" \
        --model models/siglip_base \
        --batch-size 64 \
        --device cuda
else
    echo "  frame index already present ($(wc -l < "$FRAMES_INDEX") rows)"
fi
echo "  Total frames: $(wc -l < "$FRAMES_INDEX")"

# Step 2: Serve Qwen2.5-VL-7B via transformers
echo "--- Step 2: Serve Qwen2.5-VL-7B ---"
PORT=8000
MODEL_PATH="models/qwen2_5_vl_7b"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "killing VLM server (pid=$SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES=0 \
    $PY scripts/serve_transformers_vlm.py \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --dtype bfloat16 \
    > "logs/vlm_server_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
echo "VLM server pid: $SERVER_PID"

echo "Waiting for server..."
ready=0
for attempt in $(seq 1 240); do
    if $PY -c "
import json, urllib.request
url = 'http://127.0.0.1:${PORT}/v1/models'
with urllib.request.urlopen(url, timeout=5) as r:
    data = json.loads(r.read().decode())
    print(data.get('data', [{}])[0].get('id', 'unknown'))
" 2>/dev/null; then
        ready=1
        echo "VLM ready after $attempt attempts ($(($attempt * 5))s)"
        break
    fi
    sleep 5
done

if [[ "$ready" -ne 1 ]]; then
    echo "ERROR: VLM failed to start" >&2
    tail -50 "logs/vlm_server_${SLURM_JOB_ID}.log" >&2
    exit 1
fi

# Step 3: Run zero-shot evaluation
echo "--- Step 3: Zero-shot evaluation on NExT-GQA 150 ---"
$PY -m vidcurator.run evaluate \
    --config configs/gpu_nextgqa_qwen2_5_zero_shot.json \
    2>&1 | tee "logs/eval_nextgqa_qwen25_${SLURM_JOB_ID}.log"

# Step 4: Export artifacts
echo "--- Step 4: Export paper artifacts ---"
$PY scripts/export_paper_artifacts.py --run-dir runs/gpu_nextgqa_qwen2_5_zero_shot
$PY scripts/export_qualitative_traces.py \
    --run-dir runs/gpu_nextgqa_qwen2_5_zero_shot \
    --qa-path data/processed/nextgqa_val_150.jsonl \
    --frames-path data/indices/nextgqa_siglip_frames.jsonl \
    --policy scripted_curator --max-examples 10

echo "--- Results summary ---"
$PY -c "
import json
s = json.load(open('runs/gpu_nextgqa_qwen2_5_zero_shot/summary.json'))
for key in sorted(s['groups']):
    g = s['groups'][key]
    print(f'  {key}: acc={g[\"accuracy\"]:.3f} frames={g[\"avg_frames\"]:.1f} fep={g.get(\"fep\",\"n/a\")} fer={g.get(\"fer\",\"n/a\")}')
print()
print('CE-AUC:', {k: round(v,3) if v else None for k,v in sorted(s.get('context_efficiency',{}).items())})
print('SF-delta:', {k: round(v,3) for k,v in sorted(s.get('sf_delta',{}).items())})
"

echo "=== [$SLURM_JOB_ID] finished at $(date) ==="
