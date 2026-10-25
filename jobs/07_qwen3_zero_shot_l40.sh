#!/bin/bash
#SBATCH --job-name=vidcurator-e2e-qwen3
#SBATCH --output=logs/e2e_qwen3_%j.out
#SBATCH --error=logs/e2e_qwen3_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/videos/nextgqa data/frames/nextgqa_val_150 data/indices models runs/zero_shot

export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_TOKEN="${HF_TOKEN:-}"

echo "=== [$SLURM_JOB_ID] end-to-end Qwen3-VL zero-shot started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"

# ─── Step 1: Environment setup ───────────────────────────────────────────────
echo "--- Step 1: Environment setup ---"

if [[ -f .venv/bin/python ]]; then
    PY=".venv/bin/python"
    echo "Using .venv: $PY"
    $PY --version
else
    echo "Creating .venv..."
    python3 -m venv .venv
    PY=".venv/bin/python"
    $PY -m pip install --upgrade pip
fi

if ! $PY -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "Installing GPU deps (this may take ~10 min)..."
    $PY -m pip install --upgrade pip
    $PY -m pip install "torch>=2.4.0" --index-url https://download.pytorch.org/whl/cu121 || \
    $PY -m pip install "torch>=2.4.0"
    $PY -m pip install \
        "transformers>=4.57.0" \
        "accelerate>=1.0.0" \
        "qwen-vl-utils>=0.0.8" \
        "sentence-transformers>=3.0.0" \
        "pillow>=10.0.0" \
        "huggingface-hub>=0.26.0" \
        "imageio-ffmpeg>=0.5.1" \
        "decord>=0.6.0"
    $PY -m pip install "vllm>=0.6.0" || echo "WARNING: vllm install failed; will try transformers fallback"
    $PY -m pip install -e .
fi

FFMPEG="$($PY -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null || echo '')"
if [[ -z "$FFMPEG" ]]; then
    FFMPEG=$(which ffmpeg 2>/dev/null || echo '')
fi
echo "ffmpeg: ${FFMPEG:-not found}"

# ─── Step 2: Download model weights ──────────────────────────────────────────
echo "--- Step 2: Download Qwen3-VL-8B-Instruct weights ---"
MODEL_PATH="models/qwen3_vl_8b_instruct"
if [[ ! -f "$(ls $MODEL_PATH/*.safetensors 2>/dev/null | head -1)" ]]; then
    echo "Downloading Qwen3-VL-8B-Instruct weights (~16 GB)..."
    $PY -c "
from huggingface_hub import snapshot_download
import os
token = os.environ.get('HF_TOKEN') or None
snapshot_download(
    repo_id='Qwen/Qwen3-VL-8B-Instruct',
    repo_type='model',
    local_dir='$MODEL_PATH',
    token=token,
)
print('Model weights downloaded')
"
else
    echo "Model weights already present"
fi
ls -lh $MODEL_PATH/*.safetensors 2>/dev/null | head -5

# ─── Step 3: Download NExT-GQA videos for 150 split (if not already present) ─
echo "--- Step 3: Check/download NExT-GQA videos ---"
VIDEO_COUNT=$(ls data/videos/nextgqa/*.mp4 2>/dev/null | wc -l)
if [[ "$VIDEO_COUNT" -lt 100 ]]; then
    $PY -c "
import json, os
from huggingface_hub import hf_hub_download

rows = [json.loads(l) for l in open('data/processed/nextgqa_val_150.jsonl') if l.strip()]
vids = sorted(set(r['video_id'] for r in rows))
print(f'Downloading {len(vids)} videos...')

token = os.environ.get('HF_TOKEN') or None
dest = 'data/videos/nextgqa'
os.makedirs(dest, exist_ok=True)

ok = 0
fail = 0
for i, vid in enumerate(vids, 1):
    local_path = os.path.join(dest, f'{vid}.mp4')
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        ok += 1
        continue
    repo_path = f'NExTVideo/{vid[:4]}/{vid}.mp4'
    try:
        p = hf_hub_download(
            repo_id='jinyoungkim/NExT-GQA',
            repo_type='dataset',
            filename=repo_path,
            local_dir='data/raw/nextgqa',
            token=token,
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path):
            os.symlink(os.path.abspath(p), local_path)
        ok += 1
    except Exception as e:
        print(f'  FAIL [{i}/{len(vids)}] {vid}: {repr(e)[:100]}')
        fail += 1
    if i % 20 == 0:
        print(f'  progress: {i}/{len(vids)} (ok={ok}, fail={fail})')

print(f'Done: {ok} ok, {fail} failed out of {len(vids)}')
"
else
    echo "Already have $VIDEO_COUNT videos, skipping download"
fi

# ─── Step 4: Decode frames + build index + SigLIP embeddings ─────────────────
echo "--- Step 4: Decode frames + build index + SigLIP embeddings ---"

FRAMES_INDEX="data/indices/nextgqa_siglip_frames.jsonl"
if [[ ! -f "$FRAMES_INDEX" || $(wc -l < "$FRAMES_INDEX") -lt 100 ]]; then
    $PY scripts/decode_split_frames.py \
        --qa-path data/processed/nextgqa_val_150.jsonl \
        --video-root data/videos/nextgqa \
        --out-root data/frames/nextgqa_val_150 \
        --fps 1.0 \
        --ffmpeg "$FFMPEG" \
        --manifest runs/decode_nextgqa_val_150_manifest.json

    $PY scripts/build_frame_index.py \
        --frames-root data/frames/nextgqa_val_150 \
        --output data/indices/nextgqa_val_150_decoded_frames.jsonl \
        --fps 1.0 \
        --dataset nextgqa

    $PY scripts/enrich_frame_index.py \
        --input data/indices/nextgqa_val_150_decoded_frames.jsonl \
        --output "$FRAMES_INDEX" \
        --model google/siglip-base-patch16-224 \
        --batch-size 64 \
        --device cuda
else
    echo "Frame index already present ($(wc -l < "$FRAMES_INDEX") rows), skipping decode+index"
fi

FRAMES_COUNT=$(wc -l < "$FRAMES_INDEX")
echo "  Total frames indexed: $FRAMES_COUNT"

# ─── Step 5: Serve Qwen3-VL-8B-Instruct via vLLM ─────────────────────────────
echo "--- Step 5: Serve Qwen3-VL-8B-Instruct via vLLM ---"
PORT=8001

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "killing vLLM server (pid=$SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES=0 \
    $PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-model-len 32768 \
    > "logs/vllm_qwen3_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
echo "vLLM server pid: $SERVER_PID"

echo "Waiting for server..."
ready=0
for attempt in $(seq 1 180); do
    if $PY -c "
import json, urllib.request
url = 'http://127.0.0.1:${PORT}/v1/models'
with urllib.request.urlopen(url, timeout=5) as r:
    data = json.loads(r.read().decode())
    print(data.get('data', [{}])[0].get('id', 'unknown'))
" 2>/dev/null; then
        ready=1
        echo "vLLM ready after $attempt attempts"
        break
    fi
    sleep 5
done

if [[ "$ready" -ne 1 ]]; then
    echo "ERROR: vLLM failed to start" >&2
    tail -50 "logs/vllm_qwen3_${SLURM_JOB_ID}.log" >&2
    exit 1
fi

# ─── Step 6: Run zero-shot evaluation ────────────────────────────────────────
echo "--- Step 6: Run zero-shot evaluation on NExT-GQA 150 ---"

$PY -m vidcurator.run evaluate \
    --config configs/gpu_nextgqa_qwen3_zero_shot.json \
    2>&1 | tee "logs/eval_nextgqa_qwen3_${SLURM_JOB_ID}.log"

# ─── Step 7: Export paper artifacts ──────────────────────────────────────────
echo "--- Step 7: Export paper artifacts ---"

$PY scripts/export_paper_artifacts.py \
    --run-dir runs/gpu_nextgqa_qwen3_zero_shot

$PY scripts/export_qualitative_traces.py \
    --run-dir runs/gpu_nextgqa_qwen3_zero_shot \
    --qa-path data/processed/nextgqa_val_150.jsonl \
    --frames-path data/indices/nextgqa_siglip_frames.jsonl \
    --policy scripted_curator \
    --max-examples 10

echo "--- Results summary ---"
$PY -c "
import json
s = json.load(open('runs/gpu_nextgqa_qwen3_zero_shot/summary.json'))
for key in sorted(s['groups']):
    g = s['groups'][key]
    print(f'  {key}: acc={g[\"accuracy\"]:.3f} frames={g[\"avg_frames\"]:.1f} fep={g.get(\"fep\",\"n/a\")} fer={g.get(\"fer\",\"n/a\")} if={g.get(\"if\",\"n/a\")}')
print()
print('Context efficiency:')
for p, v in sorted(s.get('context_efficiency', {}).items()):
    print(f'  {p}: {v}')
print()
print('SF-delta:')
for k, v in sorted(s.get('sf_delta', {}).items()):
    print(f'  {k}: {v:.3f}')
"

echo "=== [$SLURM_JOB_ID] end-to-end Qwen3-VL zero-shot finished at $(date) ==="
