#!/bin/bash
#SBATCH --job-name=vc-decode-full
#SBATCH --output=logs/decode_full_%x_%j.out
#SBATCH --error=logs/decode_full_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

# Decode + SigLIP-index ONE full split. Parameterized via env:
#   QA_PATH, VIDEO_ROOT, OUT_ROOT, DATASET_NAME (index name), FPS
# Writes data/indices/${DATASET_NAME}_siglip_frames.jsonl
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs data/frames data/indices runs

export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
PY="$HOME/miniconda3/envs/vidcurator/bin/python"
FFMPEG="$($PY -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null || echo '')"
export FFMPEG
export HF_HOME="${HF_HOME:-/home/vis-comp/22b3929/vidcurator/.hf_cache}"

QA_PATH="${QA_PATH:?QA_PATH required}"
VIDEO_ROOT="${VIDEO_ROOT:?VIDEO_ROOT required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"
DATASET_NAME="${DATASET_NAME:?DATASET_NAME required}"
FPS="${FPS:-1.0}"
# Use local SigLIP weights — compute nodes have no internet (HF id fails offline).
SIGLIP_MODEL="${SIGLIP_MODEL:-models/siglip_base}"

echo "=== [$SLURM_JOB_ID] decode+index FULL: $DATASET_NAME at $(date) ==="
echo "QA=$QA_PATH ($(wc -l < "$QA_PATH") rows) VIDEO_ROOT=$VIDEO_ROOT FPS=$FPS"
echo "CUDA: $($PY -c 'import torch;print(torch.cuda.is_available())' 2>/dev/null)"
echo "ffmpeg: ${FFMPEG:-not found}"

DECODED_INDEX="data/indices/${DATASET_NAME}_decoded_frames.jsonl"
if [[ -f "$DECODED_INDEX" ]]; then
    echo "--- decoded index exists ($(wc -l < "$DECODED_INDEX") rows) — skipping decode+build ---"
else
    echo "--- decoding $DATASET_NAME ---"
    $PY scripts/decode_split_frames.py \
        --qa-path "$QA_PATH" \
        --video-root "$VIDEO_ROOT" \
        --out-root "$OUT_ROOT" \
        --fps "$FPS" \
        ${FFMPEG:+--ffmpeg "$FFMPEG"} \
        --manifest "runs/decode_${DATASET_NAME}_manifest.json"

    echo "--- building frame index ---"
    $PY scripts/build_frame_index.py \
        --frames-root "$OUT_ROOT" \
        --output "$DECODED_INDEX" \
        --fps "$FPS" \
        --dataset "$DATASET_NAME"
fi

echo "--- enriching with SigLIP embeddings ---"
$PY scripts/enrich_frame_index.py \
    --input "data/indices/${DATASET_NAME}_decoded_frames.jsonl" \
    --output "data/indices/${DATASET_NAME}_siglip_frames.jsonl" \
    --model "$SIGLIP_MODEL" \
    --batch-size 64 \
    --device cuda

echo "  $DATASET_NAME: $(wc -l < "data/indices/${DATASET_NAME}_siglip_frames.jsonl") frames indexed"
$PY -c "
import json
from collections import Counter
c=Counter()
for line in open('data/indices/${DATASET_NAME}_siglip_frames.jsonl'):
    c[json.loads(line)['video_id']]+=1
qv=set(json.loads(l)['video_id'] for l in open('$QA_PATH'))
print(f'  videos indexed={len(c)} | QA videos={len(qv)} | coverage={100*len(set(c)&qv)/max(1,len(qv)):.0f}%')
"
echo "=== [$SLURM_JOB_ID] finished $DATASET_NAME at $(date) ==="
