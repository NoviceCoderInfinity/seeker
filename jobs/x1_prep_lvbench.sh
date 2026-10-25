#!/bin/bash
#SBATCH --job-name=x1-prep-lvbench
#SBATCH --output=logs/x1_prep_%j.out
#SBATCH --error=logs/x1_prep_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=cn30-a40,cn32-a40,cn37-a40,cn38-a40

# X1 prep: decode LVBench @1fps -> frame index -> SigLIP frame embeddings -> question embeddings.
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
export HF_HOME="/home/vis-comp/22b3929/vidcurator/.hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="$HOME/miniconda3/envs/vidcurator/bin/python"
FFMPEG=$($PY -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")

echo "--- decode @1fps ---"
$PY scripts/decode_split_frames.py \
    --qa-path data/processed/lvbench_val.jsonl \
    --video-root data/lvbench/videos \
    --out-root data/frames/lvbench \
    --fps 1 --ffmpeg "$FFMPEG" \
    --manifest runs/decode_lvbench_manifest.json

echo "--- build frame index ---"
$PY scripts/build_frame_index.py \
    --frames-root data/frames/lvbench \
    --output data/indices/lvbench_frames.jsonl \
    --fps 1 --dataset lvbench

echo "--- SigLIP frame embeddings (GPU) ---"
$PY scripts/enrich_frame_index.py \
    --input data/indices/lvbench_frames.jsonl \
    --output data/indices/lvbench_siglip_frames.jsonl \
    --model models/siglip_base --device cuda --batch-size 256

echo "--- SigLIP question embeddings ---"
$PY scripts/embed_questions_siglip.py \
    --input data/processed/lvbench_val.jsonl \
    --output data/processed/lvbench_val_siglip.jsonl \
    --model models/siglip_base --device cuda

$PY - <<'EOF'
import json
n=sum(1 for _ in open("data/indices/lvbench_siglip_frames.jsonl"))
q=sum(1 for _ in open("data/processed/lvbench_val_siglip.jsonl"))
print(f"X1 PREP OK: {n} frames indexed, {q} questions embedded")
EOF
