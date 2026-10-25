#!/bin/bash
#SBATCH --job-name=vidcurator-decode-index
#SBATCH --output=logs/decode_index_%j.out
#SBATCH --error=logs/decode_index_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/frames data/indices runs

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

PY="$HOME/miniconda3/envs/vidcurator/bin/python"
FFMPEG="$($PY -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null || echo '')"
export FFMPEG

export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "=== [$SLURM_JOB_ID] frame decode + index started at $(date) ==="
echo "CUDA available: $($PY -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'no torch')"
echo "ffmpeg: ${FFMPEG:-not found}"

DECODE_FPS="${DECODE_FPS:-1.0}"
SIGLIP_MODEL="${SIGLIP_MODEL:-google/siglip-base-patch16-224}"

decode_dataset() {
    local qa_path="$1"
    local video_root="$2"
    local out_root="$3"
    local dataset_name="$4"
    local fps="${5:-$DECODE_FPS}"

    if [[ ! -f "$qa_path" ]]; then
        echo "skip $dataset_name: missing QA file $qa_path"
        return 0
    fi
    if [[ ! -d "$video_root" ]]; then
        echo "skip $dataset_name: missing video root $video_root"
        return 0
    fi

    echo "--- decoding $dataset_name at ${fps} fps ---"
    $PY scripts/decode_split_frames.py \
        --qa-path "$qa_path" \
        --video-root "$video_root" \
        --out-root "$out_root" \
        --fps "$fps" \
        ${FFMPEG:+--ffmpeg "$FFMPEG"} \
        --manifest "runs/decode_${dataset_name}_manifest.json"

    echo "--- building frame index for $dataset_name ---"
    $PY scripts/build_frame_index.py \
        --frames-root "$out_root" \
        --output "data/indices/${dataset_name}_decoded_frames.jsonl" \
        --fps "$fps" \
        --dataset "$dataset_name"

    echo "--- enriching with SigLIP embeddings for $dataset_name ---"
    $PY scripts/enrich_frame_index.py \
        --input "data/indices/${dataset_name}_decoded_frames.jsonl" \
        --output "data/indices/${dataset_name}_siglip_frames.jsonl" \
        --model "$SIGLIP_MODEL" \
        --batch-size 64 \
        --device cuda

    local frame_count
    frame_count=$(wc -l < "data/indices/${dataset_name}_siglip_frames.jsonl")
    echo "  $dataset_name: $frame_count frames indexed + embedded"
}

decode_dataset data/processed/nextgqa_val_150.jsonl \
    data/raw/nextgqa/NExTVideo \
    data/frames/nextgqa \
    nextgqa_val_150 \
    1.0

decode_dataset data/processed/videomme_long_150.jsonl \
    data/videos/videomme \
    data/frames/videomme \
    videomme_long_150 \
    0.1

decode_dataset data/processed/egoschema_val_150.jsonl \
    data/videos/egoschema \
    data/frames/egoschema \
    egoschema_val_150 \
    0.5

decode_dataset data/processed/nextqa_temporal_val_150.jsonl \
    data/videos/nextqa \
    data/frames/nextqa \
    nextqa_temporal_150 \
    1.0

if [[ -d data/videos/longvideobench ]]; then
    decode_dataset data/processed/longvideobench_val_150.jsonl \
        data/videos/longvideobench \
        data/frames/longvideobench \
        longvideobench_150 \
        0.1
fi

if [[ -d data/videos/mlvu ]]; then
    decode_dataset data/processed/mlvu_val_150.jsonl \
        data/videos/mlvu \
        data/frames/mlvu \
        mlvu_150 \
        0.1
fi

echo "--- frame index summary ---"
for f in data/indices/*_siglip_frames.jsonl; do
    if [[ -f "$f" ]]; then
        echo "  $(basename $f): $(wc -l < "$f") rows"
    fi
done

echo "=== [$SLURM_JOB_ID] frame decode + index finished at $(date) ==="
