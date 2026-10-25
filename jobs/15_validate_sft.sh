#!/bin/bash
#SBATCH --job-name=vc-sft-val
#SBATCH --output=logs/sft_val_%x_%j.out
#SBATCH --error=logs/sft_val_%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Validate the SFT cold-start adapter: measure action-schema format validity and
# action-type match on held-out SFT val conversations, vs the un-adapted base.
# Intended to run after jobs/14_sft_cold_start.sh (submit with
#   sbatch --dependency=afterok:<sft_jobid> jobs/15_validate_sft.sh).
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs

TRAIN_ENV="$HOME/miniconda3/envs/vidcurator-train"
PY="$TRAIN_ENV/bin/python"
export PATH="$TRAIN_ENV/bin:$PATH"
export HF_HOME="${HF_HOME:-/home/vis-comp/22b3929/vidcurator/.hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_PATH="${MODEL_PATH:-models/qwen2_5_vl_7b}"
ADAPTER_DIR="${ADAPTER_DIR:-runs/sft/$(basename "$MODEL_PATH")_curator_lora}"
VAL_DATA="${VAL_DATA:-data/rl/vidcurator_val_sft.jsonl}"
LIMIT="${LIMIT:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
REPORT_NAME="${REPORT_NAME:-validation.json}"

echo "=== [$SLURM_JOB_ID] SFT validation at $(date) ==="
echo "Model:   $MODEL_PATH"
echo "Adapter: $ADAPTER_DIR"
echo "Val:     $VAL_DATA"
echo "CUDA:    $($PY -c 'import torch;print(torch.cuda.is_available())' 2>/dev/null)"

if [[ ! -d "$ADAPTER_DIR" ]]; then
    echo "ERROR: adapter dir not found: $ADAPTER_DIR (did SFT finish?)" >&2
    exit 1
fi

$PY scripts/validate_sft_adapter.py \
    --model "$MODEL_PATH" \
    --adapter "$ADAPTER_DIR" \
    --val "$VAL_DATA" \
    --limit "$LIMIT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --compare-base \
    --dump-failures "${DUMP_FAILURES:-0}" \
    --output "$ADAPTER_DIR/$REPORT_NAME"

echo "=== [$SLURM_JOB_ID] SFT validation finished at $(date) ==="
echo "Report: $ADAPTER_DIR/$REPORT_NAME"
