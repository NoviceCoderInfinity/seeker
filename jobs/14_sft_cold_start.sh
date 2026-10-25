#!/bin/bash
#SBATCH --job-name=vc-sft-cold
#SBATCH --output=logs/sft_cold_%x_%j.out
#SBATCH --error=logs/sft_cold_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=12

# LoRA SFT cold-start of the curator action policy (pre-GRPO).
# Trains a LoRA adapter on the base VLM from scripted-curator action traces.
# Uses the dedicated training env (vidcurator-train) which adds peft to the
# pinned inference stack, so the frozen-VLM serving env stays untouched.
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs runs/sft

TRAIN_ENV="$HOME/miniconda3/envs/vidcurator-train"
PY="$TRAIN_ENV/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "ERROR: training env not found at $TRAIN_ENV" >&2
    echo "Create it: conda create -y -n vidcurator-train --clone vidcurator && \\" >&2
    echo "           $TRAIN_ENV/bin/pip install peft" >&2
    exit 1
fi
export PATH="$TRAIN_ENV/bin:$PATH"
export HF_HOME="${HF_HOME:-/home/vis-comp/22b3929/vidcurator/.hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_PATH="${MODEL_PATH:-models/qwen2_5_vl_7b}"
TRAIN_DATA="${TRAIN_DATA:-data/rl/vidcurator_train_sft.jsonl}"
VAL_DATA="${VAL_DATA:-data/rl/vidcurator_val_sft.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/sft/$(basename "$MODEL_PATH")_curator_lora}"
EPOCHS="${EPOCHS:-2}"
LORA_RANK="${LORA_RANK:-32}"

echo "=== [$SLURM_JOB_ID] SFT cold-start at $(date) ==="
echo "Model:  $MODEL_PATH"
echo "Train:  $TRAIN_DATA ($(wc -l < "$TRAIN_DATA" 2>/dev/null || echo '?') rows)"
echo "Val:    $VAL_DATA ($(wc -l < "$VAL_DATA" 2>/dev/null || echo '?') rows)"
echo "Output: $OUTPUT_DIR"
echo "CUDA:   $($PY -c 'import torch;print(torch.cuda.is_available())' 2>/dev/null)"
echo "peft:   $($PY -c 'import peft;print(peft.__version__)' 2>/dev/null || echo 'MISSING')"

if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "ERROR: training data not found: $TRAIN_DATA" >&2
    echo "Run jobs/13_prep_train_inputs.sh first to generate SFT traces." >&2
    exit 1
fi

$PY scripts/sft_train.py \
    --model "$MODEL_PATH" \
    --train "$TRAIN_DATA" \
    --val "$VAL_DATA" \
    --output "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --lora-rank "$LORA_RANK" \
    --batch-size 2 \
    --grad-accum 8 \
    --max-len 2048

echo "=== [$SLURM_JOB_ID] SFT cold-start finished at $(date) ==="
echo "Adapter: $OUTPUT_DIR"
