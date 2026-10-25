#!/bin/bash
#SBATCH --job-name=vc-rft
#SBATCH --output=logs/rft_%j.out
#SBATCH --error=logs/rft_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
# RFT: LoRA SFT on an outcome-reward-optimal dataset (text-only action policy). Env: REWARD, EPOCHS.
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs runs/rft
export HF_HOME="/home/vis-comp/22b3929/vidcurator/.hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Training needs the dedicated env with peft (the frozen-inference env lacks it).
PY="${TRAIN_PY:-$HOME/miniconda3/envs/vidcurator-train/bin/python}"
if [[ ! -x "$PY" ]]; then echo "ERROR: training env (vidcurator-train) not found at $PY" >&2; exit 1; fi
REWARD="${REWARD:?REWARD required}"
MODEL="${MODEL:-models/qwen2_5_vl_7b}"
TRAIN="data/rl/rft_${REWARD}_sft.jsonl"
VAL="${VAL:-data/rl/vidcurator_val_sft.jsonl}"
OUT="runs/rft/${REWARD}_lora"
EPOCHS="${EPOCHS:-3}"
echo "=== [$SLURM_JOB_ID] RFT-SFT reward=$REWARD on $(hostname) $(date) ==="
$PY -c "import torch; assert torch.cuda.is_available(); print('cuda OK:', torch.cuda.get_device_name(0))"
if [[ -f "$OUT/lora_adapter/adapter_config.json" ]]; then echo "already trained: $OUT"; exit 0; fi
VALARG=""; [[ -f "$VAL" ]] && VALARG="--val $VAL"
$PY scripts/sft_train.py --model "$MODEL" --train "$TRAIN" $VALARG \
  --output "$OUT" --epochs "$EPOCHS" --lora-rank 32 --lora-alpha 64
echo "=== [$SLURM_JOB_ID] RFT-SFT $REWARD done $(date) → $OUT ==="
