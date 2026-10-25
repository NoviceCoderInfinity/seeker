#!/bin/bash
#SBATCH --job-name=vc-grpo-val
#SBATCH --output=logs/grpo_validate_%j.out
#SBATCH --error=logs/grpo_validate_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Merge latest GRPO FSDP checkpoint -> HF LoRA dir, then validate action metrics on full SFT val.
# Submit after full GRPO finishes:
#   sbatch --dependency=afterok:<grpo_jobid> jobs/24_post_grpo_validate.sh
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs runs/grpo

GRPO_PY="${GRPO_PY:-$HOME/miniconda3/envs/vidcurator-grpo/bin/python}"
TRAIN_PY="${TRAIN_PY:-$HOME/miniconda3/envs/vidcurator-train/bin/python}"
export PATH="$(dirname "$GRPO_PY"):$PATH"
export HF_HOME="${HF_HOME:-/home/vis-comp/22b3929/vidcurator/.hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

GRPO_RUN_DIR="${GRPO_RUN_DIR:-runs/grpo/qwen2_5_vl_7b}"
LATEST_FILE="$GRPO_RUN_DIR/latest_checkpointed_iteration.txt"
MODEL_PATH="${MODEL_PATH:-models/qwen2_5_vl_7b}"
VAL_DATA="${VAL_DATA:-data/rl/vidcurator_val_sft.jsonl}"
MERGED_DIR="${MERGED_DIR:-runs/grpo/qwen2_5_vl_7b_grpo_lora}"
# verl.model_merger writes PEFT weights under lora_adapter/, not the merge root.
ADAPTER_DIR="${ADAPTER_DIR:-$MERGED_DIR/lora_adapter}"

echo "=== [$SLURM_JOB_ID] post-GRPO validate at $(date) ==="

if [[ ! -f "$LATEST_FILE" ]]; then
  echo "ERROR: missing $LATEST_FILE" >&2
  exit 1
fi
STEP=$(tr -d '[:space:]' < "$LATEST_FILE")
ACTOR_DIR="$GRPO_RUN_DIR/global_step_${STEP}/actor"
if [[ ! -d "$ACTOR_DIR" ]]; then
  echo "ERROR: missing actor checkpoint: $ACTOR_DIR" >&2
  exit 1
fi

if [[ -f "$ADAPTER_DIR/adapter_config.json" && "${SKIP_MERGE:-0}" != "0" ]]; then
  echo "SKIP_MERGE=1 and adapter exists at $ADAPTER_DIR — skipping FSDP merge"
else
  echo "Merging FSDP actor checkpoint global_step_${STEP} -> $MERGED_DIR"
  rm -rf "$MERGED_DIR"
  "$GRPO_PY" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$ACTOR_DIR" \
    --target_dir "$MERGED_DIR"
fi

if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ]]; then
  echo "ERROR: missing adapter_config.json at $ADAPTER_DIR (merge failed?)" >&2
  exit 1
fi

echo "Validating merged GRPO adapter on full SFT val (1941 turns, max_new_tokens=384)"
"$TRAIN_PY" scripts/validate_sft_adapter.py \
  --model "$MODEL_PATH" \
  --adapter "$ADAPTER_DIR" \
  --val "$VAL_DATA" \
  --limit 1941 \
  --max-new-tokens 384 \
  --compare-base \
  --output "$MERGED_DIR/validation_grpo.json"

echo "=== [$SLURM_JOB_ID] post-GRPO validate finished at $(date) ==="
echo "Report: $MERGED_DIR/validation_grpo.json"
