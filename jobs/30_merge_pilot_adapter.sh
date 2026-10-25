#!/bin/bash
#SBATCH --job-name=vc-merge-pilot
#SBATCH --output=logs/merge_pilot_%j.out
#SBATCH --error=logs/merge_pilot_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

# Merge a verl FSDP GRPO checkpoint into a HF LoRA adapter dir (merge only — the
# downstream eval is the validation). Defaults target the lambda=-0.2 pilot.
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs

GRPO_PY="${GRPO_PY:-$HOME/miniconda3/envs/vidcurator-grpo/bin/python}"
export HF_HOME="${HF_HOME:-/home/vis-comp/22b3929/vidcurator/.hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

GRPO_RUN_DIR="${GRPO_RUN_DIR:-runs/grpo/pilot_lamneg2}"
MERGED_DIR="${MERGED_DIR:-runs/grpo/pilot_lamneg2_lora}"
STEP="${STEP:-$(tr -d '[:space:]' < "$GRPO_RUN_DIR/latest_checkpointed_iteration.txt")}"
ACTOR_DIR="$GRPO_RUN_DIR/global_step_${STEP}/actor"

if [[ ! -d "$ACTOR_DIR" ]]; then
  echo "ERROR: missing actor checkpoint: $ACTOR_DIR" >&2
  exit 1
fi

echo "=== [$SLURM_JOB_ID] merging $ACTOR_DIR -> $MERGED_DIR at $(date) ==="
rm -rf "$MERGED_DIR"
"$GRPO_PY" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$ACTOR_DIR" \
  --target_dir "$MERGED_DIR"

if [[ ! -f "$MERGED_DIR/lora_adapter/adapter_config.json" ]]; then
  echo "ERROR: merge produced no lora_adapter/adapter_config.json" >&2
  exit 1
fi
echo "=== merge OK: $MERGED_DIR/lora_adapter ==="
