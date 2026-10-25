#!/bin/bash
#SBATCH --job-name=vidcurator-grpo-s1
#SBATCH --output=logs/grpo_stage1_%j.out
#SBATCH --error=logs/grpo_stage1_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

# GRPO stage 1: LoRA-GRPO on Qwen2.5-VL-7B using SFT cold-start adapter.
# Requires vidcurator-grpo env (bash env/setup_grpo_env.sh on login node first).
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs runs/grpo

GRPO_PY="${GRPO_PY:-$HOME/miniconda3/envs/vidcurator-grpo/bin/python}"
if [[ ! -x "$GRPO_PY" ]]; then
  echo "ERROR: vidcurator-grpo env missing. Run: bash env/setup_grpo_env.sh" >&2
  exit 1
fi
export PATH="$(dirname "$GRPO_PY"):$PATH"

export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

export GRPO_CONFIG="${GRPO_CONFIG:-configs/grpo_stage1_pilot.json}"
export TRAIN_DATA="${TRAIN_DATA:-data/rl/vidcurator_train_sft.jsonl}"
export EVAL_DATA="${EVAL_DATA:-data/rl/vidcurator_val_sft.jsonl}"
export MODEL_INDEX="${MODEL_INDEX:-0}"
export VERL_ROOT="/home/vis-comp/22b3929/vidcurator/third_party/verl"
if [[ ! -f "$VERL_ROOT/examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh" ]]; then
  echo "ERROR: missing verl entrypoint at $VERL_ROOT/examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh" >&2
  exit 1
fi

echo "=== [$SLURM_JOB_ID] GRPO stage 1 started at $(date) ==="
echo "GRPO_CONFIG=$GRPO_CONFIG"
echo "TRAIN_DATA=$TRAIN_DATA"
echo "EVAL_DATA=$EVAL_DATA"
echo "VERL_ROOT=$VERL_ROOT"

exec bash jobs/04_grpo_dgx.sh
