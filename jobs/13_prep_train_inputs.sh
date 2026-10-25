#!/bin/bash
#SBATCH --job-name=vc-prep-train
#SBATCH --output=logs/prep_train_%x_%j.out
#SBATCH --error=logs/prep_train_%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

# Prepare training inputs (no model training here — this is the data-prep gate):
#   1. Embed full-split QA questions into the SigLIP text space (label-free, T1.b)
#      -> data/processed/<split>_siglip.jsonl  (powers deployable curator_siglip)
#   2. Build the SFT cold-start train/val set from existing scripted_curator traces
#      -> data/rl/vidcurator_train_sft.jsonl + data/rl/vidcurator_val_sft.jsonl
#
# MUST use models/siglip_base (768-dim) to match the existing frame indices.
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs data/processed data/rl

export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
PY="$HOME/miniconda3/envs/vidcurator/bin/python"
export HF_HOME="${HF_HOME:-/home/vis-comp/22b3929/vidcurator/.hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SIGLIP_MODEL="${SIGLIP_MODEL:-models/siglip_base}"

echo "=== [$SLURM_JOB_ID] prep training inputs at $(date) ==="
echo "CUDA: $($PY -c 'import torch;print(torch.cuda.is_available())' 2>/dev/null)"
echo "SigLIP model: $SIGLIP_MODEL"

# ----------------------------------------------------------------------------
# 1) Embed full-split questions into SigLIP text space
# ----------------------------------------------------------------------------
# Only the question is embedded (label-free); the frame index is NOT required here.
FULL_SPLITS=(
  "data/processed/nextgqa_val.jsonl"
  "data/processed/nextqa_temporal_test.jsonl"
  "data/processed/nextqa_temporal_test_3500.jsonl"
  "data/processed/longvideobench_val.jsonl"
  "data/processed/videomme_long.jsonl"
  "data/processed/mlvu_val.jsonl"
  "data/processed/egoschema_val.jsonl"
)

for qa in "${FULL_SPLITS[@]}"; do
  if [[ ! -f "$qa" ]]; then
    echo "--- SKIP (missing): $qa"
    continue
  fi
  out="${qa%.jsonl}_siglip.jsonl"
  if [[ -f "$out" ]]; then
    echo "--- exists, skipping: $out ($(wc -l < "$out") rows)"
    continue
  fi
  echo "--- embedding questions: $qa -> $out ($(wc -l < "$qa") rows)"
  $PY scripts/embed_questions_siglip.py \
    --input "$qa" \
    --output "$out" \
    --model "$SIGLIP_MODEL" \
    --device cuda \
    --batch-size 128
done

# ----------------------------------------------------------------------------
# 2) Build SFT cold-start train/val from existing scripted_curator runs
# ----------------------------------------------------------------------------
# Training mix follows docs/GRPO_PLAN.md (broad long-video + temporal + grounded).
# Each (run, qa) pair must align. Runs without predictions.jsonl are skipped.
echo "--- building SFT cold-start dataset ---"
RUNS=(
  "runs/gpu_longvideobench_qwen2_5_full"
  "runs/gpu_videomme_qwen2_5_full"
  "runs/gpu_mlvu_qwen2_5_full"
  "runs/gpu_egoschema_qwen2_5_full"
  "runs/gpu_nextqa_qwen2_5_analysis"
  "runs/gpu_nextgqa_qwen2_5_analysis"
)
QAS=(
  "data/processed/longvideobench_val.jsonl"
  "data/processed/videomme_long.jsonl"
  "data/processed/mlvu_val.jsonl"
  "data/processed/egoschema_val.jsonl"
  "data/processed/nextqa_temporal_val_150.jsonl"
  "data/processed/nextgqa_val_150.jsonl"
)

# include nextgqa/nextqa FULL runs too once they finish (auto-added if present)
for pair in "gpu_nextgqa_qwen2_5_full:data/processed/nextgqa_val.jsonl" \
            "gpu_nextqa_qwen2_5_full:data/processed/nextqa_temporal_test.jsonl"; do
  rd="runs/${pair%%:*}"; qp="${pair##*:}"
  if [[ -f "$rd/predictions.jsonl" ]]; then
    RUNS+=("$rd"); QAS+=("$qp")
    echo "  + including finished full run: $rd"
  fi
done

$PY scripts/export_sft_traces.py \
  --runs "${RUNS[@]}" \
  --qa-paths "${QAS[@]}" \
  --policy scripted_curator \
  --output data/rl/vidcurator_train_sft.jsonl \
  --val-output data/rl/vidcurator_val_sft.jsonl \
  --val-frac 0.1 \
  --split-seed 13

echo "--- SFT dataset summary ---"
echo "  train rows: $(wc -l < data/rl/vidcurator_train_sft.jsonl 2>/dev/null || echo 0)"
echo "  val rows:   $(wc -l < data/rl/vidcurator_val_sft.jsonl 2>/dev/null || echo 0)"
echo "=== [$SLURM_JOB_ID] prep finished at $(date) ==="
