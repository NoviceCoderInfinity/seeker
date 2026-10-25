#!/bin/bash
#SBATCH --job-name=vidcurator-siglip
#SBATCH --output=logs/siglip_analysis_%j.out
#SBATCH --error=logs/siglip_analysis_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Re-run Qwen2.5 analysis with curator_siglip + flexframe (configs updated after prior runs).
# Uses fresh RUN_DIRs so existing summaries are not overwritten.
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs runs

export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
PY="$HOME/miniconda3/envs/vidcurator/bin/python"

run_one() {
  local cfg="$1" model="$2" qa="$3" frames="$4" run_dir="$5" port="$6" label="$7"
  if [[ -f "$run_dir/summary.json" ]]; then
    echo "Skip (done): $run_dir"
    return 0
  fi
  export CONFIG_PATH="$cfg" MODEL_PATH="$model" QA_PATH="$qa" FRAMES_PATH="$frames"
  export RUN_DIR="$run_dir" PORT="$port" JOB_LABEL="$label"
  bash jobs/12_zero_shot_single.sh
}

echo "=== [$SLURM_JOB_ID] SigLIP / curator_siglip analysis at $(date) ==="

# NExT-GQA analysis (n=150) — headline FEP + CE-AUC with curator_siglip
run_one \
  configs/gpu_nextgqa_qwen2_5_analysis.json \
  models/qwen2_5_vl_7b \
  data/processed/nextgqa_val_150_siglip.jsonl \
  data/indices/nextgqa_siglip_frames.jsonl \
  runs/gpu_nextgqa_qwen2_5_siglip_analysis \
  8010 \
  nextgqa-siglip-analysis

# NExT-QA analysis (n=150)
run_one \
  configs/gpu_nextqa_qwen2_5_analysis.json \
  models/qwen2_5_vl_7b \
  data/processed/nextqa_temporal_val_150_siglip.jsonl \
  data/indices/nextqa_temporal_siglip_frames.jsonl \
  runs/gpu_nextqa_qwen2_5_siglip_analysis \
  8011 \
  nextqa-siglip-analysis

# NExT-GQA full — acceptance gate: curator_siglip FEP > uniform
run_one \
  configs/gpu_nextgqa_qwen2_5_full_siglip.json \
  models/qwen2_5_vl_7b \
  data/processed/nextgqa_val_siglip.jsonl \
  data/indices/nextgqa_siglip_frames.jsonl \
  runs/gpu_nextgqa_qwen2_5_full_siglip \
  8012 \
  nextgqa-siglip-full

echo "=== [$SLURM_JOB_ID] SigLIP analysis finished at $(date) ==="
