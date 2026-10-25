#!/usr/bin/env bash
# Queue all remaining eval + training jobs for paper readiness (Qwen2.5 + Qwen3).
set -euo pipefail
ROOT="/home/vis-comp/22b3929/vidcurator"
cd "$ROOT"
mkdir -p logs

echo "=== paper-ready queue $(date) ==="
squeue -u "$USER" 2>/dev/null || true

submit_l40() {
  local label="$1"; shift
  sbatch "$@" && echo "  QUEUED $label" || echo "  FAILED  $label"
}

submit_dgx() {
  local label="$1"; shift
  sbatch "$@" && echo "  QUEUED $label (DGX)" || echo "  FAILED  $label"
}

# Skip duplicate nextgqa qwen3 if one is already running
if squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -q "vidcurator-zs-single"; then
  running_zs=$(squeue -u "$USER" -h -o "%j %R" 2>/dev/null | grep -c vidcurator-zs-single || true)
  echo "  INFO: $running_zs zero-shot jobs already active"
fi

# Remaining Qwen3 full splits
if [[ ! -f runs/gpu_nextqa_qwen3_full/summary.json ]]; then
  submit_l40 "nextqa-qwen3-full" --partition=l40 --qos=l40 \
    --export=ALL,CONFIG_PATH=configs/gpu_nextqa_qwen3_full.json,MODEL_PATH=models/qwen3_vl_8b_instruct,QA_PATH=data/processed/nextqa_temporal_test_3500_siglip.jsonl,FRAMES_PATH=data/indices/nextqa_temporal_siglip_frames.jsonl,RUN_DIR=runs/gpu_nextqa_qwen3_full,PORT=8025,JOB_LABEL=nextqa-qwen3-full \
    jobs/12_zero_shot_single.sh
fi

if [[ ! -f runs/gpu_nextgqa_qwen3_full/summary.json ]] && ! squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -q "143230\|nextgqa-qwen3"; then
  submit_l40 "nextgqa-qwen3-full" --partition=l40 --qos=l40 \
    --export=ALL,CONFIG_PATH=configs/gpu_nextgqa_qwen3_full.json,MODEL_PATH=models/qwen3_vl_8b_instruct,QA_PATH=data/processed/nextgqa_val_siglip.jsonl,FRAMES_PATH=data/indices/nextgqa_siglip_frames.jsonl,RUN_DIR=runs/gpu_nextgqa_qwen3_full,PORT=8024,JOB_LABEL=nextgqa-qwen3-full \
    jobs/12_zero_shot_single.sh
fi

# SigLIP FEP runs (Qwen2.5) — resumes from _shards if a prior job timed out
if { [[ ! -f runs/gpu_nextgqa_qwen2_5_siglip_analysis/summary.json ]] || \
     [[ ! -f runs/gpu_nextqa_qwen2_5_siglip_analysis/summary.json ]] || \
     [[ ! -f runs/gpu_nextgqa_qwen2_5_full_siglip/summary.json ]]; } && \
   ! squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -q vidcurator-siglip; then
  submit_l40 "siglip-analysis" --partition=l40 --qos=l40 jobs/17_siglip_analysis.sh
fi

# GRPO stage 1 (DGX) — requires third_party/verl + vidcurator-grpo
VERL_ROOT="$ROOT/third_party/verl"
if [[ -f "$VERL_ROOT/examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh" ]] && \
   [[ -x "$HOME/miniconda3/envs/vidcurator-grpo/bin/python" ]]; then
  if ! squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -q vidcurator-grpo; then
    submit_dgx "grpo-stage1" --partition=dgx --qos=dgx \
      --export=ALL,VERL_ROOT=$VERL_ROOT,GRPO_CONFIG=configs/grpo_stage1_pilot.json,TRAIN_DATA=data/rl/vidcurator_train_sft.jsonl,EVAL_DATA=data/rl/vidcurator_val_sft.jsonl \
      jobs/16_grpo_stage1.sh
  fi
fi

# Paper tables refresh (CPU-light, no GPU)
submit_l40 "paper-refresh" --partition=l40 --qos=l40 jobs/20_paper_refresh.sh

echo "=== final queue ==="
squeue -u "$USER" 2>/dev/null || true
