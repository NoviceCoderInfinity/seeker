#!/usr/bin/env bash
# Submit parallel SLURM jobs for VID-CURATOR (Qwen2.5 + Qwen3 only).
# Respects l40 QoS (~4 concurrent); extra jobs queue automatically.
set -euo pipefail
ROOT="/home/vis-comp/22b3929/vidcurator"
cd "$ROOT"
mkdir -p logs

echo "=== VID-CURATOR parallel queue $(date) ==="
echo "Current queue:"
squeue -u "$USER" 2>/dev/null || true
echo

submit_l40() {
  local label="$1"
  shift
  if sbatch "$@" >/tmp/vc_jobid.txt 2>&1; then
    local jid
    jid=$(cat /tmp/vc_jobid.txt)
    echo "  QUEUED $label -> job $jid"
  else
    echo "  FAILED  $label: $(cat /tmp/vc_jobid.txt)"
  fi
}

submit_dgx() {
  local label="$1"
  shift
  if sbatch "$@" >/tmp/vc_jobid.txt 2>&1; then
    local jid
    jid=$(cat /tmp/vc_jobid.txt)
    echo "  QUEUED $label -> job $jid (DGX)"
  else
    echo "  FAILED  $label: $(cat /tmp/vc_jobid.txt)"
  fi
}

# --- P1: SigLIP / curator_siglip analysis (Qwen2.5) ---
echo "[P1] SigLIP FEP analysis (Qwen2.5, 3 evals in one job)..."
submit_l40 "siglip-analysis" --partition=l40 --qos=l40 jobs/17_siglip_analysis.sh

# --- P1: Qwen3 full splits (6 datasets, parallel) ---
echo "[P1] Qwen3 full-split evals..."
QWEN3_FULL=(
  "gpu_videomme_qwen3_full:models/qwen3_vl_8b_instruct:data/processed/videomme_long_siglip.jsonl:data/indices/videomme_long_siglip_frames.jsonl:runs/gpu_videomme_qwen3_full:8020"
  "gpu_longvideobench_qwen3_full:models/qwen3_vl_8b_instruct:data/processed/longvideobench_val_siglip.jsonl:data/indices/longvideobench_siglip_frames.jsonl:runs/gpu_longvideobench_qwen3_full:8021"
  "gpu_mlvu_qwen3_full:models/qwen3_vl_8b_instruct:data/processed/mlvu_val_siglip.jsonl:data/indices/mlvu_siglip_frames.jsonl:runs/gpu_mlvu_qwen3_full:8022"
  "gpu_egoschema_qwen3_full:models/qwen3_vl_8b_instruct:data/processed/egoschema_val_siglip.jsonl:data/indices/egoschema_siglip_frames.jsonl:runs/gpu_egoschema_qwen3_full:8023"
  "gpu_nextgqa_qwen3_full:models/qwen3_vl_8b_instruct:data/processed/nextgqa_val_siglip.jsonl:data/indices/nextgqa_siglip_frames.jsonl:runs/gpu_nextgqa_qwen3_full:8024"
  "gpu_nextqa_qwen3_full:models/qwen3_vl_8b_instruct:data/processed/nextqa_temporal_test_3500_siglip.jsonl:data/indices/nextqa_temporal_siglip_frames.jsonl:runs/gpu_nextqa_qwen3_full:8025"
)
for entry in "${QWEN3_FULL[@]}"; do
  IFS=':' read -r cfg model qa frames run_dir port <<< "$entry"
  if [[ -f "$run_dir/summary.json" ]]; then
    echo "  SKIP $cfg (already done)"
    continue
  fi
  if [[ ! -f "configs/${cfg}.json" ]]; then
    echo "  SKIP $cfg (config missing)"
    continue
  fi
  submit_l40 "$cfg" \
    --partition=l40 --qos=l40 \
    --export=ALL,CONFIG_PATH=configs/${cfg}.json,MODEL_PATH=$model,QA_PATH=$qa,FRAMES_PATH=$frames,RUN_DIR=$run_dir,PORT=$port,JOB_LABEL=$cfg \
    jobs/12_zero_shot_single.sh
done

# --- P0: GRPO (DGX) if vidcurator-grpo env is ready ---
echo "[P0] GRPO stage 1 (DGX, if env ready)..."
if source ~/miniconda3/etc/profile.d/conda.sh && conda activate vidcurator-grpo 2>/dev/null && python -c "import verl" 2>/dev/null; then
  VERL_ROOT="/home/vis-comp/22b3929/vidcurator/third_party/verl"
  if [[ ! -f "$VERL_ROOT/examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh" ]]; then
    echo "  SKIP GRPO — third_party/verl missing GRPO entrypoint."
    VERL_ROOT=""
  fi
  conda deactivate
  if [[ -n "$VERL_ROOT" ]]; then
    submit_dgx "grpo-stage1" \
      --partition=dgx --qos=dgx \
      --export=ALL,VERL_ROOT=$VERL_ROOT,GRPO_CONFIG=configs/grpo_stage1_pilot.json,TRAIN_DATA=data/rl/vidcurator_train_sft.jsonl,EVAL_DATA=data/rl/vidcurator_val_sft.jsonl \
      jobs/16_grpo_stage1.sh
  fi
else
  echo "  SKIP GRPO — vidcurator-grpo / verl not installed yet."
  echo "  Run: bash env/setup_grpo_env.sh"
fi

echo
echo "=== submission complete ==="
squeue -u "$USER" 2>/dev/null || true
