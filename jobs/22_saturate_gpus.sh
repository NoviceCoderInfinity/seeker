#!/usr/bin/env bash
# Saturate idle GPUs: refresh reports, export artifacts, siglip gaps, SFT val, GRPO chain.
set -euo pipefail
ROOT="/home/vis-comp/22b3929/vidcurator"
cd "$ROOT"
mkdir -p logs

PILOT_JOB="${PILOT_JOB:-144037}"
SIGLIP_JOB="${SIGLIP_JOB:-143764}"

echo "=== GPU saturate queue $(date) ==="
echo "Pilot GRPO job: $PILOT_JOB | SigLIP job: $SIGLIP_JOB"
squeue -u "$USER" 2>/dev/null || true
echo

submit_a40() {
  local label="$1"
  shift
  if sbatch --partition=a40 --qos=a40 "$@" >/tmp/vc_sat_jobid.txt 2>&1; then
    echo "  QUEUED $label -> job $(cat /tmp/vc_sat_jobid.txt | tail -1) [a40]"
  else
    echo "  FAILED  $label: $(cat /tmp/vc_sat_jobid.txt)"
  fi
}

submit_l40() {
  local label="$1"
  shift
  if sbatch --partition=l40 --qos=l40 "$@" >/tmp/vc_sat_jobid.txt 2>&1; then
    echo "  QUEUED $label -> job $(cat /tmp/vc_sat_jobid.txt) [l40]"
  else
    echo "  FAILED  $label: $(cat /tmp/vc_sat_jobid.txt)"
  fi
}

submit_dgx() {
  local label="$1"
  shift
  if sbatch --partition=dgx --qos=dgx "$@" >/tmp/vc_sat_jobid.txt 2>&1; then
    echo "  QUEUED $label -> job $(cat /tmp/vc_sat_jobid.txt) [dgx]"
  else
    echo "  FAILED  $label: $(cat /tmp/vc_sat_jobid.txt)"
  fi
}

# --- CPU-light: paper tables + figure refresh (a40 has 12 idle nodes) ---
echo "[1] Paper refresh + artifact export..."
submit_a40 "paper-refresh" jobs/20_paper_refresh.sh
submit_a40 "artifact-export" jobs/05_export_artifacts.sh

# --- SFT full-val action metrics (pre/post GRPO comparison) ---
if [[ ! -f runs/sft/qwen2_5_vl_7b_curator_lora/validation_full.json ]]; then
  echo "[2] SFT validation (full 1941 val turns)..."
  submit_a40 "sft-val-full" \
    --export=ALL,LIMIT=1941,MAX_NEW_TOKENS=256,REPORT_NAME=validation_full.json \
    jobs/15_validate_sft.sh
else
  echo "[2] SKIP sft-val-full (validation_full.json exists)"
fi

# --- Missing siglip full split: NExT-QA (NExT-GQA siglip full in flight on 143764) ---
if [[ ! -f runs/gpu_nextqa_qwen2_5_full_siglip/summary.json ]]; then
  echo "[3] NExT-QA full siglip (curator_siglip FEP on temporal 3500)..."
  submit_a40 "nextqa-full-siglip" \
    --export=ALL,CONFIG_PATH=configs/gpu_nextqa_qwen2_5_full_siglip.json,MODEL_PATH=models/qwen2_5_vl_7b,QA_PATH=data/processed/nextqa_temporal_test_3500_siglip.jsonl,FRAMES_PATH=data/indices/nextqa_temporal_siglip_frames.jsonl,RUN_DIR=runs/gpu_nextqa_qwen2_5_full_siglip,PORT=8030,JOB_LABEL=nextqa-full-siglip \
    jobs/12_zero_shot_single.sh
else
  echo "[3] SKIP nextqa-full-siglip (done)"
fi

# --- Re-refresh reports after siglip jobs land ---
if squeue -h -j "$SIGLIP_JOB" &>/dev/null; then
  echo "[4] Chained paper refresh after siglip job $SIGLIP_JOB..."
  submit_a40 "paper-refresh-post-siglip" \
    --dependency=afterok:"$SIGLIP_JOB" \
    jobs/20_paper_refresh.sh
fi

# --- Full GRPO: resume from pilot checkpoint, 3 epochs, full train data ---
VERL_ROOT="$ROOT/third_party/verl"
if [[ -f "$VERL_ROOT/examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh" ]] && \
   [[ -x "$HOME/miniconda3/envs/vidcurator-grpo/bin/python" ]]; then
  if squeue -h -j "$PILOT_JOB" &>/dev/null; then
    echo "[5] Full GRPO chained after pilot $PILOT_JOB (auto-resume, 3 epochs, full data)..."
    submit_dgx "grpo-full" \
      --dependency=afterok:"$PILOT_JOB" \
      --export=ALL,VERL_ROOT=$VERL_ROOT,GRPO_CONFIG=configs/grpo_stage1_full.json,GRPO_EXPERIMENT=stage1_full,TRAIN_DATA=data/rl/vidcurator_train_sft.jsonl,EVAL_DATA=data/rl/vidcurator_val_sft.jsonl \
      jobs/16_grpo_stage1.sh
    FULL_GRPO_DEP="afterok:$PILOT_JOB"
  elif ! squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -q vidcurator-grpo; then
    echo "[5] Pilot not running — submit full GRPO immediately..."
    submit_dgx "grpo-full" \
      --export=ALL,VERL_ROOT=$VERL_ROOT,GRPO_CONFIG=configs/grpo_stage1_full.json,GRPO_EXPERIMENT=stage1_full,TRAIN_DATA=data/rl/vidcurator_train_sft.jsonl,EVAL_DATA=data/rl/vidcurator_val_sft.jsonl \
      jobs/16_grpo_stage1.sh
    FULL_GRPO_DEP=""
  else
    echo "[5] SKIP grpo-full (GRPO already active)"
    FULL_GRPO_DEP=""
  fi

  # Post-GRPO: refresh tables + SFT-style action validation on merged adapter (when export script lands)
  if [[ -n "${FULL_GRPO_DEP:-}" ]]; then
    echo "[6] Chained post-GRPO paper refresh..."
    submit_a40 "paper-refresh-post-grpo" \
      --dependency=afterok:"$PILOT_JOB" \
      jobs/20_paper_refresh.sh
  fi
else
  echo "[5] SKIP GRPO — verl env not ready"
fi

echo
echo "=== final queue ==="
squeue -u "$USER" 2>/dev/null || true
