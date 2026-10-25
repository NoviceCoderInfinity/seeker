#!/bin/bash
# Deferred submissions blocked by QOS submit limits (l40: 5 concurrent/user incl. pending;
# a40: 6). Run at every watcher wake-up; idempotent — each block checks its output/queue
# before submitting. Priority order: L4 portability > IV mlvu ladder > OV mlvu ladder.
set -uo pipefail
cd /home/vis-comp/22b3929/vidcurator

inq() { squeue -u "$(whoami)" -h -n "$1" | grep -q .; }
l40_free() { [[ "$(squeue -u "$(whoami)" -p l40 -h | wc -l)" -lt 5 ]]; }
a40_free() { [[ "$(squeue -u "$(whoami)" -p a40 -h | wc -l)" -lt 6 ]]; }

# 1) L4 ledger-portability all-family replay (one sequential job)
if [[ ! -f reports/l4_ledger_port_mlvu_g3.md ]] && ! inq vc-l4-all; then
  if l40_free; then
    sbatch --partition=l40 --qos=l40 --job-name=vc-l4-all --exclude=cn42-l40 jobs/64_l4_all.sh \
      && echo "submitted vc-l4-all"
  elif a40_free; then
    sbatch --partition=a40 --qos=a40 --job-name=vc-l4-all jobs/64_l4_all.sh \
      && echo "submitted vc-l4-all (a40)"
  else
    echo "deferred: vc-l4-all (no slots)"
  fi
fi

# 2) IV mlvu K=16/32 ladder (win-set fragility, family 2). b16 cells cached-complete;
#    b32 needs 4 GPUs + device_map=balanced: InternVL's .chat() path has no flash-attn,
#    so a 32-image generation allocates ~15GB of quadratic attention on ONE card —
#    auto packing left only 15.17GB free and missed by 50MB (266816/266817).
if false; then  # DROPPED 2026-07-20: b32 infeasible (no-flash-attn OOM x2); b16 salvaged via sigpack
  if l40_free; then
    sbatch --partition=l40 --qos=l40 --gres=gpu:4 --time=24:00:00 --job-name=vc-iv-mlvu-lad \
      --exclude=cn42-l40 \
      --export=ALL,CONFIGS="configs/xfam_iv_mlvu_b16b32.json",MODEL_PATH=models/internvl3_5_8b,PORT=8550,VLM_DEVICE_MAP=balanced,GPUS_VISIBLE=all \
      jobs/60_pilot_multi.sh && echo "submitted vc-iv-mlvu-lad"
  else
    echo "deferred: vc-iv-mlvu-lad (no l40 slots; needs gpu:4)"
  fi
fi

# 3) OV mlvu K=16/32 ladder (win-set fragility, family 3; video path, 1 GPU fine)
if [[ ! -f runs/xfam_ov_mlvu_b16b32/summary.json ]] && ! inq vc-ov-mlvu-lad; then
  if l40_free; then
    sbatch --partition=l40 --qos=l40 --gres=gpu:1 --time=24:00:00 --job-name=vc-ov-mlvu-lad \
      --exclude=cn42-l40 \
      --export=ALL,CONFIGS="configs/xfam_ov_mlvu_b16b32.json",MODEL_PATH=models/llava_onevision_qwen2_7b_ov,PORT=8551 \
      jobs/60_pilot_multi.sh && echo "submitted vc-ov-mlvu-lad"
  elif a40_free; then
    sbatch --partition=a40 --qos=a40 --gres=gpu:1 --time=24:00:00 --job-name=vc-ov-mlvu-lad \
      --export=ALL,CONFIGS="configs/xfam_ov_mlvu_b16b32.json",MODEL_PATH=models/llava_onevision_qwen2_7b_ov,PORT=8551 \
      jobs/60_pilot_multi.sh && echo "submitted vc-ov-mlvu-lad (a40)"
  else
    echo "deferred: vc-ov-mlvu-lad (no slots)"
  fi
fi

# 4) IV egoschema b32: DROPPED 2026-07-20 after 3 failed attempts (OOM: no-flash-attn
#    quadratic attention at 32 images; 266817 missed by 50MB even on gpu:4/auto).
#    Paper uses the b16 cell per the pre-registered fallback. Do not resubmit.
