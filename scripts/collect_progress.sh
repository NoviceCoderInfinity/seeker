#!/bin/bash
set -uo pipefail
cd /home/vis-comp/22b3929/vidcurator || exit 1
mkdir -p logs
OUT="logs/progress_$(date +%Y%m%d_%H%M%S).txt"
PY="${HOME}/miniconda3/envs/vidcurator/bin/python"
exec > >(tee "$OUT") 2>&1
echo "VID-CURATOR PROGRESS — $(date) — $OUT"
echo "========== 1. SLURM =========="
squeue -u "$USER" -o "%.8i %.24j %.2t %.10M %.12L %R" 2>/dev/null || true
echo "========== 2. COMPLETED RUNS =========="
ls runs/gpu_*/summary.json 2>/dev/null | wc -l | xargs echo Total:
ls runs/gpu_*/summary.json 2>/dev/null | sed 's|^runs/||;s|/summary.json||'
echo "========== 3. PENDING FULL SPLITS =========="
for r in runs/gpu_nextgqa_qwen3_full runs/gpu_nextqa_qwen3_full runs/gpu_nextgqa_qwen2_5_siglip_analysis; do
  [[ -f "$r/summary.json" ]] && echo "  $r: DONE" || echo "  $r: pending ($(ls "$r/_shards" 2>/dev/null | wc -l) shards)"
done
echo "========== 4. GRPO/SFT =========="
ls -d runs/sft/* runs/grpo/* 2>/dev/null || echo "  (no grpo output yet)"
echo "DONE -> $OUT"
