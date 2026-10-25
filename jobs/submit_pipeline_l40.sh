#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

echo "=== VID-CURATOR L40 pipeline submission ==="
echo "Submitting jobs in dependency order..."
echo

echo "[1/4] data prep (CPU)..."
DATA_JOB=$(sbatch --parsable jobs/01_prepare_data.sh)
echo "  job_id=$DATA_JOB"

echo "[2/4] frame decode + SigLIP index (1 GPU, L40)..."
DECODE_JOB=$(sbatch --parsable --dependency=afterok:$DATA_JOB jobs/02_decode_index_l40.sh)
echo "  job_id=$DECODE_JOB"

echo "[3a/4] zero-shot Qwen2.5-VL (1 GPU, L40)..."
ZS_QWEN25_JOB=$(sbatch --parsable --dependency=afterok:$DECODE_JOB jobs/03a_zero_shot_qwen2_5_l40.sh)
echo "  job_id=$ZS_QWEN25_JOB"

echo "[3b/4] zero-shot Qwen3-VL (1 GPU, L40)..."
ZS_QWEN3_JOB=$(sbatch --parsable --dependency=afterok:$DECODE_JOB jobs/03b_zero_shot_qwen3_l40.sh)
echo "  job_id=$ZS_QWEN3_JOB"

echo "[4/4] export artifacts (CPU, after both zero-shot jobs)..."
ARTIFACT_JOB=$(sbatch --parsable --dependency=afterok:$ZS_QWEN25_JOB:$ZS_QWEN3_JOB jobs/05_export_artifacts.sh)
echo "  job_id=$ARTIFACT_JOB"

echo
echo "=== pipeline submitted ==="
echo "  data prep:     $DATA_JOB"
echo "  decode+index:  $DECODE_JOB (after $DATA_JOB)"
echo "  zs Qwen2.5-VL: $ZS_QWEN25_JOB (after $DECODE_JOB)"
echo "  zs Qwen3-VL:   $ZS_QWEN3_JOB (after $DECODE_JOB)"
echo "  artifacts:     $ARTIFACT_JOB (after $ZS_QWEN25_JOB + $ZS_QWEN3_JOB)"
echo
echo "Monitor: squeue -u \$USER"
echo "Logs:    logs/<stage>_<jobid>.out / .err"
echo "Cancel:  scancel <job_id>"
