#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

echo "=== VID-CURATOR DGX pipeline submission ==="
echo "Submitting jobs in dependency order..."
echo

echo "[1/4] data prep (CPU)..."
DATA_JOB=$(sbatch --parsable jobs/01_prepare_data.sh)
echo "  job_id=$DATA_JOB"

echo "[2/4] frame decode + SigLIP index (1 GPU, L40)..."
DECODE_JOB=$(sbatch --parsable --parsable --dependency=afterok:$DATA_JOB jobs/02_decode_index_l40.sh)
echo "  job_id=$DECODE_JOB"

echo "[3/4] zero-shot dual-model DGX (2 GPUs)..."
ZS_JOB=$(sbatch --parsable --dependency=afterok:$DECODE_JOB jobs/03c_zero_shot_dgx.sh)
echo "  job_id=$ZS_JOB"

echo "[4/4] export artifacts (CPU, after zero-shot)..."
ARTIFACT_JOB=$(sbatch --parsable --dependency=afterok:$ZS_JOB jobs/05_export_artifacts.sh)
echo "  job_id=$ARTIFACT_JOB"

echo
echo "=== pipeline submitted ==="
echo "  data prep:     $DATA_JOB"
echo "  decode+index:  $DECODE_JOB (after $DATA_JOB)"
echo "  zs dual-model: $ZS_JOB (after $DECODE_JOB)"
echo "  artifacts:     $ARTIFACT_JOB (after $ZS_JOB)"
echo
echo "Optional: GRPO training after zero-shot passes kill test:"
echo "  sbatch --dependency=afterok:$ZS_JOB jobs/04_grpo_dgx.sh"
echo
echo "Monitor: squeue -u \$USER"
echo "Logs:    logs/<stage>_<jobid>.out / .err"
