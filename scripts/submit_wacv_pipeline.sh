#!/usr/bin/env bash
# Submit the full WACV experiment pipeline with GPU-aware partitioning.
# a40 (11 idle): parallel 7-8B inference — best throughput
# l40: decode/index + downloads
# dgx: reserved for GRPO later
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs

echo "=== VID-CURATOR WACV pipeline submission $(date) ==="
echo "Partition strategy: a40=inference, l40=download/decode, dgx=GRPO (later)"
sinfo -o "%P %D %t" 2>&1 | head -8
echo

# ── Phase A: downloads (CPU-light, l40 queue) ─────────────────────────────────
DL_MODELS=""
DL_DATA=""
if ! ls models/internvl3_5_8b/*.safetensors >/dev/null 2>&1; then
    DL_MODELS=$(sbatch --parsable jobs/01_download_models.sh)
    echo "[A1] model download: job $DL_MODELS"
else
    echo "[A1] models mostly present — skipping bulk download"
fi

if [[ ! -d data/videos/longvideobench ]] || [[ $(du -s data/videos/longvideobench 2>/dev/null | cut -f1) -lt 1000 ]]; then
    DL_DATA=$(sbatch --parsable jobs/01b_download_datasets.sh)
    echo "[A2] dataset download (LongVideoBench+MLVU): job $DL_DATA"
else
    echo "[A2] LongVideoBench videos present — skipping download"
fi

# ── Phase B: decode + SigLIP index (GPU, after data download) ───────────────
DECODE_DEP=""
if [[ -n "$DL_DATA" ]]; then
    DECODE_DEP="--dependency=afterok:$DL_DATA"
fi
DECODE=$(sbatch --parsable $DECODE_DEP jobs/02_decode_index_l40.sh)
echo "[B] decode+index: job $DECODE${DECODE_DEP:+ (after $DL_DATA)}"

# ── Phase C: Qwen3 zero-shot on missing datasets (parallel on a40) ──────────
submit_zs() {
    local label="$1" config="$2" model="$3" qa="$4" frames="$5" run_dir="$6" port="$7"
    if [[ -f "$run_dir/summary.json" ]]; then
        echo "[C] SKIP $label (already done)"
        return 0
    fi
    local jid
    jid=$(sbatch --parsable \
        --job-name="vc-zs-${label}" \
        --export=ALL,CONFIG_PATH="$config",MODEL_PATH="$model",QA_PATH="$qa",FRAMES_PATH="$frames",RUN_DIR="$run_dir",PORT="$port",JOB_LABEL="$label" \
        jobs/12_zero_shot_single.sh)
    echo "[C] $label → job $jid (a40)"
    echo "$jid"
}

QWEN3_JOBS=()
for spec in \
    "videomme_qwen3|configs/gpu_videomme_qwen3_zero_shot.json|models/qwen3_vl_8b_instruct|data/processed/videomme_long_150.jsonl|data/indices/videomme_long_siglip_frames.jsonl|runs/gpu_videomme_qwen3_zero_shot|8001" \
    "nextqa_qwen3|configs/gpu_nextqa_qwen3_zero_shot.json|models/qwen3_vl_8b_instruct|data/processed/nextqa_temporal_val_150.jsonl|data/indices/nextqa_temporal_siglip_frames.jsonl|runs/gpu_nextqa_qwen3_zero_shot|8002" \
    "egoschema_qwen3|configs/gpu_egoschema_qwen3_zero_shot.json|models/qwen3_vl_8b_instruct|data/processed/egoschema_val_150.jsonl|data/indices/egoschema_siglip_frames.jsonl|runs/gpu_egoschema_qwen3_zero_shot|8003"
do
    IFS='|' read -r label config model qa frames run_dir port <<< "$spec"
    jid=$(submit_zs "$label" "$config" "$model" "$qa" "$frames" "$run_dir" "$port")
    [[ -n "$jid" ]] && QWEN3_JOBS+=("$jid")
done

# ── Phase D: LongVideoBench + MLVU zero-shot (after decode, both backbones) ───
LVB_MLVU_JOBS=()
for spec in \
    "lvb_qwen25|configs/gpu_longvideobench_qwen2_5_zero_shot.json|models/qwen2_5_vl_7b|data/processed/longvideobench_val_150.jsonl|data/indices/longvideobench_150_siglip_frames.jsonl|runs/gpu_longvideobench_qwen2_5_zero_shot|8010" \
    "lvb_qwen3|configs/gpu_longvideobench_qwen3_zero_shot.json|models/qwen3_vl_8b_instruct|data/processed/longvideobench_val_150.jsonl|data/indices/longvideobench_150_siglip_frames.jsonl|runs/gpu_longvideobench_qwen3_zero_shot|8011" \
    "mlvu_qwen25|configs/gpu_mlvu_qwen2_5_zero_shot.json|models/qwen2_5_vl_7b|data/processed/mlvu_val_150.jsonl|data/indices/mlvu_150_siglip_frames.jsonl|runs/gpu_mlvu_qwen2_5_zero_shot|8012" \
    "mlvu_qwen3|configs/gpu_mlvu_qwen3_zero_shot.json|models/qwen3_vl_8b_instruct|data/processed/mlvu_val_150.jsonl|data/indices/mlvu_150_siglip_frames.jsonl|runs/gpu_mlvu_qwen3_zero_shot|8013"
do
    IFS='|' read -r label config model qa frames run_dir port <<< "$spec"
    if [[ -f "$run_dir/summary.json" ]]; then
        echo "[D] SKIP $label (already done)"
        continue
    fi
    jid=$(sbatch --parsable --dependency=afterok:$DECODE \
        --job-name="vc-zs-${label}" \
        --export=ALL,CONFIG_PATH="$config",MODEL_PATH="$model",QA_PATH="$qa",FRAMES_PATH="$frames",RUN_DIR="$run_dir",PORT="$port",JOB_LABEL="$label" \
        jobs/12_zero_shot_single.sh)
    echo "[D] $label → job $jid (a40, after decode $DECODE)"
    LVB_MLVU_JOBS+=("$jid")
done

# ── Phase E: artifact export (after all zero-shot) ────────────────────────────
EXPORT_DEPS=""
ALL_ZS=("${QWEN3_JOBS[@]}" "${LVB_MLVU_JOBS[@]}")
if [[ ${#ALL_ZS[@]} -gt 0 ]]; then
    EXPORT_DEPS="--dependency=afterok:$(IFS=:; echo "${ALL_ZS[*]}")"
fi
EXPORT=$(sbatch --parsable ${EXPORT_DEPS:+$EXPORT_DEPS} jobs/05_export_artifacts.sh)
echo "[E] artifact export: job $EXPORT"

echo
echo "=== submitted ==="
echo "  decode:     $DECODE"
echo "  qwen3 jobs: ${QWEN3_JOBS[*]:-none}"
echo "  export:     $EXPORT"
echo
echo "Monitor:  squeue -u \$USER"
echo "Logs:     logs/zs_single_*_<id>.out"
echo "State:    runs/pipeline_state.json (update manually or via orchestrator)"
echo
echo "GPU choice: a40 (48GB, 11 idle) > l40 (48GB, 2 idle) > dgx (80GB, for GRPO)"
echo "Fix applied: VLM now loads on cuda:0 explicitly (was loading on CPU → timeout)"
