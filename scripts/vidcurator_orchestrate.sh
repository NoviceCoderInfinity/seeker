#!/usr/bin/env bash
# Full vidcurator pipeline: smoke -> sbatch GPU jobs -> watch -> export.
# Run inside tmux so it survives SSH logout.
set -euo pipefail

ROOT="$HOME/vidcurator"
cd "$ROOT"
source "$ROOT/scripts/vidcurator_watch.sh"

PY="${VIDCURATOR_PY:-$HOME/miniconda3/envs/vidcurator/bin/python}"
mkdir -p logs runs

log "=== VID-CURATOR orchestrator started ==="
write_state "init" "running" "orchestrator started"

# ── Phase 0: smoke tests (CPU, login node) ───────────────────────────────────
log "Phase 0: smoke tests"
write_state "smoke" "running" "unit tests + fixture kill test"

if ! "$PY" -m unittest discover -s tests -q 2>&1 | tee -a "$LOG"; then
  log "ERROR: unit tests failed"; write_state "smoke" "failed" "unit tests"; exit 1
fi

if ! "$PY" -m vidcurator.run evaluate --config configs/kill_test.json 2>&1 | tee -a "$LOG"; then
  log "ERROR: fixture kill test failed"; write_state "smoke" "failed" "kill test"; exit 1
fi

if ! "$PY" scripts/check_research_readiness.py 2>&1 | tee -a "$LOG"; then
  log "WARN: readiness check reported gaps (expected on login node)"
fi

log "Phase 0: smoke PASSED"
write_state "smoke" "done" "all smoke tests passed"

# ── Phase 1: submit Qwen3 multi-dataset on DGX (P0/T0.2) ─────────────────────
log "Phase 1: submitting Qwen3-VL multi-dataset zero-shot (DGX)"
write_state "submit_qwen3" "running" "sbatch jobs/11_zero_shot_qwen3_multi_dgx.sh"

QWEN3_JOB=""
if [[ ! -f runs/gpu_videomme_qwen3_zero_shot/summary.json ]] || \
   [[ ! -f runs/gpu_nextqa_qwen3_zero_shot/summary.json ]] || \
   [[ ! -f runs/gpu_egoschema_qwen3_zero_shot/summary.json ]]; then
  QWEN3_JOB=$(sbatch --parsable jobs/11_zero_shot_qwen3_multi_dgx.sh)
  log "Submitted Qwen3 multi job: $QWEN3_JOB"
  write_state "submit_qwen3" "submitted" "job=$QWEN3_JOB"
  wait_job "$QWEN3_JOB" "qwen3-multi-dataset" || {
    log "ERROR: Qwen3 multi job failed — check logs/zs_qwen3_multi_${QWEN3_JOB}.err"
    exit 1
  }
else
  log "Qwen3 multi-dataset runs already complete — skipping"
  write_state "submit_qwen3" "skipped" "already done"
fi

# ── Phase 2: export all artifacts ────────────────────────────────────────────
log "Phase 2: exporting paper artifacts"
write_state "export" "running" "artifact export"

ARTIFACT_JOB=$(sbatch --parsable jobs/05_export_artifacts.sh)
log "Submitted artifact export job: $ARTIFACT_JOB"
wait_job "$ARTIFACT_JOB" "artifact-export" || log "WARN: artifact export had issues"

# ── Phase 3: consolidate results into experiments.md ─────────────────────────
log "Phase 3: consolidating results"
write_state "consolidate" "running" "updating result summaries"

"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path

root = Path(".")
runs = sorted((root / "runs").glob("gpu_*/summary.json"))
print(f"\n=== GPU run summary ({len(runs)} runs) ===")
for sp in runs:
    s = json.load(open(sp))
    name = sp.parent.name
    groups = s.get("groups", {})
    ce = s.get("context_efficiency", {})
    sf = s.get("sf_delta", {})
    print(f"\n{name}:")
    for k in sorted(groups):
        g = groups[k]
        print(f"  {k}: acc={g['accuracy']:.3f} frames={g['avg_frames']:.1f} fep={g.get('fep','n/a')}")
    if ce:
        print(f"  CE-AUC: { {k: round(v,3) if v else None for k,v in sorted(ce.items())} }")
    if sf:
        print(f"  SF-delta: { {k: round(v,3) for k,v in sorted(sf.items())} }")
PY

write_state "done" "complete" "orchestrator finished — check logs/pipeline_watch.log and runs/gpu_*/summary.json"
log "=== VID-CURATOR orchestrator COMPLETE ==="
log "Next manual steps: LongVideoBench/MLVU decode (P0/T0.3), external baselines (P1), GRPO (P3)"
