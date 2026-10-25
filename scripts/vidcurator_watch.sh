#!/usr/bin/env bash
# Watch SLURM jobs and log status; re-submit next stage when deps complete.
set -euo pipefail

ROOT="${VIDCURATOR_ROOT:-$HOME/vidcurator}"
STATE="$ROOT/runs/pipeline_state.json"
LOG="$ROOT/logs/pipeline_watch.log"
POLL="${POLL_SECONDS:-60}"

mkdir -p "$ROOT/logs" "$ROOT/runs"
cd "$ROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

write_state() {
  local phase="$1" status="$2" detail="${3:-}"
  python3 - <<PY
import json, datetime
from pathlib import Path
p = Path("$STATE")
state = {}
if p.exists():
    try: state = json.loads(p.read_text())
    except Exception: state = {}
state.update({
    "phase": "$phase",
    "status": "$status",
    "detail": """$detail""",
    "updated_at": datetime.datetime.now().isoformat(),
    "jobs": """$(squeue -u "$USER" -h -o '%i|%j|%T|%M' 2>/dev/null | tr '\n' ';')""",
})
p.write_text(json.dumps(state, indent=2))
PY
}

wait_job() {
  local job_id="$1" label="$2"
  log "Watching $label (job $job_id)..."
  while squeue -j "$job_id" -h 2>/dev/null | grep -q .; do
    write_state "watch" "running" "$label job=$job_id"
    sleep "$POLL"
  done
  local st
  st=$(sacct -j "$job_id" --format=State -n -P 2>/dev/null | head -1 | cut -d'|' -f1 || echo UNKNOWN)
  log "$label job $job_id finished: $st"
  if [[ "$st" != "COMPLETED" ]]; then
    write_state "watch" "failed" "$label job=$job_id state=$st"
    return 1
  fi
  write_state "watch" "done" "$label job=$job_id"
  return 0
}

export -f log write_state wait_job
export ROOT STATE LOG POLL USER

log "Watcher started (poll=${POLL}s)"
