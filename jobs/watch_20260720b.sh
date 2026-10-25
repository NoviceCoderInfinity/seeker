#!/bin/bash
# Watcher: evening wave (G2 window-oracle, IV mlvu b32 balanced retry, M4 episodic).
cd /home/vis-comp/22b3929/vidcurator
JOBS="267049 267155 267157"
while true; do
  LEFT=$(squeue -u "$(whoami)" -h -o %i | grep -cFf <(tr ' ' '\n' <<< "$JOBS") || true)
  bash jobs/queue_next.sh >/dev/null 2>&1
  [[ "$LEFT" -eq 0 ]] && break
  sleep 300
done
echo "WATCHER: evening wave drained at $(date)"
sacct -j ${JOBS// /,} --format=JobID,JobName%16,State,Elapsed -X
