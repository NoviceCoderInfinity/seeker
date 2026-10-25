#!/bin/bash
# Watcher: wait until the 2026-07-20 wave drains, re-firing queue_next.sh each poll.
cd /home/vis-comp/22b3929/vidcurator
JOBS="266479 266480 266482 266816 266817 266819 266822 266823 266824"
while true; do
  LEFT=$(squeue -u "$(whoami)" -h -o %i | grep -cFf <(tr ' ' '\n' <<< "$JOBS") || true)
  bash jobs/queue_next.sh >/dev/null 2>&1
  [[ "$LEFT" -eq 0 ]] && break
  sleep 300
done
echo "WATCHER: all 2026-07-20 wave jobs left the queue at $(date)"
sacct -j ${JOBS// /,} --format=JobID,JobName%16,State,Elapsed -X
