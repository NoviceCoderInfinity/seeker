#!/bin/bash
cd /home/vis-comp/22b3929/vidcurator
JOBS="267343 267344"
while true; do
  LEFT=$(squeue -u "$(whoami)" -h -o %i | grep -cFf <(tr ' ' '\n' <<< "$JOBS") || true)
  [[ "$LEFT" -eq 0 ]] && break
  sleep 600
done
echo "WATCHER: M2 caption-variant jobs drained at $(date)"
sacct -j ${JOBS// /,} --format=JobID,JobName%16,State,Elapsed -X
for f in reports/rft_captions_nextgqa.jsonl reports/rft_captions_egoschema.jsonl; do
  echo "-- $f: $(wc -l < "$f" 2>/dev/null || echo 0) videos"
done
