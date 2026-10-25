#!/bin/bash
# X1: download LVBench videos (360p mp4) listed in needed_videos.txt; resumable, throttle-friendly.
cd /home/vis-comp/22b3929/vidcurator
YT=$HOME/miniconda3/envs/vidcurator/bin/yt-dlp
ok=0; fail=0
while read -r vid; do
  [ -z "$vid" ] && continue
  out="data/lvbench/videos/${vid}.mp4"
  [ -s "$out" ] && { ok=$((ok+1)); continue; }
  if $YT -f 'bestvideo[height<=360][ext=mp4]/best[height<=360]' --no-playlist \
     --sleep-interval 3 --max-sleep-interval 8 -o "$out" "https://www.youtube.com/watch?v=$vid" \
     >> logs/x1_download.log 2>&1; then ok=$((ok+1)); else fail=$((fail+1)); echo "$vid" >> data/lvbench/failed_videos.txt; fi
done < data/lvbench/needed_videos.txt
echo "DOWNLOAD COMPLETE ok=$ok fail=$fail" | tee -a logs/x1_download.log
