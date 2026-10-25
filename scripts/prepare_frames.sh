#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: scripts/prepare_frames.sh <video_dir> <out_frame_dir> <fps>" >&2
  exit 2
fi

video_dir="$1"
out_dir="$2"
fps="$3"

command -v ffmpeg >/dev/null 2>&1 || {
  echo "ffmpeg is required for real video decoding" >&2
  exit 1
}

mkdir -p "$out_dir"

find "$video_dir" -type f \( -name '*.mp4' -o -name '*.mkv' -o -name '*.webm' -o -name '*.mov' \) | sort | while read -r video; do
  base="$(basename "$video")"
  stem="${base%.*}"
  target="$out_dir/$stem"
  mkdir -p "$target"
  ffmpeg -hide_banner -loglevel error -y -i "$video" -vf "fps=$fps" -q:v 2 "$target/frame_%06d.jpg"
  echo "decoded $video -> $target"
done

cat <<'NOTE'
Next step: create data/indices/<dataset>_frames.jsonl with fields:
video_id, frame_id, timestamp, path, caption, embedding, metadata.
Use SigLIP/CLIP or the VLM vision tower for embeddings; keep captions if using the dependency-free retriever.
NOTE
