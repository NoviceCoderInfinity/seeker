#!/usr/bin/env python3
"""Robust HF dataset/model downloader with 429 rate-limit backoff + resume.

snapshot_download resumes (skips files already on disk), so on a 429 we just
sleep and retry; eventually all files land. xet is disabled because its
xet-read-token endpoint burns the 1000-req/5min API quota very fast.

Usage:
  python scripts/robust_download.py --datasets nextgqa videomme_long longvideobench mlvu
  python scripts/robust_download.py --models internvl3_5_8b llava_video_7b_qwen2
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")


def load_manifest(kind: str) -> dict:
    fn = "manifests/datasets.json" if kind == "dataset" else "manifests/models.json"
    key = "datasets" if kind == "dataset" else "models"
    rows = json.load(open(fn))[key]
    return {r["name"]: r for r in rows}


VIDEO_IGNORE_NONE = None
META_IGNORE = ["*.mp4", "*.mkv", "*.webm", "*.mov"]


def download_one(name: str, hf_id: str, repo_type: str, local_dir: Path,
                 token: str, allow_video: bool, max_retries: int = 40) -> bool:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    local_dir.mkdir(parents=True, exist_ok=True)
    ignore = None if allow_video else META_IGNORE
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            snapshot_download(
                repo_id=hf_id,
                repo_type=repo_type,
                local_dir=str(local_dir),
                token=token,
                ignore_patterns=ignore,
                max_workers=4,
            )
            print(f"[ok] {name}: complete -> {local_dir}", flush=True)
            return True
        except HfHubHTTPError as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code == 429 or "429" in str(exc) or "rate limit" in str(exc).lower():
                wait = min(300, 30 * attempt)
                print(f"[429] {name}: rate-limited (attempt {attempt}/{max_retries}); "
                      f"sleeping {wait}s then resuming...", flush=True)
                time.sleep(wait)
                continue
            print(f"[err] {name}: {exc}", flush=True)
            time.sleep(30)
        except Exception as exc:  # transient network etc.
            print(f"[retry] {name}: {type(exc).__name__}: {exc}; sleeping 30s", flush=True)
            time.sleep(30)
    print(f"[FAIL] {name}: exhausted {max_retries} retries", flush=True)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--no-video", action="store_true", help="metadata only (no mp4)")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set")

    ok = True
    if args.datasets:
        ds = load_manifest("dataset")
        for name in args.datasets:
            row = ds[name]
            print(f"=== dataset {name} ({row['hf_id']}) ===", flush=True)
            ok &= download_one(name, row["hf_id"], row.get("hf_type", "dataset"),
                               Path(row["raw_dir"]), token, allow_video=not args.no_video)
    if args.models:
        ms = load_manifest("model")
        for name in args.models:
            row = ms[name]
            print(f"=== model {name} ({row['hf_id']}) ===", flush=True)
            ok &= download_one(name, row["hf_id"], "model",
                               Path("models") / name, token, allow_video=True)
    print("ALL_DOWNLOADS_DONE" if ok else "SOME_DOWNLOADS_FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
