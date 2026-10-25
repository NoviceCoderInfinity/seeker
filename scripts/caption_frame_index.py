#!/usr/bin/env python3
"""Caption frames in a frame index through an OpenAI-compatible VLM endpoint."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def image_content(path: Path) -> dict[str, Any]:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": "low"}}


def post(endpoint: str, payload: dict[str, Any], timeout: float, api_key_env: str) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(api_key_env)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    prompt = (
        "Describe the visual evidence in this video frame in one compact sentence. "
        "Mention people, objects, actions, scene text, and state changes if visible."
    )
    done = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            path = Path(row.get("path", ""))
            if path.exists():
                payload = {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}, image_content(path)],
                        }
                    ],
                    "temperature": 0.0,
                    "max_tokens": 80,
                }
                try:
                    raw = post(args.endpoint, payload, args.timeout_seconds, args.api_key_env)
                    caption = raw["choices"][0]["message"]["content"].strip()
                    row["caption"] = caption
                    metadata = dict(row.get("metadata", {}))
                    metadata["caption_model"] = args.model
                    row["metadata"] = metadata
                    done += 1
                except (urllib.error.URLError, KeyError, TimeoutError) as exc:
                    metadata = dict(row.get("metadata", {}))
                    metadata["caption_error"] = str(exc)
                    row["metadata"] = metadata
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)
    print(f"wrote {len(rows)} rows to {out}; captioned {done}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
