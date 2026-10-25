#!/usr/bin/env python3
"""One-shot GPU smoke test for the InternVL serving adapter.

Loads models/internvl3_5_8b via serve_transformers_vlm.load_model_and_processor,
feeds 2 real frames + a question through generate_response, and prints the reply.
Non-HTTP: exercises exactly the code path the server uses.
"""
import sys
from pathlib import Path

ROOT = Path("/home/vis-comp/22b3929/vidcurator")
sys.path.insert(0, str(ROOT / "scripts"))
import serve_transformers_vlm as srv

FRAME_DIR = ROOT / "data/frames/egoschema_val_150/8670020b-15c9-4ea9-acb5-71613dac2e9d"
frames = sorted(str(p) for p in FRAME_DIR.glob("*.jpg"))[:2]

print(f"frames: {frames}", flush=True)
model, tok = srv.load_model_and_processor("models/internvl3_5_8b", dtype="bfloat16")
print(f"_is_internvl = {getattr(model, '_is_internvl', False)}", flush=True)

q = "Briefly, in one sentence, what is happening in these frames? Then answer with the option letter. A. cooking  B. driving  C. gardening  D. reading"
out = srv.generate_response(model, tok, q, frames, max_tokens=64)
print("=" * 60, flush=True)
print("RESPONSE:", repr(out), flush=True)
print("=" * 60, flush=True)
assert out and len(out.strip()) > 0, "empty response — adapter FAILED"
print("SMOKE OK: non-empty coherent response from InternVL adapter", flush=True)
