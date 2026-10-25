#!/usr/bin/env python3
"""Validate the SFT cold-start LoRA adapter for the curator action policy.

The cold-start's job is to teach the base VLM to emit the strict action schema
(probe / retrieve / answer ... as JSON). This script measures, on the held-out
SFT val conversations, how well the adapter does that vs the un-adapted base:

  - format_valid_rate : output parses as JSON with a known "action" field
  - action_match_rate : predicted action type == gold (teacher) action type
  - per-action breakdown

A large format/action gap between adapter and base is the signal that the
cold-start worked and that GRPO will start from a well-formed policy.

Usage:
    python scripts/validate_sft_adapter.py \\
        --model models/qwen2_5_vl_7b \\
        --adapter runs/sft/qwen2_5_vl_7b_curator_lora \\
        --val data/rl/vidcurator_val_sft.jsonl \\
        --limit 200 --compare-base \\
        --output runs/sft/qwen2_5_vl_7b_curator_lora/validation.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

KNOWN_ACTIONS = {"probe", "retrieve", "answer", "keep", "compress", "drop", "abstain"}
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_conversations(path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object with a known 'action' field."""
    candidates = []
    m = _JSON_RE.search(text)
    if m:
        candidates.append(m.group(0))
    candidates.append(text.strip())
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict) and "action" in obj:
            return obj
    return None


def gold_action(messages: List[Dict[str, Any]]) -> Optional[str]:
    target = messages[-1]["content"]
    obj = parse_action(target)
    if obj is None:
        return None
    return str(obj.get("action", "")).lower()


def generate(model, tokenizer, messages, max_new_tokens: int) -> str:
    import torch

    prompt_msgs = messages[:-1]
    input_ids = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate(model, tokenizer, rows, max_new_tokens: int, label: str, dump_failures: int = 0) -> Dict[str, Any]:
    from collections import Counter

    n = len(rows)
    fmt_valid = 0
    action_match = 0
    pred_actions = Counter()
    per_gold_match = Counter()
    per_gold_total = Counter()
    failure_samples: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        messages = row["messages"]
        g = gold_action(messages)
        if g is not None:
            per_gold_total[g] += 1
        text = generate(model, tokenizer, messages, max_new_tokens)
        obj = parse_action(text)
        valid = obj is not None and str(obj.get("action", "")).lower() in KNOWN_ACTIONS
        if valid:
            fmt_valid += 1
            pred = str(obj["action"]).lower()
            pred_actions[pred] += 1
            if g is not None and pred == g:
                action_match += 1
                per_gold_match[g] += 1
        elif len(failure_samples) < dump_failures:
            # Heuristic: looks_truncated if it starts a JSON object but never closes.
            stripped = text.strip()
            looks_truncated = stripped.startswith("{") and stripped.count("{") > stripped.count("}")
            failure_samples.append({
                "gold_action": g,
                "output_len_chars": len(text),
                "looks_truncated": looks_truncated,
                "output": text[:600],
            })
        if (i + 1) % 25 == 0:
            print(f"  [{label}] {i+1}/{n} | fmt_valid={fmt_valid/(i+1):.3f} match={action_match/(i+1):.3f}", flush=True)
    report = {
        "label": label,
        "n": n,
        "format_valid_rate": round(fmt_valid / n, 4) if n else 0.0,
        "action_match_rate": round(action_match / n, 4) if n else 0.0,
        "pred_action_dist": dict(pred_actions),
        "per_action_match": {
            a: round(per_gold_match[a] / per_gold_total[a], 4)
            for a in sorted(per_gold_total) if per_gold_total[a]
        },
        "per_action_support": dict(per_gold_total),
    }
    if dump_failures:
        report["failure_samples"] = failure_samples
        report["failures_truncated"] = sum(1 for f in failure_samples if f["looks_truncated"])
    return report


def load_base(model_path: str):
    import json as _json
    import torch
    from transformers import AutoModelForCausalLM

    cfg = Path(model_path) / "config.json"
    mtype = _json.loads(cfg.read_text()).get("model_type", "") if cfg.exists() else ""
    kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
    if mtype == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    if mtype == "qwen3_vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError:
            from transformers import Qwen3_VLForConditionalGeneration as Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--compare-base", action="store_true")
    ap.add_argument("--dump-failures", type=int, default=0,
                    help="Save up to N format-invalid generations (with truncation heuristic)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_conversations(args.val, args.limit)
    print(f"loaded {len(rows)} val conversations", flush=True)

    print("loading base model...", flush=True)
    base = load_base(args.model)
    base.eval()

    results: Dict[str, Any] = {}
    if args.compare_base:
        print("evaluating BASE (no adapter)...", flush=True)
        results["base"] = evaluate(base, tokenizer, rows, args.max_new_tokens, "base", args.dump_failures)

    print(f"loading adapter from {args.adapter}...", flush=True)
    adapted = PeftModel.from_pretrained(base, args.adapter)
    adapted.eval()
    print("evaluating ADAPTER...", flush=True)
    results["adapter"] = evaluate(adapted, tokenizer, rows, args.max_new_tokens, "adapter", args.dump_failures)

    print("\n=== SFT VALIDATION SUMMARY ===")
    for key in ("base", "adapter"):
        if key in results:
            r = results[key]
            print(f"[{key:7s}] format_valid={r['format_valid_rate']:.3f} "
                  f"action_match={r['action_match_rate']:.3f}  pred_dist={r['pred_action_dist']}")
    if "base" in results and "adapter" in results:
        df = results["adapter"]["format_valid_rate"] - results["base"]["format_valid_rate"]
        dm = results["adapter"]["action_match_rate"] - results["base"]["action_match_rate"]
        print(f"[delta  ] format_valid {df:+.3f} | action_match {dm:+.3f}")
        results["delta"] = {"format_valid_rate": round(df, 4), "action_match_rate": round(dm, 4)}

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
