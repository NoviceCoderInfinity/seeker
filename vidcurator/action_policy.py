"""Load and run the LoRA action policy (SFT / GRPO curator adapter)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .grpo import _SYSTEM_PROMPT, _make_user_turn

KNOWN_ACTIONS = {"probe", "retrieve", "answer", "keep", "compress", "drop", "abstain"}
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    candidates = []
    match = _JSON_RE.search(text)
    if match:
        candidates.append(match.group(0))
    candidates.append(text.strip())
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict) and "action" in obj:
            return obj
    return None


def load_action_model(model_path: str, adapter_path: str):
    import os

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    cfg = Path(model_path) / "config.json"
    mtype = json.loads(cfg.read_text()).get("model_type", "") if cfg.exists() else ""
    kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
    device = os.environ.get("ACTION_POLICY_DEVICE", "cuda").lower()
    if device == "cpu" or not torch.cuda.is_available():
        kwargs["device_map"] = "cpu"
    elif device.startswith("cuda:"):
        # explicit GPU index, e.g. "cuda:1" to keep the adapter off the served VLM's GPU
        kwargs["device_map"] = {"": int(device.split(":", 1)[1])}
    else:
        kwargs["device_map"] = {"": 0}

    if mtype == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    elif mtype == "qwen3_vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError:
            from transformers import Qwen3_VLForConditionalGeneration as Qwen3VLForConditionalGeneration

        base = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    else:
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


class ActionPolicyModel:
    """Text-only curator action generator (probe / retrieve / answer JSON)."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 384, temperature: float = 0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        # temperature > 0 enables sampling (self-consistency trajectories); 0 = greedy.
        self.temperature = temperature

    def generate(self, messages: Sequence[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        import torch

        input_ids = self.tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        gen_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=float(self.temperature), top_p=0.95)
        else:
            gen_kwargs["do_sample"] = False
        with torch.inference_mode():
            out = self.model.generate(**gen_kwargs)
        new_tokens = out[0][input_ids.shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return parse_action(text)

    def build_turn_messages(
        self,
        question: str,
        options: List[str],
        held_frames: List[str],
        running_summary: str,
        budget: int,
        hop: int,
    ) -> List[Dict[str, str]]:
        user_text = _make_user_turn(question, options, held_frames, running_summary, budget, hop)
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]


def build_action_policy(model_path: str, adapter_path: str, max_new_tokens: int = 384) -> ActionPolicyModel:
    model, tokenizer = load_action_model(model_path, adapter_path)
    return ActionPolicyModel(model, tokenizer, max_new_tokens=max_new_tokens)
