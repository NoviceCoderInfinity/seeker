#!/usr/bin/env python3
"""LoRA SFT cold-start for the VID-CURATOR action policy (Stage-1 pre-GRPO).

Teaches the base VLM the curator action schema (probe / retrieve / answer as
strict JSON) from scripted-curator traces produced by export_sft_traces.py.
This is the cold-start that precedes LoRA-GRPO (docs/GRPO_PLAN.md): GRPO is far
more stable when the policy already emits well-formed actions.

The traces are text-only conversations (system + user state + assistant action),
so this trains a LoRA adapter on the language-model attention projections of the
VLM. The adapter is loadable on the same Qwen2.5-VL checkpoint at inference; the
vision tower stays untouched and frozen.

Label masking: only assistant-response tokens contribute to the loss; the system
prompt and the user state are masked out (-100).

Usage:
    python scripts/sft_train.py \\
        --model models/qwen2_5_vl_7b \\
        --train data/rl/vidcurator_train_sft.jsonl \\
        --val   data/rl/vidcurator_val_sft.jsonl \\
        --output runs/sft/qwen2_5_curator_lora \\
        --epochs 2 --lora-rank 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_conversations(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_examples(rows, tokenizer, max_len: int):
    """Tokenize (system,user,assistant) triples with assistant-only labels."""
    import torch

    features = []
    skipped = 0
    for row in rows:
        messages = row["messages"]
        # Split into prompt (everything up to and incl. the generation prompt)
        # and the assistant target so we can mask the prompt tokens.
        if not messages or messages[-1]["role"] != "assistant":
            skipped += 1
            continue
        prompt_msgs = messages[:-1]
        target_text = messages[-1]["content"]

        prompt_ids = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=True, add_generation_prompt=True
        )
        target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
        eos = tokenizer.eos_token_id
        if eos is not None:
            target_ids = target_ids + [eos]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + list(target_ids)
        if len(input_ids) > max_len:
            # Truncate from the left of the prompt, keep the full target.
            overflow = len(input_ids) - max_len
            if overflow >= len(prompt_ids):
                skipped += 1
                continue
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]
        features.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        })
    print(f"  built {len(features)} examples (skipped {skipped})", flush=True)
    return features


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch

        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def load_model(model_path: str):
    import json as _json
    import torch
    from transformers import AutoModelForCausalLM

    config_path = Path(model_path) / "config.json"
    model_type = ""
    if config_path.exists():
        model_type = _json.loads(config_path.read_text()).get("model_type", "")

    kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}

    if model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    elif model_type == "qwen3_vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError:
            from transformers import Qwen3_VLForConditionalGeneration as Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading datasets...", flush=True)
    train_rows = load_conversations(args.train)
    train_feats = build_examples(train_rows, tokenizer, args.max_len)
    eval_feats = None
    if args.val and Path(args.val).exists():
        eval_feats = build_examples(load_conversations(args.val), tokenizer, args.max_len)

    print("loading model...", flush=True)
    model = load_model(args.model)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=20,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
        seed=args.seed,
        eval_strategy="steps" if eval_feats else "no",
        eval_steps=args.save_steps if eval_feats else None,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_feats,
        eval_dataset=eval_feats,
        data_collator=Collator(tokenizer.pad_token_id),
    )

    print("starting SFT...", flush=True)
    trainer.train()
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"saved LoRA adapter to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
