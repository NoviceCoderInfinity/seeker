#!/usr/bin/env python3
"""Embed QA questions into the SigLIP text space for the label-free FEP fix (T1.b).

The deployable ``curator_siglip`` policy retrieves frames by cosine similarity in
the shared SigLIP space: cosine(question_text_embedding, frame_image_embedding).
This script produces the question-side embedding and stores it on each example
under ``metadata['query_embedding']``, so retrieval relevance comes from real
vision-language similarity rather than caption-lexical overlap — WITHOUT ever
reading gold evidence spans.

CRITICAL: use the SAME SigLIP model here that was used to embed the frame index
(see scripts/enrich_frame_index.py), otherwise the image/text embeddings live in
different spaces and cosine similarity is meaningless. The script records the
model id under metadata['query_embedding_model'] for auditability.

Usage:
    python scripts/embed_questions_siglip.py \\
        --input data/processed/nextgqa_val_150.jsonl \\
        --output data/processed/nextgqa_val_150_siglip.jsonl \\
        --model models/siglip_base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def load_rows(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_query_text(row: dict, include_options: bool, max_chars: int) -> str:
    question = str(row.get("question", "")).strip()
    if not include_options:
        return question[:max_chars]
    options = row.get("options", []) or []
    opt_text = " ".join(str(o) for o in options)
    combined = f"{question} {opt_text}".strip()
    return combined[:max_chars]


def as_text_features(model, inputs):
    """Extract pooled text features from a SigLIP/CLIP-style model."""
    if hasattr(model, "get_text_features"):
        return model.get_text_features(**inputs)
    outputs = model(**inputs)
    for attr in ("text_embeds", "pooler_output", "last_hidden_state"):
        value = getattr(outputs, attr, None)
        if value is not None:
            if attr == "last_hidden_state" and value.ndim == 3:
                return value[:, 0]
            return value
    raise TypeError(f"cannot extract text features from {type(outputs)!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="QA JSONL (project schema)")
    parser.add_argument("--output", required=True, help="Output QA JSONL with query_embedding")
    parser.add_argument(
        "--model", default="models/siglip_base",
        help=(
            "SigLIP model id/path — MUST match the frame-index embedding model. "
            "The existing indices were built with models/siglip_base (768-dim); use "
            "the same here. For the SigLIP-so400m re-index upgrade (plan T1.b), "
            "re-run enrich_frame_index.py AND this script with the so400m model."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--include-options", action="store_true",
        help="Fuse options into the query text (default: question only)",
    )
    parser.add_argument(
        "--max-chars", type=int, default=300,
        help="Truncate query text before tokenization (SigLIP text len is short)",
    )
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise SystemExit(
            "Install GPU deps first: pip install torch transformers"
        ) from exc

    rows = load_rows(Path(args.input))
    if not rows:
        print(f"no rows in {args.input}")
        return 1

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device)
    model.eval()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    embedded = 0
    with out.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            texts = [build_query_text(row, args.include_options, args.max_chars) for row in batch]
            # SigLIP text encoder expects fixed-length padding.
            inputs = processor(
                text=texts,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(args.device)
            features = as_text_features(model, inputs)
            features = torch.nn.functional.normalize(features, dim=-1).detach().cpu().float().tolist()
            for row, embedding in zip(batch, features):
                metadata = dict(row.get("metadata", {}))
                metadata["query_embedding"] = embedding
                metadata["query_embedding_model"] = args.model
                metadata["query_embedding_includes_options"] = bool(args.include_options)
                row["metadata"] = metadata
                embedded += 1
            for row in batch:
                handle.write(json.dumps(row, sort_keys=True))
                handle.write("\n")
            print(f"  embedded {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    print(f"wrote {len(rows)} rows to {out}; embedded {embedded} questions with {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
