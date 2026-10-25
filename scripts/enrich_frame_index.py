#!/usr/bin/env python3
"""Add optional CLIP/SigLIP embeddings to a frame index JSONL.

This is the H100/GPU indexing pass. It keeps the base schema stable and only
fills the `embedding` and metadata fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def as_feature_tensor(outputs):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - already checked by caller
        raise RuntimeError("torch is required") from exc
    if isinstance(outputs, torch.Tensor):
        return outputs
    for attr in ("image_embeds", "pooler_output", "last_hidden_state"):
        value = getattr(outputs, attr, None)
        if isinstance(value, torch.Tensor):
            if attr == "last_hidden_state" and value.ndim == 3:
                return value[:, 0]
            return value
    if isinstance(outputs, (tuple, list)) and outputs:
        return as_feature_tensor(outputs[0])
    raise TypeError(f"cannot extract image features from {type(outputs)!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="google/siglip-base-patch16-224")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise SystemExit(
            "Install GPU deps first: pip install torch transformers pillow"
        ) from exc

    rows = load_rows(Path(args.input))
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device)
    model.eval()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    enriched = 0
    with out.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            images = []
            valid_indices = []
            for idx, row in enumerate(batch):
                path = Path(row.get("path", ""))
                if not path.exists():
                    continue
                images.append(Image.open(path).convert("RGB"))
                valid_indices.append(idx)
            if images:
                inputs = processor(images=images, return_tensors="pt").to(args.device)
                outputs = model.get_image_features(**inputs) if hasattr(model, "get_image_features") else model(**inputs)
                features = as_feature_tensor(outputs)
                outputs = torch.nn.functional.normalize(features, dim=-1).detach().cpu().float().tolist()
                for local_idx, embedding in zip(valid_indices, outputs):
                    batch[local_idx]["embedding"] = embedding
                    metadata = dict(batch[local_idx].get("metadata", {}))
                    metadata["embedding_model"] = args.model
                    batch[local_idx]["metadata"] = metadata
                    enriched += 1
            for row in batch:
                handle.write(json.dumps(row, sort_keys=True))
                handle.write("\n")
    print(f"wrote {len(rows)} rows to {out}; enriched {enriched} with embeddings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
