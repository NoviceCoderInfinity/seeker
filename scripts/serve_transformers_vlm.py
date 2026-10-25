#!/usr/bin/env python3
"""Lightweight OpenAI-compatible VLM server using transformers.

This avoids the heavy vllm dependency and works with any HuggingFace VLM.
Supports multimodal (text + image) inputs via the OpenAI chat completions API.

Usage:
    python scripts/serve_transformers_vlm.py --model models/qwen2_5_vl_7b --port 8000
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import tempfile
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any


def load_model_and_processor(
    model_path: str,
    dtype: str = "bfloat16",
    min_pixels: int | None = None,
    max_pixels: int | None = None,
):
    import torch
    from transformers import AutoProcessor, AutoModel

    print(f"Loading model from {model_path} (dtype={dtype})...", flush=True)
    print(f"  torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    torch_dtype = getattr(torch, dtype, torch.bfloat16)

    load_kwargs = {"torch_dtype": torch_dtype, "trust_remote_code": True}
    if torch.cuda.is_available():
        # VLM_DEVICE_MAP=auto shards across all visible GPUs (needed for InternVL at K=32,
        # which OOMs a single 44GB card); default pins to GPU 0.
        load_kwargs["device_map"] = os.environ.get("VLM_DEVICE_MAP") or {"": 0}
    else:
        print("  WARNING: CUDA not available — model will run on CPU (very slow)", flush=True)

    # Read model_type first so we can pick the right (processor, model, generation) path.
    config_path = Path(model_path) / "config.json"
    model_type = ""
    if config_path.exists():
        model_type = json.loads(config_path.read_text()).get("model_type", "")
    print(f"  model_type: {model_type}", flush=True)

    # --- InternVL family (internvl_chat): custom .chat() API, AutoTokenizer, 448px tiles. ---
    # It does NOT use the Qwen AutoProcessor/apply_chat_template path, so it is handled
    # entirely separately from generation onward (see generate_response / _generate_internvl).
    if model_type == "internvl_chat":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        model = AutoModel.from_pretrained(model_path, **load_kwargs)
        model._is_internvl = True
        model.eval()
        device = next(model.parameters()).device
        print(f"InternVL model loaded on {device} (custom .chat() path, 448px tiles, 1 tile/frame)", flush=True)
        if torch.cuda.is_available() and device.type == "cpu":
            raise RuntimeError("CUDA is available but model loaded on CPU — aborting")
        return model, tokenizer

    # --- LLaVA-OneVision (llava_onevision): HF-native; frames must go through the VIDEO
    # path (~196 tokens/frame after pooling). The multi-image path applies anyres tiling
    # per image and explodes the visual token budget at K>=8, so generate_response routes
    # frame lists through processor(videos=...) for this family.
    if model_type == "llava_onevision":
        from transformers import LlavaOnevisionForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
        model._is_llava_ov = True
        model.eval()
        device = next(model.parameters()).device
        print(f"LLaVA-OneVision model loaded on {device} (video path, ~196 tok/frame)", flush=True)
        if torch.cuda.is_available() and device.type == "cpu":
            raise RuntimeError("CUDA is available but model loaded on CPU — aborting")
        return model, processor

    # Cap per-frame resolution so multi-frame contexts (K=16/32) do not blow up the
    # visual token budget and degrade generation. Qwen2.5/Qwen3-VL processors accept
    # min_pixels/max_pixels; each visual token covers a 28x28 patch after merge.
    proc_kwargs = {"trust_remote_code": True}
    # min/max_pixels are Qwen-processor knobs; Gemma3 resizes every image to a fixed
    # 896px square (256 visual tokens/image) and takes no pixel budget.
    if model_type != "gemma3":
        if min_pixels is not None:
            proc_kwargs["min_pixels"] = int(min_pixels)
        if max_pixels is not None:
            proc_kwargs["max_pixels"] = int(max_pixels)
    processor = AutoProcessor.from_pretrained(model_path, **proc_kwargs)
    if min_pixels is not None or max_pixels is not None:
        print(f"  pixel budget: min_pixels={min_pixels} max_pixels={max_pixels}", flush=True)

    # Try model-specific classes first, fall back to AutoModel
    model = None
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if model_type == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
        elif model_type == "qwen3_vl":
            try:
                from transformers import Qwen3VLForConditionalGeneration
            except ImportError:
                from transformers import Qwen3_VLForConditionalGeneration as Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
        elif model_type == "gemma3":
            # AutoModel would return the headless base; the conditional-generation class
            # carries the vision tower + LM head. Multi-image via the generic path is safe:
            # fixed 256 tokens/image, so even K=32 stays ~8k visual tokens.
            from transformers import Gemma3ForConditionalGeneration
            model = Gemma3ForConditionalGeneration.from_pretrained(model_path, **load_kwargs)

    if model is None:
        print("  falling back to AutoModel...", flush=True)
        model = AutoModel.from_pretrained(model_path, **load_kwargs)

    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}", flush=True)
    if torch.cuda.is_available() and device.type == "cpu":
        raise RuntimeError("CUDA is available but model loaded on CPU — aborting")
    return model, processor


def extract_content(messages: list[dict]) -> tuple[str, list[str]]:
    """Extract text and image paths from OpenAI-format messages."""
    text_parts: list[str] = []
    image_paths: list[str] = []

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part["text"])
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:"):
                        header, data = url.split(",", 1)
                        mime = header.split(":")[1].split(";")[0]
                        suffix = mimetypes.guess_extension(mime) or ".jpg"
                        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                        tmp.write(base64.b64decode(data))
                        tmp.close()
                        image_paths.append(tmp.name)
                    else:
                        image_paths.append(url)
    return "\n".join(text_parts), image_paths


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _internvl_transform(image, input_size: int = 448):
    """Standard InternVL frame preprocessing: 1 tile/frame at 448px, ImageNet-normalized.

    Matches the model's `load_video` convention (max_num=1 tile per frame) so many-frame
    contexts stay within budget instead of exploding via dynamic tiling.
    """
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    transform = T.Compose([
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transform(image.convert("RGB"))


def _generate_internvl(model, tokenizer, text: str, image_paths: list[str], max_tokens: int,
                       temperature: float = 0.0) -> str:
    """Generation via InternVLChatModel.chat() with <image> token expansion."""
    import torch
    from PIL import Image

    if temperature > 0:
        gen_config = dict(max_new_tokens=max_tokens, do_sample=True, temperature=temperature, top_p=0.95)
    else:
        gen_config = dict(max_new_tokens=max_tokens, do_sample=False)

    tiles = []
    for p in image_paths:
        try:
            if Path(p).exists():
                tiles.append(_internvl_transform(Image.open(p)))
        except Exception as e:
            print(f"  WARNING: failed to load image {p}: {e}", flush=True)

    if not tiles:
        response = model.chat(tokenizer, None, text, gen_config, history=None, return_history=False)
        return response.strip()

    dtype = next(model.parameters()).dtype
    pixel_values = torch.stack(tiles).to(model.device, dtype=dtype)
    num_patches_list = [1] * len(tiles)
    # One <image> placeholder per frame; model.chat expands each to num_patches tokens.
    prefix = "".join(f"Image-{i + 1}: <image>\n" for i in range(len(tiles)))
    question = prefix + text
    response = model.chat(
        tokenizer, pixel_values, question, gen_config,
        num_patches_list=num_patches_list, history=None, return_history=False,
    )
    return response.strip()


def _generate_llava_ov(model, processor, text: str, image_paths: list[str], max_tokens: int,
                       temperature: float = 0.0) -> str:
    """LLaVA-OneVision generation: selected frames are sent as ONE video clip.

    The video path pools each frame to ~196 visual tokens; the multi-image path would
    anyres-tile every frame (up to ~10x more tokens each) and overflow at K>=8.
    """
    import numpy as np
    import torch
    from PIL import Image

    frames = []
    for p in image_paths:
        try:
            if Path(p).exists():
                frames.append(np.array(Image.open(p).convert("RGB")))
        except Exception as e:
            print(f"  WARNING: failed to load image {p}: {e}", flush=True)

    if frames:
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": text}]}]
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if frames:
        inputs = processor(text=chat_text, videos=[frames], return_tensors="pt").to(model.device)
    else:
        inputs = processor(text=chat_text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        if temperature > 0:
            output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True,
                                        temperature=temperature, top_p=0.95)
        else:
            output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


def generate_response(model, processor, text: str, image_paths: list[str], max_tokens: int = 128,
                      temperature: float = 0.0) -> str:
    import torch

    if getattr(model, "_is_internvl", False):
        return _generate_internvl(model, processor, text, image_paths, max_tokens, temperature)

    if getattr(model, "_is_llava_ov", False):
        return _generate_llava_ov(model, processor, text, image_paths, max_tokens, temperature)

    images = None
    if image_paths:
        from PIL import Image
        images = []
        for p in image_paths:
            try:
                if Path(p).exists():
                    images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"  WARNING: failed to load image {p}: {e}", flush=True)
        if not images:
            images = None

    if images:
        messages = [{"role": "user", "content": [{"type": "text", "text": text}] + [{"type": "image"} for _ in images]}]
    else:
        messages = [{"role": "user", "content": text}]

    try:
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        chat_text = text

    if images:
        inputs = processor(text=chat_text, images=images, return_tensors="pt", padding=True).to(model.device)
    else:
        inputs = processor(text=chat_text, return_tensors="pt", padding=True).to(model.device)

    with torch.inference_mode():
        if temperature > 0:
            output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True,
                                        temperature=temperature, top_p=0.95)
        else:
            output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, temperature=0.0)

    input_len = inputs.get("input_ids", inputs.get("input_ids", None))
    if input_len is not None:
        input_len = input_len.shape[1]
        response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    else:
        response = processor.decode(output_ids[0], skip_special_tokens=True)

    return response.strip()


class VLMHandler(BaseHTTPRequestHandler):
    server_version = "VLMServer/1.0"
    model = None
    processor = None
    model_id = "unknown"

    def _send_json(self, code: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [{"id": self.model_id, "object": "model"}]})
        elif self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            payload = json.loads(body)
        except Exception as e:
            self._send_json(400, {"error": f"bad request: {e}"})
            return

        messages = payload.get("messages", [])
        max_tokens = payload.get("max_tokens", 128)
        temperature = float(payload.get("temperature", 0.0) or 0.0)
        text, image_paths = extract_content(messages)

        t0 = time.time()
        try:
            response_text = generate_response(self.model, self.processor, text, image_paths, max_tokens, temperature)
            elapsed = time.time() - t0
            print(f"  generated in {elapsed:.1f}s: {response_text[:100]}", flush=True)
        except Exception as e:
            print(f"  GENERATION ERROR: {e}", flush=True)
            self._send_json(500, {"error": str(e)})
            return
        finally:
            for p in image_paths:
                if p.startswith(tempfile.gettempdir()):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

        self._send_json(200, {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "model": self.model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    def log_message(self, fmt, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path or HF id")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=int(os.environ.get("VLM_MAX_PIXELS", 200704)),
        help="Max pixels per frame after resize (default 200704 = 256 visual tokens / 448x448).",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=int(os.environ.get("VLM_MIN_PIXELS", 3136)),
        help="Min pixels per frame after resize (default 3136 = 4 visual tokens).",
    )
    args = parser.parse_args()

    VLMHandler.model, VLMHandler.processor = load_model_and_processor(
        args.model, args.dtype, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )
    VLMHandler.model_id = args.model

    server = HTTPServer((args.host, args.port), VLMHandler)
    print(f"Server listening on {args.host}:{args.port}", flush=True)
    print(f"  GET  /v1/models", flush=True)
    print(f"  POST /v1/chat/completions", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
