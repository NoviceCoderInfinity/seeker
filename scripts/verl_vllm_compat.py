"""Runtime compatibility shims for verl + vLLM on the cluster.

verl 0.8+ imports ``run_headless`` from ``vllm.entrypoints.cli.serve`` at
import time, but vLLM wheels below 0.12 do not export that symbol. Single-node
GRPO on DGX never calls ``run_headless`` (node_rank == 0), so a stub is safe.

Loaded via PYTHONSTARTUP from jobs/04_grpo_dgx.sh so Ray worker subprocesses
also get the patch.
"""

from __future__ import annotations


def _patch_run_headless() -> None:
    try:
        from vllm.entrypoints.cli.serve import run_headless  # noqa: F401
        return
    except ImportError:
        pass

    import vllm.entrypoints.cli.serve as serve_mod

    def run_headless(args):  # type: ignore[no-untyped-def]
        raise RuntimeError(
            "vLLM run_headless is unavailable in this vLLM build; "
            "multi-node rollout is not supported. Use a single-node DGX job."
        )

    serve_mod.run_headless = run_headless


def _patch_flash_attn_bert_padding() -> None:
    """verl attention_utils imports flash_attn on CUDA; cluster env has no flash_attn wheel."""
    try:
        from flash_attn.bert_padding import index_first_axis  # noqa: F401
        return
    except ImportError:
        pass

    import sys
    import types

    from verl.utils.npu_flash_attn_utils import (
        index_first_axis,
        pad_input,
        rearrange,
        unpad_input,
    )

    flash_attn = types.ModuleType("flash_attn")
    bert_padding = types.ModuleType("flash_attn.bert_padding")
    bert_padding.index_first_axis = index_first_axis
    bert_padding.pad_input = pad_input
    bert_padding.rearrange = rearrange
    bert_padding.unpad_input = unpad_input
    flash_attn.bert_padding = bert_padding
    sys.modules["flash_attn"] = flash_attn
    sys.modules["flash_attn.bert_padding"] = bert_padding


_patch_run_headless()
_patch_flash_attn_bert_padding()
