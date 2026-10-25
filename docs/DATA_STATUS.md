# Data And Model Status

Updated: June 22, 2026.

This file tracks what is actually present in the workspace and what still blocks full paper experiments. There are two execution environments: the **local workspace** (code, configs, manifests, tests, lightweight `.venv`, metadata-only data) and the **H100 workspace** (full weights, videos, frames, GPU inference). The local workspace is now end-to-end ready for everything that does not require a GPU; the H100-only items are listed at the bottom.

## Local Workspace Environment

- `.venv/` exists with Python 3.11.7 and pip 26.1.2.
- Editable `vidcurator` package installed (`pip install -e .`).
- Local deps installed: pillow, pytest, ruff, huggingface_hub, pandas, pyarrow, numpy.
- GPU deps (torch, transformers, vllm, sglang, decord, peft, trl) are NOT installed locally — they require a CUDA GPU and are H100-only.
- No `ffmpeg`, no `nvidia-smi`, no conda on this box; these are H100-only.
- `HF_TOKEN` is not set in the local shell, so gated repos cannot be downloaded from here.

## Local Workspace Readiness (verified end-to-end)

- `python -m vidcurator.run evaluate --config configs/kill_test.json` passes: `scripted_curator@2` acc=1.0 vs `uniform@2` acc=0.0 (expected kill-test signal).
- `pytest tests/` → 7 passed.
- `scripts/run_local_research_smoke.sh .venv/bin/python` runs clean: fixture zero-shot + NExT-GQA metadata-only zero-shot + paper artifacts (CSV/SVG) + qualitative traces.
- `scripts/check_research_readiness.py` reports only the H100-only items as missing (see below).

## H100 Workspace Environment

- `.conda-env/` exists with Python 3.10, ffmpeg, git-lfs, PyTorch CUDA, Transformers, Accelerate, vLLM, SGLang, PEFT, TRL, pandas, pyarrow, and qwen-vl-utils.
- PyTorch sees CUDA GPUs in the current shell.
- `torchaudio` was corrected to match PyTorch CUDA 12.8.
- Qwen2.5-VL vLLM server smoke passed with `VLLM_USE_FLASHINFER_SAMPLER=0`, including an OpenAI-compatible image request.
- `pip check` still reports `decord 0.6.0 is not supported on this platform`; the repo uses ffmpeg for decoding, so this is not blocking.

## Downloaded And Extracted Assets

| Asset | Local Path | Status |
|---|---|---|
| NExT-GQA raw | `data/raw/nextgqa/` | downloaded, including `NExTVideo/` with 5,440 MP4s |
| Video-MME raw | `data/raw/videomme/` | downloaded, 20 video zip chunks plus metadata |
| Video-MME videos | `data/videos/videomme/` | extracted, 900 MP4s |
| EgoSchema raw | `data/raw/egoschema/` | downloaded, 5 video zip chunks plus parquet metadata |
| EgoSchema videos | `data/videos/egoschema/` | extracted, 5,031 MP4s |
| NExTQA raw | `data/raw/nextqa/` | downloaded, including `videos.zip` |
| NExTQA videos | `data/videos/nextqa/` | extracted, 1,570 MP4s |
| LongVideoBench | `data/raw/longvideobench/` | repo visible, file downloads blocked by HF token permission |
| MLVU | `data/raw/mlvu/` | repo visible, file downloads blocked by HF token permission |

## Downloaded Models

| Model | Local Path | Status |
|---|---|---|
| Qwen2.5-VL-7B-Instruct | `models/qwen2_5_vl_7b/` | full weights downloaded |
| Qwen3-VL-8B-Instruct | `models/qwen3_vl_8b_instruct/` | full weights downloaded |
| InternVL3.5-8B | `models/internvl3_5_8b/` | full weights downloaded |
| LLaVA-Video-7B-Qwen2 | `models/llava_video_7b_qwen2/` | full weights downloaded; verify license before final tables |

## Processed Splits

| Split | Path | Rows |
|---|---|---:|
| NExT-GQA val | `data/processed/nextgqa_val.jsonl` | 3,358 |
| NExT-GQA val 150 | `data/processed/nextgqa_val_150.jsonl` | 150 |
| Video-MME all | `data/processed/videomme_all.jsonl` | 2,700 |
| Video-MME long | `data/processed/videomme_long.jsonl` | 900 |
| Video-MME long 150 | `data/processed/videomme_long_150.jsonl` | 150 |
| EgoSchema labeled | `data/processed/egoschema_val.jsonl` | 500 |
| EgoSchema labeled 150 | `data/processed/egoschema_val_150.jsonl` | 150 |
| EgoSchema hidden test | `data/processed/egoschema_test_hidden.jsonl` | 5,031 |
| NExTQA temporal/causal | `data/processed/nextqa_temporal_val.jsonl` | 7,159 |
| NExTQA temporal/causal 150 | `data/processed/nextqa_temporal_val_150.jsonl` | 150 |

## Frame Indices Ready For Zero-Shot

| Dataset | Frames | Path | Notes |
|---|---:|---|---|
| NExT-GQA timestamp sanity | 18,240 | `data/indices/nextgqa_timestamp_frames.jsonl` | metadata-only sanity baseline |
| NExT-GQA decoded pilot | 6,232 | `data/indices/nextgqa_siglip_frames.jsonl` | 130 videos at 1 fps, SigLIP embeddings |
| Video-MME long pilot | 29,599 | `data/indices/videomme_long_siglip_frames.jsonl` | 123 videos at 0.1 fps, SigLIP embeddings |
| EgoSchema pilot | 13,500 | `data/indices/egoschema_siglip_frames.jsonl` | 150 videos at 0.5 fps, SigLIP embeddings |
| NExTQA temporal/causal pilot | 5,746 | `data/indices/nextqa_temporal_siglip_frames.jsonl` | 139 videos at 1 fps, SigLIP embeddings |

Decode manifests:

- `runs/decode_nextgqa_val_150_manifest.json`
- `runs/decode_videomme_long_150_manifest.json`
- `runs/decode_egoschema_val_150_manifest.json`
- `runs/decode_nextqa_temporal_150_manifest.json`

Extraction manifests:

- `runs/extract_videomme_manifest.json`
- `runs/extract_egoschema_manifest.json`
- `runs/extract_nextqa_manifest.json`

## Access Blockers

- The HF token (user `AmanTech1`) now has `canReadGatedRepos: True` and all six datasets are downloaded and normalized, including the previously gated LongVideoBench (1,337 val examples) and MLVU (502 test examples with ground truth).
- The local workspace has no `HF_TOKEN` persisted (it is used only as a session env var), so no secret is written into the repo.

## Local Workspace Downloaded Assets (metadata-only, no token required)

| Asset | Local Path | Status |
|---|---|---|
| NExT-GQA metadata | `data/raw/nextgqa/` | downloaded (CSV + grounding + frame2time JSON) |
| Video-MME metadata | `data/raw/videomme/` | downloaded (test parquet + subtitle.zip) |
| NExTQA metadata | `data/raw/nextqa/` | downloaded (MC test parquet) |
| EgoSchema metadata | `data/raw/egoschema/` | downloaded (Subset + MC parquets) |
| LongVideoBench metadata | `data/raw/longvideobench/` | downloaded (lvb_val.json + val parquet, gated repo, token with canReadGatedRepos) |
| MLVU metadata | `data/raw/mlvu/` | downloaded (test_question.json + ground truth JSONs, gated repo, token with canReadGatedRepos) |
| Qwen2.5-VL-7B config/tokenizer | `models/qwen2_5_vl_7b/` | metadata only (no safetensors weights) |
| Qwen3-VL-8B-Instruct config/tokenizer | `models/qwen3_vl_8b_instruct/` | metadata only (no safetensors weights) |
| InternVL3.5-8B config/tokenizer | `models/internvl3_5_8b/` | metadata only (no safetensors weights) |
| LLaVA-Video-7B-Qwen2 config/tokenizer | `models/llava_video_7b_qwen2/` | metadata only (no safetensors weights) |

Local weights are intentionally not pulled: each model is ~15 GB, there is no GPU on this box to serve them, and the H100 workspace already holds the full weights (see table below).

## Local Workspace Processed Splits (re-derived from metadata)

| Split | Path | Rows |
|---|---|---:|
| NExT-GQA val | `data/processed/nextgqa_val.jsonl` | 3,358 |
| NExT-GQA val 150 | `data/processed/nextgqa_val_150.jsonl` | 150 |
| Video-MME all | `data/processed/videomme_all.jsonl` | 2,700 |
| Video-MME long | `data/processed/videomme_long.jsonl` | 900 |
| Video-MME long 150 | `data/processed/videomme_long_150.jsonl` | 150 |
| EgoSchema labeled | `data/processed/egoschema_val.jsonl` | 500 |
| EgoSchema labeled 150 | `data/processed/egoschema_val_150.jsonl` | 150 |
| EgoSchema hidden test | `data/processed/egoschema_test_hidden.jsonl` | 5,031 |
| NExTQA temporal/causal | `data/processed/nextqa_temporal_val.jsonl` | 7,159 |
| NExTQA temporal/causal 150 | `data/processed/nextqa_temporal_val_150.jsonl` | 150 |
| LongVideoBench val | `data/processed/longvideobench_val.jsonl` | 1,337 |
| LongVideoBench val 150 | `data/processed/longvideobench_val_150.jsonl` | 150 |
| MLVU val | `data/processed/mlvu_val.jsonl` | 502 |
| MLVU val 150 | `data/processed/mlvu_val_150.jsonl` | 150 |
| NExT-GQA timestamp frame index | `data/indices/nextgqa_timestamp_frames.jsonl` | 18,240 |

## H100-Only Blockers (cannot be completed on the local workspace)

These require the GPU box, disk, ffmpeg, and/or an HF token with gated-repo access:

1. **Model weights** — full safetensors for Qwen2.5-VL-7B, Qwen3-VL-8B-Instruct, InternVL3.5-8B, LLaVA-Video-7B-Qwen2 (and Qwen3-VL-8B-Thinking if it exists). Already present on the H100 workspace per the June 21 state below.
2. **Video files** — NExT-GQA / Video-MME / EgoSchema / NExTQA MP4s. Already extracted on the H100 workspace. LongVideoBench and MLVU video tar files need download with the HF token (gated, large — ~20 GB each).
3. **Frame decoding + SigLIP embeddings** — needs ffmpeg + GPU. Already built on the H100 workspace for 4 datasets; LongVideoBench and MLVU need decoding after video download. Run via `sbatch jobs/02_decode_index_l40.sh`.
4. **Real VLM zero-shot runs** — needs served Qwen2.5-VL / Qwen3-VL via vLLM/SGLang on GPU. Run via `sbatch jobs/03a_zero_shot_qwen2_5_l40.sh` or `sbatch jobs/03c_zero_shot_dgx.sh`.
5. **LoRA-GRPO** — needs GPU + verl. Only after zero-shot kill tests pass. Run via `sbatch jobs/04_grpo_dgx.sh`.

## SLURM Job Pipeline

All GPU experiments run through SLURM with proper logging to `logs/`:

```text
jobs/01_prepare_data.sh          CPU     download + normalize + kill splits
jobs/02_decode_index_l40.sh      L40 1G  decode frames + SigLIP embeddings
jobs/03a_zero_shot_qwen2_5_l40   L40 1G  serve Qwen2.5-VL + zero-shot matrix
jobs/03b_zero_shot_qwen3_l40     L40 1G  serve Qwen3-VL + zero-shot matrix
jobs/03c_zero_shot_dgx           DGX 2G  serve both + full zero-shot matrix
jobs/04_grpo_dgx                 DGX 4G  LoRA-GRPO training
jobs/05_export_artifacts         CPU     CSV/SVG/trace export from all runs
jobs/submit_pipeline_l40.sh      wrapper: submit full L40 pipeline with deps
jobs/submit_pipeline_dgx.sh      wrapper: submit full DGX pipeline with deps
```

Quick start:
```bash
export HF_TOKEN=<token>
bash jobs/submit_pipeline_l40.sh   # or submit_pipeline_dgx.sh
```

## Next Data Actions

1. On H100: run Qwen2.5-VL and Qwen3-VL zero-shot configs on NExT-GQA, Video-MME long, EgoSchema, and NExTQA.
2. Fix the HF token permission for public gated repositories, then rerun `scripts/download_priority_assets.py --datasets --allow-large-video --only longvideobench mlvu`.
3. Generate VLM captions for selected frame indices if caption-aware retrieval outperforms raw SigLIP+metadata.
4. Freeze split manifests after first accepted zero-shot numbers.
