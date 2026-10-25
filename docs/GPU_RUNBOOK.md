# GPU Experiment Runbook

This is the operational path from downloaded assets to paper-grade zero-shot and training runs.

## SLURM Job Pipeline (L40 / DGX)

All GPU experiments run through SLURM. The numbered job scripts in `jobs/` form a dependency chain:

```text
01_prepare_data.sh          CPU     download + normalize + kill splits
02_decode_index_l40.sh      L40 1G  decode frames + SigLIP embeddings
03a_zero_shot_qwen2_5_l40   L40 1G  serve Qwen2.5-VL + zero-shot matrix
03b_zero_shot_qwen3_l40     L40 1G  serve Qwen3-VL + zero-shot matrix
03c_zero_shot_dgx           DGX 2G  serve both + full zero-shot matrix
04_grpo_dgx                 DGX 4G  LoRA-GRPO training
05_export_artifacts         CPU     CSV/SVG/trace export from all runs
```

### Quick start — submit the full L40 pipeline

```bash
export HF_TOKEN=<your_token>
bash jobs/submit_pipeline_l40.sh
```

### Quick start — submit the full DGX pipeline

```bash
export HF_TOKEN=<your_token>
bash jobs/submit_pipeline_dgx.sh
```

### Submit individual jobs

```bash
sbatch jobs/01_prepare_data.sh
sbatch jobs/02_decode_index_l40.sh
sbatch jobs/03a_zero_shot_qwen2_5_l40.sh
sbatch jobs/03b_zero_shot_qwen3_l40.sh
sbatch jobs/03c_zero_shot_dgx.sh
sbatch jobs/04_grpo_dgx.sh
sbatch jobs/05_export_artifacts.sh
```

### Logs

All SLURM stdout/stderr goes to `logs/`:

```text
logs/data_prep_<jobid>.out / .err
logs/decode_index_<jobid>.out / .err
logs/zero_shot_qwen25_<jobid>.out / .err
logs/zero_shot_qwen3_<jobid>.out / .err
logs/zero_shot_dgx_<jobid>.out / .err
logs/grpo_<jobid>.out / .err
logs/artifacts_<jobid>.out / .err
logs/vllm_qwen25_<jobid>.log     (vLLM server log)
logs/vllm_qwen3_<jobid>.log      (vLLM server log)
```

### Override defaults via env vars

```bash
sbatch --export=ALL,MODEL_PATH=models/qwen3_vl_8b_instruct,PORT=8001 jobs/03a_zero_shot_qwen2_5_l40.sh
sbatch --export=ALL,DECODE_FPS=0.5,SIGLIP_MODEL=google/siglip-so400m-patch14-384 jobs/02_decode_index_l40.sh
sbatch --export=ALL,VERL_ROOT=/opt/verl,MODEL_INDEX=1 jobs/04_grpo_dgx.sh
```

### Monitor and cancel

```bash
squeue -u $USER
scancel <job_id>
sacct -j <job_id> --format=JobID,State,Elapsed,MaxMemory
```

## Hardware State

Observed locally on June 20, 2026:

- `nvidia-smi` sees 3 RTX PRO 6000 Blackwell GPUs with about 98 GB each.
- The shell has `CUDA_VISIBLE_DEVICES=0,1`, so PyTorch currently sees 2 GPUs.
- PyTorch 2.11.0+cu128 works with CUDA matmul.
- `ffmpeg` and `git-lfs` are installed in `.conda-env`.
- vLLM 0.23.0 and SGLang 0.5.2 import successfully.

For 3-GPU runs, unset or change `CUDA_VISIBLE_DEVICES` before launching:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
```

## Asset Layout

Models:

```text
models/qwen2_5_vl_7b/
models/qwen3_vl_8b_instruct/
models/internvl3_5_8b/
models/llava_video_7b_qwen2/
```

Data:

```text
data/raw/<dataset>/
data/videos/<dataset>/
data/frames/<dataset>/<video_id>/frame_000001.jpg
data/indices/<dataset>_<encoder>_frames.jsonl
data/processed/<dataset>_<split>.jsonl
```

Ready pilot indices:

```text
data/indices/nextgqa_siglip_frames.jsonl          # 6,232 frames
data/indices/videomme_long_siglip_frames.jsonl    # 29,599 frames
data/indices/egoschema_siglip_frames.jsonl        # 13,500 frames
data/indices/nextqa_temporal_siglip_frames.jsonl  # 5,746 frames
```

## Serve A VLM

Qwen2.5 first:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 .conda-env/bin/python -m vllm.entrypoints.openai.api_server \
  --model models/qwen2_5_vl_7b \
  --port 8000 \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 32768
```

Qwen3-VL:

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 .conda-env/bin/python -m vllm.entrypoints.openai.api_server \
  --model models/qwen3_vl_8b_instruct \
  --port 8001 \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 32768
```

If vLLM rejects a multimodal model argument, fall back to the Transformers path for that backbone and keep vLLM for Qwen.

On the current Blackwell workstation, vLLM 0.23.0 loads Qwen2.5-VL successfully only with `VLLM_USE_FLASHINFER_SAMPLER=0`; the default FlashInfer sampler path fails its architecture check during warmup. The helper [scripts/serve_vllm_qwen.sh](../scripts/serve_vllm_qwen.sh) sets this by default.

## Build Or Rebuild Pilot Frames

The current workspace already has decoded and SigLIP-enriched pilots. To rebuild them:

```bash
.conda-env/bin/python scripts/decode_split_frames.py \
  --qa-path data/processed/nextgqa_val_150.jsonl \
  --video-root data/raw/nextgqa/NExTVideo \
  --out-root data/frames/nextgqa_val_150 \
  --fps 1 \
  --manifest runs/decode_nextgqa_val_150_manifest.json \
  --require-all

.conda-env/bin/python scripts/build_frame_index.py \
  --frames-root data/frames/nextgqa_val_150 \
  --output data/indices/nextgqa_val_150_decoded_frames.jsonl \
  --fps 1 \
  --dataset nextgqa

.conda-env/bin/python scripts/decode_split_frames.py \
  --qa-path data/processed/videomme_long_150.jsonl \
  --video-root data/videos/videomme \
  --out-root data/frames/videomme_long_150 \
  --fps 0.1 \
  --manifest runs/decode_videomme_long_150_manifest.json \
  --require-all

.conda-env/bin/python scripts/build_frame_index.py \
  --frames-root data/frames/videomme_long_150 \
  --output data/indices/videomme_long_decoded_frames.jsonl \
  --fps 0.1 \
  --dataset videomme

.conda-env/bin/python scripts/decode_split_frames.py \
  --qa-path data/processed/egoschema_val_150.jsonl \
  --video-root data/videos/egoschema \
  --out-root data/frames/egoschema_val_150 \
  --fps 0.5 \
  --manifest runs/decode_egoschema_val_150_manifest.json \
  --require-all

.conda-env/bin/python scripts/build_frame_index.py \
  --frames-root data/frames/egoschema_val_150 \
  --output data/indices/egoschema_decoded_frames.jsonl \
  --fps 0.5 \
  --dataset egoschema

.conda-env/bin/python scripts/decode_split_frames.py \
  --qa-path data/processed/nextqa_temporal_val_150.jsonl \
  --video-root data/videos/nextqa \
  --out-root data/frames/nextqa_temporal_150 \
  --fps 1 \
  --manifest runs/decode_nextqa_temporal_150_manifest.json \
  --require-all

.conda-env/bin/python scripts/build_frame_index.py \
  --frames-root data/frames/nextqa_temporal_150 \
  --output data/indices/nextqa_temporal_decoded_frames.jsonl \
  --fps 1 \
  --dataset nextqa
```

Embedding enrichment:

```bash
.conda-env/bin/python scripts/enrich_frame_index.py \
  --input data/indices/nextgqa_val_150_decoded_frames.jsonl \
  --output data/indices/nextgqa_siglip_frames.jsonl \
  --model google/siglip-base-patch16-224 \
  --batch-size 128 \
  --device cuda
```

Optional VLM captions:

```bash
.conda-env/bin/python scripts/caption_frame_index.py \
  --input data/indices/nextgqa_siglip_frames.jsonl \
  --output data/indices/nextgqa_qwen_captioned_frames.jsonl \
  --endpoint http://localhost:8000 \
  --model models/qwen2_5_vl_7b
```

## Run Zero-Shot With Images

Use configs with `send_images: true` once decoded frames exist.

```bash
.conda-env/bin/python -m vidcurator.run evaluate \
  --config configs/gpu_nextgqa_qwen2_5_zero_shot.json

.conda-env/bin/python -m vidcurator.run evaluate \
  --config configs/gpu_videomme_qwen2_5_zero_shot.json

.conda-env/bin/python -m vidcurator.run evaluate \
  --config configs/gpu_egoschema_qwen2_5_zero_shot.json

.conda-env/bin/python -m vidcurator.run evaluate \
  --config configs/gpu_nextqa_qwen2_5_zero_shot.json
```

Qwen3 configs are available with the same naming pattern:

```text
configs/gpu_nextgqa_qwen3_zero_shot.json
configs/gpu_videomme_qwen3_zero_shot.json
configs/gpu_egoschema_qwen3_zero_shot.json
configs/gpu_nextqa_qwen3_zero_shot.json
```

Then export paper artifacts:

```bash
.conda-env/bin/python scripts/export_paper_artifacts.py \
  --run-dir runs/gpu_nextgqa_qwen2_5_zero_shot

.conda-env/bin/python scripts/export_qualitative_traces.py \
  --run-dir runs/gpu_nextgqa_qwen2_5_zero_shot \
  --qa-path data/processed/nextgqa_val_150.jsonl \
  --frames-path data/indices/nextgqa_siglip_frames.jsonl \
  --policy scripted_curator \
  --max-examples 20
```

## Training Gate

Do not start LoRA-GRPO until:

- Qwen2.5 and Qwen3 zero-shot runs finish on NExT-GQA 150.
- At least one long-video dataset small split is runnable.
- Full VID-CURATOR beats uniform or MMR on at least one tight-budget gate.
- Evidence metrics are not worse than the retrieval-only baseline.

Once the gate passes:

```bash
.conda-env/bin/python scripts/render_verl_grpo_command.py \
  --config configs/grpo_stage1.json
```

Then adapt the rendered command to the installed verl path.
