# Dataset Plan

The repo does not vendor benchmark data. Put raw downloads under `data/raw/`, normalized QA under `data/processed/`, decoded frames under `data/videos/` or `data/frames/`, and frame indices under `data/indices/`.

## Primary Benchmarks

| Dataset | Role | Why It Matters | Current Handling |
|---|---|---|---|
| LongVideoBench | Main long-video Pareto | Referring reasoning questions explicitly need long frame inputs | `normalize --dataset longvideobench` |
| Video-MME / Video-MME-long | Main long-video Pareto | Broad video domains, long videos up to about an hour | `normalize --dataset videomme` |
| MLVU | Multi-task long-video understanding | Multiple genres and task types | `normalize --dataset mlvu` |
| NExT-QA | Temporal/causal QA | Good for single-frame shortcut control | `normalize --dataset nextqa` |
| NExT-GQA | Grounded VideoQA | Evidence spans power FEP/FER, IF, and reward term | `normalize --dataset nextgqa` |
| EgoSchema | Egocentric long-form QA | Domain transfer and temporal certificate stress test | `normalize --dataset egoschema` |
| VSI-Bench | Spatial optional | Useful if Video-R1-style spatial reasoning is included | use `generic` adapter until fields are finalized |

## Required Fields After Normalization

Every row must have `qid`, `video_id`, and `question`. MCQ rows should include `options` and either `answer_index` or `answer`. Grounded rows should include `evidence_spans`.

## Evidence Span Audit

Before using FEP/FER or reward weights on a benchmark:

1. Check whether span timestamps are in seconds or frame indices.
2. Check whether spans are inclusive at both ends.
3. Confirm the video decoding FPS and timestamp origin.
4. Spot-check 30 examples visually.
5. Set `frame_evidence_precision` reward weight to `0` for any dataset without trustworthy spans.

## License/Access Checklist

Do this before large runs and before paper submission:

- Confirm benchmark licenses and redistribution limits.
- Keep raw data outside the repo.
- Store only normalized metadata and generated predictions in repo-visible artifacts.
- Record exact commit/hash/version for each benchmark download.
- Record whether subtitles/audio were used; Video-MME and LongVideoBench comparisons can change depending on modality.

## Verified Source Notes

As of June 20, 2026:

- LongVideoBench is a NeurIPS 2024 Datasets & Benchmarks benchmark with 6,678 MCQ questions over 3,763 videos and a public project page/leaderboard.
- NExT-GQA repository is MIT licensed and targets visually grounded VideoQA with temporal evidence.
- Qwen2.5-VL-7B-Instruct and Qwen3-VL-8B-Instruct model cards list Apache-2.0 licenses.
- Qwen3-VL model card emphasizes native 256K multimodal context and second-level video indexing; this makes it a strong backbone but also raises the bar for showing VID-CURATOR wins under tight budgets.
