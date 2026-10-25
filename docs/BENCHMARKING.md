# Benchmarking Strategy Against Existing Models And Papers

This project must benchmark both models and frame-selection methods. The paper should not compare only against uniform sampling.

## Required Method Comparisons

Internal:

- Uniform sampling.
- MMR retrieval.
- Best single frame.
- Oracle temporal span where gold spans exist.
- VID-CURATOR scripted zero-shot.
- VID-CURATOR probe-off.
- VID-CURATOR curation-off.
- VID-CURATOR summary-off.
- VID-CURATOR LoRA-GRPO after zero-shot passes.

External:

- AKS.
- VideoTree.
- Flexible Frame Selection.
- FrameThinker.
- VideoBrain.
- Adaptive Greedy Frame Selection.
- A.I.R.
- F2C/key-clip selection if reviewers push on frame-vs-clip continuity.

The tracking manifest is [manifests/external_baselines.json](../manifests/external_baselines.json).

## Required Backbone Comparisons

Start with:

- Qwen2.5-VL-7B-Instruct.
- Qwen3-VL-8B-Instruct.

Then add:

- InternVL3.5-8B.
- LLaVA-Video-7B-Qwen2 after license/runtime check.
- Qwen3-VL-8B-Thinking if exact weights and server support are available.

Backbone manifest: [manifests/models.json](../manifests/models.json).

## Fairness Rules

- Same decoded candidate frame pool per dataset.
- Same FPS/candidate cap per method unless reproducing a paper requires otherwise.
- Same downstream VLM prompt and answer parser for selectors.
- Same frame budget and visual-token budget reporting.
- Same subtitle/audio setting per dataset.
- External reported numbers can be cited, but main claims need our own runs when code/data permit.

## Reporting Tables

1. Main Pareto: accuracy vs K frames.
2. Evidence table: FEP/FER/IF on NExT-GQA.
3. Shortcut table: best single frame vs curated context.
4. Ablation table: full vs probe-off vs curation-off vs summary-off.
5. Backbone table.
6. External-baseline table.
7. Cost table: frames, visual tokens, latency, GPU hours.

## Target Standing

These are success thresholds, not claimed results:

- Minimum: `scripted_curator@8 >= uniform@16` or `scripted_curator@16 >= uniform@32` on a main long-video benchmark.
- Evidence: scripted curator improves FEP over uniform and MMR on NExT-GQA.
- Ablation: full method beats at least two of probe-off, curation-off, and summary-off.
- Shortcut: curated context beats best-single-frame on temporal/causal subsets.
- Strong: trained LoRA-GRPO curator improves scripted by 1-3 accuracy points on two datasets without reducing FEP or increasing cost too much.
- External: match/beat AKS or VideoTree in a controlled run, and compare against FrameThinker/VideoBrain as reproduced or clearly reported-only learned-agent baselines.

## What Makes VID-CURATOR Different

VideoTree and AKS are strong training-free frame selectors. FrameThinker and VideoBrain are strong adaptive learned policies. VID-CURATOR needs to win by making the context state and the probe measurable:

- buffer state is persistent across hops;
- keyframes and compressed summary are budgeted;
- probe declares missing evidence;
- sufficiency is calibrated;
- evidence precision and single-frame shortcut gap are explicit metrics.
