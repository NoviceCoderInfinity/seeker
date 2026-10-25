# seeker

Code and paper source for:

> **Seeing but Not Speaking: Most Token-Mass Interference in Budget-Bounded
> Video Question Answering Is a Decoding Artifact**
> Aman Verma, Anupam Rawat — Indian Institute of Technology Bombay
> WACV 2027 (Applications Track), paper #949.

This is an **audit / measurement** study of budget-bounded video question
answering. It shows that (1) comparing frame-selection methods at a matched
*frame count* `K` is confounded — the per-frame pixel budget alone moves
visual-token consumption by `4.2×` on one answerer — and (2) most of the
"visual interference" attributed to high token mass is a **decoding artifact**:
reading the answer at the token position where the model emits it (forced
choice) rather than from free generation removes 83% of the effect on the one
checkpoint that shows it.

The released protocol — the iso-token grid, firewall splits, null calculators,
modality labels, and both readouts' per-question outputs — is **TokenBudgetBench**.

> The name is a nod to the Seeker in Quidditch: the job is to find the one frame
> that matters among many, under a strict budget.

## Pipeline

![Budget-bounded VQA measurement pipeline: frames are indexed and sampled under a
fixed token budget, passed to a served VLM, and the answer is read out two ways —
free generation vs. forced choice — so the token-mass effect can be decomposed.](paper/assets/figure_pipeline.png)

Frames are indexed (SigLIP, duration-proportional) and sampled under a fixed
visual-token budget; the selected frames go to a VLM served over an
OpenAI-compatible endpoint; the answer is then **read out two ways** — free
generation and forced choice — which is what lets us separate a genuine
visual-interference effect from the decoding artifact. Vector source:
[`paper/assets/figure_pipeline.drawio`](paper/assets/figure_pipeline.drawio)
(also exported as `.pdf` and `.png`).

## Repository layout

```
vidcurator/     Core Python package (dependency-free core; schemas, buffer,
                baselines, scripted curator, metrics, normalizers, reward helper).
scripts/        Research scripts: evaluation, paired significance tests,
                figure/table export, dataset normalizers, iso-token bake-off,
                readout (generative vs forced-choice) re-scoring, routers.
jobs/           SLURM launchers for the full grid (named e16_resdecomp,
                e17_curve, x14_fc_decomp, a1_bakeoff, … — these names appear
                verbatim in the paper's captions and in-text run paths).
configs/        Experiment configs (one JSON per run / matrix cell).
tests/          Unit tests that run on Python 3.10 with no installed packages.
docs/           Technical reference: benchmarks, datasets, schema, GPU runbook.
paper/          Camera-ready LaTeX source (paper.tex + refs.bib + WACV kit),
                figure data (fig1_data.csv, fig2_data.csv), the pipeline
                schematic (paper/assets/figure_pipeline.*), and the compiled
                camera-ready PDF.
```

## Where the data lives

The code here is self-contained, but the real experiments were run on an HPC
cluster. The authoritative run tree, logs, and per-question outputs live on the
server at:

```
/home/vis-comp/22b3929/vidcurator
```

Paths referenced in the paper (e.g. `runs/e16_resdecomp/{FF,FT,TF,TT}.jsonl`,
`runs/e17_curve_nextqa_ho/`, `runs/x14_fc_decomp_{FF,FT}_full/cells.jsonl`,
`runs/a1_bakeoff_*`, `slurm-*.out` logs) are relative to that root. Large
artifacts (`data/`, `models/`, `runs/`, `logs/`, `outputs/`) are **not** checked
in — they are reproduced from the scripts below or released as **TokenBudgetBench**.

## Quick start (no GPU, no installed packages)

The fixture run and tests use only the Python 3.10 standard library:

```bash
python3 -m vidcurator.run evaluate --config configs/kill_test.json
python3 -m unittest discover -s tests
```

## Reproducing the study

Real experiments plug in decoded frames, SigLIP/CLIP frame indices, and a VLM
served over an OpenAI-compatible endpoint (vLLM/SGLang with Qwen2.5-VL / Qwen3-VL):

1. **Normalize** a benchmark into the project schema:
   `python3 -m vidcurator.run normalize --dataset nextgqa --input <raw> --output data/processed/nextgqa_val.jsonl`
2. **Index frames** (SigLIP, fps 0.1, duration-proportional) — see `scripts/prepare_frames.sh`.
3. **Serve a VLM** (`scripts/serve_transformers_vlm.py` or an external vLLM endpoint).
4. **Run the grid** via the `jobs/` SLURM launchers (A40 for the harness grid,
   A100-80GB for LoRA training), or `scripts/run_zero_shot_matrix.py`.
5. **Readout decomposition** (generative vs forced choice) and **paired
   significance** (exact-binomial McNemar): `scripts/paired_significance.py`.
6. **Export tables and figures**: `scripts/make_figures.py`, `scripts/export_paper_artifacts.py`.

Environment specs are in `env/` (`environment.h100.yml`, `requirements-*.txt`).

## Paper

`paper/paper.tex` is the WACV 2027 camera-ready, exactly as submitted — an
8-page body plus references. The two figures in the paper are inline
TikZ/pgfplots driven by `fig1_data.csv` and `fig2_data.csv`. The pipeline
schematic shown above is maintained here in the repository
(`paper/assets/figure_pipeline.{drawio,pdf,png}`) rather than in the paper.

Build with [tectonic](https://tectonic-typesetting.github.io/):

```bash
cd paper && tectonic -X compile paper.tex
```

## Citation

```bibtex
@inproceedings{verma2027seeing,
  title     = {Seeing but Not Speaking: Most Token-Mass Interference in
               Budget-Bounded Video Question Answering Is a Decoding Artifact},
  author    = {Verma, Aman and Rawat, Anupam},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## License

MIT — see [LICENSE](LICENSE).
