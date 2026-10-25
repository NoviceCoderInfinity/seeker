"""P1 significance pass (plan4 §3-P1): paired per-question tests on the POWERED runs.

For each (dataset, policyA=learned, policyB in {uniform, scripted_curator}) at matched n:
  - per-budget McNemar exact test on paired correctness (binomial on discordant pairs)
  - paired bootstrap (10k) over the budget-averaged accuracy delta (CE-AUC delta)
Pairing is by (qid, budget). Learned predictions come from the powered learned runs
(consolidated predictions.jsonl, or _shards/*.jsonl if not yet consolidated); baselines from the
matched r2_v2 / pow452 / x1 runs. Gold: data/rl/qid_gold.json (6 train datasets) or the QA file
(lvbench). Zero GPU. Output: reports/significance.md
"""
from __future__ import annotations
import json, glob, os, math, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = json.loads((ROOT / "data/rl/qid_gold.json").read_text())

# --- TRAIN/TEST FIREWALL (added 2026-07-10) ---------------------------------
# The GRPO/RFT curator was trained on traces exported from the val files themselves, so many
# train qids collide (same question AND video) with eval sets. Every learned-curator number is
# reported on the STRICT held-out set frozen in data/rl/firewall_split.json (eval qid not a train
# qid AND video not a train video). We build a HELDOUT allowlist; any qid outside it is skipped.
def _heldout_allow():
    p = ROOT / "data/rl/firewall_split.json"
    allow = set()
    if p.exists():
        fw = json.loads(p.read_text())
        for ds, rec in fw.get("eval", {}).items():
            allow |= set(map(str, rec.get("heldout_qids", [])))
    return allow
HELDOUT_QIDS = _heldout_allow()

def gold_for(ds):
    if ds == "lvbench":
        g = {}
        for l in open(ROOT / "data/processed/lvbench_val_siglip.jsonl"):
            r = json.loads(l); g[str(r["qid"])] = r["answer_index"]
        return g
    return GOLD

def load_preds(run_dir, policy, budgets):
    """-> {(qid,budget): correct} using consolidated predictions or shards."""
    out = {}
    ds_gold = None  # filled by caller via closure-free param; we pass gold in score()
    files = []
    p = ROOT / run_dir / "predictions.jsonl"
    if p.exists():
        files = [p]
    else:
        files = [Path(f) for f in glob.glob(str(ROOT / run_dir / "_shards" / "*.jsonl"))]
    rows = []
    for f in files:
        for l in open(f):
            try: rows.append(json.loads(l))
            except: pass
    for r in rows:
        if r.get("policy") != policy: continue
        b = r.get("budget")
        if budgets and b not in budgets: continue
        out[(str(r.get("qid")), b)] = r.get("answer_index")
    return out

def mcnemar_p(b, c):
    """exact binomial two-sided on discordant pairs (b = A right/B wrong, c = A wrong/B right)."""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    # two-sided exact binomial p
    p = sum(math.comb(n, i) for i in range(0, k + 1)) * 2 / (2 ** n)
    return min(1.0, p)

def run_pair(ds, learned_dir, base_dir, base_pol, budgets, gold, pol_a="learned_curator"):
    L = load_preds(learned_dir, pol_a, budgets)
    B = load_preds(base_dir, base_pol, budgets)
    lines = []
    deltas_by_q = {}  # qid -> list of (l_correct - b_correct) across budgets
    total_pairs = 0
    for b in budgets:
        n = w = l = both = neither = 0
        for (qid, bb), la in L.items():
            if bb != b: continue
            if (qid, b) not in B: continue
            if qid not in HELDOUT_QIDS: continue  # strict firewall: held-out (qid+video) only
            g = gold.get(qid)
            if g is None: continue
            lc = int(la == g); bc = int(B[(qid, b)] == g)
            n += 1
            if lc and not bc: w += 1
            elif bc and not lc: l += 1
            elif lc: both += 1
            else: neither += 1
            deltas_by_q.setdefault(qid, []).append(lc - bc)
        total_pairs += n
        if n == 0:
            lines.append(f"| {ds} | b{b} | — | no paired data |"); continue
        p = mcnemar_p(w, l)
        acc_d = (w - l) / n
        sig = "**SIG**" if p < 0.05 else "ns"
        lines.append(f"| {ds} vs {base_pol} | b{b} | n={n} | Δacc={acc_d:+.3f} (win {w}/lose {l}) | p={p:.4f} {sig} |")
    # paired bootstrap on per-question mean delta (across budgets)
    qids = [q for q, ds_ in deltas_by_q.items() if ds_]
    if qids:
        per_q = [sum(deltas_by_q[q]) / len(deltas_by_q[q]) for q in qids]
        rng = random.Random(0)
        n = len(per_q); boots = []
        for _ in range(10000):
            s = [per_q[rng.randrange(n)] for _ in range(n)]
            boots.append(sum(s) / n)
        boots.sort()
        lo, hi = boots[249], boots[9749]
        mean = sum(per_q) / n
        star = "**CI excludes 0**" if (lo > 0 or hi < 0) else "CI includes 0"
        lines.append(f"| {ds} vs {base_pol} | CE-Δ boot | n={n}q | mean={mean:+.3f} | 95% CI [{lo:+.3f},{hi:+.3f}] {star} |")
    return lines

def main():
    PAIRS = [
        # ds, learned_run, baseline_run, baseline_policy, budgets
        ("egoschema(252)", "runs/gpu_egoschema_qwen2_5_learned_grpo_full", "runs/r2_v2_egoschema_qwen2_5_full", "uniform", [4,8,16,32]),
        ("egoschema(252)", "runs/gpu_egoschema_qwen2_5_learned_grpo_full", "runs/r2_v2_egoschema_qwen2_5_full", "scripted_curator", [4,8,16,32]),
        ("mlvu(253)", "runs/gpu_mlvu_qwen2_5_learned_grpo_full", "runs/r2_v2_mlvu_qwen2_5_full", "uniform", [4,8,16,32]),
        ("mlvu(253)", "runs/gpu_mlvu_qwen2_5_learned_grpo_full", "runs/r2_v2_mlvu_qwen2_5_full", "scripted_curator", [4,8,16,32]),
        ("videomme(453)", "runs/gpu_videomme_qwen2_5_learned_grpo_full453", "runs/r2_v2_videomme_qwen2_5_full", "uniform", [4,8,16,32]),
        ("videomme(453)", "runs/gpu_videomme_qwen2_5_learned_grpo_full453", "runs/r2_v2_videomme_qwen2_5_full", "scripted_curator", [4,8,16,32]),
        ("lvb(669)", "runs/gpu_longvideobench_qwen2_5_learned_grpo_full", "runs/r2_v2_longvideobench_qwen2_5_full", "uniform", [4,8,16,32]),
        ("nextgqa(452)", "runs/s1181_nextgqa_452", "runs/gpu_nextgqa_qwen2_5_learned_grpo_pow452", "uniform", [4,8,16]),
        ("nextgqa(452)", "runs/s1181_nextgqa_452", "runs/gpu_nextgqa_qwen2_5_learned_grpo_pow452", "scripted_curator", [4,8,16]),
        ("nextqa(452)", "runs/s1181_nextqa_452", "runs/gpu_nextqa_qwen2_5_learned_grpo_pow452", "uniform", [4,8,16]),
        ("nextqa(452)", "runs/s1181_nextqa_452", "runs/gpu_nextqa_qwen2_5_learned_grpo_pow452", "scripted_curator", [4,8,16]),
        # scripted vs uniform at power on localizable (the other side of complementarity)
        ("SCRIPTEDvU nextgqa(452)", "runs/gpu_nextgqa_qwen2_5_learned_grpo_pow452", "runs/gpu_nextgqa_qwen2_5_learned_grpo_pow452", "uniform", [4,8,16]),
        ("SCRIPTEDvU nextqa(452)", "runs/gpu_nextqa_qwen2_5_learned_grpo_pow452", "runs/gpu_nextqa_qwen2_5_learned_grpo_pow452", "uniform", [4,8,16]),
        # s1181 camera-ready vs r2_v2
        ("s1181 videomme(453)", "runs/s1181_videomme_453", "runs/r2_v2_videomme_qwen2_5_full", "uniform", [4,8,16,32]),
        ("s1181 lvb(669)", "runs/s1181_lvb_669", "runs/r2_v2_longvideobench_qwen2_5_full", "uniform", [4,8,16,32]),
        ("s1181 mlvu(253)", "runs/s1181_mlvu_253", "runs/r2_v2_mlvu_qwen2_5_full", "uniform", [4,8,16,32]),
        ("s1181 egoschema(252)", "runs/s1181_egoschema_252", "runs/r2_v2_egoschema_qwen2_5_full", "uniform", [4,8,16,32]),
        ("lvbench(234)", "runs/s1181_lvbench_learned", "runs/x1_lvbench", "uniform", [8,16]),
        ("lvbench(234)", "runs/s1181_lvbench_learned", "runs/x1_lvbench", "scripted_curator", [8,16]),
    ]
    L = ["# P1 — Paired significance tests (McNemar exact + paired bootstrap), powered runs\n",
         f"_STRICT HELD-OUT ONLY ({len(HELDOUT_QIDS)} qids, firewall_split.json): eval qid not a train qid AND video not a train video.",
         "Learned curator vs baseline, paired by (qid,budget) at matched n. McNemar exact binomial",
         "on discordant pairs; bootstrap = 10k resamples of per-question mean delta (budget-averaged),",
         "95% CI. step-100 runs = conservative adapter; s1181 = camera-ready (step-1181)._\n",
         "| comparison | cell | n | effect | test |", "|---|---|---|---|---|"]
    for row in PAIRS:
        ds, lr, br, bp, budgets = row[:5]
        pol_a = "scripted_curator" if ds.startswith("SCRIPTEDvU") else "learned_curator"
        if not (ROOT / lr).exists(): continue
        has_data = (ROOT/lr/"predictions.jsonl").exists() or glob.glob(str(ROOT/lr/"_shards/*.jsonl"))
        if not has_data: continue
        g = gold_for("lvbench" if "lvbench" in lr else "any")
        L += run_pair(ds, lr, br, bp, budgets, g, pol_a)
    out = ROOT / "reports/significance.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L)); print("\nWROTE", out)

if __name__ == "__main__":
    main()
