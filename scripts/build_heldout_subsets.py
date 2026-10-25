"""Materialize strict-firewall held-out QA subset files.

Filters each evaluated QA file down to the firewall allowlist
(`data/rl/firewall_split.json` -> eval[ds].heldout_qids): the question is not a GRPO
train/val qid AND its video is disjoint from every training video. These files let new
policy evals (DdE, lambda-sweep adapters) run ONLY on clean questions instead of paying
for contaminated rows we would filter out at scoring time anyway.

Output: data/processed/heldout/<ds>_heldout.jsonl (+ a manifest with counts).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIREWALL = ROOT / "data/rl/firewall_split.json"
OUT_DIR = ROOT / "data/processed/heldout"


def main() -> None:
    fw = json.loads(FIREWALL.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for ds, spec in fw["eval"].items():
        allow = set(spec["heldout_qids"]) if isinstance(spec["heldout_qids"], list) else None
        if allow is None:
            raise SystemExit(f"{ds}: heldout_qids is not a list — rebuild firewall_split.json")
        src = ROOT / spec["file"]
        rows = [json.loads(line) for line in src.open() if line.strip()]
        kept = [r for r in rows if str(r["qid"]) in allow]
        if len(kept) != spec["heldout_n"]:
            raise SystemExit(f"{ds}: kept {len(kept)} != expected {spec['heldout_n']}")
        out = OUT_DIR / f"{ds}_heldout.jsonl"
        with out.open("w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        manifest[ds] = {"source": spec["file"], "out": str(out.relative_to(ROOT)),
                        "n": len(kept), "source_n": len(rows)}
        print(f"{ds}: {len(kept)}/{len(rows)} -> {out.name}")
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
