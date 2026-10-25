#!/usr/bin/env python3
"""N7: subsample a SigLIP frame index temporally to simulate a sparser decode (fps) pool.
Keeps every Nth frame per video (ordered by timestamp). Usage:
  python scripts/subsample_index.py --in <idx.jsonl> --out <idx.jsonl> --stride 2
"""
import argparse, json
from collections import defaultdict
ap=argparse.ArgumentParser(); ap.add_argument("--in",dest="inp",required=True)
ap.add_argument("--out",required=True); ap.add_argument("--stride",type=int,required=True)
a=ap.parse_args()
byvid=defaultdict(list)
for l in open(a.inp):
    r=json.loads(l); byvid[r["video_id"]].append(r)
kept=0; tot=0
with open(a.out,"w") as f:
    for vid,rows in byvid.items():
        rows.sort(key=lambda r: float(r.get("timestamp",0)))
        tot+=len(rows)
        for i,r in enumerate(rows):
            if i % a.stride == 0:
                f.write(json.dumps(r)+"\n"); kept+=1
print(f"{a.inp} stride={a.stride}: kept {kept}/{tot} frames -> {a.out}")
