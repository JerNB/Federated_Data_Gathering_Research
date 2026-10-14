"""Collapse the raw per-draw rows into one compact table per stage: mean and standard error per cell.

    .venv/bin/python3 scripts/ssp/aggregate.py [--runs DIR] [--out DIR]

Writes <out>/03_tables/cells_<stage>.csv. These are the files to share: small, one row per cell.
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

CELL_KEYS = {
    "e1": ["dataset", "arm", "N", "k"],
    "e2": ["dataset", "N", "k"],
    "e4": ["dataset", "N", "k"],
    "e5": ["dataset", "group", "N_req", "k"],
    "e6": ["dataset", "N", "k", "eps"],
    "ref": ["dataset"],
}
SKIP = {"key", "draw", "sec", "model"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/small_sample_popularity/02_runs")
    ap.add_argument("--out", default="results/small_sample_popularity")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/03_tables", exist_ok=True)
    for stage, keys in CELL_KEYS.items():
        path = f"{a.runs}/{stage}.jsonl"
        if not os.path.exists(path):
            continue
        groups = defaultdict(list)
        for ln in open(path):
            r = json.loads(ln)
            if "error" in r:
                continue
            r.setdefault("arm", "uniform")
            groups[tuple(r.get(k) for k in keys)].append(r)
        # stage e1 rows from the biased arms live in e7.jsonl, same schema
        if stage == "e1" and os.path.exists(f"{a.runs}/e7.jsonl"):
            for ln in open(f"{a.runs}/e7.jsonl"):
                r = json.loads(ln)
                if "error" in r:
                    continue
                r.setdefault("arm", "uniform")
                groups[tuple(r.get(k) for k in keys)].append(r)
        cols = []
        for rows in groups.values():
            for r in rows:
                for c, v in r.items():
                    if c not in keys and c not in SKIP and isinstance(v, (int, float)) and c not in cols:
                        cols.append(c)
        with open(f"{a.out}/03_tables/cells_{stage}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys + ["n_draws"] + [x for c in cols for x in (c, c + ".se")])
            for g, rows in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
                out = list(g) + [len(rows)]
                for c in cols:
                    v = [r[c] for r in rows if isinstance(r.get(c), (int, float))]
                    if not v:
                        out += ["", ""]
                        continue
                    m = sum(v) / len(v)
                    se = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1) / len(v)) if len(v) > 1 else ""
                    out += [round(m, 5), round(se, 5) if se != "" else ""]
                w.writerow(out)
        print(stage, len(groups), "cells")


if __name__ == "__main__":
    main()
