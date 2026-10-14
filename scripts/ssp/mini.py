"""Tiny serial run of every stage (ml-1m + amazon-software, few draws) for developing analysis scripts.

    .venv/bin/python3 scripts/ssp/mini.py     ->  results/small_sample_popularity/02_runs_mini/<stage>.jsonl
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import quiet  # noqa: F401  one BLAS thread
import run_all as RA
import exp_core as X

OUT = "results/small_sample_popularity/02_runs_mini"
os.makedirs(OUT, exist_ok=True)
for stage in ["ref", "e1", "e2", "e4", "e5", "e6", "e7"]:
    tasks = [t for t in RA.build(stage) if t["dataset"] in ("ml-1m", "amazon-software") and t.get("draw", 0) < 8]
    tasks = [t for t in tasks if t.get("N", 0) <= 2000]
    with open(f"{OUT}/{stage}.jsonl", "w") as f:
        for t in tasks:
            r = X.run_task(t)
            r["key"] = t["key"]
            if "arm" in t:
                r["arm"] = t["arm"]
            f.write(json.dumps(r) + "\n")
    print(stage, len(tasks), flush=True)
