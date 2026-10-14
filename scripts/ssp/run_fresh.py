"""Run the fresh-seed replication tasks written by an_hunt.py (stage scan) under the quiet governor.

    scripts/ssp/run_overnight.sh  ...                      # main stages first
    .venv/bin/python3 scripts/ssp/an_hunt.py --stage scan
    nice -n 10 caffeinate -i .venv/bin/python3 scripts/ssp/run_fresh.py
    .venv/bin/python3 scripts/ssp/an_hunt.py --stage check --fresh results/small_sample_popularity/02_runs_fresh

Rows go to 02_runs_fresh/{e1,e2,e5,e6,e7}_fresh.jsonl (draw ids 1000+, never used by the main stages).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import quiet as Q  # noqa: E402

import exp_core as X  # noqa: E402

TASKS = "results/small_sample_popularity/03_tables/hunt_replication_tasks.json"
OUT = "results/small_sample_popularity/02_runs_fresh"


def main():
    tasks = json.load(open(TASKS))
    seen, uniq = set(), []
    for t in tasks:
        if t["key"] not in seen:
            seen.add(t["key"])
            uniq.append(t)
    groups = {}
    for t in uniq:
        name = "e7" if (t["kind"] == "e1" and t.get("arm", "uniform") != "uniform") else t["kind"]
        groups.setdefault(name, []).append(t)
    os.makedirs(OUT, exist_ok=True)
    gov = Q.Governor(Q.Limits.from_json("configs/ssp_quiet.json"),
                     log_path="results/small_sample_popularity/00_safety/governor_log.jsonl")
    try:
        for name, ts in groups.items():
            n = Q.run_tasks(X.run_task, ts, gov, f"{OUT}/{name}_fresh.jsonl")
            print(f"fresh {name}: ran {n} of {len(ts)}", flush=True)
    finally:
        gov.close()


if __name__ == "__main__":
    main()
