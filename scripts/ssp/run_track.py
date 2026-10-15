"""Run the tracking study under the quiet governor (2 workers). Resumable.

    nice -n 10 caffeinate -i .venv/bin/python3 scripts/ssp/run_track.py            # everything
    .venv/bin/python3 scripts/ssp/run_track.py --count
    .venv/bin/python3 scripts/ssp/run_track.py --smoke ml-1m                        # one small dataset, own output file

Rows go to results/small_sample_popularity/02_runs/track.jsonl (gitignored).
"""
from __future__ import annotations

import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import quiet as Q  # noqa: E402  (sets one BLAS thread per process before numpy)

import numpy as np  # noqa: E402

RUNS = "results/small_sample_popularity/02_runs"


def datasets() -> list[str]:
    return sorted(os.path.basename(f)[:-4] for f in glob.glob("datasets/npz/*.npz"))


def eligible_n(active: np.ndarray, first: int = 0) -> list[int]:
    """n values (people per round) allowed for a log: n <= 20% of the 10th-percentile active people per window
    over the scored windows 8..29 (the finite-population correction stays above 0.8)."""
    import track_core as T
    p10 = float(np.percentile(active[T.FIRST_STEP:], 10))
    return [n for n in T.N_GRID if n <= 0.2 * p10]


def build(only: list[str] | None = None) -> list[dict]:
    tasks = []
    for ds in only or datasets():
        tasks.append(dict(key=f"full|{ds}", kind="full", dataset=ds))
    return tasks


def main():
    only = None
    out = f"{RUNS}/track.jsonl"
    if "--smoke" in sys.argv:
        only = [sys.argv[sys.argv.index("--smoke") + 1]]
        out = f"/tmp/track_smoke.jsonl"
    import track_core as T
    lim = Q.Limits.from_json("configs/ssp_quiet.json")
    lim.max_workers = 2
    os.makedirs(RUNS, exist_ok=True)
    # Phase 1: full-community tasks (they also give the active-people counts that decide each log's n grid)
    gov = Q.Governor(lim, log_path="results/small_sample_popularity/00_safety/governor_log_track.jsonl")
    try:
        t0 = time.time()
        full = build(only)
        if "--count" in sys.argv:
            print("full tasks", len(full))
            return
        Q.run_tasks(T.run_task, full, gov, out)
        import json
        act = {}
        for ln in open(out):
            r = json.loads(ln)
            if r.get("kind") == "full":
                act[r["dataset"]] = np.array(r["active"])
        tasks = [dict(key=f"n|{ds}|{n}", kind="n", dataset=ds, n=n) for ds in sorted(act) for n in eligible_n(act[ds])]
        print(f"[{time.strftime('%H:%M:%S')}] full tasks done in {time.time()-t0:.0f}s; {len(tasks)} (dataset, n) cells",
              flush=True)
        done = Q.run_tasks(T.run_task, tasks, gov, out)
        print(f"[{time.strftime('%H:%M:%S')}] ran {done} of {len(tasks)} cells; governor {gov.last}", flush=True)
    finally:
        gov.close()


if __name__ == "__main__":
    main()
