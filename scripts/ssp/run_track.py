"""Run the tracking study under the quiet governor (2 workers). Resumable.

    nice -n 10 caffeinate -i .venv/bin/python3 scripts/ssp/run_track.py                     # primary (30 windows, P10 rule)
    .venv/bin/python3 scripts/ssp/run_track.py --elig steps                                 # extension E1: more n per log
    .venv/bin/python3 scripts/ssp/run_track.py --nwin 60 --elig steps                       # extension E2: shorter windows
    .venv/bin/python3 scripts/ssp/run_track.py --nwin 20 --elig steps                       # extension E2: longer windows
    .venv/bin/python3 scripts/ssp/run_track.py --smoke ml-1m                                # one small dataset, own file

Eligibility of n (people per round):
  p10   (pre-registered, primary): n <= 20% of the 10th-percentile active people per window over windows 8..end.
  steps (extension, post hoc):     at least 10 steps in which every window w-7..w+1 has >= 5n active people.
Rows go to results/small_sample_popularity/02_runs/track.jsonl (gitignored). Keys of the primary run stay unchanged.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import quiet as Q  # noqa: E402  (sets one BLAS thread per process before numpy)

import numpy as np  # noqa: E402

RUNS = "results/small_sample_popularity/02_runs"


def datasets() -> list[str]:
    return sorted(os.path.basename(f)[:-4] for f in glob.glob("datasets/npz/*.npz"))


def eligible_n(active: np.ndarray, elig: str = "p10") -> list[int]:
    import track_core as T
    if elig == "p10":
        p10 = float(np.percentile(active[T.FIRST_STEP:], 10))
        return [n for n in T.N_GRID if n <= 0.2 * p10]
    out = []
    n_win = len(active)
    for n in T.N_GRID:
        valid = sum(1 for w in range(T.FIRST_STEP, n_win - 1) if all(active[w - 7: w + 2] >= 5 * n))
        if valid >= 10:
            out.append(n)
    return out


def key_of(kind: str, ds: str, n: int | None, nwin: int) -> str:
    base = f"full|{ds}" if kind == "full" else f"n|{ds}|{n}"
    return base if nwin == 30 else f"{base}|w{nwin}"


def arg(name: str, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    import track_core as T
    nwin = int(arg("--nwin", 30))
    elig = arg("--elig", "p10")
    only = [arg("--smoke")] if "--smoke" in sys.argv else None
    out = "/tmp/track_smoke.jsonl" if only else f"{RUNS}/track.jsonl"
    full = [dict(key=key_of("full", ds, None, nwin), kind="full", dataset=ds, n_win=nwin) for ds in (only or datasets())]
    if "--count" in sys.argv:
        print("full tasks", len(full))
        return
    lim = Q.Limits.from_json("configs/ssp_quiet.json")
    lim.max_workers = 2
    os.makedirs(RUNS, exist_ok=True)
    gov = Q.Governor(lim, log_path="results/small_sample_popularity/00_safety/governor_log_track.jsonl")
    try:
        t0 = time.time()
        Q.run_tasks(T.run_task, full, gov, out)
        act = {}
        for ln in open(out):
            r = json.loads(ln)
            if r.get("kind") == "full" and r.get("n_win", 30) == nwin:
                act[r["dataset"]] = np.array(r["active"])
        tasks = [dict(key=key_of("n", ds, n, nwin), kind="n", dataset=ds, n=n, n_win=nwin)
                 for ds in sorted(act) for n in eligible_n(act[ds], elig)]
        print(f"[{time.strftime('%H:%M:%S')}] windows={nwin} elig={elig}: full tasks done in {time.time()-t0:.0f}s; "
              f"{len(tasks)} (dataset, n) cells", flush=True)
        done = Q.run_tasks(T.run_task, tasks, gov, out)
        print(f"[{time.strftime('%H:%M:%S')}] ran {done} new cells; governor {gov.last}", flush=True)
    finally:
        gov.close()


if __name__ == "__main__":
    main()
