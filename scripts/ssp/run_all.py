"""Build every task list and run them under the quiet governor.  Resumable: rerun = continue.

    nice -n 10 caffeinate -i .venv/bin/python3 scripts/ssp/run_all.py            # everything
    nice -n 10 caffeinate -i .venv/bin/python3 scripts/ssp/run_all.py e1 e2      # some stages
    .venv/bin/python3 scripts/ssp/run_all.py --count                             # only count tasks

Raw rows go to results/small_sample_popularity/02_runs/<stage>.jsonl (one JSON row per task).
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import quiet as Q  # sets one BLAS thread per process before numpy is imported

import numpy as np

RUNS = "results/small_sample_popularity/02_runs"
DATASETS = ["ml-latest", "ml-1m", "amazon-video-games", "amazon-musical-instruments", "amazon-software",
            "amazon-movies-tv", "amazon-books", "gowalla", "brightkite", "kuairand-pure", "lastfm-1k"]
N_GRID = [20, 40, 100, 200, 400, 1000, 2000, 4000, 10000, 20000]
K_GRID = [1, 2, 5, 10, 20, None]          # None = every earlier choice
R_E1 = 60                                 # draws per cell, popularity stage
R_E2 = 20                                 # draws per cell, model-training stage
R_E4 = 30
R_E5 = 40
R_E6 = 20
R_E7 = 30
R_E2P = 12
R_E8 = 20
E8_DATASETS = ["ml-latest", "ml-1m", "amazon-video-games", "amazon-musical-instruments", "amazon-software", "kuairand-pure"]
E2_DATASETS = ["ml-latest", "ml-1m", "amazon-video-games", "amazon-musical-instruments",
               "amazon-software", "kuairand-pure", "gowalla"]
E4_DATASETS = ["ml-latest", "ml-1m", "amazon-video-games", "kuairand-pure", "gowalla", "amazon-movies-tv"]
E6_DATASETS = ["ml-latest", "ml-1m", "kuairand-pure", "amazon-video-games", "gowalla"]
GENRES = ["Horror", "Documentary", "Sci-Fi", "Romance", "Children", "Western", "Animation", "Film-Noir"]
EPS = [0.5, 1.0, 2.0, 4.0, 8.0]

_meta = {}


def meta(ds):
    if not _meta:
        for r in json.load(open("results/small_sample_popularity/01_datasets/datasets.json")):
            _meta[r["name"]] = r
    return _meta[ds]


def n_grid(ds, cap_frac=0.4):
    return [n for n in N_GRID if n <= cap_frac * meta(ds)["n_pop"]]


def k_grid(ds, ks=K_GRID):
    mr = meta(ds)["min_rows"]
    return [k for k in ks if k is None or k <= mr]


def T(kind, ds, draw, **kw):
    key = f"{kind}|{ds}|" + "|".join(f"{k}={kw[k]}" for k in sorted(kw)) + f"|d={draw}"
    return dict(kind=kind, key=key, dataset=ds, draw=draw, **kw)


def build(stage):
    out = []
    if stage == "ref":
        out = [dict(kind="ref", key=f"ref|{ds}", dataset=ds) for ds in DATASETS]
    elif stage == "e1":
        for ds in DATASETS:
            for N in n_grid(ds):
                for k in k_grid(ds):
                    for d in range(R_E1):
                        out.append(T("e1", ds, d, N=N, k=k))
    elif stage == "e2":
        for ds in E2_DATASETS:
            for N in [n for n in (100, 200, 400, 1000, 2000, 4000) if n <= 0.4 * meta(ds)["n_pop"]]:
                for k in k_grid(ds, [5, 10, 20, None]):
                    for d in range(R_E2):
                        out.append(T("e2", ds, d, N=N, k=k))
    elif stage == "e2c":      # same cells as e2 on fresh draws (100+), also storing the in-sample hold-out gaps
        for ds in E2_DATASETS:
            for N in [n for n in (100, 200, 400, 1000, 2000, 4000) if n <= 0.4 * meta(ds)["n_pop"]]:
                for k in k_grid(ds, [5, 10, 20, None]):
                    for d in range(100, 100 + R_E2):
                        out.append(dict(T("e2c", ds, d, N=N, k=k), cvgap=True))
    elif stage == "e2p":
        for ds in E2_DATASETS:
            for N in [n for n in (100, 200, 400, 1000, 2000, 4000) if n <= 0.4 * meta(ds)["n_pop"]]:
                for k in (5, None):
                    for d in range(R_E2P):
                        out.append(T("e2p", ds, d, N=N, k=k))
    elif stage == "e8":
        for ds in E8_DATASETS:
            for N in [n for n in (100, 200, 400, 1000, 2000, 4000) if n <= 0.4 * meta(ds)["n_pop"]]:
                for k in (5, None):
                    for d in range(R_E8):
                        out.append(T("e8", ds, d, N=N, k=k))
    elif stage == "e4":
        for ds in E4_DATASETS:
            for N in [n for n in (100, 400, 1000, 4000) if n <= 0.4 * meta(ds)["n_pop"]]:
                for k in (5, None):
                    for d in range(R_E4):
                        out.append(T("e4", ds, d, N=N, k=k))
    elif stage == "e5":
        for ds in DATASETS:
            groups = [f"activity:{i}" for i in range(3)] + [f"joined:{i}" for i in range(3)]
            if ds in ("ml-latest", "ml-1m"):
                groups += [f"genre:{g}" for g in GENRES]
            for g in groups:
                # genre fans = top 15% by genre share; terciles = one third of the group
                pool = meta(ds)["n_pop"] * (0.15 if g.startswith("genre:") else 1 / 3)
                for N in [n for n in (10, 20, 50, 100, 200, 500, 1000) if n <= 0.4 * pool]:
                    for k in (5, None):
                        for d in range(R_E5):
                            out.append(T("e5", ds, d, group=g, N=N, k=k))
    elif stage == "e6":
        for ds in E6_DATASETS:
            for N in [n for n in (200, 1000, 5000, 20000) if n <= 0.4 * meta(ds)["n_pop"]]:
                for k in (1, 5, 20):
                    for eps in EPS:
                        for d in range(R_E6):
                            out.append(T("e6", ds, d, N=N, k=k, eps=eps))
    elif stage == "e7":      # biased samplers, same estimators as e1
        arms = {"activity": dict(users="activity"), "inv_activity": dict(users="inv_activity"),
                "recent_rows": dict(rows="recent"), "recent_window": dict(window=True),
                "boot": dict(users="replace")}
        for ds in DATASETS:
            for arm, kw in arms.items():
                for N in n_grid(ds):
                    # 'boot' (people drawn with replacement) carries the whole k grid: it is the clean C1 check
                    for k in k_grid(ds, K_GRID if arm == "boot" else [5, 20, None]):
                        if arm == "recent_rows" and k is None:
                            continue
                        for d in range(20 if arm == "boot" else R_E7):
                            out.append(dict(T("e1", ds, d, N=N, k=k, arm=arm), **kw))
    else:
        raise SystemExit(f"unknown stage {stage}")
    return out


STAGES = ["ref", "e2", "e1", "e2p", "e2c", "e8", "e4", "e6", "e5", "e7"]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    stages = args or STAGES
    if "--count" in sys.argv:
        for s in stages:
            print(s, len(build(s)))
        return
    import exp_core as X  # noqa: E402  (imported late so quiet.py's env vars apply first)
    lim = Q.Limits.from_json("configs/ssp_quiet.json")
    os.makedirs(RUNS, exist_ok=True)
    gov = Q.Governor(lim, log_path="results/small_sample_popularity/00_safety/governor_log.jsonl")
    try:
        for s in stages:
            tasks = build(s)
            out = f"{RUNS}/{s}.jsonl"
            t0 = time.time()
            n = Q.run_tasks(X.run_task, tasks, gov, out)
            print(f"[{time.strftime('%H:%M:%S')}] stage {s}: ran {n} of {len(tasks)} tasks "
                  f"in {time.time()-t0:.0f}s; governor {gov.last}", flush=True)
    finally:
        gov.close()


if __name__ == "__main__":
    main()
