"""Programme G: window-3 cells under the quiet governor. Resumable.

    nice -n 10 caffeinate -i .venv/bin/python3 -W ignore scripts/ssp/run_guard.py [--workers 3] [--plan] [--smoke LOG]

Rows go to results/small_sample_popularity/02_runs/guard.jsonl (gitignored); --smoke writes to /tmp. Registry: configs/selfsel_guard_registry.json.
One cell = one log, one n (30, 100, 300, 1000), one round length (7 or 28 days) on the log's third window (window3_core). --plan writes
02_runs/guard_plan.json (keys and the start week of every w3 window) and prints the counts; it runs no simulation.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import quiet as Q  # noqa: E402  (one BLAS thread per process before numpy)

import numpy as np  # noqa: E402

RUNS = "results/small_sample_popularity/02_runs"
REG = "configs/selfsel_guard_registry.json"
KREG = "configs/selfsel_confirm_registry.json"


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def build(reg, only=None):
    """(tasks, info): every (log, n, round length) cell of window 3 the registry allows."""
    import track_core as T
    import window3_core as X
    kreg = json.load(open(KREG))
    days_by_n = kreg["stage"]["days_by_n"]
    st = reg["stage"]
    tasks, info = [], {}
    for log in kreg["logs"]["w2"]:
        if only and log != only:
            continue
        if not os.path.exists(f"datasets/npz/{log}.npz"):
            info[log] = "log file missing"
            continue
        got = X.load_window3(log)
        if got is None:
            info[log] = "no third window"
            continue
        D = X.prepare_window3(log, 7)
        med = float(np.median(D["active"]))
        cover = float(np.mean(D["rows"][9:] >= T.MIN_TRUTH_ROWS))          # share of 7-day rounds whose next-round truth has enough rows
        if cover < st["min_truth_coverage"]:
            info[log] = f"truth coverage {cover:.2f} below {st['min_truth_coverage']}"
            continue
        ns = [n for n in st["n"] if med >= 3 * n]
        info[log] = dict(start_week=int(got[4]), n_weeks=int(got[3]), median_weekly_active=med, truth_coverage=round(cover, 3), eligible_n=ns)
        for rd in st["round_days"]:
            for n in ns:
                arms = ["decay:h=2"] + [f"decay:days={d}" for d in days_by_n[str(n)]]
                tasks.append(dict(key=f"g|{log}@w3|{n}|{rd}", dataset=f"{log}@w3", log=log, n=n, n_win=got[3] * 7 // rd, round_days=rd,
                                  draws=st["draws"], arms=arms, seed_tag=st["seed_tag"]))
    T._CACHE.clear()
    return tasks, info


def main():
    reg = json.load(open(REG))
    smoke = arg("--smoke")
    tasks, info = build(reg, smoke)
    print(f"[{time.strftime('%H:%M:%S')}] {len(tasks)} cells over {len(set(t['log'] for t in tasks))} logs; "
          f"7-day {sum(t['round_days'] == 7 for t in tasks)}, 28-day {sum(t['round_days'] == 28 for t in tasks)}", flush=True)
    os.makedirs(RUNS, exist_ok=True)
    if not smoke:
        json.dump(dict(keys=[t["key"] for t in tasks], info=info, meta=dict(seed_tag=reg["stage"]["seed_tag"], draws=reg["stage"]["draws"])), open(f"{RUNS}/guard_plan.json", "w"), indent=1)
    if "--plan" in sys.argv:
        for log, v in info.items():
            print(log, v if isinstance(v, str) else (v["start_week"], v["median_weekly_active"], v["eligible_n"]))
        return
    import window3_core as X
    out = "/tmp/guard_smoke.jsonl" if smoke else f"{RUNS}/guard.jsonl"
    lim = Q.Limits.from_json("configs/ssp_quiet.json")
    lim.max_workers = int(arg("--workers", 3))
    gov = Q.Governor(lim, log_path="results/small_sample_popularity/00_safety/governor_log_guard.jsonl")
    try:
        done = Q.run_tasks(X.run_task, tasks, gov, out)
        print(f"[{time.strftime('%H:%M:%S')}] ran {done} new cells; governor {gov.last}", flush=True)
    finally:
        gov.close()


if __name__ == "__main__":
    main()
