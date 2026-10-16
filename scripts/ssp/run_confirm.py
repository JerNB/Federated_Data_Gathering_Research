"""Selector confirmation (stage K) under the quiet governor. Resumable.

    nice -n 10 caffeinate -i .venv/bin/python3 -W ignore scripts/ssp/run_confirm.py [--plan] [--workers 3] [--smoke LOG]

Rows go to results/small_sample_popularity/02_runs/confirm.jsonl (gitignored); --smoke writes to /tmp instead.
Registry: configs/selfsel_confirm_registry.json. --plan only cuts the windows, prints the cell list and writes
02_runs/confirm_logs.json; it runs no simulation.
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


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def build(reg, only=None):
    """(tasks, info): every (log, n, round length) cell the registry allows, in an order that reuses the cut window."""
    import confirm_core as K
    import track_core as T
    st = reg["stage"]
    arms_of = lambda n: ["decay:h=2"] + [f"decay:days={d}" for d in st["days_by_n"][str(n)]]
    tasks, info = [], {}
    for group, logs in (("w2", reg["logs"]["w2"]), ("whole", reg["logs"]["whole"])):
        for ds in logs:
            if only and ds != only:
                continue
            if not os.path.exists(f"datasets/npz/{ds}.npz"):
                info[ds] = "log file missing"
                continue
            got = K.load_window(ds)
            if got is None:
                info[ds] = "no free 156-week span"
                continue
            _, _, _, kind, n_weeks = got
            if kind != group:
                info[ds] = f"window kind {kind} but registry lists it under {group}"
                continue
            D = K.prepare_window(ds, 7)
            act = D["active"]
            med = float(np.median(act))
            ns = [n for n in st["n"] if med >= 3 * n]
            info[ds] = dict(kind=kind, n_weeks=n_weeks, median_weekly_active=med, p10=float(np.percentile(act, 10)),
                            p90=float(np.percentile(act, 90)), eligible_n=ns)
            key = K.dataset_key(ds, kind)
            for rd in st["round_days"][group]:
                for n in ns:
                    tasks.append(dict(key=f"k|{key}|{n}|{rd}", dataset=key, log=ds, n=n, n_win=n_weeks * 7 // rd, round_days=rd,
                                      draws=st["draws"], arms=arms_of(n), seed_tag=st["seed_tag"]))
    T._CACHE.clear()
    return tasks, info


def main():
    reg = json.load(open("configs/selfsel_confirm_registry.json"))
    smoke = arg("--smoke")
    tasks, info = build(reg, smoke)
    by = {}
    for t in tasks:
        by.setdefault((t["round_days"], t["dataset"].endswith("@w2")), []).append(t)
    print(f"[{time.strftime('%H:%M:%S')}] {len(tasks)} cells over {len(set(t['log'] for t in tasks))} logs; "
          + "; ".join(f"{rd}-day {'w2' if w else 'whole'}: {len(v)}" for (rd, w), v in sorted(by.items())), flush=True)
    if "--plan" in sys.argv:
        os.makedirs(RUNS, exist_ok=True)
        json.dump(info, open(f"{RUNS}/confirm_logs.json", "w"), indent=1)
        for ds, v in info.items():
            print(ds, v if isinstance(v, str) else (v["kind"], v["n_weeks"], v["median_weekly_active"], v["eligible_n"]))
        return
    import confirm_core as K
    out = "/tmp/confirm_smoke.jsonl" if smoke else f"{RUNS}/confirm.jsonl"
    lim = Q.Limits.from_json("configs/ssp_quiet.json")
    lim.max_workers = int(arg("--workers", 3))
    os.makedirs(RUNS, exist_ok=True)
    gov = Q.Governor(lim, log_path="results/small_sample_popularity/00_safety/governor_log_confirm.jsonl")
    try:
        if not smoke:
            json.dump(info, open(f"{RUNS}/confirm_logs.json", "w"), indent=1)
        done = Q.run_tasks(K.run_task, tasks, gov, out)
        print(f"[{time.strftime('%H:%M:%S')}] ran {done} new cells; governor {gov.last}", flush=True)
    finally:
        gov.close()


if __name__ == "__main__":
    main()
