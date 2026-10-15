"""Run one stage of the deep programme under the quiet governor. Resumable.

    nice -n 10 caffeinate -i .venv/bin/python3 scripts/ssp/run_deep.py --stage agg|plan|cal|seas|dp [--workers 3] [--smoke LOG]
Rows go to results/small_sample_popularity/02_runs/deep.jsonl (gitignored). Registry: configs/deep_registry.json.
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
BASE = "decay:h=2"
HG = ["decay:h=0.25", "decay:h=0.5", "decay:h=1", "decay:h=4", "decay:h=8", "decay:h=inf"]


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def logs_of(reg, groups):
    sp, out = reg["split"], []
    for g in groups:
        v = sp[g]
        out += (v["amazon"] + v["other"]) if isinstance(v, dict) else v
    return list(dict.fromkeys(out))


def arms_for(reg, stage):
    arms = [reg["baseline"]]
    for h in reg["hypotheses"]:
        if h["stage"] != stage:
            continue
        if h["kind"] == "paired":
            arms += [h["primary"]] + h["secondary"] + [h["ref"]]
        else:
            arms += h["arms"]
    if stage == "plan":
        arms += HG + ["fulldecay:h=2"] + [f"full:lag={l}" for l in (0, 1, 2, 3, 4, 6, 8)]
    if stage == "agg":
        arms += ["decay:h=2@varn", "fulldecay:h=2"]
    return list(dict.fromkeys(arms))


def eligible_n(active, grid):
    nw = len(active)
    return [n for n in grid if sum(1 for w in range(8, nw - 1) if all(active[w - 7: w + 2] >= 5 * n)) >= 10]


def main():
    import deep_core as C
    import track_core as T
    reg = json.load(open("configs/deep_registry.json"))
    stage = arg("--stage")
    st = reg["stages"][stage]
    smoke = arg("--smoke")
    out = "/tmp/deep_smoke.jsonl" if smoke else f"{RUNS}/deep.jsonl"
    logs = [smoke] if smoke else [l for l in logs_of(reg, st["logs"]) if os.path.exists(f"datasets/npz/{l}.npz")]
    arms_all = arms_for(reg, stage)
    lim = Q.Limits.from_json("configs/ssp_quiet.json")
    lim.max_workers = int(arg("--workers", 3))
    os.makedirs(RUNS, exist_ok=True)
    gov = Q.Governor(lim, log_path="results/small_sample_popularity/00_safety/governor_log_deep.jsonl")
    try:
        tasks = []
        for nw in st["n_win"]:
            for ds in logs:
                D = T.prepare(ds, nw)
                lag = int(round(365.0 / D["window_days"]))
                span = (nw - 2) - 8 + 1
                active = max(0, (nw - 2) - max(8, lag + 1) + 1) / span if span > 0 else 0.0
                seas_ok = D["window_days"] <= 182 and lag >= 2 and active >= 0.5           # seasonal term active in at least half of the steady steps
                arms = [a for a in arms_all if not (a.startswith("seas") and not seas_ok)]
                if stage == "seas" and len(arms) <= 1:
                    continue
                for n in eligible_n(D["active"], st["n"]):
                    tasks.append(dict(key=f"{stage}|{ds}|{n}|{nw}", dataset=ds, n=n, n_win=nw, draws=st["draws"], arms=arms, seed_tag=st["seed_tag"]))
        if not smoke:
            json.dump({t["key"]: t["arms"] for t in tasks}, open(f"{RUNS}/deep_expected_{stage}.json", "w"))
        print(f"[{time.strftime('%H:%M:%S')}] stage {stage}: {len(tasks)} cells over {len(set(t['dataset'] for t in tasks))} logs, {len(arms_all)} arms", flush=True)
        done = Q.run_tasks(C.run_task, tasks, gov, out)
        print(f"[{time.strftime('%H:%M:%S')}] stage {stage}: ran {done} new cells; governor {gov.last}", flush=True)
    finally:
        gov.close()


if __name__ == "__main__":
    main()
