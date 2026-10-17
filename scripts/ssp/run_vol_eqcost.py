"""Post hoc equal-cost baseline for programme V (NOT registered; added after the critic's review of doc 19).

Question: the audit sample of programme V costs max(10, n/2) extra random people per round. Is it better to use those people as random reporters?
Random-people cells of window w3 (7-day rounds, std sampler, the selector of docs 14-15 on the reporters' self-score) are run at 50 and 150 people (the
audit sizes of the 100- and 300-people cells, and 1.5 times 100), for every log with median weekly active >= 3 n (the rule of programme G). The cells use
the same seed tag, draws and arms rule as programme G (days list of the nearest registered n: 50 -> 30, 150 -> 100).

    nice -n 10 caffeinate -i .venv/bin/python3 -W ignore scripts/ssp/run_vol_eqcost.py --plan | --run [--workers 3] | --report

Rows go to 02_runs/vol_eqcost.jsonl (gitignored); --report writes 03_tables/volunteer/posthoc_equal_cost*.csv and needs 02_runs/vol.jsonl.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import quiet as Q  # noqa: E402  (one BLAS thread per process before numpy)

RUNS = "results/small_sample_popularity/02_runs"
OUT = "results/small_sample_popularity/03_tables/volunteer"
KREG = "configs/selfsel_confirm_registry.json"
NS = [50, 150]
NEAREST = {50: "30", 150: "100"}
SEED_TAG = "guard-w3"
DRAWS = 8


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def build():
    import numpy as np
    import track_core as T
    import window3_core as X
    days_by_n = json.load(open(KREG))["stage"]["days_by_n"]
    logs = sorted({k.split("|")[2][:-3] for k in json.load(open(f"{RUNS}/vol_plan.json"))})
    tasks = []
    for log in logs:
        got = X.load_window3(log)
        D = X.prepare_window3(log, 7)
        med = float(np.median(D["active"]))
        for n in NS:
            if med >= 3 * n:
                arms = ["decay:h=2"] + [f"decay:days={d}" for d in days_by_n[NEAREST[n]]]
                tasks.append(dict(key=f"eq|{log}@w3|{n}|7", dataset=f"{log}@w3", log=log, n=n, n_win=got[3] * 7 // 7, round_days=7, draws=DRAWS, arms=arms,
                                  seed_tag=SEED_TAG, family="eqcost", window="w3"))
    T._CACHE.clear()
    tasks.sort(key=lambda t: (-t["n"], t["dataset"]))
    return tasks


def run():
    tasks = build()
    print(f"[{time.strftime('%H:%M:%S')}] {len(tasks)} cells over {len(set(t['log'] for t in tasks))} logs; n=50: {sum(t['n'] == 50 for t in tasks)}, n=150: {sum(t['n'] == 150 for t in tasks)}", flush=True)
    if "--plan" in sys.argv:
        return
    import weight_core as V
    lim = Q.Limits.from_json("configs/ssp_quiet.json")
    lim.max_workers = int(arg("--workers", 3))
    gov = Q.Governor(lim, log_path="results/small_sample_popularity/00_safety/governor_log_eqcost.jsonl")
    try:
        done = Q.run_tasks(V.run_task, tasks, gov, f"{RUNS}/vol_eqcost.jsonl")
        print(f"[{time.strftime('%H:%M:%S')}] ran {done} new cells; governor {gov.last}", flush=True)
    finally:
        gov.close()


def std_value(r: dict) -> float:
    """Selector value of a random-people cell: the selector of docs 14-15 on the reporters' self-score over the days arms (as rep0 in an_vol)."""
    import numpy as np
    import explore_selfsel_weekly as E
    nd, se = np.array(r["nd100"], float), np.array(r["selfnd"], float)
    st = [i for i, s in enumerate(r["steps"]) if s >= 8]
    idx = [i for i, a in enumerate(r["arms"]) if a.startswith("decay:days=") and "@" not in a]
    ds = np.array([int(r["arms"][i].split("=")[1]) for i in idx])
    order = np.argsort(ds)
    idx, ds = [idx[j] for j in order], ds[order]
    return E.chosen_value(nd[:, idx, :], E.choose(se[:, idx, :], st, E.HL, E.nearest(ds, 14)), st)


def report():
    import numpy as np
    import an_vol as A
    new = {}
    errs = f"{RUNS}/vol_eqcost.jsonl.errors"
    if os.path.exists(errs) and os.path.getsize(errs) > 0:
        sys.exit(f"{errs} is not empty")
    for line in open(f"{RUNS}/vol_eqcost.jsonl"):
        r = json.loads(line)
        new[(r["log"], r["n"])] = std_value(r)
    tasks = build()
    if sorted(new) != sorted((t["log"], t["n"]) for t in tasks):
        sys.exit("the equal-cost cells are not exactly the planned ones")
    per = A.build_per(A.load_rows(f"{RUNS}/vol.jsonl"))
    val = lambda sampler, n, rule: {k[0]: per[k][sampler][rule] for k in per if k[1] == n}
    ref = {n: val("std", n, "rep0") for n in (100, 300)}
    base = {n: {L: v for (L, m), v in new.items() if m == n} for n in NS}
    comps = []                                       # (label, {log: value of the first}, {log: value of the second})
    for n, audit, big in ((100, 50, 150), (300, 150, None)):
        for s, name in (("A+", "heavy volunteers"), ("A-", "light volunteers"), ("std", "random people")):
            aw = val(s, n, "audw")
            comps.append((f"{name}, {n} reporters + audit of {audit} (audit-weighted) minus the audit alone as {audit} random reporters", aw, base[audit]))
            comps.append((f"{name}, {n} reporters + audit of {audit} (audit-weighted) minus {n} random reporters (no audit)", aw, ref[n]))
            if big:
                comps.append((f"{name}, {n} reporters + audit of {audit} (audit-weighted) minus {big} random reporters (same head count)", aw, base[big]))
    body = []
    for label, a, b in comps:
        logs = sorted(set(a) & set(b))
        d = {L: a[L] - b[L] for L in logs}
        st = A.stat(d) if d else None
        body.append([label, len(logs), round(st["mean"], 6), round(st["lo"], 6), round(st["hi"], 6), round(st["share_positive"], 3)] if st else [label, 0, "", "", "", ""])
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/posthoc_equal_cost.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["comparison (selector values, paired per log)", "logs", "mean", "ci_lo", "ci_hi", "share_logs_positive"])
        w.writerows(body)
    with open(f"{OUT}/posthoc_equal_cost_values.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["quantity", "logs", "mean_selector_value"])
        for n in NS:
            w.writerow([f"random people, {n} reporters (new cells)", len(base[n]), round(float(np.mean(list(base[n].values()))), 6)])
        for n in (100, 300):
            w.writerow([f"random people, {n} reporters (programme V std cells)", len(ref[n]), round(float(np.mean(list(ref[n].values()))), 6)])
            for s in ("A+", "A-"):
                v = val(s, n, "audw")
                w.writerow([f"{s} audit-weighted, {n} reporters + audit of {max(10, n // 2)}", len(v), round(float(np.mean(list(v.values()))), 6)])
    for row in body:
        print(row)


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif "--run" in sys.argv or "--plan" in sys.argv:
        run()
    else:
        sys.exit(__doc__)
