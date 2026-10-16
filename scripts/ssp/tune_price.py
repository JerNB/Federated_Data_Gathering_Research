"""Programme P, development stage: sweep the contribution bound of the central-DP sampler on the DEVELOPMENT windows (w1 = the weekly-study
windows, 7-day rounds). Nothing here is a registered result.

    nice -n 10 caffeinate -i .venv/bin/python3 -W ignore scripts/ssp/tune_price.py --run [--grid 1|2|3] [--workers 3] [--smoke LOG]
    .venv/bin/python3 -W ignore scripts/ssp/tune_price.py --report [--grid 1|2|3]

Grid 1: 1,000 people (stage a, 17 logs), eps = 4, hard caps c = 1, 2, 3, 5, 10 and norm clips C = 1, 1.4142, 2, 3.1623.
Grid 2: 300 people (stage d, 24 logs) and 3,000 people (stage d, 7 logs) at eps = 2, 4, 8, and 1,000 people at eps = 2 and 8, with the registered
        cap c = 10 and norm clips C = 1 and C = 2.
Grid 3: the noise-free references (eps = 1e6) of cap c = 10 and of the norm clips C = 1 and C = 2 at 300, 1,000 and 3,000 people: the loss from
        the bound alone (bias), without noise.
Grid 4: the hard cap c = 1 (one random row per person, sensitivity 1) at eps = 2, 4, 8 and 300, 1,000, 3,000 people: the like-for-like comparison of a
        hard cap with the unit-length clip (same sensitivity).
Cells are built from the STORED weekly cells (02_runs/weekly.jsonl): same logs, same std arms, same seed tag, so the std arms replicate the stored
cell and the `gauss:eps=4,c=10` arms replicate the stress cells of doc 16. Rows go to 02_runs/price_dev.jsonl, price_dev2.jsonl and
price_dev3.jsonl (gitignored); the reports go to 03_tables/price/p_dev_grid.csv, p_dev_grid2.csv and p_dev_grid3.csv.
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
OUT = "results/small_sample_popularity/03_tables/price"
SEED_TAG = {"a": "weekly-a", "d": "weekly-d"}
REF = 1000000


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def grid():
    return int(arg("--grid", 1))


def schemes_for(n: int):
    if grid() == 1:
        return [f"gauss:eps=4,c={c}" for c in (1, 2, 3, 5, 10)] + [f"gaussn:eps=4,C={x:g}" for x in (1.0, 1.4142, 2.0, 3.1623)]
    if grid() == 3:
        return [f"gauss:eps={REF},c=10"] + [f"gaussn:eps={REF},C={x:g}" for x in (1.0, 2.0)]
    if grid() == 4:
        return [f"gauss:eps={e},c=1" for e in (2, 4, 8)]
    eps = (2, 8) if n == 1000 else (2, 4, 8)
    return [f"gauss:eps={e},c=10" for e in eps] + [f"gaussn:eps={e},C={x:g}" for e in eps for x in (1.0, 2.0)]


def all_schemes():
    out = []
    for n in ((1000,) if grid() == 1 else (300, 1000, 3000)):
        out += [s for s in schemes_for(n) if s not in out]
    return out


def wanted():
    return {(1000, "a")} if grid() == 1 else {(300, "d"), (3000, "d"), (1000, "a")}


def build(only=None):
    tasks = []
    for line in open(f"{RUNS}/weekly.jsonl"):
        r = json.loads(line)
        if r.get("kind") != "deep" or r["round_days"] != 7 or "@frac" in r["arms"][0]:
            continue
        stage = r["key"].split("|", 1)[0]
        if (r["n"], stage) not in wanted():
            continue
        if only and r["dataset"] != only:
            continue
        arms = list(r["arms"])
        days = [a for a in arms if a.startswith("decay:days=") and "@" not in a]
        tasks.append(dict(family="pdev", window="w1", dataset=r["dataset"], n=r["n"], n_win=r["n_win"], round_days=7, draws=len(r["nd100"]),
                          seed_tag=SEED_TAG[stage], arms=arms + [f"{a}@{s}" for s in schemes_for(r["n"]) for a in days],
                          key=f"p|dev{grid()}|{r['dataset']}|{r['n']}|7"))
    tasks.sort(key=lambda t: (t["n"], t["dataset"]))
    return tasks


def out_path():
    return f"{RUNS}/price_dev{'' if grid() == 1 else grid()}.jsonl"


def run():
    only = arg("--smoke")
    tasks = build(only)
    print(f"[{time.strftime('%H:%M:%S')}] grid {grid()}: {len(tasks)} dev cells; arms per cell {sorted(set(len(t['arms']) for t in tasks))}", flush=True)
    import price_core as P
    out = "/tmp/price_dev_smoke.jsonl" if only else out_path()
    lim = Q.Limits.from_json("configs/ssp_quiet.json")
    lim.max_workers = int(arg("--workers", 3))
    gov = Q.Governor(lim, log_path="results/small_sample_popularity/00_safety/governor_log_price.jsonl")
    try:
        done = Q.run_tasks(P.run_task, tasks, gov, out)
        print(f"[{time.strftime('%H:%M:%S')}] ran {done} new cells; governor {gov.last}", flush=True)
    finally:
        gov.close()


def report():
    import numpy as np
    import an_stress as AS
    import explore_selfsel_weekly as E
    rows = {}
    for line in open(out_path()):
        r = json.loads(line)
        rows[r["key"]] = r
    per = {}                                        # per[(n, scheme)][log] = rule values
    for r in rows.values():
        r["_nd"], r["_se"] = np.array(r["nd100"], float), np.array(r["selfnd"], float)
        r["_sa"] = None
        for suffix in [""] + [s for s in all_schemes()]:
            c = AS.view(r, suffix)
            if c is None:
                continue
            per.setdefault((r["n"], suffix or "std"), {})[r["dataset"]] = AS.rule_values(c)
    head = ["n", "scheme", "logs", "selector_value", "best_fixed_value", "selector_regret", "price_selector_vs_std", "price_best_vs_std", "amazon_price_selector", "other_price_selector"]
    body = []
    for n in sorted({k[0] for k in per}):
        std = per[(n, "std")]
        for s in ["std"] + all_schemes():
            if (n, s) not in per:
                continue
            ls = sorted(per[(n, s)])
            sel = np.array([per[(n, s)][L]["v_sel"] for L in ls])
            best = np.array([per[(n, s)][L]["best"] for L in ls])
            std_sel = np.array([std[L]["v_sel"] for L in ls])
            std_best = np.array([std[L]["best"] for L in ls])
            am = [i for i, L in enumerate(ls) if E.stratum(L) == "amazon"]
            ot = [i for i, L in enumerate(ls) if E.stratum(L) != "amazon"]
            body.append([n, s, len(ls), round(float(sel.mean()), 5), round(float(best.mean()), 5), round(float((best - sel).mean()), 5),
                         round(float((std_sel - sel).mean()), 5), round(float((std_best - best).mean()), 5),
                         round(float((std_sel - sel)[am].mean()), 5) if am else "", round(float((std_sel - sel)[ot].mean()), 5) if ot else ""])
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/p_dev_grid{'' if grid() == 1 else grid()}.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(head)
        w.writerows(body)
    for row in body:
        print(row)


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif "--run" in sys.argv:
        run()
    else:
        sys.exit(__doc__)
