"""Programme P (price of privacy) under the quiet governor. Resumable.

    nice -n 10 caffeinate -i .venv/bin/python3 -W ignore scripts/ssp/run_price.py [--workers 3] [--plan] [--smoke LOG]

Rows go to results/small_sample_popularity/02_runs/price.jsonl (gitignored); --smoke writes to /tmp instead. Registry:
configs/selfsel_price_registry.json. The test cells are built from the STORED confirmation cells (02_runs/confirm.jsonl, windows w2, 7-day
rounds): same logs, same people per round, same std arms, same seed tag, so the std arms reproduce the stored cell exactly. Each cell adds
the days arms under the registered samplers (hard cap c = 10, norm clips) at every registered eps and the capped / clipped noise-free
references (eps = 1e6). --plan only builds the cells, writes 02_runs/price_plan.json (every key) and prints the counts.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import quiet as Q  # noqa: E402  (one BLAS thread per process before numpy)

RUNS = "results/small_sample_popularity/02_runs"
REG = "configs/selfsel_price_registry.json"


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def rows(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def days_arms(arms):
    return [a for a in arms if a.startswith("decay:days=") and "@" not in a]


def scheme_list(reg):
    """Every sampler scheme string of the registry, in the registry's order."""
    return list(reg["stage"]["schemes"])


def build(reg, only=None):
    st = reg["stage"]
    schemes = scheme_list(reg)
    tasks = []
    for r in rows(f"{RUNS}/confirm.jsonl"):
        if not r["dataset"].endswith("@w2") or r["round_days"] != st["round_days"] or r["n"] not in st["n"]:
            continue
        if only and r["log"] != only:
            continue
        arms = list(r["arms"])
        if "decay:days=14" not in arms:
            sys.exit(f"the 2-round arm (14 days) is missing from {r['key']}")
        tasks.append(dict(family="price", window="w2", dataset=r["dataset"], log=r["log"], n=r["n"], n_win=r["n_win"], round_days=r["round_days"],
                          draws=len(r["nd100"]), seed_tag=st["seed_tag"], arms=arms + [f"{a}@{s}" for s in schemes for a in days_arms(arms)],
                          key=f"p|test|{r['dataset']}|{r['n']}|{r['round_days']}"))
    tasks.sort(key=lambda t: (-t["n"], t["dataset"]))               # the cells that carry hypotheses (n = 1000, 300) run first
    return tasks


def main():
    reg = json.load(open(REG))
    smoke = arg("--smoke")
    tasks = build(reg, smoke)
    print(f"[{time.strftime('%H:%M:%S')}] {len(tasks)} cells over {len(set(t['dataset'] for t in tasks))} logs; "
          f"arms per cell {sorted(set(len(t['arms']) for t in tasks))}", flush=True)
    os.makedirs(RUNS, exist_ok=True)
    if "--plan" in sys.argv:
        json.dump([t["key"] for t in tasks], open(f"{RUNS}/price_plan.json", "w"), indent=1)
        return
    import price_core as P
    out = "/tmp/price_smoke.jsonl" if smoke else f"{RUNS}/price.jsonl"
    lim = Q.Limits.from_json("configs/ssp_quiet.json")
    lim.max_workers = int(arg("--workers", 3))
    gov = Q.Governor(lim, log_path="results/small_sample_popularity/00_safety/governor_log_price.jsonl")
    try:
        done = Q.run_tasks(P.run_task, tasks, gov, out)
        print(f"[{time.strftime('%H:%M:%S')}] ran {done} new cells; governor {gov.last}", flush=True)
    finally:
        gov.close()


if __name__ == "__main__":
    main()
