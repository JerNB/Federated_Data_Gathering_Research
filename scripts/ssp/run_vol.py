"""Programme V (volunteer-bias weights) under the quiet governor. Resumable.

    nice -n 10 caffeinate -i .venv/bin/python3 -W ignore scripts/ssp/run_vol.py [--workers 3] [--plan] [--smoke LOG]

Rows go to results/small_sample_popularity/02_runs/vol.jsonl (gitignored); --smoke writes to /tmp instead. Registry: configs/selfsel_vol_registry.json.
The test cells are built from the STORED window-3 rows of programme G (02_runs/guard.jsonl: 7-day rounds): same logs, same people per round, same std arms,
same seed tag, so the std arms reproduce the stored cell exactly. Each cell adds the days arms under every registered (sampler, weight exponent) pair and
draws an unbiased audit sample of max(10, n/2) random people per round. --plan only builds the cells, writes 02_runs/vol_plan.json and prints the counts.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import quiet as Q  # noqa: E402  (one BLAS thread per process before numpy)

RUNS = "results/small_sample_popularity/02_runs"
REG = "configs/selfsel_vol_registry.json"


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def rows(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def days_arms(arms):
    return [a for a in arms if a.startswith("decay:days=") and "@" not in a]


def scheme(template: str, b: float) -> str:
    return template.format(b=f"{b:g}")


def scheme_list(reg):
    """(sampler name, beta, scheme string) for every non-stored pair, in the registry's order. std with beta = 0 is the stored cell itself."""
    st = reg["stage"]
    out = []
    for name, tpl in st["samplers"].items():
        for b in st["betas"]:
            if name == "std" and b == 0:
                continue
            out.append((name, b, scheme(tpl, b)))
    return out


def build(reg, only=None):
    st = reg["stage"]
    pairs = scheme_list(reg)
    tasks = []
    for r in rows(f"{RUNS}/guard.jsonl"):
        if not r["dataset"].endswith("@w3") or r["round_days"] != st["round_days"] or r["n"] not in st["n"]:
            continue
        if only and r["log"] != only:
            continue
        arms = list(r["arms"])
        if "decay:days=14" not in arms:
            sys.exit(f"the 2-round arm (14 days) is missing from {r['key']}")
        tasks.append(dict(family="vol", window="w3", dataset=r["dataset"], log=r["log"], n=r["n"], n_win=r["n_win"], round_days=r["round_days"],
                          draws=len(r["nd100"]), seed_tag=st["seed_tag"], arms=arms + [f"{a}@{s}" for _, _, s in pairs for a in days_arms(arms)],
                          audit_n=max(10, r["n"] // 2), key=f"v|test|{r['dataset']}|{r['n']}|{r['round_days']}"))
    tasks.sort(key=lambda t: (-t["n"], t["dataset"]))
    return tasks


def main():
    reg = json.load(open(REG))
    smoke = arg("--smoke")
    tasks = build(reg, smoke)
    print(f"[{time.strftime('%H:%M:%S')}] {len(tasks)} cells over {len(set(t['dataset'] for t in tasks))} logs; "
          f"arms per cell {sorted(set(len(t['arms']) for t in tasks))}", flush=True)
    os.makedirs(RUNS, exist_ok=True)
    if "--plan" in sys.argv:
        json.dump([t["key"] for t in tasks], open(f"{RUNS}/vol_plan.json", "w"), indent=1)
        return
    import weight_core as V
    out = "/tmp/vol_smoke.jsonl" if smoke else f"{RUNS}/vol.jsonl"
    lim = Q.Limits.from_json("configs/ssp_quiet.json")
    lim.max_workers = int(arg("--workers", 3))
    gov = Q.Governor(lim, log_path="results/small_sample_popularity/00_safety/governor_log_volunteer.jsonl")
    try:
        done = Q.run_tasks(V.run_task, tasks, gov, out)
        print(f"[{time.strftime('%H:%M:%S')}] ran {done} new cells; governor {gov.last}", flush=True)
    finally:
        gov.close()


if __name__ == "__main__":
    main()
