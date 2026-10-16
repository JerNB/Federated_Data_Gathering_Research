"""Stress tests of the selector (stage S) under the quiet governor. Resumable.

    nice -n 10 caffeinate -i .venv/bin/python3 -W ignore scripts/ssp/run_stress.py --family bias|churn|dp [--workers 3] [--plan] [--smoke LOG]

Rows go to results/small_sample_popularity/02_runs/stress.jsonl (gitignored); --smoke writes to /tmp instead. Registry:
configs/selfsel_stress_registry.json. The cells are built from the STORED rows of earlier runs (02_runs/confirm.jsonl for windows w2,
02_runs/weekly.jsonl for windows w1): same logs, same n, same std arms, same seed tag, so the std arms reproduce the stored cells.
--plan only builds the cells, writes 02_runs/stress_plan.json (every key) and prints the counts.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import quiet as Q  # noqa: E402  (one BLAS thread per process before numpy)

RUNS = "results/small_sample_popularity/02_runs"
REG = "configs/selfsel_stress_registry.json"


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def rows(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def days_arms(arms):
    return [a for a in arms if a.startswith("decay:days=") and "@" not in a]


def build(reg, family, only=None):
    st = reg["stage"]
    tasks = []
    if family in ("bias", "churn"):
        c = st[family]
        for r in rows(f"{RUNS}/confirm.jsonl"):
            if not r["dataset"].endswith("@w2") or r["round_days"] != c["round_days"] or r["n"] not in c["n"]:
                continue
            if only and r["log"] != only:
                continue
            arms = list(r["arms"])
            if "decay:days=14" not in arms:
                sys.exit(f"the 2-round arm (14 days) is missing from {r['key']}")
            t = dict(family=family, window="w2", dataset=r["dataset"], log=r["log"], n=r["n"], n_win=r["n_win"], round_days=r["round_days"],
                     draws=len(r["nd100"]), seed_tag=c["seed_tag"], key=f"s|{family}|{r['dataset']}|{r['n']}|{r['round_days']}")
            if family == "bias":
                t["arms"] = arms + [f"{a}@{s}" for s in c["samplers"] for a in days_arms(arms)]
                t["audit_n"] = max(10, r["n"] // 2)
            else:
                t["arms"], t["save_overlap"] = arms, True
            tasks.append(t)
    elif family == "dp":
        c = st["dp"]
        for r in rows(f"{RUNS}/weekly.jsonl"):
            if r.get("kind") != "deep" or r["round_days"] != c["round_days"] or r["n"] not in c["n"] or "@frac" in r["arms"][0]:
                continue
            stage = r["key"].split("|", 1)[0]
            if (r["n"], stage) not in ((1000, "a"), (3000, "d")):
                continue
            if only and r["dataset"] != only:
                continue
            arms = list(r["arms"])
            if "decay:days=14" not in arms:
                sys.exit(f"the 2-round arm (14 days) is missing from {r['key']}")
            dp = [f"{a}@gauss:eps={e},c={c['c']}" for e in c["eps"] for a in days_arms(arms)]
            dp += [f"{a}@gauss:eps={c['fresh_eps']},c={c['c']},fresh=1" for a in days_arms(arms)]
            dp += [f"{a}@gauss:eps={c['reference_eps']},c={c['c']}" for a in days_arms(arms)]
            tasks.append(dict(family="dp", window="w1", dataset=r["dataset"], n=r["n"], n_win=r["n_win"], round_days=r["round_days"],
                              draws=len(r["nd100"]), seed_tag=c["seed_tag_by_n"][str(r["n"])], arms=arms + dp,
                              key=f"s|dp|{r['dataset']}|{r['n']}|{r['round_days']}"))
    else:
        sys.exit("family must be bias, churn or dp")
    order = {"w1": 0, "w2": 1}
    tasks.sort(key=lambda t: (order[t["window"]], t["dataset"], t["n"]))
    return tasks


def main():
    reg = json.load(open(REG))
    family, smoke = arg("--family"), arg("--smoke")
    tasks = build(reg, family, smoke)
    print(f"[{time.strftime('%H:%M:%S')}] family {family}: {len(tasks)} cells over {len(set(t['dataset'] for t in tasks))} logs; "
          f"arms per cell {sorted(set(len(t['arms']) for t in tasks))}", flush=True)
    os.makedirs(RUNS, exist_ok=True)
    if "--plan" in sys.argv:
        plan_path = f"{RUNS}/stress_plan.json"
        plan = json.load(open(plan_path)) if os.path.exists(plan_path) else {}
        plan[family] = [t["key"] for t in tasks]
        json.dump(plan, open(plan_path, "w"), indent=1)
        return
    import stress_core as S
    out = "/tmp/stress_smoke.jsonl" if smoke else f"{RUNS}/stress.jsonl"
    lim = Q.Limits.from_json("configs/ssp_quiet.json")
    lim.max_workers = int(arg("--workers", 3))
    gov = Q.Governor(lim, log_path="results/small_sample_popularity/00_safety/governor_log_stress.jsonl")
    try:
        done = Q.run_tasks(S.run_task, tasks, gov, out)
        print(f"[{time.strftime('%H:%M:%S')}] ran {done} new cells; governor {gov.last}", flush=True)
    finally:
        gov.close()


if __name__ == "__main__":
    main()
