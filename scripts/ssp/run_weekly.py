"""Weekly-cadence programme under the quiet governor. Resumable.

    nice -n 10 caffeinate -i .venv/bin/python3 -W ignore scripts/ssp/run_weekly.py --stage a|b|c [--workers 3] [--smoke LOG]
Rows go to results/small_sample_popularity/02_runs/weekly.jsonl (gitignored). Registry: configs/weekly_registry.json.
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


def all_logs(reg):
    sp, out = reg["split"], []
    for g in ("discovery", "holdout_old", "holdout_fresh", "new_discovery", "new_holdout"):
        v = sp[g]
        out += (v["amazon"] + v["other"]) if isinstance(v, dict) else v
    return list(dict.fromkeys(out))


def main():
    import track_core as T
    import weekly_core as W
    reg = json.load(open("configs/weekly_registry.json"))
    stage = arg("--stage")
    if stage in ("d", "e"):
        reg2 = json.load(open("configs/weekly_registry_2.json"))
        st = reg2["stages"][stage]
    else:
        st = reg["stages"][stage]
    smoke = arg("--smoke")
    out = "/tmp/weekly_smoke.jsonl" if smoke else f"{RUNS}/weekly.jsonl"
    logs = [smoke] if smoke else [l for l in all_logs(reg) if os.path.exists(f"datasets/npz/{l}.npz")]
    lim = Q.Limits.from_json("configs/ssp_quiet.json")
    lim.max_workers = int(arg("--workers", 3))
    os.makedirs(RUNS, exist_ok=True)
    gov = Q.Governor(lim, log_path="results/small_sample_popularity/00_safety/governor_log_weekly.jsonl")
    try:
        tasks, info = [], {}
        for ds in logs:
            if W.load_span(ds) is None:
                info[ds] = "shorter than 156 weeks"
                continue
            D = W.prepare_weekly(ds, 7)
            act = D["active"]
            med = float(np.median(act))
            info[ds] = dict(median_weekly_active=med, p10=float(np.percentile(act, 10)), p90=float(np.percentile(act, 90)))
            if stage == "a":
                ns = [n for n in st["n"] if med >= 3 * n]
                for rd in reg["round_days"]:
                    arms = ["decay:h=2"] + [f"decay:days={d}" for d in reg["days_grid"]]
                    for n in ns:
                        tasks.append(dict(key=f"a|{ds}|{n}|{rd}", dataset=ds, n=n, n_win=156 * 7 // rd, round_days=rd, draws=st["draws"],
                                          arms=arms, seed_tag=st["seed_tag"]))
            elif stage == "d":
                ns = [n for n in st["n"] if med >= 3 * n]
                for rd in st["round_days"]:
                    arms = ["decay:h=2"] + [f"decay:days={d}" for d in st["days"]]
                    for n in ns:
                        tasks.append(dict(key=f"d|{ds}|{n}|{rd}", dataset=ds, n=n, n_win=156 * 7 // rd, round_days=rd, draws=st["draws"],
                                          arms=arms, seed_tag=st["seed_tag"]))
            elif stage == "e":
                for n in st["n"]:
                    if med >= 3 * n:
                        tasks.append(dict(key=f"e|{ds}|{n}", kind="cadence", match_rows=True, dataset=ds, n=n, draws=st["draws"], days=st["days"]))
            elif stage == "b":
                n = st["n"][0]
                if med < 3 * n:
                    continue
                f = round(n / med, 5)
                sch = f"@frac:f={f}"
                rd = 7
                d = reg["ref_days"]
                arms = [f"decay:days={d}{sch}"] + [f"ivw:g={g},days={d}{sch}" for g in (0.5, 1, 2)]
                tasks.append(dict(key=f"b|{ds}|{n}|{rd}", dataset=ds, n=n, n_win=156, round_days=rd, draws=st["draws"], arms=arms,
                                  seed_tag=st["seed_tag"], f=f))
            elif stage == "c":
                for n in st["n"]:
                    if med >= 3 * n:
                        tasks.append(dict(key=f"c|{ds}|{n}", kind="cadence", dataset=ds, n=n, draws=st["draws"], days=st["days"]))
        if not smoke:
            json.dump(info, open(f"{RUNS}/weekly_logs.json", "w"), indent=1)
        print(f"[{time.strftime('%H:%M:%S')}] stage {stage}: {len(tasks)} cells over {len(set(t['dataset'] for t in tasks))} logs", flush=True)
        T._CACHE.clear()
        done = Q.run_tasks(W.run_task, tasks, gov, out)
        print(f"[{time.strftime('%H:%M:%S')}] stage {stage}: ran {done} new cells; governor {gov.last}", flush=True)
    finally:
        gov.close()


if __name__ == "__main__":
    main()
