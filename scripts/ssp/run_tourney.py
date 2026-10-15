"""Run one stage of the hypothesis tournament under the quiet governor. Resumable.

    nice -n 10 caffeinate -i .venv/bin/python3 scripts/ssp/run_tourney.py --stage screen
    ... --stage full | holdout | cal
    ... --stage screen --smoke ml-latest        # one log, own file in /tmp

Stages and gates are fixed in configs/tourney_registry.json. Later stages run only the hypotheses that passed the
earlier gate (results/small_sample_popularity/03_tables/tourney_survivors.json, written by an_tourney.py).
Rows go to results/small_sample_popularity/02_runs/tourney.jsonl (gitignored).
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import quiet as Q  # noqa: E402  (sets one BLAS thread per process before numpy)

import numpy as np  # noqa: E402

GEN = os.environ.get("TOURNEY_GEN", "")        # "" = generation 1, "2" = generation 2 (own registry, rows and tables)
RUNS = "results/small_sample_popularity/02_runs"
TABLES = "results/small_sample_popularity/03_tables"
HGRID = ["decay:h=0.25", "decay:h=0.5", "decay:h=1", "decay:h=2", "decay:h=4", "decay:h=8", "decay:h=inf"]


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def eligible_n(active, grid):
    out = []
    nw = len(active)
    for n in grid:
        valid = sum(1 for w in range(8, nw - 1) if all(active[w - 7: w + 2] >= 5 * n))
        if valid >= 10:
            out.append(n)
    return out


def logs_for(reg, groups):
    sp = reg["split"]
    out = []
    for g in groups:
        if g.endswith("_amazon"):
            out += sp[g[:-len("_amazon")]]["amazon"]
        elif g == "all_eligible":
            for k in ("discovery", "holdout_old"):
                out += sp[k]["amazon"] + sp[k]["other"]
        elif g == "holdout_fresh":
            out += sp[g]
        else:
            out += sp[g]["amazon"] + sp[g]["other"]
    return out


def survivors(reg, stage):
    p = f"{TABLES}/tourney{GEN}_survivors.json"
    prev = {"full": "screen", "holdout": "full", "robust": "holdout"}[stage]
    s = json.load(open(p)) if os.path.exists(p) else {}
    if prev not in s:
        sys.exit(f"missing '{prev}' in {p}: run an_tourney.py {prev} first (refusing to treat this as 'all hypotheses killed')")
    ids = s[prev]
    if stage == "robust":   # the 60-window check runs at n=100: only hypotheses confirmed at n=100 can be tested there
        ids = [i for i in ids if 100 in s.get("full_n", {}).get(i, []) and 100 in s.get("holdout_n", {}).get(i, [])]
    return [h for h in reg["hypotheses"] if h["id"] in ids]


def policies_of(hyps):
    out = []
    for h in hyps:
        out += [h["primary"]] + h["secondary"]
    return list(dict.fromkeys(out))


def main():
    import tourney_core as C
    import track_core as T
    reg = json.load(open(f"configs/tourney_registry{GEN}.json"))
    stage = arg("--stage")
    st = reg["stages"][stage]
    smoke = arg("--smoke")
    out = "/tmp/tourney_smoke.jsonl" if smoke else f"{RUNS}/tourney{GEN}.jsonl"
    logs = [smoke] if smoke else logs_for(reg, st["logs"])
    if stage == "screen":
        pol = policies_of(reg["hypotheses"])
    elif stage in ("full", "holdout", "robust"):
        pol = policies_of(survivors(reg, stage))
        if stage in ("full", "robust") and not pol:
            print(f"stage {stage}: no surviving hypotheses; nothing to run")
            return
    else:
        pol = []
    widths = st.get("n_win", [30])
    lim = Q.Limits.from_json("configs/ssp_quiet.json")
    lim.max_workers = int(arg("--workers", 3))
    os.makedirs(RUNS, exist_ok=True)
    gov = Q.Governor(lim, log_path="results/small_sample_popularity/00_safety/governor_log_tourney.jsonl")
    try:
        tasks = []
        for nw in widths:
            for ds in logs:
                D = T.prepare(ds, nw)
                for n in eligible_n(D["active"], st["n"]):
                    plist = [reg["baseline"]] + pol
                    if stage == "cal":
                        plist = HGRID
                    elif stage == "holdout" and n == st.get("hgrid_at_n"):
                        plist = plist + [p for p in HGRID if p not in plist]
                    tasks.append(dict(key=f"{stage}|{ds}|{n}|{nw}", dataset=ds, n=n, n_win=nw, draws=st["draws"], policies=plist,
                                  seed_tag=st.get("seed_tag", "")))
        if not smoke:
            json.dump({t["key"]: t["policies"] for t in tasks}, open(f"{RUNS}/tourney{GEN}_expected_{stage}.json", "w"))
        print(f"[{time.strftime('%H:%M:%S')}] stage {stage}: {len(tasks)} cells over {len(set(t['dataset'] for t in tasks))} logs; "
              f"{len(pol)} candidate rules", flush=True)
        done = Q.run_tasks(C.run_task, tasks, gov, out)
        print(f"[{time.strftime('%H:%M:%S')}] stage {stage}: ran {done} new cells; governor {gov.last}", flush=True)
    finally:
        gov.close()


if __name__ == "__main__":
    main()
