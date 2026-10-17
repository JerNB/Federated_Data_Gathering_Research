"""Programme V, development stage: weight reports by rows^(-beta) under random and volunteer sampling on the DEVELOPMENT windows (w1 = the weekly-study
windows, 7-day rounds, 100 people (stage a, 27 logs) and 300 people (stage d, 24 logs)). Nothing here is a registered result.

    nice -n 10 caffeinate -i .venv/bin/python3 -W ignore scripts/ssp/tune_weights.py --run [--workers 3] [--smoke LOG]
    .venv/bin/python3 -W ignore scripts/ssp/tune_weights.py --report

Cells are built from the STORED weekly cells (02_runs/weekly.jsonl): same logs, same std arms, same seed tag, so the std arms replicate the stored cell
(an audit sample of max(10, n/2) random people is drawn in every cell, as in doc 16). Rows go to 02_runs/vol_dev.jsonl (gitignored); the report goes to
03_tables/volunteer/v_dev_grid.csv.
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
SEED_TAG = {"a": "weekly-a", "d": "weekly-d"}
BETAS = [-1.0, -0.5, 0.0, 0.5, 1.0]
SAMPLERS = {"std": "vstd:w={b}", "A+": "vpps:a=1,w={b}", "A-": "vpps:a=-1,w={b}"}
WANTED = {(100, "a"), (300, "d")}


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def scheme(sampler: str, b: float):
    """Scheme string of (sampler, beta); None for the stored std arms (beta = 0 under std)."""
    if sampler == "std" and b == 0.0:
        return ""
    return SAMPLERS[sampler].format(b=f"{b:g}")


def new_schemes():
    return [scheme(s, b) for s in SAMPLERS for b in BETAS if scheme(s, b)]


def build(only=None):
    tasks = []
    for line in open(f"{RUNS}/weekly.jsonl"):
        r = json.loads(line)
        if r.get("kind") != "deep" or r["round_days"] != 7 or "@frac" in r["arms"][0]:
            continue
        stage = r["key"].split("|", 1)[0]
        if (r["n"], stage) not in WANTED or (only and r["dataset"] != only):
            continue
        arms = list(r["arms"])
        days = [a for a in arms if a.startswith("decay:days=") and "@" not in a]
        tasks.append(dict(family="vdev", window="w1", dataset=r["dataset"], n=r["n"], n_win=r["n_win"], round_days=7, draws=len(r["nd100"]),
                          seed_tag=SEED_TAG[stage], arms=arms + [f"{a}@{s}" for s in new_schemes() for a in days], audit_n=max(10, r["n"] // 2),
                          key=f"v|dev|{r['dataset']}|{r['n']}|7"))
    tasks.sort(key=lambda t: (t["n"], t["dataset"]))
    return tasks


def run():
    only = arg("--smoke")
    tasks = build(only)
    print(f"[{time.strftime('%H:%M:%S')}] {len(tasks)} dev cells; arms per cell {sorted(set(len(t['arms']) for t in tasks))}", flush=True)
    import weight_core as V
    out = "/tmp/vol_dev_smoke.jsonl" if only else f"{RUNS}/vol_dev.jsonl"
    lim = Q.Limits.from_json("configs/ssp_quiet.json")
    lim.max_workers = int(arg("--workers", 3))
    gov = Q.Governor(lim, log_path="results/small_sample_popularity/00_safety/governor_log_volunteer.jsonl")
    try:
        done = Q.run_tasks(V.run_task, tasks, gov, out)
        print(f"[{time.strftime('%H:%M:%S')}] ran {done} new cells; governor {gov.last}", flush=True)
    finally:
        gov.close()


def rules_of(r: dict, samplers=None, betas=None, scheme_of=None) -> dict:
    """Per (sampler, beta) and across beta: mean values (steady steps, draws) of the best fixed arm, the reporter selector and the audit selector.
    The reporter selector is only meaningful within one beta (the self-score target is that scheme's own weighted sample); across beta only the audit
    self-score (an unbiased random sample) is comparable, so 'rep' of the 'all' record is a diagnostic."""
    import numpy as np
    import explore_selfsel_weekly as E
    samplers = list(SAMPLERS) if samplers is None else samplers
    betas = BETAS if betas is None else betas
    scheme_of = scheme if scheme_of is None else scheme_of
    nd, se, sa = np.array(r["nd100"], float), np.array(r["selfnd"], float), np.array(r["selfaud"], float)
    steps = r["steps"]
    st = [i for i, s in enumerate(steps) if s >= 8]
    out = {}
    for sampler in samplers:
        parts = []
        for b in betas:
            suffix = scheme_of(sampler, b)
            idx = [i for i, a in enumerate(r["arms"]) if a.partition("@")[2] == suffix and a.startswith("decay:days=")]
            ds = np.array([int(r["arms"][i].partition("@")[0].split("=")[1]) for i in idx])
            order = np.argsort(ds)
            idx, ds = [idx[j] for j in order], ds[order]
            c = dict(nd=nd[:, idx, :], se=se[:, idx, :], sa=sa[:, idx, :], ds=ds)
            fb = E.nearest(ds, 14)
            rec = dict(best=max(float(np.nanmean(c["nd"][:, a, :][:, st])) for a in range(len(idx))),
                       rep=E.chosen_value(c["nd"], E.choose(c["se"], st, E.HL, fb), st),
                       aud=E.chosen_value(c["nd"], E.choose(c["sa"], st, E.HL, fb), st))
            out[(sampler, b)] = rec
            parts.append((b, c, fb))
        nd_all = np.concatenate([p[1]["nd"] for p in parts], axis=1)
        sa_all = np.concatenate([p[1]["sa"] for p in parts], axis=1)
        se_all = np.concatenate([p[1]["se"] for p in parts], axis=1)
        offs = np.cumsum([0] + [p[1]["nd"].shape[1] for p in parts])
        b0 = [i for i, p in enumerate(parts) if p[0] == 0.0][0]
        fb_all = int(offs[b0] + parts[b0][2])
        out[(sampler, "all")] = dict(best=max(float(np.nanmean(nd_all[:, a, :][:, st])) for a in range(nd_all.shape[1])),
                                     rep=E.chosen_value(nd_all, E.choose(se_all, st, E.HL, fb_all), st),
                                     aud=E.chosen_value(nd_all, E.choose(sa_all, st, E.HL, fb_all), st))
    return out


def report():
    import numpy as np
    import explore_selfsel_weekly as E
    rows = [json.loads(line) for line in open(f"{RUNS}/vol_dev.jsonl")]
    per = {}                                          # per[(n, sampler, beta)][log] = rec
    for r in rows:
        for key, rec in rules_of(r).items():
            per.setdefault((r["n"],) + key, {})[r["dataset"]] = rec
    std_ref = {(n, L): per[(n, "std", 0.0)][L]["rep"] for (n, s, b) in per if s == "std" and b == 0.0 for L in per[(n, s, b)]}
    head = ["n", "sampler", "beta", "logs", "best_fixed", "reporter_selector", "audit_selector", "std_reporter_selector_same_cells",
            "reporter_selector_minus_std", "audit_selector_minus_std", "best_fixed_minus_best_fixed_beta0"]
    body = []
    for key in sorted(per, key=lambda k: (k[0], list(SAMPLERS).index(k[1]), 99 if k[2] == "all" else k[2])):
        n, sampler, b = key
        ls = sorted(per[key])
        g = lambda f: np.array([per[key][L][f] for L in ls])
        ref = np.array([std_ref[(n, L)] for L in ls])
        b0 = np.array([per[(n, sampler, 0.0)][L]["best"] for L in ls])
        body.append([n, sampler, b, len(ls), round(float(g("best").mean()), 5), round(float(g("rep").mean()), 5), round(float(g("aud").mean()), 5),
                     round(float(ref.mean()), 5), round(float((g("rep") - ref).mean()), 5), round(float((g("aud") - ref).mean()), 5), round(float((g("best") - b0).mean()), 5)])
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/v_dev_grid.csv", "w", newline="") as f:
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
