"""Pooled-list pilot (programme E, development stage, window w1). NOT a test and NOT registered: it only decides whether a registered test is worth writing.

Question: if the audit's people ALSO enter the list (pooled with the volunteers), do 100 heavy volunteers plus an audit of 50 reach what 150 random reporters reach?
(Doc 19, section 4, point 9: when the audit only scored lists they were 0.0203 short of 150 random reporters; a pooled list was not tested.)

    nice -n 10 caffeinate -i .venv/bin/python3 -W ignore scripts/ssp/tune_pool.py --run [--grid] [--workers 3] [--smoke LOG] | --plan [--grid]
    .venv/bin/python3 -W ignore scripts/ssp/tune_pool.py --report [--grid]      |      --overlap

Window w1 (the weekly-study windows; the test windows of programmes P and V are not used), 7-day rounds, the logs of stage a at 100 people, the draws of the stored cell.
Rows go to 02_runs/pool_dev.jsonl (gitignored; --smoke writes to /tmp). Cells per log:
  p|pool|LOG|NV|7    NV heavy volunteers (vpool:a=1), audit_n = 50 (it scores the lists AND enters the pooled list, prequentially), the stored days arms (NV random
                     reporters: the std sampler) and the days arms under vpool:a=1,w=B,na=50 for B in -1, -0.5, 0, 0.5, 1;  NV = 100, and with --grid 150, 200, 300
  p|rN|LOG|N|7       N random reporters, the stored days arms: N = 150 (the equal-head-count comparator), and with --grid 200, 300, 400 (the curve behind the break-even)
  (--grid cells need a median of at least 3 NV (3 N) weekly active people, the rule of programmes G and V.)
The audit-scoring-only design (b) is read from the programme V development cells (02_runs/vol_dev.jsonl, A+ arms, audit_n = 50): the same volunteers (paired).
Values are those of tune_weights.rules_of: the best fixed arm in hindsight, the reporter selector and the audit selector (steady steps >= 8, mean over draws).
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import quiet as Q  # noqa: E402  (one BLAS thread per process before numpy)

RUNS = "results/small_sample_popularity/02_runs"
SEED_TAG = "weekly-a"
BETAS = [-1.0, -0.5, 0.0, 0.5, 1.0]
AUDIT = 50                     # the audit people per round: the engine's audit sample (audit_n of the cell) AND the audit part of the pooled list (na of the scheme) must be equal
POOL = "vpool:a=1,w={b},na=" + str(AUDIT)
NV_GRID = [150, 200, 300]       # more heavy volunteers with the same audit of 50: cost ratios 100 / NV of a random reporter at a budget of 150 random reporters
R_GRID = [200, 300, 400]        # random reporters at larger head counts: the curve behind the break-even cost ratio
CURVE = [100, 150, 200, 300, 400]
OUT = "results/small_sample_popularity/03_tables/volunteer"
SUMMARY = []                    # rows of pilot_pool_summary.csv: block, name, logs, mean, lo, hi, logs positive, logs within -0.005


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def pool_schemes():
    return [POOL.format(b=f"{b:g}") for b in BETAS]


def eligible(r, n):
    """The rule of programmes G and V: the log's median weekly active people must be at least 3 n."""
    import numpy as np
    return float(np.median(r["active"])) >= 3 * n


def build(only=None, grid=False):
    tasks = []
    for line in open(f"{RUNS}/weekly.jsonl"):
        r = json.loads(line)
        if r.get("kind") != "deep" or r["round_days"] != 7 or "@frac" in r["arms"][0]:
            continue
        if (r["n"], r["key"].split("|", 1)[0]) != (100, "a") or (only and r["dataset"] != only):
            continue
        arms = list(r["arms"])
        days = [a for a in arms if a.startswith("decay:days=") and "@" not in a]
        base = dict(family="pdev", window="w1", dataset=r["dataset"], n_win=r["n_win"], round_days=7, draws=len(r["nd100"]), seed_tag=SEED_TAG)
        pooled = arms + [f"{a}@{s}" for s in pool_schemes() for a in days]
        tasks.append(dict(base, n=100, arms=pooled, audit_n=AUDIT, key=f"p|pool|{r['dataset']}|100|7"))
        tasks.append(dict(base, n=150, arms=arms, key=f"p|r150|{r['dataset']}|150|7"))
        if grid:
            for nv in NV_GRID:
                if eligible(r, nv):
                    tasks.append(dict(base, n=nv, arms=pooled, audit_n=AUDIT, key=f"p|pool|{r['dataset']}|{nv}|7"))
            for nr in R_GRID:
                if eligible(r, nr):
                    tasks.append(dict(base, n=nr, arms=arms, key=f"p|r{nr}|{r['dataset']}|{nr}|7"))
    tasks.sort(key=lambda t: (t["n"], t["dataset"]))
    return tasks


def run():
    only = arg("--smoke")
    tasks = build(only, "--grid" in sys.argv)
    print(f"[{time.strftime('%H:%M:%S')}] {len(tasks)} cells over {len(set(t['dataset'] for t in tasks))} logs; arms per cell {sorted(set(len(t['arms']) for t in tasks))}", flush=True)
    if "--plan" in sys.argv:
        return
    import pool_core as P
    out = "/tmp/pool_dev_smoke.jsonl" if only else f"{RUNS}/pool_dev.jsonl"
    lim = Q.Limits.from_json("configs/ssp_quiet.json")
    lim.max_workers = int(arg("--workers", 3))
    gov = Q.Governor(lim, log_path="results/small_sample_popularity/00_safety/governor_log_pool.jsonl")
    try:
        done = Q.run_tasks(P.run_task, tasks, gov, out)
        print(f"[{time.strftime('%H:%M:%S')}] ran {done} new cells; governor {gov.last}", flush=True)
    finally:
        gov.close()


def scheme_of(sampler: str, b: float) -> str:
    import tune_weights as TW
    if sampler == "std" and b == 0.0:
        return ""
    if sampler == "pool":
        return POOL.format(b=f"{b:g}")
    return TW.scheme(sampler, b)


def load(path, want=None):
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if want is None or r["key"] in want:
                rows[r["key"]] = r
    return rows


def best_days(r):
    """Hindsight best fixed days arm of a random-people cell (mean over draws and steady steps)."""
    import numpy as np
    nd = np.array(r["nd100"], float)
    st = [i for i, s in enumerate(r["steps"]) if s >= 8]
    idx = [i for i, a in enumerate(r["arms"]) if a.startswith("decay:days=") and "@" not in a]
    return max(float(np.nanmean(nd[:, i, :][:, st])) for i in idx)


def pooled_values(rp):
    """Values of a pooled cell: audit selector and best fixed arm over all B (c_*), and at B = 0 (no weights: c_b0_*)."""
    import tune_weights as TW
    c = TW.rules_of(rp, ["pool"], BETAS, scheme_of)
    return dict(c_aud=c[("pool", "all")]["aud"], c_best=c[("pool", "all")]["best"], c_b0_rep=c[("pool", 0.0)]["rep"], c_b0_aud=c[("pool", 0.0)]["aud"],
                c_b0_best=c[("pool", 0.0)]["best"])


def per_log(pool_rows, vol_rows):
    """{log: record of values} for every log that has the pooled (100), the 150-reporter and the programme-V cell."""
    import numpy as np
    import run_vol_eqcost as EQ
    import tune_weights as TW
    out = {}
    for k, rp in sorted(pool_rows.items()):
        if not k.startswith("p|pool|") or rp["n"] != 100:
            continue
        log = rp["dataset"]
        r150, rv = pool_rows.get(f"p|r150|{log}|150|7"), vol_rows.get(f"v|dev|{log}|100|7")
        if r150 is None or rv is None:
            continue
        b = TW.rules_of(rv, ["A+"], BETAS, scheme_of)
        act = np.array(rp["active"], float)
        steady = act[[i for i, s in enumerate(rp["steps"]) if s >= 8]] if len(act) > max(rp["steps"]) else act[8:]
        out[log] = dict(a100_sel=EQ.std_value(rp), a100_best=best_days(rp), a150_sel=EQ.std_value(r150), a150_best=best_days(r150),
                        b_aud=b[("A+", "all")]["aud"], b_best=b[("A+", "all")]["best"], b_unc=b[("A+", 0.0)]["rep"], active=float(np.median(act)),
                        census=float(np.mean(steady <= 150)), **pooled_values(rp))
    return out


def diffs(pl, rows, label="", block="all"):
    from an_weekly import boot
    for name, x, y in rows:
        d = [r[x] - r[y] for r in pl.values()]
        m, lo, hi = boot(d)
        pos, within = sum(v > 0 for v in d), sum(v >= -0.005 for v in d)
        SUMMARY.append([block, name, len(d), round(float(m), 4), round(float(lo), 4), round(float(hi), 4), pos, within])
        print(f"  {label}{name:22s} {m:+.4f} ({lo:+.4f} to {hi:+.4f})  positive {pos}/{len(d)}  within -0.005: {within}/{len(d)}")


def base_report(pool_rows, vol_rows):
    import numpy as np
    pl = per_log(pool_rows, vol_rows)
    print(f"{len(pl)} logs with the pooled, the 150-reporter and the programme-V cell")
    fields = ["a100_sel", "a150_sel", "b_unc", "b_aud", "c_b0_rep", "c_b0_aud", "c_aud", "a100_best", "a150_best", "b_best", "c_b0_best", "c_best"]
    for f in fields:
        print(f"  {f:10s} mean {np.mean([r[f] for r in pl.values()]):.4f}")
    print(f"  census rounds (steady rounds with at most 150 active people): mean share {np.mean([r['census'] for r in pl.values()]):.3f}, logs with any: {sum(r['census'] > 0 for r in pl.values())}")
    print("paired differences (mean over logs, 95% bootstrap interval over logs, logs positive):")
    diffs(pl, (("c_aud - a150_sel", "c_aud", "a150_sel"), ("c_aud - a100_sel", "c_aud", "a100_sel"), ("c_aud - b_aud", "c_aud", "b_aud"),
               ("c_best - a150_best", "c_best", "a150_best"), ("c_best - a100_best", "c_best", "a100_best"), ("b_aud - a150_sel", "b_aud", "a150_sel"),
               ("c_b0_rep - a150_sel", "c_b0_rep", "a150_sel"), ("c_b0_aud - a150_sel", "c_b0_aud", "a150_sel"), ("c_aud - c_b0_aud", "c_aud", "c_b0_aud")),
          block=f"all {len(pl)} logs")
    rng = np.random.default_rng(7)
    rows_ = [pl[l] for l in sorted(pl)]

    def r_star2(rows, key):                   # random head count R* with the same mean value on the line through 100 and 150 random reporters; r* = (R* - 50) / 100
        a1, a2, p = (float(np.mean([r[k] for r in rows])) for k in ("a100_sel", "a150_sel", key))
        return (100 + (p - a1) / (a2 - a1) * 50 - 50) / 100
    for key, lab in (("b_aud", "audit scores only (programme V design)"), ("c_aud", "pooled")):
        r0 = r_star2(rows_, key)
        bs = np.array([r_star2([rows_[i] for i in rng.integers(0, len(rows_), len(rows_))], key) for _ in range(2000)])
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        SUMMARY.append([f"break-even two-point, all {len(pl)} logs", f"{lab}: r* = (R* - 50) / 100", len(pl), round(r0, 3), round(lo, 3), round(hi, 3), "", ""])
        print(f"  break-even cost ratio, {lab}: r* = {r0:.3f} (95% interval {lo:.3f} to {hi:.3f}; line through 100 and 150 random reporters)")
    amz = {k: v for k, v in pl.items() if k.startswith("amazon")}
    oth = {k: v for k, v in pl.items() if not k.startswith("amazon")}
    med = float(np.median([r["active"] for r in pl.values()]))
    big = {k: v for k, v in pl.items() if v["active"] >= med}
    small = {k: v for k, v in pl.items() if v["active"] < med}
    for lab, sub in (("Amazon", amz), ("other", oth), (f"median weekly active >= {med:.0f}", big), (f"median weekly active < {med:.0f}", small)):
        if len(sub) >= 3:
            print(f"{lab} ({len(sub)} logs)")
            diffs(sub, (("c_aud - a150_sel", "c_aud", "a150_sel"), ("c_aud - b_aud", "c_aud", "b_aud")), "  ", f"{lab} ({len(sub)} logs)")
    write_csv("pilot_pool_per_log.csv", ["log", "group", "median_weekly_active", "census_share"] + fields,
              [[l, "amazon" if l.startswith("amazon") else "other", round(r["active"], 1), round(r["census"], 4)] + [round(r[f], 5) for f in fields] for l, r in sorted(pl.items())])
    return pl


def write_csv(name, head, body):
    import csv
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(head)
        w.writerows(body)
    print("wrote", f"{OUT}/{name}")


def break_even(curve_logs, pool_by_nv, rng_seed=7, nb=2000):
    """curve_logs[log][n] = (selector value, best value) of n random reporters; pool_by_nv[nv][log] = (audit-selected value, best value) of the pooled list.
    Break-even head count R* = the random head count (linear interpolation on the mean curve over logs) with the same mean value; cost ratio r* = (R* - 50) / NV."""
    import numpy as np
    logs = sorted(curve_logs)

    def r_star(sel_idx, nv, rows):
        curve = [float(np.mean([curve_logs[l][n][sel_idx] for l in rows])) for n in CURVE]
        p = float(np.mean([pool_by_nv[nv][l][sel_idx] for l in rows]))
        if p < curve[0]:
            return float("nan"), curve, p
        for i in range(len(CURVE) - 1):
            if curve[i] <= p <= curve[i + 1]:
                rs = CURVE[i] + (p - curve[i]) / (curve[i + 1] - curve[i]) * (CURVE[i + 1] - CURVE[i])
                return (rs - 50) / nv, curve, p
        return float("nan"), curve, p

    rng = np.random.default_rng(rng_seed)
    out = {}
    for nv in sorted(pool_by_nv):
        for sel_idx, lab in ((0, "feasible"), (1, "hindsight")):
            r0, curve, p = r_star(sel_idx, nv, logs)
            bs = []
            for _ in range(nb):
                rows = [logs[i] for i in rng.integers(0, len(logs), len(logs))]
                bs.append(r_star(sel_idx, nv, rows)[0])
            bs = np.array(bs)
            ok = bs[np.isfinite(bs)]
            out[(nv, lab)] = (r0, float(np.percentile(ok, 2.5)) if len(ok) else float("nan"), float(np.percentile(ok, 97.5)) if len(ok) else float("nan"), len(ok) / nb, p, curve)
    return out


def grid_report(pool_rows):
    import numpy as np
    import run_vol_eqcost as EQ
    from an_weekly import boot
    logs = sorted({r["dataset"] for r in pool_rows.values()})
    curve_logs, pool_by_nv, sizes = {}, {nv: {} for nv in [100] + NV_GRID}, {}
    worst = 0.0
    for log in logs:
        need = [f"p|pool|{log}|{nv}|7" for nv in [100] + NV_GRID] + [f"p|r150|{log}|150|7"] + [f"p|r{n}|{log}|{n}|7" for n in R_GRID]
        if not all(k in pool_rows for k in need):
            continue
        cl = {100: (EQ.std_value(pool_rows[f"p|pool|{log}|100|7"]), best_days(pool_rows[f"p|pool|{log}|100|7"])),
              150: (EQ.std_value(pool_rows[f"p|r150|{log}|150|7"]), best_days(pool_rows[f"p|r150|{log}|150|7"]))}
        for n in R_GRID:
            cl[n] = (EQ.std_value(pool_rows[f"p|r{n}|{log}|{n}|7"]), best_days(pool_rows[f"p|r{n}|{log}|{n}|7"]))
        for nv in NV_GRID:                               # the std arms of a pooled cell are NV random reporters: the same draw as the random cell of that size
            same = cl.get(nv) or cl[150]
            worst = max(worst, abs(EQ.std_value(pool_rows[f"p|pool|{log}|{nv}|7"]) - same[0]))
        curve_logs[log] = cl
        for nv in [100] + NV_GRID:
            v = pooled_values(pool_rows[f"p|pool|{log}|{nv}|7"])
            pool_by_nv[nv][log] = (v["c_aud"], v["c_best"])
    print(f"{len(curve_logs)} logs have every grid cell (median weekly active >= 1,200); largest |std arm of a pooled cell - the random cell of that size| = {worst:.2e}")
    if not curve_logs:
        return
    gridrows = [[l, nv, round(pool_by_nv[nv][l][0], 5), round(pool_by_nv[nv][l][1], 5)] + [round(curve_logs[l][n][0], 5) for n in CURVE] + [round(curve_logs[l][n][1], 5) for n in CURVE]
                for l in sorted(curve_logs) for nv in [100] + NV_GRID]
    write_csv("pilot_pool_grid.csv", ["log", "nv", "pooled_audit_selected", "pooled_best"] + [f"random{n}_selector" for n in CURVE] + [f"random{n}_best" for n in CURVE], gridrows)
    for nv in [100] + NV_GRID:
        d = [pool_by_nv[nv][l][0] - curve_logs[l][150][0] for l in curve_logs]
        m, lo, hi = boot(d)
        pos = sum(v > 0 for v in d)
        SUMMARY.append([f"grid {len(curve_logs)} logs", f"pooled NV={nv} (cost ratio {100 / nv:.2f}) - 150 random", len(d), round(float(m), 4), round(float(lo), 4), round(float(hi), 4), pos, sum(v >= -0.005 for v in d)])
        print(f"  NV = {nv:3d} pooled (audit-selected) minus 150 random reporters: {m:+.4f} ({lo:+.4f} to {hi:+.4f}), positive {pos}/{len(d)}; cost ratio at equal cost {100 / nv:.2f}")
    print("mean random-reporter curve (selector value) over these logs:", {n: round(float(np.mean([curve_logs[l][n][0] for l in curve_logs])), 4) for n in CURVE})
    for (nv, lab), (r0, lo, hi, share, p, curve) in sorted(break_even(curve_logs, pool_by_nv).items()):
        SUMMARY.append([f"break-even {lab}", f"NV={nv}: pooled value {p:.4f}, r* (R* - 50) / NV", len(curve_logs), round(r0, 3), round(lo, 3), round(hi, 3), "", ""])
        print(f"  NV = {nv:3d} {lab:9s} pooled value {p:.4f}: break-even cost ratio r* = {r0:.3f} (95% interval {lo:.3f} to {hi:.3f}; {share:.0%} of resamples bracketed)")


def overlap():
    """Mean number of people in BOTH the volunteer draw (vpps:a=1) and the audit draw of the same round (they are drawn independently), steady rounds, all draws."""
    import numpy as np
    import deep_core as C
    import pool_core  # noqa: F401  (installs the patches)
    import weekly_core as W
    rows = [json.loads(l) for l in open(f"{RUNS}/weekly.jsonl")]
    rows = [r for r in rows if r.get("kind") == "deep" and r["round_days"] == 7 and "@frac" not in r["arms"][0] and (r["n"], r["key"].split("|", 1)[0]) == (100, "a")]
    res = []
    for r in sorted(rows, key=lambda r: r["dataset"]):
        D = W.prepare_weekly(r["dataset"], 7)                     # the densest 156-week span, as in the cells themselves
        ov, cnt = [], []
        for dr in range(len(r["nd100"])):
            seed = (r["dataset"], 100, dr, r["n_win"], SEED_TAG)
            vrng = np.random.default_rng(C.seed_of(*seed, "vol", "vpps:a=1"))
            arng = np.random.default_rng(C.seed_of(r["dataset"], 50, dr, r["n_win"], SEED_TAG, "audit", "sel"))
            for k in range(len(D["off"]) - 2):                    # rounds 0..n_win-2: the cells never draw the last round (its truth would be round n_win)
                a, b = D["off"][k], D["off"][k + 1]
                uk, rows_ = np.unique(D["user"][a:b], return_counts=True)
                if len(uk) <= 100:
                    sv = uk
                else:
                    keys = vrng.random(len(uk)) ** (1.0 / rows_.astype(np.float64))
                    sv = uk[np.argpartition(-keys, 99)[:100]]
                uq = D["users_in"][k]
                sa = uq if len(uq) <= 50 else arng.choice(uq, 50, replace=False)
                if k >= 8:
                    ov.append(len(np.intersect1d(sv, sa)))
                    cnt.append(len(uq))
        res.append((r["dataset"], float(np.mean(ov)), float(np.median(cnt))))
        print(f"{r['dataset']:45s} overlap {res[-1][1]:5.2f} people per round (median active {res[-1][2]:.0f})", flush=True)
    print(f"mean over {len(res)} logs: {np.mean([x[1] for x in res]):.2f} people in both draws; largest {max(x[1] for x in res):.2f}")
    write_csv("pilot_pool_overlap.csv", ["log", "mean_people_in_both_draws_per_steady_round", "median_weekly_active"], [[a, round(b, 3), round(c, 1)] for a, b, c in res])


def report():
    pool_rows = load(f"{RUNS}/pool_dev.jsonl")
    vol_rows = load(f"{RUNS}/vol_dev.jsonl", {f"v|dev|{r['dataset']}|100|7" for r in pool_rows.values()})
    base_report(pool_rows, vol_rows)
    if "--grid" in sys.argv:
        print()
        grid_report(pool_rows)
    write_csv("pilot_pool_summary.csv", ["block", "contrast", "logs", "mean", "ci_lo", "ci_hi", "logs_positive", "logs_within_0.005"], SUMMARY)


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif "--overlap" in sys.argv:
        overlap()
    elif "--run" in sys.argv or "--plan" in sys.argv:
        run()
    else:
        sys.exit(__doc__)
