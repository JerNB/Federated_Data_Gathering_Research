"""Analysis of the weekly-cadence programme: verdicts for W1-W7 against configs/weekly_registry.json.

    .venv/bin/python3 -W ignore scripts/ssp/an_weekly.py
Writes results/small_sample_popularity/03_tables/weekly/*.csv and weekly_verdicts.json.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

RUNS = "results/small_sample_popularity/02_runs"
OUT = "results/small_sample_popularity/03_tables/weekly"
STEADY = 8
REG = json.load(open("configs/weekly_registry.json"))
SP = REG["split"]
DAYS = REG["days_grid"]
REFD = REG["ref_days"]


def flat(v):
    return (v["amazon"] + v["other"]) if isinstance(v, dict) else list(v)


GROUPS = {
    "discovery": flat(SP["discovery"]) + SP["new_discovery"],
    "holdout": flat(SP["holdout_old"]) + SP["holdout_fresh"] + SP["new_holdout"],
    "clean": SP["holdout_fresh"] + SP["new_holdout"],
}


def boot(x, seed=7, B=4000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [float("nan")] * 3
    r = np.random.default_rng(seed)
    m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def load():
    cells, cad, bad = {}, {}, 0
    for l in open(f"{RUNS}/weekly.jsonl"):
        r = json.loads(l)
        if r["kind"] == "cadence":
            cad[(r["dataset"], r["n"])] = r
            continue
        steps = np.array(r["steps"])
        st = np.flatnonzero(steps >= STEADY)
        nd = np.array(r["nd100"], float)                    # [draw][arm][step]
        se = np.array(r["selfnd"], float)
        val = np.nanmean(nd[:, :, st], axis=(0, 2))
        key = (r["dataset"], r["n"], r["round_days"], "frac" if "@frac" in r["arms"][0] else "std")
        cells[key] = dict(arms=r["arms"], val=dict(zip(r["arms"], val)), nd=nd, se=se, st=st, active=r.get("weekly_active"))
    return cells, cad


def share_report(d):
    x = list(d.values())
    return dict(n=len(x), share=float(np.mean(x)) if x else None)


def diff_report(d):
    out = dict(all=dict(zip(("mean", "lo", "hi"), boot(list(d.values()))), n=len(d), share_pos=float(np.mean([v > 0 for v in d.values()])) if d else None))
    for g, ls in GROUPS.items():
        x = [d[l] for l in ls if l in d]
        out[g] = dict(zip(("mean", "lo", "hi"), boot(x)), n=len(x), share_pos=float(np.mean([v > 0 for v in x])) if x else None)
    return out


def w(path, rows, head):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{path}", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(head)
        cw.writerows(rows)


def main():
    cells, cad = load()
    V = {}
    logs_info = json.load(open(f"{RUNS}/weekly_logs.json"))
    used = sorted({k[0] for k in cells})
    V["logs_used"] = used
    V["n_cells"] = len(cells)

    def best_days(l, n, rd):
        c = cells.get((l, n, rd, "std"))
        if not c:
            return None
        v = [c["val"][f"decay:days={d}"] for d in DAYS]
        return DAYS[int(np.argmax(v))]

    # ---- best half-life table (all cells)
    rows = []
    for (l, n, rd, sch), c in sorted(cells.items()):
        if sch != "std":
            continue
        rows.append([l, n, rd, best_days(l, n, rd)] + [round(c["val"][f"decay:days={d}"], 5) for d in DAYS] + [round(c["val"]["decay:h=2"], 5)])
    w("a_cells.csv", rows, ["log", "n", "round_days", "best_days"] + [f"nd100_days{d}" for d in DAYS] + ["nd100_h2rounds"])

    # ---- W1
    b1 = {l: best_days(l, 100, 7) for l in used if best_days(l, 100, 7) is not None}
    ge28 = {l: b >= 28 for l, b in b1.items()}
    V["W1"] = dict(share_ge28=share_report(ge28), best_days_counts={str(d): sum(1 for b in b1.values() if b == d) for d in DAYS},
                   median_best_days=float(np.median(list(b1.values()))),
                   by_group={g: share_report({l: ge28[l] for l in ls if l in ge28}) for g, ls in GROUPS.items()})
    # ---- W2
    b2 = {l: (best_days(l, 100, 7), best_days(l, 1000, 7)) for l in used if best_days(l, 100, 7) and best_days(l, 1000, 7)}
    V["W2"] = dict(n=len(b2), share_ge=float(np.mean([a >= b for a, b in b2.values()])) if b2 else None,
                   share_strict=float(np.mean([a > b for a, b in b2.values()])) if b2 else None,
                   median_100=float(np.median([a for a, b in b2.values()])) if b2 else None,
                   median_1000=float(np.median([b for a, b in b2.values()])) if b2 else None)
    # best half-life by round length (days) at n=100 and n=1000
    V["best_days_by_round_length"] = {f"n{n}|{rd}d": dict(median=float(np.median([best_days(l, n, rd) for l in used if best_days(l, n, rd)])),
                                                         n=sum(1 for l in used if best_days(l, n, rd))) for n in (100, 1000) for rd in (7, 14, 28)
                                      if any(best_days(l, n, rd) for l in used)}
    # ---- W3
    V["W3"] = {}
    for n in (100, 1000):
        for rd in (7, 14, 28):
            d = {l: cells[(l, n, rd, "std")]["val"][f"decay:days={REFD}"] - cells[(l, n, rd, "std")]["val"]["decay:h=2"] for l in used if (l, n, rd, "std") in cells}
            if d:
                V["W3"][f"n{n}|{rd}d"] = diff_report(d)
    reg_rows = []
    for rd in (7, 14, 28):
        for n in (100, 1000):
            L = [l for l in used if (l, n, rd, "std") in cells]
            if not L:
                continue
            def regret(arm):
                return float(np.mean([max(cells[(l, n, rd, "std")]["val"][f"decay:days={d}"] for d in DAYS) - cells[(l, n, rd, "std")]["val"][arm] for l in L]))
            for arm in [f"decay:days={d}" for d in DAYS] + ["decay:h=2"]:
                reg_rows.append([rd, n, len(L), arm, round(regret(arm), 5)])
    w("a_regret_by_rule.csv", reg_rows, ["round_days", "n", "logs", "arm", "mean_regret_vs_best_days"])
    # ---- W4
    d4, gain_vs_spread = {}, []
    for l in used:
        c = cells.get((l, 100, 7, "frac"))
        if not c:
            continue
        a = c["arms"]
        base = [x for x in a if x.startswith("decay:")][0]
        d4[l] = c["val"][[x for x in a if x.startswith("ivw:g=1,")][0]] - c["val"][base]
        act = np.array(c["active"], float)
        gain_vs_spread.append((l, d4[l], float(np.percentile(act, 90) / max(np.percentile(act, 10), 1)), float(np.std(np.log(np.maximum(act, 1))))))
    V["W4"] = diff_report(d4)
    V["W4"]["by_g"] = {}
    for g in (0.5, 1, 2):
        dd = {}
        for l in used:
            c = cells.get((l, 100, 7, "frac"))
            if c:
                a = c["arms"]
                dd[l] = c["val"][[x for x in a if x.startswith(f"ivw:g={g},")][0]] - c["val"][[x for x in a if x.startswith("decay:")][0]]
        V["W4"]["by_g"][str(g)] = dict(zip(("mean", "lo", "hi"), boot(list(dd.values()))), share_pos=float(np.mean([v > 0 for v in dd.values()])) if dd else None)
    if len(gain_vs_spread) > 3:
        rr = spearmanr([g[1] for g in gain_vs_spread], [g[3] for g in gain_vs_spread])
        V["W4"]["rho_gain_vs_sd_log_weekly_active"] = [float(rr.statistic), float(rr.pvalue)]
        V["W4"]["median_log_sd_weekly_active"] = float(np.median([g[3] for g in gain_vs_spread]))
        V["W4"]["median_p90_over_p10"] = float(np.median([g[2] for g in gain_vs_spread]))
    w("b_gain.csv", [[l, round(g, 5), round(r, 2), round(s, 3)] for l, g, r, s in gain_vs_spread], ["log", "gain_ivw_g1", "p90_over_p10_weekly_active", "sd_log_weekly_active"])
    # ---- W5, W6
    rho, pool_hit, pool_alarm, rows = {}, 0, 0, []
    per_log6 = {}
    for l in used:
        c = cells.get((l, 100, 7, "std"))
        if not c:
            continue
        ai = c["arms"].index(f"decay:days={REFD}")
        nd = np.nanmean(c["nd"][:, ai, :][:, c["st"]], axis=0)
        se = np.nanmean(c["se"][:, ai, :][:, c["st"]], axis=0)
        ok = np.isfinite(nd) & np.isfinite(se)
        if ok.sum() < 20:
            continue
        nd, se = nd[ok], se[ok]
        rho[l] = float(spearmanr(se, nd).statistic)
        alarm = se <= np.percentile(se, 10)
        low = nd <= np.percentile(nd, 25)
        hit, na = int((alarm & low).sum()), int(alarm.sum())
        pool_hit += hit
        pool_alarm += na
        per_log6[l] = hit / max(na, 1)
        rows.append([l, int(ok.sum()), round(rho[l], 3), hit, na])
    w("e_within_log_selfscore.csv", rows, ["log", "rounds", "spearman_selfscore_vs_nd100", "alarm_hits", "alarm_weeks"])
    V["W5"] = dict(n=len(rho), median_rho=float(np.median(list(rho.values()))), q25=float(np.percentile(list(rho.values()), 25)),
                   q75=float(np.percentile(list(rho.values()), 75)), share_ge_05=float(np.mean([v >= 0.5 for v in rho.values()])),
                   share_positive=float(np.mean([v > 0 for v in rho.values()])),
                   by_group={g: float(np.median([rho[l] for l in ls if l in rho])) if any(l in rho for l in ls) else None for g, ls in GROUPS.items()})
    V["W6"] = dict(pooled_share=pool_hit / max(pool_alarm, 1), alarms=pool_alarm, hits=pool_hit, chance=0.25,
                   median_per_log=float(np.median(list(per_log6.values()))), n=len(per_log6))
    # ---- W7
    d7, rows = {}, []
    for n in (25, 100, 300):
        dd, dfix = {}, {}
        for l in used:
            r = cad.get((l, n))
            if not r:
                continue
            m = {k: float(np.nanmean(v)) for k, v in r["mean"].items()}
            wk = max(v for k, v in m.items() if k.startswith("week|"))
            mo = max(v for k, v in m.items() if k.startswith("month|"))
            dd[l] = wk - mo
            dfix[l] = m["week|56"] - m["month|56"]
            rows.append([l, n, round(wk, 5), round(mo, 5), round(dd[l], 5), round(dfix[l], 5)])
        if dd:
            d7[f"n{n}"] = dict(best=diff_report(dd), fixed56=diff_report(dfix))
    V["W7"] = d7
    w("c_cadence.csv", rows, ["log", "n_weekly", "weekly_best", "monthly_best", "diff_best", "diff_halflife56"])
    json.dump(V, open(f"{OUT}/weekly_verdicts.json", "w"), indent=1)
    print(json.dumps({k: V[k] for k in ("n_cells", "W1", "W2", "W5", "W6")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
