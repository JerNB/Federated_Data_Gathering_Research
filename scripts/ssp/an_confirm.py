"""Verdicts for the selector confirmation (stage K); mechanical, thresholds read from configs/selfsel_confirm_registry.json.

    .venv/bin/python3 -W ignore scripts/ssp/an_confirm.py [--selftest]

Reads 02_runs/confirm.jsonl (fresh windows), 02_runs/confirm_logs.json (the planned cells) and 02_runs/weekly.jsonl (the weekly study's
window-1 cells). Writes results/small_sample_popularity/03_tables/confirm/: k_cells.csv, k_logs.csv, k_summary.csv, k_paired.csv,
k_tails.csv, k_verdicts.json (verdicts plus the sha256 of the registry and of the scripts, so an edit after the run leaves a trace).
It aborts when the loaded cells are not exactly the planned cells or when confirm.jsonl.errors is not empty.

Comparator tables. "pooled" / "same" are fitted leave-one-log-out on the window-2 cells (the exploration's contrast, same period as the
cell they are scored on); "pooled_w1" / "same_w1" are fitted on the window-1 cells of the other logs (a deployment contrast: tables
fitted on the past). Rule definitions (selector, tables, regret) are imported from explore_selfsel_weekly.py, not re-implemented.

--selftest feeds the window-1 cells in as if they were the new windows (own-history and window-1 tables skipped, nothing written),
checks that two / pooled / same / selector and the paired rows reproduce i_selfsel_rules.csv / i_selfsel_paired.csv to 1e-5, and prints
the verdict numbers the window-1 data give for K1-K4, K8-K10 and K13 (anchors for the registry thresholds).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import explore_selfsel_weekly as E
from an_weekly import OUT as EXPLORE_OUT
from an_weekly import RUNS, STEADY, boot, load

REGISTRY = "configs/selfsel_confirm_registry.json"
REG = json.load(open(REGISTRY))
CR = REG["criteria"]
OUT = "results/small_sample_popularity/03_tables/confirm"
RULES = ["two", "pooled", "same", "pooled_w1", "same_w1", "own", "selector"]
WARM = (8, 20)
NAN = float("nan")
HASHED = [REGISTRY, "scripts/ssp/confirm_core.py", "scripts/ssp/run_confirm.py", "scripts/ssp/an_confirm.py", "scripts/ssp/explore_selfsel_weekly.py",
          "scripts/ssp/an_weekly.py"]


def load_confirm(path: str):
    cells, kind_of, rows = {}, {}, 0
    for line in open(path):
        r = json.loads(line)
        rows += 1
        st = np.flatnonzero(np.array(r["steps"]) >= STEADY)
        nd = np.array(r["nd100"], float)
        se = np.array(r["selfnd"], float)
        val = np.nanmean(nd[:, :, st], axis=(0, 2))
        key = (r["log"], r["n"], r["round_days"], "std")
        if key in cells:
            sys.exit(f"duplicate row for {key}")
        cells[key] = dict(arms=r["arms"], val=dict(zip(r["arms"], val)), nd=nd, se=se, st=st)
        kind_of[r["log"]] = "w2" if r["dataset"].endswith("@w2") else "whole"
    return cells, kind_of


def planned_cells() -> set:
    info = json.load(open(f"{RUNS}/confirm_logs.json"))
    out = set()
    for log, v in info.items():
        if isinstance(v, str):
            continue
        for rd in REG["stage"]["round_days"][v["kind"]]:
            for n in v["eligible_n"]:
                out.add((log, n, rd, "std"))
    return out


def check_complete(cells2: dict) -> None:
    errs = f"{RUNS}/confirm.jsonl.errors"
    if os.path.exists(errs) and os.path.getsize(errs) > 0:
        sys.exit(f"{errs} is not empty: fix the failed cells before analysing")
    want = planned_cells()
    missing, extra = sorted(want - set(cells2)), sorted(set(cells2) - want)
    if missing or extra:
        sys.exit(f"cell set differs from the plan: {len(missing)} missing {missing[:3]}, {len(extra)} unplanned {extra[:3]}")


def evaluate(cells2: dict, cells1: dict, kind_of: dict, with_own: bool = True, with_w1: bool = True) -> dict:
    """{(log, n, round_days): regrets of every rule} for each cell of cells2.

    pooled/same: tables fitted on the other window-2 logs (leave one out); pooled_w1/same_w1: fitted on the window-1 cells of the other logs.
    """
    per = {}
    for (L, n, rd, _), c in sorted(cells2.items()):
        b = E.best_value(c)
        ds, nd, se = E.prep(c)
        st = c["st"]
        fb = E.nearest(ds, 2 * rd)
        rec = dict(group=E.stratum(L), kind=kind_of[L], two=b - E.snap_value(c, 2 * rd), pooled=NAN, same=NAN, pooled_w1=NAN, same_w1=NAN,
                   own=NAN, r_pool=NAN, r_same=NAN, own_days=NAN, warm_selector=NAN, warm_pooled=NAN)
        others = sorted({l for (l, n1, rd1, s) in cells2 if n1 == n and rd1 == rd and l != L and kind_of[l] == "w2"})
        same = [l for l in others if E.stratum(l) == E.stratum(L)]
        if others:
            rec["r_pool"] = E.fit_rounds(cells2, others, n, rd)
            rec["pooled"] = b - E.snap_value(c, rd * rec["r_pool"])
        if len(same) >= E.MIN_GROUP:
            rec["r_same"] = E.fit_rounds(cells2, same, n, rd)
            rec["same"] = b - E.snap_value(c, rd * rec["r_same"])
        if with_w1:
            others1 = sorted({l for (l, n1, rd1, s) in cells1 if n1 == n and rd1 == rd and s == "std" and l != L})
            same1 = [l for l in others1 if E.stratum(l) == E.stratum(L)]
            if others1:
                rec["pooled_w1"] = b - E.snap_value(c, rd * E.fit_rounds(cells1, others1, n, rd))
            if len(same1) >= E.MIN_GROUP:
                rec["same_w1"] = b - E.snap_value(c, rd * E.fit_rounds(cells1, same1, n, rd))
        c1 = cells1.get((L, n, rd, "std")) if with_own else None
        if c1 is not None:
            a_best = max((a for a in c1["val"] if a.startswith("decay:days=")), key=lambda a: (c1["val"][a], -int(a.split("=")[1])))
            rec["own_days"] = int(a_best.split("=")[1])
            rec["own"] = b - E.snap_value(c, rec["own_days"])
        ch = E.choose(se, st, E.HL, fb)
        rec["selector"] = b - E.chosen_value(nd, ch, st)
        ts = [t for t in st if WARM[0] <= t < WARM[1]]
        if rd == 7 and len(ts) >= 5 and np.isfinite(rec["r_pool"]):
            ref = E.fixed_value(nd, int(np.argmax([E.fixed_value(nd, a, st) for a in range(len(ds))])), ts)
            rec["warm_selector"] = ref - E.chosen_value(nd, ch, ts)
            rec["warm_pooled"] = ref - E.fixed_value(nd, E.nearest(ds, rd * rec["r_pool"]), ts)
        per[(L, n, rd)] = rec
    return per


def pdiff(per: dict, keys: list, a: str, b: str):
    """({log: mean over the log's cells of (rule a - rule b)}, number of cells dropped because a rule is missing)."""
    d, dropped = {}, 0
    for k in keys:
        x, y = per[k][a], per[k][b]
        if np.isfinite(x) and np.isfinite(y):
            d.setdefault(k[0], []).append(x - y)
        else:
            dropped += 1
    return {l: float(np.mean(v)) for l, v in d.items()}, dropped


def paired_row(per, keys, a, b):
    d, dropped = pdiff(per, keys, a, b)
    m, lo, hi = boot(list(d.values()))
    return d, dict(mean=m, lo=lo, hi=hi, share_negative=float(np.mean([v < 0 for v in d.values()])) if d else NAN, logs=len(d), dropped_cells=dropped)


def vals(per, keys, rule):
    v = np.array([per[k][rule] for k in keys], float)
    return v[np.isfinite(v)]


def best_simple(rec) -> float:
    """Regret of the better of the two simple rules (2 rounds, same-group table); the table is skipped when it is missing."""
    return min(x for x in (rec["two"], rec["same"]) if np.isfinite(x))


def verdicts(per: dict) -> dict:
    A7 = [k for k in per if k[2] == 7 and per[k]["kind"] == "w2"]
    A28 = [k for k in per if k[2] == 28 and per[k]["kind"] == "w2"]
    C7 = [k for k in per if k[2] == 7 and per[k]["kind"] == "whole"]
    V = {}

    c = CR["K1"]
    d, row = paired_row(per, A7, "selector", "same")
    row["by_group"] = {}
    for g in ("amazon", "other"):
        dg = [v for l, v in d.items() if E.stratum(l) == g]
        if dg:
            m, lo, hi = boot(dg)
            row["by_group"][g] = dict(mean=m, lo=lo, hi=hi, logs=len(dg))
    row["met"] = bool(d and row["mean"] <= c["max_mean_diff"] and row["hi"] < c["ci_upper_below"]
                      and (not c["both_groups_negative"] or (len(row["by_group"]) == 2 and all(v["mean"] < 0 for v in row["by_group"].values()))))
    V["K1"] = row

    d, row = paired_row(per, A7, "selector", "pooled")
    row["met"] = bool(d and row["hi"] < CR["K2"]["ci_upper_below"])
    V["K2"] = row

    c = CR["K3"]
    v = vals(per, A7, "selector")
    V["K3"] = dict(cells=len(v), share_over_0_01=float(np.mean(v > 0.01)) if len(v) else NAN, largest=float(v.max()) if len(v) else NAN,
                   mean=float(v.mean()) if len(v) else NAN,
                   met=bool(len(v) and np.mean(v > 0.01) <= c["max_share_over_0.01"] and v.max() <= c["max_cell_regret"]))

    c = CR["K4"]
    dd = np.array([per[k]["selector"] - best_simple(per[k]) for k in A7])
    vs_two = np.array([per[k]["selector"] - per[k]["two"] for k in A7])
    V["K4"] = dict(cells=len(dd), share_over=float(np.mean(dd > c["margin"])) if len(dd) else NAN, largest=float(dd.max()) if len(dd) else NAN,
                   cells_over=[list(k) for k, x in zip(A7, dd) if x > c["margin"]],
                   cells_without_table=int(sum(not np.isfinite(per[k]["same"]) for k in A7)),
                   cells_worse_than_two_by_margin=int(np.sum(vs_two > c["margin"])), largest_vs_two=float(vs_two.max()) if len(vs_two) else NAN,
                   met=bool(len(dd) and np.mean(dd > c["margin"]) <= c["max_share"]))

    d, row = paired_row(per, A7, "selector", "same_w1")
    row["met"] = bool(d and row["hi"] < CR["K5"]["ci_upper_below"])
    V["K5"] = row

    d, row = paired_row(per, A7, "selector", "pooled_w1")
    row["met"] = bool(d and row["hi"] < CR["K6"]["ci_upper_below"])
    V["K6"] = row

    c = CR["K7"]
    ok = [k for k in A7 if np.isfinite(per[k]["pooled_w1"])]
    pw, tw = np.array([per[k]["pooled_w1"] for k in ok]), np.array([per[k]["two"] for k in ok])
    V["K7"] = dict(cells=len(ok), pooled_w1=float(pw.mean()) if len(ok) else NAN, two=float(tw.mean()) if len(ok) else NAN,
                   ratio=float(pw.mean() / tw.mean()) if len(ok) else NAN,
                   met=bool(len(ok) and pw.mean() <= c["max_regret"] and pw.mean() <= c["max_ratio"] * tw.mean()))

    c = CR["K8"]
    ks = [k for k in A7 if per[k]["group"] == "other" and k[1] >= c["min_n"] and np.isfinite(per[k]["pooled"])]
    d, _ = pdiff(per, ks, "pooled", "two")
    p, t = vals(per, ks, "pooled"), vals(per, ks, "two")
    V["K8"] = dict(cells=len(ks), logs=len(d), pooled=float(p.mean()) if len(ks) else NAN, two=float(t.mean()) if len(ks) else NAN,
                   ratio=float(p.mean() / t.mean()) if len(ks) else NAN, gap=float(p.mean() - t.mean()) if len(ks) else NAN,
                   share_logs_pooled_worse=float(np.mean([v > 0 for v in d.values()])) if d else NAN,
                   per_log_mean_pooled_minus_two={l: round(v, 5) for l, v in d.items()},
                   met=bool(len(ks) and p.mean() >= c["min_ratio"] * t.mean() and p.mean() - t.mean() >= c["min_gap"]))

    c = CR["K9"]
    d, row = paired_row(per, [k for k in A28 if k[1] >= c["min_n"]], "selector", "same")
    row["met"] = bool(d and row["hi"] < c["ci_upper_below"])
    V["K9"] = row

    c = CR["K10"]
    d, row = paired_row(per, [k for k in A28 if k[1] == c["n"]], "selector", "same")
    row["share_positive"] = float(np.mean([v > 0 for v in d.values()])) if d else NAN
    row["met"] = bool(d and row["mean"] > c["min_mean_diff"] and row["share_positive"] >= c["min_share_logs"])
    V["K10"] = row

    c = CR["K11"]
    ok = [k for k in C7 if np.isfinite(per[k]["same"])]
    le = [per[k]["selector"] <= per[k]["same"] for k in ok]
    d, _ = pdiff(per, ok, "selector", "same")
    harm = [per[k]["selector"] - per[k]["two"] > c["margin"] for k in C7]
    V["K11"] = dict(cells=len(C7), cells_with_table=len(ok), share_selector_le_same=float(np.mean(le)) if ok else NAN,
                    per_log_mean_selector_minus_same={l: round(v, 5) for l, v in d.items()}, cells_worse_than_two=int(sum(harm)),
                    met=bool(ok and np.mean(le) >= c["min_share"] and len(d) == len({k[0] for k in C7}) and all(v < 0 for v in d.values()) and not any(harm)))

    d, row = paired_row(per, A7, "selector", "own")
    row["met"] = bool(d and row["hi"] < CR["K12"]["ci_upper_below"])
    V["K12"] = row

    ok = [k for k in A7 if np.isfinite(per[k]["warm_selector"]) and np.isfinite(per[k]["warm_pooled"])]
    ws, wp = np.array([per[k]["warm_selector"] for k in ok]), np.array([per[k]["warm_pooled"] for k in ok])
    V["K13"] = dict(cells=len(ok), warm_selector=float(ws.mean()) if len(ok) else NAN, warm_pooled=float(wp.mean()) if len(ok) else NAN,
                    met=bool(len(ok) and ws.mean() < wp.mean()))

    k1 = V["K1"]
    if not np.isfinite(k1["hi"]) or k1["hi"] >= 0:
        V["decision"] = "NOT CONFIRMED"
    elif all(V[k]["met"] for k in ("K1", "K2", "K3", "K4")):
        V["decision"] = "CONFIRMED"
    else:
        V["decision"] = "WEAKENED"
    return V


def write_csv(name: str, head: list, rows: list) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}", "w", newline="") as f:
        cw = csv.writer(f, lineterminator="\n")
        cw.writerow(head)
        cw.writerows(rows)


def r6(x):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 6)


def tables(per: dict) -> None:
    write_csv("k_cells.csv", ["log", "kind", "group", "n", "round_days"] + RULES + ["r_pool", "r_same", "own_days", "warm_selector", "warm_pooled"],
              [[k[0], v["kind"], v["group"], k[1], k[2]] + [r6(v[r]) for r in RULES] + [r6(v["r_pool"]), r6(v["r_same"]), r6(v["own_days"]),
                                                                                       r6(v["warm_selector"]), r6(v["warm_pooled"])]
               for k, v in sorted(per.items())])
    rows = []
    for rd in sorted({k[2] for k in per}):
        for kind in ("w2", "whole"):
            for grp in ("amazon", "other", "all"):
                for n in ["all"] + E.NS:
                    keys = [k for k in per if k[2] == rd and per[k]["kind"] == kind and (grp == "all" or per[k]["group"] == grp) and (n == "all" or k[1] == n)]
                    for rule in RULES if keys else []:
                        v = vals(per, keys, rule)
                        rows.append([rd, kind, grp, n, rule, len(v), r6(v.mean()) if len(v) else ""])
    write_csv("k_summary.csv", ["round_days", "window", "group", "n", "rule", "cells", "mean_regret"], rows)
    rows = []
    for rd in sorted({k[2] for k in per}):
        for kind in ("w2", "whole"):
            keys = [k for k in per if k[2] == rd and per[k]["kind"] == kind]
            for rule in RULES if keys else []:
                v = vals(per, keys, rule)
                if len(v):
                    rows.append([rd, kind, rule, len(v), r6(v.mean()), r6(np.median(v)), r6(np.percentile(v, 90)), r6(v.max()),
                                 r6(np.mean(v > 0.01)), r6(np.mean(v > 0.02)), r6(np.mean(v > 0.03))])
    write_csv("k_tails.csv", ["round_days", "window", "rule", "cells", "mean", "median", "p90", "max", "share_over_0.01", "share_over_0.02", "share_over_0.03"], rows)
    rows = []
    scopes = (("w2 all", lambda k: per[k]["kind"] == "w2"), ("w2 amazon", lambda k: per[k]["kind"] == "w2" and per[k]["group"] == "amazon"),
              ("w2 other", lambda k: per[k]["kind"] == "w2" and per[k]["group"] == "other"),
              ("w2 n>=100", lambda k: per[k]["kind"] == "w2" and k[1] >= 100), ("w2 n=30", lambda k: per[k]["kind"] == "w2" and k[1] == 30))
    for rd in sorted({k[2] for k in per}):
        for scope, f in scopes:
            keys = [k for k in per if k[2] == rd and f(k)]
            for other in ("two", "pooled", "same", "pooled_w1", "same_w1", "own"):
                d, row = paired_row(per, keys, "selector", other)
                if d:
                    rows.append([rd, scope, f"selector minus {other}", r6(row["mean"]), r6(row["lo"]), r6(row["hi"]), r6(row["share_negative"]),
                                 row["logs"], row["dropped_cells"]])
    write_csv("k_paired.csv", ["round_days", "scope", "comparison", "mean_diff", "ci_lo", "ci_hi", "share_logs_selector_better", "logs", "dropped_cells"], rows)
    rows = []
    for rd in sorted({k[2] for k in per}):
        for L in sorted({k[0] for k in per if k[2] == rd}):
            keys = [k for k in per if k[2] == rd and k[0] == L]
            rows.append([L, per[keys[0]]["group"], rd, len(keys)] + [r6(np.mean(vals(per, keys, r))) if len(vals(per, keys, r)) else "" for r in RULES])
    write_csv("k_logs.csv", ["log", "group", "round_days", "cells"] + RULES, rows)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    return o


def show(V, ids) -> None:
    for k in ids:
        print(k, "MET" if V[k]["met"] else "NOT MET", {a: (round(b, 4) if isinstance(b, float) else b) for a, b in V[k].items()
                                                         if a not in ("met", "cells_over", "per_log_mean_pooled_minus_two", "per_log_mean_selector_minus_same")})


def selftest() -> None:
    cells1, _ = load()
    cells2 = {k: v for k, v in cells1.items() if k[3] == "std" and k[2] in E.CADENCES and k[1] in E.NS}
    per = evaluate(cells2, cells1, {k[0]: "w2" for k in cells2}, with_own=False, with_w1=False)
    bad = 0
    want = {}
    for r in csv.DictReader(open(f"{EXPLORE_OUT}/i_selfsel_rules.csv")):
        if r["group"] == "all" and r["n"] == "all":
            want[(int(r["round_days"]), r["rule"])] = float(r["mean_regret"])
    for rd in E.CADENCES:
        for rule in ("two", "pooled", "same", "selector"):
            got = float(np.nanmean([per[k][rule] for k in per if k[2] == rd]))
            ok = abs(got - want[(rd, rule)]) < 1e-5
            bad += not ok
            print(f"selftest means  {rd}-day {rule:9s} new {got:.6f}  exploration {want[(rd, rule)]:.6f}  {'OK' if ok else 'MISMATCH'}")
    for r in csv.DictReader(open(f"{EXPLORE_OUT}/i_selfsel_paired.csv")):
        rd, other = int(r["round_days"]), r["comparison"].split()[-1]
        _, row = paired_row(per, [k for k in per if k[2] == rd], "selector", other)
        got = [row["mean"], row["lo"], row["hi"], row["share_negative"]]
        exp = [float(r["mean_diff"]), float(r["ci_lo"]), float(r["ci_hi"]), float(r["share_logs_selector_better"])]
        ok = (all(abs(a - b) < 1e-5 for a, b in zip(got[:3], exp[:3])) and abs(got[3] - exp[3]) < 6e-4 and row["logs"] == int(r["logs"]))
        bad += not ok
        print(f"selftest paired {rd}-day selector minus {other:7s} new {[round(x, 6) for x in got]} exploration {exp} {'OK' if ok else 'MISMATCH'}")
    print("SELFTEST", "FAIL" if bad else "PASS")
    print("-- window-1 anchors (exploration data standing in for window 2; the tables are the same-window contrast) --")
    V = verdicts(per)
    show(V, ["K1", "K2", "K3", "K4", "K8", "K9", "K10", "K13"])
    sys.exit(1 if bad else 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        selftest()
    path = f"{RUNS}/confirm.jsonl"
    if not os.path.exists(path):
        sys.exit(f"{path} is missing: run scripts/ssp/run_confirm.py first")
    cells2, kind_of = load_confirm(path)
    check_complete(cells2)
    cells1, _ = load()
    per = evaluate(cells2, cells1, kind_of)
    tables(per)
    V = verdicts(per)
    os.makedirs(OUT, exist_ok=True)
    sha = {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in HASHED}
    json.dump(clean(dict(registry=REG["version"], cells=len(per), sha256=sha, verdicts=V)), open(f"{OUT}/k_verdicts.json", "w"), indent=1)
    show(V, [f"K{i}" for i in range(1, 14)])
    print("decision:", V["decision"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
