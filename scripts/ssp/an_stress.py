"""Verdicts for the selector stress tests (stage S); mechanical, thresholds read from configs/selfsel_stress_registry.json.

    .venv/bin/python3 -W ignore scripts/ssp/an_stress.py                       # all verdicts (needs the complete run)
    .venv/bin/python3 -W ignore scripts/ssp/an_stress.py --check-only [--partial] [--rows PATH]    # integrity checks only, prints no result

Reads 02_runs/stress.jsonl (stress cells), 02_runs/confirm.jsonl and 02_runs/weekly.jsonl (the stored cells whose std arms every stress
cell must reproduce exactly) and rebuilds the plan from them (run_stress.build). Writes results/small_sample_popularity/03_tables/stress/:
s_cells_bias.csv, s_cells_dp.csv, s_cells_churn.csv, s_summary.csv, s_verdicts.json (verdicts plus sha256 of the registry, the scripts, the
simulation code, the plan keys and the rows file) and post hoc tables posthoc_*.csv.
Aborts when: a key is duplicated or does not match its row content; the cells are not exactly the planned cells; a family is empty;
stress.jsonl.errors is not empty; a std arm does not reproduce its stored cell; the std regrets do not reproduce k_cells.csv.
--partial and --rows are only allowed together with --check-only and never write a verdict.

Rule definitions (selector, regret, tables, bootstrap) are those of explore_selfsel_weekly.py / an_weekly.py and are imported, not copied.
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
import run_stress as RS
from an_weekly import RUNS, STEADY, boot

REGISTRY = "configs/selfsel_stress_registry.json"
REG = json.load(open(REGISTRY))
CR = REG["criteria"]
OUT = "results/small_sample_popularity/03_tables/stress"
CONFIRM_TABLES = "results/small_sample_popularity/03_tables/confirm"
RD = 7
NAN = float("nan")
FAMILIES = ("bias", "churn", "dp")
BIAS = {"std": "", "A+": "pps", "Ap": "ppsprev", "A-": "pps:a=-1"}
DPS = {"std": "", "eps2": "gauss:eps=2,c=10", "eps4": "gauss:eps=4,c=10", "eps8": "gauss:eps=8,c=10", "fresh4": "gauss:eps=4,c=10,fresh=1",
       "ref": "gauss:eps=1000000,c=10"}
HASHED = [REGISTRY] + [f"scripts/ssp/{f}" for f in (
    "stress_core.py", "run_stress.py", "an_stress.py", "explore_selfsel_weekly.py", "an_weekly.py", "deep_core.py", "track_core.py",
    "weekly_core.py", "confirm_core.py", "quiet.py")]


def sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------ loading
def load_rows(path: str) -> dict:
    """{key: row} with the big arrays turned into numpy arrays (_nd, _se, _sa, _ovl); duplicates and key/content mismatches abort."""
    rows = {}
    for line in open(path):
        r = json.loads(line)
        want = f"s|{r['family']}|{r['dataset']}|{r['n']}|{r['round_days']}"
        if r["key"] != want:
            sys.exit(f"row key {r['key']} does not match its content ({want})")
        if r["key"] in rows:
            sys.exit(f"duplicate row for {r['key']}")
        r["_nd"] = np.array(r.pop("nd100"), float)
        r["_se"] = np.array(r.pop("selfnd"), float)
        r["_sa"] = np.array(r.pop("selfaud"), float) if "selfaud" in r else None
        r["_ovl"] = np.array(r.pop("ovl"), float) if "ovl" in r else None
        r.pop("metrics", None)
        rows[r["key"]] = r
    return rows


def rebuilt_plan():
    """(plan {family: [keys]}, tasks {key: task}) rebuilt from the stored rows; aborts on an empty family or when the plan file differs."""
    built = {f: RS.build(REG, f) for f in FAMILIES}
    plan = {f: [t["key"] for t in built[f]] for f in FAMILIES}
    empty = [f for f, v in plan.items() if not v]
    if empty:
        sys.exit(f"plan rebuilt from the stored rows is empty for {empty}")
    path = f"{RUNS}/stress_plan.json"
    if os.path.exists(path) and json.load(open(path)) != plan:
        sys.exit(f"{path} differs from the plan rebuilt from the stored rows")
    return plan, {t["key"]: t for f in FAMILIES for t in built[f]}


def stored_arrays(path: str, want: set) -> dict:
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("kind") != "deep" or "@frac" in r["arms"][0]:
            continue
        stage = r["key"].split("|", 1)[0] if len(r["key"]) > 1 and r["key"][1] == "|" else None
        k = (r["dataset"], r["n"], r["round_days"], stage)
        if (k[0], k[1], k[2]) in want or (k[0], k[1], k[3]) in want:
            out[k] = (r["arms"], np.array(r["nd100"], float), np.array(r["selfnd"], float))
    return out


def check_integrity(rows: dict, plan: dict, tasks: dict, rows_path: str, partial: bool) -> None:
    errs = rows_path + ".errors"
    if os.path.exists(errs) and os.path.getsize(errs) > 0:
        sys.exit(f"{errs} is not empty: fix the failed cells before analysing")
    want = {k for v in plan.values() for k in v}
    extra = sorted(set(rows) - want)
    if extra:
        sys.exit(f"unplanned cells {extra[:3]}")
    if not partial:
        missing = sorted(want - set(rows))
        if missing:
            sys.exit(f"cell set differs from the plan: {len(missing)} missing {missing[:3]}")
        for f in FAMILIES:
            if not any(r["family"] == f for r in rows.values()):
                sys.exit(f"no rows for family {f}")
    for k, r in rows.items():
        t = tasks[k]
        if r["arms"] != t["arms"] or r.get("audit_n") != t.get("audit_n") or r["window"] != t["window"] or r["log"] != r["dataset"].replace("@w2", ""):
            sys.exit(f"{k}: arms, audit_n, window or log differ from the planned task")
        if r["family"] == "bias" and r["_sa"] is None:
            sys.exit(f"{k} has no selfaud (audit self-score)")
        if r["family"] == "churn" and (r["_ovl"] is None or not r.get("ovl_arms")):
            sys.exit(f"{k} has no ovl or ovl_arms (list overlaps)")
    need_w2 = {(r["dataset"], r["n"], r["round_days"]) for r in rows.values() if r["window"] == "w2"}
    need_w1 = {(r["dataset"], r["n"], "a" if r["n"] == 1000 else "d") for r in rows.values() if r["window"] == "w1"}
    st2 = stored_arrays(f"{RUNS}/confirm.jsonl", need_w2) if need_w2 else {}
    st1 = stored_arrays(f"{RUNS}/weekly.jsonl", need_w1) if need_w1 else {}
    bad = 0
    for k, r in rows.items():
        if r["window"] == "w2":
            hit = [v for kk, v in st2.items() if kk[0] == r["dataset"] and kk[1] == r["n"] and kk[2] == r["round_days"]]
        else:
            tag = "a" if r["n"] == 1000 else "d"
            hit = [v for kk, v in st1.items() if kk[0] == r["dataset"] and kk[1] == r["n"] and kk[3] == tag and kk[2] == r["round_days"]]
        if len(hit) != 1:
            print(f"no unique stored cell for {k}")
            bad += 1
            continue
        arms, nd, se = hit[0]
        for i, a in enumerate(arms):
            j = r["arms"].index(a)
            if not (np.array_equal(r["_nd"][:, j, :], nd[:, i, :], equal_nan=True) and np.array_equal(r["_se"][:, j, :], se[:, i, :], equal_nan=True)):
                print(f"std arm {a} of {k} does not reproduce the stored cell")
                bad += 1
                break
    if bad:
        sys.exit(f"{bad} cells fail the replication check")
    print(f"integrity: {len(rows)} cells ({'partial' if partial else 'complete'}), std arms reproduce the stored cells exactly")


# --------------------------------------------------------------------------------------------------------------------- views
def view(r: dict, suffix: str):
    """Cell of one sampler (arms renamed without the '@suffix'), in the layout explore_selfsel_weekly expects."""
    idx = [i for i, a in enumerate(r["arms"]) if a.partition("@")[2] == suffix]
    if not idx:
        return None
    names = [r["arms"][i].partition("@")[0] for i in idx]
    st = np.flatnonzero(np.array(r["steps"]) >= STEADY)
    nd, se = r["_nd"][:, idx, :], r["_se"][:, idx, :]
    sa = r["_sa"][:, idx, :] if r["_sa"] is not None else None
    val = dict(zip(names, np.nanmean(nd[:, :, st], axis=(0, 2))))
    return dict(arms=names, val=val, nd=nd, se=se, sa=sa, st=st, active=float(np.median(r["weekly_active"])), n=r["n"],
                rpp=(r["rows_per_round"][idx[0]] / max(r["people_per_round"][idx[0]], 1e-9)), ppr=r["people_per_round"][idx[0]])


def day_arrays(c):
    idx, ds = E.arms_of(c)
    return ds, c["nd"][:, idx, :], c["se"][:, idx, :], (c["sa"][:, idx, :] if c["sa"] is not None else None)


def best_days(c) -> int:
    return max((int(a.split("=")[1]) for a in c["val"] if a.startswith("decay:days=")), key=lambda d: (c["val"][f"decay:days={d}"], -d))


def rule_values(c) -> dict:
    """Values (mean nd100 over steady steps and draws) and regrets of the rules on one view; audit selector if the audit exists."""
    ds, nd, se, sa = day_arrays(c)
    st, fb = c["st"], E.nearest(ds, 2 * RD)
    b = E.best_value(c)
    sel = E.chosen_value(nd, E.choose(se, st, E.HL, fb), st)
    rec = dict(best=b, v_two=E.snap_value(c, 2 * RD), v_sel=sel, two=b - E.snap_value(c, 2 * RD), selector=b - sel, best_days=best_days(c),
               top_days=int(ds.max()), rpp=c["rpp"], ppr=c["ppr"], audit=NAN, v_aud=NAN, active=c["active"], n=c["n"])
    if sa is not None:
        va = E.chosen_value(nd, E.choose(sa, st, E.HL, fb), st)
        rec["v_aud"], rec["audit"] = va, b - va
    return rec


def with_tables(per: dict, cells: dict) -> None:
    """Add pooled / same-group leave-one-log-out table regrets to per[(log, n)] using the same-sampler cells[(log, n, RD, 'std')]."""
    for (L, n), rec in per.items():
        c = cells[(L, n, RD, "std")]
        others = sorted({l for (l, n1, rd1, s) in cells if n1 == n and l != L})
        same = [l for l in others if E.stratum(l) == E.stratum(L)]
        rec["pooled"] = rec["same"] = NAN
        if others:
            rec["pooled"] = rec["best"] - E.snap_value(c, RD * E.fit_rounds(cells, others, n, RD))
        if len(same) >= E.MIN_GROUP:
            rec["same"] = rec["best"] - E.snap_value(c, RD * E.fit_rounds(cells, same, n, RD))


def pdiff(per: dict, keys: list, a: str, b: str):
    d, dropped = {}, 0
    for k in keys:
        x, y = per[k][a], per[k][b]
        if np.isfinite(x) and np.isfinite(y):
            d.setdefault(k[0], []).append(x - y)
        else:
            dropped += 1
    return {l: float(np.mean(v)) for l, v in d.items()}, dropped


def paired(per, keys, a, b):
    d, dropped = pdiff(per, keys, a, b)
    m, lo, hi = boot(list(d.values()))
    return d, dict(mean=m, lo=lo, hi=hi, share_negative=float(np.mean([v < 0 for v in d.values()])) if d else NAN, logs=len(d), dropped_cells=dropped)


def mean_of(per, keys, field):
    v = np.array([per[k][field] for k in keys], float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else NAN


def log_level(per, keys, field):
    """Mean over logs of the log's mean over its cells (the log is the unit)."""
    d, _ = pdiff({k: dict(x=per[k][field], z=0.0) for k in keys}, keys, "x", "z")
    return float(np.mean(list(d.values()))) if d else NAN


def tri(flag):
    return None if flag is None else bool(flag)


# ------------------------------------------------------------------------------------------------------------------ family B
def family_b(rows: dict) -> dict:
    cells = {name: {} for name in BIAS}
    for r in rows.values():
        if r["family"] == "bias":
            for name, suf in BIAS.items():
                cells[name][(r["log"], r["n"], RD, "std")] = view(r, suf)
    per = {name: {} for name in BIAS}
    for name in BIAS:
        for (L, n, rd, _), c in cells[name].items():
            per[name][(L, n)] = rule_values(c)
        with_tables(per[name], cells[name])
    return per


def verdicts_b(per: dict) -> dict:
    V = {}
    keys = sorted(per["std"])
    c1, c2 = CR["S1"], CR["S2"]
    ratio = {x: float(np.mean([per[x][k]["rpp"] / per["std"][k]["rpp"] for k in keys])) for x in ("A+", "Ap", "A-")}
    gain = {x: float(np.mean([per[x][k]["best"] - per["std"][k]["best"] for k in keys])) for x in ("A+", "Ap", "A-")}
    ok = {"A+": ratio["A+"] >= c1["A+"]["min_ratio"], "Ap": ratio["Ap"] >= c1["Ap"]["min_ratio"], "A-": ratio["A-"] <= c1["A-"]["max_ratio"]}
    V["S1"] = dict(ratio_rows_per_reporter=ratio, cells=len(keys), met_by_sampler={x: bool(v) for x, v in ok.items()}, met=bool(all(ok.values())))
    s2 = {"A+": gain["A+"] >= c2["A+"]["min_gain"], "A-": gain["A-"] <= c2["A-"]["max_gain"]}
    V["S2"] = dict(mean_best_gain_vs_std=gain, cells=len(keys), met_by_sampler={x: bool(v) for x, v in s2.items()},
                   interpreted_by_sampler={"A+": bool(ok["A+"]), "A-": bool(ok["A-"])},
                   met=bool(s2["A+"] and s2["A-"]) if (ok["A+"] and ok["A-"]) else None)
    c3, c4 = CR["S3"], CR["S4"]
    for sid3, sid4, x in (("S3a", "S4a", "A+"), ("S3b", "S4b", "Ap"), ("S3c", "S4c", "A-")):
        d, row = paired(per[x], keys, "selector", "same")
        mr = log_level(per[x], keys, "selector")
        row.update(sampler=x, interpreted=bool(ok[x]), mean_selector_regret_over_logs=mr, mean_same_table_regret=mean_of(per[x], keys, "same"),
                   mean_two_regret=mean_of(per[x], keys, "two"),
                   selector_regret_minus_std_selector_regret=float(np.mean([per[x][k]["selector"] - per["std"][k]["selector"] for k in keys])))
        row["met"] = tri(bool(np.isfinite(mr) and mr <= c3["max_mean_regret"] and d and row["mean"] < 0 and row["hi"] < c3["ci_upper_below"]) if ok[x] else None)
        V[sid3] = row
        g = {k: dict(g=per[x][k]["selector"] - per[x][k]["audit"], z=0.0) for k in keys}
        dl, grow = paired(g, keys, "g", "z")
        grow.update(sampler=x, interpreted=bool(ok[x]), mean_audit_gain=float(np.nanmean([g[k]["g"] for k in keys])), cells=len(keys),
                    cells_with_n_at_least_half_the_active=int(sum(per[x][k]["n"] >= 0.5 * per[x][k]["active"] for k in keys)))
        grow["met"] = tri(bool(grow["mean_audit_gain"] < c4["max_audit_gain"] and grow["hi"] < c4["ci_upper_below"]) if ok[x] else None)
        V[sid4] = grow
    g = np.array([per["std"][k]["audit"] - per["std"][k]["selector"] for k in keys], float)
    V["S5"] = dict(mean_audit_loss=float(np.nanmean(g)), cells=len(keys), met=bool(np.nanmean(g) <= CR["S5"]["max_audit_loss"]))
    return V


# ------------------------------------------------------------------------------------------------------------------ family D
def family_d(rows: dict) -> dict:
    per, cells = {name: {} for name in DPS}, {name: {} for name in DPS}
    for r in rows.values():
        if r["family"] == "dp":
            for name, suf in DPS.items():
                cells[name][(r["log"], r["n"], RD, "std")] = view(r, suf)
    for name in DPS:
        for (L, n, rd, _), c in cells[name].items():
            per[name][(L, n)] = rule_values(c)
        for n in (1000, 3000):                                   # tables only within the same n; groups are too small, pooled only
            sub = {k: v for k, v in cells[name].items() if k[1] == n}
            tabs = {kk: per[name][kk] for kk in per[name] if kk[1] == n}
            with_tables(tabs, sub)
    return per


def price(per, name, n, field, ref="ref"):
    """{log: value of the rule with the capped noise-free reference minus its value with the noise of `name`} (ref='std': uncapped)."""
    return {k[0]: per[ref][k][field] - per[name][k][field] for k in per[name] if k[1] == n}


def verdicts_d(per: dict, rows: dict) -> dict:
    V = {}
    c6, c7, c8, c9, c10, c11 = (CR[s] for s in ("S6", "S7", "S8", "S9", "S10", "S11"))
    keys_all = sorted(per["std"])
    ev = [k for k in keys_all if not (c6["exclude_nonprivate_best_at_grid_top"] and per["ref"][k]["best_days"] >= per["ref"][k]["top_days"])]
    ratio = [per["eps4"][k]["best_days"] / per["ref"][k]["best_days"] for k in ev]
    share = float(np.mean([x >= c6["min_factor"] for x in ratio])) if ratio else NAN
    V["S6"] = dict(cells=len(keys_all), evaluable_cells=len(ev), not_evaluable_nonprivate_best_at_grid_top=len(keys_all) - len(ev),
                   share_best_halflife_at_least_double=share, median_ratio=float(np.median(ratio)) if ratio else NAN,
                   met=(bool(share >= c6["min_share"]) if ratio else None))
    k1000 = [k for k in keys_all if k[1] == c7["n"]]
    V["S7"] = dict(cells=len(k1000), mean_regret_two_rounds=mean_of(per["eps4"], k1000, "two"), mean_regret_selector=mean_of(per["eps4"], k1000, "selector"),
                   met=bool(mean_of(per["eps4"], k1000, "two") >= c7["min_regret"]))
    d, row = paired(per["eps4"], k1000, "selector", "pooled")
    mr = log_level(per["eps4"], k1000, "selector")
    row.update(mean_selector_regret_over_logs=mr, mean_pooled_table_regret=mean_of(per["eps4"], k1000, "pooled"), mean_two_regret=mean_of(per["eps4"], k1000, "two"))
    row["met"] = bool(np.isfinite(mr) and mr <= c8["max_mean_regret"] and d and row["mean"] < 0 and row["hi"] < c8["ci_upper_below"])
    V["S8"] = row
    ps, p2 = price(per, "eps4", c9["n"], "v_sel"), price(per, "eps4", c9["n"], "v_two")
    logs = sorted(ps)
    scored = [l for l in logs if p2[l] > c9["skip_logs_with_two_round_price_at_most"]]
    skipped = sorted(set(logs) - set(scored))
    ok9 = [ps[l] <= c9["max_cost_ratio"] * p2[l] for l in scored]
    ps_std, p2_std = price(per, "eps4", c9["n"], "v_sel", "std"), price(per, "eps4", c9["n"], "v_two", "std")
    cap_loss = {l: per["std"][(l, c9["n"])]["v_sel"] - per["ref"][(l, c9["n"])]["v_sel"] for l in logs}
    m_sel, m_two = float(np.mean([ps[l] for l in logs])), float(np.mean([p2[l] for l in logs]))
    V["S9"] = dict(logs=len(logs), logs_in_share=len(scored), skipped_logs_in_share=skipped, mean_price_selector=m_sel, mean_price_two_rounds=m_two,
                   share_logs_selector_at_most_half=float(np.mean(ok9)) if ok9 else NAN,
                   mean_price_selector_vs_uncapped_std=float(np.mean(list(ps_std.values()))), mean_price_two_rounds_vs_uncapped_std=float(np.mean(list(p2_std.values()))),
                   mean_cap_loss_selector=float(np.mean(list(cap_loss.values()))),
                   met=bool(m_sel <= c9["max_cost_ratio"] * m_two and ok9 and np.mean(ok9) >= c9["min_share_logs"]))
    fill = {k: per["fresh4"][k]["ppr"] / k[1] for k in k1000}
    q = [k for k in k1000 if fill[k] >= c10["min_fill"]]
    diff = [per["fresh4"][k]["v_sel"] - per["eps4"][k]["v_sel"] for k in q]
    V["S10"] = dict(cells=len(k1000), qualifying_cells=len(q), mean_fill_share_of_n=float(np.mean(list(fill.values()))) if fill else NAN,
                    mean_diff=float(np.mean(diff)) if diff else NAN, mean_abs_diff=float(np.mean(np.abs(diff))) if diff else NAN,
                    met=(bool(abs(float(np.mean(diff))) <= c10["max_abs_diff"]) if len(q) >= c10["min_cells"] and np.isfinite(np.mean(diff)) else None))
    strong = float(np.mean(list(price(per, "eps2", c11["strong_privacy"]["n"], "v_sel").values())))
    weak = float(np.mean(list(price(per, "eps8", c11["weak_privacy"]["n"], "v_sel").values())))
    V["S11"] = dict(price_selector_eps2_n1000=strong, price_selector_eps8_n3000=weak, logs_n3000=len(price(per, "eps8", 3000, "v_sel")),
                    met=bool(strong >= c11["strong_privacy"]["min_cost"] and weak <= c11["weak_privacy"]["max_cost"]))
    return V


# ------------------------------------------------------------------------------------------------------------------ family C
def churn_of_cell(r: dict) -> dict:
    c = view(r, "")
    ds, nd, se, _ = day_arrays(c)
    st, fb = c["st"], E.nearest(ds, 2 * RD)
    S = nd.shape[2]
    ch = E.choose(se, list(range(1, S)), E.HL, fb)
    pos = {a: i for i, a in enumerate(r["ovl_arms"])}
    names = [f"decay:days={int(d)}" for d in ds]
    ovl = r["_ovl"]
    sel = []
    for t in st:
        if t - 1 not in ch or t not in ch:
            continue
        for d in range(nd.shape[0]):
            sel.append(1.0 - ovl[d, pos[names[ch[t - 1][d]]], pos[names[ch[t][d]]], t])

    def fixed(a):
        i = pos[a]
        return float(np.nanmean(1.0 - ovl[:, i, i, :][:, st]))
    best_arm, two_arm = f"decay:days={best_days(c)}", names[fb]
    return dict(churn_selector=float(np.nanmean(sel)), churn_best=fixed(best_arm), churn_two=fixed(two_arm), best_arm=best_arm, two_arm=two_arm)


def verdicts_c(rows: dict):
    per = {(r["log"], r["n"]): churn_of_cell(r) for r in rows.values() if r["family"] == "churn"}
    keys = sorted(per)
    ex = np.array([per[k]["churn_selector"] - per[k]["churn_best"] for k in keys])
    sh = float(np.mean([per[k]["churn_selector"] < per[k]["churn_two"] for k in keys]))
    V = {"S12": dict(cells=len(keys), mean_extra_churn_vs_best_fixed=float(ex.mean()), mean_churn_selector=mean_of(per, keys, "churn_selector"),
                     mean_churn_best_fixed=mean_of(per, keys, "churn_best"), met=bool(ex.mean() <= CR["S12"]["max_extra_churn"])),
         "S13": dict(cells=len(keys), share_selector_steadier_than_two_rounds=sh, mean_churn_two_rounds=mean_of(per, keys, "churn_two"),
                     met=bool(sh >= CR["S13"]["min_share"]))}
    return V, per


# ------------------------------------------------------------------------------------------------------------------- outputs
def r6(x):
    return "" if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)) else (round(float(x), 6) if isinstance(x, (float, np.floating)) else x)


def write_csv(name, head, body):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}", "w", newline="") as f:
        cw = csv.writer(f, lineterminator="\n")
        cw.writerow(head)
        cw.writerows(body)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


FIELDS = ["best", "best_days", "top_days", "two", "selector", "audit", "pooled", "same", "v_two", "v_sel", "v_aud", "rpp", "ppr", "active"]


def tables(pb, pd_, pc):
    write_csv("s_cells_bias.csv", ["sampler", "log", "n", "group"] + FIELDS,
              [[name, L, n, E.stratum(L)] + [r6(rec.get(f)) for f in FIELDS] for name in pb for (L, n), rec in sorted(pb[name].items())])
    write_csv("s_cells_dp.csv", ["sampler", "log", "n", "group"] + FIELDS,
              [[name, L, n, E.stratum(L)] + [r6(rec.get(f)) for f in FIELDS] for name in pd_ for (L, n), rec in sorted(pd_[name].items())])
    write_csv("s_cells_churn.csv", ["log", "n", "group", "churn_selector", "churn_best_fixed", "churn_two_rounds", "best_arm", "two_arm"],
              [[L, n, E.stratum(L), r6(v["churn_selector"]), r6(v["churn_best"]), r6(v["churn_two"]), v["best_arm"], v["two_arm"]] for (L, n), v in sorted(pc.items())])
    rows = []
    for fam, per in (("bias", pb), ("dp", pd_)):
        for name in per:
            for n in sorted({k[1] for k in per[name]}):
                keys = [k for k in per[name] if k[1] == n]
                rows.append([fam, name, n, len(keys)] + [r6(mean_of(per[name], keys, f)) for f in ("two", "selector", "audit", "pooled", "same", "best")])
    write_csv("s_summary.csv", ["family", "sampler", "n", "cells", "regret_two", "regret_selector", "regret_audit_selector", "regret_pooled_table", "regret_same_group_table", "best_fixed_value"], rows)


def posthoc(pb, pd_):
    rows = []
    keys = sorted(pb["std"])
    for x in ("A+", "Ap", "A-"):
        for n in sorted({k[1] for k in keys}) + ["all"]:
            ks = [k for k in keys if n == "all" or k[1] == n]
            d, row = paired(pb[x], ks, "selector", "same")
            g = np.array([pb[x][k]["selector"] - pb[x][k]["audit"] for k in ks], float)
            rows.append([x, n, len(ks), r6(mean_of(pb[x], ks, "selector")), r6(row["mean"]), r6(row["lo"]), r6(row["hi"]), r6(float(np.nanmean(g)))])
    write_csv("posthoc_bias_by_n.csv", ["sampler", "n", "cells", "mean_selector_regret", "selector_minus_same_mean", "ci_lo", "ci_hi", "mean_audit_gain"], rows)
    rows = []
    for e in ("eps2", "eps4", "eps8", "fresh4", "ref"):
        for n in (1000, 3000):
            ks = [k for k in sorted(pd_["std"]) if k[1] == n]
            if not ks:
                continue
            ps, p2 = price(pd_, e, n, "v_sel"), price(pd_, e, n, "v_two")
            d, row = paired(pd_[e], ks, "selector", "pooled")
            top = float(np.mean([pd_[e][k]["best_days"] >= pd_[e][k]["top_days"] for k in ks]))
            rows.append([e, n, len(ks), r6(mean_of(pd_[e], ks, "two")), r6(mean_of(pd_[e], ks, "selector")), r6(mean_of(pd_[e], ks, "pooled")),
                         r6(row["mean"]), r6(row["lo"]), r6(row["hi"]), r6(float(np.mean(list(ps.values())))), r6(float(np.mean(list(p2.values())))),
                         r6(float(np.median([pd_[e][k]["best_days"] / pd_["std"][k]["best_days"] for k in ks]))), r6(top)])
    write_csv("posthoc_dp_by_eps_n.csv", ["sampler", "n", "cells", "regret_two", "regret_selector", "regret_pooled_table", "selector_minus_pooled_mean", "ci_lo", "ci_hi",
                                          "price_selector", "price_two_rounds", "median_best_halflife_ratio_vs_nonprivate", "share_best_at_grid_top"], rows)


def check_against_confirmation(pb) -> None:
    """The std rows of family B are the confirmation cells: selector, 2-round, pooled and same-group regrets must equal k_cells.csv."""
    path = f"{CONFIRM_TABLES}/k_cells.csv"
    if not os.path.exists(path):
        sys.exit(f"{path} is missing: the std regrets cannot be checked against the confirmation")
    ref = {(r["log"], int(r["n"])): r for r in csv.DictReader(open(path)) if r["kind"] == "w2" and int(r["round_days"]) == RD}
    bad = 0
    for (L, n), rec in pb["std"].items():
        r = ref.get((L, n))
        if r is None:
            sys.exit(f"cell {(L, n)} is not in {path}")
        for mine in ("selector", "two", "pooled", "same"):
            a, b = rec[mine], (float(r[mine]) if r[mine] != "" else NAN)
            if (np.isfinite(a) != np.isfinite(b)) or (np.isfinite(a) and abs(a - b) > 2e-6):
                bad += 1
    if bad:
        sys.exit(f"{bad} std regrets differ from the confirmation tables (k_cells.csv)")
    print("std regrets of family B reproduce the confirmation tables (k_cells.csv)")


def show(V):
    for k in sorted(V, key=lambda s: (int("".join(ch for ch in s if ch.isdigit())), s)):
        v = V[k]
        print(k, {None: "NOT EVALUABLE / NOT INTERPRETED", True: "MET", False: "NOT MET"}[v["met"]],
              {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items() if a != "met" and not isinstance(b, (dict, list))},
              {a: {x: (round(y, 4) if isinstance(y, float) else y) for x, y in b.items()} for a, b in v.items() if isinstance(b, dict)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--rows", default=f"{RUNS}/stress.jsonl")
    a = ap.parse_args()
    if (a.partial or a.rows != f"{RUNS}/stress.jsonl") and not a.check_only:
        sys.exit("--partial and --rows are only allowed together with --check-only (a partial analysis never writes verdicts)")
    if not os.path.exists(a.rows):
        sys.exit(f"{a.rows} is missing: run scripts/ssp/run_stress.py first")
    plan, tasks = rebuilt_plan()
    rows = load_rows(a.rows)
    check_integrity(rows, plan, tasks, a.rows, a.partial)
    if a.check_only:
        return
    pb, pd_ = family_b(rows), family_d(rows)
    check_against_confirmation(pb)
    Vc, pc = verdicts_c(rows)
    V = {**verdicts_b(pb), **verdicts_d(pd_, rows), **Vc}
    tables(pb, pd_, pc)
    posthoc(pb, pd_)

    def gate(sids):
        return None if any(V[s]["met"] is None for s in sids) else all(V[s]["met"] for s in sids)
    V["families"] = dict(B_robust=gate(("S3a", "S3b", "S3c")), B_audit_unhelpful=gate(("S4a", "S4b", "S4c")),
                         D_memory_cuts_price=bool(V["S8"]["met"] and V["S9"]["met"]), C_switching_cheap=bool(V["S12"]["met"] and V["S13"]["met"]))
    os.makedirs(OUT, exist_ok=True)
    sha = {p: sha_file(p) for p in HASHED}
    sha["plan_keys"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    sha["stress.jsonl"] = sha_file(a.rows)
    for p in (f"{RUNS}/confirm.jsonl", f"{RUNS}/weekly.jsonl", f"{CONFIRM_TABLES}/k_cells.csv"):
        sha[p] = sha_file(p)
    json.dump(clean(dict(registry=REG["version"], cells=len(rows), sha256=sha, verdicts=V)), open(f"{OUT}/s_verdicts.json", "w"), indent=1)
    show({k: v for k, v in V.items() if k != "families"})
    print("families:", V["families"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
