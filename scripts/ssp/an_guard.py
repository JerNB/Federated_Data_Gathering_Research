"""Verdicts for programme G (history-seeded and guarded selectors on window 3); mechanical, thresholds and parameters from configs/selfsel_guard_registry.json.

    .venv/bin/python3 -W ignore scripts/ssp/an_guard.py                 # all verdicts (needs the complete run)
    .venv/bin/python3 -W ignore scripts/ssp/an_guard.py --check-only [--partial] [--rows PATH]   # integrity checks only, prints no result

Reads 02_runs/guard.jsonl (w3 cells), 02_runs/weekly.jsonl (w1 history cells) and 02_runs/confirm.jsonl (w2 history cells); rebuilds the plan with
run_guard.build and aborts when the cells differ from it, when a key is duplicated or does not match its content, when the errors file is not
empty, when the arms, draws, shapes or n_win of a row differ from the plan, when the plan file is missing, differs or records another seed tag or draw count than the registry, or when a history file has a
duplicated key. --partial and --rows are only allowed with --check-only and never write a verdict.
Writes results/small_sample_popularity/03_tables/guard/: g_cells.csv, g_summary.csv, g_paired.csv, g_verdicts.json (with sha256 of the registry,
the scripts, the simulation code, the history files and the plan) and the post hoc table posthoc_causal.csv.
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

import an_confirm as AC
import an_weekly as AW
import confirm_core as K
import explore_selfsel_weekly as E
import guard_core as G
import run_guard as RG
import weekly_core as W

REGISTRY = "configs/selfsel_guard_registry.json"
REG = json.load(open(REGISTRY))
CR, PAR = REG["criteria"], REG["params"]
RUNS = AW.RUNS
OUT = "results/small_sample_popularity/03_tables/guard"
STEADY = AW.STEADY
NAN = float("nan")
REGIONS = {"n30_7d": lambda n, rd: n == 30 and rd == 7, "n30_28d": lambda n, rd: n == 30 and rd == 28,
           "n100plus_7d": lambda n, rd: n >= 100 and rd == 7, "n100plus_28d": lambda n, rd: n >= 100 and rd == 28}
RULES = ["plain", "two", "warm", "warm104", "ownself", "margin_ownself", "own_oracle", "table_only", "margin_table", "fixedshare", "same_w3"]
MIN_LOGS = REG["design"]["min_logs"]
HASHED = [REGISTRY, "configs/selfsel_confirm_registry.json"] + [f"scripts/ssp/{f}" for f in (
    "window3_core.py", "run_guard.py", "guard_core.py", "tune_guard.py", "an_guard.py", "an_confirm.py", "explore_selfsel_weekly.py", "an_weekly.py",
    "deep_core.py", "track_core.py", "weekly_core.py", "confirm_core.py", "quiet.py")]


def sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------ loading
def load_rows(path: str) -> dict:
    rows = {}
    for line in open(path):
        r = json.loads(line)
        if r["key"] != f"g|{r['dataset']}|{r['n']}|{r['round_days']}":
            sys.exit(f"row key {r['key']} does not match its content")
        if r["key"] in rows:
            sys.exit(f"duplicate row for {r['key']}")
        r["_nd"], r["_se"] = np.array(r.pop("nd100"), float), np.array(r.pop("selfnd"), float)
        r.pop("metrics", None)
        rows[r["key"]] = r
    return rows


def check_history_unique() -> None:
    for path in (f"{RUNS}/weekly.jsonl", f"{RUNS}/confirm.jsonl"):
        seen = set()
        for line in open(path):
            r = json.loads(line)
            if r.get("kind") != "deep":
                continue
            k = (r["dataset"], r["n"], r["round_days"], "@frac" in r["arms"][0])
            if k in seen:
                sys.exit(f"duplicate cell {k} in {path}")
            seen.add(k)


def check_integrity(rows: dict, tasks: dict, info: dict, rows_path: str, partial: bool) -> None:
    errs = rows_path + ".errors"
    if os.path.exists(errs) and os.path.getsize(errs) > 0:
        sys.exit(f"{errs} is not empty: fix the failed cells before analysing")
    extra = sorted(set(rows) - set(tasks))
    if extra:
        sys.exit(f"unplanned cells {extra[:3]}")
    if not partial:
        missing = sorted(set(tasks) - set(rows))
        if missing:
            sys.exit(f"cell set differs from the plan: {len(missing)} missing {missing[:3]}")
        if not rows:
            sys.exit("no rows")
    for k, r in rows.items():
        t = tasks[k]
        if (r["arms"] != t["arms"] or r["log"] != t["log"] or r["round_days"] != t["round_days"] or r["n"] != t["n"] or r["_nd"].shape[0] != t["draws"]
                or r["n_win"] != t["n_win"]):
            sys.exit(f"{k}: arms, log, round_days, n, draws or n_win differ from the planned task")
        if not (r["_nd"].shape == r["_se"].shape == (t["draws"], len(r["arms"]), len(r["steps"]))):
            sys.exit(f"{k}: nd100, selfnd and steps have inconsistent shapes")
    path = f"{RUNS}/guard_plan.json"
    if not os.path.exists(path):
        sys.exit(f"{path} is missing (run_guard.py writes it when a run starts)")
    stored = json.load(open(path))
    if stored.get("meta") != dict(seed_tag=REG["stage"]["seed_tag"], draws=REG["stage"]["draws"]):
        sys.exit(f"{path} records a seed tag or draw count that differs from the registry")
    if sorted(stored["keys"]) != sorted(tasks):
        sys.exit(f"{path} lists different keys from the plan rebuilt from the registry")
    starts = lambda d: {l: v["start_week"] for l, v in d.items() if isinstance(v, dict)}
    if starts(stored["info"]) != starts(info):
        sys.exit(f"{path} has different w3 start weeks from the rebuilt plan")
    print(f"integrity: {len(rows)} cells ({'partial' if partial else 'complete'})")


# ------------------------------------------------------------------------------------------------------------------ views and priors
def view(r: dict) -> dict:
    st = np.flatnonzero(np.array(r["steps"]) >= STEADY)
    val = dict(zip(r["arms"], np.nanmean(r["_nd"][:, :, st], axis=(0, 2))))
    return dict(arms=r["arms"], val=val, nd=r["_nd"], se=r["_se"], st=st)


def history_cells():
    c1, _ = AW.load()
    c1 = {k: v for k, v in c1.items() if k[3] == "std" and k[2] in (7, 28)}
    c2, kind = AC.load_confirm(f"{RUNS}/confirm.jsonl")
    c2 = {k: v for k, v in c2.items() if kind[k[0]] == "w2" and k[2] in (7, 28)}
    return c1, c2


def pooled_best_days(cells, what):
    """Half-life with the best mean value over the history cells, on the arms common to all of them (the grids of w1 and w2 can differ at n = 100, 1000)."""
    vecs = [G.days_vec(c, what) for c in cells]
    vecs = [v for v in vecs if v]
    if not vecs:
        return None
    common = sorted(set.intersection(*[set(v) for v in vecs]))
    if not common:
        return None
    mean = {d: float(np.mean([v[d] for v in vecs])) for d in common}
    return max(common, key=lambda d: (mean[d], -d))


def table_days(L, n, rd, c1, c2):
    pairs = [(l, c) for cc in (c1, c2) for (l, n1, rd1, s), c in cc.items() if n1 == n and rd1 == rd and l != L]
    logs = {l for l, _ in pairs}
    same = {l for l in logs if E.stratum(l) == E.stratum(L)}
    use = same if len(same) >= E.MIN_GROUP else logs
    cells = [c for l, c in pairs if l in use]
    if not cells:
        return None
    r = min(E.RS, key=lambda r: np.mean([E.best_value(c) - E.snap_value(c, rd * r) for c in cells]))
    return rd * r


def cell_rules(L, n, rd, c3, cells3, c1, c2) -> dict:
    ds, nd, se = E.prep(c3)
    st, b = c3["st"], E.best_value(c3)
    fb = E.nearest(ds, 2 * rd)
    rec = {r: NAN for r in RULES}
    rec["plain"] = b - E.chosen_value(nd, E.choose(se, st, E.HL, fb), st)
    rec["two"] = b - E.fixed_value(nd, fb, st)
    rec["fixedshare"] = b - G.value_of(nd, G.choose_fixedshare(se, st, PAR["fixedshare_eta"], PAR["fixedshare_alpha"], fb), st)
    hist = [c for c in (c1.get((L, n, rd, "std")), c2.get((L, n, rd, "std"))) if c is not None]
    hs = G.hist_scores(hist, ds)
    if hs is not None:
        rec["warm"] = b - G.value_of(nd, G.choose_warm(se, st, E.HL, hs, PAR["warm_m"]), st)
        rec["warm104"] = b - G.value_of(nd, G.choose_warm(se, st, E.HL, hs, PAR["warm_m_large"]), st)
    d_or, d_os = pooled_best_days(hist, "val"), pooled_best_days(hist, "se")
    if d_or is not None:
        rec["own_oracle"] = b - E.fixed_value(nd, E.nearest(ds, d_or), st)
    if d_os is not None:
        i = E.nearest(ds, d_os)
        rec["ownself"] = b - E.fixed_value(nd, i, st)
        rec["margin_ownself"] = b - G.value_of(nd, G.choose_margin(se, st, E.HL, i, PAR["delta_ownself"]), st)
    dt = table_days(L, n, rd, c1, c2)
    if dt is not None:
        i = E.nearest(ds, dt)
        rec["table_only"] = b - E.fixed_value(nd, i, st)
        rec["margin_table"] = b - G.value_of(nd, G.choose_margin(se, st, E.HL, i, PAR["delta_table"]), st)
    logs = sorted({l for (l, n1, rd1, s) in cells3 if n1 == n and rd1 == rd and l != L})
    same = [l for l in logs if E.stratum(l) == E.stratum(L)]
    if len(same) >= E.MIN_GROUP:
        rec["same_w3"] = b - E.snap_value(c3, rd * E.fit_rounds(cells3, same, n, rd))
    rec["n_hist_windows"] = len(hist)
    return rec


# ------------------------------------------------------------------------------------------------------------------ statistics
def log_means(per, keys, rule):
    d = {}
    for k in keys:
        v = per[k][rule]
        if np.isfinite(v):
            d.setdefault(k[0], []).append(v)
    return {l: float(np.mean(x)) for l, x in d.items()}


def paired(per, keys, a, b):
    d, dropped = {}, 0
    for k in keys:
        x, y = per[k][a], per[k][b]
        if np.isfinite(x) and np.isfinite(y):
            d.setdefault(k[0], []).append(x - y)
        else:
            dropped += 1
    d = {l: float(np.mean(v)) for l, v in d.items()}
    m, lo, hi = AW.boot(list(d.values()))
    return dict(mean=m, lo=lo, hi=hi, logs=len(d), dropped_cells=dropped, share_negative=float(np.mean([v < 0 for v in d.values()])) if d else NAN)


def region_keys(per, region):
    f = REGIONS[region]
    return [k for k in per if f(k[1], k[2])]


def verdicts(per) -> dict:
    V = {}

    def pr(a, b, region, keys=None):
        return paired(per, keys if keys is not None else region_keys(per, region), a, b)

    def ev(*rows):
        return all(r["logs"] >= MIN_LOGS for r in rows)

    def sel(cond, *rows):
        return bool(cond) if ev(*rows) else None
    c = CR["G1"]; row = pr("warm", "plain", c["region"]); V["G1"] = dict(row, met=sel(row["mean"] <= c["max_mean_diff"] and row["hi"] < c["ci_upper_below"], row))
    c = CR["G2"]; row = pr("warm", "plain", c["region"]); V["G2"] = dict(row, met=sel(row["mean"] <= c["max_mean_diff"] and row["hi"] < c["ci_upper_below"], row))
    r7, r28 = pr("warm", "plain", "n100plus_7d"), pr("warm", "plain", "n100plus_28d")
    V["G3"] = dict(diff_7d=r7["mean"], ci_hi_7d=r7["hi"], diff_28d=r28["mean"], ci_hi_28d=r28["hi"], logs_7d=r7["logs"], logs_28d=r28["logs"],
                   met=sel(r7["hi"] < CR["G3"]["ci_upper_below_7d"] and r28["hi"] < CR["G3"]["ci_upper_below_28d"], r7, r28))
    rr = {rg: pr("ownself", "warm", rg) for rg in REGIONS}
    V["G4"] = dict(diff_by_region={rg: v["mean"] for rg, v in rr.items()}, ci_lo_by_region={rg: v["lo"] for rg, v in rr.items()},
                   logs_by_region={rg: v["logs"] for rg, v in rr.items()}, met=sel(all(v["lo"] > CR["G4"]["ci_lower_above"] for v in rr.values()), *rr.values()))
    c = CR["G5"]; row = pr("margin_table", "plain", c["region"]); V["G5"] = dict(row, met=sel(row["mean"] <= c["max_mean_diff"] and row["hi"] < c["ci_upper_below"], row))
    r7, r28 = pr("margin_table", "plain", "n100plus_7d"), pr("margin_table", "plain", "n100plus_28d")
    V["G6"] = dict(diff_7d=r7["mean"], diff_28d=r28["mean"], ci_hi_7d=r7["hi"], ci_hi_28d=r28["hi"], logs_7d=r7["logs"], logs_28d=r28["logs"],
                   met=sel(r7["mean"] <= CR["G6"]["max_diff_7d"] and r28["mean"] <= CR["G6"]["max_diff_28d"], r7, r28))
    rr = {rg: pr("fixedshare", "plain", rg) for rg in REGIONS}
    V["G7"] = dict(diff_by_region={rg: v["mean"] for rg, v in rr.items()}, logs_by_region={rg: v["logs"] for rg, v in rr.items()},
                   met=sel(all(v["mean"] > -CR["G7"]["min_gain_to_count"] for v in rr.values()), *rr.values()))
    k7 = [k for k in per if k[2] == 7]
    c = CR["G8"]; row = pr("own_oracle", "plain", None, k7); V["G8"] = dict(row, met=sel(row["mean"] <= c["max_mean_diff"] and row["hi"] < c["ci_upper_below"], row))
    rr = {rg: pr("ownself", "own_oracle", rg) for rg in REGIONS}
    V["G9"] = dict(gap_by_region={rg: v["mean"] for rg, v in rr.items()}, logs_by_region={rg: v["logs"] for rg, v in rr.items()},
                   met=sel(all(abs(v["mean"]) <= CR["G9"]["max_abs_gap"] for v in rr.values()), *rr.values()))
    v = np.array([per[k]["warm"] for k in k7], float)
    v = v[np.isfinite(v)]
    n_logs10 = len({k[0] for k in k7 if np.isfinite(per[k]["warm"])})
    V["G10"] = dict(cells=len(v), logs=n_logs10, share_over_0_01=float(np.mean(v > 0.01)) if len(v) else NAN, largest=float(v.max()) if len(v) else NAN,
                    met=(bool(np.mean(v > 0.01) <= CR["G10"]["max_share_over_0.01"] and v.max() <= CR["G10"]["max_cell_regret"]) if n_logs10 >= MIN_LOGS else None))
    rr = {rg: pr("warm104", "warm", rg) for rg in REGIONS}
    V["G11"] = dict(diff_by_region={rg: v["mean"] for rg, v in rr.items()}, logs_by_region={rg: v["logs"] for rg, v in rr.items()},
                    met=sel(all(v["mean"] <= CR["G11"]["max_diff"] for v in rr.values()), *rr.values()))
    def agg(*sids):
        mets = [V[s]["met"] for s in sids]
        return False if any(m is False for m in mets) else (None if any(m is None for m in mets) else True)
    V["decisions"] = dict(history_seeded_selector=agg("G1", "G2", "G3"), freezing_not_better=V["G4"]["met"], table_guard=agg("G5", "G6"))
    return V


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


def tables(per, info):
    write_csv("g_cells.csv", ["log", "group", "n", "round_days", "n_hist_windows"] + RULES,
              [[k[0], E.stratum(k[0]), k[1], k[2], per[k]["n_hist_windows"]] + [r6(per[k][r]) for r in RULES] for k in sorted(per)])
    body = []
    for rg in REGIONS:
        ks = region_keys(per, rg)
        for rule in RULES:
            m = log_means(per, ks, rule)
            body.append([rg, rule, len(m), r6(float(np.mean(list(m.values()))) if m else NAN)])
    write_csv("g_summary.csv", ["region", "rule", "logs", "mean_regret_over_logs"], body)
    body = []
    for a, b in (("warm", "plain"), ("warm104", "plain"), ("warm104", "warm"), ("ownself", "plain"), ("ownself", "warm"), ("margin_ownself", "plain"), ("margin_ownself", "ownself"),
                 ("margin_table", "plain"), ("fixedshare", "plain"), ("own_oracle", "plain"), ("ownself", "own_oracle"), ("warm", "same_w3"), ("margin_table", "same_w3")):
        for rg in REGIONS:
            row = paired(per, region_keys(per, rg), a, b)
            body.append([f"{a} minus {b}", rg, r6(row["mean"]), r6(row["lo"]), r6(row["hi"]), row["logs"], row["dropped_cells"]])
    write_csv("g_paired.csv", ["comparison", "region", "mean_diff", "ci_lo", "ci_hi", "logs", "dropped_cells"], body)


def posthoc_causal(per, info):
    """Post hoc: logs whose history windows (w1 and w2) both end before the w3 window starts."""
    causal = []
    for log, v in info.items():
        if not isinstance(v, dict):
            continue
        u, i, t = K._positive_events(log)
        st1 = W.densest_start(t, int(t.min()))
        got2 = K.window_of(t)
        if st1 is not None and got2 is not None and st1 + 156 <= v["start_week"] and got2[1] + 156 <= v["start_week"]:
            causal.append(log)
    rows, gate = [], {}
    for rg in REGIONS:
        ks = [k for k in region_keys(per, rg) if k[0] in causal]
        row = [rg, len(causal), len({k[0] for k in ks})]
        for a, b in (("warm", "plain"), ("ownself", "plain"), ("margin_table", "plain")):
            p = paired(per, ks, a, b)
            row += [r6(p["mean"]), p["logs"]]
            if (a, b) == ("warm", "plain"):
                gate[rg] = bool(p["logs"] >= MIN_LOGS and p["mean"] < 0)
        rows.append(row)
    write_csv("posthoc_causal.csv", ["region", "causal_logs", "logs_in_region", "warm_minus_plain", "logs", "ownself_minus_plain", "logs2", "margin_table_minus_plain", "logs3"], rows)
    return causal, dict(by_region=gate, history_seeded_selector=bool(all(gate.values())))


def show(V):
    for k in sorted((k for k in V if k.startswith("G")), key=lambda s: int(s[1:])):
        v = V[k]
        print(k, {None: "NOT EVALUABLE", True: "MET", False: "NOT MET"}[v["met"]], {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items() if a != "met" and not isinstance(b, dict)},
              {a: {x: round(y, 4) for x, y in b.items()} for a, b in v.items() if isinstance(b, dict)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--rows", default=f"{RUNS}/guard.jsonl")
    a = ap.parse_args()
    if (a.partial or a.rows != f"{RUNS}/guard.jsonl") and not a.check_only:
        sys.exit("--partial and --rows are only allowed together with --check-only (a partial analysis never writes verdicts)")
    if not os.path.exists(a.rows):
        sys.exit(f"{a.rows} is missing: run scripts/ssp/run_guard.py first")
    built, info = RG.build(REG)
    tasks = {t["key"]: t for t in built}
    if not tasks:
        sys.exit("the rebuilt plan is empty")
    check_history_unique()
    rows = load_rows(a.rows)
    check_integrity(rows, tasks, info, a.rows, a.partial)
    if a.check_only:
        return
    c1, c2 = history_cells()
    cells3 = {(r["log"], r["n"], r["round_days"], "std"): view(r) for r in rows.values()}
    per = {(L, n, rd): cell_rules(L, n, rd, c, cells3, c1, c2) for (L, n, rd, _), c in sorted(cells3.items())}
    V = verdicts(per)
    tables(per, info)
    V["causal_logs_posthoc"], V["causal_gate_posthoc"] = posthoc_causal(per, info)
    os.makedirs(OUT, exist_ok=True)
    sha = {p: sha_file(p) for p in HASHED}
    sha["guard.jsonl"] = sha_file(a.rows)
    sha["plan_keys"] = hashlib.sha256(json.dumps(sorted(tasks), sort_keys=True).encode()).hexdigest()
    for p in (f"{RUNS}/weekly.jsonl", f"{RUNS}/confirm.jsonl", f"{RUNS}/guard_plan.json"):
        sha[p] = sha_file(p)
    json.dump(clean(dict(registry=REG["version"], cells=len(rows), sha256=sha, verdicts=V)), open(f"{OUT}/g_verdicts.json", "w"), indent=1)
    show(V)
    print("decisions:", V["decisions"], "| causal logs (post hoc):", len(V["causal_logs_posthoc"]), "| causal gate (>= %d logs and warm < plain):" % MIN_LOGS, V["causal_gate_posthoc"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
