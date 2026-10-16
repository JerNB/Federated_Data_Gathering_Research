"""Analysis of programme P (price of privacy: bounding each person's contribution), mechanical and registered.

    .venv/bin/python3 -W ignore scripts/ssp/an_price.py [--check-only [--partial] [--rows PATH]]

Reads 02_runs/price.jsonl (test cells on the confirmation windows w2) and configs/selfsel_price_registry.json. The plan is rebuilt from the stored
confirmation rows (run_price.build) and the analysis aborts when the loaded cells differ from it, when a key is duplicated or does not match its
content, when the errors file is not empty, when the plan file differs, or when the std arms of a price cell do not reproduce the stored
confirmation cell they copy (exact equality). --partial and --rows are only allowed with --check-only and never write a verdict.
Writes results/small_sample_popularity/03_tables/price/: p_cells.csv, p_summary.csv, p_verdicts.json (with sha256 of the registry, the scripts, the
simulation code and the rows) and the post hoc tables posthoc_*.csv.
Price of a rule = its NDCG@100 with plain random sampling and no noise (the stored std cell) minus its NDCG@100 under the sampler.
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

import an_stress as AS
import explore_selfsel_weekly as E
import run_price as RP
from an_weekly import RUNS, boot

REGISTRY = RP.REG
REG = json.load(open(REGISTRY))
CR = REG["criteria"]
OUT = "results/small_sample_popularity/03_tables/price"
RD = 7
NAN = float("nan")
HASHED = [REGISTRY] + [f"scripts/ssp/{f}" for f in (
    "price_core.py", "run_price.py", "an_price.py", "tune_price.py", "stress_core.py", "an_stress.py", "explore_selfsel_weekly.py", "an_weekly.py",
    "deep_core.py", "track_core.py", "weekly_core.py", "confirm_core.py", "quiet.py")]


def sampler_name(scheme: str) -> str:
    """'gauss:eps=4,c=10' -> 'cap10_e4'; 'gaussn:eps=4,C=1' -> 'clip1_e4'; eps 1000000 -> 'ref'."""
    kind, _, rest = scheme.partition(":")
    par = dict(kv.split("=") for kv in rest.split(","))
    eps = "ref" if float(par["eps"]) >= 1e5 else f"e{int(float(par['eps']))}"
    if kind == "gauss":
        return f"cap{int(float(par['c']))}_{eps}"
    if kind == "gaussn":
        return f"clip{float(par['C']):g}_{eps}"
    raise ValueError(scheme)


SCHEMES = {sampler_name(s): s for s in REG["stage"]["schemes"]}
NAMES = ["std"] + list(SCHEMES)


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
        want = f"p|test|{r['dataset']}|{r['n']}|{r['round_days']}"
        if r["key"] != want:
            sys.exit(f"row key {r['key']} does not match its content ({want})")
        if r["key"] in rows:
            sys.exit(f"duplicate row for {r['key']}")
        r["_nd"] = np.array(r.pop("nd100"), float)
        r["_se"] = np.array(r.pop("selfnd"), float)
        r["_sa"] = None
        r.pop("metrics", None)
        rows[r["key"]] = r
    return rows


def stored_confirm(want: set) -> dict:
    out = {}
    for line in open(f"{RUNS}/confirm.jsonl"):
        r = json.loads(line)
        if not r["dataset"].endswith("@w2") or (r["dataset"], r["n"], r["round_days"]) not in want:
            continue
        if (r["dataset"], r["n"], r["round_days"]) in out:
            sys.exit(f"duplicate stored confirmation cell {(r['dataset'], r['n'], r['round_days'])}")
        out[(r["dataset"], r["n"], r["round_days"])] = (r["arms"], np.array(r["nd100"], float), np.array(r["selfnd"], float))
    return out


def check_integrity(rows: dict, tasks: dict, rows_path: str, partial: bool) -> None:
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
    path = f"{RUNS}/price_plan.json"
    if not os.path.exists(path):
        sys.exit(f"{path} is missing (run run_price.py --plan first)")
    if sorted(json.load(open(path))) != sorted(tasks):
        sys.exit(f"{path} differs from the plan rebuilt from the stored confirmation rows")
    for k, r in rows.items():
        t = tasks[k]
        if r["arms"] != t["arms"] or r["window"] != t["window"] or r["log"] != t["log"] or r["n"] != t["n"]:
            sys.exit(f"{k}: arms, window, log or n differ from the planned task")
        if not (r["_nd"].shape == r["_se"].shape == (t["draws"], len(r["arms"]), len(r["steps"]))):
            sys.exit(f"{k}: nd100, selfnd and steps have inconsistent shapes")
    stored = stored_confirm({(r["dataset"], r["n"], r["round_days"]) for r in rows.values()})
    bad = 0
    for k, r in rows.items():
        hit = stored.get((r["dataset"], r["n"], r["round_days"]))
        if hit is None:
            print(f"no stored confirmation cell for {k}")
            bad += 1
            continue
        arms, nd, se = hit
        for i, a in enumerate(arms):
            j = r["arms"].index(a)
            if not (np.array_equal(r["_nd"][:, j, :], nd[:, i, :], equal_nan=True) and np.array_equal(r["_se"][:, j, :], se[:, i, :], equal_nan=True)):
                print(f"std arm {a} of {k} does not reproduce the stored confirmation cell")
                bad += 1
                break
    if bad:
        sys.exit(f"{bad} cells fail the replication check")
    print(f"integrity: {len(rows)} cells ({'partial' if partial else 'complete'}), std arms reproduce the stored cells exactly")


# --------------------------------------------------------------------------------------------------------------------- values
def build_per(rows: dict) -> dict:
    """per[name][(log, n)] = rule values of the sampler `name`; pooled-table regrets are added within each n."""
    cells = {name: {} for name in NAMES}
    for r in rows.values():
        for name in NAMES:
            suf = "" if name == "std" else SCHEMES[name]
            c = AS.view(r, suf)
            if c is None:
                sys.exit(f"{r['key']} has no arms for sampler {name}")
            cells[name][(r["log"], r["n"], RD, "std")] = c
    per = {name: {(L, n): AS.rule_values(c) for (L, n, rd, _), c in cells[name].items()} for name in NAMES}
    for name in NAMES:
        for n in sorted({k[1] for k in per[name]}):
            sub = {k: v for k, v in cells[name].items() if k[1] == n}
            AS.with_tables({kk: per[name][kk] for kk in per[name] if kk[1] == n}, sub)
    return per


def vals(per, name, n, field="v_sel"):
    return {k[0]: per[name][k][field] for k in per[name] if k[1] == n}


def price(per, name, n, field="v_sel"):
    s = vals(per, "std", n, field)
    return {l: s[l] - v for l, v in vals(per, name, n, field).items() if l in s}


def gain(per, a, b, n, field="v_sel"):
    """{log: value under a minus value under b}: positive means a is better."""
    va, vb = vals(per, a, n, field), vals(per, b, n, field)
    return {l: va[l] - vb[l] for l in va if l in vb}


def stat(d: dict) -> dict:
    v = list(d.values())
    m, lo, hi = boot(v)
    return dict(mean=float(m), lo=float(lo), hi=float(hi), logs=len(v), share_positive=float(np.mean([x > 0 for x in v])) if v else NAN)


def sel(flag, *stats):
    """None (not evaluable) when any comparison has fewer than min_logs logs."""
    if any(s["logs"] < CR["min_logs"] for s in stats):
        return None
    return bool(flag)


def verdicts(per: dict) -> dict:
    V = {}
    for pid in ("P1", "P2", "P3", "P4"):
        c = CR[pid]
        s = stat(gain(per, f"clip1_e{c['eps']}", f"cap10_e{c['eps']}", c["n"]))
        V[pid] = dict(s, eps=c["eps"], n=c["n"], met=sel(s["mean"] >= c["min_gain"] and s["lo"] > 0, s))
    c = CR["P5"]
    s = stat(gain(per, f"clip1_e{c['eps']}", f"clip2_e{c['eps']}", c["n"]))
    V["P5"] = dict(s, met=sel(s["mean"] >= c["min_gain"] and s["lo"] > 0, s))
    c = CR["P6"]
    p_hi, p_lo = stat(price(per, f"clip1_e{c['strong']['eps']}", c["strong"]["n"])), stat(price(per, f"clip1_e{c['weak']['eps']}", c["weak"]["n"]))
    V["P6"] = dict(price_strong=p_hi["mean"], price_weak=p_lo["mean"], logs=min(p_hi["logs"], p_lo["logs"]),
                   met=sel(p_hi["mean"] >= c["strong"]["min_price"] and p_lo["mean"] <= c["weak"]["max_price"], p_hi, p_lo))
    c = CR["P7"]
    nm = f"clip1_e{c['eps']}"
    k = [kk for kk in per[nm] if kk[1] == c["n"]]
    d, row = AS.paired(per[nm], k, "selector", "pooled")
    mr = AS.log_level(per[nm], k, "selector")
    row.update(mean_selector_regret_over_logs=mr, mean_pooled_table_regret=AS.mean_of(per[nm], k, "pooled"))
    row["met"] = sel(np.isfinite(mr) and mr <= c["max_mean_regret"] and row["mean"] <= c["max_diff_to_pooled"] and row["hi"] < 0, row)
    V["P7"] = row
    c = CR["P8"]
    s = stat(price(per, "clip1_ref", c["n"]))
    V["P8"] = dict(s, met=sel(s["mean"] <= c["max_clip_loss"], s))
    c = CR["P9"]
    s = stat(gain(per, f"clip1_e{c['eps']}", f"cap{c['cap']}_e{c['eps']}", c["n"]))
    V["P9"] = dict(s, met=sel(s["mean"] >= c["min_gain"] and s["lo"] > 0, s))
    c = CR["P10"]
    ss = {e: stat(gain(per, f"clip1_e{e}", f"cap{c['cap']}_e{e}", c["n"])) for e in c["eps"]}
    V["P10"] = dict(by_eps={str(e): s for e, s in ss.items()}, logs=min(s["logs"] for s in ss.values()),
                    met=sel(all(s["mean"] >= c["min_gain"] and s["lo"] > 0 for s in ss.values()), *ss.values()))
    ok = lambda ids: (None if any(V[i]["met"] is None for i in ids) else bool(all(V[i]["met"] for i in ids)))
    V["decisions"] = {name: ok(ids) for name, ids in CR["decisions"].items()}
    return V


# ------------------------------------------------------------------------------------------------------------------ outputs
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
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


FIELDS = ["best", "best_days", "top_days", "v_two", "v_sel", "selector", "two", "pooled", "same"]


def tables(per: dict) -> None:
    write_csv("p_cells.csv", ["sampler", "log", "n", "group", "price_selector", "price_two"] + FIELDS,
              [[name, L, n, E.stratum(L), r6(price(per, name, n)[L]) if name != "std" else 0.0, r6(price(per, name, n, "v_two")[L]) if name != "std" else 0.0]
               + [r6(rec.get(f)) for f in FIELDS] for name in NAMES for (L, n), rec in sorted(per[name].items())])
    body = []
    for name in NAMES:
        for n in sorted({k[1] for k in per[name]}):
            ps = price(per, name, n)
            p2 = price(per, name, n, "v_two")
            k = [kk for kk in per[name] if kk[1] == n]
            body.append([name, n, len(k), r6(np.mean([per[name][kk]["v_sel"] for kk in k])), r6(np.mean([per[name][kk]["best"] for kk in k])),
                         r6(np.mean([per[name][kk]["selector"] for kk in k])), r6(np.mean(list(ps.values()))), r6(np.mean(list(p2.values()))),
                         r6(AS.mean_of(per[name], k, "pooled")), r6(AS.mean_of(per[name], k, "same"))])
    write_csv("p_summary.csv", ["sampler", "n", "cells", "selector_value", "best_fixed_value", "selector_regret", "price_selector", "price_two_rounds",
                                "pooled_table_regret", "same_group_table_regret"], body)


def posthoc(per: dict) -> None:
    rows = []
    for n in sorted({k[1] for k in per["std"]}):
        for eps in sorted({int(s.split("_e")[1]) for s in SCHEMES if "_e" in s}):
            for a, b in (("clip1", "cap10"), ("clip2", "cap10"), ("clip1", "clip2"), ("clip1", "cap1")):
                na, nb = f"{a}_e{eps}", f"{b}_e{eps}"
                if na not in per or nb not in per:
                    continue
                d = gain(per, na, nb, n)
                for scope in ("all", "amazon", "other"):
                    dd = {l: v for l, v in d.items() if scope == "all" or (E.stratum(l) == "amazon") == (scope == "amazon")}
                    if len(dd) >= 3:
                        s = stat(dd)
                        rows.append([n, eps, f"{a} minus {b}", scope, s["logs"], r6(s["mean"]), r6(s["lo"]), r6(s["hi"]), r6(s["share_positive"])])
    write_csv("posthoc_gain_by_scope.csv", ["n", "eps", "comparison", "scope", "logs", "mean_gain_in_selector_value", "ci_lo", "ci_hi", "share_logs_positive"], rows)
    rows = []
    for n in sorted({k[1] for k in per["std"]}):
        for name in ("cap10_ref", "clip1_ref", "clip2_ref", "cap1_ref"):
            if name in per:
                s = stat(price(per, name, n))
                rows.append([n, name, s["logs"], r6(s["mean"]), r6(s["lo"]), r6(s["hi"])])
    write_csv("posthoc_bound_loss_without_noise.csv", ["n", "sampler", "logs", "loss_vs_std_selector", "ci_lo", "ci_hi"], rows)


def show(V):
    for k in sorted((k for k in V if k.startswith("P")), key=lambda s: int(s[1:])):
        v = V[k]
        print(k, {None: "NOT EVALUABLE", True: "MET", False: "NOT MET"}[v["met"]], {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items() if a != "met"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--rows", default=f"{RUNS}/price.jsonl")
    a = ap.parse_args()
    if (a.partial or a.rows != f"{RUNS}/price.jsonl") and not a.check_only:
        sys.exit("--partial and --rows are only allowed together with --check-only (a partial analysis never writes verdicts)")
    if not os.path.exists(a.rows):
        sys.exit(f"{a.rows} is missing: run scripts/ssp/run_price.py first")
    tasks = {t["key"]: t for t in RP.build(REG)}
    if not tasks:
        sys.exit("the rebuilt plan is empty")
    rows = load_rows(a.rows)
    check_integrity(rows, tasks, a.rows, a.partial)
    if a.check_only:
        return
    per = build_per(rows)
    V = verdicts(per)
    tables(per)
    posthoc(per)
    os.makedirs(OUT, exist_ok=True)
    sha = {p: sha_file(p) for p in HASHED}
    sha["price.jsonl"] = sha_file(a.rows)
    sha["plan_keys"] = hashlib.sha256(json.dumps(sorted(tasks), sort_keys=True).encode()).hexdigest()
    for p in (f"{RUNS}/confirm.jsonl", f"{RUNS}/price_plan.json"):
        sha[p] = sha_file(p)
    json.dump(clean(dict(registry=REG["version"], cells=len(rows), sha256=sha, verdicts=V)), open(f"{OUT}/p_verdicts.json", "w"), indent=1)
    show(V)
    print("decisions:", V["decisions"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
