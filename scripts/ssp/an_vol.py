"""Analysis of programme V (volunteer-bias weights chosen on an audit sample), mechanical and registered.

    .venv/bin/python3 -W ignore scripts/ssp/an_vol.py [--check-only [--partial] [--rows PATH]]

Reads 02_runs/vol.jsonl (test cells on the third window w3, 7-day rounds) and configs/selfsel_vol_registry.json. The plan is rebuilt from the stored
window-3 rows of programme G (run_vol.build) and the analysis aborts when the loaded cells differ from it, when a key is duplicated or does not match its
content, when the errors file is not empty, when the plan file differs, when an array has an inconsistent shape, or when the std arms of a cell do not
reproduce the stored window-3 cell they copy (exact equality). --partial and --rows are only allowed with --check-only and never write a verdict.
Writes results/small_sample_popularity/03_tables/volunteer/: v_cells.csv, v_summary.csv, v_verdicts.json (with sha256 of the registry, the scripts, the
simulation code and the rows) and the post hoc tables posthoc_*.csv.

Rules, per cell and sampler X (std = random people, A+ = same-week heavy volunteers, A- = light volunteers), values = mean true NDCG@100 over steady steps
and draws; reports are weighted by rows^(-beta):
  rep0   the selector of docs 14-15 on the reporters' self-score, beta = 0 (the uncorrected baseline)
  aud0   the same selector on the audit self-score, beta = 0
  audw   the selector over ALL (beta, days) arms on the audit self-score (the proposal)
  best0  hindsight best fixed arm at beta = 0;  bestw  hindsight best fixed arm over all (beta, days)
  ref    rep0 under random people (std): the value a correct estimate would give
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
import run_vol as RV
import tune_weights as TW
from an_weekly import RUNS, boot

REGISTRY = RV.REG
REG = json.load(open(REGISTRY))
CR = REG["criteria"]
ST = REG["stage"]
OUT = "results/small_sample_popularity/03_tables/volunteer"
NAN = float("nan")
SAMPLERS = list(ST["samplers"])
BETAS = [float(b) for b in ST["betas"]]
HASHED = [REGISTRY] + [f"scripts/ssp/{f}" for f in (
    "weight_core.py", "run_vol.py", "an_vol.py", "tune_weights.py", "window3_core.py", "run_guard.py", "explore_selfsel_weekly.py", "an_weekly.py",
    "deep_core.py", "track_core.py", "weekly_core.py", "confirm_core.py", "quiet.py")]


def scheme_of(sampler: str, b: float) -> str:
    return "" if (sampler == "std" and b == 0.0) else RV.scheme(ST["samplers"][sampler], b)


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
        want = f"v|test|{r['dataset']}|{r['n']}|{r['round_days']}"
        if r["key"] != want:
            sys.exit(f"row key {r['key']} does not match its content ({want})")
        if r["key"] in rows:
            sys.exit(f"duplicate row for {r['key']}")
        for f in ("nd100", "selfnd", "selfaud"):
            if f not in r:
                sys.exit(f"{r['key']} has no {f}")
            r[f] = np.array(r[f], float)
        r.pop("metrics", None)
        rows[r["key"]] = r
    return rows


def stored_guard(want: set) -> dict:
    out = {}
    for line in open(f"{RUNS}/guard.jsonl"):
        r = json.loads(line)
        k = (r["dataset"], r["n"], r["round_days"])
        if k not in want:
            continue
        if k in out:
            sys.exit(f"duplicate stored window-3 cell {k}")
        out[k] = (r["arms"], np.array(r["nd100"], float), np.array(r["selfnd"], float))
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
    path = f"{RUNS}/vol_plan.json"
    if not os.path.exists(path):
        sys.exit(f"{path} is missing (run run_vol.py --plan first)")
    if sorted(json.load(open(path))) != sorted(tasks):
        sys.exit(f"{path} differs from the plan rebuilt from the stored window-3 rows")
    for k, r in rows.items():
        t = tasks[k]
        if r["arms"] != t["arms"] or r["window"] != t["window"] or r["log"] != t["log"] or r["n"] != t["n"] or r.get("audit_n") != t["audit_n"]:
            sys.exit(f"{k}: arms, window, log, n or audit_n differ from the planned task")
        shape = (t["draws"], len(r["arms"]), len(r["steps"]))
        if not (r["nd100"].shape == r["selfnd"].shape == r["selfaud"].shape == shape):
            sys.exit(f"{k}: nd100, selfnd, selfaud and steps have inconsistent shapes")
    stored = stored_guard({(r["dataset"], r["n"], r["round_days"]) for r in rows.values()})
    bad = 0
    for k, r in rows.items():
        hit = stored.get((r["dataset"], r["n"], r["round_days"]))
        if hit is None:
            print(f"no stored window-3 cell for {k}")
            bad += 1
            continue
        arms, nd, se = hit
        for i, a in enumerate(arms):
            j = r["arms"].index(a)
            if not (np.array_equal(r["nd100"][:, j, :], nd[:, i, :], equal_nan=True) and np.array_equal(r["selfnd"][:, j, :], se[:, i, :], equal_nan=True)):
                print(f"std arm {a} of {k} does not reproduce the stored window-3 cell")
                bad += 1
                break
    if bad:
        sys.exit(f"{bad} cells fail the replication check")
    print(f"integrity: {len(rows)} cells ({'partial' if partial else 'complete'}), std arms reproduce the stored cells exactly")


# --------------------------------------------------------------------------------------------------------------------- values
def cell_values(r: dict) -> dict:
    out = TW.rules_of(r, SAMPLERS, BETAS, scheme_of)
    rec = {}
    ref = out[("std", 0.0)]["rep"]
    for s in SAMPLERS:
        rec[s] = dict(rep0=out[(s, 0.0)]["rep"], aud0=out[(s, 0.0)]["aud"], audw=out[(s, "all")]["aud"], best0=out[(s, 0.0)]["best"], bestw=out[(s, "all")]["best"], ref=ref,
                      betas={f"{b:g}": dict(best=out[(s, b)]["best"], aud=out[(s, b)]["aud"], rep=out[(s, b)]["rep"]) for b in BETAS})
    return rec


def build_per(rows: dict) -> dict:
    """per[(log, n)][sampler] = values; cells are taken in sorted (log, n) order so that the bootstrap does not depend on the order in which the cells finished."""
    return {(r["log"], r["n"]): cell_values(r) for r in sorted(rows.values(), key=lambda r: (r["log"], r["n"]))}


def log_means(per: dict, fn, n_list=None) -> dict:
    """{log: mean over the log's cells (n in n_list) of fn(cell)}; the log is the unit."""
    d = {}
    for (L, n), rec in per.items():
        if n_list is None or n in n_list:
            v = fn(rec)
            if np.isfinite(v):
                d.setdefault(L, []).append(v)
    return {l: float(np.mean(v)) for l, v in d.items()}


def stat(d: dict) -> dict:
    v = list(d.values())
    m, lo, hi = boot(v)
    return dict(mean=float(m), lo=float(lo), hi=float(hi), logs=len(v), share_positive=float(np.mean([x > 0 for x in v])) if v else NAN)


def sel(flag, *stats):
    if any(s["logs"] < CR["min_logs"] for s in stats):
        return None
    return bool(flag)


def verdicts(per: dict) -> dict:
    V = {}
    nl = CR["n_hyp"]
    best_gain = lambda smp: stat(log_means(per, lambda rec: rec[smp]["bestw"] - rec[smp]["best0"], nl))     # hindsight: best (beta, days) arm minus best days arm at beta = 0
    aud_gain = lambda smp: stat(log_means(per, lambda rec: rec[smp]["audw"] - rec[smp]["rep0"], nl))        # audit-weighted selector minus the uncorrected reporter selector
    aud_ref = lambda smp: stat(log_means(per, lambda rec: rec[smp]["audw"] - rec[smp]["ref"], nl))          # audit-weighted selector minus the uncorrected selector under random people
    c = CR["V1"]
    s = best_gain(c["sampler"])
    V["V1"] = dict(s, sampler=c["sampler"], met=sel(s["mean"] >= c["min_gain"] and s["lo"] > 0, s))
    for pid in ("V2", "V3"):                                    # registered nulls
        c = CR[pid]
        s = best_gain(c["sampler"])
        V[pid] = dict(s, sampler=c["sampler"], met=sel(s["mean"] < c["max_gain"], s))
    for pid in ("V4", "V5"):
        c = CR[pid]
        s = aud_gain(c["sampler"])
        V[pid] = dict(s, sampler=c["sampler"], met=sel(s["mean"] >= c["min_gain"] and s["lo"] > 0, s))
    c = CR["V6"]
    s = aud_gain(c["sampler"])
    V["V6"] = dict(s, sampler=c["sampler"], met=sel(s["lo"] > c["ci_lower_above"], s))
    c = CR["V7"]
    s = aud_ref(c["sampler"])
    V["V7"] = dict(s, sampler=c["sampler"], met=sel(s["lo"] > c["ci_lower_above"], s))
    c = CR["V8"]
    s = aud_ref(c["sampler"])
    V["V8"] = dict(s, sampler=c["sampler"], met=sel(s["mean"] <= c["max_mean_diff"], s))
    c = CR["V9"]
    ss = {smp: stat(log_means(per, lambda rec, smp=smp: rec[smp]["bestw"] - rec[smp]["audw"], nl)) for smp in SAMPLERS}
    V["V9"] = dict(by_sampler={k: v["mean"] for k, v in ss.items()}, logs=min(s["logs"] for s in ss.values()),
                   met=sel(all(s["mean"] <= c["max_gap"] for s in ss.values()), *ss.values()))
    c = CR["V10"]                                               # what the weights add to the audit selector without weights (aud0)
    s = stat(log_means(per, lambda rec: rec[c["sampler"]]["audw"] - rec[c["sampler"]]["aud0"], nl))
    V["V10"] = dict(s, sampler=c["sampler"], met=sel(s["mean"] >= c["min_gain"] and s["lo"] > 0, s))
    ok_ = lambda ids: (None if any(V[i]["met"] is None for i in ids) else bool(all(V[i]["met"] for i in ids)))
    V["decisions"] = {name: ok_(ids) for name, ids in CR["decisions"].items()}
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


FIELDS = ["rep0", "aud0", "audw", "best0", "bestw", "ref"]


def tables(per: dict) -> None:
    write_csv("v_cells.csv", ["log", "n", "group", "sampler"] + FIELDS,
              [[L, n, E.stratum(L), s] + [r6(rec[s][f]) for f in FIELDS] for (L, n), rec in sorted(per.items()) for s in SAMPLERS])
    body = []
    for n in sorted({k[1] for k in per}):
        for s in SAMPLERS:
            cells = [rec[s] for k, rec in per.items() if k[1] == n]
            body.append([n, s, len(cells)] + [r6(float(np.mean([c[f] for c in cells]))) for f in FIELDS])
    write_csv("v_summary.csv", ["n", "sampler", "cells"] + FIELDS, body)


def posthoc(per: dict) -> None:
    rows = []
    for n in sorted({k[1] for k in per}):
        for s in SAMPLERS:
            for b in BETAS:
                v = [rec[s]["betas"][f"{b:g}"] for k, rec in per.items() if k[1] == n]
                rows.append([n, s, b, len(v), r6(float(np.mean([x["best"] for x in v]))), r6(float(np.mean([x["aud"] for x in v]))), r6(float(np.mean([x["rep"] for x in v])))])
    write_csv("posthoc_by_beta.csv", ["n", "sampler", "beta", "cells", "best_fixed_value", "audit_selector_value", "reporter_selector_value"], rows)
    rows = []
    for scope in ("all", "amazon", "other"):
        for s in SAMPLERS:
            for name, fn in (("audw minus rep0", lambda rec, s=s: rec[s]["audw"] - rec[s]["rep0"]), ("bestw minus best0", lambda rec, s=s: rec[s]["bestw"] - rec[s]["best0"]),
                             ("audw minus aud0", lambda rec, s=s: rec[s]["audw"] - rec[s]["aud0"])):
                d = log_means({k: v for k, v in per.items() if scope == "all" or (E.stratum(k[0]) == "amazon") == (scope == "amazon")}, fn, CR["n_hyp"])
                if len(d) >= 3:
                    st_ = stat(d)
                    rows.append([scope, s, name, st_["logs"], r6(st_["mean"]), r6(st_["lo"]), r6(st_["hi"])])
    write_csv("posthoc_gain_by_scope.csv", ["scope", "sampler", "comparison", "logs", "mean", "ci_lo", "ci_hi"], rows)
    rows = []
    for n in sorted({k[1] for k in per}):
        for s in SAMPLERS:
            for name, fn in (("audw minus rep0", lambda rec, s=s: rec[s]["audw"] - rec[s]["rep0"]), ("bestw minus best0", lambda rec, s=s: rec[s]["bestw"] - rec[s]["best0"]),
                             ("ref minus rep0 (loss from the volunteer rule)", lambda rec, s=s: rec[s]["ref"] - rec[s]["rep0"]),
                             ("audw minus ref", lambda rec, s=s: rec[s]["audw"] - rec[s]["ref"])):
                d = log_means(per, fn, [n])
                if len(d) >= 3:
                    st_ = stat(d)
                    rows.append([n, s, name, st_["logs"], r6(st_["mean"]), r6(st_["lo"]), r6(st_["hi"])])
    write_csv("posthoc_gain_by_n.csv", ["n", "sampler", "comparison", "logs", "mean", "ci_lo", "ci_hi"], rows)


def show(V):
    for k in sorted((k for k in V if k.startswith("V")), key=lambda s: int(s[1:])):
        v = V[k]
        print(k, {None: "NOT EVALUABLE", True: "MET", False: "NOT MET"}[v["met"]], {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items() if a != "met"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--rows", default=f"{RUNS}/vol.jsonl")
    a = ap.parse_args()
    if (a.partial or a.rows != f"{RUNS}/vol.jsonl") and not a.check_only:
        sys.exit("--partial and --rows are only allowed together with --check-only (a partial analysis never writes verdicts)")
    if not os.path.exists(a.rows):
        sys.exit(f"{a.rows} is missing: run scripts/ssp/run_vol.py first")
    tasks = {t["key"]: t for t in RV.build(REG)}
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
    sha["vol.jsonl"] = sha_file(a.rows)
    sha["plan_keys"] = hashlib.sha256(json.dumps(sorted(tasks), sort_keys=True).encode()).hexdigest()
    for p in (f"{RUNS}/guard.jsonl", f"{RUNS}/vol_plan.json"):
        sha[p] = sha_file(p)
    json.dump(clean(dict(registry=REG["version"], cells=len(rows), sha256=sha, verdicts=V)), open(f"{OUT}/v_verdicts.json", "w"), indent=1)
    show(V)
    print("decisions:", V["decisions"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
