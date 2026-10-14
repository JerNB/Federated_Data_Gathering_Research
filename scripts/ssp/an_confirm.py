#!/usr/bin/env python3
"""Confirmatory claims C1, C2, C3 of the Small-Sample Popularity pre-registration (section 6).

Usage:
    python3 scripts/ssp/an_confirm.py --runs <dir with e1.jsonl, e2.jsonl> --out <out dir>

Reads  <runs>/e1.jsonl (popularity estimators) and <runs>/e2.jsonl (model training).
Writes <out>/03_tables/an_confirm_*.csv, <out>/03_tables/c_summary.csv,
       <out>/03_tables/an_confirm_verdicts.json and <out>/04_figures/F-C*.png.

Rules implemented literally (pre-registration section 6 and 8):

C1  count NDCG@100 vs Truth A.  Per (dataset, budget B = N*k, k numeric) with >= 2 cells:
    difference = widest cell (largest N) minus deepest cell (largest k).  Each cell's draws are
    resampled separately (2000 resamples).  Supported: >= 75% of pairs positive AND pooled interval
    excludes 0 (above).  Falsified: <= 50% of pairs positive OR pooled interval entirely below 0.
    Pooled estimate = mean over datasets of the mean pair difference (datasets weigh equally).
    Pooled interval = two-level percentile bootstrap: pairs are resampled within each dataset and
    each chosen pair's difference is itself a draw-level bootstrap replicate (this is the wider of
    the two intervals reported; the draws-only interval is reported next to it).
C2  panel NDCG@100.  Per (dataset, N, k) cell: mean over draws of hyb_time - max(pop, pers_time).
    (a) share of cells with mean >= -0.002, (b) share of cells whose paired 95% interval over draws
    is above 0, (c) share of cells with mean(hyb_random.w) >= mean(hyb_time.w).
    Supported: a >= 90%, b >= 20%, c >= 70%.  Falsified: a < 75% or b < 10%.
C3  N*(dataset, k) = smallest N whose paired-by-draw 95% interval of pers_time - pop is above 0.
    Supported: N* exists for k = 20 and for k = all in >= 5 of the 7 model-stage datasets and N* never
    increases with k.  Falsified: N* absent at k = all in >= 4 of 7 datasets.

Shares are judged against the number of EXPECTED units (cells/pairs the run grid should contain, from
datasets.json when found).  A rule is only decided when the available data already fixes the outcome
whatever the missing units would show; otherwise the verdict is 'not testable' and the numbers carry
an 'indicative_verdict' computed on the available units.  Thresholds are constants below and never tuned.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import metrics  # noqa: E402
except ImportError:  # worktree without the helper: use the main checkout
    sys.path.insert(0, "/Users/LeoQin/Documents/GitHub/Federated_Data_Gathering_Research/scripts/ssp")
    import metrics  # noqa: E402

REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
DEFAULT_RUNS = os.path.join(REPO, "results", "small_sample_popularity", "02_runs")
DEFAULT_OUT = os.path.join(REPO, "results", "small_sample_popularity")

# ---------------------------------------------------------------- fixed design (pre-registration)
E1_DATASETS = ["ml-latest", "ml-1m", "amazon-video-games", "amazon-musical-instruments", "amazon-software",
               "amazon-movies-tv", "amazon-books", "gowalla", "brightkite", "kuairand-pure", "lastfm-1k"]
E2_DATASETS = ["ml-latest", "ml-1m", "amazon-video-games", "amazon-musical-instruments",
               "amazon-software", "kuairand-pure", "gowalla"]
N_GRID = [20, 40, 100, 200, 400, 1000, 2000, 4000, 10000, 20000]
K_GRID = [1, 2, 5, 10, 20, None]
E2_N = [100, 200, 400, 1000, 2000, 4000]
E2_K = [5, 10, 20, None]
N_BOOT = 2000
SEED = 20261014
MIN_DRAWS = 5            # a cell with fewer valid draws is treated as missing (cannot carry an interval)
C1_SHARE_SUPPORT, C1_SHARE_FALSIFY = 0.75, 0.50
C2_TOL = 0.002
C2_A_SUPPORT, C2_A_FALSIFY = 0.90, 0.75
C2_B_SUPPORT, C2_B_FALSIFY = 0.20, 0.10
C2_C_SUPPORT = 0.70
C3_MIN_DATASETS = 5
C3_FALSIFY_ABSENT = 4
EPS = 1e-12

C1_METRICS = {"A": "A_ndcg100", "B": "B_ndcg100", "U": "U_ndcg"}

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


# ---------------------------------------------------------------- generic helpers
def kname(k):
    return "all" if k is None else str(int(k))


def korder(k):
    return math.inf if k is None else k


def to_float(v):
    if v is None:
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def read_jsonl(path):
    """Rows of a jsonl file; None if the file is missing. A truncated last line (file still growing) is skipped."""
    if not os.path.isfile(path):
        return None
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_meta(runs, out):
    for base in (os.path.join(runs, "..", "01_datasets"), os.path.join(out, "01_datasets"),
                 os.path.join(DEFAULT_OUT, "01_datasets")):
        p = os.path.join(base, "datasets.json")
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as fh:
                return {r["name"]: r for r in json.load(fh)}
    return None


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in rows:
            out = []
            for f in fields:
                v = r.get(f, "")
                if isinstance(v, (float, np.floating)):
                    v = "" if math.isnan(float(v)) else f"{float(v):.6g}"
                out.append(v)
            w.writerow(out)


def boot_means(x, n_boot, rng):
    """n_boot bootstrap means of a 1-D sample (resampling its draws)."""
    n = x.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    return x[idx].mean(axis=1)


def ci95(reps):
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(lo), float(hi)


def pct(x):
    return f"{100.0 * x:.0f}%"


def decide(sup, fals, complete):
    if sup:
        return "supported"
    if fals:
        return "not supported"
    return "mixed" if complete else "not testable"


def share_bounds(n_true, n_present, n_missing):
    """Lowest and highest share the claim could end with once the missing units are in."""
    tot = n_present + n_missing
    if tot == 0:
        return math.nan, math.nan, math.nan
    return n_true / tot, (n_true + n_missing) / tot, (n_true / n_present if n_present else math.nan)


def expected_cells(meta, ds, ns, ks, cap=0.4):
    """Expected (N, k) cells of a dataset (N capped at 40% of the group, k capped at min_rows)."""
    if meta is None or ds not in meta:
        return None
    m = meta[ds]
    return [(n, k) for n in ns if n <= cap * m["n_pop"] for k in ks if k is None or k <= m["min_rows"]]


# ---------------------------------------------------------------- loading
def e1_cells(rows):
    """{(dataset, N, k): {draw: row}} for the main grid (arm uniform, uniform users, random rows), k numeric."""
    cells = defaultdict(dict)
    for r in rows:
        key = r.get("key", "") or ""
        arm = r.get("arm")
        if arm is None:
            arm = "uniform"
            for part in key.split("|"):
                if part.startswith("arm="):
                    arm = part[4:]
        if arm != "uniform" or r.get("users", "uniform") != "uniform" or r.get("rows", "random") != "random":
            continue
        if r.get("k") is None or r.get("dataset") not in E1_DATASETS:
            continue
        cells[(r["dataset"], int(r["N"]), int(r["k"]))][int(r["draw"])] = r   # a repeated key: last row wins
    return cells



def e2_cells(rows):
    """{(dataset, N, k): {draw: row}}; k None = all."""
    cells = defaultdict(dict)
    for r in rows:
        if r.get("dataset") not in E2_DATASETS:
            continue
        k = None if r.get("k") is None else int(r["k"])
        cells[(r["dataset"], int(r["N"]), k)][int(r["draw"])] = r
    return cells


def col(cell, field):
    return np.array([to_float(r.get(field)) for _, r in sorted(cell.items())], dtype=float)


def cols_valid(cell, fields):
    """Columns restricted to draws where every listed field is finite."""
    arrs = [col(cell, f) for f in fields]
    ok = np.ones(arrs[0].shape[0], dtype=bool)
    for a in arrs:
        ok &= np.isfinite(a)
    return [a[ok] for a in arrs]


# ================================================================ C1
def pool_two_level(D_by_ds, points_by_ds, n_boot, rng):
    """Pooled mean of per-dataset mean pair differences with two intervals (two-level, draws-only)."""
    two = np.zeros(n_boot)
    drw = np.zeros(n_boot)
    for ds, D in D_by_ds.items():
        m = D.shape[0]
        pi = rng.integers(0, m, size=(n_boot, m))
        ri = rng.integers(0, n_boot, size=(n_boot, m))
        two += D[pi, ri].mean(axis=1)
        drw += D.mean(axis=0)
    nd = len(D_by_ds)
    two /= nd
    drw /= nd
    point = float(np.mean([np.mean(points_by_ds[ds]) for ds in D_by_ds]))
    return point, ci95(two), ci95(drw)


def run_c1(e1, meta, rng):
    cells = e1_cells(e1)
    budgets = defaultdict(list)
    for (ds, n, k), d in cells.items():
        if len(d) >= MIN_DRAWS:
            budgets[(ds, n * k)].append((n, k))
    pair_rows = []
    D = {m: defaultdict(list) for m in C1_METRICS}
    P = {m: defaultdict(list) for m in C1_METRICS}
    for (ds, b), lst in sorted(budgets.items(), key=lambda t: (E1_DATASETS.index(t[0][0]), t[0][1])):
        if len(lst) < 2:
            continue
        wide = max(lst)
        deep = max(lst, key=lambda t: t[1])
        row = dict(dataset=ds, budget=b, n_cells=len(lst), N_wide=wide[0], k_wide=wide[1],
                   N_deep=deep[0], k_deep=deep[1])
        cw, cd = cells[(ds, *wide)], cells[(ds, *deep)]
        row["draws_wide"], row["draws_deep"] = len(cw), len(cd)
        for tag, met in C1_METRICS.items():
            xw = col(cw, f"count.{met}")
            xw = xw[np.isfinite(xw)]
            xd = col(cd, f"count.{met}")
            xd = xd[np.isfinite(xd)]
            if xw.size < MIN_DRAWS or xd.size < MIN_DRAWS:
                row[f"{tag}_wide"] = row[f"{tag}_deep"] = row[f"{tag}_diff"] = math.nan
                row[f"{tag}_lo"] = row[f"{tag}_hi"] = math.nan
                continue
            reps = boot_means(xw, N_BOOT, rng) - boot_means(xd, N_BOOT, rng)
            lo, hi = ci95(reps)
            diff = float(xw.mean() - xd.mean())
            row.update({f"{tag}_wide": float(xw.mean()), f"{tag}_deep": float(xd.mean()),
                        f"{tag}_diff": diff, f"{tag}_lo": lo, f"{tag}_hi": hi})
            D[tag][ds].append(reps)
            P[tag][ds].append(diff)
        row["positive"] = int(row["A_diff"] > 0) if not math.isnan(row["A_diff"]) else ""
        pair_rows.append(row)

    # expected pairs per dataset from the run grid
    exp_pairs = {}
    for ds in E1_DATASETS:
        ec = expected_cells(meta, ds, N_GRID, [k for k in K_GRID if k is not None])
        if ec is None:
            continue
        cnt = defaultdict(int)
        for n, k in ec:
            cnt[n * k] += 1
        exp_pairs[ds] = sum(1 for v in cnt.values() if v >= 2)
    valid = [r for r in pair_rows if not math.isnan(r["A_diff"])]
    present_by_ds = defaultdict(int)
    pos_by_ds = defaultdict(int)
    for r in valid:
        present_by_ds[r["dataset"]] += 1
        pos_by_ds[r["dataset"]] += int(r["A_diff"] > 0)
    missing_pairs = sum(max(0, exp_pairs.get(ds, 0) - present_by_ds.get(ds, 0)) for ds in E1_DATASETS)
    ds_missing = [ds for ds in E1_DATASETS if present_by_ds.get(ds, 0) == 0]
    unknown_open = meta is None and bool(ds_missing)
    n_pairs, n_pos = len(valid), sum(pos_by_ds.values())

    ds_rows = []
    for ds in E1_DATASETS:
        if present_by_ds.get(ds, 0) == 0:
            continue
        dd = [r["A_diff"] for r in valid if r["dataset"] == ds]
        ds_rows.append(dict(dataset=ds, n_pairs=present_by_ds[ds], n_expected_pairs=exp_pairs.get(ds, ""),
                            n_positive=pos_by_ds[ds], share_positive=pos_by_ds[ds] / present_by_ds[ds],
                            mean_diff_A=float(np.mean(dd))))

    res = dict(pairs=pair_rows, datasets=ds_rows, cells=cells)
    numbers = dict(n_pairs=n_pairs, n_positive=n_pos, n_expected_pairs=sum(exp_pairs.values()) if exp_pairs else None,
                   n_missing_pairs=missing_pairs, datasets_present=sorted(present_by_ds, key=E1_DATASETS.index),
                   datasets_missing=ds_missing)
    if n_pairs == 0:
        res["verdict"] = "not testable"
        res["numbers"] = numbers
        res["why"] = "no (dataset, budget) pair with two usable cells in e1.jsonl"
        return res
    for tag in C1_METRICS:
        if D[tag]:
            pt, (lo2, hi2), (lod, hid) = pool_two_level({ds: np.array(v) for ds, v in D[tag].items()},
                                                        P[tag], N_BOOT, rng)
            numbers[f"pooled_{tag}"] = dict(mean=pt, lo=lo2, hi=hi2, lo_draws_only=lod, hi_draws_only=hid,
                                            n_pairs=sum(len(v) for v in D[tag].values()),
                                            share_positive=float(np.mean([r[f"{tag}_diff"] > 0 for r in pair_rows
                                                                          if not math.isnan(r[f"{tag}_diff"])])))
    pa = numbers["pooled_A"]
    lo_b, hi_b, share = share_bounds(n_pos, n_pairs, missing_pairs)
    numbers["share_positive"] = share
    numbers["share_positive_if_missing_all_negative"] = lo_b
    numbers["share_positive_if_missing_all_positive"] = hi_b
    sup = lo_b >= C1_SHARE_SUPPORT and pa["lo"] > 0
    fals = hi_b <= C1_SHARE_FALSIFY or pa["hi"] < 0
    complete = missing_pairs == 0 and not unknown_open
    numbers["prediction_share_met"] = bool(lo_b >= C1_SHARE_SUPPORT)
    numbers["prediction_interval_excludes_zero"] = bool(pa["lo"] > 0)
    numbers["falsifier_share"] = bool(hi_b <= C1_SHARE_FALSIFY)
    numbers["falsifier_interval_below_zero"] = bool(pa["hi"] < 0)
    numbers["indicative_verdict"] = decide(share >= C1_SHARE_SUPPORT and pa["lo"] > 0,
                                           share <= C1_SHARE_FALSIFY or pa["hi"] < 0, True)
    numbers["per_dataset_share_positive"] = {r["dataset"]: r["share_positive"] for r in ds_rows}
    res["verdict"] = decide(sup, fals, complete)
    res["numbers"] = numbers
    res["complete"] = complete
    return res


# ================================================================ C2 / C3 shared: model-stage cells
def e2_expected(meta, ds):
    ec = expected_cells(meta, ds, E2_N, E2_K)
    return ec


def run_c2(e2c, meta, rng):
    cell_rows = []
    reps_by_ds = defaultdict(list)       # draw-level bootstrap replicates of each cell's mean difference
    points_by_ds = defaultdict(list)
    for (ds, n, k) in sorted(e2c, key=lambda t: (E2_DATASETS.index(t[0]), t[1], korder(t[2]))):
        if n not in E2_N or k not in E2_K:
            continue
        cell = e2c[(ds, n, k)]
        pop, pers, ht, hr = cols_valid(cell, ["pop.U_ndcg", "pers_time.U_ndcg", "hyb_time.U_ndcg",
                                              "hyb_random.U_ndcg"])
        if pop.size < MIN_DRAWS:
            continue
        diff = ht - np.maximum(pop, pers)
        mean, lo, hi = metrics.paired_bootstrap_ci(diff, n_boot=N_BOOT, rng=rng)
        wr, wt = cols_valid(cell, ["hyb_random.w", "hyb_time.w"])
        wrm = float(wr.mean()) if wr.size >= MIN_DRAWS else math.nan
        wtm = float(wt.mean()) if wt.size >= MIN_DRAWS else math.nan
        row = dict(dataset=ds, N=n, k=kname(k), n_draws=int(pop.size), pop=float(pop.mean()),
                   pers_time=float(pers.mean()), hyb_time=float(ht.mean()), hyb_random=float(hr.mean()),
                   diff_mean=float(mean), diff_lo=float(lo), diff_hi=float(hi),
                   a_ok=int(mean >= -C2_TOL - EPS), b_ok=int(lo > 0),
                   w_random=wrm, w_time=wtm,
                   c_ok=(int(wrm >= wtm - EPS) if not (math.isnan(wrm) or math.isnan(wtm)) else ""))
        cell_rows.append(row)
        reps_by_ds[ds].append(boot_means(diff, N_BOOT, rng))
        points_by_ds[ds].append(float(mean))

    exp = {}
    for ds in E2_DATASETS:
        ec = e2_expected(meta, ds)
        if ec is not None:
            exp[ds] = len(ec)
    present_by_ds = defaultdict(int)
    for r in cell_rows:
        present_by_ds[r["dataset"]] += 1
    missing = sum(max(0, exp.get(ds, 0) - present_by_ds.get(ds, 0)) for ds in E2_DATASETS)
    ds_missing = [ds for ds in E2_DATASETS if present_by_ds.get(ds, 0) == 0]
    unknown_open = meta is None and bool(ds_missing)

    ds_rows = []
    for ds in E2_DATASETS:
        rr = [r for r in cell_rows if r["dataset"] == ds]
        if not rr:
            continue
        cc = [r for r in rr if r["c_ok"] != ""]
        ds_rows.append(dict(dataset=ds, n_cells=len(rr), n_expected_cells=exp.get(ds, ""),
                            share_a=np.mean([r["a_ok"] for r in rr]), share_b=np.mean([r["b_ok"] for r in rr]),
                            share_c=np.mean([r["c_ok"] for r in cc]) if cc else math.nan,
                            mean_diff=float(np.mean([r["diff_mean"] for r in rr]))))
    res = dict(cells=cell_rows, datasets=ds_rows)
    numbers = dict(n_cells=len(cell_rows), n_expected_cells=sum(exp.values()) if exp else None,
                   n_missing_cells=missing, datasets_present=[ds for ds in E2_DATASETS if present_by_ds.get(ds)],
                   datasets_missing=ds_missing)
    if not cell_rows:
        res["verdict"] = "not testable"
        res["numbers"] = numbers
        res["why"] = "no model-stage cell with hyb_time / pers_time / pop results in e2.jsonl"
        return res
    n = len(cell_rows)
    na = sum(r["a_ok"] for r in cell_rows)
    nb = sum(r["b_ok"] for r in cell_rows)
    crow = [r for r in cell_rows if r["c_ok"] != ""]
    nc, nc_present = sum(r["c_ok"] for r in crow), len(crow)
    miss_c = missing + (n - nc_present)
    a_lo, a_hi, a = share_bounds(na, n, missing)
    b_lo, b_hi, b = share_bounds(nb, n, missing)
    c_lo, c_hi, c = share_bounds(nc, nc_present, miss_c)
    sup = a_lo >= C2_A_SUPPORT and b_lo >= C2_B_SUPPORT and c_lo >= C2_C_SUPPORT
    fals = a_hi < C2_A_FALSIFY or b_hi < C2_B_FALSIFY
    complete = missing == 0 and not unknown_open
    pt, (lo2, hi2), _ = pool_two_level({ds: np.array(v) for ds, v in reps_by_ds.items()}, points_by_ds, N_BOOT, rng)
    numbers.update(
        a_share_within_0p002=a, a_cells=na, b_share_above_both=b, b_cells=nb, c_share_random_ge_time=c, c_cells=nc,
        c_cells_with_weights=nc_present,
        a_share_range_if_missing=[a_lo, a_hi], b_share_range_if_missing=[b_lo, b_hi], c_share_range_if_missing=[c_lo, c_hi],
        prediction_a=bool(a_lo >= C2_A_SUPPORT), prediction_b=bool(b_lo >= C2_B_SUPPORT), prediction_c=bool(c_lo >= C2_C_SUPPORT),
        falsifier_a=bool(a_hi < C2_A_FALSIFY), falsifier_b=bool(b_hi < C2_B_FALSIFY),
        pooled_diff_mean=pt, pooled_diff_lo=lo2, pooled_diff_hi=hi2,
        n_datasets_positive_mean_diff=int(sum(1 for r in ds_rows if r["mean_diff"] > 0)), n_datasets=len(ds_rows),
        mean_w_random=float(np.nanmean([r["w_random"] for r in crow])) if crow else None,
        mean_w_time=float(np.nanmean([r["w_time"] for r in crow])) if crow else None,
        min_draws=min(r["n_draws"] for r in cell_rows))
    numbers["indicative_verdict"] = decide(a >= C2_A_SUPPORT and b >= C2_B_SUPPORT and (c >= C2_C_SUPPORT),
                                           a < C2_A_FALSIFY or b < C2_B_FALSIFY, True)
    res["verdict"] = decide(sup, fals, complete)
    res["numbers"] = numbers
    res["complete"] = complete
    return res


def run_c3(e2c, meta, rng):
    cell_rows = []
    series = defaultdict(dict)           # (ds, k) -> {N: row}
    for (ds, n, k) in sorted(e2c, key=lambda t: (E2_DATASETS.index(t[0]), korder(t[2]), t[1])):
        if n not in E2_N or k not in E2_K:
            continue
        pop, pers = cols_valid(e2c[(ds, n, k)], ["pop.U_ndcg", "pers_time.U_ndcg"])
        if pop.size < MIN_DRAWS:
            continue
        diff = pers - pop
        mean, lo, hi = metrics.paired_bootstrap_ci(diff, n_boot=N_BOOT, rng=rng)
        row = dict(dataset=ds, k=kname(k), N=n, n_draws=int(pop.size), pop=float(pop.mean()),
                   pers_time=float(pers.mean()), diff_mean=float(mean), diff_lo=float(lo), diff_hi=float(hi),
                   above_zero=int(lo > 0))
        cell_rows.append(row)
        series[(ds, k)][n] = row

    nstar = {}      # (ds, k) -> (status, value)
    for ds in E2_DATASETS:
        ds_has = any(key[0] == ds for key in series)
        ec = e2_expected(meta, ds)
        for k in E2_K:
            if not ds_has:
                nstar[(ds, k)] = ("missing", None)
                continue
            if ec is not None:
                k_in_grid = any(kk == k for _, kk in ec)
                exp_ns = sorted({n for n, kk in ec if kk == k})
            else:
                k_in_grid = (ds, k) in series
                exp_ns = sorted({n for (d2, _), s in series.items() if d2 == ds for n in s})
            if not k_in_grid:
                nstar[(ds, k)] = ("n/a", None)
                continue
            s = series.get((ds, k), {})
            hit = [n for n in sorted(s) if s[n]["above_zero"]]
            if hit:
                nstar[(ds, k)] = ("exists", hit[0])
            elif all(n in s for n in exp_ns):
                nstar[(ds, k)] = ("none", None)
            else:
                nstar[(ds, k)] = ("incomplete", None)

    def text(st, v):
        return {"exists": str(v), "none": "none", "n/a": "n/a", "incomplete": "none so far",
                "missing": "no data"}[st]

    wide_rows = []
    long_rows = []
    viol = []
    for ds in E2_DATASETS:
        wr = dict(dataset=ds)
        seq = []
        for k in E2_K:
            st, v = nstar[(ds, k)]
            wr[f"k={kname(k)}"] = text(st, v)
            maxn = max(series[(ds, k)]) if (ds, k) in series else ""
            long_rows.append(dict(dataset=ds, k=kname(k), status=st, N_star=v if v is not None else "",
                                  max_N_tested=maxn))
            if st == "exists":
                seq.append((k, v))
        bad = any(seq[i + 1][1] > seq[i][1] for i in range(len(seq) - 1))
        wr["n_star_increases_with_k"] = "yes" if bad else ("no" if len(seq) >= 2 else "n/a")
        if bad:
            viol.append(ds)
        wide_rows.append(wr)

    def tally(k):
        sts = [nstar[(ds, k)][0] for ds in E2_DATASETS]
        return {s: sts.count(s) for s in ("exists", "none", "n/a", "incomplete", "missing")}

    t20, tall = tally(20), tally(None)
    open20 = t20["incomplete"] + t20["missing"]
    openall = tall["incomplete"] + tall["missing"]
    n_present = sum(1 for ds in E2_DATASETS if nstar[(ds, 5)][0] != "missing")
    sup = t20["exists"] >= C3_MIN_DATASETS and tall["exists"] >= C3_MIN_DATASETS and not viol
    fals = tall["none"] + tall["n/a"] >= C3_FALSIFY_ABSENT
    # could missing/incomplete series still change the outcome?  (monotonicity is judged on what exists)
    sup_possible = t20["exists"] + open20 >= C3_MIN_DATASETS and tall["exists"] + openall >= C3_MIN_DATASETS and not viol
    fals_possible = tall["none"] + tall["n/a"] + openall >= C3_FALSIFY_ABSENT
    complete = openall == 0 and open20 == 0
    verdict = "supported" if sup else "not supported" if fals else (
        "not testable" if (sup_possible or fals_possible) and not complete else "mixed")
    ind_sup = t20["exists"] >= C3_MIN_DATASETS and tall["exists"] >= C3_MIN_DATASETS and not viol
    ind_fals = tall["none"] + tall["n/a"] + tall["incomplete"] + tall["missing"] >= C3_FALSIFY_ABSENT
    # indicative: treat every unresolved series as absent
    ind = "supported" if ind_sup else "not supported" if ind_fals else "mixed"
    k20_grid = sum(1 for ds in E2_DATASETS if meta is not None and ds in meta and meta[ds]["min_rows"] >= 20) \
        if meta is not None else sum(1 for ds in E2_DATASETS if nstar[(ds, 20)][0] not in ("n/a", "missing"))
    numbers = dict(
        n_datasets_expected=len(E2_DATASETS), n_datasets_with_data=n_present,
        n_star_exists_k20=t20["exists"], n_star_none_k20=t20["none"], k20_not_in_grid=t20["n/a"],
        k20_open=open20, n_star_exists_kall=tall["exists"], n_star_none_kall=tall["none"],
        kall_open=openall,
        n_star_exists_k5=tally(5)["exists"], n_star_exists_k10=tally(10)["exists"],
        datasets_where_n_star_increases_with_k=viol,
        prediction_k20=bool(t20["exists"] >= C3_MIN_DATASETS), prediction_kall=bool(tall["exists"] >= C3_MIN_DATASETS),
        prediction_monotone=bool(not viol), falsifier_kall=bool(tall["none"] + tall["n/a"] >= C3_FALSIFY_ABSENT),
        datasets_with_k20_in_grid=k20_grid,
        n_star_table={f"{ds}|k={kname(k)}": text(*nstar[(ds, k)]) for ds in E2_DATASETS for k in E2_K},
        indicative_verdict=ind)
    res = dict(cells=cell_rows, nstar=wide_rows, long=long_rows, series=series, nstar_map=nstar,
               verdict=verdict, numbers=numbers, complete=complete)
    if not cell_rows:
        res["verdict"] = "not testable"
        res["why"] = "no model-stage cell with pop / pers_time results in e2.jsonl"
    return res


# ================================================================ figures
def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style(ax):
    ax.grid(True, color="#dddddd", lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def fig_c1(c1, path):
    cells = c1["cells"]
    dss = [ds for ds in E1_DATASETS if any(key[0] == ds for key in cells)]
    if not dss:
        return False
    ncol = min(3, len(dss))
    nrow = math.ceil(len(dss) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(max(7.2, 3.5 * ncol), 2.9 * nrow + 1.4), squeeze=False)
    ks_all = sorted({key[2] for key in cells})
    cmap = {k: COLORS[i % len(COLORS)] for i, k in enumerate(ks_all)}
    for ax in axes.ravel():
        ax.set_visible(False)
    for i, ds in enumerate(dss):
        ax = axes[i // ncol][i % ncol]
        ax.set_visible(True)
        style(ax)
        for k in ks_all:
            pts = sorted((key[1] * k, np.nanmean(col(c, "count.A_ndcg100"))) for key, c in cells.items()
                         if key[0] == ds and key[2] == k and len(c) >= MIN_DRAWS)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=3.5, lw=1.5, color=cmap[k],
                        label=f"k = {k}")
        ax.set_xscale("log")
        ax.set_title(ds, fontsize=11)
        if i // ncol == nrow - 1 or i + ncol >= len(dss):
            ax.set_xlabel("total rows kept, N x k (log scale)", fontsize=10)
        if i % ncol == 0:
            ax.set_ylabel("NDCG@100 vs Truth A", fontsize=10)
    handles, labels = None, None
    for ax in axes.ravel():
        if ax.get_visible() and ax.get_legend_handles_labels()[0]:
            handles, labels = ax.get_legend_handles_labels()
            break
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=10, frameon=False,
                   title="choices kept per person: at the same x, small k = many people (breadth), large k = few people (depth)",
                   title_fontsize=10)
    n = c1["numbers"]
    fig.suptitle(f"At equal total rows, many people with few choices beat few people with many choices "
                 f"in {n['n_positive']} of {n['n_pairs']} dataset-budget pairs", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    save(fig, path)
    return True


def largest_datasets(meta, e2c, n=3):
    present = [ds for ds in E2_DATASETS if any(key[0] == ds for key in e2c)]
    if meta is not None:
        present.sort(key=lambda d: -meta[d]["n_pop"] if d in meta else 0)
    else:
        present.sort(key=lambda d: -max(key[1] for key in e2c if key[0] == d))
    return present[:n]


def series_ci(cell_by_n, fields, rng):
    """Mean and 95% interval over draws of one metric (cell_by_n: {N: cell})."""
    xs, m, lo, hi = [], [], [], []
    for n in sorted(cell_by_n):
        (v,) = cols_valid(cell_by_n[n], fields)
        if v.size < MIN_DRAWS:
            continue
        mm, l, h = metrics.paired_bootstrap_ci(v, n_boot=N_BOOT, rng=rng)
        xs.append(n)
        m.append(mm)
        lo.append(l)
        hi.append(h)
    return xs, m, lo, hi


def fig_c2(c2, e2c, meta, rng, path_a, path_b):
    top = largest_datasets(meta, e2c)
    top = [ds for ds in top if any(key[0] == ds and key[2] is None for key in e2c)]
    if not top:
        return []
    out = []
    n = c2["numbers"]
    spec = [("pop.U_ndcg", "popularity", COLORS[6], "-"), ("pers_time.U_ndcg", "best pure personalised (time)", COLORS[0], "-"),
            ("hyb_time.U_ndcg", "hybrid, time hold-out", COLORS[1], "-"), ("hyb_random.U_ndcg", "hybrid, random hold-out", COLORS[2], "--")]
    fig, axes = plt.subplots(1, len(top), figsize=(3.7 * len(top) + 0.8, 4.6), squeeze=False)
    for ax, ds in zip(axes[0], top):
        style(ax)
        by_n = {key[1]: c for key, c in e2c.items() if key[0] == ds and key[2] is None}
        for f, lab, colr, ls in spec:
            xs, m, lo, hi = series_ci(by_n, [f], rng)
            if xs:
                ax.plot(xs, m, ls, marker="o", ms=3.5, lw=1.6, color=colr, label=lab)
                ax.fill_between(xs, lo, hi, color=colr, alpha=0.15, lw=0)
        ax.set_xscale("log")
        ax.set_title(ds, fontsize=11)
        ax.set_xlabel("people sampled, N (log scale)", fontsize=10)
    axes[0][0].set_ylabel("panel NDCG@100 (higher = better)", fontsize=10)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9.5, frameon=False)
    fig.suptitle(f"Hybrid (time hold-out) is within 0.002 of the better pure list in {pct(n['a_share_within_0p002'])} "
                 f"of all cells; panels show k = all, bands are 95% over draws", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    save(fig, path_a)
    out.append(path_a)

    fig, axes = plt.subplots(1, len(top), figsize=(3.7 * len(top) + 0.8, 4.3), squeeze=False)
    for ax, ds in zip(axes[0], top):
        style(ax)
        by_n = {key[1]: c for key, c in e2c.items() if key[0] == ds and key[2] is None}
        for f, lab, colr, ls in [("hyb_random.w", "random hold-out", COLORS[2], "--"),
                                 ("hyb_time.w", "time hold-out", COLORS[1], "-")]:
            xs, m, lo, hi = series_ci(by_n, [f], rng)
            if xs:
                ax.plot(xs, m, ls, marker="o", ms=3.5, lw=1.6, color=colr, label=lab)
                ax.fill_between(xs, lo, hi, color=colr, alpha=0.15, lw=0)
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(ds, fontsize=11)
        ax.set_xlabel("people sampled, N (log scale)", fontsize=10)
    axes[0][0].set_ylabel("weight on personalisation, w (0 to 1)", fontsize=10)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=10, frameon=False, title="how the sample hid choices to pick w",
               title_fontsize=10)
    fig.suptitle(f"Random hold-out gives personalisation at least as much weight as time hold-out in "
                 f"{pct(n['c_share_random_ge_time'])} of cells; panels show k = all", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    save(fig, path_b)
    out.append(path_b)
    return out


def fig_c3(c3, path_table, path_lines):
    out = []
    nstar = c3["nstar_map"]
    present = [ds for ds in E2_DATASETS if nstar[(ds, 5)][0] != "missing"]
    if not present:
        return out
    colour = {"exists": "#c7e9c0", "none": "#f4cccc", "n/a": "#eeeeee", "incomplete": "#fff2cc", "missing": "#ffffff"}
    texts, fills = [], []
    for ds in E2_DATASETS:
        rt, rf = [ds], ["#ffffff"]
        for k in E2_K:
            st, v = nstar[(ds, k)]
            rt.append(str(v) if st == "exists" else {"none": "none", "n/a": "n/a", "incomplete": "none so far",
                                                      "missing": "no data"}[st])
            rf.append(colour[st])
        texts.append(rt)
        fills.append(rf)
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(E2_DATASETS) + 1.6))
    ax.axis("off")
    tb = ax.table(cellText=texts, cellColours=fills, colLabels=["dataset"] + [f"k = {kname(k)}" for k in E2_K],
                  colWidths=[0.34] + [0.165] * len(E2_K),
                  loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(11)
    tb.scale(1, 1.6)
    n = c3["numbers"]
    ax.set_title(f"A crossover N* exists at k = all in {n['n_star_exists_kall']} of {n['n_datasets_expected']} datasets "
                 f"and at k = 20 in {n['n_star_exists_k20']}", fontsize=12, pad=10)
    fig.text(0.5, 0.02, "N* = smallest N (people) where personalised(time) beats popularity on the panel, 95% interval above 0.\n"
             "none = never in the tested grid; n/a = k not in that dataset's grid; none so far / no data = run incomplete.",
             ha="center", fontsize=9.5)
    save(fig, path_table)
    out.append(path_table)

    dss = [ds for ds in E2_DATASETS if (ds, None) in c3["series"]]
    if dss:
        ncol = 4
        nrow = math.ceil(len(dss) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.1 * nrow + 0.9), squeeze=False)
        for ax in axes.ravel():
            ax.set_visible(False)
        for i, ds in enumerate(dss):
            ax = axes[i // ncol][i % ncol]
            ax.set_visible(True)
            style(ax)
            s = c3["series"][(ds, None)]
            ns = sorted(s)
            ax.axhline(0, color="#444444", lw=1)
            ax.plot(ns, [s[x]["diff_mean"] for x in ns], "-o", ms=3.5, lw=1.6, color=COLORS[0])
            ax.fill_between(ns, [s[x]["diff_lo"] for x in ns], [s[x]["diff_hi"] for x in ns], color=COLORS[0],
                            alpha=0.18, lw=0)
            st, v = nstar[(ds, None)]
            if st == "exists":
                ax.axvline(v, color=COLORS[1], ls="--", lw=1.3)
            ax.set_xscale("log")
            ax.set_title(ds + (f"  (N* = {v})" if st == "exists" else "  (no N*)"), fontsize=10.5)
            if i // ncol == nrow - 1 or i + ncol >= len(dss):
                ax.set_xlabel("people sampled, N (log scale)", fontsize=10)
            if i % ncol == 0:
                ax.set_ylabel("panel NDCG@100,\npersonalised - popularity", fontsize=10)
        fig.suptitle(f"At k = all, personalised(time) overtakes popularity in {n['n_star_exists_kall']} of "
                     f"{n['n_datasets_expected']} datasets (band: 95% over draws; dashed line: N*)", fontsize=12, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, path_lines)
        out.append(path_lines)
    return out


# ================================================================ verdict text
def partial_note(res, kind):
    if res.get("complete", True):
        return ("", "")
    n = res["numbers"]
    ind = n.get("indicative_verdict")
    if kind == "C1":
        have, want, unit = n["n_pairs"], n.get("n_expected_pairs"), ("pairs", "组合")
    elif kind == "C2":
        have, want, unit = n["n_cells"], n.get("n_expected_cells"), ("cells", "格")
    else:
        have, want, unit = n["n_datasets_with_data"], n["n_datasets_expected"], ("datasets", "个数据集")
    wtxt = "all" if want is None else str(want)
    return (f" The data are incomplete ({have} of {wtxt} expected {unit[0]} present), so the verdict is withheld; "
            f"on what is present the answer would be '{ind}'.",
            f"数据还不完整（预期 {wtxt} 个{unit[1]}，目前只有 {have} 个），所以暂不下结论；按现有数据，结论会是“{ind}”。")


def build_verdicts(c1, c2, c3):
    out = []
    # ---- C1
    n = c1["numbers"]
    if "pooled_A" in n:
        pa = n["pooled_A"]
        pb, pu = n.get("pooled_B"), n.get("pooled_U")
        pn_en, pn_zh = partial_note(c1, "C1")
        en = (f"The list from many people with few choices each beat the list from few people with many choices in "
              f"{n['n_positive']} of {n['n_pairs']} dataset-budget pairs ({pct(n['share_positive'])}). "
              f"The pooled difference (breadth minus depth) is {pa['mean']:+.4f} NDCG@100 (95% interval {pa['lo']:+.4f} to {pa['hi']:+.4f}).")
        zh = (f"在 {n['n_pairs']} 个（数据集，预算）组合中，有 {n['n_positive']} 个（{pct(n['share_positive'])}）是“人多、每人记录少”的名单比“人少、每人记录多”的名单更好。"
              f"合并后的差（人多 减 记录多）是 {pa['mean']:+.4f}（95% 区间 {pa['lo']:+.4f} 到 {pa['hi']:+.4f}）。")
        if pb and pu:
            en += (f" Against later choices the same difference is {pb['mean']:+.4f} (Truth B) and {pu['mean']:+.4f} (panel), shown only as extra information.")
            zh += f" 对未来选择，同样的差是 {pb['mean']:+.4f}（真值 B）和 {pu['mean']:+.4f}（测试人群），仅作补充。"
        en += pn_en
        zh += pn_zh
    else:
        en = "No (dataset, budget) pair with two usable cells was found in the popularity stage (e1), so C1 could not be tested."
        zh = "流行度阶段（e1）里没有找到可比较的（数据集，预算）组合，所以无法检验 C1。"
    out.append(dict(
        id="C1", title_en="C1. Breadth beats depth for estimating the list",
        title_zh="C1. 广度胜过深度：同样的记录总数，人多比每人记录多更能估准名单",
        verdict=c1["verdict"], numbers=n, reading_en=en, reading_zh=zh,
        rule="difference (largest-N cell minus largest-k cell) of count NDCG@100 vs Truth A > 0 in >= 75% of (dataset, budget) pairs "
             "and pooled interval excludes 0; falsified if <= 50% positive or pooled interval entirely below 0"))
    # ---- C2
    n = c2["numbers"]
    if "a_share_within_0p002" in n:
        pn_en, pn_zh = partial_note(c2, "C2")
        en = (f"Over {n['n_cells']} cells, the time-hold-out hybrid was within 0.002 of the better pure list in {pct(n['a_share_within_0p002'])} "
              f"(rule: at least 90%), clearly above both in {pct(n['b_share_above_both'])} (rule: at least 20%), "
              f"and the random hold-out gave personalisation at least as much weight in {pct(n['c_share_random_ge_time'])} (rule: at least 70%).")
        zh = (f"在 {n['n_cells']} 个格子中，用时间留出法选权重的混合名单与较好的单一名单相差不超过 0.002 的占 {pct(n['a_share_within_0p002'])}（要求至少 90%）；"
              f"明显同时胜过两者的占 {pct(n['b_share_above_both'])}（要求至少 20%）；随机留出法给个性化的权重不低于时间留出法的占 {pct(n['c_share_random_ge_time'])}（要求至少 70%）。")
        en += pn_en
        zh += pn_zh
    else:
        en = "No model-stage cell with hybrid results was found in e2, so C2 could not be tested."
        zh = "模型训练阶段（e2）里没有找到带混合名单结果的格子，所以无法检验 C2。"
    out.append(dict(
        id="C2", title_en="C2. A hybrid with honest weights is never worse than the better pure list, and sometimes better than both",
        title_zh="C2. 只用样本自己选权重的混合名单，不比较好的单一名单差，有时还胜过两者",
        verdict=c2["verdict"], numbers=n, reading_en=en, reading_zh=zh,
        rule="(a) hybrid_time >= best pure list - 0.002 in >= 90% of cells; (b) above both with paired interval > 0 in >= 20%; "
             "(c) hybrid_random weight >= hybrid_time weight in >= 70%; falsified if (a) < 75% or (b) < 10%"))
    # ---- C3
    n = c3["numbers"]
    if "n_star_exists_kall" in n and c3.get("cells"):
        pn_en, pn_zh = partial_note(c3, "C3")
        en = (f"A crossover size N* (personalised beats popularity, 95% interval above 0) exists for k = all in {n['n_star_exists_kall']} of "
              f"{n['n_datasets_expected']} datasets and for k = 20 in {n['n_star_exists_k20']} (rule: at least 5 each; "
              f"k = 20 is in the grid for only {n['datasets_with_k20_in_grid']} datasets). "
              + ("N* never increases with k where it is defined." if not n["datasets_where_n_star_increases_with_k"]
                 else f"N* increases with k in {', '.join(n['datasets_where_n_star_increases_with_k'])}."))
        zh = (f"在 k = 全部 时，{n['n_datasets_expected']} 个数据集中有 {n['n_star_exists_kall']} 个存在交叉样本量 N*（个性化模型显著胜过流行度）；"
              f"k = 20 时有 {n['n_star_exists_k20']} 个（要求各至少 5 个；k = 20 只出现在 {n['datasets_with_k20_in_grid']} 个数据集的网格里）。"
              + ("在能确定的地方，N* 没有随 k 增大而增大。" if not n["datasets_where_n_star_increases_with_k"]
                 else f"在 {', '.join(n['datasets_where_n_star_increases_with_k'])} 里，N* 随 k 增大而增大。"))
        en += pn_en
        zh += pn_zh
    else:
        en = "No model-stage cell with personalised results was found in e2, so C3 could not be tested."
        zh = "模型训练阶段（e2）里没有找到个性化模型的结果，所以无法检验 C3。"
    out.append(dict(
        id="C3", title_en="C3. A crossover sample size exists, and shrinks as k grows",
        title_zh="C3. 存在一个交叉样本量，并且随每人记录数 k 增大而变小",
        verdict=c3["verdict"], numbers=n, reading_en=en, reading_zh=zh,
        rule="N*(k) = smallest N where personalised_time beats popularity with 95% interval above 0; exists for k=20 and k=all in >= 5 of 7 "
             "datasets and does not increase with k; falsified if absent at k=all in >= 4 of 7"))
    for v, res in zip(out, (c1, c2, c3)):
        if res.get("why"):
            v["numbers"] = dict(v["numbers"], reason=res["why"])
    return out


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    runs, out = args.runs, args.out
    tdir, fdir = os.path.join(out, "03_tables"), os.path.join(out, "04_figures")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)
    meta = load_meta(runs, out)
    if meta is None:
        print("note: datasets.json not found; expected grids unknown, missing datasets keep verdicts 'not testable'")

    e1 = read_jsonl(os.path.join(runs, "e1.jsonl"))
    e2 = read_jsonl(os.path.join(runs, "e2.jsonl"))
    for name, rows in (("e1", e1), ("e2", e2)):
        print(f"{name}.jsonl: " + ("missing" if rows is None else f"{len(rows)} rows"))

    rng1, rng2, rng3 = (np.random.default_rng([SEED, i]) for i in (1, 2, 3))
    empty = dict(verdict="not testable", numbers={}, cells=[], datasets=[], pairs=[], nstar=[], long=[], series={},
                 nstar_map={})
    c1 = run_c1(e1, meta, rng1) if e1 is not None else dict(empty, why="e1.jsonl not found in the runs directory")
    e2c = e2_cells(e2) if e2 is not None else {}
    if e2 is not None:
        c2 = run_c2(e2c, meta, rng2)
        c3 = run_c3(e2c, meta, rng3)
    else:
        c2 = dict(empty, why="e2.jsonl not found in the runs directory")
        c3 = dict(empty, why="e2.jsonl not found in the runs directory")

    # tables
    if c1["pairs"]:
        f = ["dataset", "budget", "n_cells", "N_wide", "k_wide", "N_deep", "k_deep", "draws_wide", "draws_deep"]
        for tag in C1_METRICS:
            f += [f"{tag}_wide", f"{tag}_deep", f"{tag}_diff", f"{tag}_lo", f"{tag}_hi"]
        write_csv(os.path.join(tdir, "an_confirm_c1_pairs.csv"), c1["pairs"], f + ["positive"])
        write_csv(os.path.join(tdir, "an_confirm_c1_datasets.csv"), c1["datasets"],
                  ["dataset", "n_pairs", "n_expected_pairs", "n_positive", "share_positive", "mean_diff_A"])
    if c2["cells"]:
        write_csv(os.path.join(tdir, "an_confirm_c2_cells.csv"), c2["cells"],
                  ["dataset", "N", "k", "n_draws", "pop", "pers_time", "hyb_time", "hyb_random", "diff_mean", "diff_lo",
                   "diff_hi", "a_ok", "b_ok", "w_random", "w_time", "c_ok"])
        write_csv(os.path.join(tdir, "an_confirm_c2_datasets.csv"), c2["datasets"],
                  ["dataset", "n_cells", "n_expected_cells", "share_a", "share_b", "share_c", "mean_diff"])
    if c3["cells"]:
        write_csv(os.path.join(tdir, "an_confirm_c3_cells.csv"), c3["cells"],
                  ["dataset", "k", "N", "n_draws", "pop", "pers_time", "diff_mean", "diff_lo", "diff_hi", "above_zero"])
        write_csv(os.path.join(tdir, "an_confirm_c3_nstar.csv"), c3["nstar"],
                  ["dataset"] + [f"k={kname(k)}" for k in E2_K] + ["n_star_increases_with_k"])
        write_csv(os.path.join(tdir, "an_confirm_c3_nstar_long.csv"), c3["long"],
                  ["dataset", "k", "status", "N_star", "max_N_tested"])

    # figures
    made = []
    if c1["pairs"] and fig_c1(c1, os.path.join(fdir, "F-C1_breadth_vs_depth.png")):
        made.append("F-C1_breadth_vs_depth.png")
    if c2["cells"]:
        for p in fig_c2(c2, e2c, meta, np.random.default_rng([SEED, 4]),
                        os.path.join(fdir, "F-C2a_hybrid_vs_pure.png"), os.path.join(fdir, "F-C2b_weight_random_vs_time.png")):
            made.append(os.path.basename(p))
    if c3["cells"]:
        for p in fig_c3(c3, os.path.join(fdir, "F-C3a_crossover_table.png"), os.path.join(fdir, "F-C3b_personalised_minus_popularity.png")):
            made.append(os.path.basename(p))

    verdicts = build_verdicts(c1, c2, c3)
    with open(os.path.join(tdir, "an_confirm_verdicts.json"), "w", encoding="utf-8") as fh:
        json.dump(clean(verdicts), fh, ensure_ascii=False, indent=2)

    # c_summary.csv
    n1, n2, n3 = (v["numbers"] for v in verdicts)
    pa = n1.get("pooled_A", {})
    summ = [
        dict(claim="C1", verdict=verdicts[0]["verdict"], units=n1.get("n_pairs", ""), share_or_count=n1.get("share_positive", ""),
             headline_1_name="pairs_positive", headline_1=n1.get("n_positive", ""),
             headline_2_name="pooled_diff_A_ndcg100", headline_2=pa.get("mean", ""),
             headline_3_name="pooled_ci", headline_3=(f"[{pa['lo']:.4f}, {pa['hi']:.4f}]" if pa else "")),
        dict(claim="C2", verdict=verdicts[1]["verdict"], units=n2.get("n_cells", ""), share_or_count=n2.get("a_share_within_0p002", ""),
             headline_1_name="share_a_within_0.002", headline_1=n2.get("a_share_within_0p002", ""),
             headline_2_name="share_b_above_both", headline_2=n2.get("b_share_above_both", ""),
             headline_3_name="share_c_random_ge_time", headline_3=n2.get("c_share_random_ge_time", "")),
        dict(claim="C3", verdict=verdicts[2]["verdict"], units=n3.get("n_datasets_with_data", ""),
             share_or_count=n3.get("n_star_exists_kall", ""),
             headline_1_name="datasets_with_N*_k_all", headline_1=n3.get("n_star_exists_kall", ""),
             headline_2_name="datasets_with_N*_k_20", headline_2=n3.get("n_star_exists_k20", ""),
             headline_3_name="datasets_N*_increases_with_k", headline_3=len(n3.get("datasets_where_n_star_increases_with_k", []))),
    ]
    write_csv(os.path.join(tdir, "c_summary.csv"), summ,
              ["claim", "verdict", "units", "share_or_count", "headline_1_name", "headline_1", "headline_2_name",
               "headline_2", "headline_3_name", "headline_3"])

    for v in verdicts:
        print(f"{v['id']}: {v['verdict']} | {v['reading_en']}")
    print("figures: " + (", ".join(made) if made else "none"))
    print("tables in " + tdir)


if __name__ == "__main__":
    main()
