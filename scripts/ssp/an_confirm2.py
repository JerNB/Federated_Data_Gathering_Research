#!/usr/bin/env python3
"""Confirmatory claims C1, C2, C3 of the Small-Sample Popularity pre-registration under the AMENDED rules
(section 11: A9-A12 and A14; second set of amendments).

Usage:
    python3 scripts/ssp/an_confirm2.py --runs <dir with e1.jsonl, e2.jsonl, e7.jsonl> --out <out dir>

Reads  <runs>/e1.jsonl (main grid), <runs>/e2.jsonl (model training), <runs>/e7.jsonl (arm `boot`),
       <runs>/../01_datasets/datasets.json and, when present, <runs>/../03_tables/cand_share.csv.
Writes <out>/03_tables/an_confirm2_*.csv, <out>/03_tables/an_confirm2_summary.csv,
       <out>/03_tables/an_confirm2_verdicts.json and <out>/04_figures/F2-C*.png.

Rules implemented:

C1 (A10)  Literal test: count.A_ndcg100, per (dataset, budget B = N*k, k numeric) with >= 2 cells, difference =
    widest cell (largest N) minus deepest cell (largest k); each cell's draws are resampled separately (2000
    resamples).  NEW: only pairs whose widest cell has N <= 5% of the group (datasets.json n_pop) are judged.
    Supported: >= 75% of judged pairs positive AND pooled interval excludes 0 AND pooled difference >= 0.005.
    Falsified ('not supported'): <= 50% positive OR pooled interval entirely below 0.  50-75%: 'mixed'.
    Pooled estimate = mean over datasets of the mean pair difference; pooled interval = two-level percentile
    bootstrap (pairs resampled within a dataset, each chosen pair's difference a draw-level replicate).
    The same contrast, same rule, is reported as secondary lines for ht.A_ndcg100, count.Au_ndcg100 (e1, same 5%
    restriction) and for the `boot` arm of e7 (count/ht x A/Au, no 5% restriction).
C2 (A11)  Per (dataset, N, k) cell of the model stage, panel NDCG@100 (U_ndcg).  (a) mean(hyb_time) minus the larger
    of the cell means of pop and pers_time >= -0.002, share of cells >= 90%.  (b) paired-by-draw 95% interval of
    hyb_time minus the pure list with the larger CELL MEAN above 0, share of cells >= 20%.  (c) per cell count
    draws with hyb_random.w > hyb_time.w versus <, ties dropped; cells need >= 5 untied draws; cells whose
    hyb_random and hyb_time model and w agree in every draw (identical splits) are excluded and counted;
    supported if 'random more personalised' is the larger count in >= 70% of eligible cells, falsified at <= 50%.
    Verdict: supported if a >= 90% and b >= 20% and c >= 70%; falsified if a < 75% or b < 10% or c <= 50%.
C3 (A12)  N*(dataset, k), k in {10, all}: smallest N whose paired-by-draw 95% interval of
    pers_time - pop is above 0 AND whose mean difference stays > 0 at every larger tested N.  Absent N* counts as
    larger than any present one.  Supported: N* at k = all in >= 5 of 7 datasets AND at k = 10 in >= 4 of 7 AND no
    violation (N*(all) > N*(10)).  Falsified: N* absent at k = all in >= 4 of 7.  Else mixed.  N_max per dataset =
    largest N tested in e2; optional cand_share column from cand_share.csv ('cap-limited' when the share < 0.5).
A14       'most datasets' = strictly more than half of the datasets that could be judged (reported as a
    descriptive companion wherever a share of datasets is read).

Partial data: shares are judged against the number of EXPECTED units (cells/pairs the run grid should contain,
from datasets.json).  A claim is only decided when the available data already fix the outcome whatever the
missing units would show; otherwise the verdict is 'not testable' and the numbers carry an 'indicative_verdict'
computed on the available units.  Thresholds are constants below and never tuned.
"""
from __future__ import annotations

import argparse
import csv
import itertools
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
C3_K = [10, None]
N_BOOT = 2000
SEED = 20261014
MIN_DRAWS = 5            # a cell with fewer valid draws is treated as missing (cannot carry an interval)
EPS = 1e-12
INF = math.inf

# C1 (A10)
C1_SHARE_SUPPORT, C1_SHARE_FALSIFY = 0.75, 0.50
C1_MIN_POOLED = 0.005
C1_NPOP_FRAC = 0.05
# C2 (A11)
C2_TOL = 0.002
C2_A_SUPPORT, C2_A_FALSIFY = 0.90, 0.75
C2_B_SUPPORT, C2_B_FALSIFY = 0.20, 0.10
C2_C_SUPPORT, C2_C_FALSIFY = 0.70, 0.50
C2_MIN_UNTIED = 5
W_TOL = 1e-9
# C3 (A12)
C3_MIN_ALL, C3_MIN_TEN = 5, 4
C3_FALSIFY_ABSENT = 4
CAP_LIMITED_BELOW = 0.5

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]

# (id, source, column, restricted to N <= 5% of the group, English label, panel title)
C1_VARIANTS = [
    dict(id="count_A", src="e1", field="count.A_ndcg100", restrict=True, primary=True,
         label="count vs Truth A (literal C1)"),
    dict(id="ht_A", src="e1", field="ht.A_ndcg100", restrict=True, primary=False,
         label="ht vs Truth A"),
    dict(id="count_Au", src="e1", field="count.Au_ndcg100", restrict=True, primary=False,
         label="count vs Truth A_u (user-normalised)"),
    dict(id="boot_count_A", src="boot", field="count.A_ndcg100", restrict=False, primary=False,
         label="boot arm: count vs Truth A"),
    dict(id="boot_ht_A", src="boot", field="ht.A_ndcg100", restrict=False, primary=False,
         label="boot arm: ht vs Truth A"),
    dict(id="boot_count_Au", src="boot", field="count.Au_ndcg100", restrict=False, primary=False,
         label="boot arm: count vs Truth A_u"),
    dict(id="boot_ht_Au", src="boot", field="ht.Au_ndcg100", restrict=False, primary=False,
         label="boot arm: ht vs Truth A_u"),
]
C1_PANELS = ["count_A", "ht_A", "count_Au", "boot_count_Au"]


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


def load_cand_share(runs):
    """{dataset: share} from <runs>/../03_tables/cand_share.csv (columns dataset, share), {} when absent."""
    p = os.path.join(runs, "..", "03_tables", "cand_share.csv")
    if not os.path.isfile(p):
        return {}
    out = {}
    with open(p, "r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                out[r["dataset"]] = float(r["share"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
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
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100.0 * x:.0f}%"


def fnum(x, fmt="+.4f"):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)


def most(n_yes, n_judged):
    """A14: 'most datasets' = strictly more than half of the datasets that could be judged."""
    return bool(n_judged > 0 and 2 * n_yes > n_judged)


def decide(sup, fals, complete):
    if sup:
        return "supported"
    if fals:
        return "not supported"
    return "mixed" if complete else "not testable"


def share_bounds(n_true, n_present, n_missing):
    """Lowest and highest share the claim could end with once the missing units are in, and the present share."""
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
def row_arm(r):
    arm = r.get("arm")
    if arm is None:
        arm = "uniform"
        for part in (r.get("key", "") or "").split("|"):
            if part.startswith("arm="):
                arm = part[4:]
    return arm


def numeric_k_cells(rows, arm, uniform_only):
    """{(dataset, N, k): {draw: row}} for one arm of e1/e7, k numeric."""
    cells = defaultdict(dict)
    for r in rows:
        if row_arm(r) != arm:
            continue
        if uniform_only and (r.get("users", "uniform") != "uniform" or r.get("rows", "random") != "random"):
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


def has_field(cells, field):
    return any(field in r for c in cells.values() for r in c.values())


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


def make_source(cells, meta, restrict, filename):
    """Pairs (widest vs deepest cell at equal budget) of one e1/e7 arm, plus the expected judged pairs per dataset."""
    src = dict(cells=cells, file=filename, restrict=restrict, meta=meta, pairs=[], exp_pairs={})
    if cells is None:
        return src
    budgets = defaultdict(list)
    for (ds, n, k), d in cells.items():
        if len(d) >= MIN_DRAWS:
            budgets[(ds, n * k)].append((n, k))
    exp_sets = {}
    numeric_ks = [k for k in K_GRID if k is not None]
    if meta is not None:
        for ds in E1_DATASETS:
            ec = expected_cells(meta, ds, N_GRID, numeric_ks)
            if ec is None:
                continue
            by_b = defaultdict(set)
            for n, k in ec:
                by_b[n * k].add((n, k))
            cnt = 0
            for b, s in by_b.items():
                exp_sets[(ds, b)] = s
                if len(s) >= 2 and (not restrict or max(s)[0] <= C1_NPOP_FRAC * meta[ds]["n_pop"]):
                    cnt += 1
            src["exp_pairs"][ds] = cnt
    for (ds, b), lst in sorted(budgets.items(), key=lambda t: (E1_DATASETS.index(t[0][0]), t[0][1])):
        if len(lst) < 2:
            continue
        wide = max(lst)
        deep = max(lst, key=lambda t: t[1])
        if meta is not None and ds in meta:
            in_scope = (not restrict) or wide[0] <= C1_NPOP_FRAC * meta[ds]["n_pop"]
            complete = (ds, b) in exp_sets and exp_sets[(ds, b)] <= set(lst)
        else:
            in_scope = (not restrict)          # cannot be judged without n_pop when restricted
            complete = True
        src["pairs"].append(dict(dataset=ds, budget=b, n_cells=len(lst), N_wide=wide[0], k_wide=wide[1],
                                 N_deep=deep[0], k_deep=deep[1], in_scope=in_scope, complete=complete,
                                 draws_wide=len(cells[(ds, *wide)]), draws_deep=len(cells[(ds, *deep)])))
    return src


def eval_variant(var, src, rng):
    """The widest-minus-deepest contrast of one column, judged with the amended C1 rule."""
    field, vid = var["field"], var["id"]
    res = dict(id=vid, field=field, label=var["label"], source=var["src"], restricted=var["restrict"],
               n_pairs=0, n_positive=0, share_positive=math.nan, verdict="not testable", complete=False,
               forest=[], datasets=[], pooled=None)
    if src["cells"] is None:
        res["why"] = f"{src['file']} not found in the runs directory"
        return res
    if var["restrict"] and src["meta"] is None:
        res["why"] = "datasets.json (n_pop) not found, so the 5%-of-group restriction cannot be applied"
        return res
    if not src["cells"]:
        res["why"] = f"no usable rows of this arm in {src['file']} (arm not run yet or earlier code version)"
        return res
    if not has_field(src["cells"], field):
        res["why"] = f"column {field} is absent from {src['file']} (rows from an earlier code version)"
        return res
    cells, meta = src["cells"], src["meta"]
    D, P = defaultdict(list), defaultdict(list)
    valid = []                 # judged pairs with a usable difference
    for p in src["pairs"]:
        if not p["in_scope"]:
            continue
        ds = p["dataset"]
        xw = col(cells[(ds, p["N_wide"], p["k_wide"])], field)
        xw = xw[np.isfinite(xw)]
        xd = col(cells[(ds, p["N_deep"], p["k_deep"])], field)
        xd = xd[np.isfinite(xd)]
        st = dict(wide=math.nan, deep=math.nan, diff=math.nan, lo=math.nan, hi=math.nan)
        if xw.size >= MIN_DRAWS and xd.size >= MIN_DRAWS:
            reps = boot_means(xw, N_BOOT, rng) - boot_means(xd, N_BOOT, rng)
            lo, hi = ci95(reps)
            st = dict(wide=float(xw.mean()), deep=float(xd.mean()), diff=float(xw.mean() - xd.mean()), lo=lo, hi=hi)
            D[ds].append(reps)
            P[ds].append(st["diff"])
            valid.append(p)
        p[f"{vid}_wide"], p[f"{vid}_deep"], p[f"{vid}_diff"] = st["wide"], st["deep"], st["diff"]
        p[f"{vid}_lo"], p[f"{vid}_hi"] = st["lo"], st["hi"]
    n_pairs = len(valid)
    n_pos = sum(1 for p in valid if p[f"{vid}_diff"] > 0)
    n_comp = sum(1 for p in valid if p["complete"])
    n_comp_pos = sum(1 for p in valid if p["complete"] and p[f"{vid}_diff"] > 0)
    exp_total = sum(src["exp_pairs"].values()) if src["exp_pairs"] else None
    missing = sum(max(0, src["exp_pairs"].get(ds, 0) - sum(1 for p in valid if p["dataset"] == ds and p["complete"]))
                  for ds in E1_DATASETS) if src["exp_pairs"] else 0
    present_ds = [ds for ds in E1_DATASETS if any(p["dataset"] == ds for p in valid)]
    unknown_open = meta is None and len(present_ds) < len(E1_DATASETS)
    res.update(n_pairs=n_pairs, n_positive=n_pos, n_expected_pairs=exp_total, n_missing_pairs=missing,
               datasets_present=present_ds, datasets_missing=[d for d in E1_DATASETS if d not in present_ds])
    if n_pairs == 0:
        res["why"] = "no judged (dataset, budget) pair with two usable cells"
        return res
    res["share_positive"] = n_pos / n_pairs
    ds_rows = []
    for ds in present_ds:
        vv = [p for p in valid if p["dataset"] == ds]
        dd = [p[f"{vid}_diff"] for p in vv]
        ds_rows.append(dict(dataset=ds, n_pairs=len(vv), n_expected_pairs=src["exp_pairs"].get(ds, ""),
                            n_positive=sum(1 for x in dd if x > 0), share_positive=float(np.mean([x > 0 for x in dd])),
                            mean_diff=float(np.mean(dd))))
    pt, (lo2, hi2), (lod, hid) = pool_two_level({ds: np.array(v) for ds, v in D.items()}, P, N_BOOT, rng)
    res["pooled"] = dict(mean=pt, lo=lo2, hi=hi2, lo_draws_only=lod, hi_draws_only=hid)
    forest = []
    for ds in present_ds:
        m = np.array(D[ds]).mean(axis=0)
        l, h = ci95(m)
        forest.append((ds, float(np.mean(P[ds])), l, h))
    res["forest"] = forest
    res["datasets"] = ds_rows
    n_dpos = sum(1 for r in ds_rows if r["mean_diff"] > 0)
    res["datasets_positive"], res["datasets_judged"] = n_dpos, len(ds_rows)
    res["most_datasets_positive"] = most(n_dpos, len(ds_rows))      # A14, descriptive
    # decision: complete data decide on all three conditions; incomplete data can only falsify on the share
    complete = missing == 0 and not unknown_open
    res["complete"] = complete
    share = res["share_positive"]
    sup_i = share >= C1_SHARE_SUPPORT and lo2 > 0 and pt >= C1_MIN_POOLED
    fals_i = share <= C1_SHARE_FALSIFY or hi2 < 0
    res["indicative_verdict"] = decide(sup_i, fals_i, True)
    if complete:
        res["verdict"] = decide(sup_i, fals_i, True)
    else:
        _, hi_b, _ = share_bounds(n_comp_pos, n_comp, missing)
        fals_sure = (not unknown_open) and hi_b <= C1_SHARE_FALSIFY
        res["verdict"] = "not supported" if fals_sure else "not testable"
        res["share_positive_upper_bound_if_missing_all_positive"] = hi_b
    res["prediction_share_met"] = bool(share >= C1_SHARE_SUPPORT)
    res["prediction_interval_excludes_zero"] = bool(lo2 > 0)
    res["prediction_pooled_ge_0p005"] = bool(pt >= C1_MIN_POOLED)
    res["falsifier_share"] = bool(share <= C1_SHARE_FALSIFY)
    res["falsifier_interval_below_zero"] = bool(hi2 < 0)
    return res


def run_c1(e1, e7, meta):
    srcs = dict(
        e1=make_source(numeric_k_cells(e1, "uniform", True) if e1 is not None else None, meta, True, "e1.jsonl"),
        boot=make_source(numeric_k_cells(e7, "boot", False) if e7 is not None else None, meta, False, "e7.jsonl (arm boot)"))
    results = {}
    for i, var in enumerate(C1_VARIANTS):
        results[var["id"]] = eval_variant(var, srcs[var["src"]], np.random.default_rng([SEED, 1, i]))
    prim = results["count_A"]
    res = dict(variants=results, sources=srcs, verdict=prim["verdict"], complete=prim["complete"])
    if prim.get("why"):
        res["why"] = prim["why"]
    # A10 finding: literal C1 fails and the clean boot arm shows depth at least as good as breadth
    b = results["boot_count_A"]
    finding = dict(applies=False, provisional=False)
    if prim["verdict"] == "not supported" and b.get("pooled") is not None and b["pooled"]["mean"] <= 0:
        finding = dict(applies=True, provisional=not b["complete"], boot_pooled_mean=b["pooled"]["mean"],
                       boot_pooled_lo=b["pooled"]["lo"], boot_pooled_hi=b["pooled"]["hi"])
    res["depth_finding"] = finding
    n = dict(n_pairs=prim["n_pairs"], n_positive=prim["n_positive"], share_positive=prim["share_positive"],
             n_expected_pairs=prim.get("n_expected_pairs"), n_missing_pairs=prim.get("n_missing_pairs"),
             datasets_present=prim.get("datasets_present", []), datasets_missing=prim.get("datasets_missing", []),
             restriction="widest cell N <= 5% of the group (datasets.json n_pop)")
    if prim.get("pooled"):
        n["pooled_A"] = dict(prim["pooled"], n_pairs=prim["n_pairs"], share_positive=prim["share_positive"])
        for kk in ("prediction_share_met", "prediction_interval_excludes_zero", "prediction_pooled_ge_0p005",
                   "falsifier_share", "falsifier_interval_below_zero", "indicative_verdict",
                   "datasets_positive", "datasets_judged", "most_datasets_positive"):
            n[kk] = prim.get(kk)
        n["per_dataset_share_positive"] = {r["dataset"]: r["share_positive"] for r in prim["datasets"]}
    lines, sec = [], {}
    for var in C1_VARIANTS:
        r = results[var["id"]]
        entry = dict(label=r["label"], column=r["field"], verdict=r["verdict"], n_pairs=r["n_pairs"],
                     n_positive=r["n_positive"], share_positive=r["share_positive"], complete=r["complete"],
                     n_expected_pairs=r.get("n_expected_pairs"))
        if r.get("pooled"):
            entry.update(pooled_mean=r["pooled"]["mean"], pooled_lo=r["pooled"]["lo"], pooled_hi=r["pooled"]["hi"],
                         datasets_positive=r["datasets_positive"], datasets_judged=r["datasets_judged"],
                         most_datasets_positive=r["most_datasets_positive"], indicative_verdict=r["indicative_verdict"])
        if r.get("why"):
            entry["reason"] = r["why"]
        if var["primary"]:
            continue
        sec[var["id"]] = entry
        if r.get("pooled"):
            lines.append(f"{r['label']}: {r['n_positive']} of {r['n_pairs']} pairs positive ({pct(r['share_positive'])}), "
                         f"pooled breadth-minus-depth {r['pooled']['mean']:+.4f} "
                         f"(95% interval {r['pooled']['lo']:+.4f} to {r['pooled']['hi']:+.4f}) -> {r['verdict']}"
                         + ("" if r["complete"] else f" (data incomplete; on present data: {r['indicative_verdict']})"))
        else:
            lines.append(f"{r['label']}: not testable ({r.get('why', 'no pairs')})")
    n["secondary"] = sec
    n["secondary_verdict_lines"] = lines
    n["depth_finding"] = finding
    res["numbers"] = n
    if not prim["n_pairs"]:
        res["verdict"] = "not testable"
    return res


# ================================================================ C2 / C3 shared: model-stage cells
def e2_expected(meta, ds):
    return expected_cells(meta, ds, E2_N, E2_K)


def run_c2(e2c, meta, rng):
    cell_rows = []
    reps_by_ds = defaultdict(list)       # draw-level bootstrap replicates of each cell's mean difference
    points_by_ds = defaultdict(list)
    w01 = dict(time=[0, 0], random=[0, 0])
    n_identical = n_few = n_noweights = 0
    for (ds, n, k) in sorted(e2c, key=lambda t: (E2_DATASETS.index(t[0]), t[1], korder(t[2]))):
        if n not in E2_N or k not in E2_K:
            continue
        cell = e2c[(ds, n, k)]
        pop, pers, hyt, hyr = cols_valid(cell, ["pop.U_ndcg", "pers_time.U_ndcg", "hyb_time.U_ndcg", "hyb_random.U_ndcg"])
        if pop.size < MIN_DRAWS:
            continue
        # (a) cell means; (b) paired against the pure list with the larger cell mean
        pers_wins = pers.mean() > pop.mean()
        comp = pers if pers_wins else pop
        a_gap = float(hyt.mean() - max(pop.mean(), pers.mean()))
        diff = hyt - comp
        mean, lo, hi = metrics.paired_bootstrap_ci(diff, n_boot=N_BOOT, rng=rng)
        rmt_mean, rmt_lo, rmt_hi = metrics.paired_bootstrap_ci(hyr - hyt, n_boot=N_BOOT, rng=rng)
        # (c) draws where random hold-out puts more / less weight on personalisation than time hold-out
        draws = [r for _, r in sorted(cell.items())]
        wr = np.array([to_float(r.get("hyb_random.w")) for r in draws])
        wt = np.array([to_float(r.get("hyb_time.w")) for r in draws])
        ok = np.isfinite(wr) & np.isfinite(wt)
        wr, wt = wr[ok], wt[ok]
        same_model = all(r.get("hyb_random.model") == r.get("hyb_time.model") for r, o in zip(draws, ok) if o)
        n_more = int(np.sum(wr > wt + W_TOL))
        n_less = int(np.sum(wr < wt - W_TOL))
        untied = n_more + n_less
        if wr.size == 0:
            status = "no_weights"
            n_noweights += 1
        elif same_model and untied == 0:
            status = "identical_splits"
            n_identical += 1
        elif untied < C2_MIN_UNTIED:
            status = "few_untied"
            n_few += 1
        else:
            status = "eligible"
        for tag, w in (("time", wt), ("random", wr)):
            w01[tag][0] += int(np.sum((w <= W_TOL) | (w >= 1 - W_TOL)))
            w01[tag][1] += int(w.size)
        row = dict(dataset=ds, N=n, k=kname(k), n_draws=int(pop.size), pop=float(pop.mean()),
                   pers_time=float(pers.mean()), hyb_time=float(hyt.mean()), hyb_random=float(hyr.mean()),
                   comparator="pers_time" if pers_wins else "pop", a_gap=a_gap,
                   diff_mean=float(mean), diff_lo=float(lo), diff_hi=float(hi),
                   a_ok=int(a_gap >= -C2_TOL - EPS), b_ok=int(lo > 0),
                   rand_minus_time=float(rmt_mean), rand_minus_time_lo=float(rmt_lo), rand_minus_time_hi=float(rmt_hi),
                   w_random=float(wr.mean()) if wr.size else math.nan, w_time=float(wt.mean()) if wt.size else math.nan,
                   w01_time=float(np.mean((wt <= W_TOL) | (wt >= 1 - W_TOL))) if wt.size else math.nan,
                   w01_random=float(np.mean((wr <= W_TOL) | (wr >= 1 - W_TOL))) if wr.size else math.nan,
                   draws_random_more=n_more, draws_time_more=n_less, draws_untied=untied, c_status=status,
                   c_ok=(int(n_more > n_less) if status == "eligible" else ""))
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
                            share_a=float(np.mean([r["a_ok"] for r in rr])), share_b=float(np.mean([r["b_ok"] for r in rr])),
                            n_eligible_c=len(cc), share_c=float(np.mean([r["c_ok"] for r in cc])) if cc else math.nan,
                            mean_diff=float(np.mean([r["diff_mean"] for r in rr])),
                            mean_rand_minus_time=float(np.mean([r["rand_minus_time"] for r in rr]))))
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
    crow = [r for r in cell_rows if r["c_status"] == "eligible"]
    nc, n_elig = sum(r["c_ok"] for r in crow), len(crow)
    a_lo, a_hi, a = share_bounds(na, n, missing)
    b_lo, b_hi, b = share_bounds(nb, n, missing)
    c_lo, c_hi, c = share_bounds(nc, n_elig, missing)          # missing cells could still be eligible
    sup = a_lo >= C2_A_SUPPORT and b_lo >= C2_B_SUPPORT and c_lo >= C2_C_SUPPORT      # nan compares False
    fals = a_hi < C2_A_FALSIFY or b_hi < C2_B_FALSIFY or c_hi <= C2_C_FALSIFY
    c_defined = n_elig > 0
    complete = missing == 0 and not unknown_open and c_defined
    pt, (lo2, hi2), _ = pool_two_level({ds: np.array(v) for ds, v in reps_by_ds.items()}, points_by_ds, N_BOOT, rng)
    n_ds_a_ok = sum(1 for r in ds_rows if r["share_a"] >= C2_A_SUPPORT)
    numbers.update(
        a_share_within_0p002=a, a_cells=na, b_share_above_comparator=b, b_cells=nb,
        c_share_random_more=c, c_cells=nc, c_cells_eligible=n_elig,
        c_cells_identical_splits=n_identical, c_cells_few_untied=n_few, c_cells_no_weights=n_noweights,
        a_share_range_if_missing=[a_lo, a_hi], b_share_range_if_missing=[b_lo, b_hi], c_share_range_if_missing=[c_lo, c_hi],
        prediction_a=bool(a_lo >= C2_A_SUPPORT), prediction_b=bool(b_lo >= C2_B_SUPPORT), prediction_c=bool(c_lo >= C2_C_SUPPORT),
        falsifier_a=bool(a_hi < C2_A_FALSIFY), falsifier_b=bool(b_hi < C2_B_FALSIFY), falsifier_c=bool(c_hi <= C2_C_FALSIFY),
        pooled_diff_mean=pt, pooled_diff_lo=lo2, pooled_diff_hi=hi2,
        n_datasets_positive_mean_diff=int(sum(1 for r in ds_rows if r["mean_diff"] > 0)), n_datasets=len(ds_rows),
        datasets_with_a_share_ge_90=n_ds_a_ok, most_datasets_a_ok=most(n_ds_a_ok, len(ds_rows)),     # A14, descriptive
        mean_rand_minus_time_panel=float(np.mean([r["rand_minus_time"] for r in cell_rows])),
        share_cells_random_panel_above_time=float(np.mean([r["rand_minus_time"] > 0 for r in cell_rows])),
        share_draws_w01_time=(w01["time"][0] / w01["time"][1]) if w01["time"][1] else None,
        share_draws_w01_random=(w01["random"][0] / w01["random"][1]) if w01["random"][1] else None,
        mean_w_random=float(np.nanmean([r["w_random"] for r in cell_rows])),
        mean_w_time=float(np.nanmean([r["w_time"] for r in cell_rows])),
        share_cells_comparator_pers_time=float(np.mean([r["comparator"] == "pers_time" for r in cell_rows])),
        min_draws=min(r["n_draws"] for r in cell_rows))
    sup_i = a >= C2_A_SUPPORT and b >= C2_B_SUPPORT and c >= C2_C_SUPPORT
    fals_i = a < C2_A_FALSIFY or b < C2_B_FALSIFY or (c <= C2_C_FALSIFY)
    numbers["indicative_verdict"] = decide(sup_i, fals_i, c_defined)
    res["verdict"] = decide(sup, fals, complete)
    if not c_defined and missing == 0 and not unknown_open:
        res["why"] = "no cell with at least 5 untied draws and non-identical splits, so part (c) cannot be judged"
    res["numbers"] = numbers
    res["complete"] = complete
    return res


def nstar_value(ns, d):
    """Smallest N whose interval is above 0 and whose mean difference stays > 0 at every larger N (INF if none).
    d: {N: (lo_positive, mean_positive)}."""
    best, ok_suffix = INF, True
    for n in sorted(ns, reverse=True):
        lo_pos, mean_pos = d[n]
        if lo_pos and ok_suffix:
            best = n
        ok_suffix = ok_suffix and mean_pos
    return best


MISSING_STATES = [(True, True), (False, True), (False, False)]


def series_outcomes(ns_expected, present):
    """Every N* a series could end with, whatever the missing cells would show."""
    missing = [n for n in ns_expected if n not in present]
    allns = sorted(set(ns_expected) | set(present))
    outs = set()
    for states in itertools.product(MISSING_STATES, repeat=len(missing)):
        d = dict(present)
        d.update(zip(missing, states))
        outs.add(nstar_value(allns, d))
    return outs


def c3_verdict(n_all, n_ten, viol, n_ds):
    sup = n_all >= C3_MIN_ALL and n_ten >= C3_MIN_TEN and not viol
    fals = (n_ds - n_all) >= C3_FALSIFY_ABSENT
    return "supported" if sup else "not supported" if fals else "mixed"


def run_c3(e2c, meta, cand, rng):
    cell_rows = []
    series = defaultdict(dict)           # (ds, k) -> {N: row}
    for (ds, n, k) in sorted(e2c, key=lambda t: (E2_DATASETS.index(t[0]), korder(t[2]), t[1])):
        if n not in E2_N or k not in C3_K:
            continue
        pop, pers = cols_valid(e2c[(ds, n, k)], ["pop.U_ndcg", "pers_time.U_ndcg"])
        if pop.size < MIN_DRAWS:
            continue
        diff = pers - pop
        mean, lo, hi = metrics.paired_bootstrap_ci(diff, n_boot=N_BOOT, rng=rng)
        row = dict(dataset=ds, k=kname(k), N=n, n_draws=int(pop.size), pop=float(pop.mean()),
                   pers_time=float(pers.mean()), diff_mean=float(mean), diff_lo=float(lo), diff_hi=float(hi),
                   above_zero=int(lo > 0), mean_above_zero=int(mean > 0))
        cell_rows.append(row)
        series[(ds, k)][n] = row

    n_max_tested = {}
    for (ds, n, k), cell in e2c.items():
        if len(cell) >= MIN_DRAWS:
            n_max_tested[ds] = max(n_max_tested.get(ds, 0), n)
    ds_with_data = [ds for ds in E2_DATASETS if any(key[0] == ds for key in series)]

    info = {}                           # (ds, k) -> dict(outcomes, indicative, in_grid, ns_expected)
    for ds in E2_DATASETS:
        for k in C3_K:
            if meta is not None and ds in meta:
                ec = e2_expected(meta, ds)
                ns_exp = sorted({n for n, kk in ec if kk == k})
                in_grid = bool(ns_exp)
            else:
                ds_ns = sorted({n for (d2, _), s in series.items() if d2 == ds for n in s})
                ns_exp, in_grid = (ds_ns if ds_ns else list(E2_N)), True
            present = {n: (bool(r["above_zero"]), bool(r["mean_above_zero"])) for n, r in series.get((ds, k), {}).items()}
            if not in_grid:
                outs, ind = {INF}, INF
            else:
                outs = series_outcomes(ns_exp, present)
                ind = nstar_value(sorted(present), present) if present else INF
            info[(ds, k)] = dict(outcomes=outs, indicative=ind, in_grid=in_grid, ns_expected=ns_exp, present=sorted(present))

    def status(ds, k):
        o = info[(ds, k)]["outcomes"]
        if len(o) == 1:
            return "none" if INF in o else "exists"
        return "open"

    def value(ds, k):
        o = info[(ds, k)]["outcomes"]
        return next(iter(o)) if len(o) == 1 else None

    # exact verdict: every reachable (n_all, n_10, violation) combination must give the same verdict
    n_ds = len(E2_DATASETS)
    states = {(0, 0, False)}
    for ds in E2_DATASETS:
        step = set()
        for oa in info[(ds, None)]["outcomes"]:
            for ot in info[(ds, 10)]["outcomes"]:
                step.add((oa < INF, ot < INF, oa > ot))
        states = {(a + int(x), t + int(y), v or z) for a, t, v in states for x, y, z in step}
    reach = {c3_verdict(a, t, v, n_ds) for a, t, v in states}
    # indicative: present data only, unresolved series as computed from present cells
    ind_all = sum(1 for ds in E2_DATASETS if info[(ds, None)]["indicative"] < INF)
    ind_ten = sum(1 for ds in E2_DATASETS if info[(ds, 10)]["indicative"] < INF)
    ind_viol = [ds for ds in E2_DATASETS if info[(ds, None)]["indicative"] > info[(ds, 10)]["indicative"]]
    indicative = c3_verdict(ind_all, ind_ten, bool(ind_viol), n_ds)
    open_series = sum(1 for ds in E2_DATASETS for k in C3_K if status(ds, k) == "open")
    complete = open_series == 0
    verdict = next(iter(reach)) if len(reach) == 1 else "not testable"
    if not cell_rows:
        verdict = "not testable"

    def text(ds, k):
        st = status(ds, k)
        if not info[(ds, k)]["in_grid"]:
            return "n/a (k not in grid)"
        nmax = n_max_tested.get(ds)
        if st == "exists":
            v = int(value(ds, k))
            lo_n = min(info[(ds, k)]["ns_expected"]) if info[(ds, k)]["ns_expected"] else None
            if lo_n is not None and v == lo_n:
                return f"<= {v} (smallest tested)"     # left-censored: the crossover may be lower
            if nmax is not None and v == nmax:
                return f"{v} (= largest tested)"       # nothing larger to confirm 'stays above 0'
            return str(v)
        if st == "none":
            return f"none (> {nmax})" if nmax is not None else "none (> N_max)"
        ind = info[(ds, k)]["indicative"]
        return f"open (so far {'none' if ind == INF else int(ind)})"

    wide_rows, long_rows, viol = [], [], []
    for ds in E2_DATASETS:
        sh = cand.get(ds)
        oa, ot = info[(ds, None)]["outcomes"], info[(ds, 10)]["outcomes"]
        if status(ds, None) != "open" and status(ds, 10) != "open":
            bad = value(ds, None) > value(ds, 10)
            incr = "yes" if bad else "no"
            if bad:
                viol.append(ds)
        else:
            incr = "open"
        wide_rows.append(dict(dataset=ds, N_star_k10=text(ds, 10), N_star_kall=text(ds, None),
                              n_max_tested=n_max_tested.get(ds, ""),
                              n_max_expected=max((n for n, _ in (e2_expected(meta, ds) or [])), default="") if meta else "",
                              n_star_increases_with_k=incr, cand_share=sh if sh is not None else "",
                              label=("cap-limited" if sh is not None and sh < CAP_LIMITED_BELOW else "")))
        for k in C3_K:
            long_rows.append(dict(dataset=ds, k=kname(k), status=status(ds, k) if ds in ds_with_data or info[(ds, k)]["in_grid"] is False
                                  else "missing",
                                  N_star=value(ds, k) if status(ds, k) == "exists" else "",
                                  n_max_tested=n_max_tested.get(ds, ""),
                                  n_cells_present=len(info[(ds, k)]["present"]),
                                  n_cells_expected=len(info[(ds, k)]["ns_expected"])))

    def tally(k):
        sts = [status(ds, k) for ds in E2_DATASETS]
        return {s: sts.count(s) for s in ("exists", "none", "open")}

    tall, tten = tally(None), tally(10)
    exists_all_ds = [ds for ds in E2_DATASETS if status(ds, None) == "exists"]
    exists_ten_ds = [ds for ds in E2_DATASETS if status(ds, 10) == "exists"]
    cap_limited = [ds for ds in E2_DATASETS if cand.get(ds) is not None and cand[ds] < CAP_LIMITED_BELOW]
    numbers = dict(
        n_datasets_expected=n_ds, n_datasets_with_data=len(ds_with_data),
        n_star_exists_kall=tall["exists"], n_star_none_kall=tall["none"], kall_open=tall["open"],
        n_star_exists_k10=tten["exists"], n_star_none_k10=tten["none"], k10_open=tten["open"],
        datasets_where_n_star_increases_with_k=viol,
        prediction_kall=bool(tall["exists"] >= C3_MIN_ALL), prediction_k10=bool(tten["exists"] >= C3_MIN_TEN),
        prediction_monotone=bool(not viol and open_series == 0), falsifier_kall=bool(tall["none"] >= C3_FALSIFY_ABSENT),
        most_datasets_exist_kall=most(tall["exists"], n_ds - tall["open"]),        # A14, descriptive
        most_datasets_exist_k10=most(tten["exists"], n_ds - tten["open"]),
        n_max_tested=dict(n_max_tested), cand_share=dict(cand), cap_limited_datasets=cap_limited,
        cand_share_note=("cand_share.csv found" if cand else "cand_share.csv not found; cap-limited labels not available"),
        n_star_table={f"{ds}|k={kname(k)}": text(ds, k) for ds in E2_DATASETS for k in C3_K},
        reachable_verdicts=sorted(reach), indicative_verdict=indicative,
        indicative_exists_kall=ind_all, indicative_exists_k10=ind_ten, indicative_violations=ind_viol,
        exists_kall_datasets=exists_all_ds, exists_k10_datasets=exists_ten_ds)
    res = dict(cells=cell_rows, nstar=wide_rows, long=long_rows, series=series, info=info, status=status, value=value,
               text=text, verdict=verdict, numbers=numbers, complete=complete, n_max_tested=n_max_tested)
    if not cell_rows:
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


PANEL_TITLES = {
    "count_A": "plain count vs Truth A (literal C1)",
    "ht_A": "ht (re-weighted) vs Truth A",
    "count_Au": "plain count vs Truth A_u (user-normalised)",
    "boot_count_Au": "boot arm: plain count vs Truth A_u",
}


def fig_c1_contrasts(c1, path):
    res = c1["variants"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.4), squeeze=False)
    any_data = False
    for ax, vid in zip(axes.ravel(), C1_PANELS):
        style(ax)
        r = res[vid]
        ax.set_title(PANEL_TITLES[vid], fontsize=11.5)
        if not r.get("pooled"):
            ax.text(0.5, 0.5, "not testable\n" + r.get("why", "no usable pairs").replace(" (rows", "\n(rows"),
                    ha="center", va="center", fontsize=10.5, color="#777777", transform=ax.transAxes, wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            continue
        any_data = True
        rows = r["forest"]
        ys = list(range(len(rows), 0, -1))
        for y, (ds, m, lo, hi) in zip(ys, rows):
            ax.plot([lo, hi], [y, y], "-", color="#8a8a8a", lw=2)
            ax.plot([m], [y], "o", color=COLORS[0], ms=6)
        p = r["pooled"]
        ax.plot([p["lo"], p["hi"]], [0, 0], "-", color=COLORS[1], lw=3.2)
        ax.plot([p["mean"]], [0], "D", color=COLORS[1], ms=8)
        ax.axvline(0, color="#444444", lw=1)
        if vid == "count_A":
            ax.axvline(C1_MIN_POOLED, color=COLORS[2], ls="--", lw=1.2)
            ax.text(C1_MIN_POOLED, len(rows) + 0.9, " +0.005 (support threshold)", color=COLORS[2], fontsize=8.5,
                    va="bottom", ha="left")
        ax.set_yticks([0] + ys)
        ax.set_yticklabels(["pooled (all datasets)"] + [x[0] for x in rows], fontsize=9.5)
        ax.set_ylim(-0.8, len(rows) + 1.4)
        ax.set_xlabel("breadth minus depth, NDCG@100 (positive = breadth better)", fontsize=10)
        ax.set_title(f"{PANEL_TITLES[vid]}\n{r['n_positive']} of {r['n_pairs']} pairs positive ({pct(r['share_positive'])}); "
                     f"pooled {p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]", fontsize=10.5)
    if not any_data:
        plt.close(fig)
        return False
    pv = res["count_A"]
    fig.suptitle("Breadth minus depth at equal budget N x k: same contrast for four columns\n"
                 f"literal C1 verdict: {pv['verdict']}. Plain counts at small k estimate the user-normalised list (A_u); "
                 "compare columns before reading", fontsize=12, y=0.998)
    fig.text(0.5, 0.005, "grey bar: 95% interval of the dataset's mean pair difference (over draws); orange diamond: pooled "
             "estimate with two-level 95% interval. Panels 1-3: widest cell N <= 5% of the group; boot arm: no restriction.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.025, 1, 0.95))
    save(fig, path)
    return True


def fig_c1_curves(c1, meta, path):
    src = c1["sources"]["e1"]
    cells = src["cells"]
    if not cells or not has_field(cells, "count.A_ndcg100"):
        return False
    dss = [ds for ds in E1_DATASETS if any(key[0] == ds for key in cells)]
    ncol = min(3, len(dss))
    nrow = math.ceil(len(dss) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(max(7.2, 3.6 * ncol), 3.0 * nrow + 1.5), squeeze=False)
    ks_all = sorted({key[2] for key in cells})
    cmap = {k: COLORS[i % len(COLORS)] for i, k in enumerate(ks_all)}
    for ax in axes.ravel():
        ax.set_visible(False)
    for i, ds in enumerate(dss):
        ax = axes[i // ncol][i % ncol]
        ax.set_visible(True)
        style(ax)
        lim = C1_NPOP_FRAC * meta[ds]["n_pop"] if meta is not None and ds in meta else math.inf
        for k in ks_all:
            pts = sorted((key[1], np.nanmean(col(c, "count.A_ndcg100"))) for key, c in cells.items()
                         if key[0] == ds and key[2] == k and len(c) >= MIN_DRAWS)
            if not pts:
                continue
            ax.plot([p[0] * k for p in pts], [p[1] for p in pts], "-", lw=1.4, color=cmap[k], label=f"k = {k}")
            inn = [p for p in pts if p[0] <= lim]
            out = [p for p in pts if p[0] > lim]
            ax.plot([p[0] * k for p in inn], [p[1] for p in inn], "o", ms=4, color=cmap[k])
            ax.plot([p[0] * k for p in out], [p[1] for p in out], "o", ms=4, mfc="white", color=cmap[k])
        ax.set_xscale("log")
        ax.set_title(ds, fontsize=11)
        if i // ncol == nrow - 1 or i + ncol >= len(dss):
            ax.set_xlabel("total rows kept, N x k (log scale)", fontsize=10)
        if i % ncol == 0:
            ax.set_ylabel("count NDCG@100 vs Truth A", fontsize=10)
    handles, labels = None, None
    for ax in axes.ravel():
        if ax.get_visible() and ax.get_legend_handles_labels()[0]:
            handles, labels = ax.get_legend_handles_labels()
            break
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=10, frameon=False,
                   title="k = choices kept per person. Filled marker: N <= 5% of the group (judged); open marker: N above 5% (not judged)",
                   title_fontsize=9.5)
    n = c1["numbers"]
    fig.suptitle(f"Literal C1 on the judged pairs: breadth better in {n['n_positive']} of {n['n_pairs']} dataset-budget pairs",
                 fontsize=12, y=0.995)
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
    fig.suptitle(f"Hybrid (time hold-out) is within 0.002 of the better pure list (cell means) in {pct(n['a_share_within_0p002'])} "
                 f"of cells and above it in {pct(n['b_share_above_comparator'])}; panels show k = all, bands 95% over draws",
                 fontsize=10.5, y=0.995)
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
    fig.suptitle(f"Draw-by-draw, random hold-out put more weight on personalisation than time hold-out in "
                 f"{pct(n['c_share_random_more'])} of {n['c_cells_eligible']} eligible cells "
                 f"({n['c_cells_identical_splits']} cells where both hold-outs chose the same model and weight in every draw excluded); panels show k = all",
                 fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    save(fig, path_b)
    out.append(path_b)
    return out


def fig_c3(c3, path_table, path_lines):
    out = []
    ds_data = [ds for ds in E2_DATASETS if any(key[0] == ds for key in c3["series"])]
    if not ds_data:
        return out
    colour = {"exists": "#c7e9c0", "none": "#f4cccc", "open": "#fff2cc"}
    texts, fills = [], []
    for ds in E2_DATASETS:
        wr = next(r for r in c3["nstar"] if r["dataset"] == ds)
        rt, rf = [ds], ["#ffffff"]
        for k in C3_K:
            st = c3["status"](ds, k)
            rt.append(c3["text"](ds, k))
            rf.append(colour[st] if c3["info"][(ds, k)]["in_grid"] else "#eeeeee")
        nm = wr["n_max_tested"]
        rt.append(str(nm) if nm != "" else "no data")
        rf.append("#ffffff")
        sh = wr["cand_share"]
        rt.append("n/a" if sh == "" else f"{sh:.2f}")
        rf.append("#ffe0b3" if wr["label"] else "#ffffff")
        rt.append(wr["label"] or ("" if sh != "" else "n/a"))
        rf.append("#ffe0b3" if wr["label"] else "#ffffff")
        texts.append(rt)
        fills.append(rf)
    fig, ax = plt.subplots(figsize=(11.2, 0.4 * len(E2_DATASETS) + 1.5))
    ax.axis("off")
    cols = ["dataset"] + [f"N* at k = {kname(k)}" for k in C3_K] + ["N_max tested", "cand_share", "label"]
    tb = ax.table(cellText=texts, cellColours=fills, colLabels=cols,
                  colWidths=[0.27, 0.17, 0.17, 0.12, 0.12, 0.15], loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(10.5)
    tb.scale(1, 1.6)
    n = c3["numbers"]
    ax.set_title(f"A crossover N* exists at k = all in {n['n_star_exists_kall']} of {n['n_datasets_expected']} datasets "
                 f"and at k = 10 in {n['n_star_exists_k10']}  (rule: at least {C3_MIN_ALL} and {C3_MIN_TEN}, no violation)",
                 fontsize=11.5, pad=10)
    fig.text(0.5, 0.01, "N* = smallest N where the 95% interval of personalised(time) - popularity on the panel is above 0 AND the mean "
             "difference stays above 0 at every larger tested N.\nnone (> N_max) = no such N in the tested grid; open = run incomplete; "
             "cap-limited = under half of the panel's future choices fall in the candidate universe (cand_share).",
             ha="center", fontsize=9)
    save(fig, path_table)
    out.append(path_table)

    ncol = min(4, len(ds_data))
    nrow = math.ceil(len(ds_data) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(max(7.0, 3.0 * ncol + 1.0), 3.2 * nrow + 1.3), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for i, ds in enumerate(ds_data):
        ax = axes[i // ncol][i % ncol]
        ax.set_visible(True)
        style(ax)
        ax.axhline(0, color="#444444", lw=1)
        title = ds
        for k, colr in ((None, COLORS[0]), (10, COLORS[1])):
            s = c3["series"].get((ds, k), {})
            if not s:
                continue
            ns = sorted(s)
            ax.plot(ns, [s[x]["diff_mean"] for x in ns], "-o", ms=3.5, lw=1.5, color=colr, label=f"k = {kname(k)}")
            ax.fill_between(ns, [s[x]["diff_lo"] for x in ns], [s[x]["diff_hi"] for x in ns], color=colr, alpha=0.15, lw=0)
            if c3["status"](ds, k) == "exists":
                ax.axvline(c3["value"](ds, k), color=colr, ls="--", lw=1.1)
        title += f"\nN*  k = 10: {c3['text'](ds, 10)}\nN*  k = all: {c3['text'](ds, None)}"
        ax.set_xscale("log")
        ax.set_title(title, fontsize=9)
        if i // ncol == nrow - 1 or i + ncol >= len(ds_data):
            ax.set_xlabel("people sampled, N (log scale)", fontsize=10)
        if i % ncol == 0:
            ax.set_ylabel("panel NDCG@100,\npersonalised - popularity", fontsize=10)
    h, l = None, None
    for ax in axes.ravel():
        if ax.get_visible() and ax.get_legend_handles_labels()[0]:
            h, l = ax.get_legend_handles_labels()
            break
    if h:
        fig.legend(h, l, loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.suptitle("Personalised(time) minus popularity on the panel (bands: 95% over draws; dashed lines: N*)", fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
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
    if res.get("verdict") not in ("not testable",):
        return ("", "")
    return (f" The data are incomplete ({have} of {wtxt} expected {unit[0]} present), so the verdict is withheld; "
            f"on what is present the answer would be '{ind}'.",
            f"数据还不完整（预期 {wtxt} 个{unit[1]}，目前只有 {have} 个），所以暂不下结论；按现有数据，结论会是“{ind}”。")


def build_verdicts(c1, c2, c3):
    out = []
    # ---- C1
    n = c1["numbers"]
    if "pooled_A" in n:
        pa = n["pooled_A"]
        pn_en, pn_zh = partial_note(c1, "C1")
        en = (f"Literal test (plain count, Truth A, widest cell N at most 5% of the group): breadth beat depth in {n['n_positive']} of "
              f"{n['n_pairs']} judged dataset-budget pairs ({pct(n['share_positive'])}; rule: at least 75% for support, at most 50% "
              f"falsifies, in between is inconclusive). Pooled difference (breadth minus depth) {pa['mean']:+.4f} NDCG@100 "
              f"(95% interval {pa['lo']:+.4f} to {pa['hi']:+.4f}; rule: interval above 0 and difference at least 0.005); "
              f"{n['datasets_positive']} of {n['datasets_judged']} datasets have a positive mean difference"
              f" ({'most' if n['most_datasets_positive'] else 'not most'} datasets). ")
        zh = (f"字面检验（普通计数，真值 A，只看最宽的格子 N 不超过群体 5% 的组合）：在 {n['n_pairs']} 个组合中，人多的名单更好的有 {n['n_positive']} 个"
              f"（{pct(n['share_positive'])}；支持要求至少 75%，不超过 50% 则被证伪，介于两者之间为无定论）。合并差（人多 减 记录多）为 {pa['mean']:+.4f}"
              f"（95% 区间 {pa['lo']:+.4f} 到 {pa['hi']:+.4f}；要求区间在 0 之上且差不小于 0.005）；"
              f"{n['datasets_judged']} 个数据集中有 {n['datasets_positive']} 个平均差为正。")
        if n["secondary_verdict_lines"]:
            en += "Secondary lines (same contrast, same rule): " + " | ".join(n["secondary_verdict_lines"]) + ". "
            zh += "次要结果（同一对比、同一规则）：" + "；".join(n["secondary_verdict_lines"]) + "。"
        if n["depth_finding"]["applies"]:
            en += ("Finding (A10): the literal C1 fails and the with-replacement boot arm shows depth at least as good as breadth, so depth is "
                   "not worse than breadth here, opposite to the cluster-sampling intuition"
                   + (" (provisional: the boot arm is incomplete). " if n["depth_finding"]["provisional"] else ". "))
            zh += "结论（A10）：字面 C1 不成立，且有放回的 boot 组显示“记录多”不比“人多”差；这与整群抽样的直觉相反" \
                  + ("（暂定：boot 组数据不完整）。" if n["depth_finding"]["provisional"] else "。")
        en += ("Mechanism (A9): plain counts at small k estimate the user-normalised list A_u rather than Truth A, so compare the "
               "count.A, ht.A and count.Au columns before interpreting the sign.")
        zh += "机制（A9）：k 较小时普通计数估计的是“按人归一化”的名单 A_u，而不是真值 A；解读符号之前请先对照 count.A、ht.A、count.Au 三列。"
        en += pn_en
        zh += pn_zh
    else:
        en = ("No judged (dataset, budget) pair with two usable cells (widest cell N at most 5% of the group) was found in the "
              "popularity stage (e1), so C1 could not be tested.")
        zh = "流行度阶段（e1）里没有找到可比较的（数据集，预算）组合（最宽格 N 不超过群体 5%），所以无法检验 C1。"
        if n["secondary_verdict_lines"]:
            en += " Secondary lines: " + " | ".join(n["secondary_verdict_lines"]) + "."
    out.append(dict(
        id="C1", title_en="C1. Breadth beats depth for estimating the list (amended rule A10)",
        title_zh="C1. 广度胜过深度：同样的记录总数，人多比每人记录多更能估准名单（修订规则 A10）",
        verdict=c1["verdict"], numbers=n, reading_en=en, reading_zh=zh,
        rule="difference (largest-N cell minus largest-k cell) of count NDCG@100 vs Truth A at equal budget N*k, judged only when the widest "
             "cell has N <= 5% of the group: supported if >= 75% of pairs positive AND pooled interval excludes 0 AND pooled difference >= 0.005; "
             "falsified if <= 50% positive OR pooled interval entirely below 0; otherwise inconclusive (mixed). Secondary lines: same contrast "
             "for ht.A, count.Au and the boot arm (no 5% restriction)"))
    # ---- C2
    n = c2["numbers"]
    if "a_share_within_0p002" in n:
        pn_en, pn_zh = partial_note(c2, "C2")
        en = (f"Over {n['n_cells']} cells, the time-hold-out hybrid was within 0.002 of the better pure list (comparator = the larger cell mean of "
              f"popularity and personalised_time) in {pct(n['a_share_within_0p002'])} (rule: at least 90%), above that list with a paired 95% interval "
              f"above 0 in {pct(n['b_share_above_comparator'])} (rule: at least 20%), and the random hold-out put more weight on personalisation "
              f"than the time hold-out in more draws in {pct(n['c_share_random_more'])} of {n['c_cells_eligible']} eligible cells "
              f"(rule: at least 70%; falsified at 50% or below; {n['c_cells_identical_splits']} cells where both hold-outs chose the same model and weight in every draw and "
              f"{n['c_cells_few_untied']} cells with fewer than 5 untied draws were excluded). "
              f"Mean panel NDCG@100 of hybrid_random minus hybrid_time per cell: {fnum(n['mean_rand_minus_time_panel'], '+.4f')}. "
              f"Share of draws where the hybrid picked w = 0 or 1: {pct(n['share_draws_w01_time'])} (time), "
              f"{pct(n['share_draws_w01_random'])} (random).")
        zh = (f"在 {n['n_cells']} 个格子中，用时间留出法选权重的混合名单与较好的单一名单（取流行度与个性化(时间)两者格子均值较大者）相差不超过 0.002 的占 "
              f"{pct(n['a_share_within_0p002'])}（要求至少 90%）；配对 95% 区间在 0 之上的占 {pct(n['b_share_above_comparator'])}（要求至少 20%）；"
              f"在 {n['c_cells_eligible']} 个可判定的格子中，随机留出法比时间留出法给个性化更大权重的抽样次数占多数的格子占 {pct(n['c_share_random_more'])}"
              f"（要求至少 70%，不超过 50% 则被证伪；另有 {n['c_cells_identical_splits']} 个格子因两种留出法每次都选了相同的模型和权重、{n['c_cells_few_untied']} 个格子因不并列的抽样少于 5 次而被排除）。")
        en += pn_en
        zh += pn_zh
        if c2.get("why"):
            en += " " + c2["why"] + "."
    else:
        en = "No model-stage cell with hybrid results was found in e2, so C2 could not be tested."
        zh = "模型训练阶段（e2）里没有找到带混合名单结果的格子，所以无法检验 C2。"
    out.append(dict(
        id="C2", title_en="C2. A hybrid with honest weights is never worse than the better pure list, and sometimes better than both (amended rule A11)",
        title_zh="C2. 只用样本自己选权重的混合名单，不比较好的单一名单差，有时还胜过两者（修订规则 A11）",
        verdict=c2["verdict"], numbers=n, reading_en=en, reading_zh=zh,
        rule="(a) hybrid_time cell mean >= larger of (popularity, personalised_time) cell means - 0.002 in >= 90% of cells; (b) paired-by-draw 95% "
             "interval of hybrid_time minus that pure list above 0 in >= 20% of cells; (c) per cell, draws with hybrid_random.w > hybrid_time.w vs <, "
             "ties dropped, >= 5 untied draws, identical-split cells excluded; 'random more' larger in >= 70% of eligible cells. Supported if a>=90%, "
             "b>=20%, c>=70%; falsified if a<75% or b<10% or c<=50%; else mixed"))
    # ---- C3
    n = c3["numbers"]
    if c3.get("cells"):
        pn_en, pn_zh = partial_note(c3, "C3")
        cap = n["cap_limited_datasets"]
        en = (f"A crossover size N* (personalised(time) beats popularity, 95% interval above 0 and mean difference above 0 at every larger tested N) "
              f"exists for k = all in {n['n_star_exists_kall']} of {n['n_datasets_expected']} datasets (rule: at least {C3_MIN_ALL}; absent in "
              f"{n['n_star_none_kall']}, falsified at {C3_FALSIFY_ABSENT}) and for k = 10 in {n['n_star_exists_k10']} (rule: at least {C3_MIN_TEN}). "
              + ("N*(all) does not exceed N*(10) in any dataset where both are known (no monotonicity violation)." if not n["datasets_where_n_star_increases_with_k"]
                 else f"N*(all) exceeds N*(10) in {', '.join(n['datasets_where_n_star_increases_with_k'])}.")
              + f" Largest N tested per dataset: {', '.join(f'{d} {v}' for d, v in n['n_max_tested'].items())}."
              + (f" Candidate-universe share below 0.5 (cap-limited): {', '.join(cap)}." if cap else
                 (" No dataset is cap-limited by the candidate-universe share." if n["cand_share"] else " Candidate-universe share not available.")))
        zh = (f"在 k = 全部 时，{n['n_datasets_expected']} 个数据集中有 {n['n_star_exists_kall']} 个存在交叉样本量 N*（个性化模型显著胜过流行度，且在更大的 N 上均值差保持为正；"
              f"要求至少 {C3_MIN_ALL} 个；至少 {C3_FALSIFY_ABSENT} 个不存在则被证伪）；k = 10 时有 {n['n_star_exists_k10']} 个（要求至少 {C3_MIN_TEN} 个）。"
              + ("N* 没有出现 k = 全部 大于 k = 10 的情况。" if not n["datasets_where_n_star_increases_with_k"]
                 else f"在 {', '.join(n['datasets_where_n_star_increases_with_k'])} 里，N*(全部) 大于 N*(10)。")
              + (f" 候选集占比低于 0.5（受候选集上限限制）：{', '.join(cap)}。" if cap else ""))
        en += pn_en
        zh += pn_zh
    else:
        en = "No model-stage cell with personalised results was found in e2, so C3 could not be tested."
        zh = "模型训练阶段（e2）里没有找到个性化模型的结果，所以无法检验 C3。"
    out.append(dict(
        id="C3", title_en="C3. A crossover sample size exists, and does not grow with k (amended rule A12)",
        title_zh="C3. 存在一个交叉样本量，并且 k = 全部 时不比 k = 10 时更大（修订规则 A12）",
        verdict=c3["verdict"], numbers=n, reading_en=en, reading_zh=zh,
        rule="N*(k), k in {10, all} = smallest N whose paired-by-draw 95% interval of personalised_time - popularity is above 0 AND whose mean "
             "difference stays > 0 at every larger tested N; absent counts as larger than any present N*; supported if N* exists at k=all in >= 5 of 7 "
             "and at k=10 in >= 4 of 7 and N*(all) > N*(10) never happens; falsified if absent at k=all in >= 4 of 7; else mixed"))
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
        print("note: datasets.json not found; the 5% rule and the expected grids are unknown, claims stay 'not testable'")
    cand = load_cand_share(runs)

    e1 = read_jsonl(os.path.join(runs, "e1.jsonl"))
    e2 = read_jsonl(os.path.join(runs, "e2.jsonl"))
    e7 = read_jsonl(os.path.join(runs, "e7.jsonl"))
    for name, rows in (("e1", e1), ("e2", e2), ("e7", e7)):
        print(f"{name}.jsonl: " + ("missing" if rows is None else f"{len(rows)} rows"))

    rng2, rng3 = (np.random.default_rng([SEED, i]) for i in (2, 3))
    c1 = run_c1(e1, e7, meta)
    if e1 is None:
        c1["why"] = "e1.jsonl not found in the runs directory"
        c1["numbers"]["reason"] = c1["why"]
    empty = dict(verdict="not testable", numbers={}, cells=[], datasets=[], nstar=[], long=[], series={})
    e2c = e2_cells(e2) if e2 is not None else {}
    if e2 is not None:
        c2 = run_c2(e2c, meta, rng2)
        c3 = run_c3(e2c, meta, cand, rng3)
    else:
        c2 = dict(empty, why="e2.jsonl not found in the runs directory")
        c3 = dict(empty, why="e2.jsonl not found in the runs directory")

    # ---- tables
    srcs = c1["sources"]
    for tag, src, fname in (("e1", srcs["e1"], "an_confirm2_c1_pairs.csv"), ("boot", srcs["boot"], "an_confirm2_c1_pairs_boot.csv")):
        if not src["pairs"]:
            continue
        f = ["dataset", "budget", "n_cells", "N_wide", "k_wide", "N_deep", "k_deep", "draws_wide", "draws_deep", "in_scope", "complete"]
        for var in C1_VARIANTS:
            if var["src"] == tag:
                f += [f"{var['id']}_{s}" for s in ("wide", "deep", "diff", "lo", "hi")]
        write_csv(os.path.join(tdir, fname), src["pairs"], f)
    crow = []
    for var in C1_VARIANTS:
        r = c1["variants"][var["id"]]
        p = r.get("pooled") or {}
        crow.append(dict(variant=var["id"], label=r["label"], column=r["field"], source=var["src"],
                         restricted_to_5pct=int(var["restrict"]), verdict=r["verdict"], complete=int(bool(r["complete"])),
                         n_pairs=r["n_pairs"], n_expected_pairs=r.get("n_expected_pairs", ""), n_positive=r["n_positive"],
                         share_positive=r["share_positive"], pooled_diff=p.get("mean", math.nan), pooled_lo=p.get("lo", math.nan),
                         pooled_hi=p.get("hi", math.nan), datasets_positive=r.get("datasets_positive", ""),
                         datasets_judged=r.get("datasets_judged", ""),
                         most_datasets_positive=r.get("most_datasets_positive", ""), reason=r.get("why", "")))
    write_csv(os.path.join(tdir, "an_confirm2_c1_contrasts.csv"), crow,
              ["variant", "label", "column", "source", "restricted_to_5pct", "verdict", "complete", "n_pairs", "n_expected_pairs",
               "n_positive", "share_positive", "pooled_diff", "pooled_lo", "pooled_hi", "datasets_positive", "datasets_judged",
               "most_datasets_positive", "reason"])
    drow = []
    for var in C1_VARIANTS:
        for d in c1["variants"][var["id"]]["datasets"]:
            drow.append(dict(d, variant=var["id"]))
    if drow:
        write_csv(os.path.join(tdir, "an_confirm2_c1_datasets.csv"), drow,
                  ["variant", "dataset", "n_pairs", "n_expected_pairs", "n_positive", "share_positive", "mean_diff"])
    if c2["cells"]:
        write_csv(os.path.join(tdir, "an_confirm2_c2_cells.csv"), c2["cells"],
                  ["dataset", "N", "k", "n_draws", "pop", "pers_time", "hyb_time", "hyb_random", "comparator", "a_gap", "diff_mean",
                   "diff_lo", "diff_hi", "a_ok", "b_ok", "rand_minus_time", "rand_minus_time_lo", "rand_minus_time_hi",
                   "w_random", "w_time", "w01_time", "w01_random", "draws_random_more", "draws_time_more", "draws_untied",
                   "c_status", "c_ok"])
        write_csv(os.path.join(tdir, "an_confirm2_c2_datasets.csv"), c2["datasets"],
                  ["dataset", "n_cells", "n_expected_cells", "share_a", "share_b", "n_eligible_c", "share_c", "mean_diff",
                   "mean_rand_minus_time"])
    if c3["cells"]:
        write_csv(os.path.join(tdir, "an_confirm2_c3_cells.csv"), c3["cells"],
                  ["dataset", "k", "N", "n_draws", "pop", "pers_time", "diff_mean", "diff_lo", "diff_hi", "above_zero", "mean_above_zero"])
        write_csv(os.path.join(tdir, "an_confirm2_c3_nstar.csv"), c3["nstar"],
                  ["dataset", "N_star_k10", "N_star_kall", "n_max_tested", "n_max_expected", "n_star_increases_with_k",
                   "cand_share", "label"])
        write_csv(os.path.join(tdir, "an_confirm2_c3_nstar_long.csv"), c3["long"],
                  ["dataset", "k", "status", "N_star", "n_max_tested", "n_cells_present", "n_cells_expected"])

    # ---- figures
    made = []
    if fig_c1_contrasts(c1, os.path.join(fdir, "F2-C1_contrasts.png")):
        made.append("F2-C1_contrasts.png")
    if fig_c1_curves(c1, meta, os.path.join(fdir, "F2-C1_count_curves.png")):
        made.append("F2-C1_count_curves.png")
    if c2["cells"]:
        for p in fig_c2(c2, e2c, meta, np.random.default_rng([SEED, 4]),
                        os.path.join(fdir, "F2-C2a_hybrid_vs_pure.png"), os.path.join(fdir, "F2-C2b_weight_random_vs_time.png")):
            made.append(os.path.basename(p))
    if c3["cells"]:
        for p in fig_c3(c3, os.path.join(fdir, "F2-C3a_crossover_table.png"), os.path.join(fdir, "F2-C3b_personalised_minus_popularity.png")):
            made.append(os.path.basename(p))

    verdicts = build_verdicts(c1, c2, c3)
    with open(os.path.join(tdir, "an_confirm2_verdicts.json"), "w", encoding="utf-8") as fh:
        json.dump(clean(verdicts), fh, ensure_ascii=False, indent=2)

    # ---- summary
    n1, n2, n3 = (v["numbers"] for v in verdicts)
    pa = n1.get("pooled_A", {})
    summ = [
        dict(claim="C1", verdict=verdicts[0]["verdict"], units=n1.get("n_pairs", ""), share_or_count=n1.get("share_positive", ""),
             headline_1_name="pairs_positive", headline_1=n1.get("n_positive", ""),
             headline_2_name="pooled_diff_A_ndcg100", headline_2=pa.get("mean", ""),
             headline_3_name="pooled_ci", headline_3=(f"[{pa['lo']:.4f}, {pa['hi']:.4f}]" if pa else "")),
    ]
    for var in C1_VARIANTS[1:]:
        r = c1["variants"][var["id"]]
        p = r.get("pooled")
        summ.append(dict(claim=f"C1:{var['id']}", verdict=r["verdict"], units=r["n_pairs"], share_or_count=r["share_positive"],
                         headline_1_name="pairs_positive", headline_1=r["n_positive"],
                         headline_2_name=f"pooled_diff_{r['field']}", headline_2=p["mean"] if p else "",
                         headline_3_name="pooled_ci", headline_3=(f"[{p['lo']:.4f}, {p['hi']:.4f}]" if p else r.get("why", ""))))
    summ += [
        dict(claim="C2", verdict=verdicts[1]["verdict"], units=n2.get("n_cells", ""), share_or_count=n2.get("a_share_within_0p002", ""),
             headline_1_name="share_a_within_0.002", headline_1=n2.get("a_share_within_0p002", ""),
             headline_2_name="share_b_above_comparator", headline_2=n2.get("b_share_above_comparator", ""),
             headline_3_name="share_c_random_more", headline_3=n2.get("c_share_random_more", "")),
        dict(claim="C3", verdict=verdicts[2]["verdict"], units=n3.get("n_datasets_with_data", ""),
             share_or_count=n3.get("n_star_exists_kall", ""),
             headline_1_name="datasets_with_N*_k_all", headline_1=n3.get("n_star_exists_kall", ""),
             headline_2_name="datasets_with_N*_k_10", headline_2=n3.get("n_star_exists_k10", ""),
             headline_3_name="datasets_N*_all_above_N*_10", headline_3=len(n3.get("datasets_where_n_star_increases_with_k", []))),
    ]
    write_csv(os.path.join(tdir, "an_confirm2_summary.csv"), summ,
              ["claim", "verdict", "units", "share_or_count", "headline_1_name", "headline_1", "headline_2_name",
               "headline_2", "headline_3_name", "headline_3"])

    for v in verdicts:
        print(f"{v['id']}: {v['verdict']} | {v['reading_en']}")
    print("figures: " + (", ".join(made) if made else "none"))
    print("tables in " + tdir)


if __name__ == "__main__":
    main()
