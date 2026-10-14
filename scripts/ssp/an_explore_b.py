#!/usr/bin/env python3
"""Exploratory families X5-X9 of the small-sample popularity pre-registration (section 7).

Every output of this script is labelled "exploratory".

  X5  rank confidence sets (e4): naive bootstrap intervals vs Mogstad et al. sets, coverage by rank band, widths
  X6  activity-biased samplers (e7 vs e1): uncorrected count vs Horvitz-Thompson count, crossover N
  X7  recency (e7 vs e1): sign flip between Truth A and Truth B when sampling recent choices
  X8  subgroups (e5): sample-only vs global vs shrunk/blend lists, N** vs distinctiveness
  X9  privacy (e6): effective sample size ratio of private mechanisms, head vs tail cost

Usage:
    python3 scripts/ssp/an_explore_b.py --runs <dir with e*.jsonl> --out <dir>

Writes <out>/03_tables/an_explore_b_*.csv, <out>/04_figures/an_explore_b_*.png and
<out>/03_tables/an_explore_b_verdicts.json. Missing stages / too few draws give "not testable".
All intervals are 95% percentile bootstraps over sampling draws (2,000 resamples, seeded);
people are never resampled.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import textwrap
import traceback
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics  # noqa: E402

NAME = "an_explore_b"
K_ALL = 10**6            # stands for k = null ("all earlier choices")
N_BOOT = 2000
NAN = float("nan")
STR_COLS = {"dataset", "group", "arm", "users", "rows", "key"}
INT_COLS = {"N", "draw"}
FIG_W = 6.9              # inches; 130 dpi -> about 900 px

# Pre-registered rule constants (fixed before looking at any number).
X5_NAIVE_BELOW = 0.93    # naive coverage in ranks 51-200 must be below this in most cells
X5_MOGSTAD_AT_LEAST = 0.95
X9_EPS_MIN = 2.0         # "central DP at eps >= 2"
X9_NREF_CENTRAL = 2000   # "once N is a few thousand"
X9_CENTRAL_FREE = 0.8    # "nearly free": effective N is at least 80% of the real N
X9_LOCAL_FACTOR = 30.0   # "~100x more people": read as at least 30x (a third of 100)
CENTRAL = ("laplace", "gaussian")
LOCAL = ("ldp_oue",)
MECHS = ("laplace", "gaussian", "ldp_oue")


# ----------------------------------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------------------------------
def klab(k: int) -> str:
    return "all" if k == K_ALL else str(k)


def nmean(x) -> float:
    a = np.asarray(x, dtype=np.float64)
    a = a[~np.isnan(a)]
    return float(a.mean()) if a.size else NAN


def nmedian(x) -> float:
    a = np.asarray(list(x), dtype=np.float64)
    a = a[~np.isnan(a)]
    return float(np.median(a)) if a.size else NAN


def fmt(x, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def clean(o):
    """Make a structure JSON-safe: numpy -> python, NaN/inf -> None, floats rounded."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def read_jsonl(path: Path, cols: list[str]) -> tuple[list[dict], int]:
    """Read only the wanted columns; k=null -> K_ALL; null numbers -> NaN; skip broken (partial) lines."""
    out: list[dict] = []
    bad = 0
    want_arm = "arm" in cols
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            rec: dict = {}
            for c in cols:
                v = r.get(c)
                if c == "k":
                    v = K_ALL if v is None else int(v)
                elif c == "arm":
                    continue
                elif c in STR_COLS:
                    v = "" if v is None else str(v)
                elif c in INT_COLS:
                    v = int(v)
                else:
                    v = NAN if v is None else float(v)
                rec[c] = v
            if want_arm:
                arm = r.get("arm")
                if not arm:
                    arm = "uniform"
                    for part in str(r.get("key", "")).split("|"):
                        if part.startswith("arm="):
                            arm = part[4:]
                rec["arm"] = arm
            out.append(rec)
    return out, bad


def collect(rows: list[dict], keyfn, cols: list[str]) -> dict:
    """key -> {col -> float array (one entry per row of that cell)}."""
    d: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = keyfn(r)
        for c in cols:
            d[key][c].append(r[c])
    return {k: {c: np.asarray(v, dtype=np.float64) for c, v in g.items()} for k, g in d.items()}


def valid(x: np.ndarray) -> np.ndarray:
    return x[~np.isnan(x)]


def boot_vec(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, x.size, size=(N_BOOT, x.size))
    return x[idx].mean(1)


def pct(v: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def combine(parts: list) -> str:
    """Verdict from sub-claims: True / False / None (not testable)."""
    t = [p for p in parts if p is not None]
    if not t:
        return "not testable"
    if all(t):
        return "supported" if len(t) == len(parts) else "mixed"
    if not any(t):
        return "not supported"
    return "mixed"


class Ctx:
    def __init__(self, runs: Path, out: Path, min_draws: int, seed: int):
        self.runs = runs
        self.out = out
        self.tables = out / "03_tables"
        self.figs = out / "04_figures"
        self.tables.mkdir(parents=True, exist_ok=True)
        self.figs.mkdir(parents=True, exist_ok=True)
        self.min_draws = min_draws
        self.seed = seed
        self.written: list[str] = []

    def rng(self, tag: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, tag])

    def table(self, name: str, rows: list[dict]):
        if not rows:
            return
        fields: list[str] = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        path = self.tables / f"{NAME}_{name}.csv"
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["label"] + fields)
            for r in rows:
                vals = []
                for f in fields:
                    v = r.get(f, "")
                    if isinstance(v, (float, np.floating)):
                        v = "" if (math.isnan(v) or math.isinf(v)) else f"{float(v):.6g}"
                    vals.append(v)
                w.writerow(["exploratory"] + vals)
        self.written.append(str(path))

    def finish(self, fig, name: str, title: str, handles=None, legend_ncol: int = 3):
        """Add finding title, shared legend and the 'exploratory' tag; save PNG at 130 dpi."""
        h = fig.get_size_inches()[1]
        wrapped = textwrap.fill(title, 78)
        top = 1.0
        bottom = 0.1 / h
        if handles:
            rows = math.ceil(len(handles) / legend_ncol)
            bottom = (0.2 * rows + 0.22) / h
        fig.suptitle(wrapped, fontsize=9.5, y=1.0 - 0.05 / h, va="top")
        fig.tight_layout(rect=(0, bottom, 1, top))
        if handles:
            fig.legend(handles=handles, loc="lower center", ncol=legend_ncol, fontsize=8, frameon=False,
                       bbox_to_anchor=(0.5, 0.12 / h))
        fig.text(0.995, 0.004, "exploratory", ha="right", va="bottom", fontsize=7, color="grey")
        path = self.figs / f"{NAME}_{name}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        self.written.append(str(path))


def grid(n: int, ncols: int = 3, h_per: float = 2.6, w: float = FIG_W, sharex=False, sharey=False):
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    if nrows > 3:
        h_per = min(h_per, 2.1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h_per * nrows + 1.6), squeeze=False,
                             sharex=sharex, sharey=sharey)
    flat = list(axes.ravel())
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat[:n]


def style(ax, xlabel=None, ylabel=None, title=None):
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=7.5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    if title:
        ax.set_title(title, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def verdict(id_, title_en, title_zh, v, numbers, en, zh, rule):
    return {"id": id_, "title_en": title_en, "title_zh": title_zh, "verdict": v, "numbers": clean(numbers),
            "reading_en": en, "reading_zh": zh, "rule": rule, "label": "exploratory"}


def not_testable(id_, title_en, title_zh, why_en, why_zh, rule, numbers=None):
    return verdict(id_, title_en, title_zh, "not testable", numbers or {}, f"Not testable: {why_en}.",
                   f"无法检验：{why_zh}。", rule)


def load(ctx: Ctx, fname: str, cols: list[str], notes: list[str]):
    p = ctx.runs / fname
    if not p.exists():
        notes.append(f"{fname} is missing")
        return None
    rows, bad = read_jsonl(p, cols)
    if bad:
        notes.append(f"{fname}: skipped {bad} unreadable line(s)")
    if not rows:
        notes.append(f"{fname} has no rows")
        return None
    print(f"  loaded {fname}: {len(rows)} rows")
    return rows


# ----------------------------------------------------------------------------------------------
# X5  rank confidence sets
# ----------------------------------------------------------------------------------------------
X5_RULE = ("Naive coverage below 0.93 in the ranks 51-200 band in most cells (> 50% of (dataset,N,k) cells); "
           "Mogstad coverage at least 0.95 in most cells.")
X5_TITLE = ("X5 (exploratory): rank intervals", "X5（探索性）：排名置信区间")


def x5(ctx: Ctx):
    notes: list[str] = []
    stats = ("cov", "width", "width_top10", "cov_1_10", "cov_11_50", "cov_51_200")
    cols = ["dataset", "N", "k", "draw"] + [f"{m}.{s}" for m in ("naive", "mogstad") for s in stats]
    rows = load(ctx, "e4.jsonl", cols, notes)
    if rows is None:
        return not_testable("X5", *X5_TITLE, "the rank-set stage (e4) has not produced rows yet",
                            "排名置信区间阶段（e4）还没有数据", X5_RULE)
    rng = ctx.rng(5)
    groups = collect(rows, lambda r: (r["dataset"], r["k"], r["N"]), cols[4:])
    tab: list[dict] = []
    skipped = 0
    for (ds, k, N), g in sorted(groups.items()):
        n = g["naive.cov"].size
        if n < ctx.min_draws:
            skipped += 1
            continue
        rec = {"dataset": ds, "k": klab(k), "N": N, "n_draws": n}
        for m in ("naive", "mogstad"):
            for s in stats:
                rec[f"{m}_{s}"] = nmean(g[f"{m}.{s}"])
        for m, s in (("naive", "cov_51_200"), ("mogstad", "cov_51_200"), ("mogstad", "cov")):
            x = valid(g[f"{m}.{s}"])
            if x.size:
                _, lo, hi = metrics.paired_bootstrap_ci(x, N_BOOT, rng=rng)
                rec[f"{m}_{s}_lo"], rec[f"{m}_{s}_hi"] = lo, hi
        tab.append(rec)
    ctx.table("x5_cells", tab)
    if skipped:
        notes.append(f"{skipped} cell(s) dropped for fewer than {ctx.min_draws} draws")
    if not tab:
        return not_testable("X5", *X5_TITLE, f"no (dataset,N,k) cell has {ctx.min_draws} or more draws",
                            f"没有任何格子有 {ctx.min_draws} 次以上的抽样", X5_RULE)

    naive_band = np.array([r["naive_cov_51_200"] for r in tab])
    mog_cov = np.array([r["mogstad_cov"] for r in tab])
    mog_band = np.array([r["mogstad_cov_51_200"] for r in tab])
    ok_n, ok_m = ~np.isnan(naive_band), ~np.isnan(mog_cov)
    frac_naive = float(np.mean(naive_band[ok_n] < X5_NAIVE_BELOW)) if ok_n.any() else NAN
    frac_mog = float(np.mean(mog_cov[ok_m] >= X5_MOGSTAD_AT_LEAST)) if ok_m.any() else NAN
    frac_mog_band = float(np.mean(mog_band[~np.isnan(mog_band)] >= X5_MOGSTAD_AT_LEAST)) if (~np.isnan(mog_band)).any() else NAN
    part_n = None if math.isnan(frac_naive) else frac_naive > 0.5
    part_m = None if math.isnan(frac_mog) else frac_mog > 0.5
    v = combine([part_n, part_m])
    wr = [r["mogstad_width"] / r["naive_width"] for r in tab if r["naive_width"] > 0]
    nums = {"n_cells": len(tab), "n_datasets": len({r["dataset"] for r in tab}),
            "median_draws_per_cell": nmedian(r["n_draws"] for r in tab),
            "share_cells_naive_51_200_below_0.93": frac_naive,
            "share_cells_mogstad_cov_at_least_0.95": frac_mog,
            "share_cells_mogstad_51_200_at_least_0.95": frac_mog_band,
            "median_naive_cov": nmedian(r["naive_cov"] for r in tab),
            "median_naive_cov_51_200": nmedian(naive_band),
            "median_naive_cov_1_10": nmedian(r["naive_cov_1_10"] for r in tab),
            "median_mogstad_cov": nmedian(mog_cov),
            "median_mogstad_cov_51_200": nmedian(mog_band),
            "median_width_ratio_mogstad_over_naive": nmedian(wr),
            "notes": notes}
    en = (f"Across {len(tab)} cells, the naive intervals cover only {fmt(nums['median_naive_cov_51_200'], 2)} of true "
          f"ranks 51-200 (median) and fall below 0.93 in {frac_naive * 100:.0f}% of cells; the Mogstad sets cover "
          f"{fmt(nums['median_mogstad_cov'], 2)} (median) and reach 0.95 in {frac_mog * 100:.0f}% of cells. "
          f"The Mogstad sets are {fmt(nums['median_width_ratio_mogstad_over_naive'], 1)} times as wide as the naive "
          f"intervals (median).")
    zh = (f"在 {len(tab)} 个格子里，朴素区间对真实第 51-200 名的覆盖率中位数只有 {fmt(nums['median_naive_cov_51_200'], 2)}，"
          f"{frac_naive * 100:.0f}% 的格子低于 0.93；Mogstad 区间覆盖率中位数 {fmt(nums['median_mogstad_cov'], 2)}，"
          f"{frac_mog * 100:.0f}% 的格子达到 0.95。Mogstad 区间的宽度约为朴素区间的 "
          f"{fmt(nums['median_width_ratio_mogstad_over_naive'], 1)} 倍。")
    plot_x5(ctx, tab, nums)
    return verdict("X5", "X5 (exploratory): coverage and width of rank intervals, naive bootstrap versus Mogstad sets",
                   "X5（探索性）：排名区间的覆盖率与宽度，朴素自助法与 Mogstad 区间对比",
                   v, nums, en, zh, X5_RULE)


def plot_x5(ctx: Ctx, tab: list[dict], nums: dict):
    ks = sorted({r["k"] for r in tab}, key=lambda s: K_ALL if s == "all" else int(s))
    bands = (("cov_1_10", "ranks 1-10", ":"), ("cov_11_50", "ranks 11-50", "--"), ("cov_51_200", "ranks 51-200", "-"))
    cols = {"naive": "tab:red", "mogstad": "tab:blue"}

    def by_n(k, key):
        d = defaultdict(list)
        for r in tab:
            if r["k"] == k and not math.isnan(r[key]):
                d[r["N"]].append(r[key])
        xs = sorted(d)
        return xs, [float(np.mean(d[x])) for x in xs]

    fig, axes = grid(len(ks), ncols=3, sharey=True)
    for ax, k in zip(axes, ks):
        for m, c in cols.items():
            for key, _, ls in bands:
                xs, ys = by_n(k, f"{m}_{key}")
                if xs:
                    ax.plot(xs, ys, ls=ls, color=c, marker="o", ms=3, lw=1.3)
        ax.axhline(0.95, color="grey", lw=0.8, ls="-.")
        ax.set_xscale("log")
        lo_y = min([r[f"{m}_{b}"] for r in tab for m in cols for b, _, _ in bands
                    if not math.isnan(r[f"{m}_{b}"])] + [0.95])
        ax.set_ylim(max(0.0, lo_y - 0.05), 1.03)
        style(ax, "N (people sampled)", "coverage of true rank", f"k = {k}")
    handles = [Line2D([], [], color=cols[m], ls=ls, marker="o", ms=3, label=f"{m.capitalize()}, {lab}")
               for m in cols for _, lab, ls in bands]
    handles.append(Line2D([], [], color="grey", ls="-.", label="nominal 0.95"))
    title = (f"Naive intervals cover {fmt(nums['median_naive_cov_51_200'], 2)} of true ranks 51-200 (median over cells); "
             f"Mogstad sets cover {fmt(nums['median_mogstad_cov_51_200'], 2)}. Exploratory.")
    ctx.finish(fig, "x5_coverage", title, handles, legend_ncol=3)

    fig, axes = grid(len(ks), ncols=3, sharey=True)
    for ax, k in zip(axes, ks):
        first = None
        for m, c in cols.items():
            xs, ys = by_n(k, f"{m}_width")
            if xs:
                ax.plot(xs, ys, color=c, marker="o", ms=3, lw=1.3)
                if m == "naive":
                    first = (xs[0], ys[0], xs)
        if first:
            x0, y0, xs = first
            ref = np.array(xs, dtype=float)
            ax.plot(ref, y0 * (ref / x0) ** -0.5, color="grey", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        style(ax, "N (people sampled)", "mean interval width (rank positions)", f"k = {k}")
    handles = [Line2D([], [], color="tab:red", marker="o", ms=3, label="Naive bootstrap"),
               Line2D([], [], color="tab:blue", marker="o", ms=3, label="Mogstad et al. set"),
               Line2D([], [], color="grey", ls="--", label="slope -1/2 reference (anchored at naive, smallest N)")]
    title = (f"Mogstad sets are {fmt(nums['median_width_ratio_mogstad_over_naive'], 1)} times as wide as naive "
             f"intervals (median over cells); mean width per N on log-log axes, dashed line = N^-1/2. Exploratory.")
    ctx.finish(fig, "x5_widths", title, handles, legend_ncol=3)


# ----------------------------------------------------------------------------------------------
# X6  biased samplers
# ----------------------------------------------------------------------------------------------
X6_RULE = ("A crossover N exists in most datasets: the weighted (HT) count beats the uncorrected count (95% CI of the "
           "paired difference above 0) from some N on, while the uncorrected count is not behind at the smallest N.")
X6_TITLE = ("X6 (exploratory): biased samplers, uncorrected vs weighted count",
            "X6（探索性）：有偏抽样下，未校正计数与加权计数")
BIAS_ARMS = ("activity", "inv_activity")


def x6(ctx: Ctx):
    notes: list[str] = []
    c7 = ["dataset", "N", "k", "draw", "arm", "count.A_ndcg100", "ht.A_ndcg100"]
    c1 = ["dataset", "N", "k", "draw", "arm", "count.A_ndcg100"]
    r7 = load(ctx, "e7.jsonl", c7, notes)
    r1 = load(ctx, "e1.jsonl", c1, notes)
    if r7 is None:
        return not_testable("X6", *X6_TITLE, "the biased-sampler stage (e7) has not produced rows yet",
                            "有偏抽样阶段（e7）还没有数据", X6_RULE)
    r7 = [r for r in r7 if r["arm"] in BIAS_ARMS]
    if not r7:
        return not_testable("X6", *X6_TITLE, "e7 has no 'activity' / 'inv_activity' rows yet",
                            "e7 里还没有 activity / inv_activity 行", X6_RULE)
    uni = collect([r for r in (r1 or []) if r["arm"] == "uniform"], lambda r: (r["dataset"], r["k"], r["N"]),
                  ["count.A_ndcg100"])
    if r1 is None:
        notes.append("uniform reference (e1) missing: plots lack the unbiased curve")
    bia = collect(r7, lambda r: (r["arm"], r["dataset"], r["k"], r["N"]), ["count.A_ndcg100", "ht.A_ndcg100"])
    rng = ctx.rng(6)
    curves: list[dict] = []
    series: dict = defaultdict(list)
    for (arm, ds, k, N), g in sorted(bia.items()):
        cnt, ht = g["count.A_ndcg100"], g["ht.A_ndcg100"]
        ok = ~np.isnan(cnt) & ~np.isnan(ht)
        if ok.sum() < ctx.min_draws:
            continue
        cnt, ht = cnt[ok], ht[ok]
        mc, lc, hc = metrics.paired_bootstrap_ci(cnt, N_BOOT, rng=rng)
        mh, lh, hh = metrics.paired_bootstrap_ci(ht, N_BOOT, rng=rng)
        md, ld, hd = metrics.paired_bootstrap_ci(ht - cnt, N_BOOT, rng=rng)
        rec = {"arm": arm, "dataset": ds, "k": klab(k), "N": N, "n_draws": int(ok.sum()),
               "biased_count": mc, "biased_count_lo": lc, "biased_count_hi": hc,
               "biased_ht": mh, "biased_ht_lo": lh, "biased_ht_hi": hh,
               "ht_minus_count": md, "ht_minus_count_lo": ld, "ht_minus_count_hi": hd}
        u = uni.get((ds, k, N))
        if u is not None and valid(u["count.A_ndcg100"]).size >= ctx.min_draws:
            mu, lu, hu = metrics.paired_bootstrap_ci(u["count.A_ndcg100"], N_BOOT, rng=rng)
            rec.update({"uniform_count": mu, "uniform_count_lo": lu, "uniform_count_hi": hu})
        curves.append(rec)
        series[(arm, ds, k)].append(rec)
    ctx.table("x6_curves", curves)
    if not curves:
        return not_testable("X6", *X6_TITLE, f"no biased cell has {ctx.min_draws} or more draws with HT values",
                            f"没有任何有偏格子同时具有 {ctx.min_draws} 次以上抽样与 HT 数值", X6_RULE,
                            {"notes": notes})
    cross: list[dict] = []
    for (arm, ds, k), recs in sorted(series.items()):
        recs = sorted(recs, key=lambda r: r["N"])
        if len(recs) < 2:
            continue
        first_pos = next((r for r in recs if r["ht_minus_count_lo"] > 0), None)
        count_ahead_small = recs[0]["ht_minus_count"] <= 0
        if first_pos is None:
            status, cN = "never", "never"
        else:
            cN = first_pos["N"]
            status = "crossover" if (count_ahead_small and cN != recs[0]["N"]) else "ht_ahead_from_smallest_N"
        cross.append({"arm": arm, "dataset": ds, "k": klab(k), "n_N_points": len(recs),
                      "smallest_N": recs[0]["N"], "diff_at_smallest_N": recs[0]["ht_minus_count"],
                      "largest_N": recs[-1]["N"], "diff_at_largest_N": recs[-1]["ht_minus_count"],
                      "crossover_N": cN, "status": status})
    ctx.table("x6_crossover", cross)
    if not cross:
        return not_testable("X6", *X6_TITLE, "no (arm, dataset, k) series has two or more N values with enough draws",
                            "没有任何系列拥有两个以上带足够抽样数的 N", X6_RULE)

    per_arm: dict = {}
    for arm in BIAS_ARMS:
        cs = [c for c in cross if c["arm"] == arm]
        if not cs:
            continue
        dsets = sorted({c["dataset"] for c in cs})
        has = []
        for ds in dsets:
            ss = [c for c in cs if c["dataset"] == ds]
            has.append(np.mean([c["status"] == "crossover" for c in ss]) > 0.5)
        per_arm[arm] = {"n_series": len(cs), "n_series_crossover": sum(c["status"] == "crossover" for c in cs),
                        "n_series_never": sum(c["status"] == "never" for c in cs),
                        "n_series_ht_ahead_from_start": sum(c["status"] == "ht_ahead_from_smallest_N" for c in cs),
                        "n_datasets": len(dsets), "n_datasets_with_crossover": int(sum(has)),
                        "median_crossover_N": nmedian(c["crossover_N"] for c in cs if c["status"] == "crossover")}
    parts = [per_arm[a]["n_datasets_with_crossover"] / per_arm[a]["n_datasets"] > 0.5 for a in per_arm]
    if len(parts) == 2:
        v = "supported" if all(parts) else ("not supported" if not any(parts) else "mixed")
    else:
        v = "supported" if parts[0] else "not supported"
        notes.append(f"only arm '{list(per_arm)[0]}' is available")
    nums = {"per_arm": per_arm, "notes": notes}
    bits_en = "; ".join(f"{a}: {p['n_datasets_with_crossover']} of {p['n_datasets']} datasets "
                        f"({p['n_series_crossover']} of {p['n_series']} dataset-k series)" for a, p in per_arm.items())
    bits_zh = "；".join(f"{a}：{p['n_datasets']} 个数据集中有 {p['n_datasets_with_crossover']} 个"
                       f"（{p['n_series']} 条数据集-k 系列中有 {p['n_series_crossover']} 条）" for a, p in per_arm.items())
    med = nmedian(c["crossover_N"] for c in cross if c["status"] == "crossover")
    en = (f"A crossover (uncorrected count ahead or level at the smallest N, weighted count clearly ahead from some N on) "
          f"was found in: {bits_en}. Where it exists, the median crossover is at N = {fmt(med, 0)}; "
          f"{sum(c['status'] == 'never' for c in cross)} series never show the weighted count winning.")
    zh = (f"出现“交叉点”（最小 N 时未校正计数不落后，之后加权计数明显领先）的情况：{bits_zh}。"
          f"有交叉点时，交叉 N 的中位数约为 {fmt(med, 0)}；有 {sum(c['status'] == 'never' for c in cross)} 条系列加权计数始终没有赢。")
    plot_x6(ctx, curves, cross, per_arm)
    return verdict("X6", "X6 (exploratory): activity-biased samplers, uncorrected count versus weighted (HT) count",
                   "X6（探索性）：按活跃度有偏抽样下，未校正计数与加权（HT）计数对比",
                   v, nums, en, zh, X6_RULE)


def plot_x6(ctx: Ctx, curves: list[dict], cross: list[dict], per_arm: dict):
    for arm, p in per_arm.items():
        cs = [c for c in curves if c["arm"] == arm]
        keys = sorted({(c["dataset"], c["k"]) for c in cs},
                      key=lambda t: (t[0], K_ALL if t[1] == "all" else int(t[1])))
        fig, axes = grid(len(keys), ncols=3, sharey=False)
        for ax, (ds, k) in zip(axes, keys):
            pts = sorted((c for c in cs if c["dataset"] == ds and c["k"] == k), key=lambda c: c["N"])
            xs = np.array([c["N"] for c in pts], dtype=float)

            def line(key, color, ls, label=None):
                ys = np.array([c.get(key, NAN) for c in pts])
                lo = np.array([c.get(key + "_lo", NAN) for c in pts])
                hi = np.array([c.get(key + "_hi", NAN) for c in pts])
                if np.isnan(ys).all():
                    return
                ax.plot(xs, ys, color=color, ls=ls, marker="o", ms=3, lw=1.3)
                ax.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0)

            line("uniform_count", "tab:gray", "-")
            line("biased_count", "tab:orange", "-")
            line("biased_ht", "tab:blue", "--")
            cr = next((c for c in cross if c["arm"] == arm and c["dataset"] == ds and c["k"] == k), None)
            if cr and cr["crossover_N"] != "never":
                ax.axvline(cr["crossover_N"], color="tab:green", ls=":", lw=1.2)
            ax.set_xscale("log")
            style(ax, "N (people sampled)", "NDCG@100 vs Truth A", f"{ds}, k={k}")
            ax.title.set_fontsize(7.5)
        handles = [Line2D([], [], color="tab:gray", marker="o", ms=3, label="uniform sampling, count"),
                   Line2D([], [], color="tab:orange", marker="o", ms=3, label=f"{arm} sampling, uncorrected count"),
                   Line2D([], [], color="tab:blue", ls="--", marker="o", ms=3, label=f"{arm} sampling, weighted (HT) count"),
                   Line2D([], [], color="tab:green", ls=":", label="first N where weighted wins (95% CI)")]
        title = (f"Under {arm.replace('_', '-')}-biased sampling the weighted count beats the plain count (95% CI) at "
                 f"some N in {p['n_series'] - p['n_series_never']} of {p['n_series']} dataset-k series; "
                 f"the crossover pattern holds in {p['n_series_crossover']}. Exploratory.")
        ctx.finish(fig, f"x6_{arm}", title, handles, legend_ncol=2)


# ----------------------------------------------------------------------------------------------
# X7  recency
# ----------------------------------------------------------------------------------------------
X7_RULE = ("Opposite signs in most datasets: recent-choice samples are worse than uniform ones on Truth A and better on "
           "Truth B (difference of mean NDCG@100, averaged over the (N,k) cells).")
X7_TITLE = ("X7 (exploratory): sampling recent choices", "X7（探索性）：抽取最近的选择")
RECENT_ARMS = ("recent_rows", "recent_window")
X7_METRICS = (("A", "count.A_ndcg100"), ("B", "count.B_ndcg100"), ("U", "count.U_ndcg"))


def x7(ctx: Ctx):
    notes: list[str] = []
    cols = ["dataset", "N", "k", "draw", "arm"] + [m for _, m in X7_METRICS]
    r7 = load(ctx, "e7.jsonl", cols, notes)
    r1 = load(ctx, "e1.jsonl", cols, notes)
    if r7 is None or r1 is None:
        return not_testable("X7", *X7_TITLE, "needs both the biased-sampler stage (e7) and the main grid (e1): "
                            + "; ".join(notes), "需要 e7 和 e1 两个阶段的数据", X7_RULE)
    rec = collect([r for r in r7 if r["arm"] in RECENT_ARMS], lambda r: (r["arm"], r["dataset"], r["k"], r["N"]),
                  [m for _, m in X7_METRICS])
    uni = collect([r for r in r1 if r["arm"] == "uniform"], lambda r: (r["dataset"], r["k"], r["N"]),
                  [m for _, m in X7_METRICS])
    if not rec:
        return not_testable("X7", *X7_TITLE, "e7 has no 'recent_rows' / 'recent_window' rows yet",
                            "e7 里还没有 recent_rows / recent_window 行", X7_RULE)
    rng = ctx.rng(7)
    cells: list[dict] = []
    acc: dict = {}  # (arm, ds, metric) -> [sum vec, n cells, sum point]
    for (arm, ds, k, N), g in sorted(rec.items()):
        u = uni.get((ds, k, N))
        if u is None:
            continue
        row = {"arm": arm, "dataset": ds, "k": klab(k), "N": N}
        for tag, m in X7_METRICS:
            xr, xu = valid(g[m]), valid(u[m])
            if xr.size < ctx.min_draws or xu.size < ctx.min_draws:
                continue
            vec = boot_vec(xr, rng) - boot_vec(xu, rng)
            d = float(xr.mean() - xu.mean())
            lo, hi = pct(vec)
            row.update({f"n_recent_{tag}": xr.size, f"n_uniform_{tag}": xu.size, f"d{tag}": d,
                        f"d{tag}_lo": lo, f"d{tag}_hi": hi})
            a = acc.setdefault((arm, ds, tag), [np.zeros(N_BOOT), 0, 0.0])
            a[0] += vec
            a[1] += 1
            a[2] += d
        if "dA" in row or "dB" in row or "dU" in row:
            cells.append(row)
    ctx.table("x7_cells", cells)
    if not cells:
        return not_testable("X7", *X7_TITLE, f"no (arm,dataset,N,k) cell has {ctx.min_draws} or more draws in both "
                            "e7 and e1", f"没有任何格子在 e7 和 e1 中同时有 {ctx.min_draws} 次以上抽样", X7_RULE)
    ds_rows: list[dict] = []
    for arm in RECENT_ARMS:
        for ds in sorted({k[1] for k in acc if k[0] == arm}):
            row = {"arm": arm, "dataset": ds}
            for tag, _ in X7_METRICS:
                a = acc.get((arm, ds, tag))
                if a is None or a[1] == 0:
                    continue
                lo, hi = pct(a[0] / a[1])
                row.update({f"n_cells_{tag}": a[1], f"d{tag}": a[2] / a[1], f"d{tag}_lo": lo, f"d{tag}_hi": hi})
            if "dA" in row and "dB" in row:
                row["flip_B"] = bool(row["dA"] < 0 < row["dB"])
            if "dA" in row and "dU" in row:
                row["flip_U"] = bool(row["dA"] < 0 < row["dU"])
            ds_rows.append(row)
    ctx.table("x7_datasets", ds_rows)

    per_arm: dict = {}
    brng = ctx.rng(70)
    for arm in RECENT_ARMS:
        rs = [r for r in ds_rows if r["arm"] == arm and "flip_B" in r]
        if not rs:
            continue
        p = {"n_datasets": len(rs), "n_flip_B": int(sum(r["flip_B"] for r in rs)),
             "n_flip_U": int(sum(r.get("flip_U", False) for r in rs)),
             "n_datasets_with_U": int(sum("flip_U" in r for r in rs))}
        for tag, _ in X7_METRICS:
            vals = np.array([r[f"d{tag}"] for r in rs if f"d{tag}" in r])
            p[f"pooled_d{tag}"] = float(vals.mean()) if vals.size else NAN
            p[f"n_datasets_positive_{tag}"] = int((vals > 0).sum())
            if vals.size >= 3:
                bv = vals[brng.integers(0, vals.size, size=(N_BOOT, vals.size))].mean(1)
                p[f"pooled_d{tag}_lo"], p[f"pooled_d{tag}_hi"] = pct(bv)
        per_arm[arm] = p
    if not per_arm:
        return not_testable("X7", *X7_TITLE, "no dataset has both a Truth A and a Truth B difference",
                            "没有数据集同时具有真值 A 与真值 B 的差值", X7_RULE)
    parts = [p["n_flip_B"] / p["n_datasets"] > 0.5 for p in per_arm.values()]
    v = "supported" if all(parts) else ("not supported" if not any(parts) else "mixed")
    if len(per_arm) < 2:
        notes.append(f"only arm '{list(per_arm)[0]}' is available")
    nums = {"per_arm": per_arm, "notes": notes}
    bits_en = "; ".join(f"{a}: Truth A {fmt(p['pooled_dA'], 3)}, Truth B {fmt(p['pooled_dB'], 3)} (mean difference in "
                        f"NDCG@100, recent minus uniform), sign flip in {p['n_flip_B']} of {p['n_datasets']} datasets"
                        for a, p in per_arm.items())
    bits_zh = "；".join(f"{a}：真值 A 差 {fmt(p['pooled_dA'], 3)}，真值 B 差 {fmt(p['pooled_dB'], 3)}"
                       f"（最近 减 均匀，NDCG@100 均值），{p['n_datasets']} 个数据集中 {p['n_flip_B']} 个符号相反"
                       for a, p in per_arm.items())
    en = f"{bits_en}. A sign flip means worse on Truth A and better on Truth B."
    zh = f"{bits_zh}。符号相反指对真值 A 更差、对真值 B 更好。"
    plot_x7(ctx, ds_rows, per_arm)
    return verdict("X7", "X7 (exploratory): recent-choice samples versus uniform samples on Truth A and Truth B",
                   "X7（探索性）：最近选择的样本与均匀样本在真值 A、真值 B 上的比较", v, nums, en, zh, X7_RULE)


def plot_x7(ctx: Ctx, ds_rows: list[dict], per_arm: dict):
    arms = list(per_arm)
    fig, axes = plt.subplots(len(arms), 1, figsize=(FIG_W, 3.1 * len(arms) + 1.2), squeeze=False)
    spec = (("A", "tab:orange", "Truth A (full earlier community list, all choices before T0)"),
            ("B", "tab:blue", "Truth B (community's later choices)"),
            ("U", "tab:green", "Truth B (held-out panel's later choices)"))
    for ax, arm in zip(axes[:, 0], arms):
        rs = [r for r in ds_rows if r["arm"] == arm]
        w = 0.27
        for j, (tag, color, _) in enumerate(spec):
            xs, ys, err = [], [], [[], []]
            for i, r in enumerate(rs):
                if f"d{tag}" in r:
                    xs.append(i + (j - 1) * w)
                    ys.append(r[f"d{tag}"])
                    err[0].append(max(0.0, r[f"d{tag}"] - r[f"d{tag}_lo"]))
                    err[1].append(max(0.0, r[f"d{tag}_hi"] - r[f"d{tag}"]))
            ax.bar(xs, ys, w, color=color, yerr=err, capsize=2, error_kw={"lw": 0.8})
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(rs)))
        ax.set_xticklabels([r["dataset"] for r in rs], rotation=40, ha="right", fontsize=7.5)
        style(ax, None, "NDCG@100: recent minus uniform", arm.replace("_", " "))
    handles = [Line2D([], [], color=c, lw=6, label=lab) for _, c, lab in spec]
    parts = [f"{a.replace('_', ' ')}: A {fmt(p['pooled_dA'], 3)}, B {fmt(p['pooled_dB'], 3)}, "
             f"flip in {p['n_flip_B']}/{p['n_datasets']} datasets" for a, p in per_arm.items()]
    title = "Recent-choice samples minus uniform samples, mean over (N,k) cells (" + "; ".join(parts) + "). Exploratory."
    ctx.finish(fig, "x7_recency", title, handles, legend_ncol=1)


# ----------------------------------------------------------------------------------------------
# X8  subgroups
# ----------------------------------------------------------------------------------------------
X8_RULE = ("Positive rank correlation (Spearman, across groups) between distinctiveness (1 - group_vs_global_ov100) and "
           "N** (smallest N where the sample-only list beats the global list). Supported if rho > 0 and the 95% "
           "bootstrap interval (over groups) excludes 0; mixed if rho > 0 but the interval includes 0.")
X8_TITLE = ("X8 (exploratory): subgroups", "X8（探索性）：子群体")
EST8 = ("sample", "global", "shrunk", "blend25", "blend50", "blend75")


def nstar(cells: list[dict], metric: str, ci: bool):
    """Smallest N where mean(sample) > mean(global) (ci=False) or the paired 95% CI of the difference is above 0."""
    for c in cells:
        if ci:
            if c[f"diff_{metric}_lo"] > 0:
                return c["N"]
        elif c[f"sample_{metric}"] > c[f"global_{metric}"]:
            return c["N"]
    return None


def x8(ctx: Ctx):
    notes: list[str] = []
    cols = ["dataset", "group", "N", "k", "draw", "group_vs_global_ov100"] + \
           [f"{e}.{m}" for e in EST8 for m in ("ndcg100", "ov100")]
    rows = load(ctx, "e5.jsonl", cols, notes)
    if rows is None:
        return not_testable("X8", *X8_TITLE, "the subgroup stage (e5) has not produced rows yet",
                            "子群体阶段（e5）还没有数据", X8_RULE)
    rng = ctx.rng(8)
    groups = collect(rows, lambda r: (r["dataset"], r["group"], r["k"], r["N"]), cols[5:])
    dist_acc: dict = defaultdict(list)
    for r in rows:
        dist_acc[(r["dataset"], r["group"])].append(r["group_vs_global_ov100"])
    distinct = {k: 1.0 - nmean(v) for k, v in dist_acc.items()}

    curves: list[dict] = []
    units: dict = defaultdict(list)
    for (ds, grp, k, N), g in sorted(groups.items()):
        n = g["sample.ndcg100"].size
        if n < ctx.min_draws:
            continue
        rec = {"dataset": ds, "group": grp, "k": klab(k), "N": N, "n_draws": n,
               "distinctiveness": distinct[(ds, grp)]}
        for e in EST8:
            for m in ("ndcg100", "ov100"):
                rec[f"{e}_{m}"] = nmean(g[f"{e}.{m}"])
        for m in ("ndcg100", "ov100"):
            d = g[f"sample.{m}"] - g[f"global.{m}"]
            if valid(d).size >= ctx.min_draws:
                _, lo, hi = metrics.paired_bootstrap_ci(d, N_BOOT, rng=rng)
            else:
                lo = hi = NAN
            rec[f"diff_{m}_lo"], rec[f"diff_{m}_hi"] = lo, hi
        curves.append(rec)
        units[(ds, grp, k)].append(rec)
    ctx.table("x8_curves", curves)
    if not curves:
        return not_testable("X8", *X8_TITLE, f"no subgroup cell has {ctx.min_draws} or more draws",
                            f"没有任何子群体格子有 {ctx.min_draws} 次以上抽样", X8_RULE)

    max_n = max(c["N"] for c in curves)
    cens = 2 * max_n
    urows: list[dict] = []
    for (ds, grp, k), cs in sorted(units.items()):
        cs = sorted(cs, key=lambda c: c["N"])
        if len(cs) < 3:
            continue
        row = {"dataset": ds, "group": grp, "k": klab(k), "n_N_points": len(cs), "smallest_N": cs[0]["N"],
               "largest_N": cs[-1]["N"], "distinctiveness": distinct[(ds, grp)]}
        for m in ("ndcg100", "ov100"):
            a, b = nstar(cs, m, False), nstar(cs, m, True)
            row[f"Nstar2_{m}"] = a if a is not None else "never"
            row[f"Nstar2_ci_{m}"] = b if b is not None else "never"
            gaps = [c[f"shrunk_{m}"] - max(c[f"sample_{m}"], c[f"global_{m}"]) for c in cs]
            row[f"shrunk_ok_cells_{m}"] = int(sum(g >= 0 for g in gaps))
            row[f"shrunk_all_ok_{m}"] = bool(all(g >= 0 for g in gaps))
            row[f"shrunk_worst_gap_{m}"] = min(gaps)
        urows.append(row)
    ctx.table("x8_nstar", urows)
    if len(urows) < 4:
        return not_testable("X8", *X8_TITLE, f"only {len(urows)} (dataset, group, k) series with 3 or more N values; "
                            "at least 4 needed for a rank correlation", "可用的（数据集、群体、k）系列少于 4 条，无法算秩相关",
                            X8_RULE)

    def rho_of(rs, key):
        x = np.array([r["distinctiveness"] for r in rs])
        y = np.array([cens if r[key] == "never" else r[key] for r in rs], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(spearmanr(x, y).statistic) if x.size >= 3 else NAN

    def rho_boot(rs, key):
        cl = np.array([f"{r['dataset']}|{r['group']}" for r in rs])
        uq = np.unique(cl)
        mem = {c: np.where(cl == c)[0] for c in uq}
        x = np.array([r["distinctiveness"] for r in rs])
        y = np.array([cens if r[key] == "never" else r[key] for r in rs], dtype=float)
        vals = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(N_BOOT):
                pick = rng.integers(0, uq.size, size=uq.size)
                idx = np.concatenate([mem[uq[i]] for i in pick])
                vals.append(spearmanr(x[idx], y[idx]).statistic)
        vals = np.array(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        return pct(vals) if vals.size > 20 else (NAN, NAN)

    rho = rho_of(urows, "Nstar2_ndcg100")
    lo, hi = rho_boot(urows, "Nstar2_ndcg100")
    rho_ov = rho_of(urows, "Nstar2_ov100")
    ks = sorted({r["k"] for r in urows}, key=lambda s: K_ALL if s == "all" else int(s))
    rho_by_k = {k: rho_of([r for r in urows if r["k"] == k], "Nstar2_ndcg100") for k in ks}
    n_never = sum(r["Nstar2_ndcg100"] == "never" for r in urows)
    sh_all = float(np.mean([r["shrunk_all_ok_ndcg100"] for r in urows]))
    n_cells = sum(r["n_N_points"] for r in urows)
    sh_cell = sum(r["shrunk_ok_cells_ndcg100"] for r in urows) / n_cells
    sh_all_ov = float(np.mean([r["shrunk_all_ok_ov100"] for r in urows]))
    if math.isnan(rho):
        v = "not testable"
    elif rho > 0 and lo > 0:
        v = "supported"
    elif rho > 0:
        v = "mixed"
    else:
        v = "not supported"
    nums = {"n_series": len(urows), "n_groups": len({(r["dataset"], r["group"]) for r in urows}),
            "n_datasets": len({r["dataset"] for r in urows}), "spearman_rho_ndcg100": rho,
            "rho_ci_lo": lo, "rho_ci_hi": hi, "spearman_rho_ov100": rho_ov, "rho_by_k": rho_by_k,
            "n_series_sample_never_beats_global_ndcg100": n_never,
            "censoring_value_used_for_never": cens,
            "share_series_shrunk_at_least_best_of_both_at_every_N_ndcg100": sh_all,
            "share_cells_shrunk_at_least_best_of_both_ndcg100": sh_cell,
            "share_series_shrunk_at_least_best_of_both_at_every_N_ov100": sh_all_ov, "notes": notes}
    en = (f"Across {len(urows)} (group, k) series, the Spearman correlation between distinctiveness and N** is "
          f"{fmt(rho, 2)} (95% interval {fmt(lo, 2)} to {fmt(hi, 2)}); the sample-only list never beat the global list "
          f"in {n_never} series (counted as above the largest N). The shrunk list was at least as good as the better of "
          f"sample and global at every N in {sh_all * 100:.0f}% of series ({sh_cell * 100:.0f}% of individual cells).")
    zh = (f"在 {len(urows)} 条（群体，k）系列中，差异度与 N** 的 Spearman 相关为 {fmt(rho, 2)}"
          f"（95% 区间 {fmt(lo, 2)} 到 {fmt(hi, 2)}）；有 {n_never} 条系列中仅用样本的榜单始终没有超过全局榜单"
          f"（记为大于最大 N）。收缩榜单在每个 N 上都不差于样本榜与全局榜中较好者的系列占 {sh_all * 100:.0f}%"
          f"（单个格子占 {sh_cell * 100:.0f}%）。")
    plot_x8(ctx, curves, urows, distinct, cens, rho, lo, hi)
    return verdict("X8", "X8 (exploratory): subgroup lists, sample-only versus global versus shrunk, and N** versus distinctiveness",
                   "X8（探索性）：子群体榜单（仅样本、全局、收缩）以及 N** 与差异度的关系", v, nums, en, zh, X8_RULE)


def plot_x8(ctx: Ctx, curves, urows, distinct, cens, rho, lo, hi):
    ks_num = sorted({r["k"] for r in urows if r["k"] != "all"}, key=int)
    kk = ks_num[0] if ks_num else "all"
    cand: dict = defaultdict(list)
    for c in curves:
        if c["k"] == kk:
            cand[(c["dataset"], c["group"])].append(c)
    cand = {k: sorted(v, key=lambda c: c["N"]) for k, v in cand.items() if len(v) >= 3}
    order = sorted(cand, key=lambda g: distinct[g], reverse=True)
    pick: list = []
    for g in order[:2] + order[::-1][:2]:
        if g not in pick:
            pick.append(g)
    fig, axes = grid(len(pick), ncols=2, h_per=2.6)
    spec = (("sample", "tab:red", "-", "sample only"), ("global", "tab:gray", "-", "global list"),
            ("shrunk", "tab:green", "-", "shrunk (sample + global)"), ("blend50", "tab:blue", "--", "50/50 blend"))
    for ax, g in zip(axes, pick):
        cs = cand[g]
        xs = [c["N"] for c in cs]
        for e, col, ls, _ in spec:
            ax.plot(xs, [c[f"{e}_ndcg100"] for c in cs], color=col, ls=ls, marker="o", ms=3, lw=1.3)
        ax.set_xscale("log")
        style(ax, "N (subgroup people sampled)", "NDCG@100 vs subgroup's own list",
              f"{g[0]} / {g[1]} (distinctiveness {distinct[g]:.2f}, k={kk})")
        ax.title.set_fontsize(7.5)
    handles = [Line2D([], [], color=c, ls=ls, marker="o", ms=3, label=lab) for _, c, ls, lab in spec]
    title = (f"For the two most and two least distinctive subgroups (k={kk}): sample-only, global, shrunk and blended "
             f"lists against the subgroup's own full list. Exploratory.")
    ctx.finish(fig, "x8_curves", title, handles, legend_ncol=4)

    fig, ax = plt.subplots(figsize=(FIG_W, 4.6))
    fam_col = {"genre": "tab:purple", "activity": "tab:orange", "joined": "tab:green"}
    ks = sorted({r["k"] for r in urows}, key=lambda s: K_ALL if s == "all" else int(s))
    marks = ["o", "s", "^", "D", "v", "P"]
    for r in urows:
        y = cens if r["Nstar2_ndcg100"] == "never" else r["Nstar2_ndcg100"]
        fam = r["group"].split(":")[0]
        ax.scatter(r["distinctiveness"], y, color=fam_col.get(fam, "black"), marker=marks[ks.index(r["k"]) % len(marks)],
                   facecolors="none" if r["Nstar2_ndcg100"] == "never" else fam_col.get(fam, "black"), s=30, alpha=0.8,
                   linewidths=1)
    ns = sorted({c["N"] for c in curves})
    ax.set_yscale("log")
    ax.set_yticks(ns + [cens])
    ax.set_yticklabels([str(n) for n in ns] + ["never"])
    ax.minorticks_off()
    style(ax, "distinctiveness = 1 - overlap of subgroup list with the global list (top 100)",
          "N** (people), smallest N where sample beats global")
    handles = [Line2D([], [], color=c, marker="o", ls="", label=f) for f, c in fam_col.items()
               if any(r["group"].startswith(f) for r in urows)]
    handles += [Line2D([], [], color="grey", marker=marks[i % len(marks)], ls="", label=f"k = {k}")
                for i, k in enumerate(ks)]
    handles.append(Line2D([], [], color="grey", marker="o", mfc="none", ls="", label="never (hollow)"))
    title = (f"Spearman correlation of N** with distinctiveness: {fmt(rho, 2)} "
             f"(95% interval {fmt(lo, 2)} to {fmt(hi, 2)}) over {len(urows)} series. Exploratory.")
    ctx.finish(fig, "x8_nstar_vs_distinctiveness", title, handles, legend_ncol=4)


# ----------------------------------------------------------------------------------------------
# X9  privacy
# ----------------------------------------------------------------------------------------------
X9_RULE = ("Statements in sample-size ratios, per dataset. Operational reading (fixed in advance): (a) head: private "
           "relative loss in ov10 exceeds that in ov100 in most cells; (b) central DP (laplace, gaussian) at eps >= 2 and "
           "N >= 2000 has median effective-N ratio >= 0.8; (c) local DP (ldp_oue) at eps >= 2 needs a median of at least "
           "30x more people.")
X9_TITLE = ("X9 (exploratory): privacy cost in sample sizes", "X9（探索性）：隐私代价（用样本量衡量）")


def effective_n(target: float, ns: np.ndarray, ys: np.ndarray):
    """N where the (monotone) nonprivate curve reaches `target`, interpolated in log N; returns (N_eff, flag)."""
    i = int(np.argmax(ys >= target)) if (ys >= target).any() else -1
    if i == -1:
        return float(ns[-1]), "above_curve"
    if i == 0:
        return float(ns[0]), "below_curve"
    t = (target - ys[i - 1]) / (ys[i] - ys[i - 1])
    return float(math.exp(math.log(ns[i - 1]) + t * (math.log(ns[i]) - math.log(ns[i - 1])))), ""


def x9(ctx: Ctx):
    notes: list[str] = []
    mets = ("ndcg100", "ov10", "ov100")
    cols = ["dataset", "N", "k", "eps", "draw"] + [f"{m}.{x}" for m in ("nonprivate",) + MECHS for x in mets]
    rows = load(ctx, "e6.jsonl", cols, notes)
    if rows is None:
        return not_testable("X9", *X9_TITLE, "the privacy stage (e6) has not produced rows yet",
                            "隐私阶段（e6）还没有数据", X9_RULE)
    e1 = None
    p1 = ctx.runs / "e1.jsonl"
    if p1.exists():
        e1, bad = read_jsonl(p1, ["dataset", "N", "k", "draw", "arm", "count.A_ndcg100"])
        e1 = [r for r in e1 if r["arm"] == "uniform"]
    else:
        notes.append("e1.jsonl missing: nonprivate curve uses e6 N values only")
    rng = ctx.rng(9)
    cell_cols = [f"{m}.{x}" for m in ("nonprivate",) + MECHS for x in mets]
    cells = collect(rows, lambda r: (r["dataset"], r["k"], r["eps"], r["N"]), cell_cols)
    npool = collect(rows, lambda r: (r["dataset"], r["k"], r["N"]),
                    [f"nonprivate.{x}" for x in mets])  # nonprivate is independent of eps: pool all eps
    e1c = collect(e1 or [], lambda r: (r["dataset"], r["k"], r["N"]), ["count.A_ndcg100"])
    curve: dict = {}
    for (ds, k, N), g in npool.items():
        x = valid(g["nonprivate.ndcg100"])
        if x.size >= ctx.min_draws:
            curve.setdefault((ds, k), {})[N] = (float(x.mean()), "e6")
    for (ds, k, N), g in e1c.items():
        x = valid(g["count.A_ndcg100"])
        if x.size >= ctx.min_draws and N not in curve.get((ds, k), {}):
            curve.setdefault((ds, k), {})[N] = (float(x.mean()), "e1")
    curve_arr = {}
    for key, d in curve.items():
        ns = np.array(sorted(d), dtype=float)
        ys = np.maximum.accumulate(np.array([d[int(n)][0] for n in ns]))  # enforce a non-decreasing curve
        curve_arr[key] = (ns, ys)

    ratio_rows: list[dict] = []
    ht_rows: list[dict] = []
    cell_means: list[dict] = []
    for (ds, k, eps, N), g in sorted(cells.items()):
        ncurve = curve_arr.get((ds, k))
        xn = valid(g["nonprivate.ndcg100"])
        if xn.size >= ctx.min_draws:
            cell_means.append({"dataset": ds, "k": klab(k), "eps": eps, "N": N, "mechanism": "nonprivate",
                               "n_draws": xn.size, "ndcg100": float(xn.mean())})
        for mech in MECHS:
            x = valid(g[f"{mech}.ndcg100"])
            if x.size < ctx.min_draws:
                continue
            m, lo, hi = metrics.paired_bootstrap_ci(x, N_BOOT, rng=rng)
            cell_means.append({"dataset": ds, "k": klab(k), "eps": eps, "N": N, "mechanism": mech, "n_draws": x.size,
                               "ndcg100": m, "ndcg100_lo": lo, "ndcg100_hi": hi})
            if ncurve is not None and ncurve[0].size >= 2:
                ne, flag = effective_n(m, *ncurve)
                ne_lo, _ = effective_n(lo, *ncurve)
                ne_hi, _ = effective_n(hi, *ncurve)
                ratio_rows.append({"dataset": ds, "k": klab(k), "eps": eps, "N_ref": N, "mechanism": mech,
                                   "klass": "central" if mech in CENTRAL else "local", "n_draws": x.size,
                                   "private_ndcg100": m, "N_eff": ne, "ratio": ne / N, "ratio_lo": ne_lo / N,
                                   "ratio_hi": ne_hi / N, "cost_factor": N / ne,
                                   "censored": flag if flag else "no",
                                   "curve_N_min": float(ncurve[0][0]), "curve_N_max": float(ncurve[0][-1])})
            npg = npool.get((ds, k, N))
            if npg is not None:
                n10, n100 = nmean(npg["nonprivate.ov10"]), nmean(npg["nonprivate.ov100"])
                p10, p100 = nmean(g[f"{mech}.ov10"]), nmean(g[f"{mech}.ov100"])
                if n10 > 0 and n100 > 0 and not math.isnan(p10) and not math.isnan(p100):
                    l10, l100 = 1 - p10 / n10, 1 - p100 / n100
                    ht_rows.append({"dataset": ds, "k": klab(k), "eps": eps, "N": N, "mechanism": mech,
                                    "nonprivate_ov10": n10, "private_ov10": p10, "nonprivate_ov100": n100,
                                    "private_ov100": p100, "loss_head_ov10": l10, "loss_tail_ov100": l100,
                                    "head_loss_larger": bool(l10 > l100)})
    ctx.table("x9_cell_means", cell_means)
    ctx.table("x9_ratios", ratio_rows)
    ctx.table("x9_head_tail", ht_rows)
    curve_tab = [{"dataset": ds, "k": klab(k), "N": int(n), "nonprivate_ndcg100_monotone": y,
                  "source": curve[(ds, k)][int(n)][1]}
                 for (ds, k), (ns, ys) in sorted(curve_arr.items()) for n, y in zip(ns, ys)]
    ctx.table("x9_nonprivate_curve", curve_tab)
    if not ratio_rows:
        return not_testable("X9", *X9_TITLE, f"no private cell has {ctx.min_draws} or more draws and a nonprivate curve "
                            "with two N values", f"没有任何隐私格子有 {ctx.min_draws} 次以上抽样和可用的非隐私曲线",
                            X9_RULE, {"notes": notes})

    summ: list[dict] = []
    keyf = lambda r: (r["dataset"], r["mechanism"], r["eps"])  # noqa: E731
    gg: dict = defaultdict(list)
    for r in ratio_rows:
        gg[keyf(r)].append(r)
    for (ds, mech, eps), rs in sorted(gg.items()):
        summ.append({"dataset": ds, "mechanism": mech, "klass": rs[0]["klass"], "eps": eps, "n_cells": len(rs),
                     "median_ratio": nmedian(r["ratio"] for r in rs),
                     "median_cost_factor": nmedian(r["cost_factor"] for r in rs),
                     "share_censored": float(np.mean([r["censored"] != "no" for r in rs]))})
    ctx.table("x9_summary", summ)

    def sel(klass, eps_min=None, n_min=None):
        return [r for r in ratio_rows if r["klass"] == klass and (eps_min is None or r["eps"] >= eps_min)
                and (n_min is None or r["N_ref"] >= n_min)]

    cen_big = sel("central", X9_EPS_MIN, X9_NREF_CENTRAL)
    cen_all = sel("central", X9_EPS_MIN)
    loc = sel("local", X9_EPS_MIN)
    med_cen_big = nmedian(r["ratio"] for r in cen_big)
    med_cen_all = nmedian(r["ratio"] for r in cen_all)
    med_loc_f = nmedian(r["cost_factor"] for r in loc)
    med_loc_r = nmedian(r["ratio"] for r in loc)
    def frac(rows_, flag):
        return float(np.mean([r["censored"] == flag for r in rows_])) if rows_ else 0.0

    # A median that is only a bound cannot confirm (ratio, below the curve) or refute (cost, below the curve) a threshold.
    if not cen_big:
        part_c = None
    elif med_cen_big >= X9_CENTRAL_FREE:
        part_c = None if frac(cen_big, "below_curve") >= 0.5 else True
    else:
        part_c = None if frac(cen_big, "above_curve") >= 0.5 else False
    if not loc:
        part_l = None
    elif med_loc_f >= X9_LOCAL_FACTOR:
        part_l = None if frac(loc, "above_curve") >= 0.5 else True
    else:
        part_l = None if frac(loc, "below_curve") >= 0.5 else False
    head_share = float(np.mean([r["head_loss_larger"] for r in ht_rows])) if ht_rows else NAN
    part_h = None if math.isnan(head_share) else head_share > 0.5
    v = combine([part_h, part_c, part_l])
    med_head = nmedian(r["loss_head_ov10"] for r in ht_rows)
    med_tail = nmedian(r["loss_tail_ov100"] for r in ht_rows)
    cen_loss_ratio = nmedian(r["cost_factor"] for r in sel("central", X9_EPS_MIN))
    nums = {"n_ratio_rows": len(ratio_rows), "n_datasets": len({r["dataset"] for r in ratio_rows}),
            "eps_values": sorted({r["eps"] for r in ratio_rows}),
            "N_ref_values": sorted({r["N_ref"] for r in ratio_rows}),
            "central_eps_ge2_Nref_ge2000_n_rows": len(cen_big), "central_eps_ge2_Nref_ge2000_median_ratio": med_cen_big,
            "central_eps_ge2_all_Nref_median_ratio": med_cen_all,
            "central_eps_ge2_all_Nref_median_cost_factor": cen_loss_ratio,
            "local_eps_ge2_median_ratio": med_loc_r, "local_eps_ge2_median_cost_factor": med_loc_f,
            "local_eps_ge2_share_censored": float(np.mean([r["censored"] != "no" for r in loc])) if loc else NAN,
            "share_cells_head_loss_larger_than_tail": head_share, "median_relative_loss_head_ov10": med_head,
            "median_relative_loss_tail_ov100": med_tail,
            "part_head": part_h, "part_central_free": part_c, "part_local_factor": part_l, "notes": notes}
    if part_c is None:
        notes.append(f"central claim not tested: no private cell with N >= {X9_NREF_CENTRAL} "
                     f"(e6 N values: {nums['N_ref_values']})")
    cen_txt_en = (f"central DP (eps >= 2, N >= 2000) keeps a median {fmt(med_cen_big, 2)} of the people's worth"
                  if part_c is not None else
                  f"central DP at eps >= 2 keeps a median {fmt(med_cen_all, 2)} of the people's worth at the N tested "
                  f"(no N >= 2000 in the data, so the 'few thousand' claim is untested)")
    cen_txt_zh = (f"中心化差分隐私（eps≥2，N≥2000）保留约 {fmt(med_cen_big, 2)} 的有效人数"
                  if part_c is not None else
                  f"中心化差分隐私（eps≥2）在已测的 N 上保留约 {fmt(med_cen_all, 2)} 的有效人数（数据里没有 N≥2000，“几千人”这一说法未检验）")
    cens_loc = nums["local_eps_ge2_share_censored"]
    en = (f"{cen_txt_en}; local DP (eps >= 2) needs a median {fmt(med_loc_f, 0)} times more people"
          f"{' (a lower bound: ' + format(cens_loc * 100, '.0f') + '% of these cells are below the nonprivate curve)' if cens_loc and cens_loc > 0 else ''}. "
          f"Privacy noise removes a median {fmt(med_head * 100, 0)}% of the top-10 overlap but {fmt(med_tail * 100, 0)}% "
          f"of the top-100 overlap (head loss larger in {fmt(head_share * 100, 0)}% of cells).")
    zh = (f"{cen_txt_zh}；本地差分隐私（eps≥2）需要多出约 {fmt(med_loc_f, 0)} 倍的人数"
          f"{'（只是下限：这些格子中有 ' + format(cens_loc * 100, '.0f') + '% 低于非隐私曲线的最小 N）' if cens_loc and cens_loc > 0 else ''}。"
          f"隐私噪声使前 10 名重合度中位下降 {fmt(med_head * 100, 0)}%，前 100 名重合度下降 {fmt(med_tail * 100, 0)}%"
          f"（{fmt(head_share * 100, 0)}% 的格子里头部损失更大）。")
    miss_en = [t for t, p in (("the central 'nearly free at a few thousand people' claim", part_c),
                              ("the local '~100x' claim", part_l)) if p is None]
    if miss_en:
        en += " Not judged: " + " and ".join(miss_en) + " (no N >= 2000, or only a bound on the cost is available)."
        zh += "未能判断：" + ("中心化“几千人几乎免费”" if part_c is None else "") + \
              ("、" if part_c is None and part_l is None else "") + ("本地化“约 100 倍”" if part_l is None else "") + \
              "（没有 N≥2000，或只得到代价的一个界）。"
    plot_x9(ctx, cell_means, ratio_rows, nums)
    return verdict("X9", "X9 (exploratory): privacy noise measured in sample sizes, central versus local, head versus tail",
                   "X9（探索性）：用样本量衡量隐私噪声的代价，中心化与本地化、头部与尾部", v, nums, en, zh, X9_RULE)


def pick_ks(ks: list[int], n: int = 3) -> list[int]:
    ks = sorted(ks)
    if len(ks) <= n:
        return ks
    return [ks[0], ks[len(ks) // 2], ks[-1]]


def plot_x9(ctx: Ctx, cell_means, ratio_rows, nums):
    mech_style = {"nonprivate": ("black", "o", "nonprivate"), "laplace": ("tab:blue", "s", "central: Laplace"),
                  "gaussian": ("tab:green", "^", "central: Gaussian"), "ldp_oue": ("tab:red", "D", "local: OUE")}
    ksall = sorted({r["k"] for r in cell_means}, key=lambda s: K_ALL if s == "all" else int(s))
    ksel = [klab(k) for k in pick_ks([K_ALL if s == "all" else int(s) for s in ksall])]
    eps_show = [e for e in (2.0, 8.0) if any(abs(r["eps"] - e) < 1e-9 for r in cell_means)]
    if eps_show:
        fig, axes = plt.subplots(len(eps_show), len(ksel), figsize=(FIG_W, 2.5 * len(eps_show) + 1.1),
                                 squeeze=False, sharey=True)
        for i, e in enumerate(eps_show):
            for j, k in enumerate(ksel):
                ax = axes[i, j]
                for mech, (c, mk, lab) in mech_style.items():
                    d = defaultdict(list)
                    for r in cell_means:
                        if r["k"] == k and abs(r["eps"] - e) < 1e-9 and r["mechanism"] == mech:
                            d[r["N"]].append(r["ndcg100"])
                    xs = sorted(d)
                    if xs:
                        ax.plot(xs, [float(np.mean(d[x])) for x in xs], color=c, marker=mk, ms=3.5, lw=1.3)
                ax.set_xscale("log")
                ax.set_ylim(0, 1.02)
                n_ticks = sorted({r["N"] for r in cell_means})
                ax.set_xticks(n_ticks)
                ax.set_xticklabels([str(n) for n in n_ticks])
                ax.minorticks_off()
                style(ax, "N (people sampled)" if i == len(eps_show) - 1 else None,
                      "NDCG@100 vs Truth A" if j == 0 else None, f"eps = {e:g}, k = {k}")
        handles = [Line2D([], [], color=c, marker=mk, ms=3.5, label=lab) for c, mk, lab in mech_style.values()]
        e_t = eps_show[-1]
        n_t = max(r["N"] for r in cell_means if abs(r["eps"] - e_t) < 1e-9)
        mean_at = {m: nmean([r["ndcg100"] for r in cell_means if abs(r["eps"] - e_t) < 1e-9 and r["N"] == n_t
                             and r["mechanism"] == m and r["k"] in ksel]) for m in mech_style}
        title = (f"At eps = {e_t:g} and N = {n_t}, mean NDCG@100: nonprivate {fmt(mean_at['nonprivate'], 2)}, "
                 f"Laplace {fmt(mean_at['laplace'], 2)}, Gaussian {fmt(mean_at['gaussian'], 2)}, "
                 f"local OUE {fmt(mean_at['ldp_oue'], 2)} (mean over datasets and the k shown). Exploratory.")
        ctx.finish(fig, "x9_ndcg_by_mechanism", title, handles, legend_ncol=4)

    fig, axes = grid(len(ksel), ncols=3, sharey=True)
    for ax, k in zip(axes, ksel):
        for mech, (c, mk, lab) in mech_style.items():
            if mech == "nonprivate":
                continue
            eps_vals = sorted({r["eps"] for r in ratio_rows if r["k"] == k and r["mechanism"] == mech})
            xs, ys, hol = [], [], []
            for e in eps_vals:
                rs = [r for r in ratio_rows if r["k"] == k and r["mechanism"] == mech and r["eps"] == e]
                xs.append(e)
                ys.append(nmedian(r["ratio"] for r in rs))
                hol.append(np.mean([r["censored"] != "no" for r in rs]) >= 0.5)
            ax.plot(xs, ys, color=c, lw=1.3)
            for x, y, h in zip(xs, ys, hol):
                ax.plot([x], [y], color=c, marker=mk, ms=4.5, mfc="none" if h else c, ls="")
        ax.axhline(1.0, color="grey", lw=0.8, ls="--")
        ax.axhline(0.01, color="grey", lw=0.8, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        epsall = sorted({r["eps"] for r in ratio_rows})
        ax.set_xticks(epsall)
        ax.set_xticklabels([f"{e:g}" for e in epsall])
        ax.minorticks_off()
        style(ax, "privacy budget eps", "effective N / real N (median)", f"k = {k}")
    handles = [Line2D([], [], color=c, marker=mk, ms=4.5, label=lab) for m, (c, mk, lab) in mech_style.items()
               if m != "nonprivate"]
    handles += [Line2D([], [], color="grey", ls="--", label="no cost (1)"),
                Line2D([], [], color="grey", ls=":", label="100x more people (0.01)"),
                Line2D([], [], color="grey", marker="o", mfc="none", ls="", label="hollow: bound only")]
    cf = nums["local_eps_ge2_median_cost_factor"]
    cens_l = nums["local_eps_ge2_share_censored"]
    bound = "at least " if (cens_l == cens_l and cens_l >= 0.5) else "about "
    title = (f"At eps >= 2 central DP keeps a median {fmt(nums['central_eps_ge2_all_Nref_median_ratio'], 2)} of the "
             f"people's worth; local DP needs {bound}{fmt(cf, 0)} times more people (median over datasets and N; "
             f"hollow markers are bounds). Exploratory.")
    ctx.finish(fig, "x9_effective_n_ratio", title, handles, legend_ncol=3)


# ----------------------------------------------------------------------------------------------
def main():
    root = HERE.parent.parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, default=root / "results/small_sample_popularity/02_runs")
    ap.add_argument("--out", type=Path, default=root / "results/small_sample_popularity")
    ap.add_argument("--min-draws", type=int, default=5, help="minimum draws for a cell to be used")
    ap.add_argument("--seed", type=int, default=20261014)
    a = ap.parse_args()
    ctx = Ctx(a.runs, a.out, a.min_draws, a.seed)
    fams = (("X5", x5, X5_TITLE, X5_RULE), ("X6", x6, X6_TITLE, X6_RULE), ("X7", x7, X7_TITLE, X7_RULE),
            ("X8", x8, X8_TITLE, X8_RULE), ("X9", x9, X9_TITLE, X9_RULE))
    out, failed = [], False
    for id_, fn, title, rule in fams:
        print(f"{id_} ...")
        try:
            out.append(fn(ctx))
        except Exception as exc:  # keep the other families; report loudly
            failed = True
            traceback.print_exc()
            out.append(not_testable(id_, *title, f"the analysis crashed ({type(exc).__name__}: {exc})",
                                    f"分析程序出错（{type(exc).__name__}）", rule))
        print(f"  -> {out[-1]['verdict']}")
    vp = ctx.tables / f"{NAME}_verdicts.json"
    with open(vp, "w") as fh:
        json.dump(clean(out), fh, ensure_ascii=False, indent=2)
    print("wrote", vp)
    for p in ctx.written:
        print("  ", p)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
