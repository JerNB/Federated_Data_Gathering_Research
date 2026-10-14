#!/usr/bin/env python3
"""Window-truth analysis (families X1, X2 as corrected by amendment A1, and the new family X12).

Everything written by this script is EXPLORATORY: no confirmatory claim, no p-value, and
nothing here has been re-run on fresh seeds (replication status: not replicated).

Why this script exists (pre-registration section 11, amendments A1 and A7): the stale list
(choices before the 55th percentile of time) is a nested part of Truth A (choices before the
cut-off T0), so "stale beats a fresh sample" was partly mechanical. The corrected test scores
every list against the WINDOW truth (item counts over choices between the 55th percentile and
T0 only) and draws the fresh samples ONLY from that window (the `recent_window` arm of e7).

  X1  stale list vs fresh window sample : N at which the plain count list first matches the stale
                                          list (window score), as a share of the group
  X2  shrinkage toward the stale list   : gain of `shrunk` over `count` (window score), where it peaks
                                          against the X1 crossover, median kappa by N
  X12 recency weighting (new)           : `decay` minus `count` on window score, Truth B and the panel

Usage:
  python3 scripts/ssp/an_window.py --runs <dir with e7.jsonl + ref.jsonl> --out <dir>
Writes <out>/03_tables/an_window_*.csv, <out>/04_figures/an_window_*.png and
<out>/03_tables/an_window_verdicts.json (entries X1, X2, X12).

Thresholds fixed BEFORE looking at any number (my reading of the vague words in the
pre-registration; each one is repeated in the matching verdict `rule`):
  "most datasets"  = strictly more than half of the datasets that can be judged;
                     exactly half = 'mixed'; no judgeable dataset = 'not testable'.
  X1 judged case   = crossover reached inside the sampled grid, or not reached although the
                     largest sampled N is already >= 30% of the group (then N_s > 30%: 'met').
                     Not reached and largest N < 30% of the group = indeterminate, not judged.
  X1 verdict       = the W_ndcg100 majorities for k=5 and k=all must agree ('supported' only if both
                     are 'supported', 'not supported' only if both are, else 'mixed'). The same
                     rule on W_ov100 is reported beside it as a robustness check.
  "near N*" (X2)   = peak of the gain is at the crossover grid point or one grid step away; judged
                     only where the crossover is reached inside the grid.
  X12 cell         = one (dataset, N, k). It is a WIN when the 95% bootstrap interval of
                     decay-minus-count (over draws) lies above 0, a LOSS when it lies below 0,
                     otherwise a TIE. Verdict on Truth B (B_ndcg100): 'supported' if more than
                     half of the judged cells are wins, 'not supported' if fewer than a quarter
                     are, else 'mixed'.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import textwrap
import time
import traceback
import warnings
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    import metrics  # noqa: E402
except ImportError:  # helper not in this worktree: read it from the main repository
    sys.path.insert(1, "/Users/LeoQin/Documents/GitHub/Federated_Data_Gathering_Research/scripts/ssp")
    import metrics  # noqa: E402

REPO = HERE.parent.parent
NAME = "an_window"
NAN = float("nan")
N_BOOT = 2000           # interval resamples over draws (metrics.paired_bootstrap_ci)
SHARE_THRESHOLD = 0.30  # X1 prediction
NEAR_STEPS = 1          # X2: 'near' = within one grid step
WIN_SHARE = 0.50        # X12 'supported' above this share of cells
LOSE_SHARE = 0.25       # X12 'not supported' below this share of cells
KAPPA_CAP = 1e6         # the estimator's cap on kappa (= all weight on the stale list)
KS_MAIN = ("all", 5)
ARM = "recent_window"
PRIORITY = ["ml-1m", "amazon-software", "gowalla", "lastfm-1k", "kuairand-pure", "amazon-books",
            "ml-latest", "brightkite", "amazon-video-games", "amazon-musical-instruments", "amazon-movies-tv"]

E7_FIELDS = ["count.W_ndcg100", "count.W_ov100", "shrunk.W_ndcg100",
             "count.B_ndcg100", "count.U_ndcg",
             "decay.W_ndcg100", "decay.B_ndcg100", "decay.U_ndcg",
             "kappa", "n_rows", "n_users"]
REF_PREFIXES = ("stale.", "truthA.", "truthB.")
X12_MEASURES = (("W_ndcg100", "window score (NDCG@100 vs window truth)"),
                ("B_ndcg100", "Truth B (NDCG@100 on later choices)"),
                ("U_ndcg", "panel (NDCG@100 on held-out people)"))

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "figure.dpi": 130, "savefig.dpi": 130,
                     "axes.spines.top": False, "axes.spines.right": False})


# ----------------------------------------------------------------------------- helpers
def fnum(v) -> float:
    return NAN if v is None else float(v)


def kkey(k):
    return "all" if k is None or k == "all" else int(k)


def kord(k) -> float:
    return math.inf if k == "all" else float(k)


def klabel(k) -> str:
    return "k=all" if k == "all" else f"k={k}"


def seed_for(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def fin(x) -> bool:
    return x is not None and isinstance(x, (int, float, np.floating, np.integer)) and math.isfinite(float(x))


def f3(x, nd=3) -> str:
    return "n/a" if not fin(x) else f"{x:.{nd}f}"


def fN(x) -> str:
    return "n/a" if not fin(x) else f"{x:,.0f}"


def pct(x) -> str:
    return "n/a" if not fin(x) else (f"{100 * x:.0f}%" if x >= 0.01 else f"{100 * x:.1f}%")


def clean(o):
    """JSON-safe copy: NaN/inf -> None, numpy -> python."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for r in rows:
        for k in r:
            if k not in names:
                names.append(k)
    if "status" not in names:
        names.append("status")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        for r in rows:
            out = {}
            for k in names:
                v = r.get(k, "exploratory" if k == "status" else "")
                if isinstance(v, (float, np.floating)):
                    v = "" if not math.isfinite(float(v)) else f"{float(v):.6g}"
                out[k] = v
            w.writerow(out)


def majority(n_yes: int, n_det: int) -> str:
    if n_det == 0:
        return "not testable"
    f = n_yes / n_det
    return "supported" if f > 0.5 else ("not supported" if f < 0.5 else "mixed")


def combine(vs: list[str]) -> str:
    vs = [v for v in vs if v != "not testable"]
    if not vs:
        return "not testable"
    if all(v == "supported" for v in vs):
        return "supported"
    if all(v == "not supported" for v in vs):
        return "not supported"
    return "mixed"


def entry(id_, title_en, title_zh, verdict, numbers, reading_en, reading_zh, rule) -> dict:
    return clean({"id": id_, "title_en": title_en, "title_zh": title_zh, "verdict": verdict,
                  "numbers": numbers, "reading_en": reading_en, "reading_zh": reading_zh, "rule": rule})


def not_testable(id_, title_en, title_zh, rule, why_en, why_zh, numbers=None) -> dict:
    return entry(id_, title_en, title_zh, "not testable", numbers or {}, why_en, why_zh, rule)


def pick4(avail: list[str], order: list[str]) -> list[str]:
    pri = [d for d in PRIORITY if d in avail]
    rest = [d for d in order if d in avail and d not in pri]
    return (pri + rest)[:4]


# ----------------------------------------------------------------------------- data
class Data:
    """ref.jsonl (constant reference lists) and the recent_window arm of e7.jsonl."""

    def __init__(self, runs: Path, datasets_path: Path):
        self.runs = runs
        self.messages: list[str] = []
        self.ds_meta: dict[str, dict] = {}
        self.ds_order: list[str] = []
        self.ref: dict[str, dict] = {}
        self.cells: dict[tuple, dict] = {}
        self.grid: dict[tuple, list[int]] = {}
        self._load_datasets(datasets_path)
        self._load_ref(runs / "ref.jsonl")
        self._load_e7(runs / "e7.jsonl")
        seen = list(self.ds_order)
        for d in sorted({d for (d, _, _) in self.cells} | set(self.ref)):
            if d not in seen:
                seen.append(d)
        self.ds_w = [d for d in seen if any(c[0] == d for c in self.cells)]

    def _load_datasets(self, p: Path):
        if not p.exists():
            self.messages.append(f"datasets.json not found at {p}: group sizes unavailable")
            return
        for d in json.loads(p.read_text()):
            self.ds_meta[d["name"]] = d
            self.ds_order.append(d["name"])

    def _load_ref(self, p: Path):
        if not p.exists():
            self.messages.append(f"ref.jsonl not found at {p}: no stale/truth reference lists")
            return
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.ref[r["dataset"]] = {k: (fnum(v) if k.startswith(REF_PREFIXES) else v) for k, v in r.items()}

    def _load_e7(self, p: Path):
        if not p.exists():
            self.messages.append(f"e7.jsonl not found at {p}: sampling-arm stage missing")
            return
        raw: dict[tuple, dict] = defaultdict(dict)
        bad = n_arm = n_rows = 0
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                n_rows += 1
                arm = r.get("arm")
                if arm is None:  # older rows: the arm is only in the key string
                    for part in str(r.get("key", "")).split("|"):
                        if part.startswith("arm="):
                            arm = part[4:]
                if arm != ARM:
                    continue
                n_arm += 1
                key = (r["dataset"], kkey(r.get("k")), int(r["N"]))
                raw[key][int(r["draw"])] = [fnum(r.get(fl)) for fl in E7_FIELDS]
        if bad:
            self.messages.append(f"e7.jsonl: {bad} unreadable line(s) skipped")
        if not n_arm:
            self.messages.append(f"e7.jsonl holds no `{ARM}` rows ({n_rows} rows of other arms): the window analysis cannot run")
            return
        grid: dict[tuple, list[int]] = defaultdict(list)
        for key, byd in raw.items():
            draws = np.array(sorted(byd))
            mat = np.array([byd[d] for d in draws])
            cell = {"draws": draws}
            for j, fl in enumerate(E7_FIELDS):
                cell[fl] = mat[:, j]
            self.cells[key] = cell
            grid[(key[0], key[1])].append(key[2])
        self.grid = {k: sorted(v) for k, v in grid.items()}

    # accessors ------------------------------------------------------------------
    def npop(self, ds) -> float:
        return fnum(self.ds_meta.get(ds, {}).get("n_pop"))

    def ks(self, ds) -> list:
        return sorted({k for (d, k) in self.grid if d == ds}, key=kord)

    def mean_curve(self, ds, k, field):
        Ns = self.grid.get((ds, k), [])
        m = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for N in Ns:
                v = self.cells[(ds, k, N)][field]
                m.append(np.nanmean(v) if v.size and np.isfinite(v).any() else NAN)
        return np.array(Ns, dtype=float), np.array(m)


def ci_of(values: np.ndarray, rng) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (NAN, NAN, NAN)
    if v.size == 1:
        return (float(v[0]), NAN, NAN)
    return metrics.paired_bootstrap_ci(v, n_boot=N_BOOT, rng=rng)


def curve_ci(D: Data, ds, k, Ns, field, tag):
    """(mean, lo, hi) arrays of one field over draws at each N."""
    mean, lo, hi = (np.full(len(Ns), NAN) for _ in range(3))
    for i, N in enumerate(Ns):
        rng = np.random.default_rng(seed_for(tag, ds, k, field, int(N)))
        mean[i], lo[i], hi[i] = ci_of(D.cells[(ds, k, int(N))][field], rng)
    return mean, lo, hi


def crossover(Ns: np.ndarray, means: np.ndarray, ref: float) -> dict:
    if len(Ns) == 0:
        return {"status": "no_data", "N_s": NAN, "censored": False}
    if not fin(ref):
        return {"status": "no_reference", "N_s": NAN, "censored": False}
    for N, m in zip(Ns, means):
        if np.isfinite(m) and m >= ref:
            return {"status": "ok", "N_s": float(N), "censored": False}
    return {"status": "ok", "N_s": NAN, "censored": True}


# ----------------------------------------------------------------------------- figures
def savefig(fig, figdir: Path, name: str) -> str:
    figdir.mkdir(parents=True, exist_ok=True)
    p = figdir / f"{NAME}_{name}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p.name


def grid_axes(n: int, ncols: int, w: float, h: float):
    nrows = max(1, math.ceil(n / ncols))
    ncols = min(ncols, max(n, 1))
    w = max(w, 6.6 / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def finish(fig, title: str, handles=None, labels=None, ncol: int | None = None) -> None:
    """Grow the figure by the title and legend heights, then lay out the panels in the remaining box.

    The wrapped finding-title sits on top (tight_layout reserves room for it), the shared legend at the bottom.
    """
    W, H0 = fig.get_size_inches()
    t = "\n".join(textwrap.wrap(title, max(40, int(W * 9.5))))
    title_in = 0.2 * (t.count("\n") + 1) + 0.2
    leg_in = 0.0
    if handles:
        ncol = ncol or min(len(handles), max(2, int(W / 3.3)))
        leg_in = 0.24 * math.ceil(len(handles) / ncol) + 0.25
    H = H0 + title_in + leg_in
    fig.set_size_inches(W, H)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.04 / H))
    fig.suptitle(t, fontsize=11, y=1 - 0.05 / H, va="top")
    fig.tight_layout(rect=(0, leg_in / H, 1, 1))


def _kfmt(v, _pos=None) -> str:
    return f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"


def log_n_axis(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.xaxis.set_major_formatter(FuncFormatter(_kfmt))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=0.25, which="major")


def shared_legend(axes):
    seen: dict = {}
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(ll, hh)
    return list(seen.values()), list(seen.keys())


COL_K = {"all": "#1f77b4", 5: "#ff7f0e", 1: "#9467bd", 2: "#8c564b", 10: "#2ca02c", 20: "#d62728"}
COL_M = {"W_ndcg100": "#1f77b4", "B_ndcg100": "#d62728", "U_ndcg": "#2ca02c"}


# ----------------------------------------------------------------------------- X1
X1_RULE = ("Crossover N_s >= 30% of the group in most datasets (N_s = smallest N whose mean window-sample count score "
           ">= the stale list's score on the window truth; judged on W_ndcg100 at k=5 and k=all, both must agree; "
           "W_ov100 reported as a robustness check)")
X1_TITLE = ("Exploratory X1 (window truth): the stale list beats a fresh sample from the same window until N is a large share of the group",
            "探索性 X1（窗口真相）：在样本人数达到群体的较大比例之前，旧榜单优于来自同一时间窗口的新样本榜单")
X1_MEASURES = ("W_ndcg100", "W_ov100")


def run_x1(D: Data, tabdir: Path, figdir: Path) -> dict:
    if not D.ds_w or not D.ref:
        return not_testable("X1", *X1_TITLE, X1_RULE,
                            f"Needs `{ARM}` rows in e7.jsonl and ref.jsonl (stale lists); at least one is missing.",
                            f"需要 e7.jsonl 中的 `{ARM}` 数据和 ref.jsonl（旧榜单），至少缺少其中一个。")
    rows, curves = [], {}
    for ds in D.ds_w:
        ref = D.ref.get(ds, {})
        npop = D.npop(ds)
        for k in KS_MAIN:
            if (ds, k) not in D.grid:
                continue
            for mt in X1_MEASURES:
                field = f"count.{mt}"
                Ns, means = D.mean_curve(ds, k, field)
                _, lo, hi = curve_ci(D, ds, k, Ns, field, "x1")
                stale = ref.get(f"stale.{mt}", NAN)
                curves[(ds, k, mt)] = (Ns, means, lo, hi)
                cx = crossover(Ns, means, stale)
                maxN = float(Ns[-1])
                share = cx["N_s"] / npop if fin(cx["N_s"]) and fin(npop) else NAN
                lb = maxN / npop if fin(npop) else NAN
                if cx["status"] != "ok":
                    outcome = cx["status"]
                elif not fin(npop):
                    outcome = "no_group_size"
                elif cx["censored"]:
                    outcome = "met" if lb >= SHARE_THRESHOLD else "indeterminate"
                else:
                    outcome = "met" if share >= SHARE_THRESHOLD else "not_met"
                robust = NAN
                if fin(stale):
                    for N, l in zip(Ns, lo):
                        if np.isfinite(l) and l > stale:
                            robust = float(N)
                            break
                at = np.flatnonzero(Ns == cx["N_s"])
                a = at[0] if at.size else None
                rows.append({
                    "dataset": ds, "k": k, "measure": mt, "n_pop": npop, "stale_reference": stale,
                    "truthA_reference": ref.get(f"truthA.{mt}", NAN), "truthB_reference": ref.get(f"truthB.{mt}", NAN),
                    "largest_N": maxN, "mean_count_at_largest_N": means[-1],
                    "crossover_N_s": cx["N_s"], "crossover_reached": (not cx["censored"]) and cx["status"] == "ok",
                    "crossover_share_of_group": share,
                    "share_lower_bound_if_not_reached": lb if cx["censored"] else NAN,
                    "mean_count_at_N_s": means[a] if a is not None else NAN,
                    "ci_lo_at_N_s": lo[a] if a is not None else NAN, "ci_hi_at_N_s": hi[a] if a is not None else NAN,
                    "ci_clears_stale_at_N_s": bool(a is not None and np.isfinite(lo[a]) and fin(stale) and lo[a] > stale),
                    "N_where_ci_lower_exceeds_stale": robust,
                    "share_where_ci_lower_exceeds_stale": robust / npop if fin(robust) and fin(npop) else NAN,
                    "outcome_vs_30pct": outcome,
                    "n_draws_at_largest_N": int(len(D.cells[(ds, k, int(Ns[-1]))]["draws"])),
                    "median_n_users_at_N_s": float(np.nanmedian(D.cells[(ds, k, int(cx["N_s"]))]["n_users"])) if a is not None else NAN,
                })
    write_csv(tabdir / f"{NAME}_x1_window_crossover.csv", rows)

    combo_v, nums = {}, {}
    for mt in X1_MEASURES:
        for k in KS_MAIN:
            sub = [r for r in rows if r["k"] == k and r["measure"] == mt]
            det = [r for r in sub if r["outcome_vs_30pct"] in ("met", "not_met")]
            yes = sum(r["outcome_vs_30pct"] == "met" for r in det)
            tag = f"{mt}_{k}"
            combo_v[tag] = majority(yes, len(det))
            nums[f"{tag}_met_of_judged"] = f"{yes}/{len(det)}"
            nums[f"{tag}_reached_in_grid"] = f"{sum(r['crossover_reached'] for r in sub)}/{sum(fin(r['stale_reference']) for r in sub)}"
            nums[f"{tag}_ci_clears_stale_at_N_s"] = f"{sum(r['ci_clears_stale_at_N_s'] for r in sub)}/{sum(r['crossover_reached'] for r in sub)}"
            shares = [r["crossover_share_of_group"] for r in sub if fin(r["crossover_share_of_group"])]
            nums[f"{tag}_median_share_where_reached"] = float(np.median(shares)) if shares else None
            nums[f"{tag}_indeterminate"] = sum(r["outcome_vs_30pct"] == "indeterminate" for r in sub)
    verdict = combine([combo_v[f"W_ndcg100_{k}"] for k in KS_MAIN])
    nums["verdict_per_measure"] = combo_v
    nums["n_datasets_with_window_sample"] = len(D.ds_w)
    nums["n_datasets_without_stale_reference"] = sum(not fin(D.ref.get(d, {}).get("stale.W_ndcg100", NAN)) for d in D.ds_w)
    n_all = [r for r in rows if r["k"] == "all" and r["measure"] == "W_ndcg100"]
    reach = sum(r["crossover_reached"] for r in n_all)
    nstale = sum(fin(r["stale_reference"]) for r in n_all)
    ms = nums["W_ndcg100_all_median_share_where_reached"]
    reading_en = (f"Scored on the window (the same stretch of time the fresh sample comes from), a fresh sample with every earlier choice kept (k=all) "
                  f"matched the stale list in {reach} of {nstale} datasets within the sampled N (median {pct(ms)} of the group where it did). "
                  f"Judged datasets where this took at least 30% of the group: k=all {nums['W_ndcg100_all_met_of_judged']}, k=5 {nums['W_ndcg100_5_met_of_judged']} "
                  f"(top-100 overlap: k=all {nums['W_ov100_all_met_of_judged']}, k=5 {nums['W_ov100_5_met_of_judged']}). Exploratory; few draws per cell make single crossovers noisy.")
    reading_zh = (f"以“窗口”（新样本所取自的同一段时间）为标准，保留全部过往选择（k=all）时，在已抽样的 N 范围内，新样本在 {nstale} 个数据集中有 {reach} 个追平了旧榜单；"
                  f"追平发生处的中位数约为群体的 {pct(ms)}。需要群体至少 30% 才追平的判定数：k=all {nums['W_ndcg100_all_met_of_judged']}，k=5 {nums['W_ndcg100_5_met_of_judged']}"
                  f"（前 100 名重合度：k=all {nums['W_ov100_all_met_of_judged']}，k=5 {nums['W_ov100_5_met_of_judged']}）。这是探索性结果，每格抽样次数不多，单个交叉点有噪声。")
    out = entry("X1", *X1_TITLE, verdict, nums, reading_en, reading_zh, X1_RULE)

    # figure: small multiples on W_ndcg100
    mt = "W_ndcg100"
    dss = [d for d in D.ds_w if any((d, k, mt) in curves for k in KS_MAIN)]
    if dss:
        fig, axes = grid_axes(len(dss), 3, 3.0, 2.7)
        for ax, ds in zip(axes, dss):
            ref = D.ref.get(ds, {})
            for k in KS_MAIN:
                if (ds, k, mt) not in curves:
                    continue
                Ns, m, lo, hi = curves[(ds, k, mt)]
                ax.plot(Ns, m, "-o", ms=3, color=COL_K[k], label=f"fresh window sample, {klabel(k)}")
                ax.fill_between(Ns, lo, hi, color=COL_K[k], alpha=0.15, lw=0)
                cx = crossover(Ns, m, ref.get(f"stale.{mt}", NAN))
                if fin(cx["N_s"]):
                    ax.plot([cx["N_s"]], [m[list(Ns).index(cx["N_s"])]], "o", ms=9, mfc="none", mec="k", mew=1.3)
            for key, col, lab in ((f"stale.{mt}", "#d62728", "stale list"),
                                  (f"truthA.{mt}", "#2ca02c", "Truth A list (full collection)")):
                if fin(ref.get(key, NAN)):
                    ax.axhline(ref[key], ls="--", color=col, lw=1.2, label=lab)
            log_n_axis(ax)
            ax.set_title(ds)
            ax.set_xlabel("N (people sampled)")
            ax.set_ylabel("NDCG@100 vs window truth")
        h, l = shared_legend(axes)
        h.append(plt.Line2D([], [], marker="o", ms=9, mfc="none", mec="k", ls=""))
        l.append("first N where the fresh list >= stale list")
        r5 = [r for r in rows if r["k"] == 5 and r["measure"] == mt]
        finish(fig, f"Exploratory X1: a fresh window sample matches the stale list in {reach} of {nstale} datasets with k=all "
                    f"and {sum(r['crossover_reached'] for r in r5)} of {sum(fin(r['stale_reference']) for r in r5)} with k=5, within the sampled N", h, l, ncol=2)
        savefig(fig, figdir, "x1_window_stale_vs_fresh")
    return out


# ----------------------------------------------------------------------------- X2
X2_RULE = ("Gain of shrinking toward the stale list (shrunk minus count, window score W_ndcg100) peaks near the X1 crossover "
           "(peak at the crossover grid point or one grid step away; judged only where the crossover is reached inside the grid; "
           "k=5 and k=all must agree, verdict from the majority of judged datasets per k)")
X2_TITLE = ("Exploratory X2 (window truth): shrinking toward the stale list helps most near the crossover",
            "探索性 X2（窗口真相）：向旧榜单收缩在交叉点附近帮助最大")


def run_x2(D: Data, tabdir: Path, figdir: Path) -> dict:
    if not D.ds_w:
        return not_testable("X2", *X2_TITLE, X2_RULE, f"Needs `{ARM}` rows in e7.jsonl; none found.",
                            f"需要 e7.jsonl 中的 `{ARM}` 数据，但没有找到。")
    gain_rows, kappa_rows, peak_rows, gains = [], [], [], {}
    for ds in D.ds_w:
        for k in D.ks(ds):
            Ns = D.grid[(ds, k)]
            mean, lo, hi = (np.full(len(Ns), NAN) for _ in range(3))
            for i, N in enumerate(Ns):
                c = D.cells[(ds, k, N)]
                diff = c["shrunk.W_ndcg100"] - c["count.W_ndcg100"]
                rng = np.random.default_rng(seed_for("x2", ds, k, N))
                mean[i], lo[i], hi[i] = ci_of(diff, rng)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    cnt = float(np.nanmean(c["count.W_ndcg100"]))
                    shr = float(np.nanmean(c["shrunk.W_ndcg100"]))
                gain_rows.append({"dataset": ds, "k": k, "N": N, "mean_count_W_ndcg100": cnt, "mean_shrunk_W_ndcg100": shr,
                                  "gain_shrunk_minus_count": mean[i], "ci_lo": lo[i], "ci_hi": hi[i],
                                  "n_draws": int(np.isfinite(diff).sum())})
                kap, nr = c["kappa"], c["n_rows"]
                okk = np.isfinite(kap)
                pw = kap / (kap + nr)
                kappa_rows.append({"dataset": ds, "k": k, "N": N,
                                   "kappa_median": float(np.median(kap[okk])) if okk.any() else NAN,
                                   "share_draws_kappa_at_cap": float(np.mean(kap[okk] >= KAPPA_CAP)) if okk.any() else NAN,
                                   "prior_weight_median_approx_kappa_over_kappa_plus_n_rows": float(np.nanmedian(pw)) if np.isfinite(pw).any() else NAN,
                                   "n_draws": int(okk.sum())})
            gains[(ds, k)] = (np.array(Ns, dtype=float), mean, lo, hi)
    write_csv(tabdir / f"{NAME}_x2_shrinkage_gain.csv", gain_rows)
    write_csv(tabdir / f"{NAME}_x2_kappa.csv", kappa_rows)

    for (ds, k), (Ns, mean, lo, hi) in gains.items():
        if not np.isfinite(mean).any():
            continue
        ip = int(np.nanargmax(mean))
        ref = D.ref.get(ds, {}).get("stale.W_ndcg100", NAN)
        cx = crossover(*D.mean_curve(ds, k, "count.W_ndcg100"), ref)
        row = {"dataset": ds, "k": k, "peak_N": Ns[ip], "peak_gain": mean[ip], "peak_ci_lo": lo[ip], "peak_ci_hi": hi[ip],
               "gain_at_peak_ci_excludes_0": bool(np.isfinite(lo[ip]) and lo[ip] > 0),
               "x1_crossover_N_s": cx["N_s"], "grid_steps_peak_vs_crossover": NAN}
        if cx["status"] != "ok":
            row["outcome"] = "not judged (no stale reference)"
        elif cx["censored"]:
            row["outcome"] = "not judged (crossover not reached within sampled N)"
        else:
            steps = abs(int(np.flatnonzero(Ns == cx["N_s"])[0]) - ip)
            row.update({"grid_steps_peak_vs_crossover": steps, "outcome": "near" if steps <= NEAR_STEPS else "far"})
        peak_rows.append(row)
    write_csv(tabdir / f"{NAME}_x2_peak_vs_crossover.csv", peak_rows)

    combo_v, nums = {}, {}
    for k in KS_MAIN:
        sub = [r for r in peak_rows if r["k"] == k and r["outcome"] in ("near", "far")]
        yes = sum(r["outcome"] == "near" for r in sub)
        combo_v[str(k)] = majority(yes, len(sub))
        nums[f"k{k}_peak_near_crossover"] = f"{yes}/{len(sub)}"
        nums[f"k{k}_not_judged"] = sum(r["outcome"].startswith("not judged") for r in peak_rows if r["k"] == k)
    verdict = combine(list(combo_v.values()))
    nums["verdict_per_k"] = combo_v
    kap_all = [r["kappa_median"] for r in kappa_rows if fin(r["kappa_median"])]
    nums["kappa_median_over_cells"] = float(np.median(kap_all)) if kap_all else None
    nums["kappa_median_by_N"] = {}
    for N in sorted({r["N"] for r in kappa_rows}):
        v = [r["kappa_median"] for r in kappa_rows if r["N"] == N and fin(r["kappa_median"])]
        if v:
            nums["kappa_median_by_N"][str(N)] = float(np.median(v))
    nums["kappa_note"] = f"kappa is capped at {KAPPA_CAP:g}, which means 'all weight on the stale list'"
    peaks = [r for r in peak_rows if r["k"] in KS_MAIN]
    best = max((r["peak_gain"] for r in peaks), default=NAN)
    medpk = float(np.median([r["peak_N"] for r in peaks])) if peaks else NAN
    nums["largest_mean_gain_W_ndcg100"] = best
    nums["median_peak_N"] = medpk
    nums["share_peak_ci_excludes_0"] = (f"{sum(r['gain_at_peak_ci_excludes_0'] for r in peaks)}/{len(peaks)}")
    reading_en = (f"Shrinking a fresh window sample toward the stale list gains up to {f3(best)} NDCG@100 on the window truth, with the gain peaking at a median N of {fN(medpk)}. "
                  f"Peak within one grid step of the X1 crossover: k=all {nums['kall_peak_near_crossover']}, k=5 {nums['k5_peak_near_crossover']} "
                  f"(only datasets whose crossover was reached can be judged). The median shrinkage strength kappa over cells is {fN(nums['kappa_median_over_cells'])}; "
                  f"a high kappa means the method leans on the stale list.")
    reading_zh = (f"把来自窗口的新样本向旧榜单收缩，对窗口真相的 NDCG@100 最多提高 {f3(best)}，增益峰值的中位数 N 为 {fN(medpk)}。"
                  f"增益峰值与 X1 交叉点相差不超过一格的情况：k=all {nums['kall_peak_near_crossover']}，k=5 {nums['k5_peak_near_crossover']}"
                  f"（只有交叉点已出现的数据集才能判定）。各格收缩强度 kappa 的中位数为 {fN(nums['kappa_median_over_cells'])}；kappa 越大，方法越依赖旧榜单。")
    out = entry("X2", *X2_TITLE, verdict, nums, reading_en, reading_zh, X2_RULE)

    show = pick4(list(D.ds_w), D.ds_order)
    if show:
        fig, axes = grid_axes(len(show), 2, 3.4, 3.0)
        pk = []
        for ax, ds in zip(axes, show):
            for k in D.ks(ds):
                Ns, m, lo, hi = gains[(ds, k)]
                if not np.isfinite(m).any():
                    continue
                ax.plot(Ns, m, "-o", ms=3, color=COL_K.get(k, "k"), label=klabel(k))
                ax.fill_between(Ns, lo, hi, color=COL_K.get(k, "k"), alpha=0.10, lw=0)
                pk.append((Ns[int(np.nanargmax(m))], np.nanmax(m)))
                cx = crossover(*D.mean_curve(ds, k, "count.W_ndcg100"), D.ref.get(ds, {}).get("stale.W_ndcg100", NAN))
                if k in KS_MAIN and fin(cx["N_s"]):
                    ax.axvline(cx["N_s"], color=COL_K.get(k, "k"), ls=":", lw=1.4,
                               label="X1 crossover (dotted, same colour)")
            ax.axhline(0, color="k", lw=0.8)
            log_n_axis(ax)
            ax.set_title(ds)
            ax.set_xlabel("N (people sampled)")
            ax.set_ylabel("gain in NDCG@100 vs window truth")
        h, l = shared_legend(axes)
        if pk:
            ttl = (f"Exploratory X2: shrinking toward the stale list gains up to {max(p[1] for p in pk):+.3f} NDCG@100, "
                   f"with a median peak at N = {np.median([p[0] for p in pk]):,.0f}")
        else:
            ttl = "Exploratory X2: no shrinkage gain available"
        finish(fig, ttl, h, l, ncol=4)
        savefig(fig, figdir, "x2_window_shrinkage_gain")
    return out


# ----------------------------------------------------------------------------- X12
X12_RULE = (f"Share of (dataset, N, k) cells where recency-weighted counts beat plain counts on Truth B (B_ndcg100) with the 95% bootstrap "
            f"interval of the difference above 0 (cells with >= 2 draws): supported if more than {WIN_SHARE:.0%} of judged cells, "
            f"not supported if fewer than {LOSE_SHARE:.0%}, else mixed; cells with the interval below 0 are reported as losses")
X12_TITLE = ("Exploratory X12: weighting recent choices more helps predict later choices",
             "探索性 X12：更看重近期选择有助于预测之后的选择")


def run_x12(D: Data, tabdir: Path, figdir: Path) -> dict:
    if not D.ds_w:
        return not_testable("X12", *X12_TITLE, X12_RULE, f"Needs `{ARM}` rows in e7.jsonl; none found.",
                            f"需要 e7.jsonl 中的 `{ARM}` 数据，但没有找到。")
    rows, curves = [], {}
    for ds in D.ds_w:
        for k in D.ks(ds):
            Ns = D.grid[(ds, k)]
            for mt, _ in X12_MEASURES:
                mean, lo, hi = (np.full(len(Ns), NAN) for _ in range(3))
                for i, N in enumerate(Ns):
                    c = D.cells[(ds, k, N)]
                    diff = c[f"decay.{mt}"] - c[f"count.{mt}"]
                    rng = np.random.default_rng(seed_for("x12", ds, k, mt, N))
                    mean[i], lo[i], hi[i] = ci_of(diff, rng)
                    if np.isfinite(lo[i]):
                        outcome = "win" if lo[i] > 0 else ("loss" if hi[i] < 0 else "tie")
                    else:
                        outcome = "not judged"
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        cnt = float(np.nanmean(c[f"count.{mt}"]))
                    rows.append({"dataset": ds, "k": k, "N": N, "measure": mt, "mean_count": cnt,
                                 "diff_decay_minus_count": mean[i], "ci_lo": lo[i], "ci_hi": hi[i],
                                 "outcome": outcome, "n_draws": int(np.isfinite(diff).sum())})
                curves[(ds, k, mt)] = (np.array(Ns, dtype=float), mean, lo, hi)
    write_csv(tabdir / f"{NAME}_x12_decay_vs_count.csv", rows)

    def tally(sub):
        j = [r for r in sub if r["outcome"] != "not judged"]
        return {"cells": len(j), "wins": sum(r["outcome"] == "win" for r in j),
                "losses": sum(r["outcome"] == "loss" for r in j), "ties": sum(r["outcome"] == "tie" for r in j)}

    nums = {}
    for mt, _ in X12_MEASURES:
        t = tally([r for r in rows if r["measure"] == mt])
        t["win_share"] = t["wins"] / t["cells"] if t["cells"] else None
        t["median_diff"] = float(np.median([r["diff_decay_minus_count"] for r in rows if r["measure"] == mt and fin(r["diff_decay_minus_count"])] or [NAN]))
        nums[mt] = t
    tb = nums["B_ndcg100"]
    verdict = "not testable"
    if tb["cells"]:
        verdict = ("supported" if tb["win_share"] > WIN_SHARE else
                   ("not supported" if tb["win_share"] < LOSE_SHARE else "mixed"))
    # per dataset and per k (Truth B)
    share_rows = []
    brows = [r for r in rows if r["measure"] == "B_ndcg100"]
    for ds in D.ds_w:
        for kk in ["all-k"] + list(D.ks(ds)):
            sub = [r for r in brows if r["dataset"] == ds and (kk == "all-k" or r["k"] == kk)]
            t = tally(sub)
            share_rows.append({"dataset": ds, "k": kk, **t, "win_share": t["wins"] / t["cells"] if t["cells"] else NAN,
                               "loss_share": t["losses"] / t["cells"] if t["cells"] else NAN,
                               "tie_share": t["ties"] / t["cells"] if t["cells"] else NAN})
    write_csv(tabdir / f"{NAME}_x12_win_share.csv", share_rows)
    per_k = {}
    for k in sorted({r["k"] for r in brows}, key=kord):
        t = tally([r for r in brows if r["k"] == k])
        per_k[klabel(k)] = f"{t['wins']} wins / {t['losses']} losses / {t['ties']} ties of {t['cells']}"
    nums["truthB_per_k"] = per_k
    nums["truthB_datasets_with_a_win"] = sum(1 for r in share_rows if r["k"] == "all-k" and r["wins"] > 0)
    nums["truthB_datasets_with_a_loss"] = [r["dataset"] for r in share_rows if r["k"] == "all-k" and r["losses"] > 0]
    worst = sorted([r for r in brows if r["outcome"] == "loss"], key=lambda r: r["diff_decay_minus_count"])[:5]
    nums["truthB_largest_losses"] = [f"{r['dataset']} {klabel(r['k'])} N={r['N']}: {r['diff_decay_minus_count']:+.3f}" for r in worst]
    nums["truthB_largest_wins"] = [f"{r['dataset']} {klabel(r['k'])} N={r['N']}: {r['diff_decay_minus_count']:+.3f}"
                                   for r in sorted([r for r in brows if r["outcome"] == "win"], key=lambda r: -r["diff_decay_minus_count"])[:5]]
    nums["decay_half_life"] = "the length of the window"
    ws, ls_, ts = tb["wins"], tb["losses"], tb["ties"]
    ndat = len(D.ds_w)
    if tb["cells"]:
        reading_en = (f"Weighting recent choices more beat the plain count list on later real choices (Truth B) in {ws} of {tb['cells']} cells "
                      f"({pct(tb['win_share'])}), lost in {ls_}, and was indistinguishable in {ts}; the median gain over all cells is {tb['median_diff']:+.3f} NDCG@100. "
                      f"On the window score it wins {nums['W_ndcg100']['wins']} and loses {nums['W_ndcg100']['losses']} of {nums['W_ndcg100']['cells']}, and on the held-out panel it wins {nums['U_ndcg']['wins']} and loses {nums['U_ndcg']['losses']} of {nums['U_ndcg']['cells']}. "
                      f"Exploratory, {ndat} dataset(s) with window samples.")
        reading_zh = (f"更看重近期选择，在 {tb['cells']} 个格子中有 {ws} 个（{pct(tb['win_share'])}）预测之后真实选择（Truth B）的效果好于普通计数榜单，{ls_} 个更差，{ts} 个看不出差别；"
                      f"所有格子的中位差为 {tb['median_diff']:+.3f}（NDCG@100）。窗口得分上胜 {nums['W_ndcg100']['wins']}、负 {nums['W_ndcg100']['losses']}（共 {nums['W_ndcg100']['cells']}），留出面板上胜 {nums['U_ndcg']['wins']}、负 {nums['U_ndcg']['losses']}（共 {nums['U_ndcg']['cells']}）。"
                      f"这是探索性结果，共 {ndat} 个有窗口样本的数据集。")
    else:
        reading_en = "No cell has enough draws (at least 2) with the recency-weighted estimate to judge."
        reading_zh = "没有任何格子有足够的抽样次数（至少 2 次）来判断近期加权估计。"
    out = entry("X12", *X12_TITLE, verdict, nums, reading_en, reading_zh, X12_RULE)

    # figure 1: per-dataset curves, k = all
    dss = [d for d in D.ds_w if (d, "all", "B_ndcg100") in curves]
    if dss:
        fig, axes = grid_axes(len(dss), 3, 3.0, 2.7)
        for ax, ds in zip(axes, dss):
            for mt, lab in X12_MEASURES:
                Ns, m, lo, hi = curves[(ds, "all", mt)]
                ax.plot(Ns, m, "-o", ms=3, color=COL_M[mt], label=lab)
                ax.fill_between(Ns, lo, hi, color=COL_M[mt], alpha=0.15, lw=0)
            ax.axhline(0, color="k", lw=0.8)
            log_n_axis(ax)
            ax.set_title(ds)
            ax.set_xlabel("N (people sampled)")
            ax.set_ylabel("recency-weighted minus plain")
        h, l = shared_legend(axes)
        ball = [r for r in brows if r["k"] == "all" and r["outcome"] != "not judged"]
        bw = sum(r["outcome"] == "win" for r in ball)
        bl = sum(r["outcome"] == "loss" for r in ball)
        mb = float(np.median([r["diff_decay_minus_count"] for r in ball])) if ball else NAN
        finish(fig, f"Exploratory X12: with every choice kept, recency weighting beats plain counts on later choices in {bw} of {len(ball)} cells "
                    f"and loses in {bl} (median {mb:+.3f})", h, l, ncol=1)
        savefig(fig, figdir, "x12_decay_minus_count_kall")

    # figure 2: stacked win / tie / loss shares per dataset (Truth B, all k pooled)
    pooled = [r for r in share_rows if r["k"] == "all-k" and r["cells"] > 0]
    if pooled:
        fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(pooled) + 0.9))
        y = np.arange(len(pooled))[::-1]
        win = np.array([r["win_share"] for r in pooled])
        tie = np.array([r["tie_share"] for r in pooled])
        los = np.array([r["loss_share"] for r in pooled])
        ax.barh(y, win, color="#2ca02c", label="recency weighting wins (interval above 0)")
        ax.barh(y, tie, left=win, color="#bbbbbb", label="no clear difference")
        ax.barh(y, los, left=win + tie, color="#d62728", label="recency weighting loses (interval below 0)")
        for yi, r in zip(y, pooled):
            ax.text(1.01, yi, f"{r['wins']}/{r['cells']}", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels([r["dataset"] for r in pooled])
        ax.set_xlim(0, 1.0)
        ax.axvline(WIN_SHARE, color="k", ls=":", lw=1)
        ax.set_xlabel("share of (N, k) cells, Truth B (NDCG@100 on later choices); right: wins/cells")
        h, l = ax.get_legend_handles_labels()
        finish(fig, f"Exploratory X12: recency weighting wins in {ws} of {tb['cells']} cells on later choices, loses in {ls_}", h, l, ncol=1)
        savefig(fig, figdir, "x12_win_share_by_dataset")
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(REPO / "results/small_sample_popularity/02_runs"))
    ap.add_argument("--out", default=str(REPO / "results/small_sample_popularity"))
    ap.add_argument("--datasets", default=None, help="datasets.json (default: <runs>/../01_datasets/datasets.json)")
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    dpath = Path(a.datasets) if a.datasets else runs.parent / "01_datasets" / "datasets.json"
    tabdir, figdir = out / "03_tables", out / "04_figures"
    tabdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    D = Data(runs, dpath)
    for m in D.messages:
        print("NOTE:", m)
    print(f"loaded {len(D.cells)} `{ARM}` cells, {len(D.ref)} ref rows, datasets with window samples: {D.ds_w}  ({time.time() - t0:.1f}s)")

    def guarded(id_, title, rule, fn):
        t = time.time()
        try:
            r = fn()
        except Exception as e:  # keep the other families alive
            traceback.print_exc()
            r = not_testable(id_, *title, rule, f"Internal error while computing this family: {type(e).__name__}: {e}",
                             f"计算该族时出现内部错误：{type(e).__name__}: {e}")
        print(f"{id_}: {r['verdict']}  ({time.time() - t:.1f}s)")
        return r

    verdicts = [guarded("X1", X1_TITLE, X1_RULE, lambda: run_x1(D, tabdir, figdir)),
                guarded("X2", X2_TITLE, X2_RULE, lambda: run_x2(D, tabdir, figdir)),
                guarded("X12", X12_TITLE, X12_RULE, lambda: run_x12(D, tabdir, figdir))]
    if D.messages:  # surface missing stages inside the file itself
        for v in verdicts:
            if v["verdict"] == "not testable":
                v["numbers"] = {**(v.get("numbers") or {}), "missing_inputs": D.messages}
    (tabdir / f"{NAME}_verdicts.json").write_text(json.dumps(clean(verdicts), indent=1, ensure_ascii=False, allow_nan=False))
    print(f"wrote {tabdir / (NAME + '_verdicts.json')}  total {time.time() - t0:.1f}s")
    for v in verdicts:
        print(f"  {v['id']:4s} {v['verdict']:14s} {v['reading_en'][:150]}")


if __name__ == "__main__":
    main()
