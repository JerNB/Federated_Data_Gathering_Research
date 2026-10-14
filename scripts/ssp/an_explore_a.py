#!/usr/bin/env python3
"""Exploratory analysis A (families X1, X2, X3, X4, X10 of the pre-registration, section 7).

Everything written by this script is EXPLORATORY: no confirmatory claim, no p-value, and
nothing here has been re-run on fresh seeds (replication status: not replicated).

  X1  stale list vs fresh sample  : N at which the plain count list first matches the stale list
  X2  shrinkage toward stale list : gain of `shrunk` over `count`, where it peaks, kappa
  X3  scaling                     : power-law fit with floor, exponent vs Zipf slope
  X4  Dyer-Owen bound             : predicted resolve-N beside the observed N where top-10 overlap hits 0.8 (proxy)
  X10 drift vs sampling error     : Truth A -> Truth B gap vs the sampling gap of a fresh sample

Usage:
  python3 scripts/ssp/an_explore_a.py --runs <dir with e1.jsonl + ref.jsonl> --out <dir>
Writes <out>/03_tables/an_explore_a_*.csv, <out>/04_figures/an_explore_a_*.png and
<out>/03_tables/an_explore_a_verdicts.json (entries X1, X2, X3, X4, X10).

Thresholds fixed BEFORE looking at any number (they are my reading of the vague words in
the pre-registration; each one is repeated in the matching verdict `rule`):
  "most datasets"  = strictly more than half of the datasets that can be judged;
                     exactly half = 'mixed'; no judgeable dataset = 'not testable'.
  "near N*" (X2)   = peak of the gain is at the crossover grid point or one grid step away.
  X3 rank correlation is judged only with >= 5 datasets with a converged fit.
A family with several measures (e.g. k = all / k = 5, Truth B / panel) is 'supported' only if
every judgeable measure is 'supported', 'not supported' only if every one is, else 'mixed'.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    import metrics  # noqa: E402
    import scaling  # noqa: E402
except ImportError:  # helper not in this worktree: read it from the main repository
    sys.path.insert(1, "/Users/LeoQin/Documents/GitHub/Federated_Data_Gathering_Research/scripts/ssp")
    import metrics  # noqa: E402
    import scaling  # noqa: E402

REPO = HERE.parent.parent
NAME = "an_explore_a"
NAN = float("nan")
N_BOOT = 2000          # interval resamples over draws (metrics.*_bootstrap_ci)
N_BOOT_FIT = 300       # replicate-curve resamples for the exponent CI (scaling default)
SHARE_THRESHOLD = 0.30  # X1 prediction
NEAR_STEPS = 1         # X2: 'near' = within one grid step
MIN_DATASETS_RHO = 5   # X3
OV_TARGET = 0.8        # X4
KS_MAIN = ("all", 5)
PRIORITY = ["ml-1m", "amazon-software", "gowalla", "lastfm-1k", "kuairand-pure", "amazon-books",
            "ml-latest", "brightkite", "amazon-video-games", "amazon-musical-instruments", "amazon-movies-tv"]

E1_FIELDS = ["count.A_ndcg100", "count.A_ov10", "count.A_ov100", "count.A_rbo90", "count.B_ndcg100",
             "count.U_ndcg", "shrunk.A_ndcg100", "shrunk.B_ndcg100", "shrunk.U_ndcg", "kappa", "n_rows"]
REF_PREFIXES = ("stale.", "truthA.", "truthB.")

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "legend.fontsize": 9, "figure.dpi": 130, "savefig.dpi": 130,
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
        self._load_e1(runs / "e1.jsonl")
        seen = [d for d in self.ds_order]
        extra = sorted({d for (d, _, _) in self.cells} | set(self.ref))
        for d in extra:
            if d not in seen:
                seen.append(d)
        self.ds_all = [d for d in seen if d in self.ref or any(c[0] == d for c in self.cells)]
        self.ds_e1 = [d for d in self.ds_all if any(c[0] == d for c in self.cells)]

    def _load_datasets(self, p: Path):
        if not p.exists():
            self.messages.append(f"datasets.json not found at {p}: group sizes, Zipf slopes and Dyer-Owen figures unavailable")
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

    def _load_e1(self, p: Path):
        if not p.exists():
            self.messages.append(f"e1.jsonl not found at {p}: popularity-estimator stage missing")
            return
        raw: dict[tuple, dict] = defaultdict(dict)
        bad = 0
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
                arm = r.get("arm")
                if arm is None:
                    arm = "uniform"
                    for part in str(r.get("key", "")).split("|"):
                        if part.startswith("arm="):
                            arm = part[4:]
                if arm != "uniform" or r.get("users", "uniform") != "uniform" or r.get("rows", "random") != "random":
                    continue
                key = (r["dataset"], kkey(r.get("k")), int(r["N"]))
                raw[key][int(r["draw"])] = [fnum(r.get(fl)) for fl in E1_FIELDS]
        if bad:
            self.messages.append(f"e1.jsonl: {bad} unreadable line(s) skipped")
        grid: dict[tuple, list[int]] = defaultdict(list)
        for key, byd in raw.items():
            draws = np.array(sorted(byd))
            mat = np.array([byd[d] for d in draws]) if len(draws) else np.empty((0, len(E1_FIELDS)))
            cell = {"draws": draws}
            for j, fl in enumerate(E1_FIELDS):
                cell[fl] = mat[:, j]
            self.cells[key] = cell
            grid[(key[0], key[1])].append(key[2])
        self.grid = {k: sorted(v) for k, v in grid.items()}
        if not self.cells:
            self.messages.append("e1.jsonl holds no uniform-arm main-grid rows")

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

    def reps(self, ds, k, field, Ns):
        """(draws, len(Ns)) matrix, NaN where a draw is missing at that N."""
        ids = sorted(set().union(*(set(self.cells[(ds, k, N)]["draws"].tolist()) for N in Ns)))
        pos = {d: i for i, d in enumerate(ids)}
        M = np.full((len(ids), len(Ns)), NAN)
        for j, N in enumerate(Ns):
            c = self.cells[(ds, k, N)]
            for d, v in zip(c["draws"], c[field]):
                M[pos[int(d)], j] = v
        return M


def ci_of(values: np.ndarray, rng) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (NAN, NAN, NAN)
    if v.size == 1:
        return (float(v[0]), NAN, NAN)
    return metrics.paired_bootstrap_ci(v, n_boot=N_BOOT, rng=rng)


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
    w = max(w, 9.0 / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows + 0.8), squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def finish(fig, title: str, handles=None, labels=None, ncol: int | None = None) -> None:
    """Wrapped finding-title on top, shared legend at the bottom, then tight layout in the remaining box."""
    import textwrap
    W, H = fig.get_size_inches()
    t = "\n".join(textwrap.wrap(title, max(40, int(W * 9.5))))
    title_in = 0.26 * (t.count("\n") + 1) + 0.15
    leg_in = 0.0
    if handles:
        ncol = ncol or min(len(handles), max(2, int(W / 3.3)))
        leg_in = 0.27 * math.ceil(len(handles) / ncol) + 0.2
        fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.05 / H))
    fig.suptitle(t, fontsize=11, y=1 - 0.05 / H, va="top")
    fig.tight_layout(rect=(0, leg_in / H, 1, 1))


def log_n_axis(ax):
    ax.set_xscale("log")
    ax.grid(alpha=0.25, which="major")


COL_K = {"all": "#1f77b4", 5: "#ff7f0e", 1: "#9467bd", 2: "#8c564b", 10: "#2ca02c", 20: "#d62728"}


# ----------------------------------------------------------------------------- X1
X1_RULE = "Crossover N_s >= 30% of the group in most datasets (N_s = smallest N whose mean count score >= the stale list's score)"
X1_TITLE = ("Exploratory X1: the stale list beats a small fresh sample until N is a large share of the group",
            "探索性 X1：在样本人数达到群体的较大比例之前，旧榜单优于小样本新榜单")


def run_x1(D: Data, tabdir: Path, figdir: Path) -> tuple[dict, dict]:
    if not D.ds_e1 or not D.ref:
        return not_testable("X1", *X1_TITLE, X1_RULE,
                            "Needs e1 rows and ref.jsonl (stale lists); at least one is missing.",
                            "需要 e1 数据和 ref.jsonl（旧榜单），至少缺少其中一个。"), {}
    rows, curves = [], {}
    for ds in D.ds_e1:
        ref = D.ref.get(ds, {})
        npop = D.npop(ds)
        for k in KS_MAIN:
            if (ds, k) not in D.grid:
                continue
            for mt in ("B_ndcg100", "U_ndcg"):
                Ns, means = D.mean_curve(ds, k, f"count.{mt}")
                stale = ref.get(f"stale.{mt}", NAN)
                lo = np.full(len(Ns), NAN)
                hi = np.full(len(Ns), NAN)
                for i, N in enumerate(Ns):
                    rng = np.random.default_rng(seed_for("x1", ds, k, mt, N))
                    _, lo[i], hi[i] = ci_of(D.cells[(ds, k, int(N))][f"count.{mt}"], rng)
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
                rows.append({
                    "dataset": ds, "k": k, "measure": mt, "n_pop": npop, "stale_reference": stale,
                    "truthA_reference": ref.get(f"truthA.{mt}", NAN), "truthB_reference": ref.get(f"truthB.{mt}", NAN),
                    "largest_N": maxN, "mean_count_at_largest_N": means[-1] if len(means) else NAN,
                    "crossover_N_s": cx["N_s"], "crossover_reached": (not cx["censored"]) and cx["status"] == "ok",
                    "crossover_share_of_group": share,
                    "share_lower_bound_if_not_reached": lb if cx["censored"] else NAN,
                    "mean_count_at_N_s": means[at[0]] if at.size else NAN,
                    "ci_lo_at_N_s": lo[at[0]] if at.size else NAN, "ci_hi_at_N_s": hi[at[0]] if at.size else NAN,
                    "N_where_ci_lower_exceeds_stale": robust, "outcome_vs_30pct": outcome,
                    "n_draws_at_largest_N": int(len(D.cells[(ds, k, int(Ns[-1]))]["draws"])),
                })
    write_csv(tabdir / f"{NAME}_x1_stale_crossover.csv", rows)

    combo_v, nums = {}, {}
    for k in KS_MAIN:
        for mt in ("B_ndcg100", "U_ndcg"):
            sub = [r for r in rows if r["k"] == k and r["measure"] == mt]
            det = [r for r in sub if r["outcome_vs_30pct"] in ("met", "not_met")]
            yes = sum(r["outcome_vs_30pct"] == "met" for r in det)
            tag = f"{mt.split('_')[0]}_{k}"
            combo_v[tag] = majority(yes, len(det))
            nums[f"{tag}_met_of_judged"] = f"{yes}/{len(det)}"
            nums[f"{tag}_reached_in_grid"] = f"{sum(r['crossover_reached'] for r in sub)}/{sum(fin(r['stale_reference']) for r in sub)}"
            shares = [r["crossover_share_of_group"] for r in sub if fin(r["crossover_share_of_group"])]
            nums[f"{tag}_median_share"] = float(np.median(shares)) if shares else None
    verdict = combine(list(combo_v.values()))
    nums["verdict_per_measure"] = combo_v
    nums["n_datasets"] = len(D.ds_e1)
    pb = [r for r in rows if r["k"] == "all" and r["measure"] == "B_ndcg100"]
    reach = sum(r["crossover_reached"] for r in pb)
    nstale = sum(fin(r["stale_reference"]) for r in pb)
    ms = nums["B_all_median_share"]
    reading_en = (f"With every earlier choice kept (k=all), a fresh sample matched the stale list on Truth B in {reach} of {nstale} datasets "
                  f"within the sampled N (median crossover {pct(ms)} of the group where it happened). "
                  f"Judged cases with a crossover of at least 30% of the group, B/k=all: {nums['B_all_met_of_judged']}, "
                  f"B/k=5: {nums['B_5_met_of_judged']}, panel/k=all: {nums['U_all_met_of_judged']}, panel/k=5: {nums['U_5_met_of_judged']}. "
                  f"Exploratory; small draws per cell make single crossovers noisy.")
    reading_zh = (f"保留全部过往选择（k=all）时，在已抽样的 N 范围内，新样本在 {nstale} 个数据集中有 {reach} 个追平了旧榜单（对后续真实选择 Truth B 的得分）；"
                  f"追平发生处的中位数约为群体的 {pct(ms)}。达到“至少群体 30%”的判定数：B/k=all {nums['B_all_met_of_judged']}，"
                  f"B/k=5 {nums['B_5_met_of_judged']}，面板/k=all {nums['U_all_met_of_judged']}，面板/k=5 {nums['U_5_met_of_judged']}。"
                  f"这是探索性结果，每格抽样次数不多，单个交叉点有噪声。")
    out = entry("X1", *X1_TITLE, verdict, nums, reading_en, reading_zh, X1_RULE)

    # figures: one small-multiples figure per measure
    for mt, lab, ylab in (("B_ndcg100", "B", "NDCG@100 vs Truth B"),
                          ("U_ndcg", "U", "panel NDCG@100")):
        dss = [d for d in D.ds_e1 if any((d, k, mt) in curves for k in KS_MAIN)]
        if not dss:
            continue
        fig, axes = grid_axes(len(dss), 4, 3.6, 3.8)
        for ax, ds in zip(axes, dss):
            ref = D.ref.get(ds, {})
            for k in KS_MAIN:
                if (ds, k, mt) not in curves:
                    continue
                Ns, m, lo, hi = curves[(ds, k, mt)]
                ax.plot(Ns, m, "-o", ms=3.5, color=COL_K[k], label=f"fresh count list, {klabel(k)}")
                ax.fill_between(Ns, lo, hi, color=COL_K[k], alpha=0.15, lw=0)
                cx = crossover(Ns, m, ref.get(f"stale.{mt}", NAN))
                if fin(cx["N_s"]):
                    ax.plot([cx["N_s"]], [m[list(Ns).index(cx["N_s"])]], "o", ms=9, mfc="none", mec="k", mew=1.4)
            for key, col, ls_lab in ((f"stale.{mt}", "#d62728", "stale list"), (f"truthA.{mt}", "#2ca02c", "Truth A list (full community, same window)"),
                                     (f"truthB.{mt}", "#555555", "Truth B list (later choices)")):
                if fin(ref.get(key, NAN)):
                    ax.axhline(ref[key], ls="--", color=col, lw=1.2, label=ls_lab)
            log_n_axis(ax)
            ax.set_title(ds)
            ax.set_xlabel("N (people sampled)")
            ax.set_ylabel(ylab)
        seen = {}
        for ax in axes:
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(ll, hh)
        h, l = list(seen.values()), list(seen.keys())
        h.append(plt.Line2D([], [], marker="o", ms=9, mfc="none", mec="k", ls=""))
        l.append("first N where fresh list >= stale list")
        sub = [r for r in rows if r["k"] == "all" and r["measure"] == mt]
        reach = sum(r["crossover_reached"] for r in sub)
        nst = sum(fin(r["stale_reference"]) for r in sub)
        finish(fig, f"Exploratory X1: a fresh sample (k=all) matches the stale list in {reach} of {nst} datasets within the sampled N", h, l)
        savefig(fig, figdir, f"x1_stale_vs_fresh_{lab}")
    return out, curves


# ----------------------------------------------------------------------------- X2
X2_RULE = ("Flat-prior ranks identical to raw counts (exact control, mathematical fact, not tested); "
           "gain peaks near N* (here: peak of the shrunk-minus-count gain at the X1 crossover grid point or one grid step away)")
X2_TITLE = ("Exploratory X2: shrinking toward the stale list helps most near the crossover",
            "探索性 X2：向旧榜单收缩在交叉点附近帮助最大")


def run_x2(D: Data, x1curves: dict, x1rows_path_unused, tabdir: Path, figdir: Path) -> dict:
    if not D.ds_e1:
        return not_testable("X2", *X2_TITLE, X2_RULE, "Needs e1 rows; none found.", "需要 e1 数据，但没有找到。")
    gain_rows, kappa_rows, peak_rows = [], [], []
    gains = {}
    for ds in D.ds_e1:
        for k in D.ks(ds):
            Ns = D.grid[(ds, k)]
            for mt in ("A_ndcg100", "B_ndcg100", "U_ndcg"):
                mean = np.full(len(Ns), NAN)
                lo = np.full(len(Ns), NAN)
                hi = np.full(len(Ns), NAN)
                for i, N in enumerate(Ns):
                    c = D.cells[(ds, k, N)]
                    diff = c[f"shrunk.{mt}"] - c[f"count.{mt}"]
                    rng = np.random.default_rng(seed_for("x2", ds, k, mt, N))
                    mean[i], lo[i], hi[i] = ci_of(diff, rng)
                    gain_rows.append({"dataset": ds, "k": k, "measure": mt, "N": N, "gain_shrunk_minus_count": mean[i],
                                      "ci_lo": lo[i], "ci_hi": hi[i],
                                      "n_draws": int(np.isfinite(diff).sum())})
                gains[(ds, k, mt)] = (np.array(Ns, dtype=float), mean, lo, hi)
            for N in Ns:
                c = D.cells[(ds, k, N)]
                kap, nr = c["kappa"], c["n_rows"]
                okk = np.isfinite(kap)
                pw = kap / (kap + nr)
                kappa_rows.append({"dataset": ds, "k": k, "N": N,
                                   "kappa_median": float(np.median(kap[okk])) if okk.any() else NAN,
                                   "prior_weight_median_approx_kappa_over_kappa_plus_n_rows":
                                       float(np.nanmedian(pw)) if np.isfinite(pw).any() else NAN,
                                   "n_draws": int(okk.sum())})
    write_csv(tabdir / f"{NAME}_x2_shrinkage_gain.csv", gain_rows)
    write_csv(tabdir / f"{NAME}_x2_kappa.csv", kappa_rows)

    # peak vs the X1 crossover
    for (ds, k, mt), (Ns, mean, lo, hi) in gains.items():
        if not np.isfinite(mean).any():
            continue
        ip = int(np.nanargmax(mean))
        row = {"dataset": ds, "k": k, "measure": mt, "peak_N": Ns[ip], "peak_gain": mean[ip],
               "peak_ci_lo": lo[ip], "peak_ci_hi": hi[ip], "gain_at_peak_ci_excludes_0": bool(np.isfinite(lo[ip]) and lo[ip] > 0)}
        row.update({"x1_crossover_N_s": NAN, "grid_steps_peak_vs_crossover": NAN, "outcome": "not compared (no X1 crossover defined for this k/measure)"})
        if (ds, k, mt) in x1curves and mt in ("B_ndcg100", "U_ndcg"):
            ref = D.ref.get(ds, {}).get(f"stale.{mt}", NAN)
            cx = crossover(*x1curves[(ds, k, mt)][:2], ref)
            if cx["status"] != "ok":
                row["outcome"] = "not judged (no stale reference)"
            elif cx["censored"]:
                row["outcome"] = "not judged (crossover not reached within sampled N)"
            else:
                steps = abs(int(np.flatnonzero(Ns == cx["N_s"])[0]) - ip)
                row.update({"x1_crossover_N_s": cx["N_s"], "grid_steps_peak_vs_crossover": steps,
                            "outcome": "near" if steps <= NEAR_STEPS else "far"})
        peak_rows.append(row)
    write_csv(tabdir / f"{NAME}_x2_peak_vs_crossover.csv", peak_rows)

    combo_v, nums = {}, {}
    for k in KS_MAIN:
        for mt in ("B_ndcg100", "U_ndcg"):
            sub = [r for r in peak_rows if r["k"] == k and r["measure"] == mt and r["outcome"] in ("near", "far")]
            yes = sum(r["outcome"] == "near" for r in sub)
            tag = f"{mt.split('_')[0]}_{k}"
            combo_v[tag] = majority(yes, len(sub))
            nums[f"{tag}_peak_near_crossover"] = f"{yes}/{len(sub)}"
    verdict = combine(list(combo_v.values()))
    nums["verdict_per_measure"] = combo_v
    allk = [r["kappa_median"] for r in kappa_rows if fin(r["kappa_median"])]
    nums["kappa_median_over_cells"] = float(np.median(allk)) if allk else None
    nums["kappa_median_by_N"] = {}
    for N in sorted({r["N"] for r in kappa_rows}):
        v = [r["kappa_median"] for r in kappa_rows if r["N"] == N and fin(r["kappa_median"])]
        if v:
            nums["kappa_median_by_N"][str(N)] = float(np.median(v))
    bpeaks = [r for r in peak_rows if r["measure"] == "B_ndcg100"]
    best = max((r["peak_gain"] for r in bpeaks), default=NAN)
    medpk = float(np.median([r["peak_N"] for r in bpeaks])) if bpeaks else NAN
    nums["largest_mean_gain_B_ndcg100"] = best
    nums["median_peak_N_B_ndcg100"] = medpk
    nums["control_flat_prior"] = "mathematical fact, not tested: a flat prior adds the same amount to every item, so ranks are unchanged"
    reading_en = (f"Mean gain from shrinking toward the stale list on Truth B peaks at a median N of {fN(medpk)} (largest gain {f3(best)} NDCG@100). "
                  f"Peak within one grid step of the X1 crossover: B/k=all {nums['B_all_peak_near_crossover']}, B/k=5 {nums['B_5_peak_near_crossover']}, "
                  f"panel/k=all {nums['U_all_peak_near_crossover']}, panel/k=5 {nums['U_5_peak_near_crossover']}; the median shrinkage strength kappa over cells is {fN(nums['kappa_median_over_cells'])}. "
                  f"Control: shrinking toward a flat prior leaves ranks unchanged (a mathematical fact, not tested here).")
    reading_zh = (f"向旧榜单收缩带来的平均增益（对 Truth B）在中位数 N={fN(medpk)} 处最大，最大增益为 {f3(best)}（NDCG@100）。"
                  f"增益峰值与 X1 交叉点相差不超过一格的情况：B/k=all {nums['B_all_peak_near_crossover']}，B/k=5 {nums['B_5_peak_near_crossover']}，"
                  f"面板/k=all {nums['U_all_peak_near_crossover']}，面板/k=5 {nums['U_5_peak_near_crossover']}；各格收缩强度 kappa 的中位数为 {fN(nums['kappa_median_over_cells'])}。"
                  f"对照：向“平坦先验”收缩不会改变排名（数学事实，此处未做检验）。")
    out = entry("X2", *X2_TITLE, verdict, nums, reading_en, reading_zh, X2_RULE)

    show = pick4([d for d in D.ds_e1], D.ds_order)
    for mt, lab, ylab in (("B_ndcg100", "B", "NDCG@100 gain vs Truth B"),
                          ("U_ndcg", "U", "panel NDCG@100 gain")):
        if not show:
            break
        fig, axes = grid_axes(len(show), 2, 5.4, 3.7)
        peaks = []
        for ax, ds in zip(axes, show):
            for k in D.ks(ds):
                Ns, m, lo, hi = gains[(ds, k, mt)]
                if not np.isfinite(m).any():
                    continue
                ax.plot(Ns, m, "-o", ms=3, color=COL_K.get(k, "k"), label=klabel(k))
                ax.fill_between(Ns, lo, hi, color=COL_K.get(k, "k"), alpha=0.10, lw=0)
                peaks.append((Ns[int(np.nanargmax(m))], np.nanmax(m)))
            ax.axhline(0, color="k", lw=0.8)
            if "all" in D.ks(ds) and (ds, "all", mt) in x1curves and mt in ("B_ndcg100", "U_ndcg"):
                cx = crossover(*x1curves[(ds, "all", mt)][:2], D.ref.get(ds, {}).get(f"stale.{mt}", NAN))
                if fin(cx["N_s"]):
                    ax.axvline(cx["N_s"], color=COL_K["all"], ls=":", lw=1.5, label="X1 crossover, k=all")
            log_n_axis(ax)
            ax.set_title(ds)
            ax.set_xlabel("N (people sampled)")
            ax.set_ylabel(ylab)
        h, l = {}, {}
        for ax in axes:
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                h.setdefault(ll, hh)
        if peaks:
            pn = float(np.median([p[0] for p in peaks]))
            pg = max(p[1] for p in peaks)
            ttl = f"Exploratory X2: shrinking toward the stale list gains up to {pg:+.3f}, with the median peak at N = {pn:,.0f}"
        else:
            ttl = "Exploratory X2: no shrinkage gain available"
        finish(fig, ttl, list(h.values()), list(h.keys()))
        savefig(fig, figdir, f"x2_shrinkage_gain_{lab}")
    return out


# ----------------------------------------------------------------------------- X3
X3_RULE = "Rank correlation of the power-law exponent and the Zipf slope above 0.6 across the datasets (judged on NDCG@100 error, k=all and k=5; needs >= 5 converged datasets)"
X3_TITLE = ("Exploratory X3: error falls as a power law with a floor, and the exponent follows the Zipf slope",
            "探索性 X3：误差按带下限的幂律下降，且指数与齐普夫斜率同序")
X3_METRICS = (("A_ndcg100", "NDCG@100"), ("A_ov100", "top-100 overlap"), ("A_rbo90", "RBO(0.9)"))


def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return NAN
    return float(spearmanr(x, y).statistic)


def run_x3(D: Data, tabdir: Path, figdir: Path) -> dict:
    if not D.ds_e1:
        return not_testable("X3", *X3_TITLE, X3_RULE, "Needs e1 rows; none found.", "需要 e1 数据，但没有找到。")
    fits, slopes = [], {}
    for ds in D.ds_e1:
        zs = fnum(D.ds_meta.get(ds, {}).get("zipf_s"))
        for k in KS_MAIN:
            Ns = D.grid.get((ds, k), [])
            for mt, _ in X3_METRICS:
                base = {"dataset": ds, "k": k, "measure": mt, "zipf_s": zs, "n_points": len(Ns)}
                if len(Ns) < 4:
                    fits.append({**base, "note": "fewer than 4 N values: no fit"})
                    continue
                E = 1.0 - D.reps(ds, k, f"count.{mt}", Ns)
                Na = np.array(Ns, dtype=float)
                rng = np.random.default_rng(seed_for("x3", ds, k, mt))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    boot = scaling.fit_power_law_boot(Na, E, n_boot=N_BOOT_FIT, rng=rng)
                    meanE = np.nanmean(E, axis=0)
                    cmp = scaling.compare_power_vs_exp(Na, meanE)
                fits.append({**base, "beta": boot["beta"], "beta_ci_lo": boot["ci_lo"], "beta_ci_hi": boot["ci_hi"],
                             "a": boot["a"], "floor_c": boot["c"], "converged": boot["converged"],
                             "frac_boot_nonconverged": boot["frac_nonconverged"], "n_draws": E.shape[0],
                             "aic_power": cmp["aic_power"], "aic_exp": cmp["aic_exp"], "delta_aic_exp_minus_power": cmp["delta_aic"],
                             "better_model": cmp["winner"] or "none converged", "note": ""})
                if k == "all":
                    slopes[(ds, mt)] = (Na, meanE, scaling.local_slope(Na, meanE))
    write_csv(tabdir / f"{NAME}_x3_scaling_fits.csv", fits)

    corr_rows, nums, combo_v = [], {}, {}
    for k in KS_MAIN:
        for mt, mlab in X3_METRICS:
            sub = [r for r in fits if r["k"] == k and r["measure"] == mt and r.get("converged") is True and fin(r["zipf_s"])]
            n = len(sub)
            rho = spearman(np.array([r["beta"] for r in sub]), np.array([r["zipf_s"] for r in sub])) if n else NAN
            lo = hi = NAN
            if n >= MIN_DATASETS_RHO and fin(rho):
                rng = np.random.default_rng(seed_for("x3rho", k, mt))
                b = np.array([r["beta"] for r in sub])
                z = np.array([r["zipf_s"] for r in sub])
                bs = []
                for _ in range(N_BOOT):
                    idx = rng.integers(0, n, n)
                    bs.append(spearman(b[idx], z[idx]))
                bs = np.array(bs)
                if np.isfinite(bs).any():
                    lo, hi = (float(v) for v in np.nanpercentile(bs, [2.5, 97.5]))
            allfit = [r for r in fits if r["k"] == k and r["measure"] == mt and "better_model" in r]
            pw = sum(r["better_model"] == "power" for r in allfit)
            corr_rows.append({"k": k, "measure": mt, "n_datasets_converged": n, "spearman_rho_beta_vs_zipf_s": rho,
                              "rho_boot_ci_lo": lo, "rho_boot_ci_hi": hi,
                              "note": "n is small: rank correlation over datasets is very noisy" if n < 12 else "",
                              "power_law_better_than_exponential_by_AIC": f"{pw}/{len(allfit)}"})
            nums[f"rho_{mt}_{k}"] = rho
            nums[f"n_{mt}_{k}"] = n
            nums[f"power_wins_{mt}_{k}"] = f"{pw}/{len(allfit)}"
            if mt == "A_ndcg100":
                if n < MIN_DATASETS_RHO or not fin(rho):
                    combo_v[k] = "not testable"
                else:
                    combo_v[k] = "supported" if rho > 0.6 else "not supported"
    write_csv(tabdir / f"{NAME}_x3_exponent_vs_zipf.csv", corr_rows)
    verdict = combine(list(combo_v.values())) if all(v != "not testable" for v in combo_v.values()) else \
        ("not testable" if all(v == "not testable" for v in combo_v.values()) else combine(list(combo_v.values())))
    nums["verdict_per_k_on_NDCG"] = combo_v
    r_all, r_5 = nums.get("rho_A_ndcg100_all", NAN), nums.get("rho_A_ndcg100_5", NAN)
    n_all, n_5 = nums.get("n_A_ndcg100_all", 0), nums.get("n_A_ndcg100_5", 0)
    if verdict == "not testable":
        reading_en = (f"Only {n_all} dataset(s) (k=all) and {n_5} (k=5) have a converged power-law fit with a Zipf slope; a rank correlation needs at least "
                      f"{MIN_DATASETS_RHO} (fixed beforehand), so the rule cannot be judged here. Fits and power-vs-exponential comparisons are in the tables.")
        reading_zh = (f"只有 {n_all} 个数据集（k=all）和 {n_5} 个（k=5）有收敛的幂律拟合和齐普夫斜率；等级相关至少需要 {MIN_DATASETS_RHO} 个（事先规定），"
                      f"因此这里无法判定。拟合与幂律对指数的比较见表格。")
    else:
        reading_en = (f"Rank correlation between the NDCG@100 error exponent and the Zipf slope is {f3(r_all, 2)} at k=all (n={n_all} datasets) and {f3(r_5, 2)} at k=5 (n={n_5}); "
                      f"the prediction was above 0.6. n is small, so these correlations are noisy.")
        reading_zh = (f"NDCG@100 误差指数与齐普夫斜率的等级相关：k=all 为 {f3(r_all, 2)}（{n_all} 个数据集），k=5 为 {f3(r_5, 2)}（{n_5} 个）；预测值是高于 0.6。数据集数量少，相关系数噪声大。")
    out = entry("X3", *X3_TITLE, verdict, nums, reading_en, reading_zh, X3_RULE)

    # figure 1: exponent vs Zipf slope
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), squeeze=False)
    axes = axes.ravel()
    for ax, k in zip(axes, KS_MAIN):
        sub = [r for r in fits if r["k"] == k and r["measure"] == "A_ndcg100" and r.get("converged") is True and fin(r["zipf_s"])]
        for r in sub:
            yerr = None
            if fin(r["beta_ci_lo"]) and fin(r["beta_ci_hi"]):
                yerr = [[max(r["beta"] - r["beta_ci_lo"], 0)], [max(r["beta_ci_hi"] - r["beta"], 0)]]
            ax.errorbar([r["zipf_s"]], [r["beta"]], yerr=yerr, fmt="o", color=COL_K[k], capsize=3)
            ax.annotate(r["dataset"], (r["zipf_s"], r["beta"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
        rho = nums.get(f"rho_A_ndcg100_{k}", NAN)
        ax.set_title(f"{klabel(k)}: Spearman rho = {f3(rho, 2)}, n = {len(sub)} datasets (n small)")
        ax.set_xlabel("Zipf slope s of the dataset (unitless)")
        ax.set_ylabel("fitted exponent beta of NDCG@100 error (unitless)")
        ax.grid(alpha=0.25)
        if not sub:
            ax.text(0.5, 0.5, "no converged fit", transform=ax.transAxes, ha="center")
    if fin(r_all) or fin(r_5):
        t3 = (f"Exploratory X3: exponent vs Zipf slope, rank correlation {f3(r_all, 2)} at k=all and {f3(r_5, 2)} at k=5 (prediction: > 0.6); "
              "points = fit to the mean curve, bars = 95% bootstrap interval over draws")
    else:
        t3 = (f"Exploratory X3: exponent vs Zipf slope, only {n_all} dataset(s) with a fit, too few for a rank correlation (prediction: > 0.6); "
              "points = fit to the mean curve, bars = 95% bootstrap interval over draws")
    finish(fig, t3)
    savefig(fig, figdir, "x3_exponent_vs_zipf")

    # figure 2: local slope curves
    show = [d for d in pick4(D.ds_e1, D.ds_order) if any((d, mt) in slopes for mt, _ in X3_METRICS)]
    if show:
        fig, axes = grid_axes(len(show), 2, 5.4, 3.7)
        cols = {"A_ndcg100": "#1f77b4", "A_ov100": "#ff7f0e", "A_rbo90": "#2ca02c"}
        first, last = [], []
        for ax, ds in zip(axes, show):
            for mt, mlab in X3_METRICS:
                if (ds, mt) not in slopes:
                    continue
                Na, _, sl = slopes[(ds, mt)]
                mid = np.sqrt(Na[:-1] * Na[1:])
                ax.plot(mid, sl, "-o", ms=3.5, color=cols[mt], label=f"{mlab} error")
                if mt == "A_ndcg100" and np.isfinite(sl).any():
                    f_ = sl[np.isfinite(sl)]
                    first.append(f_[0])
                    last.append(f_[-1])
            ax.axhline(0, color="k", lw=0.8)
            log_n_axis(ax)
            ax.set_title(f"{ds} (k=all)")
            ax.set_xlabel("N (people; middle of a grid step)")
            ax.set_ylabel("local log-log slope of error")
        h, l = {}, {}
        for ax in axes:
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                h.setdefault(ll, hh)
        finish(fig, (f"Exploratory X3: local slope of the NDCG@100 error moves from {np.median(first):.2f} (smallest N) to "
                     f"{np.median(last):.2f} (largest N), median over shown datasets") if first else "Exploratory X3: local slope of the error",
               list(h.values()), list(h.keys()))
        savefig(fig, figdir, "x3_local_slope")
    return out


# ----------------------------------------------------------------------------- X4
X4_RULE = "Observed correctly ordered depth >= the Dyer-Owen bound at every N (cannot be read from e1 rows; proxy only)"
X4_TITLE = ("Exploratory X4: the Dyer-Owen bound is conservative",
            "探索性 X4：Dyer-Owen 界是保守的")


def first_reach(Ns, means, target):
    for N, m in zip(Ns, means):
        if np.isfinite(m) and m >= target:
            return float(N)
    return NAN


def run_x4(D: Data, tabdir: Path, figdir: Path) -> dict:
    dss = [d for d in D.ds_e1 if d in D.ds_meta]
    why_en = ("The observed number of correctly ordered top items is not stored in the e1 rows, so the rule as written cannot be tested. "
              "Only a proxy is reported: the predicted resolve-N beside the N where top-10 overlap first reaches 0.8 (k=all).")
    why_zh = ("e1 数据中没有保存“实际排对的前几名数量”，因此该规则无法按原文检验。这里只给出代理指标：预测的“排清所需 N”与前 10 名重合度首次达到 0.8 的 N（k=all）。")
    if not dss:
        return not_testable("X4", *X4_TITLE, X4_RULE, "Needs e1 rows and datasets.json; missing. " + why_en,
                            "缺少 e1 数据或 datasets.json。" + why_zh)
    rows = []
    for ds in dss:
        m = D.ds_meta[ds]
        dep = m.get("depth_at_N", {})
        row = {"dataset": ds, "n_pop": m.get("n_pop"), "predicted_depth_N50": dep.get("50"), "predicted_depth_N200": dep.get("200"),
               "predicted_depth_N1000": dep.get("1000"), "predicted_depth_N5000": dep.get("5000"),
               "predicted_n_resolve_top10": fnum(m.get("n_resolve_top10")), "predicted_n_resolve_top100": fnum(m.get("n_resolve_top100"))}
        Ns, ov10 = D.mean_curve(ds, "all", "count.A_ov10")
        _, ov100 = D.mean_curve(ds, "all", "count.A_ov100")
        n10, n100 = first_reach(Ns, ov10, OV_TARGET), first_reach(Ns, ov100, OV_TARGET)
        maxN = float(Ns[-1]) if len(Ns) else NAN
        row.update({"largest_N_sampled": maxN, "mean_ov10_at_largest_N": ov10[-1] if len(ov10) else NAN,
                    "observed_N_ov10_reaches_0.8": n10, "ov10_0.8_reached": fin(n10),
                    "observed_N_ov100_reaches_0.8": n100, "ov100_0.8_reached": fin(n100),
                    "predicted_resolve_top10_over_n_pop": row["predicted_n_resolve_top10"] / fnum(m.get("n_pop")) if fin(m.get("n_pop")) else NAN})
        pred = row["predicted_n_resolve_top10"]
        if not len(Ns) or not fin(pred):
            oc = "no data"
        elif fin(n10):
            oc = "bound not contradicted (observed N <= predicted N)" if n10 <= pred else "observed N exceeds predicted N"
        else:
            oc = "observed N exceeds predicted N" if maxN >= pred else "uninformative (0.8 not reached and largest N below predicted N)"
        row["proxy_outcome"] = oc
        row["note"] = "PROXY: top-10 overlap 0.8 is not the same event as the bound's 'top-10 set resolved'; exploratory"
        rows.append(row)
    write_csv(tabdir / f"{NAME}_x4_dyer_owen_proxy.csv", rows)
    nok = sum(r["proxy_outcome"].startswith("bound not") for r in rows)
    nbad = sum(r["proxy_outcome"].startswith("observed N exceeds") for r in rows)
    nun = sum(r["proxy_outcome"].startswith("uninformative") for r in rows)
    nreach = sum(bool(r["ov10_0.8_reached"]) for r in rows)
    preds = [r["predicted_n_resolve_top10"] for r in rows if fin(r["predicted_n_resolve_top10"])]
    nums = {"n_datasets": len(rows), "proxy_not_contradicted": nok, "proxy_observed_exceeds_predicted": nbad, "proxy_uninformative": nun,
            "datasets_where_ov10_reaches_0.8": nreach,
            "predicted_n_resolve_top10_range": [min(preds), max(preds)] if preds else None,
            "depth_at_N5000": {r["dataset"]: r["predicted_depth_N5000"] for r in rows}}
    reading_en = (f"{why_en} Top-10 overlap reached 0.8 within the sampled N in {nreach} of {len(rows)} datasets, "
                  f"while the bound predicts {fN(min(preds)) if preds else 'n/a'} to {fN(max(preds)) if preds else 'n/a'} people to settle the top-10; "
                  f"proxy outcome: {nok} not contradicted, {nbad} exceeding, {nun} uninformative.")
    reading_zh = (f"{why_zh}在已抽样的 N 范围内，{len(rows)} 个数据集中有 {nreach} 个的前 10 名重合度达到 0.8；"
                  f"而该界预测需要 {fN(min(preds)) if preds else 'n/a'} 到 {fN(max(preds)) if preds else 'n/a'} 人才能排清前 10 名；"
                  f"代理结果：{nok} 个未被推翻，{nbad} 个超出预测，{nun} 个无法判断。")
    out = entry("X4", *X4_TITLE, "not testable", nums, reading_en, reading_zh, X4_RULE)

    # figure: dumbbell per dataset
    order = sorted(rows, key=lambda r: r["predicted_n_resolve_top10"] if fin(r["predicted_n_resolve_top10"]) else math.inf)
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(order) + 2.6))
    for i, r in enumerate(order):
        y = i
        pred = r["predicted_n_resolve_top10"]
        obs = r["observed_N_ov10_reaches_0.8"] if fin(r["observed_N_ov10_reaches_0.8"]) else r["largest_N_sampled"]
        ax.plot([obs, pred], [y, y], color="#bbbbbb", lw=1, zorder=1)
        ax.scatter([pred], [y], marker="s", s=45, color="#d62728", zorder=3, label="predicted N to resolve the top-10 set (bound)")
        if r["ov10_0.8_reached"]:
            ax.scatter([obs], [y], marker="o", s=45, color="#1f77b4", zorder=3, label="observed N where top-10 overlap first reaches 0.8")
        else:
            ax.scatter([obs], [y], marker=">", s=60, facecolor="white", edgecolor="#1f77b4", zorder=3,
                       label="largest N sampled, overlap 0.8 not reached (arrow: observed N is larger)")
        if fin(r["n_pop"]):
            ax.scatter([r["n_pop"]], [y], marker="|", s=120, color="k", zorder=2, label="group size")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([r["dataset"] for r in order])
    ax.set_xscale("log")
    ax.set_xlabel("number of people N (log scale)")
    ax.grid(alpha=0.25, axis="x")
    h, l = {}, {}
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        h.setdefault(ll, hh)
    finish(fig, f"Exploratory X4 (proxy): top-10 overlap reaches 0.8 in {nreach} of {len(rows)} datasets within the sampled N; "
                f"the bound predicts {fN(min(preds)) if preds else 'n/a'} to {fN(max(preds)) if preds else 'n/a'} people to settle the top-10",
           list(h.values()), list(h.keys()), ncol=1)
    savefig(fig, figdir, "x4_dyer_owen_proxy")
    return out


# ----------------------------------------------------------------------------- X10
X10_RULE = "Drift gap (Truth A vs Truth B oracle lists) exceeds the sampling gap in most datasets (judged at N=1000, k=all; measures: panel NDCG@100 and NDCG@100 vs the truth lists)"
X10_TITLE = ("Exploratory X10: the drift between Truth A and Truth B is larger than the sampling error",
             "探索性 X10：真相 A 与真相 B 之间的漂移大于抽样误差")


def run_x10(D: Data, tabdir: Path, figdir: Path) -> dict:
    dss = [d for d in D.ds_all if d in D.ref]
    if not dss or not D.cells:
        return not_testable("X10", *X10_TITLE, X10_RULE, "Needs ref.jsonl and e1 rows; at least one is missing.",
                            "需要 ref.jsonl 和 e1 数据，至少缺少其中一个。")
    rows = []
    for ds in dss:
        ref = D.ref[ds]
        Ns = D.grid.get((ds, "all"), [])
        for mt, tname in (("U_ndcg", "panel_ndcg"), ("B_ndcg100", "ndcg_vs_truth_lists")):
            tA, tB = ref.get(f"truthA.{mt}", NAN), ref.get(f"truthB.{mt}", NAN)
            row = {"dataset": ds, "measure": tname, "truthA_ref": tA, "truthB_ref": tB, "drift_gap_B_minus_A": tB - tA}
            for lab, N in (("N1000", 1000), ("Nmax", Ns[-1] if Ns else None)):
                if N is None or N not in Ns:
                    row.update({f"{lab}": N if N is not None else NAN, f"sampling_gap_{lab}": NAN, f"sampling_gap_{lab}_ci_lo": NAN,
                                f"sampling_gap_{lab}_ci_hi": NAN})
                    continue
                diff = tA - D.cells[(ds, "all", N)][f"count.{mt}"]
                rng = np.random.default_rng(seed_for("x10", ds, mt, N))
                m, lo, hi = ci_of(diff, rng)
                row.update({lab: N, f"sampling_gap_{lab}": m, f"sampling_gap_{lab}_ci_lo": lo, f"sampling_gap_{lab}_ci_hi": hi})
            d = row["drift_gap_B_minus_A"]
            g = row["sampling_gap_N1000"]
            row["drift_exceeds_sampling_gap_N1000"] = (bool(d > g) if fin(d) and fin(g) else "not judged")
            g2 = row["sampling_gap_Nmax"]
            row["drift_exceeds_sampling_gap_Nmax"] = (bool(d > g2) if fin(d) and fin(g2) else "not judged")
            rows.append(row)
    write_csv(tabdir / f"{NAME}_x10_drift_vs_sampling.csv", rows)

    nums, combo_v = {}, {}
    for mt, tname in (("U_ndcg", "panel_ndcg"), ("B_ndcg100", "ndcg_vs_truth_lists")):
        sub = [r for r in rows if r["measure"] == tname]
        for col, lab in (("drift_exceeds_sampling_gap_N1000", "N1000"), ("drift_exceeds_sampling_gap_Nmax", "Nmax")):
            det = [r for r in sub if r[col] != "not judged"]
            yes = sum(r[col] is True for r in det)
            nums[f"{tname}_{lab}_drift_exceeds"] = f"{yes}/{len(det)}"
            if lab == "N1000":
                combo_v[tname] = majority(yes, len(det))
        dr = [r["drift_gap_B_minus_A"] for r in sub if fin(r["drift_gap_B_minus_A"])]
        sg = [r["sampling_gap_N1000"] for r in sub if fin(r["sampling_gap_N1000"])]
        nums[f"{tname}_median_drift_gap"] = float(np.median(dr)) if dr else None
        nums[f"{tname}_median_sampling_gap_N1000"] = float(np.median(sg)) if sg else None
    verdict = combine(list(combo_v.values()))
    nums["verdict_per_measure"] = combo_v
    if verdict == "not testable":
        reading_en = ("No dataset has both a Truth A / Truth B reference and a k=all cell at N=1000 in the data available, so the rule cannot be judged yet. "
                      "Gaps at the largest N are in the table.")
        reading_zh = "现有数据中没有任何数据集同时具备 Truth A/Truth B 参考和 N=1000、k=all 的格子，因此暂时无法判定；最大 N 处的差距见表格。"
    else:
        reading_en = (f"On the panel score, the drift gap is larger than the sampling gap at N=1000 in {nums['panel_ndcg_N1000_drift_exceeds']} datasets "
                      f"(median drift {f3(nums['panel_ndcg_median_drift_gap'])} vs sampling {f3(nums['panel_ndcg_median_sampling_gap_N1000'])}); "
                      f"on NDCG@100 against the truth lists it is {nums['ndcg_vs_truth_lists_N1000_drift_exceeds']} "
                      f"(median drift {f3(nums['ndcg_vs_truth_lists_median_drift_gap'])} vs sampling {f3(nums['ndcg_vs_truth_lists_median_sampling_gap_N1000'])}). "
                      f"At the largest N the counts are {nums['panel_ndcg_Nmax_drift_exceeds']} and {nums['ndcg_vs_truth_lists_Nmax_drift_exceeds']}.")
        reading_zh = (f"按面板得分，在 N=1000 时漂移差距大于抽样差距的数据集为 {nums['panel_ndcg_N1000_drift_exceeds']}"
                      f"（漂移中位数 {f3(nums['panel_ndcg_median_drift_gap'])}，抽样中位数 {f3(nums['panel_ndcg_median_sampling_gap_N1000'])}）；"
                      f"按对真相榜单的 NDCG@100 为 {nums['ndcg_vs_truth_lists_N1000_drift_exceeds']}"
                      f"（漂移中位数 {f3(nums['ndcg_vs_truth_lists_median_drift_gap'])}，抽样中位数 {f3(nums['ndcg_vs_truth_lists_median_sampling_gap_N1000'])}）。"
                      f"在最大 N 处分别为 {nums['panel_ndcg_Nmax_drift_exceeds']} 和 {nums['ndcg_vs_truth_lists_Nmax_drift_exceeds']}。")
    out = entry("X10", *X10_TITLE, verdict, nums, reading_en, reading_zh, X10_RULE)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharex=False)
    for ax, (tname, ylab) in zip(axes, (("panel_ndcg", "gap in panel NDCG@100"), ("ndcg_vs_truth_lists", "gap in NDCG@100 vs the truth lists"))):
        sub = [r for r in rows if r["measure"] == tname]
        x = np.arange(len(sub))
        w = 0.27
        ax.bar(x - w, [r["drift_gap_B_minus_A"] for r in sub], w, color="#d62728", label="drift gap: Truth B list minus Truth A list")
        for off, lab, col, txt in ((0, "N1000", "#1f77b4", "sampling gap, N = 1000 (k=all)"), (w, "Nmax", "#9ecae1", "sampling gap, largest N (k=all)")):
            vals = np.array([r[f"sampling_gap_{lab}"] for r in sub], dtype=float)
            lo = np.array([r[f"sampling_gap_{lab}_ci_lo"] for r in sub], dtype=float)
            hi = np.array([r[f"sampling_gap_{lab}_ci_hi"] for r in sub], dtype=float)
            yerr = np.vstack([np.where(np.isfinite(lo), vals - lo, 0), np.where(np.isfinite(hi), hi - vals, 0)])
            ax.bar(x + off, np.nan_to_num(vals), w, yerr=yerr, color=col, label=txt, capsize=2)
            for xi, v in zip(x + off, vals):
                if not np.isfinite(v):
                    ax.text(xi, 0, "n/a", ha="center", va="bottom", fontsize=7, rotation=90)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([r["dataset"] for r in sub], rotation=45, ha="right")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, axis="y")
    d_u = nums["panel_ndcg_N1000_drift_exceeds"]
    d_b = nums["ndcg_vs_truth_lists_N1000_drift_exceeds"]
    axes[0].set_title("panel (people's real later choices)", fontsize=10)
    axes[1].set_title("list quality vs the truth lists", fontsize=10)
    h, l = axes[0].get_legend_handles_labels()
    finish(fig, f"Exploratory X10: the drift gap exceeds the sampling gap at N=1000 in {d_u} datasets (panel) and {d_b} (truth lists)", h, l)
    savefig(fig, figdir, "x10_drift_vs_sampling")
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
    print(f"loaded {len(D.cells)} e1 cells, {len(D.ref)} ref rows, datasets with e1: {D.ds_e1}  ({time.time() - t0:.1f}s)")

    verdicts = []
    x1curves = {}

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

    def do_x1():
        nonlocal x1curves
        r, x1curves = run_x1(D, tabdir, figdir)
        return r

    verdicts.append(guarded("X1", X1_TITLE, X1_RULE, do_x1))
    verdicts.append(guarded("X2", X2_TITLE, X2_RULE, lambda: run_x2(D, x1curves, None, tabdir, figdir)))
    verdicts.append(guarded("X3", X3_TITLE, X3_RULE, lambda: run_x3(D, tabdir, figdir)))
    verdicts.append(guarded("X4", X4_TITLE, X4_RULE, lambda: run_x4(D, tabdir, figdir)))
    verdicts.append(guarded("X10", X10_TITLE, X10_RULE, lambda: run_x10(D, tabdir, figdir)))
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
