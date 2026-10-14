#!/usr/bin/env python3
"""Extra exploratory analysis (families X13, X14, X15; amendments A9, A13 of the pre-registration).

Everything written by this script is EXPLORATORY: no confirmatory claim, no p-value, and nothing
here has been re-run on fresh seeds (replication status: not replicated).

  X13  profile length vs training depth (stage e2p): crossover N*(k_train, kprof) of a trained
       model over the popularity list on the panel; by history-length tercile; tercile policy
  X14  self-assessment (stage e8): can ONE sample tell how good its own list is?
       (user-level vs row-level bootstrap prediction of the list's NDCG@100 against Truth A)
  X15  unbiased vs plain count (stage e1): does `ht` (each kept choice weighted by history
       length / k, unbiased for Truth A) ever beat the plain `count` on Truth A?

Usage:
  python3 scripts/ssp/an_extra.py --runs <dir with e2p.jsonl, e8.jsonl, e1.jsonl> --out <dir>
Writes <out>/03_tables/an_extra_*.csv, <out>/04_figures/an_extra_*.png and
<out>/03_tables/an_extra_verdicts.json (entries X13, X13b, X13c, X14, X15). A missing stage gives
'not testable' for its family; the other families still run.

Thresholds fixed BEFORE looking at any number (each is repeated in the matching verdict `rule`):
  "most datasets"  = strictly more than half of the datasets that can be judged (A14);
                     exactly half = 'mixed'; none judgeable = 'not testable'.
  crossover N*     = smallest N whose paired-by-draw 95% interval of (model - pop) is above 0 AND
                     whose mean difference stays above 0 at every larger tested N (A12 form);
                     'none' = no such N up to the largest tested N (a lower bound, not a zero).
  a grid cell enters a curve only with >= 4 usable draws.
A family with several measures (e.g. two models, three parts) is 'supported' only if every
judgeable measure is 'supported', 'not supported' only if every one is, else 'mixed'.
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
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    import metrics  # noqa: E402
except ImportError:  # helper not in this worktree: read it from the main repository
    sys.path.insert(1, "/Users/LeoQin/Documents/GitHub/Federated_Data_Gathering_Research/scripts/ssp")
    import metrics  # noqa: E402

REPO = HERE.parent.parent
NAME = "an_extra"
NAN = float("nan")
N_BOOT = 2000             # interval resamples over draws
MIN_DRAWS = 4             # a cell needs this many usable draws to enter a curve
KPROFS = ["1", "2", "5", "10", "20", "all"]
STRATA = ["all", "0", "1", "2"]
MODELS = ["pop", "ease100", "knn100", "ease50", "knn50"]
MODELS_CROSS = ["knn100", "ease100"]          # models compared with pop in X13
STRATUM_NAME = {"all": "all people", "0": "light history", "1": "middle history", "2": "heavy history"}
X13_WITHIN = 1            # 'within one grid step'
X13_FAR = 2               # 'differ by >= 2 grid steps'
X13_MIN_YES = 5           # pre-registered counts (of the datasets that have an N* for both)
X13_MIN_NO = 4
RATIO_LO, RATIO_HI = 0.67, 1.5   # X14 calibration band
SPEARMAN_MIN = 0.5
N_CAL = 200               # X14 rule is judged for N >= 200
SHARE_HT = 0.10           # X15: ht 'wins late' threshold = 10% of the group
PRIORITY = ["ml-1m", "amazon-software", "gowalla", "lastfm-1k", "kuairand-pure", "amazon-books",
            "ml-latest", "brightkite", "amazon-video-games", "amazon-musical-instruments", "amazon-movies-tv"]

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "legend.fontsize": 9, "figure.dpi": 130, "savefig.dpi": 130,
                     "axes.spines.top": False, "axes.spines.right": False})


# ----------------------------------------------------------------------------- helpers
def fnum(v) -> float:
    try:
        return NAN if v is None else float(v)
    except (TypeError, ValueError):
        return NAN


def kkey(k):
    return "all" if k is None or k == "all" else int(k)


def kord(k) -> float:
    return math.inf if k == "all" else float(k)


def seed_for(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def fin(x) -> bool:
    return x is not None and isinstance(x, (int, float, np.floating, np.integer)) and math.isfinite(float(x))


def f3(x, nd=3) -> str:
    return "n/a" if not fin(x) else f"{x:.{nd}f}"


def fN(x) -> str:
    return "none" if not fin(x) else f"{x:,.0f}"


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


def order_ds(names, meta_order) -> list[str]:
    pri = [d for d in PRIORITY if d in names]
    rest = [d for d in meta_order if d in names and d not in pri]
    rest += sorted(d for d in names if d not in pri and d not in rest)
    return pri + rest


def ci_of(values, rng) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (NAN, NAN, NAN)
    if v.size == 1:
        return (float(v[0]), NAN, NAN)
    return metrics.paired_bootstrap_ci(v, n_boot=N_BOOT, rng=rng)


def crossover(Ns, means, los) -> float:
    """Smallest N with interval above 0 whose mean stays above 0 at every larger N; NaN = none."""
    for i in range(len(Ns)):
        if los[i] > 0 and np.all(means[i:] > 0):
            return float(Ns[i])
    return NAN


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return NAN
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return NAN
    return float(np.corrcoef(rx, ry)[0, 1])


def read_jsonl(p: Path):
    bad = 0
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"NOTE: {p.name}: {bad} unreadable line(s) skipped")


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, runs: Path, datasets_path: Path):
        self.messages: list[str] = []
        self.ds_meta: dict[str, dict] = {}
        self.ds_order: list[str] = []
        self.e2p: dict[tuple, dict] = {}      # (ds, k_train, N) -> {draws, cols}
        self.e8: dict[tuple, dict] = {}       # (ds, k, N) -> arrays by field
        self.e1: dict[tuple, dict] = {}       # (ds, k, N) -> arrays by field
        self.e1_fields_seen: set[str] = set()
        self._load_datasets(datasets_path)
        self._load_e2p(runs / "e2p.jsonl")
        self._load_e8(runs / "e8.jsonl")
        self._load_e1(runs / "e1.jsonl")
        self._cache: dict[tuple, dict | None] = {}

    def _load_datasets(self, p: Path):
        if not p.exists():
            self.messages.append(f"datasets.json not found at {p}: group sizes unavailable")
            return
        for d in json.loads(p.read_text()):
            self.ds_meta[d["name"]] = d
            self.ds_order.append(d["name"])

    def npop(self, ds) -> float:
        return fnum(self.ds_meta.get(ds, {}).get("n_pop"))

    @staticmethod
    def _stack(byd: dict[int, dict], fields: list[str]) -> dict:
        draws = np.array(sorted(byd))
        cell = {"draws": draws}
        for fl in fields:
            cell[fl] = np.array([byd[d].get(fl, NAN) for d in draws], dtype=float)
        return cell

    def _load_e2p(self, p: Path):
        if not p.exists():
            self.messages.append(f"e2p.jsonl not found at {p}: profile-length stage missing")
            return
        raw: dict[tuple, dict] = defaultdict(dict)
        for r in read_jsonl(p):
            key = (r["dataset"], kkey(r.get("k")), int(r["N"]))
            raw[key][int(r["draw"])] = {c: fnum(v) for c, v in r.items() if c.startswith("kp")}
        for key, byd in raw.items():
            cols = sorted({c for d in byd.values() for c in d})
            cell = self._stack(byd, cols)
            self.e2p[key] = {"draws": cell.pop("draws"), "cols": cell}
        if not self.e2p:
            self.messages.append("e2p.jsonl holds no rows")

    def _load_e8(self, p: Path):
        if not p.exists():
            self.messages.append(f"e8.jsonl not found at {p}: self-assessment stage missing")
            return
        fields = ["real_A", "real_Au", "real_B", "pred_user", "pred_row", "n_users", "n_rows"]
        raw: dict[tuple, dict] = defaultdict(dict)
        for r in read_jsonl(p):
            raw[(r["dataset"], kkey(r.get("k")), int(r["N"]))][int(r["draw"])] = {f: fnum(r.get(f)) for f in fields}
        for key, byd in raw.items():
            self.e8[key] = self._stack(byd, fields)
        if not self.e8:
            self.messages.append("e8.jsonl holds no rows")

    def _load_e1(self, p: Path):
        if not p.exists():
            self.messages.append(f"e1.jsonl not found at {p}: popularity-estimator stage missing")
            return
        fields = ["count.A_ndcg100", "ht.A_ndcg100", "count.Au_ndcg100", "ht.Au_ndcg100"]
        raw: dict[tuple, dict] = defaultdict(dict)
        for r in read_jsonl(p):
            arm = r.get("arm")
            if arm is None:
                arm = "uniform"
                for part in str(r.get("key", "")).split("|"):
                    if part.startswith("arm="):
                        arm = part[4:]
            if arm != "uniform" or r.get("users", "uniform") != "uniform" or r.get("rows", "random") != "random":
                continue
            for fl in fields:
                if fl in r:
                    self.e1_fields_seen.add(fl)
            raw[(r["dataset"], kkey(r.get("k")), int(r["N"]))][int(r["draw"])] = {f: fnum(r.get(f)) for f in fields}
        for key, byd in raw.items():
            self.e1[key] = self._stack(byd, fields)
        if not self.e1:
            self.messages.append("e1.jsonl holds no uniform-arm main-grid rows")

    # ---- e2p accessors
    def e2p_ds(self) -> list[str]:
        return order_ds({d for (d, _, _) in self.e2p}, self.ds_order)

    def e2p_kts(self, ds) -> list:
        return sorted({k for (d, k, _) in self.e2p if d == ds}, key=kord)

    def e2p_grid(self, ds) -> list[int]:
        return sorted({n for (d, _, n) in self.e2p if d == ds})

    def kprofs(self, ds) -> list[str]:
        have = {c.split(".")[0][2:] for (d, _, _), c in self.e2p.items() if d == ds for c in c["cols"]}
        return [k for k in KPROFS if k in have]

    def curve(self, ds, kt, kprof, stratum, model) -> dict | None:
        """Paired-by-draw curve of (model - pop) over N for one (dataset, k_train, kprof, stratum)."""
        key = (ds, kt, kprof, stratum, model)
        if key in self._cache:
            return self._cache[key]
        cm, cp = f"kp{kprof}.s{stratum}.{model}", f"kp{kprof}.s{stratum}.pop"
        rows = []
        for N in sorted(n for (d, k, n) in self.e2p if d == ds and k == kt):
            c = self.e2p[(ds, kt, N)]["cols"]
            if cm not in c or cp not in c:
                continue
            ok = np.isfinite(c[cm]) & np.isfinite(c[cp])
            if ok.sum() < MIN_DRAWS:
                continue
            rng = np.random.default_rng(seed_for("x13", *key, N))
            m, lo, hi = ci_of(c[cm][ok] - c[cp][ok], rng)
            rows.append((N, int(ok.sum()), m, lo, hi, float(c[cp][ok].mean()), float(c[cm][ok].mean())))
        out = None
        if rows:
            a = np.array(rows, dtype=float)
            out = {"N": a[:, 0], "n": a[:, 1], "mean": a[:, 2], "lo": a[:, 3], "hi": a[:, 4],
                   "mean_pop": a[:, 5], "mean_model": a[:, 6]}
            out["nstar"] = crossover(out["N"], out["mean"], out["lo"])
        self._cache[key] = out
        return out


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
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows + 1.2), squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def finish(fig, title: str, handles=None, labels=None, ncol: int | None = None) -> None:
    """Wrapped finding-title on top, shared legend at the bottom, then tight layout in the remaining box."""
    W, H = fig.get_size_inches()
    t = "\n".join(textwrap.wrap(title, max(40, int(W * 9.5))))
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


# ----------------------------------------------------------------------------- X13
X13_TITLE = ("Exploratory X13: the crossover sample size is set by the served profile length, not by the training depth",
             "探索性 X13：交叉点样本量主要由展示给用户的历史长度决定，而不是由训练深度决定")
X13_RULE = ("Per dataset and model (knn100, ease100) vs pop at stratum=all, kprof=all: N*(k_train) = smallest N whose paired-by-draw 95% interval "
            "of (model - pop) is above 0 and whose mean stays above 0 at every larger N. Supported if N*(k_train=5) is within one grid step of "
            "N*(k_train=all) in >= 5 of the datasets that have an N* for both; not supported if they differ by >= 2 steps in >= 4 of them; "
            "not testable if fewer than 4 datasets have an N* for both; else mixed. Models are combined (supported only if both are).")
X13B_TITLE = ("Exploratory X13b: people with a long history benefit from personalisation at smaller samples than people with a short one",
              "探索性 X13b：历史长的人比历史短的人在更小的样本量下就能从个性化中受益")
X13B_RULE = ("Per dataset and model, kprof=all, k_train=all: ratio = N*(heavy tercile) / N*(light tercile). 'Far smaller' = ratio <= 0.5; a light "
             "tercile with no N* counts as N* above the largest tested N (so a heavy N* at most half the largest N counts as far smaller; a heavy "
             "N* above that is not judged). Datasets with no N* in either tercile are not judged. Supported if far smaller in strictly more than "
             "half of the judged datasets; models combined.")
X13C_TITLE = ("Exploratory X13c: serving each history tercile its own best model beats the best single model",
              "探索性 X13c：按历史长度三分位分别选最优模型，优于只用一个最优模型")
X13C_RULE = ("Per dataset at the largest N with >= 4 draws, kprof=all, k_train=all: tercile policy = for each tercile the model (of pop, ease100, knn100, "
             "ease50, knn50) with the best cell mean; best single policy = the one model with the best mean over the three terciles. Gain = per-draw "
             "paired difference of the tercile-average scores. A dataset counts when the 95% interval of the gain is above 0 AND the split-half "
             "gain (choose on even draws, score on odd draws, and the reverse) is above 0. Supported if so in strictly more than half of the datasets.")


def nstar_label(v) -> str:
    return fN(v) if fin(v) else "none"


def run_x13(D: Data, tabdir: Path, figdir: Path) -> list[dict]:
    dss = D.e2p_ds()
    if not dss:
        why = ("Needs the e2p stage (panel scores for served profile lengths), which is missing or empty in the data available.",
               "需要 e2p 阶段（不同展示长度下的面板得分），现有数据中缺失或为空。")
        return [not_testable("X13", *X13_TITLE, X13_RULE, *why),
                not_testable("X13b", *X13B_TITLE, X13B_RULE, *why),
                not_testable("X13c", *X13C_TITLE, X13C_RULE, *why)]

    # ---- tables: every N*, and the curves behind them
    nrows, crow = [], []
    for ds in dss:
        grid = D.e2p_grid(ds)
        for kt in D.e2p_kts(ds):
            for kp in D.kprofs(ds):
                for st in STRATA:
                    for mdl in MODELS_CROSS:
                        c = D.curve(ds, kt, kp, st, mdl)
                        base = {"dataset": ds, "k_train": kt, "kprof": kp, "stratum": st, "model": mdl}
                        if c is None:
                            nrows.append({**base, "N_star": NAN, "N_star_label": "no data", "N_max_tested": NAN})
                            continue
                        ns = c["nstar"]
                        nrows.append({**base, "N_star": ns, "N_star_label": nstar_label(ns), "N_max_tested": c["N"][-1],
                                      "N_star_grid_index": grid.index(int(ns)) if fin(ns) and int(ns) in grid else NAN,
                                      "mean_diff_at_Nmax": c["mean"][-1], "ci_lo_at_Nmax": c["lo"][-1], "ci_hi_at_Nmax": c["hi"][-1]})
                        for i in range(len(c["N"])):
                            crow.append({**base, "N": c["N"][i], "n_draws": c["n"][i], "mean_pop": c["mean_pop"][i],
                                         "mean_model": c["mean_model"][i], "mean_diff": c["mean"][i],
                                         "ci_lo": c["lo"][i], "ci_hi": c["hi"][i]})
    write_csv(tabdir / f"{NAME}_x13_nstar.csv", nrows)
    write_csv(tabdir / f"{NAME}_x13_diff_curves.csv", crow)

    def nstar(ds, kt, kp, st, mdl):
        c = D.curve(ds, kt, kp, st, mdl)
        return None if c is None else c["nstar"]       # None = no data, NaN = none

    # ---- part 1: k_train 5 vs all at kprof=all
    krows, per_model = [], {}
    for mdl in MODELS_CROSS:
        both = within = far = one_sided = 0
        for ds in dss:
            grid = D.e2p_grid(ds)
            a, b = nstar(ds, 5, "all", "all", mdl), nstar(ds, "all", "all", "all", mdl)
            row = {"dataset": ds, "model": mdl, "N_star_k5": a if a is not None else NAN, "N_star_kall": b if b is not None else NAN,
                   "label_k5": "no data" if a is None else nstar_label(a), "label_kall": "no data" if b is None else nstar_label(b)}
            if a is not None and b is not None:
                if fin(a) and fin(b) and int(a) in grid and int(b) in grid:
                    steps = abs(grid.index(int(a)) - grid.index(int(b)))
                    row.update({"grid_steps_apart": steps, "within_one_step": steps <= X13_WITHIN, "differ_2plus_steps": steps >= X13_FAR})
                    both += 1
                    within += steps <= X13_WITHIN
                    far += steps >= X13_FAR
                elif fin(a) != fin(b):
                    row["note"] = "N* exists for one k_train only"
                    one_sided += 1
                else:
                    row["note"] = "no N* for either k_train"
            krows.append(row)
        if both < X13_MIN_NO:
            v = "not testable"
        elif within >= X13_MIN_YES:
            v = "supported"
        elif far >= X13_MIN_NO:
            v = "not supported"
        else:
            v = "mixed"
        per_model[mdl] = {"verdict": v, "datasets_with_N_star_for_both": both, "within_one_step": within,
                          "differ_2plus_steps": far, "N_star_for_one_k_train_only": one_sided, "datasets_in_data": len(dss)}
    write_csv(tabdir / f"{NAME}_x13_ktrain_vs_kprof.csv", krows)
    v13 = combine([p["verdict"] for p in per_model.values()])
    # how N* moves with kprof (kt=all, stratum all): pooled summary for the reading
    kp_med = {}
    for mdl in MODELS_CROSS:
        for kt in ("all", 5):
            for kp in KPROFS:
                vals = [nstar(ds, kt, kp, "all", mdl) for ds in dss if kp in D.kprofs(ds)]
                vals = [v for v in vals if v is not None]
                kp_med[f"{mdl}|k_train={kt}|kprof={kp}"] = f"{sum(fin(v) for v in vals)}/{len(vals)} datasets have an N*"
    nums13 = {"per_model": per_model, "datasets": dss, "datasets_with_N_star_by_kprof": kp_med}
    if v13 == "not testable":
        en = (f"Fewer than {X13_MIN_NO} datasets have a crossover N* for both k_train = 5 and k_train = all at kprof = all "
              f"(knn100: {per_model['knn100']['datasets_with_N_star_for_both']}, ease100: {per_model['ease100']['datasets_with_N_star_for_both']} "
              f"of {len(dss)} datasets in the data), so the rule cannot be judged yet. The N* values are in the table.")
        zh = (f"在 kprof=all 时，同时有 k_train=5 和 k_train=all 交叉点的数据集少于 {X13_MIN_NO} 个"
              f"（knn100：{per_model['knn100']['datasets_with_N_star_for_both']}，ease100：{per_model['ease100']['datasets_with_N_star_for_both']}，"
              f"现有数据共 {len(dss)} 个数据集），所以暂时无法判定。各 N* 见表格。")
    else:
        parts_en = "; ".join(f"{m}: within one step in {p['within_one_step']} of {p['datasets_with_N_star_for_both']}, "
                             f"two or more steps apart in {p['differ_2plus_steps']}" for m, p in per_model.items())
        parts_zh = "；".join(f"{m}：相差不超过一档的有 {p['within_one_step']}/{p['datasets_with_N_star_for_both']}，"
                            f"相差两档及以上的有 {p['differ_2plus_steps']}" for m, p in per_model.items())
        en = (f"At kprof = all, comparing N* for k_train = 5 and k_train = all among datasets that have both: {parts_en}. "
              f"Verdict per model: knn100 {per_model['knn100']['verdict']}, ease100 {per_model['ease100']['verdict']}.")
        zh = (f"在 kprof=all 时，比较同时有 N* 的数据集里 k_train=5 与 k_train=all 的交叉点：{parts_zh}。"
              f"分模型结论：knn100 {per_model['knn100']['verdict']}，ease100 {per_model['ease100']['verdict']}。")
    out13 = entry("X13", *X13_TITLE, v13, nums13, en, zh, X13_RULE)

    # ---- part 2: heavy vs light tercile
    trows, tmodel = [], {}
    for mdl in MODELS_CROSS:
        yes = judged = 0
        for ds in dss:
            hv, lt, md = (nstar(ds, "all", "all", s, mdl) for s in ("2", "0", "1"))
            c_h = D.curve(ds, "all", "all", "2", mdl)
            nmax = float(c_h["N"][-1]) if c_h is not None else NAN
            row = {"dataset": ds, "model": mdl, "N_star_light": lt if lt is not None else NAN, "N_star_middle": md if md is not None else NAN,
                   "N_star_heavy": hv if hv is not None else NAN, "label_light": "no data" if lt is None else nstar_label(lt),
                   "label_middle": "no data" if md is None else nstar_label(md), "label_heavy": "no data" if hv is None else nstar_label(hv),
                   "N_max_tested": nmax}
            if hv is None or lt is None:
                case, far = "no data", None
            elif fin(hv) and fin(lt):
                row["ratio_heavy_over_light"] = hv / lt
                case, far = "both have N*", hv / lt <= 0.5
            elif fin(hv) and not fin(lt):
                case = "light has no N* (ratio is an upper bound)"
                far = True if hv <= 0.5 * nmax else None
                row["ratio_heavy_over_light_upper_bound"] = hv / nmax if fin(nmax) else NAN
            elif not fin(hv) and fin(lt):
                case, far = "heavy has no N* (heavy needs more)", False
            else:
                case, far = "no N* in either tercile", None
            row["case"] = case
            row["heavy_far_smaller"] = "not judged" if far is None else bool(far)
            if far is not None:
                judged += 1
                yes += bool(far)
            trows.append(row)
        tmodel[mdl] = {"verdict": majority(yes, judged), "heavy_far_smaller": yes, "judged": judged}
    write_csv(tabdir / f"{NAME}_x13_tercile_ratio.csv", trows)
    v13b = combine([p["verdict"] for p in tmodel.values()])
    rat = [r["ratio_heavy_over_light"] for r in trows if fin(r.get("ratio_heavy_over_light"))]
    nums13b = {"per_model": tmodel, "median_ratio_where_both_have_N_star": float(np.median(rat)) if rat else None,
               "n_ratios": len(rat)}
    if v13b == "not testable":
        en = ("No dataset can be judged: in none does the heavy tercile have an N* (or a light-tercile N* that would show heavy needs more) "
              "at kprof = all, k_train = all. The N* values by tercile are in the table.")
        zh = "没有任何数据集可以判定：在 kprof=all、k_train=all 时，历史最长的一组没有交叉点，也没有可以显示它需要更大样本的对照。各三分位的 N* 见表格。"
    else:
        parts_en = "; ".join(f"{m}: {p['heavy_far_smaller']} of {p['judged']} judged datasets" for m, p in tmodel.items())
        parts_zh = "；".join(f"{m}：{p['judged']} 个可判定数据集中有 {p['heavy_far_smaller']} 个" for m, p in tmodel.items())
        med = f" Where both terciles have an N*, the median heavy/light ratio is {f3(nums13b['median_ratio_where_both_have_N_star'], 2)} ({len(rat)} ratios)." if rat else ""
        med_zh = f"两组都有 N* 时，重度/轻度的中位数比值为 {f3(nums13b['median_ratio_where_both_have_N_star'], 2)}（共 {len(rat)} 个）。" if rat else ""
        en = f"Heavy-history people reach a crossover at no more than half the sample size of light-history people in: {parts_en}.{med}"
        zh = f"历史长的人所需样本量不超过历史短的人的一半的情况：{parts_zh}。{med_zh}"
    out13b = entry("X13b", *X13B_TITLE, v13b, nums13b, en, zh, X13B_RULE)

    # ---- part 3: tercile policy
    prow, pnum = [], {}
    for kt in ("all", 5):
        for ds in dss:
            cand = []
            for N in sorted(n for (d, k, n) in D.e2p if d == ds and k == kt):
                c = D.e2p[(ds, kt, N)]["cols"]
                cols = [f"kpall.s{s}.{m}" for s in ("0", "1", "2") for m in MODELS]
                if not all(x in c for x in cols):
                    continue
                M = np.vstack([c[x] for x in cols])                # (15, draws)
                ok = np.isfinite(M).all(0)
                if ok.sum() >= MIN_DRAWS:
                    cand.append((N, M[:, ok].reshape(3, len(MODELS), -1)))
            if not cand:
                prow.append({"dataset": ds, "k_train": kt, "status_cell": "no data"})
                continue
            N, S = cand[-1]                                         # S[stratum, model, draw]

            def choose(sub):
                mean = sub.mean(2)                                  # (3, 5)
                per_stratum = mean.argmax(1)
                single = int(mean.mean(0).argmax())
                return per_stratum, single

            def score(sub, per_stratum, single):
                pol = np.mean([sub[s, per_stratum[s]] for s in range(3)], axis=0)
                sg = sub[:, single].mean(0)
                return pol, sg

            ps, sg_m = choose(S)
            pol, sg = score(S, ps, sg_m)
            rng = np.random.default_rng(seed_for("x13c", ds, kt, N))
            gain, lo, hi = ci_of(pol - sg, rng)
            idx = np.arange(S.shape[2])
            halves = []
            for tr, te in ((idx % 2 == 0, idx % 2 == 1), (idx % 2 == 1, idx % 2 == 0)):
                if tr.sum() >= 2 and te.sum() >= 2:
                    p2, s2 = choose(S[:, :, tr])
                    a, b = score(S[:, :, te], p2, s2)
                    halves.append(float((a - b).mean()))
            split = float(np.mean(halves)) if len(halves) == 2 else NAN
            prow.append({"dataset": ds, "k_train": kt, "N": N, "n_draws": S.shape[2],
                         "model_light": MODELS[ps[0]], "model_middle": MODELS[ps[1]], "model_heavy": MODELS[ps[2]],
                         "best_single_model": MODELS[sg_m], "score_tercile_policy": float(pol.mean()), "score_best_single": float(sg.mean()),
                         "gain": gain, "gain_ci_lo": lo, "gain_ci_hi": hi, "gain_split_half": split,
                         "counts": bool(fin(lo) and lo > 0 and fin(split) and split > 0)})
    write_csv(tabdir / f"{NAME}_x13_tercile_policy.csv", prow)
    sel = [r for r in prow if r["k_train"] == "all" and "N" in r]
    yes = sum(r["counts"] for r in sel)
    v13c = majority(yes, len(sel))
    other = [r for r in prow if r["k_train"] == 5 and "N" in r]
    pnum = {"datasets_judged": len(sel), "datasets_counting": yes, "k_train_5_datasets_counting": f"{sum(r['counts'] for r in other)}/{len(other)}",
            "median_gain": float(np.median([r["gain"] for r in sel])) if sel else None,
            "median_gain_split_half": float(np.median([r["gain_split_half"] for r in sel if fin(r["gain_split_half"])])) if any(fin(r["gain_split_half"]) for r in sel) else None,
            "choices": {r["dataset"]: f"light={r['model_light']}, middle={r['model_middle']}, heavy={r['model_heavy']}, single={r['best_single_model']}" for r in sel}}
    if v13c == "not testable":
        en = "No dataset has panel scores for all three terciles and all five models in the e2p stage, so the policy gain cannot be computed yet."
        zh = "e2p 阶段中没有任何数据集同时具备三个三分位和五个模型的面板得分，暂时无法计算策略增益。"
    else:
        en = (f"At the largest N, the tercile policy beats the best single model with a 95% interval above 0 and a positive split-half gain in {yes} of "
              f"{len(sel)} datasets (median in-sample gain {f3(pnum['median_gain'], 4)} NDCG@100, split-half {f3(pnum['median_gain_split_half'], 4)}). "
              f"The choice is made on the same draws it is scored on, so the in-sample gain is optimistic; the split-half column is the fair check.")
        zh = (f"在最大 N 处，按三分位选模型的策略优于最优单一模型（95% 区间在 0 以上且折半检验为正）的数据集有 {yes}/{len(sel)} 个"
              f"（样本内增益中位数 {f3(pnum['median_gain'], 4)} NDCG@100，折半检验 {f3(pnum['median_gain_split_half'], 4)}）。"
              f"选择与打分用的是同一批抽样，样本内增益偏乐观；折半检验才是公平的检查。")
    out13c = entry("X13c", *X13C_TITLE, v13c, pnum, en, zh, X13C_RULE)

    # ---- figure 1: N* vs kprof
    fig, axes = grid_axes(len(dss), 4, 3.0, 2.9)
    xs = {k: i for i, k in enumerate(KPROFS)}
    styles = {("knn100", "all"): ("#1f77b4", "-", "o"), ("knn100", 5): ("#ff7f0e", "-", "s"),
              ("ease100", "all"): ("#1f77b4", "--", "^"), ("ease100", 5): ("#ff7f0e", "--", "v")}
    for ax, ds in zip(axes, dss):
        grid = D.e2p_grid(ds)
        nmax = grid[-1]
        top = nmax * 1.8
        for (mdl, kt), (col, ls, mk) in styles.items():
            if kt not in D.e2p_kts(ds):
                continue
            off = (0.07 if mdl == "knn100" else -0.07) * (1 if kt == "all" else -1)
            px, py, nx, ny = [], [], [], []
            for kp in D.kprofs(ds):
                v = nstar(ds, kt, kp, "all", mdl)
                if v is None:
                    continue
                (px if fin(v) else nx).append(xs[kp] + off)
                (py if fin(v) else ny).append(v if fin(v) else top)
            ax.plot(px, py, color=col, ls=ls, marker=mk, ms=5, lw=1.2)
            ax.plot(nx, ny, color=col, ls="none", marker=mk, ms=5, mfc="white")
        ax.axhline(nmax * 1.3, color="gray", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_yticks(grid + [top])
        ax.set_yticklabels([f"{g:,}" for g in grid] + ["none"], fontsize=8)
        ax.minorticks_off()
        ax.set_ylim(grid[0] * 0.7, top * 1.4)
        ax.set_xticks(range(len(KPROFS)))
        ax.set_xticklabels(KPROFS, fontsize=8)
        ax.set_xlim(-0.4, len(KPROFS) - 0.6)
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("served profile length (kprof)")
        ax.grid(alpha=0.25)
    for ax in axes[::4]:
        ax.set_ylabel("crossover N*")
    handles = [Line2D([], [], color="#1f77b4", marker="o", lw=1.2), Line2D([], [], color="#ff7f0e", marker="s", lw=1.2),
               Line2D([], [], color="k", ls="-"), Line2D([], [], color="k", ls="--"),
               Line2D([], [], color="gray", marker="o", mfc="white", ls="none")]
    labels = ["trained on all choices", "trained on 5 choices each", "knn100", "ease100", "hollow = no crossover up to the largest N"]
    pm = per_model
    finish(fig, (f"Exploratory X13: N* for k_train = 5 stays within one grid step of k_train = all in {pm['knn100']['within_one_step']} of "
                 f"{pm['knn100']['datasets_with_N_star_for_both']} (knn100) and {pm['ease100']['within_one_step']} of "
                 f"{pm['ease100']['datasets_with_N_star_for_both']} (ease100) datasets that have both"), handles, labels)
    savefig(fig, figdir, "x13_nstar_vs_kprof")

    # ---- figure 2: tercile contrast
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), squeeze=False)
    axes = axes[0]
    colst = {"0": "#1f77b4", "1": "#7f7f7f", "2": "#d62728"}
    mk = {"0": "o", "1": "s", "2": "^"}
    topv = max(D.e2p_grid(ds)[-1] for ds in dss) * 1.8
    for ax, mdl in zip(axes[:2], MODELS_CROSS):
        for j, ds in enumerate(dss):
            nmax = D.e2p_grid(ds)[-1]
            ax.plot([j], [nmax], color="lightgray", marker="_", ms=14, ls="none")
            for si, st in enumerate(("0", "1", "2")):
                v = nstar(ds, "all", "all", st, mdl)
                if v is None:
                    continue
                x = j + (si - 1) * 0.2
                y = v if fin(v) else topv
                ax.plot([x], [y], color=colst[st], marker=mk[st], ms=6, ls="none", mfc=colst[st] if fin(v) else "white")
        ax.set_yscale("log")
        allg = sorted({g for ds in dss for g in D.e2p_grid(ds)})
        ax.set_yticks(allg + [topv])
        ax.set_yticklabels([f"{g:,}" for g in allg] + ["none"], fontsize=8)
        ax.minorticks_off()
        ax.set_xticks(range(len(dss)))
        ax.set_xticklabels(dss, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{mdl}: crossover N* (k_train = all, kprof = all)", fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    ax = axes[2]
    pk = [r for r in prow if r["k_train"] == "all" and "N" in r]
    xx = np.arange(len(pk))
    g = np.array([r["gain"] for r in pk], dtype=float)
    lo = np.array([r["gain_ci_lo"] for r in pk], dtype=float)
    hi = np.array([r["gain_ci_hi"] for r in pk], dtype=float)
    if len(pk):
        ax.bar(xx - 0.2, np.nan_to_num(g), 0.4, color="#2ca02c", label="in-sample gain (95% interval)",
               yerr=np.vstack([np.where(np.isfinite(lo), g - lo, 0), np.where(np.isfinite(hi), hi - g, 0)]), capsize=2)
        ax.bar(xx + 0.2, np.nan_to_num([r["gain_split_half"] for r in pk]), 0.4, color="#98df8a", label="split-half gain")
        ax.set_xticks(xx)
        ax.set_xticklabels([r["dataset"] for r in pk], rotation=60, ha="right", fontsize=7)
        ax.legend(fontsize=8, frameon=False)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("panel NDCG@100 gain")
    ax.set_title("tercile policy minus best single model", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    handles = [Line2D([], [], color=colst[s], marker=mk[s], ls="none") for s in ("0", "1", "2")] + \
              [Line2D([], [], color="gray", marker="o", mfc="white", ls="none"), Line2D([], [], color="lightgray", marker="_", ms=10, ls="none")]
    labels = ["light history", "middle history", "heavy history", "hollow = no crossover", "largest N tested"]
    s_en = f"{tmodel['knn100']['heavy_far_smaller']} of {tmodel['knn100']['judged']} (knn100) and {tmodel['ease100']['heavy_far_smaller']} of {tmodel['ease100']['judged']} (ease100)"
    finish(fig, f"Exploratory X13b/c: heavy-history people need at most half the sample of light-history people in {s_en} judged datasets; "
                f"the tercile policy gains in {yes} of {len(sel)}", handles, labels)
    savefig(fig, figdir, "x13_tercile")
    return [out13, out13b, out13c]


# ----------------------------------------------------------------------------- X14
X14_TITLE = ("Exploratory X14: one sample's own bootstrap predicts its list quality at the right scale, and the row-level bootstrap does not",
             "探索性 X14：单个样本自己的自助法能按正确尺度预测榜单质量，而按行的自助法不能")
X14_RULE = ("Against Truth A (real_A), cells with N >= 200. (1) user-level bootstrap 'calibrated' in a dataset if the median pred/real ratio over draws lies "
            "in [0.67, 1.5] in every (N >= 200, k) cell; supported if so in strictly more than half of the datasets. (2) row-level bootstrap 'mis-scaled "
            "relative to the user-level one' in a dataset if the paired-by-draw 95% interval of (pred_row - pred_user) excludes 0 in more than half of its "
            "N >= 200 cells; supported if so in strictly more than half of the datasets. (3) Spearman correlation across draws of pred_user vs real_A: a dataset "
            "passes if the median over its N >= 200 cells is >= 0.5 (cells with >= 5 draws and non-constant values); supported if so in strictly more than half "
            "of the datasets. The three parts are combined (supported only if all are).")


def run_x14(D: Data, tabdir: Path, figdir: Path) -> dict:
    if not D.e8:
        return not_testable("X14", *X14_TITLE, X14_RULE, "Needs the e8 stage (self-assessment rows), which is missing or empty in the data available.",
                            "需要 e8 阶段（自我评估数据），现有数据中缺失或为空。")
    dss = order_ds({d for (d, _, _) in D.e8}, D.ds_order)
    crows, cellmap = [], {}
    for ds in dss:
        for k in sorted({k for (d, k, _) in D.e8 if d == ds}, key=kord):
            for N in sorted(n for (d, kk, n) in D.e8 if d == ds and kk == k):
                c = D.e8[(ds, k, N)]
                real = c["real_A"]
                ok = np.isfinite(real) & (real > 0)
                row = {"dataset": ds, "N": N, "k": k, "n_draws": int(np.isfinite(real).sum())}
                if not ok.any():
                    crows.append(row)
                    continue
                rng = np.random.default_rng(seed_for("x14", ds, k, N))
                for tag in ("user", "row"):
                    pred = c[f"pred_{tag}"]
                    good = ok & np.isfinite(pred)
                    row[f"median_ratio_{tag}"] = float(np.median(pred[good] / real[good])) if good.any() else NAN
                    m, lo, hi = ci_of((pred - real)[good], rng) if good.sum() >= 2 else (NAN, NAN, NAN)
                    row.update({f"mean_signed_error_{tag}": m, f"signed_error_{tag}_ci_lo": lo, f"signed_error_{tag}_ci_hi": hi,
                                f"median_signed_error_{tag}": float(np.median((pred - real)[good])) if good.any() else NAN,
                                f"spearman_{tag}": spearman(pred[good], real[good]) if good.sum() >= 5 else NAN})
                both = ok & np.isfinite(c["pred_user"]) & np.isfinite(c["pred_row"])
                m, lo, hi = ci_of((c["pred_row"] - c["pred_user"])[both], rng) if both.sum() >= 2 else (NAN, NAN, NAN)
                row.update({"mean_row_minus_user": m, "row_minus_user_ci_lo": lo, "row_minus_user_ci_hi": hi,
                            "row_differs_from_user": bool(lo > 0 or hi < 0) if fin(lo) and fin(hi) else "not judged",
                            "mean_real_A": float(real[ok].mean())})
                mu = row.get("median_ratio_user", NAN)
                row["user_calibrated"] = bool(RATIO_LO <= mu <= RATIO_HI) if fin(mu) else "not judged"
                mr = row.get("median_ratio_row", NAN)
                row["row_calibrated"] = bool(RATIO_LO <= mr <= RATIO_HI) if fin(mr) else "not judged"
                crows.append(row)
                cellmap[(ds, k, N)] = row
    write_csv(tabdir / f"{NAME}_x14_cells.csv", crows)

    # per-dataset rule parts (N >= 200)
    drows = []
    for ds in dss:
        cells = [r for (d, k, N), r in cellmap.items() if d == ds and N >= N_CAL]
        row = {"dataset": ds, "n_cells_N_ge_200": len(cells)}
        cal = [r["user_calibrated"] for r in cells if r["user_calibrated"] != "not judged"]
        row["user_calibrated"] = ("not judged" if not cal else bool(all(cal)))
        row["user_cells_in_band"] = f"{sum(cal)}/{len(cal)}"
        rcal = [r["row_calibrated"] for r in cells if r["row_calibrated"] != "not judged"]
        row["row_cells_in_band"] = f"{sum(rcal)}/{len(rcal)}"
        row["row_calibrated"] = "not judged" if not rcal else bool(all(rcal))
        dif = [r["row_differs_from_user"] for r in cells if r["row_differs_from_user"] != "not judged"]
        row["row_misscaled_vs_user"] = ("not judged" if not dif else bool(sum(dif) / len(dif) > 0.5))
        row["row_differs_cells"] = f"{sum(dif)}/{len(dif)}"
        sp = [r["spearman_user"] for r in cells if fin(r.get("spearman_user"))]
        row["median_spearman_user"] = float(np.median(sp)) if sp else NAN
        row["median_spearman_row"] = float(np.median([r["spearman_row"] for r in cells if fin(r.get("spearman_row"))])) if any(fin(r.get("spearman_row")) for r in cells) else NAN
        row["spearman_ok"] = bool(row["median_spearman_user"] >= SPEARMAN_MIN) if sp else "not judged"
        row["median_ratio_user_over_cells"] = float(np.median([r["median_ratio_user"] for r in cells if fin(r.get("median_ratio_user"))])) if cells else NAN
        row["median_ratio_row_over_cells"] = float(np.median([r["median_ratio_row"] for r in cells if fin(r.get("median_ratio_row"))])) if cells else NAN
        drows.append(row)
    write_csv(tabdir / f"{NAME}_x14_datasets.csv", drows)

    def part(col):
        det = [r for r in drows if r[col] != "not judged"]
        yes = sum(r[col] is True for r in det)
        return majority(yes, len(det)), yes, len(det)

    pu, yu, du = part("user_calibrated")
    pr, yr, dr = part("row_misscaled_vs_user")
    ps, ys, ds_ = part("spearman_ok")
    verdict = combine([pu, pr, ps])

    # pooled by N (and k): cluster bootstrap over draws, equal weight per dataset
    brows = []
    for k in sorted({k for (_, k, _) in D.e8}, key=kord):
        for N in sorted({n for (_, kk, n) in D.e8 if kk == k}):
            sub = [(ds, D.e8[(ds, k, N)]) for ds in dss if (ds, k, N) in D.e8]
            row = {"k": k, "N": N, "n_datasets": len(sub)}
            for tag in ("user", "row"):
                vals, grp = [], []
                for ds, c in sub:
                    err = c[f"pred_{tag}"] - c["real_A"]
                    vals.append(err)
                    grp.append(c["draws"])
                v, g = np.concatenate(vals), np.concatenate(grp)
                rng = np.random.default_rng(seed_for("x14pool", tag, k, N))
                m, lo, hi = metrics.cluster_bootstrap_ci(v, g, n_boot=N_BOOT, rng=rng) if np.isfinite(v).sum() >= 2 else (NAN, NAN, NAN)
                row.update({f"signed_error_{tag}": m, f"signed_error_{tag}_ci_lo": lo, f"signed_error_{tag}_ci_hi": hi})
                rs = [r[f"median_ratio_{tag}"] for (d, kk, nn), r in cellmap.items() if kk == k and nn == N and fin(r.get(f"median_ratio_{tag}"))]
                row[f"median_of_cell_median_ratios_{tag}"] = float(np.median(rs)) if rs else NAN
            brows.append(row)
    write_csv(tabdir / f"{NAME}_x14_by_n.csv", brows)

    err_by_n = {}
    for r in brows:
        err_by_n[f"k={r['k']}|N={r['N']}"] = {"user": r["signed_error_user"], "row": r["signed_error_row"]}
    sp_all = [r["median_spearman_user"] for r in drows if fin(r["median_spearman_user"])]
    nums = {"user_calibrated_datasets": f"{yu}/{du}", "row_misscaled_datasets": f"{yr}/{dr}", "spearman_ok_datasets": f"{ys}/{ds_}",
            "verdict_per_part": {"user_calibrated": pu, "row_misscaled": pr, "spearman": ps},
            "median_ratio_user_N_ge_200": float(np.median([r["median_ratio_user_over_cells"] for r in drows if fin(r["median_ratio_user_over_cells"])])) if drows else None,
            "median_ratio_row_N_ge_200": float(np.median([r["median_ratio_row_over_cells"] for r in drows if fin(r["median_ratio_row_over_cells"])])) if drows else None,
            "median_spearman_user": float(np.median(sp_all)) if sp_all else None,
            "mean_signed_error_pred_minus_real_by_k_and_N": err_by_n}
    if verdict == "not testable":
        en = "No dataset has an e8 cell with N >= 200 and a usable real_A, so the calibration rule cannot be judged yet."
        zh = "没有任何数据集在 N≥200 时具备可用的 real_A，所以暂时无法判定校准规则。"
    else:
        en = (f"The user-level bootstrap has a median pred/real ratio inside [0.67, 1.5] in every N >= 200 cell in {nums['user_calibrated_datasets']} datasets "
              f"(median ratio {f3(nums['median_ratio_user_N_ge_200'], 2)}); the row-level one differs from it by a paired interval that excludes zero in most cells in "
              f"{nums['row_misscaled_datasets']} datasets (median ratio {f3(nums['median_ratio_row_N_ge_200'], 2)}). Across draws, predicted and real quality are "
              f"rank-correlated at 0.5 or more in {nums['spearman_ok_datasets']} datasets (median Spearman {f3(nums['median_spearman_user'], 2)}). "
              f"Signed error (pred - real) by N is in the table.")
        zh = (f"按用户的自助法在 N≥200 的每个格子里，预测/真实比值的中位数都落在 [0.67, 1.5] 内的数据集有 {nums['user_calibrated_datasets']} 个"
              f"（比值中位数 {f3(nums['median_ratio_user_N_ge_200'], 2)}）；按行的自助法与之相比、配对区间不含 0 的格子占多数的数据集有 {nums['row_misscaled_datasets']} 个"
              f"（比值中位数 {f3(nums['median_ratio_row_N_ge_200'], 2)}）。跨抽样来看，预测与真实质量的秩相关达到 0.5 以上的数据集有 {nums['spearman_ok_datasets']} 个"
              f"（Spearman 中位数 {f3(nums['median_spearman_user'], 2)}）。按 N 的有符号误差（预测减真实）见表格。")
    out = entry("X14", *X14_TITLE, verdict, nums, en, zh, X14_RULE)

    # ---- figure: 4 scatters + ratio vs N
    ks = sorted({k for (_, k, _) in D.e8}, key=kord)
    kpick = 5 if 5 in ks else ks[0]
    picks = []
    for ds in dss[:2]:
        Ns = sorted(n for (d, k, n) in D.e8 if d == ds and k == kpick)
        small = [n for n in Ns if n >= N_CAL]
        if small:
            picks.append((ds, kpick, small[0]))
        if Ns and (not small or Ns[-1] != small[0]):
            picks.append((ds, kpick, Ns[-1]))
    picks = picks[:4]
    fig = plt.figure(figsize=(11.5, 7.2))
    gs = fig.add_gridspec(2, 4)
    for i, (ds, k, N) in enumerate(picks):
        ax = fig.add_subplot(gs[0, i])
        c = D.e8[(ds, k, N)]
        ax.scatter(c["real_A"], c["pred_user"], s=22, color="#1f77b4", label="user-level bootstrap")
        ax.scatter(c["real_A"], c["pred_row"], s=22, color="#ff7f0e", marker="s", label="row-level bootstrap")
        allv = np.concatenate([c["real_A"], c["pred_user"], c["pred_row"]])
        allv = allv[np.isfinite(allv)]
        lo_, hi_ = (allv.min(), allv.max()) if allv.size else (0, 1)
        pad = 0.05 * (hi_ - lo_ + 1e-9)
        ax.plot([lo_ - pad, hi_ + pad], [lo_ - pad, hi_ + pad], color="k", lw=0.9, ls=":", label="identity")
        r = cellmap.get((ds, k, N), {})
        ax.set_title(f"{ds}, N={N:,}, k={k}\nmedian ratio user {f3(r.get('median_ratio_user'), 2)}, row {f3(r.get('median_ratio_row'), 2)}", fontsize=9)
        ax.set_xlabel("real NDCG@100 vs Truth A")
        if i == 0:
            ax.set_ylabel("predicted by bootstrap")
        ax.grid(alpha=0.25)
    cmap = plt.get_cmap("tab10")
    for j, k in enumerate(ks[:2]):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        ax.axhspan(RATIO_LO, RATIO_HI, color="#2ca02c", alpha=0.10)
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        for di, ds in enumerate(dss):
            Ns = sorted(n for (d, kk, n) in D.e8 if d == ds and kk == k and (ds, kk, n) in cellmap)
            for tag, ls, mk_ in (("user", "-", "o"), ("row", "--", "s")):
                y = [cellmap[(ds, k, n)].get(f"median_ratio_{tag}", NAN) for n in Ns]
                ax.plot(Ns, y, color=cmap(di % 10), ls=ls, marker=mk_, ms=4, lw=1.2)
        log_n_axis(ax)
        ax.set_xlabel("sample size N")
        ax.set_ylabel("median predicted / real")
        ax.set_title(f"k = {k}: green band = ratio 0.67 to 1.5", fontsize=9)
    handles = [Line2D([], [], color=cmap(i % 10), lw=1.5) for i in range(len(dss))] + \
              [Line2D([], [], color="k", ls="-", marker="o", ms=4), Line2D([], [], color="k", ls="--", marker="s", ms=4)]
    labels = list(dss) + ["user-level bootstrap (solid)", "row-level bootstrap (dashed)"]
    pr_cal = [r["row_calibrated"] for r in drows if r["row_calibrated"] != "not judged"]
    n_row_cal = f"{sum(x is True for x in pr_cal)}/{len(pr_cal)}"
    finish(fig, f"Exploratory X14: the user-level bootstrap's median predicted/real ratio is inside 0.67-1.5 for N >= 200 in {nums['user_calibrated_datasets']} datasets, "
                f"the row-level one in {n_row_cal}",
           handles, labels, ncol=4)
    savefig(fig, figdir, "x14_self_assessment")
    return out


# ----------------------------------------------------------------------------- X15
X15_TITLE = ("Exploratory X15: the unbiased weighted count only beats the plain count at very large samples, if ever",
             "探索性 X15：无偏的加权计数即便能胜过普通计数，也只发生在样本很大时")
X15_RULE = ("For each dataset and numeric k: N_win is the smallest N where the independent-draw bootstrap 95% interval of (ht - count) for A_ndcg100 is "
            "above 0 (2000 resamples); 'never' if no N qualifies. Supported if, in strictly more than half of the dataset-k series, ht never wins or wins "
            "only at N >= 10% of the group. A_u results are shown beside it and do not enter the verdict.")


def indep_diff_ci(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float, float]:
    """Mean(b) - mean(a) with a percentile interval from resampling each group independently."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return (NAN, NAN, NAN)
    ma = a[rng.integers(0, a.size, size=(N_BOOT, a.size))].mean(1)
    mb = b[rng.integers(0, b.size, size=(N_BOOT, b.size))].mean(1)
    lo, hi = np.percentile(mb - ma, [2.5, 97.5])
    return float(b.mean() - a.mean()), float(lo), float(hi)


def run_x15(D: Data, tabdir: Path, figdir: Path) -> dict:
    if not D.e1:
        return not_testable("X15", *X15_TITLE, X15_RULE, "Needs the e1 stage (main grid), which is missing or empty in the data available.",
                            "需要 e1 阶段（主网格），现有数据中缺失或为空。")
    if "ht.A_ndcg100" not in D.e1_fields_seen:
        return not_testable("X15", *X15_TITLE, X15_RULE,
                            "The e1 rows in the data available carry no `ht.A_ndcg100` column (the weighted count was not recorded), so ht cannot be compared with count.",
                            "现有 e1 数据中没有 `ht.A_ndcg100` 列（没有记录加权计数），所以无法比较 ht 与 count。")
    dss = order_ds({d for (d, _, _) in D.e1}, D.ds_order)
    srows, curves = [], {}
    targets = [("A", "A_ndcg100")]
    if "ht.Au_ndcg100" in D.e1_fields_seen and "count.Au_ndcg100" in D.e1_fields_seen:
        targets.append(("A_u", "Au_ndcg100"))
    for tname, suf in targets:
        for ds in dss:
            npop = D.npop(ds)
            for k in sorted({k for (d, k, _) in D.e1 if d == ds and k != "all"}, key=kord):
                Ns, ms, los, his, mc, mh, cl, ch, hl, hh = ([] for _ in range(10))
                for N in sorted(n for (d, kk, n) in D.e1 if d == ds and kk == k):
                    c = D.e1[(ds, k, N)]
                    a, b = c[f"count.{suf}"], c[f"ht.{suf}"]
                    if np.isfinite(a).sum() < MIN_DRAWS or np.isfinite(b).sum() < MIN_DRAWS:
                        continue
                    rng = np.random.default_rng(seed_for("x15", tname, ds, k, N))
                    m, lo, hi = indep_diff_ci(a, b, rng)
                    Ns.append(N)
                    ms.append(m)
                    los.append(lo)
                    his.append(hi)
                    mc.append(float(np.nanmean(a)))
                    mh.append(float(np.nanmean(b)))
                    cl.append(ci_of(a, np.random.default_rng(seed_for("x15c", tname, ds, k, N, "count")))[1:])
                    hl.append(ci_of(b, np.random.default_rng(seed_for("x15c", tname, ds, k, N, "ht")))[1:])
                if not Ns:
                    continue
                Ns_a, ms_a, los_a = np.array(Ns, float), np.array(ms), np.array(los)
                wins = np.flatnonzero(los_a > 0)
                nw = float(Ns_a[wins[0]]) if wins.size else NAN
                share = nw / npop if fin(nw) and fin(npop) else NAN
                late = None
                if not fin(nw):
                    late = True
                elif fin(share):
                    late = bool(share >= SHARE_HT)
                srows.append({"target": tname, "dataset": ds, "k": k, "n_cells": len(Ns), "N_max_tested": Ns[-1],
                              "N_max_share_of_group": Ns[-1] / npop if fin(npop) else NAN, "group_size": npop,
                              "N_ht_wins": nw, "N_ht_wins_label": "never" if not fin(nw) else fN(nw), "N_ht_wins_share_of_group": share,
                              "mean_diff_ht_minus_count_at_Nmax": ms[-1], "ci_lo_at_Nmax": los[-1], "ci_hi_at_Nmax": his[-1],
                              "mean_count_at_Nmax": mc[-1], "mean_ht_at_Nmax": mh[-1],
                              "prediction_holds": "not judged" if late is None else late})
                curves[(tname, ds, k)] = {"N": Ns_a, "count": np.array(mc), "ht": np.array(mh), "count_ci": np.array(cl), "ht_ci": np.array(hl),
                                         "nwin": nw}
    write_csv(tabdir / f"{NAME}_x15_ht_vs_count.csv", srows)
    crow = [{"target": t, "dataset": ds, "k": k, "N": float(c["N"][i]), "mean_count": c["count"][i], "mean_ht": c["ht"][i],
             "count_ci_lo": c["count_ci"][i][0], "count_ci_hi": c["count_ci"][i][1], "ht_ci_lo": c["ht_ci"][i][0], "ht_ci_hi": c["ht_ci"][i][1]}
            for (t, ds, k), c in curves.items() for i in range(len(c["N"]))]
    write_csv(tabdir / f"{NAME}_x15_curves.csv", crow)

    sa = [r for r in srows if r["target"] == "A"]
    det = [r for r in sa if r["prediction_holds"] != "not judged"]
    yes = sum(r["prediction_holds"] is True for r in det)
    verdict = majority(yes, len(det))
    never = sum(not fin(r["N_ht_wins"]) for r in sa)
    early = sum(fin(r["N_ht_wins"]) and fin(r["N_ht_wins_share_of_group"]) and r["N_ht_wins_share_of_group"] < SHARE_HT for r in sa)
    cens = sum(1 for r in sa if not fin(r["N_ht_wins"]) and fin(r["N_max_share_of_group"]) and r["N_max_share_of_group"] < SHARE_HT)
    diffs = [r["mean_diff_ht_minus_count_at_Nmax"] for r in sa if fin(r["mean_diff_ht_minus_count_at_Nmax"])]
    nums = {"series": len(sa), "series_judged": len(det), "series_prediction_holds": yes, "series_ht_never_wins": never,
            "series_ht_wins_before_10pct_of_group": early, "never_wins_but_largest_N_below_10pct_of_group": cens,
            "median_diff_ht_minus_count_at_Nmax": float(np.median(diffs)) if diffs else None,
            "series_ht_better_at_Nmax_mean": sum(d > 0 for d in diffs)}
    au = [r for r in srows if r["target"] == "A_u"]
    if au:
        nums["A_u_series_ht_never_wins"] = f"{sum(not fin(r['N_ht_wins']) for r in au)}/{len(au)}"
        d_au = [r["mean_diff_ht_minus_count_at_Nmax"] for r in au if fin(r["mean_diff_ht_minus_count_at_Nmax"])]
        nums["A_u_median_diff_ht_minus_count_at_Nmax"] = float(np.median(d_au)) if d_au else None
    if verdict == "not testable":
        en = "No dataset-k series can be judged (no numeric-k cell holds both count and ht scores with enough draws)."
        zh = "没有任何数据集与 k 的组合可以判定（没有数值 k 的格子同时具备足够抽样的 count 与 ht 得分）。"
    else:
        en = (f"In {yes} of {len(det)} dataset-k series the weighted `ht` count never beats the plain `count` on Truth A, or only does so once N is at least 10% of the group "
              f"(never: {never}; wins earlier than 10% of the group: {early}; of the never cases, {cens} stop below 10% of the group, so they are lower bounds). "
              f"At the largest N the median difference (ht - count) is {f3(nums['median_diff_ht_minus_count_at_Nmax'], 4)} NDCG@100, and ht is ahead in "
              f"{nums['series_ht_better_at_Nmax_mean']} of {len(diffs)} series.")
        zh = (f"在 {len(det)} 个数据集与 k 的组合中，有 {yes} 个里加权的 `ht` 在对 Truth A 的得分上从不胜过普通 `count`，或者只有在 N 达到群体 10% 以上时才胜过"
              f"（从不胜过：{never}；在 10% 以前就胜过：{early}；从不胜过的组合中有 {cens} 个测试的最大 N 不到群体的 10%，只能算下界）。"
              f"在最大 N 处 ht 减 count 的中位数为 {f3(nums['median_diff_ht_minus_count_at_Nmax'], 4)} NDCG@100，ht 领先的组合为 "
              f"{nums['series_ht_better_at_Nmax_mean']}/{len(diffs)}。")
    out = entry("X15", *X15_TITLE, verdict, nums, en, zh, X15_RULE)

    # ---- figure: count vs ht at k = 5, 4 datasets
    avail = [ds for ds in dss if ("A", ds, 5) in curves]
    pri = [d for d in PRIORITY if d in avail]
    pick = (pri + [d for d in avail if d not in pri])[:4]
    if pick:
        fig, axes = grid_axes(len(pick), 4, 3.0, 3.0)
        for ax, ds in zip(axes, pick):
            c = curves[("A", ds, 5)]
            for key, col, lab in (("count", "#1f77b4", "plain count"), ("ht", "#ff7f0e", "ht (unbiased)")):
                ci = c[f"{key}_ci"]
                ax.plot(c["N"], c[key], color=col, marker="o", ms=4, lw=1.4, label=lab)
                ax.fill_between(c["N"], ci[:, 0], ci[:, 1], color=col, alpha=0.18, lw=0)
            npop = D.npop(ds)
            if fin(npop) and c["N"][0] <= 0.1 * npop <= c["N"][-1] * 3:
                ax.axvline(0.1 * npop, color="gray", ls=":", lw=1.0)
                ax.text(0.1 * npop, ax.get_ylim()[0], " 10% of group", rotation=90, va="bottom", ha="right", fontsize=7, color="gray")
            if fin(c["nwin"]):
                ax.plot([c["nwin"]], [c["ht"][list(c["N"]).index(c["nwin"])]], marker="*", ms=12, color="#d62728", ls="none")
            log_n_axis(ax)
            ax.set_title(ds, fontsize=10)
            ax.set_xlabel("sample size N")
        axes[0].set_ylabel("NDCG@100 vs Truth A (k = 5)")
        h, l = axes[0].get_legend_handles_labels()
        h.append(Line2D([], [], color="#d62728", marker="*", ms=10, ls="none"))
        l.append("first N with significant ht gain")
        finish(fig, f"Exploratory X15: at k = 5 the unbiased ht count beats the plain count in {sum(fin(curves[('A', d, 5)]['nwin']) for d in pick)} of the {len(pick)} datasets shown; "
                    f"in {yes} of {len(det)} dataset-k series it never wins or wins only at N of 10% of the group or more", h, l)
        savefig(fig, figdir, "x15_ht_vs_count")
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
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    D = Data(runs, dpath)
    for m in D.messages:
        print("NOTE:", m)
    print(f"loaded {len(D.e2p)} e2p cells, {len(D.e8)} e8 cells, {len(D.e1)} e1 cells  ({time.time() - t0:.1f}s)")

    verdicts: list[dict] = []

    def guarded(ids, title, rule, fn):
        t = time.time()
        try:
            r = fn()
            r = r if isinstance(r, list) else [r]
        except Exception as e:  # keep the other families alive
            traceback.print_exc()
            r = [not_testable(i, *title, rule, f"Internal error while computing this family: {type(e).__name__}: {e}",
                              f"计算该族时出现内部错误：{type(e).__name__}: {e}") for i in ids]
        print(f"{'/'.join(ids)}: {', '.join(x['verdict'] for x in r)}  ({time.time() - t:.1f}s)")
        verdicts.extend(r)

    guarded(["X13"], X13_TITLE, X13_RULE, lambda: run_x13(D, tabdir, figdir))
    guarded(["X14"], X14_TITLE, X14_RULE, lambda: run_x14(D, tabdir, figdir))
    guarded(["X15"], X15_TITLE, X15_RULE, lambda: run_x15(D, tabdir, figdir))
    if D.messages:  # surface missing stages inside the file itself
        for v in verdicts:
            if v["verdict"] == "not testable":
                v["numbers"] = {**(v.get("numbers") or {}), "missing_inputs": D.messages}
    (tabdir / f"{NAME}_verdicts.json").write_text(json.dumps(clean(verdicts), indent=1, ensure_ascii=False, allow_nan=False))
    print(f"wrote {tabdir / (NAME + '_verdicts.json')}  total {time.time() - t0:.1f}s")
    for v in verdicts:
        print(f"  {v['id']:5s} {v['verdict']:14s} {v['reading_en'][:150]}")


if __name__ == "__main__":
    main()
