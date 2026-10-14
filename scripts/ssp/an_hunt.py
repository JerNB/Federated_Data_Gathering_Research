#!/usr/bin/env python3
"""X11 surprise hunt plus replication handshake (small-sample popularity study).

    python3 scripts/ssp/an_hunt.py --stage scan  --runs <02_runs> --out <results dir>
    python3 scripts/ssp/an_hunt.py --stage check --runs <02_runs> --out <results dir> --fresh <dir>

Stage scan  : scans the raw e1 / e2 (and e5, e6, e7 when present) rows for patterns a researcher would call
              surprising.  Every pattern is an explicit statistic over sampling DRAWS (never over people):
              Welch / paired t on per-draw values (z = effect / standard error; the p-value uses the t law
              with Satterthwaite df, which equals the normal law at 60 draws).  Benjamini-Hochberg at q = 0.10
              is applied across ALL hypotheses of the same pattern class (hits with p > q are never filtered
              before BH).  Writes 03_tables/hunt_candidates.csv, hunt_candidates_zh.csv,
              hunt_replication_tasks.json, hunt_scan_counts.csv, an_hunt_verdicts.json (replication pending)
              and 04_figures/hunt_counts.png, hunt_best.png.
Stage check : re-estimates every FDR hit ONLY on fresh draws (draw ids >= 1000, new seeds) and calls it
              replicated when the effect has the same sign and the one-sided p < 0.05 (for multi-part patterns:
              every part, i.e. the maximum of the part p-values).  Writes hunt_replication.csv and the final
              an_hunt_verdicts.json.  An unreplicated hit is never reported as a finding.

Pattern classes (all exploratory; the pre-registered X11):
  1 non-monotone learning curve        mean of M falls by > 2 SE from one N to the next larger N
  2 sign flip between targets          estimator difference significant with opposite signs on two targets
                                       (A_ndcg100 vs U_ndcg as pre-registered; A vs B added as an extension)
  3 estimator order reversal           two estimators cross as N grows (best ordered pair of N, Bonferroni)
  4 interaction                        (a) N-effect differs by k (difference of differences);
                                       (b) breadth/depth at equal budget N*k reverses between A_ndcg100 and
                                           another metric
  5 cross-dataset heterogeneity        effect significantly positive in some datasets, negative in others
  6 mixing valley            (added)   a mixture is significantly WORSE than both of its pure ends (a hybrid
                                       below popularity and below the pure personalised list)
  7 diverging targets        (added)   one more N helps on one target and hurts on another
  8 noise helps              (added)   a privacy-noised list beats the non-private list (should be impossible
                                       by design, so a hit points at a metric or small-sample artefact)

Why the three added classes: 6 is the mirror image of confirmatory claim C2 (hybrids "never worse"); 7 is the
N-axis version of the A-versus-B drift question at the heart of the project; 8 is a sanity check that a
privacy mechanism cannot win by accident of the metric.

Weakest-part convention: `effect` and `z` of a candidate are those of its weakest part, oriented so that
positive means "the surprising pattern holds" (class 1: the drop in the metric; class 2/7: the smaller of the
two opposite changes; class 3/5: the smaller of the two opposite differences; class 4a: |difference of
differences|; class 6: the smaller shortfall; class 8: the gain over non-private).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import textwrap
import time
from collections import defaultdict

import numpy as np
from scipy.special import stdtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import metrics as MT  # noqa: E402
except ImportError:  # worktree without the committed helper: use the shared checkout
    sys.path.insert(0, "/Users/LeoQin/Documents/GitHub/Federated_Data_Gathering_Research/scripts/ssp")
    import metrics as MT  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 20261014
NBOOT = 2000
Q_FDR = 0.10
P_REP = 0.05
MIN_N = 3                 # draws needed per group
Z_NONMONO = 2.0           # class 1: "more than 2 standard errors"
P_STORE = 0.10            # a hypothesis with q <= 0.10 necessarily has p <= 0.10
FRESH_DRAW0, N_FRESH, TOP_TASKS = 1000, 20, 40

E1_EST = ["count", "shrunk", "lcb"]
E1_METRICS = ["A_ov10", "A_ov100", "A_rbo90", "A_rbo98", "A_tau100", "A_ndcg100", "B_ndcg100", "B_mass10",
              "B_ov100", "U_ndcg", "U_recall", "U_hit"]
CURVE_METRICS = ["A_ndcg100", "B_ndcg100", "U_ndcg"]
E2_DUP = {"knn.w0", "ease.w0"}                       # identical to pop by construction
E2_PAIR_SET = ["pop", "ease.w50", "ease.w100", "knn.w50", "knn.w100", "ease_r25.w100", "hyb_time",
               "hyb_random", "pers_time", "pers_random"]
E2_VS_POP = ["ease.w50", "ease.w100", "knn.w50", "knn.w100", "hyb_time", "hyb_random", "pers_time",
             "pers_random"]
E5_EST = ["sample", "global", "shrunk", "blend25", "blend50", "blend75"]
E6_EST = ["nonprivate", "laplace", "gaussian", "ldp_oue"]
ARM_KW = {"activity": dict(users="activity"), "inv_activity": dict(users="inv_activity"),
          "recent_rows": dict(rows="recent"), "recent_window": dict(window=True)}

CLASSES = {
    1: ("x11_c1_nonmonotone", "1 non-monotone learning curve", "1 学习曲线随样本变大反而变差"),
    2: ("x11_c2_signflip", "2 sign flip between targets", "2 不同评价目标上结论相反"),
    3: ("x11_c3_reversal", "3 estimator order reversal", "3 两种估计方法随样本变大互换名次"),
    4: ("x11_c4_interaction", "4 interaction (N x k, breadth vs depth)", "4 交互作用（人数×每人条数；广度与深度）"),
    5: ("x11_c5_heterogeneity", "5 cross-dataset heterogeneity", "5 不同数据集上效果方向相反"),
    6: ("x11_c6_valley", "6 mixing valley", "6 混合比两端都差"),
    7: ("x11_c7_diverging", "7 diverging targets", "7 样本变大对两个目标作用相反"),
    8: ("x11_c8_noisehelps", "8 noise helps", "8 加噪声反而更好"),
}
RULE_COMMON = ("X11 (exploratory, pre-registered): scan every curve for non-monotone shapes and sign reversals; "
               "Benjamini-Hochberg q=0.10 within the class; each hit re-run on fresh seeds, reported as "
               "found / replicated / not replicated.")

# --------------------------------------------------------------------------------------------- naming
EST_EN = {"count": "the plain count list", "shrunk": "the stale-shrunk list", "lcb": "the lower-bound list",
          "ht": "the Horvitz-Thompson count list", "pop": "the popularity list",
          "oracle_best": "the panel-tuned upper bound", "sample": "the subgroup-sample list",
          "global": "the global list", "nonprivate": "the non-private list",
          "laplace": "the Laplace-noised list", "gaussian": "the Gaussian-noised list",
          "ldp_oue": "the local-DP (OUE) list", "hyb_time": "the time-hidden hybrid",
          "hyb_random": "the random-hidden hybrid", "pers_time": "the best pure personalised list (time-hidden)",
          "pers_random": "the best pure personalised list (random-hidden)"}
EST_ZH = {"count": "朴素计数榜", "shrunk": "向旧榜收缩的榜", "lcb": "下界榜", "ht": "加权(HT)计数榜",
          "pop": "热门榜", "oracle_best": "用留出人群调参的上限", "sample": "仅用子群样本的榜",
          "global": "全体榜", "nonprivate": "无隐私保护的榜", "laplace": "拉普拉斯加噪的榜",
          "gaussian": "高斯加噪的榜", "ldp_oue": "本地隐私(OUE)榜", "hyb_time": "按时间隐藏选权重的混合榜",
          "hyb_random": "按随机隐藏选权重的混合榜", "pers_time": "同法（时间隐藏）选出的纯个性化榜",
          "pers_random": "同法（随机隐藏）选出的纯个性化榜"}
MET_EN = {"A_ndcg100": "list quality against Truth A (NDCG@100)", "B_ndcg100": "list quality against Truth B (NDCG@100)",
          "U_ndcg": "panel NDCG@100 on real later choices", "ndcg100": "NDCG@100 against the reference list",
          "A_ov10": "top-10 overlap with Truth A", "A_ov100": "top-100 overlap with Truth A",
          "A_rbo90": "rank-biased overlap (0.9) with Truth A", "A_rbo98": "rank-biased overlap (0.98) with Truth A",
          "A_tau100": "rank agreement with Truth A on its top 100", "B_mass10": "Truth-B top-10 mass captured",
          "B_ov100": "top-100 overlap with Truth B", "U_recall": "panel recall@100", "U_hit": "panel hit@100",
          "ov10": "top-10 overlap", "ov100": "top-100 overlap", "mass10": "top-10 mass captured",
          "rbo90": "rank-biased overlap (0.9)"}
MET_ZH = {"A_ndcg100": "对真值A的榜单质量(NDCG@100)", "B_ndcg100": "对真值B的榜单质量(NDCG@100)",
          "U_ndcg": "留出人群真实后续选择上的NDCG@100", "ndcg100": "对参考榜的NDCG@100",
          "A_ov10": "与真值A的前10重合", "A_ov100": "与真值A的前100重合", "A_rbo90": "与真值A的RBO(0.9)",
          "A_rbo98": "与真值A的RBO(0.98)", "A_tau100": "在真值A前100上的名次一致度",
          "B_mass10": "真值B前10所占份额", "B_ov100": "与真值B的前100重合", "U_recall": "留出人群recall@100",
          "U_hit": "留出人群hit@100", "ov10": "前10重合", "ov100": "前100重合", "mass10": "前10份额",
          "rbo90": "RBO(0.9)"}


def est_en(e):
    if e in EST_EN:
        return EST_EN[e]
    m = re.fullmatch(r"(ease|knn|ease_r25)\.w(\d+)", e)
    if m:
        name = {"ease": "EASE", "knn": "item-kNN", "ease_r25": "EASE (25% random hold-out)"}[m.group(1)]
        return f"{name} with {m.group(2)}% weight on personalisation"
    m = re.fullmatch(r"blend(\d+)", e)
    return f"the {m.group(1)}% blend of sample and global" if m else e


def est_zh(e):
    if e in EST_ZH:
        return EST_ZH[e]
    m = re.fullmatch(r"(ease|knn|ease_r25)\.w(\d+)", e)
    if m:
        name = {"ease": "EASE", "knn": "近邻(kNN)", "ease_r25": "EASE(25%随机留出)"}[m.group(1)]
        return f"{name}个性化模型(个性化权重{m.group(2)}%)"
    m = re.fullmatch(r"blend(\d+)", e)
    return f"样本与全体榜{m.group(1)}%混合的榜" if m else e


def met_en(m):
    return MET_EN.get(m, m)


def met_zh(m):
    return MET_ZH.get(m, m)


def kname(k):
    return "all" if k is None else str(k)


def ctx_en(f):
    """f = family tuple (src, ds, arm, group, k, eps)."""
    src, ds, arm, group, k, eps = f
    parts = []
    if arm not in ("", "uniform"):
        parts.append(f"{arm} sampler")
    if group:
        parts.append(f"subgroup {group}")
    if eps is not None:
        parts.append(f"eps={eps:g}")
    parts.append(f"k={kname(k)}")
    return ", ".join(parts)


def ctx_zh(f):
    src, ds, arm, group, k, eps = f
    parts = []
    if arm not in ("", "uniform"):
        parts.append(f"{arm}抽样")
    if group:
        parts.append(f"子群{group}")
    if eps is not None:
        parts.append(f"eps={eps:g}")
    parts.append("每人全部条数" if k is None else f"每人{k}条")
    return "，".join(parts)


def sk(t):
    """Sort key that tolerates None (k = all, eps unset) inside key tuples."""
    return tuple(-1 if x is None else x for x in t)


def p_one(z, df):
    """Upper-tail P(T > z) of Student t."""
    return float(stdtr(df, -z))


# --------------------------------------------------------------------------------------------- data store
def get_arm(r):
    if r.get("arm"):
        return r["arm"]
    m = re.search(r"\|arm=([^|]+)", r.get("key", ""))
    if m:
        return m.group(1)
    if r.get("users", "uniform") != "uniform":
        return r["users"]
    if r.get("rows", "random") != "random":
        return "recent_rows"
    return "uniform"


def row_key(src, r):
    ds = r["dataset"]
    k = r.get("k")
    k = None if k is None else int(k)
    if src in ("e1", "e7"):
        arm = get_arm(r)
        return ("e1" if arm == "uniform" else "e7", ds, arm, "", int(r["N"]), k, None)
    if src == "e2":
        return ("e2", ds, "", "", int(r["N"]), k, None)
    if src == "e5":
        return ("e5", ds, "", r["group"], int(r.get("N_req", r["N"])), k, None)
    if src == "e6":
        return ("e6", ds, "", "", int(r["N"]), k, float(r["eps"]))
    raise ValueError(src)


class Cell:
    __slots__ = ("draw", "X")

    def __init__(self, draw, X):
        self.draw, self.X = draw, X


class Store:
    """cell key = (src, dataset, arm, group, N, k, eps) -> Cell with per-draw columns (nan = missing)."""

    def __init__(self):
        self.pending = defaultdict(lambda: ([], []))
        self.colidx = defaultdict(dict)       # src -> column name -> index
        self.cells = {}
        self.fams = defaultdict(list)         # (src,ds,arm,group,k,eps) -> sorted Ns
        self._g = {}
        self.n_rows = 0

    def load(self, path, src):
        n = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue                  # partial last line of a file that is still being written
                try:
                    key = row_key(src, r)
                    draw = int(r["draw"])
                except (KeyError, TypeError, ValueError):
                    continue
                ci = self.colidx[key[0]]
                vals = {}
                for c, v in r.items():
                    if "." in c and not isinstance(v, (str, bool)) and (v is None or isinstance(v, (int, float))):
                        if c not in ci:
                            ci[c] = len(ci)
                        vals[ci[c]] = np.nan if v is None else float(v)
                arr = np.full(len(ci), np.nan)
                for i, v in vals.items():
                    arr[i] = v
                d, vs = self.pending[key]
                d.append(draw)
                vs.append(arr)
                n += 1
        self.n_rows += n
        return n

    def finalize(self):
        for key, (d, vs) in self.pending.items():
            W = len(self.colidx[key[0]])
            last = {}
            for i, dr in enumerate(d):
                last[dr] = i                  # duplicated draw ids: the last row wins
            order = sorted(last)
            X = np.full((len(order), W), np.nan)
            for j, dr in enumerate(order):
                v = vs[last[dr]]
                X[j, :len(v)] = v
            self.cells[key] = Cell(np.array(order), X)
        self.pending.clear()
        for key in self.cells:
            src, ds, arm, group, N, k, eps = key
            self.fams[(src, ds, arm, group, k, eps)].append(N)
        for f in self.fams:
            self.fams[f].sort()

    def has(self, src):
        return any(f[0] == src for f in self.fams)

    def ks_of(self, src, ds, arm, group, eps):
        ks = {f[4] for f in self.fams if f[:4] == (src, ds, arm, group) and f[5] == eps}
        return sorted(ks, key=lambda k: math.inf if k is None else k)

    def datasets(self, src):
        return sorted({f[1] for f in self.fams if f[0] == src})

    def columns(self, src):
        return self.colidx.get(src, {})

    def vec(self, key, col):
        c = self.cells.get(key)
        if c is None:
            return None
        i = self.colidx[key[0]].get(col)
        if i is None:
            return None
        return c.X[:, i]

    def pvec(self, key, x, y):
        a, b = self.vec(key, x), self.vec(key, y)
        return None if a is None or b is None else a - b

    def gvec(self, g):
        return self.vec(g[1], g[2]) if g[0] == "c" else self.pvec(g[1], g[2], g[3])

    def gstat(self, g):
        """(mean, variance, n) of a group; None if fewer than MIN_N finite draws."""
        h = (g[0], g[1]) + tuple(g[2:])
        if h in self._g:
            return self._g[h]
        v = self.gvec(g)
        out = None
        if v is not None:
            v = v[np.isfinite(v)]
            if v.size >= MIN_N:
                out = (float(v.mean()), float(v.var(ddof=1)), int(v.size))
        self._g[h] = out
        return out


def cg(key, col):
    return ["c", tuple(key), col]


def pg(key, x, y):
    return ["p", tuple(key), x, y]


def dcomp(ka, kb, col, d=1):
    return {"dir": d, "terms": [[1, cg(ka, col)], [-1, cg(kb, col)]]}


def pcomp(key, x, y, d=1):
    return {"dir": d, "terms": [[1, pg(key, x, y)]]}


def dodcomp(a1, a2, b1, b2, col, d=1):
    return {"dir": d, "terms": [[1, cg(a1, col)], [-1, cg(a2, col)], [-1, cg(b1, col)], [1, cg(b2, col)]]}


def ev_comp(S, comp):
    """(mean, se, df, n) of the signed sum of independent group means; None if untestable."""
    mean = vs = den = 0.0
    nmin = 10 ** 9
    for sign, g in comp["terms"]:
        st = S.gstat(g)
        if st is None:
            return None
        m, v, n = st
        t = v / n
        mean += sign * m
        vs += t
        den += t * t / (n - 1)
        nmin = min(nmin, n)
    if vs <= 1e-24 or den <= 0:
        return None
    return mean, math.sqrt(vs), vs * vs / den, nmin


def eval_spec(S, spec, two_sided=False):
    """Intersection-union evaluation of an oriented spec on store S."""
    evs = []
    for c in spec["comps"]:
        e = ev_comp(S, c)
        if e is None:
            return None
        evs.append(e)
    ps, zs, effs = [], [], []
    for e, c in zip(evs, spec["comps"]):
        z = c["dir"] * e[0] / e[1]
        ps.append(min(1.0, p_one(z, e[2]) * (2.0 if two_sided else 1.0)))
        zs.append(z)
        effs.append(c["dir"] * e[0])
    return {"p": max(ps), "z": min(zs), "effect": min(effs), "n": min(e[3] for e in evs),
            "means": [e[0] for e in evs]}


# --------------------------------------------------------------------------------------------- collector
class Collector:
    def __init__(self):
        self.ps = defaultdict(list)           # cls -> p of every scanned hypothesis (BH denominator)
        self.stored = defaultdict(list)       # cls -> (index into ps, record) for p <= P_STORE
        self.skipped = defaultdict(int)
        self.notes = defaultdict(list)
        self.seen = set()
        self.dups = defaultdict(int)

    def skip(self, cls):
        self.skipped[cls] += 1

    def add(self, cls, p, rec):
        i = len(self.ps[cls])
        self.ps[cls].append(p)
        if p <= P_STORE:
            rec["idx"] = i
            rec["p"] = p
            self.stored[cls].append(rec)


def hyp(S, C, cls, comps, two_sided, info, mult=1.0, orient=False, opposite=False, labels=None, curve=None):
    """Evaluate one hypothesis and file it.  orient: set each part's direction from the observed sign;
    opposite: the pattern requires the two parts to have opposite observed signs."""
    evs = [ev_comp(S, c) for c in comps]
    if any(e is None for e in evs):
        C.skip(cls)
        return None
    sig = (cls, info["cell"], tuple((round(e[0], 10), round(e[1], 10)) for e in evs))
    if sig in C.seen:                          # numerically identical series (e.g. pers_time == knn.w100)
        C.dups[cls] += 1
        return None
    C.seen.add(sig)
    if orient:
        for c, e in zip(comps, evs):
            c["dir"] = 1 if e[0] >= 0 else -1
        if opposite and comps[0]["dir"] == comps[1]["dir"]:
            C.add(cls, 1.0, {})
            return None
    spec = {"comps": comps}
    r = eval_spec(S, spec, two_sided)
    p = min(1.0, r["p"] * mult)
    rec = {"cls": cls, "info": info, "spec": {"comps": comps, "two_sided": False, "labels": labels or [],
                                               "curve": curve}, "z": r["z"], "effect": r["effect"],
           "n": r["n"], "means": r["means"], "mult": mult}
    C.add(cls, p, rec)
    return rec


# --------------------------------------------------------------------------------------------- scan classes
def metric_cols(src, S):
    """[(estimator, metrics)] to scan for learning curves in a source."""
    cols = S.columns(src)
    if src in ("e1", "e7"):
        est = E1_EST + (["ht"] if src == "e7" else [])
        return [(e, [m for m in CURVE_METRICS if f"{e}.{m}" in cols]) for e in est]
    if src == "e2":
        ests = sorted({c[:-len(".U_ndcg")] for c in cols if c.endswith(".U_ndcg")} - E2_DUP)
        return [(e, ["U_ndcg"]) for e in ests]
    if src == "e5":
        return [(e, ["ndcg100"]) for e in E5_EST if f"{e}.ndcg100" in cols]
    if src == "e6":
        return [(e, ["ndcg100"]) for e in E6_EST if f"{e}.ndcg100" in cols]
    return []


def scan_c1(S, C):
    for src in ("e1", "e7", "e2", "e5", "e6"):
        if not S.has(src):
            continue
        mc = metric_cols(src, S)
        for fam, Ns in sorted(S.fams.items(), key=lambda kv: str(kv[0])):
            if fam[0] != src or len(Ns) < 2:
                continue
            _, ds, arm, group, k, eps = fam
            for est, mets in mc:
                for met in mets:
                    col = f"{est}.{met}"
                    for N1, N2 in zip(Ns[:-1], Ns[1:]):
                        ka, kb = (src, ds, arm, group, N1, k, eps), (src, ds, arm, group, N2, k, eps)
                        comp = dcomp(ka, kb, col, 1)
                        rec = hyp(S, C, 1, [comp], False,
                                  dict(fam=list(fam), est=est, metric=met, N1=N1, N2=N2,
                                       cell=f"{ds}|{ctx_en(fam)}|N {N1}->{N2}"),
                                  labels=[f"{col}: N={N1} minus N={N2}"],
                                  curve=dict(series=[dict(label=col, kind="c", fam=list(fam), col=col)]))
                        if rec is not None and rec["z"] <= Z_NONMONO:
                            rec["gate_fail"] = True


def scan_c2(S, C):
    if not S.has("e1"):
        return
    C.notes[2].append("e2 rows hold only panel (U) metrics, so 'hyb_time - pop' cannot be compared with a "
                      "Truth-A metric: that half of the pre-registered pair is not testable on these data.")
    pairs = [("shrunk", "count"), ("lcb", "count")]
    targets = [("A_ndcg100", "U_ndcg"), ("A_ndcg100", "B_ndcg100")]
    cols = S.columns("e1")
    for key in sorted((k for k in S.cells if k[0] == "e1"), key=sk):
        _, ds, arm, group, N, k, eps = key
        for x, y in pairs:
            for m1, m2 in targets:
                if not all(f"{e}.{m}" in cols for e in (x, y) for m in (m1, m2)):
                    continue
                comps = [pcomp(key, f"{x}.{m1}", f"{y}.{m1}"), pcomp(key, f"{x}.{m2}", f"{y}.{m2}")]
                hyp(S, C, 2, comps, True,
                    dict(key=list(key), x=x, y=y, m1=m1, m2=m2, cell=f"{ds}|{ctx_en(('e1', ds, arm, group, k, eps))}|N={N}"),
                    orient=True, opposite=True, labels=[f"{x}-{y} on {m1}", f"{x}-{y} on {m2}"])


def _series(S, src, ds, arm, group, k, eps, xcol, ycol, Ns):
    out = []
    for N in Ns:
        key = (src, ds, arm, group, N, k, eps)
        st = S.gstat(pg(key, xcol, ycol))
        if st is None:
            continue
        m, v, n = st
        if v <= 1e-24:
            continue
        se = math.sqrt(v / n)
        out.append((N, key, m / se, n - 1))
    return out


def scan_c3(S, C):
    jobs = []
    if S.has("e1"):
        for x, y in [("shrunk", "count"), ("lcb", "count"), ("shrunk", "lcb")]:
            for m in CURVE_METRICS:
                jobs.append(("e1", f"{x}.{m}", f"{y}.{m}", x, y, m))
    if S.has("e2"):
        cols = S.columns("e2")
        have = [e for e in E2_PAIR_SET if f"{e}.U_ndcg" in cols]
        for i in range(len(have)):
            for j in range(i + 1, len(have)):
                jobs.append(("e2", f"{have[i]}.U_ndcg", f"{have[j]}.U_ndcg", have[i], have[j], "U_ndcg"))
    for src, xc, yc, x, y, m in jobs:
        for fam in sorted((f for f in S.fams if f[0] == src), key=sk):
            _, ds, arm, group, k, eps = fam
            Ns = S.fams[fam]
            ser = _series(S, src, ds, arm, group, k, eps, xc, yc, Ns)
            if len(ser) < 2:
                C.skip(3)
                continue
            nv = len(ser)
            # p of "difference > 0" and "difference < 0" at each N
            pp = [p_one(z, df) for _, _, z, df in ser]
            pn = [p_one(-z, df) for _, _, z, df in ser]
            best = None
            for sgn, p_first, p_second in ((1, pp, pn), (-1, pn, pp)):
                run_min, run_arg = 2.0, -1
                for b in range(1, nv):
                    if p_first[b - 1] < run_min:
                        run_min, run_arg = p_first[b - 1], b - 1
                    p = max(run_min, p_second[b])
                    if best is None or p < best[0]:
                        best = (p, sgn, run_arg, b)
            _, sgn, a, b = best
            ka, kb = ser[a][1], ser[b][1]
            comps = [pcomp(ka, xc, yc, sgn), pcomp(kb, xc, yc, -sgn)]
            fam_s = (src, ds, arm, group, k, eps)
            hyp(S, C, 3, comps, False,
                dict(fam=list(fam_s), x=x, y=y, metric=m, Na=ser[a][0], Nb=ser[b][0], sgn=sgn,
                     cell=f"{ds}|{ctx_en(fam_s)}|N {ser[a][0]} vs {ser[b][0]}"),
                mult=float(nv * (nv - 1)), labels=[f"{x}-{y} at N={ser[a][0]}", f"{x}-{y} at N={ser[b][0]}"],
                curve=dict(series=[dict(label=f"{x} - {y} ({m})", kind="p", fam=list(fam_s), x=xc, y=yc)]))


def scan_c4(S, C):
    # (a) N-effect differs by k: difference of differences over adjacent N and adjacent k
    jobs = []
    if S.has("e1"):
        jobs += [("e1", f"{e}.{m}", e, m) for e in E1_EST for m in CURVE_METRICS]
    if S.has("e2"):
        cols = S.columns("e2")
        jobs += [("e2", f"{e}.U_ndcg", e, "U_ndcg")
                 for e in sorted({c[:-len('.U_ndcg')] for c in cols if c.endswith('.U_ndcg')} - E2_DUP)]
    for src, col, est, met in jobs:
        for ds in S.datasets(src):
            ks = S.ks_of(src, ds, "uniform" if src == "e1" else "", "", None)
            arm = "uniform" if src == "e1" else ""
            for k1, k2 in zip(ks[:-1], ks[1:]):
                f1, f2 = (src, ds, arm, "", k1, None), (src, ds, arm, "", k2, None)
                Ns = sorted(set(S.fams.get(f1, [])) & set(S.fams.get(f2, [])))
                for N1, N2 in zip(Ns[:-1], Ns[1:]):
                    ka1, ka2 = (src, ds, arm, "", N1, k1, None), (src, ds, arm, "", N2, k1, None)
                    kb1, kb2 = (src, ds, arm, "", N1, k2, None), (src, ds, arm, "", N2, k2, None)
                    # N-effect = larger N minus smaller N, for k1 minus for k2
                    comp = dodcomp(ka2, ka1, kb2, kb1, col)
                    hyp(S, C, 4, [comp], True,
                        dict(sub="a", ds=ds, est=est, metric=met, k1=k1, k2=k2, N1=N1, N2=N2, f1=list(f1),
                             f2=list(f2), cell=f"{ds}|k {kname(k1)} vs {kname(k2)}|N {N1}->{N2}"),
                        orient=True, labels=[f"[{col} N {N1}->{N2} at k={kname(k1)}] minus [same at k={kname(k2)}]"],
                        curve=dict(series=[dict(label=f"{col} k={kname(k1)}", kind="c", fam=list(f1), col=col),
                                           dict(label=f"{col} k={kname(k2)}", kind="c", fam=list(f2), col=col)]))
    # (b) breadth vs depth at equal budget N*k: A_ndcg100 direction reversed by another metric
    if not S.has("e1"):
        return
    cols = S.columns("e1")
    for ds in S.datasets("e1"):
        by_b = defaultdict(list)
        for key in S.cells:
            if key[:4] == ("e1", ds, "uniform", "") and key[5] is not None:
                by_b[key[4] * key[5]].append((key[4], key[5]))
        for B, cs in sorted(by_b.items()):
            if len(cs) < 2:
                continue
            wide, deep = max(cs), min(cs)
            kw = ("e1", ds, "uniform", "", wide[0], wide[1], None)
            kd = ("e1", ds, "uniform", "", deep[0], deep[1], None)
            for est in E1_EST:
                if f"{est}.A_ndcg100" not in cols:
                    continue
                for met in E1_METRICS:
                    if met == "A_ndcg100" or f"{est}.{met}" not in cols:
                        continue
                    comps = [dcomp(kw, kd, f"{est}.A_ndcg100"), dcomp(kw, kd, f"{est}.{met}")]
                    hyp(S, C, 4, comps, True,
                        dict(sub="b", ds=ds, est=est, metric=met, B=B, wide=list(wide), deep=list(deep),
                             cell=f"{ds}|budget {B}|wide N={wide[0]},k={wide[1]} vs deep N={deep[0]},k={deep[1]}"),
                        orient=True, opposite=True,
                        labels=[f"wide-deep on {est}.A_ndcg100", f"wide-deep on {est}.{met}"])


def scan_c5(S, C):
    jobs = []
    if S.has("e1"):
        for x, y in [("shrunk", "count"), ("lcb", "count")]:
            for m in CURVE_METRICS:
                jobs.append(("e1", f"{x}.{m}", f"{y}.{m}", f"{x}-{y}", m, "uniform"))
    if S.has("e2"):
        cols = S.columns("e2")
        for x in E2_VS_POP:
            if f"{x}.U_ndcg" in cols and "pop.U_ndcg" in cols:
                jobs.append(("e2", f"{x}.U_ndcg", "pop.U_ndcg", f"{x}-pop", "U_ndcg", ""))
    for src, xc, yc, name, met, arm in jobs:
        by_nk = defaultdict(list)
        for key in S.cells:
            if key[0] == src:
                by_nk[(key[4], key[5])].append(key)
        for (N, k), keys in sorted(by_nk.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            rows = []
            for key in sorted(keys, key=sk):
                st = S.gstat(pg(key, xc, yc))
                if st is None or st[1] <= 1e-24:
                    C.skip(5)
                    continue
                m, v, n = st
                z = m / math.sqrt(v / n)
                rows.append((key, z, n - 1))
            nd = len(rows)
            if nd < 2:
                continue
            ppos = [(p_one(z, df), key) for key, z, df in rows]
            pneg = [(p_one(-z, df), key) for key, z, df in rows]
            ip, jn = min(ppos, key=lambda t: t[0])[1], min(pneg, key=lambda t: t[0])[1]
            if ip == jn:
                continue
            comps = [pcomp(ip, xc, yc, 1), pcomp(jn, xc, yc, -1)]
            hyp(S, C, 5, comps, False,
                dict(name=name, metric=met, N=N, k=k, ds_pos=ip[1], ds_neg=jn[1], nds=nd, src=src,
                     cell=f"all datasets|k={kname(k)}|N={N}"),
                mult=float(nd), labels=[f"{name} on {ip[1]}", f"{name} on {jn[1]}"])


def scan_c6(S, C):
    if not S.has("e2"):
        return
    cols = S.columns("e2")
    specs = []                                  # (label, mixed col, [other col, ...])
    for model in ("ease", "knn"):
        for w in (25, 50, 75):
            specs.append((f"{model}.w{w}", f"{model}.w{w}.U_ndcg", ["pop.U_ndcg", f"{model}.w100.U_ndcg"],
                          ("pop", f"{model}.w100")))
    for mode in ("time", "random"):
        specs.append((f"hyb_{mode}", f"hyb_{mode}.U_ndcg", ["pop.U_ndcg", f"pers_{mode}.U_ndcg"],
                      ("pop", f"pers_{mode}")))
    for fam in sorted((f for f in S.fams if f[0] == "e2"), key=sk):
        _, ds, arm, group, k, eps = fam
        for N in S.fams[fam]:
            key = ("e2", ds, arm, group, N, k, eps)
            for label, mixed, others, onames in specs:
                if mixed not in cols or any(o not in cols for o in others):
                    continue
                comps = [pcomp(key, mixed, o, -1) for o in others]
                hyp(S, C, 6, comps, False,
                    dict(ds=ds, mixed=label, ends=list(onames), N=N, k=k,
                         cell=f"{ds}|k={kname(k)}|N={N}"),
                    labels=[f"{label} - {n}" for n in onames])


def scan_c7(S, C):
    if not S.has("e1"):
        return
    cols = S.columns("e1")
    targets = [("A_ndcg100", "U_ndcg"), ("A_ndcg100", "B_ndcg100")]
    for fam, Ns in sorted(S.fams.items(), key=lambda kv: str(kv[0])):
        if fam[0] != "e1" or len(Ns) < 2:
            continue
        _, ds, arm, group, k, eps = fam
        for est in E1_EST:
            for m1, m2 in targets:
                if f"{est}.{m1}" not in cols or f"{est}.{m2}" not in cols:
                    continue
                for N1, N2 in zip(Ns[:-1], Ns[1:]):
                    ka, kb = ("e1", ds, arm, group, N1, k, eps), ("e1", ds, arm, group, N2, k, eps)
                    # effect of the extra people = larger N minus smaller N
                    comps = [dcomp(kb, ka, f"{est}.{m1}"), dcomp(kb, ka, f"{est}.{m2}")]
                    hyp(S, C, 7, comps, True,
                        dict(fam=list(fam), est=est, m1=m1, m2=m2, N1=N1, N2=N2,
                             cell=f"{ds}|{ctx_en(fam)}|N {N1}->{N2}"),
                        orient=True, opposite=True, labels=[f"{est}.{m1}: N={N2} minus N={N1}",
                                                            f"{est}.{m2}: N={N2} minus N={N1}"],
                        curve=dict(series=[dict(label=f"{est}.{m1}", kind="c", fam=list(fam), col=f"{est}.{m1}"),
                                           dict(label=f"{est}.{m2}", kind="c", fam=list(fam), col=f"{est}.{m2}")]))


def scan_c8(S, C):
    if not S.has("e6"):
        return
    cols = S.columns("e6")
    for key in sorted((k for k in S.cells if k[0] == "e6"), key=sk):
        _, ds, arm, group, N, k, eps = key
        for est in ("laplace", "gaussian", "ldp_oue"):
            for met in ("ndcg100", "ov100"):
                if f"{est}.{met}" not in cols or f"nonprivate.{met}" not in cols:
                    continue
                hyp(S, C, 8, [pcomp(key, f"{est}.{met}", f"nonprivate.{met}", 1)], False,
                    dict(ds=ds, est=est, metric=met, N=N, k=k, eps=eps,
                         cell=f"{ds}|k={kname(k)}|eps={eps:g}|N={N}"),
                    labels=[f"{est}-nonprivate on {met}"])


# --------------------------------------------------------------------------------------------- BH + text
def bh(ps):
    p = np.asarray(ps, dtype=float)
    m = p.size
    if m == 0:
        return p
    o = np.argsort(p, kind="stable")
    q = p[o] * m / (np.arange(m) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(q, 1.0)
    return out


def rise(d, zh=False):
    return ("提高" if d > 0 else "降低") if zh else ("raises" if d > 0 else "lowers")


def describe(S, rec):
    c, i, m = rec["cls"], rec["info"], rec["means"]
    if c == 1:
        f = tuple(i["fam"])
        N1k = (f[0], f[1], f[2], f[3], i["N1"], f[4], f[5])
        N2k = (f[0], f[1], f[2], f[3], i["N2"], f[4], f[5])
        col = f"{i['est']}.{i['metric']}"
        m1, m2 = S.gstat(cg(N1k, col))[0], S.gstat(cg(N2k, col))[0]
        en = (f"On {f[1]} ({ctx_en(f)}), the {met_en(i['metric'])} of {est_en(i['est'])} falls from {m1:.3f} at "
              f"N={i['N1']} to {m2:.3f} at N={i['N2']}, although more people were sampled.")
        zh = (f"在{f[1]}（{ctx_zh(f)}）上，{est_zh(i['est'])}的{met_zh(i['metric'])}从N={i['N1']}时的{m1:.3f}"
              f"降到N={i['N2']}时的{m2:.3f}，尽管抽的人更多了。")
    elif c == 2:
        en = (f"On {i['cell'].split('|')[0]} at N={i['key'][4]}, k={kname(i['key'][5])}, using {est_en(i['x'])} instead of "
              f"{est_en(i['y'])} {rise(m[0])} the {met_en(i['m1'])} ({m[0]:+.3f}) but {rise(m[1])} the "
              f"{met_en(i['m2'])} ({m[1]:+.3f}).")
        zh = (f"在{i['key'][1]}、N={i['key'][4]}、{ctx_zh(('e1', i['key'][1], 'uniform', '', i['key'][5], None))}时，"
              f"用{est_zh(i['x'])}代替{est_zh(i['y'])}，{met_zh(i['m1'])}{rise(m[0], True)}{abs(m[0]):.3f}，"
              f"而{met_zh(i['m2'])}却{rise(m[1], True)}{abs(m[1]):.3f}。")
    elif c == 3:
        f = tuple(i["fam"])
        en = (f"On {f[1]} ({ctx_en(f)}), {est_en(i['x'])} {'beats' if m[0] > 0 else 'trails'} {est_en(i['y'])} on "
              f"{met_en(i['metric'])} at N={i['Na']} ({m[0]:+.3f}) but {'trails' if m[0] > 0 else 'beats'} it at "
              f"N={i['Nb']} ({m[1]:+.3f}).")
        zh = (f"在{f[1]}（{ctx_zh(f)}）上，就{met_zh(i['metric'])}而言，N={i['Na']}时{est_zh(i['x'])}"
              f"{'优于' if m[0] > 0 else '劣于'}{est_zh(i['y'])}（{m[0]:+.3f}），N={i['Nb']}时则"
              f"{'劣于' if m[0] > 0 else '优于'}它（{m[1]:+.3f}）。")
    elif c == 4 and i["sub"] == "a":
        f1 = tuple(i["f1"])
        en = (f"On {i['ds']}, going from N={i['N1']} to N={i['N2']} changes the {met_en(i['metric'])} of "
              f"{est_en(i['est'])} by a different amount when k={kname(i['k1'])} than when k={kname(i['k2'])} "
              f"(difference of differences {m[0]:+.3f}).")
        zh = (f"在{i['ds']}上，从N={i['N1']}增到N={i['N2']}，{est_zh(i['est'])}的{met_zh(i['metric'])}变化量"
              f"在每人{kname(i['k1'])}条与每人{kname(i['k2'])}条时不同（差之差{m[0]:+.3f}）。")
    elif c == 4:
        w, d = i["wide"], i["deep"]
        en = (f"On {i['ds']} at equal budget N*k={i['B']}, many shallow samples (N={w[0]}, k={w[1]}) vs few deep ones "
              f"(N={d[0]}, k={d[1]}) differ by {m[0]:+.3f} on Truth-A NDCG@100 for {est_en(i['est'])} but by "
              f"{m[1]:+.3f} on {met_en(i['metric'])}, i.e. in the opposite direction.")
        zh = (f"在{i['ds']}上，总预算N*k={i['B']}相同，{est_zh(i['est'])}的“人多条少”(N={w[0]},k={w[1]})比“人少条多”"
              f"(N={d[0]},k={d[1]})在真值A的NDCG@100上相差{m[0]:+.3f}，而在{met_zh(i['metric'])}上相差{m[1]:+.3f}，方向相反。")
    elif c == 5:
        en = (f"For {i['name']} on {met_en(i['metric'])} at N={i['N']}, k={kname(i['k'])}, the effect is "
              f"{m[0]:+.3f} on {i['ds_pos']} but {m[1]:+.3f} on {i['ds_neg']}.")
        nm = i["name"].split("-")
        zh = (f"在N={i['N']}、{ctx_zh(('', '', '', '', i['k'], None))}时，“{est_zh(nm[0])}减{est_zh(nm[1])}”"
              f"在{met_zh(i['metric'])}上，于{i['ds_pos']}为{m[0]:+.3f}，于{i['ds_neg']}为{m[1]:+.3f}，方向相反。")
    elif c == 6:
        a, b = i["ends"]
        en = (f"On {i['ds']} (k={kname(i['k'])}, N={i['N']}), {est_en(i['mixed'])} scores below both {est_en(a)} "
              f"({m[0]:+.3f}) and {est_en(b)} ({m[1]:+.3f}) on real later choices.")
        zh = (f"在{i['ds']}（每人{kname(i['k'])}条，N={i['N']}）上，{est_zh(i['mixed'])}在真实后续选择上比{est_zh(a)}"
              f"（{m[0]:+.3f}）和{est_zh(b)}（{m[1]:+.3f}）都更差。")
    elif c == 7:
        f = tuple(i["fam"])
        en = (f"On {f[1]} ({ctx_en(f)}), going from N={i['N1']} to N={i['N2']} {rise(m[0])} the {met_en(i['m1'])} of "
              f"{est_en(i['est'])} ({m[0]:+.3f}) but {rise(m[1])} its {met_en(i['m2'])} ({m[1]:+.3f}).")
        zh = (f"在{f[1]}（{ctx_zh(f)}）上，从N={i['N1']}增到N={i['N2']}，{est_zh(i['est'])}的{met_zh(i['m1'])}"
              f"{rise(m[0], True)}{abs(m[0]):.3f}，但{met_zh(i['m2'])}{rise(m[1], True)}{abs(m[1]):.3f}。")
    else:
        en = (f"On {i['ds']} (k={kname(i['k'])}, N={i['N']}, eps={i['eps']:g}), {est_en(i['est'])} scores {m[0]:+.3f} "
              f"higher than the non-private list on {met_en(i['metric'])}.")
        zh = (f"在{i['ds']}（每人{kname(i['k'])}条，N={i['N']}，eps={i['eps']:g}）上，{est_zh(i['est'])}的"
              f"{met_zh(i['metric'])}比无隐私的榜高{m[0]:+.3f}。")
    return en, zh


# --------------------------------------------------------------------------------------------- fresh tasks
def cells_of_spec(spec):
    seen, out = set(), []
    for c in spec["comps"]:
        for _, g in c["terms"]:
            key = tuple(g[1])
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def make_task(key, draw):
    """Task dict formatted exactly like run_all.T / build (key = kind|ds|sorted k=v ...|d=draw)."""
    src, ds, arm, group, N, k, eps = key
    if src == "e1":
        kind, kw = "e1", dict(N=N, k=k)
    elif src == "e7":
        kind, kw = "e1", dict(N=N, k=k, arm=arm)
    elif src == "e2":
        kind, kw = "e2", dict(N=N, k=k)
    elif src == "e5":
        kind, kw = "e5", dict(group=group, N=N, k=k)
    else:
        kind, kw = "e6", dict(N=N, k=k, eps=eps)
    t = dict(kind=kind, key=f"{kind}|{ds}|" + "|".join(f"{a}={kw[a]}" for a in sorted(kw)) + f"|d={draw}",
             dataset=ds, draw=draw, **kw)
    if src == "e7":
        t.update(ARM_KW[arm])
    return t


def fresh_tasks(hits):
    tasks, seen = [], set()
    for rec in hits[:TOP_TASKS]:
        for key in cells_of_spec(rec["spec"]):
            for d in range(FRESH_DRAW0, FRESH_DRAW0 + N_FRESH):
                t = make_task(key, d)
                if t["key"] not in seen:
                    seen.add(t["key"])
                    tasks.append(t)
    return tasks


# --------------------------------------------------------------------------------------------- io helpers
def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def fnum(x, nd=6):
    return "" if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)


def tuplify(spec):
    for c in spec["comps"]:
        for t in c["terms"]:
            t[1][1] = tuple(t[1][1])
    return spec


def load_runs(runs, names):
    S = Store()
    got = []
    for fn, src in names:
        p = os.path.join(runs, fn)
        if os.path.exists(p):
            n = S.load(p, src)
            got.append(f"{fn}:{n}")
    S.finalize()
    return S, got


def scan_names():
    return [("e1.jsonl", "e1"), ("e2.jsonl", "e2"), ("e5.jsonl", "e5"), ("e6.jsonl", "e6"), ("e7.jsonl", "e7")]


def fresh_names():
    return [("e1_fresh.jsonl", "e1"), ("e2_fresh.jsonl", "e2"), ("e5_fresh.jsonl", "e5"),
            ("e6_fresh.jsonl", "e6"), ("e7_fresh.jsonl", "e7")]


# --------------------------------------------------------------------------------------------- scan stage
def run_scan(S, out, log):
    C = Collector()
    for fn in (scan_c1, scan_c2, scan_c3, scan_c4, scan_c5, scan_c6, scan_c7, scan_c8):
        t0 = time.time()
        fn(S, C)
        log(f"  {fn.__name__}: {time.time() - t0:.1f}s")
    cands, counts = [], {}
    for c in CLASSES:
        ps = C.ps.get(c, [])
        q = bh(ps)
        n_hit = 0
        for rec in C.stored.get(c, []):
            rec["q"] = float(q[rec["idx"]])
            rec["hit"] = bool(rec["q"] <= Q_FDR and not rec.get("gate_fail"))
            n_hit += rec["hit"]
            cands.append(rec)
        counts[c] = dict(scanned=len(ps), skipped=C.skipped.get(c, 0), duplicates=C.dups.get(c, 0),
                         nominal_p05=int(sum(1 for p in ps if p < 0.05)), significant=int(n_hit))
    cands.sort(key=lambda r: (not r["hit"], r["q"], -r["z"], r["cls"]))
    for n, rec in enumerate(cands):
        rec["cid"] = f"C{rec['cls']}-{n:05d}"
        rec["desc_en"], rec["desc_zh"] = describe(S, rec)
    return C, cands, counts


def write_scan_tables(out, cands, counts, C, S):
    td = os.path.join(out, "03_tables")
    os.makedirs(td, exist_ok=True)
    head = ["cid", "class", "dataset", "cell", "estimators", "metric", "effect", "z", "p", "q", "n_draws_min",
            "fdr_hit", "description", "spec"]

    def names(r):
        i, c = r["info"], r["cls"]
        if c in (1, 4, 7):
            e = i.get("est", "")
        elif c in (2, 3):
            e = f"{i['x']} vs {i['y']}"
        elif c == 5:
            e = i["name"]
        elif c == 6:
            e = f"{i['mixed']} vs {' & '.join(i['ends'])}"
        else:
            e = f"{i['est']} vs nonprivate"
        return e

    def metric(r):
        i = r["info"]
        return i.get("metric") or "/".join(x for x in (i.get("m1"), i.get("m2")) if x)

    def ds_of(r):
        i = r["info"]
        if "ds" in i:
            return i["ds"]
        if "fam" in i:
            return i["fam"][1]
        if "key" in i:
            return i["key"][1]
        return "multiple"

    with open(os.path.join(td, "hunt_candidates.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for r in cands:
            w.writerow([r["cid"], CLASSES[r["cls"]][1], ds_of(r), r["info"]["cell"], names(r), metric(r),
                        fnum(r["effect"]), fnum(r["z"], 3), f"{r['p']:.3g}", f"{r['q']:.3g}", r["n"],
                        "FDR hit" if r["hit"] else "nominal only (not FDR)", r["desc_en"],
                        json.dumps(jsonable(r["spec"]))])
    with open(os.path.join(td, "hunt_candidates_zh.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cid", "class", "dataset", "cell", "estimators", "metric", "effect", "z", "p", "q",
                    "fdr_hit", "description_zh"])
        for r in cands:
            w.writerow([r["cid"], CLASSES[r["cls"]][2], ds_of(r), r["info"]["cell"], names(r), metric(r),
                        fnum(r["effect"]), fnum(r["z"], 3), f"{r['p']:.3g}", f"{r['q']:.3g}",
                        "FDR命中" if r["hit"] else "仅名义显著(未过FDR)", r["desc_zh"]])
    hits = [r for r in cands if r["hit"]]
    tasks = fresh_tasks(hits)
    with open(os.path.join(td, "hunt_replication_tasks.json"), "w") as f:
        json.dump(jsonable(tasks), f, indent=0)
    with open(os.path.join(td, "hunt_scan_counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class", "scanned", "skipped_untestable", "duplicates_removed",
                    "nominal_p_lt_0.05", "fdr_hits_q_le_0.10", "notes"])
        for c, v in counts.items():
            w.writerow([CLASSES[c][0], CLASSES[c][1], v["scanned"], v["skipped"], v["duplicates"],
                        v["nominal_p05"], v["significant"], " ".join(C.notes.get(c, []))])
    return hits, tasks


# --------------------------------------------------------------------------------------------- replication
def replicate(F, rec):
    """Re-estimate a candidate on fresh draws only."""
    spec = rec["spec"]
    r = eval_spec(F, spec, two_sided=False)
    if r is None:
        return dict(status="no fresh data", n=0, effect=None, z=None, p=None, replicated=False)
    return dict(status="replicated" if r["p"] < P_REP else "not replicated", n=r["n"], effect=r["effect"],
                z=r["z"], p=r["p"], replicated=bool(r["p"] < P_REP))


def read_candidates(td):
    zh = {}
    p = os.path.join(td, "hunt_candidates_zh.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                zh[r["cid"]] = r["description_zh"]
    out = []
    with open(os.path.join(td, "hunt_candidates.csv")) as f:
        for r in csv.DictReader(f):
            spec = tuplify(json.loads(r["spec"]))
            cls = int(r["class"].split()[0])
            out.append(dict(cid=r["cid"], cls=cls, q=float(r["q"]), p=float(r["p"]), z=float(r["z"] or 0),
                            effect=float(r["effect"] or 0), hit=r["fdr_hit"] == "FDR hit", spec=spec,
                            desc_en=r["description"], desc_zh=zh.get(r["cid"], ""), cell=r["cell"],
                            dataset=r["dataset"]))
    return out


def read_counts(td):
    out = {}
    with open(os.path.join(td, "hunt_scan_counts.csv")) as f:
        for r in csv.DictReader(f):
            cid = [c for c, v in CLASSES.items() if v[0] == r["class_id"]][0]
            out[cid] = dict(scanned=int(r["scanned"]), skipped=int(r["skipped_untestable"]),
                            duplicates=int(r["duplicates_removed"]),
                            nominal_p05=int(r["nominal_p_lt_0.05"]), significant=int(r["fdr_hits_q_le_0.10"]),
                            notes=r["notes"])
    return out


# --------------------------------------------------------------------------------------------- verdicts
def class_verdict(c, cnt, rep_rows, total, have_fresh):
    sig = cnt["significant"]
    hits = [r for r in rep_rows if r["cls"] == c]
    tested = [r for r in hits if r["status"] != "no fresh data"]
    nrep = sum(1 for r in tested if r["replicated"])
    nums = {"scanned": cnt["scanned"], "skipped_untestable": cnt["skipped"],
            "duplicates_removed": cnt.get("duplicates", 0), "nominal_p_lt_0.05": cnt["nominal_p05"],
            "significant_q_le_0.10": sig, "rerun_on_fresh": len(tested), "replicated": nrep,
            "total_hypotheses_all_classes": total}
    nm = CLASSES[c][1][2:]
    note = (" " + cnt["notes"]) if cnt.get("notes") else ""
    if cnt["scanned"] == 0:
        v = "not testable"
        en = f"No hypothesis of this kind could be tested: the needed stages or cells are missing in the data.{note}"
        zh = "这一类假设无法检验：数据里缺少所需的实验阶段或格子。"
    elif sig == 0:
        v = "not supported"
        en = (f"Of {cnt['scanned']} hypotheses scanned ({total} in all classes), none survives the false-discovery "
              f"control, so nothing of this kind is reported as a finding.{note}")
        zh = f"共扫描{cnt['scanned']}个这类假设（所有类别合计{total}个），没有一个通过假发现率控制，因此不作为发现报告。"
    elif not have_fresh:
        v = "not testable"
        en = (f"Of {cnt['scanned']} hypotheses scanned ({total} in all classes), {sig} pass the false-discovery control "
              f"(q <= 0.10); they have NOT yet been re-run on fresh draws, so none is a finding.{note}")
        zh = (f"共扫描{cnt['scanned']}个这类假设（所有类别合计{total}个），{sig}个通过假发现率控制(q<=0.10)；"
              f"尚未用新抽样重跑，所以都还不算发现。")
    elif not tested:
        v = "not testable"
        en = (f"Of {cnt['scanned']} hypotheses scanned, {sig} pass the false-discovery control but none could be "
              f"re-run because the fresh data lack their cells; none is a finding.{note}")
        zh = f"共扫描{cnt['scanned']}个，{sig}个通过假发现率控制，但新数据里缺少对应格子，无法重跑，因此都不算发现。"
    elif nrep == 0:
        v = "not supported"
        en = (f"Of {cnt['scanned']} hypotheses scanned ({total} in all classes), {sig} pass the false-discovery control; "
              f"of the {len(tested)} re-run on fresh draws none replicated, so these are chance patterns.{note}")
        zh = (f"共扫描{cnt['scanned']}个这类假设（所有类别合计{total}个），{sig}个通过假发现率控制；"
              f"其中{len(tested)}个用新抽样重跑，无一重现，因此属于偶然现象。")
    else:
        v = "supported" if nrep >= 0.5 * len(hits) else "mixed"
        en = (f"Of {cnt['scanned']} hypotheses scanned ({total} in all classes), {sig} pass the false-discovery control; "
              f"{nrep} of the {len(tested)} re-run on fresh draws replicated (same sign, one-sided p < 0.05). "
              f"Only the replicated ones count as findings; all remain exploratory.{note}")
        zh = (f"共扫描{cnt['scanned']}个这类假设（所有类别合计{total}个），{sig}个通过假发现率控制；"
              f"用新抽样重跑的{len(tested)}个中有{nrep}个重现（同方向且单侧p<0.05）。只有重现的才算发现，且仍属探索性。")
    left = len(hits) - len(tested)
    if tested and left > 0:
        en += (f" {left} further hits could not be re-run (fresh tasks were issued for the top {TOP_TASKS} hits "
               f"only) and count as unreplicated.")
        zh += f"另有{left}个命中未能重跑（新抽样任务只覆盖q最小的前{TOP_TASKS}个），按未重现计。"
    ex = sorted(hits, key=lambda r: (not r["replicated"], r["status"] == "no fresh data", r["q"]))[:3]
    examples = [dict(cid=r["cid"], status=r["status"], reading_en=r["desc_en"], reading_zh=r["desc_zh"])
                for r in ex]
    rule = (RULE_COMMON + " Class verdict rule (fixed before use): supported = at least half of ALL FDR hits of the "
            "class replicate (hits that could not be re-run count as not replicated); mixed = some but fewer; "
            "not supported = no FDR hit or none replicated; not testable = hits not yet re-run or cells missing.")
    return dict(id=CLASSES[c][0], title_en=f"Surprise hunt, pattern class {CLASSES[c][1]} (exploratory)",
                title_zh=f"意外发现搜索：{CLASSES[c][2]}（探索性）", verdict=v, numbers=nums, reading_en=en,
                reading_zh=zh, rule=rule, examples=examples)


# --------------------------------------------------------------------------------------------- figures
def fig_counts(out, counts, reps):
    fd = os.path.join(out, "04_figures")
    os.makedirs(fd, exist_ok=True)
    cl = list(CLASSES)
    lab = [CLASSES[c][1] for c in cl]
    scanned = [counts[c]["scanned"] for c in cl]
    sig = [counts[c]["significant"] for c in cl]
    rep = None if reps is None else [sum(1 for r in reps if r["cls"] == c and r["replicated"]) for c in cl]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    y = np.arange(len(cl))[::-1]
    series = [("scanned", scanned, "#9aa5b1"), ("significant (BH q <= 0.10)", sig, "#d98c2b")]
    if rep is not None:
        series.append(("replicated on fresh draws", rep, "#2a7f62"))
    hh = 0.8 / len(series)
    for j, (nm, vals, col) in enumerate(series):
        yy = y + 0.4 - hh * (j + 0.5)
        ax.barh(yy, [max(v, 0.5) - 0.5 if v > 0 else 0 for v in vals], left=0.5, height=hh * 0.9, color=col, label=nm)
        for yv, v in zip(yy, vals):
            ax.text(max(v, 0.5) * 1.12, yv, str(v), va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.5, max(scanned + [10]) * 6)
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=8)
    ax.set_xlabel("number of hypotheses (log scale)")
    tot, nsig = sum(scanned), sum(sig)
    if rep is None:
        title = f"{tot:,} hypotheses scanned, {nsig} pass FDR control; none re-run on fresh draws yet"
    else:
        title = f"{tot:,} hypotheses scanned, {nsig} pass FDR control, {sum(rep)} replicate on fresh draws"
    ax.set_title(textwrap.fill(title, 70), fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=len(series), frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(fd, "hunt_counts.png"), dpi=130)
    plt.close(fig)


def boot_comp(S, comp, rng):
    """(effect, lo, hi) of a part (signed sum of group means), resampling draws within every group."""
    groups = []
    for sign, g in comp["terms"]:
        v = S.gvec(g)
        if v is None:
            return None
        v = v[np.isfinite(v)]
        if v.size < MIN_N:
            return None
        groups.append((sign, v))
    est = sum(s * v.mean() for s, v in groups)
    b = np.zeros(NBOOT)
    for s, v in groups:
        b += s * v[rng.integers(0, v.size, size=(NBOOT, v.size))].mean(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def series_vec(S, ser, key):
    v = S.vec(key, ser["col"]) if ser["kind"] == "c" else S.pvec(key, ser["x"], ser["y"])
    return None if v is None else v[np.isfinite(v)]


def fig_best(out, S, F, rec, rep, desc_status):
    fd = os.path.join(out, "04_figures")
    os.makedirs(fd, exist_ok=True)
    rng = np.random.default_rng(SEED)
    spec = rec["spec"]
    curve = spec.get("curve")
    has_curve = bool(curve and S is not None)
    fig, axes = plt.subplots(1, 2 if has_curve else 1, figsize=(9.0 if has_curve else 7.4, 4.6))
    axes = np.atleast_1d(axes)
    if has_curve:
        ax = axes[0]
        cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for si, ser in enumerate(curve["series"]):
            fam = tuple(ser["fam"])
            Ns = S.fams.get(fam, [])
            xs, ms, los, his = [], [], [], []
            for N in Ns:
                key = (fam[0], fam[1], fam[2], fam[3], N, fam[4], fam[5])
                v = series_vec(S, ser, key)
                if v is None or v.size < MIN_N:
                    continue
                m, lo, hi = MT.paired_bootstrap_ci(v, NBOOT, 0.05, rng)
                xs.append(N), ms.append(m), los.append(lo), his.append(hi)
            if not xs:
                continue
            col = cols[si % len(cols)]
            ax.plot(xs, ms, "-o", color=col, ms=4, label=f"{ser['label']} (scan draws)")
            ax.fill_between(xs, los, his, color=col, alpha=0.18)
            if F is not None:
                fx, fm = [], []
                for N in Ns:
                    key = (fam[0], fam[1], fam[2], fam[3], N, fam[4], fam[5])
                    v = series_vec(F, ser, key) if key in F.cells else None
                    if v is not None and v.size >= 1:
                        fx.append(N), fm.append(float(v.mean()))
                if fx:
                    ax.plot(fx, fm, "D", color=col, mfc="white", ms=7, label=f"{ser['label']} (fresh draws)")
        ax.set_xscale("log")
        ax.set_xlabel("people sampled, N (log scale)")
        ax.set_ylabel("mean score over draws (band: 95% bootstrap CI)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        axf = axes[1]
    else:
        axf = axes[0]
    labels = spec.get("labels") or [f"part {i + 1}" for i in range(len(spec["comps"]))]
    ypos = np.arange(len(spec["comps"]))[::-1]
    for st, store, col, off, mk in (("scan draws", S, "#33618f", 0.12, "o"), ("fresh draws", F, "#2a7f62", -0.12, "D")):
        if store is None:
            continue
        for y, c in zip(ypos, spec["comps"]):
            r = boot_comp(store, c, rng)
            if r is None:
                continue
            m, lo, hi = r
            axf.errorbar([m], [y + off], xerr=[[m - lo], [hi - m]], fmt=mk, color=col, capsize=3,
                         label=st if y == ypos[0] else None)
    axf.axvline(0, color="k", lw=0.8)
    axf.set_ylim(-0.7, len(ypos) - 0.3)
    axf.set_yticks(ypos)
    axf.set_yticklabels([textwrap.fill(s, 26) for s in labels], fontsize=7)
    axf.set_xlabel("difference of per-draw means (95% bootstrap CI)")
    axf.legend(fontsize=7, loc="best")
    axf.grid(axis="x", alpha=0.25)
    fig.suptitle(textwrap.fill(f"{rec['desc_en']} [{desc_status}]", 100), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(fd, "hunt_best.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default="results/small_sample_popularity/02_runs")
    ap.add_argument("--out", default="results/small_sample_popularity")
    ap.add_argument("--stage", choices=["scan", "check"], default="scan")
    ap.add_argument("--fresh", default=None, help="dir with e1_fresh.jsonl / e2_fresh.jsonl (stage check)")
    a = ap.parse_args()
    td = os.path.join(a.out, "03_tables")
    os.makedirs(td, exist_ok=True)
    t0 = time.time()

    def log(s):
        print(s, flush=True)

    if a.stage == "scan":
        S, got = load_runs(a.runs, scan_names())
        log(f"loaded {', '.join(got) or 'NOTHING'} from {a.runs}: {len(S.cells)} cells")
        C, cands, counts = run_scan(S, a.out, log)
        hits, tasks = write_scan_tables(a.out, cands, counts, C, S)
        total = sum(v["scanned"] for v in counts.values())
        log("=" * 78)
        log(f"TOTAL HYPOTHESES SCANNED: {total}  (per class: " +
            ", ".join(f"{c}:{v['scanned']}" for c, v in counts.items()) + ")")
        log(f"untestable (too few draws / zero variance): {sum(v['skipped'] for v in counts.values())};  "
            f"identical duplicates removed before BH: {sum(v['duplicates'] for v in counts.values())}")
        log(f"nominal p<0.05: {sum(v['nominal_p05'] for v in counts.values())};  "
            f"BH q<=0.10 hits: {len(hits)}  (expect ~5% nominal hits by chance alone)")
        log(f"fresh replication tasks written: {len(tasks)} for the top {min(TOP_TASKS, len(hits))} hits")
        log("=" * 78)
        for n, c in C.notes.items():
            for s in c:
                log(f"note class {n}: {s}")
        rows = [dict(cid=r["cid"], cls=r["cls"], status="not run", replicated=False, q=r["q"], desc_en=r["desc_en"],
                     desc_zh=r["desc_zh"]) for r in hits]
        verd = []
        for c in CLASSES:
            counts[c]["notes"] = " ".join(C.notes.get(c, []))
            verd.append(class_verdict(c, counts[c], rows, total, have_fresh=False))
        with open(os.path.join(td, "an_hunt_verdicts.json"), "w") as f:
            json.dump(jsonable(verd), f, indent=1, ensure_ascii=False)
        fig_counts(a.out, counts, None)
        best = hits[0] if hits else (cands[0] if cands else None)
        if best is not None:
            fig_best(a.out, S, None, best, None, "FDR hit, replication not run" if best["hit"] else "nominal only")
        log(f"done in {time.time() - t0:.0f}s")
        return

    # ------------------------------------------------------------------ stage check
    if not a.fresh:
        raise SystemExit("--stage check needs --fresh <dir>")
    if not os.path.exists(os.path.join(td, "hunt_candidates.csv")):
        raise SystemExit("run --stage scan first (hunt_candidates.csv missing)")
    cands = read_candidates(td)
    counts = read_counts(td)
    F, got = load_runs(a.fresh, fresh_names())
    log(f"fresh data: {', '.join(got) or 'NOTHING'} from {a.fresh}: {len(F.cells)} cells")
    hits = [r for r in cands if r["hit"]]
    rows = []
    for r in hits:
        rp = replicate(F, r)
        rows.append(dict(r, status=rp["status"], replicated=rp["replicated"], f_n=rp["n"],
                         f_effect=rp["effect"], f_z=rp["z"], f_p=rp["p"]))
    scan = {c["cid"]: c for c in hits}
    with open(os.path.join(td, "hunt_replication.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cid", "class", "dataset", "cell", "q_scan", "scan_effect", "scan_z", "fresh_n_draws_min",
                    "fresh_effect", "fresh_z", "fresh_p_one_sided", "status", "replicated", "description"])
        for r in rows:
            s = scan[r["cid"]]
            w.writerow([r["cid"], CLASSES[r["cls"]][1], s["dataset"], s["cell"], f"{s['q']:.3g}", fnum(s["effect"]),
                        fnum(s["z"], 3), r["f_n"], fnum(r["f_effect"]), fnum(r["f_z"], 3),
                        "" if r["f_p"] is None else f"{r['f_p']:.3g}", r["status"], r["replicated"], s["desc_en"]])
    total = sum(v["scanned"] for v in counts.values())
    verd = [class_verdict(c, counts[c], rows, total, have_fresh=len(F.cells) > 0) for c in CLASSES]
    with open(os.path.join(td, "an_hunt_verdicts.json"), "w") as f:
        json.dump(jsonable(verd), f, indent=1, ensure_ascii=False)
    nrep = sum(1 for r in rows if r["replicated"])
    log("=" * 78)
    log(f"TOTAL HYPOTHESES SCANNED: {total};  FDR hits: {len(hits)};  re-run on fresh data: "
        f"{sum(1 for r in rows if r['status'] != 'no fresh data')};  replicated: {nrep}")
    log("=" * 78)
    fig_counts(a.out, counts, rows)
    S, got = load_runs(a.runs, scan_names())
    pick = sorted([r for r in rows if r["replicated"]], key=lambda r: r["f_p"])
    if pick:
        best, status = pick[0], "replicated on fresh draws"
    elif hits:
        best, status = min((r for r in rows if r["status"] != "no fresh data"), key=lambda r: r["q"], default=rows[0]), \
            "NOT replicated"
    else:
        best, status = None, ""
    if best is not None:
        rec = scan[best["cid"]]
        fig_best(a.out, S, F, rec, best, status)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
