"""Tracking study analysis. Predictions are FIXED HERE, before the run (see docs/small_sample/06_tracking_design_en.md).

Reads  <runs>/track.jsonl (from run_track.py). Writes <out>/03_tables/an_track_*.csv|json, <out>/04_figures/an_track_*.png

Definitions. Window = 1/30 of a log's time span. n = people per round. A tracker is a decayed sum of per-window shares
with half-life h windows (H = 0.25, 0.5, 1, 2, 4, 8, inf). Score = NDCG@100 of its top-100 against the WHOLE community's
counts in the next window. A step (window w -> w+1, w = 8..28) is VALID for n if every window w-7..w+1 has at least 5n
active people and the target window has >= 500 rows; a (dataset, n) cell needs >= 10 valid steps.
Equivalent age L_eq(s): the lag (windows, linear interpolation) at which the full-community list from L windows
earlier scores s on the same valid steps; '<=0' if s beats the full list of the last window, '>8' if below lag 8.
Tuned h is chosen by two-fold cross-fitting on time (first half of valid steps -> score second half, and back).
Self-tuned h (the server's own rule): before each step choose the h with the smallest mean share error against the
server's own SAMPLED later windows, using only earlier valid steps (>= 3 needed); evaluated from the 4th valid step on.
Strata: Amazon (16 logs) and Other (MovieLens x2, Gowalla, Brightkite, KuaiRand, Last.fm). A claim is "supported" only
when it passes in both strata; if fewer than 3 Other logs are eligible the verdict is "Amazon only".

PRIMARY predictions (verdicts), n = 100 people per round unless stated:
  PA  A tracker with n=100 and the fixed half-life h=2 is worse than a full-community list two windows old:
      L_eq >= 2 (censored '>8' counts) in >= 2/3 of the eligible logs of each stratum.
  PB  Tuning does not matter: cross-fit tuned minus fixed h=2 is NOT (mean >= 0.01 and block-bootstrap 95% lower bound > 0)
      in >= 2/3 of the eligible logs of each stratum.
  PC  Matching a full list one window old needs at least 1% of the active people each round: with n_L1 = smallest n whose
      fixed-h=2 L_eq <= 1, the share n_L1 / median active people (windows 8..29) is >= 1% in >= 2/3 of the logs of each
      stratum for which the share is determined (n_L1 reached inside the grid, or the grid ends at >= 1% of active people
      without reaching it, or the smallest eligible n is below 1% of active people and already reaches it).
  PD  The server's own prequential rule (share error) works: over all cells where cross-fit tuning beats fixed h=2 by
      >= 0.01 (measured on the steps the self-tuned rule is scored on, i.e. from the 4th valid step), the self-tuned score
      minus fixed h=2 is >= half of the tuned gain in >= 2/3 of those cells, per stratum;
      and over all cells the self-tuned score is more than 0.01 below fixed h=2 in <= 10% of cells.
REPORTED, not judged: h* (parabola in log h) and its slope against n; h* vs new-item share of the next head (P2 of the
design); the NDCG-based selector; information ceiling M0 (a list from half the people of the target window scored on the
other half vs lagged full lists); drift profile M1 (noise-corrected squared share change V(k) and TV(k) floor-corrected,
log-log slopes); people-per-round table (n_L1, and n_80gap = smallest n whose fixed-h=2 score closes 80% of the gap between
the 1-window-old full list and the best full-community tracker; h* = 16 means 'at the 8 / infinity edge', a proxy).
"""
import argparse
import csv
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

H = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, float("inf")]
I2, IALL, ILAST = 3, 6, 0
FIRST = 8
MIN_STEPS = 10
RNG = np.random.default_rng(20261020)


def stratum(name: str) -> str:
    return "Amazon" if name.startswith("amazon") else "Other"


def load(path, nwin=30):
    full, cells = {}, {}
    for ln in open(path):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if "error" in r or r.get("n_win", 30) != nwin:
            continue
        if r.get("kind") == "full":
            full[r["dataset"]] = r
        elif r.get("kind") == "n":
            cells[(r["dataset"], r["n"])] = r
    return full, cells


def p10_ok(active, n):
    """Pre-registered eligibility: n <= 20% of the 10th-percentile active people per window over windows 8..end."""
    return n <= 0.2 * float(np.percentile(np.asarray(active)[FIRST:], 10))


def block_ci(x, B=1000, block=3):
    x = np.asarray(x, float)
    n = len(x)
    if n < 4:
        return float("nan"), float("nan")
    nb = int(np.ceil(n / block))
    means = []
    for _ in range(B):
        starts = RNG.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        means.append(x[idx].mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def eq_age(score, f):
    """f: mean full-list score by lag 0..8 (decreasing). Returns (L, flag) flag in '', '<=0', '>8'."""
    f = np.minimum.accumulate(np.asarray(f, float))        # enforce non-increasing
    if score >= f[0]:
        return 0.0, "<=0"
    if score < f[-1]:
        return 9.0, ">8"
    for L in range(1, len(f)):
        if score >= f[L]:
            a, b = f[L - 1], f[L]
            return (L - 1) + (a - score) / (a - b) if a != b else float(L), ""
    return 9.0, ">8"


def hstar(means):
    """Continuous h*: parabola through the best grid point and its neighbours in log2 h; None at the grid edge."""
    k = int(np.argmax(means))
    if k == 0 or k == len(means) - 1:
        return None, ("lo" if k == 0 else "hi")
    grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]         # inf is placed at 16 windows for the fit only
    x = np.log2(grid[k - 1: k + 2])
    y = means[k - 1: k + 2]
    A = np.polyfit(x, y, 2)
    if A[0] >= 0:
        return 2.0 ** x[1], ""
    xs = -A[1] / (2 * A[0])
    return float(2.0 ** np.clip(xs, x[0], x[2])), ""

def cell_stats(dsf, cell):
    n = cell["n"]
    steps = cell["steps"]
    act = np.array(dsf["active"])
    T3 = np.array(cell["truth_ndcg"], float)                 # draws, H, steps
    SM = np.array(cell["self_mse"], float)
    SN = np.array(cell["self_ndcg"], float)
    T = np.nanmean(T3, 0)
    ok = np.array([all(act[w - 7: w + 2] >= 5 * n) for w in steps]) & np.all(np.isfinite(T), 0)
    idx = np.flatnonzero(ok)
    if len(idx) < MIN_STEPS:
        return None
    lag = np.array(dsf["lag_ndcg"], float)[:, idx]
    f = np.nanmean(lag, 1)
    half = len(idx) // 2
    A, B = np.arange(half), np.arange(half, len(idx))
    Ti = T[:, idx]
    hA, hB = int(np.argmax(Ti[:, A].mean(1))), int(np.argmax(Ti[:, B].mean(1)))
    X = np.empty(len(idx))
    X[B] = Ti[hA, B]
    X[A] = Ti[hB, A]
    F2 = Ti[I2]
    d_tuned = X - F2
    lo, hi = block_ci(d_tuned)
    # selectors (per draw, own history)
    sel_mse, sel_nd, sel_h = [], [], []
    for d in range(T3.shape[0]):
        sm, sn, tr = SM[d][:, idx], SN[d][:, idx], T3[d][:, idx]
        a_scores, b_scores, chosen = [], [], []
        for p in range(3, len(idx)):
            hist = np.arange(p)
            ok_h = np.all(np.isfinite(sm[:, hist]), 0)
            if ok_h.sum() < 3:
                a_scores.append(np.nan); b_scores.append(np.nan); chosen.append(np.nan)
                continue
            hh = hist[ok_h]
            c = int(np.argmin(np.mean(sm[:, hh], 1)))
            a_scores.append(tr[c, p]); chosen.append(c)
            okn = np.all(np.isfinite(sn[:, hist]), 0)
            cn = int(np.argmax(np.mean(sn[:, hist[okn]], 1))) if okn.sum() >= 3 else c
            b_scores.append(tr[cn, p])
        sel_mse.append(a_scores); sel_nd.append(b_scores); sel_h.append(chosen)
    sel_mse = np.nanmean(np.array(sel_mse), 0)
    sel_nd = np.nanmean(np.array(sel_nd), 0)
    pos = np.arange(3, len(idx))
    hs, flag = hstar(Ti.mean(1))
    fb = np.nanmean(np.array(dsf["lag_ndcg"], float)[:, idx], 1)
    res = dict(dataset=cell["dataset"], n=n, n_valid_steps=len(idx),
               best_fixed=float(Ti.mean(1).max()), fixed2=float(F2.mean()), last=float(Ti[ILAST].mean()),
               allhist=float(Ti[IALL].mean()), tuned=float(X.mean()), tuned_minus_fixed2=float(d_tuned.mean()),
               tuned_ci_lo=lo, tuned_ci_hi=hi, hstar=hs, hstar_edge=flag,
               argmax_h=H[int(np.argmax(Ti.mean(1)))],
               sel_minus_fixed2=float(np.nanmean(sel_mse - F2[pos])),
               sel_ndcg_minus_fixed2=float(np.nanmean(sel_nd - F2[pos])),
               tuned_gain_on_sel_steps=float(np.mean(X[pos] - F2[pos])),
               sel_mean_h_idx=float(np.nanmean(np.array(sel_h))) if np.isfinite(np.array(sel_h, float)).any() else float("nan"),
               Leq_fixed2=eq_age(float(F2.mean()), fb)[0], Leq_fixed2_flag=eq_age(float(F2.mean()), fb)[1],
               Leq_tuned=eq_age(float(X.mean()), fb)[0], Leq_best=eq_age(float(Ti.mean(1).max()), fb)[0],
               Leq_last=eq_age(float(Ti[ILAST].mean()), fb)[0], full_lag1=float(fb[1]), full_lag0=float(fb[0]),
               full_best_tracker=float(np.nanmean(np.array(dsf["tracker_ndcg"], float)[:, idx], 1).max()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/small_sample_popularity/02_runs/track.jsonl")
    ap.add_argument("--out", default="results/small_sample_popularity")
    ap.add_argument("--nwin", type=int, default=30)
    ap.add_argument("--elig", default="p10", choices=["p10", "steps"])
    a = ap.parse_args()
    suf = "" if (a.nwin == 30 and a.elig == "p10") else f"_w{a.nwin}_{a.elig}"
    full, cells = load(a.runs, a.nwin)
    if a.elig == "p10":
        cells = {k: c for k, c in cells.items() if k[0] in full and p10_ok(full[k[0]]["active"], k[1])}
    rows, skipped = [], []
    for (ds, n), c in sorted(cells.items()):
        if ds not in full:
            continue
        r = cell_stats(full[ds], c)
        if r is None:
            skipped.append((ds, n))
        else:
            rows.append(r)
    os.makedirs(f"{a.out}/03_tables", exist_ok=True)
    os.makedirs(f"{a.out}/04_figures", exist_ok=True)
    with open(f"{a.out}/03_tables/an_track_cells{suf}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by.setdefault(r["dataset"], {})[r["n"]] = r
    med_active = {d: float(np.median(np.array(full[d]["active"])[FIRST:])) for d in full}

    def stratum_vote(passfn, names):
        out = {}
        for s in ("Amazon", "Other"):
            v = [(d, passfn(d)) for d in names if stratum(d) == s]
            v = [(d, p) for d, p in v if p is not None]
            out[s] = dict(passed=sum(1 for _, p in v if p), judged=len(v), datasets=[d for d, p in v if p],
                          failed=[d for d, p in v if not p])
        am, ot = out["Amazon"], out["Other"]
        am_ok = am["judged"] > 0 and am["passed"] >= 2 / 3 * am["judged"]
        ot_ok = ot["judged"] >= 3 and ot["passed"] >= 2 / 3 * ot["judged"]
        if am["judged"] == 0:
            verdict = "not judged"
        elif ot["judged"] < 3:
            verdict = "Amazon only: supported" if am_ok else "Amazon only: not supported"
        else:
            verdict = "supported" if (am_ok and ot_ok) else ("not supported" if not (am_ok or ot_ok) else "mixed")
        return dict(verdict=verdict, **out)

    names = sorted(by)
    # PA
    pa = stratum_vote(lambda d: (by[d][100]["Leq_fixed2"] >= 2.0) if 100 in by[d] else None, names)
    # PB
    def pb_fn(d):
        r = by[d].get(100)
        if r is None:
            return None
        return not (r["tuned_minus_fixed2"] >= 0.01 and r["tuned_ci_lo"] > 0)
    pb = stratum_vote(pb_fn, names)
    # PC: n_L1
    calc = []

    def pc_fn(d):
        ns = sorted(by[d])
        if not ns:
            return None
        ma = med_active[d]
        reach = [n for n in ns if by[d][n]["Leq_fixed2"] <= 1.0]
        n1, status = None, ""
        if reach:
            n1 = reach[0]
            status = "<=smallest" if n1 == ns[0] else "reached"
        else:
            status = ">largest"
        tgt_hit = [n for n in ns if by[d][n]["fixed2"] >= by[d][n]["full_lag1"]
                   + 0.8 * (by[d][n]["full_best_tracker"] - by[d][n]["full_lag1"])]
        n80 = tgt_hit[0] if tgt_hit else None
        calc.append(dict(dataset=d, stratum=stratum(d), n_L1=n1 if n1 else f">{ns[-1]}", status=status,
                         median_active=ma, share=(n1 / ma if n1 else float("nan")),
                         n_80gap=(n80 if n80 else f">{ns[-1]}"), n_80gap_share=(n80 / ma if n80 else float("nan")),
                         largest_n=ns[-1], largest_share=ns[-1] / ma))
        if status == "reached":
            return n1 / ma >= 0.01
        if status == ">largest":
            return True if ns[-1] / ma >= 0.01 else None
        return False if ns[0] / ma < 0.01 else None
    pc = stratum_vote(pc_fn, names)
    # PD
    pd = {}
    for s in ("Amazon", "Other"):
        rs = [r for r in rows if stratum(r["dataset"]) == s]
        tun = [r for r in rs if r["tuned_gain_on_sel_steps"] >= 0.01]
        ok = [r for r in tun if r["sel_minus_fixed2"] >= 0.5 * r["tuned_gain_on_sel_steps"]]
        harm = [r for r in rs if r["sel_minus_fixed2"] < -0.01]
        pd[s] = dict(cells=len(rs), tuning_matters_cells=len(tun), selector_recovers_half=len(ok),
                     frac_recovers=(len(ok) / len(tun) if tun else float("nan")), harmful_cells=len(harm),
                     frac_harm=len(harm) / max(len(rs), 1))
    def pd_ok(x):
        return (x["tuning_matters_cells"] == 0 or x["frac_recovers"] >= 2 / 3) and x["frac_harm"] <= 0.10
    n_other = len({r["dataset"] for r in rows if stratum(r["dataset"]) == "Other"})
    if pd["Amazon"]["cells"] == 0:
        pd_verdict = "not judged"
    elif n_other < 3:
        pd_verdict = "Amazon only: supported" if pd_ok(pd["Amazon"]) else "Amazon only: not supported"
    else:
        pd_verdict = ("supported" if pd_ok(pd["Amazon"]) and pd_ok(pd["Other"]) else
                      "mixed" if pd_ok(pd["Amazon"]) or pd_ok(pd["Other"]) else "not supported")
    # reported: h* vs n slope, h* vs new share
    slopes = {}
    for d in names:
        pts = [(n, r["hstar"]) for n, r in sorted(by[d].items()) if r["hstar"] is not None]
        edge = {n: r["hstar_edge"] for n, r in by[d].items()}
        if len(pts) >= 3:
            sl = float(np.polyfit(np.log2([p[0] for p in pts]), np.log2([p[1] for p in pts]), 1)[0])
        else:
            sl = None
        slopes[d] = dict(slope=sl, n_interior=len(pts), edges=edge)
    sl_vals = {s: [v["slope"] for d, v in slopes.items() if stratum(d) == s and v["slope"] is not None] for s in ("Amazon", "Other")}
    new_sh = {d: float(np.nanmean(full[d]["new_share"])) for d in full}
    p2 = {}
    for s in ("Amazon", "Other", "All"):
        pts = [(new_sh[d], by[d][100]["hstar"]) for d in names if 100 in by[d] and by[d][100]["hstar"] is not None
               and (s == "All" or stratum(d) == s)]
        p2[s] = dict(n=len(pts), spearman=(float(spearmanr([p[0] for p in pts], [p[1] for p in pts])[0]) if len(pts) >= 4 else None))
    # M0, M1
    m01 = []
    for d, r in sorted(full.items()):
        m0h = np.nanmean(r["m0_half"])
        m0l = np.nanmean(np.array(r["m0_lag"], float), 1)
        ks = sorted(int(k) for k in r["V"])
        Vp = [(k, r["V"][str(k)]) for k in ks if r["V"][str(k)] > 0]
        Tp = [(k, r["TV"][str(k)]) for k in ks if r["TV"][str(k)] > 0]
        slope = lambda pts: float(np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)[0]) if len(pts) >= 3 else None
        m01.append(dict(dataset=d, stratum=stratum(d), window_days=r["window_days"], median_active=med_active[d],
                        half_list_score=float(m0h), full_lag0_on_half=float(m0l[0]), ratio_lag0_over_half=float(m0l[0] / m0h) if m0h > 0 else None,
                        V_slope=slope(Vp), TV_slope=slope(Tp),
                        new_share_next_head=new_sh[d], born_share=float(np.nanmean(r["born_share"]))))
    with open(f"{a.out}/03_tables/an_track_m0m1{suf}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(m01[0]))
        w.writeheader()
        w.writerows(m01)
    with open(f"{a.out}/03_tables/an_track_calculator{suf}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(calc[0]))
        w.writeheader()
        w.writerows(sorted(calc, key=lambda x: x["dataset"]))
    verdicts = dict(PA=pa, PB=pb, PC=pc, PD=dict(verdict=pd_verdict, **pd), skipped_cells=skipped,
                    reported=dict(hstar_slope_by_stratum={s: dict(median=(float(np.median(v)) if v else None), n=len(v)) for s, v in sl_vals.items()},
                                  hstar_slopes=slopes, hstar_vs_new_share=p2),
                    n_cells=len(rows), n_datasets=len(by))
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    json.dump(clean(verdicts), open(f"{a.out}/03_tables/an_track_verdicts{suf}.json", "w"), indent=1, allow_nan=False)
    # figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for d in names:
        ns = sorted(by[d])
        c = "0.6" if stratum(d) == "Amazon" else None
        ax[0].plot(ns, [by[d][n]["Leq_fixed2"] for n in ns], "o-", color=c, lw=1, label=None if c else d)
        ax[1].plot(ns, [by[d][n]["tuned_minus_fixed2"] for n in ns], "o-", color=c, lw=1)
    ax[0].set_xscale("log"); ax[1].set_xscale("log")
    ax[0].set_xlabel("people per round n"); ax[0].set_ylabel("equivalent age of the tracker (windows), h=2")
    ax[0].axhline(1, color="k", lw=0.5)
    ax[1].set_xlabel("people per round n"); ax[1].set_ylabel("cross-fit tuned minus fixed h=2 (NDCG@100)")
    ax[1].axhline(0.01, color="k", lw=0.5); ax[1].axhline(0, color="k", lw=0.5)
    ax[0].legend(fontsize=6)
    lab = [d for d in names if 100 in by[d]]
    x = np.arange(len(lab))
    ax[2].bar(x - 0.25, [by[d][100]["fixed2"] for d in lab], 0.25, label="fixed h=2")
    ax[2].bar(x, [by[d][100]["fixed2"] + by[d][100]["tuned_minus_fixed2"] for d in lab], 0.25, label="cross-fit tuned")
    ax[2].bar(x + 0.25, [by[d][100]["fixed2"] + by[d][100]["sel_minus_fixed2"] for d in lab], 0.25, label="self-tuned")
    ax[2].set_xticks(x); ax[2].set_xticklabels([d.replace("amazon-", "a-") for d in lab], rotation=70, ha="right", fontsize=6)
    ax[2].set_ylabel("NDCG@100 at n=100"); ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{a.out}/04_figures/an_track_main{suf}.png", dpi=130)
    print(json.dumps({k: (v if k in ("PD",) else {kk: vv for kk, vv in v.items() if kk in ("verdict",)} if isinstance(v, dict) and "verdict" in v else None)
                      for k, v in verdicts.items() if k in ("PA", "PB", "PC", "PD")}, indent=1, default=str))
    print("cells", len(rows), "datasets", len(by), "skipped", skipped)


if __name__ == "__main__":
    main()
