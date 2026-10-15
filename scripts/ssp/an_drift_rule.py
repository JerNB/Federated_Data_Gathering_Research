"""H16 (offline, pre-registered): can the log's own drift speed predict the best half-life, in days?

Statement. How fast the full community's item shares drift apart over time (the "drift half-life") predicts the best
forgetting half-life of the sampled tracker, across very different logs (Amazon about 534 days, Gowalla about 83 days,
KuaiRand a few days).

Drift half-life of a log at a given round length. For round lags L = 1..8 let S(L) be the mean cosine similarity between
the item-count vectors of the whole community in round k and round k+L (rounds with at least 500 rows only). Fit
log S(L) = a - b L by least squares; drift half-life = ln2 / b rounds (infinite if b <= 0), times the round length in days.

Best half-life. For every (log, round length) cell at n = 100 people per round, the grid half-life (0.25, 0.5, 1, 2, 4, 8, inf
rounds) with the highest mean NDCG@100: tracking runs (20, 30 and 60 rounds), tournament calendar runs (45 and 90 rounds)
and fresh-log holdout runs (30 rounds). Cells where the best grid value is inf, or where the drift half-life is infinite, are
left out of the correlations and counted separately.

Pre-registered predictions (written before this script was run):
  P1  Spearman correlation between drift half-life (days) and best half-life (days) over (log, round length) cells is at
      least 0.7, with at least 40 cells.
  P2  Rule 'half-life in days = c x drift half-life in days', c fitted on the other logs only (leave one LOG out, median of
      the ratios best/drift), snapped to the nearest grid value, has lower mean regret (best grid NDCG minus rule NDCG) than
      BOTH fixed alternatives: 2 rounds, and 534 days. Paired one-sided t test over cells, p < 0.05 for each.
  P3  The same holds on the non-Amazon logs alone (mean regret of the drift rule below that of 534 days), because 534 days
      is Amazon-specific by construction. Reported with its number of cells; no test is run below 8 cells.
Anything not met is reported as not met. The drift half-life uses the whole community's counts (ceiling for what a server could
estimate from its own samples); an estimate from samples is not tested here.
"""
import csv
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
import track_core as T  # noqa: E402

OUT = "results/small_sample_popularity"
TAB = f"{OUT}/03_tables"
HG = ["0.25", "0.5", "1", "2", "4", "8", "inf"]
HV = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf]
DAYS_FIXED = 534.0


def cos(a, b):
    na, nb = float(np.sqrt(a @ a)), float(np.sqrt(b @ b))
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def drift(name, nw):
    D = T.prepare(name, nw)
    ok = [k for k in range(nw) if D["counts"][k].sum() >= T.MIN_TRUTH_ROWS]
    S = []
    for L in range(1, min(8, nw - 9) + 1):
        v = [cos(D["counts"][k], D["counts"][k + L]) for k in ok if k + L in set(ok)]
        S.append(np.nanmean(v) if v else np.nan)
    S = np.array(S)
    good = np.isfinite(S) & (S > 0)
    if good.sum() < 3:
        return float("inf"), D["window_days"]
    x = np.arange(1, len(S) + 1)[good]
    b = -np.polyfit(x, np.log(S[good]), 1)[0]
    return (float(np.log(2) / b) if b > 1e-9 else float("inf")), D["window_days"]


def grid_means():
    """{(log, n_win): array(7) of mean NDCG per grid half-life} at n = 100."""
    out = {}
    for ln in open(f"{OUT}/02_runs/track.jsonl"):
        r = json.loads(ln)
        if r.get("kind") == "n" and r["n"] == 100:
            out[(r["dataset"], r.get("n_win", 30))] = np.nanmean(np.array(r["truth_ndcg"], float), axis=(0, 2))
    for fn, stages in (("tourney.jsonl", ("cal", "holdout")),):
        for ln in open(f"{OUT}/02_runs/{fn}"):
            r = json.loads(ln)
            st = r["key"].split("|")[0]
            if st in stages and r["n"] == 100 and all(f"decay:h={h}" in r["policies"] for h in HG):
                a = np.array(r["ndcg"], float)
                out[(r["dataset"], r["n_win"])] = np.array([np.nanmean(a[:, r["policies"].index(f"decay:h={h}"), :]) for h in HG])
    return out


def snap(h):
    return int(np.argmin([abs(np.log2(min(x, 16.0)) - np.log2(min(h, 16.0))) for x in HV]))


def main():
    gm = grid_means()
    rows = []
    for (name, nw), m in sorted(gm.items()):
        if not np.isfinite(m).all():
            continue
        tau, wd = drift(name, nw)
        bi = int(np.argmax(m))
        rows.append(dict(log=name, n_win=nw, amazon=name.startswith("amazon-"), window_days=wd, drift_rounds=tau,
                         drift_days=tau * wd if np.isfinite(tau) else float("inf"), best_h=HV[bi],
                         best_days=HV[bi] * wd if np.isfinite(HV[bi]) else float("inf"), m=m))
    use = [r for r in rows if np.isfinite(r["drift_days"]) and np.isfinite(r["best_days"])]
    print(f"cells {len(rows)}; usable for correlation {len(use)}; best=inf {sum(not np.isfinite(r['best_days']) for r in rows)}; "
          f"drift=inf {sum(not np.isfinite(r['drift_days']) for r in rows)}; logs {len({r['log'] for r in use})}")
    rho = stats.spearmanr([r["drift_days"] for r in use], [r["best_days"] for r in use])
    print(f"P1 Spearman drift_days vs best_days = {rho.statistic:.3f} (p={rho.pvalue:.2g}), cells {len(use)}  -> {'MET' if rho.statistic >= 0.7 and len(use) >= 40 else 'NOT MET'}")
    # leave-one-log-out ratio
    logs = sorted({r["log"] for r in use})
    for r in rows:
        r["regret_fixed2"] = float(r["m"].max() - r["m"][3])
        r["regret_534"] = float(r["m"].max() - r["m"][snap(DAYS_FIXED / r["window_days"])])
        oth = [x["best_days"] / x["drift_days"] for x in use if x["log"] != r["log"]]
        if np.isfinite(r["drift_days"]) and oth:
            c = float(np.median(oth))
            r["c"] = c
            r["regret_drift"] = float(r["m"].max() - r["m"][snap(c * r["drift_days"] / r["window_days"])])
        else:
            r["c"] = r["regret_drift"] = float("nan")
    cells = [r for r in rows if np.isfinite(r["regret_drift"])]
    d1 = np.array([r["regret_drift"] - r["regret_fixed2"] for r in cells])
    d2 = np.array([r["regret_drift"] - r["regret_534"] for r in cells])
    t1 = stats.ttest_1samp(d1, 0.0, alternative="less")
    t2 = stats.ttest_1samp(d2, 0.0, alternative="less")
    print(f"P2 cells {len(cells)}: mean regret drift {np.mean([r['regret_drift'] for r in cells]):.4f}  fixed-2-rounds {np.mean([r['regret_fixed2'] for r in cells]):.4f}  "
          f"534-days {np.mean([r['regret_534'] for r in cells]):.4f}")
    print(f"   drift vs fixed-2: mean diff {d1.mean():+.4f} p={t1.pvalue:.3g}; drift vs 534 days: mean diff {d2.mean():+.4f} p={t2.pvalue:.3g}  -> "
          f"{'MET' if t1.pvalue < 0.05 and t2.pvalue < 0.05 else 'NOT MET'}")
    oc = [r for r in cells if not r["amazon"]]
    am = [r for r in cells if r["amazon"]]
    for tag, grp in (("Amazon", am), ("non-Amazon", oc)):
        if grp:
            print(f"   {tag} cells {len(grp)}: regret drift {np.mean([r['regret_drift'] for r in grp]):.4f}  fixed-2 {np.mean([r['regret_fixed2'] for r in grp]):.4f}  534-days {np.mean([r['regret_534'] for r in grp]):.4f}")
    print(f"P3 non-Amazon cells {len(oc)} (no test below 8): drift below 534-days: {np.mean([r['regret_drift'] for r in oc]) < np.mean([r['regret_534'] for r in oc]) if oc else 'n/a'}")
    with open(f"{TAB}/tourney_drift_rule.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["log", "n_win", "amazon", "window_days", "drift_rounds", "drift_days", "best_h_rounds", "best_days", "c_loo", "regret_drift", "regret_fixed2_rounds", "regret_534_days"])
        for r in rows:
            w.writerow([r["log"], r["n_win"], r["amazon"], round(r["window_days"], 2), r["drift_rounds"], r["drift_days"], r["best_h"], r["best_days"],
                        r["c"], r["regret_drift"], r["regret_fixed2"], r["regret_534"]])


if __name__ == "__main__":
    main()
