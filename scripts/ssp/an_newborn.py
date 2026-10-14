"""R2 (exploratory, written before running): is a stale list stale because of NEW items?

Same rolling-origin setting as an_refresh.py: target = what people choose in (t, t+W], W = span/30, origins at
35%..80% of the span. A list built L = 4 windows ago (everyone's choices in (t-5W, t-4W]) can only contain items
that already existed then. "Newborn" = first choice of the item in the whole log is later than t-4W.
Measured per dataset (mean over origins):
  head_newborn_share : share of the target's top-100 mass (sum of true counts) held by newborn items
  gap_full           : NDCG@100 of the fresh list (everyone, (t-W, t]) minus NDCG@100 of the 4-window-old list, on the full truth
  gap_survivors      : same gap with the truth restricted to items that already existed 4 windows ago
  gap_random         : same gap after removing the same NUMBER of randomly chosen top-100 items (control: is the shrink
                       just because mass was removed?)
Caveat to print with the result: "new" = first seen in the log, not necessarily newly released (matters on ml-1m / ml-latest).
Predictions fixed before the run:
  Q1  removing newborn items shrinks the gap by >= 50% (gap_survivors <= 0.5 gap_full) in >= 8 of 11 datasets, and
      the shrink is larger than the random-removal control in the same datasets;
  Q2  Spearman over datasets between head_newborn_share and staleness (gap_full / fresh score) >= 0.6;
  Q3  head_newborn_share >= 0.40 in >= 4 of the 5 Amazon sets and < 0.10 in gowalla and lastfm-1k.
Writes <out>/03_tables/an_newborn_cells.csv, an_newborn_verdicts.json, <out>/04_figures/an_newborn.png
"""
import csv
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import spearmanr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import frame as F  # noqa: E402
import metrics as M  # noqa: E402

DATASETS = ["ml-latest", "ml-1m", "amazon-video-games", "amazon-musical-instruments", "amazon-software",
            "amazon-movies-tv", "amazon-books", "gowalla", "brightkite", "kuairand-pure", "lastfm-1k"]
AMAZON = [d for d in DATASETS if d.startswith("amazon")]
OUT = "results/small_sample_popularity"
LAG = 4
N_ORIGINS = 8


def wc(idx, ts, lo, hi, n_items):
    m = (ts > lo) & (ts <= hi)
    return np.bincount(idx[m], minlength=n_items).astype(np.float64)


def ndcg(c, truth, rng):
    return M.ndcg_vs_gain(M.top_m(c, 100, rng), truth, 100)


def main():
    rng = np.random.default_rng(20261016)
    rows = []
    for d in DATASETS:
        fr = F.load_frame(d)
        ts = fr.c_ts.astype(np.float64)
        idx = fr.c_idx
        lo, hi = float(ts.min()), float(ts.max())
        span = hi - lo
        W = span / 30.0
        first = np.full(fr.n_items, np.inf)
        np.minimum.at(first, idx, ts)
        for t in lo + span * np.linspace(0.35, 0.80, N_ORIGINS):
            if t + W > hi or t - (LAG + 1) * W < lo:
                continue
            truth = wc(idx, ts, t, t + W, fr.n_items)
            if truth.sum() < 100:
                continue
            fresh = wc(idx, ts, t - W, t, fr.n_items)
            old = wc(idx, ts, t - LAG * W - W, t - LAG * W, fr.n_items)
            alive = first <= t - LAG * W            # could be in the old list
            head = M.top_m(truth, 100, rng)
            head_mass = truth[head].sum()
            newborn_head = head[~alive[head]]
            share = float(truth[newborn_head].sum() / head_mass)
            gap_full = ndcg(fresh, truth, rng) - ndcg(old, truth, rng)
            surv = truth * alive
            gap_surv = ndcg(fresh, surv, rng) - ndcg(old, surv, rng) if surv.sum() > 0 else float("nan")
            gr = []
            for _ in range(5):
                drop = rng.choice(head, len(newborn_head), replace=False)
                tr = truth.copy()
                tr[drop] = 0
                gr.append(ndcg(fresh, tr, rng) - ndcg(old, tr, rng))
            rows.append(dict(dataset=d, origin=round((t - lo) / span, 3), n_newborn_in_head=len(newborn_head),
                             head_newborn_share=share, fresh_score=ndcg(fresh, truth, rng),
                             gap_full=gap_full, gap_survivors=gap_surv, gap_random=float(np.mean(gr))))
        print(d, len(rows), flush=True)
        del fr
    os.makedirs(f"{OUT}/03_tables", exist_ok=True)
    with open(f"{OUT}/03_tables/an_newborn_cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarise(rows)


def summarise(rows):
    per = {}
    for d in DATASETS:
        R = [r for r in rows if r["dataset"] == d]
        if not R:
            continue
        m = lambda k: float(np.nanmean([r[k] for r in R]))
        o = {k: m(k) for k in ("head_newborn_share", "fresh_score", "gap_full", "gap_survivors", "gap_random")}
        o["n_origins"] = len(R)
        o["gap_removed_by_dropping_newborn"] = 1 - o["gap_survivors"] / o["gap_full"] if o["gap_full"] > 0 else float("nan")
        o["gap_removed_by_random_drop"] = 1 - o["gap_random"] / o["gap_full"] if o["gap_full"] > 0 else float("nan")
        o["staleness"] = o["gap_full"] / o["fresh_score"] if o["fresh_score"] > 0 else float("nan")
        per[d] = o
    ok = [d for d, o in per.items() if np.isfinite(o["gap_removed_by_dropping_newborn"])]
    q1 = [d for d in ok if per[d]["gap_removed_by_dropping_newborn"] >= 0.5
          and per[d]["gap_removed_by_dropping_newborn"] > per[d]["gap_removed_by_random_drop"]]
    xs = [per[d]["head_newborn_share"] for d in ok]
    ys = [per[d]["staleness"] for d in ok]
    rho = float(spearmanr(xs, ys)[0]) if len(ok) > 2 else float("nan")
    q3a = [d for d in AMAZON if d in per and per[d]["head_newborn_share"] >= 0.40]
    q3b = {d: per[d]["head_newborn_share"] < 0.10 for d in ("gowalla", "lastfm-1k") if d in per}
    verdict = dict(
        Q1=dict(rule="dropping newborn items removes >= 50% of the gap, more than random drop, in >= 8 of 11",
                met=len(q1), judged=len(ok), datasets=q1),
        Q2=dict(rule="Spearman(head_newborn_share, staleness) >= 0.6", rho=rho, n=len(ok)),
        Q3=dict(rule=">=0.40 in >=4 of 5 Amazon, <0.10 in gowalla and lastfm-1k", amazon_met=len(q3a), low=q3b),
        per_dataset=per)
    json.dump(verdict, open(f"{OUT}/03_tables/an_newborn_verdicts.json", "w"), indent=1,
              default=lambda x: None if not np.isfinite(x) else x)
    os.makedirs(f"{OUT}/04_figures", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    names = list(per)
    ax[0].scatter([per[n]["head_newborn_share"] for n in names], [per[n]["staleness"] for n in names])
    for n in names:
        ax[0].annotate(n.replace("amazon-", "a-"), (per[n]["head_newborn_share"], per[n]["staleness"]), fontsize=7)
    ax[0].set_xlabel("share of next-window head held by new items")
    ax[0].set_ylabel("how much the 4-window-old list loses (fraction)")
    x = np.arange(len(names))
    ax[1].bar(x - 0.2, [per[n]["gap_removed_by_dropping_newborn"] for n in names], 0.4, label="drop new items")
    ax[1].bar(x + 0.2, [per[n]["gap_removed_by_random_drop"] for n in names], 0.4, label="drop random head items (control)")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names, rotation=50, ha="right", fontsize=7)
    ax[1].set_ylabel("share of the gap removed")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/04_figures/an_newborn.png", dpi=130)
    print(json.dumps({k: verdict[k] for k in ("Q1", "Q2", "Q3")}, indent=1))
    for d, o in per.items():
        print(f"{d[:24]:24} newborn_head {o['head_newborn_share']:.2f} stale {o['staleness']:.2f} gap {o['gap_full']:.2f}->{o['gap_survivors']:.2f} (random {o['gap_random']:.2f})")


if __name__ == "__main__":
    main()
