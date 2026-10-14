"""R1 (exploratory, written before running): how old can a list get before a small FRESH sample beats it?

Setting. Timestamped choices of the whole community (collection period only, so every window is fully observed).
At origin t, the target is what people choose in the next window (t, t+W], W = span/30. Lists scored on that target
with NDCG@100 and overlap@100 (truth = counts in the target window over the whole community).
  fresh_full  : everyone's choices in (t-W, t]             (ceiling: ask everybody, just now)
  stale_full(L): everyone's choices in (t-L-W, t-L]        (same window width, only older by L = 1,2,4,8,16 windows)
  alltime     : everyone's choices up to t                 (the plain "most popular ever" list)
  fresh_N     : N random people (N = 100, 1000, 10000), their choices in (t-W, t]
Quantity: L*(N) = smallest lag (in windows, linear interpolation) at which stale_full scores below fresh_N,
read per dataset as days. Origins: 8 evenly spaced between the 35% and 80% points of the span, 30 draws each.

Predictions fixed before the run (no tuning afterwards):
  P1  fresh_N=1000 beats stale_full(L=4 windows) in >= 8 of 11 datasets;
  P2  alltime is worse than fresh_full in >= 9 of 11 datasets (recency is worth having);
  P3  L*(N=100) < L*(N=1000) < L*(N=10000) in every dataset where all three are defined (more people tolerate older lists less).
Writes <out>/03_tables/an_refresh_cells.csv, an_refresh_verdicts.json, <out>/04_figures/an_refresh.png
"""
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

import frame as F  # noqa: E402
import metrics as M  # noqa: E402

DATASETS = ["ml-latest", "ml-1m", "amazon-video-games", "amazon-musical-instruments", "amazon-software",
            "amazon-movies-tv", "amazon-books", "gowalla", "brightkite", "kuairand-pure", "lastfm-1k"]
LAGS = [1, 2, 4, 8, 16]
NS = [100, 1000, 10000]
N_ORIGINS, N_DRAWS = 8, 30
OUT = "results/small_sample_popularity"


def window_counts(idx, ts, lo, hi, n_items, users=None, row_user=None):
    m = (ts > lo) & (ts <= hi)
    if users is not None:
        m &= users[row_user]
    return np.bincount(idx[m], minlength=n_items).astype(np.float64)


def score(c, truth, rng):
    top = M.top_m(c, 100, rng)
    return M.ndcg_vs_gain(top, truth, 100), M.overlap_at_m(c, truth, 100, rng)


def main():
    rng = np.random.default_rng(20261015)
    rows = []
    for d in DATASETS:
        fr = F.load_frame(d)
        if not fr.has_time:
            continue
        ts = fr.c_ts.astype(np.float64)
        idx = fr.c_idx
        row_user = np.repeat(np.arange(fr.n_pop), fr.n_c)
        lo, hi = float(ts.min()), float(ts.max())
        span = hi - lo
        W = span / 30.0
        day = 86400.0
        origins = lo + span * np.linspace(0.35, 0.80, N_ORIGINS)
        origins = origins[origins + W <= hi]
        for t in origins:
            truth = window_counts(idx, ts, t, t + W, fr.n_items)
            if truth.sum() < 100:
                continue
            base = dict(dataset=d, origin=round((t - lo) / span, 3), W_days=round(W / day, 2))
            c_full = window_counts(idx, ts, t - W, t, fr.n_items)
            n, o = score(c_full, truth, rng)
            rows.append({**base, "list": "fresh_full", "N": "", "lag": 0, "ndcg": n, "ov100": o})
            c_all = window_counts(idx, ts, -1, t, fr.n_items)
            n, o = score(c_all, truth, rng)
            rows.append({**base, "list": "alltime", "N": "", "lag": "", "ndcg": n, "ov100": o})
            for L in LAGS:
                if t - L * W - W < lo:
                    continue
                c = window_counts(idx, ts, t - L * W - W, t - L * W, fr.n_items)
                n, o = score(c, truth, rng)
                rows.append({**base, "list": "stale_full", "N": "", "lag": L, "ndcg": n, "ov100": o})
            for N in NS:
                if N > fr.n_pop // 2:
                    continue
                for _ in range(N_DRAWS):
                    pick = np.zeros(fr.n_pop, bool)
                    pick[rng.choice(fr.n_pop, N, replace=False)] = True
                    c = window_counts(idx, ts, t - W, t, fr.n_items, pick, row_user)
                    n, o = score(c, truth, rng)
                    rows.append({**base, "list": "fresh_N", "N": N, "lag": 0, "ndcg": n, "ov100": o})
        print(d, len(rows), flush=True)
        del fr
    os.makedirs(f"{OUT}/03_tables", exist_ok=True)
    with open(f"{OUT}/03_tables/an_refresh_cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarise(rows)


def summarise(rows):
    res, verdict = {}, {}
    for d in DATASETS:
        R = [r for r in rows if r["dataset"] == d]
        if not R:
            continue
        W_days = R[0]["W_days"]

        def mean(lst, N=None, lag=None, key="ndcg"):
            v = [r[key] for r in R if r["list"] == lst and (N is None or r["N"] == N)
                 and (lag is None or r["lag"] == lag)]
            return float(np.mean(v)) if v else float("nan")

        out = dict(W_days=W_days, fresh_full=mean("fresh_full"), alltime=mean("alltime"))
        for L in LAGS:
            out[f"stale_{L}"] = mean("stale_full", lag=L)
        for N in NS:
            out[f"fresh_{N}"] = mean("fresh_N", N=N)
        lags = [0] + LAGS
        stale = [out["fresh_full"]] + [out[f"stale_{L}"] for L in LAGS]
        for N in NS:
            f = out[f"fresh_{N}"]
            star = float("nan")
            if np.isfinite(f):
                for i in range(1, len(lags)):
                    if np.isfinite(stale[i]) and stale[i] < f:
                        a, b = stale[i - 1], stale[i]
                        star = lags[i - 1] + (lags[i] - lags[i - 1]) * (a - f) / (a - b) if a != b else lags[i]
                        break
                else:
                    star = float("inf")
            out[f"Lstar_{N}_windows"] = star
            out[f"Lstar_{N}_days"] = star * W_days
        res[d] = out
    p1 = [d for d, o in res.items() if np.isfinite(o["fresh_1000"]) and np.isfinite(o["stale_4"]) and o["fresh_1000"] > o["stale_4"]]
    p1_n = [d for d, o in res.items() if np.isfinite(o["fresh_1000"]) and np.isfinite(o["stale_4"])]
    p2 = [d for d, o in res.items() if o["alltime"] < o["fresh_full"]]
    p3_ok, p3_n = [], []
    for d, o in res.items():
        v = [o[f"Lstar_{N}_windows"] for N in NS]
        if all(np.isfinite(x) for x in v):
            p3_n.append(d)
            if v[0] < v[1] < v[2]:
                p3_ok.append(d)
    verdict = dict(
        P1=dict(rule="fresh N=1000 beats 4-window-old full list in >= 8 of 11", met=len(p1), judged=len(p1_n), datasets=p1),
        P2=dict(rule="alltime worse than fresh_full in >= 9 of 11", met=len(p2), judged=len(res)),
        P3=dict(rule="L*(100) < L*(1000) < L*(10000) wherever all defined", met=len(p3_ok), judged=len(p3_n)),
        per_dataset=res)
    json.dump(verdict, open(f"{OUT}/03_tables/an_refresh_verdicts.json", "w"), indent=1, default=lambda x: None if not np.isfinite(x) else x)
    os.makedirs(f"{OUT}/04_figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(res)
    x = np.arange(len(names))
    for j, (key, lab) in enumerate([("fresh_full", "everyone, last window"), ("stale_4", "everyone, 4 windows older"),
                                    ("alltime", "everyone, all history"), ("fresh_1000", "1,000 people, last window"),
                                    ("fresh_100", "100 people, last window")]):
        ax.bar(x + (j - 2) * 0.16, [res[n][key] for n in names], 0.16, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylabel("NDCG@100 on the next window")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/04_figures/an_refresh.png", dpi=130)
    print(json.dumps({k: verdict[k] for k in ("P1", "P2", "P3")}, indent=1))


if __name__ == "__main__":
    main()
