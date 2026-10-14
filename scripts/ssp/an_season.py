"""S1 (exploratory, written before running): does last year's list beat last month's list in gift season?

Product-review logs (Amazon 2023, positive = rating >= 4), the WHOLE community's choices.
Target window = 30 days starting at origin t; scored with NDCG@100 and overlap@100 against that window's true counts.
  last   : everyone's choices in the 30 days before t
  yearago: everyone's choices in the same 30 days one year (365 d) earlier than the target window
  blend  : last + yearago, each normalised to sum 1
A window is used only when the target, last and year-ago windows each hold >= 300 choices; from 2 years after the log
starts. "Gift season" = target windows whose midpoint is in November, December or January.
Predictions fixed before the run:
  S1  in gift-season windows, yearago beats last (mean NDCG@100 difference, bootstrap interval over windows above 0)
      in >= 8 of the product datasets with enough windows (>= 6 gift windows);
  S2  in all other windows, the blend beats last by >= +0.01 NDCG@100 (interval above 0) in >= 8 datasets;
  S3  in non-gift windows yearago does NOT beat last (interval above 0) in >= 12 datasets.
Writes <out>/03_tables/an_season_windows.csv, an_season_verdicts.json, <out>/04_figures/an_season.png
"""
import csv
import datetime as dt
import glob
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import metrics as M  # noqa: E402

OUT = "results/small_sample_popularity"
DAY = 86400
WIN = 30 * DAY
MIN_CH = 300


def counts(item, ts, lo, hi, n):
    m = (ts >= lo) & (ts < hi)
    return np.bincount(item[m], minlength=n).astype(np.float64)


def sc(c, truth, rng):
    top = M.top_m(c, 100, rng)
    return M.ndcg_vs_gain(top, truth, 100), M.overlap_at_m(c, truth, 100, rng)


def boot(v, rng, n=2000):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float("nan"), float("nan"), float("nan")
    m = np.array([rng.choice(v, len(v)).mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    rng = np.random.default_rng(20261017)
    rows = []
    for f in sorted(glob.glob("datasets/npz/amazon-*.npz")):
        name = os.path.basename(f)[:-4]
        z = np.load(f)
        pos = z["r"] >= float(z["positive_threshold"])
        item, ts = z["item"][pos].astype(np.int64), z["ts"][pos].astype(np.int64)
        n = int(item.max()) + 1
        t0, t1 = int(ts.min()), int(ts.max())
        start = t0 + 2 * 365 * DAY
        for t in range(start, t1 - WIN, WIN):
            ya = t - 365 * DAY
            truth = counts(item, ts, t, t + WIN, n)
            last = counts(item, ts, t - WIN, t, n)
            yago = counts(item, ts, ya, ya + WIN, n)
            if min(truth.sum(), last.sum(), yago.sum()) < MIN_CH:
                continue
            mid = dt.datetime.fromtimestamp(t + WIN // 2, dt.timezone.utc)
            blend = last / last.sum() + yago / yago.sum()
            r = dict(dataset=name, window_start=dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat(),
                     month=mid.month, gift=int(mid.month in (11, 12, 1)))
            for lab, c in (("last", last), ("yearago", yago), ("blend", blend)):
                a, b = sc(c, truth, rng)
                r[f"{lab}_ndcg"], r[f"{lab}_ov100"] = a, b
            rows.append(r)
        print(name, sum(1 for r in rows if r["dataset"] == name), flush=True)
    os.makedirs(f"{OUT}/03_tables", exist_ok=True)
    with open(f"{OUT}/03_tables/an_season_windows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarise(rows, rng)


def summarise(rows, rng):
    per = {}
    for d in sorted({r["dataset"] for r in rows}):
        R = [r for r in rows if r["dataset"] == d]
        o = {}
        for lab, sel in (("gift", [r for r in R if r["gift"]]), ("other", [r for r in R if not r["gift"]])):
            o[f"n_{lab}"] = len(sel)
            for key, a, b in (("yearago_minus_last", "yearago", "last"), ("blend_minus_last", "blend", "last")):
                m, lo, hi = boot([r[f"{a}_ndcg"] - r[f"{b}_ndcg"] for r in sel], rng)
                o[f"{lab}_{key}"] = dict(mean=m, lo=lo, hi=hi)
        per[d] = o
    ok = lambda o, lab: o[f"n_{lab}"] >= 6
    s1 = [d for d, o in per.items() if ok(o, "gift") and o["gift_yearago_minus_last"]["lo"] > 0]
    s1n = [d for d, o in per.items() if ok(o, "gift")]
    s2 = [d for d, o in per.items() if ok(o, "other") and o["other_blend_minus_last"]["lo"] > 0
          and o["other_blend_minus_last"]["mean"] >= 0.01]
    s2n = [d for d, o in per.items() if ok(o, "other")]
    s3 = [d for d, o in per.items() if ok(o, "other") and not (o["other_yearago_minus_last"]["lo"] > 0)]
    verdict = dict(S1=dict(rule="gift windows: yearago beats last, interval > 0, in >= 8 datasets", met=len(s1), judged=len(s1n), datasets=s1),
                   S2=dict(rule="other windows: blend beats last by >= 0.01, interval > 0, in >= 8", met=len(s2), judged=len(s2n), datasets=s2),
                   S3=dict(rule="other windows: yearago does not beat last in >= 12", met=len(s3), judged=len(s2n)),
                   per_dataset=per)
    json.dump(verdict, open(f"{OUT}/03_tables/an_season_verdicts.json", "w"), indent=1,
              default=lambda x: None if not np.isfinite(x) else x)
    os.makedirs(f"{OUT}/04_figures", exist_ok=True)
    names = [d for d in per if per[d]["n_gift"] >= 1]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(names))
    ax.bar(x - 0.2, [per[n]["gift_yearago_minus_last"]["mean"] for n in names], 0.4, label="gift-season windows (Nov-Jan)")
    ax.bar(x + 0.2, [per[n]["other_yearago_minus_last"]["mean"] for n in names], 0.4, label="all other windows")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("amazon-", "") for n in names], rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("NDCG@100: year-ago list minus last-30-days list")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/04_figures/an_season.png", dpi=130)
    print(json.dumps({k: verdict[k] for k in ("S1", "S2", "S3")}, indent=1))
    for d, o in per.items():
        g, q = o["gift_yearago_minus_last"], o["other_yearago_minus_last"]
        print(f"{d[:34]:34} gift n={o['n_gift']:3d} {g['mean']:+.3f} [{g['lo']:+.3f},{g['hi']:+.3f}] | other n={o['n_other']:3d} {q['mean']:+.3f}")


if __name__ == "__main__":
    main()
