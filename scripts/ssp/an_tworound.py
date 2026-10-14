"""T1 (exploratory, written before running): ask in two rounds instead of one.

Same people budget, same message size per person (at most k items). N people are drawn without replacement.
  one-round : all N people each report k random choices; the list is the top 100 by count.
  two-round : half of the N people report k random choices -> shortlist S = top 250 items;
              the other half each report k random choices among their choices INSIDE S;
              the final list ranks the items in S by round-2 counts.
Scored by top-100 overlap with the whole community's list (truth A, all choices) and with the user-normalised list
(truth Au, the one a k-capped sample really targets). Primary = Au overlap@100 (as registered in A9 for k < all).
Saving factor at a cell = (one-round N giving the same overlap, log-linear interpolation) / N; right-censored at the
largest N when the two-round list beats everything on the one-round curve.
Predictions fixed before the run:
  T1  at k = 5, two-round beats one-round in >= 60% of (dataset, N) cells with N >= 400 and a gain interval above 0
      in >= half of the datasets;
  T2  median saving factor at k = 5 is >= 1.5;
  T3  the saving factor is larger at k = 1 than at k = 5, and larger at k = 5 than at k = 20 (median over datasets).
Sensitivity (reported, not judged): shortlist 150 and 500 at k = 5.
Writes <out>/03_tables/an_tworound_cells.csv, an_tworound_verdicts.json, <out>/04_figures/an_tworound.png
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
import est as E  # noqa: E402

OUT = "results/small_sample_popularity"
NS = [100, 200, 400, 800, 1600, 3200, 6400]
KS = [1, 5, 20]
DRAWS = 30
SHORT = 250


def report(fr, users, k, rng, allowed=None):
    """Count vector: each user reports at most k random choices (optionally only items in `allowed`)."""
    starts, lens = fr.c_ptr[users], fr.n_c[users]
    total = int(lens.sum())
    grp = np.repeat(np.arange(len(users)), lens)
    off = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
    idx = np.repeat(starts, lens) + off
    items = fr.c_idx[idx]
    if allowed is not None:
        keep0 = allowed[items]
        items, grp = items[keep0], grp[keep0]
    rnd = rng.random(len(items))
    o = np.lexsort((rnd, grp))
    items, grp = items[o], grp[o]
    cnt = np.bincount(grp, minlength=len(users))
    off2 = np.arange(len(items)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    items = items[off2 < k]
    return np.bincount(items, minlength=fr.n_items).astype(np.float64)


def run_dataset(name, rng):
    fr = F.load_frame(name)
    tA, tU = fr.count_A, E.user_norm_counts(fr)
    out = []
    for k in KS:
        for N in NS:
            if N > fr.n_pop:
                continue
            for sz in ([SHORT] if k != 5 else [150, SHORT, 500]):
                for dr in range(DRAWS):
                    u = rng.choice(fr.n_pop, N, replace=False)
                    c1 = report(fr, u, k, rng)
                    one = dict(U=M.overlap_at_m(c1, tU, 100, rng), A=M.overlap_at_m(c1, tA, 100, rng))
                    h = N // 2
                    ca = report(fr, u[:h], k, rng)
                    S = M.top_m(ca, sz, rng)
                    allowed = np.zeros(fr.n_items, bool)
                    allowed[S] = True
                    cb = report(fr, u[h:], k, rng, allowed)
                    sc2 = np.where(allowed, cb, -1.0)
                    two = dict(U=M.overlap_at_m(sc2, tU, 100, rng), A=M.overlap_at_m(sc2, tA, 100, rng))
                    if sz == SHORT:
                        out.append(dict(dataset=name, k=k, N=N, shortlist=sz, draw=dr, arm="one", **one))
                    out.append(dict(dataset=name, k=k, N=N, shortlist=sz, draw=dr, arm="two", **two))
        print(name, k, flush=True)
    del fr
    return out


def main():
    rng = np.random.default_rng(20261018)
    names = sys.argv[1:] or sorted(os.path.basename(f)[:-4] for f in __import__("glob").glob("datasets/npz/*.npz"))
    rows = []
    for n in names:
        rows += run_dataset(n, rng)
    os.makedirs(f"{OUT}/03_tables", exist_ok=True)
    with open(f"{OUT}/03_tables/an_tworound_cells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarise(rows, rng)


def mean_by(rows, **kw):
    v = [r["U"] for r in rows if all(r[a] == b for a, b in kw.items())]
    return float(np.mean(v)) if v else float("nan")


def summarise(rows, rng):
    datasets = sorted({r["dataset"] for r in rows})
    cells, savings = [], {}
    for d in datasets:
        R = [r for r in rows if r["dataset"] == d]
        for k in KS:
            curve = [(N, mean_by(R, k=k, N=N, shortlist=SHORT, arm="one")) for N in NS]
            curve = [(N, v) for N, v in curve if np.isfinite(v)]
            if len(curve) < 3:
                continue
            Ns, vs = np.array([c[0] for c in curve], float), np.array([c[1] for c in curve])
            vs_m = np.maximum.accumulate(vs)
            for N, v1 in curve:
                v2 = mean_by(R, k=k, N=N, shortlist=SHORT, arm="two")
                if not np.isfinite(v2):
                    continue
                # paired gain with interval over draws
                a = np.array([r["U"] for r in R if r["k"] == k and r["N"] == N and r["shortlist"] == SHORT and r["arm"] == "one"])
                b = np.array([r["U"] for r in R if r["k"] == k and r["N"] == N and r["shortlist"] == SHORT and r["arm"] == "two"])
                diff = b - a
                bs = np.array([rng.choice(diff, len(diff)).mean() for _ in range(500)])
                if v2 > vs_m[-1]:
                    eq, cens = Ns[-1], True
                elif v2 <= vs_m[0]:
                    eq, cens = Ns[0], False
                else:
                    eq, cens = float(np.exp(np.interp(v2, vs_m, np.log(Ns)))), False
                cells.append(dict(dataset=d, k=k, N=N, one=v1, two=v2, gain=v2 - v1, lo=float(np.percentile(bs, 2.5)),
                                  hi=float(np.percentile(bs, 97.5)), saving=eq / N, saving_censored=cens))
    json_cells = cells
    with open(f"{OUT}/03_tables/an_tworound_summary_cells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0]))
        w.writeheader()
        w.writerows(cells)
    c5 = [c for c in cells if c["k"] == 5 and c["N"] >= 400]
    t1_frac = float(np.mean([c["gain"] > 0 for c in c5])) if c5 else float("nan")
    t1_ds = {d: float(np.mean([c["gain"] > 0 and c["lo"] > 0 for c in c5 if c["dataset"] == d])) for d in datasets if any(c["dataset"] == d for c in c5)}
    t1_half = sum(1 for v in t1_ds.values() if v >= 0.5)
    med = lambda k: float(np.median([c["saving"] for c in cells if c["k"] == k and c["N"] >= 400])) if any(c["k"] == k for c in cells) else float("nan")
    sens = {}
    for sz in (150, SHORT, 500):
        g = []
        for d in datasets:
            R = [r for r in rows if r["dataset"] == d and r["k"] == 5]
            for N in NS:
                if N < 400:
                    continue
                b = mean_by(R, N=N, shortlist=sz, arm="two"); a = mean_by(R, N=N, shortlist=SHORT, arm="one")
                if np.isfinite(a) and np.isfinite(b):
                    g.append(b - a)
        sens[str(sz)] = float(np.mean(g)) if g else float("nan")
    verdict = dict(
        T1=dict(rule="k=5: two-round > one-round in >=60% of cells (N>=400) and interval>0 in >=half of cells for >=half of datasets",
                frac_cells_gain_positive=t1_frac, datasets_with_half_cells_significant=t1_half, n_datasets=len(t1_ds), per_dataset=t1_ds),
        T2=dict(rule="median saving factor at k=5 >= 1.5", median=med(5)),
        T3=dict(rule="saving k=1 > k=5 > k=20", k1=med(1), k5=med(5), k20=med(20)),
        sensitivity_mean_gain_k5_by_shortlist=sens)
    json.dump(verdict, open(f"{OUT}/03_tables/an_tworound_verdicts.json", "w"), indent=1,
              default=lambda x: None if not np.isfinite(x) else x)
    os.makedirs(f"{OUT}/04_figures", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for k, col in zip(KS, ("C0", "C1", "C2")):
        xs = NS
        ys = [np.mean([c["gain"] for c in cells if c["k"] == k and c["N"] == N]) for N in xs if any(c["k"] == k and c["N"] == N for c in cells)]
        ax[0].plot(xs[:len(ys)], ys, "o-", color=col, label=f"k={k}")
        ys2 = [np.median([c["saving"] for c in cells if c["k"] == k and c["N"] == N]) for N in xs if any(c["k"] == k and c["N"] == N for c in cells)]
        ax[1].plot(xs[:len(ys2)], ys2, "o-", color=col, label=f"k={k}")
    ax[0].set_xscale("log"); ax[1].set_xscale("log")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_ylabel("two-round minus one-round (overlap@100, mean over datasets)")
    ax[1].axhline(1, color="k", lw=0.6)
    ax[1].set_ylabel("saving factor (median over datasets)")
    for a in ax:
        a.set_xlabel("people N"); a.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/04_figures/an_tworound.png", dpi=130)
    print(json.dumps(verdict, indent=1, default=str))


if __name__ == "__main__":
    main()
