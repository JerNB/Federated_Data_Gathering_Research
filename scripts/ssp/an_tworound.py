"""T1 (exploratory): can a two-round shortlist reduce the number of people needed?

Amendment A16 (after the first run): the first implementation discarded round-1 counts and sampled round 2 without
correcting its within-shortlist inclusion probabilities. That run is retained as ``an_tworound_uncorrected_*`` but is not used
as the primary result. This run uses the corrected estimator below and writes ``an_tworound_amended_*``.

Same people budget and at most k uploaded items per person. N people are drawn without replacement.
  one-round : all N people report k random choices, inverse-probability weighted for user-normalised truth Au.
  two-round : half report k random choices; their weighted counts form shortlist S; the other half report k random
              choices among their choices inside S, with the matching Horvitz-Thompson weight. The final score on S
              is the sum of both halves' weighted counts, so round 1 is not discarded.
Primary metric: overlap@100 with Au, the list where each person contributes total weight one. A = whole-row truth is
also reported. Saving factor = one-round N needed for the same Au overlap divided by two-round N; endpoint cells are
censored rather than treated as exact.

Predictions fixed before the first run:
  T1  at k = 5, corrected two-round beats one-round in >= 60% of (dataset, N) cells with N >= 400 and in at least
      half of the cells for at least half of the datasets; the gain interval must be above zero;
  T2  median saving factor at k = 5 (using endpoint bounds for censored cells) is >= 1.5;
  T3  saving is larger at k = 1 than at k = 5, and larger at k = 5 than at k = 20.
Sensitivity (reported, not judged): shortlist 150 and 500 at k = 5.
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
TAG = "an_tworound_amended"


def report(fr, users, k, rng, allowed=None):
    """Estimate user-normalised truth with exact inclusion-probability weights."""
    starts, lens = fr.c_ptr[users], fr.n_c[users]
    total = int(lens.sum())
    grp = np.repeat(np.arange(len(users)), lens)
    off = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
    idx = np.repeat(starts, lens) + off
    items = fr.c_idx[idx]
    if allowed is not None:
        keep0 = allowed[items]
        items, grp = items[keep0], grp[keep0]
    n_available = np.bincount(grp, minlength=len(users)).astype(np.int64)
    rnd = rng.random(len(items))
    order = np.lexsort((rnd, grp))
    items, grp = items[order], grp[order]
    n_for_row = n_available[grp]
    off2 = np.arange(len(items)) - np.repeat(np.cumsum(n_available) - n_available, n_available)
    keep = off2 < k
    items, grp, n_for_row = items[keep], grp[keep], n_for_row[keep]
    n_original = lens[grp].astype(np.float64)
    n_reported = np.minimum(k, n_for_row).astype(np.float64)
    if allowed is None:
        weights = 1.0 / n_reported
    else:
        weights = n_for_row.astype(np.float64) / (n_reported * n_original)
    return np.bincount(items, weights=weights, minlength=fr.n_items).astype(np.float64)


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
                    sc2 = np.where(allowed, ca + cb, -1.0)
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
    with open(f"{OUT}/03_tables/{TAG}_cells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarise(rows, rng)


def mean_by(rows, **kw):
    v = [r["U"] for r in rows if all(r[a] == b for a, b in kw.items())]
    return float(np.mean(v)) if v else float("nan")


def summarise(rows, _rng=None):
    datasets = sorted({r["dataset"] for r in rows})
    cells = []
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
                seed = 20261018 + sum((i + 1) * ord(ch) for i, ch in enumerate(d)) + 1000 * k + N
                bs_rng = np.random.default_rng(seed)
                bs = np.array([bs_rng.choice(diff, len(diff)).mean() for _ in range(500)])
                if v2 > vs_m[-1]:
                    eq, cens = Ns[-1], True
                elif v2 <= vs_m[0]:
                    eq, cens = Ns[0], True
                else:
                    eq, cens = float(np.exp(np.interp(v2, vs_m, np.log(Ns)))), False
                cells.append(dict(dataset=d, k=k, N=N, one=v1, two=v2, gain=v2 - v1, lo=float(np.percentile(bs, 2.5)),
                                  hi=float(np.percentile(bs, 97.5)), saving=eq / N, saving_censored=cens))
    json_cells = cells
    with open(f"{OUT}/03_tables/{TAG}_summary_cells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0]))
        w.writeheader()
        w.writerows(cells)
    c5 = [c for c in cells if c["k"] == 5 and c["N"] >= 400]
    t1_frac = float(np.mean([c["gain"] > 0 for c in c5])) if c5 else float("nan")
    t1_ds = {d: float(np.mean([c["gain"] > 0 and c["lo"] > 0 for c in c5 if c["dataset"] == d])) for d in datasets if any(c["dataset"] == d for c in c5)}
    t1_half = sum(1 for v in t1_ds.values() if v >= 0.5)
    med = lambda k, include_censored: float(np.median([c["saving"] for c in cells if c["k"] == k and c["N"] >= 400
                                                        and (include_censored or not c["saving_censored"])])) if any(
        c["k"] == k and c["N"] >= 400 and (include_censored or not c["saving_censored"]) for c in cells) else float("nan")
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
        T2=dict(rule="median saving factor at k=5 >= 1.5; censored cells use their endpoint bounds",
                median=med(5, True), median_uncensored=med(5, False),
                censored_cells=sum(c["k"] == 5 and c["N"] >= 400 and c["saving_censored"] for c in cells)),
        T3=dict(rule="saving k=1 > k=5 > k=20; censored cells use endpoint bounds",
                k1=med(1, True), k5=med(5, True), k20=med(20, True),
                uncensored=dict(k1=med(1, False), k5=med(5, False), k20=med(20, False))),
        sensitivity_mean_gain_k5_by_shortlist=sens)
    json.dump(verdict, open(f"{OUT}/03_tables/{TAG}_verdicts.json", "w"), indent=1,
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
    fig.savefig(f"{OUT}/04_figures/{TAG}.png", dpi=130)
    print(json.dumps(verdict, indent=1, default=str))


if __name__ == "__main__":
    main()
