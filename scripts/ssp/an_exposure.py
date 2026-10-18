"""How often is one person drawn? (development analysis for programme L, nothing registered)

The privacy prices of docs 16, 18 and programme L are per round (nominal accounting): a person drawn in k rounds loses more. This script replays the pilot's own draw 0 (same
selection streams as the stored weekly cells: random reporters `std` and heavy volunteers `vpps:a=1`; steady rounds 8 to 59 = 52 weeks) on every log that has a stored weekly cell
at 100 people (stage a, 27 logs), 300 people (stage d, 24 logs) and 1,000 people (stage a, 17 logs), and counts, per person who is drawn at least once, in how many of the 52 rounds
they are drawn. A round with at most n active people draws everybody.

    .venv/bin/python3 -W ignore scripts/ssp/an_exposure.py
Table: results/small_sample_popularity/03_tables/price/laplace_dev_exposure.csv (per n, log and sampler) and the medians over logs on stdout.
"""
from __future__ import annotations

import csv
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import deep_core as C
import tune_lap as TL
import weekly_core as W

OUT = "results/small_sample_popularity/03_tables/price"
DRAW = 0
FIRST, LAST = 8, 60                      # rounds 8..59: 52 steady rounds


def stats(sel_lists):
    _, c = np.unique(np.concatenate(sel_lists), return_counts=True)
    return dict(people=len(c), mean=float(c.mean()), ge2=float((c >= 2).mean()), ge5=float((c >= 5).mean()), ge10=float((c >= 10).mean()), p95=float(np.percentile(c, 95)), max=int(c.max()),
                share_draws_from_ge5=float(c[c >= 5].sum() / c.sum()))


def main():
    cells = TL.stored_cells()
    out = []
    for (log, n), (stage, _) in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        tag = TL.SEED_TAG[stage]
        user, item, ts = W.load_span(log)
        nw = W.N_WEEKS
        win = np.minimum(ts // (7 * 86400), nw - 1).astype(np.int16)
        order = np.argsort(win, kind="stable")
        user, win = user[order], win[order]
        off = np.searchsorted(win, np.arange(nw + 1))
        rr = np.random.default_rng(C.seed_of(log, n, DRAW, nw, tag, "sel"))
        vrng = np.random.default_rng(C.seed_of(log, n, DRAW, nw, tag, "vol", "vpps:a=1"))
        sr_all, sv_all = [], []
        for k in range(nw - 1):
            a, b = off[k], off[k + 1]
            uk, rows = np.unique(user[a:b], return_counts=True)
            sr = uk if len(uk) <= n else rr.choice(uk, n, replace=False)
            if len(uk) <= n:
                sv = uk
            else:
                keys = vrng.random(len(uk)) ** (1.0 / rows.astype(np.float64))
                sv = uk[np.argpartition(-keys, n - 1)[:n]]
            if FIRST <= k < LAST:
                sr_all.append(sr)
                sv_all.append(sv)
        for nm, lst in (("random", sr_all), ("heavy_volunteers", sv_all)):
            out.append(dict(n=n, log=log, sampler=nm, **stats(lst)))
        print(n, log, "done", flush=True)
    head = ["n", "log", "sampler", "people_drawn", "mean_draws", "share_ge2", "share_ge5", "share_ge10", "p95_draws", "max_draws", "share_of_draws_from_people_drawn_ge5"]
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/laplace_dev_exposure.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(head)
        for r in out:
            w.writerow([r["n"], r["log"], r["sampler"], r["people"], round(r["mean"], 3), round(r["ge2"], 4), round(r["ge5"], 4), round(r["ge10"], 4), r["p95"], r["max"], round(r["share_draws_from_ge5"], 4)])
    print("wrote", f"{OUT}/laplace_dev_exposure.csv")
    for n in sorted({r["n"] for r in out}):
        for nm in ("random", "heavy_volunteers"):
            sub = [r for r in out if r["n"] == n and r["sampler"] == nm]
            med = lambda k: float(np.median([r[k] for r in sub]))
            print(n, nm, {k: round(med(k), 3) for k in ("mean", "ge2", "ge5", "ge10", "p95", "max", "share_draws_from_ge5")}, "| largest max", max(r["max"] for r in sub), "logs", len(sub),
                  "| logs with >=5% of drawn people drawn >=5 times:", sum(r["ge5"] >= 0.05 for r in sub))


if __name__ == "__main__":
    main()
