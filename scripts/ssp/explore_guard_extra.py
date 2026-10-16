#!/usr/bin/env python3
"""Post hoc extras for programme G (doc 17). NOT registered: nothing here changes a verdict.

Reads results/small_sample_popularity/03_tables/guard/g_cells.csv (regret per cell and rule) and the raw rows, and writes, in the same folder:
  posthoc_warm_by_n.csv        history-seeded selector (warm) minus plain selector by group size and round length: mean over logs, 95% bootstrap
                               interval over logs (seed 7, 4,000 resamples), share of logs where warm is better
  posthoc_warm_by_group.csv    the same for the 18 Amazon logs and the 6 other logs, by region
  posthoc_warm_vs_oracle.csv   fixed arm chosen with TRUE NDCG on the other eras (own_oracle) minus warm, by region (positive: warm is better)
  posthoc_warm_other_log.csv   is the gain from the log's OWN other eras or from a generic prior? 'warm_other' seeds the selector with the mean
                               self-score per arm of OTHER logs (same group if at least 3, else all; same n and round length; their w1 and w2
                               cells), 38 pseudo-rounds as in 'warm'. It needs no truth and no history of the log itself.
Run: .venv/bin/python3 -W ignore scripts/ssp/explore_guard_extra.py
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

OUT = "results/small_sample_popularity/03_tables/guard"


def boot_ci(v: np.ndarray, seed: int = 7, nb: int = 4000):
    rng = np.random.default_rng(seed)
    m = len(v)
    bs = np.array([v[rng.integers(0, m, m)].mean() for _ in range(nb)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def region(n: int, rd: int) -> str:
    return f"{'n30' if n == 30 else 'n100plus'}_{rd}d"


def per_log(rows, a, b, keep):
    d = {}
    for r in rows:
        if not keep(r):
            continue
        x, y = float(r[a]), float(r[b])
        if np.isfinite(x) and np.isfinite(y):
            d.setdefault(r["log"], []).append(x - y)
    return np.array([np.mean(v) for v in d.values()])


def write(name, header, out):
    with open(f"{OUT}/{name}", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(out)
    print("wrote", f"{OUT}/{name}")


def other_log_seed() -> list:
    """warm_other: the selector seeded with the mean self-scores of OTHER logs' history cells (same n, round length; same group if >= 3 logs)."""
    import an_guard as AG
    import explore_selfsel_weekly as E
    import guard_core as G
    rows = AG.load_rows(f"{AG.RUNS}/guard.jsonl")
    c1, c2 = AG.history_cells()
    cells3 = {(r["log"], r["n"], r["round_days"], "std"): AG.view(r) for r in rows.values()}
    per = {}
    for (L, n, rd, _), c in sorted(cells3.items()):
        ds, nd, se = E.prep(c)
        st, b, fb = c["st"], E.best_value(c), E.nearest(ds, 2 * rd)
        rec = dict(plain=b - E.chosen_value(nd, E.choose(se, st, E.HL, fb), st))
        own = [x for x in (c1.get((L, n, rd, "std")), c2.get((L, n, rd, "std"))) if x is not None]
        hs = G.hist_scores(own, ds)
        rec["warm"] = b - G.value_of(nd, G.choose_warm(se, st, E.HL, hs, AG.PAR["warm_m"]), st) if hs is not None else float("nan")
        pairs = [(l, x) for cc in (c1, c2) for (l, n1, rd1, s), x in cc.items() if n1 == n and rd1 == rd and l != L]
        logs = {l for l, _ in pairs}
        same = {l for l in logs if E.stratum(l) == E.stratum(L)}
        use = same if len(same) >= E.MIN_GROUP else logs
        hso = G.hist_scores([x for l, x in pairs if l in use], ds) if use else None
        rec["warm_other"] = b - G.value_of(nd, G.choose_warm(se, st, E.HL, hso, AG.PAR["warm_m"]), st) if hso is not None else float("nan")
        per[(L, n, rd)] = rec
    out = []
    for a, b in (("warm_other", "plain"), ("warm", "plain"), ("warm_other", "warm")):
        for rg, f in AG.REGIONS.items():
            keys = [k for k in per if f(k[1], k[2])]
            p = AG.paired(per, keys, a, b)
            out.append([f"{a} minus {b}", rg, round(p["mean"], 6), round(p["lo"], 6), round(p["hi"], 6), p["logs"], p["dropped_cells"]])
    return out


def main() -> None:
    rows = list(csv.DictReader(open(f"{OUT}/g_cells.csv")))
    out = []
    for rd in (7, 28):
        for n in (30, 100, 300, 1000):
            v = per_log(rows, "warm", "plain", lambda r: int(r["n"]) == n and int(r["round_days"]) == rd)
            if len(v) >= 3:
                lo, hi = boot_ci(v)
                out.append([rd, n, len(v), round(float(v.mean()), 6), round(lo, 6), round(hi, 6), round(float(np.mean(v < 0)), 3)])
    write("posthoc_warm_by_n.csv", ["round_days", "n", "logs", "warm_minus_plain", "ci_lo", "ci_hi", "share_logs_warm_better"], out)

    out = []
    for grp in ("amazon", "other"):
        for rg in ("n30_7d", "n30_28d", "n100plus_7d", "n100plus_28d"):
            v = per_log(rows, "warm", "plain", lambda r: r["group"] == grp and region(int(r["n"]), int(r["round_days"])) == rg)
            if len(v) >= 3:
                lo, hi = boot_ci(v)
                out.append([grp, rg, len(v), round(float(v.mean()), 6), round(lo, 6), round(hi, 6)])
    write("posthoc_warm_by_group.csv", ["group", "region", "logs", "warm_minus_plain", "ci_lo", "ci_hi"], out)

    out = []
    for rg in ("n30_7d", "n30_28d", "n100plus_7d", "n100plus_28d"):
        v = per_log(rows, "own_oracle", "warm", lambda r: region(int(r["n"]), int(r["round_days"])) == rg)
        lo, hi = boot_ci(v)
        out.append([rg, len(v), round(float(v.mean()), 6), round(lo, 6), round(hi, 6)])
    write("posthoc_warm_vs_oracle.csv", ["region", "logs", "own_oracle_minus_warm", "ci_lo", "ci_hi"], out)

    write("posthoc_warm_other_log.csv", ["comparison", "region", "mean_diff", "ci_lo", "ci_hi", "logs", "dropped_cells"], other_log_seed())


if __name__ == "__main__":
    main()
