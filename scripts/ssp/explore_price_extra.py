#!/usr/bin/env python3
"""Post hoc extras for programme P (doc 18). NOT registered: nothing here changes a verdict.

Reads results/small_sample_popularity/03_tables/price/p_cells.csv (written by an_price.py) and writes, in the same folder:
  posthoc_price_list.csv            the selector's price (selector value of the stored non-private std cell minus its value under the sampler) by n,
                                    eps and bound: mean over logs with a 95% bootstrap interval over logs (seed 7, 4,000 resamples)
  posthoc_hard_cap_k.csv            is one row per person better than ten? price(cap 1) minus price(cap 10) per (n, eps): negative = cap 1 is better
  posthoc_share_of_gain.csv         which share of the gain of unit-length reports over the 10-row cap does a hard cap of one row already achieve?
  posthoc_best_memory.csv           median best fixed memory (days) per sampler, n and eps: less noise should allow a shorter memory
  posthoc_bound_loss_by_group.csv   the loss of the bound alone (eps = 10^6) by log group (Amazon / other): where does a bound cost most?
Run: .venv/bin/python3 -W ignore scripts/ssp/explore_price_extra.py
"""
from __future__ import annotations

import csv

import numpy as np

OUT = "results/small_sample_popularity/03_tables/price"


def boot_ci(v, seed: int = 7, nb: int = 4000):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(nb)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def write(name, head, body):
    with open(f"{OUT}/{name}", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(head)
        w.writerows(body)
    print("wrote", f"{OUT}/{name}")


def main() -> None:
    rows = list(csv.DictReader(open(f"{OUT}/p_cells.csv")))
    group = {r["log"]: r["group"] for r in rows}
    price = {}                                   # price[(sampler, n)][log]
    best_days = {}
    for r in rows:
        k = (r["sampler"], int(r["n"]))
        price.setdefault(k, {})[r["log"]] = float(r["price_selector"])
        best_days.setdefault(k, {})[r["log"]] = float(r["best_days"])
    ns = sorted({k[1] for k in price})
    out = []
    for n in ns:
        for eps in (2, 4, 8):
            for lever in ("cap10", "clip2", "clip1", "cap1"):
                d = price.get((f"{lever}_e{eps}", n))
                if d:
                    v = np.array(list(d.values()))
                    lo, hi = boot_ci(v)
                    out.append([n, eps, lever, len(v), round(float(v.mean()), 4), round(lo, 4), round(hi, 4)])
        for lever in ("cap10", "clip2", "clip1", "cap1"):
            d = price.get((f"{lever}_ref", n))
            if d:
                v = np.array(list(d.values()))
                lo, hi = boot_ci(v)
                out.append([n, "none", lever, len(v), round(float(v.mean()), 4), round(lo, 4), round(hi, 4)])
    write("posthoc_price_list.csv", ["n", "eps", "bound", "logs", "price_mean", "ci_lo", "ci_hi"], out)

    out = []
    for n in ns:
        for eps in (2, 4, 8):
            a, b = price.get((f"cap1_e{eps}", n)), price.get((f"cap10_e{eps}", n))
            if a and b:
                v = np.array([a[l] - b[l] for l in a if l in b])
                lo, hi = boot_ci(v)
                out.append([n, eps, len(v), round(float(v.mean()), 4), round(lo, 4), round(hi, 4)])
    write("posthoc_hard_cap_k.csv", ["n", "eps", "logs", "price_cap1_minus_price_cap10", "ci_lo", "ci_hi"], out)

    out = []
    for n in ns:
        for eps in (2, 4, 8):
            c1, c10, u1 = price.get((f"cap1_e{eps}", n)), price.get((f"cap10_e{eps}", n)), price.get((f"clip1_e{eps}", n))
            if c1 and c10 and u1:
                ls = [l for l in c10 if l in c1 and l in u1]
                tot = np.mean([c10[l] - u1[l] for l in ls])
                part = np.mean([c10[l] - c1[l] for l in ls])
                out.append([n, eps, len(ls), round(float(tot), 4), round(float(part), 4), round(float(part / tot), 3) if tot > 0 else ""])
    write("posthoc_share_of_gain.csv", ["n", "eps", "logs", "gain_unit_length_over_cap10", "gain_cap1_over_cap10", "share_reached_by_cap1"], out)

    out = []
    for n in ns:
        for eps in (2, 4, 8):
            for lever in ("cap10", "clip1"):
                d = best_days.get((f"{lever}_e{eps}", n))
                if d:
                    out.append([n, eps, lever, len(d), float(np.median(list(d.values())))])
        d = best_days.get(("std", n))
        if d:
            out.append([n, "none", "std", len(d), float(np.median(list(d.values())))])
    write("posthoc_best_memory.csv", ["n", "eps", "sampler", "logs", "median_best_days"], out)

    out = []
    for n in ns:
        for lever in ("cap10", "clip2", "clip1", "cap1"):
            d = price.get((f"{lever}_ref", n))
            if not d:
                continue
            for scope in ("amazon", "other"):
                v = np.array([x for l, x in d.items() if (group[l] == "amazon") == (scope == "amazon")])
                if len(v) >= 3:
                    lo, hi = boot_ci(v)
                    out.append([n, f"{lever}_ref", scope, len(v), round(float(v.mean()), 4), round(lo, 4), round(hi, 4)])
    write("posthoc_bound_loss_by_group.csv", ["n", "sampler", "scope", "logs", "loss_vs_std_selector", "ci_lo", "ci_hi"], out)


if __name__ == "__main__":
    main()
