#!/usr/bin/env python3
"""Post hoc for programme P (doc 18). NOT registered: nothing here changes a verdict.

How much of the noise-free list survives at shorter cutoffs? The stored rows of price.jsonl (window w2, 7-day rounds, 63 cells) keep, for
every arm, the steady NDCG at cutoffs 10, 30 and 300 and nd100 per step. For each cell (a log at a number of people), each sampler and each
cutoff k the value is the best fixed memory chosen in hindsight AT THAT CUTOFF (the day arms of the row: 7 to 896 days, plus 1,792 at 300
people), taken as the mean over draws of the steady value (steps >= 8; nd100 is averaged over those steps here). "Retained" is the value under
the sampler divided by the value of plain sampling (no bound, no noise); ratio of the means over logs.

Writes, in results/small_sample_popularity/03_tables/price/:
  posthoc_list_length.csv                  n, eps, sampler, k, logs, plain, value, retained, loss        (mean over logs)
  posthoc_list_length_head_vs_tail.csv     per log: retained at the top 10 minus retained at the top 100, mean over logs with a 95% bootstrap
                                           interval over logs (seed 7, 4,000 resamples), for unit-length reports and the 10-row cap
Sampler names follow p_cells.csv: cap10 (10 random rows), cap1 (one random row), clip2 (L2 norm at most 2), clip1 (unit length).
eps "none" is the noise-free reference of the same bound (eps = 10^6).
Needs the raw rows (not committed): results/small_sample_popularity/02_runs/price.jsonl.
Run: .venv/bin/python3 -W ignore scripts/ssp/explore_price_k.py
"""
from __future__ import annotations

import csv
import json

import numpy as np

RUNS = "results/small_sample_popularity/02_runs/price.jsonl"
OUT = "results/small_sample_popularity/03_tables/price"
STEADY = 8
KS = (10, 30, 100, 300)
SUFFIX = {"cap10": "gauss:eps={e},c=10", "cap1": "gauss:eps={e},c=1", "clip2": "gaussn:eps={e},C=2", "clip1": "gaussn:eps={e},C=1"}
EPS = (("2", "2"), ("4", "4"), ("8", "8"), ("none", "1000000"))      # (label, text of eps in the arm name)


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


def arr(v):
    return np.array([[np.nan if x is None else x for x in d] for d in v], float)


def best_values(r) -> dict:
    """{(sampler, eps_label, k): best fixed memory value} for one row; the plain-sampling arms are ('plain', 'none', k)."""
    arms = r["arms"]
    steps = np.array(r["steps"])
    days = sorted({int(a.split("days=")[1].split("@")[0]) for a in arms if a.startswith("decay:days=")})
    st = np.flatnonzero(steps >= STEADY)
    nd100 = np.array(r["nd100"], float)                                   # draw, arm, step
    mets = {10: arr(r["metrics"]["nd10"]), 30: arr(r["metrics"]["nd30"]),
            100: np.nanmean(nd100[:, :, st], axis=2), 300: arr(r["metrics"]["nd300"])}
    out = {}
    variants = [("plain", "none", "")]
    for s, tpl in SUFFIX.items():
        for lab, e in EPS:
            variants.append((s, lab, "@" + tpl.format(e=e)))
    for s, lab, suf in variants:
        idx = [arms.index(f"decay:days={d}{suf}") for d in days if f"decay:days={d}{suf}" in arms]
        if len(idx) != len(days):
            continue
        for k in KS:
            out[(s, lab, k)] = float(np.nanmax(np.nanmean(mets[k][:, idx], axis=0)))
    return out


def main() -> None:
    cells = {}                                   # cells[n][log] = best_values
    with open(RUNS) as f:
        for line in f:
            r = json.loads(line)
            cells.setdefault(int(r["n"]), {})[r["dataset"]] = best_values(r)
    out, ht = [], []
    for n in sorted(cells):
        logs = cells[n]
        for s in SUFFIX:
            for lab, _ in EPS:
                for k in KS:
                    ls = [l for l, v in logs.items() if np.isfinite(v.get((s, lab, k), np.nan)) and np.isfinite(v.get(("plain", "none", k), np.nan))]
                    if not ls:
                        continue
                    p = float(np.mean([logs[l][("plain", "none", k)] for l in ls]))
                    v = float(np.mean([logs[l][(s, lab, k)] for l in ls]))
                    out.append([n, lab, s, k, len(ls), round(p, 4), round(v, 4), round(v / p, 3), round(p - v, 4)])
        for s in ("clip1", "cap10"):
            for lab in ("2", "4", "8"):
                d = []
                for l, v in logs.items():
                    try:
                        d.append(v[(s, lab, 10)] / v[("plain", "none", 10)] - v[(s, lab, 100)] / v[("plain", "none", 100)])
                    except KeyError:
                        pass
                d = np.array(d)
                d = d[np.isfinite(d)]
                if len(d):
                    lo, hi = boot_ci(d)
                    ht.append([n, lab, s, len(d), round(float(d.mean()), 4), round(lo, 4), round(hi, 4), round(float((d > 0).mean()), 3)])
    write("posthoc_list_length.csv", ["n", "eps", "sampler", "k", "logs", "plain", "value", "retained", "loss"], out)
    write("posthoc_list_length_head_vs_tail.csv", ["n", "eps", "sampler", "logs", "retained10_minus_retained100", "ci_lo", "ci_hi", "share_logs_positive"], ht)


if __name__ == "__main__":
    main()
