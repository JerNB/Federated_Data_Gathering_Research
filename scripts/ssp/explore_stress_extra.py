#!/usr/bin/env python3
"""Post hoc extras for the stress programme (doc 16). NOT registered: nothing here changes a verdict.

1. posthoc_bias_best_by_n.csv        best fixed list quality under each volunteer sampler minus under plain random sampling (std), by n and group
                                     (cells of s_cells_bias.csv; paired per (log, n); bootstrap over logs, seed 7, 4,000 resamples).
2. posthoc_bias_population.csv       a mechanism check without sampling noise. If people volunteer with probability proportional to their rows in
                                     the round (A+) and send all their rows, the EXPECTED count of an item (small sample fraction) is proportional to
                                     sum over people of rows_u * rows_{u,i}, not to the row-weighted truth sum_u rows_{u,i}. For A- (probability
                                     proportional to 1 / rows) it is proportional to sum_u rows_{u,i} / rows_u (every person counts once). Ap uses the
                                     person's rows in the previous round. These expected counts are normalised to shares each round, decayed with
                                     half-life h (rounds) and ranked; the NDCG@100 against the next round's full counts is averaged over steady
                                     steps (>= 8). Per log the best h is taken separately for each sampler, as in the cells. The population gain
                                     (sampler minus std) is compared with the gain observed in the cells at n = 300 and n = 1,000.
3. posthoc_bias_population_n.csv     the same check at finite n: the inclusion probability of every person in every round is estimated by
                                     simulating the volunteer draw (Efraimidis-Spirakis keys, 64 draws), and the expected count of an item is
                                     sum over people of (inclusion probability) * (their rows of the item). With n close to the number of active
                                     people the heavy users are almost certainly drawn, which shrinks the bias of the small-fraction limit.
Run: nice -n 10 .venv/bin/python3 -W ignore scripts/ssp/explore_stress_extra.py
"""
from __future__ import annotations

import csv
import os
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

import confirm_core as K

OUT = "results/small_sample_popularity/03_tables/stress"
HL = [1, 2, 4, 8, 16, 32, 64, 128, 256]          # half-lives in rounds (7-day rounds: 7 ... 1,792 days)
STEADY = 8
DISC = 1.0 / np.log2(np.arange(100) + 2.0)
SAMPLERS = ("A+", "Ap", "A-")


def top100(s: np.ndarray) -> np.ndarray:
    idx = np.argpartition(-s, 99)[:100]
    return idx[np.argsort(-s[idx], kind="stable")]


def nd100(s: np.ndarray, g: np.ndarray) -> float:
    ti = top100(g)
    idcg = float(np.sum(np.sort(g[ti])[::-1] * DISC))
    return float(np.sum(g[top100(s)] * DISC)) / idcg if idcg > 0 else 0.0


def population_values(log: str, n: int | None = None, reps: int = 64) -> dict:
    """{sampler: {h: mean nd100 over steady steps}} for the expected-count lists of std, A+, Ap, A-.
    n=None: small-fraction limit (inclusion probability proportional to the weight). n given: inclusion probabilities simulated."""
    D = K.prepare_window(log, 7)
    nw, off, user, item, n_items = D["n_win"], D["off"], D["user"], D["item"], D["n_items"]
    names = ("std",) + SAMPLERS
    state = {(x, h): np.zeros(n_items) for x in names for h in HL}
    fac = {h: 2.0 ** (-1.0 / h) for h in HL}
    acc = {(x, h): [] for x in names for h in HL}
    prev_u, prev_rows = np.empty(0, np.int32), np.empty(0, np.int64)
    for k in range(nw - 1):
        a, b = off[k], off[k + 1]
        u, it = user[a:b], item[a:b]
        if len(u) == 0:
            for key in state:
                state[key] *= fac[key[1]]
            prev_u, prev_rows = np.empty(0, np.int32), np.empty(0, np.int64)
            continue
        uq, inv, rows = np.unique(u, return_inverse=True, return_counts=True)
        wp = np.ones(len(uq))
        if len(prev_u):
            pos = np.searchsorted(prev_u, uq)
            pos[pos >= len(prev_u)] = 0
            wp = np.where(prev_u[pos] == uq, prev_rows[pos], 1).astype(np.float64)
        weights = {"A+": rows.astype(np.float64), "Ap": wp, "A-": 1.0 / rows}
        if n is None:
            pu = weights
        else:
            pu = {}
            rng = np.random.default_rng([7, n, k, zlib.crc32(log.encode())])
            for x in SAMPLERS:
                if len(uq) <= n:
                    pu[x] = np.ones(len(uq))
                else:
                    keys = rng.random((reps, len(uq))) ** (1.0 / weights[x])
                    top = np.argpartition(-keys, n - 1, axis=1)[:, :n]
                    pu[x] = np.bincount(top.ravel(), minlength=len(uq)) / reps
        vec = {"std": np.bincount(it, minlength=n_items).astype(np.float64)}
        for x in SAMPLERS:
            vec[x] = np.bincount(it, weights=pu[x][inv], minlength=n_items)
        share = {x: v / v.sum() for x, v in vec.items()}
        truth = D["counts"][k + 1].astype(np.float64)
        for key, s in state.items():
            s *= fac[key[1]]
            s += share[key[0]]
            if k >= STEADY and truth.sum() > 0:
                acc[key].append(nd100(s, truth))
        prev_u, prev_rows = uq, rows
    return {x: {h: float(np.mean(acc[(x, h)])) if acc[(x, h)] else float("nan") for h in HL} for x in names}


def read_cells(path: str):
    with open(path) as f:
        return list(csv.DictReader(f))


def boot_ci(v: np.ndarray, seed: int = 7, nb: int = 4000):
    rng = np.random.default_rng(seed)
    m = len(v)
    bs = np.array([v[rng.integers(0, m, m)].mean() for _ in range(nb)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def write_csv(name: str, header: list, rows: list) -> None:
    with open(f"{OUT}/{name}", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    print("wrote", f"{OUT}/{name}")


def best_gains(pv: dict) -> dict:
    """per sampler: (best half-life in rounds, best value minus the std best value)."""
    def top(x):
        h = max(pv[x], key=lambda q: pv[x][q] if pv[x][q] == pv[x][q] else -1.0)
        return h, pv[x][h]
    base = top("std")[1]
    return {x: (top(x)[0], top(x)[1] - base) for x in SAMPLERS}, base


def by_n(cells: list) -> None:
    best = {(c["sampler"], c["log"], int(c["n"])): float(c["best"]) for c in cells}
    group = {c["log"]: c["group"] for c in cells}
    rows = []
    for smp in SAMPLERS:
        for scope in ("all", "30", "100", "300", "1000", "amazon", "other"):
            per_log, ncell = {}, 0
            for (s, log, n), v in best.items():
                if s != smp or ("std", log, n) not in best:
                    continue
                if scope in ("30", "100", "300", "1000") and n != int(scope):
                    continue
                if scope in ("amazon", "other") and group[log] != scope:
                    continue
                per_log.setdefault(log, []).append(v - best[("std", log, n)])
                ncell += 1
            if not per_log:
                continue
            logv = np.array([np.mean(x) for x in per_log.values()])
            lo, hi = boot_ci(logv)
            rows.append([smp, scope, len(per_log), ncell, round(float(logv.mean()), 6), round(lo, 6), round(hi, 6)])
    write_csv("posthoc_bias_best_by_n.csv", ["sampler", "scope", "logs", "cells", "mean_best_gain_vs_std", "ci_lo", "ci_hi"], rows)

def summary() -> None:
    """Rank correlations and means of the bias check, read back from the two population tables (no recomputation of the populations)."""
    from scipy.stats import spearmanr
    out = []
    for mode, path, pcols, ccols in (
            ("small_fraction", "posthoc_bias_population.csv", {x: f"{x}_pop_gain" for x in SAMPLERS}, {(x, n): f"{x}_cell_gain_n{n}" for x in SAMPLERS for n in (300, 1000)}),
            ("finite_n", "posthoc_bias_population_n.csv", {x: f"{x}_expected_gain" for x in SAMPLERS}, {(x, n): f"{x}_cell_gain" for x in SAMPLERS for n in (300, 1000)})):
        rows = list(csv.DictReader(open(f"{OUT}/{path}")))
        for x in SAMPLERS:
            for n in (300, 1000):
                pairs = []
                for r in rows:
                    if mode == "finite_n" and int(r["n"]) != n:
                        continue
                    c = r[ccols[(x, n)]]
                    if c != "":
                        pairs.append((float(r[pcols[x]]), float(c)))
                if len(pairs) >= 5:
                    a, b = np.array(pairs).T
                    out.append([mode, x, n, len(pairs), round(float(a.mean()), 6), round(float(b.mean()), 6), round(float(spearmanr(a, b)[0]), 3)])
    write_csv("posthoc_bias_population_summary.csv", ["mode", "sampler", "n", "logs", "mean_expected_gain", "mean_cell_gain", "spearman"], out)
    for row in out:
        print(row)



def main() -> None:
    from scipy.stats import spearmanr
    cells = read_cells(f"{OUT}/s_cells_bias.csv")
    by_n(cells)
    best = {(c["sampler"], c["log"], int(c["n"])): float(c["best"]) for c in cells}
    logs = sorted({c["log"] for c in cells})
    group = {c["log"]: c["group"] for c in cells}

    def cell_gain(x, log, n):
        k1, k0 = (x, log, n), ("std", log, n)
        return round(best[k1] - best[k0], 6) if k1 in best and k0 in best else ""

    rows, rows_n = [], []
    for i, log in enumerate(logs):
        g, base = best_gains(population_values(log))
        row = [log, group[log], round(base, 6)]
        for x in SAMPLERS:
            row += [g[x][0], round(g[x][1], 6)]
        for n in (300, 1000):
            row += [cell_gain(x, log, n) for x in SAMPLERS]
        rows.append(row)
        for n in (300, 1000):
            if ("std", log, n) not in best:
                continue
            gn, _ = best_gains(population_values(log, n=n))
            rows_n.append([log, group[log], n] + [round(gn[x][1], 6) for x in SAMPLERS] + [cell_gain(x, log, n) for x in SAMPLERS])
        print(f"{i + 1}/{len(logs)} {log}: small-fraction gain " + " ".join(f"{x} {row[4 + 2 * j]:+.4f}" for j, x in enumerate(SAMPLERS)), flush=True)
    write_csv("posthoc_bias_population.csv",
              ["log", "group", "population_best_std"] + [f"{x}_{c}" for x in SAMPLERS for c in ("best_h", "pop_gain")]
              + [f"{x}_cell_gain_n{n}" for n in (300, 1000) for x in SAMPLERS], rows)
    write_csv("posthoc_bias_population_n.csv",
              ["log", "group", "n"] + [f"{x}_expected_gain" for x in SAMPLERS] + [f"{x}_cell_gain" for x in SAMPLERS], rows_n)
    for j, x in enumerate(SAMPLERS):
        for n, ccol in ((300, 9 + j), (1000, 12 + j)):
            pairs = [(r[4 + 2 * j], r[ccol]) for r in rows if r[ccol] != ""]
            if len(pairs) >= 5:
                a, b = np.array(pairs).T
                print(f"small-fraction limit  {x} n={n}: logs {len(pairs)}; mean population gain {a.mean():+.4f}; mean cell gain {b.mean():+.4f}; Spearman {spearmanr(a, b)[0]:+.2f}")
    for j, x in enumerate(SAMPLERS):
        for n in (300, 1000):
            pairs = [(r[3 + j], r[6 + j]) for r in rows_n if r[2] == n and r[6 + j] != ""]
            if len(pairs) >= 5:
                a, b = np.array(pairs).T
                print(f"finite n (simulated)  {x} n={n}: logs {len(pairs)}; mean expected gain {a.mean():+.4f}; mean cell gain {b.mean():+.4f}; Spearman {spearmanr(a, b)[0]:+.2f}")


if __name__ == "__main__":
    if "--summary-only" in sys.argv:
        summary()
    else:
        main()
        summary()
