"""Programme G, step 1: tune the parameters of the guarded rules on the DEVELOPMENT windows (w1 = weekly study, w2 = confirmation).

    .venv/bin/python3 -W ignore scripts/ssp/tune_guard.py

Development cells are the stored std cells at 7- and 28-day rounds, n = 30, 100, 300, 1000, of the 27 logs in both windows. For a cell of one window
the priors come from the OTHER window or from OTHER logs only: the own prior is the log's best fixed arm in the other window; the table prior is the
half-life in rounds fitted on the other logs' cells of the other window (same-group when at least 3 logs, else pooled). Writes
results/small_sample_popularity/03_tables/guard/g_dev_grid.csv (mean regret of every rule and parameter by region) and prints the chosen parameters.
The chosen values are then written into configs/selfsel_guard_registry.json BEFORE any window-3 cell is run. This is development, not a test:
the in-sample numbers here are optimistic and are not used as results.
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import an_confirm as AC
import an_weekly as AW
import explore_selfsel_weekly as E
import guard_core as G

OUT = "results/small_sample_popularity/03_tables/guard"
RUNS = AW.RUNS
NS = (30, 100, 300, 1000)
RDS = (7, 28)
DELTAS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08]
WARM_M = [13.0, 38.0, 104.0]
ETAS = [1.0, 3.0, 10.0, 30.0]
ALPHAS = [0.001, 0.01, 0.05, 0.2]
REGIONS = {"n30_7d": lambda n, rd: n == 30 and rd == 7, "n30_28d": lambda n, rd: n == 30 and rd == 28,
           "n100plus_7d": lambda n, rd: n >= 100 and rd == 7, "n100plus_28d": lambda n, rd: n >= 100 and rd == 28}


def load_dev():
    c1, _ = AW.load()
    c2, kind = AC.load_confirm(f"{RUNS}/confirm.jsonl")
    dev = {1: {k: v for k, v in c1.items() if k[3] == "std" and k[1] in NS and k[2] in RDS},
           2: {k: v for k, v in c2.items() if kind[k[0]] == "w2" and k[1] in NS and k[2] in RDS}}
    return dev


def prior_arms(cell, key, other):
    """(own prior by TRUE NDCG, table prior, own prior by SELF-SCORE) as indices into the days-sorted arms of `cell`; None where the other window has no information.

    own (truth) is an oracle: it uses the full community's counts of the other window. own_self uses only the server's own self-scores of the other window."""
    L, n, rd, _ = key
    ds, _, _ = E.prep(cell)
    own = tab = own_self = hs = None
    c_o = other.get((L, n, rd, "std"))
    if c_o is not None:
        b_days = max((int(a.split("=")[1]) for a in c_o["val"] if a.startswith("decay:days=")), key=lambda d: (c_o["val"][f"decay:days={d}"], -d))
        own = E.nearest(ds, b_days)
        ds_o, _, se_o = E.prep(c_o)
        m = np.nanmean(se_o[:, :, c_o["st"]], axis=(0, 2))
        if np.isfinite(m).any():
            own_self = E.nearest(ds, int(ds_o[int(np.nanargmax(m))]))
    logs = sorted({l for (l, n1, rd1, s) in other if n1 == n and rd1 == rd and l != L})
    same = [l for l in logs if E.stratum(l) == E.stratum(L)]
    use = same if len(same) >= E.MIN_GROUP else logs
    if use:
        tab = E.nearest(ds, rd * E.fit_rounds(other, use, n, rd))
    if c_o is not None:
        hs = G.hist_scores([c_o], ds)
    return own, tab, own_self, hs


def window_order(logs):
    """{log: True when window 1 lies entirely before window 2 in time}."""
    import confirm_core as K
    import weekly_core as W
    out = {}
    for log in logs:
        u, i, t = K._positive_events(log)
        st1 = W.densest_start(t, int(t.min()))
        _, st2, _ = K.window_of(t)
        out[log] = bool(st1 + 156 <= st2)
    return out


def evaluate(dev):
    prepped = []
    order = window_order(sorted({k[0] for w in (1, 2) for k in dev[w]}))
    for w in (1, 2):
        other = dev[3 - w]
        for key, c in dev[w].items():
            ds, nd, se = E.prep(c)
            st, b = c["st"], E.best_value(c)
            fb = E.nearest(ds, 2 * key[2])
            own, tab, own_self, hs = prior_arms(c, key, other)
            earlier = bool(order[key[0]]) if w == 2 else bool(not order[key[0]])     # is the window the prior comes from earlier in time than this cell?
            prepped.append((w, key, nd, se, st, b, fb, own, tab, own_self, hs, earlier))
    print("development cells:", len(prepped), flush=True)
    results = {}                                           # rule -> list of (region, regret) ; rule = (name, params)

    cur = {}

    def add(rule, key, reg):
        results.setdefault(rule, []).append((key[1], key[2], reg, cur["earlier"]))
    for (w, key, nd, se, st, b, fb, own, tab, own_self, hs, earlier) in prepped:
        cur["earlier"] = earlier
        add(("plain", ()), key, b - E.chosen_value(nd, E.choose(se, st, E.HL, fb), st))
        add(("two", ()), key, b - E.fixed_value(nd, fb, st))
        if hs is not None:
            for m in WARM_M:
                add(("warm", (m,)), key, b - G.value_of(nd, G.choose_warm(se, st, E.HL, hs, m), st))
        if own_self is not None:
            add(("ownself_only", ()), key, b - E.fixed_value(nd, own_self, st))
            for dl in DELTAS:
                add(("margin_ownself", (dl,)), key, b - G.value_of(nd, G.choose_margin(se, st, E.HL, own_self, dl), st))
        if own is not None:
            add(("own_only", ()), key, b - E.fixed_value(nd, own, st))
            for dl in DELTAS:
                add(("margin_own", (dl,)), key, b - G.value_of(nd, G.choose_margin(se, st, E.HL, own, dl), st))
        if tab is not None:
            add(("table_only", ()), key, b - E.fixed_value(nd, tab, st))
            for dl in DELTAS:
                add(("margin_table", (dl,)), key, b - G.value_of(nd, G.choose_margin(se, st, E.HL, tab, dl), st))
        for eta in ETAS:
            for al in ALPHAS:
                add(("fixedshare", (eta, al)), key, b - G.value_of(nd, G.choose_fixedshare(se, st, eta, al, fb), st))
    return results


def summarize(results):
    table = {}
    for rule, lst in results.items():
        row = {}
        for rg, f in REGIONS.items():
            v = [x[2] for x in lst if f(x[0], x[1])]
            row[rg] = (float(np.mean(v)) if v else float("nan"), len(v))
        table[rule] = row
    return table


def split_by_order(results, rules):
    out = {}
    for rule in rules:
        lst = results.get(rule, [])
        for flag in (True, False):
            for rg, f in REGIONS.items():
                v = [x[2] for x in lst if f(x[0], x[1]) and x[3] == flag]
                out[(rule, flag, rg)] = (float(np.mean(v)) if v else float("nan"), len(v))
    return out


def main():
    dev = load_dev()
    res = evaluate(dev)
    table = summarize(res)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/g_dev_grid.csv", "w", newline="") as f:
        cw = csv.writer(f, lineterminator="\n")
        cw.writerow(["rule", "params"] + [f"{r}_{x}" for r in REGIONS for x in ("mean_regret", "cells")])
        for rule in sorted(table, key=lambda r: (r[0], r[1])):
            cw.writerow([rule[0], "|".join(str(p) for p in rule[1])] + [round(v, 6) if x == 0 else v for r in REGIONS for x, v in enumerate(table[rule][r])])
    base = table[("plain", ())]
    print("plain:", {r: round(v[0], 5) for r, v in base.items()})
    print("two:", {r: round(v[0], 5) for r, v in table[("two", ())].items()})
    print("own_only (oracle, true NDCG of the other window):", {r: round(v[0], 5) for r, v in table[("own_only", ())].items()})
    print("ownself_only (self-scores of the other window):", {r: round(v[0], 5) for r, v in table[("ownself_only", ())].items()})
    print("table_only:", {r: round(v[0], 5) for r, v in table[("table_only", ())].items()})
    for m in WARM_M:
        print(f"warm m={m}:", {r: round(v[0], 5) for r, v in table[("warm", (m,))].items()})
    sp = split_by_order(res, [("plain", ()), ("ownself_only", ()), ("own_only", ()), ("warm", (38.0,))])
    for rule in (("plain", ()), ("ownself_only", ()), ("own_only", ()), ("warm", (38.0,))):
        for flag in (True, False):
            print(f"  split {rule[0]:13s} prior window {'EARLIER' if flag else 'LATER  '} in time:",
                  {rg: (round(sp[(rule, flag, rg)][0], 5), sp[(rule, flag, rg)][1]) for rg in REGIONS})
    # choice rule: minimise the mean of the two n=30 regions subject to no harm at n >= 100 (7-day) of more than 0.0005 against plain
    for name in ("margin_own", "margin_ownself", "margin_table", "fixedshare"):
        cands = [r for r in table if r[0] == name]
        ok = [r for r in cands if table[r]["n100plus_7d"][0] <= base["n100plus_7d"][0] + 0.0005 and table[r]["n100plus_28d"][0] <= base["n100plus_28d"][0] + 0.001]
        pool = ok if ok else cands
        best = min(pool, key=lambda r: (table[r]["n30_7d"][0] + table[r]["n30_28d"][0]) / 2)
        print(name, "chosen", best[1], "(no-harm set non-empty)" if ok else "(NO candidate met the no-harm constraint)",
              {r: round(v[0], 5) for r, v in table[best].items()})


if __name__ == "__main__":
    main()
