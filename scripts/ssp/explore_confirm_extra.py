"""Post-hoc extras on the confirmation cells (stage K). NOT pre-registered: everything here is exploratory.

    .venv/bin/python3 -W ignore scripts/ssp/explore_confirm_extra.py

Needs 02_runs/confirm.jsonl and 02_runs/weekly.jsonl. Writes results/small_sample_popularity/03_tables/confirm/:
  l_selector_picks.csv      what half-life (in rounds) the selector settles on (steps >= 40, all draws) next to the hindsight-best fixed
                            half-life and the tables' choice, medians over cells by round length, group and n
  l_best_arm_stability.csv  how far the hindsight-best fixed half-life of the same log and n moved between the old and the new window
  l_pooled_windows.csv      mean regret of the five rules over the old windows (exploration cells) and the new windows, 7-day and 28-day rounds
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import an_confirm as A
import explore_selfsel_weekly as E
from an_weekly import RUNS, load

OUT = A.OUT
START = 40


def write_csv(name, head, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}", "w", newline="") as f:
        cw = csv.writer(f, lineterminator="\n")
        cw.writerow(head)
        cw.writerows(rows)


def best_days(c) -> int:
    return max((int(a.split("=")[1]) for a in c["val"] if a.startswith("decay:days=")), key=lambda d: (c["val"][f"decay:days={d}"], -d))


def picks(cells2, per):
    rows, recs = [], []
    for (L, n, rd, _), c in sorted(cells2.items()):
        ds, nd, se = E.prep(c)
        ch = E.choose(se, c["st"], E.HL, E.nearest(ds, 2 * rd))
        used = [t for t in c["st"] if t >= START and t in ch]
        if not used:
            continue
        chosen_days = np.concatenate([ds[ch[t]] for t in used])
        rec = per[(L, n, rd)]
        recs.append((rd, rec["kind"], rec["group"], n, float(np.median(chosen_days)) / rd, best_days(c) / rd,
                     rec["r_pool"] if np.isfinite(rec["r_pool"]) else np.nan, rec["r_same"] if np.isfinite(rec["r_same"]) else np.nan))
    for rd in (7, 28):
        for kind in ("w2", "whole"):
            for grp in ("amazon", "other", "all"):
                for n in ["all"] + E.NS:
                    sel = [r for r in recs if r[0] == rd and r[1] == kind and (grp == "all" or r[2] == grp) and (n == "all" or r[3] == n)]
                    if sel:
                        rows.append([rd, kind, grp, n, len(sel)] + [round(float(np.nanmedian([r[i] for r in sel])), 2) if np.isfinite([r[i] for r in sel]).any() else ""
                                                                    for i in (4, 5, 6, 7)])
    write_csv("l_selector_picks.csv", ["round_days", "window", "group", "n", "cells", "selector_median_rounds", "best_fixed_rounds",
                                       "pooled_table_rounds", "same_group_table_rounds"], rows)


def stability(cells2, cells1):
    rows, recs = [], []
    for (L, n, rd, _), c in sorted(cells2.items()):
        c1 = cells1.get((L, n, rd, "std"))
        if c1 is None:
            continue
        d1, d2 = best_days(c1), best_days(c)
        recs.append((rd, E.stratum(L), n, float(np.log2(d2 / d1))))
    for rd in (7, 28):
        for grp in ("amazon", "other", "all"):
            for n in ["all"] + E.NS:
                v = np.array([r[3] for r in recs if r[0] == rd and (grp == "all" or r[1] == grp) and (n == "all" or r[2] == n)])
                if len(v):
                    rows.append([rd, grp, n, len(v), round(float(np.mean(v == 0)), 3), round(float(np.mean(np.abs(v) <= 1)), 3),
                                 round(float(np.mean(np.abs(v) >= 3)), 3), round(float(np.median(v)), 2)])
    write_csv("l_best_arm_stability.csv", ["round_days", "group", "n", "cells", "share_same_best_half_life", "share_within_factor_2",
                                           "share_factor_8_or_more", "median_log2_ratio_new_over_old"], rows)


def pooled(cells2, cells1, kind_of):
    old = {k: v for k, v in cells1.items() if k[3] == "std" and k[2] in E.CADENCES and k[1] in E.NS}
    per_old = A.evaluate(old, cells1, {k[0]: "w2" for k in old}, with_own=False, with_w1=False)
    per_new = A.evaluate(cells2, cells1, kind_of, with_own=False, with_w1=False)
    rows = []
    for label, per in (("old windows (exploration)", per_old), ("new windows", per_new)):
        for rd in (7, 28):
            keys = [k for k in per if k[2] == rd and per[k]["kind"] == "w2"]
            rows.append([label, rd, len(keys)] + [round(float(np.nanmean([per[k][r] for k in keys])), 6) for r in ("two", "pooled", "same", "selector")])
    write_csv("l_pooled_windows.csv", ["windows", "round_days", "cells", "two", "pooled", "same", "selector"], rows)


def main():
    cells2, kind_of = A.load_confirm(f"{RUNS}/confirm.jsonl")
    A.check_complete(cells2)
    cells1, _ = load()
    per = A.evaluate(cells2, cells1, kind_of)
    picks(cells2, per)
    stability(cells2, cells1)
    pooled(cells2, cells1, kind_of)
    print("wrote", OUT, "l_selector_picks.csv l_best_arm_stability.csv l_pooled_windows.csv")


if __name__ == "__main__":
    main()
