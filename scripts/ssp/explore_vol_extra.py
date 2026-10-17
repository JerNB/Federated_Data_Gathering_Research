"""Post hoc extras for programme V (NOT registered; every table written here is labelled posthoc_*). Run after an_vol.py has passed its integrity checks:

    .venv/bin/python3 -W ignore scripts/ssp/explore_vol_extra.py [--rows PATH]

Writes into results/small_sample_popularity/03_tables/volunteer/:
  posthoc_chosen_beta.csv    share of the audit selector's picks that fall on each weight exponent (per n and sampler; steady steps, draws, logs)
  posthoc_recovery.csv       how much of the loss caused by a volunteer rule (ref - rep0) the audit-weighted selector recovers (audw - rep0), per n and scope
Self-check: the chosen value recomputed here must equal the registered audw of an_vol.py for every cell (assertion).
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import an_vol as A
import explore_selfsel_weekly as E

OUT = A.OUT
BETAS = A.BETAS
SAMPLERS = A.SAMPLERS


def chosen_betas(r: dict, sampler: str):
    """(share of audit picks per beta [len(BETAS)], value of those picks) over steady steps and draws, replicating tune_weights.rules_of('all')."""
    nd, sa = np.array(r["nd100"], float), np.array(r["selfaud"], float)
    st = [i for i, s in enumerate(r["steps"]) if s >= 8]
    parts, offs, k = [], [], 0
    fb = None
    for b in BETAS:
        suffix = A.scheme_of(sampler, b)
        idx = [i for i, a in enumerate(r["arms"]) if a.partition("@")[2] == suffix and a.startswith("decay:days=")]
        ds = np.array([int(r["arms"][i].partition("@")[0].split("=")[1]) for i in idx])
        order = np.argsort(ds)
        idx, ds = [idx[j] for j in order], ds[order]
        if b == 0.0:
            fb = k + E.nearest(ds, 14)
        parts.append((nd[:, idx, :], sa[:, idx, :]))
        offs.append(k)
        k += len(idx)
    nd_all = np.concatenate([p[0] for p in parts], axis=1)
    sa_all = np.concatenate([p[1] for p in parts], axis=1)
    choice = E.choose(sa_all, st, E.HL, fb)
    val = E.chosen_value(nd_all, choice, st)
    bounds = offs + [k]
    shares = np.zeros(len(BETAS))
    cnt = 0
    for t in st:
        if t not in choice:
            continue
        for a in choice[t]:
            j = int(np.searchsorted(bounds, a, side="right") - 1)
            shares[j] += 1
            cnt += 1
    return shares / max(cnt, 1), val


def ratio_ci(num: dict, den: dict, reps: int = 4000, seed: int = 7):
    ls = sorted(num)
    a, b = np.array([num[l] for l in ls]), np.array([den[l] for l in ls])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ls), (reps, len(ls)))
    rs = a[draws].sum(1) / np.where(b[draws].sum(1) != 0, b[draws].sum(1), np.nan)
    return float(a.sum() / b.sum()), float(np.nanpercentile(rs, 2.5)), float(np.nanpercentile(rs, 97.5)), len(ls)


def main() -> None:
    path = sys.argv[sys.argv.index("--rows") + 1] if "--rows" in sys.argv else f"{A.RUNS}/vol.jsonl"
    rows = A.load_rows(path)
    per = A.build_per(rows)
    body = []
    shares_by = {}
    for r in sorted(rows.values(), key=lambda r: (r["log"], r["n"])):
        for s in SAMPLERS:
            sh, val = chosen_betas(r, s)
            want = per[(r["log"], r["n"])][s]["audw"]
            assert abs(val - want) < 1e-9, (r["key"], s, val, want)
            shares_by.setdefault((r["n"], s), []).append(sh)
    for (n, s), v in sorted(shares_by.items()):
        m = np.mean(v, axis=0)
        body.append([n, s, len(v)] + [round(float(x), 4) for x in m])
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/posthoc_chosen_beta.csv", "w", newline="") as f:
        cw = csv.writer(f, lineterminator="\n")
        cw.writerow(["n", "sampler", "logs"] + [f"share_beta_{b:g}" for b in BETAS])
        cw.writerows(body)
    out = []
    for n in sorted({k[1] for k in per}):
        for scope in ("all", "amazon", "other"):
            for s in ("A+", "A-"):
                keys = [k for k in per if k[1] == n and (scope == "all" or (E.stratum(k[0]) == "amazon") == (scope == "amazon"))]
                num = {k[0]: per[k][s]["audw"] - per[k][s]["rep0"] for k in keys}
                den = {k[0]: per[k][s]["ref"] - per[k][s]["rep0"] for k in keys}
                if len(num) < 3:
                    continue
                share, lo, hi, nl = ratio_ci(num, den)
                out.append([n, scope, s, nl, round(float(np.mean(list(num.values()))), 6), round(float(np.mean(list(den.values()))), 6), round(share, 3), round(lo, 3), round(hi, 3)])
    with open(f"{OUT}/posthoc_recovery.csv", "w", newline="") as f:
        cw = csv.writer(f, lineterminator="\n")
        cw.writerow(["n", "scope", "sampler", "logs", "audw_minus_rep0", "ref_minus_rep0", "recovered_share_ratio_of_means", "ci_lo", "ci_hi"])
        cw.writerows(out)
    print("wrote posthoc_chosen_beta.csv and posthoc_recovery.csv")


if __name__ == "__main__":
    main()
