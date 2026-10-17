"""Post hoc extras for programme V (NOT registered; every table written here is labelled posthoc_*). Run after an_vol.py has passed its integrity checks:

    .venv/bin/python3 -W ignore scripts/ssp/explore_vol_extra.py [--rows PATH]

Writes into results/small_sample_popularity/03_tables/volunteer/:
  posthoc_chosen_beta.csv    share of the audit selector's picks that fall on each weight exponent (per n and sampler; steady steps, draws, logs)
  posthoc_recovery.csv       how much of the loss caused by a volunteer rule (ref - rep0) the audit-weighted selector recovers (audw - rep0), per n and scope
  posthoc_per_log.csv        per-log gains under A+ and A- (mean over the log's cells with n in n_hyp)
  posthoc_leave_one_log_out.csv, posthoc_leave_one_log_out_flips.csv   every verdict recomputed with one log dropped; which single-log removals flip a verdict
  posthoc_rows_per_reporter.csv   mean rows per reporter of the unweighted 14-day arm of each sampler, per n; mean of the per-cell ratios to std and ratio of the means
  posthoc_audit_memory_only.csv   aud0 - rep0 (what the audit's choice of memory adds without weights), log-level mean over n_hyp with a 95% interval
  posthoc_loss_split.csv          per volunteer rule: logs split by the observed sign of the loss (random minus volunteers, uncorrected): mean loss, mean audit-weighted gain (indicative only: selection on the outcome)
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
    logs = sorted({k[0] for k in per})
    nl = A.CR["n_hyp"]
    rows_pl = []
    for L in logs:
        for s in ("A+", "A-"):
            vals = [per[k][s] for k in per if k[0] == L and k[1] in nl]
            if not vals:
                continue
            m = lambda f: float(np.mean([f(v) for v in vals]))
            rows_pl.append([L, E.stratum(L), s, len(vals), round(m(lambda v: v["bestw"] - v["best0"]), 6), round(m(lambda v: v["audw"] - v["rep0"]), 6),
                            round(m(lambda v: v["audw"] - v["aud0"]), 6), round(m(lambda v: v["audw"] - v["ref"]), 6), round(m(lambda v: v["ref"] - v["rep0"]), 6)])
    with open(f"{OUT}/posthoc_per_log.csv", "w", newline="") as f:
        cw = csv.writer(f, lineterminator="\n")
        cw.writerow(["log", "group", "sampler", "cells_n100_300", "bestw_minus_best0", "audw_minus_rep0", "audw_minus_aud0", "audw_minus_ref", "ref_minus_rep0"])
        cw.writerows(rows_pl)
    full = A.verdicts(per)
    ids = [f"V{i}" for i in range(1, 11)]
    loo = []
    flips = {i: [] for i in ids}
    for L in logs:
        sub = A.verdicts({k: v for k, v in per.items() if k[0] != L})
        for i in ids:
            if sub[i]["met"] != full[i]["met"]:
                flips[i].append(L)
        loo.append([L] + [("" if i == "V9" else round(sub[i]["mean"], 6)) for i in ids] + [round(max(sub["V9"]["by_sampler"].values()), 6)] + [str(sub[i]["met"]) for i in ids])
    with open(f"{OUT}/posthoc_leave_one_log_out.csv", "w", newline="") as f:
        cw = csv.writer(f, lineterminator="\n")
        cw.writerow(["dropped_log"] + [f"{i}_mean" for i in ids] + ["V9_max_gap"] + [f"{i}_met" for i in ids])
        cw.writerows(loo)
    with open(f"{OUT}/posthoc_leave_one_log_out_flips.csv", "w", newline="") as f:
        cw = csv.writer(f, lineterminator="\n")
        cw.writerow(["hypothesis", "met_with_all_24_logs", "single_log_removals_that_flip_it", "logs_whose_removal_flips_it"])
        for i in ids:
            cw.writerow([i, str(full[i]["met"]), len(flips[i]), ";".join(flips[i])])
    rpr = {}
    for r in sorted(rows.values(), key=lambda r: (r["log"], r["n"])):
        rp, pp = np.array(r["rows_per_round"], float), np.array(r["people_per_round"], float)
        for s in SAMPLERS:
            arm = "decay:days=14" + ("" if s == "std" else "@" + A.scheme_of(s, 0.0))
            i = r["arms"].index(arm)                           # the unweighted 14-day arm: its rows are real rows
            rpr.setdefault((r["n"], s), {})[r["log"]] = float(rp[i] / pp[i])
    with open(f"{OUT}/posthoc_rows_per_reporter.csv", "w", newline="") as f:
        cw = csv.writer(f, lineterminator="\n")
        cw.writerow(["n", "sampler", "cells", "mean_rows_per_reporter", "mean_of_cell_ratios_to_std", "ratio_of_means_to_std"])
        for (n, s), d in sorted(rpr.items()):
            cw.writerow([n, s, len(d), round(float(np.mean(list(d.values()))), 3), round(float(np.mean([d[l] / rpr[(n, "std")][l] for l in d])), 3),
                         round(float(np.mean(list(d.values())) / np.mean(list(rpr[(n, "std")].values()))), 3)])
    rows_am = []                                       # what the audit's choice of memory alone (no weights) adds: aud0 - rep0, log-level mean over n_hyp
    for s in SAMPLERS:
        d = A.log_means(per, lambda rec, s=s: rec[s]["aud0"] - rec[s]["rep0"], nl)
        st_ = A.stat(d)
        rows_am.append([s, st_["logs"], round(st_["mean"], 6), round(st_["lo"], 6), round(st_["hi"], 6)])
    with open(f"{OUT}/posthoc_audit_memory_only.csv", "w", newline="") as f:
        cw = csv.writer(f, lineterminator="\n")
        cw.writerow(["sampler", "logs", "aud0_minus_rep0_mean", "ci_lo", "ci_hi"])
        cw.writerows(rows_am)
    split = []                                         # heavy volunteers: logs split by the OBSERVED sign of the loss (selection on the outcome: indicative only)
    for s in ("A+", "A-"):
        loss = A.log_means(per, lambda rec, s=s: rec[s]["ref"] - rec[s]["rep0"], nl)
        gain = A.log_means(per, lambda rec, s=s: rec[s]["audw"] - rec[s]["rep0"], nl)
        for label, keep in (("loss > 0", [l for l in loss if loss[l] > 0]), ("loss <= 0", [l for l in loss if loss[l] <= 0])):
            if keep:
                split.append([s, label, len(keep), round(float(np.mean([loss[l] for l in keep])), 6), round(float(np.mean([gain[l] for l in keep])), 6),
                              round(float(np.mean([gain[l] - loss[l] for l in keep])), 6)])
    with open(f"{OUT}/posthoc_loss_split.csv", "w", newline="") as f:
        cw = csv.writer(f, lineterminator="\n")
        cw.writerow(["sampler", "logs_with", "logs", "mean_loss_ref_minus_rep0", "mean_gain_audw_minus_rep0", "mean_audw_minus_ref"])
        cw.writerows(split)
    print("wrote posthoc_chosen_beta.csv, posthoc_recovery.csv, posthoc_per_log.csv, posthoc_leave_one_log_out.csv, posthoc_leave_one_log_out_flips.csv, posthoc_rows_per_reporter.csv, posthoc_audit_memory_only.csv, posthoc_loss_split.csv")


if __name__ == "__main__":
    main()
