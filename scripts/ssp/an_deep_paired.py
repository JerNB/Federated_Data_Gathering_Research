"""Paired hypotheses D01-D14 of the deep programme: arm minus reference, per log, on identical reporters.

Rules are those of configs/deep_registry.json `protocol.paired` (fixed before the run):
  * per log: diff = mean over draws of steady-state (steps >= 8) NDCG@100 of the arm minus the reference arm;
  * primary combination: n = 100 and 30 rounds (D10: n = 100 and 90 rounds);
  * discovery logs: one-sided t test of mean diff > 0; Benjamini-Hochberg q = 0.10 across the claims of a gain; selected if mean >= margin;
  * holdout logs: same sign, mean >= margin, one-sided p < 0.05 / (number selected) -> CONFIRMED;
  * plain label from the 95% t interval of the mean over ALL logs (GAIN / SMALL GAIN / NO DIFFERENCE / SMALL LOSS / LOSS / INCONCLUSIVE);
  * prediction check against the registered numeric prediction.
"""
from __future__ import annotations

import csv
import os

import numpy as np
from scipy import stats

MARGIN = 0.01
MIN_LOGS = 6


def _t(v, alt="greater"):
    v = np.asarray(v, float)
    if len(v) < MIN_LOGS:
        return float("nan")
    r = stats.ttest_1samp(v, 0.0, alternative=alt)
    return float(r.pvalue)


def _ci(v, level=0.95):
    v = np.asarray(v, float)
    if len(v) < 3:
        return float("nan"), float("nan")
    se = v.std(ddof=1) / np.sqrt(len(v))
    t = stats.t.ppf(0.5 + level / 2, len(v) - 1)
    return float(v.mean() - t * se), float(v.mean() + t * se)


def label(v):
    v = np.asarray(v, float)
    if len(v) < MIN_LOGS:
        return "NOT TESTABLE"
    lo, hi = _ci(v)
    m = float(v.mean())
    if lo > 0 and m >= MARGIN:
        return "GAIN"
    if lo > 0:
        return "SMALL GAIN"
    if hi < 0 and m <= -MARGIN:
        return "LOSS"
    if hi < 0:
        return "SMALL LOSS"
    if lo > -MARGIN and hi < MARGIN:
        return "NO DIFFERENCE"
    return "INCONCLUSIVE"


def diffs(deep, stage, arm, ref, n, nw, metric="nd100"):
    """{log: diff} for one combination."""
    out = {}
    for r in deep.cells(stage):
        if r["n"] != n or r["n_win"] != nw or not deep.has(r, arm) or not deep.has(r, ref):
            continue
        d = np.nanmean(deep.steady(r, arm, metric) - deep.steady(r, ref, metric))
        if np.isfinite(d):
            out[r["dataset"]] = float(d)
    return out


def bh(pvals, q):
    items = sorted((p, k) for k, p in pvals.items() if np.isfinite(p))
    m, sel = len(items), set()
    kmax = 0
    for i, (p, k) in enumerate(items, 1):
        if p <= q * i / m:
            kmax = i
    for i, (p, k) in enumerate(items, 1):
        if i <= kmax:
            sel.add(k)
    return sel


def run(deep, reg, outdir):
    os.makedirs(outdir, exist_ok=True)
    H = [h for h in reg["hypotheses"] if h["kind"] == "paired"]
    stage_of = {h["id"]: h["stage"] for h in H}
    hyp_of = {h["id"]: h for h in H}
    rows, res = [], {}
    for h in H:
        stage = h["stage"]
        combos = sorted({(r["n"], r["n_win"]) for r in deep.cells(stage)})
        prim = (100, 90) if stage == "seas" else (100, 30)
        info = dict(id=h["id"], primary=h["primary"], ref=h["ref"], combo=prim)
        for role, arm in [("primary", h["primary"])] + [("secondary", a) for a in h["secondary"]]:
            for (n, nw) in combos:
                d = diffs(deep, stage, arm, h["ref"], n, nw)
                if not d:
                    continue
                v = np.array(list(d.values()))
                lo, hi = _ci(v)
                am = [x for k, x in d.items() if deep.stratum(k) == "amazon"]
                ot = [x for k, x in d.items() if deep.stratum(k) == "other"]
                di = [x for k, x in d.items() if deep.split(k) == "discovery"]
                ho = [x for k, x in d.items() if deep.split(k) == "holdout"]
                rows.append(dict(hypothesis=h["id"], arm=arm, role=role, exploratory=(role != "primary" or (n, nw) != prim), ref=h["ref"], n=n, n_win=nw, n_logs=len(v), mean=round(float(v.mean()), 5),
                                 lo=round(lo, 5), hi=round(hi, 5), frac_pos=round(float((v > 0).mean()), 3),
                                 amazon_mean=round(float(np.mean(am)), 5) if am else None, other_mean=round(float(np.mean(ot)), 5) if ot else None,
                                 discovery_mean=round(float(np.mean(di)), 5) if di else None, holdout_mean=round(float(np.mean(ho)), 5) if ho else None,
                                 label=label(v)))
        d0 = diffs(deep, stage, h["primary"], h["ref"], *prim)
        info["diffs"] = d0
        res[h["id"]] = info
    # discovery BH over claims of a gain
    claims = {h["id"]: h for h in H if h["direction"] == "sup"}
    pv = {}
    for i in claims:
        excl = set(next(h for h in H if h["id"] == i).get("exclude_discovery", []))
        dd = [x for k, x in res[i]["diffs"].items() if deep.split(k) == "discovery" and k not in excl]
        pv[i] = _t(dd, "greater")
        res[i]["disc_p"] = pv[i]
        res[i]["disc_mean"] = float(np.mean(dd)) if dd else float("nan")
        res[i]["n_disc"] = len(dd)
    sel = {i for i in bh(pv, reg["bh_q"]) if res[i]["disc_mean"] >= MARGIN}
    clean_groups = set()
    for g in reg.get("clean_set", []):
        v = reg["split"][g]
        clean_groups.update((v["amazon"] + v["other"]) if isinstance(v, dict) else v)
    for i in claims:
        hh = [x for k, x in res[i]["diffs"].items() if deep.split(k) == "holdout"]
        res[i]["hold_mean"] = float(np.mean(hh)) if hh else float("nan")
        res[i]["n_hold"] = len(hh)
        res[i]["hold_p"] = _t(hh, "greater")
        res[i]["selected_discovery"] = i in sel
        alpha = 0.05 / max(len(sel), 1)
        cl = [x for k, x in res[i]["diffs"].items() if k in clean_groups]
        res[i]["clean_mean"] = float(np.mean(cl)) if cl else float("nan")
        res[i]["n_clean"] = len(cl)
        res[i]["confirmed"] = bool(i in sel and res[i]["hold_mean"] >= MARGIN and res[i]["hold_p"] < alpha and cl and res[i]["clean_mean"] > 0)
    # verdicts and predictions
    out = []
    for h in H:
        i = h["id"]
        d0 = res[i]["diffs"]
        v = np.array(list(d0.values())) if d0 else np.array([])
        lab = label(v) if len(v) else "NOT TESTABLE"
        m = float(v.mean()) if len(v) else float("nan")
        pr = h["prediction"]
        sec = {r["arm"]: r for r in rows if r["hypothesis"] == i and r["n"] == res[i]["combo"][0] and r["n_win"] == res[i]["combo"][1]}
        pred_met, why = None, ""
        hv = [x for k, x in d0.items() if deep.split(k) == "holdout"]
        hlab = label(np.array(hv)) if len(hv) >= MIN_LOGS else "NOT TESTABLE"
        lo_, hi_ = _ci(v) if len(v) >= MIN_LOGS else (float("nan"), float("nan"))
        if len(v) < MIN_LOGS:
            pred_met, why = None, f"fewer than {MIN_LOGS} logs at the primary combination"
        elif i == "D01":
            und = sec.get("holt:a=0.3,b=0.1,p=1.0", {}).get("mean")
            pred_met = hi_ < MARGIN and abs(m) <= 0.005 and und is not None and und <= -0.02
            why = f"damped {m:+.4f} (upper bound {hi_:+.4f}); undamped {und}"
        elif i == "D02":
            pred_met = lo_ > -0.003 and hi_ < 0.003
            why = f"mean diff {m:+.4f}, 95% interval {lo_:+.4f} to {hi_:+.4f} (needs inside +/-0.003)"
        elif i == "D07":
            am = [x for k, x in d0.items() if deep.stratum(k) == "amazon"]
            ot = [x for k, x in d0.items() if deep.stratum(k) == "other"]
            pred_met = (np.mean(am) <= 0.005 if am else False) and (np.mean(ot) >= 0.005 if ot else False)
            why = f"Amazon {np.mean(am) if am else float('nan'):+.4f}; other {np.mean(ot) if ot else float('nan'):+.4f}"
        elif i == "D08":
            parts = []
            for sp_ in ("discovery", "holdout"):
                ch_ = _churn_cut(deep, h, sp_)
                dv = np.array([x for k, x in d0.items() if deep.split(k) == sp_])
                lo2 = _ci(dv)[0] if len(dv) >= MIN_LOGS else float("nan")
                parts.append((sp_, ch_, lo2))
            res[i]["churn_cut"] = parts
            pred_met = all(c is not None and c >= 0.30 and np.isfinite(l2) and l2 >= -0.005 for _, c, l2 in parts)
            why = "; ".join(f"{sp_}: churn cut {c if c is None else round(c, 3)}, NDCG lower bound {l2:+.4f}" for sp_, c, l2 in parts) + " (needs cut >= 0.30 and lower bound >= -0.005 on both)"
        elif i == "D10":
            am = [x for k, x in d0.items() if deep.stratum(k) == "amazon"]
            ot = [x for k, x in d0.items() if deep.stratum(k) == "other"]
            hl = [x for k, x in d0.items() if deep.split(k) == "holdout" and deep.stratum(k) == "amazon"]
            pred_met = (np.mean(hl) >= MARGIN if hl else False) and (np.mean(ot) < MARGIN if ot else True)
            why = f"Amazon {np.mean(am) if am else float('nan'):+.4f} (holdout Amazon {np.mean(hl) if hl else float('nan'):+.4f}); other {np.mean(ot) if ot else float('nan'):+.4f}"
        elif i == "D11":
            p10 = diffs(deep, "agg", "decay:h=2@panel:rho=0.1", h["ref"], 100, 30)
            common = [k for k in p10 if k in d0]
            share = float(np.mean([p10[k] < d0[k] for k in common])) if common else float("nan")
            pred_met = hi_ < MARGIN and share >= 0.6
            why = f"rho=0.25 diff {m:+.4f} (upper bound {hi_:+.4f}); rho=0.1 worse than 0.25 in {share:.0%} of logs"
        elif pr.get("gain"):
            pred_met = hlab == "GAIN"
            why = f"holdout label {hlab} (holdout mean {np.mean(hv) if hv else float('nan'):+.4f}); all-log label {lab}"
        elif i in ("D04", "D06", "D09", "D14"):
            pred_met = hi_ < MARGIN and m <= 0.005
            why = f"mean {m:+.4f} (needs <= +0.005), upper 95% bound {hi_:+.4f} (a gain of 0.01 is {'ruled out' if hi_ < MARGIN else 'NOT ruled out'})"
        else:
            pred_met = hi_ < MARGIN
            why = f"mean {m:+.4f}, upper 95% bound {hi_:+.4f} (a gain of 0.01 is {'ruled out' if hi_ < MARGIN else 'NOT ruled out'})"
        lo, hi = _ci(v) if len(v) >= 3 else (float("nan"), float("nan"))
        rr_, pr_ = [], []
        for r in deep.cells(stage_of[i]):
            if r["n"] == res[i]["combo"][0] and r["n_win"] == res[i]["combo"][1] and deep.has(r, hyp_of[i]["primary"]) and deep.has(r, hyp_of[i]["ref"]):
                a1, a2 = r["arms"].index(hyp_of[i]["primary"]), r["arms"].index(hyp_of[i]["ref"])
                if r["rows_per_round"][a2] > 0:
                    rr_.append(r["rows_per_round"][a1] / r["rows_per_round"][a2])
                if r["people_per_round"][a2] > 0:
                    pr_.append(r["people_per_round"][a1] / r["people_per_round"][a2])
        if rr_ and pr_:
            why += f"; rows per round x{np.mean(rr_):.2f} and people x{np.mean(pr_):.2f} of the reference"
        oth = [r for r in rows if r["hypothesis"] == i and r["role"] == "primary" and (r["n"], r["n_win"]) != tuple(res[i]["combo"]) and r["n_logs"] >= 3]
        sgn = 1 if m >= 0 else -1
        same = sum(1 for r in oth if r["mean"] * sgn > 0)
        out.append(dict(id=i, kind="paired", clean_mean=res[i].get("clean_mean"), n_clean=res[i].get("n_clean"), robust_same=same, robust_total=len(oth), label=lab, mean=m, lo=lo, hi=hi, n_logs=len(v), prediction_met=pred_met, why=why,
                        confirmed=res[i].get("confirmed"), selected_discovery=res[i].get("selected_discovery"),
                        disc_mean=res[i].get("disc_mean"), hold_mean=res[i].get("hold_mean"), disc_p=res[i].get("disc_p"), hold_p=res[i].get("hold_p"),
                        frac_pos=float((v > 0).mean()) if len(v) else None,
                        amazon=float(np.mean([x for k, x in d0.items() if deep.stratum(k) == "amazon"])) if any(deep.stratum(k) == "amazon" for k in d0) else None,
                        other=float(np.mean([x for k, x in d0.items() if deep.stratum(k) != "amazon"])) if any(deep.stratum(k) != "amazon" for k in d0) else None))
    with open(f"{outdir}/deep_paired.csv", "w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return out


BASE_ = "decay:h=2"


def _churn_cut(deep, h, split=None):
    """Relative churn reduction of the hysteresis arm vs the baseline at the primary combination (mean over logs)."""
    vals = []
    for r in deep.cells(h["stage"]):
        if r["n"] != 100 or r["n_win"] != 30 or not deep.has(r, h["primary"]) or not deep.has(r, h["ref"]):
            continue
        if split is not None and deep.split(r["dataset"]) != split:
            continue
        a, b = np.nanmean(deep.steady(r, h["primary"], "churn")), np.nanmean(deep.steady(r, h["ref"], "churn"))
        if np.isfinite(a) and np.isfinite(b) and b > 0:
            vals.append(1.0 - a / b)
    return float(np.mean(vals)) if vals else None
