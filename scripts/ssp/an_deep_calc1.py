"""Calculator analyses D15-D23 of the deep programme (configs/deep_registry.json; every rule below is the registry text).

    RUNNERS = {"D15": fn, ...};   fn(deep, reg, outdir) -> dict(id, verdict, numbers, summary_en, summary_zh, tables)

verdict is "MET", "NOT MET" or "NOT TESTABLE" (too few eligible logs or arms missing; the reason is in the summaries and in
numbers["reason"]). A share- or median-based verdict needs at least reg["min_logs"] eligible logs at the deciding cell(s).
Every table is a CSV under outdir. numpy + standard library only. Run `python an_deep_calc1.py [deep.jsonl] [outdir]` for a smoke test.

Conventions shared by all runners
  * "NDCG" = nd100 (or nd10/nd30/nd300 where a runner says so): steady mean (steps >= 8), then mean over draws -> one value per (log, n, arm).
  * A log is "eligible" at an n if the run produced a row for it (the engine only runs cells that pass the 5n-active-people rule).
  * Stages plan and dp are read at their only round count (n_win = 30).
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import warnings

import numpy as np

STEADY = 8
LAGS = [0, 1, 2, 3, 4, 6, 8]
PLAN_GRID = [10, 30, 100, 300, 1000, 3000]
DP_GRID = [1000, 3000, 10000]
HALF_LIVES = ["0.25", "0.5", "1", "2", "4", "8", "inf"]


# ----------------------------------------------------------------------------------------------------- helpers
def _finite(x) -> bool:
    return x is not None and np.isfinite(x)


def _nanmean(a) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def _median(vals) -> float:
    v = np.array([x for x in vals if x is not None and not np.isnan(x)], float)
    return float(np.median(v)) if v.size else float("nan")


def _share(flags) -> float:
    flags = list(flags)
    return float(np.mean(flags)) if flags else float("nan")


def _num(x, nd=4):
    """JSON-safe number: NaN -> "NA", +inf -> "inf"; numpy scalars -> python."""
    if isinstance(x, (str, bool)) or x is None:
        return x
    if isinstance(x, (int, np.integer)):
        return int(x)
    x = float(x)
    if np.isnan(x):
        return "NA"
    if np.isinf(x):
        return "inf" if x > 0 else "-inf"
    return round(x, nd)


def _f(x, nd=3) -> str:
    return "NA" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _pct(x) -> str:
    return "NA" if x is None or not np.isfinite(x) else f"{100 * x:.0f}%"


def _cell(x):
    if isinstance(x, (float, np.floating)):
        return "" if not np.isfinite(x) else f"{x:.6g}"
    return "" if x is None else x


def _write_csv(outdir, name, header, rows) -> str:
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([_cell(x) for x in r])
    return name


def _index(deep, reg, stage) -> dict:
    """{(log, n): row} for the stage at its (single) registered n_win."""
    nw = int(reg["stages"][stage]["n_win"][0])
    out = {}
    for r in deep.cells(stage):
        if stage == "dp":                       # dp is run at several round counts: keep every one as its own "log@n_win" unit
            out[(f"{r['dataset']}@{int(r['n_win'])}", int(r["n"]))] = r
        elif int(r["n_win"]) == nw:
            out[(r["dataset"], int(r["n"]))] = r
    return out


def _val(deep, row, arm, metric="nd100") -> float:
    """Steady mean over draws for one arm of one row; NaN if the arm or metric is missing."""
    if row is None or arm not in row["arms"]:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _nanmean(deep.steady(row, arm, metric))


def _stratum(deep, log) -> str:
    return deep.stratum(str(log).split("@")[0])


def _min_logs(reg) -> int:
    return int(reg.get("min_logs", 6))


def _res(id_, verdict, numbers, en, zh, tables):
    return {"id": id_, "verdict": verdict, "numbers": {k: _num(v) if not isinstance(v, (list, dict)) else v for k, v in numbers.items()},
            "summary_en": en, "summary_zh": zh, "tables": tables}


def _nt(id_, reason_en, reason_zh, numbers, tables):
    numbers = dict(numbers)
    numbers["reason"] = reason_en
    return _res(id_, "NOT TESTABLE", numbers, f"Not testable: {reason_en}.", f"无法检验：{reason_zh}。", tables)


def _hnum(h: str) -> float:
    return float("inf") if h == "inf" else float(h)


# ----------------------------------------------------------------------------------------------------- D15
def run_d15(deep, reg, outdir):
    idx = _index(deep, reg, "plan")
    logs = sorted({k[0] for k in idx})
    base = reg.get("baseline", "decay:h=2")
    by_n, per_log = [], []
    shares, medians, elig = {}, {}, {}
    for m in (4, 8):
        census = f"censold:m={m}"
        be_vals, flags = [], []
        for log in logs:
            vals = {}
            for n in PLAN_GRID:
                row = idx.get((log, n))
                s, c = _val(deep, row, base), _val(deep, row, census)
                if _finite(s) and _finite(c):
                    vals[n] = (s, c)
                    by_n.append([log, _stratum(deep, log), m, n, s, c, s - c, int(s >= c)])
            if 300 not in vals:       # cannot say whether break-even is above 300 without the n=300 cell
                per_log.append([log, _stratum(deep, log), m, "", "", "", "not eligible (no n=300 cell)"])
                continue
            be = next((n for n in PLAN_GRID if n in vals and vals[n][0] >= vals[n][1]), None)
            above = be is None or be > 300
            be_vals.append(float("inf") if be is None else float(be))
            flags.append(above)
            per_log.append([log, _stratum(deep, log), m, "" if be is None else be, int(be is None), int(above), f"largest n seen {max(vals)}"])
        elig[m] = len(be_vals)
        shares[m] = _share(flags)
        medians[m] = _median(be_vals)
    tables = [
        _write_csv(outdir, "D15_by_n.csv", ["log", "stratum", "m", "n", "ndcg_sampled", "ndcg_census", "diff", "sampled_ge_census"], by_n),
        _write_csv(outdir, "D15_breakeven.csv", ["log", "stratum", "m", "breakeven_n", "censored_never", "above_300", "note"], per_log),
    ]

    def med_txt(m):
        return "NA" if not np.isfinite(medians[m]) and medians[m] != np.inf else (">3000" if medians[m] == np.inf else f"{medians[m]:.0f}")

    numbers = {"interpretation": "Census value taken from the same row (same n) as the sampled arm; break-even = smallest n with sampled >= census among the n present; a log is eligible if its n=300 cell exists; censored (never) counts as > 300; median treats censored as +inf. Verdict uses m=4.",
               "eligible_logs": elig[4], "share_above_300_m4": shares[4], "share_above_300_m8": shares[8],
               "median_breakeven_m4": med_txt(4), "median_breakeven_m8": med_txt(8), "threshold_share": 0.7, "min_logs": _min_logs(reg)}
    if elig[4] < _min_logs(reg):
        return _nt("D15", f"only {elig[4]} eligible log(s) with an n=300 cell, at least {_min_logs(reg)} are needed",
                   f"只有 {elig[4]} 个日志有 n=300 的结果，至少需要 {_min_logs(reg)} 个", numbers, tables)
    met = shares[4] >= 0.7
    en = (f"The sampled tracker is compared with a full count taken every 4 rounds (the 'census') in {elig[4]} logs. "
          f"The smallest number of people per round at which the sampled list is at least as good is above 300 (or never reached on the grid up to 3000) in {_pct(shares[4])} of logs; "
          f"the median break-even is {med_txt(4)} people (for a census every 8 rounds: {_pct(shares[8])} above 300, median {med_txt(8)}). "
          f"The prediction needed at least 70% above 300, so it is {'met' if met else 'not met'}.")
    zh = (f"在 {elig[4]} 个日志中，把抽样追踪器与每 4 轮一次的全量统计（普查）比较。抽样榜单第一次不差于普查所需的每轮人数，"
          f"在 {_pct(shares[4])} 的日志里高于 300（或在最多 3000 的网格内始终达不到）；追平所需人数的中位数是 {med_txt(4)}"
          f"（对每 8 轮一次的普查：{_pct(shares[8])} 高于 300，中位数 {med_txt(8)}）。预测要求至少 70% 高于 300，因此{'成立' if met else '不成立'}。")
    return _res("D15", "MET" if met else "NOT MET", numbers, en, zh, tables)


# ----------------------------------------------------------------------------------------------------- D16 / D17
_GAUSS = re.compile(r"^decay:h=2@gauss:eps=([0-9.]+),c=([0-9.]+)$")
_OUE = re.compile(r"^decay:h=2@oue:eps=([0-9.]+)$")


def _dp_tables(deep, reg, outdir, pid, kind, thr, direction):
    """Loss (baseline minus private arm) per log/n/arm; summary per (eps, c, n); smallest n on the grid meeting the criterion."""
    idx = _index(deep, reg, "dp")
    base = reg.get("baseline", "decay:h=2")
    logs = sorted({k[0] for k in idx})
    arms = set()
    for row in idx.values():
        for a in row["arms"]:
            mm = (_GAUSS if kind == "gauss" else _OUE).match(a)
            if mm and float(mm.group(1)) < 1e5:          # eps=1e6 arms are the noise-free capped references, not private arms
                arms.add((a, float(mm.group(1)), float(mm.group(2)) if kind == "gauss" else 0.0))
    arms = sorted(arms, key=lambda t: (t[2], t[1]))
    ok = (lambda loss: loss < thr) if direction == "lt" else (lambda loss: loss > thr)
    per_log, summ, smallest = [], [], []
    loss_by = {}                                   # (arm, n) -> {log: loss}
    for a, eps, c in arms:
        for n in DP_GRID:
            for log in logs:
                row = idx.get((log, n))
                b, p = _val(deep, row, base), _val(deep, row, a)
                if _finite(b) and _finite(p):
                    loss_by.setdefault((a, n), {})[log] = b - p
                    per_log.append([log, _stratum(deep, log), kind, eps, c if kind == "gauss" else "", n, b, p, b - p, int(ok(b - p))])
    for a, eps, c in arms:
        for n in DP_GRID:
            d = loss_by.get((a, n), {})
            v = list(d.values())
            summ.append([kind, eps, c if kind == "gauss" else "", n, len(v), _nanmean(v), _median(v), _share(ok(x) for x in v) if v else float("nan")])
        pernl = []
        for log in logs:
            hit = next((n for n in DP_GRID if log in loss_by.get((a, n), {}) and ok(loss_by[(a, n)][log])), None)
            avail = [n for n in DP_GRID if log in loss_by.get((a, n), {})]
            if avail:
                pernl.append(float("inf") if hit is None else float(hit))
                smallest.append([kind, eps, c if kind == "gauss" else "", log, "" if hit is None else hit, int(hit is None)])
        maj = next((n for n in DP_GRID if len(loss_by.get((a, n), {})) >= _min_logs(reg)
                    and _share(ok(x) for x in loss_by[(a, n)].values()) >= 0.5), None)
        smallest.append([kind, eps, c if kind == "gauss" else "", "MAJORITY (>=50% of eligible logs, needs min_logs)", "" if maj is None else maj, int(maj is None)])
        smallest.append([kind, eps, c if kind == "gauss" else "", "MEDIAN OVER LOGS (inf = never)", _median(pernl), ""])
    ref_rows = []
    for a, eps, c in arms:
        ref = f"decay:h=2@gauss:eps=1000000,c={int(c)}" if kind == "gauss" else "decay:h=2@cap:c=1"
        for n in DP_GRID:
            for log in logs:
                row = idx.get((log, n))
                r0, p = _val(deep, row, ref), _val(deep, row, a)
                if _finite(r0) and _finite(p):
                    ref_rows.append([log, _stratum(deep, log), kind, eps, c if kind == "gauss" else "", n, r0, p, r0 - p])
    _write_csv(outdir, f"{pid}_noise_only_loss.csv", ["log@n_win", "stratum", "mechanism", "eps", "c", "n", "ndcg_noise_free_reference", "ndcg_private", "noise_only_loss"], ref_rows)
    t1 = _write_csv(outdir, f"{pid}_loss_per_log.csv", ["log", "stratum", "mechanism", "eps", "c", "n", "ndcg_baseline", "ndcg_private", "loss", "meets_criterion"], per_log)
    t2 = _write_csv(outdir, f"{pid}_loss_by_eps_c_n.csv", ["mechanism", "eps", "c", "n", "n_logs", "mean_loss", "median_loss", "share_meeting_criterion"], summ)
    t3 = _write_csv(outdir, f"{pid}_smallest_n.csv", ["mechanism", "eps", "c", "log", "smallest_n_meeting_criterion", "never_on_grid"], smallest)
    return loss_by, [t1, t2, t3]


def _cell_share(loss_by, arm, n, ok):
    d = loss_by.get((arm, n), {})
    return len(d), (_share(ok(x) for x in d.values()) if d else float("nan"))


def run_d16(deep, reg, outdir):
    loss_by, tables = _dp_tables(deep, reg, outdir, "D16", "gauss", 0.05, "lt")
    ok = lambda x: x < 0.05
    a2, a8 = "decay:h=2@gauss:eps=2,c=10", "decay:h=2@gauss:eps=8,c=10"
    n2, s2 = _cell_share(loss_by, a2, 10000, ok)
    n8, s8 = _cell_share(loss_by, a8, 1000, ok)
    numbers = {"interpretation": "Primary arms are the c=10 ones (the statement says each person limited to 10 rows); 'loses < 0.05' = baseline NDCG minus private NDCG < 0.05; 'does NOT reach <0.05 in at least 50%' = share of logs with loss < 0.05 is below 50%; shares are over the logs eligible at that n. c=3 arms are in the tables only.",
               "n10000_eps2_c10_eligible_logs": n2, "n10000_eps2_c10_share_loss_lt_0.05": s2,
               "n1000_eps8_c10_eligible_logs": n8, "n1000_eps8_c10_share_loss_lt_0.05": s8, "min_logs": _min_logs(reg)}
    for n in DP_GRID:                    # extra context: share per eps at c=10
        for e in ("1", "2", "4", "8"):
            nn, ss = _cell_share(loss_by, f"decay:h=2@gauss:eps={e},c=10", n, ok)
            numbers[f"n{n}_eps{e}_c10_share_lt_0.05"] = ss
    if n2 < _min_logs(reg) or n8 < _min_logs(reg):
        return _nt("D16", f"eligible logs: {n2} at n=10000 and {n8} at n=1000; at least {_min_logs(reg)} are needed at both",
                   f"n=10000 时有 {n2} 个日志、n=1000 时有 {n8} 个日志可用，两处都至少需要 {_min_logs(reg)} 个", numbers, tables)
    met = s2 >= 0.5 and s8 < 0.5
    en = (f"Central privacy (noise added once to the counts, each person limited to 10 rows) is compared with the same list without noise. "
          f"With 10000 people per round and privacy level eps=2, the loss is below 0.05 NDCG in {_pct(s2)} of {n2} logs; "
          f"with 1000 people per round and eps=8 the loss is below 0.05 in {_pct(s8)} of {n8} logs. "
          f"The prediction needed at least 50% in the first case and fewer than 50% in the second, so it is {'met' if met else 'not met'}.")
    zh = (f"中心化隐私（对计数加一次噪声，每人最多 10 条）与同样但无噪声的榜单比较。"
          f"每轮 10000 人、隐私水平 eps=2 时，{n2} 个日志中有 {_pct(s2)} 的损失低于 0.05 NDCG；每轮 1000 人、eps=8 时，{n8} 个日志中有 {_pct(s8)} 的损失低于 0.05。"
          f"预测要求前者至少 50%、后者不足 50%，因此{'成立' if met else '不成立'}。")
    return _res("D16", "MET" if met else "NOT MET", numbers, en, zh, tables)


def run_d17(deep, reg, outdir):
    loss_by, tables = _dp_tables(deep, reg, outdir, "D17", "oue", 0.1, "gt")
    ok = lambda x: x > 0.1
    a4 = "decay:h=2@oue:eps=4"
    n4, s4 = _cell_share(loss_by, a4, 10000, ok)
    d = loss_by.get((a4, 10000), {})
    numbers = {"interpretation": "Loss = baseline NDCG minus private NDCG; share over logs eligible at n=10000; criterion loss > 0.1 (strict).",
               "n10000_eps4_eligible_logs": n4, "n10000_eps4_share_loss_gt_0.1": s4,
               "n10000_eps4_mean_loss": _nanmean(list(d.values())), "n10000_eps4_median_loss": _median(list(d.values())), "min_logs": _min_logs(reg)}
    for n in DP_GRID:
        for e in ("1", "2", "4", "8"):
            v = list(loss_by.get((f"decay:h=2@oue:eps={e}", n), {}).values())
            numbers[f"n{n}_eps{e}_mean_loss"] = _nanmean(v)
    if n4 < _min_logs(reg):
        return _nt("D17", f"only {n4} eligible log(s) at n=10000, at least {_min_logs(reg)} are needed",
                   f"n=10000 时只有 {n4} 个日志可用，至少需要 {_min_logs(reg)} 个", numbers, tables)
    met = s4 >= 0.5
    en = (f"Local privacy (each person randomises their own report, so the server never sees true counts) is compared with the same list without noise. "
          f"With 10000 people per round and privacy level eps=4, the list loses more than 0.1 NDCG in {_pct(s4)} of {n4} logs "
          f"(mean loss {_f(numbers['n10000_eps4_mean_loss'])}, median {_f(numbers['n10000_eps4_median_loss'])}). "
          f"The prediction needed at least 50% of logs, so it is {'met' if met else 'not met'}.")
    zh = (f"本地隐私（每个人自己对汇报加噪声，服务器看不到真实计数）与同样但无噪声的榜单比较。"
          f"每轮 10000 人、隐私水平 eps=4 时，{n4} 个日志中有 {_pct(s4)} 的榜单损失超过 0.1 NDCG（平均损失 {_f(numbers['n10000_eps4_mean_loss'])}，中位数 {_f(numbers['n10000_eps4_median_loss'])}）。"
          f"预测要求至少 50% 的日志，因此{'成立' if met else '不成立'}。")
    return _res("D17", "MET" if met else "NOT MET", numbers, en, zh, tables)


# ----------------------------------------------------------------------------------------------------- D18
_ATHR = re.compile(r"^athr:t=([0-9.]+)$")


def run_d18(deep, reg, outdir):
    idx = _index(deep, reg, "plan")
    logs = sorted({k[0] for k in idx})
    ts = sorted({float(_ATHR.match(a).group(1)) for r in idx.values() for a in r["arms"] if _ATHR.match(a)})
    per_log, summ, smallest = [], [], []
    med_len, med_prec, n_logs, full_n = {}, {}, {}, {}
    for t in ts:
        arm = f"athr:t={t:g}"
        full_by_log = []
        for n in PLAN_GRID:
            L, P = [], []
            for log in logs:
                row = idx.get((log, n))
                ln, pr = _val(deep, row, arm, "len100"), _val(deep, row, arm, "prec100")
                if _finite(ln):
                    L.append(ln)
                    P.append(pr)
                    per_log.append([log, _stratum(deep, log), t, n, ln, pr])
            med_len[(t, n)], med_prec[(t, n)], n_logs[(t, n)] = _median(L), _median(P), len(L)
            summ.append([t, n, len(L), _median(L), _median(P), _nanmean(L), _nanmean(P)])
        for log in logs:
            avail = [n for n in PLAN_GRID if _finite(_val(deep, idx.get((log, n)), arm, "len100"))]
            if avail:
                hit = next((n for n in avail if _val(deep, idx[(log, n)], arm, "len100") >= 100 - 1e-6), None)
                full_by_log.append(float("inf") if hit is None else float(hit))
                smallest.append([t, log, "" if hit is None else hit, int(hit is None)])
        maj = next((n for n in PLAN_GRID if n_logs[(t, n)] >= _min_logs(reg) and med_len[(t, n)] >= 100 - 1e-6), None)
        full_n[t] = (maj, _median(full_by_log))
        smallest.append([t, "SMALLEST n WHERE MEDIAN LENGTH OVER LOGS = 100 (needs min_logs)", "" if maj is None else maj, int(maj is None)])
        smallest.append([t, "MEDIAN OVER LOGS OF PER-LOG SMALLEST n (inf = never)", _median(full_by_log), ""])
    tables = [
        _write_csv(outdir, "D18_per_log.csv", ["log", "stratum", "t", "n", "len100", "prec100"], per_log),
        _write_csv(outdir, "D18_by_t_n.csv", ["t", "n", "n_logs", "median_len100", "median_prec100", "mean_len100", "mean_prec100"], summ),
        _write_csv(outdir, "D18_smallest_n_full_list.csv", ["t", "log", "smallest_n_publishing_100", "never_on_grid"], smallest),
    ]
    k = (5.0, 100)
    maj5, medlog5 = full_n.get(5.0, (None, float("nan")))
    numbers = {"interpretation": "Median over logs of each log's draw-mean of len100 and prec100 (athr:t=5, n=100); 'publishes a full 100 items' = mean published length 100 (>= 99.999999); smallest n reported both as first grid n where the median over logs is 100 and as the median of per-log smallest n.",
               "n100_t5_eligible_logs": n_logs.get(k, 0), "n100_t5_median_len100": med_len.get(k, float("nan")), "n100_t5_median_prec100": med_prec.get(k, float("nan")),
               "t5_smallest_n_median_list_full": "none on grid" if maj5 is None else maj5,
               "t5_median_of_per_log_smallest_n": "never" if medlog5 == np.inf else medlog5, "min_logs": _min_logs(reg)}
    for t in ts:
        numbers[f"n100_t{t:g}_median_len100"] = med_len.get((t, 100), float("nan"))
    if n_logs.get(k, 0) < _min_logs(reg):
        return _nt("D18", f"only {n_logs.get(k, 0)} eligible log(s) with t=5 at n=100, at least {_min_logs(reg)} are needed",
                   f"n=100、t=5 时只有 {n_logs.get(k, 0)} 个日志可用，至少需要 {_min_logs(reg)} 个", numbers, tables)
    L, P = med_len[k], med_prec[k]
    met = L < 50 and P >= 0.5
    full_txt = "no n on the grid up to 3000" if maj5 is None else f"{maj5} people per round"
    full_zh = "网格内（最多 3000）没有" if maj5 is None else f"每轮 {maj5} 人"
    en = (f"A published item must have at least 5 reports in the running count. With 100 people per round, the median published list has {_f(L, 1)} items "
          f"(the full list would be 100) and {_pct(P)} of the published items are truly in the top 100 (median over {n_logs[k]} logs). "
          f"A full 100-item list is first reached at {full_txt}. The prediction needed fewer than 50 items and at least 50% correct items, so it is {'met' if met else 'not met'}.")
    zh = (f"每个发布的物品要求累计至少 5 次汇报。每轮 100 人时，发布榜单的中位长度是 {_f(L, 1)} 个物品（完整榜单应为 100 个），"
          f"其中 {_pct(P)} 的物品确实在真实前 100 名里（{n_logs[k]} 个日志的中位数）。能发满 100 个物品的最小规模：{full_zh}。"
          f"预测要求长度少于 50 且准确率至少 50%，因此{'成立' if met else '不成立'}。")
    return _res("D18", "MET" if met else "NOT MET", numbers, en, zh, tables)


# ----------------------------------------------------------------------------------------------------- D19
def equivalent_age(b, f):
    """Lag at which full:lag=L equals the baseline value b, f = NDCG at LAGS. Returns (age, status).

    status: 'censored_0' if b > f[0] (baseline better than the freshest full count), 'censored_8' if b <= f[-1] (as bad as lag 8),
    else 'interpolated' (linear between the first pair of neighbouring lags, going up from lag 0, that brackets b); b == f[0] -> exactly 0."""
    if not _finite(b) or any(not _finite(x) for x in f):
        return float("nan"), "NA"
    if b > f[0]:
        return 0.0, "censored_0"
    if b == f[0]:
        return 0.0, "interpolated"
    if b <= f[-1]:
        return float(LAGS[-1]), "censored_8"
    for i in range(len(LAGS) - 1):
        if f[i] > b >= f[i + 1]:
            return LAGS[i] + (f[i] - b) / (f[i] - f[i + 1]) * (LAGS[i + 1] - LAGS[i]), "interpolated"
    return float("nan"), "NA"       # unreachable: b < f[0] and b > f[-1] always brackets


def run_d19(deep, reg, outdir):
    idx = _index(deep, reg, "plan")
    logs = sorted({k[0] for k in idx})
    sp = reg["split"]
    fresh_new = set(sp.get("holdout_fresh", [])) | set(sp.get("new_discovery", [])) | set(sp.get("new_holdout", []))
    base = reg.get("baseline", "decay:h=2")

    def group(log):
        if _stratum(deep, log) == "other":
            return "non-Amazon"
        return "Amazon discovery" if deep.split(log) == "discovery" else "Amazon holdout and fresh"

    per_log, ages = [], {}          # ages[(log, n)] = (age, status)
    for n in (30, 100, 300, 1000):
        for log in logs:
            row = idx.get((log, n))
            b = _val(deep, row, base)
            f = [_val(deep, row, f"full:lag={L}") for L in LAGS]
            age, st = equivalent_age(b, f)
            if st == "NA":
                continue
            ages[(log, n)] = (age, st)
            per_log.append([log, group(log), int(log in fresh_new), n, b] + f + [age, st])
    summ = []
    for n in (30, 100, 300, 1000):
        sets = [("all logs", lambda l: True), ("fresh and new logs (verdict set)", lambda l: l in fresh_new)] + \
               [(g, (lambda gg: lambda l: group(l) == gg)(g)) for g in ("Amazon discovery", "Amazon holdout and fresh", "non-Amazon")]
        for name, pred in sets:
            v = [(a, s) for (l, nn), (a, s) in ages.items() if nn == n and pred(l)]
            vals = [a for a, _ in v]
            summ.append([n, name, len(v), _median(vals), _nanmean(vals), float(np.percentile(vals, 25)) if vals else float("nan"),
                         float(np.percentile(vals, 75)) if vals else float("nan"),
                         sum(s == "censored_0" for _, s in v), sum(s == "censored_8" for _, s in v)])
    tables = [
        _write_csv(outdir, "D19_per_log.csv", ["log", "group", "fresh_or_new", "n", "ndcg_baseline"] + [f"ndcg_full_lag{L}" for L in LAGS] + ["equivalent_age", "status"], per_log),
        _write_csv(outdir, "D19_by_n_group.csv", ["n", "group", "n_logs", "median_age", "mean_age", "p25", "p75", "n_censored_at_0", "n_censored_at_8"], summ),
    ]
    sel = [(a, s) for (l, n), (a, s) in ages.items() if n == 100 and l in fresh_new]
    med = _median([a for a, _ in sel])
    allmed = _median([a for (l, n), (a, s) in ages.items() if n == 100])
    numbers = {"interpretation": "Verdict set = holdout_fresh + new_discovery + new_holdout logs at n=100 (fresh categories and new non-Amazon logs); equivalent age interpolated linearly between lags 0,1,2,3,4,6,8 from lag 0 upwards, 0 if baseline > lag 0 NDCG, 8 if baseline <= lag 8 NDCG (censored values enter the median as 0 / 8); interval [3, 6] inclusive.",
               "n100_fresh_new_logs": len(sel), "n100_fresh_new_median_age": med, "n100_fresh_new_censored_at_0": sum(s == "censored_0" for _, s in sel),
               "n100_fresh_new_censored_at_8": sum(s == "censored_8" for _, s in sel), "n100_all_logs_median_age": allmed, "min_logs": _min_logs(reg)}
    if len(sel) < _min_logs(reg):
        return _nt("D19", f"only {len(sel)} fresh/new log(s) at n=100, at least {_min_logs(reg)} are needed",
                   f"n=100 时只有 {len(sel)} 个新类别或新日志，至少需要 {_min_logs(reg)} 个", numbers, tables)
    lo, hi = reg_hyp(reg, "D19")["prediction"]["lo"], reg_hyp(reg, "D19")["prediction"]["hi"]
    met = lo <= med <= hi
    en = (f"The 'equivalent age' is how old a full count of all users would have to be to be as good as the sampled list: it is the lag at which the full count's NDCG equals the sampled list's NDCG (interpolated between lags 0 to 8, capped at 0 and 8). "
          f"With 100 people per round, the median equivalent age over {len(sel)} fresh-category and new non-Amazon logs is {_f(med, 2)} rounds "
          f"({numbers['n100_fresh_new_censored_at_0']} logs capped at 0, {numbers['n100_fresh_new_censored_at_8']} capped at 8; median over all logs {_f(allmed, 2)}). "
          f"The prediction needed a median between {lo} and {hi}, so it is {'met' if met else 'not met'}.")
    zh = (f"“等效年龄”指：全量统计要旧到什么程度，才和抽样榜单一样好；即全量统计的 NDCG 等于抽样榜单 NDCG 时的滞后轮数（在 0 到 8 轮之间插值，并以 0 和 8 为上下限）。"
          f"每轮 100 人时，{len(sel)} 个新类别和新的非 Amazon 日志的等效年龄中位数为 {_f(med, 2)} 轮"
          f"（{numbers['n100_fresh_new_censored_at_0']} 个被截在 0，{numbers['n100_fresh_new_censored_at_8']} 个被截在 8；全部日志的中位数 {_f(allmed, 2)}）。"
          f"预测要求中位数在 {lo} 到 {hi} 之间，因此{'成立' if met else '不成立'}。")
    return _res("D19", "MET" if met else "NOT MET", numbers, en, zh, tables)


def reg_hyp(reg, hid):
    return next(h for h in reg["hypotheses"] if h["id"] == hid)


# ----------------------------------------------------------------------------------------------------- D20
def run_d20(deep, reg, outdir):
    idx = _index(deep, reg, "plan")
    logs = sorted({k[0] for k in idx})
    KS = [(10, "nd10"), (30, "nd30"), (100, "nd100"), (300, "nd300")]
    arms = [f"decay:h={h}" for h in HALF_LIVES]
    per_log, mean_rows, ok_logs = [], [], []
    tab = {}
    for log in logs:
        row = idx.get((log, 100))
        grid = {k: [_val(deep, row, a, m) for a in arms] for k, m in KS}
        if row is None or any(not _finite(x) for g in grid.values() for x in g):
            continue
        ok_logs.append(log)
        best = {}
        for k, _ in KS:
            g = grid[k]
            best[k] = HALF_LIVES[int(np.argmax(g))]         # first maximum in ascending half-life; exact ties are not expected for 5-decimal means
            for h, v in zip(HALF_LIVES, g):
                tab.setdefault((k, h), []).append(v)
        per_log.append([log, _stratum(deep, log)] + [best[k] for k, _ in KS] + [int(_hnum(best[10]) <= _hnum(best[300])), int(_hnum(best[10]) < _hnum(best[300]))])
    for (k, _) in KS:
        for h in HALF_LIVES:
            v = tab.get((k, h), [])
            mean_rows.append([k, h, len(v), _nanmean(v)])
    tables = [
        _write_csv(outdir, "D20_best_half_life_per_log.csv", ["log", "stratum", "best_h_k10", "best_h_k30", "best_h_k100", "best_h_k300", "best10_le_best300", "best10_lt_best300"], per_log),
        _write_csv(outdir, "D20_mean_ndcg_by_k_h.csv", ["k", "half_life", "n_logs", "mean_ndcg"], mean_rows),
    ]
    n = len(per_log)
    s_le = _share(r[6] for r in per_log)
    s_lt = _share(r[7] for r in per_log)
    numbers = {"interpretation": "n=100; best half-life = argmax of the draw-mean steady NDCG over the seven decay arms for each cutoff k (nd10, nd30, nd100, nd300); half-life inf is the longest; logs need all seven arms and four metrics.",
               "eligible_logs": n, "share_best10_le_best300": s_le, "share_best10_lt_best300": s_lt, "threshold_le": 0.6, "threshold_lt": 0.25, "min_logs": _min_logs(reg)}
    for kk in (10, 30, 100, 300):
        vs = [_hnum(r[2 + [10, 30, 100, 300].index(kk)]) for r in per_log]
        numbers[f"median_best_h_k{kk}"] = _median([v if np.isfinite(v) else 16.0 for v in vs])
    numbers["note_median_best_h"] = "half-life inf is coded as 16 for this median only"
    if n < _min_logs(reg):
        return _nt("D20", f"only {n} log(s) have all seven decay arms at n=100, at least {_min_logs(reg)} are needed",
                   f"n=100 时只有 {n} 个日志具备全部七个衰减方案，至少需要 {_min_logs(reg)} 个", numbers, tables)
    met = s_le >= 0.6 and s_lt >= 0.25
    en = (f"For each log we find the half-life (how fast old reports fade) that gives the best list at n=100 people per round, once for a top-10 list and once for a top-300 list. "
          f"In {_pct(s_le)} of {n} logs the best half-life for the top 10 is no longer than for the top 300, and it is strictly shorter in {_pct(s_lt)}. "
          f"The prediction needed at least 60% and at least 25%, so it is {'met' if met else 'not met'}.")
    zh = (f"对每个日志，分别找出每轮 100 人时使前 10 名榜单和前 300 名榜单最好的半衰期（旧汇报衰减的速度）。"
          f"在 {n} 个日志中，{_pct(s_le)} 的日志里前 10 名的最佳半衰期不长于前 300 名，{_pct(s_lt)} 的日志里严格更短。"
          f"预测要求分别至少 60% 和 25%，因此{'成立' if met else '不成立'}。")
    return _res("D20", "MET" if met else "NOT MET", numbers, en, zh, tables)


# ----------------------------------------------------------------------------------------------------- D21
def run_d21(deep, reg, outdir):
    idx = _index(deep, reg, "plan")
    logs = sorted({k[0] for k in idx})
    base = reg.get("baseline", "decay:h=2")
    full = "fulldecay:h=2"
    KS = [(10, "nd10"), (30, "nd30"), (100, "nd100"), (300, "nd300")]
    ratio = {}                       # (k, log, n) -> ratio
    rows = []
    for k, m in KS:
        for log in logs:
            for n in PLAN_GRID:
                row = idx.get((log, n))
                b, f = _val(deep, row, base, m), _val(deep, row, full, m)
                if _finite(b) and _finite(f) and f > 0:
                    ratio[(k, log, n)] = b / f
                    rows.append([log, _stratum(deep, log), k, n, b, f, b / f])
    small, by_n, slopes = [], [], []
    reach = {}                       # (k, log) -> smallest n or inf
    for k, _ in KS:
        for log in logs:
            ns = [n for n in PLAN_GRID if (k, log, n) in ratio]
            if not ns:
                continue
            hit = next((n for n in ns if ratio[(k, log, n)] >= 0.95), None)
            reach[(k, log)] = float("inf") if hit is None else float(hit)
            small.append([log, _stratum(deep, log), k, "" if hit is None else hit, int(hit is None), max(ns)])
            if len(ns) >= 3:
                sl = float(np.polyfit(np.log10(ns), [1 - ratio[(k, log, n)] for n in ns], 1)[0])
                slopes.append([log, k, len(ns), sl])
    for k, _ in KS:
        for n in PLAN_GRID:
            lg = [l for l in logs if (k, l, n) in ratio]
            v = [ratio[(k, l, n)] for l in lg]
            by_n.append([k, n, len(lg), _nanmean(v), _median(v), _share(x >= 0.95 for x in v), _share(reach[(k, l)] <= n for l in lg)])
    pooled = {}
    for k, _ in KS:
        pts = [(n, 1 - r[3]) for r in by_n if r[0] == k and r[2] > 0 for n in [r[1]]]
        pooled[k] = float(np.polyfit(np.log10([p[0] for p in pts]), [p[1] for p in pts], 1)[0]) if len(pts) >= 3 else float("nan")
    slope_med = {k: _median([s[3] for s in slopes if s[1] == k]) for k, _ in KS}
    tables = [
        _write_csv(outdir, "D21_ratio_per_log.csv", ["log", "stratum", "k", "n", "ndcg_baseline", "ndcg_full_decay", "ratio"], rows),
        _write_csv(outdir, "D21_smallest_n_95.csv", ["log", "stratum", "k", "smallest_n_ratio_ge_0.95", "never_on_grid", "largest_n_seen"], small),
        _write_csv(outdir, "D21_by_k_n.csv", ["k", "n", "n_logs", "mean_ratio", "median_ratio", "share_ratio_ge_0.95_at_n", "share_reached_95_by_n"], by_n),
        _write_csv(outdir, "D21_slope_1_minus_ratio_vs_log10n.csv", ["log", "k", "n_points", "slope_per_decade"], slopes),
        _write_csv(outdir, "D21_slope_summary.csv", ["k", "median_per_log_slope_per_decade", "slope_of_cross_log_mean_per_decade"], [[k, slope_med[k], pooled[k]] for k, _ in KS]),
    ]
    n3 = [l for l in logs if (100, l, 3000) in ratio]
    s_lt = _share(ratio[(100, l, 3000)] < 0.95 for l in n3)
    numbers = {"interpretation": "ratio = steady NDCG of decay:h=2 at n divided by steady NDCG of fulldecay:h=2 (same row); k=100 uses nd100, k=10/30/300 the nd10/nd30/nd300 metrics; verdict = share of logs eligible at n=3000 with ratio < 0.95 (k=100) >= 70%; slope = OLS slope of (1 - ratio) on log10(n) (per tenfold increase of n).",
               "eligible_logs_n3000": len(n3), "share_ratio_lt_0.95_at_n3000": s_lt,
               "mean_ratio_k100_n3000": _nanmean([ratio[(100, l, 3000)] for l in n3]), "min_logs": _min_logs(reg),
               "slope_per_decade_k100_median_log": slope_med[100], "slope_per_decade_k100_cross_log_mean": pooled[100]}
    for k, _ in KS:
        for n in (100, 3000):
            v = [r for r in by_n if r[0] == k and r[1] == n]
            numbers[f"k{k}_n{n}_share_ge_0.95"] = v[0][5] if v else float("nan")
    if len(n3) < _min_logs(reg):
        return _nt("D21", f"only {len(n3)} log(s) eligible at n=3000, at least {_min_logs(reg)} are needed",
                   f"n=3000 时只有 {len(n3)} 个日志可用，至少需要 {_min_logs(reg)} 个", numbers, tables)
    met = s_lt >= 0.7
    en = (f"The 'ceiling' is the list built from a full count of every user with the same half-life. At 3000 people per round, the sampled list scores below 95% of this ceiling (top-100 list) "
          f"in {_pct(s_lt)} of {len(n3)} logs (mean ratio {_f(numbers['mean_ratio_k100_n3000'], 3)}). "
          f"Per tenfold increase in people the gap to the ceiling (1 minus ratio) {'shrinks' if slope_med[100] < 0 else 'grows'} by {_f(abs(slope_med[100]), 3)} (median of per-log fitted slopes). "
          f"The prediction needed at least 70% of logs below 95%, so it is {'met' if met else 'not met'}.")
    zh = (f"“上限”是指用全体用户的全量统计、同样半衰期所得的榜单。每轮 3000 人时，抽样榜单（前 100 名）在 {len(n3)} 个日志中有 {_pct(s_lt)} 低于上限的 95%"
          f"（平均比值 {_f(numbers['mean_ratio_k100_n3000'], 3)}）。人数每增加十倍，与上限的差距（1 减比值）{'缩小' if slope_med[100] < 0 else '扩大'} {_f(abs(slope_med[100]), 3)}（各日志拟合斜率的中位数）。"
          f"预测要求至少 70% 的日志低于 95%，因此{'成立' if met else '不成立'}。")
    return _res("D21", "MET" if met else "NOT MET", numbers, en, zh, tables)


# ----------------------------------------------------------------------------------------------------- D22
def run_d22(deep, reg, outdir):
    idx = _index(deep, reg, "plan")
    logs = sorted({k[0] for k in idx})
    arms = [f"decay:h={h}" for h in HALF_LIVES]
    per_log, burn, lost = [], {}, {}  # burn[(h, n)] = {log: round}; lost[(h, n)] = [n_left_censored, n_never]
    for h, arm in zip(HALF_LIVES, arms):
        for n in (30, 100, 300):
            for log in logs:
                row = idx.get((log, n))
                if row is None or arm not in row["arms"]:
                    continue
                steps = np.array(row["steps"], int)
                a = deep.steps(row, arm, "nd100")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    curve = np.nanmean(a, axis=0)            # mean over draws per step
                    late = curve[steps >= STEADY]
                steady = _nanmean(late)
                fin = np.where(np.isfinite(curve))[0]
                if not _finite(steady) or steady <= 0 or not fin.size:
                    continue
                hit = np.where(np.isfinite(curve) & (curve >= 0.9 * steady))[0]
                s0 = int(steps[fin[0]])                       # first step with a truth list (early rounds of a growing log have none)
                lc = bool(hit.size and hit[0] == fin[0] and s0 > 3)   # already at 90% at the first evaluated step, which is after round 3: burn-in unobservable
                rnd = float("nan") if (lc or not hit.size) else float(steps[hit[0]])
                c = lost.setdefault((h, n), [0, 0])
                if lc:
                    c[0] += 1
                elif not hit.size:
                    c[1] += 1
                else:
                    burn.setdefault((h, n), {})[log] = rnd
                per_log.append([log, _stratum(deep, log), h, n, steady, s0, float(curve[fin[0]] / steady), int(lc), rnd])
    summ = []
    for h in HALF_LIVES:
        for n in (30, 100, 300):
            v = list(burn.get((h, n), {}).values())
            c = lost.get((h, n), [0, 0])
            summ.append([h, n, len(v), _median(v), _nanmean(v), _share(x <= 3 for x in v), c[0], c[1]])
    tables = [
        _write_csv(outdir, "D22_burnin_per_log.csv", ["log", "stratum", "half_life", "n", "steady_ndcg", "first_evaluated_step", "first_evaluated_over_steady", "left_censored_excluded", "round_reaching_90pct"], per_log),
        _write_csv(outdir, "D22_burnin_summary.csv", ["half_life", "n", "n_logs_used", "median_round", "mean_round", "share_within_3_rounds", "n_excluded_left_censored", "n_never_reached"], summ),
    ]
    d2, di = burn.get(("2", 100), {}), burn.get(("inf", 100), {})
    m2, mi = _median(list(d2.values())), _median(list(di.values()))
    numbers = {"interpretation": "Draw-mean nd100 per step; steady = mean over steps >= 8; burn-in = round number (= step, steps start at round 2) of the first step whose draw-mean is >= 0.9 x steady; median over logs; verdict at n=100: h=2 <= 3 and h=inf >= 4. Steps without a truth list (NaN, early rounds of a growing log) cannot be evaluated; a log whose first evaluated step is after round 3 and already at >= 90% of steady is left-censored (burn-in unobservable) and excluded and counted.",
               "n100_h2_logs": len(d2), "n100_h2_median_round": m2, "n100_hinf_logs": len(di), "n100_hinf_median_round": mi, "min_logs": _min_logs(reg),
               "n100_h2_excluded_left_censored": lost.get(("2", 100), [0, 0])[0], "n100_hinf_excluded_left_censored": lost.get(("inf", 100), [0, 0])[0]}
    for h in HALF_LIVES:
        numbers[f"n100_h{h}_median_round"] = _median(list(burn.get((h, 100), {}).values()))
    if len(d2) < _min_logs(reg) or len(di) < _min_logs(reg):
        return _nt("D22", f"usable logs at n=100: {len(d2)} (h=2), {len(di)} (h=inf); at least {_min_logs(reg)} are needed for both ({numbers['n100_h2_excluded_left_censored']} and {numbers['n100_hinf_excluded_left_censored']} logs were left out because their first evaluated round after round 3 was already at 90%)",
                   f"n=100 时可用日志：h=2 有 {len(d2)} 个、h=inf 有 {len(di)} 个，两者都至少需要 {_min_logs(reg)} 个（分别有 {numbers['n100_h2_excluded_left_censored']} 个和 {numbers['n100_hinf_excluded_left_censored']} 个日志因第 3 轮之后首次评估时就已达 90% 而被排除）", numbers, tables)
    met = m2 <= 3
    en = (f"Burn-in is the first round at which the list's average NDCG reaches 90% of its steady level (steady = average over rounds 8 and later), at 100 people per round. "
          f"With half-life 2 the median round is {_f(m2, 1)} ({len(d2)} logs); with no forgetting (half-life infinite) it is {_f(mi, 1)} ({len(di)} logs); logs whose first evaluated round was already at 90% after round 3 are left out ({numbers['n100_h2_excluded_left_censored']} and {numbers['n100_hinf_excluded_left_censored']}). "
          f"The prediction needed at most 3 for half-life 2 and at least 4 without forgetting, so it is {'met' if met else 'not met'}.")
    zh = (f"冷启动轮数是指榜单的平均 NDCG 第一次达到稳态水平（第 8 轮及以后的平均）90% 的那一轮，条件为每轮 100 人。"
          f"半衰期为 2 时中位轮数是 {_f(m2, 1)}（{len(d2)} 个日志）；不遗忘（半衰期无穷大）时是 {_f(mi, 1)}（{len(di)} 个日志）；第 3 轮之后首次评估时就已达 90% 的日志无法观察冷启动，被排除（分别 {numbers['n100_h2_excluded_left_censored']} 个和 {numbers['n100_hinf_excluded_left_censored']} 个）。"
          f"预测要求半衰期 2 不超过 3 轮、不遗忘至少 4 轮，因此{'成立' if met else '不成立'}。")
    return _res("D22", "MET" if met else "NOT MET", numbers, en, zh, tables)


# ----------------------------------------------------------------------------------------------------- D23
def run_d23(deep, reg, outdir):
    idx = _index(deep, reg, "plan")
    logs = sorted({k[0] for k in idx})
    base = reg.get("baseline", "decay:h=2")
    per_log, gaps = [], {}           # gaps[(n, log)] = (gap10, gap5)
    for n in PLAN_GRID:
        for log in logs:
            row = idx.get((log, n))
            if row is None or base not in row["arms"]:
                continue
            steps = np.array(row["steps"], int)
            a = deep.steps(row, base, "nd100")[:, steps >= STEADY].ravel()
            a = a[np.isfinite(a)]
            if a.size < 10:
                continue
            mean, p10, p5 = float(a.mean()), float(np.percentile(a, 10)), float(np.percentile(a, 5))
            gaps[(n, log)] = (mean - p10, mean - p5)
            per_log.append([log, _stratum(deep, log), n, a.size, mean, p10, p5, mean - p10, mean - p5])
    summ = []
    for n in PLAN_GRID:
        for name, pred in (("all", lambda l: True), ("amazon", lambda l: _stratum(deep, l) == "amazon"), ("other", lambda l: _stratum(deep, l) == "other")):
            g = [v for (nn, l), v in gaps.items() if nn == n and pred(l)]
            summ.append([n, name, len(g), _median([x[0] for x in g]), _median([x[1] for x in g]),
                         _median([r[4] for r in per_log if r[2] == n and pred(r[0])]), _median([r[5] for r in per_log if r[2] == n and pred(r[0])])])
    tables = [
        _write_csv(outdir, "D23_per_log.csv", ["log", "stratum", "n", "n_values", "mean", "p10", "p5", "gap_mean_minus_p10", "gap_mean_minus_p5"], per_log),
        _write_csv(outdir, "D23_by_n_stratum.csv", ["n", "stratum", "n_logs", "median_gap_p10", "median_gap_p5", "median_mean", "median_p10"], summ),
    ]
    g300 = [v[0] for (n, l), v in gaps.items() if n == 300]
    g30 = [v[0] for (n, l), v in gaps.items() if n == 30]
    m300, m30 = _median(g300), _median(g30)
    numbers = {"interpretation": "Per log: per-round nd100 of decay:h=2 pooled over draws and steps >= 8; gap = mean - 10th percentile (linear interpolation); median over all logs eligible at that n; verdict: gap(n=300) <= 0.10 and gap(n=30) >= 0.15.",
               "n300_logs": len(g300), "n300_median_gap_p10": m300, "n30_logs": len(g30), "n30_median_gap_p10": m30, "min_logs": _min_logs(reg)}
    for n in PLAN_GRID:
        numbers[f"n{n}_median_gap_p10"] = _median([v[0] for (nn, l), v in gaps.items() if nn == n])
    if len(g300) < _min_logs(reg) or len(g30) < _min_logs(reg):
        return _nt("D23", f"logs available: {len(g30)} at n=30 and {len(g300)} at n=300; at least {_min_logs(reg)} are needed at both",
                   f"n=30 时有 {len(g30)} 个、n=300 时有 {len(g300)} 个日志，两处都至少需要 {_min_logs(reg)} 个", numbers, tables)
    met = m300 <= 0.10 and m30 >= 0.15
    en = (f"A 'bad round' is a round whose NDCG is in the worst 10% of rounds (steady rounds 8 and later, all draws pooled); we measure how far it sits below the average round. "
          f"The median gap over logs is {_f(m300)} NDCG at 300 people per round ({len(g300)} logs) and {_f(m30)} at 30 people per round ({len(g30)} logs). "
          f"The prediction needed at most 0.10 at n=300 and at least 0.15 at n=30, so it is {'met' if met else 'not met'}.")
    zh = (f"“糟糕的轮次”指 NDCG 处于最差 10% 的那些轮（第 8 轮及以后，合并所有重复），我们度量它比平均轮低多少。"
          f"各日志差距的中位数：每轮 300 人时为 {_f(m300)} NDCG（{len(g300)} 个日志），每轮 30 人时为 {_f(m30)}（{len(g30)} 个日志）。"
          f"预测要求 n=300 时不超过 0.10、n=30 时至少 0.15，因此{'成立' if met else '不成立'}。")
    return _res("D23", "MET" if met else "NOT MET", numbers, en, zh, tables)


RUNNERS = {"D15": run_d15, "D16": run_d16, "D17": run_d17, "D18": run_d18, "D19": run_d19, "D20": run_d20,
           "D21": run_d21, "D22": run_d22, "D23": run_d23}


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    from an_deep_lib import Deep
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/deep_smoke.jsonl"
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/an_deep_calc1_out"
    reg = json.load(open(os.path.join(here, "..", "..", "configs", "deep_registry.json")))
    deep = Deep(path, reg)
    for hid, fn in RUNNERS.items():
        r = fn(deep, reg, out)
        json.dumps(r)                                   # must be serialisable
        print(f"== {hid}: {r['verdict']}")
        print("   numbers:", json.dumps({k: v for k, v in r["numbers"].items() if k != "interpretation"}, ensure_ascii=False))
        print("   EN:", r["summary_en"])
        print("   ZH:", r["summary_zh"])
        print("   tables:", r["tables"])
