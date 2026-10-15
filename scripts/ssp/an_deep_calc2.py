"""Calculator analyses D24-D30 of the deep programme (self-monitoring, calendar half-life, metric choice).

    RUNNERS = {"D24": fn, ..., "D30": fn};   fn(deep, reg, outdir) -> dict(id, verdict, numbers, summary_en, summary_zh, tables)

Verdicts are "MET", "NOT MET" or "NOT TESTABLE" (too few cells / logs, with the reason in the summary and in numbers["reason"]).
Common conventions (the registry text is the pre-registration; where it is ambiguous the literal reading is used and written to
numbers["interpretation"]):
  * a (log, n_win, n) cell is one row of the stage; its value is the steady-state (steps >= 8) mean over draws (deep.mean);
  * duplicate rows with the same key keep the last one;
  * minimum evidence: D24/D25/D26 need at least 12 cells; D27-D30 need at least reg["min_logs"] logs in the stratum judged;
  * a rule made of several clauses is NOT MET as soon as one clause is clearly false, MET only if all are true, otherwise NOT TESTABLE.
Run `python an_deep_calc2.py [deep.jsonl] [outdir]` to execute all runners on a results file.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import warnings

import numpy as np
from scipy import stats

BASE = "decay:h=2"
FULLDECAY = "fulldecay:h=2"
SELFSEL = "selfsel"
MIN_CELLS = 12
DAYS_GRID = [3, 10, 30, 100, 200, 300, 400, 534, 700, 1000, 1500]
DAYS_ARMS = [f"decay:days={d}" for d in DAYS_GRID]
H_GRID = ["0.25", "0.5", "1", "2", "4", "8", "inf"]            # chosen_h_idx 0..6
H_ARMS = [f"decay:h={h}" for h in H_GRID]
EPS = 1e-9


# ------------------------------------------------------------------ helpers

def _ok(x) -> bool:
    return x is not None and np.isfinite(x)


def _f(x, d=2) -> str:
    return f"{x:.{d}f}" if _ok(x) else "n/a"


def _num(x):
    """JSON-safe number: numpy -> python, NaN/inf -> None, floats rounded."""
    if isinstance(x, (np.floating, float)):
        return round(float(x), 5) if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _clean(d: dict) -> dict:
    return {k: _num(v) for k, v in d.items()}


def _cells(deep, stages) -> list:
    """Rows of the given stage(s), one per key (last wins), as (stage, row)."""
    if isinstance(stages, str):
        stages = [stages]
    out = []
    for st in stages:
        seen = {}
        for r in deep.cells(st):
            seen[r["key"]] = r
        out += [(st, r) for r in seen.values()]
    return out


def _m(deep, row, arm, metric="nd100") -> float:
    """Steady mean over draws of one arm/metric; NaN if the arm is not in the row."""
    if not deep.has(row, arm):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return deep.mean(row, arm, metric)


def _spear(x, y):
    """(rho, p, n) on the pairs where both are finite; NaN when n < 3 or one side is constant."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    k = np.isfinite(x) & np.isfinite(y)
    x, y = x[k], y[k]
    n = int(k.sum())
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan"), n
    r = stats.spearmanr(x, y)
    return float(r[0]), float(r[1]), n


def _csv(outdir, name, header, rows) -> str:
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if (isinstance(v, (float, np.floating)) and not np.isfinite(v))
                        else (round(float(v), 6) if isinstance(v, (float, np.floating)) else v) for v in r])
    return name


def _combine(clauses) -> str:
    """clauses: list of True / False / None (None = cannot be judged)."""
    if any(c is False for c in clauses):
        return "NOT MET"
    if any(c is None for c in clauses):
        return "NOT TESTABLE"
    return "MET"


def _result(id_, verdict, numbers, en, zh, tables) -> dict:
    return dict(id=id_, verdict=verdict, numbers=_clean(numbers), summary_en=en, summary_zh=zh, tables=tables)


def _hyp(reg, id_) -> dict:
    return next(h for h in reg["hypotheses"] if h["id"] == id_)


def _new_logs(reg) -> set:
    return set(v for g, v in reg["split"].items() if g.startswith("new_") for v in v)


def _by_group(vals, key):
    out = {}
    for v in vals:
        out.setdefault(key(v), []).append(v)
    return out


def _days_nd(deep, row):
    return np.array([_m(deep, row, a) for a in DAYS_ARMS])


def _best_days_by_log(deep, cal, nwins, logs_ok):
    """{log: {n_win: nd-by-days vector}} for logs with all nwins present (n=100, all eight days arms finite)."""
    tmp = {}
    for _, r in cal:
        if r["n"] != 100 or r["n_win"] not in nwins or not logs_ok(r["dataset"]):
            continue
        v = _days_nd(deep, r)
        if np.isfinite(v).all():
            tmp.setdefault(r["dataset"], {})[r["n_win"]] = (v, r["window_days"])
    return {lg: d for lg, d in tmp.items() if all(w in d for w in nwins)}


# ------------------------------------------------------------------ D24

def run_d24(deep, reg, outdir):
    h = _hyp(reg, "D24")
    thr = h["prediction"]["rho"]
    recs = []
    for st, r in _cells(deep, ["cal", "plan"]):
        if not deep.has(r, BASE):
            continue
        x = _m(deep, r, BASE, "selfnd")
        y = _m(deep, r, BASE, "nd100")
        recs.append(dict(stage=st, log=r["dataset"], n=r["n"], n_win=r["n_win"], selfnd=x, nd100=y,
                         ratio=x / y if _ok(x) and _ok(y) and y > 0 else float("nan")))
    t1 = _csv(outdir, "d24_cells.csv", ["stage", "log", "n", "n_win", "selfnd", "nd100", "selfnd_over_nd100"],
              [[c["stage"], c["log"], c["n"], c["n_win"], c["selfnd"], c["nd100"], c["ratio"]] for c in recs])
    X = [c["selfnd"] for c in recs]
    Y = [c["nd100"] for c in recs]
    rho, p, n = _spear(X, Y)
    rows = [["pooled", "all", n, rho, p]]
    nums = dict(n_cells=n, rho_pooled=rho, threshold=thr, interpretation=(
        "every row of stages cal and plan is one cell (a (log, n, n_win) present in both stages counts twice: independent draws); "
        "rho_pooled_merged averages such duplicates"))
    within = {}
    for field, tag in (("n", "n"), ("n_win", "nwin")):
        for v, cs in sorted(_by_group(recs, lambda c: c[field]).items()):
            rr, pp, nn = _spear([c["selfnd"] for c in cs], [c["nd100"] for c in cs])
            rows.append([f"by_{field}", v, nn, rr, pp])
            nums[f"rho_{tag}{v}"] = rr
            if field == "n":
                within[v] = (rr, nn)
    wv = [rr for rr, nn in within.values() if nn >= MIN_CELLS and _ok(rr)]
    rho_med = float(np.median(wv)) if wv else float("nan")
    nums["rho_median_within_n"] = rho_med
    nums["n_values_used"] = len(wv)
    rb, pb, nb = _spear([c["n"] for c in recs], [c["ratio"] for c in recs])
    rows.append(["ratio_vs_n", "all", nb, rb, pb])
    nums["rho_ratio_vs_n"] = rb
    mer = {}
    for c in recs:
        mer.setdefault((c["log"], c["n"], c["n_win"]), []).append(c)
    rm, _, nm = _spear([np.nanmean([c["selfnd"] for c in v]) for v in mer.values()],
                       [np.nanmean([c["nd100"] for c in v]) for v in mer.values()])
    nums.update(rho_pooled_merged=rm, n_cells_merged=nm)
    ratios = np.array([c["ratio"] for c in recs])
    nums["mean_ratio_selfnd_over_nd100"] = float(np.nanmean(ratios)) if np.isfinite(ratios).any() else float("nan")
    t2 = _csv(outdir, "d24_spearman.csv", ["group", "value", "n_cells", "rho", "p"], rows)
    if not _ok(rho_med):
        verdict, nums["reason"] = "NOT TESTABLE", f"no n value with at least {MIN_CELLS} usable cells ({n} cells in total)"
        en = (f"Only {n} cells (log x round length x n) have both scores, fewer than the {MIN_CELLS} needed, so the self-score "
              f"cannot be judged; the correlation on these cells is {_f(rho)}.")
        zh = f"只有 {n} 个单元（日志 x 轮长 x n）同时有两种评分，少于所需的 {MIN_CELLS} 个，所以无法判断；这些单元上的相关系数为 {_f(rho)}。"
    else:
        verdict = "MET" if rho_med >= thr else "NOT MET"
        en = (f"Across {n} cells we compared the server's own score of its list (scored against the next round's reporter sample, "
              f"no ground truth needed) with the true NDCG@100 (0 to 1, higher means closer to the real top 100). The Spearman rank "
              f"correlation (1 = same ordering, 0 = unrelated) is {_f(rho_med)} (median of the within-n correlations over {len(wv)} values of n; pooled over all n it is {_f(rho)}, inflated because both scores rise with n) against a bar of {thr}, so the prediction is {verdict}. "
              f"The self-score is on average {_f(nums['mean_ratio_selfnd_over_nd100'])} times the true score, and this ratio moves "
              f"with n with rank correlation {_f(rb)}.")
        zh = (f"在 {n} 个单元中，我们把服务器对自己榜单的评分（用下一轮的汇报者样本打分，不需要真值）与真实 NDCG@100（0 到 1，越高越接近真实前 100）比较。"
              f"秩相关系数（1 表示排序完全一致，0 表示无关）为 {_f(rho_med)}（对 {len(wv)} 个 n 值分别计算后取中位数；不分 n 合并时为 {_f(rho)}，因为两种评分都随 n 上升而被抬高），门槛为 {thr}，因此预测{'成立' if verdict == 'MET' else '不成立'}。"
              f"自评分平均是真实分数的 {_f(nums['mean_ratio_selfnd_over_nd100'])} 倍，这个比值与 n 的秩相关为 {_f(rb)}。")
    return _result("D24", verdict, nums, en, zh, [t1, t2])


# ------------------------------------------------------------------ D25

def run_d25(deep, reg, outdir):
    h = _hyp(reg, "D25")
    thr = h["prediction"]["rho"]
    recs = []
    for _, r in _cells(deep, "cal"):
        if r["n"] != 100 or not (deep.has(r, BASE) and deep.has(r, FULLDECAY)):
            continue
        recs.append(dict(log=r["dataset"], n_win=r["n_win"], churn=_m(deep, r, BASE, "churn"),
                         gap=_m(deep, r, FULLDECAY) - _m(deep, r, BASE)))
    t1 = _csv(outdir, "d25_cells.csv", ["log", "n_win", "churn_baseline", "nd100_fulldecay_minus_baseline"],
              [[c["log"], c["n_win"], c["churn"], c["gap"]] for c in recs])
    rho, p, n = _spear([c["churn"] for c in recs], [c["gap"] for c in recs])
    rows = [["pooled", "all", n, rho, p]]
    nums = dict(n_cells=n, rho_pooled=rho, abs_rho=abs(rho) if _ok(rho) else float("nan"), threshold=thr, interpretation=(
        "statement: rank correlation >= 0.5; prediction text: not met expected when |rho| < 0.5, so MET iff |rho_pooled| >= 0.5 "
        "(signed rho reported). 'Best-days stability' = per log max/min of the best days over n_win 45, 60, 90 (decay:days arms), "
        "correlated across logs with the log's mean churn; descriptive only, not used for the verdict"))
    for v, cs in sorted(_by_group(recs, lambda c: c["n_win"]).items()):
        rr, pp, nn = _spear([c["churn"] for c in cs], [c["gap"] for c in cs])
        rows.append(["by_n_win", v, nn, rr, pp])
        nums[f"rho_nwin{v}"] = rr
    cal = _cells(deep, "cal")
    bd = _best_days_by_log(deep, cal, (45, 60, 90), lambda lg: True)
    lg_rows, lg_ch, lg_st = [], [], []
    for lg, d in sorted(bd.items()):
        best = [DAYS_GRID[int(np.argmax(d[w][0]))] for w in (45, 60, 90)]
        ratio = max(best) / min(best)
        ch = np.nanmean([c["churn"] for c in recs if c["log"] == lg]) if any(c["log"] == lg for c in recs) else float("nan")
        lg_rows.append([lg, ch, *best, ratio])
        lg_ch.append(ch)
        lg_st.append(math.log(ratio))
    rs, ps, ns = _spear(lg_ch, lg_st)
    rows.append(["churn_vs_best_days_ratio", "per log", ns, rs, ps])
    nums.update(rho_churn_vs_best_days_instability=rs, n_logs_best_days=ns)
    t2 = _csv(outdir, "d25_spearman.csv", ["group", "value", "n_cells", "rho", "p"], rows)
    t3 = _csv(outdir, "d25_best_days_by_log.csv", ["log", "mean_churn", "best_days_45", "best_days_60", "best_days_90", "max_over_min"],
              lg_rows)
    if n < MIN_CELLS or not _ok(rho):
        verdict, nums["reason"] = "NOT TESTABLE", f"only {n} usable cells at n=100 (need at least {MIN_CELLS})"
        en = (f"Only {n} cells at n=100 have both the baseline and the full-count list, fewer than the {MIN_CELLS} needed, so "
              f"churn versus staleness cannot be judged (correlation on these cells: {_f(rho)}).")
        zh = f"在 n=100 下只有 {n} 个单元同时有基线榜单和全量榜单，少于所需的 {MIN_CELLS} 个，无法判断榜单变动与过时程度的关系（这些单元上的相关系数：{_f(rho)}）。"
    else:
        verdict = "MET" if abs(rho) >= thr else "NOT MET"
        en = (f"Across {n} cells at n=100 we compared how much the published top-100 changes from one round to the next (churn, "
              f"share of the list replaced) with how far the list is behind a full count (NDCG of the full-count list minus NDCG of "
              f"the baseline). Their Spearman rank correlation is {_f(rho)}; the statement needs a size of {thr} or more, so it is "
              f"{verdict} (this study expected NOT MET). Against how much the best half-life in days differs between round counts, "
              f"the log-level correlation is {_f(rs)} over {ns} logs.")
        zh = (f"在 n=100 的 {n} 个单元中，我们比较了榜单相邻两轮之间的变动（前 100 名被替换的比例）与榜单落后于全量统计的程度（全量榜单 NDCG 减基线 NDCG）。"
              f"两者的秩相关系数为 {_f(rho)}；命题要求绝对值不低于 {thr}，因此结果为{'成立' if verdict == 'MET' else '不成立'}（本研究事先预期不成立）。"
              f"与最佳天数半衰期随轮数变化的幅度相比，{ns} 个日志上的相关系数为 {_f(rs)}。")
    return _result("D25", verdict, nums, en, zh, [t1, t2, t3])


# ------------------------------------------------------------------ D26

def run_d26(deep, reg, outdir):
    h = _hyp(reg, "D26")
    hi, lo = h["prediction"]["rho"], h["prediction"]["lo_rho"]
    arm = "half:h=2"
    recs = []
    for _, r in _cells(deep, "plan"):
        if not deep.has(r, arm):
            continue
        recs.append(dict(log=r["dataset"], n=r["n"], agree=_m(deep, r, arm, "agree"), nd100=_m(deep, r, arm)))
    t1 = _csv(outdir, "d26_cells.csv", ["log", "n", "agree", "nd100_half"], [[c["log"], c["n"], c["agree"], c["nd100"]] for c in recs])
    big = [c for c in recs if c["n"] >= 300]
    _, _, n_hi = _spear([c["agree"] for c in big], [c["nd100"] for c in big])
    rows = []
    hv = []
    for nv in (300, 1000, 3000):
        cs_ = [c for c in recs if c["n"] == nv]
        rr_, pp_, nn_ = _spear([c["agree"] for c in cs_], [c["nd100"] for c in cs_])
        rows.append(["within_n_for_median", nv, nn_, rr_, pp_])
        if nn_ >= MIN_CELLS and _ok(rr_):
            hv.append(rr_)
    rho_hi, p_hi = (float(np.median(hv)) if hv else float("nan")), float("nan")
    n_hi = len(hv)
    nums = dict(n_cells_n300plus=n_hi, rho_n300plus=rho_hi, threshold_hi=hi, threshold_lo=lo, interpretation=(
        "both clauses of the prediction text are required: MEDIAN over n in {300,1000,3000} (with >= 12 cells each) of the within-n rho >= 0.7 AND within-n rho(n=30) < 0.5; "
        "n_cells_n300plus here counts the n values used for the median, not cells"))
    per_n = {}
    for v, cs in sorted(_by_group(recs, lambda c: c["n"]).items()):
        rr, pp, nn = _spear([c["agree"] for c in cs], [c["nd100"] for c in cs])
        rows.append(["by_n", v, nn, rr, pp])
        per_n[v] = (rr, nn)
        if v in (30, 100):
            nums[f"rho_n{v}"] = rr
            nums[f"n_cells_n{v}"] = nn
    t2 = _csv(outdir, "d26_spearman.csv", ["group", "value", "n_cells", "rho", "p"], rows)
    rho30, n30 = per_n.get(30, (float("nan"), 0))
    c_hi = None if (n_hi < 1 or not _ok(rho_hi)) else bool(rho_hi >= hi)
    c_lo = None if (n30 < MIN_CELLS or not _ok(rho30)) else bool(rho30 < lo)
    verdict = _combine([c_hi, c_lo])
    if verdict == "NOT TESTABLE":
        nums["reason"] = (f"{n_hi} cells with n>=300 and {n30} cells at n=30 (need at least {MIN_CELLS} for each clause)")
        en = (f"The split-half agreement cannot be judged: only {n_hi} cells with n of 300 or more and {n30} cells at n=30 are "
              f"available (at least {MIN_CELLS} each are needed). The correlations on these cells are {_f(rho_hi)} and {_f(rho30)}.")
        zh = (f"无法判断半样本一致度：n 不小于 300 的单元只有 {n_hi} 个，n=30 的单元只有 {n30} 个（各至少需要 {MIN_CELLS} 个）。"
              f"这些单元上的相关系数分别为 {_f(rho_hi)} 和 {_f(rho30)}。")
    else:
        en = (f"Split-half agreement is the overlap of the top-100 lists built from two random halves of the reporters. Its Spearman "
              f"rank correlation (1 = same ordering) with the true NDCG@100 is {_f(rho_hi)} (median over the {n_hi} value(s) of n in 300, 1000, 3000 that have at least 12 logs) "
              f"(bar {hi}) and {_f(rho30)} at n=30 over {n30} cells (must be below {lo}). The prediction is {verdict}.")
        zh = (f"半样本一致度是用随机分成的两半汇报者各自建出的前 100 名榜单的重合程度。它与真实 NDCG@100 的秩相关（1 表示排序一致）"
              f"在 n 取 300、1000、3000 且日志数不少于 12 的 {n_hi} 个取值上取中位数为 {_f(rho_hi)}（门槛 {hi}），在 n=30 的 {n30} 个单元上为 {_f(rho30)}（须低于 {lo}）。"
              f"预测{'成立' if verdict == 'MET' else '不成立'}。")
    return _result("D26", verdict, nums, en, zh, [t1, t2])


# ------------------------------------------------------------------ D27

def run_d27(deep, reg, outdir):
    h = _hyp(reg, "D27")
    share_thr, margin = h["prediction"]["share"], h["prediction"]["margin"]
    judged = [300, 400, 534, 700]
    nwins = (30, 45, 60, 90)
    cells = []
    for _, r in _cells(deep, "cal"):
        if r["n"] != 100 or r["n_win"] not in nwins or deep.stratum(r["dataset"]) != "amazon":
            continue
        v = _days_nd(deep, r)
        if np.isfinite(v).all():
            cells.append((r["dataset"], r["n_win"], v))
    cells.sort(key=lambda c: (c[0], c[1]))
    reg_rows, regrets = [], []
    for lg, nw, v in cells:
        best = int(np.argmax(v))
        rg = v[best] - v
        regrets.append(rg)
        reg_rows.append([lg, nw, *v, DAYS_GRID[best], *rg])
    t1 = _csv(outdir, "d27_cells.csv", ["log", "n_win"] + [f"nd100_days{d}" for d in DAYS_GRID] + ["best_days"] +
              [f"regret_days{d}" for d in DAYS_GRID], reg_rows)
    R = np.array(regrets) if regrets else np.zeros((0, len(DAYS_GRID)))
    nc = len(cells)
    nlogs = len({c[0] for c in cells})
    nums = dict(n_cells=nc, n_logs=nlogs, share_required=share_thr, margin=margin, interpretation=(
        "'each of 300, 400, 534, 700 days' read as a separate share per days value (all four must reach the required share); "
        "regret = best of the eight days arms minus that arm, steady NDCG@100 mean over draws; regret <= 0.01 inclusive; "
        "min_logs Amazon logs required. Figure data = d27_mean_regret.csv (no image written)"))
    shares = {}
    for d in DAYS_GRID:
        j = DAYS_GRID.index(d)
        shares[d] = float(np.mean(R[:, j] <= margin + EPS)) if nc else float("nan")
        if d in judged:
            nums[f"share_regret_ok_{d}"] = shares[d]
    nums["share_all_four_ok"] = float(np.mean((R[:, [DAYS_GRID.index(d) for d in judged]] <= margin + EPS).all(axis=1))) if nc else float("nan")
    nums["worst_mean_regret_judged"] = float(max(R[:, DAYS_GRID.index(d)].mean() for d in judged)) if nc else float("nan")
    rows_s = [[d, shares[d], float(R[:, i].mean()) if nc else float("nan"), float(R[:, i].max()) if nc else float("nan")]
              for i, d in enumerate(DAYS_GRID)]
    t2 = _csv(outdir, "d27_mean_regret.csv", ["days", "share_cells_regret_le_margin", "mean_regret", "max_regret"], rows_s)
    rows_b = []
    for w in nwins:
        sel = [reg_rows[i][2 + len(DAYS_GRID)] for i, c in enumerate(cells) if c[1] == w]
        for d in DAYS_GRID:
            rows_b.append([w, d, sum(1 for b in sel if b == d), len(sel)])
    t3 = _csv(outdir, "d27_best_days_by_nwin.csv", ["n_win", "best_days", "n_cells_best", "n_cells"], rows_b)
    if nlogs < reg["min_logs"] or nc == 0:
        verdict, nums["reason"] = "NOT TESTABLE", f"{nlogs} Amazon logs with the full days grid (need at least {reg['min_logs']})"
        en = (f"Only {nlogs} Amazon log(s) ({nc} cells) have the full days grid, fewer than the {reg['min_logs']} required, so the "
              f"flatness of the best half-life cannot be judged. On these cells the share within {margin} NDCG of the best is "
              + ", ".join(f"{d} days {_f(shares[d])}" for d in judged) + ".")
        zh = (f"只有 {nlogs} 个 Amazon 日志（{nc} 个单元）具备完整的天数网格，少于要求的 {reg['min_logs']} 个，无法判断最佳半衰期是否平坦。"
              f"在这些单元上与最佳值相差不超过 {margin} NDCG 的比例为：" + "，".join(f"{d} 天 {_f(shares[d])}" for d in judged) + "。")
    else:
        verdict = "MET" if all(shares[d] >= share_thr for d in judged) else "NOT MET"
        en = (f"On {nc} Amazon cells (log x round length of 3 to 9 months, {nlogs} logs) we measured how much NDCG@100 each half-life in "
              f"days loses against the best of the eight tried values (the regret). The share of cells with regret of at most {margin} "
              f"is " + ", ".join(f"{_f(shares[d])} for {d} days" for d in judged) + f"; every one must reach {share_thr}, so the "
              f"prediction is {verdict}.")
        zh = (f"在 {nc} 个 Amazon 单元（{nlogs} 个日志，每轮 3 到 9 个月）上，我们计算了每个天数半衰期比八个候选值中的最佳者低多少 NDCG@100（遗憾值）。"
              f"遗憾值不超过 {margin} 的单元比例为：" + "，".join(f"{d} 天 {_f(shares[d])}" for d in judged) +
              f"；每一项都须达到 {share_thr}，因此预测{'成立' if verdict == 'MET' else '不成立'}。")
    return _result("D27", verdict, nums, en, zh, [t1, t2, t3])


# ------------------------------------------------------------------ D28

def run_d28(deep, reg, outdir):
    h = _hyp(reg, "D28")
    share_thr, factor = h["prediction"]["share"], h["prediction"]["factor"]
    nwins = (45, 60, 90)
    cal = _cells(deep, "cal")
    new = _new_logs(reg)
    bd = _best_days_by_log(deep, cal, nwins, lambda lg: deep.stratum(lg) == "other")
    # sensitivity: combined days + rounds grid (h rounds * window_days of the cell)
    rowsr = {(r["dataset"], r["n_win"]): r for _, r in cal if r["n"] == 100}
    rows, ratios, ratios_c, grp = [], {}, {}, {}
    censored = []
    for lg, d in sorted(bd.items()):
        edge = any(int(np.argmax(d[w][0])) in (0, len(DAYS_GRID) - 1) for w in nwins)
        flat = any(np.nanmax(d[w][0]) - np.nanmin(d[w][0]) < 0.001 for w in nwins)
        if edge or flat:                                   # best days at the grid edge or all days arms within 0.001: ratio not identified
            censored.append([lg, "edge" if edge else "flat"])
            continue
        best = [DAYS_GRID[int(np.argmax(d[w][0]))] for w in nwins]
        ratio = max(best) / min(best)
        bc = []
        for w in nwins:
            row = rowsr[(lg, w)]
            cand = [(d[w][0][i], float(DAYS_GRID[i])) for i in range(len(DAYS_GRID))]
            for hs in H_GRID[:-1]:
                v = _m(deep, row, f"decay:h={hs}")
                if _ok(v):
                    cand.append((v, float(hs) * row["window_days"]))
            bc.append(max(cand, key=lambda t: t[0])[1])
        ratio_c = max(bc) / min(bc)
        g = "new" if lg in new else "old"
        ratios[lg], ratios_c[lg], grp[lg] = ratio, ratio_c, g
        rows.append([lg, g, *best, ratio, ratio <= factor + EPS, *bc, ratio_c])
    t1 = _csv(outdir, "d28_best_days_by_log.csv",
              ["log", "group", "best_days_45", "best_days_60", "best_days_90", "max_over_min", "within_factor",
               "combined_best_days_45", "combined_best_days_60", "combined_best_days_90", "combined_max_over_min"], rows)
    n_el = len(ratios)
    _csv(outdir, "d28_censored_logs.csv", ["log", "reason"], censored)
    n_ok = sum(1 for v in ratios.values() if v <= factor + EPS)
    share = n_ok / n_el if n_el else float("nan")
    share_c = sum(1 for v in ratios_c.values() if v <= factor + EPS) / n_el if n_el else float("nan")
    srows = [["all", n_el, n_ok, share]]
    nums = dict(n_eligible_logs=n_el, n_logs_within_factor=n_ok, share_within_factor=share, share_required=share_thr, factor=factor,
                share_within_factor_combined_grid=share_c, n_censored=len(censored), interpretation=(
                    "primary: best days among the eight decay:days arms at n=100 for n_win 45, 60, 90, ratio max/min <= 2 per log; "
                    "eligible = non-Amazon logs with all three n_win and all eight arms; at least min_logs eligible logs required. "
                    "Sensitivity: combined grid adds decay:h=0.25..8 arms converted to days (h x days per round)"))
    for g in ("old", "new"):
        ls = [lg for lg in ratios if grp[lg] == g]
        k = sum(1 for lg in ls if ratios[lg] <= factor + EPS)
        srows.append([g, len(ls), k, k / len(ls) if ls else float("nan")])
        nums[f"n_logs_{g}"] = len(ls)
        nums[f"share_{g}"] = k / len(ls) if ls else float("nan")
    arr = np.array(list(ratios.values()))
    nums["median_max_over_min"] = float(np.median(arr)) if len(arr) else float("nan")
    nums["max_max_over_min"] = float(arr.max()) if len(arr) else float("nan")
    t2 = _csv(outdir, "d28_share.csv", ["group", "n_logs", "n_within_factor", "share"], srows)
    if n_el < reg["min_logs"]:
        verdict, nums["reason"] = "NOT TESTABLE", f"{n_el} eligible non-Amazon logs (need at least {reg['min_logs']})"
        en = (f"Only {n_el} non-Amazon log(s) have the days grid at all of 45, 60 and 90 rounds, fewer than the {reg['min_logs']} "
              f"required, so the stability of the best half-life in days cannot be judged.")
        zh = f"只有 {n_el} 个非 Amazon 日志在 45、60、90 轮下都有完整的天数网格，少于要求的 {reg['min_logs']} 个，无法判断最佳半衰期（按天）的稳定性。"
    else:
        verdict = "MET" if share >= share_thr else "NOT MET"
        en = (f"For each of {n_el} non-Amazon logs we found the best half-life in days at 45, 60 and 90 rounds and divided the "
              f"largest by the smallest. {n_ok} logs ({_f(share)}) stay within a factor of {factor:g} against a bar of {share_thr}, "
              f"so the prediction is {verdict}. The median ratio is {_f(nums['median_max_over_min'])} and the largest is "
              f"{_f(nums['max_max_over_min'])}; old logs {_f(nums['share_old'])}, new logs {_f(nums['share_new'])}.")
        zh = (f"对 {n_el} 个非 Amazon 日志，我们分别找出 45、60、90 轮时最佳的半衰期（按天），再用最大值除以最小值。"
              f"有 {n_ok} 个日志（{_f(share)}）的比值不超过 {factor:g} 倍，门槛为 {share_thr}，因此预测{'成立' if verdict == 'MET' else '不成立'}。"
              f"比值中位数为 {_f(nums['median_max_over_min'])}，最大为 {_f(nums['max_max_over_min'])}；旧日志 {_f(nums['share_old'])}，新日志 {_f(nums['share_new'])}。")
    return _result("D28", verdict, nums, en, zh, [t1, t2])


# ------------------------------------------------------------------ D29

def run_d29(deep, reg, outdir):
    h = _hyp(reg, "D29")
    share_thr = h["prediction"]["share"]
    need_gain, harm = 0.01, -0.01
    recs, chosen = [], {"amazon": np.zeros(7), "other": np.zeros(7)}
    for _, r in _cells(deep, "cal"):
        if r["n"] != 100 or not all(deep.has(r, a) for a in [BASE, SELFSEL] + H_ARMS):
            continue
        base = _m(deep, r, BASE)
        sel = _m(deep, r, SELFSEL)
        best_fixed = np.nanmax([_m(deep, r, a) for a in H_ARMS]) if np.isfinite([_m(deep, r, a) for a in H_ARMS]).any() else float("nan")
        gain, d_sel = best_fixed - base, sel - base
        st = deep.stratum(r["dataset"])
        ci = deep.steady(r, SELFSEL, "chosen_h_idx")
        ci = np.rint(ci[np.isfinite(ci)]).astype(int)
        for k in ci:
            if 0 <= k <= 6:
                chosen[st][k] += 1
        recs.append(dict(log=r["dataset"], n_win=r["n_win"], stratum=st, base=base, selfsel=sel, best_fixed=best_fixed, gain=gain,
                         d_sel=d_sel, recovered=bool(d_sel >= 0.5 * gain - EPS) if _ok(gain) and _ok(d_sel) else None,
                         harmful=bool(d_sel < harm) if _ok(d_sel) else None,
                         mean_idx=_m(deep, r, SELFSEL, "chosen_h_idx")))
    t1 = _csv(outdir, "d29_cells.csv",
              ["log", "n_win", "stratum", "nd100_base", "nd100_selfsel", "nd100_best_fixed", "best_fixed_minus_base",
               "selfsel_minus_base", "recovers_half", "harmful", "mean_chosen_h_idx"],
              [[c["log"], c["n_win"], c["stratum"], c["base"], c["selfsel"], c["best_fixed"], c["gain"], c["d_sel"], c["recovered"],
                c["harmful"], c["mean_idx"]] for c in recs])
    oth = [c for c in recs if c["stratum"] == "other"]
    ama = [c for c in recs if c["stratum"] == "amazon"]
    gain_cells = [c for c in oth if c["gain"] >= need_gain - EPS]
    n_rec = sum(1 for c in gain_cells if c["recovered"])
    share_rec = n_rec / len(gain_cells) if gain_cells else float("nan")
    harm_oth = sum(1 for c in oth if c["harmful"])
    harm_ama = sum(1 for c in ama if c["harmful"])
    share_harm_ama = harm_ama / len(ama) if ama else float("nan")
    logs_oth = len({c["log"] for c in oth})
    logs_ama = len({c["log"] for c in ama})
    ml = reg["min_logs"]
    nums = dict(n_cells_other=len(oth), n_logs_other=logs_oth, n_gain_cells_other=len(gain_cells), n_recovered_other=n_rec,
                share_recovered_other=share_rec, n_harmful_other=harm_oth, n_cells_amazon=len(ama), n_logs_amazon=logs_ama,
                n_harmful_amazon=harm_ama, share_harmful_amazon=share_harm_ama, share_required=share_thr, interpretation=(
                    "best fixed = best of the seven decay:h arms (0.25..inf, includes the baseline h=2); gain = best fixed - baseline; "
                    "clause 1 (non-Amazon, cells with gain >= 0.01): share with selfsel-base >= 0.5 x gain is >= 0.6; clause 2 (all "
                    "non-Amazon cells): none with selfsel-base < -0.01; clause 3 (Amazon cells): share with selfsel-base < -0.01 >= 0.3. "
                    "Share clauses need at least min_logs logs of the stratum and at least one qualifying cell; a harmful non-Amazon "
                    "cell falsifies clause 2 at any size"))
    c1 = None if (logs_oth < ml or not gain_cells) else bool(share_rec >= share_thr)
    c2 = False if harm_oth > 0 else (None if logs_oth < ml else True)
    c3 = None if (logs_ama < ml or not ama) else bool(share_harm_ama >= 0.3)
    verdict = _combine([c1, c2, c3])
    rows = []
    for w in sorted({c["n_win"] for c in recs}):
        for st in ("other", "amazon"):
            cs = [c for c in recs if c["n_win"] == w and c["stratum"] == st]
            gc = [c for c in cs if c["gain"] >= need_gain - EPS]
            rows.append([w, st, len(cs), len(gc), sum(1 for c in gc if c["recovered"]), sum(1 for c in cs if c["harmful"])])
    t2 = _csv(outdir, "d29_by_nwin.csv", ["n_win", "stratum", "n_cells", "n_gain_cells", "n_recovered", "n_harmful"], rows)
    crow = []
    for st in ("amazon", "other"):
        tot = chosen[st].sum()
        for k, lab in enumerate(H_GRID):
            crow.append([st, k, lab, chosen[st][k], chosen[st][k] / tot if tot else float("nan")])
    t3 = _csv(outdir, "d29_chosen_h.csv", ["stratum", "chosen_h_idx", "half_life_rounds", "n_draws", "share_draws"], crow)
    for st in ("amazon", "other"):
        tot = chosen[st].sum()
        nums[f"mean_chosen_h_idx_{st}"] = float((chosen[st] * np.arange(7)).sum() / tot) if tot else float("nan")
    if verdict == "NOT TESTABLE":
        nums["reason"] = (f"non-Amazon: {logs_oth} logs, {len(gain_cells)} cells with a gain of 0.01 or more; Amazon: {logs_ama} logs "
                          f"(need at least {ml} logs per stratum and at least one gain cell)")
        en = (f"The self-tuning half-life cannot be fully judged: the data hold {logs_oth} non-Amazon log(s) ({len(gain_cells)} cells where a "
              f"fixed half-life gains 0.01 or more over half-life 2) and {logs_ama} Amazon log(s) (at least {ml} logs per group are needed). "
              f"Where it can be measured, the self-tuning server is worse than half-life 2 by more than 0.01 in {_f(share_harm_ama)} of "
              f"{len(ama)} Amazon cells and in {harm_oth} non-Amazon cells.")
        zh = (f"无法完整判断自调半衰期：数据中有 {logs_oth} 个非 Amazon 日志（其中固定半衰期比 2 好 0.01 以上的单元有 {len(gain_cells)} 个）"
              f"和 {logs_ama} 个 Amazon 日志（每组至少需要 {ml} 个日志）。能测的部分是：自调服务器比半衰期 2 差 0.01 以上的"
              f"在 {len(ama)} 个 Amazon 单元中占 {_f(share_harm_ama)}，在非 Amazon 单元中有 {harm_oth} 个。")
    else:
        en = (f"On non-Amazon logs, among {len(gain_cells)} cells where the best fixed half-life beats half-life 2 by 0.01 NDCG or more, "
              f"the self-tuning server recovers at least half of that gain in {n_rec} ({_f(share_rec)}; bar {share_thr}), and it is worse "
              f"than half-life 2 by more than 0.01 in {harm_oth} of {len(oth)} non-Amazon cells (must be 0). On Amazon logs it is that much "
              f"worse in {harm_ama} of {len(ama)} cells ({_f(share_harm_ama)}; bar 0.3). The prediction is {verdict}.")
        zh = (f"在非 Amazon 日志上，最佳固定半衰期比半衰期 2 好 0.01 NDCG 以上的单元有 {len(gain_cells)} 个，自调服务器在其中 {n_rec} 个（{_f(share_rec)}；门槛 {share_thr}）"
              f"追回了至少一半的收益；在 {len(oth)} 个非 Amazon 单元中，比半衰期 2 差 0.01 以上的有 {harm_oth} 个（须为 0）。"
              f"在 Amazon 日志上，{len(ama)} 个单元中有 {harm_ama} 个（{_f(share_harm_ama)}；门槛 0.3）差这么多。预测{'成立' if verdict == 'MET' else '不成立'}。")
    return _result("D29", verdict, nums, en, zh, [t1, t2, t3])


# ------------------------------------------------------------------ D30

def run_d30(deep, reg, outdir):
    h = _hyp(reg, "D30")
    thr = h["prediction"]["rho"]
    flip_min = 0.005
    metrics = ["nd100", "nd10", "nd30", "nd300", "rec100"]
    cells = [r for _, r in _cells(deep, "agg") if r["n"] == 100 and deep.has(r, BASE)]
    logs = sorted({r["dataset"] for r in cells})
    arms = sorted({a for r in cells for a in r["arms"] if a != BASE}, key=lambda a: [x for r in cells for x in r["arms"]].index(a))
    D = {}                                                   # (arm, metric) -> mean over logs of per-log diff
    nlog_arm = {}
    for a in arms:
        for m in metrics:
            per_log = {}
            for r in cells:
                if deep.has(r, a):
                    d = _m(deep, r, a, m) - _m(deep, r, BASE, m)
                    if _ok(d):
                        per_log.setdefault(r["dataset"], []).append(d)
            v = [np.mean(x) for x in per_log.values()]
            D[(a, m)] = float(np.mean(v)) if v else float("nan")
            if m == "nd100":
                nlog_arm[a] = len(v)
    arms = [a for a in arms if not a.startswith(("fulldecay", "cens")) and _ok(D[(a, "nd100")]) and abs(D[(a, "nd100")]) <= 0.02]   # no oracle/census arms, no large-effect arms
    rows, flips = [], {m: [] for m in metrics[1:]}
    for a in arms:
        d100 = D[(a, "nd100")]
        fl = []
        for m in metrics[1:]:
            dm = D[(a, m)]
            if _ok(d100) and _ok(dm) and abs(d100) > flip_min and d100 * dm < 0:
                flips[m].append(a)
                fl.append(m)
        rows.append([a, nlog_arm[a]] + [D[(a, m)] for m in metrics] + [";".join(fl)])
    t1 = _csv(outdir, "d30_arm_diffs.csv", ["arm", "n_logs"] + [f"diff_{m}" for m in metrics] + ["sign_flips_in"], rows)
    nums = dict(n_logs=len(logs), n_arms=len(arms), threshold=thr, flip_min_abs_diff=flip_min, interpretation=(
        "arms = every arm in stage agg except the baseline, the oracle and census arms (fulldecay, cens, censold) and arms with |mean NDCG@100 diff| > 0.02; diff = arm minus baseline at n=100, "
        "mean over logs of the per-log steady mean; Spearman over arms between NDCG@100 and each other metric; a sign flip = opposite "
        "sign (a zero is not a flip) with |diff under NDCG@100| > 0.005; MET needs all four rho >= 0.9 and no flip under any metric; "
        "at least min_logs logs required"))
    srows, rhos = [], {}
    for m in metrics[1:]:
        rr, pp, nn = _spear([D[(a, "nd100")] for a in arms], [D[(a, m)] for a in arms])
        rhos[m] = rr
        nums[f"rho_{m}"] = rr
        nums[f"n_flips_{m}"] = len(flips[m])
        srows.append([m, nn, rr, pp, len(flips[m]), ";".join(flips[m])])
    flipped = sorted({a for v in flips.values() for a in v})
    nums["n_arms_with_flip"] = len(flipped)
    t2 = _csv(outdir, "d30_spearman.csv", ["metric", "n_arms", "rho_vs_nd100", "p", "n_sign_flips", "flipped_arms"], srows)
    if len(logs) < reg["min_logs"] or len(arms) < 5 or not all(_ok(v) for v in rhos.values()):
        verdict, nums["reason"] = "NOT TESTABLE", f"{len(logs)} logs and {len(arms)} arms at n=100 (need at least {reg['min_logs']} logs)"
        en = (f"Only {len(logs)} log(s) at n=100 are available, fewer than the {reg['min_logs']} required, so metric dependence cannot be "
              f"judged. On this log the rank correlations with NDCG@100 are " + ", ".join(f"{_f(rhos[m])} for {m}" for m in metrics[1:]) +
              f", and {len(flipped)} arms change sign.")
        zh = (f"在 n=100 下只有 {len(logs)} 个日志，少于要求的 {reg['min_logs']} 个，无法判断结论是否依赖指标。在这些数据上，与 NDCG@100 的秩相关为："
              + "，".join(f"{m} {_f(rhos[m])}" for m in metrics[1:]) + f"；有 {len(flipped)} 个聚合方式的符号发生反转。")
    else:
        ok = all(rhos[m] >= thr for m in metrics[1:]) and not flipped
        verdict = "MET" if ok else "NOT MET"
        en = (f"We ranked {len(arms)} aggregation arms by their mean NDCG@100 difference from the baseline over {len(logs)} logs at n=100, "
              f"and repeated this with NDCG@10, NDCG@30, NDCG@300 and recall@100. The Spearman rank correlations (1 = same ordering) with the "
              f"NDCG@100 ordering are " + ", ".join(f"{_f(rhos[m])} ({m})" for m in metrics[1:]) + f", all required to be at least {thr}. "
              f"{len(flipped)} arms flip sign with an NDCG@100 difference above {flip_min} (must be 0), so the prediction is {verdict}.")
        zh = (f"我们在 n=100、{len(logs)} 个日志上，按各聚合方式相对基线的平均 NDCG@100 差值给 {len(arms)} 个方式排序，并用 NDCG@10、NDCG@30、NDCG@300 和 recall@100 重复。"
              f"它们与 NDCG@100 排序的秩相关（1 表示排序一致）分别为 " + "，".join(f"{m} {_f(rhos[m])}" for m in metrics[1:]) + f"，均须不低于 {thr}。"
              f"NDCG@100 差值绝对值大于 {flip_min} 而符号反转的方式有 {len(flipped)} 个（须为 0），因此预测{'成立' if verdict == 'MET' else '不成立'}。")
    return _result("D30", verdict, nums, en, zh, [t1, t2])


RUNNERS = {"D24": run_d24, "D25": run_d25, "D26": run_d26, "D27": run_d27, "D28": run_d28, "D29": run_d29, "D30": run_d30}


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from an_deep_lib import Deep
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    reg_ = json.load(open(os.path.join(root, "configs", "deep_registry.json")))
    path_ = sys.argv[1] if len(sys.argv) > 1 else "/tmp/deep_smoke.jsonl"
    out_ = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.environ.get("TMPDIR", "/tmp"), "an_deep_calc2_out")
    dp = Deep(path_, reg_)
    for k_, fn_ in RUNNERS.items():
        res = fn_(dp, reg_, out_)
        json.dumps(res)                                           # must be serialisable
        print(k_, res["verdict"])
        print("  numbers:", json.dumps(res["numbers"], ensure_ascii=False))
        print("  EN:", res["summary_en"])
        print("  ZH:", res["summary_zh"])
        print("  tables:", res["tables"])
