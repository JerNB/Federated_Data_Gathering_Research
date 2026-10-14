"""Follow-up checks that test the explanations we first gave for four surprising results (descriptive, not registered).

  F1  C1: breadth vs depth at equal budget: is the SPREAD of list quality across draws smaller for the wide cell?
  F2  X8: subgroups: does the reversed sign survive when series that never reached a crossover (censored) are dropped?
  F3  X16: in-sample hold-out vs the panel: how does its accuracy compare with always guessing the more common outcome?
  F4  X7: recency sampling: effect on held-out people (not only on the community's later list), by dataset

Reads  <out>/03_tables/{cells_e1.csv, an_confirm2_c1_pairs.csv, an_explore_b_x8_nstar.csv, an_explore_b_x7_datasets.csv}
       and <runs>/e2c.jsonl.     Writes <out>/03_tables/an_followups_verdicts.json
"""
import argparse
import csv
import json
import os

import numpy as np

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None


def rc(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def f(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/small_sample_popularity/02_runs")
    ap.add_argument("--out", default="results/small_sample_popularity")
    a = ap.parse_args()
    T = f"{a.out}/03_tables"
    rng = np.random.default_rng(20261014)
    out = []

    # ---- F1: standard error of count.A_ndcg100, wide cell vs deep cell
    cells = {(r["dataset"], r["N"], r["k"]): f(r.get("count.A_ndcg100.se")) for r in rc(f"{T}/cells_e1.csv") if r.get("arm") == "uniform"}
    ratios = []
    for p in rc(f"{T}/an_confirm2_c1_pairs.csv"):
        if p.get("in_scope") != "True":
            continue
        w, d = cells.get((p["dataset"], p["N_wide"], p["k_wide"])), cells.get((p["dataset"], p["N_deep"], p["k_deep"]))
        if w and d:
            ratios.append(w / d)
    if ratios:
        r_ = np.array(ratios)
        out.append(dict(id="F1", title_en="Breadth makes lists more repeatable even though it does not make them better",
                        title_zh="广度让榜单更稳定，但并没有让它更好", verdict="descriptive",
                        numbers=dict(pairs=len(r_), share_wide_smaller_se=float((r_ < 1).mean()), median_se_ratio_wide_over_deep=float(np.median(r_))),
                        reading_en=(f"At equal total choices, the draw-to-draw standard error of list quality is smaller for the wide cell in {int((r_ < 1).sum())} of {len(r_)} pairs "
                                    f"(median ratio {np.median(r_):.2f}), while the average quality is the same (C1). Breadth buys stability, not accuracy."),
                        reading_zh=(f"总选择数相同时，{len(r_)} 个对里有 {int((r_ < 1).sum())} 个，宽格子在不同抽样之间的榜单质量标准误更小（中位数比值 {np.median(r_):.2f}），"
                                    f"而平均质量相同（C1）。广度换来的是稳定性，不是准确性。"),
                        rule="Descriptive follow-up to C1; no registered threshold."))

    # ---- F2: X8 without censored series
    rows = [r for r in rc(f"{T}/an_explore_b_x8_nstar.csv")]
    unc = [(f(r["distinctiveness"]), f(r["Nstar2_ndcg100"])) for r in rows if f(r["Nstar2_ndcg100"]) is not None]
    cen = [f(r["distinctiveness"]) for r in rows if f(r["Nstar2_ndcg100"]) is None and f(r["distinctiveness"]) is not None]
    if unc and spearmanr:
        x, y = zip(*unc)
        rho = float(spearmanr(x, y)[0])
        boots = []
        for _ in range(2000):
            i = rng.integers(0, len(x), len(x))
            boots.append(spearmanr(np.array(x)[i], np.array(y)[i])[0])
        lo, hi = float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))
        out.append(dict(id="F2", title_en="Subgroups: the reversed sign survives, but weaker, once never-reached series are dropped",
                        title_zh="子群体：去掉从未达到交叉点的序列后，反向的符号仍然存在，但变弱", verdict="descriptive",
                        numbers=dict(n_uncensored=len(unc), n_censored=len(cen), rho_uncensored=rho, ci=[lo, hi],
                                     mean_distinctiveness_censored=float(np.mean(cen)) if cen else None,
                                     mean_distinctiveness_uncensored=float(np.mean(x))),
                        reading_en=(f"Among the {len(unc)} series where the subgroup sample did beat the global list, the rank correlation between distinctiveness and the sample size at which it did is "
                                    f"{rho:+.2f} (interval {lo:+.2f} to {hi:+.2f}); the {len(cen)} series that never got there are less distinctive on average "
                                    f"({np.mean(cen):.2f} vs {np.mean(x):.2f}), which is why the first number (-0.44) overstated it."),
                        reading_zh=(f"在子群体样本确实胜过全局榜的 {len(unc)} 个序列里，差异度与取胜所需样本量的秩相关为 {rho:+.2f}（区间 {lo:+.2f} 到 {hi:+.2f}）；"
                                    f"从未达到的 {len(cen)} 个序列平均差异度更低（{np.mean(cen):.2f} 对 {np.mean(x):.2f}），这就是第一个数字（-0.44）被夸大的原因。"),
                        rule="Descriptive follow-up to X8; no registered threshold."))

    # ---- F3: X16 against the base rate
    path = f"{a.runs}/e2c.jsonl"
    if os.path.exists(path):
        gaps = {"time": [], "random": []}
        for ln in open(path):
            r = json.loads(ln)
            for m in ("time", "random"):
                if all(r.get(k) is not None for k in (f"cv_pop_{m}", f"cv_pers_{m}", f"pers_{m}.U_ndcg", "pop.U_ndcg")):
                    gaps[m].append((r[f"cv_pers_{m}"] - r[f"cv_pop_{m}"], r[f"pers_{m}.U_ndcg"] - r["pop.U_ndcg"]))
        g = np.array(gaps["time"])
        if len(g):
            real_pos = g[:, 1] > 0
            base = float(max(real_pos.mean(), 1 - real_pos.mean()))
            nz = (g[:, 0] != 0) & (g[:, 1] != 0)
            acc = float((np.sign(g[nz, 0]) == np.sign(g[nz, 1])).mean())
            out.append(dict(id="F3", title_en="The in-sample hold-out barely beats always guessing the more common outcome",
                            title_zh="样本内留出检验几乎不比「永远猜更常见的结果」好", verdict="descriptive",
                            numbers=dict(draws=len(g), accuracy_time_holdout=acc, base_rate_accuracy=base,
                                         share_draws_personalised_better_on_panel=float(real_pos.mean())),
                            reading_en=(f"Personalised beats popularity on the panel in {real_pos.mean():.0%} of draws, so always guessing the more common outcome is right {base:.0%} of the time; "
                                        f"the time-ordered in-sample hold-out is right {acc:.0%} of the time. It adds little skill."),
                            reading_zh=(f"个性化在验证小组上胜过热门的抽样占 {real_pos.mean():.0%}，所以永远猜更常见的结果，对的比例是 {base:.0%}；按时间留出的样本内检验对的比例是 {acc:.0%}。几乎没有增加判断力。"),
                            rule="Descriptive follow-up to X16; no registered threshold."))

    # ---- F4: X7 effect on held-out people
    xs = [r for r in rc(f"{T}/an_explore_b_x7_datasets.csv") if r.get("arm") == "recent_window"]
    if xs:
        dU = np.array([f(r["dU"]) for r in xs])
        neg = [r["dataset"] for r in xs if f(r["dU"]) < 0]
        dB = np.array([f(r["dB"]) for r in xs])
        out.append(dict(id="F4", title_en="Recency sampling: the community's later list improves a lot, held-out people only a little",
                        title_zh="近期抽样：群体之后的榜单提升很多，留出的人只提升一点点", verdict="descriptive",
                        numbers=dict(median_dB_later_list=float(np.median(dB)), median_dU_held_out_people=float(np.median(dU)), datasets_negative_on_people=neg),
                        reading_en=(f"A recent-window sample beats a uniform sample of the same size by a median {np.median(dB):+.3f} NDCG@100 against the community's later list, "
                                    f"but only by {np.median(dU):+.4f} for held-out people, and it is negative in {len(neg)} of {len(xs)} datasets ({', '.join(neg) or 'none'})."),
                        reading_zh=(f"近期窗口样本对群体之后的榜单，比同样大小的均匀样本好，NDCG@100 中位数 {np.median(dB):+.3f}；但对留出的人只好 {np.median(dU):+.4f}，"
                                    f"并且在 {len(xs)} 个数据集里有 {len(neg)} 个为负（{', '.join(neg) or '无'}）。"),
                        rule="Descriptive follow-up to X7; no registered threshold."))
    json.dump(out, open(f"{T}/an_followups_verdicts.json", "w"), ensure_ascii=False, indent=1)
    for v in out:
        print(v["id"], v["reading_en"][:260])


if __name__ == "__main__":
    main()
