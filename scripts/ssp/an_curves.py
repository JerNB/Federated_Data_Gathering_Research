"""X17 (exploratory): two speeds of learning. Does the popularity list stop improving with N long before a personalised model does?

Reads  <runs>/e1.jsonl (count.A_ndcg100, count.U_ndcg, k = all or numeric)  and  <runs>/e2.jsonl (pop.U_ndcg, pers_time.U_ndcg)
Writes <out>/03_tables/an_curves_*.csv, an_curves_verdicts.json, <out>/04_figures/an_curves_*.png

Definitions (fixed here, before the full data were read):
  N95(curve)  = smallest tested N at which the mean curve reaches 95% of its own value at the largest tested N
                (curves are only compared on datasets where that largest N is the same in both).
  popularity-saturation claim (X17): in more than half of the judged datasets, N95(popularity, panel) is at least
  two grid steps below N95(best pure personalised, panel), at k = all. Exploratory.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read(path):
    out = []
    if os.path.exists(path):
        for ln in open(path):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if "error" not in r:
                out.append(r)
    return out


def cellmeans(rows, field, k):
    d = defaultdict(list)
    for r in rows:
        if r.get("k") == k and r.get(field) is not None and r.get("arm", "uniform") == "uniform":
            d[(r["dataset"], r["N"])].append(r[field])
    by = defaultdict(dict)
    for (ds, N), v in d.items():
        by[ds][N] = float(np.mean(v))
    return by


def n95(curve, frac=0.95):
    Ns = sorted(curve)
    if len(Ns) < 3:
        return None, None
    top = curve[Ns[-1]]
    if top <= 0:
        return None, None
    for N in Ns:
        if curve[N] >= frac * top:
            return N, Ns[-1]
    return None, Ns[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/small_sample_popularity/02_runs")
    ap.add_argument("--out", default="results/small_sample_popularity")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/03_tables", exist_ok=True)
    os.makedirs(f"{a.out}/04_figures", exist_ok=True)
    e1, e2 = read(f"{a.runs}/e1.jsonl"), read(f"{a.runs}/e2.jsonl")
    vid = {"id": "X17", "title_en": "Two speeds: the popularity list saturates early, a personalised model keeps improving (exploratory)",
           "title_zh": "两种速度：热门榜很早就饱和，个性化模型还在继续提升（探索性）",
           "rule": "In more than half of the judged datasets, the sample size where the popularity list reaches 95% of its own best is at least two grid steps below that of the best pure personalised model (k = all, panel NDCG@100)."}
    pop = cellmeans(e2, "pop.U_ndcg", None)
    pers = cellmeans(e2, "pers_time.U_ndcg", None)
    estA = cellmeans(e1, "count.A_ndcg100", None)
    rows, judged, yes = [], 0, 0
    grid = [100, 200, 400, 1000, 2000, 4000]
    for ds in sorted(set(pop) & set(pers)):
        p, pm = n95(pop[ds])
        q, qm = n95(pers[ds])
        e, em = n95(estA.get(ds, {}), 0.95)
        steps = None
        if p and q:
            steps = (sorted(pers[ds]).index(q) - sorted(pop[ds]).index(p))
            judged += 1
            yes += steps >= 2
        rows.append(dict(dataset=ds, N_max=pm, N95_popularity_panel=p, N95_personalised_panel=q, steps_apart=steps,
                         N95_list_vs_truthA=e, panel_gain_pop_last_vs_first=round(pop[ds][max(pop[ds])] / pop[ds][min(pop[ds])], 3),
                         panel_gain_pers_last_vs_first=round(pers[ds][max(pers[ds])] / max(pers[ds][min(pers[ds])], 1e-9), 3)))
    if rows:
        with open(f"{a.out}/03_tables/an_curves_n95.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if judged == 0:
        verdict = "not testable"
    else:
        verdict = "supported" if yes / judged > 0.5 else ("not supported" if yes == 0 else "mixed")
    gp = [r["panel_gain_pop_last_vs_first"] for r in rows]
    gq = [r["panel_gain_pers_last_vs_first"] for r in rows]
    nums = dict(datasets_judged=judged, datasets_two_or_more_steps=yes,
                median_gain_pop=float(np.median(gp)) if gp else None, median_gain_pers=float(np.median(gq)) if gq else None)
    out = [dict(vid, verdict=verdict, numbers=nums,
                reading_en=(f"From N=100 to the largest tested N, the popularity list's panel score grows by a median factor of {nums['median_gain_pop']:.2f} "
                            f"while the best personalised model's grows by {nums['median_gain_pers']:.2f}; the popularity list reached 95% of its best at least two grid steps earlier in {yes} of {judged} datasets."
                            if gp else "Not enough data yet."),
                reading_zh=(f"从 N=100 到最大被测 N，热门榜在验证小组上的得分中位数增长 {nums['median_gain_pop']:.2f} 倍，"
                            f"而最好的纯个性化模型增长 {nums['median_gain_pers']:.2f} 倍；{judged} 个数据集中，有 {yes} 个里热门榜至少提早两个网格步就达到了自身最好成绩的 95%。"
                            if gp else "数据还不够。"))]
    json.dump(out, open(f"{a.out}/03_tables/an_curves_verdicts.json", "w"), ensure_ascii=False, indent=1)
    # figure: curves normalised by their own value at the largest N
    if rows:
        ds_list = [r["dataset"] for r in rows]
        fig, axes = plt.subplots(1, len(ds_list), figsize=(2.6 * len(ds_list), 3.2), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, ds in zip(axes, ds_list):
            for lab, cur, col in (("popularity", pop[ds], "#000000"), ("best pure personalised", pers[ds], "#0072B2")):
                Ns = sorted(cur)
                ax.plot(Ns, [cur[n] / cur[Ns[-1]] for n in Ns], "-o", ms=3, color=col, label=lab)
            ax.axhline(0.95, color="grey", lw=0.6, ls=":")
            ax.set_xscale("log")
            ax.set_title(ds, fontsize=8)
            ax.set_xlabel("people sampled N", fontsize=8)
        axes[0].set_ylabel("panel NDCG@100,\nshare of own value at largest N", fontsize=8)
        axes[0].legend(fontsize=7)
        earlier = sum(1 for r in rows if (r["steps_apart"] or 0) >= 1)
        fig.suptitle(f"The popularity list reaches 95% of its best at a smaller N than the personalised model in {earlier} of {len(rows)} datasets (k = all)", fontsize=9)
        fig.tight_layout()
        fig.savefig(f"{a.out}/04_figures/an_curves_two_speeds.png", dpi=130)
    print("X17:", verdict, nums)


if __name__ == "__main__":
    main()
