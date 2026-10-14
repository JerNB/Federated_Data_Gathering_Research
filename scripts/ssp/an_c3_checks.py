"""Two companions to the confirmatory claims C2 and C3 (amendment A15 and the left-censoring note).

  (1) in-universe: the same C2(a) share and C3 N* table, but on the in-universe NDCG (`*.U_ndcg_in`), which only
      counts relevant items a model could reach (inside the 1,500-item candidate universe).
  (2) small N: C3's N* with the grid extended downward by the e2s stage (N = 20, 40, 60), because every k = all
      crossover found at the smallest registered N (100) is only known to be "at or below 100".

Reads  <runs>/e2.jsonl and <runs>/e2s.jsonl     Writes <out>/03_tables/an_c3_checks_*.csv, an_c3_checks_verdicts.json,
<out>/04_figures/an_c3_checks_*.png.   Exploratory companions: the registered verdicts stay those of an_confirm2.py.
N* rule (A12): smallest N where the paired 95% interval of (personalised_time - popularity) is above 0 AND the mean
difference stays above 0 at every larger tested N.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics  # noqa: E402

DATASETS = ["ml-latest", "ml-1m", "amazon-video-games", "amazon-musical-instruments", "amazon-software", "kuairand-pure", "gowalla"]
KS = [10, None]


def kn(k):
    return "all" if k is None else str(k)


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


def cells(rows, f_pers, f_pop):
    d = defaultdict(list)
    for r in rows:
        if r.get(f_pers) is not None and r.get(f_pop) is not None:
            d[(r["dataset"], r.get("k"), r["N"])].append(r[f_pers] - r[f_pop])
    return d


def nstar(diffs_by_n, rng):
    """diffs_by_n: {N: list of per-draw differences}. Returns (N* or None, [(N, mean, lo, hi)])."""
    ns = sorted(diffs_by_n)
    stats = []
    for n in ns:
        v = np.asarray(diffs_by_n[n], float)
        m, lo, hi = metrics.paired_bootstrap_ci(v, n_boot=2000, rng=rng)
        stats.append((n, m, lo, hi))
    for i, (n, m, lo, hi) in enumerate(stats):
        if lo > 0 and all(x[1] > 0 for x in stats[i:]):
            return n, stats
    return None, stats


def fmt(n, ns):
    if n is None:
        return f"none (> {max(ns)})" if ns else "n/a"
    return f"<= {n} (smallest tested)" if n == min(ns) else str(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/small_sample_popularity/02_runs")
    ap.add_argument("--out", default="results/small_sample_popularity")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/03_tables", exist_ok=True)
    os.makedirs(f"{a.out}/04_figures", exist_ok=True)
    e2, e2s = read(f"{a.runs}/e2.jsonl"), read(f"{a.runs}/e2s.jsonl")
    rng = np.random.default_rng(20261014)
    verdicts, table = [], []
    # ---------------- (1) in-universe
    std = cells(e2, "pers_time.U_ndcg", "pop.U_ndcg")
    inu = cells(e2, "pers_time.U_ndcg_in", "pop.U_ndcg_in")
    n_std, n_in = {}, {}
    for ds in DATASETS:
        for k in KS:
            for tag, src, store in (("standard", std, n_std), ("in_universe", inu, n_in)):
                by = {n: v for (d, kk, n), v in src.items() if d == ds and kk == k and len(v) >= 5}
                if by:
                    store[(ds, k)] = nstar(by, rng)[0]
                    table.append(dict(check="in_universe", dataset=ds, k=("all" if k is None else k), metric=tag,
                                      N_star=fmt(store[(ds, k)], list(by))))
    changed = [(ds, k) for (ds, k) in n_in if (ds, k) in n_std and n_in[(ds, k)] != n_std[(ds, k)]]
    cnt = lambda store, k: sum(1 for ds in DATASETS if store.get((ds, k)) is not None)
    if n_in:
        verdicts.append(dict(
            id="C3-in-universe", title_en="C3 crossover on the in-universe score (companion to C3, amendment A15)",
            title_zh="C3 交叉样本量：候选集内得分版（C3 的配套，修订 A15）", verdict="not testable" if len(n_in) < 14 else "mixed",
            numbers=dict(exists_kall_standard=cnt(n_std, None), exists_kall_in_universe=cnt(n_in, None),
                         exists_k10_standard=cnt(n_std, 10), exists_k10_in_universe=cnt(n_in, 10),
                         series_that_change=[f"{ds}/k={'all' if k is None else k}" for ds, k in changed]),
            reading_en=(f"Counting only relevant items a model can reach (inside the 1,500-item candidate universe), N* exists at k=all in "
                        f"{cnt(n_in, None)} of 7 datasets (standard score: {cnt(n_std, None)}) and at k=10 in {cnt(n_in, 10)} (standard: {cnt(n_std, 10)}). "
                        f"Series whose answer changes: {', '.join(f'{ds}/k={kn(k)}' for ds, k in changed) or 'none'}. "
                        f"This shows how much of the 'popularity wins' result is the candidate cap talking."),
            reading_zh=(f"只统计模型够得着的相关物品（候选集 1,500 个物品之内）时，k=all 下 7 个数据集中 {cnt(n_in, None)} 个存在 N*"
                        f"（标准得分：{cnt(n_std, None)} 个），k=10 下 {cnt(n_in, 10)} 个（标准：{cnt(n_std, 10)} 个）。结果发生变化的序列："
                        f"{', '.join(f'{ds}/k={kn(k)}' for ds, k in changed) or '无'}。"
                        f"这说明'热门更好'的结果有多少是候选集上限造成的。"),
            rule="Same N* rule as C3 (A12) on U_ndcg_in; companion, not a registered verdict."))
    # ---------------- (2) small N
    allrows = e2 + e2s
    ext = cells(allrows, "pers_time.U_ndcg", "pop.U_ndcg")
    n_ext = {}
    curves = {}
    for ds in (DATASETS if e2s else []):     # only meaningful once the e2s stage exists
        for k in KS:
            by = {n: v for (d, kk, n), v in ext.items() if d == ds and kk == k and len(v) >= 5}
            if not by:
                continue
            n, stats = nstar(by, rng)
            n_ext[(ds, k)] = n
            curves[(ds, k)] = stats
            table.append(dict(check="small_N", dataset=ds, k=("all" if k is None else k), metric="standard, grid from N=20",
                              N_star=fmt(n, list(by))))
    if e2s and n_ext:
        lower = [(ds, k) for (ds, k) in n_ext if n_std.get((ds, k)) is not None and n_ext[(ds, k)] is not None
                 and n_ext[(ds, k)] < n_std[(ds, k)]]
        verdicts.append(dict(
            id="C3-small-N", title_en="Where does the crossover really start? C3 with the grid extended down to N=20",
            title_zh="交叉点到底从哪里开始？把 C3 的网格向下延伸到 N=20", verdict="mixed",
            numbers=dict(exists_kall=sum(1 for ds in DATASETS if n_ext.get((ds, None)) is not None),
                         exists_k10=sum(1 for ds in DATASETS if n_ext.get((ds, 10)) is not None),
                         series_with_lower_crossover=len(lower),
                         kall_N_star={ds: fmt(n_ext.get((ds, None)), [n for (d, kk, n) in ext if d == ds and kk is None]) for ds in DATASETS}),
            reading_en=("With three smaller sample sizes added (20, 40, 60 people), the crossover at k=all moves below 100 in "
                        f"{sum(1 for ds, k in lower if k is None)} datasets; "
                        + "; ".join(f"{ds}: {fmt(n_ext.get((ds, None)), [n for (d, kk, n) in ext if d == ds and kk is None])}" for ds in DATASETS) + "."),
            reading_zh=("加入三个更小的样本量（20、40、60 人）后，k=all 时交叉点移到 100 以下的数据集有 "
                        f"{sum(1 for ds, k in lower if k is None)} 个；"
                        + "；".join(f"{ds}：{fmt(n_ext.get((ds, None)), [n for (d, kk, n) in ext if d == ds and kk is None])}" for ds in DATASETS) + "。"),
            rule="N* as in A12, on the grid {20, 40, 60} + the registered grid; exploratory."))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
        for ax, k in zip(axes, KS):
            for i, ds in enumerate(DATASETS):
                st = curves.get((ds, k))
                if not st:
                    continue
                ns = [x[0] for x in st]
                ax.plot(ns, [x[1] for x in st], "-o", ms=3, label=ds)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xscale("log")
            ax.set_xlabel("people sampled N (log)")
            ax.set_title(f"k = {'all' if k is None else k}", fontsize=10)
        axes[0].set_ylabel("personalised(time) minus popularity, panel NDCG@100")
        axes[1].legend(fontsize=6)
        fig.suptitle("Personalised minus popularity on the panel, with sample sizes down to N=20", fontsize=10)
        fig.tight_layout()
        fig.savefig(f"{a.out}/04_figures/an_c3_checks_small_n.png", dpi=130)
    if table:
        with open(f"{a.out}/03_tables/an_c3_checks_nstar.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    json.dump(verdicts, open(f"{a.out}/03_tables/an_c3_checks_verdicts.json", "w"), ensure_ascii=False, indent=1)
    for v in verdicts:
        print(v["id"], v["verdict"], v["reading_en"][:300])
    if not verdicts:
        print("nothing to report yet (needs e2 with U_ndcg_in, and e2s)")


if __name__ == "__main__":
    main()
