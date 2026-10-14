"""X16 (exploratory): can the sample itself say when to switch from the popularity list to a personalised model?

For every draw the sample hides part of each sampled person's choices (random rows, or each person's latest rows)
and measures how much a personalised model beats popularity on what it hid: the in-sample gap. We ask whether the
SIGN of that gap predicts the sign of the real gap on the held-out panel.

Reads  <runs>/e2c.jsonl   (fields cv_pop_<mode>, cv_pers_<mode>, pop.U_ndcg, pers_<mode>.U_ndcg; mode = random | time)
Writes <out>/03_tables/an_stoprule_*.csv, an_stoprule_verdicts.json, <out>/04_figures/an_stoprule_*.png
Rule (fixed in advance, pre-registration X16 as added with e2c): the time-ordered in-sample gap predicts the sign of
the panel gap with accuracy >= 0.80 pooled over draws, and its AUC exceeds the random hold-out AUC (95% interval
over datasets above 0). Exploratory.
"""
import argparse
import json
import os
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    rows = []
    if os.path.exists(path):
        for ln in open(path):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if "error" not in r:
                rows.append(r)
    return rows


def auc(score, label):
    """P(score of a positive > score of a negative), ties half; nan if one class is empty."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def boot_ci(vals, rng, n=2000):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) == 0:
        return (float("nan"),) * 3
    m = np.array([rng.choice(v, len(v)).mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/small_sample_popularity/02_runs")
    ap.add_argument("--out", default="results/small_sample_popularity")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/03_tables", exist_ok=True)
    os.makedirs(f"{a.out}/04_figures", exist_ok=True)
    rows = load(f"{a.runs}/e2c.jsonl")
    rng = np.random.default_rng(20261014)
    vid = {"id": "X16", "title_en": "Can the sample tell when personalising starts to pay? (exploratory)",
           "title_zh": "样本自己能否判断何时个性化开始划算？（探索性）",
           "rule": "Time-ordered in-sample gap predicts the sign of the panel gap with accuracy >= 0.80, and its AUC beats the random hold-out's (interval over datasets above 0)."}
    cells = defaultdict(list)
    for r in rows:
        for mode in ("time", "random"):
            need = (f"cv_pop_{mode}", f"cv_pers_{mode}", f"pers_{mode}.U_ndcg", "pop.U_ndcg")
            if all(r.get(k) is not None for k in need):
                cells[(r["dataset"], r["N"], r["k"], mode)].append(
                    (r[f"cv_pers_{mode}"] - r[f"cv_pop_{mode}"], r[f"pers_{mode}.U_ndcg"] - r["pop.U_ndcg"]))
    if not cells:
        out = [dict(vid, verdict="not testable", numbers={}, reading_en="No e2c rows with in-sample gap columns yet.",
                    reading_zh="还没有带样本内差值列的 e2c 数据。")]
        json.dump(out, open(f"{a.out}/03_tables/an_stoprule_verdicts.json", "w"), ensure_ascii=False, indent=1)
        print("X16: not testable")
        return
    per_ds = defaultdict(lambda: {"time": ([], []), "random": ([], [])})
    table = []
    for (ds, N, k, mode), v in sorted(cells.items(), key=lambda kv: [str(x) for x in kv[0]]):
        g = np.array(v)
        cv, real = g[:, 0], g[:, 1]
        per_ds[ds][mode][0].extend(cv)
        per_ds[ds][mode][1].extend(real)
        nz = (cv != 0) & (real != 0)
        table.append(dict(dataset=ds, N=N, k=("all" if k is None else k), mode=mode, n_draws=len(g),
                          sign_accuracy=float(np.mean(np.sign(cv[nz]) == np.sign(real[nz]))) if nz.any() else None,
                          mean_cv_gap=float(cv.mean()), mean_panel_gap=float(real.mean()),
                          inflation=float(cv.mean() - real.mean())))
    import csv
    with open(f"{a.out}/03_tables/an_stoprule_cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    summ = {}
    for mode in ("time", "random"):
        cv = np.concatenate([np.array(per_ds[d][mode][0]) for d in per_ds])
        real = np.concatenate([np.array(per_ds[d][mode][1]) for d in per_ds])
        nz = (cv != 0) & (real != 0)
        accs = [np.mean(np.sign(np.array(per_ds[d][mode][0])[(np.array(per_ds[d][mode][0]) != 0) & (np.array(per_ds[d][mode][1]) != 0)])
                        == np.sign(np.array(per_ds[d][mode][1])[(np.array(per_ds[d][mode][0]) != 0) & (np.array(per_ds[d][mode][1]) != 0)]))
                for d in per_ds if ((np.array(per_ds[d][mode][0]) != 0) & (np.array(per_ds[d][mode][1]) != 0)).any()]
        aucs = [auc(per_ds[d][mode][0], np.array(per_ds[d][mode][1]) > 0) for d in per_ds]
        summ[mode] = dict(pooled_accuracy=float(np.mean(np.sign(cv[nz]) == np.sign(real[nz]))) if nz.any() else float("nan"),
                          auc_by_dataset=aucs, accuracy_by_dataset=accs,
                          mean_inflation=float(np.mean(cv) - np.mean(real)))
    d_auc = [t - r for t, r in zip(summ["time"]["auc_by_dataset"], summ["random"]["auc_by_dataset"])]
    m, lo, hi = boot_ci(d_auc, rng)
    acc_t = summ["time"]["pooled_accuracy"]
    verdict = ("supported" if acc_t >= 0.80 and lo > 0 else
               "not supported" if (acc_t < 0.65 or hi < 0) else "mixed")
    nums = dict(accuracy_time=round(acc_t, 3), accuracy_random=round(summ["random"]["pooled_accuracy"], 3),
                auc_time_minus_random=round(m, 3), auc_diff_ci_lo=round(lo, 3), auc_diff_ci_hi=round(hi, 3),
                inflation_time=round(summ["time"]["mean_inflation"], 4), inflation_random=round(summ["random"]["mean_inflation"], 4),
                n_datasets=len(per_ds), n_draws=int(sum(len(per_ds[d]["time"][0]) for d in per_ds)))
    out = [dict(vid, verdict=verdict, numbers=nums,
                reading_en=(f"Hiding each sampled person's latest choices predicts whether personalising helps on the real panel with accuracy {acc_t:.0%} "
                            f"(random hiding: {summ['random']['pooled_accuracy']:.0%}). Random hiding overstates the benefit of personalising by "
                            f"{summ['random']['mean_inflation']:+.3f} NDCG on average, time-ordered hiding by {summ['time']['mean_inflation']:+.3f}."),
                reading_zh=(f"把每个被抽中者最新的选择藏起来，预测个性化在真实验证小组上是否有帮助，准确率为 {acc_t:.0%}"
                            f"（随机藏：{summ['random']['pooled_accuracy']:.0%}）。随机藏平均高估个性化的好处 {summ['random']['mean_inflation']:+.3f} NDCG，按时间藏为 {summ['time']['mean_inflation']:+.3f}。"))]
    json.dump(out, open(f"{a.out}/03_tables/an_stoprule_verdicts.json", "w"), ensure_ascii=False, indent=1)
    # figure: in-sample gap vs real gap
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for i, mode in enumerate(("random", "time")):
        cv = np.concatenate([np.array(per_ds[d][mode][0]) for d in per_ds])
        real = np.concatenate([np.array(per_ds[d][mode][1]) for d in per_ds])
        ax[i].scatter(cv, real, s=4, alpha=0.3)
        ax[i].axhline(0, color="k", lw=0.5)
        ax[i].axvline(0, color="k", lw=0.5)
        ax[i].set_title(f"hide {'random rows' if mode == 'random' else 'latest rows'}: accuracy {summ[mode]['pooled_accuracy']:.0%}", fontsize=10)
        ax[i].set_xlabel("in-sample gap (personalised - popularity), NDCG@100")
    ax[0].set_ylabel("real gap on the panel, NDCG@100")
    better = "latest-row" if summ["time"]["pooled_accuracy"] >= summ["random"]["pooled_accuracy"] else "random-row"
    fig.suptitle(f"The {better} hold-out predicts the sign of the real gap better "
                 f"({summ['time']['pooled_accuracy']:.0%} latest-row vs {summ['random']['pooled_accuracy']:.0%} random-row)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{a.out}/04_figures/an_stoprule_gap_scatter.png", dpi=130)
    print("X16:", verdict, nums)


if __name__ == "__main__":
    main()
