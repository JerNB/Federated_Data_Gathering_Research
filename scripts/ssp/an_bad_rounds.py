"""What makes a round hard? (development analysis, nothing registered; no new sampling cell is run.)

Doc 12 section 4: the 10th-percentile round sits 0.09 (30 people) to 0.12 (300 people) NDCG below the mean round and the gap does not shrink with n; the
interpretation (drift between rounds, not sampling noise) was not tested. Suspects: (1) the community changed between the last round the list has seen and the round that
scores it (drift), (2) the scoring round is thin (few rows) or its own finite size adds noise, (3) the scoring round is dominated by items born in it (newborn mass), against
(4) sampling luck of the server's sample.

For every (cadence, log, n) the per-round NDCG@100 of the SAMPLED list (draws x rounds, stored cells) is compared with the NDCG@100 of the FULL-community list with the same
memory (no sampling at all, recomputed from the log's own counts exactly as deep_core.FullDecay does; for the long-round deep cells the same recomputation, which agrees with the
stored `fulldecay:h=2` arm up to tie-breaking).
  within_share    share of one draw's variance over steady rounds that is draw-to-draw noise (unbiased: between-round part = var of draw means - within / draws)
  rho_yf          Spearman correlation across steady rounds between the sampled draw-mean and the full list
  gap10_y/_f      mean minus 10th percentile over steady rounds (draws pooled for y), as doc 12's bad-round gap
  r2_f_pred       R^2 of the full list's per-round NDCG on three standardised predictors of the SCORING round: drift = 1 - cos(counts[w], counts[w+1]) (this nearly restates the score
                  of a list built from round w alone: descriptive, not a forecast), thin = log10(rows of round w+1), newborn = share of round w+1's rows on items first seen in w+1
                  (weekly: first positive anywhere in the log, not only in the 156-week span; deep: first round of the prepared window)
  rho_f_lagdrift  Spearman between the full list's NDCG and the LAGGED drift 1 - cos(counts[w-1], counts[w]), which is known before the scoring round: the forecastable part
  r2_f_lagdrift   squared Pearson correlation of the same pair
  rel_scoring     reliability of the scoring round: the same full list is scored against two random halves of the scoring round's people; the Pearson correlation of the two
                  half-scores across steady rounds, stepped up to the whole round by Spearman-Brown; 1 - rel_scoring is the share of the full list's round-to-round variance
                  that is noise of the finite scoring round, not change of the community
Cadences: weekly (7-day rounds, window w1, n = 100 stage a and n = 300 stage d, the hindsight best fixed memory) and deep (the stored `plan` cells of doc 12, 30 rounds of 1 to 350
days, baseline decay:h=2 against the full list with h = 2, n = 30 and 300).

    .venv/bin/python3 -W ignore scripts/ssp/an_bad_rounds.py [--only LOG]
Tables: results/small_sample_popularity/03_tables/badrounds/
"""
from __future__ import annotations

import csv
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import spearmanr

import deep_core as C
import track_core as T
import weekly_core as W

RUNS = "results/small_sample_popularity/02_runs"
OUT = "results/small_sample_popularity/03_tables/badrounds"
STEADY = 8
MIN_ROUNDS = 12
PRED = ("drift", "thin", "newborn")


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def ndcg100(lst, gain, idcg):
    g = gain[lst].astype(np.float64)
    return float(np.sum(g[:100] * C.DISC[:len(g[:100])])) / idcg if idcg > 0 else 0.0


def full_and_half(D, h, key):
    """NDCG@100 of the full-community decayed list (half-life h rounds) after round w, scored against round w + 1, and the NDCG of the SAME list against two random halves of round
    w + 1's people (NaN where the scoring round, or a half of it, is too thin). The halves use their own ideal ranking."""
    nw = len(D["counts"])
    d, s = C.decay_factor(h), np.zeros(D["n_items"], np.float64)
    full, ha, hb = (np.full(nw - 1, np.nan) for _ in range(3))
    mask = np.zeros(D["n_users"] + 1, bool)
    for w in range(nw - 1):
        s *= d
        s += C.norm(D["counts"][w].astype(np.float64))
        gain = D["counts"][w + 1]
        if gain.sum() < T.MIN_TRUTH_ROWS:
            continue
        lst = C.top_idx(s, 300)
        ig = gain[C.top_idx(gain, 300)].astype(np.float64)
        full[w] = ndcg100(lst, gain, float(np.sum(ig[:100] * C.DISC[:100])))
        a, b = D["off"][w + 1], D["off"][w + 2]
        uk = D["users_in"][w + 1]
        perm = np.random.default_rng(C.seed_of(key, w)).permutation(len(uk))
        cut = len(uk) // 2
        for out, sel in ((ha, uk[perm[:cut]]), (hb, uk[perm[cut:]])):
            mask[sel] = True
            m = mask[D["user"][a:b]]
            mask[sel] = False
            g = np.bincount(D["item"][a:b][m], minlength=D["n_items"]).astype(np.float64)
            if g.sum() < 0.5 * T.MIN_TRUTH_ROWS:
                continue
            igh = g[C.top_idx(g, 300)]
            out[w] = ndcg100(lst, g, float(np.sum(igh[:100] * C.DISC[:100])))
    return full, ha, hb


def whole_log_birth_round(log, D):
    """For every item of the prepared window: the 7-day round (relative to the window start) of its first positive anywhere in the whole log; negative = born before the window."""
    z = np.load(f"{T.NPZ}/{log}.npz")
    pos = z["r"] >= float(z["positive_threshold"])
    item_all, ts_all = z["item"][pos].astype(np.int64), z["ts"][pos].astype(np.int64)
    if ts_all.max() > 10 ** 11:
        ts_all //= 1000
    lo = int(ts_all.min())
    t0 = lo + W.densest_start(ts_all, lo) * W.WEEK
    order = np.argsort(item_all, kind="stable")
    it_s, ts_s = item_all[order], ts_all[order]
    starts = np.flatnonzero(np.r_[True, it_s[1:] != it_s[:-1]])
    first_ts = np.full(int(it_s.max()) + 1, np.iinfo(np.int64).max, np.int64)
    first_ts[it_s[starts]] = np.minimum.reduceat(ts_s, starts)
    span_items = np.unique(W.load_span(log)[1])                     # the window's items, in the order prepare_weekly numbers them
    assert len(span_items) == D["n_items"]
    return (first_ts[span_items] - t0) // W.WEEK


def predictors(D, birth=None):
    nw = len(D["counts"])
    out = {k: np.full(nw - 1, np.nan) for k in PRED + ("drift_lag", "drift_lag2")}
    born = np.asarray(D["first"]) if birth is None else birth
    for w in range(nw - 1):
        a, b = D["counts"][w].astype(np.float64), D["counts"][w + 1].astype(np.float64)
        tot, na, nb = float(b.sum()), float(np.sqrt(a @ a)), float(np.sqrt(b @ b))
        if w >= 1:
            p = D["counts"][w - 1].astype(np.float64)
            npv = float(np.sqrt(p @ p))
            if npv > 0 and na > 0:
                out["drift_lag"][w] = 1.0 - float(p @ a) / (npv * na)             # round w-1 to w: shares round w with the drift into the scoring round
        if w >= 2:
            p2, q = D["counts"][w - 2].astype(np.float64), D["counts"][w - 1].astype(np.float64)
            n2, nq = float(np.sqrt(p2 @ p2)), float(np.sqrt(q @ q))
            if n2 > 0 and nq > 0:
                out["drift_lag2"][w] = 1.0 - float(p2 @ q) / (n2 * nq)            # round w-2 to w-1: shares no round with the drift into the scoring round
        if tot < T.MIN_TRUTH_ROWS or na == 0 or nb == 0:
            continue
        out["drift"][w] = 1.0 - float(a @ b) / (na * nb)
        out["thin"][w] = np.log10(tot)
        out["newborn"][w] = float(b[born == w + 1].sum()) / tot
    return out


def analyse(y, f, X, steps, ha, hb):
    """y [draws][steps] sampled NDCG, f [steps] full-list NDCG, X name -> [steps], ha/hb [steps] half-round scores; steady rounds only."""
    st = np.array(steps) >= STEADY
    y, f, ha, hb = y[:, st], f[st], ha[st], hb[st]
    X = {k: v[st] for k, v in X.items()}
    ok = np.isfinite(f) & np.all(np.isfinite(y), axis=0)
    for k in PRED:
        ok &= np.isfinite(X[k])
    if ok.sum() < MIN_ROUNDS:
        return None
    okl = ok & np.isfinite(X["drift_lag"])
    okh = ok & np.isfinite(ha) & np.isfinite(hb)
    yo, fo, Xo = y[:, ok], f[ok], {k: X[k][ok] for k in PRED}
    draws = yo.shape[0]
    ybar = yo.mean(0)
    within = float(yo.var(axis=0, ddof=1).mean())
    between = max(float(ybar.var(ddof=1)) - within / draws, 0.0)
    pooled = yo.ravel()
    Z = np.column_stack([np.ones(len(fo))] + [(Xo[k] - Xo[k].mean()) / (Xo[k].std() or 1.0) for k in PRED])
    coef, *_ = np.linalg.lstsq(Z, fo, rcond=None)
    sse, sst = float(((fo - Z @ coef) ** 2).sum()), float(((fo - fo.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    g = fo - ybar                                           # the sampling gap of every round: full list minus the sampled list
    rel = float("nan")
    if okh.sum() >= MIN_ROUNDS:
        r_half = float(np.corrcoef(ha[okh], hb[okh])[0, 1])
        rel = 2.0 * r_half / (1.0 + r_half)
    lag_ok = okl.sum() >= MIN_ROUNDS
    okl2 = ok & np.isfinite(X["drift_lag2"])
    g_all = np.full(len(f), np.nan)
    g_all[ok] = g
    both = np.flatnonzero(ok[1:] & ok[:-1]) + 1                                    # consecutive steady rounds that are both usable
    lag2_ok = okl2.sum() >= MIN_ROUNDS
    q10 = float(np.percentile(fo, 10))                                              # runs test: is a bottom-decile round followed by another one more often than 10%?
    bot = np.zeros(len(f), bool)
    bot[ok] = f[ok] <= q10
    after_bot = bot[both - 1]
    persist = float(bot[both][after_bot].mean()) if after_bot.sum() >= 3 else float("nan")
    tri = np.flatnonzero(ok[2:] & ok[1:-1] & ok[:-2]) + 2                           # detrended: autocorrelation of the first differences (negative = noise or mean reversion)
    rho_diff = float(spearmanr(f[tri] - f[tri - 1], f[tri - 1] - f[tri - 2]).statistic) if len(tri) >= MIN_ROUNDS else float("nan")
    return dict(rounds=int(ok.sum()), mean_y=float(pooled.mean()), mean_f=float(fo.mean()), within_share=within / (within + between) if within + between > 0 else float("nan"),
                rho_yf=float(spearmanr(ybar, fo).statistic), r2_yf=float(np.corrcoef(ybar, fo)[0, 1] ** 2),
                gap10_y=float(pooled.mean() - np.percentile(pooled, 10)), gap10_f=float(fo.mean() - np.percentile(fo, 10)),
                r2_f_pred=r2, r2_f_pred_adj=1.0 - (1.0 - r2) * (len(fo) - 1) / (len(fo) - 1 - len(PRED)),
                rho_f_lagdrift=float(spearmanr(f[okl], X["drift_lag"][okl]).statistic) if lag_ok else float("nan"),
                r2_f_lagdrift=float(np.corrcoef(f[okl], X["drift_lag"][okl])[0, 1] ** 2) if lag_ok else float("nan"),
                rho_f_lag2drift=float(spearmanr(f[okl2], X["drift_lag2"][okl2]).statistic) if lag2_ok else float("nan"),
                rho_f_prev=float(spearmanr(f[both], f[both - 1]).statistic) if len(both) >= MIN_ROUNDS else float("nan"), p_bottom_persist=persist, rho_f_diff=rho_diff,
                rho_g_lag2drift=float(spearmanr(g_all[okl2], X["drift_lag2"][okl2]).statistic) if lag2_ok else float("nan"),
                rho_g_prev=float(spearmanr(g_all[both], g_all[both - 1]).statistic) if len(both) >= MIN_ROUNDS else float("nan"),
                rel_scoring=rel, f_noise_share=1.0 - rel if np.isfinite(rel) else float("nan"),
                gapS_mean=float(g.mean()), gapS_sd_over_sd_f=float(g.std(ddof=1) / fo.std(ddof=1)), rho_g_drift=float(spearmanr(g, Xo["drift"]).statistic),
                **{f"b_{k}": float(coef[i + 1] / (fo.std() or 1.0)) for i, k in enumerate(PRED)})


def weekly(only=None):
    out = []
    cells = {}
    for line in open(f"{RUNS}/weekly.jsonl"):
        r = json.loads(line)
        if r.get("kind") != "deep" or r["round_days"] != 7 or "@frac" in r["arms"][0]:
            continue
        stage = r["key"].split("|", 1)[0]
        if (r["n"], stage) in {(100, "a"), (300, "d")} and (not only or r["dataset"] == only):
            cells.setdefault(r["dataset"], []).append(r)
    for log in sorted(cells):
        D = W.prepare_weekly(log, 7)
        X = predictors(D, whole_log_birth_round(log, D))
        for r in sorted(cells[log], key=lambda r: r["n"]):
            steps = r["steps"]
            assert list(steps) == list(range(len(steps))), "step index must equal round index"
            nd = np.array(r["nd100"], float)
            days = [(a, i) for i, a in enumerate(r["arms"]) if a.startswith("decay:days=") and "@" not in a]
            st = np.array(steps) >= STEADY
            best = max(days, key=lambda ai: float(np.nanmean(nd[:, ai[1], :][:, st])))
            h = float(best[0].split("=")[1]) / 7.0
            f, ha, hb = (x[:len(steps)] for x in full_and_half(D, h, (log, r["n"], "scoring-halves")))
            res = analyse(nd[:, best[1], :], f, X, steps, ha, hb)
            if res:
                out.append(dict(cadence="weekly", log=log, n=r["n"], memory_rounds=h, **res))
        print("weekly", log, "done", flush=True)
    return out


def deep(only=None):
    out = []
    rows = {}
    for line in open(f"{RUNS}/deep.jsonl"):
        r = json.loads(line)
        if r["key"].startswith("plan|") and r["n"] in (30, 300) and "decay:h=2" in r["arms"] and "fulldecay:h=2" in r["arms"] and (not only or r["dataset"] == only):
            rows.setdefault(r["dataset"], []).append(r)
    for log in sorted(rows):
        if not all(any(r["n"] == n for r in rows[log]) for n in (30, 300)):
            continue
        D = T.prepare(log, rows[log][0]["n_win"])
        X = predictors(D)
        f_full, ha, hb = full_and_half(D, 2.0, (log, "deep", "scoring-halves"))
        for r in sorted(rows[log], key=lambda r: r["n"]):
            nd = np.array(r["nd100"], float)
            y = nd[:, r["arms"].index("decay:h=2"), :]
            n_steps = len(r["steps"])
            res = analyse(y, f_full[:n_steps], X, r["steps"], ha[:n_steps], hb[:n_steps])
            if res:
                out.append(dict(cadence="deep", log=log, n=r["n"], memory_rounds=2.0, **res))
        print("deep", log, "done", flush=True)
    return out


def main():
    only = arg("--only")
    rows = weekly(only) + deep(only)
    os.makedirs(OUT, exist_ok=True)
    head = ["cadence", "log", "n", "memory_rounds", "rounds", "mean_y", "mean_f", "within_share", "rho_yf", "r2_yf", "gap10_y", "gap10_f", "r2_f_pred", "r2_f_pred_adj",
            "rho_f_lagdrift", "r2_f_lagdrift", "rho_f_lag2drift", "rho_f_prev", "p_bottom_persist", "rho_f_diff", "rho_g_lag2drift", "rho_g_prev", "rel_scoring", "f_noise_share",
            "gapS_mean", "gapS_sd_over_sd_f", "rho_g_drift"] + [f"b_{k}" for k in PRED]
    with open(f"{OUT}/badrounds_per_log.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(head)
        for r in rows:
            w.writerow([r[h] if isinstance(r[h], (str, int)) else round(r[h], 5) for h in head])
    print("wrote", f"{OUT}/badrounds_per_log.csv", len(rows), "rows")
    summ = []
    for cad in ("weekly", "deep"):
        for n in sorted({r["n"] for r in rows if r["cadence"] == cad}):
            sub = [r for r in rows if r["cadence"] == cad and r["n"] == n]
            med = lambda k: float(np.nanmedian([r[k] for r in sub]))
            q = lambda k, p: float(np.nanpercentile([r[k] for r in sub], p))
            hard = sum(1 for r in sub if r["within_share"] < 0.25 and r["rho_yf"] > 0.6)
            summ.append([cad, n, len(sub), round(med("within_share"), 3), round(med("rho_yf"), 3), round(med("r2_yf"), 3), round(med("gap10_y"), 4), round(med("gap10_f"), 4),
                         round(med("r2_f_pred"), 3), round(med("r2_f_pred_adj"), 3), round(q("r2_f_pred_adj", 25), 3), round(q("r2_f_pred_adj", 75), 3),
                         round(med("rho_f_lagdrift"), 3), round(med("r2_f_lagdrift"), 3), round(med("rho_f_lag2drift"), 3), round(med("rho_f_prev"), 3),
                         round(med("p_bottom_persist"), 3), round(med("rho_f_diff"), 3), round(med("rho_g_lag2drift"), 3), round(med("rho_g_prev"), 3),
                         round(med("rel_scoring"), 3), round(med("f_noise_share"), 3),
                         round(med("gapS_mean"), 4), round(med("gapS_sd_over_sd_f"), 3), round(med("rho_g_drift"), 3)] + [round(med(f"b_{k}"), 3) for k in PRED] + [hard])
    sh = ["cadence", "n", "logs", "median_within_share", "median_rho_yf", "median_r2_yf", "median_gap10_y", "median_gap10_f", "median_r2_f_pred", "median_r2_f_pred_adj",
          "r2_f_pred_adj_q25", "r2_f_pred_adj_q75", "median_rho_f_lagdrift", "median_r2_f_lagdrift", "median_rho_f_lag2drift", "median_rho_f_prev", "median_p_bottom_persist",
          "median_rho_f_diff", "median_rho_gap_lag2drift", "median_rho_gap_prev", "median_rel_scoring", "median_f_noise_share", "median_sampling_gap_mean",
          "median_sampling_gap_sd_over_sd_f", "median_rho_gap_drift"] + [f"median_b_{k}" for k in PRED] + ["logs_intrinsically_hard (within<0.25 and rho>0.6)"]
    with open(f"{OUT}/badrounds_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(sh)
        w.writerows(summ)
    print("wrote", f"{OUT}/badrounds_summary.csv")
    for s in summ:
        print(dict(zip(sh, s)))


if __name__ == "__main__":
    main()
