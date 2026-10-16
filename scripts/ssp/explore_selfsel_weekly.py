"""Post-hoc exploration (after the W1-W11 verdicts and the strata check): can a server choose its own memory from its own self-scores?

    .venv/bin/python3 -W ignore scripts/ssp/explore_selfsel_weekly.py [--no-calculator] [--no-d29]

Exploratory; it is in neither weekly registry. It re-reads the stored weekly cells (02_runs/weekly.jsonl through an_weekly.load) and runs
no new simulation. Two optional blocks read more data: the calculator block reads the log files (weekly_core.prepare_weekly, ~15 s) and the
D29 replay reads 02_runs/deep.jsonl (gitignored; skipped when absent).

Selector. At step t, per draw, among the decay:days arms pick the one with the highest exponentially weighted mean (half-life 26 rounds,
fixed in advance, not tuned) of its stored self-scores at steps < t. selfnd[j] is the NDCG of the list published at step j against the SAMPLE
of round j+1 (deep_core.run_cell), so every score used at step t was available when the list for step t was published. The rule's value is
the stored nd100 of the chosen arm at step t, averaged over steady steps and draws. Regret = best fixed decay:days arm (hindsight, same
cell) minus the rule's value, so negative means better than every fixed arm.
Tables (pooled / same group) are fitted leave-one-log-out on 1..256 rounds (arms that do not exist in a cell snap to the nearest one).

Writes results/small_sample_popularity/03_tables/weekly/: i_selfsel_rules.csv, i_selfsel_paired.csv, i_selfsel_checks.csv,
i_selfsel_d29_replay.csv, i_calculator_loo.csv, i_log_spans.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from an_weekly import OUT, RUNS, boot, load

NS = [30, 100, 300, 1000]
RS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
HL = 26
CADENCES = (7, 28)
START = 40
SEED = 12345
MIN_GROUP = 3
GRIDS = {"factor 2 (all arms)": None, "factor 4 (14, 56, 224, 896 d)": (14, 56, 224, 896), "factor 8 (14, 112, 896 d)": (14, 112, 896)}
BINS = [(8, 20), (20, 40), (40, 80), (80, 200)]
RULES = ["two", "pooled", "same", "selector", "selector_cumulative"]
RULE_LABEL = {"two": "2 rounds", "pooled": "pooled table (LOO)", "same": "same-group table (LOO)", "selector": "self-score selector (EWMA 26)",
              "selector_cumulative": "self-score selector (cumulative mean)"}


def stratum(log: str) -> str:
    return "amazon" if log.startswith("amazon") else "other"


def write_csv(name: str, head: list, rows: list) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}", "w", newline="") as f:
        cw = csv.writer(f, lineterminator="\n")
        cw.writerow(head)
        cw.writerows(rows)


def arms_of(c, keep=None):
    pairs = sorted((int(a.split("=")[1]), i) for i, a in enumerate(c["arms"]) if a.startswith("decay:days="))
    if keep is not None:
        pairs = [(d, i) for d, i in pairs if d in keep]
    return [i for _, i in pairs], np.array([d for d, _ in pairs])


def prep(c, keep=None):
    idx, ds = arms_of(c, keep)
    return ds, c["nd"][:, idx, :], c["se"][:, idx, :]


def nearest(ds, days) -> int:
    return int(np.argmin(np.abs(np.log2(ds) - np.log2(max(days, 1)))))


def best_value(c) -> float:
    return max(v for a, v in c["val"].items() if a.startswith("decay:days="))


def snap_value(c, days) -> float:
    _, ds = arms_of(c)
    return c["val"][f"decay:days={int(ds[nearest(ds, days)])}"]


def fit_rounds(cells, logs, n, rd) -> int:
    def mean_regret(r):
        return np.mean([best_value(cells[(l, n, rd, "std")]) - snap_value(cells[(l, n, rd, "std")], rd * r) for l in logs])
    return min(RS, key=mean_regret)


def history_scores(se, t, hl):
    past = se[:, :, :t]
    ok = np.isfinite(past)
    w = np.ones(t) if hl is None else 0.5 ** ((t - 1 - np.arange(t)) / hl)
    num = np.sum(np.where(ok, past, 0.0) * w, axis=2)
    den = np.sum(ok * w, axis=2)
    return np.where(den > 0, num / np.maximum(den, 1e-12), -np.inf)


def choose(se, steps, hl, fallback):
    """{t: arm index per draw} from self-scores of steps < t; the neutral fallback (2-round arm) is used when no score exists."""
    out = {}
    for t in steps:
        if t <= 0:
            continue
        s = history_scores(se, t, hl)
        out[t] = np.where(np.isfinite(s).any(axis=1), np.argmax(s, axis=1), fallback)
    return out


def chosen_value(nd, choice, steps) -> float:
    d = np.arange(nd.shape[0])
    return float(np.nanmean([nd[d, choice[t], t] for t in steps if t in choice]))


def fixed_value(nd, arm, steps) -> float:
    return float(np.nanmean(nd[:, arm, :][:, steps]))


# ---------------------------------------------------------------------------------------------------------------- rules
def rules_block(cells) -> dict:
    per_rd = {}
    for rd in CADENCES:
        logs = sorted({k[0] for k in cells if k[3] == "std" and k[2] == rd})
        per = {}
        for held in logs:
            for n in NS:
                if (held, n, rd, "std") not in cells:
                    continue
                c = cells[(held, n, rd, "std")]
                others = [l for l in logs if l != held and (l, n, rd, "std") in cells]
                same = [l for l in others if stratum(l) == stratum(held)]
                b = best_value(c)
                ds, nd, se = prep(c)
                fb = nearest(ds, 2 * rd)
                r_pool = fit_rounds(cells, others, n, rd)
                rec = dict(r_pool=r_pool, two=b - snap_value(c, 2 * rd), pooled=b - snap_value(c, rd * r_pool),
                           same=b - snap_value(c, rd * fit_rounds(cells, same, n, rd)) if len(same) >= MIN_GROUP else float("nan"))
                rec["selector"] = b - chosen_value(nd, choose(se, c["st"], HL, fb), c["st"])
                rec["selector_cumulative"] = b - chosen_value(nd, choose(se, c["st"], None, fb), c["st"])
                per[(held, n)] = rec
        per_rd[rd] = per
    return per_rd


def summarize(per_rd) -> list:
    rows = []
    for rd, per in per_rd.items():
        for grp in ("amazon", "other", "all"):
            for n in ["all"] + NS:
                keys = [k for k in per if (grp == "all" or stratum(k[0]) == grp) and (n == "all" or k[1] == n)]
                for rule in RULES if keys else []:
                    rows.append([rd, grp, n, rule, len(keys), round(float(np.nanmean([per[k][rule] for k in keys])), 6)])
    return rows


def paired(per_rd) -> list:
    rows = []
    for rd, per in per_rd.items():
        logs = sorted({k[0] for k in per})

        def log_mean(rule, l):
            return float(np.nanmean([per[k][rule] for k in per if k[0] == l]))
        for other in ("two", "pooled", "same"):
            diff = {l: log_mean("selector", l) - log_mean(other, l) for l in logs}
            m, lo, hi = boot(list(diff.values()))
            rows.append([rd, f"selector minus {other}", round(m, 6), round(lo, 6), round(hi, 6),
                         round(float(np.mean([v < 0 for v in diff.values()])), 3), len(logs)])
    return rows


# --------------------------------------------------------------------------------------------------------------- checks
def swap_history_regret(c, c_hist, window, hl=HL) -> float:
    """Regret of the selector on cell c when it reads the self-score history of another cell (same arms only)."""
    common = sorted(set(arms_of(c)[1]) & set(arms_of(c_hist)[1]))
    ds, nd, _ = prep(c, common)
    _, _, se = prep(c_hist, common)
    d = min(nd.shape[0], se.shape[0])
    nd, se = nd[:d], se[:d]
    ch = choose(se, window, hl, nearest(ds, 14))
    best = max(fixed_value(nd, a, window) for a in range(len(ds)))
    return best - chosen_value(nd, ch, window)


def checks_block(cells, per_rd) -> list:
    rows = []
    rng = np.random.default_rng(SEED)
    for rd in CADENCES:
        for hl in (6, 13, 26, 52, 104, None):
            v = []
            for (held, n) in per_rd[rd]:
                c = cells[(held, n, rd, "std")]
                ds, nd, se = prep(c)
                v.append(best_value(c) - chosen_value(nd, choose(se, c["st"], hl, nearest(ds, 2 * rd)), c["st"]))
            rows.append([f"{rd}-day rounds", "self-score average, half-life in rounds", "cumulative mean" if hl is None else hl, len(v), round(float(np.mean(v)), 6)])
        worse = [rec["selector"] - rec["two"] for rec in per_rd[rd].values()]
        rows.append([f"{rd}-day rounds", "selector worse than 2 rounds: largest amount in one cell", "", len(worse), round(float(max(worse)), 6)])
        rows.append([f"{rd}-day rounds", "selector worse than 2 rounds by more than 0.01: number of cells", "", len(worse), int(sum(w > 0.01 for w in worse))])

    per = per_rd[7]
    warm = {b: {"selector": [], "pooled": [], "two": []} for b in BINS}
    grid = {g: [] for g in GRIDS}
    ctl = {k: [] for k in ["2 rounds", "selector", "freeze after 8 weeks", "freeze after 12 weeks", "freeze after 20 weeks", "freeze after 40 weeks",
                           "random arm (null)", "history of the other logs (mean over logs)"]}
    half = {k: [] for k in ["2 rounds", "pooled table (LOO)", "fixed arm best on 1st half (uses truth)", "selector"]}
    logs7 = sorted({k[0] for k in per})
    for (held, n), rec in per.items():
        c = cells[(held, n, 7, "std")]
        ds, nd, se = prep(c)
        st = c["st"]
        fb = nearest(ds, 14)
        a_pool, a_two = nearest(ds, 7 * rec["r_pool"]), fb
        ch = choose(se, st, HL, fb)
        fbest = int(np.argmax([fixed_value(nd, a, st) for a in range(len(ds))]))
        for b in BINS:
            ts = [t for t in st if b[0] <= t < b[1]]
            if len(ts) < 5:
                continue
            ref = fixed_value(nd, fbest, ts)
            warm[b]["selector"].append(ref - chosen_value(nd, ch, ts))
            warm[b]["pooled"].append(ref - fixed_value(nd, a_pool, ts))
            warm[b]["two"].append(ref - fixed_value(nd, a_two, ts))
        for g, keep in GRIDS.items():
            ds2, nd2, se2 = prep(c, keep)
            if len(ds2) >= 2:
                grid[g].append(best_value(c) - chosen_value(nd2, choose(se2, st, HL, nearest(ds2, 14)), st))
        win = [t for t in st if t >= START]
        bw = max(fixed_value(nd, a, win) for a in range(len(ds)))
        d = np.arange(nd.shape[0])
        ctl["2 rounds"].append(bw - fixed_value(nd, a_two, win))
        ctl["selector"].append(bw - chosen_value(nd, ch, win))
        for k in (8, 12, 20, 40):
            ck = np.argmax(history_scores(se, k, None), axis=1)
            ctl[f"freeze after {k} weeks"].append(bw - float(np.nanmean([nd[d, ck, t] for t in win])))
        ctl["random arm (null)"].append(bw - float(np.nanmean([nd[d, rng.integers(0, len(ds), len(d)), t] for t in win])))
        ctl["history of the other logs (mean over logs)"].append(np.mean([swap_history_regret(c, cells[(o, n, 7, "std")], win)
                                                                          for o in logs7 if o != held and (o, n, 7, "std") in cells]))
        h = len(st) // 2
        s1, s2 = st[:h], st[h:]
        b2 = max(fixed_value(nd, a, s2) for a in range(len(ds)))
        half["2 rounds"].append(b2 - fixed_value(nd, a_two, s2))
        half["pooled table (LOO)"].append(b2 - fixed_value(nd, a_pool, s2))
        half["fixed arm best on 1st half (uses truth)"].append(b2 - fixed_value(nd, int(np.argmax([fixed_value(nd, a, s1) for a in range(len(ds))])), s2))
        half["selector"].append(b2 - chosen_value(nd, ch, s2))
    for b in BINS:
        for k in ("selector", "pooled", "two"):
            rows.append(["7-day rounds", f"warm-up, regret vs the cell's best fixed arm, steps {b[0]}-{b[1]}", {"selector": "selector", "pooled": "pooled table (LOO)", "two": "2 rounds"}[k],
                         len(warm[b][k]), round(float(np.mean(warm[b][k])), 6)])
    for g, v in grid.items():
        rows.append(["7-day rounds", "arm grid for the selector (regret vs all-arm best)", g, len(v), round(float(np.mean(v)), 6)])
    for k, v in ctl.items():
        rows.append(["7-day rounds", f"control, steps >= {START}", k, len(v), round(float(np.mean(v)), 6)])
    for k, v in half.items():
        rows.append(["7-day rounds", "second half of the steady steps (regret vs second-half best)", k, len(v), round(float(np.mean(v)), 6)])
    return rows


# ------------------------------------------------------------------------------------------------------------ D29 replay
def d29_replay() -> list:
    path = f"{RUNS}/deep.jsonl"
    if not os.path.exists(path):
        print("D29 replay skipped: deep.jsonl not found")
        return []
    recs = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") != "deep" or "selfsel" not in r["arms"] or r["n"] != 100:
                continue
            arms = r["arms"]
            ai = {a: i for i, a in enumerate(arms)}
            nd, se = np.array(r["nd100"], float), np.array(r["selfnd"], float)
            st = [i for i, s in enumerate(r["steps"]) if s >= 8]
            if len(st) < 3:
                continue
            hg = [a for a in arms if a.startswith("decay:h=")]
            dy = [a for a in arms if a.startswith("decay:days=")]
            rec = dict(nw=r["n_win"], s=stratum(r["dataset"]), base=fixed_value(nd, ai["decay:h=2"], st), d29=fixed_value(nd, ai["selfsel"], st),
                       best_hg=max(fixed_value(nd, ai[a], st) for a in hg))
            for lab, grp in (("new_hg", hg), ("new_days", dy), ("new_all", hg + dy)):
                ix = [ai[a] for a in grp]
                rec[lab] = chosen_value(nd[:, ix, :], choose(se[:, ix, :], st, HL, ix.index(ai["decay:h=2"]) if ai["decay:h=2"] in ix else 0), st)
            recs.append(rec)
    rows = []
    for nw in sorted({x["nw"] for x in recs}) + ["all"]:
        for s in ("amazon", "other"):
            R = [x for x in recs if (nw == "all" or x["nw"] == nw) and x["s"] == s]
            if not R:
                continue
            row = [nw, s, len(R)]
            for k in ("best_hg", "d29", "new_hg", "new_days", "new_all"):
                row.append(round(float(np.mean([x[k] - x["base"] for x in R])), 6))
            for k in ("d29", "new_hg", "new_days", "new_all"):
                row.append(round(float(np.mean([(x[k] - x["base"]) < -0.01 for x in R])), 3))
            rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------------------------- calculator
def log_features(name: str) -> dict:
    import weekly_core as W
    D = W.prepare_weekly(name, 7)
    cnt, act, rows, first = D["counts"], D["active"], D["rows"], D["first"]
    disc = 1 / np.log2(np.arange(2, 102))

    def top100(v):
        """Indices of the 100 largest values, best first; ties at the cut are broken by lowest index (deterministic)."""
        if len(v) <= 100:
            return np.argsort(-v, kind="stable")
        cut = np.partition(v, len(v) - 100)[len(v) - 100]
        above = np.flatnonzero(v > cut)
        idx = np.concatenate([above, np.flatnonzero(v == cut)[:100 - len(above)]])
        return idx[np.argsort(-v[idx], kind="stable")]
    n0, n25 = [], []
    for k in range(START, len(cnt)):
        gain = cnt[k].astype(np.float64)
        if gain.sum() <= 0:
            continue
        ideal = float(np.sum(gain[top100(gain)] * disc))
        for lag, out in ((0, n0), (25, n25)):
            lst = top100(np.sum(cnt[k - 4 - lag: k - lag], axis=0).astype(np.float64))
            out.append(float(np.sum(gain[lst] * disc) / ideal))
    W.T._CACHE.clear()
    return dict(rpp=float(np.median(rows / np.maximum(act, 1))), stale25=1 - float(np.mean(n25)) / float(np.mean(n0)))


def calculator_block(cells) -> list:
    logs = sorted({k[0] for k in cells if k[3] == "std" and k[2] == 7})
    feat = {l: log_features(l) for l in logs}
    cl = [(l, n) for l in logs for n in NS if (l, n, 7, "std") in cells]

    def hstar(l, n):
        c = cells[(l, n, 7, "std")]
        return max((v, int(a.split("=")[1])) for a, v in c["val"].items() if a.startswith("decay:days="))[1] / 7

    def x(l, n, kind):
        o = 1.0 if stratum(l) == "other" else 0.0
        f = {"n": [np.log2(n)], "n+group": [np.log2(n), o], "info": [np.log2(n * feat[l]["rpp"])],
             "info+group": [np.log2(n * feat[l]["rpp"]), o], "n+drift": [np.log2(n), np.log2(feat[l]["stale25"] + 0.02)],
             "info+drift": [np.log2(n * feat[l]["rpp"]), np.log2(feat[l]["stale25"] + 0.02)]}[kind]
        return [1.0] + f
    kinds = ["n", "n+group", "info", "info+group", "n+drift", "info+drift"]
    per = {nm: {} for nm in ["2 rounds", "pooled table", "same-group table"] + [f"regression {k}" for k in kinds]}
    for held in logs:
        tr = [(l, n) for l, n in cl if l != held]
        coef = {k: np.linalg.lstsq(np.array([x(l, n, k) for l, n in tr]), np.array([np.log2(hstar(l, n)) for l, n in tr]), rcond=None)[0] for k in kinds}
        for l, n in [(a, m) for a, m in cl if a == held]:
            c = cells[(l, n, 7, "std")]
            b = best_value(c)
            others = [o for o in logs if o != l and (o, n, 7, "std") in cells]
            same = [o for o in others if stratum(o) == stratum(l)]
            per["2 rounds"][(l, n)] = b - snap_value(c, 14)
            per["pooled table"][(l, n)] = b - snap_value(c, 7 * fit_rounds(cells, others, n, 7))
            per["same-group table"][(l, n)] = b - snap_value(c, 7 * fit_rounds(cells, same, n, 7)) if len(same) >= MIN_GROUP else float("nan")
            for k in kinds:
                per[f"regression {k}"][(l, n)] = b - snap_value(c, 7 * 2 ** float(np.dot(coef[k], x(l, n, k))))
    rows = []
    for nm, d in per.items():
        for grp in ("amazon", "other", "all", "goodreads-poetry", "steam"):
            v = [val for (l, n), val in d.items() if grp == "all" or grp == stratum(l) or grp == l]
            rows.append([nm, grp, len(v), round(float(np.nanmean(v)), 6)])
    feat_rows = [["feature", l, round(feat[l]["rpp"], 3), round(feat[l]["stale25"], 4)] for l in logs]
    return rows + feat_rows


def spans_block() -> list:
    """Weeks between the first and last positive event of every log of the weekly study (the 3 logs excluded for length included)."""
    import weekly_core as W
    rows = []
    for name in sorted(json.load(open(f"{RUNS}/weekly_logs.json"))):
        z = np.load(f"{W.T.NPZ}/{name}.npz")
        ts = z["ts"][z["r"] >= float(z["positive_threshold"])].astype(np.int64)
        if ts.max() > 10 ** 11:
            ts = ts // 1000
        rows.append([name, int(round((ts.max() - ts.min()) / (7 * 86400)))])
    return sorted(rows, key=lambda r: r[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-calculator", action="store_true")
    ap.add_argument("--no-d29", action="store_true")
    args = ap.parse_args()
    cells, _ = load()
    bad = [k for k, c in cells.items() if k[3] == "std" for a, v in c["val"].items() if a.startswith("decay:days=") and not np.isfinite(v)]
    assert not bad, f"non-finite decay values in cells {bad[:3]}"
    per_rd = rules_block(cells)
    rules = summarize(per_rd)
    write_csv("i_selfsel_rules.csv", ["round_days", "group", "n", "rule", "cells", "mean_regret"], rules)
    write_csv("i_selfsel_paired.csv", ["round_days", "comparison", "mean_diff", "ci_lo", "ci_hi", "share_logs_selector_better", "logs"], paired(per_rd))
    write_csv("i_selfsel_checks.csv", ["round_days", "check", "setting", "cells", "mean_regret"], checks_block(cells, per_rd))
    print("mean regret vs best fixed arm (rows: all cells)   amazon | other | all")
    for rd in CADENCES:
        print(f"-- {rd}-day rounds")
        for rule in RULES:
            v = {g: next(r[5] for r in rules if r[0] == rd and r[1] == g and r[2] == "all" and r[3] == rule) for g in ("amazon", "other", "all")}
            print(f"  {RULE_LABEL[rule]:40} {v['amazon']:+.4f} | {v['other']:+.4f} | {v['all']:+.4f}")
    if not args.no_d29:
        d29 = d29_replay()
        write_csv("i_selfsel_d29_replay.csv", ["n_win", "group", "cells", "best_hg_minus_base", "d29_selfsel_minus_base", "selector_hg_minus_base",
                                               "selector_days_minus_base", "selector_all_minus_base", "harm_d29", "harm_selector_hg", "harm_selector_days",
                                               "harm_selector_all"], d29)
        print("D29 replay rows:", len(d29))
    if not args.no_calculator:
        write_csv("i_calculator_loo.csv", ["rule_or_feature", "group_or_log", "cells_or_rows_per_person", "mean_regret_or_drift"], calculator_block(cells))
        write_csv("i_log_spans.csv", ["log", "weeks_of_positive_events"], spans_block())
    print("wrote tables under", OUT)


if __name__ == "__main__":
    main()
