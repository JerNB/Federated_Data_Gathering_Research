"""Gates, survivors and ledger for the hypothesis tournament. Rules are fixed in configs/tourney_registry.json.

    .venv/bin/python3 scripts/ssp/an_tourney.py screen|full|holdout|cal|offline|final

Per cell: diff = mean NDCG@100 of a rule minus the baseline rule (decay:h=2) over draws and steps (paired: same people).
Each n (people per round) is judged separately on the logs eligible at that n (needs min_logs_per_n logs). The t bound is
a one-sided t interval of the mean over logs. A missing expected cell makes a stage INCOMPLETE (exit code 3, nothing written).
Writes tourney_<stage>.csv, tourney_survivors.json, tourney_ledger.csv in results/small_sample_popularity/03_tables.
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

GEN = os.environ.get("TOURNEY_GEN", "")
OUT = "results/small_sample_popularity"
TAB = f"{OUT}/03_tables"
RUNS = f"{OUT}/02_runs/tourney{GEN}.jsonl"
REG = json.load(open(f"configs/tourney_registry{GEN}.json"))
BASE = REG["baseline"]
MARGIN = REG["effect_margin"]
K = REG["family_size"]
ALPHA = REG["alpha"]
NV = len(REG["n_values"])
MIN_LOGS = REG["min_logs_per_n"]
FRESH = set(REG["split"]["holdout_fresh"])
HGRID = ["decay:h=0.25", "decay:h=0.5", "decay:h=1", "decay:h=2", "decay:h=4", "decay:h=8", "decay:h=inf"]


def rows_of(stage):
    out = []
    if os.path.exists(RUNS):
        for ln in open(RUNS):
            r = json.loads(ln)
            if r["key"].startswith(stage + "|"):
                out.append(r)
    return out


def check_complete(stage, rows):
    p = f"{OUT}/02_runs/tourney{GEN}_expected_{stage}.json"
    if not os.path.exists(p):
        sys.exit(f"INCOMPLETE: {p} missing; run run_tourney.py --stage {stage} first")
    exp = json.load(open(p))
    have = {r["key"]: set(r["policies"]) for r in rows}
    miss = [k for k, pol in exp.items() if k not in have or not set(pol) <= have[k]]
    if miss:
        print(f"INCOMPLETE: {len(miss)} of {len(exp)} expected cells missing for stage {stage}: {miss[:8]}")
        sys.exit(3)


def cell_means(r):
    a = np.array(r["ndcg"], float)                       # draw, policy, step
    return {p: float(np.nanmean(a[:, i, :])) for i, p in enumerate(r["policies"])}


def per_n_diffs(rows, nwin=None):
    """{policy: {(n_win, n): {log: diff vs baseline}}}; nwin=None keeps every window width."""
    acc = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if nwin is not None and r["n_win"] != nwin:
            continue
        m = cell_means(r)
        for p, v in m.items():
            if p != BASE and BASE in m:
                acc[p][(r["n_win"], r["n"])][r["dataset"]] = v - m[BASE]
    return acc


def judge(kind, d, level, fresh_ok=False):
    v = np.array(list(d.values()))
    st = dict(n_logs=len(v), testable=len(v) >= MIN_LOGS)
    if len(v) == 0:
        return dict(st, mean=None, pass_=False)
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    tq = float(stats.t.ppf(1 - level, max(len(v) - 1, 1)))
    se = sd / np.sqrt(len(v))
    st.update(mean=float(v.mean()), median=float(np.median(v)), frac_pos=float((v > 0).mean()), frac_neg=float((v < 0).mean()),
              minimum=float(v.min()), lo=float(v.mean() - tq * se), hi=float(v.mean() + tq * se))
    am = [x for k, x in d.items() if k.startswith("amazon-")]
    ot = [x for k, x in d.items() if not k.startswith("amazon-")]
    fr = [x for k, x in d.items() if k in FRESH]
    st["amazon_mean"] = float(np.mean(am)) if am else None
    st["other_mean"] = float(np.mean(ot)) if ot else None
    st["fresh_mean"] = float(np.mean(fr)) if fr else None
    if kind == "superiority":
        ok = st["mean"] >= MARGIN and st["frac_pos"] >= 0.7 and st["lo"] > 0
    elif kind == "noninferiority":
        ok = st["lo"] > -MARGIN and st["minimum"] > -3 * MARGIN
    else:
        ok = st["mean"] <= -MARGIN and st["frac_neg"] >= 0.7 and st["hi"] < 0
    st["pass_"] = bool(ok and st["testable"])
    return st


def fresh_same_sign(kind, st):
    f = st.get("fresh_mean")
    if f is None:
        return True
    return f > 0 if kind == "superiority" else (f > -MARGIN if kind == "noninferiority" else f < 0)


def screen_keep(kind, mean):
    return mean > 0.005 if kind == "superiority" else (mean > -MARGIN if kind == "noninferiority" else mean < -0.005)


def load_surv():
    p = f"{TAB}/tourney{GEN}_survivors.json"
    return json.load(open(p)) if os.path.exists(p) else {}


def save(name, rows):
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(f"{TAB}/{name}", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def stage_table(stage, rows):
    sc = REG["stages"][stage]
    combos = len(sc["n"]) * len(sc.get("n_win", [30]))          # (window width, n) combinations judged separately
    level = {"screen": 0.05, "full": ALPHA / (K * combos), "holdout": ALPHA / combos, "robust": ALPHA / combos}[stage]
    diffs = per_n_diffs(rows, nwin=60 if stage == "robust" else None)
    table = []
    for h in REG["hypotheses"]:
        for role, p in [("primary", h["primary"])] + [("secondary", s) for s in h["secondary"]]:
            for nw, n in sorted(diffs.get(p, {})):
                st = judge(h["kind"], diffs[p][(nw, n)], level)
                cid = n if (nw == 30 or stage == "robust") else f"{n}@{nw}w"      # combination id; plain n means 30 windows
                row = dict(hypothesis=h["id"], rule=p, role=role, kind=h["kind"], n=n, n_win=nw, cid=cid)
                row.update({k.rstrip("_"): (round(x, 5) if isinstance(x, float) else x) for k, x in st.items()})
                if stage == "screen":
                    row["keep"] = bool(st["mean"] is not None and screen_keep(h["kind"], st["mean"]))
                if h["primary"].startswith("cad:"):
                    sc = "cad" + str(int(float(p.split("m=")[1])))
                    rr = [r["people_per_round"][sc] / r["people_per_round"]["std"] for r in rows if r["n"] == n and r["n_win"] == nw and sc in r["people_per_round"] and "std" in r["people_per_round"] and r["people_per_round"]["std"] > 0]
                    row["people_ratio_vs_baseline"] = round(float(np.mean(rr)), 3) if rr else None
                if stage in ("holdout", "robust"):
                    row["fresh_ok"] = fresh_same_sign(h["kind"], st)
                    row["n_fresh_logs"] = sum(1 for k in diffs[p][(nw, n)] if k in FRESH)
                    row["pass"] = bool(st["pass_"] and row["fresh_ok"])
                table.append(row)
    return table


def main():
    stage = sys.argv[1]
    if stage in ("screen", "full", "holdout", "robust"):
        rows = rows_of(stage)
        if not rows and stage == "full" and "screen" in load_surv() and not load_surv()["screen"]:
            surv = load_surv(); surv["full"] = []; surv["full_n"] = {}
            json.dump(surv, open(f"{TAB}/tourney{GEN}_survivors.json", "w"), indent=1)
            print("full: every hypothesis was killed at screen; nothing to run")
            return
        if not rows and stage == "robust" and not load_surv().get("holdout"):
            print("robust: nothing was confirmed on the holdout logs; nothing to run")
            return
        if not rows:
            sys.exit(f"no rows for stage {stage}")
        check_complete(stage, rows)
        table = stage_table(stage, rows)
        save(f"tourney{GEN}_{stage}.csv", table)
        surv = load_surv()
        ids = [h["id"] for h in REG["hypotheses"]]
        if stage == "screen":
            surv["screen"] = [i for i in ids if any(r["keep"] for r in table if r["hypothesis"] == i and r["role"] == "primary")]
        else:
            ns = {i: sorted({r["cid"] for r in table if r["hypothesis"] == i and r["role"] == "primary" and r["pass"]}, key=str) for i in ids}
            surv[stage + "_n"] = {i: v for i, v in ns.items() if v}
            surv[stage] = [i for i in ids if ns[i]]
        json.dump(surv, open(f"{TAB}/tourney{GEN}_survivors.json", "w"), indent=1)
        for r in table:
            if r["role"] != "primary":
                continue
            tag = ("KEEP" if r["keep"] else "KILL") if stage == "screen" else ("PASS" if r["pass"] else ("----" if r["testable"] else "n/a "))
            print(f"{r['hypothesis']} {r['rule']:13s} n={r['n']:<4d} w={r['n_win']:<3d} logs={r['n_logs']:2d} mean={r['mean']:+.4f} lo={r['lo']:+.4f} hi={r['hi']:+.4f} "
                  f"pos={r['frac_pos']:.2f} min={r['minimum']:+.4f} {tag}")
        print(stage, "->", surv[stage])
    elif stage == "cal":
        cal()
    elif stage == "offline":
        offline()
    elif stage == "final":
        final()


def best_h(m):
    return max([p for p in HGRID if p in m], key=lambda p: m[p])


def hval(p):
    s = p.split("h=")[1]
    return float("inf") if s == "inf" else float(s)


def nearest_grid(h):
    g = [hval(p) for p in HGRID]
    cand = [(abs(np.log2(min(x, 16.0)) - np.log2(min(h, 16.0))), x) for x in g]
    return min(cand)[1]


def key_of(h):
    return f"decay:h={'inf' if not np.isfinite(h) else (str(int(h)) if h == int(h) else str(h))}"


def cal():
    """H8 (a): at 90 windows calendar-predicted vs rounds-predicted half-life on the logs flagged 'same in days'.
    H8 (b): fresh Amazon logs at 30 windows: NDCG at the half-life nearest 534 days vs the best grid NDCG."""
    rows = rows_of("cal")
    check_complete("cal", rows)
    prev = {r["dataset"]: r for r in csv.DictReader(open(f"{TAB}/an_track_calendar.csv")) if r["n"] == "100"}
    out = []
    for nw in (90, 45):
        for r in rows:
            if r["n_win"] != nw:
                continue
            m = cell_means(r)
            d = r["dataset"]
            old = [float(prev[d][f"best_days_w{w}"]) for w in (30, 60) if d in prev and prev[d][f"best_days_w{w}"] not in ("", "inf")]
            flagged = d in prev and prev[d]["max_over_min_days"] == "1.0"
            b = best_h(m)
            days = hval(b) * r["window_days"] if np.isfinite(hval(b)) else float("inf")
            ref = float(np.median(old)) if old else None
            rec = dict(dataset=d, n_win=nw, window_days=round(r["window_days"], 2), best_h=b.split("=")[1], best_days=round(days, 1) if np.isfinite(days) else "inf",
                       ref_days_w30_w60=round(ref, 1) if ref else "", flagged_same_in_days=flagged)
            if flagged and ref:
                cal_p = key_of(nearest_grid(ref / r["window_days"]))
                h30 = prev[d]["best_h_w30"]
                rnd_p = key_of(float("inf") if h30 == "inf" else float(h30))
                rec.update(cal_pred=cal_p.split("=")[1], rounds_pred=rnd_p.split("=")[1], separable=cal_p != rnd_p,
                           ndcg_cal=round(m.get(cal_p, np.nan), 5), ndcg_rounds=round(m.get(rnd_p, np.nan), 5),
                           calendar_wins=bool(cal_p != rnd_p and m.get(cal_p, -1) > m.get(rnd_p, -1)))
            out.append(rec)
    if out:
        save(f"tourney{GEN}_cal.csv", out)
    for nw in (90, 45):
        sep = [o for o in out if o["n_win"] == nw and o.get("separable")]
        fl = [o for o in out if o["n_win"] == nw and o["flagged_same_in_days"]]
        k = sum(o["calendar_wins"] for o in sep)
        print(f"H8(a) at {nw} windows: flagged logs run {len(fl)}; separable {len(sep)}; calendar prediction beats rounds prediction in {k}/{len(sep)}")
    fr = []
    for r in rows_of("holdout"):
        if r["dataset"] in FRESH and r["n"] == 100 and r["n_win"] == 30:
            m = cell_means(r)
            if not all(p in m for p in HGRID):
                continue
            pk = key_of(nearest_grid(534.0 / r["window_days"]))
            b = best_h(m)
            fr.append(dict(dataset=r["dataset"], window_days=round(r["window_days"], 1), best_h=b.split("=")[1], cal_h=pk.split("=")[1],
                           regret_of_534_days=round(m[b] - m[pk], 5), within_0_005=bool(m[b] - m[pk] <= 0.005)))
    if fr:
        save(f"tourney{GEN}_cal_fresh.csv", fr)
        print("H8(b) fresh logs with calendar-rule regret <= 0.005:", sum(f["within_0_005"] for f in fr), "/", len(fr))
    else:
        print("H8(b): no fresh-log h-grid rows (holdout stage not run at n=100)")


def offline():
    """H7: leave-one-log-out line from new-item share to log2(best half-life) vs fixed half-life 2 (tracking-run data).
    Registered wording was unreachable; see registry h7.amendment. Reports mean regret (primary) and share of logs better."""
    new = {r["dataset"]: float(r["new_share_next_head"]) for r in csv.DictReader(open(f"{TAB}/an_track_m0m1_w30_steps.csv"))}
    H = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf]
    per = {}
    for ln in open(f"{OUT}/02_runs/track.jsonl"):
        r = json.loads(ln)
        if r.get("kind") == "n" and r.get("n_win", 30) == 30 and r["n"] == 100 and r["dataset"] in new:
            per[r["dataset"]] = np.nanmean(np.array(r["truth_ndcg"], float), axis=(0, 2))
    logs = sorted(per)
    ys = {d: np.log2(min(H[int(np.argmax(per[d]))], 16.0)) for d in logs}
    rows = []
    for d in logs:
        tr = [x for x in logs if x != d]
        X = np.array([new[x] for x in tr]); Y = np.array([ys[x] for x in tr])
        b, a = np.polyfit(X, Y, 1)
        pred = 2.0 ** float(np.clip(a + b * new[d], -2, 4))
        gi = int(np.argmin([abs(np.log2(min(h, 16.0)) - np.log2(pred)) for h in H]))
        best = float(per[d].max())
        rows.append(dict(dataset=d, new_share=round(new[d], 3), pred_h=round(pred, 2), rule_idx=gi, regret_rule=round(best - per[d][gi], 5),
                         regret_fixed2=round(best - per[d][3], 5), rule_better=bool(per[d][gi] > per[d][3] + 1e-9),
                         rule_worse=bool(per[d][gi] < per[d][3] - 1e-9)))
    save(f"tourney{GEN}_offline_h7.csv", rows)
    dm = np.array([r["regret_rule"] - r["regret_fixed2"] for r in rows])
    t = stats.ttest_1samp(dm, 0.0)
    print(f"H7: logs {len(rows)}; rule better in {sum(r['rule_better'] for r in rows)}, worse in {sum(r['rule_worse'] for r in rows)}, "
          f"same in {len(rows) - sum(r['rule_better'] or r['rule_worse'] for r in rows)}; mean regret rule {np.mean([r['regret_rule'] for r in rows]):.4f} "
          f"vs fixed-2 {np.mean([r['regret_fixed2'] for r in rows]):.4f}; mean diff {dm.mean():+.4f} (t p={t.pvalue:.3f}); "
          f"H7 holds if rule mean regret is lower with p<0.05")


def final():
    surv = load_surv()
    stages = ("screen", "full", "holdout", "robust")
    tabs = {s: (list(csv.DictReader(open(f"{TAB}/tourney{GEN}_{s}.csv"))) if os.path.exists(f"{TAB}/tourney{GEN}_{s}.csv") else []) for s in stages}
    prim = lambda s, i: [r for r in tabs[s] if r["hypothesis"] == i and r["role"] == "primary"]
    ledger = []
    for h in REG["hypotheses"]:
        i = h["id"]
        fn = set(surv.get("full_n", {}).get(i, []))
        hn = set(surv.get("holdout_n", {}).get(i, []))
        rn = set(surv.get("robust_n", {}).get(i, []))
        conf = sorted(fn & hn, key=str)
        robust_run = "robust" in surv
        if "screen" not in surv:
            status = "screen stage not analysed (INCOMPLETE)"
        elif i not in surv["screen"]:
            status = "killed at screen"
        elif "full" not in surv:
            status = "discovery stage not analysed (INCOMPLETE)"
        elif not fn:
            rr = [r for r in prim("full", i) if r["testable"] == "True"]
            if h["kind"] == "noninferiority" and rr and all(float(r["mean"]) > -MARGIN for r in rr):
                status = "not shown on discovery logs (mean loss inside the margin, but the interval is wider than the margin)"
            elif h["kind"] == "superiority" and any(float(r["mean"]) >= MARGIN and float(r["frac_pos"]) >= 0.7 for r in rr):
                status = "effect seen but NOT proven on discovery logs (mean gain above the margin, interval reaches 0 at the strict level)"
            else:
                status = "failed on discovery logs"
            if hn:
                status += "; the holdout logs alone pass at n=" + ",".join(map(str, sorted(hn, key=str)))
        elif "holdout_n" not in surv:
            status = "passed discovery, holdout not analysed (INCOMPLETE)"
        elif not conf:
            status = "passed discovery, failed holdout"
        elif robust_run:
            rob_n = set(REG["stages"].get("robust", {}).get("n", []))
            held, failed, untested = sorted(set(conf) & rn, key=str), sorted((set(conf) & rob_n) - rn, key=str), sorted(set(conf) - rob_n, key=str)
            parts = []
            if held:
                parts.append("CONFIRMED at n=" + ",".join(map(str, held)) + " (discovery, holdout, 60 windows; Amazon-dominated logs)")
            if failed:
                parts.append("confirmed on holdout at n=" + ",".join(map(str, failed)) + ", did NOT hold at 60 windows")
            if untested:
                parts.append("confirmed on holdout at n=" + ",".join(map(str, untested)) + " (30 windows only; the 60-window check runs at n=100)")
            status = "; ".join(parts)
        elif "robust" not in REG["stages"]:
            status = "CONFIRMED at n=" + ",".join(map(str, conf)) + " (discovery and holdout; Amazon-dominated logs)"
        else:
            status = "confirmed on holdout at n=" + ",".join(map(str, conf)) + " (60-window check not run)"
        def mean_at(s):
            rr = prim(s, i)
            return "; ".join(f"n={r['n']}:{float(r['mean']):+.4f}" for r in rr) if rr else ""
        ledger.append(dict(id=i, name=h["name"], kind=h["kind"], rule=h["primary"], status=status,
                           screen=mean_at("screen"), full=mean_at("full"), holdout=mean_at("holdout"), robust=mean_at("robust")))
    save(f"tourney{GEN}_ledger.csv", ledger)
    json.dump(ledger, open(f"{TAB}/tourney{GEN}_final.json", "w"), indent=1)
    for r in ledger:
        print(r["id"], r["rule"], "->", r["status"])


if __name__ == "__main__":
    main()
