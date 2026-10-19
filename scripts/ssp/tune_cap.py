"""Programme C (participation cap, equal-lifetime pairs), development stage on the weekly-study windows w1 (7-day rounds). Nothing here is a registered result.

    nice -n 10 caffeinate -i .venv/bin/python3 -W ignore scripts/ssp/tune_cap.py --run [--set main|k1] [--workers 4] [--smoke LOG]
    .venv/bin/python3 -W ignore scripts/ssp/tune_cap.py --plan [--set k1]
    .venv/bin/python3 -W ignore scripts/ssp/tune_cap.py --report [--file F] [--file1 F1]

Cells: the same 68 cells as programme L (100 people stage a, 27 logs; 300 people stage d, 24 logs; 1,000 people stage a, 17 logs), built from the STORED weekly cells (02_runs/weekly.jsonl): same logs, same
std arms, same seed tag (--report checks that the std arms replicate the stored cell). Each cell adds the days arms under central samplers (cap_core.py), in two sets (two row files; --report reads both):
  set main (16 samplers, 02_runs/cap_dev.jsonl):
    cap at 5 draws per person over the whole window:   gaussn eps 4 and 8, lapn eps 2 and 4 (equal lifetime epsilon: Gaussian 4 -> 10.4 and Laplace 2.08, Gaussian 8 -> 22.2 and Laplace 4.43; the grid values 2 and 4 are
                                                       slightly smaller, so the Laplace arm is protected slightly more), noise-free references of both kinds (eps 1e6)
    cap at 10 draws:                                   gaussn eps 4 and 8, lapn eps 1.6 and 3.5 (Gaussian 4 -> 16.1 and Laplace 1.61, Gaussian 8 -> 35.6 and Laplace 3.56), noise-free references of both kinds
    no cap, larger L1 bound:                           lapn eps 4 with C = 2 and C = 4, and their noise-free references (the uncapped C = 1 arms are in programme L's lap_dev.jsonl)
  set k1 (8 samplers, 02_runs/cap_dev_k1.jsonl): single-use participants, nobody is drawn twice, so the per-round eps is the whole lifetime eps: gaussn and lapn at eps 2, 4 and 8 (the SAME eps for both kinds;
    Gaussian delta 1e-5, Laplace delta 0) and the noise-free references of both kinds.
Lifetime epsilon is exact composition for the Gaussian (sigma / sqrt(k), delta 1e-5) and k * eps for Laplace (pure), valid for a person drawn at most k times (laplace_dev_lifetime.csv; the cap makes that true)
and ONLY under this neighbouring relation: a drawn person's report is replaced by an empty one, while who is active in a round and who is drawn are public (sensitivity C for both kinds). Under a swap relation
(a drawn person's rows replaced by other rows) the L1 sensitivity doubles and the L2 sensitivity is multiplied by sqrt(2), which favours the Gaussian by a further factor sqrt(2) in noise.
Rows go to 02_runs/cap_dev.jsonl and cap_dev_k1.jsonl (gitignored; --smoke writes to /tmp); the report (it also reads programme L's 02_runs/lap_dev.jsonl for the uncapped arms) goes to 03_tables/price/cap_dev_*.csv.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import quiet as Q  # noqa: E402  (one BLAS thread per process before numpy)
import tune_lap as TL  # noqa: E402

RUNS, OUT, REF = TL.RUNS, TL.OUT, TL.REF
arg, write_csv = TL.arg, TL.write_csv
# (kmax, Laplace eps, Gaussian eps): equal lifetime epsilon up to the grid (k = 1: single-use participants, the same eps for both)
LIFETIME_PAIRS = ((1, 2, 2), (1, 4, 4), (1, 8, 8), (5, 2, 4), (5, 4, 8), (10, 1.6, 4), (10, 3.5, 8))


def schemes(which="main"):
    if which == "k1":                                                         # single-use participants: nobody is drawn twice, so the per-round eps is the whole lifetime eps
        out = []
        for e in (2, 4, 8):
            out += [f"gaussn:eps={e},C=1,kmax=1", f"lapn:eps={e},C=1,kmax=1"]
        return out + [f"gaussn:eps={REF},C=1,kmax=1", f"lapn:eps={REF},C=1,kmax=1"]
    out = []
    for k in (5, 10):
        g_eps = sorted({g for kk, _, g in LIFETIME_PAIRS if kk == k})
        l_eps = sorted({l for kk, l, _ in LIFETIME_PAIRS if kk == k})
        out += [f"gaussn:eps={e:g},C=1,kmax={k}" for e in g_eps] + [f"lapn:eps={e:g},C=1,kmax={k}" for e in l_eps]
        out += [f"gaussn:eps={REF},C=1,kmax={k}", f"lapn:eps={REF},C=1,kmax={k}"]
    for c in (2, 4):
        out += [f"lapn:eps=4,C={c}", f"lapn:eps={REF},C={c}"]
    return out


def build(only=None, which="main"):
    tasks = []
    for (log, n), (stage, r) in TL.stored_cells().items():
        if only and log != only:
            continue
        arms = list(r["arms"])
        days = [a for a in arms if a.startswith("decay:days=") and "@" not in a]
        tasks.append(dict(family="pcap", window="w1", dataset=log, n=n, n_win=r["n_win"], round_days=7, draws=len(r["nd100"]), seed_tag=TL.SEED_TAG[stage],
                          arms=arms + [f"{a}@{s}" for s in schemes(which) for a in days], key=f"p|cap{'1' if which == 'k1' else ''}|{log}|{n}|7"))
    tasks.sort(key=lambda t: (t["n"], t["dataset"]))
    return tasks


def run():
    only = arg("--smoke")
    which = arg("--set", "main")
    if which not in ("main", "k1"):
        sys.exit("--set must be main or k1")
    tasks = build(only, which)
    print(f"[{time.strftime('%H:%M:%S')}] set {which}: {len(tasks)} dev cells over {len(set(t['dataset'] for t in tasks))} logs; arms per cell {sorted(set(len(t['arms']) for t in tasks))}", flush=True)
    if "--plan" in sys.argv:
        return
    import cap_core as K
    name = "cap_dev.jsonl" if which == "main" else "cap_dev_k1.jsonl"
    out = f"/tmp/{name}.smoke" if only else f"{RUNS}/{name}"
    lim = Q.Limits.from_json("configs/ssp_quiet.json")
    lim.max_workers = int(arg("--workers", 3))
    gov = Q.Governor(lim, log_path="results/small_sample_popularity/00_safety/governor_log_cap.jsonl")
    try:
        done = Q.run_tasks(K.run_task, tasks, gov, out)
        print(f"[{time.strftime('%H:%M:%S')}] ran {done} new cells; governor {gov.last}", flush=True)
    finally:
        gov.close()


def load(file, suffixes):
    """per[(n, suffix)][log] = rule values of that sampler's days arms; ppl[(n, suffix)][log] = people per round (mean over draws); the std arms must equal the stored cell."""
    import numpy as np
    import an_stress as AS
    stored = TL.stored_cells()
    per, ppl, same, bad = {}, {}, 0, 0
    for line in open(file):
        r = json.loads(line)
        stage, s = stored[(r["dataset"], r["n"])]
        a, b = np.array(r["nd100"], float)[:, :len(s["arms"]), :], np.array(s["nd100"], float)
        sa_, sb_ = np.array(r["selfnd"], float)[:, :len(s["arms"]), :], np.array(s["selfnd"], float)
        ok = a.shape == b.shape and np.allclose(a, b, atol=1e-9, equal_nan=True) and np.allclose(sa_, sb_, atol=1e-9, equal_nan=True)
        same, bad = same + ok, bad + (not ok)
        r["_nd"], r["_se"], r["_sa"] = np.array(r["nd100"], float), np.array(r["selfnd"], float), None
        for suffix in [""] + list(suffixes):
            c = AS.view(r, suffix)
            if c is None:
                continue
            key = (r["n"], suffix or "std")
            per.setdefault(key, {})[r["dataset"]] = AS.rule_values(c)
            idx = [i for i, a_ in enumerate(r["arms"]) if a_.partition("@")[2] == suffix]
            ppl.setdefault(key, {})[r["dataset"]] = float(np.mean([r["people_per_round"][i] for i in idx]))
    print(f"{file}: std arms identical to the stored cell: {same} of {same + bad} cells")
    return per, ppl


def paired(per, ns, a_key, b_key):
    from an_weekly import boot
    out = []
    for n in ns:
        A, B = per.get((n, a_key)), per.get((n, b_key))
        if not A or not B:
            continue
        ls = sorted(set(A) & set(B))
        d = [A[l]["v_sel"] - B[l]["v_sel"] for l in ls]
        m, lo, hi = boot(d)
        out.append((n, len(ls), round(m, 5), round(lo, 5), round(hi, 5), sum(x > 0 for x in d)))
    return out


def equivalents(per, ns, pairs):
    """People-equivalent of swapping the noise: how many people the Gaussian arm needs to lose as little as the Laplace arm does at 100 and at 300 people per round (log-log interpolation of the Gaussian
    arm's lost share over the three sizes, on the logs present at all three sizes: an approximation, flagged when it has to extrapolate)."""
    import numpy as np
    common = set.intersection(*[set(per[(n, "std")]) for n in ns])

    def lost(n, key):
        ls = sorted(common & set(per[(n, key)]))
        sd = float(np.mean([per[(n, "std")][l]["v_sel"] for l in ls]))
        return (sd - float(np.mean([per[(n, key)][l]["v_sel"] for l in ls]))) / sd, len(ls)

    out, x = [], np.log(np.array(ns, float))
    for label, lk, gk in pairs:
        if any((n, k_) not in per for n in ns for k_ in (lk, gk)):
            continue
        y = np.log(np.array([lost(n, gk)[0] for n in ns]))
        for n in ns[:-1]:
            tl, nl = lost(n, lk)
            if tl <= 0:
                continue
            ty = np.log(tl)
            i = next((i for i in range(len(ns) - 1) if y[i] >= ty >= y[i + 1]), None)
            flag = ""
            if i is None:
                flag = "extrapolated below the smallest size" if ty > y[0] else "extrapolated beyond the largest size"
                i = 0 if ty > y[0] else len(ns) - 2
            eq = float(np.exp(x[i] + (ty - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])))
            out.append([label, n, nl, round(tl, 4), round(eq, 1), round(eq / n, 2), flag])
    return out


def report():
    import numpy as np
    cap_file = arg("--file", f"{RUNS}/cap_dev.jsonl")
    per, ppl = load(cap_file, schemes())
    k1_file = arg("--file1", f"{RUNS}/cap_dev_k1.jsonl")
    if os.path.exists(k1_file):                                               # the single-use set; its std arms are the same stored cells, so only its own samplers are merged
        per1, ppl1 = load(k1_file, schemes("k1"))
        per.update({k: v for k, v in per1.items() if k[1] != "std"})
        ppl.update({k: v for k, v in ppl1.items() if k[1] != "std"})
    per_l, ppl_l = load(f"{RUNS}/lap_dev.jsonl", TL.schemes())
    per_all = {**per_l, **per}
    ppl_all = {**ppl_l, **ppl}
    ns = sorted({k[0] for k in per})
    head = ["n", "scheme", "logs", "selector_value", "best_fixed_value", "price_selector_vs_std", "lost_share_of_std_selector", "people_per_round_share_of_std"]
    body = []
    shown = [f"gaussn:eps={e:g},C=1" for e in (2, 4, 8)] + [f"lapn:eps={e:g},C=1" for e in (2, 4, 8)] + [f"gaussn:eps={REF},C=1", f"lapn:eps={REF},C=1"] + schemes("k1") + schemes()
    for n in ns:
        std = per_all[(n, "std")]
        for s in ["std"] + shown:
            if (n, s) not in per_all:
                continue
            ls = sorted(set(per_all[(n, s)]) & set(std))
            sel = np.array([per_all[(n, s)][l]["v_sel"] for l in ls])
            best = np.array([per_all[(n, s)][l]["best"] for l in ls])
            ssel = np.array([std[l]["v_sel"] for l in ls])
            share = float(np.mean([ppl_all[(n, s)][l] / ppl_all[(n, "std")][l] for l in ls]))
            body.append([n, s, len(ls), round(float(sel.mean()), 5), round(float(best.mean()), 5), round(float((ssel - sel).mean()), 5), round(float((ssel - sel).mean() / ssel.mean()), 4), round(share, 4)])
    write_csv("cap_dev_grid.csv", head, body)
    for r_ in body:
        print(r_)
    rows = []

    def add(kind, label, a_key, b_key):
        for n, nl, m, lo, hi, pos in paired(per_all, ns, a_key, b_key):
            rows.append([kind, label, n, a_key, b_key, nl, m, lo, hi, pos])

    for k, el, eg in LIFETIME_PAIRS:                                          # equal lifetime epsilon, both capped at k draws
        add("equal_lifetime_capped", f"k={k}: Laplace eps {el:g} minus Gaussian eps {eg:g}", f"lapn:eps={el:g},C=1,kmax={k}", f"gaussn:eps={eg:g},C=1,kmax={k}")
    for k in (1, 5, 10):                                                      # what the cap costs: capped minus uncapped with the same noise (Laplace k = 10 has no uncapped counterpart in programme L's grid)
        for eg in ((2, 4, 8) if k == 1 else (4, 8)):
            add("cap_cost", f"k={k}: Gaussian eps {eg:g} capped minus uncapped", f"gaussn:eps={eg:g},C=1,kmax={k}", f"gaussn:eps={eg:g},C=1")
    for k, els in ((1, (2, 4, 8)), (5, (2, 4))):
        for el in els:
            add("cap_cost", f"k={k}: Laplace eps {el:g} capped minus uncapped", f"lapn:eps={el:g},C=1,kmax={k}", f"lapn:eps={el:g},C=1")
    for k in (1, 5, 10):
        add("cap_cost_noise_free", f"k={k}: Gaussian unit signal, capped minus uncapped (eps 1e6)", f"gaussn:eps={REF},C=1,kmax={k}", f"gaussn:eps={REF},C=1")
        add("cap_cost_noise_free", f"k={k}: Laplace unit signal, capped minus uncapped (eps 1e6)", f"lapn:eps={REF},C=1,kmax={k}", f"lapn:eps={REF},C=1")
    for c in (2, 4):                                                          # larger L1 bound, no cap, per-round eps 4
        add("l1_bound", f"C={c} minus C=1, Laplace eps 4 (noise scale C / eps)", f"lapn:eps=4,C={c}", "lapn:eps=4,C=1")
        add("l1_bound_noise_free", f"C={c} minus C=1, signal only (eps 1e6)", f"lapn:eps={REF},C={c}", f"lapn:eps={REF},C=1")
        add("l1_bound_vs_gaussian", f"Laplace eps 4 with C={c} minus Gaussian eps 4 unit length", f"lapn:eps=4,C={c}", "gaussn:eps=4,C=1")
    seen, uniq = set(), []
    for r_ in rows:
        key = tuple(r_[:5])
        if key not in seen:
            seen.add(key)
            uniq.append(r_)
    write_csv("cap_dev_paired.csv", ["kind", "contrast (selector value, first minus second; positive = first better)", "n", "first", "second", "logs", "mean", "ci_lo", "ci_hi", "logs_positive"], uniq)
    for r_ in uniq:
        print(r_)
    prow = []
    for n in ns:
        for k in (1, 5, 10):
            key = (n, f"gaussn:eps=4,C=1,kmax={k}")
            for l in sorted(ppl_all.get(key, {})):
                prow.append([n, l, k, round(ppl_all[key][l], 3), round(ppl_all[(n, "std")][l], 3), round(ppl_all[key][l] / ppl_all[(n, "std")][l], 4)])
    write_csv("cap_dev_people.csv", ["n", "log", "kmax", "people_per_round_capped", "people_per_round_std", "ratio"], prow)
    for n in ns:
        for k in (1, 5, 10):
            rr = [r_[5] for r_ in prow if r_[0] == n and r_[2] == k]
            if rr:
                print(f"n={n} kmax={k}: people per round vs std, median {np.median(rr):.3f}, min {min(rr):.3f}, logs below 0.99: {sum(x < 0.99 for x in rr)}, below 0.9: {sum(x < 0.9 for x in rr)}, below 0.5: {sum(x < 0.5 for x in rr)} of {len(rr)}")
    pairs = [(f"per round, eps {e}: Laplace vs Gaussian, no cap", f"lapn:eps={e},C=1", f"gaussn:eps={e},C=1") for e in (2, 4, 8)]
    pairs += [(f"cap {k}: Laplace eps {el:g} vs Gaussian eps {eg:g}", f"lapn:eps={el:g},C=1,kmax={k}", f"gaussn:eps={eg:g},C=1,kmax={k}") for k, el, eg in LIFETIME_PAIRS]
    eqs = equivalents(per_all, ns, pairs)
    write_csv("cap_dev_equivalents.csv", ["contrast", "people_per_round_laplace", "logs_common_to_all_sizes", "laplace_lost_share", "gaussian_people_needed_for_the_same_lost_share", "ratio", "flag"], eqs)
    for r_ in eqs:
        print(r_)


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif "--run" in sys.argv or "--plan" in sys.argv:
        run()
    else:
        sys.exit(__doc__)
