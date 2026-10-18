"""Programme L (Laplace instead of Gaussian noise), development stage on the weekly-study windows w1 (7-day rounds). Nothing here is a registered result.

    nice -n 10 caffeinate -i .venv/bin/python3 -W ignore scripts/ssp/tune_lap.py --run [--workers 5] [--smoke LOG]
    .venv/bin/python3 -W ignore scripts/ssp/tune_lap.py --report

Cells: 100 people (stage a, 27 logs), 300 people (stage d, 24 logs) and 1,000 people (stage a, 17 logs), built from the STORED weekly cells (02_runs/weekly.jsonl): same logs, same std arms, same seed
tag, so the std arms replicate the stored cell (--report checks it). Each cell adds the days arms under 15 central samplers:
  eps = 2, 4, 8:  gaussn:eps=E,C=1 (L2 unit length + analytic Gaussian, doc 18's best design), lapn:eps=E,C=1 (L1 unit, "one vote per person", + Laplace),
                  gauss:eps=E,c=1 (one random row + Gaussian), lap:eps=E,c=1 (one random row + Laplace; same signal, other noise)
  eps = 1e6:      the noise-free references gaussn, lapn and gauss c=1 (the loss from the bound alone)
The price of a sampler is the selector value of the stored plain-sampling arm minus the selector value under the sampler (as in doc 18). Rows go to
02_runs/lap_dev.jsonl (gitignored; --smoke writes to /tmp); the report goes to 03_tables/price/laplace_dev_*.csv.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import quiet as Q  # noqa: E402  (one BLAS thread per process before numpy)

RUNS = "results/small_sample_popularity/02_runs"
OUT = "results/small_sample_popularity/03_tables/price"
SEED_TAG = {"a": "weekly-a", "d": "weekly-d"}
WANTED = {(100, "a"), (300, "d"), (1000, "a")}
EPS = (2, 4, 8)
REF = 1000000
DELTA = 1e-5
PARTICIPATIONS = (1, 5, 10, 52)
CATALOGUE = 1e5                  # items the noise sits on, for the expected-maximum yardstick


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def schemes():
    out = []
    for e in EPS:
        out += [f"gaussn:eps={e},C=1", f"lapn:eps={e},C=1", f"gauss:eps={e},c=1", f"lap:eps={e},c=1"]
    return out + [f"gaussn:eps={REF},C=1", f"lapn:eps={REF},C=1", f"gauss:eps={REF},c=1"]


def stored_cells():
    out = {}
    for line in open(f"{RUNS}/weekly.jsonl"):
        r = json.loads(line)
        if r.get("kind") != "deep" or r["round_days"] != 7 or "@frac" in r["arms"][0]:
            continue
        stage = r["key"].split("|", 1)[0]
        if (r["n"], stage) in WANTED:
            out[(r["dataset"], r["n"])] = (stage, r)
    return out


def build(only=None):
    tasks = []
    for (log, n), (stage, r) in stored_cells().items():
        if only and log != only:
            continue
        arms = list(r["arms"])
        days = [a for a in arms if a.startswith("decay:days=") and "@" not in a]
        tasks.append(dict(family="plap", window="w1", dataset=log, n=n, n_win=r["n_win"], round_days=7, draws=len(r["nd100"]), seed_tag=SEED_TAG[stage],
                          arms=arms + [f"{a}@{s}" for s in schemes() for a in days], key=f"p|lap|{log}|{n}|7"))
    tasks.sort(key=lambda t: (t["n"], t["dataset"]))
    return tasks


def run():
    only = arg("--smoke")
    tasks = build(only)
    print(f"[{time.strftime('%H:%M:%S')}] {len(tasks)} dev cells over {len(set(t['dataset'] for t in tasks))} logs; arms per cell {sorted(set(len(t['arms']) for t in tasks))}", flush=True)
    if "--plan" in sys.argv:
        return
    import lap_core as L
    out = "/tmp/lap_dev_smoke.jsonl" if only else f"{RUNS}/lap_dev.jsonl"
    lim = Q.Limits.from_json("configs/ssp_quiet.json")
    lim.max_workers = int(arg("--workers", 3))
    gov = Q.Governor(lim, log_path="results/small_sample_popularity/00_safety/governor_log_lap.jsonl")
    try:
        done = Q.run_tasks(L.run_task, tasks, gov, out)
        print(f"[{time.strftime('%H:%M:%S')}] ran {done} new cells; governor {gov.last}", flush=True)
    finally:
        gov.close()


def write_csv(name, head, body):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(head)
        w.writerows(body)
    print("wrote", f"{OUT}/{name}")


def eps_for_sigma(sigma: float, delta: float = DELTA) -> float:
    """The epsilon whose analytic-Gaussian sigma (L2 sensitivity 1, this delta) equals `sigma` (sigma(eps) decreases with eps; bisection)."""
    import deep_core as C
    lo, hi = 1e-6, 1e4
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if C.gauss_sigma(mid, 1.0, delta) > sigma else (lo, mid)
    return hi


def expected_max_constants(n: int = int(CATALOGUE)):
    """E[max |Z|] of n standard normals (numerical integral of 1 - P(|Z| <= x)^n) and E[max |X|] / b of n Laplace(b) values (= the harmonic number H_n); exact, no simulation."""
    import numpy as np
    from scipy.special import erf
    x = np.linspace(0.0, 12.0, 400001)
    c_gauss = float(np.trapezoid(1.0 - erf(x / math.sqrt(2.0)) ** n, x))
    c_laplace = float(np.sum(1.0 / np.arange(1, n + 1)))
    return c_gauss, c_laplace


def lifetime_rows():
    """Equal-lifetime comparison (arithmetic, not a certified accounting), worst case over the rows a person sends. Pure eps-DP adds up linearly (k eps; Dwork and Roth Cor. 3.15);
    k independent Gaussian releases N(0, sigma^2) whose per-round L2 sensitivity is 1 are equivalent to ONE release with sigma / sqrt(k) (only the sum of the squared sensitivities
    matters), so the lifetime epsilon (delta = 1e-5) is the epsilon whose analytic-Gaussian sigma equals sigma / sqrt(k) (Balle and Wang 2018); for k = 1 it is the nominal eps.
    The Laplace per-round epsilon with the SAME lifetime epsilon is lifetime / k (delta = 0; a zCDP route for the Laplace release would give a looser lifetime epsilon at these eps).
    Both noises are compared by their standard deviation and by the exact expected maximum |noise| over CATALOGUE items."""
    import deep_core as C
    c_gauss, c_laplace = expected_max_constants()
    rows = []
    for e in EPS:
        sig = C.gauss_sigma(float(e), 1.0)
        for k in PARTICIPATIONS:
            life = float(e) if k == 1 else eps_for_sigma(sig / math.sqrt(k))
            el = life / k                                                   # Laplace per-round epsilon with the same lifetime epsilon
            std_l, std_g = math.sqrt(2.0) / el, sig
            max_l, max_g = c_laplace / el, c_gauss * sig
            rows.append([e, k, round(sig, 4), round(life, 2), round(el, 3), round(std_g, 3), round(std_l, 3), round(std_g / std_l, 2), round(max_g, 2), round(max_l, 2), round(max_g / max_l, 2)])
    return rows


def report():
    import numpy as np
    import an_stress as AS
    from an_weekly import boot
    rows = {}
    for line in open(arg("--file", f"{RUNS}/lap_dev.jsonl")):
        r = json.loads(line)
        rows[r["key"]] = r
    stored = stored_cells()
    same = bad = 0
    per = {}                                            # per[(n, scheme)][log] = rule values
    for r in rows.values():
        stage, s = stored[(r["dataset"], r["n"])]
        a, b = np.array(r["nd100"], float)[:, :len(s["arms"]), :], np.array(s["nd100"], float)
        sa_, sb_ = np.array(r["selfnd"], float)[:, :len(s["arms"]), :], np.array(s["selfnd"], float)
        ok = a.shape == b.shape and np.allclose(a, b, atol=1e-9, equal_nan=True) and np.allclose(sa_, sb_, atol=1e-9, equal_nan=True)
        same, bad = same + ok, bad + (not ok)
        r["_nd"], r["_se"], r["_sa"] = np.array(r["nd100"], float), np.array(r["selfnd"], float), None
        for suffix in [""] + schemes():
            c = AS.view(r, suffix)
            if c is not None:
                per.setdefault((r["n"], suffix or "std"), {})[r["dataset"]] = AS.rule_values(c)
    print(f"std arms identical to the stored cell: {same} of {same + bad} cells")
    head = ["n", "scheme", "logs", "selector_value", "best_fixed_value", "price_selector_vs_std", "price_best_vs_std", "lost_share_of_std_selector"]
    body = []
    for n in sorted({k[0] for k in per}):
        std = per[(n, "std")]
        for s in ["std"] + schemes():
            if (n, s) not in per:
                continue
            ls = sorted(per[(n, s)])
            sel = np.array([per[(n, s)][l]["v_sel"] for l in ls])
            best = np.array([per[(n, s)][l]["best"] for l in ls])
            ssel = np.array([std[l]["v_sel"] for l in ls])
            sbest = np.array([std[l]["best"] for l in ls])
            body.append([n, s, len(ls), round(float(sel.mean()), 5), round(float(best.mean()), 5), round(float((ssel - sel).mean()), 5),
                         round(float((sbest - best).mean()), 5), round(float((ssel - sel).mean() / ssel.mean()), 4)])
    write_csv("laplace_dev_grid.csv", head, body)
    for r_ in body:
        print(r_)
    pairs = []
    for n in sorted({k[0] for k in per}):
        for e in EPS:
            for lab, a_, b_ in (("lapn - gaussn (L1 unit + Laplace vs L2 unit + Gaussian)", f"lapn:eps={e},C=1", f"gaussn:eps={e},C=1"),
                                ("lap c=1 - gauss c=1 (same one-row signal, Laplace vs Gaussian)", f"lap:eps={e},c=1", f"gauss:eps={e},c=1"),
                                ("lapn - lap c=1 (L1 unit vs one row, both Laplace)", f"lapn:eps={e},C=1", f"lap:eps={e},c=1")):
                A, B = per.get((n, a_)), per.get((n, b_))
                if not A or not B:
                    continue
                ls = sorted(set(A) & set(B))
                d = [A[l]["v_sel"] - B[l]["v_sel"] for l in ls]
                m, lo, hi = boot(d)
                pairs.append([n, e, lab, len(ls), round(m, 5), round(lo, 5), round(hi, 5), sum(x > 0 for x in d)])
    for n in sorted({k[0] for k in per}):                  # equal-lifetime brackets at k = 5 participations (Gaussian eps 4 and 8): the Laplace eps needed is 2.08 and 4.43, the grid has 2 and 4 (slightly less: conservative)
        for el, eg in ((2, 4), (4, 8)):
            A, B = per.get((n, f"lapn:eps={el},C=1")), per.get((n, f"gaussn:eps={eg},C=1"))
            if not A or not B:
                continue
            ls = sorted(set(A) & set(B))
            d = [A[l]["v_sel"] - B[l]["v_sel"] for l in ls]
            m, lo, hi = boot(d)
            pairs.append([n, f"{el} vs {eg}", "lapn - gaussn at about equal lifetime epsilon for k = 5 participations (Laplace eps 2.08 needed for Gaussian 4, 4.43 for Gaussian 8; the grid values are slightly smaller)", len(ls), round(m, 5), round(lo, 5), round(hi, 5), sum(x > 0 for x in d)])
    write_csv("laplace_dev_paired.csv", ["n", "eps_per_round (Laplace vs Gaussian where two values are given; Gaussian delta 1e-5, Laplace delta 0)", "contrast (selector value, first minus second; positive = first better)", "logs", "mean", "ci_lo", "ci_hi", "logs_positive"], pairs)
    for p in pairs:
        print(p)
    lhead = ["gaussian_eps_per_round", "participations_k", "gaussian_sigma", "lifetime_eps_delta_1e-5", "laplace_eps_per_round_same_lifetime", "gaussian_std", "laplace_std",
             "std_ratio_gauss_over_laplace", "gaussian_expected_max_noise", "laplace_expected_max_noise", "max_ratio_gauss_over_laplace"]
    write_csv("laplace_dev_lifetime.csv", lhead, lifetime_rows())
    for row in lifetime_rows():
        print(row)


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif "--run" in sys.argv or "--plan" in sys.argv:
        run()
    else:
        sys.exit(__doc__)
