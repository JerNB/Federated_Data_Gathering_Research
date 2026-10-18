"""Check of the Laplace samplers (programme L, development stage). Exits 1 on any failed check.

On the first rounds of one log (default amazon-musical-instruments, 100 people, draw 0, window w1, seed tag weekly-a) it verifies, for `lapn:eps=E,C=1` and
`lap:eps=E,c=1`:
  1. the noise-free arms (eps = 1e6) pick the SAME people as the std sampler: the support of their counts equals the support of the std counts;
  2. with eps = 1e6 the total weight equals the number of people: every person carries weight 1 (L1 unit) or one row (cap 1);
  3. on items nobody in the sample rated (pure noise) the added noise is Laplace with scale C / E: mean ~ 0, std ~ sqrt(2) C / E, tail P(|x| > 4 scale) ~ exp(-4);
  4. no noise on items outside the catalogue (items not yet seen are exactly 0 when nobody rated them).

    .venv/bin/python3 -W ignore scripts/ssp/check_lap.py [LOG ...]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import deep_core as C
import lap_core  # noqa: F401  (installs the price and Laplace patches)
import weekly_core as W

TAG, N, DRAW, ROUNDS = "weekly-a", 100, 0, 40
EPS = 4.0


def check(log: str) -> int:
    D = W.prepare_weekly(log, 7)
    nw = D["n_win"]
    seed = (log, N, DRAW, nw, TAG)
    mk = lambda scheme: C.Sampler(scheme, D, N, seed, False)
    std = mk("std")
    ref_n, ref_c = mk("lapn:eps=1000000,C=1"), mk("lap:eps=1000000,c=1")
    noisy_n, noisy_c = mk(f"lapn:eps={EPS:g},C=1"), mk(f"lap:eps={EPS:g},c=1")
    bad = 0
    noise = {"lapn": [], "lap": []}
    for k in range(ROUNDS):
        o_std, o_n, o_c, o_nn, o_nc = std.round(k), ref_n.round(k), ref_c.round(k), noisy_n.round(k), noisy_c.round(k)
        if not (o_std.people == o_n.people == o_c.people == o_nn.people == o_nc.people):
            bad += 1
            print(f"{log} round {k}: people differ {o_std.people, o_n.people, o_c.people, o_nn.people, o_nc.people}")
        sup = o_std.cnt > 0
        if not np.array_equal(sup, o_n.cnt > 1e-4):
            bad += 1
            print(f"{log} round {k}: lapn support differs from std")
        if np.any((o_c.cnt > 0.5) & ~sup):
            bad += 1
            print(f"{log} round {k}: lap c=1 support is not inside std")
        for nm, o in (("lapn", o_n), ("lap", o_c)):
            total = float(o.cnt.sum(dtype=np.float64))
            if abs(total - o.people) > 1e-2:
                bad += 1
                print(f"{log} round {k}: {nm} total weight {total:.4f} != people {o.people}")
        act = D["first"] <= k
        zero = act & ~sup                                       # catalogue items nobody in the sample rated: the observation is pure noise
        noise["lapn"].append(o_nn.cnt[zero].astype(np.float64))
        noise["lap"].append(o_nc.cnt[zero].astype(np.float64))
        outside = ~act & ~sup
        if np.any(o_nn.cnt[outside] != 0) or np.any(o_nc.cnt[outside] != 0):
            bad += 1
            print(f"{log} round {k}: noise outside the catalogue")
    for nm, scale in (("lapn", 1.0 / EPS), ("lap", 1.0 / EPS)):
        x = np.concatenate(noise[nm])
        sd, tail = float(x.std()), float((np.abs(x) > 4 * scale).mean())
        ok = len(x) > 1000 and abs(float(x.mean())) < 6 * np.sqrt(2) * scale / np.sqrt(len(x)) and abs(sd / (np.sqrt(2) * scale) - 1) < 0.02 and abs(tail / np.exp(-4) - 1) < 0.15
        bad += not ok
        print(f"{log}: {nm}: {len(x)} noise values, mean {float(x.mean()):+.5f}, std {sd:.4f} (expected {np.sqrt(2) * scale:.4f}), P(|x|>4 scale) {tail:.4f} (expected {np.exp(-4):.4f}) -> {'ok' if ok else 'FAIL'}")
    print(f"{log}: {ROUNDS} rounds checked, {bad} failed checks")
    return bad


if __name__ == "__main__":
    logs = sys.argv[1:] or ["amazon-musical-instruments"]
    sys.exit(1 if sum(check(l) for l in logs) else 0)
