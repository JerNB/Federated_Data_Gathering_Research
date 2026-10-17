"""Check of the pooled sampler (programme E pilot, doc 20). Exits 1 on any mismatch.

For every round the pooled share must equal  lam * share(volunteers) + (1 - lam) * share(audit people),  lam = n_v / (n_v + n_a), where the volunteers are the people
of `vpps:a=1` (same seed) and the audit people those of the engine's audit sample (same seed), on the data window the cells use (weekly_core.prepare_weekly, the
densest 156 weeks). Also checks that each pooled round sums to 1 (or is all zero) and that the audit stream is reproducible.

    .venv/bin/python3 -W ignore scripts/ssp/check_pool.py [LOG ...]       (default: amazon-arts-crafts-and-sewing and ml-latest)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import deep_core as C
import pool_core  # noqa: F401  (installs weight_core and the vpool patch)
import weekly_core as W

TAG, DRAWS, ROUNDS = "weekly-a", (0, 1), 80
NV, NA = 100, 50


def check(log: str) -> int:
    D = W.prepare_weekly(log, 7)
    nw = D["n_win"]
    bad = checked = 0
    for dr in DRAWS:
        for beta in (0.0, 0.5, -1.0):
            seed = (log, NV, dr, nw, TAG)
            pooled = C.Sampler(f"vpool:a=1,w={beta:g},na={NA}", D, NV, seed, False)
            vol = C.Sampler(f"vpps:a=1,w={beta:g}", D, NV, seed, False)
            aud = C.Sampler("std", D, NA, (log, NA, dr, nw, TAG, "audit"), False)
            for k in range(ROUNDS):
                op, ov, oa = pooled.round(k), vol.round(k), aud.round(k)
                tot = ov.people + oa.people
                lam = ov.people / tot if tot else 1.0
                exp = lam * C.norm(ov.cnt) + (1.0 - lam) * C.norm(oa.cnt)
                mass = float(op.cnt.sum())
                checked += 1
                if not np.allclose(op.cnt, exp, atol=1e-6) or op.people != tot or not (abs(mass - 1.0) < 1e-4 or mass == 0.0):
                    bad += 1
    a1, a2 = (C.Sampler("std", D, NA, (log, NA, 0, nw, TAG, "audit"), False) for _ in range(2))
    same = all(np.array_equal(a1.round(k).cnt, a2.round(k).cnt) for k in range(ROUNDS))
    print(f"{log}: {checked} rounds checked, {bad} mismatches, audit stream reproducible: {same}")
    return bad + (0 if same else 1)


if __name__ == "__main__":
    logs = sys.argv[1:] or ["amazon-arts-crafts-and-sewing", "ml-latest"]
    sys.exit(1 if sum(check(l) for l in logs) else 0)
