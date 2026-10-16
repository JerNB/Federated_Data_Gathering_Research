"""Programme P (price of privacy): a norm-clipped central-DP sampler, added to the engine WITHOUT editing deep_core.py.

Scheme `gaussn:eps=E,C=C` (a new kind, patched into deep_core.Sampler.round when this module is imported): the same people as the std and
gauss samplers (same selection stream), every person's count vector (item -> number of rows in the round) is scaled to L2 norm at most C
(vector * min(1, C / ||vector||_2); a person whose norm is already <= C is not changed), the scaled vectors are summed, and analytic Gaussian
noise with L2 sensitivity C (delta = 1e-5) is added to the count of every item already in the catalogue. Neighbouring data = one person
added or removed, so the sensitivity is C whatever the person's rows are. The existing `gauss:eps=E,c=c` scheme keeps at most c random rows
per person (hard cap, L2 sensitivity sqrt(c) when the rows are distinct items) and is used unchanged from deep_core.

run_task() is the worker entry point: it installs the patch (so spawned workers get it too) and dispatches to stress_core.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import deep_core as C

_PATCHED = False


def clipped_counts(sampler, k: int, sel: np.ndarray, clip: float) -> np.ndarray:
    """Item counts of the selected people in round k with every person's count vector scaled to L2 norm <= clip."""
    D = sampler.D
    a, b = D["off"][k], D["off"][k + 1]
    sampler.mask[sel] = True
    m = sampler.mask[D["user"][a:b]]
    sampler.mask[sel] = False
    u = D["user"][a:b][m].astype(np.int64)
    it = D["item"][a:b][m].astype(np.int64)
    if len(u) == 0:
        return np.zeros(D["n_items"], np.float32)
    n_items = int(D["n_items"])
    pair, mult = np.unique(u * n_items + it, return_counts=True)       # (person, item) -> multiplicity
    pu, pit = pair // n_items, pair % n_items
    uu, uinv = np.unique(pu, return_inverse=True)
    sq = np.bincount(uinv, weights=mult.astype(np.float64) ** 2)       # squared L2 norm of every person's vector
    scale = np.minimum(1.0, clip / np.sqrt(sq))
    return np.bincount(pit, weights=mult * scale[uinv], minlength=n_items).astype(np.float32)


def install() -> None:
    global _PATCHED
    if _PATCHED:
        return
    original = C.Sampler.round

    def round_(self, k: int):
        if self.kind != "gaussn":
            return original(self, k)
        D, n, par = self.D, self.n, self.par
        uk = D["users_in"][k]
        sel = uk if len(uk) <= n else self.rsel.choice(uk, n, replace=False)     # same people as std / gauss
        clip = float(par["C"])
        cnt = clipped_counts(self, k, sel, clip)
        sigma = C.gauss_sigma(par["eps"], clip)
        act = D["first"] <= k                                                    # noise only on items already in the catalogue
        cnt[act] += self.rng.normal(0.0, sigma, int(act.sum())).astype(np.float32)
        return C.Obs(cnt, len(sel), noisy=True)

    C.Sampler.round = round_
    _PATCHED = True


install()


def run_task(task: dict) -> dict:
    import stress_core as S
    install()
    return S.run_task(task)
