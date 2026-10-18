"""Programme L (Laplace instead of Gaussian noise), development stage: two pure-DP central samplers, added to the engine WITHOUT editing deep_core.py or price_core.py.

Two new sampler kinds, patched into deep_core.Sampler.round when this module is imported (after price_core, whose patch it wraps):

  lap:eps=E,c=c      the same people as the std / gauss / gaussn samplers (selection stream `rsel`), every person keeps at most c random rows (the hard cap
                     of the existing `gauss:eps=E,c=c` sampler, same code path), and Laplace noise of scale c / E is added to the count of every item
                     already in the catalogue. One person added or removed changes the count vector by at most c in L1, so the release is E-DP (pure,
                     delta = 0). Same signal as gauss:eps=E,c=c (only the noise differs; the capped rows are an independent draw).
  lapn:eps=E,C=C     the same people; every person's count vector (item -> rows in the round) is scaled to L1 norm at most C (vector * min(1, C / ||v||_1)),
                     the scaled vectors are summed, Laplace noise of scale C / E goes on every catalogue item. With C = 1 every person carries total weight 1:
                     a person with r rows gives each row the weight 1 / r ("one vote per person"; the L1 analogue of the unit-length (L2) clip of `gaussn`,
                     whose rows weigh 1 / sqrt(r)). L1 sensitivity C, so the release is E-DP.

Neighbouring data = one person added or removed (as in docs 16 and 18); the catalogue (items already seen) is treated as public, as in docs 12, 16, 18; E is per round
with no composition and no amplification by sampling. The eps = 1e6 arms are the noise-free references (Laplace scale 1e-6).

run_task() is the worker entry point: it installs both patches (so spawned workers get them too) and dispatches as price_core.run_task.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import deep_core as C
import price_core as P

_PATCHED = False


def l1_counts(sampler, k: int, sel: np.ndarray, clip: float) -> np.ndarray:
    """Item counts of the selected people in round k with every person's count vector scaled to L1 norm <= clip."""
    D = sampler.D
    a, b = D["off"][k], D["off"][k + 1]
    sampler.mask[sel] = True
    m = sampler.mask[D["user"][a:b]]
    sampler.mask[sel] = False
    u = D["user"][a:b][m].astype(np.int64)
    it = D["item"][a:b][m].astype(np.int64)
    if len(u) == 0:
        return np.zeros(D["n_items"], np.float32)
    _, uinv = np.unique(u, return_inverse=True)
    rows = np.bincount(uinv).astype(np.float64)                        # rows of every person = L1 norm of the person's count vector
    scale = np.minimum(1.0, clip / rows)
    return np.bincount(it, weights=scale[uinv], minlength=int(D["n_items"])).astype(np.float32)


def install() -> None:
    global _PATCHED
    P.install()
    if _PATCHED:
        return
    previous = C.Sampler.round

    def round_(self, k: int):
        if self.kind not in ("lap", "lapn"):
            return previous(self, k)
        D, n, par = self.D, self.n, self.par
        uk = D["users_in"][k]
        sel = uk if len(uk) <= n else self.rsel.choice(uk, n, replace=False)     # same people as std / gauss / gaussn
        eps = float(par["eps"])
        if not (np.isfinite(eps) and eps > 0):
            raise ValueError(f"eps must be finite and positive, got {eps}")
        if self.kind == "lap":
            c = int(par["c"])
            cnt = self._counts(k, sel, cap=c)
            scale = c / eps
        else:
            clip = float(par["C"])
            cnt = l1_counts(self, k, sel, clip)
            scale = clip / eps
        act = D["first"] <= k                                                    # noise only on items already in the catalogue
        cnt[act] += self.rng.laplace(0.0, scale, int(act.sum())).astype(np.float32)
        return C.Obs(cnt, len(sel), noisy=True)

    C.Sampler.round = round_
    _PATCHED = True


install()


def run_task(task: dict) -> dict:
    install()
    return P.run_task(task)
