"""Programme V (volunteer-bias weights): weighted reports for random and volunteer samplers, added to the engine WITHOUT editing deep_core.py.

Two new sampler kinds, patched into deep_core.Sampler (round and __init__) when this module is imported:

  vstd:w=B          the same n random active people as the std sampler (shared selection stream `rsel`), every reporter's count vector weighted by
                    rows^(-B) (rows = the reporter's rows in the round, known to the server from the report itself)
  vpps:a=A,w=B      volunteers drawn with probability proportional to rows^A in the SAME round (Efraimidis-Spirakis keys; A = 1: heavy volunteers,
                    A = -1: light volunteers), with the same weighting rows^(-B)

B = 0 is the plain count of doc 16 (no weighting). The selection stream of vpps is seeded from (seed, kind, A) and does NOT depend on B, so all
weight exponents of one volunteer rule see the SAME reporters in a draw (paired comparison). Horvitz-Thompson / Hansen-Hurwitz reasoning: when the
chance of reporting is proportional to rows^A and the target is the ROW-weighted total, the weight of a reporter's rows is 1 / chance, i.e. rows^(-A):
B = A is the right exponent for a small sampling fraction; with a larger fraction the chances saturate and the right exponent is smaller in size
(deep_core samplers use successive sampling without replacement). The server does not know A; programme V lets an unbiased audit sample choose B.

run_task() is the worker entry point: it installs the patch (so spawned workers get it too) and dispatches to weekly_core (window w1) or
window3_core (window w3).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import deep_core as C

_PATCHED = False


def weighted_counts(sampler, k: int, sel: np.ndarray, beta: float) -> np.ndarray:
    """Item counts of the selected people in round k, every person's rows weighted by rows^(-beta)."""
    D = sampler.D
    a, b = D["off"][k], D["off"][k + 1]
    sampler.mask[sel] = True
    m = sampler.mask[D["user"][a:b]]
    sampler.mask[sel] = False
    u = D["user"][a:b][m]
    it = D["item"][a:b][m]
    if len(u) == 0:
        return np.zeros(D["n_items"], np.float32)
    if beta == 0.0:
        return np.bincount(it, minlength=D["n_items"]).astype(np.float32)
    _, inv, rows = np.unique(u, return_inverse=True, return_counts=True)
    w = rows.astype(np.float64) ** (-beta)
    return np.bincount(it, weights=w[inv], minlength=D["n_items"]).astype(np.float32)


def install() -> None:
    global _PATCHED
    if _PATCHED:
        return
    original_round, original_init = C.Sampler.round, C.Sampler.__init__

    def init_(self, scheme, D, n, seed, need_half):
        original_init(self, scheme, D, n, seed, need_half)
        if self.kind == "vpps":
            self._vrng = np.random.default_rng(C.seed_of(*seed, "vol", f"vpps:a={float(self.par['a']):g}"))     # independent of the weight exponent

    def round_(self, k: int):
        if self.kind not in ("vstd", "vpps"):
            return original_round(self, k)
        D, n, par = self.D, self.n, self.par
        beta = float(par.get("w", 0.0))
        if self.kind == "vstd":
            uk = D["users_in"][k]
            sel = uk if len(uk) <= n else self.rsel.choice(uk, n, replace=False)     # same people as the std sampler
        else:
            a, b = D["off"][k], D["off"][k + 1]
            uk, rows = np.unique(D["user"][a:b], return_counts=True)
            if len(uk) <= n:
                sel = uk
            else:
                keys = self._vrng.random(len(uk)) ** (1.0 / (rows.astype(np.float64) ** float(par["a"])))
                sel = uk[np.argpartition(-keys, n - 1)[:n]]
        return C.Obs(weighted_counts(self, k, sel, beta), len(sel))

    C.Sampler.__init__ = init_
    C.Sampler.round = round_
    _PATCHED = True


install()


def run_task(task: dict) -> dict:
    install()
    win = task["window"]
    if win == "w3":
        import window3_core as X
        r = X.run_task(task)
    elif win == "w1":
        import weekly_core as W
        r = W.run_task(task)
        r["log"] = task["dataset"]
    else:
        raise ValueError(win)
    r["family"], r["window"] = task["family"], win
    return r
