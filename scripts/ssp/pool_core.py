"""Pooled-list pilot (programme E, development stage): the audit people ENTER the list. Added to the engine WITHOUT editing deep_core.py or weight_core.py.

One new sampler kind, patched into deep_core.Sampler when this module is installed (after weight_core, whose patch it wraps):

  vpool:a=A,w=B,na=M   volunteers  the n people of vpps:a=A (the SAME selection stream as the programme-V arms, so comparisons with them are paired), every
                                   reporter's count vector weighted by rows^(-B);
                       audit people the M random active people of the engine's audit sample (an identical Sampler("std") built with the audit seed: the
                                   people that score the lists in cells with audit_n = M), counts unweighted (a random person's uncorrected count is
                                   unbiased for the row-weighted truth);
                       the round's share is  lam * share(volunteers) + (1 - lam) * share(audit people),  lam = n_v / (n_v + n_a)  (the head-count share).

Prequential: run_cell scores a list against the NEXT round's sample before the list is updated with that sample, so the audit people of round t score the
list built up to t - 1 and only afterwards enter it; they never score a list they are in. rows_per_round of a pooled arm is ~1 (shares), not rows.

run_task() is the worker entry point (installs both patches, then dispatches as weight_core.run_task).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import deep_core as C
import weight_core as V

_PATCHED = False


def install() -> None:
    global _PATCHED
    V.install()
    if _PATCHED:
        return
    prev_round, prev_init = C.Sampler.round, C.Sampler.__init__

    def init_(self, scheme, D, n, seed, need_half):
        prev_init(self, scheme, D, n, seed, need_half)
        if self.kind == "vpool":
            a, m = float(self.par["a"]), int(self.par["na"])
            self._vrng = np.random.default_rng(C.seed_of(*seed, "vol", f"vpps:a={a:g}"))          # the stream of vpps:a=A
            self._aud = C.Sampler("std", D, m, (seed[0], m, seed[2], seed[3], seed[4], "audit"), False)     # the engine's audit people

    def round_(self, k: int):
        if self.kind != "vpool":
            return prev_round(self, k)
        D, n, par = self.D, self.n, self.par
        a, b = D["off"][k], D["off"][k + 1]
        uk, rows = np.unique(D["user"][a:b], return_counts=True)
        if len(uk) <= n:
            sel = uk
        else:
            keys = self._vrng.random(len(uk)) ** (1.0 / (rows.astype(np.float64) ** float(par["a"])))
            sel = uk[np.argpartition(-keys, n - 1)[:n]]
        cv = V.weighted_counts(self, k, sel, float(par.get("w", 0.0)))
        oa = self._aud.round(k)
        tot = len(sel) + oa.people
        lam = len(sel) / tot if tot > 0 else 1.0
        return C.Obs(lam * C.norm(cv) + (1.0 - lam) * C.norm(oa.cnt), tot)

    C.Sampler.__init__ = init_
    C.Sampler.round = round_
    _PATCHED = True


install()


def run_task(task: dict) -> dict:
    install()
    return V.run_task(task)
