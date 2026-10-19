"""Programme C (participation cap), development stage: an optional cap on how often one person can be drawn, added to the engine WITHOUT editing deep_core.py, price_core.py or lap_core.py.

The privacy prices of docs 16, 18 and 21 are per round: nothing stops a person from being drawn again and again, and a person drawn k times loses more than the per-round epsilon. This module adds one
optional scheme parameter `kmax=K` to the two unit-norm central-DP kinds `gaussn` (price_core: L2 unit, analytic Gaussian) and `lapn` (lap_core: L1 unit, Laplace):

  gaussn:eps=E,C=C,kmax=K        lapn:eps=E,C=C,kmax=K

One Sampler instance is one draw's sequence of rounds (the engine builds a new one per draw and scheme). In round k the eligible people are the people active in that round who have been drawn fewer
than K times so far in this sequence (warm-up rounds included: the cap counts every round the engine runs). The round draws n of them, all of them when there are fewer, with the stream `rsel` of the std
sampler: until the first person reaches the cap the draw IS the std sampler's, and every arm with the same K draws the same people whatever its noise. A round with nobody eligible releases nothing
(zero counts, no noise: a release without data is not made; the uncapped gaussn / lapn samplers would add noise to such a round, which none of the 68 development cells has: the smallest weekly pool in any round is 326
active people, 650 at 300 people per round and 1,924 at 1,000). Everything else is as in price_core and lap_core: every person's count vector (item -> rows in the round) is scaled to norm at most C (L2 for gaussn, L1 for lapn), the scaled vectors are summed, and noise
goes on every item already in the catalogue (Gaussian N(0, sigma^2) with the analytic sigma(E, C, delta = 1e-5), Laplace with scale C / E).

A person drawn at most K times is in at most K releases. Pure E-DP releases compose to a lifetime K * E (Dwork and Roth, Cor. 3.15); K Gaussian releases N(0, sigma^2) with L2 sensitivity C are one
release with sigma / sqrt(K) (doc 21, section 3). This holds under ONE neighbouring relation: a drawn person's report is replaced by an empty one, while who is active in a round and who is drawn are public
(like the catalogue), so the sensitivity is C for both kinds. Under a swap relation (the person's rows replaced by other rows) the L1 sensitivity doubles and the L2 sensitivity is multiplied by sqrt(2), and if who
is active is itself private, removing a person changes who else is drawn: neither case is covered. No amplification by sampling. Nominal accounting, not a certified guarantee.

Schemes without `kmax` are untouched (the stored cells and lap_dev.jsonl reproduce). run_task() is the worker entry point (installs the patch, then dispatches as price_core.run_task).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import deep_core as C
import lap_core as L
import price_core as P

_PATCHED = False


def cap_of(par: dict) -> int:
    k = par["kmax"]
    if not (np.isfinite(k) and k >= 1 and k == int(k)):
        raise ValueError(f"kmax must be a whole number >= 1, got {k}")
    return int(k)


def install() -> None:
    global _PATCHED
    L.install()
    if _PATCHED:
        return
    previous = C.Sampler.round

    def round_(self, k: int):
        if self.kind not in ("gaussn", "lapn") or "kmax" not in self.par:
            return previous(self, k)
        D, n, par = self.D, self.n, self.par
        kmax = cap_of(par)
        eps, clip = float(par["eps"]), float(par["C"])
        if not (np.isfinite(eps) and eps > 0):
            raise ValueError(f"eps must be finite and positive, got {eps}")
        if getattr(self, "drawn", None) is None:
            self.drawn = np.zeros(D["n_users"] + 1, np.int32)                  # times each person has been drawn in this sequence of rounds
        uk = D["users_in"][k]
        elig = uk[self.drawn[uk] < kmax]
        if len(elig) == 0:
            return C.Obs(np.zeros(D["n_items"], np.float32), 0, noisy=True)
        sel = elig if len(elig) <= n else self.rsel.choice(elig, n, replace=False)     # same stream as std; identical to std until someone reaches the cap
        self.drawn[sel] += 1
        act = D["first"] <= k                                                    # noise only on items already in the catalogue
        if self.kind == "gaussn":
            cnt = P.clipped_counts(self, k, sel, clip)
            cnt[act] += self.rng.normal(0.0, C.gauss_sigma(eps, clip), int(act.sum())).astype(np.float32)
        else:
            cnt = L.l1_counts(self, k, sel, clip)
            cnt[act] += self.rng.laplace(0.0, clip / eps, int(act.sum())).astype(np.float32)
        return C.Obs(cnt, len(sel), noisy=True)

    C.Sampler.round = round_
    _PATCHED = True


install()


def run_task(task: dict) -> dict:
    install()
    return P.run_task(task)
