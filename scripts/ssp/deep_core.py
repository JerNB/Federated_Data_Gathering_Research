"""Deep programme core: many list-building arms on identical reporters, with several metrics.

Setting (same as track_core / tourney_core): a log is cut into n_win equal-time rounds; every round n random active people
report their rows; the server builds a list from the reports so far and the list made after round w is scored against the WHOLE
community's counts in round w+1 (NDCG@k, recall@100, ...). Arms see exactly the same reporters when they share a sampling scheme.

An arm is "<aggregator>[@<sampling scheme>]", e.g. "decay:h=2", "holt:a=0.3,b=0.1@std", "decay:h=2@panel:rho=0.25".

Sampling schemes (people chosen each round; all take at most n active people):
  std                 n random active people, all their rows (default)
  pps                 n people drawn with probability proportional to their number of rows in the round (oracle: the server cannot know it before asking)
  ppsprev             feasible variant: probability proportional to the person's rows in the previous round (1 if absent)
  panel:rho=R         persistent panel; each retained member is dropped with probability R, gaps refilled with new random people
  varn                n_w = n * exp(N(0, 0.6)) clipped to [n/4, 4n]
  cap:c=C             each person reports at most C random rows
  gauss:eps=E,c=C     central DP (epsilon per round): cap C rows, analytic Gaussian noise (delta=1e-5, L2 sensitivity sqrt(C)) on every item already in the catalogue
  oue:eps=E           local DP: one random row per person, optimised unary encoding, exact binomial simulation of the aggregate

Aggregators (state over rounds, list = best items first, up to 300):
  decay:h=H | decay:days=D   decayed sum of per-round shares (baseline h=2)
  holt:a=A,b=B,p=P           Holt damped-trend smoothing of shares (forecast = level + P*trend)
  clip:c=C                   decay h=2 on counts clipped at C rows per item and round
  rrf:k=K                    decayed sum (h=2) of reciprocal-rank scores 1/(K+rank), rank in the round's top 1000
  twospeed:hf=,hs=,age=      items first seen less than `age` rounds ago use half-life hf, older items hs
  eb:a=A                     round share shrunk toward the tracker's own share: (c + A*N*prior)/(N(1+A)), decay h=2
  drift:z=Z,f=F              decay h=2; when the round's total-variation distance to the tracker exceeds Z x its running median, state *= F
  ens                        sum of normalised decays h=1, 2, 4
  seas:g=G                   decay h=2 plus G x normalised mean share of the same season one year ago (lag = 365 days)
  hyst:d=D                   decay h=2, items on the previous published list get score x (1+D)
  reserve:r=R                top 100-R of decay h=2 plus R best rising items (share h=1 minus share h=4)
  splice:k=K,h1=,h2=         first K of the h1 list, rest of the h2 list
  ivw:g=G                    decay h=2 of shares weighted by (n_w/n)^G
  cens:m=M,w=W               decay h=2 plus the full community's share added with weight W every M-th round (census)
  censold:m=M                stale list: the last census only
  full:lag=L | fulldecay:h=H lists from the whole community's counts (L rounds old / decayed): the ceiling arms
  half:h=H                   decay on all reporters; extra metric = agreement of two half-sample trackers
  selfsel                    each round picks the half-life (0.25..8, inf) with the best cumulative self-score vs the next sampled round
  athr:t=T                   decay h=2 list, but only items with decayed report count >= T are published (list may be short)
"""
from __future__ import annotations

import os
import sys
import zlib
from collections import deque

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import track_core as T

KS = (10, 30, 100, 300)
DISC = 1.0 / np.log2(np.arange(300) + 2.0)
HGRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, float("inf")]
FIRST_STEP = 0
STEADY = 8                      # steps >= STEADY are "steady state"; earlier steps are the burn-in
DELTA = 1e-5
_SIG: dict = {}


def gauss_sigma(eps: float, sens: float, delta: float = DELTA) -> float:
    """Smallest sigma of the analytic Gaussian mechanism (Balle and Wang 2018) that is (eps, delta)-DP for L2 sensitivity `sens`."""
    key = (eps, sens, delta)
    if key not in _SIG:
        from scipy.stats import norm
        ok = lambda sg: norm.cdf(sens / (2 * sg) - eps * sg / sens) - np.exp(eps) * norm.cdf(-sens / (2 * sg) - eps * sg / sens) <= delta
        lo, hi = 1e-3 * sens, 1e4 * sens
        for _ in range(100):
            mid = np.sqrt(lo * hi)
            lo, hi = (lo, mid) if ok(mid) else (mid, hi)
        _SIG[key] = hi
    return _SIG[key]


def seed_of(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def parse(spec: str):
    kind, _, rest = spec.partition(":")
    par = {}
    for kv in filter(None, rest.split(",")):
        k, v = kv.split("=")
        par[k] = float(v)
    return kind, par


def norm(x: np.ndarray) -> np.ndarray:
    s = float(x.sum())
    return x / s if s != 0 else x


def top_idx(score: np.ndarray, m: int = 300, key: np.ndarray | None = None) -> np.ndarray:
    """Indices of the m best scores, best first; ties broken by `key` (a per-draw random permutation shared by all arms), else by index."""
    n = score.shape[0]
    m = min(m, n)
    idx = np.argpartition(-score, m - 1)[:m]
    return idx[np.lexsort(((idx if key is None else key[idx]), -score[idx]))]


def decay_factor(h: float) -> float:
    return 1.0 if np.isinf(h) else float(2.0 ** (-1.0 / h))


# ---------------------------------------------------------------- sampling schemes

class Obs:
    __slots__ = ("cnt", "people", "n_w", "half", "noisy", "idcg")

    def __init__(self, cnt, people, n_w=None, half=None, noisy=False):
        self.cnt, self.people, self.n_w, self.half, self.noisy = cnt, people, n_w if n_w is not None else people, half, noisy
        self.idcg = None

    @property
    def share(self) -> np.ndarray:
        if self.noisy:                               # DP outputs are linear: noisy count / people
            return self.cnt / max(self.people, 1)
        return norm(self.cnt)


class Sampler:
    def __init__(self, scheme: str, D: dict, n: int, seed: tuple, need_half: bool):
        self.kind, self.par = parse(scheme)
        self.D, self.n, self.need_half = D, n, need_half
        self.rng = np.random.default_rng(seed_of(*seed, scheme))
        self.rsel = np.random.default_rng(seed_of(*seed, "sel"))      # same people for std / cap / gauss / oue
        self.mask = np.zeros(D["n_users"] + 1, bool)
        self.panel = np.empty(0, np.int32)

    def _counts(self, k, sel, cap=None):
        D = self.D
        a, b = D["off"][k], D["off"][k + 1]
        self.mask[sel] = True
        m = self.mask[D["user"][a:b]]
        self.mask[sel] = False
        it = D["item"][a:b][m]
        if cap is not None:
            u = D["user"][a:b][m]
            order = np.lexsort((self.rng.random(len(u)), u))
            us = u[order]
            first = np.r_[True, us[1:] != us[:-1]]
            start = np.maximum.accumulate(np.where(first, np.arange(len(us)), 0))
            keep = np.empty(len(us), bool)
            keep[order] = (np.arange(len(us)) - start) < cap
            it = it[keep]
        return np.bincount(it, minlength=D["n_items"]).astype(np.float32)

    def _pick(self, k, n):
        uk = self.D["users_in"][k]
        if len(uk) <= n:
            return uk
        return (self.rsel if self.kind in ("std", "cap", "gauss", "oue") else self.rng).choice(uk, n, replace=False)

    def round(self, k: int) -> Obs:
        kind, par, D, n = self.kind, self.par, self.D, self.n
        if kind == "std":
            sel = self._pick(k, n)
            half = None
            if self.need_half:
                p = self.rng.permutation(len(sel))
                h = len(sel) // 2
                half = (self._counts(k, sel[p[:h]]), self._counts(k, sel[p[h:]]))
            return Obs(self._counts(k, sel), len(sel), half=half)
        if kind == "varn":
            nw = int(np.clip(n * np.exp(self.rng.normal(0, 0.6)), max(n // 4, 1), 4 * n))
            sel = self._pick(k, nw)
            return Obs(self._counts(k, sel), len(sel), n_w=len(sel))
        if kind == "cap":
            sel = self._pick(k, n)
            return Obs(self._counts(k, sel, cap=int(par["c"])), len(sel))
        if kind == "pps":
            a, b = D["off"][k], D["off"][k + 1]
            uk, rows = np.unique(D["user"][a:b], return_counts=True)
            if len(uk) <= n:
                sel = uk
            else:
                keys = self.rng.random(len(uk)) ** (1.0 / rows)
                sel = uk[np.argpartition(-keys, n - 1)[:n]]
            return Obs(self._counts(k, sel), len(sel))
        if kind == "panel":
            uk = D["users_in"][k]
            keep = np.intersect1d(self.panel, uk, assume_unique=False)
            if len(keep):
                keep = keep[self.rng.random(len(keep)) >= par["rho"]]
            need = n - len(keep)
            if need > 0:
                cand = np.setdiff1d(uk, keep, assume_unique=True)
                add = cand if len(cand) <= need else self.rng.choice(cand, need, replace=False)
                keep = np.union1d(keep, add)
            self.panel = keep.astype(np.int32)
            return Obs(self._counts(k, keep), len(keep))
        if kind == "gauss":
            sel = self._pick(k, n)
            c = int(par["c"])
            cnt = self._counts(k, sel, cap=c)
            sigma = gauss_sigma(par["eps"], float(np.sqrt(c)))
            act = D["first"] <= k                                          # noise only on items already seen in the catalogue
            cnt[act] += self.rng.normal(0.0, sigma, int(act.sum())).astype(np.float32)
            return Obs(cnt, len(sel), noisy=True)
        if kind == "oue":
            sel = self._pick(k, n)
            c = self._counts(k, sel, cap=1)
            e = np.exp(par["eps"])
            p, q = 0.5, 1.0 / (e + 1.0)
            act = D["first"] <= k
            ci = c[act].astype(np.int64)
            obs = self.rng.binomial(ci, p) + self.rng.binomial(len(sel) - ci, q)
            est = np.zeros_like(c)
            est[act] = ((obs - len(sel) * q) / (p - q)).astype(np.float32)
            return Obs(est, len(sel), noisy=True)
        if kind == "ppsprev":                                              # feasible variant: weight = rows in the previous round
            uk = D["users_in"][k]
            if k == 0 or len(uk) <= n:
                sel = uk if len(uk) <= n else self.rng.choice(uk, n, replace=False)
            else:
                a, b = D["off"][k - 1], D["off"][k]
                up, rows = np.unique(D["user"][a:b], return_counts=True)
                pos = np.searchsorted(up, uk)
                pos[pos >= len(up)] = 0
                w = np.where(up[pos] == uk, rows[pos], 1).astype(np.float64)
                keys = self.rng.random(len(uk)) ** (1.0 / w)
                sel = uk[np.argpartition(-keys, n - 1)[:n]]
            return Obs(self._counts(k, sel), len(sel))
        raise ValueError(self.kind)


# ---------------------------------------------------------------- aggregators

class Agg:
    scheme = "std"

    def __init__(self, arm: str, ctx: dict):
        self.arm = arm
        spec, _, self.scheme = arm.partition("@")
        self.scheme = self.scheme or "std"
        self.kind, self.par = parse(spec)
        self.ctx = ctx
        self.n_items = ctx["n_items"]
        self.prev = None
        self.setup()

    def z(self):
        return np.zeros(self.n_items, np.float64)

    def top(self, s: np.ndarray, m: int = 300) -> np.ndarray:
        return top_idx(s, m, self.ctx["tiekey"])

    def setup(self): ...
    def update(self, obs: Obs, w: int): ...
    def score(self) -> np.ndarray: raise NotImplementedError
    def ranked(self) -> np.ndarray: return self.top(self.score())
    def extra(self) -> dict: return {}


class Decay(Agg):
    def setup(self):
        h = self.par.get("h", 2.0)
        if "days" in self.par:
            h = self.par["days"] / self.ctx["window_days"]
        self.d, self.s = decay_factor(h), self.z()

    def update(self, obs, w):
        self.s *= self.d
        self.s += obs.share

    def score(self): return self.s


class Holt(Agg):
    def setup(self):
        self.L, self.T, self.first = self.z(), self.z(), True

    def update(self, obs, w):
        a, b, p = self.par.get("a", 0.3), self.par.get("b", 0.1), self.par.get("p", 0.9)
        x = obs.share
        if self.first:
            self.L, self.first = x.astype(np.float64).copy(), False
            return
        Ln = a * x + (1 - a) * (self.L + p * self.T)
        self.T = b * (Ln - self.L) + (1 - b) * p * self.T
        self.L = Ln

    def score(self): return np.maximum(self.L + self.par.get("p", 0.9) * self.T, 0.0)


class Clip(Decay):
    def setup(self):
        self.d, self.s = decay_factor(2.0), self.z()

    def update(self, obs, w):
        c = np.minimum(obs.cnt, self.par["c"])
        self.s *= self.d
        self.s += norm(c)


class Rrf(Decay):
    def setup(self):
        self.d, self.s = decay_factor(2.0), self.z()

    def update(self, obs, w):
        self.s *= self.d
        pos = np.nonzero(obs.cnt > 0)[0]
        if len(pos):
            v = obs.cnt[pos]
            asc = np.sort(v)
            rank = 1 + (len(asc) - np.searchsorted(asc, v, side="right"))          # tied counts share a rank
            keep = rank <= 1000
            self.s[pos[keep]] += 1.0 / (self.par.get("k", 60.0) + rank[keep])


class TwoSpeed(Agg):
    def setup(self):
        self.sf, self.ss = self.z(), self.z()
        self.df, self.ds = decay_factor(self.par.get("hf", 1.0)), decay_factor(self.par.get("hs", 4.0))
        self.first = np.full(self.n_items, -1, np.int32)
        self.w = 0

    def update(self, obs, w):
        x = obs.share
        self.sf *= self.df; self.sf += x
        self.ss *= self.ds; self.ss += x
        new = (x > 0) & (self.first < 0)
        self.first[new] = w
        self.w = w

    def score(self):
        young = (self.first >= 0) & ((self.w - self.first) < self.par.get("age", 4.0))
        return np.where(young, norm(self.sf), norm(self.ss))


class EB(Decay):
    def setup(self):
        self.d, self.s = decay_factor(2.0), self.z()

    def update(self, obs, w):
        N = float(obs.cnt.sum())
        prior = norm(self.s)
        a = self.par["a"]
        post = (obs.cnt + a * N * prior) / (N * (1 + a)) if N > 0 else obs.cnt
        self.s *= self.d
        self.s += post


class Drift(Decay):
    def setup(self):
        self.d, self.s = decay_factor(2.0), self.z()
        self.tv = []

    def update(self, obs, w):
        x = obs.share
        if self.s.sum() > 0:
            tv = 0.5 * float(np.abs(x - norm(self.s)).sum())
            if len(self.tv) >= 3 and tv > self.par["z"] * float(np.median(self.tv)):
                self.s *= self.par["f"]
            self.tv.append(tv)
        self.s *= self.d
        self.s += x


class Ens(Agg):
    def setup(self):
        self.s = {h: self.z() for h in (1.0, 2.0, 4.0)}

    def update(self, obs, w):
        for h, v in self.s.items():
            v *= decay_factor(h)
            v += obs.share

    def score(self): return sum(norm(v) for v in self.s.values())


class Seas(Agg):
    def setup(self):
        self.s = self.z()
        self.lag = int(round(365.0 / self.ctx["window_days"]))
        self.hist = deque(maxlen=self.lag + 2)
        self.d = decay_factor(2.0)

    def update(self, obs, w):
        self.s *= self.d
        self.s += obs.share
        self.hist.append(obs.share.astype(np.float32))

    def score(self):
        base = norm(self.s)
        if len(self.hist) < self.lag + 2:
            return base
        old = [self.hist[-self.lag - 1], self.hist[-self.lag], self.hist[-self.lag + 1]]   # around the same season one year before the target round
        return base + self.par["g"] * norm(np.mean(old, axis=0).astype(np.float64))


class Hyst(Decay):
    def setup(self):
        self.d, self.s = decay_factor(2.0), self.z()

    def ranked(self):
        sc = self.s
        if self.prev is not None:
            sc = sc.copy()
            sc[self.prev[:100]] *= 1.0 + self.par["d"]
        return self.top(sc)


class Reserve(Agg):
    def setup(self):
        self.s = {1.0: self.z(), 2.0: self.z(), 4.0: self.z()}

    def update(self, obs, w):
        for h, v in self.s.items():
            v *= decay_factor(h)
            v += obs.share

    def score(self): return self.s[2.0]

    def ranked(self):
        r = int(self.par["r"])
        base = self.top(self.s[2.0])
        head = base[: 100 - r]
        rising = norm(self.s[1.0]) - norm(self.s[4.0])
        rising[head] = -np.inf
        rise = self.top(rising, r)
        rest = base[100 - r:]
        return np.concatenate([head, rise, rest[~np.isin(rest, rise)]])[:300]


class Splice(Agg):
    def setup(self):
        self.a, self.b = self.z(), self.z()
        self.da, self.db = decay_factor(self.par["h1"]), decay_factor(self.par["h2"])

    def update(self, obs, w):
        self.a *= self.da; self.a += obs.share
        self.b *= self.db; self.b += obs.share

    def score(self): return self.b

    def ranked(self):
        k1 = int(self.par["k"])
        first = self.top(self.a, k1)
        rest = self.top(self.b, 300)
        return np.concatenate([first, rest[~np.isin(rest, first)]])[:300]


class Ivw(Decay):
    def setup(self):
        self.d, self.s = decay_factor(2.0), self.z()

    def update(self, obs, w):
        wt = (obs.n_w / self.ctx["n"]) ** self.par["g"]
        self.s *= self.d
        self.s += wt * obs.share


class Cens(Decay):
    def setup(self):
        self.d, self.s = decay_factor(2.0), self.z()

    def update(self, obs, w):
        self.s *= self.d
        self.s += obs.share
        if (w + self.ctx["draw"]) % int(self.par["m"]) == 0:
            self.s += self.par["w"] * norm(self.ctx["full"](w).astype(np.float64))


class CensOld(Agg):
    def setup(self):
        self.last = self.z()

    def update(self, obs, w):
        if (w + self.ctx["draw"]) % int(self.par["m"]) == 0:
            self.last = self.ctx["full"](w).astype(np.float64)

    def score(self): return self.last


class FullLag(Agg):
    def setup(self):
        self.hist = deque(maxlen=int(self.par["lag"]) + 1)

    def update(self, obs, w):
        self.hist.append(self.ctx["full"](w))

    def score(self):
        if len(self.hist) < self.hist.maxlen:
            return np.zeros(self.n_items)
        return self.hist[0].astype(np.float64)


class FullDecay(Decay):
    def update(self, obs, w):
        self.s *= self.d
        self.s += norm(self.ctx["full"](w).astype(np.float64))


class Half(Decay):
    def setup(self):
        self.d, self.s = decay_factor(self.par.get("h", 2.0)), self.z()
        self.sa, self.sb = self.z(), self.z()

    def update(self, obs, w):
        self.s *= self.d; self.s += obs.share
        if obs.half is not None:
            self.sa *= self.d; self.sa += norm(obs.half[0])
            self.sb *= self.d; self.sb += norm(obs.half[1])

    def extra(self):
        if self.sa.sum() <= 0 or self.sb.sum() <= 0:
            return {}
        a, b = self.top(self.sa, 100), self.top(self.sb, 100)
        return {"agree": len(np.intersect1d(a, b)) / 100.0}


class SelfSel(Agg):
    def setup(self):
        self.s = [self.z() for _ in HGRID]
        self.d = [decay_factor(h) for h in HGRID]
        self.cum = np.zeros(len(HGRID))
        self.lists = None
        self.steps = 0
        self.choice = 3

    def update(self, obs, w):
        if self.lists is not None:                                  # score last round's lists against this round's sample
            g = obs.cnt
            if g.sum() > 0:
                ideal = np.sort(g[top_idx(g, 100)])[::-1]
                idcg = float(np.sum(ideal * DISC[:len(ideal)]))
                for i, l in enumerate(self.lists):
                    self.cum[i] += float(np.sum(g[l] * DISC[:len(l)])) / idcg if idcg > 0 else 0.0
                self.steps += 1
                self.choice = int(np.argmax(self.cum)) if self.steps >= 3 else 3
        for i in range(len(HGRID)):
            self.s[i] *= self.d[i]
            self.s[i] += obs.share
        self.lists = [self.top(v, 100) for v in self.s]

    def score(self): return self.s[self.choice]

    def extra(self): return {"chosen_h_idx": float(self.choice)}


class AThr(Decay):
    def setup(self):
        self.d, self.s, self.c = decay_factor(2.0), self.z(), self.z()

    def update(self, obs, w):
        self.s *= self.d; self.s += obs.share
        self.c *= self.d; self.c += obs.cnt

    def ranked(self):
        sc = np.where(self.c >= self.par["t"], self.s, 0.0)
        top = self.top(sc)
        return top[sc[top] > 0]


KINDS = {"decay": Decay, "holt": Holt, "clip": Clip, "rrf": Rrf, "twospeed": TwoSpeed, "eb": EB, "drift": Drift, "ens": Ens,
         "seas": Seas, "hyst": Hyst, "reserve": Reserve, "splice": Splice, "ivw": Ivw, "cens": Cens, "censold": CensOld,
         "full": FullLag, "fulldecay": FullDecay, "half": Half, "selfsel": SelfSel, "athr": AThr}


def split_arm(arm: str):
    spec, _, scheme = arm.partition("@")
    return spec, (scheme or "std")


# ---------------------------------------------------------------- one cell

def run_cell(task: dict) -> dict:
    nw = int(task.get("n_win", 30))
    D = T.prepare(task["dataset"], nw)
    n = int(task["n"])
    arms = list(task["arms"])
    draws = int(task.get("draws", 12))
    tag = task.get("seed_tag", "")
    last = nw - 2
    steps = list(range(FIRST_STEP, last + 1))
    S = len(steps)
    schemes = sorted({split_arm(a)[1] for a in arms})
    need_half = any(split_arm(a)[0].startswith("half") for a in arms)
    mnames = ["nd10", "nd30", "nd300", "rec100", "churn", "len100", "prec100", "agree", "chosen_h_idx"]
    nd100 = np.full((draws, len(arms), S), np.nan)
    selfnd = np.full((draws, len(arms), S), np.nan)
    mm = {m: np.full((draws, len(arms), S), np.nan) for m in mnames}
    people = np.zeros((draws, len(arms)))
    rows_used = np.zeros((draws, len(arms)))
    full = lambda k: D["counts"][k]
    for dr in range(draws):
        ctx = dict(n_items=D["n_items"], window_days=D["window_days"], n=n, n_win=nw, draw=dr, full=full,
                   tiekey=np.random.default_rng(seed_of(task["dataset"], n, dr, nw, tag, "tie")).permutation(D["n_items"]).astype(np.int32))
        samplers = {s: Sampler(s, D, n, (task["dataset"], n, dr, nw, tag), need_half and s == "std") for s in schemes}
        aggs = [KINDS[parse(split_arm(a)[0])[0]](a, ctx) for a in arms]
        for w in range(0, last + 1):
            obs = {s: sm.round(w) for s, sm in samplers.items()}
            for ai, ag in enumerate(aggs):
                o = obs[ag.scheme]
                if ag.prev is not None and w - 1 >= FIRST_STEP and o.cnt.sum() > 0:       # self-score of last round's list vs this sample
                    g = o.cnt
                    if o.idcg is None:
                        ideal = np.sort(g[top_idx(g, 100)])[::-1]
                        o.idcg = float(np.sum(ideal * DISC[:len(ideal)]))
                    idcg = o.idcg
                    if idcg > 0:
                        selfnd[dr, ai, w - 1 - FIRST_STEP] = float(np.sum(g[ag.prev[:100]] * DISC[:min(100, len(ag.prev))])) / idcg
                ag.update(o, w)
                people[dr, ai] += o.people
                rows_used[dr, ai] += float(o.cnt.sum()) if not o.noisy else 0.0
            if w < FIRST_STEP:
                continue
            j = w - FIRST_STEP
            gain = D["counts"][w + 1]
            if gain.sum() < T.MIN_TRUTH_ROWS:
                for ag in aggs:
                    ag.prev = ag.ranked()
                continue
            ti = top_idx(gain, 300)
            ig = gain[ti].astype(np.float64)
            idcg = {k: float(np.sum(ig[:k] * DISC[:k])) for k in KS}
            t100 = ti[:100]
            for ai, ag in enumerate(aggs):
                lst = ag.ranked()
                g = gain[lst].astype(np.float64)
                for k, name in ((100, None), (10, "nd10"), (30, "nd30"), (300, "nd300")):
                    v = float(np.sum(g[:k] * DISC[:len(g[:k])])) / idcg[k] if idcg[k] > 0 else 0.0
                    if name is None:
                        nd100[dr, ai, j] = v
                    else:
                        mm[name][dr, ai, j] = v
                top100 = lst[:100]
                mm["rec100"][dr, ai, j] = len(np.intersect1d(top100, t100)) / 100.0
                mm["len100"][dr, ai, j] = len(top100)
                mm["prec100"][dr, ai, j] = len(np.intersect1d(top100, t100)) / max(len(top100), 1)
                if ag.prev is not None:
                    mm["churn"][dr, ai, j] = 1.0 - len(np.intersect1d(top100, ag.prev[:100])) / 100.0
                for k2, v2 in ag.extra().items():
                    mm[k2][dr, ai, j] = v2
                ag.prev = lst
    steady = [i for i, s in enumerate(steps) if s >= STEADY]
    sm = lambda a: [[None if not np.isfinite(x) else round(float(x), 5) for x in row] for row in np.nanmean(a[:, :, steady], axis=2)] \
        if np.isfinite(a[:, :, steady]).any() else None
    with np.errstate(all="ignore"):
        metrics = {m: sm(a) for m, a in mm.items()}                    # [draw][arm] steady means
    r5 = lambda a: np.round(a, 5).tolist()
    return dict(kind="deep", dataset=task["dataset"], n=n, n_win=nw, arms=arms, steps=steps, window_days=D["window_days"],
               active=D["active"].tolist(), nd100=r5(nd100), selfnd=r5(selfnd), metrics=metrics,
               people_per_round=[round(float(x), 2) for x in people.mean(axis=0) / (last + 1)],
               rows_per_round=[round(float(x), 1) for x in rows_used.mean(axis=0) / (last + 1)])


def run_task(task: dict) -> dict:
    return run_cell(task)
