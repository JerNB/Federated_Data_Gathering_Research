"""Hypothesis tournament core: many candidate "list-building rules" scored on the same random reporters.

Same setting as track_core.py: a log is cut into n_win equal-time windows (rounds); in every round n random active
people report their rows; the server builds a top-100 list from the reports so far and the list is scored (NDCG@100)
against the WHOLE community's counts in the next round. Several rules see exactly the same people in the same draw,
so differences between rules are paired.

A rule is named "type:param=value". Types:
  decay:h=H        decayed sum of per-round shares, half-life H rounds (H = inf: plain sum). Baseline is decay:h=2.
  slide:K=K        plain sum of the last K rounds' shares.
  mom:beta=B       decay h=2 share + B * (recent share (h=1) - older share (h=4)): extra weight on rising items.
  blend            half recent (h=1) share + half long-run (h=inf) share.
  thresh:t=T       items with fewer than T reported rows in a round are dropped before sharing (decay h=2).
  cap:c=C          each reporter contributes at most C rows per round (C rows chosen at random), decay h=2.
  cad:m=M          budget moved to every M-th round: 0 reporters, then the people the baseline would have asked over those
                   M rounds (unspent budget carried to the next asking round) (decay h=2).
  athr:t=T         decay h=2 list, but only items with a decayed report count >= T may appear (threshold on the running count).
  pwr:p=P          decayed sum of per-round shares raised to the power P (P<1 damps heavy hitters).
"""
from __future__ import annotations

import os
import sys
import zlib
from collections import deque

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import metrics as M
import track_core as T

BASE = "decay:h=2"
FIRST_STEP = T.FIRST_STEP


def seed_of(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def parse(pid: str):
    kind, _, rest = pid.partition(":")
    par = {}
    for kv in filter(None, rest.split(",")):
        k, v = kv.split("=")
        par[k] = float(v)
    return kind, par


def scheme_of(pid: str) -> str:
    kind, par = parse(pid)
    if kind == "thresh":
        return f"thr{int(par['t'])}"
    if kind == "cap":
        return f"cap{int(par['c'])}"
    if kind == "cad":
        return f"cad{int(par['m'])}"
    return "std"


def _norm(x: np.ndarray) -> np.ndarray:
    s = float(x.sum())
    return x / s if s > 0 else x


class Policy:
    def __init__(self, pid: str, n_items: int):
        self.pid = pid
        self.kind, self.par = parse(pid)
        z = lambda: np.zeros(n_items, np.float64)
        if self.kind in ("decay", "thresh", "cap", "cad"):
            h = self.par.get("h", 2.0)
            self.d = T._decay(h)
            self.s = z()
        elif self.kind == "slide":
            self.q = deque(maxlen=int(self.par["K"]))
        elif self.kind == "mom":
            self.a, self.b, self.c = z(), z(), z()
        elif self.kind == "blend":
            self.a, self.b = z(), z()
        elif self.kind == "athr":
            self.d = T._decay(2.0)
            self.s, self.c = z(), z()
        elif self.kind == "pwr":
            self.d = T._decay(2.0)
            self.s = z()
        else:
            raise ValueError(pid)

    def update(self, share: np.ndarray, raw: np.ndarray | None = None):
        k = self.kind
        if k in ("decay", "thresh", "cap", "cad"):
            self.s *= self.d
            self.s += share
        elif k == "athr":
            self.s *= self.d
            self.s += share
            self.c *= self.d
            self.c += raw
        elif k == "pwr":
            self.s *= self.d
            self.s += _norm(np.power(share, self.par["p"]))
        elif k == "slide":
            self.q.append(share)
        elif k == "mom":
            self.a *= T._decay(1.0); self.a += share
            self.b *= T._decay(4.0); self.b += share
            self.c *= T._decay(2.0); self.c += share
        else:
            self.a *= T._decay(1.0); self.a += share
            self.b += share

    def score(self) -> np.ndarray:
        k = self.kind
        if k in ("decay", "thresh", "cap", "cad", "pwr"):
            return self.s
        if k == "athr":
            return np.where(self.c >= self.par["t"], self.s, 0.0)
        if k == "slide":
            return np.sum(self.q, axis=0)
        if k == "mom":
            return _norm(self.c) + self.par["beta"] * (_norm(self.a) - _norm(self.b))
        return _norm(self.a) + _norm(self.b)


def _counts(D: dict, k: int, sel: np.ndarray, mask: np.ndarray, cap: int | None = None, thr: int | None = None,
            rng_c: np.random.Generator | None = None) -> np.ndarray:
    a, b = D["off"][k], D["off"][k + 1]
    mask[sel] = True
    m = mask[D["user"][a:b]]
    mask[sel] = False
    it = D["item"][a:b][m]
    if cap is not None:
        u = D["user"][a:b][m]
        order = np.lexsort((rng_c.random(len(u)), u))   # by person, random order inside a person
        us = u[order]
        first = np.r_[True, us[1:] != us[:-1]]
        start = np.maximum.accumulate(np.where(first, np.arange(len(us)), 0))
        keep = np.empty(len(us), bool)
        keep[order] = (np.arange(len(us)) - start) < cap
        it = it[keep]
    c = np.bincount(it, minlength=D["n_items"]).astype(np.float32)
    if thr is not None:
        c[c < thr] = 0.0
    return c


def _select(D: dict, k: int, n: int, rng: np.random.Generator) -> np.ndarray:
    uk = D["users_in"][k]
    return uk if len(uk) <= n else rng.choice(uk, n, replace=False)


def run_cell(task: dict) -> dict:
    nw = int(task.get("n_win", 30))
    D = T.prepare(task["dataset"], nw)
    n = int(task["n"])
    pids = list(task["policies"])
    draws = int(task.get("draws", 12))
    mask = np.zeros(D["n_users"] + 1, bool)
    last_step = nw - 2
    steps = list(range(FIRST_STEP, last_step + 1))
    schemes = sorted({scheme_of(p) for p in pids})
    out = np.full((draws, len(pids), len(steps)), np.nan)
    rows_used = {s: 0.0 for s in schemes}
    people_used = {s: 0.0 for s in schemes}
    tag = task.get("seed_tag", "")                 # screen uses its own draws so full-stage draws are not reused
    base_n = np.minimum(D["active"], n)           # people the baseline realises per round
    for dr in range(draws):
        rng = np.random.default_rng(seed_of(task["dataset"], n, dr, nw, tag))
        rng_cad = {s: np.random.default_rng(seed_of(task["dataset"], n, dr, nw, s, tag)) for s in schemes if s.startswith(("cad", "cap"))}
        pol = [Policy(p, D["n_items"]) for p in pids]
        carry = {s: 0 for s in schemes}
        for w in range(0, last_step + 1):
            sel = _select(D, w, n, rng)
            cnt = {}
            for s in schemes:
                if s == "std":
                    cnt[s] = _counts(D, w, sel, mask)
                elif s.startswith("thr"):
                    cnt[s] = _counts(D, w, sel, mask, thr=int(s[3:]))
                elif s.startswith("cap"):
                    cnt[s] = _counts(D, w, sel, mask, cap=int(s[3:]), rng_c=rng_cad[s])
                else:
                    m_ = int(s[3:])
                    if (w + dr) % m_ == 0:
                        budget = int(base_n[max(0, w - m_ + 1): w + 1].sum()) + carry[s]   # baseline's realised people + unspent budget
                        sel_c = _select(D, w, budget, rng_cad[s])
                        carry[s] = budget - len(sel_c)
                        people_used[s] += len(sel_c)
                        cnt[s] = _counts(D, w, sel_c, mask)
                    else:
                        cnt[s] = np.zeros(D["n_items"], np.float32)
                if s != "cad" and not s.startswith("cad"):
                    people_used[s] += len(sel)
                rows_used[s] += float(cnt[s].sum())
            shr = {s: T._shares(c) for s, c in cnt.items()}
            for p in pol:
                p.update(shr[scheme_of(p.pid)], cnt[scheme_of(p.pid)])
            if w < FIRST_STEP:
                continue
            truth = D["counts"][w + 1]
            if truth.sum() < T.MIN_TRUTH_ROWS:
                continue
            j = w - FIRST_STEP
            for i, p in enumerate(pol):
                top = M.top_m(p.score(), 100, np.random.default_rng(seed_of(task["dataset"], n, dr, nw, "tie", w)))
                out[dr, i, j] = M.ndcg_vs_gain(top, truth, 100)
    per_round = draws * (last_step + 1)
    return dict(kind="tn", dataset=task["dataset"], n=n, n_win=nw, policies=pids, steps=steps,
                ndcg=np.round(out, 6).tolist(), rows_per_round={s: round(v / per_round, 2) for s, v in rows_used.items()},
                people_per_round={s: round(v / per_round, 2) for s, v in people_used.items()},
                window_days=D["window_days"], active=D["active"].tolist())


def run_task(task: dict) -> dict:
    return run_cell(task)
