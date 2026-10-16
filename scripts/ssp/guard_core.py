"""Guarded selection rules (programme G): offline rules over stored cells, same data layout as explore_selfsel_weekly.

A cell view is a dict with arms (decay:days=D names), nd [draw][arm][step] (true NDCG), se [draw][arm][step] (self-score), st (steady steps),
val (mean nd100 per arm over steady steps and draws). The rules choose among the decay:days arms at every step t from self-scores of steps < t
(explore_selfsel_weekly.history_scores semantics), so nothing from the future enters. Regret = best fixed arm of the cell minus the rule's value.

  plain      the selector of docs 14-15 (EWMA half-life 26, fallback arm nearest to 2 rounds)
  margin     keep a PRIOR arm unless another arm's EWMA self-score beats the prior arm's by more than delta (hysteresis around a prior)
  fixedshare Hedge weights with fixed share: w <- w * exp(eta * selfscore); w <- (1-alpha) * w + alpha / K; publish the arm of largest weight
             (Herbster and Warmuth 1998); weights start uniform
  warm       the plain EWMA selector seeded with m pseudo-steps of the HISTORY's mean self-score per arm (their weight decays like the EWMA):
             the control that gives the selector the same evidence as the fixed 'ownself' arm
  prior      a prior arm alone (no evidence used): the table arm or the log's own best arm
The priors are derived by the caller from OTHER windows or OTHER logs only.
"""
from __future__ import annotations

import numpy as np

import explore_selfsel_weekly as E


def choose_margin(se: np.ndarray, steps, hl: float, prior: int, delta: float) -> dict:
    """{t: arm index per draw}; the prior arm is kept unless another arm's EWMA self-score exceeds it by more than delta (and always while no score exists)."""
    out = {}
    for t in steps:
        if t <= 0:
            continue
        s = E.history_scores(se, t, hl)
        best = np.argmax(s, axis=1)
        d = np.arange(s.shape[0])
        gap = s[d, best] - s[:, prior]
        ok = np.isfinite(s).any(axis=1) & np.isfinite(gap) & (gap > delta)
        out[t] = np.where(ok, best, prior)
    return out


def choose_fixedshare(se: np.ndarray, steps, eta: float, alpha: float, fallback: int) -> dict:
    """{t: arm index per draw}; weights are updated with the newest self-score available when step t is published (step t-1); a missing score leaves them alone."""
    D, K, S = se.shape
    w = np.full((D, K), 1.0 / K)
    want = set(int(x) for x in steps)
    out = {}
    for t in range(1, S):
        g = se[:, :, t - 1]
        ok = np.isfinite(g).all(axis=1)
        if ok.any():
            w[ok] = w[ok] * np.exp(eta * (g[ok] - g[ok].max(axis=1, keepdims=True)))
            w[ok] = w[ok] / w[ok].sum(axis=1, keepdims=True)
            w[ok] = (1.0 - alpha) * w[ok] + alpha / K
        if t in want:
            seen = w.max(axis=1) > w.min(axis=1) + 1e-12
            out[t] = np.where(seen, np.argmax(w, axis=1), fallback)
    return out


def choose_warm(se: np.ndarray, steps, hl: float, prior_scores: np.ndarray, m: float) -> dict:
    """{t: arm index per draw}; EWMA self-score per arm with m pseudo-steps of `prior_scores` (one value per arm) before step 0, decaying like the EWMA."""
    out = {}
    for t in steps:
        if t <= 0:
            continue
        past = se[:, :, :t]
        ok = np.isfinite(past)
        w = 0.5 ** ((t - 1 - np.arange(t)) / hl)
        num = np.sum(np.where(ok, past, 0.0) * w, axis=2)
        den = np.sum(ok * w, axis=2)
        w0 = m * 0.5 ** (t / hl)
        out[t] = np.argmax((num + w0 * prior_scores[None, :]) / (den + w0), axis=1)
    return out


def value_of(nd: np.ndarray, choice: dict, steps) -> float:
    return E.chosen_value(nd, choice, steps)


def days_vec(c, what: str) -> dict:
    """{days: value} over the decay:days arms of a stored cell: 'val' = mean true NDCG, 'se' = mean self-score (steady steps, draws)."""
    idx, ds = E.arms_of(c)
    if what == "val":
        return {int(d): float(c["val"][f"decay:days={int(d)}"]) for d in ds}
    m = np.nanmean(c["se"][:, idx, :][:, :, c["st"]], axis=(0, 2))
    return {int(d): float(v) for d, v in zip(ds, m) if np.isfinite(v)}


def hist_scores(cells: list, ds: np.ndarray):
    """Mean self-score per target arm (days `ds`) over the history cells; an arm missing in a cell takes the value of the nearest half-life (log2).
    Returns None when no history cell has a finite self-score."""
    vecs = [days_vec(c, "se") for c in cells]
    vecs = [v for v in vecs if v]
    if not vecs:
        return None
    out = np.zeros(len(ds))
    for j, d in enumerate(ds):
        vals = []
        for v in vecs:
            keys = np.array(sorted(v))
            vals.append(v[int(keys[np.argmin(np.abs(np.log2(keys) - np.log2(max(float(d), 1.0))))])])
        out[j] = float(np.mean(vals))
    return out
