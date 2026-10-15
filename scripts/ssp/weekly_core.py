"""Weekly-cadence study: short rounds (7, 14, 28 days) cut from the densest 156 weeks of a log.

prepare_weekly builds the same data dict as track_core.prepare, but over a chosen calendar span and with items
re-numbered to those present in the span (keeps memory small at 156 rounds). run_task seeds track_core's cache
and calls deep_core.run_cell, so every aggregator and sampler of the deep engine works unchanged.
cadence() compares many small rounds with fewer big rounds at the same number of people per 28 days.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import deep_core as C
import track_core as T

N_WEEKS = 156
WEEK = 7 * 86400


def densest_start(ts: np.ndarray, lo: int, n_weeks: int = N_WEEKS):
    """Start (in weeks after lo) of the n_weeks span with the most events; None if the log is shorter."""
    wk = (ts - lo) // WEEK
    c = np.bincount(wk.astype(np.int64))
    if len(c) < n_weeks:
        return None
    cs = np.r_[0, np.cumsum(c)]
    sums = cs[n_weeks:] - cs[:-n_weeks]
    return int(np.argmax(sums))


def load_span(name: str):
    z = np.load(f"{T.NPZ}/{name}.npz")
    pos = z["r"] >= float(z["positive_threshold"])
    user, item, ts = z["user"][pos].astype(np.int32), z["item"][pos].astype(np.int32), z["ts"][pos].astype(np.int64)
    if ts.max() > 10 ** 11:
        ts = ts // 1000
    lo = int(ts.min())
    st = densest_start(ts, lo)
    if st is None:
        return None
    t0 = lo + st * WEEK
    m = (ts >= t0) & (ts < t0 + N_WEEKS * WEEK)
    return user[m], item[m], ts[m] - t0


def prepare_weekly(name: str, round_days: int, spans: dict | None = None) -> dict:
    nw = N_WEEKS * 7 // round_days
    if T._CACHE.get("key") == (name, nw):
        return T._CACHE["data"]
    T._CACHE.clear()
    got = (spans or {}).get(name) or load_span(name)
    user, item, ts = got
    uniq, item = np.unique(item, return_inverse=True)
    item = item.astype(np.int32)
    n_items = len(uniq)
    win = np.minimum(ts // (round_days * 86400), nw - 1).astype(np.int16)
    order = np.argsort(win, kind="stable")
    user, item, win = user[order], item[order], win[order]
    off = np.searchsorted(win, np.arange(nw + 1))
    users_in, active, counts, rows = [], [], [], []
    first = np.full(n_items, nw, np.int16)
    for k in range(nw):
        a, b = off[k], off[k + 1]
        u = np.unique(user[a:b])
        users_in.append(u)
        active.append(len(u))
        rows.append(int(b - a))
        counts.append(np.bincount(item[a:b], minlength=n_items).astype(np.float32))
        seen = counts[-1] > 0
        first[seen & (first > k)] = k
    data = dict(name=name, user=user, item=item, off=off, users_in=users_in, active=np.array(active), rows=np.array(rows),
                counts=counts, first=first, n_items=n_items, n_users=int(user.max()) + 1, window_days=float(round_days), n_win=nw)
    T._CACHE["key"], T._CACHE["data"] = (name, nw), data
    return data


def run_task(task: dict) -> dict:
    if task.get("kind") == "cadence":
        return cadence(task)
    prepare_weekly(task["dataset"], task["round_days"])
    r = C.run_cell(task)
    r["round_days"] = task["round_days"]
    r["weekly_active"] = [int(x) for x in T._CACHE["data"]["active"]]
    return r


def _ndcg(lst: np.ndarray, gain: np.ndarray, ti: np.ndarray, idcg: float) -> float:
    g = gain[lst[:100]].astype(np.float64)
    return float(np.sum(g * C.DISC[:len(g)])) / idcg if idcg > 0 else 0.0


def cadence(task: dict) -> dict:
    """Weekly rounds of n people vs one 28-day round of 4n people; both lists scored on every following week."""
    D = prepare_weekly(task["dataset"], 7)
    n, draws, nw = int(task["n"]), int(task["draws"]), D["n_win"]
    days_grid = list(task["days"])
    n_items = D["n_items"]
    out = {f"{kind}|{d}": [] for kind in ("week", "month") for d in days_grid}
    for dr in range(draws):
        rng = np.random.default_rng(T.seed_of(task["dataset"], n, dr, "cadence"))
        mask = np.zeros(D["n_users"] + 1, bool)

        def counts(k0, k1, sel):
            a, b = D["off"][k0], D["off"][k1]
            mask[sel] = True
            m = mask[D["user"][a:b]]
            mask[sel] = False
            return np.bincount(D["item"][a:b][m], minlength=n_items).astype(np.float64)

        wk_state = {d: np.zeros(n_items) for d in days_grid}
        mo_state = {d: np.zeros(n_items) for d in days_grid}
        mo_list = {d: None for d in days_grid}
        tie = rng.permutation(n_items).astype(np.int32)
        series = {k: [] for k in out}
        for w in range(0, nw - 1):
            uk = D["users_in"][w]
            sel = uk if len(uk) <= n else rng.choice(uk, n, replace=False)
            cw = counts(w, w + 1, sel)
            sw = cw / cw.sum() if cw.sum() > 0 else cw
            for d in days_grid:
                wk_state[d] = wk_state[d] * (2.0 ** (-7.0 / d)) + sw
            if w % 4 == 3 and w >= 3:                                   # a 28-day round ends here
                uu = np.unique(np.concatenate(D["users_in"][w - 3:w + 1]))
                sel = uu if len(uu) <= 4 * n else rng.choice(uu, 4 * n, replace=False)
                cm = counts(w - 3, w + 1, sel)
                sm = cm / cm.sum() if cm.sum() > 0 else cm
                for d in days_grid:
                    mo_state[d] = mo_state[d] * (2.0 ** (-28.0 / d)) + sm
                    mo_list[d] = C.top_idx(mo_state[d], 300, key=tie)[:100]
            gain = D["counts"][w + 1]
            if w < 8 or gain.sum() < T.MIN_TRUTH_ROWS or any(v is None for v in mo_list.values()):
                continue
            ti = C.top_idx(gain, 300)
            ig = gain[ti[:100]].astype(np.float64)
            idcg = float(np.sum(ig * C.DISC[:len(ig)]))
            for d in days_grid:
                series[f"week|{d}"].append(_ndcg(C.top_idx(wk_state[d], 300, key=tie)[:100], gain, ti, idcg))
                series[f"month|{d}"].append(_ndcg(mo_list[d], gain, ti, idcg))
        for k in out:
            out[k].append(float(np.mean(series[k])) if series[k] else float("nan"))
    return dict(kind="cadence", dataset=task["dataset"], n=n, days=days_grid, mean={k: out[k] for k in out})
