"""Third window (w3) for programme G: the densest 156 consecutive weeks of a log that overlap neither window 1 (weekly study, weekly_core.densest_start)
nor window 2 (confirmation, confirm_core.window_of). Weeks are counted from the log's first positive event, as in the earlier windows.

load_window3 / prepare_window3 / run_task mirror confirm_core (dataset key '<log>@w3'); items are re-numbered to those present in the span.
confirm_core.py is hash-frozen for the confirmation, so this module copies its small data-preparation loop instead of editing it.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import confirm_core as K
import deep_core as C
import track_core as T
import weekly_core as W

WEEK, N_WEEKS = W.WEEK, W.N_WEEKS


def window3_of(ts: np.ndarray):
    """(start week, 156) of the densest span free of windows 1 and 2, or None when the log has none."""
    lo = int(ts.min())
    st1 = W.densest_start(ts, lo)
    got2 = K.window_of(ts)
    if st1 is None or got2 is None or got2[0] != "w2":
        return None
    st2 = got2[1]
    c = np.bincount(((ts - lo) // WEEK).astype(np.int64))
    cs = np.r_[0, np.cumsum(c)]
    if len(c) < N_WEEKS:
        return None
    sums = (cs[N_WEEKS:] - cs[:-N_WEEKS]).astype(np.int64)
    starts = np.arange(len(sums))
    free = np.ones(len(sums), bool)
    for x in (st1, st2):
        free &= (starts + N_WEEKS <= x) | (starts >= x + N_WEEKS)
    if not free.any():
        return None
    j = int(np.argmax(np.where(free, sums, -1)))
    return j, N_WEEKS


def load_window3(name: str):
    user, item, ts = K._positive_events(name)
    got = window3_of(ts)
    if got is None:
        return None
    st, nw = got
    t0 = int(ts.min()) + st * WEEK
    m = (ts >= t0) & (ts < t0 + nw * WEEK)
    return user[m], item[m], ts[m] - t0, nw, st


def prepare_window3(name: str, round_days: int) -> dict:
    got = load_window3(name)
    if got is None:
        raise ValueError(f"{name}: no third window")
    user, item, ts, n_weeks, _ = got
    key = f"{name}@w3"
    nw = n_weeks * 7 // round_days
    if T._CACHE.get("key") == (key, nw):
        return T._CACHE["data"]
    T._CACHE.clear()
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
    data = dict(name=key, user=user, item=item, off=off, users_in=users_in, active=np.array(active), rows=np.array(rows),
                counts=counts, first=first, n_items=n_items, n_users=int(user.max()) + 1, window_days=float(round_days), n_win=nw)
    T._CACHE["key"], T._CACHE["data"] = (key, nw), data
    return data


def run_task(task: dict) -> dict:
    want = (task["dataset"], int(task["n_win"]))
    if T._CACHE.get("key") != want:
        prepare_window3(task["log"], task["round_days"])
    if T._CACHE["key"] != want:
        raise RuntimeError(f"cache key {T._CACHE['key']} does not match the task {want}")
    r = C.run_cell(task)
    r["log"], r["round_days"] = task["log"], task["round_days"]
    r["weekly_active"] = [int(x) for x in T._CACHE["data"]["active"]]
    return r
