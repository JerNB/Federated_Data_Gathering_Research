"""Selector confirmation (stage K): fresh time windows of the weekly-study logs, plus the two short check-in logs.

The weekly study cut every log to its densest 156 consecutive weeks (weekly_core.densest_start). This module cuts a DIFFERENT span:

  w2     the densest 156 consecutive weeks that do not overlap the weekly-study span (the two free sides of the log are searched,
         the side holding more positive events wins; weeks are counted from the log's first positive event, as in weekly_core);
  whole  logs shorter than 156 weeks (gowalla, brightkite): every complete week between the first and the last positive event.

prepare_window builds the same data dict as weekly_core.prepare_weekly (items re-numbered to those present in the span) for a given
number of weeks. run_task seeds track_core's cache under the key "<log>@w2" / "<log>" and calls deep_core.run_cell, so every aggregator
and sampler of the deep engine works unchanged.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import deep_core as C
import track_core as T
import weekly_core as W

WEEK = W.WEEK
N_WEEKS = W.N_WEEKS


def _positive_events(name: str):
    z = np.load(f"{T.NPZ}/{name}.npz")
    pos = z["r"] >= float(z["positive_threshold"])
    user, item, ts = z["user"][pos].astype(np.int32), z["item"][pos].astype(np.int32), z["ts"][pos].astype(np.int64)
    if ts.max() > 10 ** 11:
        ts = ts // 1000
    return user, item, ts


def window_of(ts: np.ndarray):
    """(kind, start_week, n_weeks) of the confirmation window, weeks counted from the log's first positive event."""
    lo = int(ts.min())
    st1 = W.densest_start(ts, lo)
    if st1 is None:
        return "whole", 0, int((int(ts.max()) - lo) // WEEK)
    c = np.bincount(((ts - lo) // WEEK).astype(np.int64))
    cs = np.r_[0, np.cumsum(c)]
    best = None
    for a, b in ((0, st1), (st1 + N_WEEKS, len(c))):          # earlier side first, so a tie goes to the earlier side
        if b - a >= N_WEEKS:
            sums = cs[a + N_WEEKS: b + 1] - cs[a: b - N_WEEKS + 1]
            j = int(np.argmax(sums))
            if best is None or int(sums[j]) > best[1]:
                best = (a + j, int(sums[j]))
    if best is None:
        return None
    return "w2", best[0], N_WEEKS


def load_window(name: str):
    """(user, item, ts re-based to the window start, kind, n_weeks) or None when the log has no free 156-week span."""
    user, item, ts = _positive_events(name)
    got = window_of(ts)
    if got is None:
        return None
    kind, st, n_weeks = got
    t0 = int(ts.min()) + st * WEEK
    m = (ts >= t0) & (ts < t0 + n_weeks * WEEK)
    return user[m], item[m], ts[m] - t0, kind, n_weeks


def dataset_key(name: str, kind: str) -> str:
    return f"{name}@w2" if kind == "w2" else name


def prepare_window(name: str, round_days: int) -> dict:
    """Seed track_core's cache with the window of `name` cut into n_weeks*7/round_days equal rounds; returns the data dict."""
    got = load_window(name)
    if got is None:
        raise ValueError(f"{name}: no confirmation window")
    user, item, ts, kind, n_weeks = got
    key = dataset_key(name, kind)
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
        prepare_window(task["log"], task["round_days"])
    if T._CACHE["key"] != want:
        raise RuntimeError(f"cache key {T._CACHE['key']} does not match the task {want}")
    r = C.run_cell(task)
    r["log"] = task["log"]
    r["round_days"] = task["round_days"]
    r["weekly_active"] = [int(x) for x in T._CACHE["data"]["active"]]
    return r
