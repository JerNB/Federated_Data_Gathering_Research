"""Tracking study core: n random people report per window; the server keeps a decayed sum of their window shares.

Data: datasets/npz/<name>.npz (first (user,item) event, positive rows only), cut into 30 equal-time windows.
Counting unit: raw event rows inside a window, for the tracker and for the truth alike (every kept row of a
sampled person counts, k = all within the window).

Per (dataset, n) task, per draw (12 draws): for every window an independent random set of n active people (all
active people if fewer than n) reports its rows; share_w = counts / counts.sum(). For every half-life h in
H = 0.25, 0.5, 1, 2, 4, 8, inf (windows): s_h <- s_h * 2^(-1/h) + share_w. At step w = 8..28 the list top-100(s_h)
is the prediction for window w+1 and is scored against the WHOLE community's counts in window w+1 (NDCG@100).
Self-scores for the server (no future truth): share error ||s_h/|s_h| - share_{w+1}||^2 against the SAMPLED next
window, and NDCG@100 against the sampled next window's counts.

Per dataset ("full" task): full-community lists (lag 0..8 windows; decayed trackers with the same H), the
information ceiling M0 (a list built from half of window w+1's people vs lists scored on the other half), the
drift profile M1 (noise-corrected squared share change by the product of two disjoint person groups; total
variation with a same-window floor), new-item shares of the next head, active people per window.
"""
from __future__ import annotations

import os
import sys
import zlib

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import metrics as M

NPZ = "datasets/npz"
N_WIN = 30
FIRST_STEP, LAST_STEP = 8, 28          # step w predicts window w+1
H = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, float("inf")]
LAGS = list(range(0, 9))
N_GRID = [10, 30, 100, 300, 1000, 3000, 10000]
DRAWS = 12
MIN_TRUTH_ROWS = 500
_CACHE: dict = {}


def seed_of(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def prepare(name: str) -> dict:
    """Load a dataset once per process: rows sorted by window, per-window counts and active people."""
    if _CACHE.get("name") == name:
        return _CACHE["data"]
    _CACHE.clear()
    z = np.load(f"{NPZ}/{name}.npz")
    pos = z["r"] >= float(z["positive_threshold"])
    user = z["user"][pos].astype(np.int32)
    item = z["item"][pos].astype(np.int32)
    ts = z["ts"][pos].astype(np.int64)
    n_items = int(z["item"].max()) + 1
    lo, hi = int(ts.min()), int(ts.max())
    width = (hi - lo) / N_WIN
    win = np.minimum(((ts - lo) // width).astype(np.int16), N_WIN - 1)
    order = np.argsort(win, kind="stable")
    user, item, win = user[order], item[order], win[order]
    off = np.searchsorted(win, np.arange(N_WIN + 1))
    users_in, active, counts, rows = [], [], [], []
    first = np.full(n_items, N_WIN, np.int16)
    for k in range(N_WIN):
        a, b = off[k], off[k + 1]
        u = np.unique(user[a:b])
        users_in.append(u)
        active.append(len(u))
        rows.append(int(b - a))
        counts.append(np.bincount(item[a:b], minlength=n_items).astype(np.float32))
        seen = counts[-1] > 0
        first[seen & (first > k)] = k
    data = dict(name=name, user=user, item=item, off=off, users_in=users_in, active=np.array(active),
                rows=np.array(rows), counts=counts, first=first, n_items=n_items,
                n_users=int(user.max()) + 1, window_days=width / 86400.0)
    _CACHE["name"], _CACHE["data"] = name, data
    return data


def _shares(c: np.ndarray) -> np.ndarray:
    s = float(c.sum())
    return c / s if s > 0 else c


def _decay(h: float) -> float:
    return 1.0 if np.isinf(h) else float(2.0 ** (-1.0 / h))


def sample_window(D: dict, k: int, n: int, rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    """Counts of the rows of n randomly chosen active people in window k (all active people if fewer)."""
    uk = D["users_in"][k]
    sel = uk if len(uk) <= n else rng.choice(uk, n, replace=False)
    a, b = D["off"][k], D["off"][k + 1]
    mask[sel] = True
    m = mask[D["user"][a:b]]
    c = np.bincount(D["item"][a:b][m], minlength=D["n_items"]).astype(np.float32)
    mask[sel] = False
    return c


def run_n(task: dict) -> dict:
    """One (dataset, n) cell: DRAWS draws of the sampled tracker. Returns nested lists [draw][h][step]."""
    D = prepare(task["dataset"])
    n = int(task["n"])
    mask = np.zeros(D["n_users"] + 1, bool)
    steps = list(range(FIRST_STEP, LAST_STEP + 1))
    truth_ndcg, self_mse, self_ndcg = [], [], []
    for dr in range(DRAWS):
        rng = np.random.default_rng(seed_of(task["dataset"], n, dr))
        samp = [sample_window(D, k, n, rng, mask) for k in range(N_WIN)]
        shr = [_shares(c) for c in samp]
        s = [np.zeros(D["n_items"], np.float64) for _ in H]
        t_nd = [[np.nan] * len(steps) for _ in H]
        s_mse = [[np.nan] * len(steps) for _ in H]
        s_nd = [[np.nan] * len(steps) for _ in H]
        for w in range(0, LAST_STEP + 1):
            for i, h in enumerate(H):
                s[i] = s[i] * _decay(h) + shr[w]
            if w < FIRST_STEP:
                continue
            j = w - FIRST_STEP
            truth = D["counts"][w + 1]
            if truth.sum() < MIN_TRUTH_ROWS or samp[w + 1].sum() <= 0:
                continue
            q = shr[w + 1]
            for i, h in enumerate(H):
                top = M.top_m(s[i], 100, rng)
                t_nd[i][j] = M.ndcg_vs_gain(top, truth, 100)
                tot = s[i].sum()
                p = s[i] / tot if tot > 0 else s[i]
                s_mse[i][j] = float(((p - q) ** 2).sum())
                s_nd[i][j] = M.ndcg_vs_gain(top, samp[w + 1], 100)
        truth_ndcg.append(t_nd)
        self_mse.append(s_mse)
        self_ndcg.append(s_nd)
    r = lambda a: np.round(np.asarray(a, np.float64), 6).tolist()
    return dict(kind="n", dataset=task["dataset"], n=n, steps=steps, H=[None if np.isinf(h) else h for h in H],
                truth_ndcg=r(truth_ndcg), self_mse=r(self_mse), self_ndcg=r(self_ndcg))


def _tv(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(_shares(a) - _shares(b)).sum())


def run_full(task: dict) -> dict:
    """Per-dataset full-community quantities: lags, full trackers, M0 ceiling, M1 drift profile, new-item share."""
    D = prepare(task["dataset"])
    name = task["dataset"]
    rng = np.random.default_rng(seed_of(name, "full"))
    steps = list(range(FIRST_STEP, LAST_STEP + 1))
    nI = D["n_items"]
    C = D["counts"]
    lag_nd = [[np.nan] * len(steps) for _ in LAGS]
    trk_nd = [[np.nan] * len(steps) for _ in H]
    m0_half = [np.nan] * len(steps)                         # list from half A of window w+1, scored on half B
    m0_lag = [[np.nan] * len(steps) for _ in LAGS]          # full list of window w-L, scored on half B
    m0_h2 = [np.nan] * len(steps)                           # full decayed tracker h=2, scored on half B
    new_share = [np.nan] * len(steps)
    born_share = [np.nan] * len(steps)
    s = [np.zeros(nI, np.float64) for _ in H]
    for w in range(0, LAST_STEP + 1):
        for i, h in enumerate(H):
            s[i] = s[i] * _decay(h) + _shares(C[w].astype(np.float64))
        if w < FIRST_STEP:
            continue
        j = w - FIRST_STEP
        truth = C[w + 1]
        if truth.sum() < MIN_TRUTH_ROWS:
            continue
        for L in LAGS:
            lag_nd[L][j] = M.ndcg_vs_gain(M.top_m(C[w - L], 100, rng), truth, 100)
        for i, h in enumerate(H):
            trk_nd[i][j] = M.ndcg_vs_gain(M.top_m(s[i], 100, rng), truth, 100)
        head = M.top_m(truth, 100, rng)
        mass = float(truth[head].sum())
        new_share[j] = float(truth[head][D["first"][head] >= w].sum() / mass) if mass > 0 else np.nan
        born_share[j] = float(truth[head][D["first"][head] == w + 1].sum() / mass) if mass > 0 else np.nan
        # M0: disjoint halves of the people of window w+1; three random splits
        a, b = D["off"][w + 1], D["off"][w + 2]
        uw, iw = D["user"][a:b], D["item"][a:b]
        halves, lags_, h2s = [], [[] for _ in LAGS], []
        mk = np.zeros(D["n_users"] + 1, bool)
        for _ in range(3):
            uk = D["users_in"][w + 1]
            if len(uk) < 4:
                continue
            pick = rng.permutation(uk)[: len(uk) // 2]
            mk[pick] = True
            inA = mk[uw]
            mk[pick] = False
            cA = np.bincount(iw[inA], minlength=nI).astype(np.float64)
            cB = np.bincount(iw[~inA], minlength=nI).astype(np.float64)
            if cB.sum() < 100:
                continue
            halves.append(M.ndcg_vs_gain(M.top_m(cA, 100, rng), cB, 100))
            for L in LAGS:
                lags_[L].append(M.ndcg_vs_gain(M.top_m(C[w - L], 100, rng), cB, 100))
            h2s.append(M.ndcg_vs_gain(M.top_m(s[3], 100, rng), cB, 100))
        if halves:
            m0_half[j] = float(np.mean(halves))
            for L in LAGS:
                m0_lag[L][j] = float(np.mean(lags_[L]))
            m0_h2[j] = float(np.mean(h2s))
    # ---- M1: drift profile with three disjoint person groups
    grp = np.random.default_rng(seed_of(name, "groups")).integers(0, 3, D["n_users"] + 1)
    G = np.zeros((3, N_WIN, nI), np.float32)
    for k in range(N_WIN):
        a, b = D["off"][k], D["off"][k + 1]
        g = grp[D["user"][a:b]]
        for q in range(3):
            G[q, k] = np.bincount(D["item"][a:b][g == q], minlength=nI)
    tot = G.sum(2)                                            # (3, windows) rows per group-window
    items = np.argsort(-G[0].sum(0))[:1000]                   # chosen by group 0 only
    ok = [k for k in range(FIRST_STEP, N_WIN) if tot[1, k] >= 300 and tot[2, k] >= 300]
    x1 = G[1][:, items] / np.maximum(tot[1][:, None], 1)
    x2 = G[2][:, items] / np.maximum(tot[2][:, None], 1)
    ks = [1, 2, 3, 4, 6, 8]
    V, Dtv, scale = {}, {}, {}
    for k in ks:
        pairs = [t for t in ok if t + k in ok]
        if len(pairs) < 3:
            continue
        V[k] = float(np.mean([np.mean((x1[t + k] - x1[t]) * (x2[t + k] - x2[t])) for t in pairs]))
        Dtv[k] = float(np.mean([_tv(G[1][t], G[2][t + k]) - _tv(G[1][t], G[2][t]) for t in pairs]))
    scale = float(np.mean([np.mean(x1[t] * x2[t]) for t in ok])) if ok else float("nan")
    r = lambda a: np.round(np.asarray(a, np.float64), 6).tolist()
    return dict(kind="full", dataset=name, steps=steps, H=[None if np.isinf(h) else h for h in H],
                window_days=D["window_days"], n_items=nI, n_users=D["n_users"], active=D["active"].tolist(),
                rows=D["rows"].tolist(), lag_ndcg=r(lag_nd), tracker_ndcg=r(trk_nd), m0_half=r(m0_half),
                m0_lag=r(m0_lag), m0_h2=r(m0_h2), new_share=r(new_share), born_share=r(born_share),
                V={str(k): v for k, v in V.items()}, TV={str(k): v for k, v in Dtv.items()}, V_scale=scale)


def run_task(task: dict) -> dict:
    return run_full(task) if task["kind"] == "full" else run_n(task)
