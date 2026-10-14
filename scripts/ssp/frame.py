"""Frame: one dataset turned into a population, a future, an evaluation panel and samplers.

Vocabulary (plain words)
  frame users   users with at least `min_rows` earlier choices ("choice" = a positive row)
  POP           80% of frame users: the GROUP whose popularity list we try to estimate
  EVAL          the other 20%: held-out people who receive recommendations (target B2)
  collection    rows at or before the cutoff time T0 (80th percentile of choice times);
                datasets without timestamps use a per-user random 80/20 split instead
  future        a user's first-time choices after T0 (never an item already in collection)
  truth A       item counts over ALL of POP in the collection window ("full-population list")
  truth B       item counts over ALL of POP in the future window ("the group's later choices")
  stale list    item counts over POP using only rows before T_old (a cheap prior: "last year's list")

A sample is N users drawn from POP, each keeping at most k of their collection choices.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

NPZ = "datasets/npz"


@dataclass
class Frame:
    name: str
    n_items: int
    min_rows: int
    has_time: bool
    T0: int
    T_old: int
    # POP users: CSR of collection choices, items in time order inside each user
    c_ptr: np.ndarray
    c_idx: np.ndarray
    c_ts: np.ndarray
    n_c: np.ndarray                  # collection choices per POP user
    n_f: np.ndarray                  # future choices per POP user
    count_A: np.ndarray              # float64 (n_items,)
    count_B: np.ndarray
    count_old: np.ndarray
    count_Aw: np.ndarray             # truth over the window (T_old, T0] only: disjoint from the stale list
    # EVAL panel (CSR): all collection choices, and future choices
    e_c: sp.csr_matrix
    e_f: sp.csr_matrix
    # POP user attributes for subgroups
    first_ts: np.ndarray             # time of the user's first choice
    genre_share: dict = field(default_factory=dict)   # genre -> share of the user's collection choices
    meta: dict = field(default_factory=dict)

    @property
    def n_pop(self) -> int:
        return len(self.n_c)

    def pop_csr(self, rows: np.ndarray | None = None) -> sp.csr_matrix:
        X = sp.csr_matrix((np.ones(len(self.c_idx), np.float32), self.c_idx, self.c_ptr),
                          shape=(self.n_pop, self.n_items))
        return X if rows is None else X[rows]


def _choose_min_rows(n_c_all: np.ndarray) -> int:
    for f in (20, 10, 5):
        if (n_c_all >= f).sum() >= 3000:
            return f
    return 5


def build_frame(name: str, seed: int = 20261014, eval_frac: float = 0.2, cut_q: float = 0.8,
                old_q: float = 0.55, min_rows: int | None = None, max_pop_users: int | None = None) -> Frame:
    z = np.load(f"{NPZ}/{name}.npz")
    user, item, ts, r = z["user"], z["item"], z["ts"], z["r"]
    thr = float(z["positive_threshold"])
    has_time = bool(z["has_time"])
    n_items = int(item.max()) + 1
    pos = r >= thr
    user, item, ts = user[pos], item[pos], ts[pos]
    rng = np.random.default_rng(seed)
    n_users = int(user.max()) + 1
    if has_time:
        T0 = int(np.quantile(ts, cut_q))
        T_old = int(np.quantile(ts, old_q))
        coll = ts <= T0
    else:
        # per-user random split: each row goes to collection with prob cut_q; order = random
        coll = rng.random(len(user)) < cut_q
        T0 = T_old = 0
        ts = rng.integers(0, 2**31, len(user)).astype(np.int64)  # random order stands in for time
        o = np.lexsort((ts, user))
        user, item, ts, coll = user[o], item[o], ts[o], coll[o]
    n_c_all = np.bincount(user[coll], minlength=n_users)
    f = min_rows if min_rows is not None else _choose_min_rows(n_c_all)
    frame_users = np.flatnonzero(n_c_all >= f)
    perm = rng.permutation(frame_users)
    n_eval = int(len(perm) * eval_frac)
    eval_users, pop_users = np.sort(perm[:n_eval]), np.sort(perm[n_eval:])
    if max_pop_users and len(pop_users) > max_pop_users:
        pop_users = np.sort(rng.choice(pop_users, max_pop_users, replace=False))
    role = np.full(n_users, -1, np.int8)
    role[pop_users] = 0
    role[eval_users] = 1
    prow = role[user]

    # ---- POP collection CSR (sorted by user then time already)
    sel = (prow == 0) & coll
    pu, pi, pt = user[sel], item[sel], ts[sel]
    pop_index = np.full(n_users, -1, np.int64)
    pop_index[pop_users] = np.arange(len(pop_users))
    pr = pop_index[pu]
    n_c = np.bincount(pr, minlength=len(pop_users))
    c_ptr = np.concatenate([[0], np.cumsum(n_c)]).astype(np.int64)
    sel_f = (prow == 0) & ~coll
    n_f = np.bincount(pop_index[user[sel_f]], minlength=len(pop_users))
    count_A = np.bincount(pi, minlength=n_items).astype(np.float64)
    count_B = np.bincount(item[sel_f], minlength=n_items).astype(np.float64)
    if has_time:
        count_old = np.bincount(pi[pt <= T_old], minlength=n_items).astype(np.float64)
        count_Aw = np.bincount(pi[pt > T_old], minlength=n_items).astype(np.float64)
    else:
        count_old = np.zeros(n_items)
        count_Aw = np.zeros(n_items)
    first_ts = np.zeros(len(pop_users), np.int64)
    if has_time:
        firsts = np.full(n_users, np.iinfo(np.int64).max, np.int64)
        np.minimum.at(firsts, user, ts)
        first_ts = firsts[pop_users]

    # ---- EVAL panel
    erows = np.full(n_users, -1, np.int64)
    ev_f_users = np.unique(user[(role[user] == 1) & ~coll])
    erows[ev_f_users] = np.arange(len(ev_f_users))
    m = (erows[user] >= 0) & coll
    e_c = sp.csr_matrix((np.ones(m.sum(), np.float32), (erows[user[m]], item[m])),
                        shape=(len(ev_f_users), n_items))
    m = (erows[user] >= 0) & ~coll
    e_f = sp.csr_matrix((np.ones(m.sum(), np.float32), (erows[user[m]], item[m])),
                        shape=(len(ev_f_users), n_items))
    e_c.data[:] = 1
    e_f.data[:] = 1

    genre_share = {}
    if "item_genres" in z.files:
        genres = z["item_genres"]
        names = sorted({g for s in genres for g in str(s).split("|") if g and g != "(no genres listed)"})
        ug = {}
        for g in names:
            has = np.array([g in str(s).split("|") for s in genres])
            hit = has[pi].astype(np.float32)
            share = np.bincount(pr, weights=hit, minlength=len(pop_users)) / np.maximum(n_c, 1)
            ug[g] = share
        genre_share = ug

    return Frame(name=name, n_items=n_items, min_rows=f, has_time=has_time, T0=T0, T_old=T_old,
                 c_ptr=c_ptr, c_idx=pi.astype(np.int32), c_ts=pt, n_c=n_c, n_f=n_f,
                 count_A=count_A, count_B=count_B, count_old=count_old, count_Aw=count_Aw, e_c=e_c, e_f=e_f,
                 first_ts=first_ts, genre_share=genre_share,
                 meta=dict(seed=seed, positive_threshold=thr, n_pop=len(pop_users),
                           n_eval=int(e_f.shape[0]), n_rows_coll=int(len(pi)),
                           rows_future=int(sel_f.sum())))


# ---------------------------------------------------------------- samplers

def draw_users(fr: Frame, n: int, rng: np.random.Generator, mode: str = "uniform",
               pool: np.ndarray | None = None):
    """Pick about n POP users (rows of the frame). Returns (rows, pi) with exact inclusion probabilities.

    uniform: exactly n users without replacement, pi = n / pool size.
    activity (prob ~ n_c) and inv_activity (prob ~ 1/n_c): Poisson sampling. User u joins independently with
      pi_u = min(1, lam * w_u), lam chosen so that sum(pi_u) = n. The sample size is random (about n), and the
      Horvitz-Thompson weights 1/pi_u are exactly unbiased.
    """
    pool = np.arange(fr.n_pop) if pool is None else pool
    n = min(n, len(pool))
    if mode == "uniform":
        rows = rng.choice(pool, n, replace=False)
        return rows, np.full(n, n / len(pool))
    w = fr.n_c[pool].astype(np.float64)
    w = w if mode == "activity" else 1.0 / np.maximum(w, 1)
    lo, hi = 0.0, 1.0
    while np.minimum(1.0, hi * w).sum() < n:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.minimum(1.0, mid * w).sum() < n:
            lo = mid
        else:
            hi = mid
    p = np.minimum(1.0, hi * w)
    pick = np.flatnonzero(rng.random(len(pool)) < p)
    if len(pick) == 0:
        pick = np.array([int(np.argmax(p))])
    return pool[pick], p[pick]


def draw_rows(fr: Frame, users: np.ndarray, k: int | None, rng: np.random.Generator,
              rule: str = "random", ts_min: int | None = None):
    """Keep at most k collection choices per user. Returns (S csr N x J, n_kept, n_avail, row_ts).

    rule: random | recent (latest k) | earliest. ts_min keeps only rows at or after that time.
    """
    starts, lens = fr.c_ptr[users], fr.n_c[users]
    total = int(lens.sum())
    grp = np.repeat(np.arange(len(users)), lens)
    off = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
    idx = np.repeat(starts, lens) + off
    if ts_min is not None:
        keep0 = fr.c_ts[idx] >= ts_min
        idx, grp = idx[keep0], grp[keep0]
        lens = np.bincount(grp, minlength=len(users))
        off = np.arange(len(idx)) - np.repeat(np.cumsum(lens) - lens, lens)
    n_avail = lens.copy()
    if k is not None:
        if rule == "random":
            rnd = rng.random(len(idx))
            o = np.lexsort((rnd, grp))
            idx, grp = idx[o], grp[o]
            keep = off < k
        elif rule == "recent":
            keep = off >= np.repeat(lens, lens) - k
        elif rule == "earliest":
            keep = off < k
        else:
            raise ValueError(rule)
        idx, grp = idx[keep], grp[keep]
    n_kept = np.bincount(grp, minlength=len(users))
    S = sp.csr_matrix((np.ones(len(idx), np.float32), fr.c_idx[idx], np.concatenate([[0], np.cumsum(n_kept)])),
                      shape=(len(users), fr.n_items))
    return S, n_kept, n_avail, fr.c_ts[idx]   # last: time of every kept row, aligned with S.indices


def eval_profiles(fr: Frame, k: int | None, rng: np.random.Generator, panel: np.ndarray):
    """Profile (k random collection choices) for each EVAL user in `panel` (row indices of e_c)."""
    C = fr.e_c[panel]
    if k is None:
        return C
    lens = np.diff(C.indptr)
    total = int(lens.sum())
    grp = np.repeat(np.arange(len(panel)), lens)
    off = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
    rnd = rng.random(total)
    o = np.lexsort((rnd, grp))
    items = C.indices[o]
    keep = off < k
    n_kept = np.bincount(grp[keep], minlength=len(panel))
    return sp.csr_matrix((np.ones(keep.sum(), np.float32), items[keep], np.concatenate([[0], np.cumsum(n_kept)])),
                         shape=C.shape)


def load_frame(name: str, **kw) -> Frame:
    """Build the frame once, then reuse a pickle (frames are small; the npz may be huge)."""
    import os
    import pickle
    path = f"datasets/frames/{name}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    fr = build_frame(name, **kw)
    os.makedirs("datasets/frames", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(fr, f, protocol=4)
    return fr
