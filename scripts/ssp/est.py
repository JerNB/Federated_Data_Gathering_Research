"""Estimators and scorers for the small-sample popularity study.

A "draw" is one sample S (N users x k choices). From S we build

  popularity scores    count | ht (inverse-probability weighted) | shrunk (toward a stale list) | lcb
  personalised scores  itemknn | ease  (trained on S only, honest lambda by inner hold-out)
  hybrid               w * z(personalised) + (1-w) * z(popularity), w chosen by inner hold-out

and score every one of them two ways:

  list quality   against truth A and truth B at the level of the whole list (needs no people)
  user quality   NDCG@m / recall@m for the fixed EVAL panel, who are NOT in the sample
                 (each is shown k random earlier choices; every earlier choice is hidden from the list)

Everything is evaluated inside the candidate universe Omega = the J_c items with the highest
sample count (ties random). Items outside Omega cannot be recommended by any method here;
relevant items outside Omega count as misses (the ideal DCG still uses all future choices).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import metrics as M

LOG2 = np.log2(np.arange(2, 2 + 1000))  # discount table; m <= 1000


# ------------------------------------------------------------------ popularity

def pop_count(S: sp.csr_matrix) -> np.ndarray:
    return np.asarray(S.sum(0)).ravel().astype(np.float64)


def pop_ht(S: sp.csr_matrix, user_w: np.ndarray) -> np.ndarray:
    """Horvitz-Thompson count: every kept choice weighted by 1/(user inclusion prob) * (rows available / rows kept)."""
    return np.asarray(S.T @ user_w).ravel().astype(np.float64)


def pop_decay(S: sp.csr_matrix, row_ts: np.ndarray, t_ref: int, half_life: float) -> np.ndarray:
    """Recency-weighted count: a choice made `half_life` seconds before t_ref counts half as much."""
    w = np.exp2(-(t_ref - row_ts.astype(np.float64)) / max(half_life, 1.0))
    return np.bincount(S.indices, weights=w, minlength=S.shape[1])


def pop_shrunk(c: np.ndarray, prior_counts: np.ndarray, kappa: float) -> np.ndarray:
    """(c + kappa * pi) / (n + kappa) with pi = normalised prior counts."""
    n = c.sum()
    pi = prior_counts / max(prior_counts.sum(), 1.0)
    return (c + kappa * pi) / (n + kappa)


def pop_lcb(c: np.ndarray, n_users: int, z: float = 1.0) -> np.ndarray:
    """Wilson lower bound on the inclusion rate: ranks items by a pessimistic rate."""
    p = c / max(n_users, 1)
    d = 1 + z * z / n_users
    centre = p + z * z / (2 * n_users)
    half = z * np.sqrt(np.maximum(p * (1 - p) / n_users + z * z / (4 * n_users**2), 0))
    return (centre - half) / d


# ------------------------------------------------------------------ list vs truth

def user_norm_counts(fr) -> np.ndarray:
    """Truth A_u: every person weighs 1 and spreads it over their own choices (sum_u x_uj / n_u).

    Plain sample counts at k < n_u estimate THIS list, not Truth A, because a person with a short history
    contributes a larger share of their rows (k / n_u) than a person with a long one.
    """
    if not hasattr(fr, "_count_Au"):
        w = np.repeat(1.0 / np.maximum(fr.n_c, 1), fr.n_c)
        fr._count_Au = np.bincount(fr.c_idx, weights=w, minlength=fr.n_items)
    return fr._count_Au


def list_metrics(score: np.ndarray, fr, rng: np.random.Generator, prefix: str) -> dict:
    """Quality of a score vector as a whole-list estimate of truth A (and B)."""
    out = {}
    tA, tB = fr.count_A, fr.count_B
    est_top = M.top_m(score, 100, rng)
    for m in (10, 100):
        out[f"{prefix}.A_ov{m}"] = M.overlap_at_m(score, tA, m, rng)
    out[f"{prefix}.A_rbo90"] = M.rbo_from_scores(score, tA, 0.9, 100, rng)
    out[f"{prefix}.A_rbo98"] = M.rbo_from_scores(score, tA, 0.98, 100, rng)
    out[f"{prefix}.A_tau100"] = M.kendall_topm(score, tA, 100)
    out[f"{prefix}.A_ndcg100"] = M.ndcg_vs_gain(est_top, tA, 100)
    out[f"{prefix}.B_ndcg100"] = M.ndcg_vs_gain(est_top, tB, 100)
    out[f"{prefix}.B_mass10"] = M.mass_captured(M.top_m(score, 10, rng), tB, 10)
    out[f"{prefix}.B_ov100"] = M.overlap_at_m(score, tB, 100, rng)
    tU = user_norm_counts(fr)
    out[f"{prefix}.Au_ndcg100"] = M.ndcg_vs_gain(est_top, tU, 100)
    out[f"{prefix}.Au_ov10"] = M.overlap_at_m(score, tU, 10, rng)
    out[f"{prefix}.Au_ov100"] = M.overlap_at_m(score, tU, 100, rng)
    # window truth: items counted only in (T_old, T0], disjoint from the stale list
    tW = fr.count_Aw
    if tW.sum() > 0:
        out[f"{prefix}.W_ov10"] = M.overlap_at_m(score, tW, 10, rng)
        out[f"{prefix}.W_ov100"] = M.overlap_at_m(score, tW, 100, rng)
        out[f"{prefix}.W_rbo90"] = M.rbo_from_scores(score, tW, 0.9, 100, rng)
        out[f"{prefix}.W_ndcg100"] = M.ndcg_vs_gain(est_top, tW, 100)
    return out


# ------------------------------------------------------------------ universe + panel evaluation

def universe(c: np.ndarray, j_cap: int, rng: np.random.Generator) -> np.ndarray:
    """Top j_cap items by sample count among items seen at least once (ties random)."""
    seen = np.flatnonzero(c > 0)
    if len(seen) <= j_cap:
        return seen
    jit = rng.random(len(seen)) * 0.5
    top = np.argpartition(-(c[seen] + jit), j_cap)[:j_cap]
    return np.sort(seen[top])


class Panel:
    """Fixed EVAL users, restricted to a universe. Built once per draw."""

    def __init__(self, fr, panel_rows: np.ndarray, profiles: sp.csr_matrix, omega: np.ndarray):
        self.seen = fr.e_c[panel_rows][:, omega].toarray() > 0          # E x J_c
        self.rel = fr.e_f[panel_rows][:, omega].toarray() > 0
        self.profile = profiles[:, omega].tocsr()                        # E x J_c
        self.n_rel = np.asarray(fr.e_f[panel_rows].sum(1)).ravel()       # full-catalogue relevant count
        self.E = len(panel_rows)

    def subset(self, mask: np.ndarray) -> "Panel":
        p = object.__new__(Panel)
        p.seen, p.rel, p.profile = self.seen[mask], self.rel[mask], self.profile[mask]
        p.n_rel, p.E = self.n_rel[mask], int(mask.sum())
        return p


def rank_eval(scores: np.ndarray, pn: Panel, m: int = 100, rng: np.random.Generator | None = None) -> dict:
    """NDCG@m, recall@m, hit@m for E users. scores is E x J_c (or 1 x J_c, broadcast).

    Ties are broken uniformly at random (a jitter far below any real score gap), as the pre-registration says.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    base = np.broadcast_to(np.asarray(scores, dtype=np.float32), pn.seen.shape)
    jitter = rng.random(pn.seen.shape, dtype=np.float32) * 1e-5
    s = np.where(pn.seen, -np.inf, base + jitter)
    m_eff = min(m, s.shape[1])
    part = np.argpartition(-s, m_eff - 1, axis=1)[:, :m_eff]
    vals = np.take_along_axis(s, part, 1)
    order = np.argsort(-vals, axis=1, kind="stable")
    top = np.take_along_axis(part, order, 1)
    ok = np.isfinite(np.take_along_axis(vals, order, 1))
    hit = np.take_along_axis(pn.rel, top, 1) & ok
    dcg = (hit / LOG2[:m_eff]).sum(1)
    ideal_n = np.minimum(pn.n_rel, m).astype(int)
    idcg = np.concatenate([[0.0], np.cumsum(1.0 / LOG2[:m])])[ideal_n]
    ndcg = np.where(idcg > 0, dcg / np.maximum(idcg, 1e-12), np.nan)
    rec = hit.sum(1) / np.maximum(pn.n_rel, 1)
    return {"ndcg": float(np.nanmean(ndcg)), "recall": float(np.mean(rec)),
            "hit": float(np.mean(hit.any(1)))}


def zscore_rows(a: np.ndarray) -> np.ndarray:
    a = np.atleast_2d(a).astype(np.float32)
    sd = a.std(1, keepdims=True)
    return (a - a.mean(1, keepdims=True)) / np.where(sd > 0, sd, 1.0)


# ------------------------------------------------------------------ personalised models (on Omega)

def gram(S_om: sp.csr_matrix) -> np.ndarray:
    return (S_om.T @ S_om).toarray().astype(np.float64)


def ease_B(G: np.ndarray, lam: float) -> np.ndarray:
    A = G.copy()
    A[np.diag_indices_from(A)] += lam
    P = np.linalg.inv(A)
    B = -P / np.diag(P)[None, :]
    B[np.diag_indices_from(B)] = 0.0
    return B.astype(np.float32)


def knn_W(G: np.ndarray, shrink: float = 10.0) -> np.ndarray:
    d = np.sqrt(np.maximum(np.diag(G), 1e-9))
    W = G / (d[:, None] * d[None, :] + shrink)
    np.fill_diagonal(W, 0.0)
    return W.astype(np.float32)


def apply_model(profile: sp.csr_matrix, W: np.ndarray) -> np.ndarray:
    return np.asarray(profile @ W)


def split_hidden(S: sp.csr_matrix, rng: np.random.Generator, frac: float = 0.25, row_ts=None):
    """Hide ~frac of each sampled user's choices (at least 1, leaving at least 1 visible).

    row_ts given -> hide each user's LATEST rows (time-ordered hold-out, mimics predicting the future);
    row_ts None   -> hide a random subset.
    """
    lens = np.diff(S.indptr)
    ok = lens >= 2
    grp = np.repeat(np.arange(S.shape[0]), lens)
    if row_ts is not None:
        o = np.lexsort((row_ts, grp))
        rank = np.empty(len(o), np.int64)
        within = np.arange(len(o)) - np.repeat(S.indptr[:-1], lens)
        rank[o] = within
        n_hide = np.maximum(1, np.round(lens * frac)).astype(np.int64)
        hide = (rank >= (lens - n_hide)[grp]) & ok[grp]
    else:
        hide = (rng.random(len(S.indices)) < frac) & ok[grp]
    # make sure at least one hidden and one visible per eligible user
    first = S.indptr[:-1]
    hide[first[ok & (np.bincount(grp[hide], minlength=S.shape[0]) == 0)]] = True
    vis_cnt = np.bincount(grp[~hide], minlength=S.shape[0])
    last = (S.indptr[1:] - 1)
    fix = ok & (vis_cnt == 0)
    hide[last[fix]] = False
    keep_user = ok
    Vi = sp.csr_matrix((np.ones((~hide).sum(), np.float32), S.indices[~hide],
                        np.concatenate([[0], np.cumsum(np.bincount(grp[~hide], minlength=S.shape[0]))])),
                       shape=S.shape)
    Hi = sp.csr_matrix((np.ones(hide.sum(), np.float32), S.indices[hide],
                        np.concatenate([[0], np.cumsum(np.bincount(grp[hide], minlength=S.shape[0]))])),
                       shape=S.shape)
    return Vi[keep_user], Hi[keep_user]


def _cv_scores(Vi, Hi, omega, grid_lams, rng, ws, pop_scores_fn):
    """Inner hold-out on the sample: returns dict of NDCG per (model, lam, w)."""
    cols = omega
    V = Vi[:, cols].tocsr()
    H = Hi[:, cols].toarray() > 0
    seen = V.toarray() > 0
    n_rel = np.asarray(Hi.sum(1)).ravel()
    pn = type("P", (), {})()
    pn.seen, pn.rel, pn.n_rel = seen, H, n_rel
    pop = pop_scores_fn(Vi)[cols]
    G = gram(V)
    res = {}
    zpop = zscore_rows(pop)
    for lam in grid_lams:
        B = ease_B(G, lam)
        sc = zscore_rows(apply_model(V, B))
        for w in ws:
            res[("ease", lam, w)] = rank_eval((1 - w) * zpop + w * sc, pn)["ndcg"]
    W = knn_W(G)
    sc = zscore_rows(apply_model(V, W))
    for w in ws:
        res[("knn", 0.0, w)] = rank_eval((1 - w) * zpop + w * sc, pn)["ndcg"]
    return res
