"""Distance and accuracy metrics for ranked item lists (small-sample popularity study).

Compares an estimated ranking (from N sampled users) with the full-population
ranking or with later real choices: top-m overlap, rank-biased overlap, Kendall
tau-b on the truth top-m, gain-weighted NDCG / mass captured, exact per-user
NDCG/recall/hit over the whole catalogue, and bootstrap confidence intervals.
References: Webber, Moffat & Zobel, "A Similarity Measure for Indefinite
Rankings", ACM TOIS 28(4), 2010 (RBO, eq. 32); Kendall (1945) tau-b;
Jarvelin & Kekalainen, "Cumulated gain-based evaluation of IR techniques",
ACM TOIS 20(4), 2002 (DCG/NDCG).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.stats import kendalltau

__all__ = [
    "rank_items", "top_m", "overlap_at_m", "null_overlap", "rbo",
    "rbo_from_scores", "kendall_topm", "ndcg_vs_gain", "mass_captured",
    "per_user_ranking_metrics", "paired_bootstrap_ci", "cluster_bootstrap_ci",
]


def _tie_key(n: int, rng: np.random.Generator | None) -> np.ndarray:
    """Secondary sort key: item index (rng None) or a uniform random permutation."""
    return np.arange(n) if rng is None else rng.permutation(n)


def rank_items(score: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return all item indices ordered best-first, breaking score ties at random (or by index).

    Args:
        score: 1-D float array, higher = better (NaN not supported).
        rng: generator used to break ties uniformly at random; if None, ties are
            broken by ascending item index (stable).
    """
    score = np.asarray(score)
    key = _tie_key(score.shape[0], rng)
    return np.lexsort((key, -score))


def top_m(score: np.ndarray, m: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return the indices of the m best items, best-first, without sorting the whole catalogue.

    Uses np.argpartition to find the m-th best score, keeps everything strictly
    above it, and fills the remaining slots from the tied items by the same tie
    rule as `rank_items` (so `top_m(s, m, rng)` equals `rank_items(s, rng')[:m]`
    for identically seeded generators).

    Args:
        score: 1-D float array, higher = better (NaN not supported).
        m: list length; clipped to the number of items.
        rng: tie-breaking generator (None = ascending index).
    """
    score = np.asarray(score)
    n = score.shape[0]
    m = min(int(m), n)
    if m <= 0:
        if rng is not None:
            rng.permutation(n)  # consume the stream exactly as rank_items does
        return np.empty(0, dtype=np.int64)
    neg = -score
    key = _tie_key(n, rng)
    thr = neg[np.argpartition(neg, m - 1)[m - 1]]  # m-th best (in negated score)
    strict = np.flatnonzero(neg < thr)
    need = m - strict.size
    tied = np.flatnonzero(neg == thr)
    if need < tied.size:
        tk = key[tied]
        tied = tied[np.argpartition(tk, need - 1)[:need]]
    cand = np.concatenate([strict, tied])
    return cand[np.lexsort((key[cand], neg[cand]))]


def null_overlap(m: int, n_items: int) -> float:
    """Expected top-m overlap fraction of two independent random rankings: m / n_items.

    Args:
        m: list length.
        n_items: catalogue size.
    """
    return m / n_items


def overlap_at_m(est_score: np.ndarray, truth_score: np.ndarray, m: int,
                 rng: np.random.Generator) -> float:
    """Fraction of the truth top-m items that the estimated top-m also contains.

    Args:
        est_score: estimated item scores (ties broken randomly with rng).
        truth_score: reference item scores (ties broken by item index, fixed).
        m: list length.
        rng: generator for the estimate's tie-breaking.

    Formula: |top_m(est) & top_m(truth)| / m.
    """
    a = top_m(est_score, m, rng)
    b = top_m(truth_score, m, None)
    k = a.shape[0]
    if k == 0:
        return float("nan")
    return float(np.intersect1d(a, b, assume_unique=True).size) / k


def rbo(list_a: np.ndarray, list_b: np.ndarray, p: float = 0.9,
        depth: int | None = None) -> float:
    """Rank-biased overlap (extrapolated point estimate): top-weighted agreement of two ranked lists.

    Args:
        list_a, list_b: ranked item-index arrays (best first, no repeats), equal length.
        p: persistence in (0, 1); smaller = more top-weighted.
        depth: evaluate on the first `depth` items of each list (None = full length).

    Formula (Webber, Moffat, Zobel 2010, eq. 32), k = evaluated length,
    X_d = |first d of A & first d of B|:
        RBO_EXT = (X_k/k) p^k + ((1-p)/p) sum_{d=1..k} (X_d/d) p^d
    """
    a = np.asarray(list_a)
    b = np.asarray(list_b)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("rbo needs two 1-D lists of equal length")
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    k = a.shape[0] if depth is None else min(int(depth), a.shape[0])
    if k <= 0:
        return float("nan")
    a, b = a[:k], b[:k]
    _, ia, ib = np.intersect1d(a, b, assume_unique=True, return_indices=True)
    # An item shared at positions i (in A) and j (in B) joins the overlap at depth max(i,j)+1.
    x = np.cumsum(np.bincount(np.maximum(ia, ib), minlength=k)).astype(np.float64)
    d = np.arange(1, k + 1, dtype=np.float64)
    pd = p ** d
    return float(x[-1] / k * pd[-1] + (1.0 - p) / p * np.sum(x / d * pd))


def rbo_from_scores(est_score: np.ndarray, truth_score: np.ndarray, p: float = 0.9,
                    depth: int | None = None, rng: np.random.Generator | None = None) -> float:
    """RBO between the ranking induced by estimated scores and the one induced by truth scores.

    Args:
        est_score: estimated scores (ties broken randomly with rng, by index if None).
        truth_score: reference scores (ties broken by index).
        p: RBO persistence.
        depth: list length compared (None = whole catalogue).
        rng: tie-breaking generator for the estimate.
    """
    n = np.asarray(est_score).shape[0]
    k = n if depth is None else min(int(depth), n)
    return rbo(top_m(est_score, k, rng), top_m(truth_score, k, None), p=p)


def kendall_topm(est_score: np.ndarray, truth_score: np.ndarray, m: int) -> float:
    """Rank agreement (Kendall tau-b) between estimated and true scores over the truth top-m items.

    Args:
        est_score: estimated scores, 1-D over all items.
        truth_score: reference scores; the m best (ties by index) define the item subset.
        m: size of the subset.

    Ties in either score vector are handled by tau-b. Returns nan when either
    restricted vector is constant (tau undefined).
    """
    items = top_m(truth_score, m, None)
    e = np.asarray(est_score, dtype=np.float64)[items]
    t = np.asarray(truth_score, dtype=np.float64)[items]
    if items.size < 2 or np.all(e == e[0]) or np.all(t == t[0]):
        return float("nan")
    return float(kendalltau(e, t, variant="b").statistic)


def _ideal_gains(gain: np.ndarray, m: int) -> np.ndarray:
    """The m largest gains, descending."""
    m = min(int(m), gain.shape[0])
    if m <= 0:
        return np.empty(0, dtype=np.float64)
    g = np.partition(gain, gain.shape[0] - m)[-m:] if m < gain.shape[0] else gain
    return np.sort(g)[::-1]


def ndcg_vs_gain(list_idx: np.ndarray, gain: np.ndarray, m: int = 100) -> float:
    """Discounted gain collected by a ranked list, relative to the best possible list for the same gains.

    Args:
        list_idx: ranked item indices, best first (only the first m are used).
        gain: per-item non-negative gain vector over all items.
        m: cutoff.

    Formula: DCG = sum_r gain[list[r]] / log2(r+2); result = DCG / DCG(ideal),
    ideal = gains sorted descending. Returns 0.0 if the ideal DCG is 0.
    """
    gain = np.asarray(gain, dtype=np.float64)
    lst = np.asarray(list_idx)[:m]
    ideal = _ideal_gains(gain, m)
    idcg = float(np.sum(ideal / np.log2(np.arange(ideal.shape[0]) + 2.0)))
    if idcg <= 0.0:
        return 0.0
    dcg = float(np.sum(gain[lst] / np.log2(np.arange(lst.shape[0]) + 2.0)))
    return dcg / idcg


def mass_captured(list_idx: np.ndarray, gain: np.ndarray, m: int) -> float:
    """Share of the best achievable gain mass that the first m listed items capture.

    Args:
        list_idx: ranked item indices, best first.
        gain: per-item gain vector over all items.
        m: cutoff.

    Formula: sum(gain[list[:m]]) / sum(m largest gains); 0.0 if that denominator is 0.
    """
    gain = np.asarray(gain, dtype=np.float64)
    den = float(np.sum(_ideal_gains(gain, m)))
    if den <= 0.0:
        return 0.0
    return float(np.sum(gain[np.asarray(list_idx)[:m]])) / den


def per_user_ranking_metrics(scores: np.ndarray, seen: sparse.csr_matrix,
                             relevant: sparse.csr_matrix, m: int = 100) -> dict:
    """Exact per-user NDCG@m, recall@m and hit@m of a batch of score rows against held-out relevant items.

    Args:
        scores: dense (B, J) float array; higher = better. Not modified.
        seen: csr (B, J) binary; these items are excluded from ranking (score -> -inf).
        relevant: csr (B, J) binary relevant items. All of them count in the
            recall denominator / ideal DCG (pass them disjoint from `seen`).
        m: cutoff (clipped to J). Ranking is over the full catalogue, no sampled negatives.

    Returns dict with arrays 'ndcg', 'recall', 'hit' of length B (nan for users
    with no relevant item). NDCG = sum_{r<m} rel_r / log2(r+2) divided by the
    ideal sum over min(m, n_relevant) hits. Score ties are broken by lower item
    index; excluded items never count as hits.
    """
    scores = np.array(scores, dtype=np.float32, copy=True)
    b, j = scores.shape
    seen = sparse.csr_matrix(seen)
    rows, cols = seen.nonzero()
    scores[rows, cols] = -np.inf
    rel = sparse.csr_matrix(relevant).toarray() != 0
    nrel = rel.sum(1)

    mm = min(int(m), j)
    neg = -scores
    top = np.argpartition(neg, mm - 1, axis=1)[:, :mm] if mm < j else np.tile(np.arange(j), (b, 1))
    vals = np.take_along_axis(neg, top, axis=1)
    if mm < j:
        # Rows where the boundary value is tied with items left out of the partition:
        # redo with a stable full argsort so ties resolve by index.
        thr = vals.max(1, keepdims=True)
        bad = np.flatnonzero((neg == thr).sum(1) != (vals == thr).sum(1))
        for r in bad:
            top[r] = np.argsort(neg[r], kind="stable")[:mm]
        if bad.size:
            vals = np.take_along_axis(neg, top, axis=1)
    order = np.lexsort(np.stack([top, vals]), axis=-1)
    top = np.take_along_axis(top, order, axis=1)
    vals = np.take_along_axis(vals, order, axis=1)

    hit_mat = np.take_along_axis(rel, top, axis=1) & np.isfinite(vals)
    disc = 1.0 / np.log2(np.arange(mm) + 2.0)
    dcg = (hit_mat * disc).sum(1)
    ideal_cum = np.concatenate([[0.0], np.cumsum(disc)])
    idcg = ideal_cum[np.minimum(nrel, mm)]
    nh = hit_mat.sum(1)
    valid = nrel > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        ndcg = np.where(valid, dcg / idcg, np.nan)
        recall = np.where(valid, nh / np.maximum(nrel, 1), np.nan)
    hit = np.where(valid, (nh > 0).astype(np.float64), np.nan)
    return {"ndcg": ndcg, "recall": recall, "hit": hit}


def _boot_mean_ci(x: np.ndarray, n_boot: int, alpha: float,
                  rng: np.random.Generator | None) -> tuple[float, float, float]:
    """Percentile bootstrap CI of the mean of a 1-D vector (chunked to bound memory)."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = x.shape[0]
    if n == 0:
        return (float("nan"),) * 3
    means = np.empty(n_boot)
    step = max(1, 4_000_000 // n)
    for s in range(0, n_boot, step):
        e = min(n_boot, s + step)
        means[s:e] = x[rng.integers(0, n, size=(e - s, n))].mean(1)
    lo, hi = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap_ci(diff: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                        rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Mean of a paired difference with a percentile bootstrap confidence interval over units.

    Args:
        diff: 1-D per-unit paired differences (nan entries ignored).
        n_boot: number of bootstrap resamples.
        alpha: two-sided level; interval is the alpha/2 and 1-alpha/2 percentiles.
        rng: generator (None = default_rng(0)).

    Returns (mean, lo, hi).
    """
    x = np.asarray(diff, dtype=np.float64)
    return _boot_mean_ci(x[~np.isnan(x)], n_boot, alpha, rng)


def cluster_bootstrap_ci(values: np.ndarray, groups: np.ndarray, n_boot: int = 2000,
                         alpha: float = 0.05,
                         rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Mean of per-group means with a bootstrap CI that resamples whole groups (e.g. sampling draws).

    Args:
        values: 1-D per-observation values (nan entries ignored).
        groups: group label per observation (same length).
        n_boot: number of bootstrap resamples.
        alpha: two-sided level.
        rng: generator (None = default_rng(0)).

    Returns (mean, lo, hi) where mean is the average of the per-group means and
    each replicate averages the means of G groups drawn with replacement.
    """
    v = np.asarray(values, dtype=np.float64)
    g = np.asarray(groups)
    if v.shape != g.shape:
        raise ValueError("values and groups must have the same shape")
    ok = ~np.isnan(v)
    _, inv = np.unique(g[ok], return_inverse=True)
    cnt = np.bincount(inv)
    gm = np.bincount(inv, weights=v[ok]) / np.maximum(cnt, 1)
    return _boot_mean_ci(gm, n_boot, alpha, rng)


if __name__ == "__main__":
    import math

    def zipf_scores(n: int, r: np.random.Generator, ties: bool) -> np.ndarray:
        s = 1.0 / np.arange(1, n + 1) ** 1.0 * 50
        s = r.permutation(s)
        return np.floor(s) if ties else s + r.normal(0, 0.01, n)

    r = np.random.default_rng(1)

    # --- rank_items / top_m / overlap vs brute force ---------------------------------
    for trial in range(60):
        n = int(r.integers(1, 25))
        kind = trial % 3
        s = (np.zeros(n) if kind == 0 else
             r.integers(0, 3, n).astype(float) if kind == 1 else zipf_scores(n, r, True))
        brute_idx = sorted(range(n), key=lambda i: (-s[i], i))
        assert list(rank_items(s)) == brute_idx
        for m in (0, 1, max(1, n // 2), n, n + 3):
            assert list(top_m(s, m)) == brute_idx[:m]
            seed = int(r.integers(1 << 30))
            perm = np.random.default_rng(seed).permutation(n)
            brute_rand = sorted(range(n), key=lambda i: (-s[i], perm[i]))
            assert list(rank_items(s, np.random.default_rng(seed))) == brute_rand
            assert list(top_m(s, m, np.random.default_rng(seed))) == brute_rand[:m]
        # overlap (truth tie by index, est tie random)
        t = r.integers(0, 3, n).astype(float)
        m = int(r.integers(1, n + 1))
        seed = int(r.integers(1 << 30))
        perm = np.random.default_rng(seed).permutation(n)
        a = set(sorted(range(n), key=lambda i: (-s[i], perm[i]))[:m])
        bt = set(sorted(range(n), key=lambda i: (-t[i], i))[:m])
        assert math.isclose(overlap_at_m(s, t, m, np.random.default_rng(seed)),
                            len(a & bt) / m)
    assert null_overlap(10, 200) == 0.05
    # tie-breaking is uniform: all-zero scores, each item lands first ~1/4 of the time
    rr = np.random.default_rng(5)
    first = np.bincount([top_m(np.zeros(4), 1, rr)[0] for _ in range(2000)], minlength=4)
    assert np.all(np.abs(first / 2000 - 0.25) < 0.05), first
    # all-zero est vs zero truth: random est top-m overlaps the index-fixed top-m ~ m/n
    ov = np.mean([overlap_at_m(np.zeros(100), np.zeros(100), 10, rr) for _ in range(400)])
    assert abs(ov - null_overlap(10, 100)) < 0.02, ov

    # --- RBO ---------------------------------------------------------------------------
    def rbo_brute(a, b, p):
        k = len(a)
        xs = [len(set(a[:d]) & set(b[:d])) for d in range(1, k + 1)]
        return (xs[-1] / k * p ** k
                + (1 - p) / p * sum(xs[d - 1] / d * p ** d for d in range(1, k + 1)))
    for _ in range(50):
        k = int(r.integers(1, 30))
        a = r.permutation(60)[:k]
        b = r.permutation(60)[:k]
        p = float(r.uniform(0.2, 0.99))
        assert math.isclose(rbo(a, b, p), rbo_brute(list(a), list(b), p), rel_tol=1e-12)
        dd = int(r.integers(1, k + 1))
        assert math.isclose(rbo(a, b, p, depth=dd), rbo_brute(list(a[:dd]), list(b[:dd]), p),
                            rel_tol=1e-12)
    for k, p in ((1, 0.9), (10, 0.9), (50, 0.5)):
        a = np.arange(k)
        assert math.isclose(rbo(a, a, p), 1.0, rel_tol=1e-12)           # identical -> 1
        assert rbo(a, a + k, p) == 0.0                                  # disjoint -> 0
    # known value: two lists sharing only the first item, k=2: X=(1,1)
    assert math.isclose(rbo(np.array([0, 1]), np.array([0, 2]), 0.5),
                        (1 / 2) * 0.25 + 1.0 * (1 * 0.5 + 0.5 * 0.25), rel_tol=1e-12)
    for _ in range(20):
        n = int(r.integers(2, 30))
        s = r.integers(0, 4, n).astype(float)
        t = r.integers(0, 4, n).astype(float)
        seed = int(r.integers(1 << 30))
        perm = np.random.default_rng(seed).permutation(n)
        la = sorted(range(n), key=lambda i: (-s[i], perm[i]))
        lb = sorted(range(n), key=lambda i: (-t[i], i))
        for dep in (None, max(1, n // 2)):
            kk = n if dep is None else dep
            assert math.isclose(rbo_from_scores(s, t, 0.9, dep, np.random.default_rng(seed)),
                                rbo_brute(la[:kk], lb[:kk], 0.9), rel_tol=1e-12)

    # --- Kendall tau-b on truth top-m --------------------------------------------------
    def tau_b_brute(x, y):
        n = len(x)
        c = d = tx = ty = 0
        for i in range(n):
            for j in range(i + 1, n):
                sx, sy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
                if sx == 0 and sy == 0:
                    continue
                if sx == 0:
                    tx += 1
                elif sy == 0:
                    ty += 1
                elif sx == sy:
                    c += 1
                else:
                    d += 1
        den = math.sqrt((c + d + tx) * (c + d + ty))
        return (c - d) / den if den > 0 else float("nan")
    for _ in range(40):
        n = int(r.integers(2, 30))
        s = r.integers(0, 5, n).astype(float)
        t = r.integers(0, 5, n).astype(float)
        m = int(r.integers(2, n + 1))
        items = sorted(range(n), key=lambda i: (-t[i], i))[:m]
        exp = tau_b_brute(s[items], t[items])
        got = kendall_topm(s, t, m)
        assert (math.isnan(exp) and math.isnan(got)) or math.isclose(exp, got, abs_tol=1e-12)
    assert math.isnan(kendall_topm(np.zeros(5), np.arange(5.0), 5))

    # --- NDCG vs gain / mass captured --------------------------------------------------
    for _ in range(40):
        n = int(r.integers(1, 30))
        g = r.integers(0, 4, n).astype(float) if r.random() < 0.5 else r.random(n)
        lst = r.permutation(n)[: int(r.integers(0, n + 1))]
        m = int(r.integers(1, 12))
        dcg = sum(g[lst[i]] / math.log2(i + 2) for i in range(min(m, len(lst))))
        ig = sorted(g, reverse=True)[:m]
        idcg = sum(ig[i] / math.log2(i + 2) for i in range(len(ig)))
        assert math.isclose(ndcg_vs_gain(lst, g, m), dcg / idcg if idcg > 0 else 0.0,
                            abs_tol=1e-12)
        mass = sum(g[i] for i in lst[:m])
        assert math.isclose(mass_captured(lst, g, m), mass / sum(ig) if sum(ig) > 0 else 0.0,
                            abs_tol=1e-12)
    assert ndcg_vs_gain(np.arange(5), np.zeros(5), 3) == 0.0
    g = zipf_scores(50, r, False)
    assert math.isclose(ndcg_vs_gain(rank_items(g), g, 10), 1.0)
    assert math.isclose(mass_captured(rank_items(g), g, 10), 1.0)

    # --- per_user_ranking_metrics vs naive loop ----------------------------------------
    def naive_user(sc, seen_set, rel_set, m):
        ranked = sorted((i for i in range(len(sc)) if i not in seen_set),
                        key=lambda i: (-sc[i], i))[:m]
        if not rel_set:
            return (math.nan,) * 3
        dcg = sum(1 / math.log2(k + 2) for k, i in enumerate(ranked) if i in rel_set)
        idcg = sum(1 / math.log2(k + 2) for k in range(min(m, len(rel_set))))
        h = sum(1 for i in ranked if i in rel_set)
        return dcg / idcg, h / len(rel_set), float(h > 0)
    for trial in range(30):
        b, j = int(r.integers(1, 12)), int(r.integers(2, 40))
        m = int(r.integers(1, j + 5))
        sc = (r.integers(0, 4, (b, j)) if trial % 2 else r.normal(size=(b, j))).astype(np.float32)
        if trial % 5 == 0:
            sc[:] = 0
        sn = (r.random((b, j)) < 0.3)
        rl = (r.random((b, j)) < 0.15) & ~sn
        if trial % 4 == 0:
            rl[0] = False
        res = per_user_ranking_metrics(sc, sparse.csr_matrix(sn.astype(np.float32)),
                                       sparse.csr_matrix(rl.astype(np.float32)), m)
        for u in range(b):
            exp = naive_user(sc[u], set(np.flatnonzero(sn[u])), set(np.flatnonzero(rl[u])), m)
            got = (res["ndcg"][u], res["recall"][u], res["hit"][u])
            for e, gv in zip(exp, got):
                assert (math.isnan(e) and math.isnan(gv)) or math.isclose(e, gv, abs_tol=1e-9), \
                    (trial, u, exp, got)
    # input must not be modified
    sc = np.ones((2, 5), np.float32)
    per_user_ranking_metrics(sc, sparse.csr_matrix(np.eye(2, 5, dtype=np.float32)),
                             sparse.csr_matrix(np.eye(2, 5, k=1, dtype=np.float32)), 3)
    assert np.all(sc == 1)

    # --- bootstrap CIs -----------------------------------------------------------------
    d = np.array([0.5, -0.2, np.nan, 1.0, 0.3, 0.8, np.nan, 0.1])
    clean = d[~np.isnan(d)]
    mu, lo, hi = paired_bootstrap_ci(d, 500, 0.1, np.random.default_rng(7))
    idx = np.random.default_rng(7).integers(0, clean.size, size=(500, clean.size))
    bm = clean[idx].mean(1)
    assert math.isclose(mu, clean.mean())
    assert math.isclose(lo, np.percentile(bm, 5)) and math.isclose(hi, np.percentile(bm, 95))
    assert lo <= mu <= hi
    assert paired_bootstrap_ci(np.full(10, 0.25), 100, rng=np.random.default_rng(0)) == (0.25,) * 3
    assert all(math.isnan(x) for x in paired_bootstrap_ci(np.array([np.nan])))
    # chunking path gives the same stream as one big draw
    big = np.random.default_rng(3).normal(size=3000)
    c1 = paired_bootstrap_ci(big, 2500, 0.05, np.random.default_rng(9))
    ix = np.random.default_rng(9).integers(0, 3000, size=(2500, 3000))
    bm = big[ix].mean(1)
    assert math.isclose(c1[1], np.percentile(bm, 2.5)) and math.isclose(c1[2], np.percentile(bm, 97.5))
    vals = np.array([1.0, 3.0, 2.0, np.nan, 10.0, 6.0, 8.0])
    grp = np.array([0, 0, 1, 1, 2, 2, 2])
    gmeans = np.array([2.0, 2.0, 8.0])
    mu, lo, hi = cluster_bootstrap_ci(vals, grp, 400, 0.05, np.random.default_rng(11))
    bm = gmeans[np.random.default_rng(11).integers(0, 3, size=(400, 3))].mean(1)
    assert math.isclose(mu, gmeans.mean())
    assert math.isclose(lo, np.percentile(bm, 2.5)) and math.isclose(hi, np.percentile(bm, 97.5))
    # CI shrinks with more groups and covers the true mean of a synthetic Zipf-noise draw
    z = np.random.default_rng(2).normal(0.3, 1.0, 400)
    mu, lo, hi = paired_bootstrap_ci(z, 1000, 0.05, np.random.default_rng(2))
    assert lo < 0.3 < hi and hi - lo < 0.4

    print("OK metrics")
