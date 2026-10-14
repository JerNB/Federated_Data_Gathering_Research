"""Confidence sets for item popularity ranks estimated from a user sample.

Ranks are defined from the item inclusion rate theta_j (column mean of the binary
user x item matrix).  Two interval methods are provided: the naive user-cluster
bootstrap percentile interval of the rank (inconsistent for ranks, Hall & Miller
2009, "Using the bootstrap to quantify the authority of an empirical ranking",
Ann. Statist.) and the marginal rank confidence set built from pairwise studentised
bootstrap comparisons (Mogstad, Romano, Shaikh & Wilhelm, "Inference for Ranks",
NBER w26883 / Review of Economic Studies 2024, Section 3 and Algorithm 3.1).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

_TINY_SE = 1e-12  # replacement for a zero bootstrap standard error
_CHUNK_ROWS = 100  # bootstrap replicates generated per chunk (bounds memory)


# ---------------------------------------------------------------- helpers
def _as_items(items: np.ndarray | None, n_items: int) -> np.ndarray:
    """Return the focus item indices as an int64 array (all items if None)."""
    if items is None:
        return np.arange(n_items, dtype=np.int64)
    return np.asarray(items, dtype=np.int64).ravel()


def _check(alpha: float, n_boot: int) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if n_boot < 2:
        raise ValueError("n_boot must be at least 2")


def _boot_counts(
    X: sp.csr_matrix, cols: np.ndarray, n_boot: int, rng: np.random.Generator
):
    """Yield (n_chunk, len(cols)) float64 bootstrap item counts, resampling users.

    Replicates come from a multinomial user-weight matrix times the sparse X,
    generated in chunks of ``_CHUNK_ROWS`` replicates.
    """
    n_users = X.shape[0]
    Xs = X[:, cols].tocsr()
    XsT = Xs.T.tocsr()
    pvals = np.full(n_users, 1.0 / n_users)
    done = 0
    while done < n_boot:
        c = min(_CHUNK_ROWS, n_boot - done)
        W = rng.multinomial(n_users, pvals, size=c).astype(np.float64)
        yield np.asarray(XsT @ W.T, dtype=np.float64).T
        done += c


def _ranks_of_values(sorted_asc: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Rank 1 + #{entries strictly greater} for ``values`` given an ascending sort."""
    return 1 + sorted_asc.size - np.searchsorted(sorted_asc, values, side="right")


# ---------------------------------------------------------------- public API
def true_ranks(theta: np.ndarray) -> np.ndarray:
    """Give each item its popularity rank, 1 = highest rate, ties sharing the smaller rank.

    Args:
        theta: 1-D array of item scores (e.g. inclusion rates), higher = better.

    Returns:
        int64 array r with r_j = 1 + #{k : theta_k > theta_j}.
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()
    return _ranks_of_values(np.sort(theta), theta).astype(np.int64)


def percentile_rank_intervals(
    X: sp.csr_matrix,
    alpha: float = 0.05,
    n_boot: int = 500,
    rng: np.random.Generator | None = None,
    items: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Give the naive bootstrap percentile interval of each item's rank (known to be unreliable).

    Users (rows of X) are resampled with replacement; the rank of every item
    among all items is computed in each replicate and the alpha/2 and 1-alpha/2
    percentiles of those ranks are returned (rounded outwards to integer ranks).
    The naive n-out-of-n bootstrap of ranks is inconsistent (Hall & Miller 2009).

    Args:
        X: binary csr_matrix (n_users, n_items).
        alpha: nominal miscoverage level.
        n_boot: number of bootstrap replicates.
        rng: numpy Generator; ``None`` uses ``default_rng(0)``.
        items: optional item indices whose intervals are returned (ranks are
            still computed against all items); default all items.

    Returns:
        (lo, hi) int64 arrays aligned with ``items``.
    """
    _check(alpha, n_boot)
    rng = np.random.default_rng(0) if rng is None else rng
    X = sp.csr_matrix(X)
    n_items = X.shape[1]
    focus = _as_items(items, n_items)
    all_cols = np.arange(n_items)
    ranks = np.empty((n_boot, focus.size), dtype=np.int64)
    row = 0
    for counts in _boot_counts(X, all_cols, n_boot, rng):
        for b in range(counts.shape[0]):
            ranks[row] = _ranks_of_values(np.sort(counts[b]), counts[b, focus])
            row += 1
    lo = np.percentile(ranks, 100.0 * alpha / 2.0, axis=0, method="lower")
    hi = np.percentile(ranks, 100.0 * (1.0 - alpha / 2.0), axis=0, method="higher")
    return lo.astype(np.int64), hi.astype(np.int64)


def mogstad_rank_cs(
    X: sp.csr_matrix,
    alpha: float = 0.05,
    n_boot: int = 500,
    rng: np.random.Generator | None = None,
    items: np.ndarray | None = None,
    max_compare: int = 1500,
    stepdown: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Give a marginal confidence set for each item's rank that holds with probability about 1-alpha.

    For focus item j and comparison items k, D_jk = theta_j - theta_k is estimated
    by the sample difference and studentised by its user-bootstrap standard error
    se_jk.  The critical value c_j is the (1-alpha) quantile over bootstrap
    replicates of max_k |(D*_jk - D_jk)| / se_jk.  Item k is *significantly better*
    than j if D_jk + c_j se_jk < 0 (count N-_j) and *significantly worse* if
    D_jk - c_j se_jk > 0 (count N+_j).  The rank set is
    R_j = {N-_j + 1, ..., J - N+_j}, J = total number of items (rank 1 = highest
    rate).  With ``stepdown=True`` (Algorithm 3.1) the comparisons rejected so far
    are dropped and c_j recomputed on the remaining ones until nothing changes.

    The comparison set K holds the ``max_compare`` items with the largest sample
    counts; items with zero sample count are never compared (they cannot be shown
    significantly worse, so they count towards hi).  A zero bootstrap standard
    error is replaced by 1e-12 and its studentised deviation set to 0, so a tie at
    the same estimate is never significant.

    Args:
        X: binary csr_matrix (n_users, n_items).
        alpha: nominal miscoverage level.
        n_boot: number of bootstrap replicates.
        rng: numpy Generator; ``None`` uses ``default_rng(0)``.
        items: optional item indices to return sets for; default all items.
        max_compare: maximum size of the comparison set K.
        stepdown: use the step-down refinement of Algorithm 3.1.

    Returns:
        (lo, hi) int64 arrays aligned with ``items``; lo = N-_j + 1, hi = J - N+_j.
        The sample rank of j always lies in [lo, hi].

    Reference: Mogstad, Romano, Shaikh, Wilhelm, Inference for Ranks, Section 3.
    """
    _check(alpha, n_boot)
    rng = np.random.default_rng(0) if rng is None else rng
    X = sp.csr_matrix(X)
    n_users, n_items = X.shape
    focus = _as_items(items, n_items)

    counts = np.asarray(X.sum(axis=0), dtype=np.float64).ravel()
    order = np.argsort(-counts, kind="stable")[: max(int(max_compare), 0)]
    comp = order[counts[order] > 0]  # comparison set K (items with sample count > 0)
    lo = np.ones(focus.size, dtype=np.int64)
    hi = np.full(focus.size, n_items, dtype=np.int64)
    if comp.size == 0:
        return lo, hi

    # Bootstrap means for the union of focus items and comparison items.
    cols = np.union1d(focus, comp)
    col_pos = np.full(n_items, -1, dtype=np.int64)
    col_pos[cols] = np.arange(cols.size)
    M = np.empty((n_boot, cols.size), dtype=np.float64)
    row = 0
    for chunk in _boot_counts(X, cols, n_boot, rng):
        M[row : row + chunk.shape[0]] = chunk / n_users
        row += chunk.shape[0]
    MK = M[:, col_pos[comp]]  # (n_boot, |K|)
    theta_all = counts / n_users
    thK = theta_all[comp]
    comp_pos = np.full(n_items, -1, dtype=np.int64)
    comp_pos[comp] = np.arange(comp.size)

    for i, j in enumerate(focus):
        Dstar = M[:, col_pos[j]][:, None] - MK  # (n_boot, |K|)
        Dhat = theta_all[j] - thK
        se = Dstar.std(axis=0, ddof=1)
        positive = se > _TINY_SE
        se = np.where(positive, se, _TINY_SE)
        T = np.abs(Dstar - Dhat) / se
        T[:, ~positive] = 0.0
        active = np.ones(comp.size, dtype=bool)
        if comp_pos[j] >= 0:
            active[comp_pos[j]] = False  # no self-comparison
        if not active.any():
            continue
        c = np.quantile(T[:, active].max(axis=1), 1.0 - alpha, method="higher")
        if stepdown:
            while True:
                reject = active & (np.abs(Dhat) > c * se)
                if not reject.any():
                    break
                active = active & ~reject
                if not active.any():
                    break
                c_new = np.quantile(T[:, active].max(axis=1), 1.0 - alpha, method="higher")
                if c_new >= c:
                    break
                c = c_new
        valid = np.ones(comp.size, dtype=bool)
        if comp_pos[j] >= 0:
            valid[comp_pos[j]] = False
        n_minus = int(np.count_nonzero(valid & (Dhat + c * se < 0)))
        n_plus = int(np.count_nonzero(valid & (Dhat - c * se > 0)))
        lo[i] = n_minus + 1
        hi[i] = n_items - n_plus
    return lo, hi


def coverage_table(
    lo: np.ndarray,
    hi: np.ndarray,
    truth_rank: np.ndarray,
    bins: list[tuple[int, int]],
) -> list[dict]:
    """Report how often the true rank falls inside each item's interval, by true-rank bin.

    Args:
        lo, hi: integer arrays of interval end points (inclusive), one per item.
        truth_rank: true rank per item, aligned with lo/hi.
        bins: list of inclusive (low, high) true-rank ranges, e.g.
            [(1, 10), (11, 50), (51, 200), (201, 1000)].

    Returns:
        One dict per bin with keys ``bin`` (the pair), ``n_items``, ``coverage``
        (fraction with lo <= truth <= hi, nan if the bin is empty) and
        ``mean_width`` (mean of hi - lo + 1, nan if empty).
    """
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    truth = np.asarray(truth_rank)
    covered = (lo <= truth) & (truth <= hi)
    width = (hi - lo + 1).astype(np.float64)
    out = []
    for a, b in bins:
        m = (truth >= a) & (truth <= b)
        n = int(m.sum())
        out.append(
            {
                "bin": (a, b),
                "n_items": n,
                "coverage": float(covered[m].mean()) if n else float("nan"),
                "mean_width": float(width[m].mean()) if n else float("nan"),
            }
        )
    return out


# ---------------------------------------------------------------- self-test
def _zipf_users(
    n_users: int, k: int, weights: np.ndarray, rng: np.random.Generator
) -> sp.csr_matrix:
    """Draw n_users users each with k distinct items sampled with probability ~ weights."""
    n_items = weights.size
    logw = np.log(weights)
    blocks = []
    for start in range(0, n_users, 10000):
        m = min(10000, n_users - start)
        keys = logw + rng.gumbel(size=(m, n_items))
        blocks.append(np.argpartition(-keys, k - 1, axis=1)[:, :k])
    idx = np.concatenate(blocks, axis=0)
    indptr = np.arange(0, n_users * k + 1, k)
    data = np.ones(n_users * k, dtype=np.float32)
    return sp.csr_matrix((data, idx.ravel(), indptr), shape=(n_users, n_items))


def _naive_boot_means(X: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Brute-force bootstrap item counts: explicitly resample dense rows (test helper)."""
    n = X.shape[0]
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)
    out = np.empty((n_boot, X.shape[1]))
    for b in range(n_boot):
        rows = np.repeat(np.arange(n), W[b])
        out[b] = X[rows].sum(0)
    return out


def _naive_mogstad(
    X: np.ndarray, alpha: float, n_boot: int, rng: np.random.Generator,
    max_compare: int, stepdown: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Loop-based reference implementation of mogstad_rank_cs (test helper)."""
    n, J = X.shape
    counts = X.sum(0)
    K = [k for k in np.argsort(-counts, kind="stable")[:max_compare] if counts[k] > 0]
    B = _naive_boot_means(X, n_boot, rng) / n
    th = counts / n
    lo, hi = np.ones(J, dtype=np.int64), np.full(J, J, dtype=np.int64)
    for j in range(J):
        ks = [k for k in K if k != j]
        if not ks:
            continue
        D = {k: th[j] - th[k] for k in ks}
        se = {}
        for k in ks:
            s = np.std(B[:, j] - B[:, k], ddof=1)
            se[k] = s if s > 1e-12 else 1e-12
        def tstat(k):
            if np.std(B[:, j] - B[:, k], ddof=1) <= 1e-12:
                return np.zeros(n_boot)
            return np.abs(B[:, j] - B[:, k] - D[k]) / se[k]
        act = list(ks)
        c = np.quantile(np.max([tstat(k) for k in act], axis=0), 1 - alpha, method="higher")
        if stepdown:
            while True:
                rej = [k for k in act if abs(D[k]) > c * se[k]]
                if not rej:
                    break
                act = [k for k in act if k not in rej]
                if not act:
                    break
                cn = np.quantile(np.max([tstat(k) for k in act], axis=0), 1 - alpha, method="higher")
                if cn >= c:
                    break
                c = cn
        lo[j] = 1 + sum(D[k] + c * se[k] < 0 for k in ks)
        hi[j] = J - sum(D[k] - c * se[k] > 0 for k in ks)
    return lo, hi


if __name__ == "__main__":
    import time

    t0 = time.time()
    rng = np.random.default_rng(12345)

    # --- closed forms -------------------------------------------------------
    assert np.array_equal(true_ranks(np.array([3.0, 1.0, 3.0, 2.0])), [1, 4, 1, 3])
    tbl = coverage_table(
        np.array([1, 2, 5, 9]), np.array([3, 4, 6, 12]), np.array([2, 5, 5, 20]),
        [(1, 3), (4, 10), (11, 30), (31, 40)],
    )
    assert tbl[0] == {"bin": (1, 3), "n_items": 1, "coverage": 1.0, "mean_width": 3.0}
    assert tbl[1]["n_items"] == 2 and tbl[1]["coverage"] == 0.5 and tbl[1]["mean_width"] == 2.5
    assert tbl[2]["coverage"] == 0.0 and tbl[2]["n_items"] == 1 and tbl[2]["mean_width"] == 4.0
    assert np.isnan(tbl[3]["coverage"])

    # --- brute-force agreement on a tiny problem ---------------------------
    J0, n0 = 30, 120
    w0 = 1.0 / np.arange(1, J0 + 1)
    X0 = _zipf_users(n0, 3, w0, rng)
    X0d = X0.toarray().astype(np.float64)
    seed0 = 777
    for sd in (False, True):
        lo_f, hi_f = mogstad_rank_cs(X0, 0.1, 40, np.random.default_rng(seed0), stepdown=sd)
        lo_n, hi_n = _naive_mogstad(X0d, 0.1, 40, np.random.default_rng(seed0), 1500, sd)
        assert np.array_equal(lo_f, lo_n) and np.array_equal(hi_f, hi_n), ("mogstad", sd)
    lo_s, hi_s = mogstad_rank_cs(X0, 0.1, 40, np.random.default_rng(seed0), stepdown=True)
    lo_c, hi_c = mogstad_rank_cs(X0, 0.1, 40, np.random.default_rng(seed0), stepdown=False)
    assert (lo_s >= lo_c).all() and (hi_s <= hi_c).all()  # step-down never widens sets
    # restricted comparison set and restricted items
    sub = np.array([0, 5, 29])
    lo_r, hi_r = mogstad_rank_cs(X0, 0.1, 40, np.random.default_rng(seed0), items=sub, max_compare=10)
    lo_nr, hi_nr = _naive_mogstad(X0d, 0.1, 40, np.random.default_rng(seed0), 10, False)
    assert np.array_equal(lo_r, lo_nr[sub]) and np.array_equal(hi_r, hi_nr[sub])
    # percentile intervals against brute force
    lo_p, hi_p = percentile_rank_intervals(X0, 0.1, 40, np.random.default_rng(seed0))
    Bm = _naive_boot_means(X0d, 40, np.random.default_rng(seed0))
    R = np.array([[1 + (Bm[b] > Bm[b, j]).sum() for j in range(J0)] for b in range(40)])
    assert np.array_equal(lo_p, np.percentile(R, 5, axis=0, method="lower"))
    assert np.array_equal(hi_p, np.percentile(R, 95, axis=0, method="higher"))
    lo_i, hi_i = percentile_rank_intervals(X0, 0.1, 40, np.random.default_rng(seed0), items=sub)
    assert np.array_equal(lo_i, lo_p[sub]) and np.array_equal(hi_i, hi_p[sub])
    # zero-count items are never compared: they all keep hi = J
    Xz = sp.csr_matrix(np.hstack([X0d, np.zeros((n0, 5))]).astype(np.float32))
    lo_z, hi_z = mogstad_rank_cs(Xz, 0.1, 40, np.random.default_rng(seed0), items=np.arange(J0, J0 + 5))
    assert (hi_z == J0 + 5).all() and (lo_z >= 1).all()

    # --- (a) Zipf(1.0), 300 items, 400 users, k = 5 ------------------------
    J, n_users, k = 300, 400, 5
    weights = 1.0 / np.arange(1, J + 1)
    Xa = _zipf_users(n_users, k, weights, np.random.default_rng(2024))
    theta_hat = np.asarray(Xa.sum(0)).ravel() / n_users
    sample_rank = true_ranks(theta_hat)
    for sd in (False, True):
        lo_a, hi_a = mogstad_rank_cs(Xa, 0.05, 200, np.random.default_rng(1), stepdown=sd)
        assert ((lo_a <= sample_rank) & (sample_rank <= hi_a)).all(), "sample rank outside CS"
        assert (lo_a >= 1).all() and (hi_a <= J).all() and (lo_a <= hi_a).all()

    # --- (b) coverage of the TRUE rank over 60 draws -----------------------
    k_pop = 20  # denser users than (a) so item counts are not dominated by zeros/ties
    pop = _zipf_users(60000, k_pop, weights, np.random.default_rng(99))
    theta = np.asarray(pop.sum(0)).ravel() / pop.shape[0]
    truth = true_ranks(theta)
    gap = np.array([np.min(np.abs(np.delete(theta, j) - theta[j])) for j in range(J)])
    results = {}
    n_draws, n_boot = 60, 60
    for nu in (400, 800):
        cov = {"naive": [], "mogstad": []}
        wid = {"naive": [], "mogstad": []}
        near = gap < 0.1 * np.sqrt(theta * (1 - theta) / nu)  # nearest neighbour within 0.1 standard error
        drng = np.random.default_rng(31337 + nu)
        for _ in range(n_draws):
            Xs = pop[drng.choice(pop.shape[0], size=nu, replace=False)]
            for name, fn in (("naive", percentile_rank_intervals), ("mogstad", mogstad_rank_cs)):
                lo_b, hi_b = fn(Xs, 0.05, n_boot, drng)
                ok = (lo_b <= truth) & (truth <= hi_b)
                cov[name].append((ok.mean(), ok[near].mean()))
                wid[name].append((hi_b - lo_b + 1).mean())
        for name in cov:
            c = np.mean(cov[name], axis=0)
            w = float(np.mean(wid[name]))
            results[(nu, name)] = (c, w)
            assert np.isfinite(c).all() and np.isfinite(w)
            print(
                f"n_users={nu:4d} {name:8s} coverage overall={c[0]:.3f} "
                f"near-tie={c[1]:.3f} (near-tie items={int(near.sum())}/{J}) mean width={w:.1f}"
            )
    for name in ("naive", "mogstad"):
        assert results[(800, name)][1] < results[(400, name)][1], f"{name} width did not shrink"

    assert time.time() - t0 < 20.0, "self-test too slow"
    print("OK rankcs")
