"""Differentially private estimators of an item-popularity vector from N clients.

Each client holds at most k items (contribution bounding by a uniform random
subset); the target is the count vector c = X_k.sum(0). We provide central
Laplace / analytic-Gaussian release of c and a local-DP (optimized unary
encoding with padding-and-sampling) estimator, plus closed-form noise
variances and helpers that compare privacy noise with sampling noise.
References: Wang, Blocki, Li, Jha (USENIX Security 2017); Qin et al. LDPMiner
(CCS 2016); Balle & Wang (ICML 2018); Zhu, Kairouz, McMahan, Sun, Li (AISTATS 2020).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr


def bound_contributions(X: sp.csr_matrix, k: int, rng: np.random.Generator) -> sp.csr_matrix:
    """Keep at most k randomly chosen items per client and drop the rest.

    Args:
        X: (n_users, n_items) binary csr matrix of interactions.
        k: maximum number of items a client may contribute (k >= 1).
        rng: random generator; each over-full client keeps a uniform random
            k-subset of its items, clients with <= k items are unchanged.

    Returns:
        Binary float32 csr matrix X_k with at most k non-zeros per row.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    Xc = sp.csr_matrix(X, copy=True)
    Xc.sum_duplicates()
    Xc.eliminate_zeros()
    Xc.sort_indices()
    n = Xc.shape[0]
    indptr, indices = Xc.indptr, Xc.indices
    nnz = indices.shape[0]
    rows = np.repeat(np.arange(n), np.diff(indptr))
    order = np.lexsort((rng.random(nnz), rows))  # within each row: random order
    rank = np.arange(nnz) - indptr[rows]  # position within row after the shuffle
    keep = np.zeros(nnz, dtype=bool)
    keep[order[rank < k]] = True
    new_counts = np.minimum(np.diff(indptr), k)
    new_indptr = np.concatenate(([0], np.cumsum(new_counts))).astype(indptr.dtype)
    out = sp.csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.float32), indices[keep], new_indptr),
        shape=Xc.shape,
    )
    return out


def central_laplace_counts(X_k: sp.csr_matrix, k: int, eps: float, rng: np.random.Generator) -> np.ndarray:
    """Release the item counts with Laplace noise so that the release is eps-DP (central model).

    Args:
        X_k: (n_users, n_items) binary csr matrix with at most k ones per row.
        k: per-client contribution bound.
        eps: privacy budget (> 0).
        rng: random generator.

    Returns:
        float64 vector c + Laplace(scale = k / eps), one entry per item.

    Under add/remove adjacency one client changes c by at most k in L1, so
    scale k/eps gives eps-DP. Under replace adjacency (swap one client's data
    for another's) the L1 sensitivity is 2k, i.e. the noise scale must double.
    """
    _check_eps(eps)
    c = np.asarray(X_k.sum(0), dtype=np.float64).ravel()
    return c + rng.laplace(0.0, k / eps, size=c.shape[0])


def analytic_gaussian_sigma(eps: float, delta: float, l2_sensitivity: float) -> float:
    """Smallest Gaussian noise sd that makes the mechanism (eps, delta)-DP, exactly (analytic Gaussian mechanism).

    Args:
        eps: privacy parameter (> 0).
        delta: failure probability in (0, 1).
        l2_sensitivity: L2 sensitivity D of the released vector.

    Returns:
        sigma solving Phi(D/(2s) - eps s/D) - e^eps Phi(-D/(2s) - eps s/D) = delta;
        the left side is strictly decreasing in s, so the root is the smallest
        valid sigma. Solved by Brent's method (Balle & Wang 2018, Algorithm 1
        solves the same equation by bisection on a monotone function).
    """
    _check_eps(eps)
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if l2_sensitivity <= 0:
        raise ValueError("l2_sensitivity must be > 0")

    def f(u: float) -> float:  # u = sigma / D; delta depends only on u
        return _gaussian_delta(u, eps) - delta

    lo, hi = 1e-6, 1.0
    while f(hi) > 0.0:
        hi *= 2.0
    u = brentq(f, lo, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=500)
    return float(u * l2_sensitivity)


def _gaussian_delta(u: float, eps: float) -> float:
    """delta(eps) of a Gaussian mechanism with sigma/D = u (Balle & Wang 2018, Thm 8)."""
    a = 1.0 / (2.0 * u)
    b = eps * u
    return float(ndtr(a - b) - np.exp(eps + log_ndtr(-a - b)))


def central_gaussian_counts(
    X_k: sp.csr_matrix, k: int, eps: float, delta: float, rng: np.random.Generator
) -> np.ndarray:
    """Release the item counts with Gaussian noise so that the release is (eps, delta)-DP (central model).

    Args:
        X_k: (n_users, n_items) binary csr matrix with at most k ones per row.
        k: per-client contribution bound.
        eps: privacy parameter (> 0).
        delta: failure probability in (0, 1).
        rng: random generator.

    Returns:
        float64 vector c + N(0, sigma^2), with sigma from `analytic_gaussian_sigma`
        for L2 sensitivity sqrt(k) (a client changes at most k counts by 1;
        add/remove adjacency; replace adjacency gives sqrt(2k)).
    """
    sigma = analytic_gaussian_sigma(eps, delta, float(np.sqrt(k)))
    c = np.asarray(X_k.sum(0), dtype=np.float64).ravel()
    return c + rng.normal(0.0, sigma, size=c.shape[0])


def ldp_oue_counts(X_k: sp.csr_matrix, k: int, eps: float, rng: np.random.Generator) -> np.ndarray:
    """Estimate the item counts from eps-locally-private reports (padding-and-sampling plus optimized unary encoding).

    Args:
        X_k: (n_users, n_items) binary csr matrix, canonical (as returned by
            `bound_contributions`), with at most k ones per row.
        k: padding length / contribution bound.
        eps: local privacy budget, spent entirely on the single report (> 0).
        rng: random generator.

    Returns:
        Unbiased float64 estimate of c = X_k.sum(0).

    Each client pads its item set to k slots with dummies, samples one slot
    uniformly (a real item with prob 1/k each) and reports it with OUE: the
    true bit is kept with p* = 1/2, every other bit is set with q* = 1/(e^eps+1)
    (a dummy sets only the q* noise bits). Let S_i be the number of clients whose
    sampled slot is item i, so E[S_i] = c_i / k. The number of reports with bit i
    set is T_i = Bin(S_i, p*) + Bin(N - S_i, q*), hence
    c_hat_i = k (T_i - N q*) / (p* - q*) is unbiased. The simulation is exact:
    the sampled slot is drawn per client, then the bit counts are drawn from
    the two binomials per item; O(N + J) time.
    """
    _check_eps(eps)
    n, j = X_k.shape
    indptr, indices = X_k.indptr, X_k.indices
    lengths = np.diff(indptr)
    if lengths.size and lengths.max() > k:
        raise ValueError("a client holds more than k items; call bound_contributions first")
    slot = rng.integers(0, k, size=n)
    real = slot < lengths  # slots >= number of held items are dummy slots
    chosen = indices[indptr[:-1][real] + slot[real]]
    s = np.bincount(chosen, minlength=j)
    p, q = 0.5, 1.0 / (np.exp(eps) + 1.0)
    t = rng.binomial(s, p) + rng.binomial(n - s, q)
    return k * (t - n * q) / (p - q)


def ldp_oue_noise_variance(N: int, k: int, eps: float, c: np.ndarray | float) -> np.ndarray | float:
    """Variance of the local-DP count estimate of an item with true count c (conditional on the data).

    Args:
        N: number of clients.
        k: padding length / contribution bound.
        eps: local privacy budget.
        c: true bounded count(s) c_i = sum_u X_k[u, i] (scalar or array).

    Returns:
        Var(c_hat_i) = k^2 [ Var(S_i) + (E[S_i] p(1-p) + (N - E[S_i]) q(1-q)) / (p-q)^2 ]
        with p = 1/2, q = 1/(e^eps+1), E[S_i] = c_i/k and
        Var(S_i) = (c_i/k)(1 - 1/k) summed over clients (total variance with
        respect to the slot sampling and the OUE randomness). The second term is
        the pure-protocol variance of Wang et al. (2017), scaled by k^2; the
        first, c_i (k-1), is the sampling term.
    """
    _check_eps(eps)
    c = np.asarray(c, dtype=np.float64)
    p, q = 0.5, 1.0 / (np.exp(eps) + 1.0)
    es = c / k
    sampling = es * (1.0 - 1.0 / k)
    protocol = (es * p * (1 - p) + (N - es) * q * (1 - q)) / (p - q) ** 2
    out = k * k * (sampling + protocol)
    return float(out) if out.ndim == 0 else out


def noise_to_sampling_ratio(noise_var: np.ndarray | float, p: np.ndarray | float, N: np.ndarray | float) -> np.ndarray | float:
    """How many times larger the privacy-noise variance is than the sampling variance of a popularity estimate.

    Args:
        noise_var: variance of the privacy noise on the popularity *fraction*
            estimate (count-noise variance divided by N^2).
        p: fraction of clients that hold the item.
        N: number of clients.

    Returns:
        noise_var / (p (1-p) / N); values > 1 mean privacy noise dominates the
        ordinary binomial sampling error of a sample of N clients.
    """
    p = np.asarray(p, dtype=np.float64)
    out = np.asarray(noise_var, dtype=np.float64) * N / (p * (1.0 - p))
    return float(out) if out.ndim == 0 else out


def central_crossover_N(sigma: float, p_boundary: float) -> float:
    """Number of clients at which central count noise of sd sigma equals the sampling sd for an item held by fraction p_boundary.

    Args:
        sigma: standard deviation of the noise added to each count.
        p_boundary: item popularity fraction p (e.g. at the top-m boundary).

    Returns:
        sigma^2 / (p (1-p)): the sampling variance p(1-p)/N equals the noise
        variance sigma^2/N^2 on the fraction at this N; beyond it sampling noise
        dominates.
    """
    return float(sigma) ** 2 / (p_boundary * (1.0 - p_boundary))


def topm_recovery(est: np.ndarray, truth: np.ndarray, m: int, rng: np.random.Generator) -> float:
    """Fraction of the true top-m items that also appear in the estimated top-m (ties broken at random).

    Args:
        est: estimated score per item (higher = better).
        truth: true score per item.
        m: size of the top set (1 <= m <= n_items).
        rng: random generator used to break ties in both rankings.

    Returns:
        |top_m(est) & top_m(truth)| / m in [0, 1].
    """
    est = np.asarray(est)
    truth = np.asarray(truth)
    perm = rng.permutation(est.shape[0])  # random tie-break: stable sort of a shuffled array

    def top(score: np.ndarray) -> np.ndarray:
        return perm[np.argsort(-score[perm], kind="stable")[:m]]

    return float(np.intersect1d(top(est), top(truth)).size / m)


def _check_eps(eps: float) -> None:
    if not eps > 0:
        raise ValueError("eps must be > 0")


def _zipf_matrix(n: int, j: int, draws: int, rng: np.random.Generator) -> sp.csr_matrix:
    """Synthetic binary interactions: each user makes `draws` Zipf(1) item draws (duplicates collapse)."""
    w = 1.0 / np.arange(1, j + 1)
    items = rng.choice(j, size=(n, draws), p=w / w.sum())
    rows = np.repeat(np.arange(n), draws)
    X = sp.csr_matrix((np.ones(n * draws, dtype=np.float32), (rows, items.ravel())), shape=(n, j))
    X.data[:] = 1.0
    return X


if __name__ == "__main__":
    rng = np.random.default_rng(20240607)

    # ---- bound_contributions: invariants and uniformity ----
    N, J, k = 3000, 400, 5
    X = _zipf_matrix(N, J, 9, rng)
    Xk = bound_contributions(X, k, rng)
    rl, rk = np.diff(X.indptr), np.diff(Xk.indptr)
    assert (rk == np.minimum(rl, k)).all() and Xk.dtype == np.float32
    assert (X.multiply(Xk) != Xk).nnz == 0, "X_k must be a subset of X"
    assert rl.max() > k, "test data should need bounding"
    row = np.ones((20000, 10), dtype=np.float32)
    fr = np.asarray(bound_contributions(sp.csr_matrix(row), 3, rng).mean(0)).ravel()
    assert np.abs(fr - 0.3).max() < 0.02, fr  # each item kept with prob k/m = 0.3

    c = np.asarray(Xk.sum(0), dtype=np.float64).ravel()
    top20 = np.argsort(-c)[:20]
    reps = 200

    # ---- central Laplace / Gaussian noise scales ----
    eps = 1.0
    lap = np.concatenate([central_laplace_counts(Xk, k, eps, rng) - c for _ in range(reps)])
    assert abs(lap.std() / (np.sqrt(2) * k / eps) - 1) < 0.03, lap.std()
    delta = 1e-5
    sig = analytic_gaussian_sigma(eps, delta, np.sqrt(k))
    gau = np.concatenate([central_gaussian_counts(Xk, k, eps, delta, rng) - c for _ in range(reps)])
    assert abs(gau.std() / sig - 1) < 0.03, (gau.std(), sig)

    # ---- analytic Gaussian: own equation to 1e-9 and tighter than classical for eps < 1 ----
    for e in (0.05, 0.1, 0.5, 0.9, 1.0, 2.0, 5.0):
        for d in (1e-3, 1e-5, 1e-8):
            for D in (1.0, np.sqrt(5.0)):
                s = analytic_gaussian_sigma(e, d, D)
                assert abs(_gaussian_delta(s / D, e) - d) < 1e-9, (e, d, D)
                if e < 1:
                    assert s < np.sqrt(2 * np.log(1.25 / d)) * D / e, (e, d, D)
    # sanity: eps=0 limit of the delta equation is 2 Phi(D/2s) - 1, so delta -> sigma = D/(2 Phi^-1((1+delta)/2))
    from scipy.special import ndtri

    s_small_eps = analytic_gaussian_sigma(1e-12, 1e-3, 1.0)
    assert abs(s_small_eps / (1.0 / (2 * ndtri((1 + 1e-3) / 2))) - 1) < 1e-8

    # ---- LDP OUE: unbiasedness and variance against the closed form ----
    eps = 2.0
    est = np.stack([ldp_oue_counts(Xk, k, eps, rng) for _ in range(reps)])
    mean, var = est.mean(0), est.var(0, ddof=1)
    se = np.sqrt(var / reps)
    assert (np.abs(mean - c)[top20] <= 3 * se[top20]).all(), (mean - c)[top20] / se[top20]
    theo = ldp_oue_noise_variance(N, k, eps, c)
    pooled = var.sum() / theo.sum()
    assert abs(pooled - 1) < 0.10, pooled
    assert abs(var[top20].sum() / theo[top20].sum() - 1) < 0.10

    # ---- brute force: per-client simulation of padding-and-sampling + OUE bit vectors ----
    n2, j2, k2, e2, r2 = 300, 20, 3, 1.0, 600
    X2 = bound_contributions(_zipf_matrix(n2, j2, 5, rng), k2, rng)
    c2 = np.asarray(X2.sum(0)).ravel().astype(np.float64)
    dense = X2.toarray()
    p_, q_ = 0.5, 1.0 / (np.exp(e2) + 1.0)
    naive = np.empty((r2, j2))
    for r in range(r2):
        slot = rng.integers(0, k2, size=n2)
        item = np.full(n2, -1)  # -1 = dummy
        for u in range(n2):
            held = np.flatnonzero(dense[u])
            if slot[u] < held.size:
                item[u] = held[slot[u]]
        bits = rng.random((n2, j2)) < q_
        hit = item >= 0
        bits[np.flatnonzero(hit), item[hit]] = rng.random(hit.sum()) < p_
        naive[r] = k2 * (bits.sum(0) - n2 * q_) / (p_ - q_)
    th2 = ldp_oue_noise_variance(n2, k2, e2, c2)
    assert abs(naive.var(0, ddof=1).sum() / th2.sum() - 1) < 0.10
    assert (np.abs(naive.mean(0) - c2) <= 4 * naive.std(0, ddof=1) / np.sqrt(r2)).all()
    fast = np.stack([ldp_oue_counts(X2, k2, e2, rng) for _ in range(r2)])
    assert abs(fast.var(0, ddof=1).sum() / naive.var(0, ddof=1).sum() - 1) < 0.15

    # ---- ratio / crossover helpers ----
    pb = 0.02
    n_star = central_crossover_N(sig, pb)
    assert abs(n_star - sig**2 / (pb * (1 - pb))) < 1e-12
    assert abs(noise_to_sampling_ratio(sig**2 / n_star**2, pb, n_star) - 1) < 1e-12
    assert noise_to_sampling_ratio(sig**2 / 100.0**2, pb, 100) > 1 > noise_to_sampling_ratio(sig**2 / 1e6**2, pb, 1e6)

    # ---- topm_recovery ----
    truth = np.arange(J, 0, -1).astype(float)
    assert topm_recovery(truth, truth, 10, rng) == 1.0
    assert topm_recovery(-truth, truth, 10, rng) == 0.0
    flat = np.mean([topm_recovery(np.zeros(J), truth, 40, rng) for _ in range(300)])
    assert abs(flat - 40 / J) < 0.02, flat
    print("OK dp")
