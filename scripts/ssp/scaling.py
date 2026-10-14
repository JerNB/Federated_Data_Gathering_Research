"""Learning-curve fitting and theory predictions for how list error shrinks with sample size N.

Fits power-law (E = a N^-beta + c) and exponential learning curves with bootstrap
confidence intervals, measures Zipf slopes of count vectors, evaluates the
Dyer-Owen Zipf-Poisson ordering-depth and top-m resolution bounds, and provides
Dirichlet-multinomial shrinkage tools (kappa estimation, shrunken scores, crossover N).
References: Dyer & Owen, "Correct ordering in the Zipf-Poisson ensemble" (arXiv 1101.2481);
Hutter, "Learning Curve Theory" (arXiv 2102.04074); Minka, "Estimating a Dirichlet distribution".
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import gammaln

_BETA_STARTS = (0.05, 0.15, 0.3, 0.6, 1.0, 2.0)
_LOG_BETA_BOUNDS = (math.log(1e-3), math.log(10.0))
_LOG_A_BOUNDS = (-80.0, 80.0)


# --------------------------------------------------------------------------
# Learning-curve fits
# --------------------------------------------------------------------------
def _prep(N, E, w, positive_n: bool):
    """Return finite (N, E, w) float arrays, or None if fewer than 4 usable points."""
    N = np.asarray(N, dtype=np.float64).ravel()
    E = np.asarray(E, dtype=np.float64).ravel()
    if N.shape != E.shape:
        return None
    w = np.ones_like(N) if w is None else np.asarray(w, dtype=np.float64).ravel()
    if w.shape != N.shape:
        return None
    ok = np.isfinite(N) & np.isfinite(E) & np.isfinite(w) & (w > 0)
    if positive_n:
        ok &= N > 0
    N, E, w = N[ok], E[ok], w[ok]
    if N.size < 4 or np.unique(N).size < 4:
        return None
    order = np.argsort(N, kind="stable")
    return N[order], E[order], w[order]


def _failed(keys: tuple[str, ...]) -> dict:
    out = {k: float("nan") for k in keys}
    out["converged"] = False
    return out


def _aic(rss: float, n: int, k: int) -> float:
    """Gaussian AIC n ln(rss/n) + 2k (rss floored to avoid log 0)."""
    return float(n * math.log(max(rss / n, 1e-300)) + 2 * k)


def _power_fit_core(N, E, w, floor, beta_starts):
    """Best least-squares power-law fit over several starts; returns (params, rss) or None.

    Works on N/N[0] and E/max|E| for conditioning; params = (a, beta, c) in original units.
    """
    n0 = N[0]
    x = N / n0
    s = float(np.max(np.abs(E)))
    if not s > 0:
        s = 1.0
    y = E / s
    sw = np.sqrt(w)
    lo = [_LOG_A_BOUNDS[0], _LOG_BETA_BOUNDS[0]] + ([0.0] if floor else [])
    hi = [_LOG_A_BOUNDS[1], _LOG_BETA_BOUNDS[1]] + ([np.inf] if floor else [])

    def resid(p):
        c = p[2] if floor else 0.0
        return sw * (np.exp(p[0] - np.exp(p[1]) * np.log(x)) + c - y)

    c0 = 0.5 * float(np.min(y)) if floor else 0.0
    a0 = max(float(y[0]) - c0, 1e-12)
    best = None
    for b0 in beta_starts:
        p0 = [math.log(a0), math.log(b0)] + ([c0] if floor else [])
        p0 = np.clip(p0, np.array(lo) + 1e-9, np.minimum(np.array(hi), 1e300) - 1e-9)
        try:
            r = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac", max_nfev=400)
        except (ValueError, FloatingPointError):
            continue
        if not np.all(np.isfinite(r.x)) or not r.success:
            continue
        rss = float(np.sum(r.fun**2))
        if best is None or rss < best[1]:
            best = (r.x, rss)
    if best is None:
        return None
    p, rss = best
    beta = math.exp(p[1])
    at_bound = abs(p[1] - _LOG_BETA_BOUNDS[0]) < 1e-6 or abs(p[1] - _LOG_BETA_BOUNDS[1]) < 1e-6
    a = math.exp(p[0]) * s * n0**beta  # a' (N/n0)^-beta = a' n0^beta N^-beta
    c = float(p[2]) * s if floor else 0.0
    return (a, beta, c), rss * s * s, at_bound


def fit_power_law(N, E, w=None, floor: bool = True) -> dict:
    """Fit a power-law learning curve E = a N^(-beta) + c to error-versus-sample-size data.

    Args:
        N: sample sizes (positive), 1-D.
        E: error at each N, 1-D, same length.
        w: optional non-negative weights on squared residuals (default all 1).
        floor: if True the asymptotic floor c >= 0 is fitted; if False c is fixed at 0.

    Returns:
        dict {a, beta, c, rss, aic, converged}. rss is the weighted residual sum of squares,
        aic = n ln(rss/n) + 2k with k = 3 (2 without floor). Fewer than 4 usable points, a
        failed optimisation, or beta pinned at its bounds [1e-3, 10] gives converged False
        (numeric fields nan) instead of raising.

    Method: scipy least_squares in (log a, log beta, c) with c >= 0, multi-start over
    beta in [0.05, 2]; the lowest-RSS start wins. See Hutter (arXiv 2102.04074).
    """
    keys = ("a", "beta", "c", "rss", "aic")
    d = _prep(N, E, w, positive_n=True)
    if d is None:
        return _failed(keys)
    N, E, w = d
    out = _power_fit_core(N, E, w, floor, _BETA_STARTS)
    if out is None:
        return _failed(keys)
    (a, beta, c), rss, at_bound = out
    return {
        "a": a, "beta": beta, "c": c, "rss": rss,
        "aic": _aic(rss, N.size, 3 if floor else 2),
        "converged": bool(np.isfinite(rss) and not at_bound),
    }


def fit_power_law_boot(N, E_reps, n_boot: int = 300, alpha: float = 0.05,
                       rng: np.random.Generator | None = None, floor: bool = True) -> dict:
    """Estimate the power-law exponent beta with a percentile CI from bootstrapping replicate curves.

    Args:
        N: sample sizes, length P.
        E_reps: array (R replicates, P) of error curves.
        n_boot: number of bootstrap resamples of the replicate rows.
        alpha: two-sided level; the CI is the [alpha/2, 1-alpha/2] percentile interval.
        rng: numpy Generator (None -> default_rng(0), so results stay deterministic).
        floor: passed to the fit (c >= 0 fitted if True).

    Returns:
        dict {beta, ci_lo, ci_hi, a, c, frac_nonconverged, n_boot, converged}. beta/a/c come
        from the fit to the mean of all replicates; the CI uses the converged bootstrap fits
        only (nan if none). frac_nonconverged is over the n_boot refits. converged is the
        status of the point fit.

    Each refit averages the resampled rows and fits with the point-estimate beta plus three
    fixed starts (0.15, 0.6, 1.5) rather than the full multi-start grid, for speed.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    N = np.asarray(N, dtype=np.float64).ravel()
    E_reps = np.asarray(E_reps, dtype=np.float64)
    nan = float("nan")
    base = {"beta": nan, "ci_lo": nan, "ci_hi": nan, "a": nan, "c": nan,
            "frac_nonconverged": 1.0, "n_boot": int(n_boot), "converged": False}
    if E_reps.ndim != 2 or E_reps.shape[1] != N.size or E_reps.shape[0] < 1:
        return base
    R = E_reps.shape[0]
    point = fit_power_law(N, np.nanmean(E_reps, axis=0), floor=floor)
    base.update(beta=point["beta"], a=point["a"], c=point["c"], converged=point["converged"])
    if not point["converged"]:
        return base
    starts = (point["beta"], 0.15, 0.6, 1.5)
    betas = []
    bad = 0
    for _ in range(n_boot):
        idx = rng.integers(0, R, size=R)
        d = _prep(N, np.nanmean(E_reps[idx], axis=0), None, positive_n=True)
        out = None if d is None else _power_fit_core(d[0], d[1], d[2], floor, starts)
        if out is None or out[2] or not np.isfinite(out[1]):
            bad += 1
        else:
            betas.append(out[0][1])
    base["frac_nonconverged"] = bad / n_boot if n_boot else 1.0
    if betas:
        base["ci_lo"], base["ci_hi"] = (float(v) for v in np.percentile(betas, [50 * alpha, 100 - 50 * alpha]))
    return base


def fit_exponential(N, E, w=None) -> dict:
    """Fit an exponential learning curve E = a exp(-N/N0) + c, c >= 0, for model comparison.

    Args:
        N: sample sizes, 1-D (distinct values, at least 4 usable points).
        E: error at each N.
        w: optional non-negative weights on squared residuals.

    Returns:
        dict {a, N0, c, rss, aic, converged} with aic = n ln(rss/n) + 6 (k = 3); the same
        failure convention as fit_power_law (converged False, nan fields, no raise).

    Method: least_squares in (log a, log N0, c) with N0 within [1e-3, 1e3] x the N span and
    several N0 starts; lowest RSS wins.
    """
    keys = ("a", "N0", "c", "rss", "aic")
    d = _prep(N, E, w, positive_n=False)
    if d is None:
        return _failed(keys)
    N, E, w = d
    span = float(N[-1] - N[0])
    s = float(np.max(np.abs(E))) or 1.0
    y = E / s
    sw = np.sqrt(w)
    lo = [-np.inf, math.log(1e-3 * span), 0.0]
    hi = [np.inf, math.log(1e3 * span), np.inf]

    def resid(p):
        return sw * (np.exp(p[0] - (N - N[0]) / math.exp(p[1])) + p[2] - y)

    c0 = 0.5 * float(np.min(y))
    a0 = max(float(y[0]) - c0, 1e-12)
    best = None
    for f in (0.05, 0.15, 0.4, 1.0, 3.0):
        p0 = np.array([math.log(a0), math.log(f * span), c0])
        p0[1] = np.clip(p0[1], lo[1] + 1e-9, hi[1] - 1e-9)
        try:
            r = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac", max_nfev=400)
        except (ValueError, FloatingPointError):
            continue
        if not np.all(np.isfinite(r.x)) or not r.success:
            continue
        rss = float(np.sum(r.fun**2))
        if best is None or rss < best[1]:
            best = (r.x, rss)
    if best is None:
        return _failed(keys)
    p, rss = best
    n0 = math.exp(p[1])
    at_bound = abs(p[1] - lo[1]) < 1e-6 or abs(p[1] - hi[1]) < 1e-6
    rss *= s * s
    return {
        "a": math.exp(p[0] + N[0] / n0) * s, "N0": n0, "c": float(p[2]) * s,
        "rss": rss, "aic": _aic(rss, N.size, 3),
        "converged": bool(np.isfinite(rss) and not at_bound),
    }


def compare_power_vs_exp(N, E) -> dict:
    """Decide whether a power law or an exponential describes a learning curve better (lower AIC).

    Args:
        N: sample sizes, 1-D.
        E: error at each N.

    Returns:
        dict {aic_power, aic_exp, delta_aic, winner, power, exp}: both models have a floor
        (3 parameters); delta_aic = aic_exp - aic_power (positive favours the power law);
        winner is "power", "exp", or None when neither fit converged (if only one converged
        it wins by default); power/exp hold the full fit dicts.
    """
    p = fit_power_law(N, E, floor=True)
    e = fit_exponential(N, E)
    ap = p["aic"] if p["converged"] else float("nan")
    ae = e["aic"] if e["converged"] else float("nan")
    if p["converged"] and e["converged"]:
        winner = "power" if ap <= ae else "exp"
    elif p["converged"]:
        winner = "power"
    elif e["converged"]:
        winner = "exp"
    else:
        winner = None
    return {"aic_power": ap, "aic_exp": ae, "delta_aic": ae - ap, "winner": winner,
            "power": p, "exp": e}


def local_slope(N, E) -> np.ndarray:
    """Return the local log-log slope d log E / d log N between each pair of consecutive points.

    Args:
        N: sample sizes (positive), length P.
        E: errors, length P.

    Returns:
        float array of length P-1; nan where either neighbouring E <= 0 (or N <= 0 / non-finite).
        A pure power law E ~ N^-beta gives -beta everywhere.
    """
    N = np.asarray(N, dtype=np.float64).ravel()
    E = np.asarray(E, dtype=np.float64).ravel()
    if N.size < 2:
        return np.empty(0)
    ok = (E > 0) & (N > 0) & np.isfinite(E) & np.isfinite(N)
    with np.errstate(divide="ignore", invalid="ignore"):
        lN = np.where(ok, np.log(np.where(ok, N, 1.0)), np.nan)
        lE = np.where(ok, np.log(np.where(ok, E, 1.0)), np.nan)
        return np.diff(lE) / np.diff(lN)


# --------------------------------------------------------------------------
# Zipf slope of a count vector
# --------------------------------------------------------------------------
def zipf_slope(counts, r_lo: int = 10, r_hi: int = 1000) -> dict:
    """Measure how steeply item popularity falls with rank, as a log-log slope over a rank window.

    Args:
        counts: 1-D item counts (any order; sorted descending internally).
        r_lo, r_hi: inclusive 1-based rank window (clipped to the number of positive counts).

    Returns:
        dict {s, intercept, r2}: weighted least squares of log(rate) on log(rank) with
        rate = count / sum(counts) and weight = count (large cells count more). s is the
        slope, NEGATIVE for decaying popularity (Zipf exponent = -s). r2 is the weighted R^2.
        All nan if fewer than 2 ranks fall in the window.
    """
    c = np.sort(np.asarray(counts, dtype=np.float64).ravel())[::-1]
    total = float(c.sum())
    c = c[c > 0]
    lo = max(int(r_lo), 1)
    hi = min(int(r_hi), c.size)
    nan = {"s": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    if hi - lo + 1 < 2 or total <= 0:
        return nan
    ranks = np.arange(lo, hi + 1, dtype=np.float64)
    cw = c[lo - 1:hi]
    x = np.log(ranks)
    y = np.log(cw / total)
    sw = cw.sum()
    xm = (cw * x).sum() / sw
    ym = (cw * y).sum() / sw
    sxx = (cw * (x - xm) ** 2).sum()
    if sxx <= 0:
        return nan
    syy = (cw * (y - ym) ** 2).sum()
    sxy = (cw * (x - xm) * (y - ym)).sum()
    s = sxy / sxx
    r2 = 1.0 if syy <= 0 else float(sxy * sxy / (sxx * syy))
    return {"s": float(s), "intercept": float(ym - s * xm), "r2": r2}


def zipf_slope_local(counts, ranks=(10, 100, 1000)) -> dict:
    """Measure the log-log popularity slope in a window around each of several ranks.

    Args:
        counts: 1-D item counts.
        ranks: centre ranks r; each uses the window [ceil(r/2), 2r] in zipf_slope.

    Returns:
        dict {r: slope} (nan where the window has fewer than 2 positive-count ranks).
    """
    return {int(r): zipf_slope(counts, max(1, math.ceil(r / 2)), 2 * int(r))["s"] for r in ranks}


# --------------------------------------------------------------------------
# Dyer & Owen ordering bounds
# --------------------------------------------------------------------------
def dyer_owen_depth(lam, thr: float = 0.01) -> int:
    """Count how many top-ranked items are expected to be correctly ordered (Dyer-Owen depth).

    Args:
        lam: expected counts lam_i = N * (item inclusion probability); sorted descending here.
        thr: tolerance on the adjacent-swap sum.

    Returns:
        the largest n with p(n) <= thr, where (Dyer & Owen 2011, arXiv 1101.2481, eq. 16)
        p(n) = sum_{i<n} exp(-(sqrt(lam_i) - sqrt(lam_{i+1}))^2). p(1) = 0 so the result is
        at least 1 for non-empty input; at most len(lam).
    """
    lam = np.sort(np.asarray(lam, dtype=np.float64).ravel())[::-1]
    if lam.size == 0:
        return 0
    if lam.size == 1:
        return 1
    root = np.sqrt(np.maximum(lam, 0.0))
    p = np.cumsum(np.exp(-np.diff(root) ** 2))  # p[n-2] = p(n); terms > 0 so p is increasing
    return int(1 + np.searchsorted(p, thr, side="right"))


def _topm_bound_sorted(root: np.ndarray, m: int, cutoff: float, block: int = 2_000_000) -> float:
    """Windowed Bonferroni sum for sorted-descending sqrt(lam); see dyer_owen_topm_set_bound."""
    J = root.size
    if m <= 0 or m >= J:
        return 0.0
    # j window: end of j with (root[m-1]-root[j])^2 <= cutoff (widest for i = m, root decreasing)
    neg = -root
    j1 = int(np.searchsorted(neg, -(root[m - 1] - math.sqrt(cutoff)), side="right"))
    j1 = max(j1, m + 1)  # at least j = m+1 (0-based index m)
    # i window: start of i with (root[i]-root[m])^2 <= cutoff (widest for j = m+1)
    i0 = int(np.searchsorted(neg, -(root[m] + math.sqrt(cutoff)), side="left"))
    i0 = min(i0, m - 1)
    jr = root[m:j1]
    total = 0.0
    step = max(1, block // max(jr.size, 1))
    for a in range(i0, m, step):
        gap = root[a:min(a + step, m)][:, None] - jr[None, :]
        total += float(np.exp(-gap * gap).sum())
    return min(1.0, total)


def dyer_owen_topm_set_bound(lam, m: int, cutoff: float = 50.0) -> float:
    """Bound the probability that the observed top-m SET of items differs from the true top-m set.

    Args:
        lam: expected counts lam_i = N * (inclusion probability); sorted descending here.
        m: size of the top set.
        cutoff: pairs with (sqrt(lam_i) - sqrt(lam_j))^2 > cutoff (term < exp(-cutoff) ~ 2e-22)
            are dropped. Because terms decay as j grows and as i falls, only the window
            j in (m, j1] and i in [i0, m] is evaluated, which makes the cost
            O((m_eff) * window) rather than O(m * J); the truncation error is at most
            m * J * exp(-cutoff), negligible for any realistic J.

    Returns:
        Bonferroni bound min(1, sum_{i<=m<j} exp(-(sqrt(lam_i) - sqrt(lam_j))^2); 0 if m >= J.
        Each term bounds P(item j's Poisson count >= item i's) (Dyer & Owen 2011, arXiv 1101.2481).
    """
    lam = np.sort(np.asarray(lam, dtype=np.float64).ravel())[::-1]
    return _topm_bound_sorted(np.sqrt(np.maximum(lam, 0.0)), int(m), float(cutoff))


def n_resolve_topm(q, m: int, delta: float = 0.05) -> float:
    """Find the smallest sample size N at which the top-m set is resolved with bound <= delta.

    Args:
        q: item inclusion probabilities (any order; lam = N q).
        m: size of the top set.
        delta: target value of dyer_owen_topm_set_bound.

    Returns:
        smallest N (bisection in log N to 1e-4 relative tolerance; the bound is monotone
        decreasing in N). 0.0 if m >= number of items (nothing to resolve); inf if q_m = q_(m+1)
        (tie) or the bound is still above delta at N = 1e18.
    """
    q = np.sort(np.asarray(q, dtype=np.float64).ravel())[::-1]
    m = int(m)
    if m <= 0 or m >= q.size:
        return 0.0
    if q[m - 1] <= q[m]:
        return float("inf")
    rq = np.sqrt(np.maximum(q, 0.0))

    def bound(n: float) -> bool:
        return _topm_bound_sorted(rq * math.sqrt(n), m, 50.0) <= delta

    lo, hi = 1e-9, 1.0
    if bound(lo):
        return lo
    while not bound(hi):
        lo, hi = hi, hi * 2.0
        if hi > 1e18:
            return float("inf")
    llo, lhi = math.log(lo), math.log(hi)
    while lhi - llo > 1e-4:
        mid = 0.5 * (llo + lhi)
        if bound(math.exp(mid)):
            lhi = mid
        else:
            llo = mid
    return math.exp(lhi)


# --------------------------------------------------------------------------
# Dirichlet-multinomial shrinkage
# --------------------------------------------------------------------------
def _dm_prep(counts, prior_pi, n):
    c = np.asarray(counts, dtype=np.float64).ravel()
    pi = np.asarray(prior_pi, dtype=np.float64).ravel()
    if c.shape != pi.shape:
        raise ValueError("counts and prior_pi must have the same length")
    nz = c > 0
    if np.any(pi[nz] <= 0):
        raise ValueError("prior_pi must be positive wherever counts are positive")
    return c[nz], pi[nz], float(c.sum() if n is None else n)


def _maximise_kappa(terms) -> float:
    """Maximise sum of Dirichlet-multinomial log-likelihoods over log kappa in [1e-3, 1e8].

    terms: list of (c_nz, pi_nz, n). A coarse grid guards against local optima, then
    bounded scalar minimisation refines around the best grid cell.
    """
    def nll(lk: float) -> float:
        k = math.exp(lk)
        tot = 0.0
        for c, pi, n in terms:
            kp = k * pi
            tot += gammaln(k) - gammaln(n + k) + float(np.sum(gammaln(c + kp) - gammaln(kp)))
        return -tot

    lo, hi = math.log(1e-3), math.log(1e8)
    grid = np.linspace(lo, hi, 41)
    vals = [nll(g) for g in grid]
    j = int(np.argmin(vals))
    res = minimize_scalar(nll, bounds=(grid[max(j - 1, 0)], grid[min(j + 1, grid.size - 1)]),
                          method="bounded", options={"xatol": 1e-6})
    best = res.x if res.fun <= vals[j] else grid[j]
    return float(math.exp(best))


def kappa_hat(sample_counts, prior_pi, n: float | None = None) -> float:
    """Estimate how strongly a sample's item counts concentrate around a prior popularity (Dirichlet-multinomial kappa).

    Args:
        sample_counts: 1-D item counts c_i of one sample.
        prior_pi: 1-D prior item probabilities pi_i (positive wherever c_i > 0).
        n: total count (default sum(c)).

    Returns:
        maximum-likelihood kappa in [1e-3, 1e8] (large kappa = sample close to the prior;
        a result at the upper bound means no over-dispersion is detectable). Maximises
        l(kappa) = gammaln(kappa) - gammaln(n + kappa)
        + sum_{i: c_i > 0} [gammaln(c_i + kappa pi_i) - gammaln(kappa pi_i)]
        over log kappa (Minka, "Estimating a Dirichlet distribution"). A single sample gives
        a noisy estimate; pool samples with pooled_kappa_hat.
    """
    return _maximise_kappa([_dm_prep(sample_counts, prior_pi, n)])


def pooled_kappa_hat(list_of_count_vectors, prior_pi) -> float:
    """Estimate one shared Dirichlet-multinomial kappa from many samples by pooling their likelihoods.

    Args:
        list_of_count_vectors: iterable of 1-D count vectors (same length as prior_pi).
        prior_pi: shared 1-D prior item probabilities.

    Returns:
        the kappa in [1e-3, 1e8] maximising the sum of the per-sample log-likelihoods
        defined in kappa_hat.
    """
    terms = [_dm_prep(c, prior_pi, None) for c in list_of_count_vectors]
    if not terms:
        raise ValueError("need at least one count vector")
    return _maximise_kappa(terms)


def shrink_scores(sample_counts, prior_pi, kappa: float) -> np.ndarray:
    """Blend sample counts with a prior popularity to get posterior-mean item scores.

    Args:
        sample_counts: 1-D item counts c.
        prior_pi: 1-D prior item probabilities pi (same length).
        kappa: prior strength (pseudo-count total).

    Returns:
        float64 array (c + kappa pi) / (n + kappa), n = sum(c): kappa -> 0 gives the raw
        sample shares, kappa -> inf gives the prior.
    """
    c = np.asarray(sample_counts, dtype=np.float64).ravel()
    pi = np.asarray(prior_pi, dtype=np.float64).ravel()
    return (c + kappa * pi) / (c.sum() + kappa)


def crossover_from_kappa(kappa: float, k: float, rho: float) -> float:
    """Approximate the number of users N* at which the sample starts to outweigh the prior.

    Args:
        kappa: Dirichlet-multinomial concentration (prior pseudo-counts, in interactions).
        k: interactions per user.
        rho: within-user intraclass correlation of item choices (0 = independent).

    Returns:
        N* ~ kappa (1 + (k - 1) rho) / k. This is an approximation: kappa interactions are
        converted to users through the Kish design effect 1 + (k - 1) rho, so one user is
        worth k / (1 + (k - 1) rho) effective independent interactions.
    """
    return float(kappa * (1.0 + (k - 1.0) * rho) / k)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(12345)

    # --- power-law recovery with 3% multiplicative noise
    a_t, b_t, c_t = 5.0, 0.5, 0.05
    N = np.geomspace(1e2, 1e5, 10)
    clean = a_t * N**-b_t + c_t
    est = []
    for _ in range(10):
        E = clean * (1 + 0.03 * rng.standard_normal(N.size))
        f = fit_power_law(N, E)
        assert f["converged"], f
        est.append((f["a"], f["beta"], f["c"]))
    a_e, b_e, c_e = np.median(est, axis=0)
    assert abs(b_e - b_t) < 0.05, (a_e, b_e, c_e)
    assert abs(a_e / a_t - 1) < 0.25 and abs(c_e - c_t) < 0.015, (a_e, b_e, c_e)
    f = fit_power_law(N, clean)
    assert abs(f["beta"] - b_t) < 1e-4 and abs(f["c"] - c_t) < 1e-6 and abs(f["a"] - a_t) < 1e-3, f
    assert not fit_power_law(N[:3], clean[:3])["converged"]
    assert not fit_power_law([], [])["converged"]
    f0 = fit_power_law(N, 3.0 * N**-0.7, floor=False)
    assert abs(f0["beta"] - 0.7) < 1e-5 and f0["c"] == 0.0, f0

    # --- model comparison
    Ne = np.linspace(500, 8000, 12)
    ex_clean = 1.0 * np.exp(-Ne / 2000.0) + 0.02
    pw_clean = a_t * N**-b_t + c_t
    ne = 0
    for _ in range(10):
        ex = ex_clean * (1 + 0.03 * rng.standard_normal(Ne.size))
        pw = pw_clean * (1 + 0.03 * rng.standard_normal(N.size))
        ne += compare_power_vs_exp(Ne, ex)["winner"] == "exp"
        assert compare_power_vs_exp(N, pw)["winner"] == "power"
    assert ne >= 9, ne
    fe = fit_exponential(Ne, ex_clean)
    assert abs(fe["N0"] / 2000 - 1) < 1e-3 and abs(fe["c"] - 0.02) < 1e-5 and abs(fe["a"] - 1) < 1e-3, fe
    assert not fit_exponential(Ne[:3], ex_clean[:3])["converged"]

    # --- bootstrap CI
    reps = clean[None, :] * (1 + 0.03 * rng.standard_normal((12, N.size)))
    bt = fit_power_law_boot(N, reps, n_boot=60, rng=np.random.default_rng(1))
    assert bt["ci_lo"] <= bt["beta"] <= bt["ci_hi"] or abs(bt["beta"] - b_t) < 0.1, bt
    assert bt["ci_lo"] < b_t < bt["ci_hi"], bt
    assert bt["frac_nonconverged"] < 0.2, bt

    # --- local slope
    ls = local_slope(N, 2.0 * N**-0.5)
    assert ls.shape == (N.size - 1,) and np.allclose(ls, -0.5)
    ls = local_slope([1, 10, 100, 1000], [1.0, 0.0, 0.1, 0.01])
    assert np.isnan(ls[0]) and np.isnan(ls[1]) and np.isclose(ls[2], -1.0)

    # --- Zipf slope on exact Zipf counts
    ranks = np.arange(1, 5001)
    zc = 1e9 * ranks**-1.1
    zs = zipf_slope(rng.permutation(zc))
    assert abs(zs["s"] + 1.1) < 1e-9 and zs["r2"] > 1 - 1e-12, zs
    loc = zipf_slope_local(zc)
    assert all(abs(v + 1.1) < 1e-9 for v in loc.values()) and set(loc) == {10, 100, 1000}
    # broken power law: local slopes differ
    bz = np.where(ranks < 100, ranks**-0.8, 100 ** (-0.8 + 1.6) * ranks**-1.6) * 1e9
    bl = zipf_slope_local(bz, (10, 1000))
    assert bl[10] > bl[1000] and abs(bl[1000] + 1.6) < 1e-6, bl
    naive = np.polyfit(np.log(ranks[9:1000]), np.log(zc[9:1000] / zc.sum()), 1)
    assert abs(naive[0] - zs["s"]) < 1e-9

    # --- Dyer-Owen depth: naive match and N^(1/(s+2)) scaling
    s_z = 1.1
    J = 200_000
    q = np.arange(1, J + 1, dtype=np.float64) ** -s_z
    q /= q.sum()
    lam_small = 3000.0 * q[:500]
    p_naive, d_naive = 0.0, 1
    for n in range(2, 501):
        p_naive += math.exp(-(math.sqrt(lam_small[n - 2]) - math.sqrt(lam_small[n - 1])) ** 2)
        if p_naive <= 0.01:
            d_naive = n
    assert dyer_owen_depth(lam_small[::-1]) == d_naive
    Ns = np.geomspace(1e3, 1e7, 9)
    depths = np.array([dyer_owen_depth(n * q) for n in Ns], dtype=float)
    assert np.all(np.diff(depths) >= 0) and depths[0] >= 1
    slope = np.polyfit(np.log(Ns), np.log(depths), 1)[0]
    theory = 1.0 / (s_z + 2.0)
    assert theory / 1.5 < slope < theory * 1.5, (slope, theory, depths)

    # --- top-m set bound: brute force, monotone in N, bisection consistency
    qs = np.arange(1, 301, dtype=np.float64) ** -s_z
    qs /= qs.sum()
    m = 5
    for n in (2e3, 5e4):
        lam = n * qs
        brute = min(1.0, sum(math.exp(-(math.sqrt(lam[i]) - math.sqrt(lam[j])) ** 2)
                             for i in range(m) for j in range(m, lam.size)))
        assert abs(dyer_owen_topm_set_bound(lam, m) - brute) < 1e-12, (brute,)
    grid = np.geomspace(1e2, 1e9, 15)
    bnds = np.array([dyer_owen_topm_set_bound(n * qs, m) for n in grid])
    assert np.all(np.diff(bnds) <= 1e-15) and bnds[0] == 1.0 and bnds[-1] < 1e-6, bnds
    assert dyer_owen_topm_set_bound(qs * 1e4, 300) == 0.0
    ns = n_resolve_topm(qs, m, 0.05)
    assert dyer_owen_topm_set_bound(ns * qs, m) <= 0.05
    assert dyer_owen_topm_set_bound(ns * 0.99 * qs, m) > 0.05
    assert n_resolve_topm(qs, 10, 0.05) > ns
    assert n_resolve_topm([0.3, 0.3, 0.2], 1) == float("inf")
    # windowed O(J) evaluation on a long tail agrees with the full sum
    ql = np.arange(1, 20001, dtype=np.float64) ** -s_z
    ql /= ql.sum()
    full = float(np.exp(-(np.sqrt(2e5 * ql[:20])[:, None] - np.sqrt(2e5 * ql[20:])[None, :]) ** 2).sum())
    assert abs(dyer_owen_topm_set_bound(2e5 * ql, 20) - min(1.0, full)) < 1e-12

    # --- kappa recovery on Dirichlet-multinomial draws
    Jk, nk, kap_t = 500, 2000, 300.0
    pi = np.arange(1, Jk + 1, dtype=np.float64) ** -1.0
    pi /= pi.sum()
    draws = [rng.multinomial(nk, rng.dirichlet(kap_t * pi)).astype(np.float64) for _ in range(50)]
    kp = pooled_kappa_hat(draws, pi)
    assert abs(kp / kap_t - 1) < 0.10, kp
    single = np.array([kappa_hat(d, pi) for d in draws])
    assert abs(np.median(single) / kap_t - 1) < 0.25, np.median(single)
    # likelihood matches the all-items form and is maximal at the estimate
    def ll(k, c):
        n_ = c.sum()
        return gammaln(k) - gammaln(n_ + k) + np.sum(gammaln(c + k * pi) - gammaln(k * pi))
    tot = lambda k: sum(ll(k, d) for d in draws)
    assert tot(kp) >= tot(kp * 1.1) and tot(kp) >= tot(kp / 1.1)
    # multinomial (no over-dispersion) -> kappa very large
    mult = rng.multinomial(nk, pi).astype(np.float64)
    assert kappa_hat(mult, pi) > 1e4

    # --- shrinkage and crossover
    c = np.array([5.0, 0.0, 3.0, 2.0])
    p0 = np.array([0.1, 0.4, 0.3, 0.2])
    assert np.allclose(shrink_scores(c, p0, 1e-12), c / 10, atol=1e-9)
    assert np.allclose(shrink_scores(c, p0, 1e12), p0, atol=1e-9)
    assert np.allclose(shrink_scores(c, p0, 10.0), (c + 10 * p0) / 20) and abs(shrink_scores(c, p0, 7.0).sum() - 1) < 1e-12
    assert crossover_from_kappa(100.0, 10, 0.0) == 10.0
    assert abs(crossover_from_kappa(100.0, 10, 0.5) - 100 * 5.5 / 10) < 1e-12

    print("OK scaling")
