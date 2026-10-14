"""Task functions for every experiment. Each takes one task dict and returns one flat result dict.

A task is deterministic: its random stream depends only on its key, never on run order.

  e1  grid        popularity estimators vs truth A/B, plus the fixed EVAL panel (no model training)
  e2  personal    knn / EASE and the pop+personal hybrid with inner hold-out choices of lambda and w
  e4  rankcs      rank confidence sets: naive percentile bootstrap vs Mogstad et al., coverage by rank band
  e5  subgroup    estimate a real subgroup's list from a small subgroup sample, with/without a global prior
  e6  dp          central/local differential privacy on the sample counts
"""
from __future__ import annotations

import time
import zlib

import numpy as np
import scipy.sparse as sp

import dp as DP
import est as E
import frame as F
import metrics as M
import rankcs as RC
import scaling as SC

SEED = 20261014
J_CAP = 2500       # candidate universe for popularity-only runs
J_CAP_E2 = 1500    # smaller for runs that train models (EASE inverts a J x J matrix)
PANEL_MAX = 1500
_FR: dict = {}


def frame(name: str) -> F.Frame:
    if name not in _FR:
        _FR.clear()  # one frame per worker at a time keeps memory flat
        _FR[name] = F.load_frame(name)
    return _FR[name]


def rng_for(key: str) -> np.random.Generator:
    return np.random.default_rng([SEED, zlib.crc32(key.encode())])


def panel_rows(fr: F.Frame) -> np.ndarray:
    """Fixed panel of EVAL users (same for every cell of a dataset, so cells are paired)."""
    n_c = np.diff(fr.e_c.indptr)
    ok = np.flatnonzero((n_c <= 1000) & (np.diff(fr.e_f.indptr) >= 1))
    r = np.random.default_rng([SEED, 7])
    return np.sort(r.choice(ok, min(PANEL_MAX, len(ok)), replace=False))


def prior_pi(fr: F.Frame) -> np.ndarray:
    c = fr.count_old
    return (c + 0.1) / (c.sum() + 0.1 * len(c))


_WIN: dict = {}


def window_pool(fr: F.Frame) -> np.ndarray:
    """POP users with at least one collection choice after T_old (people active in the recent window)."""
    key = id(fr)
    if key not in _WIN:
        _WIN.clear()
        n_win = np.add.reduceat((fr.c_ts >= fr.T_old).astype(np.int64), fr.c_ptr[:-1])
        n_win = np.where(np.diff(fr.c_ptr) > 0, n_win, 0)
        _WIN[key] = np.flatnonzero(n_win > 0)
    return _WIN[key]


def sample(fr, task, rng):
    """Draw the users and their rows for a task. Returns S, user weights, n_users, row_ts."""
    mode = task.get("users", "uniform")
    rule = task.get("rows", "random")
    k = task["k"]
    k = None if k in (None, 0) else int(k)
    window = bool(task.get("window")) and fr.has_time
    pool = window_pool(fr) if window else task.get("pool")
    users, pi = F.draw_users(fr, task["N"], rng, mode, pool)
    ts_min = fr.T_old if window else None
    S, kept, avail, row_ts = F.draw_rows(fr, users, k, rng, rule, ts_min)
    nonempty = kept > 0
    w = np.zeros(len(users))
    w[nonempty] = avail[nonempty] / kept[nonempty] / pi[nonempty]
    return S, w, len(users), row_ts


def _r(x):
    return round(float(x), 5) if np.isfinite(x) else None


def pop_estimators(fr, S, w, n_users, rng, with_ht: bool):
    """Dict name -> full-catalogue score vector."""
    c = E.pop_count(S)
    out = {"count": c}
    if with_ht:
        out["ht"] = E.pop_ht(S, w)
    if fr.has_time and fr.count_old.sum() > 0:
        pi = prior_pi(fr)
        kap = SC.kappa_hat(c, pi)
        kap = float(np.clip(kap, 1.0, 1e6))
        out["shrunk"] = E.pop_shrunk(c, fr.count_old + 0.1, kap)
        out["_kappa"] = kap
    out["lcb"] = E.pop_lcb(c, n_users)
    return out


# ------------------------------------------------------------------ E1

def run_e1(task: dict) -> dict:
    t0 = time.time()
    fr = frame(task["dataset"])
    rng = rng_for(task["key"])
    S, w, n_users, row_ts = sample(fr, task, rng)
    ests = pop_estimators(fr, S, w, n_users, rng, with_ht=task.get("users", "uniform") != "uniform")
    row = {"dataset": task["dataset"], "N": task["N"], "k": task["k"], "draw": task["draw"],
           "users": task.get("users", "uniform"), "rows": task.get("rows", "random"),
           "arm": task.get("arm", "uniform"), "n_users": int(n_users), "n_rows": int(S.nnz), "items_seen": int((np.asarray(S.sum(0)).ravel() > 0).sum())}
    if "_kappa" in ests:
        row["kappa"] = _r(ests.pop("_kappa"))
    # list quality vs truth A and B
    for name, sc in ests.items():
        for k_, v in E.list_metrics(sc, fr, rng, name).items():
            row[k_] = _r(v)
    # EVAL panel
    prow = panel_rows(fr)
    k = task["k"]
    prof = F.eval_profiles(fr, None if k in (None, 0) else int(k), rng, prow)
    c = ests["count"]
    om = E.universe(c, J_CAP, rng)
    pn = E.Panel(fr, prow, prof, om)
    for name, sc in ests.items():
        for k_, v in E.rank_eval(sc[om][None, :], pn, rng=rng).items():
            row[f"{name}.U_{k_}"] = _r(v)
    row["omega"] = int(len(om))
    row["sec"] = _r(time.time() - t0)
    return row


# ------------------------------------------------------------------ E2

R_GRID = (0.05, 0.25, 1.25)
W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def holdout_choice(S, row_ts, om, rng, mode: str):
    """Pick (model, lambda-ratio, w) on the sample alone by hiding part of each sampled user's rows.

    mode 'random': hide a random 25%.  mode 'time': hide each user's latest 25% (predict their future).
    Returns (best blend choice, best pure-personalised choice, best NDCG) or (None, None, None).
    """
    Vi, Hi = E.split_hidden(S, rng, 0.25, row_ts if mode == "time" else None)
    if Vi.shape[0] < 10:
        return None, None, None
    V = Vi[:, om].tocsr()
    cvp = type("P", (), {})()
    cvp.seen, cvp.rel, cvp.n_rel = V.toarray() > 0, Hi[:, om].toarray() > 0, np.asarray(Hi.sum(1)).ravel()
    Gv = E.gram(V)
    mdv = max(float(np.mean(np.diag(Gv))), 1e-6)
    zv = E.zscore_rows(np.asarray(V.sum(0)).ravel())
    table = {}
    for r in R_GRID:
        sc = E.zscore_rows(E.apply_model(V, E.ease_B(Gv, r * mdv)))
        for wt in W_GRID:
            table[("ease", r, wt)] = E.rank_eval((1 - wt) * zv + wt * sc, cvp, rng=rng)["ndcg"]
    sc = E.zscore_rows(E.apply_model(V, E.knn_W(Gv)))
    for wt in W_GRID:
        table[("knn", 0.0, wt)] = E.rank_eval((1 - wt) * zv + wt * sc, cvp, rng=rng)["ndcg"]
    ok = {kk: v for kk, v in table.items() if np.isfinite(v)}
    if not ok:
        return None, None, None
    best = max(ok, key=ok.get)
    pure = {kk: v for kk, v in ok.items() if kk[2] == 1.0}
    return best, (max(pure, key=pure.get) if pure else best), ok[best]


def run_e2(task: dict) -> dict:
    t0 = time.time()
    fr = frame(task["dataset"])
    rng = rng_for(task["key"])
    S, w, n_users, row_ts = sample(fr, task, rng)
    k = None if task["k"] in (None, 0) else int(task["k"])
    row = {"dataset": task["dataset"], "N": task["N"], "k": task["k"], "draw": task["draw"], "n_rows": int(S.nnz)}
    c = E.pop_count(S)
    om = E.universe(c, J_CAP_E2, rng)
    S_om = S[:, om].tocsr()
    prow = panel_rows(fr)
    prof = F.eval_profiles(fr, k, rng, prow)
    pn = E.Panel(fr, prow, prof, om)
    G = E.gram(S_om)
    md = max(float(np.mean(np.diag(G))) if len(G) else 1.0, 1e-6)
    zpop = E.zscore_rows(c[om])
    base = E.rank_eval(zpop, pn, rng=rng)
    row["pop.U_ndcg"], row["pop.U_recall"] = _r(base["ndcg"]), _r(base["recall"])
    # models fitted on the whole sample at each lambda ratio, scored at every blend weight
    scores = {}
    for r in R_GRID:
        scores[("ease", r)] = E.zscore_rows(E.apply_model(pn.profile, E.ease_B(G, r * md)))
    scores[("knn", 0.0)] = E.zscore_rows(E.apply_model(pn.profile, E.knn_W(G)))
    grid = {}
    for (mdl, r), sc in scores.items():
        for wt in W_GRID:
            grid[(mdl, r, wt)] = E.rank_eval((1 - wt) * zpop + wt * sc, pn, rng=rng)["ndcg"]
    # oracle (panel-tuned, an upper bound) and fixed reference points
    row["oracle_best.U_ndcg"] = _r(max(grid.values()))
    row["ease_r25.w100.U_ndcg"] = _r(grid[("ease", 0.25, 1.0)])
    row["ease_r25.w50.U_ndcg"] = _r(grid[("ease", 0.25, 0.5)])
    row["knn.w100.U_ndcg"] = _r(grid[("knn", 0.0, 1.0)])
    row["knn.w50.U_ndcg"] = _r(grid[("knn", 0.0, 0.5)])
    for wt in W_GRID:  # hybrid weight curve for the best fixed model (kNN) and EASE
        row[f"knn.w{int(wt*100)}.U_ndcg"] = _r(grid[("knn", 0.0, wt)])
        row[f"ease.w{int(wt*100)}.U_ndcg"] = _r(max(grid[("ease", r, wt)] for r in R_GRID))
    # honest choices on the sample alone: random hide vs time-ordered hide
    can_cv = (k is None or k >= 3) and S.shape[0] >= 20
    if can_cv:
        for mode in ("random", "time"):
            ch, pure, v = holdout_choice(S, row_ts, om, rng, mode)
            if ch is None:
                continue
            row[f"hyb_{mode}.w"], row[f"hyb_{mode}.model"] = ch[2], ch[0]
            row[f"hyb_{mode}.U_ndcg"] = _r(grid[ch])
            row[f"pers_{mode}.U_ndcg"] = _r(grid[(pure[0], pure[1], 1.0)])
    row["omega"] = int(len(om))
    row["sec"] = _r(time.time() - t0)
    return row


# ------------------------------------------------------------------ E4

BANDS = [(1, 10), (11, 50), (51, 200), (201, 1000)]


CATALOG = 1000


def run_e4(task: dict) -> dict:
    """Rank confidence sets for the CATALOG truly most popular items (the question 'where does each rank?').

    Restricting to a fixed catalogue keeps the upper bound meaningful (with 500k items it would be 'anywhere').
    """
    t0 = time.time()
    fr = frame(task["dataset"])
    rng = rng_for(task["key"])
    S, w, n_users, row_ts = sample(fr, task, rng)
    cat = np.argsort(-fr.count_A, kind="stable")[:CATALOG]
    Sc = S[:, cat].tocsr()
    tr = RC.true_ranks(fr.count_A[cat] / fr.n_pop)
    focus = np.flatnonzero(tr <= 200)
    n_boot = task.get("n_boot", 150)
    lo_n, hi_n = RC.percentile_rank_intervals(Sc, 0.05, n_boot, rng, items=focus)
    lo_m, hi_m = RC.mogstad_rank_cs(Sc, 0.05, n_boot, rng, items=focus, max_compare=CATALOG)
    row = {"dataset": task["dataset"], "N": task["N"], "k": task["k"], "draw": task["draw"],
           "catalog": int(len(cat))}
    for tag, lo, hi in (("naive", lo_n, hi_n), ("mogstad", lo_m, hi_m)):
        cov = (lo <= tr[focus]) & (tr[focus] <= hi)
        row[f"{tag}.cov"] = _r(cov.mean())
        row[f"{tag}.width"] = _r(np.mean(hi - lo + 1))
        row[f"{tag}.width_top10"] = _r(np.mean((hi - lo + 1)[tr[focus] <= 10]))
        for a_, b_ in BANDS[:3]:
            m = (tr[focus] >= a_) & (tr[focus] <= b_)
            row[f"{tag}.cov_{a_}_{b_}"] = _r(cov[m].mean()) if m.any() else None
    row["sec"] = _r(time.time() - t0)
    return row


# ------------------------------------------------------------------ E5

def subgroup_pool(fr: F.Frame, kind: str) -> np.ndarray:
    if kind.startswith("genre:"):
        g = kind.split(":", 1)[1]
        g = g if g in fr.genre_share else next(x for x in fr.genre_share if x.startswith(g))  # Children / Children's
        share = fr.genre_share[g]
        thr = np.quantile(share, 0.85)
        return np.flatnonzero(share >= thr)
    if kind.startswith("activity:"):
        t = int(kind.split(":")[1])
        q = np.quantile(fr.n_c, [1 / 3, 2 / 3])
        return np.flatnonzero(np.digitize(fr.n_c, q) == t)
    if kind.startswith("joined:"):
        t = int(kind.split(":")[1])
        q = np.quantile(fr.first_ts, [1 / 3, 2 / 3])
        return np.flatnonzero(np.digitize(fr.first_ts, q) == t)
    raise ValueError(kind)


def run_e5(task: dict) -> dict:
    t0 = time.time()
    fr = frame(task["dataset"])
    rng = rng_for(task["key"])
    pool = subgroup_pool(fr, task["group"])
    X = fr.pop_csr(pool)
    truth = np.asarray(X.sum(0)).ravel()          # the subgroup's own full list (truth A_g)
    others = fr.count_A - truth                   # leave-group-out global list = the prior
    n = min(task["N"], len(pool))
    rows = rng.choice(len(pool), n, replace=False)
    k = None if task["k"] in (None, 0) else int(task["k"])
    S, _, _, _ = F.draw_rows(fr, pool[rows], k, rng, "random")
    c = E.pop_count(S)
    n_total = c.sum()
    pi = (others + 0.1) / (others.sum() + 0.1 * len(others))
    kap = float(np.clip(SC.kappa_hat(c, pi), 1.0, 1e7)) if n_total > 0 else 1e7
    cands = {"sample": c, "global": others + 0.0, "shrunk": E.pop_shrunk(c, others + 0.1, kap)}
    # fixed blends so the weight curve is visible
    for w_ in (0.25, 0.5, 0.75):
        cands[f"blend{int(w_*100)}"] = w_ * c / max(n_total, 1) + (1 - w_) * others / max(others.sum(), 1)
    row = {"dataset": task["dataset"], "group": task["group"], "N": n, "N_req": task["N"], "k": task["k"],
           "draw": task["draw"], "pool": int(len(pool)), "kappa": _r(kap)}
    # distance of the subgroup from the global list (how distinctive it is)
    row["group_vs_global_ov100"] = _r(M.overlap_at_m(others, truth, 100, rng))
    for name, sc in cands.items():
        top = M.top_m(sc, 100, rng)
        row[f"{name}.ov10"] = _r(M.overlap_at_m(sc, truth, 10, rng))
        row[f"{name}.ov100"] = _r(M.overlap_at_m(sc, truth, 100, rng))
        row[f"{name}.ndcg100"] = _r(M.ndcg_vs_gain(top, truth, 100))
        row[f"{name}.mass10"] = _r(M.mass_captured(M.top_m(sc, 10, rng), truth, 10))
        row[f"{name}.rbo90"] = _r(M.rbo_from_scores(sc, truth, 0.9, 100, rng))
    row["sec"] = _r(time.time() - t0)
    return row


# ------------------------------------------------------------------ E6

def run_e6(task: dict) -> dict:
    t0 = time.time()
    fr = frame(task["dataset"])
    rng = rng_for(task["key"])
    users, _ = F.draw_users(fr, task["N"], rng, "uniform")
    k = int(task["k"])
    X = fr.pop_csr(users)
    Xk = DP.bound_contributions(X, k, rng)
    truth = fr.count_A
    row = {"dataset": task["dataset"], "N": len(users), "k": k, "eps": task["eps"], "draw": task["draw"]}
    est = {"nonprivate": np.asarray(Xk.sum(0)).ravel().astype(np.float64)}
    eps = float(task["eps"])
    est["laplace"] = DP.central_laplace_counts(Xk, k, eps, rng)
    est["gaussian"] = DP.central_gaussian_counts(Xk, k, eps, 1e-6, rng)
    est["ldp_oue"] = DP.ldp_oue_counts(Xk, k, eps, rng)
    for name, sc in est.items():
        top = M.top_m(sc, 100, rng)
        row[f"{name}.ov10"] = _r(M.overlap_at_m(sc, truth, 10, rng))
        row[f"{name}.ov100"] = _r(M.overlap_at_m(sc, truth, 100, rng))
        row[f"{name}.ndcg100"] = _r(M.ndcg_vs_gain(top, truth, 100))
        row[f"{name}.mass10"] = _r(M.mass_captured(M.top_m(sc, 10, rng), truth, 10))
        row[f"{name}.rbo90"] = _r(M.rbo_from_scores(sc, truth, 0.9, 100, rng))
    row["sec"] = _r(time.time() - t0)
    return row


# ------------------------------------------------------------------ reference rows

def run_ref(task: dict) -> dict:
    """Constant reference lists per dataset: stale list, and the two oracle lists."""
    fr = frame(task["dataset"])
    rng = rng_for(task["key"])
    row = {"dataset": task["dataset"]}
    cands = {"stale": fr.count_old if fr.count_old.sum() > 0 else None, "truthA": fr.count_A, "truthB": fr.count_B}
    prow = panel_rows(fr)
    prof = F.eval_profiles(fr, 5, rng, prow)
    om = E.universe(fr.count_A, J_CAP, rng)
    pn = E.Panel(fr, prow, prof, om)
    for name, sc in cands.items():
        if sc is None:
            continue
        for k_, v in E.list_metrics(sc.astype(np.float64), fr, rng, name).items():
            row[k_] = _r(v)
        for k_, v in E.rank_eval(sc[om][None, :].astype(np.float32), pn, rng=rng).items():
            row[f"{name}.U_{k_}"] = _r(v)
    return row


KINDS = {"e1": run_e1, "e2": run_e2, "e4": run_e4, "e5": run_e5, "e6": run_e6, "ref": run_ref}


def run_task(task: dict) -> dict:
    return KINDS[task["kind"]](task)
