"""Check of the participation cap (programme C, development stage). Exits 1 on any failed check.

  1. synthetic data (6 people active in all of 10 rounds, 2 rows each): (a) n = 6 (the whole pool) with kmax = 1, 2, 3, 10: the people per round are exactly 6 for kmax rounds and 0 afterwards; (b) n = 3 over 20
     seeds with kmax = 1, 2, 3: invariants only (overlapping draws can leave a short round before the capacity is used up): at most 3 people per round, everybody ends at exactly kmax draws, the total of
     the people is 6 * kmax, and once a round has nobody eligible every later round is exactly zero (no noise) and every earlier round is not; both kinds;
  2. kmax must be a whole number >= 1 and eps finite and positive (ValueError otherwise);
  3. real log (default amazon-musical-instruments, 100 people, draw 0, window w1, seed tag weekly-a, 60 rounds), kmax = 10**6: the same people and the same counts (eps = 1e6) as the uncapped
     price_core / lap_core samplers, round by round (Laplace counts to 1e-3, Gaussian to 0.02: the Gaussian "noise-free" arm still has sigma 7e-4);
  4. same log, kmax = 1: the same people every round, and the same set of people ever drawn, as deep_core's own single-use sampler `gauss:eps=E,c=1,fresh=1`;
  5. a log with a small pool of active people (default beeradvocate, 100 people), kmax = 2: nobody is drawn more than twice, the total number of draws equals the total of the people per round, the
     gaussn and lapn arms draw the same people, and the cap binds (an uncapped replay of the std draw uses someone more than twice, the capped draw reaches exactly 2);
  6. noise on catalogue items nobody in the sample rated, kmax = 5: Laplace std sqrt(2) C / E for lapn and the analytic Gaussian sigma for gaussn (mean ~ 0).

    .venv/bin/python3 -W ignore scripts/ssp/check_cap.py [LOG_FOR_3_4_6 [LOG_FOR_5]]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import cap_core as K
import deep_core as C
import weekly_core as W

TAG, N, DRAW, ROUNDS = "weekly-a", 100, 0, 60
EPS = 4.0


def synthetic(users=6, rounds=10, items=5, rows=2):
    ids = np.arange(1, users + 1, dtype=np.int32)
    user, item, off = [], [], [0]
    for _ in range(rounds):
        for u in ids:
            for j in range(rows):
                user.append(u)
                item.append((int(u) + j) % items)
        off.append(len(user))
    return dict(n_users=users, n_items=items, users_in=[ids] * rounds, off=np.array(off), user=np.array(user, np.int32), item=np.array(item, np.int32),
                first=np.zeros(items, np.int64))


def run(sampler, rounds):
    return [sampler.round(k) for k in range(rounds)]


def check_synthetic() -> int:
    bad = 0
    D = synthetic()
    for kmax in (1, 2, 3, 10):                                                 # (a) n = the whole pool: everybody is drawn in every round until the cap
        expect = [6] * kmax + [0] * (10 - kmax)
        for kind in ("gaussn", "lapn"):
            s = C.Sampler(f"{kind}:eps={EPS:g},C=1,kmax={kmax}", D, 6, ("synthetic", 6, 0, 10, TAG), False)
            obs = run(s, 10)
            got = [o.people for o in obs]
            ok = got == expect and int(s.drawn[1:].min()) == int(s.drawn[1:].max()) == kmax and all(float(np.abs(o.cnt).sum()) == 0.0 for o in obs[kmax:])     # drawn[0] is the unused id 0
            if not ok:
                bad += 1
                print(f"synthetic n=6 {kind} kmax={kmax}: people {got} (expected {expect}), draws {s.drawn.tolist()}")
        print(f"synthetic n=6, kmax={kmax}: people per round {expect}")
    short = 0
    for kmax in (1, 2, 3):                                                     # (b) n = half the pool: the pattern depends on the seed, the invariants do not
        for seed in range(20):
            for kind in ("gaussn", "lapn"):
                s = C.Sampler(f"{kind}:eps={EPS:g},C=1,kmax={kmax}", D, 3, ("synthetic", 3, seed, 10, TAG), False)
                obs = run(s, 10)
                got = [o.people for o in obs]
                z = got.index(0) if 0 in got else None
                ok = (z is not None and max(got) <= 3 and sum(got) == 6 * kmax and int(s.drawn[1:].min()) == int(s.drawn[1:].max()) == kmax
                      and all(g == 0 for g in got[z:]) and all(float(np.abs(o.cnt).sum()) == 0.0 for o in obs[z:]) and all(float(np.abs(o.cnt).sum()) > 0.0 for o in obs[:z]))
                short += kind == "gaussn" and got[:2 * kmax] != [3] * (2 * kmax)
                if not ok:
                    bad += 1
                    print(f"synthetic n=3 {kind} kmax={kmax} seed {seed}: people {got}, draws {s.drawn.tolist()}")
    print(f"synthetic n=3, 20 seeds x kmax 1, 2, 3: invariants checked ({short} of 60 runs had a short round before the capacity was used up)")
    return bad


def check_errors() -> int:
    bad = 0
    D = synthetic()
    for scheme in ("gaussn:eps=4,C=1,kmax=0", "lapn:eps=4,C=1,kmax=1.5", "lapn:eps=4,C=1,kmax=nan", "gaussn:eps=4,C=1,kmax=-2",
                   "lapn:eps=0,C=1,kmax=2", "gaussn:eps=nan,C=1,kmax=2", "lapn:eps=-1,C=1,kmax=2", "gaussn:eps=inf,C=1,kmax=2"):
        try:
            C.Sampler(scheme, D, 3, ("synthetic", 3, 0, 10, TAG), False).round(0)
        except ValueError:
            continue
        bad += 1
        print(f"{scheme}: no ValueError")
    print(f"error handling: {bad} failures")
    return bad


def check_real(log: str) -> int:
    D = W.prepare_weekly(log, 7)
    nw = D["n_win"]
    seed = (log, N, DRAW, nw, TAG)
    mk = lambda scheme: C.Sampler(scheme, D, N, seed, False)
    bad = 0
    big = 10 ** 6
    ga, gp = mk(f"gaussn:eps=1000000,C=1,kmax={big}"), mk("gaussn:eps=1000000,C=1")
    la, lp = mk(f"lapn:eps=1000000,C=1,kmax={big}"), mk("lapn:eps=1000000,C=1")
    for k in range(ROUNDS):                                                    # the Gaussian "noise-free" reference still has sigma 7e-4 (analytic sigma at eps 1e6), so its counts match to 0.02, the Laplace one (scale 1e-6) to 1e-3
        oga, ogp, ola, olp = ga.round(k), gp.round(k), la.round(k), lp.round(k)
        if not (oga.people == ogp.people == ola.people == olp.people and np.allclose(ola.cnt, olp.cnt, atol=1e-3) and float(np.abs(oga.cnt - ogp.cnt).max()) < 0.02):
            bad += 1
            print(f"{log} round {k}: kmax=1e6 differs from the uncapped samplers (people gaussn {oga.people} vs {ogp.people}, lapn {ola.people} vs {olp.people})")
            break
    if not np.array_equal(ga.drawn, la.drawn):
        bad += 1
        print(f"{log}: gaussn and lapn with kmax=1e6 drew different people")
    cap1, fresh = mk("gaussn:eps=1000000,C=1,kmax=1"), mk("gauss:eps=1000000,c=1,fresh=1")
    for k in range(ROUNDS):
        o1, o2 = cap1.round(k), fresh.round(k)
        if o1.people != o2.people:
            bad += 1
            print(f"{log} round {k}: kmax=1 people {o1.people} != single-use sampler {o2.people}")
            break
    if not np.array_equal(cap1.drawn > 0, fresh.used):
        bad += 1
        print(f"{log}: kmax=1 set of people drawn differs from the single-use sampler")
    print(f"{log}: kmax=1e6 equals uncapped and kmax=1 equals single-use sampler over {ROUNDS} rounds: {bad} failures so far")
    noise = {"gaussn": [], "lapn": []}
    sg, sl = mk(f"gaussn:eps={EPS:g},C=1,kmax=5"), mk(f"lapn:eps={EPS:g},C=1,kmax=5")
    rg, rl = mk("gaussn:eps=1000000,C=1,kmax=5"), mk("lapn:eps=1000000,C=1,kmax=5")
    for k in range(ROUNDS):
        og, ol, ogr, olr = sg.round(k), sl.round(k), rg.round(k), rl.round(k)
        if not (og.people == ol.people == ogr.people == olr.people):
            bad += 1
            print(f"{log} round {k}: capped arms with kmax=5 draw different numbers of people")
            break
        zero = (D["first"] <= k) & ~(olr.cnt > 1e-4)                         # catalogue items nobody in the sample rated: the observation is pure noise
        noise["gaussn"].append(og.cnt[zero].astype(np.float64))
        noise["lapn"].append(ol.cnt[zero].astype(np.float64))
    if not (np.array_equal(sg.drawn, sl.drawn) and np.array_equal(sg.drawn, rg.drawn) and np.array_equal(sg.drawn, rl.drawn)):
        bad += 1
        print(f"{log}: capped arms with kmax=5 drew different people")
    for nm, sd_exp in (("gaussn", C.gauss_sigma(EPS, 1.0)), ("lapn", np.sqrt(2) / EPS)):
        x = np.concatenate(noise[nm])
        sd = float(x.std())
        ok = len(x) > 1000 and abs(float(x.mean())) < 6 * sd_exp / np.sqrt(len(x)) and abs(sd / sd_exp - 1) < 0.02
        bad += not ok
        print(f"{log}: {nm} kmax=5: {len(x)} noise values, mean {float(x.mean()):+.5f}, std {sd:.4f} (expected {sd_exp:.4f}) -> {'ok' if ok else 'FAIL'}")
    return bad


def check_pool(log: str) -> int:
    D = W.prepare_weekly(log, 7)
    nw = D["n_win"]
    seed = (log, N, DRAW, nw, TAG)
    mk = lambda scheme: C.Sampler(scheme, D, N, seed, False)
    kmax = 2
    g, l = mk(f"gaussn:eps=1000000,C=1,kmax={kmax}"), mk(f"lapn:eps=1000000,C=1,kmax={kmax}")
    rr = np.random.default_rng(C.seed_of(*seed, "sel"))                         # replay of the std draw (same stream as the std sampler's) to see what the cap changes
    plain = np.zeros(D["n_users"] + 1, np.int32)
    bad, total, short, first_diff = 0, 0, 0, None
    for k in range(ROUNDS):
        og, ol = g.round(k), l.round(k)
        uk = D["users_in"][k]
        psel = uk if len(uk) <= N else rr.choice(uk, N, replace=False)
        plain[psel] += 1
        total += og.people
        short += og.people < min(N, len(uk))
        if og.people != ol.people:
            bad += 1
            print(f"{log} round {k}: gaussn and lapn arms draw different numbers of people")
        if first_diff is None and int(plain.max()) > kmax:
            first_diff = k                                                   # from this round on the std draw would have used someone a third time
    ok = int(g.drawn.max()) <= kmax and int(g.drawn.sum()) == total and np.array_equal(g.drawn, l.drawn)
    bound = int(g.drawn.max()) == kmax and int(plain.max()) > kmax
    bad += (not ok) + (not bound)
    print(f"{log}: kmax={kmax}, {ROUNDS} rounds: max draws {int(g.drawn.max())} (uncapped std draw: {int(plain.max())}, first exceeds {kmax} in round {first_diff}), total draws {total} "
          f"(= sum of people: {int(g.drawn.sum()) == total}), rounds short of people {short}, arms draw the same people: {np.array_equal(g.drawn, l.drawn)} -> {'ok' if ok and bound else 'FAIL'}")
    return bad


if __name__ == "__main__":
    log = sys.argv[1] if len(sys.argv) > 1 else "amazon-musical-instruments"
    pool = sys.argv[2] if len(sys.argv) > 2 else "beeradvocate"
    total = check_synthetic() + check_errors() + check_real(log) + check_pool(pool)
    print(f"{total} failed checks")
    sys.exit(1 if total else 0)
