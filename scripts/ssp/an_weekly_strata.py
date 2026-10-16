"""Post-hoc check of the weekly memory table by domain stratum (Amazon vs other logs).

    .venv/bin/python3 -W ignore scripts/ssp/an_weekly_strata.py

Written AFTER the W1-W11 verdicts (it is not in either weekly registry). It re-reads the weekly cells (02_runs/weekly.jsonl,
through an_weekly.load) and runs no new simulation. Two questions:
  1. How much does the hindsight-best memory (in rounds) differ between Amazon and other logs?
  2. Leave-one-log-out: for a held-out log, how much NDCG@100 does a memory fitted on the other logs lose against that log's
     hindsight-best half-life: (a) '2 rounds', (b) one table fitted on all other logs, (c) a table fitted on the other logs of
     the same stratum? Rounds of 7 and 28 days are pooled, as in the W9 table rule.
Writes results/small_sample_popularity/03_tables/weekly/h_strata_best_rounds.csv and h_strata_loo_regret.csv.
"""
from __future__ import annotations

import csv
import os

import numpy as np

from an_weekly import OUT, load

NS = [30, 100, 300, 1000]
RS = [1, 2, 4, 8, 16, 32]
CADENCES = (7, 28)
MIN_STRATUM_LOGS = 3


def stratum(log: str) -> str:
    return "amazon" if log.startswith("amazon") else "other"


def main() -> None:
    cells, _ = load()
    bad = [k for k, c in cells.items() if k[3] == "std" for a, v in c["val"].items() if a.startswith("decay:days=") and not np.isfinite(v)]
    assert not bad, f"non-finite decay values in cells {bad[:3]}"

    def best_value(c) -> float:
        return max(v for a, v in c["val"].items() if a.startswith("decay:days="))

    def regret(log: str, n: int, rd: int, rounds: float) -> float:
        c = cells[(log, n, rd, "std")]
        return best_value(c) - c["val"][f"decay:days={int(round(rounds * rd))}"]

    def log_regret(log: str, n: int, rounds: float) -> float:
        return float(np.mean([regret(log, n, rd, rounds) for rd in CADENCES if (log, n, rd, "std") in cells]))

    def logs_at(n: int) -> list[str]:
        return sorted({k[0] for k in cells if k[1] == n and k[3] == "std" and all((k[0], n, rd, "std") in cells for rd in CADENCES)})

    def fit(logs: list[str], n: int) -> int:
        return min(RS, key=lambda r: np.mean([log_regret(x, n, r) for x in logs]))

    def best_rounds(log: str, n: int, rd: int) -> float:
        c = cells[(log, n, rd, "std")]
        days = [(v, int(a.split("=")[1])) for a, v in c["val"].items() if a.startswith("decay:days=")]
        return max(days)[1] / rd

    rows_best, rows_loo = [], []
    for n in NS:
        L = logs_at(n)
        for s in ("amazon", "other"):
            Ls = [x for x in L if stratum(x) == s]
            if len(Ls) < MIN_STRATUM_LOGS:
                continue
            for rd in CADENCES:
                b = [best_rounds(x, n, rd) for x in Ls]
                rows_best.append([s, n, rd, len(Ls), *(float(f) for f in np.percentile(b, [25, 50, 75])), min(b), max(b)])
            p, t, two = [], [], []
            for held in Ls:
                p.append(log_regret(held, n, fit([x for x in L if x != held], n)))
                same = [x for x in Ls if x != held]
                t.append(log_regret(held, n, fit(same, n)) if len(same) >= MIN_STRATUM_LOGS else float("nan"))
                two.append(log_regret(held, n, 2))
            rstar = fit(Ls, n)
            rows_loo.append([s, n, len(Ls), rstar, round(float(np.mean([log_regret(x, n, rstar) for x in Ls])), 5),
                             round(float(np.mean(two)), 5), round(float(np.mean(p)), 5), round(float(np.nanmean(t)), 5)])

    os.makedirs(OUT, exist_ok=True)
    for name, head, rows in (
        ("h_strata_best_rounds.csv", ["stratum", "n", "round_days", "logs", "best_rounds_q25", "best_rounds_median", "best_rounds_q75", "best_rounds_min", "best_rounds_max"], rows_best),
        ("h_strata_loo_regret.csv", ["stratum", "n", "logs", "stratum_best_rounds_in_sample", "regret_in_sample_best", "regret_2rounds", "regret_pooled_table_loo", "regret_stratum_table_loo"], rows_loo),
    ):
        with open(f"{OUT}/{name}", "w", newline="") as f:
            cw = csv.writer(f, lineterminator="\n")
            cw.writerow(head)
            cw.writerows(rows)

    print("best memory in rounds (median [q25-q75]) by stratum")
    for r in rows_best:
        print(f"  {r[0]:6} n={r[1]:<5} {r[2]:>2}d logs={r[3]:<2} {r[5]:g} [{r[4]:g}-{r[6]:g}]")
    print("leave-one-log-out regret vs hindsight best (7- and 28-day rounds pooled)")
    for r in rows_loo:
        print(f"  {r[0]:6} n={r[1]:<5} logs={r[2]:<2} stratum-best r={r[3]:<2} 2rounds={r[5]:.4f} pooled-table={r[6]:.4f} stratum-table={r[7]:.4f}")


if __name__ == "__main__":
    main()
