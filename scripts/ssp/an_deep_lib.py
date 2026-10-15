"""Shared loader for the deep programme results (results/small_sample_popularity/02_runs/deep.jsonl).

    deep = Deep(path, reg)                       # path: jsonl of rows written by run_deep.py; reg: configs/deep_registry.json as dict
    for row in deep.cells("plan"): ...           # rows of one stage (dict with dataset, n, n_win, arms, steps, nd100, selfnd, metrics, ...)
    deep.steady(row, arm, "nd100")               # np.array [draw]: steady-state mean (steps >= 8) of a metric for one arm
    deep.steps(row, arm, "nd100")                # np.array [draw][step] per-step values (steps start at round 2); "nd100" and "selfnd" only
    deep.stratum(log)  -> "amazon" | "other"     deep.split(log) -> "discovery" | "holdout"
    deep.has(row, arm) -> bool

Metrics available per (draw, arm): nd100 (per step), selfnd (per step), and steady means of nd10, nd30, nd300, rec100, churn,
len100 (published list length at cut 100), prec100 (share of published items that are in the true top 100), agree (split-half
agreement, half:* arms only), chosen_h_idx (selfsel only), people (mean reporters per round, via people_per_round).
Half-life grid index for chosen_h_idx: 0..6 = 0.25, 0.5, 1, 2, 4, 8, inf rounds.
"""
from __future__ import annotations

import json
import os

import numpy as np

STEADY = 8
FIRST_STEP = 2


class Deep:
    def __init__(self, path: str, reg: dict):
        self.reg = reg
        self.rows = []
        if os.path.exists(path):
            for ln in open(path):
                try:
                    self.rows.append(json.loads(ln))
                except Exception:
                    pass
        sp = reg["split"]
        self._disc, self._hold = set(), set()
        for g, v in sp.items():
            if g in ("salt",):
                continue
            names = (v["amazon"] + v["other"]) if isinstance(v, dict) else v
            (self._disc if g in ("discovery", "new_discovery") else self._hold).update(names)

    def cells(self, stage: str) -> list:
        return [r for r in self.rows if r["key"].startswith(stage + "|")]

    def has(self, row, arm) -> bool:
        return arm in row["arms"]

    def steps(self, row, arm, metric="nd100") -> np.ndarray:
        a = np.array(row[metric], float)                       # [draw][arm][step]
        return a[:, row["arms"].index(arm), :]

    def steady(self, row, arm, metric="nd100") -> np.ndarray:
        i = row["arms"].index(arm)
        if metric in ("nd100", "selfnd"):
            a = np.array(row[metric], float)[:, i, :]
            idx = [j for j, s in enumerate(row["steps"]) if s >= STEADY]
            with np.errstate(all="ignore"):
                return np.nanmean(a[:, idx], axis=1)
        m = row["metrics"].get(metric)
        if m is None:
            return np.full(len(row["nd100"]), np.nan)
        return np.array([[np.nan if x is None else x for x in d] for d in m], float)[:, i]

    @staticmethod
    def stratum(log: str) -> str:
        return "amazon" if log.startswith("amazon-") else "other"

    def split(self, log: str) -> str:
        return "discovery" if log in self._disc else "holdout"

    def mean(self, row, arm, metric="nd100") -> float:
        with np.errstate(all="ignore"):
            return float(np.nanmean(self.steady(row, arm, metric)))
