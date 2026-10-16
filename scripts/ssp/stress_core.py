"""Stress tests of the selector (stage S): one task dispatcher over the two window types.

A stress cell is a stored cell of an earlier run (confirmation window w2, or weekly-study window w1) with extra arms added:
the std arms and the seed tag are copied so that the std arms reproduce the stored cell exactly (an_stress.py checks this),
and the added arms use the new samplers of deep_core (pps / ppsprev with an exponent, gauss with fresh=1) plus the optional
audit sample and the optional overlap storage (deep_core.run_cell options `audit_n`, `save_overlap`).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import confirm_core as K
import weekly_core as W


def run_task(task: dict) -> dict:
    win = task["window"]
    if win == "w2":                       # confirmation windows: dataset key '<log>@w2', task['log'] is the plain log name
        r = K.run_task(task)
    elif win == "w1":                     # weekly-study windows: dataset key is the log name
        r = W.run_task(task)
        r["log"] = task["dataset"]
    else:
        raise ValueError(win)
    r["family"], r["window"] = task["family"], win
    return r
