#!/usr/bin/env python3
"""Recompute the headline numbers and spot checks of docs/small_sample/23_review_package_en.md.

Standard library only. Reads only the committed tables under
results/small_sample_popularity/03_tables/ (no raw data, no network).

Every check compares a value computed from a table with the value as printed on the page.
The page prints each number rounded, so a number passes only if the table value rounds to what the
page shows: to the last printed decimal, or to the nearest integer (half up) for an integer. A
percent printed as "17" is passed as the share times 100.

This shows that the page quotes the tables correctly. It does not re-derive any table from the
raw run rows.

Usage (from anywhere): python3 scripts/ssp/check_review_numbers.py
Exit status: 0 if every number matches, 1 if any differs or a table is missing.
"""
import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results" / "small_sample_popularity" / "03_tables"

results = []


def rows(rel):
    with open(T / rel, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def js(rel):
    with open(T / rel, encoding="utf-8") as fh:
        return json.load(fh)


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def chk(card, label, got, printed):
    """Compare a computed value with the number as printed on the page."""
    if "." in printed:
        decimals = len(printed.split(".")[1])
        ok = got is not None and abs(float(got) - float(printed)) <= 0.5 * 10 ** (-decimals) + 1e-12
    else:
        ok = got is not None and math.floor(float(got) + 0.5) == int(printed)
    results.append(ok)
    print("%s %s %s: table %s, page %s" % ("PASS" if ok else "FAIL", card, label, "n/a" if got is None else "%.6g" % got, printed))


def chk_true(card, label, condition):
    results.append(bool(condition))
    print("%s %s %s" % ("PASS" if condition else "FAIL", card, label))


def pick(rs, **kw):
    """The single row whose columns equal the given strings."""
    hits = [r for r in rs if all(r.get(k) == v for k, v in kw.items())]
    if len(hits) != 1:
        raise KeyError("expected one row for %r, found %d" % (kw, len(hits)))
    return hits[0]


def rh1():
    a = rows("an_explore_a_x10_drift_vs_sampling.csv")
    lists = [r for r in a if r["measure"] == "ndcg_vs_truth_lists"]
    row = pick(a, dataset="ml-latest", measure="ndcg_vs_truth_lists")
    chk("RH1", "ml-latest drift gap", num(row["drift_gap_B_minus_A"]), "0.65803")
    chk("RH1", "ml-latest sampling gap at 1,000", num(row["sampling_gap_N1000"]), "0.000909333")
    chk("RH1", "median drift gap, list level", statistics.median(num(r["drift_gap_B_minus_A"]) for r in lists), "0.658")
    samp = [num(r["sampling_gap_N1000"]) for r in lists if num(r["sampling_gap_N1000"]) is not None]
    chk("RH1", "median sampling gap at 1,000", statistics.median(samp), "0.003")
    chk("RH1", "datasets with a 1,000-person cell", len(samp), "10")
    chk("RH1", "drift larger than sampling gap at 1,000", sum(r["drift_exceeds_sampling_gap_N1000"] == "True" for r in lists), "10")
    panel = [r for r in a if r["measure"] == "panel_ndcg"]
    chk("RH1", "median panel drift", statistics.median(num(r["drift_gap_B_minus_A"]) for r in panel), "0.021")
    chk("RH1", "median panel sampling gap", statistics.median(num(r["sampling_gap_N1000"]) for r in panel if num(r["sampling_gap_N1000"]) is not None), "0.0003")
    x7 = [r for r in rows("an_explore_b_x7_datasets.csv") if r["arm"] == "recent_window"]
    chk("RH1", "recent_window datasets", len(x7), "11")
    chk("RH1", "recent_window sign flips", sum(r["flip_B"] == "True" for r in x7), "10")
    chk("RH1", "recent_window dataset median, Truth B", statistics.median(num(r["dB"]) for r in x7), "0.037")
    chk("RH1", "recent_window dataset median, Truth A", statistics.median(num(r["dA"]) for r in x7), "-0.186")
    allk = [r for r in rows("an_window_x12_win_share.csv") if r["k"] == "all-k"]
    chk("RH1", "X12 cells", sum(int(r["cells"]) for r in allk), "223")
    chk("RH1", "X12 wins", sum(int(r["wins"]) for r in allk), "179")
    chk("RH1", "X12 losses", sum(int(r["losses"]) for r in allk), "2")
    b = rows("badrounds/badrounds_summary.csv")
    for cad, n, within, rho in (("weekly", "100", "0.125", "0.86"), ("weekly", "300", "0.074", "0.94"), ("deep", "30", "0.46", "0.25")):
        r = pick(b, cadence=cad, n=n)
        chk("RH1", "%s %s draw-to-draw share" % (cad, n), num(r["median_within_share"]), within)
        chk("RH1", "%s %s correlation sampled vs full" % (cad, n), num(r["median_rho_yf"]), rho)


def rh2():
    g = {r["n"]: r for r in rows("weekly/g_memory_table.csv")}
    chk("RH2", "n=100 half-life", num(g["100"]["table_half_life_rounds"]), "16")
    chk("RH2", "n=100 table regret", num(g["100"]["holdout_regret_table"]), "0.01827")
    chk("RH2", "n=100 2-rounds regret", num(g["100"]["holdout_regret_2rounds"]), "0.17057")
    for n, table, two in (("30", "0.012", "0.149"), ("100", "0.018", "0.171"), ("300", "0.024", "0.106"), ("1000", "0.019", "0.047")):
        chk("RH2", "n=%s table regret" % n, num(g[n]["holdout_regret_table"]), table)
        chk("RH2", "n=%s 2-rounds regret" % n, num(g[n]["holdout_regret_2rounds"]), two)
    w = js("weekly/weekly_verdicts.json")
    chk("RH2", "W9 table regret", w["W9"]["holdout_regret_table"], "0.018")
    chk("RH2", "W9 2-rounds regret", w["W9"]["holdout_regret_2rounds"], "0.118")
    chk("RH2", "W9 ratio", w["W9"]["ratio"], "0.15")
    chk("RH2", "W1 logs with best half-life >= 28 days", w["W1"]["share_ge28"]["share"] * w["W1"]["share_ge28"]["n"], "27")
    chk("RH2", "W4 gain", w["W4"]["all"]["mean"], "0.0042")
    chk("RH2", "W7 gain at 100 people", w["W7"]["n100"]["best"]["all"]["mean"], "0.012")
    chk("RH2", "W11 gain at 100 people", w["W11"]["n100"]["all"]["mean"], "0.022")
    chk("RH2", "W11 gain at 300 people", w["W11"]["n300"]["all"]["mean"], "0.028")
    s = {(r["stratum"], r["n"]): r for r in rows("weekly/h_strata_loo_regret.csv")}
    for n, pooled, two in (("30", "0.014", "0.104"), ("100", "0.036", "0.030"), ("300", "0.047", "0.014"), ("1000", "0.010", "0.002")):
        chk("RH2", "non-Amazon n=%s pooled-table regret" % n, num(s[("other", n)]["regret_pooled_table_loo"]), pooled)
        chk("RH2", "non-Amazon n=%s 2-rounds regret" % n, num(s[("other", n)]["regret_2rounds"]), two)


def rh3():
    d = rows("deep/d24_spearman.csv")
    pooled = pick(d, group="pooled", value="all")
    d24 = js("deep/deep_verdicts.json")["D24"]["numbers"]
    chk("RH3", "D24 cells", d24["n_cells"], "245")
    chk("RH3", "D24 cells after merging duplicates", d24["n_cells_merged"], "219")
    chk("RH3", "D24 median of the within-group-size correlations", d24["rho_median_within_n"], "0.95")
    chk("RH3", "D24 median of the within-group-size correlations (spot check)", d24["rho_median_within_n"], "0.94849")
    chk("RH3", "D24 group sizes used for that median", d24["n_values_used"], "4")
    chk("RH3", "D24 pooled rank correlation", d24["rho_pooled"], "0.94")
    chk("RH3", "D24 pooled rank correlation (table row)", num(pooled["rho"]), "0.935804")
    chk("RH3", "D24 pooled cells (table row)", num(pooled["n_cells"]), "245")
    chk("RH3", "D24 self-score over true score", d24["mean_ratio_selfnd_over_nd100"], "0.41")
    w = js("weekly/weekly_verdicts.json")
    chk("RH3", "W5 median within-log rank correlation", w["W5"]["median_rho"], "0.68")
    chk("RH3", "W5 median (spot check)", w["W5"]["median_rho"], "0.6813")
    chk("RH3", "W5 lower quartile", w["W5"]["q25"], "0.56")
    chk("RH3", "W5 upper quartile", w["W5"]["q75"], "0.75")
    chk("RH3", "W5 logs at or above 0.5", w["W5"]["share_ge_05"] * w["W5"]["n"], "22")
    chk("RH3", "W5 logs positive", w["W5"]["share_positive"] * w["W5"]["n"], "27")
    chk("RH3", "W6 alarm weeks", w["W6"]["alarms"], "405")
    chk("RH3", "W6 hits", w["W6"]["hits"], "299")
    chk("RH3", "W6 pooled share (spot check)", w["W6"]["pooled_share"], "0.7383")
    chk("RH3", "W6 pooled share (percent)", w["W6"]["pooled_share"] * 100, "74")


def rh4():
    k = {(r["round_days"], r["scope"], r["comparison"]): r for r in rows("confirm/k_paired.csv")}
    same = k[("7", "w2 all", "selector minus same")]
    chk("RH4", "selector minus same-group table", num(same["mean_diff"]), "-0.010311")
    chk("RH4", "  lower end", num(same["ci_lo"]), "-0.019297")
    chk("RH4", "  upper end", num(same["ci_hi"]), "-0.003865")
    chk("RH4", "  headline (4 dp)", num(same["mean_diff"]), "-0.0103")
    chk("RH4", "  headline lower end (4 dp)", num(same["ci_lo"]), "-0.0193")
    chk("RH4", "  headline upper end (4 dp)", num(same["ci_hi"]), "-0.0039")
    chk("RH4", "  logs where the selector is better", num(same["share_logs_selector_better"]) * int(same["logs"]), "19")
    pooled = k[("7", "w2 all", "selector minus pooled")]
    chk("RH4", "selector minus pooled table", num(pooled["mean_diff"]), "-0.0185")
    chk("RH4", "  lower end", num(pooled["ci_lo"]), "-0.0283")
    chk("RH4", "  upper end", num(pooled["ci_hi"]), "-0.0104")
    chk("RH4", "  logs where the selector is better", num(pooled["share_logs_selector_better"]) * int(pooled["logs"]), "22")
    chk("RH4", "selector minus 2 rounds", num(k[("7", "w2 all", "selector minus two")]["mean_diff"]), "-0.134")
    kv = js("confirm/k_verdicts.json")["verdicts"]
    chk("RH4", "K3 cells", kv["K3"]["cells"], "90")
    chk("RH4", "K3 cells with regret above 0.01", kv["K3"]["share_over_0_01"] * kv["K3"]["cells"], "2")
    chk("RH4", "K10 selector minus same-group table at 30 people, 28 days", kv["K10"]["mean"], "0.0081")
    chk("RH4", "K10 share of logs where the selector is worse (percent)", kv["K10"]["share_positive"] * 100, "81")
    chk("RH4", "K11 cells with selector <= same-group table", kv["K11"]["share_selector_le_same"] * kv["K11"]["cells"], "5")
    chk("RH4", "K12 re-picking minus other-window memory", kv["K12"]["mean"], "0.0011")


def rh5():
    g = {(r["comparison"], r["region"]): r for r in rows("guard/g_paired.csv")}
    m = g[("warm minus plain", "n30_28d")]
    chk("RH5", "seeded minus plain, 30 people monthly", num(m["mean_diff"]), "-0.007372")
    chk("RH5", "  lower end", num(m["ci_lo"]), "-0.009972")
    chk("RH5", "  upper end", num(m["ci_hi"]), "-0.004855")
    chk("RH5", "  logs", num(m["logs"]), "24")
    chk("RH5", "  headline (4 dp)", num(m["mean_diff"]), "-0.0074")
    chk("RH5", "  headline lower end (4 dp)", num(m["ci_lo"]), "-0.0100")
    chk("RH5", "  headline upper end (4 dp)", num(m["ci_hi"]), "-0.0049")
    w = g[("warm minus plain", "n30_7d")]
    chk("RH5", "seeded minus plain, 30 people weekly", num(w["mean_diff"]), "-0.0019")
    chk("RH5", "  lower end", num(w["ci_lo"]), "-0.0027")
    chk("RH5", "  upper end", num(w["ci_hi"]), "-0.0011")
    wk = g[("warm minus plain", "n100plus_7d")]
    chk("RH5", "100+ people weekly", num(wk["mean_diff"]), "-0.0006")
    chk("RH5", "  upper end", num(wk["ci_hi"]), "-0.0003")
    mo = g[("warm minus plain", "n100plus_28d")]
    chk("RH5", "100+ people monthly", num(mo["mean_diff"]), "-0.0007")
    chk("RH5", "  upper end", num(mo["ci_hi"]), "0.0015")
    gv = js("guard/g_verdicts.json")["verdicts"]
    chk("RH5", "G1 logs better (of 24)", gv["G1"]["share_negative"] * 24, "22")
    chk("RH5", "G2 logs better (of 24)", gv["G2"]["share_negative"] * 24, "19")


def rh6():
    sv = js("stress/s_verdicts.json")["verdicts"]
    for key, printed in (("A+", "-0.0094"), ("Ap", "-0.0037"), ("A-", "-0.0236")):
        chk("RH6", "S2 best-list gain vs random people, %s" % key, sv["S2"]["mean_best_gain_vs_std"][key], printed)
    for key, printed in (("A+", "2.26"), ("Ap", "1.16"), ("A-", "0.69")):
        chk("RH6", "S1 rows per reporter, %s" % key, sv["S1"]["ratio_rows_per_reporter"][key], printed)
    for key, printed in (("S3a", "0.0039"), ("S3b", "0.0025"), ("S3c", "0.0066")):
        chk("RH6", "%s selector regret" % key, sv[key]["mean_selector_regret_over_logs"], printed)
    vv = js("volunteer/v_verdicts.json")["verdicts"]
    for key, mean, lo, hi in (("V4", "0.0124", "0.0077", "0.0179"), ("V5", "0.0076", "-0.0009", "0.0217"), ("V7", "0.0039", "0.0013", "0.0065"), ("V8", "-0.0177", "-0.0232", "-0.0130")):
        chk("RH6", "%s mean" % key, vv[key]["mean"], mean)
        chk("RH6", "%s lower end" % key, vv[key]["lo"], lo)
        chk("RH6", "%s upper end" % key, vv[key]["hi"], hi)
    chk("RH6", "V4 logs", vv["V4"]["logs"], "24")


def rh7():
    p = {(r["block"], r["contrast"]): r for r in rows("volunteer/pilot_pool_summary.csv")}
    pooled = p[("all 27 logs", "c_aud - a150_sel")]
    chk("RH7", "pooled minus 150 random reporters", num(pooled["mean"]), "-0.0037")
    chk("RH7", "  lower end", num(pooled["ci_lo"]), "-0.0068")
    chk("RH7", "  upper end", num(pooled["ci_hi"]), "-0.0009")
    chk("RH7", "  logs positive", num(pooled["logs_positive"]), "8")
    chk("RH7", "  logs within -0.005", num(pooled["logs_within_0.005"]), "18")
    chk("RH7", "audit-only shortfall", num(p[("all 27 logs", "b_aud - a150_sel")]["mean"]), "-0.0241")
    w = p[("all 27 logs", "c_aud - c_b0_aud")]
    chk("RH7", "weights and memory chosen on the audit add", num(w["mean"]), "0.0077")
    chk("RH7", "  lower end", num(w["ci_lo"]), "0.0039")
    chk("RH7", "  upper end", num(w["ci_hi"]), "0.0118")
    chk("RH7", "  logs positive", num(w["logs_positive"]), "22")
    for contrast, mean, lo, hi in (("audit scores only (programme V design): r* = (R* - 50) / 100", "0.615", "0.529", "0.692"), ("pooled: r* = (R* - 50) / 100", "0.941", "0.895", "0.986")):
        r = p[("break-even two-point, all 27 logs", contrast)]
        chk("RH7", "break-even cost ratio", num(r["mean"]), mean)
        chk("RH7", "  lower end", num(r["ci_lo"]), lo)
        chk("RH7", "  upper end", num(r["ci_hi"]), hi)


def rh8():
    pv = js("price/p_verdicts.json")["verdicts"]
    chk("RH8", "P1 gain", pv["P1"]["mean"], "0.1910")
    chk("RH8", "  lower end", pv["P1"]["lo"], "0.1334")
    chk("RH8", "  upper end", pv["P1"]["hi"], "0.2446")
    chk("RH8", "  logs", pv["P1"]["logs"], "15")
    chk("RH8", "P9 gain", pv["P9"]["mean"], "0.0319")
    chk("RH8", "  lower end", pv["P9"]["lo"], "0.0249")
    chk("RH8", "  upper end", pv["P9"]["hi"], "0.0398")
    chk("RH8", "P10 gain at eps 2", pv["P10"]["by_eps"]["2"]["mean"], "0.0425")
    chk("RH8", "P10 gain at eps 8", pv["P10"]["by_eps"]["8"]["mean"], "0.0218")
    chk("RH8", "P6 price at eps 2, 1,000 people", pv["P6"]["price_strong"], "0.199")
    pl = {(r["n"], r["eps"], r["bound"]): r for r in rows("price/posthoc_price_list.csv")}
    for n, cap10, unit in (("1000", "0.2734", "0.0824"), ("300", "0.4455", "0.261"), ("100", "0.4837", "0.3836")):
        chk("RH8", "n=%s price, 10-row cap" % n, num(pl[(n, "4", "cap10")]["price_mean"]), cap10)
        chk("RH8", "n=%s price, unit length" % n, num(pl[(n, "4", "clip1")]["price_mean"]), unit)
    for n, hi, lo in (("1000", "0.273", "0.082"), ("300", "0.446", "0.261"), ("100", "0.484", "0.384")):
        chk("RH8", "n=%s headline price, 10-row cap" % n, num(pl[(n, "4", "cap10")]["price_mean"]), hi)
        chk("RH8", "n=%s headline price, unit length" % n, num(pl[(n, "4", "clip1")]["price_mean"]), lo)
    share = [r for r in rows("price/posthoc_share_of_gain.csv")]
    for n, low, high in (("1000", "79", "83"), ("300", "65", "77"), ("100", "39", "63")):
        vals = [num(r["share_reached_by_cap1"]) * 100 for r in share if r["n"] == n]
        chk("RH8", "n=%s one-row cap share of the gain, lowest over eps" % n, min(vals), low)
        chk("RH8", "n=%s one-row cap share of the gain, highest over eps" % n, max(vals), high)


def rh9():
    grid = rows("price/cap_dev_grid.csv")

    def lost(n, scheme):
        return num(pick(grid, n=n, scheme=scheme)["lost_share_of_std_selector"])

    for n, g_one, l_one, g_five, l_five in (
        ("100", "gaussn:eps=8,C=1,kmax=1", "lapn:eps=8,C=1,kmax=1", "gaussn:eps=4,C=1,kmax=5", "lapn:eps=2,C=1,kmax=5"),
        ("300", "gaussn:eps=8,C=1,kmax=1", "lapn:eps=8,C=1,kmax=1", "gaussn:eps=4,C=1,kmax=5", "lapn:eps=2,C=1,kmax=5"),
        ("1000", "gaussn:eps=8,C=1,kmax=1", "lapn:eps=8,C=1,kmax=1", "gaussn:eps=4,C=1,kmax=5", "lapn:eps=2,C=1,kmax=5"),
    ):
        chk_true("RH9", "n=%s one draw beats five draws, Gaussian (%.3f < %.3f)" % (n, lost(n, g_one), lost(n, g_five)), lost(n, g_one) < lost(n, g_five))
        chk_true("RH9", "n=%s one draw beats five draws, Laplace (%.3f < %.3f)" % (n, lost(n, l_one), lost(n, l_five)), lost(n, l_one) < lost(n, l_five))
    chk("RH9", "100 people, one draw, eps 8, Laplace lost share (percent)", lost("100", "lapn:eps=8,C=1,kmax=1") * 100, "17")
    chk("RH9", "100 people, one draw, eps 8, Gaussian lost share (percent)", lost("100", "gaussn:eps=8,C=1,kmax=1") * 100, "49")
    chk("RH9", "100 people, five draws, Laplace eps 2 lost share (percent)", lost("100", "lapn:eps=2,C=1,kmax=5") * 100, "63")
    chk("RH9", "100 people, five draws, Gaussian eps 4 lost share (percent)", lost("100", "gaussn:eps=4,C=1,kmax=5") * 100, "65")
    chk("RH9", "spot check, 100 people, lapn:eps=8,C=1,kmax=1 lost share", lost("100", "lapn:eps=8,C=1,kmax=1"), "0.1726")
    chk("RH9", "spot check, 100 people, gaussn:eps=4,C=1,kmax=5 lost share", lost("100", "gaussn:eps=4,C=1,kmax=5"), "0.6548")
    for n, g_pct, l_pct in (("100", "66", "41"), ("300", "42", "15"), ("1000", "13", "4")):
        chk("RH9", "n=%s per round eps 4, Gaussian lost share (percent)" % n, lost(n, "gaussn:eps=4,C=1") * 100, g_pct)
        chk("RH9", "n=%s per round eps 4, Laplace lost share (percent)" % n, lost(n, "lapn:eps=4,C=1") * 100, l_pct)
    paired = rows("price/cap_dev_paired.csv")
    ckey = [k for k in paired[0] if k.startswith("contrast")][0]

    def pc(kind, contrast, n):
        return pick([dict(r, c=r[ckey]) for r in paired], kind=kind, c=contrast, n=n)

    for n, mean, lo, hi in (("100", "0.156", "0.124", "0.187"), ("300", "0.187", "0.142", "0.229"), ("1000", "0.064", "0.038", "0.092")):
        r = pc("equal_lifetime_capped", "k=1: Laplace eps 4 minus Gaussian eps 4", n)
        chk("RH9", "one draw, Laplace minus Gaussian, n=%s" % n, num(r["mean"]), mean)
        chk("RH9", "  lower end", num(r["ci_lo"]), lo)
        chk("RH9", "  upper end", num(r["ci_hi"]), hi)
    r = pc("equal_lifetime_capped", "k=1: Laplace eps 4 minus Gaussian eps 4", "100")
    chk("RH9", "spot check mean", num(r["mean"]), "0.15635")
    chk("RH9", "spot check lower end", num(r["ci_lo"]), "0.1244")
    chk("RH9", "spot check upper end", num(r["ci_hi"]), "0.18739")
    chk("RH9", "spot check logs positive", num(r["logs_positive"]), "26")
    five = [r for r in paired if r["kind"] == "equal_lifetime_capped" and r[ckey].startswith("k=5")]
    ten = [r for r in paired if r["kind"] == "equal_lifetime_capped" and r[ckey].startswith("k=10")]
    chk("RH9", "five-draw contrasts", len(five), "6")
    chk("RH9", "five-draw smallest", min(num(r["mean"]) for r in five), "0.007")
    chk("RH9", "five-draw largest", max(num(r["mean"]) for r in five), "0.058")
    chk("RH9", "five-draw intervals above 0", sum(num(r["ci_lo"]) > 0 for r in five), "4")
    chk("RH9", "ten-draw smallest", min(num(r["mean"]) for r in ten), "-0.019")
    chk("RH9", "ten-draw largest", max(num(r["mean"]) for r in ten), "0.030")
    cost = [r for r in paired if r["kind"] == "cap_cost"]
    chk("RH9", "cap-cost contrasts", len(cost), "36")
    chk("RH9", "cap-cost smallest", min(num(r["mean"]) for r in cost), "-0.016")
    chk("RH9", "cap-cost largest", max(num(r["mean"]) for r in cost), "0.011")
    lap = rows("price/laplace_dev_paired.csv")
    ekey = [k for k in lap[0] if k.startswith("eps_per_round")][0]
    lkey = [k for k in lap[0] if k.startswith("contrast")][0]
    for n, printed in (("100", "0.190"), ("300", "0.228"), ("1000", "0.091")):
        hit = [r for r in lap if r["n"] == n and r[ekey] == "4" and r[lkey].startswith("lap c=1 - gauss c=1")]
        chk("RH9", "same one-row signal, Laplace minus Gaussian, n=%s" % n, num(hit[0]["mean"]) if len(hit) == 1 else None, printed)
    eq = {(r["contrast"], r["people_per_round_laplace"]): r for r in rows("price/cap_dev_equivalents.csv")}
    chk("RH9", "people-equivalent, one draw, eps 4, 100 people", num(eq[("cap 1: Laplace eps 4 vs Gaussian eps 4", "100")]["ratio"]), "3.1")
    chk("RH9", "people-equivalent, one draw, eps 4, 300 people", num(eq[("cap 1: Laplace eps 4 vs Gaussian eps 4", "300")]["ratio"]), "3.0")
    five_eq = [num(r["ratio"]) for (c, n), r in eq.items() if c.startswith("cap 5")]
    ten_eq = [num(r["ratio"]) for (c, n), r in eq.items() if c.startswith("cap 10")]
    chk_true("RH9", "five-draw people-equivalents within 1.2-1.5 (%.2f to %.2f)" % (min(five_eq), max(five_eq)), min(five_eq) >= 1.2 - 0.005 and max(five_eq) <= 1.5 + 0.005)
    chk_true("RH9", "ten-draw people-equivalents within 1.0-1.25 (%.2f to %.2f)" % (min(ten_eq), max(ten_eq)), min(ten_eq) >= 1.0 - 0.015 and max(ten_eq) <= 1.25 + 0.005)
    people = rows("price/cap_dev_people.csv")
    rb = pick(people, n="300", log="ratebeer", kmax="1")
    chk("RH9", "ratebeer keeps this share of people at one draw, 300 people (percent)", num(rb["ratio"]) * 100, "28")


def rh10():
    ll = {(r["n"], r["eps"], r["sampler"], r["k"]): r for r in rows("price/posthoc_list_length.csv")}
    for n, top10, top100 in (("100", "0.49", "0.42"), ("300", "0.79", "0.64"), ("1000", "0.96", "0.89")):
        chk("RH10", "n=%s top 10 kept quality" % n, num(ll[(n, "4", "clip1", "10")]["retained"]), top10)
        chk("RH10", "n=%s top 100 kept quality" % n, num(ll[(n, "4", "clip1", "100")]["retained"]), top100)
    chk("RH10", "spot check top 10 at 300 people", num(ll[("300", "4", "clip1", "10")]["retained"]), "0.794")
    chk("RH10", "spot check top 100 at 300 people", num(ll[("300", "4", "clip1", "100")]["retained"]), "0.642")
    ht = {(r["n"], r["eps"], r["sampler"]): r for r in rows("price/posthoc_list_length_head_vs_tail.csv")}
    r = ht[("300", "4", "clip1")]
    chk("RH10", "top 10 minus top 100 kept quality at 300 people", num(r["retained10_minus_retained100"]), "0.158")
    chk("RH10", "  lower end", num(r["ci_lo"]), "0.122")
    chk("RH10", "  upper end", num(r["ci_hi"]), "0.193")
    chk("RH10", "  logs", num(r["logs"]), "21")
    chk("RH10", "  share of logs positive (percent)", num(r["share_logs_positive"]) * 100, "100")


def main():
    for fn in (rh1, rh2, rh3, rh4, rh5, rh6, rh7, rh8, rh9, rh10):
        try:
            fn()
        except (OSError, KeyError, IndexError, ValueError) as exc:
            results.append(False)
            print("FAIL %s: %s: %s" % (fn.__name__.upper(), type(exc).__name__, exc))
    failed = results.count(False)
    print("\nchecked %d numbers: %d match, %d differ" % (len(results), len(results) - failed, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
