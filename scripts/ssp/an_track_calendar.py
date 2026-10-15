"""Post-hoc check (exploratory): is the best half-life stable in CALENDAR days when the window width changes?

For each log and n in {30, 100}: the grid-best fixed half-life (argmax_h over 0.25..8, inf) from the cell tables of the
runs with 20, 30 and 60 windows (extension eligibility), converted to days with the window length. The grid is a factor-2
grid, so values within a factor of 2 are not distinguishable.
Writes <out>/03_tables/an_track_calendar.csv and <out>/04_figures/an_track_calendar.png
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "results/small_sample_popularity"
T = f"{OUT}/03_tables"


def rd(f):
    return list(csv.DictReader(open(f"{T}/{f}")))


def main():
    cells = {w: rd(f"an_track_cells_w{w}_steps.csv") for w in (20, 30, 60)}
    days = {w: {r["dataset"]: float(r["window_days"]) for r in rd(f"an_track_m0m1_w{w}_steps.csv")} for w in (20, 30, 60)}
    rows = []
    for n in ("30", "100"):
        for d in sorted({r["dataset"] for r in cells[30]}):
            rec = dict(dataset=d, n=int(n))
            for w in (20, 30, 60):
                r = [x for x in cells[w] if x["dataset"] == d and x["n"] == n]
                if r:
                    h = float("inf") if r[0]["argmax_h"] == "inf" else float(r[0]["argmax_h"])
                    rec[f"best_h_w{w}"] = h
                    rec[f"best_days_w{w}"] = h * days[w][d]
                else:
                    rec[f"best_h_w{w}"] = rec[f"best_days_w{w}"] = ""
            vals = [rec[f"best_days_w{w}"] for w in (20, 30, 60) if rec[f"best_days_w{w}"] != "" and np.isfinite(rec[f"best_days_w{w}"])]
            rec["max_over_min_days"] = (max(vals) / min(vals)) if len(vals) >= 2 and min(vals) > 0 else ""
            rows.append(rec)
    with open(f"{T}/an_track_calendar.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ratio = [r["max_over_min_days"] for r in rows if r["n"] == 100 and r["max_over_min_days"] != ""]
    print("n=100 logs with >=2 widths:", len(ratio), "median max/min days ratio:", float(np.median(ratio)) if ratio else None,
          "share within factor 2:", float(np.mean([x <= 2.0 for x in ratio])) if ratio else None)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    r100 = [r for r in rows if r["n"] == 100]
    x = np.arange(len(r100))
    for j, w in enumerate((20, 30, 60)):
        v = [r[f"best_days_w{w}"] if r[f"best_days_w{w}"] != "" and np.isfinite(r[f"best_days_w{w}"]) else np.nan for r in r100]
        ax.bar(x + (j - 1) * 0.27, v, 0.27, label=f"{w} windows")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([r["dataset"].replace("amazon-", "a-") for r in r100], rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("grid-best half-life in days (n = 100)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/04_figures/an_track_calendar.png", dpi=130)


if __name__ == "__main__":
    main()
