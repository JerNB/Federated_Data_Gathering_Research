"""Summarise how quiet the overnight run was: fan speed, temperature, workers, memory, disk.

Reads  results/small_sample_popularity/00_safety/governor_log.jsonl and calibration.csv
Writes 00_safety/safety_summary.json and 04_figures/safety_fans_and_temperature.png

The calibration file is the pre-run test (matrix-multiply load): fans idle near 1,450 rpm; two heavy workers pushed
them to about 2,400 rpm and four to above 3,000 rpm within a few minutes. The governor exists because of that.
"""
import csv
import json
import os
from collections import Counter

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = "results/small_sample_popularity"


def main():
    rows = [json.loads(l) for l in open(f"{BASE}/00_safety/governor_log.jsonl")]
    st = [r for r in rows if "state" in r and "fan" in r]
    ev = [r for r in rows if "event" in r]
    t = np.array([r["t"] for r in st], float)
    fan = np.array([r["fan"] for r in st], float)
    temp = np.array([r["temp"] for r in st], float)
    raw = np.array([r["raw_temp"] for r in st], float)
    tgt = np.array([r["target"] for r in st], int)
    state = [r["state"] for r in st]
    gaps = np.diff(t)
    active = float(gaps[gaps < 60].sum())
    cal = list(csv.DictReader(open(f"{BASE}/00_safety/calibration.csv")))
    by = {}
    for r in cal:
        by.setdefault(int(r["workers"]), []).append(r)
    cal_sum = {w: dict(last_fan_mean=float(np.mean([float(x["fan_max_rpm"]) for x in v[-8:]])),
                       last_temp_mean=float(np.mean([float(x["cpu_temp"]) for x in v[-8:]])),
                       max_fan=max(float(x["fan_max_rpm"]) for x in v)) for w, v in by.items()}
    summ = dict(
        governed_hours=round(active / 3600, 2), samples=len(st),
        fan_rpm=dict(median=float(np.median(fan)), p99=float(np.percentile(fan, 99)), max=float(fan.max()),
                     share_above_1750=float((fan > 1750).mean()), share_above_2100=float((fan > 2100).mean())),
        smoothed_temp_c=dict(median=float(np.median(temp)), p99=float(np.percentile(temp, 99)), max=float(temp.max())),
        raw_temp_c=dict(max=float(raw.max()), share_above_78=float((raw > 78).mean())),
        workers_target=dict(Counter(map(str, tgt.tolist()))),
        freezes=sum(1 for e in ev if e.get("event") == "freeze"),
        governor_events=Counter(e.get("event") for e in ev),
        swap_gb_first_last=[st[0]["swap_gb"], st[-1]["swap_gb"]], disk_free_gb_min=float(min(r["disk_gb"] for r in st)),
        worker_memory_gb_max=float(max(r["rss_gb"] for r in st)),
        calibration_before_run=cal_sum,
    )
    json.dump(summ, open(f"{BASE}/00_safety/safety_summary.json", "w"), indent=1, default=str)
    hrs = (t - t[0]) / 3600
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    ax[0].plot(hrs, fan, lw=0.5, color="#0072B2")
    ax[0].axhline(1750, color="#D55E00", ls="--", lw=0.8, label="slow-down limit 1,750 rpm")
    ax[0].axhline(2100, color="#CC0000", ls="--", lw=0.8, label="freeze limit 2,100 rpm")
    ax[0].axhline(2360, color="grey", ls=":", lw=0.8, label="2 heavy workers, measured before the run: about 2,360 rpm")
    ax[0].set_ylim(1200, 3300)
    ax[0].set_ylabel("fan speed (rpm)")
    ax[0].legend(fontsize=7, loc="upper right")
    ax[0].set_title(f"Fans stayed near idle for {summ['governed_hours']} governed hours: median {summ['fan_rpm']['median']:.0f} rpm, "
                    f"99th percentile {summ['fan_rpm']['p99']:.0f}, maximum {summ['fan_rpm']['max']:.0f}", fontsize=9)
    ax[1].plot(hrs, raw, lw=0.3, color="#bbbbbb", label="chip temperature, raw")
    ax[1].plot(hrs, temp, lw=0.8, color="#009E73", label="smoothed (what the governor uses)")
    ax[1].axhline(72, color="#D55E00", ls="--", lw=0.8)
    ax[1].axhline(78, color="#CC0000", ls="--", lw=0.8)
    ax[1].set_ylabel("temperature (deg C)")
    ax[1].set_xlabel("hours since the run started")
    ax[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    os.makedirs(f"{BASE}/04_figures", exist_ok=True)
    fig.savefig(f"{BASE}/04_figures/safety_fans_and_temperature.png", dpi=130)
    print(json.dumps({k: summ[k] for k in ("governed_hours", "fan_rpm", "smoothed_temp_c", "freezes", "workers_target", "disk_free_gb_min")}, indent=1, default=str))


if __name__ == "__main__":
    main()
