"""Measure how fan RPM and die temperature respond to W matmul workers.

Run once before an overnight job; the result decides the worker cap.
Each step runs W single-thread numpy matmul workers at nice 10, samples
`macmon pipe` every 2 s, and writes one CSV row per sample.

    .venv/bin/python3 scripts/ssp/calibrate_fans.py --steps 0,2,4,6 --seconds 150
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

WORKER = (
    "import os;os.environ['VECLIB_MAXIMUM_THREADS']='1';"
    "import numpy as np;a=np.random.rand(1200,1200).astype('f4');b=a.copy()\n"
    "while True: a=(a@b)*1e-3+b\n"
)


def sample(proc: subprocess.Popen) -> dict:
    d = json.loads(proc.stdout.readline())
    fans = [f["rpm"] for f in d["fans"]]
    return {
        "cpu_temp": d["temp"]["cpu_temp_avg"],
        "gpu_temp": d["temp"]["gpu_temp_avg"],
        "fan_max_rpm": max(fans) if fans else 0,
        "cpu_power": d["cpu_power"],
        "sys_power": d["sys_power"],
        "ram_used_gb": d["memory"]["ram_usage"] / 2**30,
        "swap_used_gb": d["memory"]["swap_usage"] / 2**30,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="0,2,4,6")
    ap.add_argument("--seconds", type=int, default=150)
    ap.add_argument("--out", default="results/small_sample_popularity/00_safety/calibration.csv")
    ap.add_argument("--abort-rpm", type=int, default=3200, help="stop the ramp if fans exceed this")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    mac = subprocess.Popen(["macmon", "pipe", "-i", "2000"], stdout=subprocess.PIPE, text=True)
    rows = []
    try:
        for w in [int(x) for x in a.steps.split(",")]:
            procs = [
                subprocess.Popen(["nice", "-n", "10", sys.executable, "-c", WORKER])
                for _ in range(w)
            ]
            t0 = time.time()
            try:
                while time.time() - t0 < a.seconds:
                    s = sample(mac)
                    s.update(workers=w, t=round(time.time() - t0))
                    rows.append(s)
                    tail.append(s)
                    print(w, s["t"], round(s["cpu_temp"], 1), s["fan_max_rpm"], round(s["cpu_power"], 1), flush=True)
                    if s["fan_max_rpm"] > a.abort_rpm:
                        print("ABORT: fans above", a.abort_rpm)
                        raise SystemExit(2)
            finally:
                for p in procs:
                    p.kill()
                for p in procs:
                    p.wait()
            last = tail[-8:]
            print(f"== W={w}: last 16s mean temp {sum(x['cpu_temp'] for x in last)/len(last):.1f} "
                  f"fan {sum(x['fan_max_rpm'] for x in last)/len(last):.0f} rpm", flush=True)
    finally:
        mac.kill()
        keys = ["workers", "t", "cpu_temp", "gpu_temp", "fan_max_rpm", "cpu_power", "sys_power", "ram_used_gb", "swap_used_gb"]
        with open(a.out, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(round(r[k], 3)) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
