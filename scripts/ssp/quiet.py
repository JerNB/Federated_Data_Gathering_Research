"""Quiet-run governor: keeps a long CPU job below the fan, memory and swap limits.

Why: the overnight run must not wake the user. macOS starts the fans from
die temperature, so the governor reads `macmon pipe` (no sudo) every few
seconds and adapts the number of busy workers.

Rules (AIMD):
  * hard stop  : fan RPM > fan_hard, or temp > temp_hard, or free RAM / swap
                 limits broken  -> target workers = 0 until it has cooled
  * back off   : fan RPM > fan_soft or temp > temp_soft -> target = target // 2
  * ramp up    : everything under soft limits for `up_after` seconds
                 -> target + 1, never above max_workers

Usage:
    gov = Governor(Limits.from_json("configs/ssp_quiet.json"))
    run_tasks(fn, tasks, gov, out_jsonl)      # resumable, one JSON row per task
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

# One BLAS thread per worker, set before numpy is imported anywhere.
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


@dataclass
class Limits:
    max_workers: int = 4
    fan_soft: int = 2200        # rpm; idle on this Mac is about 1350-1450
    fan_hard: int = 2800
    temp_soft: float = 68.0     # deg C, cpu_temp_avg from macmon
    temp_hard: float = 76.0
    ram_budget_gb: float = 10.0  # total resident memory of our workers
    swap_growth_gb: float = 2.0  # stop if swap grows this far above the start value
    min_free_disk_gb: float = 10.0
    up_after_s: int = 90
    cool_after_s: int = 60
    poll_s: float = 3.0

    @classmethod
    def from_json(cls, path: str) -> "Limits":
        with open(path) as f:
            return cls(**json.load(f))


def _worker_rss_gb(pids: list[int]) -> float:
    if not pids:
        return 0.0
    out = subprocess.run(["ps", "-o", "rss=", "-p", ",".join(map(str, pids))],
                         capture_output=True, text=True).stdout.split()
    return sum(int(x) for x in out) / 2**20


def free_disk_gb(path: str = ".") -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2**30


class Governor:
    """Background thread; `target` is the number of workers allowed to run now."""

    def __init__(self, limits: Limits, log_path: str | None = None):
        self.lim = limits
        self.target = max(1, limits.max_workers // 2)
        self.state = "run"
        self.last = {}
        self.worker_pids: Callable[[], list[int]] = lambda: []
        self._stop = False
        self._swap0 = None
        self._ok_since = time.time()
        self._cool_since = None
        self._log = open(log_path, "a") if log_path else None
        self._mac = subprocess.Popen(["macmon", "pipe", "-i", str(int(limits.poll_s * 1000))],
                                     stdout=subprocess.PIPE, text=True)
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def close(self):
        self._stop = True
        self._mac.kill()

    def _signal(self, sig):
        for pid in self.worker_pids():
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def _loop(self):
        lim = self.lim
        while not self._stop:
            line = self._mac.stdout.readline()
            if not line:
                break
            d = json.loads(line)
            fan = max([f["rpm"] for f in d["fans"]] or [0])
            temp = max(d["temp"]["cpu_temp_avg"], d["temp"]["gpu_temp_avg"])
            swap = d["memory"]["swap_usage"] / 2**30
            self._swap0 = swap if self._swap0 is None else self._swap0
            rss = _worker_rss_gb(self.worker_pids())
            disk = free_disk_gb()
            now = time.time()
            hard = (fan > lim.fan_hard or temp > lim.temp_hard or rss > lim.ram_budget_gb * 1.2
                    or swap > self._swap0 + lim.swap_growth_gb or disk < lim.min_free_disk_gb)
            soft = fan > lim.fan_soft or temp > lim.temp_soft or rss > lim.ram_budget_gb
            cool = (fan <= lim.fan_soft - 100 and temp <= lim.temp_soft - 5 and rss <= lim.ram_budget_gb
                    and swap <= self._swap0 + lim.swap_growth_gb and disk >= lim.min_free_disk_gb)
            if hard or (soft and self.target <= 1 and self.state == "run"):
                # Too hot even at one worker: freeze the workers (SIGSTOP) and wait to cool.
                self.state, self.target = "paused", 0
                self._signal(signal.SIGSTOP)
                self._cool_since = None
                self._ok_since = now
            elif self.state == "paused":
                if cool:
                    self._cool_since = self._cool_since or now
                    if now - self._cool_since >= lim.cool_after_s:
                        self.state, self.target = "run", 1
                        self._signal(signal.SIGCONT)
                        self._ok_since = now
                else:
                    self._cool_since = None
            elif soft:
                self.target = max(1, self.target // 2)
                self._ok_since = now
            elif now - self._ok_since >= lim.up_after_s and self.target < lim.max_workers:
                self.target += 1
                self._ok_since = now
            self.last = dict(t=round(now), state=self.state, target=self.target, fan=fan,
                             temp=round(temp, 1), cpu_w=round(d["cpu_power"], 1),
                             rss_gb=round(rss, 2), swap_gb=round(swap, 2), disk_gb=round(disk, 1))
            if self._log:
                self._log.write(json.dumps(self.last) + "\n")
                self._log.flush()


def run_tasks(fn: Callable[[dict], dict], tasks: Iterable[dict], gov: Governor,
              out_jsonl: str, key: str = "key") -> int:
    """Run `fn(task)` for every task whose task[key] is not already in out_jsonl.

    Each result row is appended as one JSON line. `fn` must be a top-level,
    picklable function and should take at most a few minutes, so that a lowered
    target takes effect quickly. Returns the number of tasks run.
    """
    done = set()
    if os.path.exists(out_jsonl):
        with open(out_jsonl) as f:
            for ln in f:
                try:
                    done.add(json.loads(ln)[key])
                except Exception:
                    pass
    todo = [t for t in tasks if t[key] not in done]
    n = 0
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    with ProcessPoolExecutor(max_workers=gov.lim.max_workers) as ex, open(out_jsonl, "a") as out:
        running: dict = {}
        gov.worker_pids = lambda: [p.pid for p in getattr(ex, "_processes", {}).values()]
        it = iter(todo)
        pending = True
        while pending or running:
            while pending and len(running) < gov.target:
                try:
                    t = next(it)
                except StopIteration:
                    pending = False
                    break
                running[ex.submit(fn, t)] = t
            if not running:
                time.sleep(2)
                continue
            fin, _ = wait(list(running), timeout=5, return_when=FIRST_COMPLETED)
            for f in fin:
                t = running.pop(f)
                try:
                    row = f.result()
                    row[key] = t[key]
                except Exception as e:  # keep the queue alive; record the failure
                    row = {key: t[key], "error": repr(e)[:300]}
                out.write(json.dumps(row) + "\n")
                out.flush()
                n += 1
    return n


if __name__ == "__main__":
    print(json.dumps(asdict(Limits()), indent=2))
