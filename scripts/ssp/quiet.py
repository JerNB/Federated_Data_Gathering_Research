"""Quiet-run governor: keeps a long CPU job below the fan, memory and swap limits.

Why: the overnight run must not wake the user. macOS starts the fans from
die temperature, so the governor reads `macmon pipe` (no sudo) every few
seconds and adapts the number of busy workers.

Rules (AIMD), temperature is smoothed over about 25 s (one-second bursts reach
80 C without the fans moving), fan RPM and memory are instantaneous:
  * hard stop  : fan RPM > fan_hard, or smoothed temp > temp_hard, or worker memory / swap /
                 disk limits broken -> workers are frozen (SIGSTOP) until it has cooled
  * back off   : fan RPM > fan_soft or temp > temp_soft -> target = target // 2; at one worker -> freeze
  * ramp up    : everything under the soft limits for `up_after_s` -> target + 1, up to max_workers
  * failsafe   : if the sensor tool dies and cannot be restarted, run on ONE worker and say so in the log

The governor can never leave workers frozen: any exit path of its thread resumes them.

Usage:
    gov = Governor(Limits.from_json("configs/ssp_quiet.json"), log_path=...)
    run_tasks(fn, tasks, gov, out_jsonl)      # resumable, one JSON row per finished task
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

# One BLAS thread per worker, set before numpy is imported anywhere.
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


@dataclass
class Limits:
    max_workers: int = 4
    fan_soft: int = 1750        # rpm; idle on this Mac is about 1350-1450
    fan_hard: int = 2100
    temp_soft: float = 72.0     # deg C, smoothed max(cpu, gpu) from macmon
    temp_hard: float = 78.0
    ram_budget_gb: float = 8.0   # total resident memory of our workers
    swap_growth_gb: float = 6.0  # freeze if swap grows this far above the start value
    min_free_disk_gb: float = 10.0
    up_after_s: int = 120
    cool_after_s: int = 45
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
        self.last: dict = {}
        self.worker_pids: Callable[[], list[int]] = lambda: []
        self._stop = False
        self._swap0 = None
        self._ema = None
        self._ok_since = time.time()
        self._cool_since = None
        self._log = open(log_path, "a") if log_path else None
        self._mac = None
        self._restarts = 0
        self._start_mac()
        self._t = threading.Thread(target=self._guarded_loop, daemon=True)
        self._t.start()

    # -- plumbing -----------------------------------------------------------
    def _start_mac(self):
        self._mac = subprocess.Popen(["macmon", "pipe", "-i", str(int(self.lim.poll_s * 1000))],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def alive(self) -> bool:
        return self._t.is_alive()

    def close(self):
        self._stop = True
        try:
            self._mac.kill()
        except Exception:
            pass
        self._signal(signal.SIGCONT)

    def _signal(self, sig):
        try:
            pids = list(self.worker_pids())
        except Exception:
            pids = []
        for pid in pids:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass

    def _note(self, **kw):
        if self._log:
            self._log.write(json.dumps(dict(t=round(time.time()), **kw)) + "\n")
            self._log.flush()

    def _guarded_loop(self):
        try:
            self._loop()
        finally:
            # Whatever happens, never leave workers frozen.
            self.state, self.target = "run", 1
            self._signal(signal.SIGCONT)
            self._note(event="governor thread exited", state="run", target=1)

    # -- control loop -------------------------------------------------------
    def _loop(self):
        lim = self.lim
        while not self._stop:
            line = self._mac.stdout.readline()
            if not line:                          # sensor tool died
                if self._stop:
                    return
                self._restarts += 1
                if self._restarts > 5:
                    self.state, self.target = "run", 1
                    self._signal(signal.SIGCONT)
                    self._note(event="sensor lost for good: failsafe single worker", state="run", target=1)
                    while not self._stop:         # keep one worker, check disk only
                        time.sleep(10)
                        if free_disk_gb() < lim.min_free_disk_gb:
                            self.target = 0
                    return
                self._note(event="macmon restarted", n=self._restarts)
                time.sleep(2)
                self._start_mac()
                continue
            try:
                d = json.loads(line)
                fan = max([f["rpm"] for f in d["fans"]] or [0])
                raw = max(d["temp"]["cpu_temp_avg"], d["temp"]["gpu_temp_avg"])
                swap = d["memory"]["swap_usage"] / 2**30
                cpu_w = d["cpu_power"]
            except Exception as e:                # a bad line must not kill the control loop
                self._note(event="bad sensor line", err=repr(e)[:120])
                continue
            self._ema = raw if self._ema is None else self._ema + 0.12 * (raw - self._ema)
            temp = self._ema
            self._swap0 = swap if self._swap0 is None else self._swap0
            try:
                rss = _worker_rss_gb(list(self.worker_pids()))
            except Exception:
                rss = 0.0
            disk = free_disk_gb()
            now = time.time()
            hard = (fan > lim.fan_hard or temp > lim.temp_hard or rss > lim.ram_budget_gb * 1.2
                    or swap > self._swap0 + lim.swap_growth_gb or disk < lim.min_free_disk_gb)
            soft = fan > lim.fan_soft or temp > lim.temp_soft or rss > lim.ram_budget_gb
            cool = (fan <= lim.fan_soft - 100 and temp <= lim.temp_soft - 5 and rss <= lim.ram_budget_gb
                    and swap <= self._swap0 + lim.swap_growth_gb and disk >= lim.min_free_disk_gb)
            if hard or (soft and self.target <= 1 and self.state == "run"):
                # Too hot even at one worker: freeze the workers and wait to cool.
                if self.state != "paused":
                    self._note(event="freeze", fan=fan, temp=round(temp, 1), rss=round(rss, 2),
                               swap=round(swap, 2), disk=round(disk, 1))
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
            self.last = dict(state=self.state, target=self.target, fan=fan, temp=round(temp, 1),
                             raw_temp=round(raw, 1), cpu_w=round(cpu_w, 1), rss_gb=round(rss, 2),
                             swap_gb=round(swap, 2), disk_gb=round(disk, 1))
            self._note(**self.last)


# ---------------------------------------------------------------------------
# resumable pool

def _load_done(path: str, key: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for ln in f:
                try:
                    done.add(json.loads(ln)[key])
                except Exception:
                    pass
    return done


def run_tasks(fn: Callable[[dict], dict], tasks: Iterable[dict], gov: Governor,
              out_jsonl: str, key: str = "key", max_attempts: int = 2) -> int:
    """Run `fn(task)` for every task whose key is not already in out_jsonl.

    Successful rows go to out_jsonl (one JSON line each). Failures go to out_jsonl + '.errors' and are
    retried up to `max_attempts` times in this run; a failed key is NOT counted as done, so a later run
    retries it. A crashed worker pool (e.g. a killed worker) is rebuilt and the in-flight tasks re-queued.
    Returns the number of tasks finished successfully in this run.
    """
    tasks = list(tasks)
    done = _load_done(out_jsonl, key)
    todo = [t for t in tasks if t[key] not in done]
    attempts: dict = {}
    n_ok = 0
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    out = open(out_jsonl, "a")
    err = open(out_jsonl + ".errors", "a")
    pool_restarts = 0
    blame: dict = {}
    try:
        while todo:
            if not gov.alive():
                raise RuntimeError("governor thread is dead; refusing to run without thermal control")
            ex = ProcessPoolExecutor(max_workers=gov.lim.max_workers)
            gov.worker_pids = lambda ex=ex: [p.pid for p in list(getattr(ex, "_processes", {}).values())]
            running: dict = {}
            queue = list(todo)
            todo = []
            try:
                while queue or running:
                    while queue and len(running) < gov.target:
                        t = queue.pop(0)
                        running[ex.submit(fn, t)] = t
                    if not running:
                        time.sleep(2)
                        if not gov.alive():
                            raise RuntimeError("governor thread is dead")
                        continue
                    fin, _ = wait(list(running), timeout=5, return_when=FIRST_COMPLETED)
                    for f in fin:
                        t = running.pop(f)
                        try:
                            row = f.result()
                        except BrokenProcessPool:
                            running[f] = t          # keep it so it is re-queued below
                            raise
                        except Exception as e:
                            attempts[t[key]] = attempts.get(t[key], 0) + 1
                            err.write(json.dumps({key: t[key], "error": repr(e)[:300],
                                                  "attempt": attempts[t[key]]}) + "\n")
                            err.flush()
                            if attempts[t[key]] < max_attempts:
                                queue.append(t)
                            continue
                        row[key] = t[key]
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        n_ok += 1
            except BrokenProcessPool:
                pool_restarts += 1
                if pool_restarts > 20:
                    raise
                err.write(json.dumps({"event": "pool rebuilt", "n": pool_restarts}) + "\n")
                err.flush()
                # Re-queue everything unfinished on a fresh pool. A task in flight at three crashes is
                # blamed, logged as failed and dropped (it would otherwise crash every new pool).
                todo = list(queue)
                for t in running.values():
                    blame[t[key]] = blame.get(t[key], 0) + 1
                    if blame[t[key]] >= 3:
                        err.write(json.dumps({key: t[key], "error": "worker pool crashed 3 times on this task"}) + "\n")
                        err.flush()
                    else:
                        todo.append(t)
                gov._signal(signal.SIGCONT)
            finally:
                ex.shutdown(wait=False, cancel_futures=True)
    finally:
        out.close()
        err.close()
    return n_ok


if __name__ == "__main__":
    print(json.dumps(asdict(Limits()), indent=2))
