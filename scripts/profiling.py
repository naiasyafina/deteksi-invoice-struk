import json
import platform
import resource
import threading
import time
from pathlib import Path

import psutil

RESULTS = Path("results")


def peak_ram_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def rss_mb(process: psutil.Process) -> float:
    return process.memory_info().rss / 2**20


def used_cpu_seconds(process: psutil.Process) -> float:
    times = process.cpu_times()
    return times.user + times.system


class PeakRss:
    def __init__(self, process: psutil.Process, interval: float = 0.005):
        self.process, self.interval = process, interval
        self.peak_mb = 0.0
        self._stop = threading.Event()

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, rss_mb(self.process))
            self._stop.wait(self.interval)

    def __enter__(self) -> "PeakRss":
        self.peak_mb = rss_mb(self.process)
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, rss_mb(self.process))


def measure(process: psutil.Process, fn, *args):
    with PeakRss(process) as peak:
        wall, cpu = time.perf_counter(), used_cpu_seconds(process)
        result = fn(*args)
        wall, cpu = time.perf_counter() - wall, used_cpu_seconds(process) - cpu
    return result, {"seconds": round(wall, 3), "cpu_percent": round(cpu / wall * 100) if wall else 0, "peak_rss_mb": round(peak.peak_mb, 1)}


def machine() -> dict:
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), platform.processor())
    return {
        "cpu": cpu,
        "logical_cpus": psutil.cpu_count(),
        "physical_cores": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": platform.platform(),
        "python": platform.python_version(),
    }


def save_run(name: str, rows: list[dict], meta: dict) -> None:
    import pandas as pd

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / f"{name}.csv", index=False)
    meta = {**meta, "peak_process_rss_mb": peak_ram_mb(), "machine": machine()}
    (RESULTS / f"{name}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
