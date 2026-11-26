import resource
import threading

import psutil


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
