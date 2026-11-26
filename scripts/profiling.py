import resource

import psutil


def peak_ram_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def used_cpu_seconds(process: psutil.Process) -> float:
    times = process.cpu_times()
    return times.user + times.system
