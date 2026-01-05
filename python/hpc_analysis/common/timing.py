import statistics
import time
from typing import Callable, Dict, List

from .cuda import have_cuda, require_torch_cuda


def measure_cuda(
    run_fn: Callable[[], None],
    warmup: int = 5,
    iters: int = 50,
) -> Dict[str, float]:
    """
    Measure a callable using CUDA events when available, otherwise wall clock.
    Returns a dict with times and simple stats in milliseconds.
    """
    if have_cuda():
        import torch

        require_torch_cuda()
        torch.cuda.synchronize()
        for _ in range(max(warmup, 0)):
            run_fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times: List[float] = []
        for _ in range(max(iters, 1)):
            start.record()
            run_fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
    else:
        times = _measure_wall_clock(run_fn, warmup, iters)

    mean_ms = statistics.mean(times)
    min_ms = min(times)
    max_ms = max(times)
    std_ms = statistics.pstdev(times) if len(times) > 1 else 0.0
    return {
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_ms": std_ms,
        "samples": len(times),
    }


def _measure_wall_clock(run_fn: Callable[[], None], warmup: int, iters: int) -> List[float]:
    for _ in range(max(warmup, 0)):
        run_fn()
    times: List[float] = []
    for _ in range(max(iters, 1)):
        t0 = time.perf_counter()
        run_fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times
