"""Kernel-agnostic benchmarking utilities for JAX on TPU."""
import time
import numpy as np


def bench(fn, warmup=5, iters=20):
    """Benchmark a JAX function. Returns list of times in milliseconds.

    fn must return a JAX array (or pytree) supporting block_until_ready().
    """
    for _ in range(warmup):
        out = fn()
        out.block_until_ready()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        out.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def compute_stats(times_ms):
    """Compute summary statistics from a list of times in ms."""
    t = np.array(times_ms)
    return {
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "median": float(np.median(t)),
        "std": float(t.std()),
        "p5": float(np.percentile(t, 5)),
        "p95": float(np.percentile(t, 95)),
        "n": len(t),
    }


def format_stats(stats, label=""):
    """Format stats dict as a readable string."""
    prefix = f"{label}: " if label else ""
    return (
        f"{prefix}"
        f"median={stats['median']:.3f}ms  "
        f"mean={stats['mean']:.3f}ms  "
        f"min={stats['min']:.3f}ms  "
        f"std={stats['std']:.3f}ms  "
        f"p5={stats['p5']:.3f}ms  "
        f"p95={stats['p95']:.3f}ms  "
        f"(n={stats['n']})"
    )
