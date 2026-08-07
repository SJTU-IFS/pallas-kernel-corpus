"""JAXBench's own JAX reference for the dense GEMM in this directory.

This is `JAXBench/benchmark/8p_GEMM/baseline.py` at the commit below, unmodified apart from this
header: the file JAXBench itself measures `optimized.py` against.  It is carried
rather than rewritten because `create_inputs` is part of the contract -- it
seeds with `jax.random.key(42)` and scales the second operand by 0.02, and a
reference generating unscaled operands would be comparing different numbers,
not a different implementation.

`workload` here is `jnp.dot`, which on TPU lowers to an XLA dot rather than a
Mosaic custom call, so it is a valid non-Pallas reference; a test in
tests/test_dense_matmul_tpu.py asserts that.

Source:
  repository: https://github.com/AI-Hypercomputer/accelerator-agents
  commit: 6b6c44293c43976032ba12d2f72d6bebeaf2394f
  path: JAXBench/benchmark/8p_GEMM/baseline.py
"""

SOURCE = {
    "kind": "reference",
    "repository": "https://github.com/AI-Hypercomputer/accelerator-agents",
    "commit": "6b6c44293c43976032ba12d2f72d6bebeaf2394f",
    "path": "JAXBench/benchmark/8p_GEMM/baseline.py",
    "backend": "jax",
    "target": "portable",
    "contracts": ("dense_matmul_2d",),
}

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gemm_llama70b',
    'model': 'Llama-3.1-70B',
    'operator': 'dense_matmul',
    'M': 8192,
    'K': 8192,
    'N': 28672,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (A, B) matrices."""
    key = jax.random.key(42)
    k1, k2 = jax.random.split(key, 2)
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    A = jax.random.normal(k1, (M, K), dtype=dtype)
    B = jax.random.normal(k2, (K, N), dtype=dtype) * 0.02
    return A, B


def workload(A, B):
    """Dense matmul: C = A @ B"""
    return jnp.dot(A, B)


def benchmark(num_warmup=5, num_iters=100):
    """Benchmark and return results dict."""
    import time
    inputs = create_inputs()
    fn = jax.jit(workload)
    for _ in range(num_warmup):
        out = fn(*inputs)
        out.block_until_ready()
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = fn(*inputs)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)
    import numpy as np
    times = np.array(times) * 1000
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    flops = 2 * M * K * N
    avg = float(np.mean(times))
    return {
        'name': CONFIG['name'],
        'model': CONFIG['model'],
        'operator': CONFIG['operator'],
        'config': {k: v for k, v in CONFIG.items() if k not in ('name', 'model', 'operator')},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times)), 4),
        'tflops': round(flops / (avg / 1000) / 1e12, 2),
        'output_shape': list(out.shape),
        'status': 'success',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))
