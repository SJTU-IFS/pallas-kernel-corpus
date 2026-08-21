"""Correctness for tpu-inference's structured sparse matmul (N:M sparsity).

The kernel never sees a dense matrix. It takes the compressed `(nonzeros,
metadata)` pair `Sparsifier` produces — values with the pruned entries squeezed
out, plus bit-packed indices saying where each survivor sat — and rebuilds the
tiles it needs inside the kernel. So the test has to do two separate things:
build the compressed operands the way upstream does, and densify the *same*
matrices to get something `jnp.dot` can check against.

Both halves are upstream's. `Sparsifier` and `gen_sparse_mask` come from the
kernel file, and the reference is `jnp.dot` on the densified operands, which is
what `tests/kernels/spmm_v1_test.py` compares against.

The CPU half runs the same Pallas program under the interpreter — a real check
of the flattening and the compression logic, but not a Mosaic lowering, so it
does not make the kernel migrated. That needs the TPU half.

Run the CPU half anywhere::

    pytest tests/test_structured_sparse_matmul.py -q -k cpu
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax import random


ROOT = Path(__file__).parents[1]
FAMILY = ROOT / "kernels" / "matmul" / "structured_sparse_matmul"

HAS_TPU = any("TPU" in d.device_kind for d in jax.devices())

#: Upstream's tolerance, from spmm_v1_test.py.
ATOL = RTOL = 5e-3
#: 2 nonzeros in every 4 — the sparsity every current TPU generation accelerates.
SPARSITY = (2, 4)
STRIDE = 128

#: All eight ways the sparsity can be arranged. Upstream sweeps these because
#: which operand is sparse, and whether the sparsity runs along the contracting
#: dimension, change how the metadata is laid out and decompressed.
ARRANGEMENTS = [(r, c, t) for r in (True, False)
                for c in (True, False) for t in (True, False)]


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, FAMILY / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    return (load("spmm_kernel", "tpu_inference_optimized.py"),
            load("spmm_baseline", "baseline.py"))


def build_case(kernel, baseline, rhs_sparse, contract_sparse, rhs_transpose,
               m=256, k=256, n=256, dtype=jnp.bfloat16):
    """Upstream's own shape arithmetic: the sparse axis must fit whole strides."""
    _, y = SPARSITY
    block_m = block_n = 128
    if rhs_sparse:
        if contract_sparse:
            sparse_dim, k = int(rhs_transpose), math.lcm(STRIDE * y, k)
        else:
            sparse_dim = int(not rhs_transpose)
            n, block_n = math.lcm(STRIDE * y, n), math.lcm(STRIDE * y, block_n)
    else:
        if contract_sparse:
            sparse_dim, k = 1, math.lcm(STRIDE * y, k)
        else:
            sparse_dim = 0
            m, block_m = math.lcm(STRIDE * y, m), math.lcm(STRIDE * y, block_m)

    key = random.PRNGKey(123)
    lhs = (random.normal(key, (m, k), jnp.float32) / 100).astype(dtype)
    rhs_shape = (n, k) if rhs_transpose else (k, n)
    rhs = (random.normal(key, rhs_shape, jnp.float32) / 100).astype(dtype)
    out_dtype = kernel._infer_out_dtype(dtype, dtype)
    default_value = 0.0

    target = rhs if rhs_sparse else lhs
    mask = kernel.gen_sparse_mask(key, target.shape, SPARSITY,
                                  sparse_dim=sparse_dim, stride=STRIDE)
    sparsifier = kernel.Sparsifier(target, mask, sparsity=SPARSITY,
                                   sparse_dim=sparse_dim, stride=STRIDE)
    if rhs_sparse:
        mat, rhs = lhs, baseline.densify(rhs, mask, default_value)
    else:
        mat, lhs = rhs, baseline.densify(lhs, mask, default_value)

    call = dict(rhs_sparse=rhs_sparse, contract_sparse=contract_sparse,
                rhs_transpose=rhs_transpose, stride=STRIDE, block_m=block_m,
                block_k=k, block_n=block_n, default_value=default_value,
                out_dtype=out_dtype)
    return sparsifier, mat, lhs, rhs, call, out_dtype, mask


def check(kernel, baseline, arrangement, interpret):
    rhs_sparse, contract_sparse, rhs_transpose = arrangement
    sparsifier, mat, lhs, rhs, call, out_dtype, _ = build_case(
        kernel, baseline, rhs_sparse, contract_sparse, rhs_transpose)

    launch = kernel.structured_spmm
    if interpret:
        launch = _with_interpret(kernel)
    actual = launch(SPARSITY, sparsifier.nonzeros, sparsifier.metadata, mat, **call)
    expected = baseline.dense_matmul(lhs, rhs, rhs_transpose=rhs_transpose,
                                     out_dtype=out_dtype)

    actual = np.asarray(actual, np.float32)
    expected = np.asarray(expected, np.float32)
    assert actual.shape == expected.shape, arrangement
    peak = float(np.max(np.abs(expected)))
    assert not np.allclose(np.zeros_like(expected), expected, atol=ATOL, rtol=RTOL), (
        f"{arrangement}: a zeroed kernel would pass this comparison")
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL,
                               err_msg=f"{arrangement}, peak {peak:.4g}")


def _with_interpret(kernel):
    """Force interpret mode without editing the carried upstream file."""
    from jax.experimental import pallas as pl

    original = pl.pallas_call

    def forced(*args, **kwargs):
        return original(*args, **{**kwargs, "interpret": True})

    def run(*args, **kwargs):
        pl.pallas_call = forced
        try:
            return kernel.structured_spmm(*args, **kwargs)
        finally:
            pl.pallas_call = original

    return run


@pytest.mark.parametrize("arrangement", ARRANGEMENTS)
def test_matches_dense_matmul_on_cpu(modules, arrangement):
    """All eight sparsity arrangements, under the interpreter."""
    kernel, baseline = modules
    check(kernel, baseline, arrangement, interpret=True)


@pytest.mark.skipif(not HAS_TPU, reason="Mosaic lowering needs a TPU")
@pytest.mark.parametrize("arrangement", ARRANGEMENTS)
def test_matches_dense_matmul_on_tpu(modules, arrangement):
    """The half that makes this kernel migrated."""
    kernel, baseline = modules
    check(kernel, baseline, arrangement, interpret=False)


@pytest.mark.skipif(not HAS_TPU, reason="Mosaic lowering needs a TPU")
def test_reaches_pallas(modules):
    sys.path.insert(0, str(ROOT / "tools"))
    from profile_kernel import count_pallas_launches

    kernel, baseline = modules
    sparsifier, mat, _, _, call, _, _ = build_case(kernel, baseline, True, True, False)
    launches = count_pallas_launches(
        lambda nz, md, m: kernel.structured_spmm(SPARSITY, nz, md, m, **call),
        (sparsifier.nonzeros, sparsifier.metadata, mat))
    assert launches == 1, f"{launches} Pallas launches"


def test_the_compression_actually_drops_values(modules):
    """Guard the premise: if `Sparsifier` were the identity, the test is empty.

    The kernel is only interesting because `nonzeros` is smaller than the matrix
    it came from. At 2:4 it should be half the size — if that ever stopped being
    true, every comparison above would still pass while checking nothing about
    sparsity.
    """
    kernel, baseline = modules
    sparsifier, _, _, _, _, _, mask = build_case(kernel, baseline, True, True, False)
    dense_elements = int(np.asarray(mask).size)
    kept = int(np.asarray(sparsifier.nonzeros).size)
    x, y = SPARSITY
    assert kept * y == dense_elements * x, (
        f"expected {x}/{y} of the elements kept, got {kept} of {dense_elements}")
    # And the mask really drops that many, rather than being all-True.
    assert int(np.asarray(mask).sum()) * y == dense_elements * x


def test_the_reference_is_not_itself_pallas():
    """A reference that lowered to Mosaic would not be an independent check."""
    source = (FAMILY / "baseline.py").read_text().split('"""', 2)[-1]
    assert "pallas" not in source and "pl." not in source
