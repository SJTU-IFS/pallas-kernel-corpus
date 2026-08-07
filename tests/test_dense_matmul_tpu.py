"""TPU correctness for JAXBench's dense GEMM kernel.

Both sides of this comparison are JAXBench's own: `optimized.py` against the
`baseline.py` sitting beside it in `benchmark/8p_GEMM/`, on the inputs that
file generates.  The tolerance is JAXBench's too -- `CONFIG['atol']` and
`CONFIG['rtol']`, 1e-3 and 1e-2, which are loose because both sides run bf16 on
the MXU.

Requires a TPU.  Run with::

    uv run --frozen --with pytest python -m pytest \
        tests/test_dense_matmul_tpu.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp


ROOT = Path(__file__).parents[1]
FAMILY = ROOT / "kernels" / "matmul" / "dense_matmul"

pytestmark = pytest.mark.skipif(
    not any("TPU" in device.device_kind for device in jax.devices()),
    reason="the JAXBench GEMM is a Mosaic TPU kernel",
)


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, FAMILY / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    return (load("gemm_kernel", "jaxbench_optimized.py"),
            load("gemm_reference", "jaxbench_reference.py"))


def test_matches_jaxbench_reference(modules):
    """The Pallas GEMM against JAXBench's own `jnp.dot`, at the native shape."""
    kernel, reference = modules
    x, y = reference.create_inputs()
    actual = jax.jit(kernel.kernel)(x, y)
    expected = jax.jit(reference.workload)(x, y)
    jax.block_until_ready((actual, expected))

    assert actual.shape == expected.shape == (8192, 28672)
    assert actual.dtype == expected.dtype == jnp.bfloat16
    np.testing.assert_allclose(
        np.asarray(actual, np.float32), np.asarray(expected, np.float32),
        atol=kernel.CONFIG["atol"], rtol=kernel.CONFIG["rtol"])


def test_both_sides_generate_the_same_inputs(modules):
    """`optimized.py` and `baseline.py` must agree on what they are fed.

    Each ships its own `create_inputs`, and the scaling of the second operand
    (0.02) lives in both.  If they ever diverge, the comparison above stops
    being a comparison of implementations.
    """
    kernel, reference = modules
    kx, ky = kernel.create_inputs()
    rx, ry = reference.create_inputs()
    np.testing.assert_array_equal(np.asarray(kx), np.asarray(rx))
    np.testing.assert_array_equal(np.asarray(ky), np.asarray(ry))


def test_reaches_pallas(modules):
    """The standing rule: a passing comparison does not prove the kernel ran."""
    sys.path.insert(0, str(ROOT / "tools"))
    from profile_kernel import count_pallas_launches

    kernel, reference = modules
    assert count_pallas_launches(kernel.kernel, reference.create_inputs()) == 1


def test_the_reference_is_not_itself_pallas(modules):
    """`jnp.dot` must lower to an XLA dot, not to a Mosaic custom call.

    The flash-attention family showed this is not automatic: XLA rewrites some
    pure-JAX patterns into `tpu_custom_call`s of its own, and a reference that
    did so would not be an independent check.
    """
    sys.path.insert(0, str(ROOT / "tools"))
    from profile_kernel import count_pallas_launches

    _, reference = modules
    assert count_pallas_launches(reference.workload,
                                 reference.create_inputs()) == 0


def test_tuned_blocks_tile_the_native_shape(modules):
    """`workload` is `matmul` at JAXBench's autotuned sizes, which must divide.

    The kernel's grid is `(M // l, N // r, K // block_k)` with no remainder
    handling, so a block shape that does not divide the native shape would
    silently drop the tail rather than fail.
    """
    kernel, _ = modules
    (l, r), block_k = kernel.TUNED_PARAMS["block_shape"], kernel.TUNED_PARAMS["block_k"]
    assert kernel.CONFIG["M"] % l == 0
    assert kernel.CONFIG["N"] % r == 0
    assert kernel.CONFIG["K"] % block_k == 0
