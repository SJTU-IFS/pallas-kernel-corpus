"""Small-shape TPU correctness for the quantized-matmul family.

Each migrated file exposes two entry points, checked against two matching
pure-JAX contracts:

``quantized_matmul_kernel``            -> ``baseline.quantized_matmul_per_channel``
``blockwise_quantized_matmul_kernel``  -> ``baseline.quantized_matmul_blockwise``

Requires a TPU.  Run with::

    uv run --frozen --with pytest python -m pytest \
        tests/test_quantized_matmul_tpu.py -q
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
FAMILY = ROOT / "kernels" / "quantization" / "quantized_matmul"

pytestmark = pytest.mark.skipif(
    not any("TPU" in device.device_kind for device in jax.devices()),
    reason="quantized matmul kernels are Mosaic TPU kernels",
)


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, FAMILY / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    return {
        "baseline": load("qm_baseline", "baseline.py"),
        "tpu_inference": load("qm_tpu_inference", "tpu_inference_optimized.py"),
        "sglang_jax": load("qm_sglang_jax", "sglang_jax_optimized.py"),
    }


IMPLEMENTATIONS = ("tpu_inference", "sglang_jax")


def cosine(actual, expected) -> float:
    a = np.asarray(actual, np.float32).ravel()
    e = np.asarray(expected, np.float32).ravel()
    return float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e) + 1e-12))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("quantize_activation", [False, True])
@pytest.mark.parametrize("shape", [(256, 2048, 1024), (512, 1024, 2048)])
def test_per_channel_matches_reference(
    modules, implementation, quantize_activation, shape
):
    baseline = modules["baseline"]
    n_batch, n_in, n_out = shape
    x, w_q, w_scale = baseline.create_inputs(
        n_batch=n_batch, n_in=n_in, n_out=n_out
    )
    x_q_dtype = jnp.int8 if quantize_activation else None
    expected = jax.jit(
        baseline.quantized_matmul_per_channel, static_argnames="x_q_dtype"
    )(x, w_q, w_scale, x_q_dtype=x_q_dtype)
    actual = modules[implementation].quantized_matmul_kernel(
        x, w_q, w_scale, None, None, x_q_dtype
    )
    assert actual.shape == expected.shape
    assert cosine(actual, expected) > 0.9999


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("block_size", [128, 256])
def test_blockwise_matches_reference(modules, implementation, block_size):
    """Block-wise needs in_block_size >= block_size, so tuning is explicit.

    With the table's own choice the accumulator loop can run zero times; see
    test_blockwise_requires_in_block_at_least_block_size.
    """
    baseline = modules["baseline"]
    module = modules[implementation]
    x, w_q, w_scale, block = baseline.create_blockwise_inputs(
        n_batch=512, n_in=2048, n_out=2048, block_size=block_size
    )
    expected = jax.jit(
        baseline.quantized_matmul_blockwise,
        static_argnames=("block_size", "x_q_dtype"),
    )(x, w_q, w_scale, block_size=block, x_q_dtype=None)
    actual = module.blockwise_quantized_matmul_kernel(
        x, w_q, w_scale, None, block, None,
        tuned_value=module.TunedValue(512, 512, 512, 1),
    )
    assert actual.shape == expected.shape
    assert cosine(actual, expected) > 0.9999


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_blockwise_requires_in_block_at_least_block_size(modules, implementation):
    """Pin the failure mode found while migrating.

    The kernel loops ``steps_k = in_block_size // block_size`` times and only
    fills its accumulator list on the first trip.  When the selected
    ``in_block_size`` is smaller than ``block_size`` the loop never runs and the
    kernel fails inside ``jnp.concatenate`` on a list of ``None`` -- a confusing
    error a long way from the cause.
    """
    baseline = modules["baseline"]
    module = modules[implementation]
    x, w_q, w_scale, block = baseline.create_blockwise_inputs(
        n_batch=512, n_in=2048, n_out=2048, block_size=256
    )
    with pytest.raises((ValueError, TypeError)):
        module.blockwise_quantized_matmul_kernel(
            x, w_q, w_scale, None, block, None,
            tuned_value=module.TunedValue(512, 512, 128, 1),  # in_block < block
        )


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_zero_point_is_rejected(modules, implementation):
    """Neither upstream kernel implements asymmetric quantization."""
    baseline = modules["baseline"]
    x, w_q, w_scale = baseline.create_inputs(n_batch=256, n_in=1024, n_out=1024)
    w_zp = jnp.zeros((1024,), jnp.float32)
    with pytest.raises(NotImplementedError):
        modules[implementation].quantized_matmul_kernel(
            x, w_q, w_scale, w_zp, None, None
        )


def test_int8_weights_with_bfloat16_activations_are_untuned(modules):
    """Guard the finding that the weight-only int8 path is untuned everywhere.

    The tables are not identical -- tpu-inference has no bfloat16 activation
    entries at all, while sglang-jax has 120, but every one of those is paired
    with float8 weights.  So the specific combination "int8 weights, bfloat16
    (unquantized) activations" misses the table in both repositories and falls
    back to an untuned tiling.  Measured consequence: 66x slower than plain XLA
    at the profiled shape.

    If either table gains an (bfloat16, int8) entry, this fails so the
    performance claim gets revisited rather than silently going stale.
    """
    for name in IMPLEMENTATIONS:
        table = modules[name].TUNED_BLOCK_SIZES_RAW
        activation_dtypes = {key[4] for key in table}
        assert activation_dtypes <= {"bfloat16", "int8", "float8_e4m3fn"}, name
        int8_weight_bf16_activation = [
            key for key in table if key[4] == "bfloat16" and key[5] == "int8"
        ]
        assert not int8_weight_bf16_activation, (
            f"{name}: table now tunes int8 weights with bfloat16 activations; "
            "re-measure the weight-only path"
        )

    # tpu-inference specifically has no bfloat16 activation entries at all.
    tpu_inference_activations = {
        key[4] for key in modules["tpu_inference"].TUNED_BLOCK_SIZES_RAW
    }
    assert tpu_inference_activations == {"int8", "float8_e4m3fn"}
