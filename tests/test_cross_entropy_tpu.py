"""TPU correctness for Tokamax's linear softmax cross-entropy loss.

Both launch points are here -- the forward and the backward -- and both are
checked against upstream's own `reference.py`, at upstream's own test shapes and
tolerances (`pallas_mosaic_tpu_kernel_test.py`: 1e-4 for float32).

The pair is the point of the kernel.  Logits are `[B, V]`, larger than every
other array combined at a real vocabulary, and neither pass materialises them:
the forward blocks over V and accumulates by the log-linearity of log-sum-exp,
returning `lse` so the backward can recompute logits blockwise instead of
storing them.  So the backward is not an independent kernel that happens to live
in the same file -- it consumes the forward's residual, and a test that only
exercised the forward would leave half the contract unchecked.

Requires a TPU.  Run with::

    uv run --frozen --with pytest python -m pytest \
        tests/test_cross_entropy_tpu.py -q
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
FAMILY = ROOT / "kernels" / "loss" / "cross_entropy"

pytestmark = pytest.mark.skipif(
    not any("TPU" in device.device_kind for device in jax.devices()),
    reason="the Tokamax cross-entropy kernels are Mosaic TPU kernels",
)

#: Upstream's own "small" and "medium" parameterisations.
SMALL = (1024, 512, 2048)
MEDIUM = (4096, 1024, 4096)

#: Upstream's float32 tolerance, from pallas_mosaic_tpu_kernel_test.py.
TOL = 1e-4

REDUCTIONS = ("sum", "mean", "none")


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, FAMILY / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    return (load("ce_kernel", "tokamax_optimized.py"),
            load("ce_reference", "tokamax_reference.py"))


def inputs(shape, dtype=jnp.float32):
    """Upstream's own test inputs: normal x and w, labels uniform over V."""
    b_dim, h_dim, v_dim = shape
    kx, kl, kw = jax.random.split(jax.random.key(42), 3)
    x = jax.random.normal(kx, (b_dim, h_dim), dtype=dtype)
    labels = jax.random.randint(kl, (b_dim,), 0, v_dim, dtype=jnp.int32)
    w = jax.random.normal(kw, (h_dim, v_dim), dtype=dtype)
    return x, labels, w


def forward(kernel, x, labels, w, shape, reduction):
    config = kernel.get_heuristic_fwd_config(*shape)
    return kernel.linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu(
        x, labels, w, reduction=reduction,
        b_block_size=config.b_block_size, h_block_size=config.h_block_size,
        v_block_size=config.v_block_size)


def backward(kernel, dout, lse, x, labels, w, shape, reduction):
    config = kernel.get_heuristic_bwd_config(*shape)
    return kernel.linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu(
        dout, lse, x, labels, w, reduction=reduction,
        b_block_size=config.b_block_size, h_block_size=config.h_block_size,
        v_block_size=config.v_block_size)


def close(actual, expected, what):
    np.testing.assert_allclose(
        np.asarray(actual, np.float32), np.asarray(expected, np.float32),
        atol=TOL, rtol=TOL, err_msg=what)


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_forward_matches_upstream_reference(modules, reduction):
    """Loss *and* lse: the kernel returns both, and the backward needs both."""
    kernel, reference = modules
    x, labels, w = inputs(SMALL)

    loss, lse = forward(kernel, x, labels, w, SMALL, reduction)
    ref_loss, ref_lse = reference.linear_softmax_cross_entropy_loss_fwd_reference(
        x, labels, w, reduction=reduction)
    jax.block_until_ready((loss, lse, ref_loss, ref_lse))

    assert loss.shape == ref_loss.shape, (
        f"{reduction}: reduction changes the loss's shape, not just its scale")
    close(loss, ref_loss, f"{reduction} loss")
    close(lse, ref_lse, f"{reduction} lse")


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_backward_matches_upstream_reference(modules, reduction):
    """Both gradients, fed the forward's own lse rather than a recomputed one."""
    kernel, reference = modules
    x, labels, w = inputs(SMALL)
    _, lse = forward(kernel, x, labels, w, SMALL, reduction)

    dout = (jnp.ones((SMALL[0],), jnp.float32) if reduction == "none"
            else jnp.float32(1.0))
    x_grad, w_grad = backward(kernel, dout, lse, x, labels, w, SMALL, reduction)
    ref_x_grad, ref_w_grad = (
        reference.linear_softmax_cross_entropy_loss_bwd_reference(
            dout, lse, x, labels, w, reduction=reduction))
    jax.block_until_ready((x_grad, w_grad, ref_x_grad, ref_w_grad))

    assert x_grad.shape == x.shape and w_grad.shape == w.shape
    close(x_grad, ref_x_grad, f"{reduction} x_grad")
    close(w_grad, ref_w_grad, f"{reduction} w_grad")


def test_forward_matches_at_upstreams_medium_shape(modules):
    """The small shape is one block per dimension; this one is several."""
    kernel, reference = modules
    x, labels, w = inputs(MEDIUM)
    config = kernel.get_heuristic_fwd_config(*MEDIUM)
    assert MEDIUM[0] > config.b_block_size or MEDIUM[2] > config.v_block_size, (
        "this shape must exercise more than one block for the test to mean "
        "anything beyond the small case")

    loss, lse = forward(kernel, x, labels, w, MEDIUM, "mean")
    ref_loss, ref_lse = reference.linear_softmax_cross_entropy_loss_fwd_reference(
        x, labels, w, reduction="mean")
    jax.block_until_ready((loss, lse, ref_loss, ref_lse))
    close(loss, ref_loss, "medium loss")
    close(lse, ref_lse, "medium lse")


def test_a_vocabulary_that_does_not_divide_the_block(modules):
    """`calculate_xw_tiled` masks a ragged final V block; check that path runs.

    Real vocabularies are not multiples of 2048, and the kernel handles it by
    zeroing the tail of the last `w` block rather than by rejecting the shape.
    Without this case the masking code is never executed.

    The shape is upstream's own `fwd_v_non_aligned_multiple_of_128` case.  A
    vocabulary merely not equal to the block size is not enough: the heuristic
    picks the block size from the shape, so for V=2560 it simply chooses 2560
    and nothing is ragged.  2664 is not a multiple of 128, which `Config`
    requires of every block size, so no choice it can make divides evenly.
    """
    kernel, reference = modules
    shape = (4096, 1024, 2664)
    assert shape[2] % 128 != 0, "otherwise the heuristic can divide it evenly"
    assert shape[2] % kernel.get_heuristic_fwd_config(*shape).v_block_size != 0
    x, labels, w = inputs(shape)

    loss, lse = forward(kernel, x, labels, w, shape, "sum")
    ref_loss, ref_lse = reference.linear_softmax_cross_entropy_loss_fwd_reference(
        x, labels, w, reduction="sum")
    jax.block_until_ready((loss, lse, ref_loss, ref_lse))
    close(loss, ref_loss, "ragged-V loss")
    close(lse, ref_lse, "ragged-V lse")


def test_both_launch_points_reach_pallas(modules):
    """Two launch points in one file, so check both -- not just the forward.

    A file counted as two migrated launch points with a test that only touches
    one is exactly the gap `tools/launch_coverage.py` exists to find.
    """
    sys.path.insert(0, str(ROOT / "tools"))
    from profile_kernel import count_pallas_launches

    kernel, _ = modules
    x, labels, w = inputs(SMALL)
    _, lse = forward(kernel, x, labels, w, SMALL, "sum")

    fwd = count_pallas_launches(
        lambda a, b, c: forward(kernel, a, b, c, SMALL, "sum"), (x, labels, w))
    bwd = count_pallas_launches(
        lambda d, l, a, b, c: backward(kernel, d, l, a, b, c, SMALL, "sum"),
        (jnp.float32(1.0), lse, x, labels, w))
    assert fwd == 1, f"forward: {fwd} Pallas launches"
    assert bwd == 1, f"backward: {bwd} Pallas launches"


def test_the_reference_is_not_itself_pallas(modules):
    """A reference that lowered to Mosaic would not be an independent check."""
    sys.path.insert(0, str(ROOT / "tools"))
    from profile_kernel import count_pallas_launches

    kernel, reference = modules
    x, labels, w = inputs(SMALL)
    _, lse = forward(kernel, x, labels, w, SMALL, "sum")
    assert count_pallas_launches(
        lambda a, b, c: reference.linear_softmax_cross_entropy_loss_fwd_reference(
            a, b, c, reduction="sum"), (x, labels, w)) == 0
    assert count_pallas_launches(
        lambda d, l, a, b, c:
            reference.linear_softmax_cross_entropy_loss_bwd_reference(
                d, l, a, b, c, reduction="sum"),
        (jnp.float32(1.0), lse, x, labels, w)) == 0


def test_the_block_size_constraints_survived_the_pydantic_substitution(modules):
    """`Config`'s validation is upstream's, re-expressed as a `__post_init__`.

    The corpus swapped pydantic for stdlib dataclasses, which would silently
    drop `Field(ge=..., multiple_of=128)`.  These are the constraints upstream
    declared; a block size violating them does not lay out on a TPU lane.
    """
    kernel, _ = modules
    assert kernel.Config() == kernel.Config(1024, 512, 2048)
    for kwargs in ({"v_block_size": 100}, {"h_block_size": 64},
                   {"b_block_size": 512}, {"b_block_size": 1152 + 1}):
        with pytest.raises(ValueError):
            kernel.Config(**kwargs)
