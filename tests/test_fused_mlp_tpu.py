"""TPU correctness for sglang-jax's fused gated MLP.

The kernel fuses gate/up projection, the SiLU gate and the down projection into
one pipeline, and ends in `jax.lax.psum` over a "tensor" mesh axis because `wd`
is sharded on its contracting dimension.  This corpus validates on one device,
where that psum is the identity -- the fusion, the pipelining and the packed
weight layout are all still exercised; only the cross-shard reduction is not.

The reference is upstream's own non-fused fallback, and the weight packing is
upstream's own `post_load_weights`; both are carried in
`sglang_jax_reference.py` with the source regions they reproduce quoted.

Requires a TPU.  Run with::

    uv run --frozen --with pytest python -m pytest \
        tests/test_fused_mlp_tpu.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh


ROOT = Path(__file__).parents[1]
FAMILY = ROOT / "kernels" / "moe" / "gated_mlp"

pytestmark = pytest.mark.skipif(
    not any("TPU" in device.device_kind for device in jax.devices()),
    reason="the fused gated MLP is a Mosaic TPU kernel",
)

#: Small enough to compile quickly, but every axis is a real multiple: the
#: sequence tiles by b_seq, the intermediate by b_inter, and there is more than
#: one intermediate block so the pipeline's accumulate-across-blocks path runs.
SEQ, HIDDEN, INTER = 256, 512, 1024
B_SEQ, B_INTER = 64, 128


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, FAMILY / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    return (load("mlp_kernel", "sglang_jax_optimized.py"),
            load("mlp_reference", "sglang_jax_reference.py"))


@pytest.fixture(scope="module")
def mesh():
    """A one-device mesh carrying the "tensor" axis the kernel names."""
    return Mesh(np.array(jax.devices()[:1]), ("tensor",))


def weights(seq: int = SEQ, dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kg, ku, kd = jax.random.split(key, 4)
    x = jax.random.normal(kx, (seq, HIDDEN), dtype=dtype)
    wg = jax.random.normal(kg, (HIDDEN, INTER), dtype=dtype) * 0.05
    wu = jax.random.normal(ku, (HIDDEN, INTER), dtype=dtype) * 0.05
    wd = jax.random.normal(kd, (INTER, HIDDEN), dtype=dtype) * 0.05
    return x, wg, wu, wd


def run(kernel, mesh, x, w_gu, wd):
    return kernel.kernel(x, w_gu, wd, mesh, B_SEQ, B_INTER)


def test_matches_upstreams_own_fallback(modules, mesh):
    """The fused kernel against `down(up(x) * silu(gate(x)))`."""
    kernel, reference = modules
    x, wg, wu, wd = weights()
    w_gu = reference.pack_gate_up(wg, wu, B_INTER)

    with mesh:
        actual = run(kernel, mesh, x, w_gu, wd)
    expected = jax.jit(reference.gated_mlp)(x, wg, wu, wd)
    jax.block_until_ready((actual, expected))

    actual = np.asarray(actual, np.float32)
    expected = np.asarray(expected, np.float32)
    assert actual.shape == expected.shape == (SEQ, HIDDEN)
    # Both sides are bf16 on the MXU, through two chained matmuls.
    scale = float(np.max(np.abs(expected)))
    np.testing.assert_allclose(actual, expected, rtol=2**-7, atol=2**-7 * scale)


def test_the_packing_is_not_plain_concatenation(modules, mesh):
    """The interleaved layout is load-bearing, so prove the naive one differs.

    Without this, `test_matches_upstreams_own_fallback` would still pass if the
    kernel happened to be insensitive to how gate and up are arranged -- and the
    reader would have no way to tell whether `pack_gate_up`'s block interleaving
    was a real requirement or cargo cult.  It is real: feeding the obvious
    `concat([wg, wu], axis=1)` gives a different answer.
    """
    kernel, reference = modules
    x, wg, wu, wd = weights()
    naive = jnp.concatenate([wg, wu], axis=1)
    packed = reference.pack_gate_up(wg, wu, B_INTER)
    assert naive.shape == packed.shape
    # With more than one intermediate block the two layouts genuinely differ.
    assert INTER // B_INTER > 1
    assert not np.array_equal(np.asarray(naive), np.asarray(packed))

    with mesh:
        from_naive = np.asarray(run(kernel, mesh, x, naive, wd), np.float32)
    expected = np.asarray(jax.jit(reference.gated_mlp)(x, wg, wu, wd), np.float32)
    scale = float(np.max(np.abs(expected)))
    assert np.max(np.abs(from_naive - expected)) > 2**-7 * scale, (
        "the naive concatenation gives the same answer as the block-interleaved "
        "packing, so this kernel does not actually depend on the layout")


def test_padding_path_handles_a_ragged_sequence(modules, mesh):
    """`apply_fused_mlp_with_padding` is the entry point precisely for this."""
    kernel, reference = modules
    seq = SEQ - 8  # not a multiple of B_SEQ
    assert seq % B_SEQ != 0
    x, wg, wu, wd = weights(seq=seq)
    w_gu = reference.pack_gate_up(wg, wu, B_INTER)

    with mesh:
        actual = run(kernel, mesh, x, w_gu, wd)
    expected = jax.jit(reference.gated_mlp)(x, wg, wu, wd)
    jax.block_until_ready((actual, expected))

    actual = np.asarray(actual, np.float32)
    expected = np.asarray(expected, np.float32)
    assert actual.shape == (seq, HIDDEN), "the padding must be trimmed back off"
    scale = float(np.max(np.abs(expected)))
    np.testing.assert_allclose(actual, expected, rtol=2**-7, atol=2**-7 * scale)


def test_reaches_pallas(modules, mesh):
    """The standing rule: a passing comparison does not prove the kernel ran."""
    sys.path.insert(0, str(ROOT / "tools"))
    from profile_kernel import count_pallas_launches

    kernel, reference = modules
    x, wg, wu, wd = weights()
    w_gu = reference.pack_gate_up(wg, wu, B_INTER)
    with mesh:
        launches = count_pallas_launches(
            lambda a, b, c: run(kernel, mesh, a, b, c), (x, w_gu, wd))
    assert launches == 1, f"{launches} Pallas launches"


def test_the_reference_is_not_itself_pallas(modules):
    """A reference that lowered to Mosaic would not be an independent check."""
    sys.path.insert(0, str(ROOT / "tools"))
    from profile_kernel import count_pallas_launches

    _, reference = modules
    x, wg, wu, wd = weights()
    assert count_pallas_launches(reference.gated_mlp, (x, wg, wu, wd)) == 0
