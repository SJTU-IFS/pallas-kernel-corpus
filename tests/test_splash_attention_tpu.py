"""Small-shape TPU correctness for the Splash Attention family.

This is the corpus's **first family with migrated backward kernels**.  Each
migrated file carries three Pallas launch points -- forward, backward-dQ and
backward-dKV -- and the backward pair is reached by differentiating through the
kernel's ``custom_vjp``, which is what the gradient tests below do.

Both files ship a pure-JAX ``attention_reference`` upstream taking the same
mask object, so each is checked against its own reference *and* against the
other repository's.

Requires a TPU.  Run with::

    uv run --frozen --with pytest python -m pytest \
        tests/test_splash_attention_tpu.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)


ROOT = Path(__file__).parents[1]
FAMILY = ROOT / "kernels" / "attention" / "splash_attention"

pytestmark = pytest.mark.skipif(
    not any("TPU" in device.device_kind for device in jax.devices()),
    reason="splash attention kernels are Mosaic TPU kernels",
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
        "baseline": load("splash_baseline", "baseline.py"),
        "jaxbench": load("splash_jaxbench", "jaxbench_optimized.py"),
        "maxtext": load("splash_maxtext", "maxtext_optimized.py"),
    }


IMPLEMENTATIONS = ("jaxbench", "maxtext")
SHAPES = [(8, 2, 512, 128), (16, 4, 1024, 128)]


def cosine(actual, expected) -> float:
    a = np.asarray(actual, np.float32).ravel()
    e = np.asarray(expected, np.float32).ravel()
    return float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e) + 1e-12))


def rms_relative(actual, expected) -> float:
    """RMS error relative to the reference's own RMS magnitude."""
    a = np.asarray(actual, np.float32).ravel()
    e = np.asarray(expected, np.float32).ravel()
    return float(np.sqrt(((a - e) ** 2).mean())
                 / (np.sqrt((e ** 2).mean()) + 1e-12))


def assert_matches(actual, expected, name: str = "", *,
                   cos_bar: float = 0.9999, rms_bar: float = 2e-2) -> None:
    """Direction *and* magnitude, because `cosine` alone checks only direction.

    `cosine` is scale-invariant: an output multiplied by any constant -- 0.5,
    2, a million -- still scores 1.0 against its reference and sails past a
    0.9999 bar. A wrong dequantization scalar, fp8 scale or softmax
    normalisation constant is exactly that kind of error, and it is a plausible
    failure mode for these kernels, so direction alone is not evidence.

    The RMS-relative bar is this corpus's own existing pattern, used in the
    splash and flash backward tests and in gated linear attention. 2e-2 is the
    value those chose. It is comfortably satisfied here: at the 0.9999 cosine
    bar the angular term alone contributes sqrt(2*(1-0.9999)) = 1.4e-2, and the
    cosines this suite actually records are 0.99997 or better, putting the
    angular contribution at 7.7e-3 or less. What the bar catches is any scale
    error above ~2%.
    """
    a = np.asarray(actual, np.float32)
    e = np.asarray(expected, np.float32)
    label = f"{name}: " if name else ""
    similarity = cosine(a, e)
    assert similarity > cos_bar, f"{label}cosine {similarity} <= {cos_bar}"
    scale = rms_relative(a, e)
    assert scale < rms_bar, f"{label}RMS-relative {scale} >= {rms_bar}"


def build(modules, shape, dtype=jnp.float32):
    baseline = modules["baseline"]
    num_q_heads, num_kv_heads, seq_len, head_dim = shape
    inputs = baseline.create_inputs(
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        seq_len=seq_len, head_dim=head_dim, dtype=dtype,
    )
    mask = baseline.causal_multi_head_mask(seq_len, num_q_heads)
    return inputs, mask


def kernel_of(module, mask):
    return module.make_splash_mha_single_device(
        mask, block_sizes=module.BlockSizes.get_default()
    )


def reference_of(module, mask):
    return module.make_attention_reference(
        mask, is_mqa=False, backward_impl="vanilla"
    )


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("shape", SHAPES)
def test_forward_matches_reference(modules, implementation, shape):
    module = modules[implementation]
    (q, k, v), mask = build(modules, shape)
    expected = jax.jit(reference_of(module, mask))(q, k, v)
    actual = jax.jit(kernel_of(module, mask))(q, k, v)
    assert actual.shape == expected.shape
    assert_matches(actual, expected)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_backward_matches_reference(modules, implementation):
    """Differentiating the custom_vjp exercises backward-dQ and backward-dKV.

    Cosine is the criterion, matching the rest of the corpus.  Absolute error
    looks large (1.5-2.5) only because the gradients themselves are large here.

    Measured RMS-relative error at this shape is 5.3e-3 for dQ, 5.2e-3 for dK
    and 6.8e-4 for dV -- larger than the forward pass because a causal mask
    leaves many gradient entries near zero, which inflates any relative
    statistic.  The bound below is set from those measurements with headroom,
    not from a guess.
    """
    module = modules[implementation]
    (q, k, v), mask = build(modules, SHAPES[0])

    def as_loss(fn):
        def loss(a, b, c):
            return jnp.sum(fn(a, b, c).astype(jnp.float32) ** 2)

        return jax.grad(loss, argnums=(0, 1, 2))

    expected = jax.jit(as_loss(reference_of(module, mask)))(q, k, v)
    actual = jax.jit(as_loss(kernel_of(module, mask)))(q, k, v)

    for name, got, want in zip(("dq", "dk", "dv"), actual, expected):
        assert got.shape == want.shape, name
        assert_matches(got, want, name)
        a = np.asarray(got, np.float32)
        e = np.asarray(want, np.float32)
        rms_relative = float(
            np.sqrt(((a - e) ** 2).mean()) / (np.sqrt((e**2).mean()) + 1e-12)
        )
        assert rms_relative < 2e-2, f"{name}: RMS-relative error {rms_relative}"


def test_the_two_implementations_agree_with_each_other(modules):
    """Different code, same math: only ~15% of their definitions are shared."""
    (q, k, v), mask = build(modules, SHAPES[0])
    outputs = [
        jax.jit(kernel_of(modules[name], mask))(q, k, v) for name in IMPLEMENTATIONS
    ]
    assert_matches(*outputs)


def test_references_are_pallas_free(modules):
    """A reference that lowered to Pallas would not be a valid check."""
    import inspect

    for name in IMPLEMENTATIONS:
        source = inspect.getsource(modules[name].attention_reference)
        assert "pallas" not in source
        assert "pl." not in source


def test_default_block_sizes_are_the_slow_path(modules):
    """Pin why the profiles use JAXBench's autotuned blocks.

    ``BlockSizes.get_default()`` is 128 in every dimension and carries an
    upstream "TODO: select better parameters"; measured, it is ~18x slower than
    the autotuned forward blocks. If the default ever improves, this test
    should fail so the profiling choice gets revisited.
    """
    defaults = modules["jaxbench"].BlockSizes.get_default()
    assert defaults.block_q == 128
    assert defaults.block_kv == 128
    assert defaults.block_kv_compute == 128

    tuned = modules["jaxbench"].TUNED_PARAMS
    assert tuned["block_q"] == 2048
    assert tuned["block_kv"] == 2048
    # Upstream autotuned the forward only and left every backward block None.
    for key in ("block_q_dkv", "block_kv_dkv", "block_q_dq", "block_kv_dq"):
        assert tuned[key] is None, key


def test_backward_requires_explicit_blocks(modules):
    """The backward rejects the None blocks JAXBench ships, by design."""
    module = modules["jaxbench"]
    (q, k, v), mask = build(modules, SHAPES[0])
    blocks = module.BlockSizes(
        block_q=512, block_kv=512, block_kv_compute=512,
        block_q_dkv=None, block_kv_dkv=None, block_kv_dkv_compute=None,
        block_q_dq=None, block_kv_dq=None,
    )
    kernel = module.make_splash_mha_single_device(mask, block_sizes=blocks)

    def loss(a, b, c):
        return jnp.sum(kernel(a, b, c).astype(jnp.float32) ** 2)

    with pytest.raises(ValueError, match="backward blocks"):
        jax.grad(loss)(q, k, v)
