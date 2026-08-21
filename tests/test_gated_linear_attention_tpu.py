"""Small-shape TPU correctness for the gated linear attention family.

Two contracts share this directory and are not variants of each other: KDA
(per-channel gates plus a `beta` delta term) and Simple GLA (one scalar decay
per head). Both references are upstream's own, copied from sglang-jax.

Both are validated. Simple GLA's calling convention is not guessable from its
signature and was taken from sglang-jax's own production call sites; the
constraints are pinned in `test_simple_gla_rejects_the_conventions_that_do_not_apply`.

Requires a TPU.  Run with::

    uv run --frozen --with pytest python -m pytest \
        tests/test_gated_linear_attention_tpu.py -q
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
FAMILY = ROOT / "kernels" / "state_space" / "gated_linear_attention"

pytestmark = pytest.mark.skipif(
    not any("TPU" in device.device_kind for device in jax.devices()),
    reason="gated linear attention kernels are Mosaic TPU kernels",
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
        "baseline": load("gla_baseline", "baseline.py"),
        "kda": load("gla_kda", "sglang_jax_kda_optimized.py"),
        "simple_gla": load("gla_simple", "sglang_jax_simple_gla_optimized.py"),
    }


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


def kda_inputs(seq_len, batch=1, heads=4, k_dim=64, v_dim=64, seed=0):
    """A **contracting** KDA recurrence.

    The delta rule accumulates into the carried state, so with a weak decay the
    output grows without bound in T -- at g ~ -0.07 over 256 tokens the outputs
    reach ~1e14 and an absolute comparison becomes meaningless while cosine
    still looks fine. L2-normalising q/k (what the kernel's own
    `use_qk_l2norm_in_kernel` does) and using a full-strength decay keeps the
    recurrence contracting, so the test measures the kernel and not the setup.
    """
    keys = jax.random.split(jax.random.key(seed), 5)

    def l2(x):
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)

    shape = (batch, seq_len, heads, k_dim)
    return dict(
        q=l2(jax.random.normal(keys[0], shape, jnp.float32)),
        k=l2(jax.random.normal(keys[1], shape, jnp.float32)),
        v=jax.random.normal(keys[2], (batch, seq_len, heads, v_dim), jnp.float32) * 0.5,
        g=-jax.nn.softplus(jax.random.normal(keys[3], shape, jnp.float32)),
        beta=jax.nn.sigmoid(
            jax.random.normal(keys[4], (batch, seq_len, heads), jnp.float32)
        ),
        h0=jnp.zeros((batch, heads, k_dim, v_dim), jnp.float32),
        cu_seqlens=jnp.array([0, seq_len], jnp.int32),
        scale=k_dim**-0.5,
    )


@pytest.mark.parametrize("seq_len", [256, 512])
def test_kda_matches_upstream_reference(modules, seq_len):
    built = kda_inputs(seq_len)
    expected, expected_state = modules["baseline"].naive_recurrent_kda(
        built["q"], built["k"], built["v"], built["g"], built["beta"],
        built["scale"], built["h0"], True,
    )
    leaves = jax.tree.leaves(
        modules["kda"].chunk_kda_fwd(
            built["q"], built["k"], built["v"], built["g"], built["beta"],
            built["scale"], built["h0"], True, built["cu_seqlens"],
        )
    )
    assert leaves[0].shape == expected.shape
    assert_matches(leaves[0], expected)
    rms_relative = float(
        np.sqrt(((np.asarray(leaves[0], np.float64)
                  - np.asarray(expected, np.float64)) ** 2).mean())
        / (np.sqrt((np.asarray(expected, np.float64) ** 2).mean()) + 1e-12)
    )
    assert rms_relative < 1e-3

    for leaf in leaves[1:]:
        if hasattr(leaf, "shape") and leaf.shape == expected_state.shape:
            assert_matches(leaf, expected_state)
            break
    else:
        pytest.fail("no output leaf matched the reference final state shape")


def test_kda_reaches_pallas(modules):
    """Four launch points: gate cumsum, intra-chunk, state recurrence, output."""
    built = kda_inputs(256)
    text = (
        jax.jit(modules["kda"].chunk_kda_fwd, static_argnums=(5, 7))
        .lower(
            built["q"], built["k"], built["v"], built["g"], built["beta"],
            built["scale"], built["h0"], True, built["cu_seqlens"],
        ).compile().as_text()
    )
    assert text.count("tpu_custom_call") >= 1
    source = (FAMILY / "sglang_jax_kda_optimized.py").read_text()
    assert source.count("pl.pallas_call(") == 4
    assert modules["kda"].SOURCE["launch_points"] == 4


def test_kda_reference_is_pallas_free(modules):
    """A reference that lowered to Pallas would not be a valid baseline."""
    import inspect

    source = inspect.getsource(modules["baseline"].naive_recurrent_kda)
    assert "pallas" not in source and "pl." not in source


# Simple GLA's calling convention is not guessable from the signature; these
# constants come from sglang-jax's own production call sites in
# `srt/layers/attention/linear/lightning_backend.py`, which is where the
# corpus got them after every guessed combination asserted.
#
#   * only `cu_seqlens_dev` is ever passed -- `chunk_simple_gla_fwd_varlen`
#     asserts `cu_seqlens_cpu is None`, and the non-varlen path raises
#     NotImplementedError outright;
#   * `assert (K % 128 == 0) and (V % 128 == 0)` -- head dims of 64 fail;
#   * `assert B == 1` -- the batch axis is a formality, sequences are ragged;
#   * `g_gamma` is a **log** decay, so it must be negative or the recurrence
#     overflows to nan over T steps;
#   * `decode_simple_gla_fused` **donates** `recurrent_buffer`, so a caller
#     that reuses it afterwards gets "Array has been deleted".
SIMPLE_GLA_HEAD_DIM = 128


def simple_gla_inputs(seq_len=256, heads=4, num_seqs=2, seed=0):
    dim = SIMPLE_GLA_HEAD_DIM
    keys = jax.random.split(jax.random.key(seed), 4)
    boundaries = [0] + [
        (seq_len // num_seqs) * (i + 1) for i in range(num_seqs)
    ]
    return dict(
        q=jax.random.normal(keys[0], (seq_len, heads, dim), jnp.float32) * 0.5,
        k=jax.random.normal(keys[1], (seq_len, heads, dim), jnp.float32) * 0.5,
        v=jax.random.normal(keys[2], (seq_len, heads, dim), jnp.float32) * 0.5,
        g_gamma=-jax.nn.softplus(
            jax.random.normal(keys[3], (heads,), jnp.float32)
        ) - 0.05,
        h0=jnp.zeros((num_seqs, heads, dim, dim), jnp.float32),
        cu_seqlens=jnp.array(boundaries, jnp.int32),
        num_seqs=num_seqs,
        heads=heads,
        dim=dim,
        seed=seed,
    )


def test_simple_gla_prefill_matches_upstream_reference(modules):
    """Called exactly as sglang-jax's own lightning_backend calls it."""
    built = simple_gla_inputs()
    output, final_state = modules["simple_gla"].simple_gla_fwd(
        built["q"][None], built["k"][None], built["v"][None],
        g_gamma=built["g_gamma"], h0=built["h0"],
        cu_seqlens_dev=built["cu_seqlens"], scale=None, use_ht=True,
        chunk_size=64,
    )
    expected, expected_state = modules["baseline"].naive_gla_prefill(
        built["q"][None], built["k"][None], built["v"][None],
        built["g_gamma"], built["h0"], built["cu_seqlens"], None,
    )
    assert output.shape == expected.shape
    assert_matches(output, expected)
    assert_matches(final_state, expected_state)


def test_simple_gla_decode_matches_upstream_reference(modules):
    """The fused decode path; one token per request, donated buffer."""
    built = simple_gla_inputs()
    num_seqs, heads, dim = built["num_seqs"], built["heads"], built["dim"]
    keys = jax.random.split(jax.random.key(built["seed"] + 1), 4)
    q = jax.random.normal(keys[0], (num_seqs, heads, dim), jnp.float32) * 0.5
    k = jax.random.normal(keys[1], (num_seqs, heads, dim), jnp.float32) * 0.5
    v = jax.random.normal(keys[2], (num_seqs, heads, dim), jnp.float32) * 0.5
    buffer = jax.random.normal(
        keys[3], (num_seqs + 1, heads, dim, dim), jnp.float32
    ) * 0.1
    indices = jnp.arange(1, num_seqs + 1, dtype=jnp.int32)

    # The kernel donates `recurrent_buffer`; the reference needs its own copy.
    reference_h0 = jnp.array(buffer[indices], copy=True)
    output, new_buffer = modules["simple_gla"].decode_simple_gla_fused(
        q, k, v, recurrent_buffer=jnp.array(buffer, copy=True),
        recurrent_indices=indices,
        has_initial_state=jnp.ones((num_seqs,), jnp.bool_),
        g_gamma=built["g_gamma"], scale=None,
    )
    expected, expected_state = modules["baseline"].naive_gla_decode(
        q[:, None], k[:, None], v[:, None], built["g_gamma"], reference_h0, None,
    )
    assert_matches(output, expected)
    assert_matches(new_buffer[indices], expected_state)


def test_simple_gla_rejects_the_conventions_that_do_not_apply(modules):
    """Pin the constraints that make the signature misleading.

    Every one of these was hit by guessing before upstream's own call site was
    read; they are cheap to assert and expensive to rediscover.
    """
    built = simple_gla_inputs()
    module = modules["simple_gla"]
    common = dict(
        g_gamma=built["g_gamma"], h0=built["h0"], scale=None, use_ht=True,
        chunk_size=64,
    )
    # cu_seqlens_cpu must be None even when cu_seqlens_dev is given.
    with pytest.raises(AssertionError, match="cu_seqlens_cpu must be None"):
        module.simple_gla_fwd(
            built["q"][None], built["k"][None], built["v"][None],
            cu_seqlens_cpu=np.asarray(built["cu_seqlens"]),
            cu_seqlens_dev=built["cu_seqlens"], **common,
        )
    # Without cu_seqlens_dev there is no supported path at all.
    with pytest.raises(NotImplementedError, match="Non-varlen"):
        module.simple_gla_fwd(
            built["q"][None], built["k"][None], built["v"][None], **common
        )
    # Head dims must be multiples of 128.
    narrow = simple_gla_inputs()
    narrow_q = narrow["q"][..., :64]
    with pytest.raises(AssertionError):
        module.simple_gla_fwd(
            narrow_q[None], narrow["k"][..., :64][None],
            narrow["v"][..., :64][None],
            g_gamma=narrow["g_gamma"],
            h0=narrow["h0"][..., :64, :64],
            cu_seqlens_dev=narrow["cu_seqlens"], scale=None, use_ht=True,
            chunk_size=64,
        )


def test_simple_gla_reaches_pallas(modules):
    source = (FAMILY / "sglang_jax_simple_gla_optimized.py").read_text()
    assert source.count("pl.pallas_call(") == 3
    assert modules["simple_gla"].SOURCE["launch_points"] == 3
    built = simple_gla_inputs()
    text = (
        jax.jit(
            modules["simple_gla"].chunk_simple_gla_fwd_varlen,
            static_argnames=("use_ht", "chunk_size"),
        ).lower(
            built["q"][None], built["k"][None], built["v"][None],
            g_gamma=built["g_gamma"], h0=built["h0"],
            cu_seqlens_dev=built["cu_seqlens"], scale=None, use_ht=True,
            chunk_size=64,
        ).compile().as_text()
    )
    assert text.count("tpu_custom_call") >= 1


def test_simple_gla_is_standalone(modules):
    source = (FAMILY / "sglang_jax_simple_gla_optimized.py").read_text()
    for token in ("sgl_jax.", "tpu_inference.", "tokamax._src", "maxtext."):
        assert token not in source


def test_kda_is_standalone(modules):
    source = (FAMILY / "sglang_jax_kda_optimized.py").read_text()
    for token in ("sgl_jax.", "tpu_inference.", "tokamax._src", "maxtext."):
        assert token not in source
