"""Small-shape TPU correctness for the gated delta net family (v1 pass).

The gated delta rule is a linear-attention recurrence over ragged sequences,
carrying a per-head ``[d_k, d_v]`` recurrent state.

The reference is **upstream's own**: tpu-inference ships
``kernels/gdn/reference/ragged_gated_delta_rule_ref.py``, which imports nothing
outside jax and which upstream describes as "mainly for unit test".

Requires a TPU.  Run with::

    uv run --frozen --with pytest python -m pytest \
        tests/test_gated_delta_net_tpu.py -q
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
FAMILY = ROOT / "kernels" / "state_space" / "gated_delta_net"

pytestmark = pytest.mark.skipif(
    not any("TPU" in device.device_kind for device in jax.devices()),
    reason="gated delta net kernels are Mosaic TPU kernels",
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
        "baseline": load("gdn_baseline", "baseline.py"),
        "v1": load("gdn_v1", "tpu_inference_v1_optimized.py"),
    }


SHAPES = [(256, 4, 8, 16), (512, 8, 8, 16)]


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


def build(modules, shape):
    num_tokens, num_seqs, n_kq, n_v = shape
    built = modules["baseline"].create_inputs(
        num_tokens=num_tokens, num_seqs=num_seqs, n_kq=n_kq, n_v=n_v
    )
    static = built["static"]
    args = (
        built["mixed_qkv"], built["b"], built["a"], built["recurrent_state"],
        built["A_log"], built["dt_bias"], built["query_start_loc"],
        built["state_indices"], built["distribution"],
        built["has_initial_state"],
    )
    return built, static, args


def run(fn, args, static):
    return jax.jit(fn, static_argnames=tuple(static))(*args, **static)


def pallas_launches(fn, args, static) -> int:
    return (
        jax.jit(fn, static_argnames=tuple(static))
        .lower(*args, **static).compile().as_text()
    ).count("tpu_custom_call")


@pytest.mark.parametrize("shape", SHAPES)
def test_v1_matches_upstream_reference(modules, shape):
    """Each side is fed according to its own SiLU precondition."""
    baseline, v1 = modules["baseline"], modules["v1"]
    built, static, args = build(modules, shape)
    kargs = (baseline.to_post_silu(built["mixed_qkv"]),) + args[1:]

    ref_state, ref_out = run(baseline.ragged_gated_delta_rule, args, static)
    got_state, got_out = run(v1.ragged_gated_delta_rule, kargs, static)

    assert got_out.shape == ref_out.shape
    assert got_state.shape == ref_state.shape
    assert_matches(got_out, ref_out)
    assert_matches(got_state, ref_state)


def test_v1_reaches_pallas(modules):
    """Three launch points: chunk metadata, recurrent prefill, decode."""
    baseline, v1 = modules["baseline"], modules["v1"]
    built, static, args = build(modules, SHAPES[0])
    kargs = (baseline.to_post_silu(built["mixed_qkv"]),) + args[1:]
    assert pallas_launches(v1.ragged_gated_delta_rule, kargs, static) == 3
    # The reference must stay pure JAX or it is not a valid baseline.
    assert pallas_launches(baseline.ragged_gated_delta_rule, args, static) == 0


def test_the_silu_precondition_is_real_and_matters(modules):
    """Pin the divergence that a matching signature hides.

    Both entry points take byte-identical argument lists. The reference applies
    ``jax.nn.silu`` to ``mixed_qkv`` itself; the v1 wrapper documents its input
    as already post-SiLU. Feeding the kernel raw input therefore runs, returns
    the right shapes, and is quietly wrong -- measured at cosine ~0.72.

    If upstream ever moves the SiLU inside the kernel this test fails, which is
    the point: the corpus should notice rather than keep converting.
    """
    baseline, v1 = modules["baseline"], modules["v1"]
    built, static, args = build(modules, SHAPES[0])

    _, ref_out = run(baseline.ragged_gated_delta_rule, args, static)
    _, raw_out = run(v1.ragged_gated_delta_rule, args, static)
    _, silu_out = run(
        v1.ragged_gated_delta_rule,
        (baseline.to_post_silu(built["mixed_qkv"]),) + args[1:],
        static,
    )

    assert cosine(raw_out, ref_out) < 0.9, "raw input should be visibly wrong"
    assert_matches(silu_out, ref_out, "post-SiLU input should match")
    assert baseline.PRE_SILU_INPUT["baseline.ragged_gated_delta_rule"] is True
    assert (
        baseline.PRE_SILU_INPUT["tpu_inference_v1.ragged_gated_delta_rule"]
        is False
    )


def test_both_block_size_helpers_survived_flattening(modules):
    """Upstream defines `get_default_block_sizes` twice, with different bodies.

    Decode and recurrent each have their own; concatenating the modules would
    shadow one. Both must exist under distinct names and stay distinct.
    """
    v1 = modules["v1"]
    assert hasattr(v1, "get_default_block_sizes_decode")
    assert hasattr(v1, "get_default_block_sizes_recurrent")
    assert not hasattr(v1, "get_default_block_sizes")

    import inspect

    decode = inspect.getsource(v1.get_default_block_sizes_decode)
    recurrent = inspect.getsource(v1.get_default_block_sizes_recurrent)
    assert decode != recurrent, "the two helpers collapsed into one body"


def test_v1_is_standalone(modules):
    """No import of the originating repository package."""
    source = (FAMILY / "tpu_inference_v1_optimized.py").read_text()
    for token in ("tpu_inference.", "tokamax._src", "maxtext.", "sgl_jax."):
        assert token not in source
    assert modules["v1"].SOURCE["launch_points"] == 3


# ---------------------------------------------------------------------------
# Contract 2: fused_conv1d_gated_delta_rule (the v3 kernels)
# ---------------------------------------------------------------------------
# These fuse a depthwise causal conv1d with the gated delta rule and carry two
# caches -- conv state and recurrent state -- where v1 carries one. Different
# contract, different reference: Tokamax's own `run_jax_gdn_attention_local_ref`.

V3_IMPLEMENTATIONS = ("tokamax_v3", "tpu_inference_v3")


@pytest.fixture(scope="module")
def v3_modules():
    return {name: load(f"gdn_{name}", f"{name}_optimized.py")
            for name in V3_IMPLEMENTATIONS}


def build_fused(modules):
    built = modules["baseline"].create_inputs_fused()
    return built, built["static"], modules["baseline"].fused_args(built)


@pytest.mark.parametrize("implementation", V3_IMPLEMENTATIONS)
def test_v3_matches_tokamax_fused_reference(modules, v3_modules, implementation):
    baseline = modules["baseline"]
    built, static, args = build_fused(modules)

    reference = jax.jit(
        baseline.run_jax_gdn_attention_local_ref,
        static_argnames=tuple(static) + ("config",),
    )(*args, **static, config=baseline.GdnAttentionConfig())
    actual = jax.jit(
        v3_modules[implementation].fused_conv1d_gdn, static_argnames=tuple(static)
    )(*args, **static)

    expected_leaves = jax.tree.leaves(reference)
    actual_leaves = jax.tree.leaves(actual)
    assert len(actual_leaves) == len(expected_leaves)
    for got, want in zip(actual_leaves, expected_leaves):
        assert got.shape == want.shape
        assert_matches(got, want)


@pytest.mark.parametrize("implementation", V3_IMPLEMENTATIONS)
def test_v3_reaches_pallas(v3_modules, modules, implementation):
    built, static, args = build_fused(modules)
    assert pallas_launches(
        v3_modules[implementation].fused_conv1d_gdn, args, static
    ) >= 1


def test_the_two_v3_implementations_agree(modules, v3_modules):
    """A diverged vendored pair that still computes the same thing.

    tpu-inference `gdn/v3` and Tokamax `causal_conv1d_gated_delta_rule` share
    seven module names and 24 definition names but only ~42% AST-identical
    bodies. The divergence is in organisation, not numerics -- so this asserts
    they agree far more tightly than either does with the reference.
    """
    built, static, args = build_fused(modules)
    outputs = [
        jax.jit(v3_modules[name].fused_conv1d_gdn, static_argnames=tuple(static))(
            *args, **static
        )
        for name in V3_IMPLEMENTATIONS
    ]
    left, right = (jax.tree.leaves(o) for o in outputs)
    for got, want in zip(left, right):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_v3_is_a_different_contract_from_v1(modules, v3_modules):
    """Guard against the two contracts being conflated.

    v1's `ragged_gated_delta_rule` has no conv1d and no conv cache; v3's
    `fused_conv1d_gdn` requires both. They are not tuning variants.
    """
    for name in V3_IMPLEMENTATIONS:
        module = v3_modules[name]
        assert module.SOURCE["contract"] == "fused_conv1d_gated_delta_rule"
        assert not hasattr(module, "ragged_gated_delta_rule")
    assert modules["v1"].SOURCE["contract"] == "ragged_gated_delta_rule"

    built = modules["baseline"].create_inputs_fused()
    assert "conv_state" in built and "conv_weight" in built
    assert "conv_state" not in modules["baseline"].create_inputs()


@pytest.mark.parametrize("implementation", V3_IMPLEMENTATIONS)
def test_v3_is_standalone(implementation):
    source = (FAMILY / f"{implementation}_optimized.py").read_text()
    for token in ("tpu_inference.", "tokamax._src", "maxtext.", "sgl_jax."):
        assert token not in source


# ---------------------------------------------------------------------------
# v2: same contract as v1, two independent entry points
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def v2_module():
    return load("gdn_v2", "tpu_inference_v2_optimized.py")


def decode_batch(modules, num_seqs=4):
    """A decode batch is one token per request, and `distribution` must say so."""
    built = modules["baseline"].create_inputs(
        num_tokens=num_seqs, num_seqs=num_seqs
    )
    built["distribution"] = jnp.array(
        [num_seqs, num_seqs, num_seqs], jnp.int32
    )
    static = built["static"]
    args = (
        built["mixed_qkv"], built["b"], built["a"], built["recurrent_state"],
        built["A_log"], built["dt_bias"], built["query_start_loc"],
        built["state_indices"], built["distribution"],
        built["has_initial_state"],
    )
    return built, static, args


def test_v2_decode_only_matches_reference(modules, v2_module):
    built, static, args = decode_batch(modules)
    _, expected = run(modules["baseline"].ragged_gated_delta_rule, args, static)
    _, actual = jax.jit(
        v2_module.ragged_gated_delta_rule_decode_only,
        static_argnames=tuple(static) + ("apply_silu",),
    )(*args, **static, apply_silu=True)
    assert_matches(actual, expected)


def test_v2_apply_silu_flag_is_consistent(modules, v2_module):
    """`apply_silu=True` on raw input must equal `apply_silu=False` on SiLU'd.

    v2 turns the precondition that v1 leaves implicit into an argument. This
    checks the argument actually does what its name says.
    """
    baseline = modules["baseline"]
    built, static, args = decode_batch(modules)
    fn = jax.jit(
        v2_module.ragged_gated_delta_rule_decode_only,
        static_argnames=tuple(static) + ("apply_silu",),
    )
    _, applied = fn(*args, **static, apply_silu=True)
    _, pre = fn(
        baseline.to_post_silu(built["mixed_qkv"]), *args[1:], **static,
        apply_silu=False,
    )
    assert_matches(applied, pre)


@pytest.mark.parametrize("shape", [(256, 4, 64), (512, 4, 128)])
def test_v2_recurrent_scan_matches_reference(modules, v2_module, shape):
    """`recurrent_scan` applies SiLU itself -- the opposite of v1's wrapper."""
    num_tokens, num_seqs, chunk = shape
    built = modules["baseline"].create_inputs(
        num_tokens=num_tokens, num_seqs=num_seqs
    )
    static = built["static"]
    args = (
        built["mixed_qkv"], built["b"], built["a"], built["recurrent_state"],
        built["A_log"], built["dt_bias"], built["query_start_loc"],
        built["state_indices"], built["distribution"],
    )
    expected_state, expected = run(
        modules["baseline"].ragged_gated_delta_rule,
        args + (built["has_initial_state"],), static,
    )
    got_state, got = jax.jit(
        v2_module.recurrent_scan,
        static_argnames=tuple(static)
        + ("chunk_size", "BT", "use_qk_norm_in_gdn"),
    )(
        *args, **static, chunk_size=chunk, BT=chunk,
        use_qk_norm_in_gdn=True,
        has_initial_state=built["has_initial_state"],
    )
    assert_matches(got, expected)
    assert_matches(got_state, expected_state)


def test_v2_recurrent_scan_silu_convention_differs_from_v1(modules, v2_module):
    """The two kernels in the same repo disagree about who applies SiLU.

    Feeding `recurrent_scan` post-SiLU input scores ~0.983 -- wrong, but close
    enough to read as a tolerance problem rather than a contract error. That is
    the trap this pins.
    """
    baseline = modules["baseline"]
    built = baseline.create_inputs(num_tokens=256, num_seqs=4)
    static = built["static"]
    args = (
        built["mixed_qkv"], built["b"], built["a"], built["recurrent_state"],
        built["A_log"], built["dt_bias"], built["query_start_loc"],
        built["state_indices"], built["distribution"],
    )
    _, expected = run(
        baseline.ragged_gated_delta_rule,
        args + (built["has_initial_state"],), static,
    )
    fn = jax.jit(
        v2_module.recurrent_scan,
        static_argnames=tuple(static)
        + ("chunk_size", "BT", "use_qk_norm_in_gdn"),
    )
    kwargs = dict(
        static, chunk_size=64, BT=64, use_qk_norm_in_gdn=True,
        has_initial_state=built["has_initial_state"],
    )
    _, raw = fn(*args, **kwargs)
    _, double_silu = fn(baseline.to_post_silu(args[0]), *args[1:], **kwargs)

    assert_matches(raw, expected)
    assert cosine(double_silu, expected) < 0.99
    assert baseline.PRE_SILU_INPUT["tpu_inference_v2.recurrent_scan"] is True
    assert baseline.PRE_SILU_INPUT[
        "tpu_inference_v1.ragged_gated_delta_rule"
    ] is False


def test_v2_qk_norm_is_required_not_optional(modules, v2_module):
    """At `use_qk_norm_in_gdn=False` the kernel computes something else.

    Cosine drops to ~7e-4, not to a slightly-worse number -- so this is not a
    quality knob and must not be treated as one.
    """
    baseline = modules["baseline"]
    built = baseline.create_inputs(num_tokens=256, num_seqs=4)
    static = built["static"]
    args = (
        built["mixed_qkv"], built["b"], built["a"], built["recurrent_state"],
        built["A_log"], built["dt_bias"], built["query_start_loc"],
        built["state_indices"], built["distribution"],
    )
    _, expected = run(
        baseline.ragged_gated_delta_rule,
        args + (built["has_initial_state"],), static,
    )
    _, without = jax.jit(
        v2_module.recurrent_scan,
        static_argnames=tuple(static)
        + ("chunk_size", "BT", "use_qk_norm_in_gdn"),
    )(
        *args, **static, chunk_size=64, BT=64, use_qk_norm_in_gdn=False,
        has_initial_state=built["has_initial_state"],
    )
    assert cosine(without, expected) < 0.01
    assert baseline.V2_RECURRENT_SCAN_REQUIRES_QK_NORM is True


def test_v2_is_standalone(v2_module):
    source = (FAMILY / "tpu_inference_v2_optimized.py").read_text()
    for token in ("tpu_inference.", "tokamax._src", "maxtext.", "sgl_jax."):
        assert token not in source
    assert v2_module.SOURCE["launch_points"] == 2
    # Both `invert_triangular_matrix` definitions must have survived distinctly.
    assert hasattr(v2_module, "invert_triangular_matrix_impl")
    assert hasattr(v2_module, "invert_triangular_matrix_scan")
    assert not hasattr(v2_module, "invert_triangular_matrix")


# ---------------------------------------------------------------------------
# Contract 3: unit_lower_triangular_inverse (triangle_solver)
# ---------------------------------------------------------------------------
# Helpers for the chunked scan, and the only part of this family whose contract
# is a plain mathematical identity -- so they are checked against
# `jnp.linalg.inv`, an oracle independent of upstream. Everywhere else here the
# only reference is the one upstream wrote, and an error shared between kernel
# and reference would go unnoticed.

@pytest.fixture(scope="module")
def triangle_module():
    return load("gdn_ts", "tpu_inference_triangle_solver_optimized.py")


def unit_lower_triangular(batch, n, seed=0):
    """Unit lower triangular, kept well-conditioned.

    The inverse of a unit lower triangular matrix with strict-lower entries of
    magnitude c grows roughly like ``(1 + c)**n``, so a *fixed* c makes the
    problem exponentially ill-conditioned in n. At c=0.5, n=128 the true
    inverse has entries ~1e5 and float32 cannot represent it -- a test built
    that way measures the matrix, not the kernel. Scaling c with 1/n keeps the
    condition number flat.
    """
    scale = 0.5 / n
    matrix = jax.random.normal(jax.random.key(seed), (batch, n, n), jnp.float32)
    return jnp.tril(matrix * scale, -1) + jnp.eye(n, dtype=jnp.float32)[None]


# The validated envelope. Both kernels materialise the whole [batch, n, n] in
# VMEM, so n is bounded by the v6e's 128 MiB: newton_schulz OOMs at n>=128, and
# the blockwise kernel reaches n=128 but OOMs at n=256.
TRIANGLE_SHAPES = {
    "newton_schulz_inverse_pallas": [(4, 64)],
    "decompose_triangular_matrix_inverse_pallas": [(4, 64), (4, 128)],
}


def triangle_call(module, name, matrix, n):
    entry = getattr(module, name)
    if name == "newton_schulz_inverse_pallas":
        return jax.jit(entry, static_argnames=("block_size",))(
            matrix, block_size=64
        )
    return jax.jit(entry, static_argnames=("n_block_size", "block_size"))(
        matrix, n_block_size=64, block_size=16
    )


@pytest.mark.parametrize("name", sorted(TRIANGLE_SHAPES))
def test_triangle_solver_matches_linalg_inv(triangle_module, name):
    """Checked against an oracle upstream had no hand in."""
    for batch, n in TRIANGLE_SHAPES[name]:
        matrix = unit_lower_triangular(batch, n)
        expected = jnp.linalg.inv(matrix)
        actual = triangle_call(triangle_module, name, matrix, n)
        assert actual.shape == expected.shape
        assert_matches(actual, expected)
        assert float(jnp.max(jnp.abs(actual - expected))) < 1e-5


@pytest.mark.parametrize("name", sorted(TRIANGLE_SHAPES))
def test_triangle_solver_inverse_is_a_real_inverse(triangle_module, name):
    """`A^-1 @ A` must be the identity -- a property check, not a comparison."""
    batch, n = TRIANGLE_SHAPES[name][0]
    matrix = unit_lower_triangular(batch, n)
    inverse = triangle_call(triangle_module, name, matrix, n)
    residual = jnp.max(
        jnp.abs(jnp.einsum("bij,bjk->bik", inverse, matrix) - jnp.eye(n))
    )
    assert float(residual) < 1e-3


def test_triangle_solver_reaches_pallas(triangle_module):
    matrix = unit_lower_triangular(4, 64)
    for name in TRIANGLE_SHAPES:
        entry = getattr(triangle_module, name)
        kwargs = (
            {"block_size": 64}
            if name == "newton_schulz_inverse_pallas"
            else {"n_block_size": 64, "block_size": 16}
        )
        text = (
            jax.jit(entry, static_argnames=tuple(kwargs))
            .lower(matrix, **kwargs).compile().as_text()
        )
        assert text.count("tpu_custom_call") >= 1


def test_upstream_reference_agrees_with_the_oracle(triangle_module):
    """Upstream's own pure-JAX reference is itself correct, at this shape.

    Worth checking separately: it is the reference the other contracts in this
    family would have to trust blindly.
    """
    matrix = unit_lower_triangular(4, 64)
    reference = triangle_module.newton_schulz_inverse_ref(matrix, 64)
    assert_matches(reference, jnp.linalg.inv(matrix))


def test_triangle_solver_is_standalone(triangle_module):
    source = (FAMILY / "tpu_inference_triangle_solver_optimized.py").read_text()
    for token in ("tpu_inference.", "tokamax._src", "maxtext.", "sgl_jax."):
        assert token not in source
    assert triangle_module.SOURCE["launch_points"] == 2
