"""TPU correctness for JAXBench flash attention's **backward** launch points.

The migrated `jaxbench_optimized.py` carries three Pallas launch points --
forward, backward-dQ and backward-dKV -- but only the forward was validated
when the family was first migrated. The backward pair is reached by
differentiating through the kernel's `custom_vjp`, the same technique the
splash attention family uses.

Requires a TPU.  Run with::

    uv run --frozen --with pytest python -m pytest \
        tests/test_flash_attention_backward_tpu.py -q
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
FAMILY = ROOT / "kernels" / "attention" / "flash_attention"

pytestmark = pytest.mark.skipif(
    not any("TPU" in device.device_kind for device in jax.devices()),
    reason="flash attention kernels are Mosaic TPU kernels",
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
        "baseline": load("fa_baseline", "baseline.py"),
        "jaxbench": load("fa_jaxbench", "jaxbench_optimized.py"),
    }


SHAPE = (2, 4, 512, 128)


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


def inputs(seed: int = 0):
    batch, heads, seq, dim = SHAPE
    keys = jax.random.split(jax.random.key(seed), 3)
    return tuple(
        jax.random.normal(key, (batch, heads, seq, dim), jnp.float32) * 0.5
        for key in keys
    )


def upstream_imports(source: str) -> list[str]:
    """Modules this file actually imports from an originating repository.

    Imports are read from the AST rather than matched as substrings, because a
    standalone file may legitimately *name* the package it was flattened out
    of -- in a docstring, or in `SOURCE` to record an inlined helper.
    """
    import ast

    packages = ("tpu_inference", "tokamax", "maxtext", "sgl_jax")
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in packages:
                found.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in packages:
                    found.append(alias.name)
    return found


def as_grad(fn):
    def loss(q, k, v):
        return jnp.sum(fn(q, k, v).astype(jnp.float32) ** 2)

    return jax.grad(loss, argnums=(0, 1, 2))


def test_forward_uses_the_documented_entry_point(modules):
    """`kernel` is the corpus entry point; the raw upstream function is not.

    `flash_attention` defaults to non-causal with no `sm_scale`, so calling it
    directly against a `causal_bhsd` reference scores ~0.08 -- it runs, returns
    the right shape, and answers a different question. `kernel` fixes
    `causal=True` and `sm_scale=1/sqrt(D)`, which is the contract the baseline
    implements.
    """
    q, k, v = inputs()
    expected = jax.jit(modules["baseline"].causal_bhsd)(q, k, v)
    assert_matches(jax.jit(modules["jaxbench"].kernel)(q, k, v), expected)
    raw = jax.jit(modules["jaxbench"].flash_attention)(q, k, v)
    assert cosine(raw, expected) < 0.5, "the raw entry point should not match"


def test_backward_matches_the_reference_gradient(modules):
    """Differentiating the custom_vjp exercises backward-dQ and backward-dKV.

    Relative error is ~3e-3 on dQ and dK against ~4e-5 on dV, the same pattern
    the splash family shows and for the same reason: a causal mask leaves many
    gradient entries at or near zero, which inflates any relative statistic.
    Cosine is the criterion, as everywhere else in the corpus.
    """
    q, k, v = inputs()
    expected = jax.jit(as_grad(modules["baseline"].causal_bhsd))(q, k, v)
    actual = jax.jit(as_grad(modules["jaxbench"].kernel))(q, k, v)

    for name, got, want in zip(("dq", "dk", "dv"), actual, expected):
        assert got.shape == want.shape, name
        assert_matches(got, want, name)
        a = np.asarray(got, np.float64)
        e = np.asarray(want, np.float64)
        rms_relative = float(
            np.sqrt(((a - e) ** 2).mean()) / (np.sqrt((e**2).mean()) + 1e-12)
        )
        assert rms_relative < 2e-2, f"{name}: RMS-relative {rms_relative}"


def test_backward_reaches_all_three_launch_points(modules):
    """Forward + backward-dQ + backward-dKV, counted in the lowered HLO."""
    q, k, v = inputs()
    text = (
        jax.jit(as_grad(modules["jaxbench"].kernel))
        .lower(q, k, v).compile().as_text()
    )
    assert text.count("tpu_custom_call") == 3
    # The forward alone is one of them.
    forward = (
        jax.jit(modules["jaxbench"].kernel).lower(q, k, v).compile().as_text()
    )
    assert forward.count("tpu_custom_call") == 1
    # The reference is pure JAX in source, but **XLA rewrites it**: plain
    # einsum + softmax lowers to a `%online-softmax` tpu_custom_call on TPU.
    # So "no tpu_custom_call" is not a valid test for "not a Pallas kernel";
    # `tools/profile_kernel.py` excludes these rewrites by name for the same
    # reason. What must hold is that no *Pallas* kernel is involved.
    reference = (
        jax.jit(as_grad(modules["baseline"].causal_bhsd))
        .lower(q, k, v).compile().as_text()
    )
    pallas = [
        line for line in reference.splitlines()
        if "tpu_custom_call" in line and "online-softmax" not in line
    ]
    assert not pallas, f"reference contains a Pallas launch: {pallas[:1]}"
    assert "online-softmax" in reference, (
        "expected XLA's automatic softmax rewrite; if it is gone, the "
        "exclusion in profile_kernel.XLA_AUTOMATIC_CUSTOM_CALLS is stale"
    )


# ---------------------------------------------------------------------------
# Completing the two attention families
# ---------------------------------------------------------------------------

def test_tpu_inference_forward_matches_the_reference(modules):
    """tpu-inference's flash kernel: forward only, no backward pass.

    Unlike JAXBench's file it is inference-only -- one launch point, no
    `custom_vjp` -- so there is nothing to differentiate.
    """
    tpu_inference = load("fa_tpu_inference", "tpu_inference_optimized.py")
    for shape in ((2, 4, 512, 128), (1, 8, 1024, 128)):
        batch, heads, seq, dim = shape
        keys = jax.random.split(jax.random.key(0), 3)
        q, k, v = (
            jax.random.normal(key, (batch, heads, seq, dim), jnp.float32) * 0.5
            for key in keys
        )
        expected = jax.jit(modules["baseline"].causal_bhsd)(q, k, v)
        assert_matches(jax.jit(tpu_inference.kernel)(q, k, v), expected)

    assert tpu_inference.SOURCE["launch_points"] == 1
    source = (FAMILY / "tpu_inference_optimized.py").read_text()
    assert source.count("pl.pallas_call(") == 1
    # Check actual imports, not substrings: this file *documents* the helper it
    # inlined (`tpu_inference.utils.align_to`) in its docstring and SOURCE, and
    # a substring check cannot tell a mention from a dependency.
    assert not upstream_imports(source), upstream_imports(source)
    # The one repo-local helper is inlined, not imported.
    assert "def align_to(" in source


def test_tokamax_splash_backward_matches_the_reference(modules):
    """Tokamax splash carries forward + backward-dKV, and no separate dQ kernel.

    Its `custom_vjp` produces dQ from the dKV kernel rather than launching a
    third Pallas kernel, which is why the audit records **2** launch points
    here against JAXBench splash's 3.
    """
    tokamax = load("fa_tokamax", "tokamax_optimized.py")
    heads, seq, dim = 4, 512, 128
    keys = jax.random.split(jax.random.key(0), 3)
    # `splash_mha_hsd` is [H, S, D] and the caller pre-scales Q.
    q = jax.random.normal(keys[0], (heads, seq, dim), jnp.float32) * 0.5 * dim**-0.5
    k = jax.random.normal(keys[1], (heads, seq, dim), jnp.float32) * 0.5
    v = jax.random.normal(keys[2], (heads, seq, dim), jnp.float32) * 0.5

    single = tokamax.build_kernel(seq, block_q=128, block_kv=128)
    reference = modules["baseline"].splash_mha_hsd
    assert_matches(jax.jit(single)(q, k, v), jax.jit(reference)(q, k, v))

    expected = jax.jit(as_grad(reference))(q, k, v)
    actual = jax.jit(as_grad(single))(q, k, v)
    for name, got, want in zip(("dq", "dk", "dv"), actual, expected):
        assert_matches(got, want, name)

    hlo = jax.jit(as_grad(single)).lower(q, k, v).compile().as_text()
    pallas = sum(
        1 for line in hlo.splitlines()
        if "tpu_custom_call" in line and "online-softmax" not in line
    )
    assert pallas == 2, "expected forward + backward-dKV only"


# ---------------------------------------------------------------------------
# MaxText's copy of the Tokamax splash lineage
#
# Excluded from the corpus until 2026-08-04 as a "vendored copy of Tokamax
# splash".  Measured, only 7 of its 20 shared top-level definitions are
# AST-identical (35%) -- below the 42% at which both halves of the gdn v3 pair
# were migrated, and nowhere near the 90% that justified excluding JAXBench's
# 4p_Sparse_Attention.  It is now counted as its own migration.
# ---------------------------------------------------------------------------


def test_maxtext_tokamax_splash_backward_matches_the_reference(modules):
    """Same contract and same 2-launch-point structure as Tokamax's."""
    maxtext = load("fa_mt_tksplash", "maxtext_tokamax_splash_optimized.py")
    heads, seq, dim = 4, 512, 128
    keys = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(keys[0], (heads, seq, dim), jnp.float32) * 0.5 * dim**-0.5
    k = jax.random.normal(keys[1], (heads, seq, dim), jnp.float32) * 0.5
    v = jax.random.normal(keys[2], (heads, seq, dim), jnp.float32) * 0.5

    single = maxtext.build_kernel(seq, block_q=128, block_kv=128)
    reference = modules["baseline"].splash_mha_hsd
    assert_matches(jax.jit(single)(q, k, v), jax.jit(reference)(q, k, v))

    expected = jax.jit(as_grad(reference))(q, k, v)
    actual = jax.jit(as_grad(single))(q, k, v)
    for name, got, want in zip(("dq", "dk", "dv"), actual, expected):
        assert_matches(got, want, name)

    hlo = jax.jit(as_grad(single)).lower(q, k, v).compile().as_text()
    pallas = sum(
        1 for line in hlo.splitlines()
        if "tpu_custom_call" in line and "online-softmax" not in line
    )
    assert pallas == 2, "expected forward + backward-dKV only"
    assert maxtext.SOURCE["launch_points"] == 2


def test_the_two_tokamax_lineage_splashes_agree(modules):
    """They diverged in source but must still compute the same thing."""
    tokamax = load("fa_tokamax_pair", "tokamax_optimized.py")
    maxtext = load("fa_mt_pair", "maxtext_tokamax_splash_optimized.py")
    heads, seq, dim = 4, 512, 128
    keys = jax.random.split(jax.random.key(1), 3)
    q = jax.random.normal(keys[0], (heads, seq, dim), jnp.float32) * 0.5 * dim**-0.5
    k = jax.random.normal(keys[1], (heads, seq, dim), jnp.float32) * 0.5
    v = jax.random.normal(keys[2], (heads, seq, dim), jnp.float32) * 0.5

    a = jax.jit(tokamax.build_kernel(seq, block_q=128, block_kv=128))(q, k, v)
    b = jax.jit(maxtext.build_kernel(seq, block_q=128, block_kv=128))(q, k, v)
    assert_matches(a, b)


def test_maxtext_copy_has_diverged_from_tokamax():
    """Pin the measurement the migration decision rests on.

    If these two converge back above ~90%, the exclusion the corpus originally
    made would become the right call again and this should be revisited.
    """
    import ast

    def defs(filename):
        text = (FAMILY / filename).read_text()
        tree = ast.parse(text)
        out = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                body = ast.parse(ast.get_source_segment(text, node))
                for sub in ast.walk(body):
                    if isinstance(sub, (ast.FunctionDef, ast.ClassDef,
                                        ast.Module)):
                        if (sub.body and isinstance(sub.body[0], ast.Expr)
                                and isinstance(sub.body[0].value, ast.Constant)
                                and isinstance(sub.body[0].value.value, str)):
                            sub.body = sub.body[1:] or [ast.Pass()]
                out[node.name] = ast.dump(ast.fix_missing_locations(body))
        return out

    a = defs("tokamax_optimized.py")
    b = defs("maxtext_tokamax_splash_optimized.py")
    shared = set(a) & set(b)
    identical = sum(1 for name in shared if a[name] == b[name])
    assert shared, "no shared definitions -- did a filename change?"
    ratio = identical / len(shared)
    assert ratio < 0.6, (
        f"{identical}/{len(shared)} definitions identical ({ratio:.0%}); the "
        "two copies have converged, so re-examine whether MaxText's should "
        "still be counted separately"
    )


# ---------------------------------------------------------------------------
# PallasBench's dense_2d flash attention
#
# Recorded as migrated and PASS since it was collected, but until 2026-08-04
# no test in this suite exercised it: its correctness verdict came from the
# profiling run, which is not part of the test suite.  Same failure mode as
# grouped matmul's `tgmm_v2`, found the same way -- by recording which Pallas
# launches actually fire during the suite and comparing against the ledger.
# ---------------------------------------------------------------------------


def test_pallasbench_dense_2d_matches_the_reference(modules):
    """A third contract in this directory: non-causal, 2-D, no batch or heads."""
    pallasbench = load("fa_pallasbench", "pallasbench_optimized.py")
    baseline = modules["baseline"]
    q, k, v = baseline.create_inputs(
        contract="dense_2d", sequence=512, head_dim=64
    )
    actual = jax.jit(pallasbench.kernel)(q, k, v)
    expected = jax.jit(baseline.dense_2d)(q, k, v)
    assert actual.shape == expected.shape == (512, 64)
    assert_matches(actual, expected)


def test_pallasbench_is_not_causal(modules):
    """`dense_2d` attends over the whole sequence; `causal_bhsd` does not.

    Guards against the three contracts in this directory being treated as one:
    a causal kernel fed these inputs would disagree on every row but the last.
    """
    pallasbench = load("fa_pallasbench_causal", "pallasbench_optimized.py")
    baseline = modules["baseline"]
    q, k, v = baseline.create_inputs(
        contract="dense_2d", sequence=256, head_dim=64
    )
    actual = np.asarray(jax.jit(pallasbench.kernel)(q, k, v), np.float32)

    logits = np.asarray(q, np.float32) @ np.asarray(k, np.float32).T / 8.0
    causal = np.where(
        np.tril(np.ones((256, 256), bool)), logits, -np.inf
    )
    causal = np.exp(causal - causal.max(-1, keepdims=True))
    causal /= causal.sum(-1, keepdims=True)
    causal_out = causal @ np.asarray(v, np.float32)
    assert cosine(actual, causal_out) < 0.99, (
        "the kernel matches a causal reference; the dense_2d contract note "
        "in baseline.py would then be wrong"
    )


def test_pallasbench_reaches_pallas_and_is_standalone():
    source = (FAMILY / "pallasbench_optimized.py").read_text()
    assert source.count("pl.pallas_call(") == 1
    assert not upstream_imports(source), upstream_imports(source)
