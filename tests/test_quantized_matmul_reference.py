"""Which quantized-matmul reference is the honest one — checkable without a TPU.

`test_quantized_matmul_tpu.py` is gated on TPU hardware, as it must be: it runs
Mosaic kernels. These checks are not, because both references are pure JAX, and
they exist because an audit found the two disagree.

`kernels/quantization/quantized_matmul/baseline.py` writes its own reference
although upstream ships one — sglang-jax's `xla_quantized_matmul`, which this
corpus already carries verbatim in `sglang_jax_optimized.py`. The corpus version
truncates where upstream rounds, and its own docstring says it "mirrors the
kernels' `quantize_array`". A reference that shares the kernel's quantizer
cannot catch an error in that quantizer.

These tests pin the discrepancy so it cannot drift unnoticed, and record which
of the two is nearer the exact float32 product. They do not assert the kernel is
wrong — the kernel is not run here at all.

    pytest tests/test_quantized_matmul_reference.py -q
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


@pytest.fixture(scope="module")
def baseline():
    spec = importlib.util.spec_from_file_location(
        "qm_reference_baseline", FAMILY / "baseline.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["qm_reference_baseline"] = module
    spec.loader.exec_module(module)
    return module


def int8_case(rows=128, cols=512, outs=256, seed=0):
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.normal(size=(rows, cols)).astype(np.float32)),
            jnp.asarray(rng.normal(size=(outs, cols)).astype(np.float32)))


def similarity(a, e):
    a = np.asarray(a, np.float64).ravel()
    e = np.asarray(e, np.float64).ravel()
    return float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e)))


def test_upstream_reference_is_carried_and_importable(baseline):
    """The whole point is that upstream's is available, so check it is here."""
    for name in ("upstream_quantize_tensor", "upstream_xla_quantized_matmul",
                 "upstream_quantize_block", "upstream_get_max_min"):
        assert callable(getattr(baseline, name)), name


def test_the_corpus_reference_truncates_where_upstream_rounds(baseline):
    """Roughly half of all int8 codes come out one step apart.

    `quantize_per_token` casts, truncating toward zero; upstream's
    `quantize_block` applies `jnp.round` first on the integer path.
    """
    x, _ = int8_case()
    ours = np.asarray(baseline.quantize_per_token(x, jnp.int8)[0], np.int32)
    theirs = np.asarray(baseline.upstream_quantize_tensor(x, jnp.int8)[0], np.int32)

    delta = np.abs(ours - theirs)
    assert delta.max() <= 1, "codes should differ by at most one quantization step"
    differing = float((delta > 0).mean())
    assert 0.3 < differing < 0.7, (
        f"expected roughly half the codes to differ, got {differing:.1%}; if this "
        f"moved, one of the two quantizers changed")


def product(baseline, x, w, quantizer):
    x_q, x_scale = quantizer(x, jnp.int8)
    w_q, w_scale = baseline.upstream_quantize_tensor(w, jnp.int8)
    accumulator = jax.lax.dot_general(
        x_q, w_q, (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.int32).astype(jnp.float32)
    return (np.asarray(accumulator, np.float64)
            * np.asarray(x_scale, np.float64)
            * np.asarray(w_scale, np.float64).reshape(1, -1))


def test_upstream_is_the_closer_oracle(baseline):
    """Upstream's reference sits nearer the exact float32 product than ours.

    This is why `baseline.py`'s docstring calls the corpus reference a weakness
    rather than a stylistic choice. If a change ever makes the corpus version
    the closer one, this fails and that paragraph should be rewritten.
    """
    x, w = int8_case()
    exact = np.asarray(jnp.dot(x, w.T), np.float64)
    ours = product(baseline, x, w, baseline.quantize_per_token)
    theirs = product(baseline, x, w, baseline.upstream_quantize_tensor)

    assert similarity(theirs, exact) > similarity(ours, exact), (
        f"upstream {similarity(theirs, exact):.7f} should beat "
        f"corpus {similarity(ours, exact):.7f}")


def test_the_two_references_disagree_by_more_than_the_test_tolerance(baseline):
    """They are not interchangeable, which is the operative fact.

    The kernel tests compare at `cosine > 0.9999`. The two references sit closer
    to each other than that — so swapping one for the other is not a free
    change, and needs a TPU run to confirm the kernel still passes.
    """
    x, w = int8_case()
    ours = product(baseline, x, w, baseline.quantize_per_token)
    theirs = product(baseline, x, w, baseline.upstream_quantize_tensor)
    agreement = similarity(ours, theirs)
    assert agreement < 0.9999, (
        f"the two references now agree to {agreement:.7f}, within the kernel "
        f"tests' own bar; if that is real, the comparison can be switched to "
        f"upstream's reference")
