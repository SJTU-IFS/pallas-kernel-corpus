"""TPU correctness for JAXBench's paged-attention decode kernel.

Both sides are JAXBench's own — `optimized.py` against the `baseline.py` beside
it — at the tolerance `optimized.py` declares (atol 1e-2, rtol 2e-2, loose
because this is bf16 attention over a 4096-token cache).

Getting them to agree took two reconciliations, and neither is visible in either
signature. They are pinned by tests here rather than left as prose:

* **the kernel takes pre-scaled Q.** It has no `sm_scale` parameter and applies
  none; the baseline applies `head_dim ** -0.5` internally. Feed the kernel raw
  Q and the two disagree by 2.9 against a reference peaking at 0.14 — not a
  tolerance failure, a different function. JAXBench's own `workload` does not
  pre-scale, so its two files would not agree if anything compared them; they
  are only ever benchmarked.
* **the KV page layout differs.** The baseline indexes `(total_pages,
  page_size, num_kv_heads, head_dim)`; the kernel takes `(num_kv_heads,
  total_pages, page_size, head_dim)`. That is an axis permutation, applied
  explicitly below.

Requires a TPU.  Run with::

    uv run --frozen --with pytest python -m pytest \
        tests/test_paged_attention_tpu.py -q
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
FAMILY = ROOT / "kernels" / "attention" / "paged_attention"

pytestmark = pytest.mark.skipif(
    not any("TPU" in device.device_kind for device in jax.devices()),
    reason="the JAXBench paged-attention kernel is a Mosaic TPU kernel",
)


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, FAMILY / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    return (load("pa_kernel", "jaxbench_optimized.py"),
            load("pa_reference", "baseline.py"))


def as_reference_layout(pages):
    """`(H_kv, total_pages, page, D)` -> `(total_pages, page, H_kv, D)`.

    A relabelling of axes, not a reading of what either side means.
    """
    return jnp.transpose(pages, (1, 2, 0, 3))


def reference_output(kernel, reference, q, k_pages, v_pages, lengths, page_indices):
    cu_q_lens = jnp.arange(kernel.CONFIG["batch"] + 1, dtype=jnp.int32)
    return reference.workload(q, as_reference_layout(k_pages),
                              as_reference_layout(v_pages), lengths,
                              page_indices, cu_q_lens)


def prescale(kernel, q):
    """What the kernel expects: Q already divided by sqrt(head_dim)."""
    scale = kernel.CONFIG["head_dim"] ** -0.5
    return (q.astype(jnp.float32) * scale).astype(q.dtype)


def test_matches_jaxbench_reference(modules):
    """The Pallas kernel against JAXBench's own JAX baseline, at native shape."""
    kernel, reference = modules
    q, k_pages, v_pages, lengths, page_indices = kernel.create_inputs()

    actual = kernel.kernel(prescale(kernel, q), k_pages, v_pages, lengths,
                           page_indices)
    expected = reference_output(kernel, reference, q, k_pages, v_pages, lengths,
                                page_indices)
    jax.block_until_ready((actual, expected))

    actual = np.asarray(actual, np.float32)
    expected = np.asarray(expected, np.float32)
    assert actual.shape == expected.shape == (
        kernel.CONFIG["batch"], kernel.CONFIG["num_q_heads"],
        kernel.CONFIG["head_dim"])

    atol, rtol = kernel.CONFIG["atol"], kernel.CONFIG["rtol"]
    # Not vacuous at this magnitude: the reference peaks well above the bar.
    assert not np.allclose(np.zeros_like(expected), expected, atol=atol,
                           rtol=rtol), "a zeroed kernel would pass this"
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def test_the_kernel_takes_prescaled_q(modules):
    """Pin the convention, since neither signature carries it.

    If a future kernel started applying `sm_scale` itself, the pre-scaled call
    above would become the wrong one — and this test would fail, which is the
    point of asserting the size of the disagreement rather than just noting it.
    """
    kernel, reference = modules
    q, k_pages, v_pages, lengths, page_indices = kernel.create_inputs()
    expected = np.asarray(
        reference_output(kernel, reference, q, k_pages, v_pages, lengths,
                         page_indices), np.float32)

    raw = np.asarray(kernel.kernel(q, k_pages, v_pages, lengths, page_indices),
                     np.float32)
    scaled = np.asarray(
        kernel.kernel(prescale(kernel, q), k_pages, v_pages, lengths,
                      page_indices), np.float32)

    peak = float(np.max(np.abs(expected)))
    assert np.max(np.abs(raw - expected)) > peak, (
        "unscaled Q should disagree with the reference by more than its whole "
        "magnitude; if it no longer does, the kernel now scales internally")
    assert np.max(np.abs(scaled - expected)) < kernel.CONFIG["atol"]


def test_the_reference_is_not_argument_compatible(modules):
    """JAXBench's two files here do not take the same arguments.

    Worth a test rather than a comment: in `8p_GEMM` the pair *is* drop-in, so
    a reader who generalises from there would feed these the same arrays and
    get a shape error at best.
    """
    kernel, reference = modules
    kernel_inputs = kernel.create_inputs()
    reference_inputs = reference.create_inputs()
    assert len(kernel_inputs) == 5 and len(reference_inputs) == 6, (
        "the reference takes an extra cu_q_lens")

    _, k_kernel, *_ = kernel_inputs
    _, k_reference, *_ = reference_inputs
    assert k_kernel.shape != k_reference.shape, (
        "the two KV page layouts should differ")
    heads, pages, page_size, dim = k_kernel.shape
    assert as_reference_layout(k_kernel).shape == (pages, page_size, heads, dim)
    assert k_reference.shape[2] == heads and k_reference.shape[3] == dim, (
        "the reference layout puts heads on axis 2")


def test_reaches_pallas(modules):
    """The standing rule: a passing comparison does not prove the kernel ran."""
    sys.path.insert(0, str(ROOT / "tools"))
    from profile_kernel import count_pallas_launches

    kernel, _ = modules
    q, k_pages, v_pages, lengths, page_indices = kernel.create_inputs()
    launches = count_pallas_launches(
        kernel.kernel, (prescale(kernel, q), k_pages, v_pages, lengths,
                        page_indices))
    assert launches == 1, f"{launches} Pallas launches"


def test_the_reference_is_not_itself_pallas(modules):
    """A reference that lowered to Mosaic would not be an independent check."""
    sys.path.insert(0, str(ROOT / "tools"))
    from profile_kernel import count_pallas_launches

    kernel, reference = modules
    q, k_pages, v_pages, lengths, page_indices = kernel.create_inputs()
    cu_q_lens = jnp.arange(kernel.CONFIG["batch"] + 1, dtype=jnp.int32)
    assert count_pallas_launches(
        reference.workload,
        (q, as_reference_layout(k_pages), as_reference_layout(v_pages),
         lengths, page_indices, cu_q_lens)) == 0


def test_a_triton_compiler_param_cannot_lower_on_tpu():
    """The evidence behind excluding sglang-jax's paged attention.

    That kernel passes `plgpu.CompilerParams(num_warps=..., num_stages=...)`
    unconditionally, and its own test file says "This test suite is designed to
    be executed on GPU". Mosaic's TPU lowering asserts it was given *TPU*
    compiler params, so the kernel cannot lower here at all — which is why the
    corpus classifies it `gpu` rather than leaving it open work.

    This reproduces the mechanism rather than importing that file, since the
    file is not in the corpus precisely because it cannot run. If Mosaic ever
    accepts Triton params, this fails and the exclusion should be revisited.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    def copy_kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...]

    x = jnp.zeros((8, 128), jnp.float32)
    with pytest.raises(Exception):
        jax.jit(lambda a: pl.pallas_call(
            copy_kernel,
            out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
            compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=2),
        )(a))(x)
