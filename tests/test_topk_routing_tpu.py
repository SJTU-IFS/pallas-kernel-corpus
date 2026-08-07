"""Small-shape TPU correctness for the MoE router top-k family.

Top-k is a selection kernel, so the bar is exact: the selected expert **ids
must match bitwise** and the selected weights to 0.0.  A cosine threshold would
happily pass a kernel that routed tokens to the wrong experts.

Requires a TPU.  Run with::

    uv run --frozen --with pytest python -m pytest \
        tests/test_topk_routing_tpu.py -q
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
FAMILY = ROOT / "kernels" / "sampling" / "topk_routing"

pytestmark = pytest.mark.skipif(
    not any("TPU" in device.device_kind for device in jax.devices()),
    reason="router top-k kernels are Mosaic TPU kernels",
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
        "baseline": load("tk_baseline", "baseline.py"),
        "biased": load("tk_biased", "sglang_jax_optimized.py"),
        "grouped": load("tk_grouped", "sglang_jax_grouped_optimized.py"),
    }


SHAPES = [(4096, 256, 8), (1024, 128, 4), (2048, 384, 6)]


def assert_exact(actual, expected):
    """Ids bitwise equal, weights exactly equal."""
    got_w, got_i = (np.asarray(x) for x in actual)
    want_w, want_i = (np.asarray(x) for x in expected)
    assert got_i.shape == want_i.shape
    np.testing.assert_array_equal(got_i, want_i)
    np.testing.assert_array_equal(got_w, want_w)


@pytest.mark.parametrize("shape", SHAPES)
def test_plain_topk_matches_reference_exactly(modules, shape):
    baseline = modules["baseline"]
    batch, num_experts, topk = shape
    logits, _ = baseline.create_inputs(batch=batch, num_experts=num_experts)
    expected = jax.jit(baseline.router_topk, static_argnames="topk")(
        logits, topk=topk
    )
    actual = modules["biased"].topk_pallas(logits, topk=topk)
    assert_exact(actual, expected)


@pytest.mark.parametrize("shape", SHAPES)
def test_biased_topk_matches_reference_exactly(modules, shape):
    baseline = modules["baseline"]
    batch, num_experts, topk = shape
    logits, bias = baseline.create_inputs(batch=batch, num_experts=num_experts)
    expected = jax.jit(baseline.router_biased_topk, static_argnames="topk")(
        logits, bias, topk=topk
    )
    actual = modules["biased"].biased_topk_pallas(logits, bias, topk=topk)
    assert_exact(actual, expected)


@pytest.mark.parametrize(
    "batch,num_experts,groups,topk_group,topk",
    [(4096, 256, 8, 4, 8), (1024, 128, 8, 2, 4)],
)
def test_grouped_topk_matches_reference_exactly(
    modules, batch, num_experts, groups, topk_group, topk
):
    baseline = modules["baseline"]
    logits, bias = baseline.create_inputs(batch=batch, num_experts=num_experts)
    expected = jax.jit(
        baseline.router_grouped_topk,
        static_argnames=("num_expert_group", "topk_group", "topk"),
    )(logits, bias, num_expert_group=groups, topk_group=topk_group, topk=topk)
    actual = modules["grouped"].grouped_topk_pallas(
        logits, bias, num_expert_group=groups, topk_group=topk_group, topk=topk
    )
    assert_exact(actual, expected)


def test_biased_selection_returns_pre_bias_weights(modules):
    """The bias must steer routing without contaminating the combine weights.

    Constructed so the bias changes which experts win: without it expert 0
    would be picked, with it expert 1 is.  The returned weight must then be
    expert 1's *unbiased* logit.
    """
    logits = jnp.zeros((128, 128), jnp.float32).at[:, 0].set(1.0).at[:, 1].set(0.5)
    bias = jnp.zeros((128,), jnp.float32).at[:].set(0.0).at[1].set(10.0)

    weights, ids = modules["biased"].biased_topk_pallas(logits, bias, topk=1)
    assert int(np.asarray(ids)[0, 0]) == 1, "bias should have moved the winner"
    assert float(np.asarray(weights)[0, 0]) == pytest.approx(0.5), (
        "returned weight must be the pre-bias logit, not logit + bias"
    )


def test_num_experts_must_be_a_multiple_of_128(modules):
    """Pin the shape constraint found while migrating."""
    baseline = modules["baseline"]
    logits, _ = baseline.create_inputs(batch=512, num_experts=64)
    with pytest.raises(ValueError, match="divisible by 128"):
        modules["biased"].topk_pallas(logits, topk=2)


def test_all_three_return_batch_major_outputs(modules):
    """Guard the layout: the kernels compute transposed and transpose back."""
    baseline = modules["baseline"]
    logits, bias = baseline.create_inputs(batch=1024, num_experts=128)
    for actual in (
        modules["biased"].topk_pallas(logits, topk=4),
        modules["biased"].biased_topk_pallas(logits, bias, topk=4),
        modules["grouped"].grouped_topk_pallas(
            logits, bias, num_expert_group=8, topk_group=2, topk=4
        ),
    ):
        weights, ids = actual
        assert weights.shape == (1024, 4)
        assert ids.shape == (1024, 4)


# ---------------------------------------------------------------------------
# streamindex_topk -- DeepSeek-V4's lightning indexer, vendored into both
# tpu-inference and sglang-jax.  Not a router: it retrieves kv positions.
# ---------------------------------------------------------------------------

STREAMINDEX = ("tpu_inference_dsv4", "sglang_jax_dsa")


@pytest.fixture(scope="module")
def streamindex():
    return {
        "tpu_inference_dsv4": load(
            "topk_ti_dsv4", "tpu_inference_dsv4_optimized.py"),
        "sglang_jax_dsa": load("topk_sg_dsa", "sglang_jax_dsa_optimized.py"),
    }


# T_list, S_list, page_size, q_heads, kv_heads, head_dim, comp_ratio, bq, bkv_p
STREAMINDEX_CASES = [
    pytest.param((3, 3), (4, 6), 16, 4, 1, 64, 2, 32, 2, id="prefill"),
    pytest.param((1,), (10,), 16, 2, 1, 32, 1, 16, 1, id="decode"),
    pytest.param((1, 5), (4, 6), 16, 4, 1, 64, 2, 32, 2, id="mixed"),
    pytest.param((4, 6), (16, 24), 16, 8, 1, 64, 2, 32, 4, id="gqa_8q_1kv"),
]


def _run_streamindex(module, baseline, case, k=512):
    (T_list, S_list, page_size, q_heads, kv_heads, head_dim, comp_ratio,
     bq, bkv_p) = case
    kernel_kwargs, reference_kwargs = baseline.create_streamindex_inputs(
        T_list=T_list, S_list=S_list, page_size=page_size,
        num_q_heads=q_heads, num_kv_heads=kv_heads, head_dim=head_dim,
        k=k, compression_ratio=comp_ratio,
    )
    actual = module.streamindex_topk(
        **kernel_kwargs, k=k, compression_ratio=comp_ratio,
        num_kv_pages_per_block=bkv_p, num_queries_per_block=bq,
    )
    expected = baseline.streamindex_topk_ref(**reference_kwargs)
    return np.asarray(jax.block_until_ready(actual)), expected


@pytest.mark.parametrize("implementation", STREAMINDEX)
@pytest.mark.parametrize(
    "T_list,S_list,page_size,q_heads,kv_heads,head_dim,comp_ratio,bq,bkv_p",
    STREAMINDEX_CASES,
)
def test_streamindex_matches_reference_exactly(
    modules, streamindex, implementation, T_list, S_list, page_size, q_heads,
    kv_heads, head_dim, comp_ratio, bq, bkv_p,
):
    """A selection kernel: the retrieved index set must match, not merely be close."""
    case = (T_list, S_list, page_size, q_heads, kv_heads, head_dim,
            comp_ratio, bq, bkv_p)
    actual, expected = _run_streamindex(
        streamindex[implementation], modules["baseline"], case)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.sort(actual, -1), np.sort(expected, -1))


def test_streamindex_implementations_agree_bitwise(modules, streamindex):
    """The two are the same code, so they must not merely be close.

    tpu-inference's and sglang-jax's copies differ only in docstring
    indentation and one `0 <= x` vs `x >= 0`; see the ledger.  If this ever
    fails they have genuinely diverged and both need separate profiling.
    """
    case = STREAMINDEX_CASES[0].values
    a, _ = _run_streamindex(streamindex["tpu_inference_dsv4"],
                            modules["baseline"], case)
    b, _ = _run_streamindex(streamindex["sglang_jax_dsa"],
                            modules["baseline"], case)
    np.testing.assert_array_equal(a, b)


def test_streamindex_sources_are_semantically_identical():
    """Pin the duplication measurement the ledger records.

    Compared after `ast.unparse`, so formatting differences do not count.
    """
    import ast

    def defs(path):
        text = (FAMILY / path).read_text()
        tree = ast.parse(text)
        out = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                body = ast.parse(ast.get_source_segment(text, node))
                # Drop docstrings: the two repos indent them differently.
                for sub in ast.walk(body):
                    if isinstance(sub, (ast.FunctionDef, ast.ClassDef,
                                        ast.Module)):
                        if (sub.body and isinstance(sub.body[0], ast.Expr)
                                and isinstance(sub.body[0].value, ast.Constant)
                                and isinstance(sub.body[0].value.value, str)):
                            sub.body = sub.body[1:] or [ast.Pass()]
                out[node.name] = ast.dump(ast.fix_missing_locations(body))
        return out

    a = defs("tpu_inference_dsv4_optimized.py")
    b = defs("sglang_jax_dsa_optimized.py")
    assert set(a) == set(b)
    differing = sorted(name for name in a if a[name] != b[name])
    # `_streamindex_topk_kernel` spells one comparison the other way round:
    # `0 <= old_seq_idx` vs `old_seq_idx >= 0`.  Everything else is identical.
    assert differing == ["_streamindex_topk_kernel"], differing


def test_streamindex_requires_k_to_be_a_multiple_of_128(modules, streamindex):
    """An assert in the kernel, not a tuning preference."""
    case = STREAMINDEX_CASES[0].values
    with pytest.raises(AssertionError):
        _run_streamindex(streamindex["tpu_inference_dsv4"],
                         modules["baseline"], case, k=500)


def test_streamindex_seq_lens_are_uncompressed(modules, streamindex):
    """Passing an already-divided kv length silently shortens the search.

    The kernel divides `seq_lens` by `compression_ratio` itself.  Feeding it
    pre-divided lengths does not raise -- it just retrieves from a prefix, so
    fewer positions come back valid.
    """
    case = ((4, 6), (16, 24), 16, 8, 1, 64, 2, 32, 4)
    baseline = modules["baseline"]
    module = streamindex["tpu_inference_dsv4"]
    kernel_kwargs, _ = baseline.create_streamindex_inputs(
        T_list=(4, 6), S_list=(16, 24), page_size=16, num_q_heads=8,
        num_kv_heads=1, head_dim=64, k=512, compression_ratio=2,
    )
    full = np.asarray(module.streamindex_topk(
        **kernel_kwargs, k=512, compression_ratio=2,
        num_kv_pages_per_block=4, num_queries_per_block=32))
    halved = dict(kernel_kwargs, seq_lens=kernel_kwargs["seq_lens"] // 2)
    short = np.asarray(module.streamindex_topk(
        **halved, k=512, compression_ratio=2,
        num_kv_pages_per_block=4, num_queries_per_block=32))
    assert (short >= 0).sum() < (full >= 0).sum()
