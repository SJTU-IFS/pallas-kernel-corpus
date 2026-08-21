"""TPU correctness for the three fused expert-parallel MoE kernels.

Every part of every comparison here is upstream's: each kernel is checked
against the `ref_moe` shipped in its *own* file, on inputs from the
`gen_moe_inputs` in its *own* test, at the tolerance its own test uses
(atol = rtol = 2e-1, which is loose because these are bf16 chains of two
matmuls with a gated activation between them).

That matters more in this family than most.  "Mixture of experts" is not one
function: routing can be plain or grouped top-k, logits can be softmaxed or
sigmoided or used raw, top-k weights renormalised or not, the activation silu /
gelu / clamped SwiGLU, the weights sub-channel or per-channel quantised.  These
three files disagree on several of those, so a single corpus-written reference
would silently be checking one of them and mis-checking the others.

Expert parallelism is validated at ep_size 1.  On a one-device mesh the
dispatch to expert-holding devices is local, so the routing, the fused
activation and the blocking are all exercised, and the cross-device collective
is not.

Requires a TPU.  Run with::

    uv run --frozen --with pytest python -m pytest tests/test_fused_moe_tpu.py -q
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh


ROOT = Path(__file__).parents[1]
FAMILY = ROOT / "kernels" / "moe" / "fused_moe"

pytestmark = pytest.mark.skipif(
    not any("TPU" in device.device_kind for device in jax.devices()),
    reason="the fused MoE kernels are Mosaic TPU kernels",
)

#: Upstream's `test_basic` parameterisation, shared by all three test suites.
TOP_K, NUM_EXPERTS = 8, 128
HIDDEN, INTERMEDIATE, NUM_TOKENS = 1024, 1024, 8 * 32
SEED = 1234

#: Upstream's own tolerance for the bf16 case -- but applied *relative to the
#: reference's magnitude* rather than absolutely.  Upstream uses atol=2e-1 flat,
#: which is fine at the scales its v1 tests reach and vacuous at the scale its
#: v2 test reaches: that output peaks at 0.096, so an absolute 0.2 is satisfied
#: by a kernel returning nothing but zeros.  `tools/assertion_strength.py`
#: caught exactly that.  Measured, all six comparisons here sit at ~0.6%
#: relative error (max ratio 0.0057), so scaling 2e-1 to the reference's peak
#: stays far looser than the kernels need while no longer admitting zeros.
ATOL = RTOL = 2e-1

#: Upstream's `test_basic` block sizes, identical across the two v1 kernels.
BLOCKS = dict(bt=32, bf=1024, bd1=1024, bd2=1024,
              btc=32, bfc=256, bd1c=256, bd2c=256)

#: v2 is its own parameterisation, not a variation on v1's, and pretending
#: otherwise is what broke this test first: its `FusedMoEBlockConfig` carries
#: four fields rather than nine, its `gen_moe_inputs` returns eight arrays
#: rather than eleven, its `ref_moe` takes *pre-computed* routing rather than
#: raw gating, and its own `test_basic` runs a smaller shape.
V2_TOP_K, V2_NUM_EXPERTS = 4, 16
V2_HIDDEN, V2_INTERMEDIATE, V2_NUM_TOKENS = 512, 256, 128
V2_BLOCKS = dict(bt=32, bf=128, btc=32, bse=128)


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, FAMILY / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    return {
        "sglang_jax": load("moe_sg_v1", "sglang_jax_optimized.py"),
        "sglang_jax_v2": load("moe_sg_v2", "sglang_jax_v2_optimized.py"),
        "tpu_inference": load("moe_ti_v1", "tpu_inference_optimized.py"),
        "baseline": load("moe_baseline", "baseline.py"),
    }


@pytest.fixture(scope="module")
def meshes():
    """One device, two axis namings -- the repositories differ on the name.

    tpu-inference calls its axes ("data", "model"); sglang-jax calls them
    ("data", "tensor") and takes `tp_axis_name` explicitly.  At one device both
    are 1x1 and expert parallelism is degenerate, which is the limit recorded
    in the module docstring.
    """
    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    return {"tpu_inference": Mesh(devices, axis_names=("data", "model")),
            "sglang": Mesh(devices, axis_names=("data", "tensor"))}


def route(gating_output, top_k, renormalize):
    """`topk_weights`/`topk_ids` as upstream computes them, for the plain case.

    The sglang kernels take routing pre-computed; their tests get it from
    `sgl_jax.srt.layers.gate.TopK`, which needs flax and is not carried here.
    For the ungrouped, unbiased path that class reduces to::

        router_logits = router_logits.astype(jnp.float32)
        topk_weights, topk_ids = jax.lax.top_k(router_logits, self.topk)
        if self.renormalize:
            topk_weights = topk_weights / jnp.sum(topk_weights, -1, keepdims=True)

    which is the same routing `ref_moe` performs internally from raw gating --
    `lax.top_k` on the float32 logits, then `take_along_axis`, then the same
    renormalisation.  `test_the_routing_agrees_with_the_reference` pins that
    equivalence rather than leaving it as a claim in a comment.
    """
    logits = gating_output.astype(jnp.float32)
    topk_weights, topk_ids = jax.lax.top_k(logits, top_k)
    if renormalize:
        topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)
    return topk_weights.astype(jnp.float32), topk_ids


def sglang_inputs(baseline):
    return baseline.gen_moe_inputs_sglang_jax(
        jnp.bfloat16, TOP_K, NUM_EXPERTS, HIDDEN, INTERMEDIATE, NUM_TOKENS,
        seed=SEED)


def sglang_v2_inputs(baseline):
    """v2's own generator at v2's own default shape."""
    return baseline.gen_moe_inputs_sglang_jax_v2(
        jnp.bfloat16, V2_TOP_K, V2_NUM_EXPERTS, V2_HIDDEN, V2_INTERMEDIATE,
        V2_NUM_TOKENS, seed=SEED)


def tolerances(expected):
    """`(rtol, atol)` where peak-scaling may only ever *tighten* upstream's bar.

    Upstream uses a flat `atol=2e-1`. That is vacuous where the output peaks
    below it — the v2 case peaks at 0.096, so a kernel returning nothing but
    zeros passed — which is why the absolute term is scaled by the reference's
    magnitude here.

    But scaling cuts both ways: two of these six comparisons have references
    peaking at 62.25 and 11.06, where `ATOL * peak` would give 12.45 and 2.21 —
    62x and 11x *looser* than upstream, and in both cases larger than the mean
    output magnitude. Clamping to the minimum keeps the fix for small outputs
    without ever relaxing a bar upstream already passes at.
    """
    peak = float(np.max(np.abs(np.asarray(expected, np.float32))))
    return RTOL, ATOL * min(1.0, max(peak, 1e-6))


def close(actual, expected, what):
    actual = np.asarray(actual, np.float32)
    expected = np.asarray(expected, np.float32)
    assert actual.shape == expected.shape, what
    rtol, atol = tolerances(expected)
    # An assertion a zeroed kernel would satisfy is not evidence, and at these
    # magnitudes upstream's flat atol is one. Check the bar has teeth here
    # rather than trusting the standing sweep to notice later.
    assert not np.allclose(np.zeros_like(expected), expected,
                           rtol=rtol, atol=atol), (
        f"{what}: a kernel returning all zeros would pass this comparison")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol,
                               err_msg=what)


@pytest.mark.parametrize("renormalize", [True, False])
def test_tpu_inference_v1_matches_its_own_reference(modules, meshes, renormalize):
    """Fused `w1` of shape (E, 2, H, I), raw gating, routing done inside."""
    kernel, baseline = modules["tpu_inference"], modules["baseline"]
    a, w1, w2, b1, b2, gating = baseline.gen_moe_inputs_tpu_inference(
        jnp.bfloat16, TOP_K, NUM_EXPERTS, HIDDEN, INTERMEDIATE, NUM_TOKENS,
        seed=SEED)
    assert w1.shape == (NUM_EXPERTS, 2, HIDDEN, INTERMEDIATE), (
        "this kernel wants gate and up fused on axis 1")

    actual = kernel.fused_ep_moe(
        mesh=meshes["tpu_inference"], tokens=a, w1=w1, w2=w2,
        gating_output=gating, top_k=TOP_K,
        renormalize_topk_logits=renormalize, act_fn="silu",
        scoring_fn="softmax", b1=b1, b2=b2, **BLOCKS)
    expected = baseline.REFERENCES["fused_ep_moe_fused_w1"](
        a, w1, w2, gating, TOP_K, b1=b1, b2=b2,
        renormalize_topk_logits=renormalize, act_fn="silu",
        scoring_fn="softmax")
    jax.block_until_ready((actual, expected))
    close(actual, expected, f"tpu-inference v1 renormalize={renormalize}")


@pytest.mark.parametrize("renormalize", [True, False])
def test_sglang_v1_matches_its_own_reference(modules, meshes, renormalize):
    """Three separate weights, routing supplied by the caller."""
    kernel, baseline = modules["sglang_jax"], modules["baseline"]
    a, w1, w2, w3, b1, b2, b3, gating, *_ = sglang_inputs(baseline)
    assert w1.shape == (NUM_EXPERTS, HIDDEN, INTERMEDIATE), (
        "this kernel wants gate and up as separate arrays")
    topk_weights, topk_ids = route(gating, TOP_K, renormalize)

    config = kernel.FusedMoEBlockConfig(bse=512, **BLOCKS)
    actual = kernel.fused_ep_moe(
        mesh=meshes["sglang"], tokens=a, w1=w1, w2=w2, w3=w3,
        topk_weights=topk_weights, topk_ids=topk_ids, top_k=TOP_K,
        act_fn="silu", b1=b1, b2=b2, b3=b3,
        block_config=config, tp_axis_name="tensor")
    expected = baseline.REFERENCES["fused_ep_moe_split_weights"](
        a, w1, w2, w3, gating, TOP_K, b1=b1, b2=b2, b3=b3,
        renormalize_topk_logits=renormalize, act_fn="silu")
    jax.block_until_ready((actual, expected))
    close(actual, expected, f"sglang v1 renormalize={renormalize}")


@pytest.mark.parametrize("renormalize", [True, False])
def test_sglang_v2_matches_its_own_reference(modules, meshes, renormalize):
    """Same operand *layout* as v1, but its own everything else.

    Both kernel and reference take routing pre-computed here, so unlike the two
    v1 comparisons this one feeds the identical `topk_weights`/`topk_ids` to
    both sides -- the routing is an input to the contract rather than part of
    what is being checked.
    """
    kernel, baseline = modules["sglang_jax_v2"], modules["baseline"]
    a, w1, w2, w3, gating, *_ = sglang_v2_inputs(baseline)
    topk_weights, topk_ids = route(gating, V2_TOP_K, renormalize)

    config = kernel.FusedMoEBlockConfig(**V2_BLOCKS)
    actual = kernel.fused_ep_moe_v2(
        meshes["sglang"], a, w1, w2, w3, topk_weights, topk_ids, V2_TOP_K,
        act_fn="silu", block_config=config,
        dp_axis_name="data", tp_axis_name="tensor")
    expected = baseline.REFERENCES["fused_ep_moe_split_weights_v2"](
        a, w1, w2, w3, topk_weights, topk_ids, V2_TOP_K, act_fn="silu")
    jax.block_until_ready((actual, expected))
    close(actual, expected, f"sglang v2 renormalize={renormalize}")


def test_the_routing_agrees_with_the_reference(modules):
    """`route` must reproduce what `ref_moe` does internally, not merely resemble it.

    The sglang kernels are fed routing this helper derives while their
    reference derives its own from raw gating. If the two disagreed, every
    comparison above would be checking two different routings and the failure
    would look like a kernel bug. Upstream's generator boosts each token's
    top-k by a strictly decreasing amount, so the ordering is unambiguous and
    this is an exact check rather than a tie-dependent one.
    """
    baseline = modules["baseline"]
    *_, gating, _, _, _ = sglang_inputs(baseline)
    logits = gating.astype(jnp.float32)

    for renormalize in (True, False):
        weights, ids = route(gating, TOP_K, renormalize)
        # What ref_moe computes internally, in its own terms.
        _, ref_ids = jax.lax.top_k(logits, TOP_K)
        ref_weights = jnp.take_along_axis(logits, ref_ids, axis=-1)
        if renormalize:
            ref_weights = ref_weights / jnp.sum(ref_weights, axis=-1,
                                                keepdims=True)
        np.testing.assert_array_equal(np.asarray(ids), np.asarray(ref_ids))
        np.testing.assert_array_equal(np.asarray(weights, np.float32),
                                      np.asarray(ref_weights, np.float32))


def test_all_three_reach_pallas(modules, meshes):
    """Three implementations, three launch points, each checked separately."""
    sys.path.insert(0, str(ROOT / "tools"))
    from profile_kernel import count_pallas_launches

    baseline = modules["baseline"]
    a, w1, w2, b1, b2, gating = baseline.gen_moe_inputs_tpu_inference(
        jnp.bfloat16, TOP_K, NUM_EXPERTS, HIDDEN, INTERMEDIATE, NUM_TOKENS,
        seed=SEED)
    ti = modules["tpu_inference"]
    launches = {"tpu_inference": count_pallas_launches(
        lambda t, x, y, g: ti.fused_ep_moe(
            mesh=meshes["tpu_inference"], tokens=t, w1=x, w2=y,
            gating_output=g, top_k=TOP_K, renormalize_topk_logits=True,
            act_fn="silu", scoring_fn="softmax", **BLOCKS),
        (a, w1, w2, gating))}

    sa, sw1, sw2, sw3, *_, sgating, _, _, _ = sglang_inputs(baseline)
    v1_weights, v1_ids = route(sgating, TOP_K, True)
    v1 = modules["sglang_jax"]
    launches["sglang_jax"] = count_pallas_launches(
        lambda t, x, y, z: v1.fused_ep_moe(
            mesh=meshes["sglang"], tokens=t, w1=x, w2=y, w3=z,
            topk_weights=v1_weights, topk_ids=v1_ids, top_k=TOP_K,
            act_fn="silu",
            block_config=v1.FusedMoEBlockConfig(bse=512, **BLOCKS),
            tp_axis_name="tensor"),
        (sa, sw1, sw2, sw3))

    v2 = modules["sglang_jax_v2"]
    va, vw1, vw2, vw3, vgating, *_ = sglang_v2_inputs(baseline)
    v2_weights, v2_ids = route(vgating, V2_TOP_K, True)
    launches["sglang_jax_v2"] = count_pallas_launches(
        lambda t, x, y, z: v2.fused_ep_moe_v2(
            meshes["sglang"], t, x, y, z, v2_weights, v2_ids, V2_TOP_K,
            act_fn="silu",
            block_config=v2.FusedMoEBlockConfig(**V2_BLOCKS),
            dp_axis_name="data", tp_axis_name="tensor"),
        (va, vw1, vw2, vw3))
    assert launches == dict.fromkeys(launches, 1), launches


def test_the_references_are_not_themselves_pallas(modules, meshes):
    """A reference that lowered to Mosaic would not be an independent check.

    Two of the three can be checked the strong way, by lowering them and
    counting `tpu_custom_call`s. The third cannot: sglang v2's `ref_moe` forces
    a traced scalar to a concrete value, so it does not survive `jax.jit` at
    all and only runs eagerly -- which is upstream's property, not something
    this corpus introduced, and is why the comparison above calls it directly.
    For that one the check is structural instead, over the whole reference
    module, which is stronger in coverage if weaker in kind.
    """
    sys.path.insert(0, str(ROOT / "tools"))
    from profile_kernel import count_pallas_launches

    baseline = modules["baseline"]
    a, w1, w2, b1, b2, gating = baseline.gen_moe_inputs_tpu_inference(
        jnp.bfloat16, TOP_K, NUM_EXPERTS, HIDDEN, INTERMEDIATE, NUM_TOKENS,
        seed=SEED)
    assert count_pallas_launches(
        lambda t, x, y, g: baseline.REFERENCES["fused_ep_moe_fused_w1"](
            t, x, y, g, TOP_K, act_fn="silu", scoring_fn="softmax"),
        (a, w1, w2, gating)) == 0

    sa, sw1, sw2, sw3, *_, sgating, _, _, _ = sglang_inputs(baseline)
    assert count_pallas_launches(
        lambda t, a1, a2, a3: baseline.REFERENCES["fused_ep_moe_split_weights"](
            t, a1, a2, a3, sgating, TOP_K, act_fn="silu"),
        (sa, sw1, sw2, sw3)) == 0

    # The whole reference module, including the eager-only v2 one.
    tree = ast.parse((FAMILY / "baseline.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any("pallas" in alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "pallas" not in node.module
        if isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            assert not (isinstance(root, ast.Name) and root.id in ("pl", "pltpu"))


def test_the_v2_reference_is_eager_only(modules):
    """Pin the reason the check above has to differ for one of the three.

    If a future JAX or a future upstream makes this reference traceable, the
    weaker structural check is no longer necessary and this test says so by
    failing -- rather than the exception being carried forever as a comment.
    """
    baseline = modules["baseline"]
    a, w1, w2, w3, gating, *_ = sglang_v2_inputs(baseline)
    topk_weights, topk_ids = route(gating, V2_TOP_K, True)
    reference = baseline.REFERENCES["fused_ep_moe_split_weights_v2"]

    # Eagerly it is fine, which is how the comparison test calls it.
    eager = reference(a, w1, w2, w3, topk_weights, topk_ids, V2_TOP_K,
                      act_fn="silu")
    assert eager.shape == (V2_NUM_TOKENS, V2_HIDDEN)

    with pytest.raises(jax.errors.ConcretizationTypeError):
        jax.jit(lambda t, x, y, z: reference(
            t, x, y, z, topk_weights, topk_ids, V2_TOP_K, act_fn="silu")
        )(a, w1, w2, w3)


def test_the_two_v1_kernels_are_not_the_same_kernel():
    """Two files, both named v1, both exporting `fused_ep_moe`.

    The corpus has been wrong about this shape of thing before -- the gdn v3
    pair *is* a vendored copy, and the splash pair is not -- so measure rather
    than assume. Of the seven top-level names these two share, only the three
    smallest are identical; `_fused_ep_moe_kernel`, `fused_ep_moe` and
    `ref_moe` all differ, which is why they are two migrations with two
    contracts rather than one with a duplicate note.
    """
    def defs(name):
        source = (FAMILY / name).read_text()
        return {node.name: ast.dump(node) for node in ast.parse(source).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))}

    a = defs("sglang_jax_optimized.py")
    b = defs("tpu_inference_optimized.py")
    shared = set(a) & set(b)
    identical = {n for n in shared if a[n] == b[n]}
    assert identical == {"align_to", "broadcast_minor", "swigluoai"}, identical
    for differing in ("_fused_ep_moe_kernel", "fused_ep_moe", "ref_moe"):
        assert differing in shared - identical, differing
