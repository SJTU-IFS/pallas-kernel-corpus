"""TPU correctness for sglang-jax's EAGLE speculative-decoding kernels.

These are control flow, not arithmetic. Every output is an integer index, count
or mask, so everything here is checked by **exact equality** — a tree walk is
either right or it visits the wrong node, and a tolerance would only hide that.

The three kernels are covered three different ways, because upstream covers
them three different ways:

`verify_tree_greedy`
    Upstream ships a live exact-value test. The corpus carries its vectors and
    expectations and re-runs them.

`build_eagle_tree_structure`
    Upstream ships no correctness test at all, only a benchmark. Rather than
    invent an oracle for sglang's own tree encoding, this checks the invariants
    that follow from the inputs alone, and then feeds the kernel's output into
    `verify_tree_greedy` — which *does* have an oracle — so the pair is checked
    by composition.

`tree_speculative_sampling_target_only`
    Does not run, upstream or here. Its test upstream is dead code whose body
    begins `return`. That failure is pinned below, so a future fix surfaces as
    a test that starts passing rather than as silence.

Requires a TPU.  Run with::

    uv run --frozen --with pytest python -m pytest tests/test_speculative_tpu.py -q
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
FAMILY = ROOT / "kernels" / "sampling" / "speculative_decoding"

pytestmark = pytest.mark.skipif(
    not any("TPU" in device.device_kind for device in jax.devices()),
    reason="the EAGLE speculative-decoding kernels are Mosaic TPU kernels",
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
        "verify": load("spec_verify", "sglang_jax_verify_tree_greedy_optimized.py"),
        "sampling": load("spec_sampling", "sglang_jax_tree_sampling_optimized.py"),
        "build": load("spec_build", "sglang_jax_build_tree_optimized.py"),
        "baseline": load("spec_baseline", "baseline.py"),
    }


@pytest.fixture(scope="module")
def mesh():
    """The wrapper `shard_map`s over the surrounding mesh, so one must be set."""
    return jax.make_mesh((len(jax.devices()), 1), ("data", "model"))


def run_verify(verify, arguments):
    return verify.verify_tree_greedy(**arguments)


def test_verify_tree_greedy_matches_upstreams_oracle(modules, mesh):
    """Upstream's own tree, upstream's own expected integers, exactly.

    With one deliberate narrowing. `predicts` is an **output-only** buffer here:
    `verify_tree_greedy_pallas_call` passes the caller's array only for its
    shape and dtype, allocating a fresh one via `out_shape`, and the kernel
    writes just the slots on the accepted path. Every other slot is whatever
    that freshly allocated SMEM happened to contain.

    Upstream's expected vector nonetheless spells out zeros for those slots,
    which held on the run that produced it and does not hold in general: with
    other tests in the same process first, those slots come back carrying the
    previous tenant's values. So this asserts the two fully defined outputs
    exactly, and `predicts` exactly at the indices the kernel actually wrote --
    which is the contract -- rather than reproducing an assertion about
    uninitialised memory.
    """
    verify, baseline = modules["verify"], modules["baseline"]
    arguments = baseline.verify_tree_greedy_inputs()
    expected = baseline.VERIFY_TREE_GREEDY_EXPECTED

    with jax.set_mesh(mesh):
        accept_index, accept_token_num, predicts = run_verify(verify, arguments)

    assert accept_index.tolist() == expected["accept_index"]
    assert accept_token_num.tolist() == expected["accept_token_num"]

    written = sorted({index for row in expected["accept_index"]
                      for index in row if index != -1})
    assert written, "the oracle would be vacuous with nothing accepted"
    got = predicts.flatten().tolist()
    for index in written:
        assert got[index] == expected["predicts"][index], (
            f"predicts[{index}]: {got[index]} != {expected['predicts'][index]}")


def test_predicts_is_only_defined_where_the_kernel_wrote(modules, mesh):
    """Pin the narrowing above, so it is a finding rather than a loosened test.

    The claim is specific: the slots named by `accept_index` are written by the
    kernel and reproduce upstream's values, and the complement is not written
    at all. If a future kernel began zeroing its whole output, upstream's full
    vector would match again and this test would fail, prompting the wider
    assertion to come back.
    """
    verify, baseline = modules["verify"], modules["baseline"]
    arguments = baseline.verify_tree_greedy_inputs()
    expected = baseline.VERIFY_TREE_GREEDY_EXPECTED
    written = {index for row in expected["accept_index"]
               for index in row if index != -1}
    unwritten = [i for i in range(len(expected["predicts"])) if i not in written]
    assert unwritten, "this test assumes some slots go unwritten"

    with jax.set_mesh(mesh):
        *_, predicts = run_verify(verify, arguments)
    got = predicts.flatten().tolist()

    # Every written slot is reproducible; the unwritten ones are not asserted.
    assert all(got[i] == expected["predicts"][i] for i in sorted(written))
    assert all(expected["predicts"][i] == 0 for i in unwritten), (
        "upstream's vector expects 0 in the unwritten slots, which is what "
        "makes its full-vector assertion order-dependent")


def test_the_oracle_is_not_satisfied_by_a_degenerate_answer(modules):
    """The expectations must distinguish a real walk from an empty one.

    A tree verifier that accepted nothing would return all -1 and zero counts.
    Upstream's expected values are not that, which is what makes the comparison
    above evidence -- the integer analogue of the vacuity check the float
    comparisons get from tools/assertion_strength.py.
    """
    expected = modules["baseline"].VERIFY_TREE_GREEDY_EXPECTED
    assert expected["accept_token_num"] != [0, 0]
    assert any(index != -1 for row in expected["accept_index"] for index in row)
    assert len(set(expected["predicts"])) > 1


def build_tree(build, bs=2, draft_token_num=8, topk=4):
    """A small two-request tree, built through upstream's own preprocess."""
    steps = 4
    key = jax.random.key(0)
    k_score, k_token = jax.random.split(key)
    # (bs, 1 + (steps - 1) * topk, topk) scores, flattened by the preprocess.
    scores = jax.random.uniform(
        k_score, (bs, 1 + (steps - 1) * topk, topk), dtype=jnp.float32)
    tokens = jax.random.randint(
        k_token, (bs, (1 + (steps - 1) * topk) * topk), 0, 30000, dtype=jnp.int32)
    parents = jnp.tile(
        jnp.arange(topk + 1 + (steps - 1) * topk, dtype=jnp.int32) - 1, (bs, 1))
    verified_id = jnp.array([29974, 13], dtype=jnp.int32)[:bs]
    return scores, tokens, parents, verified_id, steps


def test_build_eagle_tree_structure_traversal_is_a_valid_tree(modules, mesh):
    """No oracle upstream, so check what the inputs alone determine.

    `retrive_next_token` / `retrive_next_sibling` encode a first-child /
    next-sibling traversal. Whatever numbering sglang uses internally, walking
    that traversal from the root must reach each of the `draft_token_num` nodes
    exactly once and terminate -- a tree, not a forest and not a cycle. That is
    a property of the encoding, not a guess at its layout.
    """
    build, baseline = modules["build"], modules["baseline"]
    bs, draft_token_num, topk = 2, 8, 4
    scores, tokens, parents, verified_id, steps = build_tree(build)
    parent_list, selected_index, _ = baseline.build_tree_kernel_efficient_preprocess(
        verified_id, scores, tokens, parents, draft_token_num, bs, steps)
    verified_seq_len = jnp.array([5, 10], dtype=jnp.int32)

    with jax.set_mesh(mesh):
        outputs = build.build_eagle_tree_structure_pallas_call(
            parent_list, selected_index, verified_seq_len,
            jnp.asarray(jnp.sum(verified_seq_len), dtype=jnp.int32),
            draft_token_num=draft_token_num, topk=topk,
            max_context_len=int(verified_seq_len.max()), tree_mask_mode=0)
    _, positions, retrive_index, next_token, next_sibling = outputs

    assert retrive_index.shape == (bs, draft_token_num)
    assert next_token.shape == next_sibling.shape == (bs, draft_token_num)
    assert positions.shape == (bs * draft_token_num,)

    for request in range(bs):
        child = np.asarray(next_token[request])
        sibling = np.asarray(next_sibling[request])
        # Depth-first walk from the root, following first-child then sibling.
        seen, stack = set(), [0]
        while stack:
            node = stack.pop()
            assert 0 <= node < draft_token_num, (request, node)
            assert node not in seen, f"request {request}: node {node} revisited"
            seen.add(node)
            for nxt in (child[node], sibling[node]):
                if nxt != -1:
                    stack.append(int(nxt))
        assert seen == set(range(draft_token_num)), (
            f"request {request}: walk reached {sorted(seen)}, not every node")


def test_build_then_verify_composes(modules, mesh):
    """The tree builder's output is the verifier's input, so run them in series.

    This is the strongest check available for a kernel upstream never tests:
    `verify_tree_greedy` has an exact-value oracle of its own, and it consumes
    precisely the three arrays the builder produces. A builder emitting a
    malformed traversal would make the verifier walk off the tree, and its
    outputs are checked for internal consistency here rather than against fixed
    values -- the tree is random, so there are no fixed values to check.
    """
    build, verify, baseline = modules["build"], modules["verify"], modules["baseline"]
    bs, draft_token_num, topk, steps = 2, 8, 4, 4
    scores, tokens, parents, verified_id, _ = build_tree(build)
    parent_list, selected_index, draft_tokens = (
        baseline.build_tree_kernel_efficient_preprocess(
            verified_id, scores, tokens, parents, draft_token_num, bs, steps))
    verified_seq_len = jnp.array([5, 10], dtype=jnp.int32)

    with jax.set_mesh(mesh):
        _, _, retrive_index, next_token, next_sibling = (
            build.build_eagle_tree_structure_pallas_call(
                parent_list, selected_index, verified_seq_len,
                jnp.asarray(jnp.sum(verified_seq_len), dtype=jnp.int32),
                draft_token_num=draft_token_num, topk=topk,
                max_context_len=int(verified_seq_len.max()), tree_mask_mode=0))

        # Make the target agree with the draft *whatever* index mapping the
        # builder chose: every candidate is the same token, and every row's
        # argmax is that token. Constructing agreement per-position instead
        # would require knowing how `retrive_index` maps nodes to logit rows,
        # which is the sglang-internal convention this test refuses to guess.
        vocab, agreed_token = 64, 7
        logits = jnp.full((bs * draft_token_num, vocab), 1.0, jnp.float32)
        logits = logits.at[:, agreed_token].set(10.0)
        candidates = jnp.full((bs, draft_token_num), agreed_token, jnp.int32)

        accept_index, accept_token_num, _ = verify.verify_tree_greedy(
            speculative_num_steps=draft_token_num - 1,
            num_draft_tokens=draft_token_num,
            draft_tokens=candidates,
            retrive_index=retrive_index,
            retrive_next_token=next_token,
            retrive_next_sibling=next_sibling,
            next_token_logits=logits)

    counts = np.asarray(accept_token_num)
    indices = np.asarray(accept_index)
    assert counts.shape == (bs,)
    for request in range(bs):
        accepted = indices[request][indices[request] != -1]
        # The count and the index list must agree, and the walk must have gone
        # somewhere: the target agrees with every draft token by construction.
        assert len(accepted) == counts[request] + 1, (request, accepted, counts)
        assert counts[request] >= 1, (
            f"request {request}: accepted nothing though the target agrees "
            f"with every drafted token, so the traversal is malformed")
        assert len(set(accepted.tolist())) == len(accepted), "repeated node"


def test_the_two_working_kernels_reach_pallas(modules, mesh):
    """The standing rule, for the two launch points the corpus counts."""
    sys.path.insert(0, str(ROOT / "tools"))
    from profile_kernel import count_pallas_launches

    verify, build, baseline = modules["verify"], modules["build"], modules["baseline"]
    arguments = baseline.verify_tree_greedy_inputs()
    with jax.set_mesh(mesh):
        verify_launches = count_pallas_launches(
            lambda *arrays: verify.verify_tree_greedy(
                draft_tokens=arrays[0], retrive_index=arrays[1],
                retrive_next_token=arrays[2], retrive_next_sibling=arrays[3],
                next_token_logits=arrays[4],
                speculative_num_steps=arguments["speculative_num_steps"],
                num_draft_tokens=arguments["num_draft_tokens"]),
            (arguments["draft_tokens"], arguments["retrive_index"],
             arguments["retrive_next_token"], arguments["retrive_next_sibling"],
             arguments["next_token_logits"]))
    assert verify_launches == 1, f"verify_tree_greedy: {verify_launches}"

    bs, draft_token_num, topk, steps = 2, 8, 4, 4
    scores, tokens, parents, verified_id, _ = build_tree(build)
    parent_list, selected_index, _ = baseline.build_tree_kernel_efficient_preprocess(
        verified_id, scores, tokens, parents, draft_token_num, bs, steps)
    verified_seq_len = jnp.array([5, 10], dtype=jnp.int32)
    # Closed over rather than traced: this wrapper sizes its output buffers
    # from the *values* of `seq_lens_sum` and `verified_seq_len`, so tracing
    # them raises ConcretizationTypeError. That is upstream's design -- it is a
    # host-side wrapper, not a jittable function -- and the launch count is the
    # same either way.
    seq_lens_sum = jnp.asarray(jnp.sum(verified_seq_len), dtype=jnp.int32)
    max_context_len = int(verified_seq_len.max())
    with jax.set_mesh(mesh):
        build_launches = count_pallas_launches(
            lambda: build.build_eagle_tree_structure_pallas_call(
                parent_list, selected_index, verified_seq_len, seq_lens_sum,
                draft_token_num=draft_token_num, topk=topk,
                max_context_len=max_context_len, tree_mask_mode=0),
            ())
    assert build_launches == 1, f"build_eagle_tree_structure: {build_launches}"


def test_tree_speculative_sampling_does_not_run(modules):
    """Pin upstream's own defect, so a fix announces itself.

    sglang-jax disabled this kernel's test with a bare `return` above the
    comment "this kernel still have some problems". Fed upstream's own
    (unreachable) vectors, the kernel raises a pytree-structure error from its
    `scan` — and upstream's *unflattened* file raises the identical error, so
    this is not the corpus's flattening. That is why this launch point is
    recorded UNVALIDATED with zero migrated launch points rather than counted.

    If this test ever fails, the kernel has started working and should be
    validated against the vectors upstream left behind.
    """
    sampling = modules["sampling"]
    candidates = jnp.array([[0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]], jnp.int32)
    retrive_index = jnp.array([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], jnp.int32)
    next_token = jnp.array([[1, 2, -1, 4, 5, -1], [4, 2, 3, -1, 5, -1]], jnp.int32)
    next_sibling = jnp.array([[-1, 3, -1, -1, -1, -1], [-1, -1, -1, -1, 1, -1]],
                             jnp.int32)
    target_probs = jax.nn.softmax(
        jnp.zeros((12, 20), jnp.float32).at[:, 18].set(1000.0), axis=-1)
    coins = jax.random.uniform(jax.random.PRNGKey(42), (2, 6), jnp.float32)

    with pytest.raises(TypeError, match="pytree structure"):
        sampling.tree_speculative_sampling_target_only_pallas_call(
            predicts=jnp.full((12,), -1, jnp.int32),
            accept_index=jnp.full((2, 4), -1, jnp.int32),
            accept_token_num=jnp.zeros((2,), jnp.int32),
            candidates=candidates, retrive_index=retrive_index,
            retrive_next_token=next_token, retrive_next_sibling=next_sibling,
            uniform_samples=coins,
            uniform_samples_for_final_sampling=jax.random.uniform(
                jax.random.PRNGKey(42), (2,), jnp.float32),
            target_probs=target_probs, draft_probs=jnp.zeros_like(target_probs),
            threshold_single=1, threshold_acc=1, deterministic=True)
