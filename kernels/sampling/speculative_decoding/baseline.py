"""References for the EAGLE speculative-decoding kernels in this directory.

There is no pure-JAX reimplementation here, and there should not be.  A tree
walk has no tolerance to hide behind: `retrive_next_token` /
`retrive_next_sibling` are sglang-jax's own encoding of a draft tree, and a
corpus-written "reference" for them would be a guess at a layout convention,
failing loudly and wrongly the moment the guess was off.

What upstream *does* supply is an oracle -- `test_verify_tree_greedy` fixes a
two-request tree by hand and asserts the exact integer outputs -- so that is
what is carried: the inputs and the expected results, as data.  The upstream
test is quoted verbatim below, extracted rather than transcribed, so the
transcription above it cannot drift from the code it claims to reproduce.

`build_tree_kernel_efficient_preprocess` is upstream's too, from
`python/sgl_jax/srt/speculative/eagle_util.py`; it turns draft-model scores and
tokens into the `parent_list` / `selected_index` pair the tree builder takes.

Coverage is deliberately uneven, because upstream's is:

``verify_tree_greedy``                     exact-value oracle, carried here
``build_eagle_tree_structure``             no upstream test; checked by
                                           traversal invariants and by
                                           composition into the kernel above
``tree_speculative_sampling_target_only``  no working test upstream, and the
                                           kernel does not run -- see the
                                           corpus notes on that file

Source:
  repository: https://github.com/sgl-project/sglang-jax
  commit: a7353325e8c00d287294c2cd679a77173f1a4594
  paths:
    python/sgl_jax/test/speculative/test_eagle_utils.py
    python/sgl_jax/srt/speculative/eagle_util.py

Upstream, `python/sgl_jax/test/speculative/test_eagle_utils.py`, `test_verify_tree_greedy`::

    def test_verify_tree_greedy(self):
            candidates = jnp.array(
                [
                    [0, 1, 2, 3, 4, 5],
                    [7, 8, 9, 10, 11, 12],
                ],
                dtype=jnp.int32,
            )
            retrive_index = jnp.array(
                [
                    [0, 1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10, 11],
                ],
                dtype=jnp.int32,
            )
            retrive_next_token = jnp.array(
                [
                    [1, 2, -1, 4, 5, -1],
                    [4, 2, 3, -1, 5, -1],
                ],
                dtype=jnp.int32,
            )
            retrive_next_sibling = jnp.array(
                [
                    [-1, 3, -1, -1, -1, -1],
                    [-1, -1, -1, -1, 1, -1],
                ],
                dtype=jnp.int32,
            )

            target_logits = jnp.full((2, 6, 20), 1, dtype=jnp.float32)
            target_logits = target_logits.at[0, 0, 3].set(10)
            target_logits = target_logits.at[0, 3, 4].set(10)
            target_logits = target_logits.at[0, 4, 5].set(10)
            target_logits = target_logits.at[1, 0, 11].set(10)
            target_logits = target_logits.at[1, 4, 12].set(10)
            for i in range(target_logits.shape[0]):
                for j in range(target_logits.shape[1]):
                    if jnp.max(target_logits[i][j]) < 10:
                        target_logits = target_logits.at[i, j, 18].set(10)

            target_logits = target_logits.reshape(-1, target_logits.shape[-1])
            predict_shape = (12,)

            bs = candidates.shape[0]
            num_spec_step = 4

            predicts = jnp.empty(predict_shape, dtype=jnp.int32)  # mutable
            accept_index = jnp.full((bs, num_spec_step), -1, dtype=jnp.int32)  # mutable
            accept_token_num = jnp.full((bs,), 0, dtype=jnp.int32)  # mutable

            from sgl_jax.srt.utils.mesh_utils import create_device_mesh

            mesh = create_device_mesh(ici_parallelism=[-1, 1], dcn_parallelism=[1, 1])
            with jax.set_mesh(mesh):
                accept_index, accept_token_num, predicts = verify_tree_greedy(
                    speculative_num_steps=4,
                    num_draft_tokens=6,
                    draft_tokens=candidates,
                    retrive_index=retrive_index,
                    retrive_next_token=retrive_next_token,
                    retrive_next_sibling=retrive_next_sibling,
                    next_token_logits=target_logits,
                )

            # Check the expected output.
            self.assertEqual(predicts.flatten().tolist(), [3, 0, 0, 4, 5, 18, 11, 0, 0, 0, 12, 18, 0])
            self.assertEqual(accept_index.tolist(), [[0, 3, 4, 5, -1], [6, 10, 11, -1, -1]])
            self.assertEqual(accept_token_num.tolist(), [3, 2])
"""

from __future__ import annotations

SOURCE = {
    "kind": "reference",
    "repository": "https://github.com/sgl-project/sglang-jax",
    "commit": "a7353325e8c00d287294c2cd679a77173f1a4594",
    "paths": ("python/sgl_jax/test/speculative/test_eagle_utils.py",
              "python/sgl_jax/srt/speculative/eagle_util.py"),
    "backend": "jax",
    "target": "portable",
    "contracts": ("verify_tree_greedy", "build_eagle_tree_structure"),
}

import functools

import jax
import jax.numpy as jnp


#: Upstream's `test_verify_tree_greedy` tree, as data.  Two requests, six draft
#: tokens each, encoded first-child / next-sibling: request 0 is the chain
#: 0->1->2 with a sibling branch 3->4->5; request 1 is 0->4->5 with 1->2->3.
VERIFY_TREE_GREEDY_INPUTS = {
    "candidates": [[0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]],
    "retrive_index": [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]],
    "retrive_next_token": [[1, 2, -1, 4, 5, -1], [4, 2, 3, -1, 5, -1]],
    "retrive_next_sibling": [[-1, 3, -1, -1, -1, -1], [-1, -1, -1, -1, 1, -1]],
    "speculative_num_steps": 4,
    "num_draft_tokens": 6,
    #: (request, position, token) triples set to logit 10; every other position
    #: gets its 10 at token 18, so each row has exactly one argmax.
    "peaks": [(0, 0, 3), (0, 3, 4), (0, 4, 5), (1, 0, 11), (1, 4, 12)],
    "vocab": 20,
    "fill_token": 18,
}

#: What upstream asserts, exactly.
VERIFY_TREE_GREEDY_EXPECTED = {
    "predicts": [3, 0, 0, 4, 5, 18, 11, 0, 0, 0, 12, 18, 0],
    "accept_index": [[0, 3, 4, 5, -1], [6, 10, 11, -1, -1]],
    "accept_token_num": [3, 2],
}


def verify_tree_greedy_target_logits():
    """Upstream's target logits for the tree above, built its way."""
    spec = VERIFY_TREE_GREEDY_INPUTS
    bs = len(spec["candidates"])
    n = spec["num_draft_tokens"]
    logits = jnp.full((bs, n, spec["vocab"]), 1, dtype=jnp.float32)
    for i, j, k in spec["peaks"]:
        logits = logits.at[i, j, k].set(10)
    for i in range(bs):
        for j in range(n):
            if jnp.max(logits[i, j]) < 10:
                logits = logits.at[i, j, spec["fill_token"]].set(10)
    return logits.reshape(-1, spec["vocab"])


def verify_tree_greedy_inputs():
    """The four int32 arrays plus the flattened target logits."""
    spec = VERIFY_TREE_GREEDY_INPUTS
    as_array = lambda key: jnp.array(spec[key], dtype=jnp.int32)
    return dict(
        draft_tokens=as_array("candidates"),
        retrive_index=as_array("retrive_index"),
        retrive_next_token=as_array("retrive_next_token"),
        retrive_next_sibling=as_array("retrive_next_sibling"),
        next_token_logits=verify_tree_greedy_target_logits(),
        speculative_num_steps=spec["speculative_num_steps"],
        num_draft_tokens=spec["num_draft_tokens"],
    )


@functools.partial(
    jax.jit, static_argnames=["num_verify_tokens", "batch_size", "speculative_num_steps"]
)
def build_tree_kernel_efficient_preprocess(
    verified_id: jax.Array,
    scores: jax.Array,
    tokens: jax.Array,
    parents: jax.Array,
    num_verify_tokens: int,
    batch_size: int,
    speculative_num_steps: int,
):
    """Upstream's own, from eagle_util.py.

    Body unmodified; upstream's inline comments are dropped and this docstring
    added. The `jax.jit` decorator is upstream's and is reproduced -- an earlier
    version of this file omitted it while still calling itself "unmodified",
    which is the same defect the corpus records and regression-tests for in the
    MLA v2 flatten (see README, "layout_transpose").

    Turns the draft model's per-step scores and tokens into the
    `parent_list` / `selected_index` pair `build_eagle_tree_structure` takes.
    """
    score_tensor = scores
    score_tensor = score_tensor.reshape(score_tensor.shape[0], -1)

    ss_token_list = tokens

    _, top_scores_index = jax.lax.top_k(score_tensor, num_verify_tokens - 1)
    top_scores_index = jnp.sort(top_scores_index, axis=-1)

    draft_tokens = jnp.take_along_axis(ss_token_list, top_scores_index, axis=1)
    draft_tokens = jnp.concatenate(
        [jnp.expand_dims(verified_id, axis=1), draft_tokens], axis=1
    ).flatten()

    if speculative_num_steps > 1:
        parent_list = parents
    else:
        parent_list = jnp.full((batch_size, 1), -1, dtype=jnp.int32)

    return parent_list, top_scores_index, draft_tokens
