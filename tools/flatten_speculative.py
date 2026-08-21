"""Flatten sglang-jax's three EAGLE speculative-decoding kernels.

These are the corpus's first kernels that are *control flow* rather than
arithmetic.  They walk a draft token tree: build its structure, verify it
greedily against target logits, or sample from it stochastically.  Their outputs
are integer indices and masks, so correctness here is exact equality rather than
a tolerance -- nothing rounds.

Flattening is nearly free.  Two of the three import a single repo-local helper
(`cdiv`), the third imports nothing repo-local, and that helper is three lines.

The references are the interesting part, and no two of the three are alike:

``verify_tree_greedy``                       a live exact-value test upstream
``build_eagle_tree_structure``               **no correctness test at all**,
                                             only a performance benchmark
``tree_speculative_sampling_target_only``    a test that exists but is **dead
                                             code** -- its body begins
                                             `return`, above the comment
                                             "this kernel still have some
                                             problems"

The first is carried into `baseline.py` as upstream's own vectors and expected
outputs.  The second is validated without an oracle -- its three routing
outputs are exactly the inputs `verify_tree_greedy` consumes, so the pair is
checked by composition, plus traversal invariants that follow from the inputs
alone and need no guess at sglang's layout conventions.

The third does not run.  Feeding it upstream's own (disabled) vectors raises
`TypeError: scan body function carry input and carry output must have the same
pytree structure`, and **upstream's unflattened file raises the identical
error**, so this is upstream's defect rather than the flattening's -- exactly
what their comment says.  It is carried, and recorded UNVALIDATED with zero
migrated launch points, in the same way as the corpus's other kernels that
cannot be checked.  See tests/test_speculative_tpu.py, which pins the failure
so that a future upstream fix shows up as a test that starts passing.

    python tools/flatten_speculative.py <sglang-jax-checkout> <corpus-dir>
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re


REPOSITORY = "https://github.com/sgl-project/sglang-jax"
COMMIT = "a7353325e8c00d287294c2cd679a77173f1a4594"
KERNELS = "python/sgl_jax/srt/kernels/speculative"
TESTS = "python/sgl_jax/test/speculative/test_eagle_utils.py"

#: `cdiv` is the whole of the repo-local surface these kernels touch.
CDIV_IMPORT = re.compile(
    r"^from sgl_jax\.srt\.utils\.common_utils import cdiv\s*$", re.M)

CDIV_INLINE = '''# ---- inlined from python/sgl_jax/srt/utils/common_utils.py ----


def cdiv(a: int, b: int) -> int:
    """Ceiling division; the only repo-local name these kernels import."""
    assert b != 0
    return (a + b - 1) // b


'''

SOURCES = {
    "verify_tree_greedy": {
        "file": "verify_tree_greedy_kernel.py",
        "entry": "verify_tree_greedy_pallas_call",
        "wrapper": "verify_tree_greedy",
        "contract": "verify_tree_greedy",
        "output": "sglang_jax_verify_tree_greedy_optimized.py",
        "summary": (
            "Greedy tree verification. Walks each request's draft tree from the\n"
            "root, accepting a child whenever the target model's argmax token\n"
            "matches the drafted one, and stopping at the first mismatch --\n"
            "then follows `retrive_next_sibling` to try the next branch. Returns\n"
            "the accepted path, its length, and the predicted tokens."
        ),
    },
    "tree_speculative_sampling": {
        "file": "tree_speculative_sampling_target_only_kernel.py",
        "entry": "tree_speculative_sampling_target_only_pallas_call",
        "wrapper": None,
        "contract": "tree_speculative_sampling_target_only",
        "output": "sglang_jax_tree_sampling_optimized.py",
        "summary": (
            "The stochastic counterpart to greedy verification: instead of\n"
            "accepting on an argmax match, it accepts a draft token when a\n"
            "uniform sample falls under the target/draft probability ratio,\n"
            "with per-token and accumulated thresholds. `target_only` means the\n"
            "draft probabilities are ignored and only the target distribution\n"
            "decides -- which is why its signature takes no draft probs."
        ),
    },
    "build_eagle_tree": {
        "file": "build_eagle_tree_structure_kernel.py",
        "entry": "build_eagle_tree_structure_pallas_call",
        "wrapper": "build_eagle_tree_structure",
        "contract": "build_eagle_tree_structure",
        "output": "sglang_jax_build_tree_optimized.py",
        "summary": (
            "Turns the draft model's `parent_list` and `selected_index` into\n"
            "the structures the verify kernels consume: a causal `tree_mask`,\n"
            "per-token `positions`, and the `retrive_index` /\n"
            "`retrive_next_token` / `retrive_next_sibling` triple that encodes\n"
            "the tree as a first-child / next-sibling traversal."
        ),
    },
}

HEADER = '''"""Standalone sglang-jax EAGLE {name} kernel.

Source:
  repository: {repository}
  commit: {commit}
  path: {path}
{inline_note}  transformation: {transformation}

Entry point: ``{entry}``{wrapper_note} (also exported as ``kernel``).

Contract ``{contract}``.
{summary}

This is index arithmetic, not float arithmetic: every output is an integer
index, count or mask, so the corpus checks it by **exact equality**.  A
tolerance would be meaningless here -- a tree walk is either right or it visits
the wrong node.

{reference_note}

Native shape: none declared; upstream's tests use bs=2 with 6 draft tokens over
a 4-step tree, and its benchmark uses larger trees without checking values.
"""

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{path}",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "{contract}",
    "family": "speculative_decoding",
    "launch_points": 1,
    "native_shape": None,
}}

'''

ORACLE_NOTE = """The reference is upstream's own: `{tests}` fixes a small tree by
hand and asserts the exact integer outputs. Those vectors and expectations are
carried into this directory's `baseline.py` rather than restated here, so the
oracle is upstream's rather than a corpus reading of what the tree should do.

One caveat found by running it: **`predicts` is an output-only buffer.** The
`predicts` argument supplies only shape and dtype -- `out_shape` allocates a
fresh array -- and the kernel writes just the slots on the accepted path, so
every other slot holds whatever that SMEM allocation happened to contain.
Upstream's expected vector nonetheless spells out zeros there, which held for
the run that produced it but not in general: with other work in the same
process first, those slots come back carrying the previous tenant's values.
The corpus therefore asserts `accept_index` and `accept_token_num` in full and
`predicts` at the indices the kernel actually writes."""

NO_ORACLE_NOTE = """**Upstream ships no correctness test for this kernel** -- only a
performance benchmark -- so there is no oracle to carry, and the corpus does not
invent one: re-deriving sglang's own `retrive_next_token` / `retrive_next_sibling`
layout would be exactly the kind of guess that produces false failures. It is
instead validated two ways that need no such guess. Its three routing outputs
are precisely the inputs `verify_tree_greedy` takes, and that kernel *does* have
an upstream oracle, so the pair is checked by composition. And the traversal
invariants -- every node reached exactly once, positions equal to depth in
`parent_list`, the mask causal with respect to ancestry -- follow from the
inputs alone. What is *not* checked is the packed `tree_mask` byte layout; the
ledger records that gap rather than implying parity with the other two."""


BASELINE_HEADER = '''"""References for the EAGLE speculative-decoding kernels in this directory.

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
  repository: {repository}
  commit: {commit}
  paths:
    {tests}
    python/sgl_jax/srt/speculative/eagle_util.py

Upstream, `{tests}`, `test_verify_tree_greedy`::

{quoted}
"""

from __future__ import annotations

SOURCE = {{
    "kind": "reference",
    "repository": "{repository}",
    "commit": "{commit}",
    "paths": ("{tests}",
              "python/sgl_jax/srt/speculative/eagle_util.py"),
    "backend": "jax",
    "target": "portable",
    "contracts": ("verify_tree_greedy", "build_eagle_tree_structure"),
}}

'''

BASELINE_BODY = '''import functools

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
'''


def _quote_test(checkout: Path) -> str:
    """Upstream's `test_verify_tree_greedy`, verbatim and indented."""
    import textwrap
    text = (checkout / TESTS).read_text()
    for node in ast.walk(ast.parse(text)):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "test_verify_tree_greedy"):
            segment = ast.get_source_segment(text, node)
            return textwrap.indent(textwrap.dedent(segment), "    ")
    raise ValueError("test_verify_tree_greedy not found")


def build_baseline(checkout: Path) -> str:
    result = BASELINE_HEADER.format(
        repository=REPOSITORY, commit=COMMIT, tests=TESTS,
        quoted=_quote_test(checkout),
    ) + BASELINE_BODY
    ast.parse(result)
    return result


def flatten(name: str, checkout: Path) -> str:
    spec = SOURCES[name]
    path = f"{KERNELS}/{spec['file']}"
    text = (checkout / path).read_text()

    tree = ast.parse(text)
    node = tree.body[0]
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        lead = "".join(text.splitlines(keepends=True)[:node.lineno - 1])
        body = "".join(text.splitlines(keepends=True)[node.end_lineno:])
    else:
        lead, body = "", text

    body, count = CDIV_IMPORT.subn("", body)
    inline = CDIV_INLINE if count else ""
    if count > 1:
        raise ValueError(f"{name}: {count} cdiv imports")

    if "sgl_jax" in body:
        offender = next(l for l in body.splitlines() if "sgl_jax" in l)
        raise ValueError(f"{name}: unresolved reference: {offender!r}")
    launches = body.count("pl.pallas_call")
    if launches != 1:
        raise ValueError(f"{name}: expected 1 pallas_call, found {launches}")
    for needed in (f"def {spec['entry']}(",):
        if needed not in body:
            raise ValueError(f"{name}: lost {needed}")
    if spec["wrapper"] and f"def {spec['wrapper']}(" not in body:
        raise ValueError(f"{name}: lost wrapper {spec['wrapper']}")

    header = HEADER.format(
        name=name.replace("_", " "), repository=REPOSITORY, commit=COMMIT,
        path=path, entry=spec["entry"], contract=spec["contract"],
        summary=spec["summary"],
        inline_note=("  also inlines: python/sgl_jax/srt/utils/common_utils.py"
                     "  (only: cdiv)\n" if count else ""),
        transformation=(
            "one repo-local import resolved -- `cdiv`, three lines,\n"
            "    inlined below. No kernel body was touched."
            if count else
            "none; the file imports only jax, so it is carried\n"
            "    verbatim below this header with a `kernel` alias appended."),
        wrapper_note=(f", with the host-side wrapper ``{spec['wrapper']}``"
                      if spec["wrapper"] else ""),
        reference_note=(NO_ORACLE_NOTE if name == "build_eagle_tree"
                        else ORACLE_NOTE.format(tests=TESTS)),
    )
    result = (lead + header + inline + body.strip("\n")
              + f"\n\n\nkernel = {spec['entry']}\n")
    ast.parse(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path, help="sglang-jax checkout root")
    parser.add_argument("corpus_dir", type=Path,
                        help="kernels/sampling/speculative_decoding")
    args = parser.parse_args()
    args.corpus_dir.mkdir(parents=True, exist_ok=True)
    (args.corpus_dir / "__init__.py").touch()
    for name, spec in SOURCES.items():
        (args.corpus_dir / spec["output"]).write_text(flatten(name, args.checkout))
    (args.corpus_dir / "baseline.py").write_text(build_baseline(args.checkout))
    print(f"wrote {len(SOURCES)} kernels and baseline.py into {args.corpus_dir}")


if __name__ == "__main__":
    main()
