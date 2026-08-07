"""Attach corpus provenance to the gated delta net reference and kernels.

The gated delta rule is a linear-attention recurrence: it carries a per-head
recurrent state ``[d_k, d_v]`` across tokens, updating it with a delta rule that
is gated per token. Unlike softmax attention it is O(T) in sequence length, and
unlike a plain RNN the update is a rank-one correction, which is what makes a
blocked TPU kernel worthwhile.

Two things make this family unusual for the corpus:

- **Upstream ships its own pure-JAX reference.** tpu-inference's
  ``kernels/gdn/reference/ragged_gated_delta_rule_ref.py`` imports nothing
  outside ``jax`` and is described upstream as being "mainly for unit test".
  It is copied here as ``baseline.py`` rather than re-derived, for the same
  reason as the sparsecore gathers: hand-writing a recurrence with ragged
  sequence boundaries and a null state block risks a subtly wrong reference
  producing false failures.
- **tpu-inference ``gdn/v3`` and Tokamax ``causal_conv1d_gated_delta_rule`` are
  a diverged vendored pair.** Same seven module names, same 24 top-level
  definition names, but only 10 of those are AST-identical (~42%). They are
  therefore two implementations of one contract, not a duplicate to drop --
  the same call made for JAXBench flash vs PallasBench flash, and the opposite
  of the call made for JAXBench ``4p_Sparse_Attention``.

Like the other ``flatten_*`` scripts this is a reproducibility aid pinned to the
audited snapshots; the generated file is the runnable artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path


TPU_INFERENCE = "https://github.com/vllm-project/tpu-inference"
TPU_INFERENCE_COMMIT = "8b9c90928c94c7230d1bc891534a301510a6a30d"

# The v1 implementation is four modules; flattening concatenates them in
# dependency order.  `get_default_block_sizes` is defined in **both** the decode
# and the recurrent kernel with different bodies, so a naive concatenation
# would silently shadow one with the other -- the renames below keep them
# distinct, the same treatment the quantized-matmul family needed.
V1_MODULES = (
    ("fused_gdn_kernel_common.py", {}),
    ("fused_gdn_decode_kernel.py",
     {"get_default_block_sizes": "get_default_block_sizes_decode"}),
    ("fused_gdn_recurrent_kernel.py",
     {"get_default_block_sizes": "get_default_block_sizes_recurrent"}),
    ("fused_gdn_kernel_wrapper.py", {}),
)

TOKAMAX = "https://github.com/openxla/tokamax"
TOKAMAX_COMMIT = "927e3f94e8ffe0430cf38bd1423112bb2f69ec66"

# The v3 implementations are seven modules each, with the same dependency order
# on both sides.  Unlike v1 there are no name collisions within an
# implementation, so no renaming is needed -- but the order matters, because the
# flattened file is executed top to bottom.
V3_MODULES = (
    "config.py",
    "compute_conv1d.py",
    "compute_gdn.py",
    "memory_ref.py",
    "metadata.py",
    "vmem_ldst.py",
    "wrapper.py",
)

# v2 is four modules with two independent public entry points -- a decode-only
# kernel and a chunked recurrent scan -- rather than one dispatching wrapper.
# `invert_triangular_matrix` is defined in both recurrent_scan_impl and
# recurrent_scan_v2, so it needs the same per-module renaming v1 needed.
V2_MODULES = (
    ("compute_schedule_v2.py", {}),
    ("recurrent_scan_impl.py",
     {"invert_triangular_matrix": "invert_triangular_matrix_impl"}),
    ("gdn_decode_kernel.py", {}),
    ("recurrent_scan_v2.py",
     {"invert_triangular_matrix": "invert_triangular_matrix_scan"}),
)

V3_VARIANTS = {
    "tokamax-v3": {
        "display": "Tokamax",
        "repository": TOKAMAX,
        "commit": TOKAMAX_COMMIT,
        "path": "tokamax/_src/ops/experimental/causal_conv1d_gated_delta_rule",
    },
    "tpu-inference-v3": {
        "display": "vLLM tpu-inference",
        "repository": TPU_INFERENCE,
        "commit": TPU_INFERENCE_COMMIT,
        "path": "tpu_inference/kernels/gdn/v3",
    },
}

SOURCES = {
    "triangle-solver": {
        "display": "vLLM tpu-inference",
        "repository": TPU_INFERENCE,
        "commit": TPU_INFERENCE_COMMIT,
        "path": "tpu_inference/kernels/gdn/triangle_solver.py",
        "entry": "decompose_triangular_matrix_inverse_pallas",
        "contract": "unit_lower_triangular_inverse",
        "kind": "kernel",
    },
    "reference": {
        "display": "vLLM tpu-inference",
        "repository": TPU_INFERENCE,
        "commit": TPU_INFERENCE_COMMIT,
        "path": "tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_ref.py",
        "entry": "ragged_gated_delta_rule",
        "contract": "ragged_gated_delta_rule",
        "kind": "reference",
    },
}

V1_HEADER = '''"""Standalone vLLM tpu-inference gated delta net kernels (v1).

Source:
  repository: {repository}
  commit: {commit}
  path: tpu_inference/kernels/gdn/v1/
  transformation: four upstream modules flattened into this file in dependency
    order -- fused_gdn_kernel_common, fused_gdn_decode_kernel,
    fused_gdn_recurrent_kernel, fused_gdn_kernel_wrapper.  Repo-local imports
    were dropped and `get_default_block_sizes` was renamed per module (see
    below); no other change.

Entry point: ``ragged_gated_delta_rule`` (also exported as ``kernel``).  Its
signature is identical to the pure-JAX reference in ``baseline.py``, so the two
are directly comparable.

**Three Pallas launch points** live here:

- ``calculate_chunk_indices`` -- builds the chunk metadata the recurrent kernel
  scans over;
- ``fused_recurrent_gdn`` -- the chunked prefill path;
- ``fused_decoding_gdn`` -- the single-token decode path.

``ragged_gated_delta_rule`` dispatches between the last two using
``distribution``, so a mixed batch can touch both.

**Renamed on flattening.** Upstream defines ``get_default_block_sizes`` in the
decode kernel *and* in the recurrent kernel, with different bodies. Both are
module-local, so upstream has no conflict; concatenating them into one file
does. They are preserved here as ``get_default_block_sizes_decode`` and
``get_default_block_sizes_recurrent``.
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "tpu_inference/kernels/gdn/v1",
    "files": (
        "fused_gdn_kernel_common.py",
        "fused_gdn_decode_kernel.py",
        "fused_gdn_recurrent_kernel.py",
        "fused_gdn_kernel_wrapper.py",
    ),
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "ragged_gated_delta_rule",
    "launch_points": 3,
    "renamed_on_flatten": {{
        "get_default_block_sizes": (
            "get_default_block_sizes_decode",
            "get_default_block_sizes_recurrent",
        ),
    }},
}}

import dataclasses
import functools

import jax
from jax._src import dtypes
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

'''


V2_HEADER = '''"""Standalone vLLM tpu-inference gated delta net kernels (v2).

Source:
  repository: {repository}
  commit: {commit}
  path: tpu_inference/kernels/gdn/v2/
  transformation: four upstream modules flattened into this file in dependency
    order -- compute_schedule_v2, recurrent_scan_impl, gdn_decode_kernel,
    recurrent_scan_v2.  Repo-local imports were dropped and
    `invert_triangular_matrix` was renamed per module (see below).

Same contract as the v1 kernels (``ragged_gated_delta_rule``), but v2 exposes
**two independent entry points** instead of one dispatching wrapper:

- ``ragged_gated_delta_rule_decode_only`` -- the decode-only path;
- ``recurrent_scan`` -- the chunked prefill / mixed path.

There is no v2 equivalent of v1's combined ``ragged_gated_delta_rule``, so a
caller picks the path rather than passing ``distribution`` to a dispatcher.
``kernel`` is bound to ``recurrent_scan``, the more general of the two.

**On the SiLU precondition.** The v1 wrapper silently expects post-SiLU input
(see ``baseline.PRE_SILU_INPUT``). v2's decode entry point makes it an explicit
``apply_silu`` argument, which is upstream resolving the same ambiguity the
corpus had to document by measurement. ``recurrent_scan`` has no such flag.

**Renamed on flattening.** ``invert_triangular_matrix`` is defined in both
``recurrent_scan_impl`` and ``recurrent_scan_v2`` with different bodies; they
are preserved as ``invert_triangular_matrix_impl`` and
``invert_triangular_matrix_scan``.
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "tpu_inference/kernels/gdn/v2",
    "files": (
        "compute_schedule_v2.py",
        "recurrent_scan_impl.py",
        "gdn_decode_kernel.py",
        "recurrent_scan_v2.py",
    ),
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "ragged_gated_delta_rule",
    "launch_points": 2,
    "entry_points": ("ragged_gated_delta_rule_decode_only", "recurrent_scan"),
    "renamed_on_flatten": {{
        "invert_triangular_matrix": (
            "invert_triangular_matrix_impl",
            "invert_triangular_matrix_scan",
        ),
    }},
}}

import dataclasses
import enum
import functools
import math
from typing import Any

import jax
from jax import lax
from jax._src import dtypes
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import numpy as np

'''


V3_HEADER = '''"""Standalone {display} fused causal-conv1d + gated delta net kernel (v3).

Source:
  repository: {repository}
  commit: {commit}
  path: {path}/
  transformation: seven upstream modules flattened into this file in dependency
    order -- config, compute_conv1d, compute_gdn, memory_ref, metadata,
    vmem_ldst, wrapper.  Repo-local imports were dropped; nothing was renamed,
    because unlike the v1 kernels these modules have no colliding definitions.

Entry point: ``fused_conv1d_gdn`` (also exported as ``kernel``).  **One** Pallas
launch point, which does the whole fused op.

This is a **different contract from the v1 kernels** in this directory. v1
implements ``ragged_gated_delta_rule`` alone; this fuses a depthwise causal
conv1d over the token stream with the gated delta rule, carrying **two** caches
-- a conv state and a recurrent state -- rather than one. It is not a tuning
variant of v1 and the two are not interchangeable.

tpu-inference ``gdn/v3`` and Tokamax ``causal_conv1d_gated_delta_rule`` are a
**diverged vendored pair**: same seven module names and the same 24 top-level
definition names, but only ~42% of those are AST-identical. Both are migrated
so the divergence can be measured rather than assumed away.
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{path}",
    "files": {files!r},
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "fused_conv1d_gated_delta_rule",
    "launch_points": 1,
    "vendored_pair": (
        "tpu_inference/kernels/gdn/v3 and "
        "tokamax/_src/ops/experimental/causal_conv1d_gated_delta_rule are the "
        "same seven modules, diverged to ~42% AST-identical"
    ),
}}

import dataclasses
import enum
import functools
import math
from typing import Any

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import numpy as np

'''


TRIANGLE_HEADER = '''"""Standalone {display} unit-lower-triangular inverse kernels.

Source:
  repository: {repository}
  commit: {commit}
  path: {path}
  transformation: copied as a standalone implementation with source metadata
    added; nothing else changed. The file imports nothing outside jax.

**Two Pallas launch points**, two algorithms for one contract:

- ``newton_schulz_inverse_pallas`` -- Newton-Schulz iteration;
- ``decompose_triangular_matrix_inverse_pallas`` -- blockwise decomposition.

``kernel`` is bound to the blockwise one, which is what the gated delta net
chunked scan calls.

Contract ``{contract}``: given a **unit** lower triangular ``A`` (ones on the
diagonal), return ``A^-1``.

Unlike everything else in this family the contract is a plain mathematical
identity, so the corpus checks it against ``jnp.linalg.inv`` -- an independent
oracle -- as well as against upstream's own references, which live in this same
file (``newton_schulz_inverse_ref`` and ``local_forward_substitution``, both
pure JAX). Everywhere else in gated_delta_net the only available reference is
the one upstream wrote, so an error shared between kernel and reference would
go unnoticed; here it would not.

These are helpers for the chunked scan rather than a model op in their own
right: the chunked algorithm needs the inverse of a triangular matrix built
from the per-chunk decay terms.
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{path}",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "{contract}",
    "launch_points": 2,
    "entry_points": (
        "newton_schulz_inverse_pallas",
        "decompose_triangular_matrix_inverse_pallas",
    ),
}}

'''


HEADER = '''"""Pure-JAX reference for the gated delta rule over ragged sequences.

This is **upstream's own reference**, not a re-derived one:

  repository: {repository}
  commit: {commit}
  path: {path}
  transformation: copied verbatim as the corpus baseline with source metadata
    and a smoke runner added; the implementation is unchanged. The file imports
    nothing outside jax.

Upstream describes it as "mainly for unit test", which is exactly the role a
corpus baseline plays. It is used here rather than a hand-written reference
because the contract has enough moving parts -- ragged sequence boundaries, a
null state block for padded tokens, a `has_initial_state` mask that zeroes
carried state for fresh prefills -- that an independently written version would
more likely be wrong than the kernels it is meant to check.

Contract ``{contract}``::

    mixed_qkv          [num_tokens, 2 * n_kq * d_k + n_v * d_v]
    b, a               [num_tokens, n_v]
    recurrent_state    [num_blocks, n_v, d_k, d_v]   block 0 is the null block
    A_log, dt_bias     [n_v]
    query_start_loc    [num_seqs + 1]    start index per sequence, last = total
    state_indices      [max_reqs]        request index -> state block
    distribution       [3] int32         (decode_end, prefill_end, mixed_end)
    has_initial_state  [max_reqs] bool   False = start from zero state
    ->                 (updated_recurrent_state, output[num_tokens, n_v * d_v])

`n_kq`, `n_v`, `d_k`, `d_v` are keyword-only and static.

The rule itself is a gated rank-one state update per token: `mixed_qkv` is
passed through SiLU and split into query/key/value, query and key are L2
normalized, and the state evolves as a decay-plus-correction recurrence driven
by the per-token gates derived from `a`, `b`, `A_log` and `dt_bias`.
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{path}",
    "kind": "reference",
    "backend": "jax",
    "target": "portable",
    "contract": "{contract}",
}}

'''


def sibling_module_aliases(text: str, stems: set[str]) -> set[str]:
    """Names by which this module refers to its siblings.

    Upstream sometimes aliases a sibling on import, and not always helpfully:
    gdn v2 has ``import compute_schedule_v2 as compute_schedule_table_v2``,
    aliasing the module to the same name as the function inside it. Flattening
    has to strip ``<alias>.`` as well as ``<module>.``, so the aliases are read
    out of the AST rather than assumed to match the file names.
    """
    import ast

    aliases: set[str] = set()
    for node in ast.parse(text).body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in stems:
                    aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                tail = alias.name.split(".")[-1]
                if tail in stems:
                    aliases.add(alias.asname or tail)
    return aliases


def _scope_bindings(node) -> set[str]:
    """Names bound in one function/lambda scope, not counting nested scopes."""
    import ast

    names: set[str] = set()
    args = getattr(node, "args", None)
    if args is not None:
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            names.add(arg.arg)
        for extra in (args.vararg, args.kwarg):
            if extra is not None:
                names.add(extra.arg)

    nested = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        if isinstance(child, nested):
            # The nested definition's *name* binds here; its body does not.
            names.add(getattr(child, "name", ""))
            continue
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                names.add((alias.asname or alias.name).split(".")[0])
        stack.extend(ast.iter_child_nodes(child))
    names.discard("")
    return names


def _module_prefix_sites(text: str, prefixes: set[str]):
    """Locate ``prefix.attr`` uses, split by whether ``prefix`` is shadowed.

    Returns ``(strippable, shadowed)``, each a list of ``(lineno, col, prefix)``.
    """
    import ast

    tree = ast.parse(text)
    strippable: list[tuple[int, int, str]] = []
    shadowed: list[tuple[int, int, str]] = []

    def walk(node, scopes):
        import ast as _ast

        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                             _ast.Lambda)):
            scopes = (*scopes, _scope_bindings(node))
        if (
            isinstance(node, _ast.Attribute)
            and isinstance(node.value, _ast.Name)
            and node.value.id in prefixes
        ):
            site = (node.value.lineno, node.value.col_offset, node.value.id)
            hidden = any(node.value.id in scope for scope in scopes)
            (shadowed if hidden else strippable).append(site)
        for child in _ast.iter_child_nodes(node):
            walk(child, scopes)

    walk(tree, ())
    return strippable, shadowed


def strip_module_prefixes(text: str, prefixes: set[str]) -> tuple[str, list]:
    """Remove ``module.`` qualifiers, except where ``module`` is a local name.

    A blind ``re.sub(rf"\\b{prefix}\\.", "", text)`` is wrong the moment a
    module's own name is also a parameter somewhere in the package, and it fails
    quietly: ``batched_rpa/schedule.py`` takes ``schedule: RpaSchedule`` as an
    argument, so blind stripping rewrote ``schedule.s_idx[step, lane] = s_idx``
    into ``s_idx[step, lane] = s_idx``.  That parses, imports, and passes every
    static check -- it only fails once the kernel runs, with a "JAX arrays are
    immutable" error pointing at a line that looks fine.

    This locates the qualifiers with the AST, skips any whose name is bound in
    an enclosing function scope, and edits by position so the surrounding
    formatting is untouched.  Returns the new text and the skipped sites, which
    the caller should keep for :func:`unbound_module_refs` to check against.
    """
    strippable, shadowed = _module_prefix_sites(text, prefixes)
    lines = text.splitlines(keepends=True)
    for lineno, col, prefix in sorted(strippable, reverse=True):
        line = lines[lineno - 1]
        if line[col:col + len(prefix) + 1] != f"{prefix}.":
            raise ValueError(
                f"line {lineno} col {col}: expected {prefix!r} qualifier, "
                f"found {line[col:col + len(prefix) + 1]!r}"
            )
        lines[lineno - 1] = line[:col] + line[col + len(prefix) + 1:]
    return "".join(lines), shadowed


def rename_top_level(text: str, renames: dict[str, str]) -> str:
    """Rename module-level definitions and their references, via the AST.

    The blind ``re.sub(rf"\\b{old}\\b", new, text)`` the ``renames`` mechanism
    started with is fine for names like ``get_default_block_sizes`` that never
    appear in prose, and wrong for ordinary words: ``proj_and_save_state.py``
    defines a function named ``kernel``, which collides with the corpus's own
    ``kernel = <entry>`` footer, and renaming it by regex would also rewrite
    every occurrence of the word "kernel" in comments and docstrings.

    Only the definition itself and ``Name`` references to it are rewritten, and
    a reference is skipped where the name is bound in an enclosing function
    scope.
    """
    import ast
    import re as _re

    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    edits: list[tuple[int, int, int, str]] = []  # lineno, col, length, new

    def walk(node, scopes):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            scopes = (*scopes, _scope_bindings(node))
        if isinstance(node, ast.Name) and node.id in renames:
            if not any(node.id in scope for scope in scopes):
                edits.append(
                    (node.lineno, node.col_offset, len(node.id),
                     renames[node.id])
                )
        for child in ast.iter_child_nodes(node):
            walk(child, scopes)

    walk(tree, ())

    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef))
            and node.name in renames
        ):
            line = lines[node.lineno - 1]
            match = _re.search(rf"\b{node.name}\b", line)
            if match is None:
                raise ValueError(
                    f"line {node.lineno}: cannot locate definition of "
                    f"{node.name!r}"
                )
            edits.append(
                (node.lineno, match.start(), len(node.name),
                 renames[node.name])
            )

    for lineno, col, length, new in sorted(set(edits), reverse=True):
        line = lines[lineno - 1]
        lines[lineno - 1] = line[:col] + new + line[col + length:]
    return "".join(lines)


def unbound_module_refs(code: str, prefixes: set[str]) -> list[tuple[int, int, str]]:
    """Any ``module.attr`` left over where ``module`` is not a local.

    The old check was ``re.search(rf"\\b{stem}\\.", code)``, which cannot tell a
    genuinely unresolved module reference from a legitimately shadowed local of
    the same name.  This re-runs the scope analysis on the flattened result, so
    the only surviving qualifiers it reports are real leftovers.
    """
    strippable, _ = _module_prefix_sites(code, prefixes)
    return strippable


def strip_module_preamble(text: str) -> str:
    """Remove the license block, module docstring and every import statement.

    Imports are located with `ast` rather than a regex.  A regex over line
    starts silently mangles parenthesised imports -- it deletes the first line
    and leaves the continuation behind as stray indented text, which then fails
    to parse a thousand lines away from the actual cause.
    """
    import ast
    import re

    text = re.sub(r"\A(?:#[^\n]*\n)+", "", text)
    tree = ast.parse(text)
    drop: set[int] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            drop.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    if tree.body and isinstance(tree.body[0], ast.Expr):
        first = tree.body[0]
        if isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            drop.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    lines = text.splitlines(keepends=True)
    return "".join(l for i, l in enumerate(lines, 1) if i not in drop)


def build(name: str, source_dir: Path, filename: str) -> str:
    spec = SOURCES[name]
    text = (source_dir / filename).read_text()
    text = text.replace("from __future__ import annotations\n", "", 1)

    for token in ("tpu_inference.", "tokamax._src", "maxtext.", "sgl_jax."):
        if token in text:
            raise ValueError(f"{name}: unresolved upstream reference {token}")
    if f"def {spec['entry']}(" not in text:
        raise ValueError(f"{name}: lost entry point {spec['entry']}")

    if name == "triangle-solver":
        for entry in (
            "newton_schulz_inverse_pallas",
            "decompose_triangular_matrix_inverse_pallas",
            "newton_schulz_inverse_ref",
            "local_forward_substitution",
        ):
            if f"def {entry}(" not in text:
                raise ValueError(f"{name}: lost {entry}")
        result = (
            TRIANGLE_HEADER.format(**spec) + text
            + "\n\nkernel = decompose_triangular_matrix_inverse_pallas\n"
        )
        import ast as _ast

        _ast.parse(result)
        return result

    return HEADER.format(**spec) + text


def build_v1(source_dir: Path) -> str:
    """Concatenate the four v1 modules into one standalone file."""
    import re

    chunks = []
    for filename, renames in V1_MODULES:
        # The generated header carries one copy of the license, docstring and
        # imports for the whole flattened file.
        text = strip_module_preamble((source_dir / filename).read_text())
        for old_name, new_name in renames.items():
            text = re.sub(rf"\b{old_name}\b", new_name, text)
        chunks.append(f"# --- from {filename} " + "-" * (56 - len(filename))
                      + "\n" + text.strip("\n") + "\n")

    body = "\n\n".join(chunks)
    for token in ("tpu_inference.", "tokamax._src", "maxtext.", "sgl_jax."):
        if token in body:
            raise ValueError(f"v1: unresolved upstream reference {token}")
    for entry in ("ragged_gated_delta_rule", "fused_decoding_gdn",
                  "fused_recurrent_gdn", "calculate_chunk_indices"):
        if f"def {entry}(" not in body:
            raise ValueError(f"v1: lost entry point {entry}")
    if body.count("def get_default_block_sizes_decode(") != 1:
        raise ValueError("v1: decode block-size helper did not survive renaming")
    if body.count("def get_default_block_sizes_recurrent(") != 1:
        raise ValueError("v1: recurrent block-size helper did not survive renaming")

    header = V1_HEADER.format(repository=TPU_INFERENCE, commit=TPU_INFERENCE_COMMIT)
    result = header + body + "\n\nkernel = ragged_gated_delta_rule\n"
    import ast as _ast
    _ast.parse(result)  # a mangled flatten must fail here, not at import time
    return result


def build_v2(source_dir: Path) -> str:
    """Concatenate the four v2 modules into one standalone file."""
    chunks = []
    for filename, renames in V2_MODULES:
        import re

        raw = (source_dir / filename).read_text()
        stems = {module[:-3] for module, _ in V2_MODULES}
        prefixes = stems | sibling_module_aliases(raw, stems)
        text = strip_module_preamble(raw)
        for old_name, new_name in renames.items():
            text = re.sub(rf"\b{old_name}\b", new_name, text)
        text, shadowed = strip_module_prefixes(text, prefixes)
        if shadowed:
            print(f"v2 {filename}: kept {len(shadowed)} shadowed qualifier(s)")
        chunks.append(f"# --- from {filename} " + "-" * (56 - len(filename))
                      + "\n" + text.strip("\n") + "\n")

    body = "\n\n".join(chunks)
    for token in ("tpu_inference.", "tokamax._src", "maxtext.", "sgl_jax."):
        if token in body:
            raise ValueError(f"v2: unresolved upstream reference {token}")
    for entry in ("ragged_gated_delta_rule_decode_only", "recurrent_scan",
                  "fused_decoding_gdn"):
        if f"def {entry}(" not in body:
            raise ValueError(f"v2: lost entry point {entry}")
    for renamed in ("invert_triangular_matrix_impl",
                    "invert_triangular_matrix_scan"):
        if body.count(f"def {renamed}(") != 1:
            raise ValueError(f"v2: {renamed} did not survive renaming")

    header = V2_HEADER.format(
        repository=TPU_INFERENCE, commit=TPU_INFERENCE_COMMIT
    )
    # A surviving `sibling_module.` reference means an alias was missed;
    # it would fail at call time, far from the cause.  The per-chunk banners
    # name the source files, so they are excluded or every run trips on them.
    import re as _re

    code = "\n".join(
        line for line in body.splitlines() if not line.startswith("# --- from ")
    )
    for _stem in {module[:-3] for module, _ in V2_MODULES}:
        if _re.search(rf"\b{_stem}\.", code):
            raise ValueError(f"v2: {_stem}. survived flattening")
    result = header + body + "\n\nkernel = recurrent_scan\n"
    import ast as _ast
    _ast.parse(result)  # a mangled flatten must fail here, not at import time
    return result


def build_v3(name: str, source_dir: Path) -> str:
    """Concatenate the seven v3 modules into one standalone file."""
    import re

    spec = V3_VARIANTS[name]
    chunks = []
    for filename in V3_MODULES:
        raw = (source_dir / filename).read_text()
        stems = {module[:-3] for module in V3_MODULES}
        prefixes = stems | sibling_module_aliases(raw, stems)
        text = strip_module_preamble(raw)
        # Upstream refers to sibling modules by name (or by an alias); flattened,
        # those names are gone and the definitions are simply in scope -- except
        # where the module's name is also a local, which the AST pass detects.
        text, shadowed = strip_module_prefixes(text, prefixes)
        if shadowed:
            print(f"{name} {filename}: kept {len(shadowed)} shadowed qualifier(s)")
        chunks.append(f"# --- from {filename} " + "-" * (56 - len(filename))
                      + "\n" + text.strip("\n") + "\n")

    body = "\n\n".join(chunks)
    for token in ("tpu_inference.", "tokamax._src", "maxtext.", "sgl_jax."):
        if token in body:
            raise ValueError(f"{name}: unresolved upstream reference {token}")
    if "def fused_conv1d_gdn(" not in body:
        raise ValueError(f"{name}: lost entry point fused_conv1d_gdn")

    header = V3_HEADER.format(files=V3_MODULES, **spec)
    # A surviving `sibling_module.` reference means an alias was missed;
    # it would fail at call time, far from the cause.  The per-chunk banners
    # name the source files, so they are excluded or every run trips on them.
    import re as _re

    code = "\n".join(
        line for line in body.splitlines() if not line.startswith("# --- from ")
    )
    for _stem in {module[:-3] for module in V3_MODULES}:
        if _re.search(rf"\b{_stem}\.", code):
            raise ValueError(f"v3: {_stem}. survived flattening")
    result = header + body + "\n\nkernel = fused_conv1d_gdn\n"
    import ast as _ast
    _ast.parse(result)  # a mangled flatten must fail here, not at import time
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name", choices=sorted(SOURCES) + ["v1", "v2"] + sorted(V3_VARIANTS)
    )
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("filename", nargs="?")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.name == "v1":
        text = build_v1(args.source_dir)
    elif args.name == "v2":
        text = build_v2(args.source_dir)
    elif args.name in V3_VARIANTS:
        text = build_v3(args.name, args.source_dir)
    else:
        text = build(args.name, args.source_dir, args.filename)
    args.output.write_text(text)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
