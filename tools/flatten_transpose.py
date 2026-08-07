"""Flatten tpu-inference's layout-transpose kernels into one standalone file.

`mla/v2/transpose.py` holds three Pallas launch points that are all about
*moving* data rather than computing on it:

``xpose_full``           whole array into VMEM, transpose, out
``xpose_pipeline``       tiled and double-buffered, for arrays that do not fit
``pin_vmem_custom_call`` an identity kernel, whose only effect is that its
                         output is pinned in VMEM

Two of its three imports are repo-local (`get_dtype_packing` and the vLLM
logger) and one is `sympy`, which is not in the pinned dependency set; all three
are resolved the same way the MLA flatten resolves them, reusing that script's
`divisors` replacement verbatim rather than writing a second one.

**This file duplicates code already in the corpus, deliberately.**
`kernels/attention/mla_attention/tpu_inference_v2_optimized.py` inlines
`xpose_pipeline` and `prev_closest_valid_divisor`, because MLA v2 physically
transposes its head-major operands on entry and exit and would not run without
them.  That copy is a *dependency* and is not counted as an MLA launch point;
this file is the *migration*, and is.  Since the two must not drift,
tests/test_layout_transpose_tpu.py compares them by AST.

    python tools/flatten_transpose.py <tpu-inference-checkout> <corpus-dir>
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re

from flatten_mla import SYMPY_DIVISORS_REPLACEMENT


REPOSITORY = "https://github.com/vllm-project/tpu-inference"
COMMIT = "8b9c90928c94c7230d1bc891534a301510a6a30d"
UPSTREAM = "tpu_inference/kernels/mla/v2/transpose.py"
UTIL = "tpu_inference/kernels/ragged_paged_attention/v3/util.py"

#: The two repo-local imports, and the sympy one.
LOCAL_IMPORTS = (
    re.compile(
        r"^from tpu_inference\.kernels\.ragged_paged_attention\.v3\.util import"
        r" \\\n    get_dtype_packing\s*$",
        re.M,
    ),
    re.compile(r"^from tpu_inference\.logger import init_logger\s*$", re.M),
    re.compile(r"^from sympy import divisors\s*$", re.M),
)

#: `get_dtype_packing` and the one helper it calls, from the RPA util module.
#: Copied rather than reimplemented: `dtypes.itemsize_bits` is the detail that
#: makes this correct for fp8 and int4, and a "32 // bits" written from memory
#: would be right until it met a sub-byte dtype.
UTIL_INLINE = '''# ---- inlined from {util}: get_dtype_bitwidth, get_dtype_packing ----

from jax._src import dtypes


def get_dtype_bitwidth(dtype):
    return dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


'''

HEADER = '''"""Standalone vLLM tpu-inference layout-transpose kernels.

Source:
  repository: {repository}
  commit: {commit}
  path: {upstream}
  also inlines: {util}  (only: get_dtype_bitwidth, get_dtype_packing)
  transformation: three imports were resolved and no kernel body was touched.
    `get_dtype_packing` and the helper it calls were inlined from the ragged
    paged attention util module; the vLLM logger became the stdlib one; and
    `sympy.divisors`, used only for host-side tile selection, was replaced by
    the trial-division equivalent shared with tools/flatten_mla.py, because
    sympy is not in the pinned dependency set.

Entry points: ``xpose_full``, ``xpose_pipeline``, ``pin_vmem_custom_call``
(three separate Pallas launches), and ``kernel`` (alias of ``xpose_pipeline``,
the one MLA v2 actually calls).  Each returns a **list**, not an array.

Contract ``layout_transpose``.  All three move data without computing on it, so
the reference is `jnp.transpose` -- which is also what upstream's own
`tests/kernels/transpose_test.py` compares against, and is exact rather than
approximate:

    xpose_full(x, transpose_axes=axes)[0]      == jnp.transpose(x, axes)
    xpose_pipeline(x, transpose_axes=axes)[0]  == jnp.transpose(x, axes)
    pin_vmem_custom_call(x)[0]                 == x

``xpose_full`` maps the whole array into VMEM at once, so it is limited by VMEM;
``xpose_pipeline`` tiles the parallel and pipeline axes and double-buffers, and
is the one to use when the array does not fit.  Its tile sizes are requests, not
commands: `prev_closest_valid_divisor` lowers `n_tile` to the largest divisor of
the axis that is also a multiple of the dtype's sublane packing, warns when it
has to, and **raises** when no such divisor exists rather than quietly using a
non-divisor tile that would leave rows unprocessed.

``pin_vmem_custom_call`` computes nothing at all -- `identity_fn_generator`
copies each input buffer to the matching output. The point is the side effect:
the result is a VMEM-resident buffer, so a later consumer does not pay to fetch
it. That makes it the one kernel here a correctness test cannot really check
beyond bitwise identity, and the ledger says so.

Native shape: none declared. Upstream's tests parameterise 2-D through 4-D,
mostly float8_e4m3fn, including a 128x2048x256 case chosen to be too large for
`xpose_full`.
"""

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{upstream}",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "layout_transpose",
    "family": "layout_transpose",
    "launch_points": 3,
    "native_shape": None,
}}

import logging

'''

FOOTER = """

# The launch MLA v2 calls, and the only one of the three that works on arrays
# too large for VMEM.
kernel = xpose_pipeline
"""

BASELINE = '''"""JAX reference for the layout-transpose kernels in this directory.

There is nothing to derive here, and that is worth stating rather than leaving
implicit.  Elsewhere in this corpus a reference is carried from upstream because
writing one risks encoding a *different task* than the kernel implements -- a
packed cache layout, a fused activation order, a reduction convention.  A
transpose has no such freedom: `jnp.transpose(x, axes)` is the definition, it is
exact rather than approximate, and it is what upstream's own
`tests/kernels/transpose_test.py` compares against.

The identity kernel is the same story with even less to say.

Source:
  repository: {repository}
  commit: {commit}
  path: tests/kernels/transpose_test.py  (upstream's own choice of reference)
"""

from __future__ import annotations

SOURCE = {{
    "kind": "reference",
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "tests/kernels/transpose_test.py",
    "backend": "jax",
    "target": "portable",
    "contracts": ("layout_transpose",),
}}

from collections.abc import Sequence

import jax
import jax.numpy as jnp


def transposed(x: jax.Array, transpose_axes: Sequence[int]) -> jax.Array:
    """What both `xpose_full` and `xpose_pipeline` must return."""
    return jnp.transpose(x, transpose_axes)


def pinned(x: jax.Array) -> jax.Array:
    """What `pin_vmem_custom_call` must return: its input, unchanged.

    The kernel's purpose is a side effect -- leaving the buffer resident in
    VMEM -- which no comparison of values can observe.  What a test can pin
    down is that the values are untouched, bitwise, and that a Pallas kernel
    ran at all rather than the call folding away.
    """
    return x
'''


def flatten(checkout: Path) -> str:
    text = (checkout / UPSTREAM).read_text()
    tree = ast.parse(text)
    license_header = "".join(
        text.splitlines(keepends=True)[:tree.body[0].lineno - 1])
    body = text[text.index("from collections.abc import Sequence"):]

    for pattern in LOCAL_IMPORTS:
        body, count = pattern.subn("", body)
        if count != 1:
            raise ValueError(f"expected 1 match for {pattern.pattern!r}, got {count}")
    body = body.replace("logger = init_logger(__name__)",
                        "logger = logging.getLogger(__name__)")

    for token in ("tpu_inference", "sympy", "init_logger"):
        if token in body:
            offender = next(l for l in body.splitlines() if token in l)
            raise ValueError(f"unresolved reference to {token}: {offender!r}")
    if body.count("pl.pallas_call") != 3:
        raise ValueError(
            f"expected 3 pallas_call, found {body.count('pl.pallas_call')}")
    for name in ("def xpose_full(", "def xpose_pipeline(",
                 "def pin_vmem_custom_call(", "def prev_closest_valid_divisor("):
        if name not in body:
            raise ValueError(f"lost entry point: {name}")

    # The helpers go above the first use, in dependency order.
    marker = "\n\n@jax.jit(static_argnames=[\n    'transpose_axes',\n])"
    if body.count(marker) != 1:
        raise ValueError("could not find the insertion point above xpose_full")
    inserted = (UTIL_INLINE.format(util=UTIL)
                + SYMPY_DIVISORS_REPLACEMENT.rstrip("\n") + "\n")
    body = body.replace(marker, "\n\n" + inserted + marker.lstrip("\n"), 1)

    result = (license_header
              + HEADER.format(repository=REPOSITORY, commit=COMMIT,
                              upstream=UPSTREAM, util=UTIL)
              + body.strip("\n") + FOOTER)
    ast.parse(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path, help="tpu-inference checkout root")
    parser.add_argument("corpus_dir", type=Path,
                        help="kernels/memory/layout_transpose")
    args = parser.parse_args()
    args.corpus_dir.mkdir(parents=True, exist_ok=True)
    (args.corpus_dir / "__init__.py").touch()
    (args.corpus_dir / "tpu_inference_optimized.py").write_text(
        flatten(args.checkout))
    (args.corpus_dir / "baseline.py").write_text(
        BASELINE.format(repository=REPOSITORY, commit=COMMIT))
    print(f"wrote tpu_inference_optimized.py and baseline.py "
          f"into {args.corpus_dir}")


if __name__ == "__main__":
    main()
