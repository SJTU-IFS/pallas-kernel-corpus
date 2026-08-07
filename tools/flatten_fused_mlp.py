"""Collect sglang-jax's fused gated-MLP kernel, and build its reference.

`python/sgl_jax/srt/kernels/fused_mlp.py` imports only `jax`, so the kernel file
needs no flattening -- a provenance header and a `kernel` alias, and that is all.

The reference is the work.  sglang-jax ships no test and no baseline for this
kernel; its single caller, `srt/models/glm5_moe.py`, is the only place the
calling convention is written down, and it holds two things the kernel cannot be
checked without:

* how `w_gu` is packed.  It is **not** `concat([w_gate, w_up])`.
  `post_load_weights` reshapes both to `(hidden, num_blocks, b_inter)` and
  concatenates on the *last* axis, so `w_gu` interleaves gate and up **per
  block**: columns `[i*2b, i*2b+b)` are gate block `i` and `[i*2b+b, (i+1)*2b)`
  are up block `i`.  The kernel relies on this -- it slices `hu[:, :b_inter]`
  and `hu[:, b_inter:]` out of one `(b_seq, 2*b_inter)` tile.  A reference built
  on the obvious whole-matrix concatenation computes a different function and
  fails for a reason that looks like a kernel bug.
* what the kernel is supposed to equal.  `__call__` keeps a non-fused fallback
  next to the fused call, and that fallback is upstream's own statement of the
  contract: `down(up(x) * silu(gate(x)))`.

So this script extracts both regions from `glm5_moe.py` verbatim and embeds them
in the generated reference, beside the corpus functions that reproduce them.
The quoted text is extracted, not transcribed, so it cannot drift from the claim
made about it.

    python tools/flatten_fused_mlp.py <sglang-jax-checkout> <corpus-dir>
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import textwrap


REPOSITORY = "https://github.com/sgl-project/sglang-jax"
COMMIT = "a7353325e8c00d287294c2cd679a77173f1a4594"
KERNEL_PATH = "python/sgl_jax/srt/kernels/fused_mlp.py"
CALLER_PATH = "python/sgl_jax/srt/models/glm5_moe.py"

KERNEL_HEADER = '''"""Standalone sglang-jax fused gated MLP (SwiGLU) kernel.

Source:
  repository: {repository}
  commit: {commit}
  path: {kernel_path}
  transformation: no repo-local imports to resolve -- the upstream file
    depends only on `jax`.  One import is repointed: upstream writes `from jax
    import shard_map` and calls it with `check_rep=False`, but in this corpus's
    pinned jax 0.10.2 that name is the newer API taking `check_vma`.  The
    import is redirected to `jax.experimental.shard_map`, which the same jax
    still ships and which still takes `check_rep`, so upstream's call site is
    left exactly as written rather than translated between two flags that do
    not mean the same thing.  A `kernel` alias is appended.

Entry points: ``apply_fused_mlp_with_padding`` (pads the sequence to a multiple
of ``b_seq``), ``apply_fused_mlp_sharded`` (requires it already aligned),
``local_fused_mlp`` (the `shard_map`ped body holding the `pallas_call`), and
``kernel`` (alias of ``apply_fused_mlp_with_padding``).

Contract ``fused_gated_mlp``::

    x     [S, H]      activations
    w_gu  [H, 2*I]    gate and up weights, interleaved per b_inter block
    wd    [I, H]      down weight
    mesh              must carry a "tensor" axis
    ->    [S, H]      down(up(x) * silu(gate(x)))

``w_gu`` is **not** `concat([w_gate, w_up], axis=1)`.  Gate and up are
interleaved in blocks of ``b_inter``: columns `[i*2b, i*2b+b)` are gate block
`i`, `[i*2b+b, (i+1)*2b)` are up block `i`.  The kernel depends on it -- it
slices both halves out of a single `(b_seq, 2*b_inter)` VMEM tile -- and
`sglang_jax_reference.pack_gate_up` in this directory builds it, reproducing
`glm5_moe.post_load_weights`.

The kernel fuses both projections, the SiLU gate and the down projection into
one pipeline (`pltpu.emit_pipeline` over the intermediate dimension, weights
triple-buffered), so no intermediate reaches HBM.  It ends in
`jax.lax.psum(..., axis_name="tensor")`: `wd` is sharded on its *contracting*
dimension, so each shard holds a partial sum. On a one-device mesh that psum is
the identity, which is how this corpus validates it -- see
tests/test_fused_mlp_tpu.py.

Native shape: sglang-jax sets `b_seq = 64 if seq_len <= 8 else 256` and takes
`b_inter` from the model config; no single shape is declared upstream.
"""

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{kernel_path}",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "fused_gated_mlp",
    "family": "gated_mlp",
    "launch_points": 1,
    "native_shape": None,
}}

'''

#: sglang-jax pins a JAX where `jax.shard_map` still took `check_rep`.  In this
#: corpus's jax 0.10.2 that name is the newer API, whose corresponding argument
#: is `check_vma` -- so upstream's call raises TypeError.  The legacy entry
#: point is still shipped at `jax.experimental.shard_map`, and it still takes
#: `check_rep`, so the import is repointed rather than the call rewritten:
#: `check_rep` and `check_vma` are not the same flag, and translating between
#: them would be a guess about intent rather than a mechanical substitution.
SHARD_MAP_IMPORT = "from jax import shard_map\n"
SHARD_MAP_IMPORT_REPLACEMENT = (
    "from jax.experimental.shard_map import shard_map  # corpus: see"
    " tools/flatten_fused_mlp.py\n"
)

#: Upstream puts this after its module docstring; inserting a `SOURCE` dict
#: above it would leave a `__future__` import below executable code, which is a
#: SyntaxError.  It is hoisted into the header instead.
FUTURE_IMPORT = "from __future__ import annotations\n"

KERNEL_FOOTER = """

# The entry point glm5_moe calls: it pads the sequence to a multiple of b_seq
# first, which `apply_fused_mlp_sharded` requires but does not enforce.
kernel = apply_fused_mlp_with_padding
"""

REFERENCE_HEADER = '''"""JAX reference for the sglang-jax fused gated MLP in this directory.

sglang-jax has no test and no baseline for `fused_mlp.py`.  Both functions here
are read off its only caller, `{caller_path}`, whose relevant regions are quoted
verbatim below -- extracted by tools/flatten_fused_mlp.py, not transcribed, so
the quotation cannot drift from the code it claims to reproduce.

`pack_gate_up` matters more than it looks.  `w_gu` is not the whole gate matrix
beside the whole up matrix; the two are interleaved in blocks of `b_inter`,
because the kernel slices gate and up out of one `(b_seq, 2*b_inter)` tile.
Packing them the obvious way computes a different function, and the failure
looks like a kernel bug rather than a calling-convention error.

Source:
  repository: {repository}
  commit: {commit}
  path: {caller_path}  (both regions below)

Upstream, `{caller_path}`, `post_load_weights` -- the packing::

{packing}

Upstream, `{caller_path}`, `__call__` -- the non-fused fallback, which is what
the fused kernel must equal::

{fallback}
"""

from __future__ import annotations

SOURCE = {{
    "kind": "reference",
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{caller_path}",
    "backend": "jax",
    "target": "portable",
    "contracts": ("fused_gated_mlp",),
}}

'''

#: The reference's own code, kept out of the formatted template above: it
#: contains an f-string, whose braces `str.format` would otherwise consume.
REFERENCE_BODY = '''import jax
import jax.numpy as jnp


def pack_gate_up(wg: jax.Array, wu: jax.Array, b_inter: int) -> jax.Array:
    """Interleave gate and up weights per `b_inter` block, as the kernel wants.

    Reproduces the reshape/concatenate/reshape in `post_load_weights` above,
    without the sharding annotations, which are what that code is mostly made
    of and which do not affect the values.
    """
    hidden_size, inter_size = wg.shape
    if inter_size % b_inter:
        raise ValueError(
            f"intermediate size {inter_size} is not a multiple of b_inter "
            f"{b_inter}; upstream pads to make it so before packing")
    num_blocks = inter_size // b_inter
    wg_blocked = wg.reshape(hidden_size, num_blocks, b_inter)
    wu_blocked = wu.reshape(hidden_size, num_blocks, b_inter)
    return jnp.concatenate([wg_blocked, wu_blocked], axis=-1).reshape(
        hidden_size, inter_size * 2)


def gated_mlp(x: jax.Array, wg: jax.Array, wu: jax.Array,
              wd: jax.Array) -> jax.Array:
    """`down(up(x) * silu(gate(x)))` -- upstream's own non-fused fallback.

    Takes the *unpacked* gate and up weights: this is the definition of the
    task, so it should not have to know how the kernel packs its operands.
    """
    a1 = x @ wg
    a2 = x @ wu
    return (a2 * jax.nn.silu(a1)) @ wd
'''


def _region(source: str, function: str, containing: str) -> str:
    """The verbatim source of a named function, picked by what is inside it.

    `glm5_moe.py` defines several `post_load_weights` and several `__call__`,
    one per module class, so a name alone selects the wrong one -- silently,
    since any of them parses.  `containing` names a line only the intended
    function has.
    """
    found = [
        segment
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == function
        and containing in (segment := ast.get_source_segment(source, node))
    ]
    if len(found) != 1:
        raise ValueError(
            f"{function} containing {containing!r}: expected 1 match, "
            f"found {len(found)}")
    return found[0]


def _quote(text: str, keep: tuple[str, ...]) -> str:
    """Indent a source region for a docstring, keeping only the lines that matter."""
    lines = textwrap.dedent(text).splitlines()
    kept, eliding = [], False
    for line in lines:
        if any(token in line for token in keep):
            if eliding:
                kept.append("    ...")
                eliding = False
            kept.append(line)
        elif kept:
            eliding = True
    return textwrap.indent("\n".join(kept), "    ")


def flatten_kernel(checkout: Path) -> str:
    text = (checkout / KERNEL_PATH).read_text()
    tree = ast.parse(text)
    body = "".join(text.splitlines(keepends=True)[tree.body[0].end_lineno:])
    if "sgl_jax" in body:
        raise ValueError("unresolved upstream reference: sgl_jax")
    if body.count("pl.pallas_call") != 1:
        raise ValueError(f"expected 1 pallas_call, found {body.count('pl.pallas_call')}")
    for name in ("def local_fused_mlp(", "def apply_fused_mlp_sharded(",
                 "def apply_fused_mlp_with_padding("):
        if name not in body:
            raise ValueError(f"lost entry point: {name}")
    if body.count(FUTURE_IMPORT) != 1:
        raise ValueError("expected exactly one __future__ import to hoist")
    body = body.replace(FUTURE_IMPORT, "")
    if body.count(SHARD_MAP_IMPORT) != 1:
        raise ValueError("expected exactly one `from jax import shard_map`")
    if "check_rep=" not in body:
        raise ValueError(
            "upstream no longer passes check_rep; the legacy shard_map import "
            "substitution is no longer the right change")
    body = body.replace(SHARD_MAP_IMPORT, SHARD_MAP_IMPORT_REPLACEMENT)
    header = KERNEL_HEADER.format(repository=REPOSITORY, commit=COMMIT,
                                  kernel_path=KERNEL_PATH)
    docstring, rest = header.split('"""\n', 2)[0] + '"""\n', header.split('"""\n', 2)[-1]
    result = (docstring + "\n" + FUTURE_IMPORT + rest
              + body.strip("\n") + KERNEL_FOOTER)
    ast.parse(result)
    return result


def flatten_reference(checkout: Path) -> str:
    caller = (checkout / CALLER_PATH).read_text()
    packing = _quote(
        _region(caller, "post_load_weights", "w_gu = jnp.concatenate"),
        ("wg =", "wu =", "wd =", "num_blocks =", "wg_reshaped", "wu_reshaped",
         "w_gu =", "reshape(", "concatenate", "self.w_gu.value", "self.w_d.value"),
    )
    fallback = _quote(
        _region(caller, "__call__", "apply_fused_mlp_with_padding("),
        ("a1, _ =", "a2, _ =", "intermediate_parallel", "output, _ =",
         "return output"),
    )
    result = REFERENCE_HEADER.format(
        repository=REPOSITORY, commit=COMMIT, caller_path=CALLER_PATH,
        packing=packing, fallback=fallback,
    ) + REFERENCE_BODY
    ast.parse(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path, help="sglang-jax checkout root")
    parser.add_argument("corpus_dir", type=Path, help="kernels/moe/gated_mlp")
    args = parser.parse_args()
    (args.corpus_dir / "sglang_jax_optimized.py").write_text(
        flatten_kernel(args.checkout))
    (args.corpus_dir / "sglang_jax_reference.py").write_text(
        flatten_reference(args.checkout))
    print(f"wrote sglang_jax_optimized.py and sglang_jax_reference.py "
          f"into {args.corpus_dir}")


if __name__ == "__main__":
    main()
