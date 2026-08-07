"""Collect Tokamax's linear softmax cross-entropy loss kernel pair.

`pallas_mosaic_tpu_kernel.py` holds both launch points -- the forward and the
backward -- and imports nothing from Tokamax, so there is no repo-local
flattening to do.  What it does import is two packages outside this corpus's
pinned dependency set, and neither can simply be dropped:

``jaxtyping``
    Every signature is annotated `Real[Array, "B H"]`.  The module has no
    `from __future__ import annotations`, so those expressions are *evaluated*
    at import time -- deleting the import turns every `def` into a NameError.
    They are replaced with subscriptable stand-ins.  Nothing is lost that was
    there: jaxtyping only checks shapes when its runtime checker is installed,
    which upstream does not do here either.

``pydantic``
    `Config` is a `pydantic.dataclasses.dataclass`, and its three fields carry
    `Field(ge=..., multiple_of=128)` constraints.  Swapping in the stdlib
    `dataclasses.dataclass` keeps the shape of the class but silently drops the
    validation, and these constraints are real: a `v_block_size` that is not a
    multiple of 128 does not lay out on a TPU lane.  So the constraints are
    reproduced as an explicit `__post_init__` rather than deleted -- the same
    check, written as code instead of as metadata.

Both substitutions are asserted against the exact upstream text, so an upstream
edit fails here rather than producing a file that quietly means something else.

The reference is upstream's own `reference.py` from the same directory, carried
to `tokamax_reference.py` with the same jaxtyping treatment.

    python tools/flatten_linear_cross_entropy.py <tokamax-op-dir> <corpus-dir>
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


REPOSITORY = "https://github.com/openxla/tokamax"
COMMIT = "927e3f94e8ffe0430cf38bd1423112bb2f69ec66"
UPSTREAM = "tokamax/_src/ops/linear_softmax_cross_entropy_loss"

#: The jaxtyping stand-ins.  `Real[Scalar, ""] | Real[Array, "B"]` appears in a
#: return annotation, so `__or__` is needed as well as `__getitem__`.
JAXTYPING_SHIM = '''
# ---- corpus substitution for `jaxtyping` ------------------------------------
# Upstream annotates every signature with jaxtyping shape types, and this module
# has no `from __future__ import annotations`, so those expressions are
# evaluated when each `def` is executed.  jaxtyping is not in this corpus's
# pinned dependency set, so these stand-ins take its place: subscriptable and
# unionable, and inert.  They check nothing -- but neither does jaxtyping
# without its runtime checker, which upstream does not install here.
class _ShapeAnnotation:
  """A subscriptable placeholder standing in for a jaxtyping shape type."""

  def __init__(self, name: str):
    self._name = name

  def __getitem__(self, item):
    return self

  def __or__(self, other):
    return self

  def __ror__(self, other):
    return self

  def __repr__(self):
    return self._name


Array = _ShapeAnnotation("Array")
Integer = _ShapeAnnotation("Integer")
Real = _ShapeAnnotation("Real")
Scalar = _ShapeAnnotation("Scalar")
'''

#: Exactly the four lines upstream uses to alias jaxtyping's names.
JAXTYPING_ALIASES = '''Array = jt.Array
Integer = jt.Integer
Real = jt.Real
Scalar = jt.Scalar
'''

#: Tokamax pins a JAX that re-exports `TensorCoreMesh` with an `axis_name=`
#: convenience constructor.  In this corpus's jax 0.10.2 the class is no longer
#: public and does not take that keyword -- it takes `(devices, axis_names)` --
#: and the supported way to build one is the factory below, which returns
#: exactly `TensorCoreMesh([TensorCore(0)], ("core",))` on a one-core device.
#: So this is a rename to the current public spelling of the same object, not a
#: change of what the kernel is given.
TENSORCORE_MESH = 'pltpu.TensorCoreMesh(axis_name="core")'
TENSORCORE_MESH_REPLACEMENT = 'pltpu.create_tensorcore_mesh("core")'

#: The backward's two outputs.  Upstream leaves their memory space implicit,
#: but its `emit_pipeline` `out_specs` are bare `pl.BlockSpec(memory_space=
#: pltpu.HBM)` -- whole-array passthrough, because the backward DMAs into those
#: refs itself rather than letting the pipeline window them.  `emit_pipeline`
#: takes that passthrough path only when the source ref's memory space already
#: equals the buffer's; under this corpus's jax 0.10.2 an implicit `out_type`
#: lands in ANY, the spaces differ, and the pipeline tries to allocate a
#: whole-array buffer in HBM -- which Mosaic rejects outright ("Cannot allocate
#: ref in non-VMEM/SMEM memory space using memref.alloca").  Naming HBM makes
#: explicit what upstream's own out_specs already asked for; it does not move
#: any data, since those refs were always HBM refs written by explicit DMA.
#: The forward needs nothing: its outputs are windowed into VMEM, so it never
#: takes this path.  The change is validated rather than argued -- the backward
#: matches upstream's own `reference.py` to 1e-4 for all three reductions, in
#: tests/test_cross_entropy_tpu.py.
BWD_OUT_TYPE = """      out_type=[
          jax.ShapeDtypeStruct(x.shape, dtype=jnp.float32),  # x_grad
          jax.ShapeDtypeStruct(w.shape, dtype=jnp.float32),  # w_grad
      ],"""

BWD_OUT_TYPE_REPLACEMENT = """      out_type=[
          # corpus: memory space named explicitly -- see
          # tools/flatten_linear_cross_entropy.py
          pltpu.HBM(x.shape, dtype=jnp.float32),  # x_grad
          pltpu.HBM(w.shape, dtype=jnp.float32),  # w_grad
      ],"""

#: Upstream's pydantic-validated field declarations, and the replacement.
PYDANTIC_FIELDS = '''  b_block_size: Annotated[int, pydantic.Field(ge=1024, multiple_of=128)] = 1024
  h_block_size: Annotated[int, pydantic.Field(ge=128, multiple_of=128)] = 512
  v_block_size: Annotated[int, pydantic.Field(ge=128, multiple_of=128)] = 2048
'''

PYDANTIC_FIELDS_REPLACEMENT = '''  b_block_size: int = 1024
  h_block_size: int = 512
  v_block_size: int = 2048

  def __post_init__(self):
    # Upstream states these as `pydantic.Field(ge=..., multiple_of=128)`.
    # pydantic is not in this corpus's pinned dependency set, so the same
    # constraints are checked here as code: dropping them would leave a
    # `Config` that constructs happily and then produces a block size that
    # does not lay out on a TPU lane.
    for name, minimum in (("b_block_size", 1024), ("h_block_size", 128),
                          ("v_block_size", 128)):
      value = getattr(self, name)
      if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
      if value % 128:
        raise ValueError(f"{name} must be a multiple of 128, got {value}")
'''

KERNEL_HEADER = '''"""Standalone Tokamax linear softmax cross-entropy loss (forward and backward).

Source:
  repository: {repository}
  commit: {commit}
  path: {upstream}/pallas_mosaic_tpu_kernel.py
  transformation: no repo-local imports to resolve -- the upstream module
    depends only on `jax` -- but two third-party packages outside this corpus's
    pinned dependency set are substituted:
      * `jaxtyping`, whose shape annotations are evaluated at definition time
        (the module has no `from __future__ import annotations`), is replaced
        by inert subscriptable stand-ins;
      * `pydantic`, whose validated dataclass backs `Config`, is replaced by
        the stdlib `dataclasses` plus an explicit `__post_init__` reproducing
        the `ge` and `multiple_of` constraints its `Field`s declared.
    Two further changes follow the pinned jax rather than the corpus's
    preferences.  `pltpu.TensorCoreMesh(axis_name="core")` becomes
    `pltpu.create_tensorcore_mesh("core")`, the current public spelling for the
    same one-core mesh.  And the backward's `out_type` names `pltpu.HBM`
    explicitly where upstream left the memory space implicit: its
    `emit_pipeline` out_specs are HBM passthroughs, which this jax only honours
    when the output ref is already in HBM -- otherwise it tries to allocate a
    whole-array buffer there and Mosaic refuses.  Neither changes what the
    kernel computes, and the backward is checked against upstream's own
    reference to 1e-4.
    Every substitution is asserted against upstream's exact text by
    tools/flatten_linear_cross_entropy.py.  No kernel logic is touched.

Entry points:
  ``linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu``   forward
  ``linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu``   backward
  ``kernel``                                                    alias of forward

Contract ``linear_softmax_cross_entropy``::

    x       [B, H]   final-layer activations
    labels  [B]      int32 class indices, not one-hot
    w       [H, V]   projection to vocabulary
    ->      loss, lse

The point of the kernel is what it does *not* materialise: logits are `[B, V]`,
which at a real vocabulary is larger than everything else combined, so the
forward blocks over B, H **and** V and accumulates across V by the log-linearity
of log-sum-exp, never writing logits to HBM.  It returns `lse` alongside the
loss so the backward can recompute the logits blockwise instead of storing them,
which is why the backward takes `lse` as an argument rather than recomputing it.

`reduction` is `"sum"`, `"mean"` or `"none"`, and it changes the *shape* of the
loss (scalar vs `[B]`) as well as its scale, so it must match between the two
passes and the reference.

The reference is upstream's own `reference.py`, carried in this directory as
`tokamax_reference.py`.

Native shape: none declared; upstream's own tests parameterise
(B, H, V) = (1024, 512, 2048), (4096, 1024, 4096) and (16384, 4096, 16384).
"""

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{upstream}/pallas_mosaic_tpu_kernel.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "linear_softmax_cross_entropy",
    "family": "cross_entropy",
    "launch_points": 2,
    "native_shape": None,
}}

import dataclasses

'''

KERNEL_FOOTER = """

kernel = linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu
"""

REFERENCE_HEADER = '''"""Tokamax's own JAX reference for the cross-entropy kernels in this directory.

This is `{upstream}/reference.py` at the commit below, unmodified apart from
this header and the same `jaxtyping` substitution the kernel file needed -- the
module upstream's own kernel tests compare against.

It defines both directions.  The backward's signature is worth reading before
using it: it takes `lse` from the forward, because the kernel it checks does not
store logits and recomputes them from `lse` instead.

Source:
  repository: {repository}
  commit: {commit}
  path: {upstream}/reference.py
"""

SOURCE = {{
    "kind": "reference",
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{upstream}/reference.py",
    "backend": "jax",
    "target": "portable",
    "contracts": ("linear_softmax_cross_entropy",),
}}

'''

#: Upstream's reference imports jaxtyping by name rather than aliasing it.
REFERENCE_IMPORT = "from jaxtyping import Array, Integer, Real  # pylint: disable=g-multiple-import\n"


def _after_docstring(text: str) -> tuple[str, str]:
    """Split into (everything before the module docstring, everything after)."""
    tree = ast.parse(text)
    node = tree.body[0]
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)):
        raise ValueError("expected a module docstring")
    lines = text.splitlines(keepends=True)
    return "".join(lines[:node.lineno - 1]), "".join(lines[node.end_lineno:])


def _unresolved(code: str, names: frozenset[str]) -> list[str]:
    """Names from `names` still *referenced* by the code, ignoring prose.

    A substring scan is not usable here: the replacement text explains what it
    replaced, so it says "pydantic" in a comment, and a naive check flags its
    own documentation.  This looks at what the parsed module actually reads.
    """
    tree = ast.parse(code)
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in names:
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in names:
                used.add(root.id)
        elif isinstance(node, ast.Import):
            used |= {a.name.split(".")[0] for a in node.names} & names
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in names:
                used.add(node.module.split(".")[0])
    return sorted(used)


def _require(text: str, needle: str, what: str) -> None:
    if text.count(needle) != 1:
        raise ValueError(
            f"{what}: expected exactly one occurrence of {needle!r}, "
            f"found {text.count(needle)} -- upstream changed, check the "
            f"substitution still means the same thing")


def flatten_kernel(source_dir: Path) -> str:
    text = (source_dir / "pallas_mosaic_tpu_kernel.py").read_text()
    license_header, body = _after_docstring(text)

    for needle, what in (
        ("import jaxtyping as jt\n", "jaxtyping import"),
        ("import pydantic\n", "pydantic import"),
        ("@pydantic.dataclasses.dataclass(frozen=True)\n", "pydantic decorator"),
        (JAXTYPING_ALIASES, "jaxtyping aliases"),
        (PYDANTIC_FIELDS, "pydantic field declarations"),
    ):
        _require(body, needle, what)

    body = body.replace("import jaxtyping as jt\n", "")
    body = body.replace("import pydantic\n", "")
    body = body.replace(JAXTYPING_ALIASES, JAXTYPING_SHIM.lstrip("\n"))
    body = body.replace("@pydantic.dataclasses.dataclass(frozen=True)\n",
                        "@dataclasses.dataclass(frozen=True)\n")
    body = body.replace(PYDANTIC_FIELDS, PYDANTIC_FIELDS_REPLACEMENT)
    # One per launch point; both are the identical expression.
    if body.count(TENSORCORE_MESH) != 2:
        raise ValueError(
            f"expected 2 TensorCoreMesh constructions, found "
            f"{body.count(TENSORCORE_MESH)}")
    body = body.replace(TENSORCORE_MESH, TENSORCORE_MESH_REPLACEMENT)
    _require(body, BWD_OUT_TYPE, "backward out_type")
    body = body.replace(BWD_OUT_TYPE, BWD_OUT_TYPE_REPLACEMENT)
    # `Annotated` was imported only for those three fields.
    body = body.replace("from typing import Annotated, Literal\n",
                        "from typing import Literal\n")

    leftover = _unresolved(body, frozenset({"pydantic", "jaxtyping", "jt",
                                            "tokamax"}))
    if leftover:
        raise ValueError(f"unresolved references after substitution: {leftover}")
    launches = body.count("@pl.kernel(")
    if launches != 2:
        raise ValueError(f"expected 2 pl.kernel launch points, found {launches}")
    for name in ("def linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu(",
                 "def linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu("):
        if name not in body:
            raise ValueError(f"lost entry point: {name}")

    result = (license_header
              + KERNEL_HEADER.format(repository=REPOSITORY, commit=COMMIT,
                                     upstream=UPSTREAM)
              + body.strip("\n") + KERNEL_FOOTER)
    ast.parse(result)
    return result


def flatten_reference(source_dir: Path) -> str:
    text = (source_dir / "reference.py").read_text()
    license_header, body = _after_docstring(text)
    _require(body, REFERENCE_IMPORT, "reference jaxtyping import")
    body = body.replace(REFERENCE_IMPORT, JAXTYPING_SHIM.lstrip("\n"))
    leftover = _unresolved(body, frozenset({"jaxtyping", "jt", "tokamax", "pl",
                                            "pltpu", "pallas"}))
    if leftover:
        raise ValueError(f"the reference must not reference: {leftover}")
    result = (license_header
              + REFERENCE_HEADER.format(repository=REPOSITORY, commit=COMMIT,
                                        upstream=UPSTREAM)
              + body.strip("\n") + "\n")
    ast.parse(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path,
                        help=f"{UPSTREAM} in a tokamax checkout")
    parser.add_argument("corpus_dir", type=Path, help="kernels/loss/cross_entropy")
    args = parser.parse_args()
    (args.corpus_dir / "tokamax_optimized.py").write_text(
        flatten_kernel(args.source_dir))
    (args.corpus_dir / "tokamax_reference.py").write_text(
        flatten_reference(args.source_dir))
    print(f"wrote tokamax_optimized.py and tokamax_reference.py "
          f"into {args.corpus_dir}")


if __name__ == "__main__":
    main()
