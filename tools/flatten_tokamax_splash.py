"""Mechanically flatten a Tokamax-lineage Splash Attention into one file.

Two copies of this lineage exist in the pinned snapshots: Tokamax's own, and
MaxText's under ``kernels/tokamax_splash_attention/``.  The directory name says
"vendored copy", and the corpus originally excluded MaxText's on that basis --
but measured, only **7 of 20 shared definitions are AST-identical (35%)**.
That is below the 42% at which the corpus migrated both halves of the gdn v3
pair as genuinely diverged, and nowhere near the 90% that justified excluding
JAXBench's 4p_Sparse_Attention.  Both are therefore flattened and counted, with
the measurement recorded rather than the directory name trusted.

Both copies expose the same public factories and both carry **two** Pallas
launch points, not three: unlike the JAXBench and MaxText ``attention/`` splash
kernels, this lineage fuses the dQ computation into the dKV kernel, which
returns ``dq_unreduced, dk, dv`` from a single ``pallas_call``.

This script is pinned to the audited snapshots; the generated file is the
runnable artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re


FILES = (
    "splash_attention_mask.py",
    "splash_attention_mask_info.py",
    "base.py",
    "splash_attention_kernel.py",
)

SOURCES = {
    "tokamax": {
        "display": "Tokamax",
        "repository": "https://github.com/openxla/tokamax",
        "commit": "927e3f94e8ffe0430cf38bd1423112bb2f69ec66",
        "path": "tokamax/_src/ops/experimental/tpu/splash_attention",
        "source_path": "tokamax/_src/ops/experimental/tpu/splash_attention",
        "implementation": "tokamax",
        "local_import": re.compile(
            r"^from tokamax\._src\.ops\.experimental\.tpu\.splash_attention import .*$",
            re.MULTILINE,
        ),
        "extra": "",
    },
    "maxtext": {
        "display": "MaxText (Tokamax splash lineage)",
        "repository": "https://github.com/AI-Hypercomputer/maxtext",
        "commit": "ca420634a9e9e73feaacc8001f605163d2d80ea1",
        "path": "src/maxtext/kernels/tokamax_splash_attention",
        "source_path": (
            "src/maxtext/kernels/tokamax_splash_attention/"
            "splash_attention_kernel.py"
        ),
        "implementation": "maxtext_tokamax_splash",
        "local_import": re.compile(
            r"^from maxtext\.kernels\.tokamax_splash_attention import .*$",
            re.MULTILINE,
        ),
        "extra": (
            "\n\nMaxText vendored this from Tokamax and then diverged: only 7 of the\n"
            "20 shared top-level definitions are AST-identical (35%), and this copy\n"
            "is 2098 lines against Tokamax's 2346.  It is counted as its own\n"
            "migration rather than excluded as a duplicate -- the same call the\n"
            "corpus makes for the gdn v3 pair (42%) and the four megablox v2\n"
            "forwards (23-57%), and the opposite of the call for JAXBench's\n"
            "4p_Sparse_Attention (90%)."
        ),
    },
}

HEADER = '''"""Standalone {display} Splash Attention TPU Pallas implementation.

Source:
  repository: {repository}
  commit: {commit}
  paths:
    {path}/
      splash_attention_mask.py
      splash_attention_mask_info.py
      base.py
      splash_attention_kernel.py
  transformation: repo-local mask, mask-info, base, and kernel modules were
    flattened in dependency order; module qualifiers were removed; Python 3.12
    type-alias statements were made Python 3.11-compatible.

**Two Pallas launch points**, not three.  This lineage fuses the dQ computation
into the dKV kernel -- one ``pallas_call`` returns ``dq_unreduced, dk, dv`` --
where the JAXBench and MaxText ``attention/`` splash kernels launch a separate
backward-dQ kernel.{extra}
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{source_path}",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "splash_mha_hsd",
    "launch_points": 2,
}}

IMPLEMENTATION = "{implementation}"

'''

RUNNER = r'''

# Corpus protocol: the default callable builds a causal, single-device Splash
# kernel.  Keeping mask construction outside the jitted call matches Tokamax.
def build_kernel(
    sequence: int = 256,
    *,
    interpret: bool = False,
    block_q: int = 128,
    block_kv: int = 128,
    block_kv_compute: int | None = None,
):
  mask = CausalMask(shape=(sequence, sequence))
  config = dataclasses.replace(
      SplashConfig.get_default(),
      block_q=block_q,
      block_kv=block_kv,
      block_kv_compute=block_kv_compute or block_kv,
      interpret=interpret,
  )
  return make_splash_mha_single_device(mask, config=config)


kernel = build_kernel


def _main() -> None:
  import argparse
  import time

  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=256)
  parser.add_argument("--heads", type=int, default=1)
  parser.add_argument("--head-dim", type=int, default=128)
  parser.add_argument("--interpret", action="store_true")
  args = parser.parse_args()

  keys = jax.random.split(jax.random.key(42), 3)
  shape = (args.heads, args.sequence, args.head_dim)
  q, k, v = (
      jax.random.normal(key, shape, dtype=jnp.bfloat16) for key in keys
  )
  attention = build_kernel(args.sequence, interpret=args.interpret)
  compiled = jax.jit(attention)
  start = time.perf_counter()
  output = compiled(q, k, v)
  output.block_until_ready()
  elapsed_ms = (time.perf_counter() - start) * 1e3
  print(json.dumps({
      "implementation": IMPLEMENTATION,
      "contract": SOURCE["contract"],
      "shape": list(output.shape),
      "compile_and_run_ms": elapsed_ms,
  }))


if __name__ == "__main__":
  _main()
'''


def flatten(name_: str, source_dir: Path) -> str:
    spec = SOURCES[name_]
    chunks = []
    for name in FILES:
        text = (source_dir / name).read_text()
        text = spec["local_import"].sub("", text)
        chunks.append(f"\n# ---- flattened from {name} ----\n\n{text}\n")
    result = "".join(chunks)
    result = re.sub(
        r"^type SplashCustomReturnType = .*$",
        "SplashCustomReturnType = Any",
        result,
        flags=re.MULTILINE,
    )
    result = re.sub(
        r"^type SplashResidualsType = tuple\[.*?^\]$",
        "SplashResidualsType = Any",
        result,
        flags=re.MULTILINE | re.DOTALL,
    )
    # Strip module qualifiers only inside the flattened code.  Applying this
    # to HEADER would also rewrite the provenance file list ("base.py" -> "py").
    for qualifier in ("mask_lib.", "mask_info_lib.", "base."):
        result = result.replace(qualifier, "")
    for token in ("tokamax._src", "maxtext."):
        if token in result:
            raise ValueError(f"{name_}: unresolved upstream reference {token}")
    if result.count("pl.pallas_call(") != 2:
        raise ValueError(
            f"{name_}: expected 2 pallas_call sites, found "
            f"{result.count('pl.pallas_call(')}"
        )
    header = HEADER.format(
        display=spec["display"], repository=spec["repository"],
        commit=spec["commit"], path=spec["path"], extra=spec["extra"],
        implementation=spec["implementation"],
        source_path=spec["source_path"],
    )
    return header + result + RUNNER


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=sorted(SOURCES))
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(flatten(args.name, args.source_dir))


if __name__ == "__main__":
    main()
