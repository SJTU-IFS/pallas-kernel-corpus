"""Mechanically flatten the MLA (multi-head latent attention) TPU kernels.

Three of the family's implementations are migrated.  Two are already
self-contained; vLLM tpu-inference's v1 kernel reaches for three helpers in the
ragged-paged-attention v3 util module, which are inlined.

Like the other ``flatten_*`` scripts this is a reproducibility aid pinned to the
audited snapshots; the generated file is the runnable artifact.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re


#: `mla/v2/transpose.py` imports `sympy.divisors` to pick a host-side tile size.
#: sympy is not in the pinned dependency set and cannot be added, so the flatten
#: substitutes a trial-division equivalent.  This is a different judgement from
#: the qwix deferral: qwix is a quantization framework whose numerics cannot be
#: reproduced in a few lines, whereas "the sorted divisors of a positive
#: integer" is a definition, is used only for integer tile arithmetic before any
#: kernel runs, and is checked against the brute-force definition by a test.
SYMPY_DIVISORS_REPLACEMENT = '''# ``sympy.divisors`` upstream; see SYMPY_DIVISORS_REPLACEMENT in
# tools/flatten_mla.py for why it is inlined rather than imported.
def divisors(n: int) -> list[int]:
    """Positive divisors of ``n``, ascending."""
    small, large = [], []
    i = 1
    while i * i <= n:
        if n % i == 0:
            small.append(i)
            if i != n // i:
                large.append(n // i)
        i += 1
    return small + large[::-1]


'''


#: True of every implementation whose Q arrives pre-split.  The DeepSeek-V4
#: kernels take one fused ``q`` instead and override this.
DEFAULT_PREAMBLE = """MLA (multi-head latent attention) compresses the KV cache into a shared latent
vector plus a small rotary part, so the kernel takes Q split into a "nope"
(non-positional) and a "pe" (rotary) half and attends against a latent cache::
"""

DSV4_PREAMBLE = """MLA (multi-head latent attention) compresses the KV cache into a shared latent
vector shared by every query head.  Unlike the other implementations here this
one takes a single fused Q -- the rotary part is already folded in -- and reads
an FP8-quantized cache::
"""


SOURCES = {
    "tokamax": {
        "display": "Tokamax",
        "repository": "https://github.com/openxla/tokamax",
        "commit": "927e3f94e8ffe0430cf38bd1423112bb2f69ec66",
        "path": "tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py",
        "kernel_file": "pallas_mosaic_tpu_kernel.py",
        "extract": (),
        "local_imports": (),
        "contract": "mla_ragged_paged_attention",
        "extra": (
            "Self-contained upstream: no repo-local imports at all, so this is\n"
            "    a straight copy with source metadata added."
        ),
    },
    "tpu_inference_dsv4": {
        "display": "vLLM tpu-inference (DeepSeek-V4 sparse)",
        "repository": "https://github.com/vllm-project/tpu-inference",
        "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
        "path": "tpu_inference/kernels/experimental/deepseek_v4/mla.py",
        "kernel_file": "mla.py",
        "extract": (),
        "local_imports": (),
        "contract": "mla_sparse_topk",
        "preamble": DSV4_PREAMBLE,
        "shapes": """    q                 [num_tokens, num_q_heads, head_dim]  ONE fused q
    cache_kv          uint8[total_num_pages, page_size // 4, 4, 640]
                          DSV4 FP8: 448B float8_e4m3fn latent, 128B bfloat16
                          rotary, 7B float8_e8m0fnu block scales, 57B pad
    kv_lens           i32[max_num_seqs]
    kv_lens_to_attend i32[max_num_tokens]     HCA mode; None for CSA
    topk_indices      i32[max_num_tokens, k]  CSA mode; None for HCA
    page_indices      i32[max_num_seqs * pages_per_seq]      flattened
    cu_q_lens         i32[max_num_seqs + 1]
    distribution      i32[3]   decode / prefill / mixed split, as in rpa_v3
    attention_sinks   f32[num_q_heads]
    swa_accumution    f32[num_tokens, num_q_heads, head_dim]  carry-in from the
    swa_l             f32[num_tokens, num_q_heads]            sliding-window
    swa_m             f32[num_tokens, num_q_heads]            kernel next door
    ->                [num_tokens, num_q_heads, head_dim]

There is no ql_nope/q_pe split and no new_kv here: a separate compressor writes
the cache before this kernel runs.
""",
        "extra": (
            "Self-contained upstream: no repo-local imports, so this is a\n"
            "    straight copy with source metadata added.  A DIFFERENT contract\n"
            "    from the migrated `mla_ragged_paged_attention`: it adds\n"
            "    `topk_indices` and `kv_lens_to_attend`, attending only to a\n"
            "    selected subset of KV rather than the whole prefix."
        ),
    },
    "tpu_inference_dsv4_swa": {
        "display": "vLLM tpu-inference (DeepSeek-V4 sliding window)",
        "repository": "https://github.com/vllm-project/tpu-inference",
        "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
        "path": "tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py",
        "kernel_file": "mla_swa.py",
        "extract": (),
        "local_imports": (),
        "entry": "mla_sliding_window_ragged_paged_attention",
        "contract": "mla_sliding_window",
        "preamble": DSV4_PREAMBLE,
        "shapes": """    q                 [num_tokens, num_q_heads, head_dim]  ONE fused q
    new_kv            [num_tokens, head_dim]   passed separately, not pre-written
    cache_kv          uint8[total_num_pages, physical_page_size // 4, 4, 640]
                          DSV4 FP8, written by this kernel; the physical page is
                          longer than the logical one (logical_page_size)
    kv_lens           i32[max_num_seqs]
    page_indices      i32[max_num_seqs * pages_per_seq]      flattened
    cu_q_lens         i32[max_num_seqs + 1]
    distribution      i32[3]   decode / prefill / mixed split, as in rpa_v3
    attention_sinks   f32[num_q_heads]
    ->                (out, cache, L, m)

L and m are the softmax denominator and running max.  They are not diagnostics:
the sparse DeepSeek-V4 kernel next door consumes them, together with `out`, as
its swa_l / swa_m / swa_accumution carry-in.  Upstream chains the two with
`unnormalized_output=True`, whose default here is False.
""",
        "extra": (
            "Self-contained upstream: no repo-local imports, so this is a\n"
            "    straight copy with source metadata added.  Entry point is\n"
            "    `mla_sliding_window_ragged_paged_attention`, which takes\n"
            "    `new_kv` separately from `cache_kv` and attends over a sliding\n"
            "    window -- again a different contract, not a tuning variant."
        ),
    },
    "tpu_inference_v2": {
        "display": "vLLM tpu-inference (v2)",
        "repository": "https://github.com/vllm-project/tpu-inference",
        "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
        "path": "tpu_inference/kernels/mla/v2/kernel.py",
        "kernel_file": "kernel.py",
        "extract": (
            (
                "kv_utils.py",
                "tpu_inference/kernels/mla/v2/kv_utils.py",
                # Only pack_new_kv is called; the seven helpers it shares with
                # kernel.py are byte-identical there, so they are not copied.
                ("pack_new_kv",),
            ),
            (
                "transpose.py",
                "tpu_inference/kernels/mla/v2/transpose.py",
                ("prev_closest_valid_divisor", "xpose_pipeline"),
            ),
        ),
        "extract_imports": {
            "kv_utils.py": "from jax.experimental.pallas import tpu as pltpu\n\n\n",
            # `import jax` is needed at *definition* time, not just at call
            # time: `xpose_pipeline` carries an `@jax.jit(static_argnames=...)`
            # decorator, and the extracted block lands above the flattened
            # kernel that would otherwise supply the import.
            "transpose.py": "import jax\n\n\n" + SYMPY_DIVISORS_REPLACEMENT,
        },
        "local_imports": (
            re.compile(r"^import tpu_inference\.envs as envs\s*$", re.M),
            re.compile(
                r"^from tpu_inference\.kernels\.mla\.v2 import kv_utils\s*$", re.M
            ),
            re.compile(
                r"^from tpu_inference\.kernels\.mla\.v2\.transpose import "
                r"xpose_pipeline\s*$",
                re.M,
            ),
            re.compile(r"^from tpu_inference\.logger import init_logger\s*$", re.M),
        ),
        "substitutions": (
            # One tuning constant read from the environment, inlined with the
            # same name and the same default as tpu_inference/envs.py.
            (
                re.compile(r"envs\.MLA_XPOSE_N_TILE_SIZE"),
                'int(os.getenv("MLA_XPOSE_N_TILE_SIZE", "160"))',
            ),
            (re.compile(r"init_logger\(__name__\)"), "logging.getLogger(__name__)"),
            # The module qualifier disappears with the import it came from.
            (re.compile(r"kv_utils\.pack_new_kv\("), "pack_new_kv("),
            (re.compile(r"^import functools\s*$", re.M), "import functools\nimport logging\nimport os"),
        ),
        "contract": "mla_ragged_paged_attention_head_major",
        "shapes": """    ql_nope       [num_q_heads, num_tokens, lkv_dim]     HEAD-MAJOR, unlike v1
    q_pe          [num_tokens, num_q_heads, r_dim]       token-major, as in v1
    new_kv_c      [num_tokens, lkv_dim]                  latent KV to append
    new_k_pe      [num_tokens, r_dim]                    rotary K to append
    cache_kv      [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    kv_lens       i32[max_num_seqs]
    page_indices  i32[max_num_seqs * pages_per_seq]      flattened
    cu_q_lens     i32[max_num_seqs + 1]
    distribution  i32[3]   decode / prefill / mixed split, as in rpa_v3
    ->            [num_q_heads, num_tokens, lkv_dim], also head-major
""",
        "transformation": (
            "four repo-local imports were resolved. `pack_new_kv` was inlined\n"
            "    from mla/v2/kv_utils.py (the other seven names that module\n"
            "    exports are byte-identical to definitions kernel.py already\n"
            "    has, and were not copied); `xpose_pipeline` and its host-side\n"
            "    helper were inlined from mla/v2/transpose.py; the\n"
            "    `MLA_XPOSE_N_TILE_SIZE` env lookup was inlined with the same\n"
            "    name and default; and the vLLM logger became the stdlib one.\n"
            "    `sympy.divisors`, used only for host-side tile selection, was\n"
            "    replaced by a trial-division equivalent because sympy is not in\n"
            "    the pinned dependency set -- a test checks it against the\n"
            "    brute-force definition. The kernel body is unmodified."
        ),
        "extra": (
            "WARNING: v2 takes the same nine argument NAMES as v1 but a\n"
            "    DIFFERENT `ql_nope` LAYOUT.  v1 wants token-major\n"
            "    [num_tokens, num_q_heads, lkv_dim]; v2 wants head-major\n"
            "    [num_q_heads, num_tokens, lkv_dim], and returns its output\n"
            "    head-major too.  `q_pe` stays token-major in both.  Nothing in\n"
            "    the argument names catches a mixed-up call, which is why v2\n"
            "    physically transposes on entry and exit via `xpose_pipeline` --\n"
            "    an inlined Pallas launch belonging to the `layout_transpose`\n"
            "    family, not counted as an MLA migration."
        ),
    },
    "tpu_inference_v1": {
        "display": "vLLM tpu-inference (v1)",
        "repository": "https://github.com/vllm-project/tpu-inference",
        "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
        "path": "tpu_inference/kernels/mla/v1/kernel.py",
        "kernel_file": "kernel.py",
        "extract": (
            (
                "util.py",
                "tpu_inference/kernels/ragged_paged_attention/v3/util.py",
                # get_dtype_bitwidth is not in the upstream import line, but
                # get_dtype_packing calls it, so it must come along.
                ("align_to", "cdiv", "get_dtype_bitwidth", "get_dtype_packing"),
            ),
        ),
        "local_imports": (
            re.compile(
                r"^from tpu_inference\.kernels\.ragged_paged_attention\.v3\.util import \(\n\s+[^)]*\)\s*$",
                re.M,
            ),
        ),
        "extract_imports": {"util.py": "from jax._src import dtypes\n\n\n"},
        "contract": "mla_ragged_paged_attention",
        "transformation": (
            "the helpers this kernel imports from the "
            "ragged-paged-attention\n    v3 util module were inlined and the "
            "repo-local import removed. The\n    kernel body is unmodified."
        ),
        "extra": (
            "This is the only migrated MLA implementation that also ships a\n"
            "    pure-JAX ``ref_mla_ragged_paged_attention`` with the matching\n"
            "    signature, so it supplies the family's baseline."
        ),
    },
    "sglang_jax_v2": {
        "display": "sglang-jax (v2)",
        "repository": "https://github.com/sgl-project/sglang-jax",
        "commit": "a7353325e8c00d287294c2cd679a77173f1a4594",
        "path": "python/sgl_jax/srt/kernels/mla/v2/kernel.py",
        "kernel_file": "kernel.py",
        # Prepended so the lazily-imported lookup resolves at module scope.
        "prepend_files": ("tuned_block_sizes.py",),
        # The tuned table itself reaches two levels further out for three
        # small helpers; extract exactly those rather than whole modules.
        "extract": (
            (
                "_rpa_util.py",
                "sgl_jax/srt/kernels/ragged_paged_attention/util.py",
                ("get_tpu_version", "next_power_of_2"),
            ),
            (
                "_jax_utils.py",
                "sgl_jax/srt/utils/jax_utils.py",
                ("get_device_name",),
            ),
        ),
        "local_imports": (
            # A lazy import *inside* the block-size lookup, not at module top.
            re.compile(
                r"^\s*from sgl_jax\.srt\.kernels\.mla\.v2\.tuned_block_sizes import \(\n\s+[^)]*\)\s*$",
                re.M,
            ),
        ),
        # Applied to the prepended table as well as the kernel.
        "prepend_local_imports": (
            re.compile(r"^from sgl_jax\.srt\.[\w.]+ import \([^)]*\)\s*$", re.M),
            re.compile(r"^from sgl_jax\.srt\.[\w.]+ import .*$", re.M),
        ),
        "contract": "mla_ragged_paged_attention_cu_kv",
        "transformation": (
            "the tuned block-size table is flattened in (the kernel imports it\n"
            "    lazily inside its lookup), together with the three helpers that\n"
            "    table reaches further out for. The kernel body is unmodified."
        ),
        "extra": (
            "Self-contained upstream. NOTE its entry point takes **ten**\n"
            "    required arguments, adding ``cu_kv_lens`` before\n"
            "    ``distribution``; the other two take nine. It is therefore a\n"
            "    separate contract, exactly as with rpa_v2 vs rpa_v3."
        ),
    },
}

HEADER = '''"""Standalone {display} MLA ragged paged attention.

Source:
  repository: {repository}
  commit: {commit}
  path: {path}
{extracted_list}  transformation: {transformation}
    {extra}

Entry point: ``{entry}`` (also exported as ``kernel``).

{preamble}
{shapes}
Unlike ordinary attention there is a single latent K/V stream shared by all
query heads, which is what makes the cache small.
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{path}",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "{contract}",
}}

'''

#: The nine-argument token-major layout shared by Tokamax, tpu-inference v1 and
#: (plus ``cu_kv_lens``) sglang-jax v2.
DEFAULT_SHAPES = """    ql_nope       [num_tokens, num_q_heads, lkv_dim]     latent query part
    q_pe          [num_tokens, num_q_heads, r_dim]       rotary query part
    new_kv_c      [num_tokens, lkv_dim]                  latent KV to append
    new_k_pe      [num_tokens, r_dim]                    rotary K to append
    cache_kv      [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    kv_lens       i32[max_num_seqs]
    page_indices  i32[max_num_seqs * pages_per_seq]      flattened
    cu_q_lens     i32[max_num_seqs + 1]
    distribution  i32[3]   decode / prefill / mixed split, as in rpa_v3
"""

FOOTER = "\n\nkernel = {entry}\n"


def extract_functions(path: Path, names: tuple[str, ...]) -> str:
    """The named top-level functions, decorators included.

    `ast.get_source_segment` on a FunctionDef starts at `def`, not at the
    decorator above it, so extracting a function this way silently drops its
    decorators.  That is how the inlined copy of `xpose_pipeline` lost its
    `@jax.jit(static_argnames=[...])` while the standalone layout_transpose
    migration of the same function kept it -- caught by the AST comparison in
    tests/test_layout_transpose_tpu.py, not by anything failing.
    """
    text = path.read_text()
    tree = ast.parse(text)
    found = {}
    for node in tree.body:
        if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in names):
            continue
        segment = ast.get_source_segment(text, node)
        decorators = [
            "@" + ast.get_source_segment(text, decorator)
            for decorator in node.decorator_list
        ]
        found[node.name] = "\n".join(decorators + [segment])
    missing = [n for n in names if n not in found]
    if missing:
        raise KeyError(f"{path}: could not extract {missing}")
    return "\n\n".join(found[n] for n in names)


def flatten(name: str, source_dir: Path) -> str:
    spec = SOURCES[name]
    chunks = []
    for filename, upstream, functions in spec["extract"]:
        source = extract_functions(source_dir / filename, functions)
        # Extracting function bodies drops the source module's imports, so
        # carry over the ones those bodies actually need.
        imports = spec.get("extract_imports", {}).get(filename, "")
        chunks.append(
            f"\n# ---- extracted from {upstream}: "
            f"{', '.join(functions)} ----\n\n{imports}{source}\n"
        )

    for filename in spec.get("prepend_files", ()):
        prepended = (source_dir / filename).read_text()
        for pattern in spec.get("prepend_local_imports", ()):
            prepended = pattern.sub("", prepended)
        prepended = re.sub(r"^from __future__ import .*$", "", prepended, flags=re.M)
        chunks.append(f"\n# ---- flattened from {filename} ----\n\n{prepended}\n")

    text = (source_dir / spec["kernel_file"]).read_text()
    for pattern in spec["local_imports"]:
        text, count = pattern.subn("", text)
        if not count:
            raise ValueError(f"{name}: expected local import not found")
    text = re.sub(r"^from __future__ import .*$", "", text, flags=re.M)
    chunks.append(f"\n# ---- flattened from {spec['kernel_file']} ----\n\n{text}\n")

    body = "".join(chunks)
    for pattern, replacement in spec.get("substitutions", ()):
        body, count = pattern.subn(replacement, body)
        if not count:
            raise ValueError(
                f"{name}: substitution {pattern.pattern!r} matched nothing"
            )
    for token in ("tpu_inference.", "sgl_jax.", "tokamax._src"):
        if token in body:
            raise ValueError(f"{name}: unresolved upstream reference {token}")
    entry = spec.get("entry", "mla_ragged_paged_attention")
    if f"def {entry}(" not in body:
        raise ValueError(f"{name}: lost entry point {entry}")

    transformation = spec.get(
        "transformation",
        "copied as a standalone implementation with source metadata added.",
    )
    header = HEADER.format(
        display=spec["display"],
        repository=spec["repository"],
        commit=spec["commit"],
        path=spec["path"],
        extracted_list="".join(
            f"  also inlines: {upstream}  (only: {', '.join(functions)})\n"
            for _, upstream, functions in spec["extract"]
        ),
        transformation=transformation,
        extra=spec["extra"],
        contract=spec["contract"],
        entry=entry,
        preamble=spec.get("preamble", DEFAULT_PREAMBLE),
        shapes=spec.get("shapes", DEFAULT_SHAPES),
    )
    return header + body + FOOTER.format(entry=entry)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=sorted(SOURCES))
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(flatten(args.name, args.source_dir))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
