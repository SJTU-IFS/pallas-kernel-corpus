"""Profiling metadata for each corpus kernel family.

``tools/profile_kernel.py`` owns the measurement protocol -- warmups, traced
iterations, device-event extraction, correctness, cost analysis.  This module
owns everything family-specific: how inputs are generated, which reference a
given implementation is allowed to be compared against, the native upstream
shape, and how FLOPs are counted.

Kernel source files never import this module.  They stay directly runnable on
their own; the registry only points at them.

To add a family, append one ``register`` block.  A case must supply:

  family          output subdirectory under the profile root
  contract        semantic contract name; two cases may only be compared when
                  these match
  build           callable(args) -> Built
  native_shape    callable(args) -> None, applied when --native is passed

``Built.flops`` maps a counting convention to a FLOP count.  Conventions are
deliberately kept separate rather than reconciled -- full-rectangle logical,
causal-useful, and scheduled-block counts are not the same number, and the
report must be able to show more than one.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import functools
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).parents[1]
KERNELS = ROOT / "kernels"

import jax  # noqa: E402  (imported after ROOT so failures name this module)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclasses.dataclass(frozen=True)
class Built:
    """One ready-to-profile implementation and its matching reference."""

    function: Callable[..., Any]
    reference: Callable[..., Any]
    inputs: tuple[Any, ...]
    contract: str
    flops: dict[str, int]
    primary_flops: str | None = "logical"
    # Byte counts for data-movement kernels, where achieved bandwidth is the
    # real metric and MXU utilization is meaningless.  Kept as a map for the
    # same reason ``flops`` is: the bytes the algorithm requires and the bytes
    # the measured configuration actually moves can differ, and collapsing them
    # into one number hides which is being reported.
    bytes_moved: dict[str, int] | None = None
    primary_bytes: str | None = None
    # The shape this case actually ran at.  Declared per case rather than
    # filtered out of the global argument namespace, so a result never records
    # another family's irrelevant defaults or omits its own dimensions.
    shape: dict[str, Any] = dataclasses.field(default_factory=dict)
    # Some upstream references validate traced values with Python control flow
    # and therefore cannot be jitted.  They are still valid correctness
    # references, just not timing denominators.
    jit_reference: bool = True
    # Index of an argument the kernel *donates* (consumes).  Reusing one across
    # iterations raises "Array has been deleted", so the profiler pre-builds a
    # fresh copy per call and swaps it in outside the timed region.
    donated_argnum: int | None = None
    config: dict[str, Any] = dataclasses.field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.primary_flops is None:
            if self.flops:
                raise ValueError("primary_flops=None requires an empty flops map")
            if not self.bytes_moved:
                raise ValueError(
                    "a case with no FLOPs must report bytes_moved, or it has no "
                    "metric at all"
                )
            if self.primary_bytes not in self.bytes_moved:
                raise KeyError(
                    f"primary_bytes={self.primary_bytes!r} not in "
                    f"{sorted(self.bytes_moved)}"
                )
        elif self.primary_flops not in self.flops:
            raise KeyError(
                f"primary_flops={self.primary_flops!r} not in {sorted(self.flops)}"
            )


@dataclasses.dataclass(frozen=True)
class Case:
    family: str
    contract: str
    build: Callable[[Any], Built]
    native_shape: Callable[[Any], None] | None = None
    is_reference: bool = False


REGISTRY: dict[str, Case] = {}


def register(name: str, case: Case) -> None:
    if name in REGISTRY:
        raise KeyError(f"duplicate implementation name: {name}")
    REGISTRY[name] = case


# ---------------------------------------------------------------------------
# attention / flash_attention
# ---------------------------------------------------------------------------

FLASH = KERNELS / "attention" / "flash_attention"


def _flash_baseline():
    return load_module("flash_attention_baseline", FLASH / "baseline.py")


def _causal_inputs(args):
    keys = jax.random.split(jax.random.key(42), 3)
    shape = (args.batch, args.heads, args.sequence, args.head_dim)
    return tuple(jax.random.normal(key, shape, dtype=jnp.bfloat16) for key in keys)


def _build_causal(args) -> Built:
    baseline = _flash_baseline()
    inputs = _causal_inputs(args)
    reference = baseline.causal_bhsd
    if args.implementation == "baseline-causal":
        function = reference
    else:
        module = load_module("flash_attention_jaxbench", FLASH / "jaxbench_optimized.py")
        function = module.workload if args.native else module.kernel
    rectangle = (
        4 * args.batch * args.heads * args.sequence * args.sequence * args.head_dim
    )
    return Built(
        function=function,
        reference=reference,
        inputs=inputs,
        contract="causal_bhsd",
        flops={"logical": rectangle, "causal_useful": rectangle // 2},
        shape={
            "batch": args.batch, "heads": args.heads,
            "sequence": args.sequence, "head_dim": args.head_dim,
        },
        notes=(
            "causal_useful halves the full rectangle; diagonal blocks still "
            "compute masked entries, so the true useful count is slightly higher",
        ),
    )


def _native_causal(args) -> None:
    args.batch, args.heads, args.sequence = 4, 64, 4096
    args.head_dim = args.value_head_dim = 128


def _build_dense_2d(args) -> Built:
    baseline = _flash_baseline()
    keys = jax.random.split(jax.random.key(42), 3)
    shape = (args.sequence, args.head_dim)
    inputs = tuple(jax.random.normal(key, shape, dtype=jnp.bfloat16) for key in keys)
    reference = baseline.dense_2d
    if args.implementation == "baseline-dense-2d":
        function = reference
    else:
        module = load_module(
            "flash_attention_pallasbench", FLASH / "pallasbench_optimized.py"
        )
        function = module.kernel
    return Built(
        function=function,
        reference=reference,
        inputs=inputs,
        contract="dense_2d",
        flops={"logical": 4 * args.sequence * args.sequence * args.head_dim},
        shape={"sequence": args.sequence, "head_dim": args.head_dim},
    )


def _native_dense_2d(args) -> None:
    args.batch, args.heads, args.sequence = 1, 1, 512
    args.head_dim = args.value_head_dim = 64


def _build_splash(args) -> Built:
    baseline = _flash_baseline()
    keys = jax.random.split(jax.random.key(42), 3)
    qk_shape = (args.batch, args.heads, args.sequence, args.head_dim)
    value_shape = (args.batch, args.heads, args.sequence, args.value_head_dim)
    raw = (
        jax.random.normal(keys[0], qk_shape, dtype=jnp.bfloat16),
        jax.random.normal(keys[1], qk_shape, dtype=jnp.bfloat16),
        jax.random.normal(keys[2], value_shape, dtype=jnp.bfloat16),
    )
    # Tokamax's public Splash contract expects Q to be pre-scaled.
    inputs = (raw[0] / math.sqrt(args.head_dim), *raw[1:])
    reference = (
        baseline.splash_mha_bhsd_blockwise
        if args.native
        else jax.vmap(baseline.splash_mha_hsd, in_axes=(0, 0, 0))
    )
    config: dict[str, Any] = {}
    if args.implementation == "baseline-splash":
        function = reference
    else:
        module = load_module("flash_attention_tokamax", FLASH / "tokamax_optimized.py")
        single = module.build_kernel(
            args.sequence,
            block_q=args.tokamax_block_q,
            block_kv=args.tokamax_block_kv,
            block_kv_compute=args.tokamax_block_kv_compute,
        )
        function = jax.vmap(single, in_axes=(0, 0, 0))
        config = {
            "block_q": args.tokamax_block_q,
            "block_kv": args.tokamax_block_kv,
            "block_kv_compute": args.tokamax_block_kv_compute,
        }
    rectangle = (
        2
        * args.batch
        * args.heads
        * args.sequence
        * args.sequence
        * (args.head_dim + args.value_head_dim)
    )
    flops = {"logical": rectangle}
    if config:
        # Tokamax schedules only the lower-triangular block pairs, so count the
        # blocks the dynamic causal grid actually visits.
        block_q = args.tokamax_block_q
        block_kv = args.tokamax_block_kv
        q_blocks = args.sequence // block_q
        kv_blocks = args.sequence // block_kv
        scheduled = sum(
            min(kv_blocks, i + 1) for i in range(q_blocks)
        )  # lower-triangular block pairs per head
        flops["scheduled_block"] = int(
            2
            * args.batch
            * args.heads
            * scheduled
            * block_q
            * block_kv
            * (args.head_dim + args.value_head_dim)
        )
    return Built(
        function=function,
        reference=reference,
        inputs=inputs,
        contract="splash_mha_hsd",
        flops=flops,
        shape={
            "batch": args.batch, "heads": args.heads,
            "sequence": args.sequence, "head_dim": args.head_dim,
            "value_head_dim": args.value_head_dim,
        },
        config=config,
        notes=(
            "scheduled_block counts the block pairs the dynamic causal grid "
            "visits, which is less than the full rectangle and more than the "
            "useful triangle",
        ),
    )


def _native_splash(args) -> None:
    args.batch, args.heads, args.sequence = 8, 128, 4096
    args.head_dim, args.value_head_dim = 192, 128


for _name, _build, _native, _reference in (
    ("baseline-causal", _build_causal, _native_causal, True),
    ("jaxbench", _build_causal, _native_causal, False),
    ("baseline-dense-2d", _build_dense_2d, _native_dense_2d, True),
    ("pallasbench", _build_dense_2d, _native_dense_2d, False),
    ("baseline-splash", _build_splash, _native_splash, True),
    ("tokamax", _build_splash, _native_splash, False),
):
    register(
        _name,
        Case(
            family="flash_attention",
            contract={
                _build_causal: "causal_bhsd",
                _build_dense_2d: "dense_2d",
                _build_splash: "splash_mha_hsd",
            }[_build],
            build=_build,
            native_shape=_native,
            is_reference=_reference,
        ),
    )


# ---------------------------------------------------------------------------
# moe / grouped_matmul
# ---------------------------------------------------------------------------

GROUPED_MATMUL = KERNELS / "moe" / "grouped_matmul"

_GMM_MODULES = {
    "gmm-jaxbench": ("grouped_matmul_jaxbench", "jaxbench_optimized.py"),
    "gmm-tpu-inference": ("grouped_matmul_tpu_inference", "tpu_inference_optimized.py"),
    "gmm-sglang-jax": ("grouped_matmul_sglang_jax", "sglang_jax_optimized.py"),
}
# The Megablox v2 kernels, which unlike the v1 ones do not depend on qwix.
_GMM_V2_MODULES = {
    "gmm-maxtext-v2": ("grouped_matmul_maxtext_v2", "maxtext_optimized.py"),
    "gmm-tokamax-v2": ("grouped_matmul_tokamax_v2", "tokamax_optimized.py"),
}


def _build_grouped_matmul(args) -> Built:
    baseline = load_module("grouped_matmul_baseline", GROUPED_MATMUL / "baseline.py")
    inputs = baseline.create_inputs(
        rows=args.rows,
        num_groups=args.groups,
        k=args.k,
        n=args.n,
        dtype=jnp.bfloat16,
        balanced=not args.unbalanced,
    )
    # jax.lax.ragged_dot is NOT the denominator: on TPU it lowers to Mosaic
    # tpu_custom_calls, so it is the Megablox kernel shipped inside JAX.  The
    # pure-XLA denominator is the batched dense matmul, which is exact and does
    # exactly the logical FLOPs whenever routing is balanced.
    if args.unbalanced:
        reference = baseline.grouped_matmul_loop
    else:
        # Verify uniformity eagerly: inside jit, group_sizes is a tracer and
        # the batched-dense baseline cannot check itself.
        baseline.check_uniform_groups(
            inputs[2], rows=args.rows, num_groups=args.groups
        )
        reference = baseline.grouped_matmul_batched_dense
    config: dict[str, Any] = {
        "balanced_groups": not args.unbalanced,
        "baseline_is_pure_xla": True,
    }

    if args.implementation == "gmm-baseline-batched-dense":
        function = baseline.grouped_matmul_batched_dense
    elif args.implementation == "gmm-baseline-loop":
        function = baseline.grouped_matmul_loop
    elif args.implementation == "gmm-xla-ragged-dot":
        # An implementation, not a baseline: JAX's own Mosaic grouped matmul.
        function = baseline.grouped_matmul_ragged_dot
        config["backend"] = "mosaic-tpu via jax.lax.ragged_dot"
    elif args.implementation in _GMM_V2_MODULES:
        module_name, filename = _GMM_V2_MODULES[args.implementation]
        module = load_module(module_name, GROUPED_MATMUL / filename)
        # v2 changed the default preferred_element_type from v1's float32 to
        # the input dtype. Left at the default, accumulation rounds to bf16 per
        # k-tile and the result drifts below the 0.9999 threshold at k=4096.
        # Pin it to float32 so v2 is measured on the same contract as v1.
        function = functools.partial(
            module.gmm_v2, preferred_element_type=jnp.float32
        )
        config["preferred_element_type"] = "float32"
        config["variant"] = "megablox v2 (no qwix)"
    else:
        module_name, filename = _GMM_MODULES[args.implementation]
        module = load_module(module_name, GROUPED_MATMUL / filename)
        if args.implementation == "gmm-jaxbench" and args.native:
            function = module.workload  # upstream-tuned tiling
            config["tiling"] = tuple(module.TUNED_PARAMS["tiling"])
        elif args.implementation == "gmm-jaxbench":
            function = module.kernel
        else:
            # tpu-inference and sglang-jax select tiling from their own tuned
            # tables, falling back to a heuristic; the profiler records which.
            function = module.kernel
            config["tiling"] = module.get_tuned_block_sizes(
                args.rows, args.k, args.n, args.groups, args.groups,
                "bfloat16", "bfloat16", args.k,
            )
    # One multiply-accumulate per output element per contraction step. Every
    # row belongs to exactly one group, so there is no ragged discount here:
    # the logical count equals a dense [m,k]@[k,n] matmul.
    logical = 2 * args.rows * args.k * args.n
    flops = {"logical": logical}
    if args.implementation == "gmm-baseline-loop":
        # The loop multiplies every row by every group's weights and masks the
        # result, so it executes num_groups times the necessary work.
        flops["executed_dense"] = logical * args.groups
    return Built(
        function=function,
        reference=reference,
        inputs=inputs,
        contract="grouped_matmul_2d",
        flops=flops,
        shape={
            "rows": args.rows, "groups": args.groups,
            "k": args.k, "n": args.n,
        },
        config=config,
        notes=(
            "sum(group_sizes) == rows, so every row is covered exactly once "
            "and the logical FLOP count matches a dense matmul of the same "
            "[m,k]@[k,n] shape",
            "tile padding means the kernels may schedule more than the logical "
            "count when group sizes are not multiples of the m-tile",
        ),
    )


def _native_grouped_matmul(args) -> None:
    """JAXBench 11p_Megablox_GMM CONFIG: Qwen3-235B-A22B MoE dimensions."""
    args.groups = 128  # num_experts
    args.k = 4096  # emb_dim
    args.n = 1536  # moe_mlp_dim
    args.rows = 4096 * 8  # seq_len * num_experts_per_tok


for _name in (
    "gmm-baseline-batched-dense",
    "gmm-baseline-loop",
    "gmm-xla-ragged-dot",
    *_GMM_MODULES,
    *_GMM_V2_MODULES,
):
    register(
        _name,
        Case(
            family="grouped_matmul",
            contract="grouped_matmul_2d",
            build=_build_grouped_matmul,
            native_shape=_native_grouped_matmul,
            is_reference=_name.startswith("gmm-baseline"),
        ),
    )


# ---------------------------------------------------------------------------
# attention / ragged_paged_attention
# ---------------------------------------------------------------------------

RPA = KERNELS / "attention" / "ragged_paged_attention"

_RPA_MODULES = {
    "rpa-jaxbench": ("rpa_jaxbench", "jaxbench_optimized.py"),
    "rpa-tpu-inference": ("rpa_tpu_inference", "tpu_inference_optimized.py"),
}


def _build_rpa_v2(args) -> Built:
    baseline = load_module("rpa_baseline", RPA / "baseline.py")
    inputs = baseline.create_inputs(
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        page_size=args.page_size,
        pages_per_seq=args.pages_per_seq,
        dtype=jnp.bfloat16,
    )
    reference = baseline.rpa_v2
    config: dict[str, Any] = {}
    if args.implementation == "rpa-baseline":
        function = reference
    else:
        module_name, filename = _RPA_MODULES[args.implementation]
        module = load_module(module_name, RPA / filename)
        if not args.native or args.rpa_auto_blocks:
            # Each kernel picks its own blocks via the JAX tuned-size table.
            function = module.kernel
            config = {"block_selection": "kernel-selected"}
        else:
            # Both implementations expose the same block-size API, so give them
            # the same tuning. Otherwise the comparison measures JAXBench's
            # autotuning against tpu-inference's table lookup, not the kernels.
            jaxbench = load_module("rpa_jaxbench_params", RPA / "jaxbench_optimized.py")
            tuned = dict(jaxbench.TUNED_PARAMS)
            scale = 1.0 / math.sqrt(args.head_dim)
            function = functools.partial(
                module.ragged_paged_attention, sm_scale=scale, **tuned
            )
            config = {"block_selection": "jaxbench-autotuned", **tuned}

    tokens_per_seq = args.max_num_batched_tokens // args.max_num_seqs
    kv_len = args.pages_per_seq * args.page_size
    rectangle = baseline.logical_flops(
        max_num_seqs=args.max_num_seqs,
        num_q_heads=args.num_q_heads,
        head_dim=args.head_dim,
        tokens_per_seq=tokens_per_seq,
        kv_len=kv_len,
    )
    # The mask is right-aligned causal: query row r sees kv positions
    # [0, kv_len - tokens_per_seq + r], so almost the whole rectangle is real
    # work at serving shapes where tokens_per_seq << kv_len.
    unmasked = sum(
        min(kv_len, kv_len - tokens_per_seq + r + 1) for r in range(tokens_per_seq)
    )
    useful = int(
        rectangle * unmasked / (tokens_per_seq * kv_len)
    )
    return Built(
        function=function,
        reference=reference,
        inputs=inputs,
        contract="rpa_v2",
        flops={"logical": rectangle, "causal_useful": useful},
        shape={
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "num_q_heads": args.num_q_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "page_size": args.page_size,
            "pages_per_seq": args.pages_per_seq,
        },
        config=config,
        notes=(
            "logical is the full q_len x kv_len rectangle, matching JAXBench's "
            "get_flops(); causal_useful subtracts the right-aligned causal "
            f"mask, which removes only {100 * (1 - useful / rectangle):.2f}% "
            "of the work at this shape",
            "all sequences share one kv_len here, so no ragged padding is "
            "discounted from either count",
        ),
    )


def _native_rpa_v2(args) -> None:
    """JAXBench 7p_Ragged_Paged_Attention CONFIG: Llama-3.1-70B serving."""
    args.max_num_batched_tokens = 4096
    args.max_num_seqs = 64
    args.num_q_heads = 64
    args.num_kv_heads = 8
    args.head_dim = 128
    args.page_size = 16
    args.pages_per_seq = 256


for _name in ("rpa-baseline", *_RPA_MODULES):
    register(
        _name,
        Case(
            family="ragged_paged_attention",
            contract="rpa_v2",
            build=_build_rpa_v2,
            native_shape=_native_rpa_v2,
            is_reference=_name == "rpa-baseline",
        ),
    )


# ---------------------------------------------------------------------------
# memory / kv_cache_update
# ---------------------------------------------------------------------------

KV_CACHE_UPDATE = KERNELS / "memory" / "kv_cache_update"

_KV_MODULES = {
    "kv-tpu-inference": ("kv_tpu_inference", "tpu_inference_optimized.py"),
    "kv-sglang-jax": ("kv_sglang_jax", "sglang_jax_optimized.py"),
}


def _build_kv_cache_update(args) -> Built:
    baseline = load_module("kv_cache_update_baseline", KV_CACHE_UPDATE / "baseline.py")
    new_kv, slices, kv_cache, num_slices, page_size = baseline.create_inputs(
        total_num_tokens=args.kv_tokens,
        num_combined_kv_heads=args.kv_combined_heads,
        head_dim=args.head_dim,
        total_num_pages=args.kv_pages,
        page_size=args.kv_page_size,
        num_slices=args.kv_slices,
        dtype=jnp.bfloat16,
    )
    inputs = (new_kv, slices, kv_cache, num_slices)

    reference = functools.partial(
        baseline.kv_cache_update, max_slice_len=page_size
    )
    reference.__name__ = "kv_cache_update"
    config: dict[str, Any] = {
        "page_size": page_size,
        "num_slices_per_block": args.kv_slices_per_block,
    }

    if args.implementation == "kv-baseline":
        function = reference
    else:
        module_name, filename = _KV_MODULES[args.implementation]
        module = load_module(module_name, KV_CACHE_UPDATE / filename)
        if args.implementation == "kv-sglang-jax":
            # sglang-jax always wraps its kernel in jax.shard_map, so it needs
            # an active mesh even on a single device.  set_mesh cannot be
            # entered inside jax.jit, and the profiler jits what it is handed,
            # so activate the mesh here and leave it active: the profiler runs
            # one implementation per process.
            mesh = jax.sharding.Mesh(
                np.asarray(jax.devices()[:1]).reshape(1), ("tensor",)
            )
            jax.sharding.set_mesh(mesh).__enter__()
            config["mesh"] = "1-device shard_map over 'tensor'"
            function = functools.partial(
                module.kv_cache_update,
                page_size=page_size,
                num_slices_per_block=args.kv_slices_per_block,
                kv_partition_axis="tensor",
            )
        else:
            function = functools.partial(
                module.kv_cache_update,
                page_size=page_size,
                num_slices_per_block=args.kv_slices_per_block,
            )

    element_bytes = jnp.dtype(jnp.bfloat16).itemsize
    per_token = args.kv_combined_heads * args.head_dim * element_bytes
    payload = int(num_slices[0]) * page_size * per_token
    cache_bytes = args.kv_pages * page_size * per_token
    return Built(
        function=function,
        reference=reference,
        inputs=inputs,
        contract="kv_cache_update",
        flops={},
        primary_flops=None,
        bytes_moved={
            # What the algorithm requires: read each live slice once, write
            # it once.  This is the cost in a serving loop, where the cache
            # buffer is genuinely donated and updated in place.
            "slice_payload": 2 * payload,
            # What this harness actually moves.  The profiler reuses its
            # inputs across iterations, so donation of kv_cache fails and XLA
            # copies the whole cache every call: read it, write it, plus the
            # payload read.  Measured device time scales linearly with cache
            # size at fixed payload, confirming this dominates.
            "harness_cache_copy": 2 * cache_bytes + payload,
        },
        primary_bytes="harness_cache_copy",
        shape={
            "total_num_tokens": args.kv_tokens,
            "num_combined_kv_heads": args.kv_combined_heads,
            "head_dim": args.head_dim,
            "total_num_pages": args.kv_pages,
            "page_size": page_size,
            "num_slices": int(num_slices[0]),
            "padded_num_slices": int(slices.shape[1]),
        },
        config=config,
        notes=(
            "this kernel performs no arithmetic, so MXU utilization is not "
            "reported; the metric is achieved HBM bandwidth",
            "the primary byte count is harness_cache_copy, not slice_payload: "
            "because the profiler reuses inputs across iterations, kv_cache "
            "donation fails and the whole cache is copied each call",
            "slice_payload is what a serving loop would move, where donation "
            "succeeds and the cache is updated in place; dividing it by this "
            "device time understates achieved bandwidth by cache_size/payload",
        ),
    )


for _name in ("kv-baseline", *_KV_MODULES):
    register(
        _name,
        Case(
            family="kv_cache_update",
            contract="kv_cache_update",
            build=_build_kv_cache_update,
            # No upstream benchmark defines a native shape for this kernel, so
            # --native is deliberately unavailable: every run is a declared
            # validation shape and says so in result.json.
            native_shape=None,
            is_reference=_name == "kv-baseline",
        ),
    )


# ---------------------------------------------------------------------------
# quantization / quantized_matmul
# ---------------------------------------------------------------------------

QUANTIZED_MATMUL = KERNELS / "quantization" / "quantized_matmul"

_QM_MODULES = {
    "qm-tpu-inference": ("qm_tpu_inference", "tpu_inference_optimized.py"),
    "qm-sglang-jax": ("qm_sglang_jax", "sglang_jax_optimized.py"),
}


def _build_quantized_matmul(args) -> Built:
    baseline = load_module("quantized_matmul_baseline", QUANTIZED_MATMUL / "baseline.py")
    x_q_dtype = jnp.int8 if args.qm_quantize_activation else None
    inputs = baseline.create_inputs(
        n_batch=args.qm_batch, n_in=args.qm_in, n_out=args.qm_out
    )
    # The exact fp32 reference is the correctness check; the XLA dequantize-
    # and-matmul path is the fair speed denominator.  Comparing against the
    # fp32 one would flatter every kernel by an upcast nobody would ship.
    reference = functools.partial(
        baseline.quantized_matmul_per_channel_xla, x_q_dtype=x_q_dtype
    )
    reference.__name__ = "quantized_matmul_per_channel_xla"
    config: dict[str, Any] = {
        "quantize_activation": args.qm_quantize_activation,
        "w_q_dtype": "int8",
    }

    if args.implementation == "qm-baseline-xla":
        function = reference
    elif args.implementation == "qm-baseline-fp32":
        function = functools.partial(
            baseline.quantized_matmul_per_channel, x_q_dtype=x_q_dtype
        )
    else:
        module_name, filename = _QM_MODULES[args.implementation]
        module = load_module(module_name, QUANTIZED_MATMUL / filename)
        function = functools.partial(
            module.quantized_matmul_kernel, x_q_dtype=x_q_dtype
        )
        # Record which tiling the kernel's own table selects. The table has no
        # bfloat16 activation entries at all, so the weight-only-quantized path
        # always misses it and falls back; capturing this makes that visible in
        # the result rather than leaving it as an unexplained slowdown.
        selected = module.get_tuned_block_sizes(
            n_batch=args.qm_batch,
            n_out=args.qm_out,
            n_in=args.qm_in,
            x_q_dtype=jnp.dtype(x_q_dtype or inputs[0].dtype).name,
            w_q_dtype=jnp.dtype(inputs[1].dtype).name,
        )
        config["tuned_value"] = list(selected) if selected is not None else None
        config["block_selection"] = "kernel tuned-size table"

    return Built(
        function=function,
        reference=reference,
        inputs=inputs,
        contract="quantized_matmul_per_channel",
        flops={"logical": 2 * args.qm_batch * args.qm_in * args.qm_out},
        shape={
            "n_batch": args.qm_batch,
            "n_in": args.qm_in,
            "n_out": args.qm_out,
        },
        config=config,
        notes=(
            "MXU utilization is computed against the 918 TFLOP/s *bf16* peak "
            "for corpus consistency; with int8 weights the hardware peak is "
            "higher, so this understates true utilization and should not be "
            "read as an int8 efficiency figure",
            "the correctness reference is the XLA dequantize-and-matmul path, "
            "not the exact fp32 one, so the reported error includes both "
            "kernels' quantization choices rather than only rounding",
        ),
    )


for _name in ("qm-baseline-xla", "qm-baseline-fp32", *_QM_MODULES):
    register(
        _name,
        Case(
            family="quantized_matmul",
            contract="quantized_matmul_per_channel",
            build=_build_quantized_matmul,
            # No upstream benchmark defines a native shape.  The default here
            # is a shape the upstream tuned-size table covers for v6e, so each
            # kernel gets its intended tiling rather than a fallback.
            native_shape=None,
            is_reference=_name.startswith("qm-baseline"),
        ),
    )


# ---------------------------------------------------------------------------
# sampling / topk_routing
# ---------------------------------------------------------------------------

TOPK_ROUTING = KERNELS / "sampling" / "topk_routing"

_TOPK_CASES = {
    "topk-baseline-plain": ("router_topk", None),
    "topk-baseline-biased": ("router_biased_topk", None),
    "topk-baseline-grouped": ("router_grouped_topk", None),
    "topk-sglang-plain": ("router_topk", "sglang_jax_optimized.py"),
    "topk-sglang-biased": ("router_biased_topk", "sglang_jax_optimized.py"),
    "topk-sglang-grouped": ("router_grouped_topk", "sglang_jax_grouped_optimized.py"),
}


def _build_topk_routing(args) -> Built:
    baseline = load_module("topk_routing_baseline", TOPK_ROUTING / "baseline.py")
    contract, filename = _TOPK_CASES[args.implementation]
    router_logits, correction_bias = baseline.create_inputs(
        batch=args.topk_batch, num_experts=args.topk_experts
    )

    if contract == "router_topk":
        inputs = (router_logits,)
        reference = functools.partial(baseline.router_topk, topk=args.topk_k)
        pallas_name = "topk_pallas"
        pallas_kwargs = {"topk": args.topk_k}
    elif contract == "router_biased_topk":
        inputs = (router_logits, correction_bias)
        reference = functools.partial(baseline.router_biased_topk, topk=args.topk_k)
        pallas_name = "biased_topk_pallas"
        pallas_kwargs = {"topk": args.topk_k}
    else:
        inputs = (router_logits, correction_bias)
        reference = functools.partial(
            baseline.router_grouped_topk,
            num_expert_group=args.topk_expert_groups,
            topk_group=args.topk_group,
            topk=args.topk_k,
        )
        pallas_name = "grouped_topk_pallas"
        pallas_kwargs = {
            "num_expert_group": args.topk_expert_groups,
            "topk_group": args.topk_group,
            "topk": args.topk_k,
        }
    reference.__name__ = contract

    config: dict[str, Any] = {"topk": args.topk_k, **pallas_kwargs}
    if filename is None:
        function = reference
    else:
        module = load_module(f"topk_{pallas_name}", TOPK_ROUTING / filename)
        function = functools.partial(getattr(module, pallas_name), **pallas_kwargs)

    # Top-k is selection, not arithmetic: there is no meaningful FLOP count, so
    # report the bytes the kernel must read and write instead.
    read = args.topk_batch * args.topk_experts * 4  # f32 router logits
    write = args.topk_batch * args.topk_k * (4 + 4)  # f32 weights + i32 ids
    return Built(
        function=function,
        reference=reference,
        inputs=inputs,
        contract=contract,
        flops={},
        primary_flops=None,
        bytes_moved={"logits_and_selection": read + write},
        primary_bytes="logits_and_selection",
        shape={
            "batch": args.topk_batch,
            "num_experts": args.topk_experts,
            "topk": args.topk_k,
        },
        config=config,
        notes=(
            "top-k is a selection kernel, so MXU utilization is not reported; "
            "the metric is achieved bandwidth over the router logits it must "
            "read and the selection it writes",
            "correctness is exact: expert ids must match bitwise and selected "
            "weights to 0.0, not merely to a cosine threshold",
            "num_experts must be a multiple of 128 -- the kernels reject "
            "anything else",
        ),
    )


for _name in _TOPK_CASES:
    register(
        _name,
        Case(
            family="topk_routing",
            contract=_TOPK_CASES[_name][0],
            build=_build_topk_routing,
            native_shape=None,  # no upstream benchmark defines one
            is_reference=_name.startswith("topk-baseline"),
        ),
    )


# ---------------------------------------------------------------------------
# attention / mla_attention
# ---------------------------------------------------------------------------

MLA = KERNELS / "attention" / "mla_attention"

_MLA_MODULES = {
    "mla-tokamax": ("mla_tokamax", "tokamax_optimized.py"),
    "mla-tpu-inference": ("mla_tpu_inference", "tpu_inference_optimized.py"),
    "mla-tpu-inference-v2": ("mla_tpu_inference_v2", "tpu_inference_v2_optimized.py"),
    "mla-sglang-jax": ("mla_sglang_jax", "sglang_jax_optimized.py"),
}

#: v2 takes the same nine argument names as v1 with ``ql_nope`` head-major, and
#: returns head-major.  Timing the kernel means timing that layout, so the
#: transpose is applied to the input rather than wrapped around the call.
_MLA_HEAD_MAJOR = ("mla-tpu-inference-v2",)


def _mla_inputs(args):
    """Build MLA inputs; the cache holds latent and rotary parts concatenated."""
    tokens = args.mla_seqs * args.mla_q_len
    pages_per_seq = args.mla_kv_len // args.mla_page_size
    total_pages = args.mla_seqs * pages_per_seq
    packing = 2  # bf16
    kv_dim = (
        (args.mla_lkv + 127) // 128 * 128 + (args.mla_r + 127) // 128 * 128
    )
    keys = jax.random.split(jax.random.key(11), 5)
    kv_lens = jnp.full((args.mla_seqs,), args.mla_kv_len, jnp.int32)
    return dict(
        ql_nope=jax.random.normal(
            keys[0], (tokens, args.mla_q_heads, args.mla_lkv), jnp.bfloat16
        ),
        q_pe=jax.random.normal(
            keys[1], (tokens, args.mla_q_heads, args.mla_r), jnp.bfloat16
        ),
        new_kv_c=jax.random.normal(keys[2], (tokens, args.mla_lkv), jnp.bfloat16),
        new_k_pe=jax.random.normal(keys[3], (tokens, args.mla_r), jnp.bfloat16),
        cache_kv=jax.random.normal(
            keys[4],
            (total_pages, args.mla_page_size // packing, packing, kv_dim),
            jnp.bfloat16,
        ),
        kv_lens=kv_lens,
        page_indices=jnp.arange(total_pages, dtype=jnp.int32),
        cu_q_lens=jnp.arange(args.mla_seqs + 1, dtype=jnp.int32) * args.mla_q_len,
        cu_kv_lens=jnp.concatenate(
            [jnp.zeros((1,), jnp.int32), jnp.cumsum(kv_lens)]
        ),
        distribution=jnp.array([0, 0, args.mla_seqs], jnp.int32),
    )


_MLA_NINE = (
    "ql_nope", "q_pe", "new_kv_c", "new_k_pe", "cache_kv",
    "kv_lens", "page_indices", "cu_q_lens", "distribution",
)


def _mla_contract(implementation: str) -> str:
    if implementation == "mla-sglang-jax":
        return "mla_ragged_paged_attention_cu_kv"
    if implementation in _MLA_HEAD_MAJOR:
        return "mla_ragged_paged_attention_head_major"
    return "mla_ragged_paged_attention"


def _build_mla(args) -> Built:
    reference_module = load_module(
        "mla_reference", MLA / "tpu_inference_optimized.py"
    )
    built = _mla_inputs(args)
    scale = 1.0 / math.sqrt(args.mla_lkv + args.mla_r)

    # Only tpu-inference ships a pure-JAX reference with the matching
    # signature, so it is the family baseline; for the other two this is a
    # genuine cross-repository check.
    reference = functools.partial(
        reference_module.ref_mla_ragged_paged_attention, sm_scale=scale
    )
    reference.__name__ = "ref_mla_ragged_paged_attention"

    blocks = {
        "num_kv_pages_per_block": args.mla_kv_pages_per_block,
        "num_queries_per_block": args.mla_queries_per_block,
    }
    config: dict[str, Any] = dict(blocks)

    module_name, filename = _MLA_MODULES[args.implementation]
    module = load_module(module_name, MLA / filename)
    kwargs = {"sm_scale": scale, **blocks}
    if args.implementation == "mla-tokamax":
        kwargs["vmem_limit_bytes"] = 64 * 1024 * 1024
    if args.implementation == "mla-sglang-jax":
        # sglang-jax takes ten required args, adding cu_kv_lens.
        order = (*_MLA_NINE[:8], "cu_kv_lens", "distribution")
        config["contract_note"] = "takes cu_kv_lens (10 required args)"
    else:
        order = _MLA_NINE
    inputs = tuple(built[k] for k in order)
    if args.implementation in _MLA_HEAD_MAJOR:
        # ql_nope is [num_q_heads, num_tokens, lkv_dim] here, not token-major.
        inputs = (jnp.transpose(inputs[0], (1, 0, 2)), *inputs[1:])
        config["contract_note"] = "ql_nope and output are head-major"
        # v2 ships no default block sizes and needs an explicit vmem limit.
        kwargs["vmem_limit_bytes"] = 100 * 1024 * 1024
        kwargs["s_dtype"] = jnp.float32
        kwargs["decode_batch_size"] = 1
    function = functools.partial(module.mla_ragged_paged_attention, **kwargs)

    # The reference always takes the nine-argument form.
    head_major = args.implementation in _MLA_HEAD_MAJOR

    def reference_on_nine(*call_args, _built=built, _ref=reference):
        out = _ref(*[_built[k] for k in _MLA_NINE])
        if not head_major:
            return out
        # Match v2's head-major output by transposing the REFERENCE, so the
        # timed function stays the kernel alone rather than kernel + transpose.
        first, *rest = out if isinstance(out, tuple) else (out,)
        first = jnp.transpose(first, (1, 0, 2))
        return (first, *rest) if rest else first

    reference_on_nine.__name__ = "ref_mla_ragged_paged_attention"

    # MLA contracts against a shared latent stream: QK is over lkv + r, and PV
    # is over lkv only, for every query head.
    tokens = args.mla_seqs * args.mla_q_len
    qk = 2 * tokens * args.mla_kv_len * args.mla_q_heads * (args.mla_lkv + args.mla_r)
    pv = 2 * tokens * args.mla_kv_len * args.mla_q_heads * args.mla_lkv
    return Built(
        function=function,
        reference=reference_on_nine,
        inputs=inputs,
        contract=_mla_contract(args.implementation),
        flops={"logical": qk + pv},
        # The upstream reference validates traced values with Python `if`, so
        # it cannot be jitted; it is a correctness reference only.
        jit_reference=False,
        # cache_kv is donated (@jax.jit(donate_argnames="cache_kv")).
        donated_argnum=order.index("cache_kv"),
        shape={
            "num_seqs": args.mla_seqs,
            "q_len": args.mla_q_len,
            "kv_len": args.mla_kv_len,
            "num_q_heads": args.mla_q_heads,
            "lkv_dim": args.mla_lkv,
            "r_dim": args.mla_r,
            "page_size": args.mla_page_size,
        },
        config=config,
        notes=(
            "there is no jittable pure-JAX MLA denominator: the only upstream "
            "reference validates traced values with Python control flow, so it "
            "is used for correctness only and speedups are reported between "
            "implementations rather than against a baseline",
            "logical counts QK over (lkv + r) and PV over lkv, for every query "
            "head against the single shared latent KV stream -- that sharing is "
            "what makes the MLA cache small",
            "no upstream benchmark defines a native shape for this contract, so "
            "this is a declared validation shape at DeepSeek-V3 MLA dims "
            "(kv_lora_rank=512, qk_rope_head_dim=64)",
        ),
    )


for _name in _MLA_MODULES:
    register(
        _name,
        Case(
            family="mla_attention",
            contract=_mla_contract(_name),
            build=_build_mla,
            native_shape=None,
            is_reference=_name == "mla-baseline",
        ),
    )


# ---------------------------------------------------------------------------
# attention / splash_attention
# ---------------------------------------------------------------------------

SPLASH = KERNELS / "attention" / "splash_attention"

_SPLASH_MODULES = {
    "splash-jaxbench": ("splash_jaxbench", "jaxbench_optimized.py"),
    "splash-maxtext": ("splash_maxtext", "maxtext_optimized.py"),
}


def _build_splash_attention(args) -> Built:
    baseline = load_module("splash_baseline", SPLASH / "baseline.py")
    name = args.implementation.removesuffix("-grad")
    module_name, filename = _SPLASH_MODULES[name]
    module = load_module(module_name, SPLASH / filename)

    inputs = baseline.create_inputs(
        num_q_heads=args.splash_q_heads,
        num_kv_heads=args.splash_kv_heads,
        seq_len=args.splash_seq_len,
        head_dim=args.splash_head_dim,
        # bf16 matches JAXBench's own workload and every other corpus family.
        dtype=jnp.bfloat16,
    )
    mask = baseline.causal_multi_head_mask(args.splash_seq_len, args.splash_q_heads)

    # Both files ship a pure-JAX attention_reference taking the same mask, so
    # the reference is exact-contract rather than re-derived here.
    forward_reference = module.make_attention_reference(
        mask, is_mqa=False, backward_impl="vanilla"
    )
    # BlockSizes.get_default() is 128x128x128 and carries an upstream
    # "TODO: select better parameters" -- roughly 15x slower per head than the
    # Tokamax splash already in the corpus. Use JAXBench's autotuned blocks for
    # both implementations so the comparison isolates the kernels, not tuning.
    jaxbench = load_module("splash_tuned_params", SPLASH / "jaxbench_optimized.py")
    if args.splash_default_blocks:
        block_sizes = module.BlockSizes.get_default()
        block_selection = "BlockSizes.get_default() (128^3)"
    else:
        tuned = jaxbench.TUNED_PARAMS
        # JAXBench autotuned only the forward blocks and left every backward
        # block None ("Not autotuned (backward-only)"). The backward pass
        # rejects None, so fall back to the upstream default of 128 for those
        # and say so, rather than inventing a tuning nobody measured.
        backward_default = 128
        block_sizes = module.BlockSizes(
            block_q=tuned["block_q"],
            block_kv=tuned["block_kv"],
            block_kv_compute=tuned["block_kv_compute"],
            block_q_dkv=tuned["block_q_dkv"] or backward_default,
            block_kv_dkv=tuned["block_kv_dkv"] or backward_default,
            block_kv_dkv_compute=(
                tuned["block_kv_dkv_compute"] or backward_default
            ),
            block_q_dq=tuned["block_q_dq"] or backward_default,
            block_kv_dq=tuned["block_kv_dq"] or backward_default,
        )
        block_selection = (
            f"jaxbench-autotuned forward q={tuned['block_q']} "
            f"kv={tuned['block_kv']} kv_compute={tuned['block_kv_compute']}; "
            f"backward untuned upstream, default {backward_default}"
        )
    forward_kernel = module.make_splash_mha_single_device(
        mask, block_sizes=block_sizes
    )

    backward = args.implementation.endswith("-grad")
    config: dict[str, Any] = {
        "mask": "causal MultiHeadMask",
        "pass": "forward+backward" if backward else "forward",
        "block_selection": block_selection,
    }

    if backward:
        # grad through the custom_vjp exercises both backward launch points.
        def as_loss(fn):
            def loss(q, k, v):
                return jnp.sum(fn(q, k, v).astype(jnp.float32) ** 2)

            return jax.grad(loss, argnums=(0, 1, 2))

        function = as_loss(forward_kernel)
        reference = as_loss(forward_reference)
    else:
        function = forward_kernel
        reference = forward_reference
    reference.__name__ = "attention_reference"

    rectangle = (
        4 * args.splash_q_heads * args.splash_seq_len * args.splash_seq_len
        * args.splash_head_dim
    )
    flops = {"logical": rectangle, "causal_useful": rectangle // 2}
    if backward:
        # A backward pass costs roughly twice the forward on top of it.
        flops = {k: v * 3 for k, v in flops.items()}
    return Built(
        function=function,
        reference=reference,
        inputs=inputs,
        contract="splash_attention_mha",
        flops=flops,
        shape={
            "num_q_heads": args.splash_q_heads,
            "num_kv_heads": args.splash_kv_heads,
            "seq_len": args.splash_seq_len,
            "head_dim": args.splash_head_dim,
        },
        config=config,
        notes=(
            "splash is block-sparse: with a causal mask it visits only the "
            "lower-triangular blocks, so causal_useful is the closer estimate "
            "of scheduled work and logical is the full-rectangle convention",
            "the -grad variants differentiate through the custom_vjp, which is "
            "what exercises the backward_dq and backward_dkv launch points; "
            "their FLOP counts are the forward count times three, a "
            "conventional estimate rather than a measured one",
            "no batch dimension: the kernel is per-device and callers vmap, so "
            "this is one device's share of a batched workload",
        ),
    )


for _name in (*_SPLASH_MODULES, *(f"{n}-grad" for n in _SPLASH_MODULES)):
    register(
        _name,
        Case(
            family="splash_attention",
            contract="splash_attention_mha",
            build=_build_splash_attention,
            native_shape=None,
            is_reference=False,
        ),
    )


# ---------------------------------------------------------------------------
# memory / sparsecore_ragged_gather
# ---------------------------------------------------------------------------

SPARSECORE = KERNELS / "memory" / "sparsecore_ragged_gather"

_SC_CASES = {
    "scg-baseline-gather": ("ragged_gather", None),
    "scg-tokamax": ("ragged_gather", "tokamax_optimized.py"),
    "scg-tokamax-v2": ("ragged_gather", "tokamax_v2_optimized.py"),
    "scg-maxtext": ("ragged_gather", "maxtext_optimized.py"),
    "scg-baseline-gather-reduce": ("ragged_gather_reduce", None),
    "scg-tokamax-gather-reduce": (
        "ragged_gather_reduce", "tokamax_gather_reduce_optimized.py"
    ),
}


# The two contracts need different shapes, and not for tuning reasons.
# `ragged_gather_reduce` refuses its own Pallas path when `x` fits comfortably
# in VMEM -- upstream falls back to XLA when `size(x) * itemsize * 2 < 0.6 *
# vmem_capacity_bytes`, which on a v6e (128 MiB VMEM) means fp32 `x` must
# exceed 38.4 MiB.  It also asserts `num_cores // num_column_partitions <=
# num_lanes`, which needs hidden >= 2048 on this device.  Profiling it at the
# gather shape measures XLA, not the kernel.
_SC_SHAPES = {
    "ragged_gather": {"num_rows": 4096, "hidden": 1024, "out_rows": 2048},
    "ragged_gather_reduce": {"num_rows": 8192, "hidden": 2048, "out_rows": 2048},
}


def _build_sparsecore_gather(args) -> Built:
    baseline = load_module("sparsecore_baseline", SPARSECORE / "baseline.py")
    contract, filename = _SC_CASES[args.implementation]
    defaults = _SC_SHAPES[contract]
    num_rows = args.scg_rows if args.scg_rows is not None else defaults["num_rows"]
    hidden = args.scg_hidden if args.scg_hidden is not None else defaults["hidden"]
    out_rows = (
        args.scg_out_rows if args.scg_out_rows is not None else defaults["out_rows"]
    )
    built = baseline.create_inputs(
        num_rows=num_rows, hidden=hidden, out_rows=out_rows
    )
    config: dict[str, Any] = {"backend": "sparsecore"}

    if contract == "ragged_gather":
        inputs = (built["x"], built["indices"], built["start"], built["end"])
        reference = baseline.ragged_gather
        if filename is None:
            function = reference
        else:
            module = load_module(f"scg_{args.implementation}", SPARSECORE / filename)
            entry = (
                module.ragged_gather
                if args.implementation == "scg-maxtext"
                else module.ragged_gather_pallas
            )
            # The kernels pad the output up to the SparseCore block size and
            # the column tile; trim to the live region so the comparison is
            # against the same thing the reference returns.
            rows = out_rows

            def function(x, indices, start, end, _entry=entry, _r=rows, _h=hidden):
                return _entry(x, indices, start, end)[:_r, :_h]

    else:
        group = args.scg_reduce_group
        inputs = (
            built["x"], built["indices"], built["weights"], built["valid_rows_mask"]
        )
        config["reduce_group_size"] = group
        reference = functools.partial(
            baseline.ragged_gather_reduce, reduce_group_size=group
        )
        reference.__name__ = "ragged_gather_reduce"
        if filename is None:
            function = reference
        else:
            module = load_module(f"scg_{args.implementation}", SPARSECORE / filename)
            def function(x, indices, w, mask, _m=module, _g=group, _h=hidden):
                return _m.ragged_gather_reduce_pallas(x, indices, w, mask, _g)[:, :_h]

    # A gather does no arithmetic: it reads one row per index and writes one,
    # so achieved bandwidth is the metric, not MXU utilization.
    moved = baseline.bytes_moved(
        out_rows=out_rows, hidden=hidden, itemsize=4
    )
    return Built(
        function=function,
        reference=reference,
        inputs=inputs,
        contract=contract,
        flops={},
        primary_flops=None,
        bytes_moved={"gathered_rows": moved},
        primary_bytes="gathered_rows",
        shape={
            "num_rows": num_rows,
            "hidden": hidden,
            "out_rows": out_rows,
        },
        config=config,
        notes=(
            "SparseCore kernel: runs on the v6e's SparseCores via "
            "plsc.VectorSubcoreMesh, not its TensorCore, so MXU utilization is "
            "not reported and the metric is achieved bandwidth",
            "the kernels pad their output to the SparseCore block size and "
            "column tile; the profiled function trims to the live region so it "
            "is compared against the same thing the reference returns",
            "the reference is upstream's own no-SparseCore fallback "
            "(x[indices]), not a re-derived one",
        ),
    )


for _name in _SC_CASES:
    register(
        _name,
        Case(
            family="sparsecore_ragged_gather",
            contract=_SC_CASES[_name][0],
            build=_build_sparsecore_gather,
            native_shape=None,
            is_reference=_name.startswith("scg-baseline"),
        ),
    )


def add_arguments(parser) -> None:
    """Register every family's shape and tuning options."""
    attention = parser.add_argument_group("attention shapes")
    attention.add_argument("--batch", type=int, default=1)
    attention.add_argument("--heads", type=int, default=8)
    attention.add_argument("--sequence", type=int, default=1024)
    attention.add_argument("--head-dim", type=int, default=128)
    attention.add_argument("--value-head-dim", type=int, default=128)
    attention.add_argument("--tokamax-block-q", type=int, default=2048)
    attention.add_argument("--tokamax-block-kv", type=int, default=2048)
    attention.add_argument("--tokamax-block-kv-compute", type=int, default=1024)

    grouped = parser.add_argument_group("grouped matmul shapes")
    grouped.add_argument("--rows", type=int, default=1024, help="m, total routed tokens")
    grouped.add_argument("--groups", type=int, default=8, help="number of experts")
    grouped.add_argument("--k", type=int, default=256)
    grouped.add_argument("--n", type=int, default=256)
    grouped.add_argument(
        "--unbalanced",
        action="store_true",
        help="use uneven group sizes that still sum to --rows",
    )

    sparsecore = parser.add_argument_group("sparsecore ragged gather shapes")
    sparsecore.add_argument("--scg-rows", type=int, default=None)
    sparsecore.add_argument("--scg-hidden", type=int, default=None)
    sparsecore.add_argument("--scg-out-rows", type=int, default=None)
    sparsecore.add_argument("--scg-reduce-group", type=int, default=4)

    splash = parser.add_argument_group("splash attention shapes")
    # One device's share of JAXBench 2p_GQA (Llama-3.1-405B): the native
    # config is 128 q-heads / 8 kv-heads / seq 4096 across batch 4.
    splash.add_argument("--splash-q-heads", type=int, default=32)
    splash.add_argument("--splash-kv-heads", type=int, default=8)
    splash.add_argument("--splash-seq-len", type=int, default=4096)
    splash.add_argument("--splash-head-dim", type=int, default=128)
    splash.add_argument(
        "--splash-default-blocks",
        action="store_true",
        help="use BlockSizes.get_default() (128^3) instead of JAXBench's "
        "autotuned blocks, to measure the tuning effect",
    )

    mla = parser.add_argument_group("mla attention shapes")
    # DeepSeek-V3 MLA dims: kv_lora_rank=512, qk_rope_head_dim=64.
    mla.add_argument("--mla-seqs", type=int, default=8)
    mla.add_argument("--mla-q-len", type=int, default=64)
    mla.add_argument("--mla-kv-len", type=int, default=1024)
    mla.add_argument("--mla-q-heads", type=int, default=16)
    mla.add_argument("--mla-lkv", type=int, default=512)
    mla.add_argument("--mla-r", type=int, default=64)
    mla.add_argument("--mla-page-size", type=int, default=16)
    mla.add_argument("--mla-kv-pages-per-block", type=int, default=2)
    mla.add_argument("--mla-queries-per-block", type=int, default=16)

    routing = parser.add_argument_group("topk routing shapes")
    # DeepSeek-V3-style router: 4096 tokens, 256 experts, top-8 over 8 groups.
    # No upstream benchmark defines a native shape, so this is declared.
    routing.add_argument("--topk-batch", type=int, default=4096)
    routing.add_argument("--topk-experts", type=int, default=256)
    routing.add_argument("--topk-k", type=int, default=8)
    routing.add_argument("--topk-expert-groups", type=int, default=8)
    routing.add_argument("--topk-group", type=int, default=4)

    quant = parser.add_argument_group("quantized matmul shapes")
    # Default is (tpu_version=6, n_batch=1024, n_out=14336, n_in=4096, int8,
    # int8), an entry the upstream tuned-size table covers for v6e -- a
    # Llama-3-70B MLP up-projection.  No upstream benchmark defines a native
    # shape, so this is a declared validation shape.
    quant.add_argument("--qm-batch", type=int, default=1024)
    quant.add_argument("--qm-in", type=int, default=4096)
    quant.add_argument("--qm-out", type=int, default=14336)
    quant.add_argument(
        "--qm-quantize-activation",
        action="store_true",
        help="dynamically quantize activations to int8 per token",
    )

    paged = parser.add_argument_group("ragged paged attention shapes")
    paged.add_argument("--max-num-batched-tokens", type=int, default=128)
    paged.add_argument("--max-num-seqs", type=int, default=4)
    paged.add_argument("--num-q-heads", type=int, default=8)
    paged.add_argument("--num-kv-heads", type=int, default=2)
    paged.add_argument("--page-size", type=int, default=16)
    paged.add_argument("--pages-per-seq", type=int, default=8)
    cache = parser.add_argument_group("kv cache update shapes")
    # Declared validation shape: one Llama-3.1-70B serving step's worth of KV,
    # matching the ragged-paged-attention family's head count. No upstream
    # benchmark defines a native shape for this kernel.
    cache.add_argument("--kv-tokens", type=int, default=4096)
    cache.add_argument("--kv-combined-heads", type=int, default=16)
    cache.add_argument("--kv-pages", type=int, default=1024)
    cache.add_argument("--kv-page-size", type=int, default=32)
    cache.add_argument("--kv-slices", type=int, default=128)
    cache.add_argument("--kv-slices-per-block", type=int, default=8)

    paged.add_argument(
        "--rpa-auto-blocks",
        action="store_true",
        help="let each kernel select its own block sizes instead of sharing "
        "JAXBench's autotuned configuration",
    )
