"""Standalone sglang-jax fused gated MLP (SwiGLU) kernel.

Source:
  repository: https://github.com/sgl-project/sglang-jax
  commit: a7353325e8c00d287294c2cd679a77173f1a4594
  path: python/sgl_jax/srt/kernels/fused_mlp.py
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

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/sgl-project/sglang-jax",
    "commit": "a7353325e8c00d287294c2cd679a77173f1a4594",
    "path": "python/sgl_jax/srt/kernels/fused_mlp.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "fused_gated_mlp",
    "family": "gated_mlp",
    "launch_points": 1,
    "native_shape": None,
}

import functools

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map  # corpus: see tools/flatten_fused_mlp.py
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P


def inner_mlp_kernel(
    x_tile,
    w_gu_tile,
    wd_tile,
    y_tile,
    y_scratch,
    *,
    b_inter: int,
    num_inter: int,
):
    i_i = pl.program_id(0)

    def _compute(is_first: bool, is_last: bool):
        # 1. Fetch & Upcast weights
        w_gu_sram = w_gu_tile[...].astype(x_tile.dtype)

        # 2. Matmul 1 (x @ W_in)
        hu_sram = jnp.matmul(x_tile[...], w_gu_sram, preferred_element_type=jnp.float32)

        # Split and Activate (SiLU Swish-gated activation)
        h_sram = hu_sram[:, :b_inter]
        u_sram = hu_sram[:, b_inter:]
        a_tile = jax.nn.silu(h_sram) * u_sram
        a_tile = a_tile.astype(x_tile.dtype)

        # 3. Matmul 2 (a @ W_out)
        wd_sram = wd_tile[...].astype(x_tile.dtype)
        y_current_sram = jnp.matmul(a_tile, wd_sram, preferred_element_type=jnp.float32)

        # 4. Accumulate
        acc = y_current_sram if is_first else y_scratch[...] + y_current_sram

        if is_last:
            y_tile[...] = acc.astype(y_tile.dtype)
        else:
            y_scratch[...] = acc

    # Define matmul wrapper scopes for XLA compiler profiling
    @jax.named_scope("compute_first_last")
    def compute_first_last():
        _compute(True, True)

    @jax.named_scope("compute_first")
    def compute_first():
        _compute(True, False)

    @jax.named_scope("compute")
    def compute():
        _compute(False, False)

    @jax.named_scope("compute_last")
    def compute_last():
        _compute(False, True)

    is_first = i_i == 0
    is_last = i_i == (num_inter - 1)

    # Explicit control flow eliminates @pl.when overhead
    jax.lax.cond(
        is_first,
        lambda: jax.lax.cond(is_last, compute_first_last, compute_first),
        lambda: jax.lax.cond(is_last, compute_last, compute),
    )


def mlp_kernel_main(x_hbm, w_gu_hbm, wd_hbm, y_hbm, y_scratch, *, b_seq, b_inter, hidden_size):
    """Entry point for Pallas grid. Wires up HBM references to the pipeline."""
    seq_idx = pl.program_id(0)
    num_inter = w_gu_hbm.shape[1] // (2 * b_inter)

    # 1. Block specs mapping the inner pipeline loop (i_i) to HBM arrays
    x_spec = pl.BlockSpec((b_seq, hidden_size), lambda i_i: (seq_idx, 0))
    y_spec = pl.BlockSpec((b_seq, hidden_size), lambda i_i: (seq_idx, 0))

    # 2. Triple buffering for weights to hide HBM latency
    w_gu_spec = pl.BlockSpec(
        (hidden_size, 2 * b_inter),
        lambda i_i: (0, i_i),
        pipeline_mode=pl.Buffered(buffer_count=3),
    )
    wd_spec = pl.BlockSpec(
        (b_inter, hidden_size),
        lambda i_i: (i_i, 0),
        pipeline_mode=pl.Buffered(buffer_count=3),
    )

    # 3. Emit the pipeline over the intermediate dimension
    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(
            inner_mlp_kernel,
            b_inter=b_inter,
            num_inter=num_inter,
        ),
        grid=(num_inter,),
        in_specs=(x_spec, w_gu_spec, wd_spec),
        out_specs=y_spec,
    )

    # Execute the pipeline, passing our scratchpad forward
    pipeline_fn(x_hbm, w_gu_hbm, wd_hbm, y_hbm, scratches=[y_scratch])


@functools.partial(jax.jit, static_argnums=(3, 4, 5))
def apply_fused_mlp_sharded(
    x: jax.Array,
    w_gu: jax.Array,
    wd: jax.Array,
    mesh: jax.sharding.Mesh,
    b_seq: int = 64,
    b_inter: int = 128,
) -> jax.Array:
    in_specs = (
        P(None, None),  # x
        P(None, "tensor"),  # w_gu (combined gate/up weight, sharded along tensor axis)
        P("tensor", None),  # wd (down weight, sharded along tensor axis)
    )
    out_specs = P(None, None)

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_rep=False,
    )
    def local_fused_mlp(x_loc, w_gu_loc, wd_loc):
        seq_len, hidden_size = x_loc.shape

        # 1D outer grid (parallelizing sequence length across TPU cores)
        grid = (seq_len // b_seq,)

        # Pass full tensors to the kernel main as HBM references
        pallas_in_specs = (
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        )
        pallas_out_specs = pl.BlockSpec(memory_space=pltpu.HBM)

        y_loc = pl.pallas_call(
            functools.partial(
                mlp_kernel_main,
                b_seq=b_seq,
                b_inter=b_inter,
                hidden_size=hidden_size,
            ),
            out_shape=jax.ShapeDtypeStruct((seq_len, hidden_size), x_loc.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=pallas_in_specs,
                out_specs=pallas_out_specs,
                scratch_shapes=[pltpu.VMEM((b_seq, hidden_size), jnp.float32)],
            ),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        )(x_loc, w_gu_loc, wd_loc)

        return jax.lax.psum(y_loc, axis_name="tensor")

    return local_fused_mlp(x, w_gu, wd)


def apply_fused_mlp_with_padding(
    x: jax.Array,
    w_gu: jax.Array,
    wd: jax.Array,
    mesh: jax.sharding.Mesh,
    b_seq: int = 64,
    b_inter: int = 128,
) -> jax.Array:
    """Pads the input sequence length to be a multiple of b_seq if necessary."""
    seq_len = x.shape[0]
    rem = seq_len % b_seq
    if rem == 0:
        return apply_fused_mlp_sharded(x, w_gu, wd, mesh, b_seq, b_inter)

    pad_len = b_seq - rem
    x_padded = jnp.pad(x, ((0, pad_len), (0, 0)), mode="constant")
    out_padded = apply_fused_mlp_sharded(x_padded, w_gu, wd, mesh, b_seq, b_inter)
    return out_padded[:seq_len, :]

# The entry point glm5_moe calls: it pads the sequence to a multiple of b_seq
# first, which `apply_fused_mlp_sharded` requires but does not enforce.
kernel = apply_fused_mlp_with_padding
