"""JAX reference for the sglang-jax fused gated MLP in this directory.

sglang-jax has no test and no baseline for `fused_mlp.py`.  Both functions here
are read off its only caller, `python/sgl_jax/srt/models/glm5_moe.py`, whose relevant regions are quoted
verbatim below -- extracted by tools/flatten_fused_mlp.py, not transcribed, so
the quotation cannot drift from the code it claims to reproduce.

`pack_gate_up` matters more than it looks.  `w_gu` is not the whole gate matrix
beside the whole up matrix; the two are interleaved in blocks of `b_inter`,
because the kernel slices gate and up out of one `(b_seq, 2*b_inter)` tile.
Packing them the obvious way computes a different function, and the failure
looks like a kernel bug rather than a calling-convention error.

Source:
  repository: https://github.com/sgl-project/sglang-jax
  commit: a7353325e8c00d287294c2cd679a77173f1a4594
  path: python/sgl_jax/srt/models/glm5_moe.py  (both regions below)

Upstream, `python/sgl_jax/srt/models/glm5_moe.py`, `post_load_weights` -- the packing::

            wg = self.gate_proj.weight.value
            wu = self.up_proj.weight.value
            wd = self.down_proj.weight.value
        ...
                wg = jnp.pad(wg, ((0, 0), (0, pad_inter)), mode="constant")
                wu = jnp.pad(wu, ((0, 0), (0, pad_inter)), mode="constant")
                wd = jnp.pad(wd, ((0, pad_inter), (0, 0)), mode="constant")
        ...
            num_blocks = local_inter_size // b_inter
        ...
            wg_reshaped = jax.lax.reshape(
        ...
            wu_reshaped = jax.lax.reshape(
        ...
            w_gu = jnp.concatenate([wg_reshaped, wu_reshaped], axis=-1)
        ...
            w_gu = jax.lax.reshape(w_gu, (hidden_size, local_inter_size * 2), out_sharding=sharding_2d)
        ...
            self.w_gu.value = w_gu
            self.w_d.value = wd

Upstream, `python/sgl_jax/srt/models/glm5_moe.py`, `__call__` -- the non-fused fallback, which is what
the fused kernel must equal::

            a1, _ = self.gate_proj(hidden_states)
            a2, _ = self.up_proj(hidden_states)
            intermediate_parallel = a2 * self.act_fn(a1)
            output, _ = self.down_proj(intermediate_parallel)
            return output
"""

from __future__ import annotations

SOURCE = {
    "kind": "reference",
    "repository": "https://github.com/sgl-project/sglang-jax",
    "commit": "a7353325e8c00d287294c2cd679a77173f1a4594",
    "path": "python/sgl_jax/srt/models/glm5_moe.py",
    "backend": "jax",
    "target": "portable",
    "contracts": ("fused_gated_mlp",),
}

import jax
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
