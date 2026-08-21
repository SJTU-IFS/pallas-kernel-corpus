"""JAX reference for the ragged causal conv1d in this directory.

`reference_causal_conv1d` is upstream's own, carried verbatim from
`tests/kernels/causal_conv1d_test.py` -- the function tpu-inference's tests compare the kernel against. It
lives in the test file rather than beside the kernel, which is why it is
extracted from there.

**It is eager-only.** The body calls `int()` on entries of `distribution` and
`query_start_loc` to decide how many sequences and tokens are real, so it does
not survive `jax.jit` with those traced. That is upstream's design -- the
reference is written to be obviously correct rather than fast, looping in Python
over every row -- and it means this module's Pallas-freeness is checked by
reading it rather than by lowering it.

Source:
  repository: https://github.com/vllm-project/tpu-inference
  commit: 8b9c90928c94c7230d1bc891534a301510a6a30d
  path: tests/kernels/causal_conv1d_test.py
"""

from __future__ import annotations

SOURCE = {
    "kind": "reference",
    "repository": "https://github.com/vllm-project/tpu-inference",
    "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
    "path": "tests/kernels/causal_conv1d_test.py",
    "backend": "jax",
    "target": "portable",
    "contracts": ("ragged_causal_conv1d",),
}

import jax
import jax.numpy as jnp


def reference_causal_conv1d(
    x: jax.Array,
    conv_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    kernel_size: int,
    has_initial_state: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    num_tokens = x.shape[0]
    num_seqs = state_indices.shape[0]
    sequences = jnp.arange(num_seqs)
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    row_to_seq_idx = jnp.repeat(sequences, query_lens)

    real_num_seqs = int(distribution[2])
    real_num_tokens = int(query_start_loc[real_num_seqs])

    new_conv_state = jnp.copy(conv_state)
    out_list = []

    for row in range(num_tokens):
        if row >= real_num_tokens:
            out_list.append(jnp.zeros_like(x[0]))
            continue

        start_row = row - kernel_size + 1
        s_idx = int(row_to_seq_idx[row])
        seq_start = int(query_start_loc[s_idx])
        state_idx = int(state_indices[s_idx])
        has_init = bool(has_initial_state[s_idx])

        row_out = jnp.zeros_like(x[0], dtype=jnp.float32)
        window_vals = []

        for k in range(kernel_size):
            idx = start_row + k

            if idx < seq_start:
                state_offset = idx - (seq_start - kernel_size + 1)
                if has_init:
                    val = conv_state[state_idx, state_offset]
                else:
                    val = jnp.zeros_like(x[0])
            else:
                val = x[idx]

            window_vals.append(val)
            row_out += val.astype(jnp.float32) * conv_weight[:, 0, k].astype(
                jnp.float32)

        if conv_bias is not None:
            row_out += conv_bias.astype(jnp.float32)

        out_list.append(row_out.astype(x.dtype))

        if row == int(query_start_loc[s_idx + 1]) - 1:
            update = jnp.stack(window_vals[1:], axis=0)
            new_conv_state = new_conv_state.at[state_idx].set(update)

    out = jnp.stack(out_list, axis=0)
    return out, new_conv_state
