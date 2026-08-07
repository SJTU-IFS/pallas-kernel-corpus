"""Standalone vLLM tpu-inference DeepSeek-V4 compressor (pure JAX).

Source:
  repository: https://github.com/vllm-project/tpu-inference
  commit: 8b9c90928c94c7230d1bc891534a301510a6a30d
  path: tpu_inference/kernels/experimental/deepseek_v4/
  files: ('compress_norm_rope.py', 'compressor.py')
  transformation: the two modules above flattened in dependency order with the
    repo-local import removed.  No other change.

**This file contains no Pallas.**  It is upstream's own reference
implementation of the same operation the Pallas compressor performs, and
upstream's ``tests/kernels/deepseek_v4/compressor_test.py`` checks it against a
NumPy ground truth.  The corpus uses it as this contract's reference rather
than re-deriving the packed cache layout, for the same reason as the
ragged-paged-attention and MLA families: a subtly wrong reference produces
false failures that look like kernel bugs.

Entry point: ``compressor_forward``.  Note its signature is **not** the Pallas
one: it takes ``kv_score`` already projected, where the Pallas path takes
``hidden_states`` and ``wkv_wgate`` and does the projection itself inside
``proj_and_save_state``.  It also packs rope into the single ``cache`` where the
Pallas path writes a separate ``rope_cache``.  ``baseline.py`` records how to
line the two up.

Also exported, and used by the corpus tests: ``unpack_state_cache`` /
``pack_state_cache`` (the fp32 state view of the uint8 buffer) and
``unpack_sparse_kv_cache`` (nope, rope and scales out of a packed record).
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/vllm-project/tpu-inference",
    "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
    "path": "tpu_inference/kernels/experimental/deepseek_v4",
    "files": ('compress_norm_rope.py', 'compressor.py'),
    "kind": "reference",
    "backend": "jax",
    "target": "portable",
    "contract": "dsv4_compress_and_store",
    "launch_points": 0,
}

import jax
import jax.numpy as jnp

# --- from compress_norm_rope.py -----------------------------------
def _to_byte_lane(x: jax.Array) -> jax.Array:
    """Reinterpret each element of ``x``'s trailing dim as raw bytes.

    ``bitcast_convert_type`` appends a trailing itemsize dim for dtypes wider
    than 8 bits (bf16 -> ``[..., 2]``, f32 -> ``[..., 4]``).
    """
    b = jax.lax.bitcast_convert_type(x, jnp.uint8)
    if b.ndim > x.ndim:
        b = b.reshape(*x.shape[:-1], -1)
    return b


def _from_byte_lane(b: jax.Array, dtype) -> jax.Array:
    """Inverse of ``_to_byte_lane``: read trailing bytes back as ``dtype``."""
    itemsize = jnp.dtype(dtype).itemsize
    if itemsize == 1:
        return jax.lax.bitcast_convert_type(b, dtype)
    grouped = b.reshape(*b.shape[:-1], b.shape[-1] // itemsize, itemsize)
    return jax.lax.bitcast_convert_type(grouped, dtype)


def quantize_fp8_ue8m0(x: jax.Array, block_size: int):
    """Block fp8 quantization with UE8M0 (power-of-two) block scales."""
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    *lead, dim = x.shape
    blocked = x.reshape(*lead, dim // block_size, block_size)
    amax = jnp.clip(jnp.max(jnp.abs(blocked), axis=-1, keepdims=True), 1e-4,
                    None)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    q = (blocked * (1.0 / scale)).astype(jnp.float8_e4m3fn).reshape(x.shape)
    scale = jnp.squeeze(scale, -1).astype(jnp.float8_e8m0fnu)
    return q, scale


def sparse_packed_width(nope_dim: int, rope_head_dim: int,
                        quant_block: int) -> int:
    """Bytes per token in the packed sparse (head_dim=512) KV cache."""
    # nope fp8 (1B) + rope bf16 (2B) + UE8M0 block scale (1B)
    return nope_dim + rope_head_dim * 2 + (nope_dim // quant_block)


def indexer_packed_width(head_dim: int, quant_block: int) -> int:
    """Bytes per token in the packed indexer (head_dim=128) KV cache."""
    # fp8 (1B) + UE8M0 block scale (1B)
    return head_dim + (head_dim // quant_block)


def unpack_sparse_kv_cache(kv_cache: jax.Array, nope_dim: int,
                           rope_head_dim: int, quant_block: int):
    """Split the packed sparse KV cache into native-dtype component views.

    The block scale is stored as UE8M0 (``float8_e8m0fnu``, one byte per
    block) and returned as the equivalent power-of-two ``float32``.
    """
    n_qb = nope_dim // quant_block
    a = nope_dim
    b = a + rope_head_dim * 2
    nope = _from_byte_lane(kv_cache[..., :a], jnp.float8_e4m3fn)
    rope = _from_byte_lane(kv_cache[..., a:b], jnp.bfloat16)
    scale = _from_byte_lane(kv_cache[..., b:b + n_qb],
                            jnp.float8_e8m0fnu).astype(jnp.float32)
    return nope, rope, scale


def unpack_indexer_kv_cache(kv_cache: jax.Array, head_dim: int,
                            quant_block: int):
    """Split the packed indexer KV cache into ``(fp8, scale)`` views."""
    n_qb = head_dim // quant_block
    fp8 = _from_byte_lane(kv_cache[..., :head_dim], jnp.float8_e4m3fn)
    scale = _from_byte_lane(kv_cache[..., head_dim:head_dim + n_qb],
                            jnp.float8_e8m0fnu).astype(jnp.float32)
    return fp8, scale


# uint8 lanes per 32-bit MLA word
PACKING = 4


def _align_to(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def shared_sparse_cache_shape(num_pages: int, page_size: int, nope_dim: int,
                              rope_head_dim: int, quant_block: int):
    """MLA-style shape of the shared state+KV ``uint8`` buffer (sparse path).

    Returns ``[num_pages, page_size // PACKING, PACKING, width]`` where
    ``width = align_to(sparse_packed_width(...), 128)`` (= 640 for DeepSeek-V4
    head_dim=512).
    """
    width = _align_to(
        sparse_packed_width(nope_dim, rope_head_dim, quant_block), 128)
    return (num_pages, _align_to(page_size, PACKING) // PACKING, PACKING,
            width)


def shared_indexer_cache_shape(num_pages: int, page_size: int, head_dim: int,
                               quant_block: int):
    """MLA-style shape of the shared state+KV ``uint8`` buffer (indexer path).

    Same layout as ``shared_sparse_cache_shape`` but sized for the indexer
    record: ``width = align_to(indexer_packed_width(...), 128)`` (= 256 for
    DeepSeek-V4 head_dim=128). 
    """
    width = _align_to(indexer_packed_width(head_dim, quant_block), 128)
    return (num_pages, _align_to(page_size, PACKING) // PACKING, PACKING,
            width)


def _state_chunk_dims(cache_shape, state_block_size: int, state_dim: int):
    """Geometry of state token in cache.

    Returns:
        kv_slots: row-slots per page (``rows * packing``, e.g. 64).
        rows_per_token: row-slots per state token (``page_size //
            state_block_size``, e.g. 16).
        f32_per_row: f32 values per row-slot (e.g. 128).
        bytes_per_row: bytes used per row-slot (``f32_per_row * 4``, e.g. 512).
    """
    num_pages, rows, packing, width = cache_shape
    kv_slots = rows * packing
    if kv_slots % state_block_size != 0:
        raise ValueError(f"page_size {kv_slots} not divisible by "
                         f"state_block_size {state_block_size}")
    rows_per_token = kv_slots // state_block_size
    if state_dim % rows_per_token != 0:
        raise ValueError(f"state_dim {state_dim} not divisible by "
                         f"rows_per_token {rows_per_token}")
    f32_per_row = state_dim // rows_per_token
    bytes_per_row = f32_per_row * 4
    if bytes_per_row > width:
        raise ValueError(
            f"state row needs {bytes_per_row}B but cache width is {width}B")
    return kv_slots, rows_per_token, f32_per_row, bytes_per_row


def unpack_state_cache(cache: jax.Array, state_block_size: int,
                       state_dim: int) -> jax.Array:
    """Read the shared buffer's f32 state view ``[num_pages, sb, state_dim]``.

    Inverse of ``pack_state_cache``. Only the leading ``bytes_per_row`` of each
    row-slot carry state; trailing pad bytes are ignored.
    """
    num_pages, rows, packing, width = cache.shape
    kv_slots, rpt, _, bpr = _state_chunk_dims(cache.shape, state_block_size,
                                              state_dim)
    slots = cache.reshape(num_pages, kv_slots, width)
    chunk = slots[:, :, :bpr].reshape(num_pages, state_block_size, rpt, bpr)
    f32 = _from_byte_lane(chunk, jnp.float32)  # [num_pages, sb, rpt, fpr]
    return f32.reshape(num_pages, state_block_size, state_dim)


def pack_state_cache(cache: jax.Array, state: jax.Array) -> jax.Array:
    """Write an f32 state view ``[num_pages, sb, state_dim]`` into ``cache``.

    Inverse of ``unpack_state_cache``; leaves each row-slot's pad bytes (and the
    KV-only row-slots) untouched.
    """
    num_pages, rows, packing, width = cache.shape
    sb, state_dim = state.shape[1], state.shape[2]
    kv_slots, rpt, fpr, bpr = _state_chunk_dims(cache.shape, sb, state_dim)
    chunk = state.reshape(num_pages, sb, rpt, fpr)
    chunk_bytes = _to_byte_lane(chunk).reshape(num_pages, kv_slots, bpr)
    slots = cache.reshape(num_pages, kv_slots, width)
    slots = slots.at[:, :, :bpr].set(chunk_bytes)
    return slots.reshape(num_pages, rows, packing, width)


def interleaved_rope(
    x: jax.Array,  # [..., head_dim] fp32
    cos_sin: jax.Array,  # [..., rope_head_dim] fp32 ([cos | sin])
    rope_head_dim: int,
) -> jax.Array:
    """Interleaved (GPT-J) RoPE on the trailing ``rope_head_dim`` elements."""
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even; got {head_dim}")
    if rope_head_dim % 2 != 0 or rope_head_dim > head_dim:
        raise ValueError(f"rope_head_dim must be even and <= head_dim; got "
                         f"rope_head_dim={rope_head_dim}, head_dim={head_dim}")

    half_rope = rope_head_dim // 2
    num_pairs = head_dim // 2
    nope_pairs = num_pairs - half_rope

    pairs = x.reshape(*x.shape[:-1], num_pairs, 2)
    even = pairs[..., 0]
    odd = pairs[..., 1]

    cos = cos_sin[..., :half_rope]
    sin = cos_sin[..., half_rope:rope_head_dim]

    pad_shape = (*cos.shape[:-1], nope_pairs)
    cos_full = jnp.concatenate([jnp.ones(pad_shape, x.dtype), cos], axis=-1)
    sin_full = jnp.concatenate([jnp.zeros(pad_shape, x.dtype), sin], axis=-1)

    new_even = even * cos_full - odd * sin_full
    new_odd = odd * cos_full + even * sin_full

    out = jnp.stack([new_even, new_odd], axis=-1)
    return out.reshape(x.shape)


def compress_norm_rope(
    kv_window: jax.Array,  # [num_tokens, window, head_dim] fp32
    score_window: jax.Array,  # [num_tokens, window, head_dim] fp32
    valid_mask: jax.Array,  # [num_tokens, window] bool
    rms_weight: jax.Array,  # [head_dim] fp32
    cos_sin_cache: jax.Array,  # [max_pos, rope_head_dim] fp32
    compressed_pos: jax.Array,  # [num_tokens] int
    rms_eps: float,
    rope_head_dim: int,
) -> jax.Array:
    """Window softmax-pool, RMSNorm, and interleaved RoPE."""
    neg_inf = jnp.array(-jnp.inf, dtype=score_window.dtype)
    masked_score = jnp.where(valid_mask[..., None], score_window, neg_inf)
    weights = jax.nn.softmax(masked_score, axis=1)

    compressed_kv = jnp.sum(weights * kv_window,
                            axis=1)  # [num_tokens, head_dim]

    # RMSNorm over head_dim. variance: [num_tokens, 1];
    # normed: [num_tokens, head_dim].
    variance = jnp.mean(jnp.square(compressed_kv), axis=-1, keepdims=True)
    normed = compressed_kv * jax.lax.rsqrt(variance + rms_eps) * rms_weight

    cos_sin = cos_sin_cache[compressed_pos]
    return interleaved_rope(normed, cos_sin, rope_head_dim)


def gather_state_windows(
    state_cache: jax.Array,  # [num_blocks, block_size, 2*state_width] fp32
    positions: jax.Array,  # [num_tokens] int
    block_table: jax.Array,  # [num_reqs, max_blocks] int
    token_to_req_indices: jax.Array,  # [num_tokens] int
    block_size: int,
    head_dim: int,
    compress_ratio: int,
    overlap: bool,
):
    """
    Gather ``[kv_window, score_window, valid_mask]`` from the paged cache.

    Returns:
      kv_window: [token, window, head_dim] -- partial kv vectors to pool 
      score_window: [tokens, window, head_dim] -- scores for softmax weight
      valid_mask: [token, window] -- False where window goes before seq start.
    
    """
    coff = 1 + int(overlap)
    window = coff * compress_ratio

    start = positions - window + 1
    w_idx = jnp.arange(window)
    pos = start[:, None] + w_idx[None, :]
    valid_mask = pos >= 0

    safe_pos = jnp.where(valid_mask, pos, 0)
    req = token_to_req_indices[:, None]
    # Gather page numbers (optimized to 1D indexing to avoid 2D index bitpacking)
    max_blocks = block_table.shape[-1]
    flat_index = req * max_blocks + (safe_pos // block_size)
    block_numbers = block_table.reshape(-1)[flat_index]
    block_offsets = safe_pos % block_size

    bn = block_numbers[:, :, None]
    bo = block_offsets[:, :, None]

    # Gather the entire state vector for the window tokens
    temp = state_cache[bn, bo].squeeze(2)

    if not overlap:
        kv_window = temp[:, :, :head_dim]
        score_window = temp[:, :, head_dim:2 * head_dim]
    else:
        # state_width * 2 = 4 * head_dim
        # Reshape to separate the 4 head-sized blocks:
        # [num_tokens, window, 4, head_dim]
        temp_reshaped = temp.reshape(temp.shape[0], window, 4, head_dim)
        w_cond = (w_idx < compress_ratio)[None, :, None]
        kv_window = jnp.where(w_cond, temp_reshaped[:, :, 0, :],
                              temp_reshaped[:, :, 1, :])
        score_window = jnp.where(w_cond, temp_reshaped[:, :, 2, :],
                                 temp_reshaped[:, :, 3, :])
    return kv_window, score_window, valid_mask


def _boundary_dest(
    positions: jax.Array,  # [num_tokens] int
    slot_mapping: jax.Array,  # [num_tokens] int
    kv_slot_mapping: jax.Array,  # [num_tokens] int
    compress_ratio: int,
    num_slots: int,
) -> jax.Array:
    is_boundary = ((positions + 1) % compress_ratio) == 0
    store = is_boundary & (slot_mapping >= 0) & (kv_slot_mapping >= 0)
    return jnp.where(store, kv_slot_mapping, num_slots)


def compress_norm_rope_store(
    cache: jax.Array,  # [num_pages, page_size//4, 4, width] uint8
    state_cache: jax.Array,  # [num_blocks, block_size, 2*state_width] fp32
    positions: jax.Array,  # [num_tokens] int
    slot_mapping: jax.Array,  # [num_tokens] int (state-cache slots)
    block_table: jax.Array,  # [num_reqs, max_blocks] int (state pages)
    token_to_req_indices: jax.Array,  # [num_tokens] int
    kv_slot_mapping: jax.Array,  # [num_tokens] int (compressed-KV slots)
    rms_weight: jax.Array,  # [head_dim] fp32
    cos_sin_cache: jax.Array,  # [max_pos, rope_head_dim] fp32
    state_block_size: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
):
    """Compress, norm, RoPE, and write boundary KV into the shared cache."""
    state_view = state_cache

    kv_window, score_window, valid_mask = gather_state_windows(
        state_cache=state_view,
        positions=positions,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        block_size=state_block_size,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
    )

    compressed_pos = (positions // compress_ratio) * compress_ratio
    compressed = compress_norm_rope(
        kv_window=kv_window,
        score_window=score_window,
        valid_mask=valid_mask,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        compressed_pos=compressed_pos,
        rms_eps=rms_eps,
        rope_head_dim=rope_head_dim,
    )

    nope_dim = head_dim - rope_head_dim
    nope = compressed[..., :nope_dim]
    rope = compressed[..., nope_dim:]

    q, scale = quantize_fp8_ue8m0(nope, quant_block)
    rope_q = rope.astype(jnp.bfloat16)

    record = jnp.concatenate(
        [_to_byte_lane(q),
         _to_byte_lane(rope_q),
         _to_byte_lane(scale)],
        axis=-1)

    num_pages, rows, packing, width = cache.shape
    pad = width - record.shape[-1]
    if pad < 0:
        raise ValueError(
            f"packed record {record.shape[-1]}B exceeds cache width {width}B")
    record = jnp.pad(record, ((0, 0), (0, pad)))

    num_slots = num_pages * rows * packing
    dest = _boundary_dest(positions, slot_mapping, kv_slot_mapping,
                          compress_ratio, num_slots)
    flat = cache.reshape(num_slots, width)
    flat = flat.at[dest].set(record, mode="drop")
    return flat.reshape(num_pages, rows, packing, width)


def compress_norm_rope_store_indexer(
    cache: jax.Array,  # [num_pages, page_size//4, 4, width] uint8
    state_cache: jax.Array,  # [num_blocks, block_size, 2*state_width] fp32
    positions: jax.Array,  # [num_tokens] int
    slot_mapping: jax.Array,  # [num_tokens] int (state-cache slots)
    block_table: jax.Array,  # [num_reqs, max_blocks] int (state pages)
    token_to_req_indices: jax.Array,  # [num_tokens] int
    kv_slot_mapping: jax.Array,  # [num_tokens] int (indexer-KV slots)
    rms_weight: jax.Array,  # [head_dim] fp32
    cos_sin_cache: jax.Array,  # [max_pos, rope_head_dim] fp32
    state_block_size: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
):
    """Indexer (head_dim=128) twin of ``compress_norm_rope_store``."""
    state_view = state_cache

    kv_window, score_window, valid_mask = gather_state_windows(
        state_cache=state_view,
        positions=positions,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        block_size=state_block_size,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
    )

    compressed_pos = (positions // compress_ratio) * compress_ratio
    compressed = compress_norm_rope(
        kv_window=kv_window,
        score_window=score_window,
        valid_mask=valid_mask,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        compressed_pos=compressed_pos,
        rms_eps=rms_eps,
        rope_head_dim=rope_head_dim,
    )

    q, scale = quantize_fp8_ue8m0(compressed, quant_block)

    record = jnp.concatenate([_to_byte_lane(q), _to_byte_lane(scale)], axis=-1)

    num_pages, rows, packing, width = cache.shape
    pad = width - record.shape[-1]
    if pad < 0:
        raise ValueError(
            f"packed record {record.shape[-1]}B exceeds cache width {width}B")
    record = jnp.pad(record, ((0, 0), (0, pad)))

    num_slots = num_pages * rows * packing
    dest = _boundary_dest(positions, slot_mapping, kv_slot_mapping,
                          compress_ratio, num_slots)
    flat = cache.reshape(num_slots, width)
    flat = flat.at[dest].set(record, mode="drop")
    return flat.reshape(num_pages, rows, packing, width)


# --- from compressor.py -------------------------------------------
def save_partial_states(
    kv_score: jax.Array,  # [num_tokens, 2 * coff * head_dim] fp32
    ape: jax.Array,  # [compress_ratio, coff * head_dim] fp32
    positions: jax.Array,  # [num_tokens] int
    state_cache: jax.Array,  # [num_blocks, block_size, 2*coff*head_dim] fp32
    slot_mapping: jax.Array,  # [num_tokens] int
    head_dim: int,
    overlap: bool,
    compress_ratio: int,
) -> jax.Array:
    """Scatter ``[kv | score + ape]`` into ``state_cache``; skip ``slot < 0``."""

    coff = 1 + int(overlap)
    state_width = coff * head_dim

    # [num_tokens, 2 * coff * head_dim]
    kv = kv_score[:, :state_width]  # [num_tokens, coff * head_dim]
    score = kv_score[:, state_width:2 *
                     state_width]  # [num_tokens, coff * head_dim]

    num_blocks, block_size, two_state_width = state_cache.shape
    state_width = two_state_width // 2

    if kv.shape[-1] != state_width or score.shape[-1] != state_width:
        raise ValueError(
            f"kv/score last dim must equal state_width={state_width}; got "
            f"kv={kv.shape}, score={score.shape}, state_cache={state_cache.shape}"
        )

    cache_dtype = state_cache.dtype
    kv = kv.astype(cache_dtype)
    score = score.astype(cache_dtype)
    ape = ape.astype(cache_dtype)

    ape_rows = jnp.mod(positions, compress_ratio)
    score_state = score + ape[ape_rows]
    packed = jnp.concatenate([kv, score_state], axis=-1)

    num_slots = num_blocks * block_size
    flat = state_cache.reshape(num_slots, two_state_width)
    valid = slot_mapping >= 0
    slots = jnp.where(valid, slot_mapping, num_slots)  # OOB sentinel for pads
    flat = flat.at[slots].set(packed, mode="drop")
    return flat.reshape(num_blocks, block_size, two_state_width)


def compressor_forward(
        kv_score: jax.Array,  # [num_tokens, 2 * coff * head_dim] fp32
        ape: jax.Array,  # [compress_ratio, coff*head_dim] fp32
        norm_weight: jax.Array,  # [head_dim] fp32 RMSNorm gamma
        cos_sin_cache: jax.Array,  # [max_pos, rope_head_dim] fp32 RoPE table
        positions: jax.Array,  # [num_tokens] int logical pos per token
        slot_mapping: jax.Array,  # [num_tokens] int flat state-cache slot
        block_table: jax.Array,  # [num_reqs, max_blocks] int state pages
        token_to_req_indices: jax.Array,  # [num_tokens] int req id per token
        kv_slot_mapping: jax.Array,  # [num_tokens] int flat compressed-KV slot
        cache: jax.Array,  # [num_pages, page_size//4, 4, width] uint8
        state_block_size: int,  # state tokens per page (4=C4, 8=C128)
        head_dim: int,  # 512 for sparse CSA/HCA main path
        rope_head_dim: int,  # 64; trailing dims get interleaved RoPE
        compress_ratio: int,  # 4 (CSA) or 128 (HCA); boundary stride
        overlap: bool,  # True for C4 (two head slices per state row)
        rms_eps: float,
        quant_block: int,  # fp8 absmax block along nope (64)
):
    """head_dim=512 path: project, save state, compress, store into one buffer.

    State cache and compressed KV cache share the same underlying ``cache``
    buffer (see ``compress_norm_rope`` for layout).
    """
    coff = 1 + int(overlap)
    state_dim = 2 * coff * head_dim

    state_view = unpack_state_cache(cache, state_block_size, state_dim)
    state_view = save_partial_states(kv_score, ape, positions, state_view,
                                     slot_mapping, head_dim, overlap,
                                     compress_ratio)
    cache = pack_state_cache(cache, state_view)

    cache = compress_norm_rope_store(
        cache=cache,
        state_cache=state_view,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=norm_weight,
        cos_sin_cache=cos_sin_cache,
        state_block_size=state_block_size,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rms_eps=rms_eps,
        quant_block=quant_block,
    )

    return cache


def compressor_forward_indexer(
        kv_score: jax.Array,  # [num_tokens, 2 * coff * head_dim] fp32
        ape: jax.Array,  # [compress_ratio, coff*head_dim] fp32
        norm_weight: jax.Array,  # [head_dim] fp32 RMSNorm gamma
        cos_sin_cache: jax.Array,  # [max_pos, rope_head_dim] fp32 RoPE table
        positions: jax.Array,  # [num_tokens] int logical pos per token
        slot_mapping: jax.Array,  # [num_tokens] int flat state-cache slot
        block_table: jax.Array,  # [num_reqs, max_blocks] int state pages
        token_to_req_indices: jax.Array,  # [num_tokens] int req id per token
        kv_slot_mapping: jax.Array,  # [num_tokens] int flat indexer-KV slot
        cache: jax.Array,  # [num_pages, page_size//4, 4, width] uint8
        state_block_size: int,  # indexer state tokens per page
        head_dim: int,  # 128 for the indexer path
        rope_head_dim: int,  # 64; trailing dims get interleaved RoPE
        compress_ratio: int,  # 4 (CSA) or 128 (HCA); boundary stride
        overlap: bool,  # True for C4 (two head slices per state row)
        rms_eps: float,
        quant_block: int,  # whole-head fp8 absmax block (128)
):
    """head_dim=128 indexer path: same as ``compressor_forward``, head_dim=128."""
    coff = 1 + int(overlap)
    state_dim = 2 * coff * head_dim

    state_view = unpack_state_cache(cache, state_block_size, state_dim)
    state_view = save_partial_states(kv_score, ape, positions, state_view,
                                     slot_mapping, head_dim, overlap,
                                     compress_ratio)
    cache = pack_state_cache(cache, state_view)

    cache = compress_norm_rope_store_indexer(
        cache=cache,
        state_cache=state_view,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=norm_weight,
        cos_sin_cache=cos_sin_cache,
        state_block_size=state_block_size,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rms_eps=rms_eps,
        quant_block=quant_block,
    )

    return cache
