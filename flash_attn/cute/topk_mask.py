# Copyright (c) 2025, Tri Dao.
# Utilities for sparse attention masking compatible with FlashAttention's
# mask_mod / block_sparse_tensors interface.

import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F

from flash_attn.cute import utils
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def pack_mask(dense_mask: torch.Tensor) -> torch.Tensor:
    """Bit-pack a dense bool/int mask into int32 words.

    Args:
        dense_mask: (B, Q, K) tensor with nonzero = attend.

    Returns:
        (B, Q, ceil(K / 32)) int32 tensor where bit *j* of word *i* indicates
        whether position ``i * 32 + j`` is attended to.
    """
    B, Q, K = dense_mask.shape
    # Pad K to a multiple of 32
    pad_k = (32 - K % 32) % 32
    if pad_k:
        dm = F.pad(dense_mask, (0, pad_k))
    else:
        dm = dense_mask
    dm = (dm != 0).view(B, Q, -1, 32).to(torch.int32)
    # bit j contributes (1 << j)
    shifts = torch.arange(32, device=dense_mask.device, dtype=torch.int32)
    return (dm << shifts).sum(dim=-1, dtype=torch.int32)


@cute.jit
def dense_mask_mod(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    q_idx: cute.TensorSSA,
    kv_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors: list,
) -> cute.TensorSSA:
    """Mask mod that reads from a bit-packed int32 mask.

    aux_tensors[0] must be a (B, Q, ceil(K / 32)) int32 tensor produced by
    ``pack_mask``.  Bit *j* of word *i* indicates position ``i * 32 + j``.
    """
    dense_mask = aux_tensors[0]
    word_idx = kv_idx[0] >> 5      # kv_idx // 32
    bit_idx = kv_idx[0] & 31       # kv_idx % 32
    word = utils.scalar_to_ssa(dense_mask[batch[0], q_idx[0], word_idx], cutlass.Int32)
    one = utils.scalar_to_ssa(1, cutlass.Int32)
    bit = (word >> bit_idx) & one
    zero = utils.scalar_to_ssa(0, cutlass.Int32)
    return bit != zero


def dense_mask_to_block_sparse(
    dense_mask: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    tile_m: int = 128,
    tile_n: int = 128,
) -> BlockSparseTensorsTorch:
    """Derive block sparsity tensors from a bit-packed (B, Q, ceil(K/32)) mask.

    For each (batch, q_tile, kv_tile), checks whether any words are nonzero.
    Tiles with no nonzero entries are skipped by the kernel.

    Args:
        dense_mask: (B, Q, ceil(K/32)) int32 bit-packed mask from ``pack_mask``.
        max_seqlen_q: Maximum query sequence length (for padding).
        max_seqlen_k: Maximum KV sequence length (for padding).
        tile_m: Query tile size (must match kernel tile_m).
        tile_n: KV tile size (must match kernel tile_n).
            Must be a multiple of 32.

    Returns:
        BlockSparseTensorsTorch with mask_block_cnt and mask_block_idx.
        Head dimension is 1 (broadcasts across heads).
    """
    assert tile_n % 32 == 0, f"tile_n must be a multiple of 32, got {tile_n}"
    B = dense_mask.shape[0]
    words_per_tile = tile_n // 32
    num_m_blocks = ceildiv(max_seqlen_q, tile_m)
    num_n_blocks = ceildiv(max_seqlen_k, tile_n)

    padded_q = num_m_blocks * tile_m
    padded_words = num_n_blocks * words_per_tile
    dm = F.pad(dense_mask,
               (0, padded_words - dense_mask.shape[2],
                0, padded_q - dense_mask.shape[1]))
    # (B, num_m_blocks, tile_m, num_n_blocks, words_per_tile)
    dm = dm.reshape(B, num_m_blocks, tile_m, num_n_blocks, words_per_tile)
    tile_active = (dm != 0).any(dim=4).any(dim=2)  # (B, num_m, num_n)

    # mask_block_cnt: (B, 1, num_m) — count of active KV tiles per Q tile
    mask_block_cnt = tile_active.sum(dim=2).unsqueeze(1).to(torch.int32)
    # mask_block_idx: (B, 1, num_m, num_n) — active tiles sorted first
    sort_keys = (~tile_active).to(torch.int32)
    sorted_indices = sort_keys.argsort(dim=2, stable=True).to(torch.int32)
    mask_block_idx = sorted_indices.unsqueeze(1)

    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=None,
        full_block_idx=None,
        block_size=(tile_m, tile_n),
    )
