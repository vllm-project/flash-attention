# Copyright (c) 2025, Tri Dao.
# Utilities for converting top-k sparse attention indices into
# packed bitmask + block sparsity tensors compatible with FlashAttention's
# mask_mod / block_sparse_tensors interface.

from typing import Tuple

import cutlass
import cutlass.cute as cute
import torch

from flash_attn.cute import utils
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def topk_to_bitmask(topk_indices: torch.Tensor, seqlen_k: int) -> torch.Tensor:
    """Convert top-k index tensor to packed bitmask.

    Args:
        topk_indices: (B, Q, k) int32 tensor of selected KV positions.
            Indices must be unique per (B, Q) pair and in range [0, seqlen_k).
        seqlen_k: Total KV sequence length.

    Returns:
        (B, Q, ceil(seqlen_k / 32)) int32 packed bitmask where bit j of word i
        indicates that KV position i*32 + j is selected.
    """
    B, Q, k = topk_indices.shape
    num_words = ceildiv(seqlen_k, 32)
    bitmask = torch.zeros(B, Q, num_words, dtype=torch.int32, device=topk_indices.device)
    word_indices = topk_indices // 32
    bit_indices = topk_indices % 32
    # Each bit_value is a distinct power of 2 (since indices are unique),
    # so scatter_add_ is equivalent to scatter_or_.
    bit_values = (1 << bit_indices).to(torch.int32)
    bitmask.scatter_add_(2, word_indices, bit_values)
    return bitmask


def bitmask_to_block_sparse(
    bitmask: torch.Tensor,
    seqlen_q: int,
    seqlen_k: int,
    tile_m: int = 128,
    tile_n: int = 128,
) -> BlockSparseTensorsTorch:
    """Derive block sparsity tensors from a packed bitmask.

    For each (batch, q_tile, kv_tile), checks whether any bits are set.
    Tiles with no set bits are skipped entirely by the kernel.

    Args:
        bitmask: (B, Q, ceil(K/32)) int32 packed bitmask from topk_to_bitmask.
        seqlen_q: Query sequence length.
        seqlen_k: KV sequence length.
        tile_m: Query tile size (must match kernel tile_m).
        tile_n: KV tile size (must match kernel tile_n).

    Returns:
        BlockSparseTensorsTorch with mask_block_cnt and mask_block_idx populated.
        full_block_cnt and full_block_idx are None (all active tiles need mask_mod).
        Head dimension is 1 (broadcasts across heads).
    """
    B = bitmask.shape[0]
    num_m_blocks = ceildiv(seqlen_q, tile_m)
    num_n_blocks = ceildiv(seqlen_k, tile_n)
    words_per_n_block = ceildiv(tile_n, 32)

    # Pad bitmask to align with tile boundaries
    padded_q = num_m_blocks * tile_m
    padded_words = num_n_blocks * words_per_n_block
    bm = bitmask
    if bm.shape[1] < padded_q or bm.shape[2] < padded_words:
        bm = torch.nn.functional.pad(
            bm,
            (0, max(0, padded_words - bm.shape[2]), 0, max(0, padded_q - bm.shape[1])),
        )

    # Reshape: (B, num_m_blocks, tile_m, num_n_blocks, words_per_n_block)
    bm = bm[:, :padded_q, :padded_words]
    bm = bm.reshape(B, num_m_blocks, tile_m, num_n_blocks, words_per_n_block)

    # A tile is active if ANY bit is set across its (tile_m, words_per_n_block) region
    # any() over the tile_m and words_per_n_block dimensions
    tile_active = (bm != 0).any(dim=4).any(dim=2)  # (B, num_m_blocks, num_n_blocks)

    # Build mask_block_cnt and mask_block_idx
    # mask_block_cnt: (B, 1, num_m_blocks) — count of active KV tiles per Q tile
    # mask_block_idx: (B, 1, num_m_blocks, num_n_blocks) — indices of active KV tiles (sorted)
    mask_block_cnt = tile_active.sum(dim=2).unsqueeze(1).to(torch.int32)  # (B, 1, num_m_blocks)

    # For mask_block_idx, we need the active tile indices packed to the left.
    # Use argsort on the negated active mask to push active tiles first.
    # Sort by (not active) to get active tiles at the start, preserving order among them.
    sort_keys = (~tile_active).to(torch.int32)  # 0 for active, 1 for inactive
    sorted_indices = sort_keys.argsort(dim=2, stable=True).to(torch.int32)
    mask_block_idx = sorted_indices.unsqueeze(1)  # (B, 1, num_m_blocks, num_n_blocks)

    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=None,
        full_block_idx=None,
        block_size=(tile_m, tile_n),
    )


@cute.jit
def bitmask_mask_mod(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    q_idx: cute.TensorSSA,
    kv_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors: list,
) -> cute.TensorSSA:
    """Mask mod that reads from a packed int32 bitmask.

    aux_tensors[0] must be a (B, Q, ceil(K/32)) int32 tensor where
    bit j of word i indicates KV position i*32 + j is selected.
    """
    bitmask = aux_tensors[0]
    thirty_two = utils.scalar_to_ssa(32, cutlass.Int32)
    word_idx = kv_idx // thirty_two
    bit_idx = kv_idx % thirty_two
    one = utils.scalar_to_ssa(1, cutlass.Int32)
    zero = utils.scalar_to_ssa(0, cutlass.Int32)
    word = utils.scalar_to_ssa(
        bitmask[batch[0], q_idx[0], word_idx[0]], cutlass.Int32
    )
    bit_mask = one << bit_idx
    return (word & bit_mask) != zero


def prepare_topk_mask(
    topk_indices: torch.Tensor,
    seqlen_q: int,
    seqlen_k: int,
    tile_m: int = 128,
    tile_n: int = 128,
) -> Tuple:
    """Convert top-k indices into mask_mod + aux_tensors + block_sparse_tensors.

    Returns a tuple (mask_mod, aux_tensors, block_sparse_tensors) that can be
    passed directly to _flash_attn_fwd:

        mask_mod, aux_tensors, block_sparse = prepare_topk_mask(topk_indices, seqlen_q, seqlen_k)
        out, lse = _flash_attn_fwd(
            q, k, v,
            mask_mod=mask_mod,
            aux_tensors=aux_tensors,
            block_sparse_tensors=block_sparse,
        )

    Args:
        topk_indices: (B, Q, k) int32 tensor of selected KV positions per query token.
        seqlen_q: Query sequence length.
        seqlen_k: KV sequence length.
        tile_m: Query tile size (default 128, must match kernel).
        tile_n: KV tile size (default 128, must match kernel).

    Returns:
        (mask_mod, aux_tensors, block_sparse_tensors) tuple.
    """
    bitmask = topk_to_bitmask(topk_indices, seqlen_k)
    block_sparse = bitmask_to_block_sparse(bitmask, seqlen_q, seqlen_k, tile_m, tile_n)
    return bitmask_mask_mod, [bitmask], block_sparse
