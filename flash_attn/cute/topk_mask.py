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


def topk_to_dense_mask(topk_indices: torch.Tensor, seqlen_k: int) -> torch.Tensor:
    """Convert top-k index tensor to dense int32 mask.

    Args:
        topk_indices: (B, Q, k) int32 tensor of selected KV positions.
        seqlen_k: Total KV sequence length.

    Returns:
        (B, Q, seqlen_k) int32 tensor with 1 at selected positions, 0 elsewhere.
    """
    B, Q, k = topk_indices.shape
    mask = torch.zeros(B, Q, seqlen_k, dtype=torch.int32, device=topk_indices.device)
    mask.scatter_(2, topk_indices.long(), 1)
    return mask


@cute.jit
def topk_mask_mod(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    q_idx: cute.TensorSSA,
    kv_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors: list,
) -> cute.TensorSSA:
    """Mask mod that reads from a dense int32 mask.

    aux_tensors[0] must be a (B, Q, K) int32 tensor with 1 at selected
    positions and 0 elsewhere.
    """
    mask = aux_tensors[0]
    val = utils.scalar_to_ssa(mask[batch[0], q_idx[0], kv_idx[0]], cutlass.Int32)
    zero = utils.scalar_to_ssa(0, cutlass.Int32)
    return val != zero


def dense_mask_to_block_sparse(
    dense_mask: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    tile_m: int = 128,
    tile_n: int = 128,
) -> BlockSparseTensorsTorch:
    """Derive block sparsity tensors from a dense (B, Q, K) int32 mask.

    For each (batch, q_tile, kv_tile), checks whether any entries are nonzero.
    Tiles with no nonzero entries are skipped by the kernel.

    Args:
        dense_mask: (B, Q, K) int32 mask (1 = attend, 0 = skip).
        max_seqlen_q: Maximum query sequence length (for padding).
        max_seqlen_k: Maximum KV sequence length (for padding).
        tile_m: Query tile size (must match kernel tile_m).
        tile_n: KV tile size (must match kernel tile_n).

    Returns:
        BlockSparseTensorsTorch with mask_block_cnt and mask_block_idx.
        Head dimension is 1 (broadcasts across heads).
    """
    B = dense_mask.shape[0]
    num_m_blocks = ceildiv(max_seqlen_q, tile_m)
    num_n_blocks = ceildiv(max_seqlen_k, tile_n)

    padded_q = num_m_blocks * tile_m
    padded_k = num_n_blocks * tile_n
    dm = F.pad(dense_mask,
               (0, padded_k - dense_mask.shape[2],
                0, padded_q - dense_mask.shape[1]))
    # (B, num_m_blocks, tile_m, num_n_blocks, tile_n)
    dm = dm.reshape(B, num_m_blocks, tile_m, num_n_blocks, tile_n)
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
