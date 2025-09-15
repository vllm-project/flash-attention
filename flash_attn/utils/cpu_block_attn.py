"""CPU block-wise attention utilities.

Provides block-wise two-pass softmax reconstruction and absolute-score share
(W_abs) computation using only O(N * d) extra memory and streaming over blocks.

Main functions:
- blockwise_softmax_lse_pass: first pass computing per-row logsumexp L.
- reconstruct_softmax_blocks: second pass yielding softmax blocks or block sums.
- blockwise_softmax_attention: convenience wrapper to materialize full softmax or block shares.
- blockwise_abs_share_two_pass: two-pass absolute value share W_abs where W_abs[i,j] = |S_ij| / sum_j |S_ij|.
- blockwise_abs_share_online: online per-block share update without storing all previous blocks.

All functions accept an optional callback to stream results instead of materializing a full matrix.

Note: These are CPU reference utilities intended for validation / analysis and
are not optimized for speed. They use NumPy-like PyTorch operations on CPU.
"""
from __future__ import annotations
from typing import Callable, Optional, Tuple, Iterable, List, Union
import math
import torch

BlockCallback = Callable[[Tuple[int,int], Tuple[int,int], torch.Tensor], None]
RowBlockShareCallback = Callable[[Tuple[int,int], torch.Tensor], None]

__all__ = [
    "blockwise_softmax_lse_pass",
    "reconstruct_softmax_blocks",
    "blockwise_softmax_attention",
    "blockwise_abs_share_two_pass",
    "blockwise_abs_share_online",
]

def _apply_causal_mask_inplace(S: torch.Tensor, r0: int, c0: int, global_N: int):
    """In-place causal mask: set S[r,c] = -inf if (c0+c) > (r0+r)."""
    Br, Bc = S.shape
    # Vectorized: build row/col indices
    rows = torch.arange(Br).unsqueeze(1)
    cols = torch.arange(Bc).unsqueeze(0)
    mask = (c0 + cols) > (r0 + rows)
    if mask.any():
        S[mask] = float('-inf')

# ---------- Softmax Two-pass ----------

def blockwise_softmax_lse_pass(Q: torch.Tensor, K: torch.Tensor, Br: int, Bc: int, *, causal: bool=False) -> torch.Tensor:
    """First pass: compute per-row logsumexp L for S = Q K^T with optional causal mask.

    Args:
        Q, K: (N, d) CPU tensors.
        Br, Bc: row/col block sizes.
        causal: apply causal mask (j>i masked).
    Returns:
        L: (N,) tensor where L[i] = logsumexp_j S[i,j].
    """
    assert Q.device.type == 'cpu' and K.device.type == 'cpu', "Use CPU tensors."
    N, d = Q.shape
    assert K.shape == (N,d)
    L = torch.empty(N, dtype=Q.dtype)
    for r0 in range(0, N, Br):
        r1 = min(N, r0+Br)
        Qi = Q[r0:r1]  # (Br,d)
        Br_eff = Qi.shape[0]
        m_local = torch.full((Br_eff,), float('-inf'), dtype=Q.dtype)
        l_local = torch.zeros(Br_eff, dtype=Q.dtype)
        for c0 in range(0, N, Bc):
            c1 = min(N, c0+Bc)
            Kj = K[c0:c1]  # (Bc,d)
            S = Qi @ Kj.T   # (Br_eff, Bc_eff)
            if causal:
                _apply_causal_mask_inplace(S, r0, c0, N)
            block_max = S.max(dim=1).values
            m_new = torch.maximum(m_local, block_max)
            # shift for stability
            S_shift = S - m_new.unsqueeze(1)
            P = torch.exp(S_shift)
            block_sum = P.sum(dim=1)
            l_local = torch.exp(m_local - m_new) * l_local + block_sum
            m_local = m_new
        L[r0:r1] = m_local + torch.log(l_local)
    return L

def reconstruct_softmax_blocks(Q: torch.Tensor, K: torch.Tensor, L: torch.Tensor, Br: int, Bc: int, *, causal: bool=False, full: bool=False, callback: Optional[BlockCallback]=None, return_block_row_sums: bool=False) -> Union[torch.Tensor, List[torch.Tensor], None]:
    """Second pass: given logsumexp L, reconstruct softmax blocks W_block = exp(S - L_row).

    Args:
        Q,K: (N,d) CPU tensors.
        L: (N,) logsumexp per row.
        Br,Bc: block sizes.
        causal: causal mask.
        full: if True, allocate full (N,N) matrix and fill.
        callback: if provided, called with ((r0,r1),(c0,c1), W_block).
        return_block_row_sums: if True, collect per-column-block row sums (list length = num col blocks).
    Returns:
        - full matrix if full=True.
        - list of row-sum tensors if return_block_row_sums=True.
        - None otherwise (stream only).
    """
    N, d = Q.shape
    assert K.shape == (N,d)
    assert L.shape == (N,)
    W_full = torch.zeros(N, N, dtype=Q.dtype) if full else None
    row_block_sums: List[torch.Tensor] = [] if return_block_row_sums else []
    for r0 in range(0, N, Br):
        r1 = min(N, r0+Br)
        Qi = Q[r0:r1]
        L_row = L[r0:r1]
        for c0 in range(0, N, Bc):
            c1 = min(N, c0+Bc)
            Kj = K[c0:c1]
            S = Qi @ Kj.T
            if causal:
                _apply_causal_mask_inplace(S, r0, c0, N)
            W_block = torch.exp(S - L_row.unsqueeze(1))  # (Br_eff, Bc_eff)
            if W_full is not None:
                W_full[r0:r1, c0:c1] = W_block
            if callback is not None:
                callback((r0,r1),(c0,c1), W_block)
            if return_block_row_sums:
                row_block_sums.append(W_block.sum(dim=1))  # list of (Br_eff,)
    if full:
        return W_full
    if return_block_row_sums:
        return row_block_sums
    return None

def blockwise_softmax_attention(Q: torch.Tensor, K: torch.Tensor, Br: int, Bc: int, *, causal: bool=False, want_full: bool=False, want_block_row_shares: bool=False, callback: Optional[BlockCallback]=None):
    """Convenience wrapper returning requested softmax outputs.

    Args:
        want_full: return full (N,N) softmax if True.
        want_block_row_shares: return list of per-block row sums (softmax mass per column block).
    Returns:
        dict with keys maybe: L, full, block_row_sums
    """
    L = blockwise_softmax_lse_pass(Q, K, Br, Bc, causal=causal)
    out: dict = {"L": L}
    res = reconstruct_softmax_blocks(Q, K, L, Br, Bc, causal=causal, full=want_full, callback=callback, return_block_row_sums=want_block_row_shares)
    if want_full:
        out["full"] = res  # type: ignore
    elif want_block_row_shares:
        out["block_row_sums"] = res
    return out

# ---------- Absolute Value Share Two-pass ----------

def blockwise_abs_share_two_pass(Q: torch.Tensor, K: torch.Tensor, Br: int, Bc: int, *, causal: bool=False, want_full: bool=False, callback: Optional[BlockCallback]=None, want_block_row_shares: bool=False):
    """Compute W_abs where W_abs[i,j] = |S_ij| / sum_j |S_ij| using two passes.

    Args similar to blockwise_softmax_attention.
    Returns dict with keys: T (row totals), maybe full, block_row_sums
    """
    assert Q.device.type == 'cpu' and K.device.type == 'cpu'
    N, d = Q.shape
    assert K.shape == (N,d)
    # Pass 1: row totals T
    T = torch.zeros(N, dtype=Q.dtype)
    for r0 in range(0, N, Br):
        r1 = min(N, r0+Br)
        Qi = Q[r0:r1]
        Br_eff = Qi.shape[0]
        row_tot = torch.zeros(Br_eff, dtype=Q.dtype)
        for c0 in range(0, N, Bc):
            c1 = min(N, c0+Bc)
            Kj = K[c0:c1]
            S = Qi @ Kj.T
            if causal:
                _apply_causal_mask_inplace(S, r0, c0, N)
                # masked become -inf; treat as 0 contribution
                S = torch.where(torch.isinf(S), torch.zeros_like(S), S)
            row_tot += S.abs().sum(dim=1)
        T[r0:r1] = row_tot
    # Pass 2: reconstruct shares
    W_full = torch.zeros(N, N, dtype=Q.dtype) if want_full else None
    block_row_sums: List[torch.Tensor] = [] if want_block_row_shares else []
    for r0 in range(0, N, Br):
        r1 = min(N, r0+Br)
        Qi = Q[r0:r1]
        T_row = T[r0:r1]
        for c0 in range(0, N, Bc):
            c1 = min(N, c0+Bc)
            Kj = K[c0:c1]
            S = Qi @ Kj.T
            if causal:
                _apply_causal_mask_inplace(S, r0, c0, N)
                S = torch.where(torch.isinf(S), torch.zeros_like(S), S)
            Wb = S.abs()
            # Normalize by row totals
            nz = T_row > 0
            Wb[nz] /= T_row[nz].unsqueeze(1)
            Wb[~nz] = 0
            if W_full is not None:
                W_full[r0:r1, c0:c1] = Wb
            if callback is not None:
                callback((r0,r1),(c0,c1), Wb)
            if want_block_row_shares:
                block_row_sums.append(Wb.sum(dim=1))
    out = {"T": T}
    if want_full:
        out["full"] = W_full
    if want_block_row_shares:
        out["block_row_sums"] = block_row_sums
    return out

# ---------- Absolute Value Share Online ----------

def blockwise_abs_share_online(Q: torch.Tensor, K: torch.Tensor, Br: int, Bc: int, *, causal: bool=False, callback_block_share: Optional[RowBlockShareCallback]=None):
    """Online streaming of per-column-block absolute value shares.

    Maintains running totals per row. For each new column block with raw
    block mass b_new (row_sum |S_block|), previous emitted shares can be
    rescaled externally if desired using: old_share *= T_prev / (T_prev + b_new)
    while new_share = b_new / (T_prev + b_new).

    This function only outputs new_share each step (not rescaling history),
    leaving any historical adjustment to the caller if needed.

    Args:
        callback_block_share: called with ((r0,r1), share_vector) where share_vector shape (Br_eff,).
    Returns:
        row_totals: final per-row total absolute mass (N,)
    """
    assert Q.device.type == 'cpu' and K.device.type == 'cpu'
    N, d = Q.shape
    assert K.shape == (N,d)
    row_totals = torch.zeros(N, dtype=Q.dtype)
    for r0 in range(0, N, Br):
        r1 = min(N, r0+Br)
        Qi = Q[r0:r1]
        Br_eff = Qi.shape[0]
        T_run = torch.zeros(Br_eff, dtype=Q.dtype)
        for c0 in range(0, N, Bc):
            c1 = min(N, c0+Bc)
            Kj = K[c0:c1]
            S = Qi @ Kj.T
            if causal:
                _apply_causal_mask_inplace(S, r0, c0, N)
                S = torch.where(torch.isinf(S), torch.zeros_like(S), S)
            b_new = S.abs().sum(dim=1)  # (Br_eff,)
            denom = T_run + b_new
            share_new = torch.zeros_like(b_new)
            nz = denom > 0
            share_new[nz] = b_new[nz] / denom[nz]
            # Emit share for this block
            if callback_block_share is not None:
                callback_block_share((r0,r1), share_new)
            T_run += b_new
        row_totals[r0:r1] = T_run
    return row_totals

# ---------- Demo ----------
if __name__ == "__main__":
    torch.manual_seed(0)
    N, d = 32, 16
    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    Br, Bc = 8, 8
    print("Softmax two-pass (block row sums):")
    soft = blockwise_softmax_attention(Q, K, Br, Bc, causal=True, want_block_row_shares=True)
    print("L shape:", soft["L"].shape, "num block row sums:", len(soft["block_row_sums"]))
    print("Abs share two-pass (block row sums):")
    abs_res = blockwise_abs_share_two_pass(Q, K, Br, Bc, causal=True, want_block_row_shares=True)
    print("T shape:", abs_res["T"].shape, "num block row sums:", len(abs_res["block_row_sums"]))
    print("Online abs share streaming:")
    def cb(idx, share):
        r0,r1 = idx
        print(f"  rows[{r0}:{r1}] block_share_mean={share.mean():.4f}")
    totals = blockwise_abs_share_online(Q, K, Br, Bc, causal=True, callback_block_share=cb)
    print("Final totals mean=", totals.mean().item())
