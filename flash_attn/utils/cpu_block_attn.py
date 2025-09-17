"""CPU blockwise softmax mass share utility.

仅保留按列块（blockwise）计算 softmax 概率质量份额 W 的实现：

    blockwise_softmax_block_share(Q, K, block_size, *, causal=False) -> W

约定：
- Q, K 为 (N, d) 的 CPU tensor；内部使用缩放 1/sqrt(d)；
- 返回 W 形状为 (N, Tc)，Tc = ceil(N / block_size)，表示每一行在各列块上的 softmax 质量份额；
- 若 causal=True，则仅累计每行 i 的 key 索引 j ≤ i 的质量。
"""
from __future__ import annotations
import torch
__all__ = ["blockwise_softmax_block_share"]


def blockwise_softmax_block_share(
    Q: torch.Tensor,
    K: torch.Tensor,
    block_size: int,
    *,
    causal: bool = False,
) -> torch.Tensor:
    """按 FlashAttention 的在线 LSE 更新，计算每行在各列块上的 softmax 概率质量份额。

    返回 W 形状 (N, Tc)，其中 Tc = ceil(N / block_size)。
    """
    assert Q.device.type == 'cpu' and K.device.type == 'cpu', "Use CPU tensors."
    N, d = Q.shape
    assert K.shape == (N, d)
    Qf = Q.float()
    Kf = K.float()
    scale = (1.0 / d**0.5)

    Tc = (N + block_size - 1) // block_size
    W = torch.zeros(N, Tc, dtype=torch.float32)
    m = torch.full((N,), float('-inf'), dtype=torch.float32)
    l = torch.zeros(N, dtype=torch.float32)
    row_idx = torch.arange(N, dtype=torch.long)

    for j in range(Tc):
        start = j * block_size
        end = min(N, (j + 1) * block_size)
        K_blk = Kf[start:end]
        S_blk = (Qf @ K_blk.T) * scale  # (N, B)

        if causal:
            key_idx = torch.arange(start, end, dtype=torch.long)
            allow = key_idx.unsqueeze(0) <= row_idx.unsqueeze(1)
            S_blk = torch.where(allow, S_blk, torch.full_like(S_blk, float('-inf')))

        blk_max, _ = S_blk.max(dim=1)
        m_new = torch.maximum(m, blk_max)

        exp_term = torch.exp(S_blk - m_new.unsqueeze(1))
        P = exp_term.sum(dim=1)

        alpha = torch.zeros_like(l)
        finite_mask = torch.isfinite(m_new)
        alpha[finite_mask] = torch.exp((m - m_new)[finite_mask])

        l_new = l * alpha + P

        wj = torch.zeros_like(l_new)
        nz = l_new > 0
        wj[nz] = P[nz] / l_new[nz]
        W[:, j] = wj

        m = m_new
        l = l_new

    return W
