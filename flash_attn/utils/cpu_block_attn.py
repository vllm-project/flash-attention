"""CPU single-pass absolute score share utility (minimal).

Only保留绝对值份额 (abs_s) 计算：

    one_pass_abs_share(Q, K, *, causal=False)

定义：
    S = Q K^T
    若 causal=True，则对 j > i 的位置设为 0（不计入强度）。
    W_abs[i,j] = |S[i,j]| / sum_j |S[i,j]|  （行归一化，若行全 0 则返回该行 0）

注意：本实现会构建完整 (N,N) 矩阵，适用于中小 N 的分析与验证，不做内存优化。
"""
from __future__ import annotations
from typing import Tuple
import torch

__all__ = ["one_pass_abs_share"]


'''
Applies a causal mask to the input tensor S.
'''
def _apply_causal_mask_full(S: torch.Tensor):
    N = S.shape[0]
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    return mask

def one_pass_abs_share(Q: torch.Tensor, K: torch.Tensor, *, causal: bool=False) -> torch.Tensor:
    """计算绝对值份额矩阵 W_abs (单趟, 返回 N x N)。

    Args:
        Q, K: 形状 (N, d) 的 CPU tensor。
        causal: 是否应用因果遮挡；若 True，则列 j>i 位置的分数置 0。

    Returns:
        W_abs: (N,N) 行归一化矩阵；若某行全 0，则该行保持 0。
    """
    assert Q.device.type == 'cpu' and K.device.type == 'cpu', "Use CPU tensors."
    N, d = Q.shape
    assert K.shape == (N, d)
    S = Q @ K.T  # (N,N)
    if causal:
        mask = _apply_causal_mask_full(S)
        S = S.masked_fill(mask, 0.0)
    A = S.abs()
    denom = A.sum(dim=-1, keepdim=True)  # (N,1)
    nonzero = denom > 0
    # 避免除 0：行和为 0 的行保持 0
    W = torch.zeros_like(A)
    if nonzero.any():
        W[nonzero.squeeze(1)] = A[nonzero.squeeze(1)] / denom[nonzero]
    return W

# ---------- Demo ----------
if __name__ == "__main__":
    torch.manual_seed(0)
    N, d = 16, 8
    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    W_abs = one_pass_abs_share(Q, K, causal=True)
    print("Abs share shape:", W_abs.shape, "row0 sum=", float(W_abs[0].sum()))
