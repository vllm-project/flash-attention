import time
import math
import pytest
import torch

from flash_attn.utils.cpu_block_attn import (
    blockwise_softmax_attention,
    blockwise_abs_share_two_pass,
)

# 设置较小规模，保证测试快速
CASES = [
    # (N, d, Br, Bc)
    (32, 16, 8, 8),
    (48, 32, 12, 16),
]

TORCH_DTYPE = torch.float64  # 提高精度，用于对比
SEED = 1234
SOFTMAX_MAX_ABS_TOL = 1e-10
SOFTMAX_MEAN_ABS_TOL = 1e-12
ABS_SHARE_MAX_ABS_TOL = 1e-12
ABS_SHARE_MEAN_ABS_TOL = 1e-13


def _baseline_softmax(Q: torch.Tensor, K: torch.Tensor, causal: bool):
    S = Q @ K.T
    if causal:
        N = S.shape[0]
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float('-inf'))
    return torch.softmax(S, dim=-1)


def _baseline_abs_share(Q: torch.Tensor, K: torch.Tensor, causal: bool):
    S = Q @ K.T
    if causal:
        N = S.shape[0]
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, 0.0)
    A = S.abs()
    row_tot = A.sum(dim=-1, keepdim=True)
    row_tot = torch.where(row_tot == 0, torch.ones_like(row_tot), row_tot)
    return A / row_tot


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("N,d,Br,Bc", CASES)
def test_blockwise_softmax_accuracy(N, d, Br, Bc, causal):
    torch.manual_seed(SEED)
    Q = torch.randn(N, d, dtype=TORCH_DTYPE)
    K = torch.randn(N, d, dtype=TORCH_DTYPE)
    # baseline
    t0 = time.time()
    W_ref = _baseline_softmax(Q, K, causal)
    t1 = time.time()
    # blockwise
    t2 = time.time()
    res = blockwise_softmax_attention(Q, K, Br, Bc, causal=causal, want_full=True)
    W_blk = res["full"]
    t3 = time.time()
    diff = (W_ref - W_blk).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"Softmax N={N} d={d} causal={causal} baseline={(t1-t0)*1e3:.2f}ms block={(t3-t2)*1e3:.2f}ms max={max_diff:.2e} mean={mean_diff:.2e}")
    assert max_diff < SOFTMAX_MAX_ABS_TOL
    assert mean_diff < SOFTMAX_MEAN_ABS_TOL


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("N,d,Br,Bc", CASES)
def test_blockwise_abs_share_accuracy(N, d, Br, Bc, causal):
    torch.manual_seed(SEED + 1)
    Q = torch.randn(N, d, dtype=TORCH_DTYPE)
    K = torch.randn(N, d, dtype=TORCH_DTYPE)
    # baseline
    t0 = time.time()
    W_ref = _baseline_abs_share(Q, K, causal)
    t1 = time.time()
    # blockwise
    t2 = time.time()
    res = blockwise_abs_share_two_pass(Q, K, Br, Bc, causal=causal, want_full=True)
    W_blk = res["full"]
    t3 = time.time()
    diff = (W_ref - W_blk).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"AbsShare N={N} d={d} causal={causal} baseline={(t1-t0)*1e3:.2f}ms block={(t3-t2)*1e3:.2f}ms max={max_diff:.2e} mean={mean_diff:.2e}")
    assert max_diff < ABS_SHARE_MAX_ABS_TOL
    assert mean_diff < ABS_SHARE_MEAN_ABS_TOL
