import time
import pytest
import torch

from flash_attn.utils.cpu_block_attn import one_pass_abs_share

# 设置较小规模，保证测试快速
CASES = [
    # (N, d)
    (32, 16),
    (48, 32),
]

TORCH_DTYPE = torch.float64  # 提高精度，用于对比
SEED = 1234
ABS_SHARE_MAX_ABS_TOL = 1e-12
ABS_SHARE_MEAN_ABS_TOL = 1e-13


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
@pytest.mark.parametrize("N,d", CASES)
def test_abs_share_single_pass_accuracy(N, d, causal):
    torch.manual_seed(SEED + 1)
    Q = torch.randn(N, d, dtype=TORCH_DTYPE)
    K = torch.randn(N, d, dtype=TORCH_DTYPE)
    # baseline
    t0 = time.time()
    W_ref = _baseline_abs_share(Q, K, causal)
    t1 = time.time()
    # single-pass abs share
    t2 = time.time()
    W_single = one_pass_abs_share(Q, K, causal=causal)
    t3 = time.time()
    diff = (W_ref - W_single).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"AbsShare N={N} d={d} causal={causal} baseline={(t1-t0)*1e3:.2f}ms single={(t3-t2)*1e3:.2f}ms max={max_diff:.2e} mean={mean_diff:.2e}")
    assert max_diff < ABS_SHARE_MAX_ABS_TOL
    assert mean_diff < ABS_SHARE_MEAN_ABS_TOL
