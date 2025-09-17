import pytest
import torch

from flash_attn.utils.cpu_block_attn import blockwise_softmax_block_share

# 较小规模，保证测试快速
CASES = [
    # (N, d)
    (32, 16),
    (48, 32),
]

TORCH_DTYPE = torch.float64  # 提高精度，用于对比
SEED = 1234


def _baseline_softmax_rows(Q: torch.Tensor, K: torch.Tensor, causal: bool):
    S = (Q @ K.T) / Q.shape[1]**0.5
    if causal:
        N = S.shape[0]
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float('-inf'))
    P = torch.softmax(S, dim=-1)
    return P


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("N,d", CASES)
@pytest.mark.parametrize("block_size", [8, 16, 33])
def test_blockwise_mass_matches_full_softmax(N, d, causal, block_size):
    torch.manual_seed(SEED + 1)
    Q = torch.randn(N, d, dtype=TORCH_DTYPE)
    K = torch.randn(N, d, dtype=TORCH_DTYPE)

    P = _baseline_softmax_rows(Q, K, causal)  # (N,N)
    Tc = (N + block_size - 1) // block_size
    # 参考每个块的概率质量：将列按块求和
    ref_blocks = []
    for j in range(Tc):
        s = j * block_size
        e = min(N, (j + 1) * block_size)
        ref_blocks.append(P[:, s:e].sum(dim=1))
    refW = torch.stack(ref_blocks, dim=1)  # (N, Tc)

    W = blockwise_softmax_block_share(Q, K, block_size, causal=causal)

    diff = (refW - W).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    # 数值精度：float64 期望 1e-12 量级
    assert max_diff < 1e-12
    assert mean_diff < 1e-13
