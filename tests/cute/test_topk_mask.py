# Tests for sparse mask utilities (flash_attn.cute.topk_mask)
#
# Verifies that dense_mask_to_block_sparse + topk_mask_mod correctly
# implement token-level sparse attention via FA4's mask_mod interface.

import math

import pytest
import torch

try:
    from flash_attn.cute.interface import _flash_attn_fwd
    from flash_attn.cute.topk_mask import (
        topk_to_dense_mask,
        dense_mask_to_block_sparse,
        topk_mask_mod,
    )
except (ImportError, ModuleNotFoundError):
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
    from vllm.vllm_flash_attn.cute.topk_mask import (
        topk_to_dense_mask,
        dense_mask_to_block_sparse,
        topk_mask_mod,
    )


COMPUTE_CAPABILITY = torch.cuda.get_device_capability()[0]
M_BLOCK_SIZE = 128
N_BLOCK_SIZE = 128
Q_STAGE = 2 if COMPUTE_CAPABILITY >= 10 else 1
SPARSE_TILE_M = Q_STAGE * M_BLOCK_SIZE
SPARSE_TILE_N = N_BLOCK_SIZE


@pytest.fixture(autouse=True)
def reset_torch_state():
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    yield
    torch._dynamo.reset()
    torch.cuda.empty_cache()


def generate_topk_indices(B, seqlen_q, seqlen_k, k, device="cuda"):
    """Generate random top-k indices: (B, seqlen_q, k) with unique indices per (B, Q)."""
    k = min(k, seqlen_k)
    topk_indices = torch.stack([
        torch.stack([
            torch.randperm(seqlen_k, device=device)[:k].sort().values
            for _ in range(seqlen_q)
        ])
        for _ in range(B)
    ]).to(torch.int32)
    return topk_indices


def compute_reference(q, k, v, topk_indices, scale):
    """Compute reference attention output with top-k masking using dense PyTorch ops."""
    B, seqlen_q, nheads, headdim = q.shape
    _, seqlen_k, nheads_kv, headdim_v = v.shape

    q_ref = q.transpose(1, 2).float()
    k_ref = k.transpose(1, 2).float()
    v_ref = v.transpose(1, 2).float()

    if nheads != nheads_kv:
        repeat_factor = nheads // nheads_kv
        k_ref = k_ref.repeat_interleave(repeat_factor, dim=1)
        v_ref = v_ref.repeat_interleave(repeat_factor, dim=1)

    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) * scale

    dense_mask = torch.zeros(B, 1, seqlen_q, seqlen_k, dtype=torch.bool, device=q.device)
    dense_mask.scatter_(3, topk_indices.unsqueeze(1).expand(-1, 1, -1, -1).long(), True)
    scores.masked_fill_(~dense_mask, float("-inf"))

    all_masked = scores.eq(float("-inf")).all(dim=-1, keepdim=True)
    probs = scores.softmax(dim=-1)
    probs = probs.masked_fill(all_masked, 0.0)

    out = torch.matmul(probs, v_ref)
    return out.transpose(1, 2).contiguous()


def compute_reference_varlen(q_packed, k_packed, v_packed, cu_seqlens_q, cu_seqlens_k,
                             topk_indices_list, scale):
    """Compute reference for varlen: per-sequence SDPA with topk mask."""
    B = len(cu_seqlens_q) - 1
    outputs = []
    for b in range(B):
        q_start, q_end = cu_seqlens_q[b], cu_seqlens_q[b + 1]
        k_start, k_end = cu_seqlens_k[b], cu_seqlens_k[b + 1]
        sq = q_end - q_start
        sk = k_end - k_start
        nheads = q_packed.shape[1]

        q_b = q_packed[q_start:q_end].unsqueeze(0).transpose(1, 2).float()
        k_b = k_packed[k_start:k_end].unsqueeze(0).transpose(1, 2).float()
        v_b = v_packed[k_start:k_end].unsqueeze(0).transpose(1, 2).float()

        nheads_kv = k_b.shape[1]
        if nheads != nheads_kv:
            k_b = k_b.repeat_interleave(nheads // nheads_kv, dim=1)
            v_b = v_b.repeat_interleave(nheads // nheads_kv, dim=1)

        scores = torch.matmul(q_b, k_b.transpose(-1, -2)) * scale

        topk_idx = topk_indices_list[b]
        dense_mask = torch.zeros(1, 1, sq, sk, dtype=torch.bool, device=q_packed.device)
        dense_mask.scatter_(3, topk_idx.unsqueeze(0).unsqueeze(0).long(), True)
        scores.masked_fill_(~dense_mask, float("-inf"))

        all_masked = scores.eq(float("-inf")).all(dim=-1, keepdim=True)
        probs = scores.softmax(dim=-1)
        probs = probs.masked_fill(all_masked, 0.0)

        out = torch.matmul(probs, v_b)
        outputs.append(out.transpose(1, 2).squeeze(0))

    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Unit tests for dense_mask_to_block_sparse
# ---------------------------------------------------------------------------


class TestDenseMaskToBlockSparse:
    def test_correct_tile_count(self):
        """Active tile count should match tiles with any selected tokens."""
        B, Q, k, seqlen_k = 1, 128, 16, 512
        tile_m, tile_n = 128, 128
        topk = generate_topk_indices(B, Q, seqlen_k, k)
        dense_mask = topk_to_dense_mask(topk, seqlen_k)
        bs = dense_mask_to_block_sparse(dense_mask, Q, seqlen_k, tile_m, tile_n)

        active_tiles = set()
        for idx in topk[0].flatten().tolist():
            active_tiles.add(idx // tile_n)

        assert bs.mask_block_cnt[0, 0, 0].item() == len(active_tiles)

    def test_no_full_blocks(self):
        """full_block_cnt and full_block_idx should be None."""
        topk = generate_topk_indices(1, 64, 256, 8)
        dense_mask = topk_to_dense_mask(topk, 256)
        bs = dense_mask_to_block_sparse(dense_mask, 64, 256, 64, 128)
        assert bs.full_block_cnt is None
        assert bs.full_block_idx is None

    def test_empty_mask(self):
        """If no tokens are selected, no tiles should be active."""
        B, Q, seqlen_k = 1, 128, 256
        dense_mask = torch.zeros(B, Q, seqlen_k, dtype=torch.int32, device="cuda")
        bs = dense_mask_to_block_sparse(dense_mask, Q, seqlen_k, 128, 128)
        assert (bs.mask_block_cnt == 0).all()


# ---------------------------------------------------------------------------
# End-to-end kernel tests (non-varlen)
# ---------------------------------------------------------------------------


SEQLEN_PAIRS = [
    (256, 256),
    (256, 512),
    (512, 1024),
    (1024, 2048),
]

TOPK_VALUES = [32, 64, 128]


@pytest.mark.skipif(
    not torch.cuda.is_available() or COMPUTE_CAPABILITY < 9,
    reason="Requires SM90+ GPU",
)
@pytest.mark.parametrize("seqlen_q,seqlen_k", SEQLEN_PAIRS)
@pytest.mark.parametrize("topk", TOPK_VALUES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_topk_mask_correctness(seqlen_q, seqlen_k, topk, dtype):
    """Verify FA4 with top-k mask matches dense PyTorch reference."""
    B, nheads, nheads_kv, headdim = 2, 4, 4, 128
    tile_m, tile_n = SPARSE_TILE_M, SPARSE_TILE_N
    topk = min(topk, seqlen_k)
    device = "cuda"
    scale = 1.0 / math.sqrt(headdim)

    torch.manual_seed(42)
    q = torch.randn(B, seqlen_q, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(B, seqlen_k, nheads_kv, headdim, dtype=dtype, device=device)
    v = torch.randn(B, seqlen_k, nheads_kv, headdim, dtype=dtype, device=device)
    topk_indices = generate_topk_indices(B, seqlen_q, seqlen_k, topk, device=device)

    dense_mask = topk_to_dense_mask(topk_indices, seqlen_k)
    block_sparse = dense_mask_to_block_sparse(dense_mask, seqlen_q, seqlen_k, tile_m, tile_n)

    out_tuple = _flash_attn_fwd(
        q, k, v,
        softmax_scale=scale,
        causal=False,
        mask_mod=topk_mask_mod,
        aux_tensors=[dense_mask],
        block_sparse_tensors=block_sparse,
        m_block_size=M_BLOCK_SIZE,
        n_block_size=N_BLOCK_SIZE,
    )
    out_kernel = out_tuple[0]
    out_ref = compute_reference(q, k, v, topk_indices, scale)

    cute_error = (out_kernel.float() - out_ref).abs().max().item()
    print(f"\ntop-k={topk} @ Q={seqlen_q}, K={seqlen_k}, H={nheads}, D={headdim}")
    print(f"  Kernel error vs FP32 ref: {cute_error:.2e}")

    assert not torch.isnan(out_kernel).any(), "NaN in kernel output"
    assert torch.isfinite(out_kernel).all(), "Inf in kernel output"
    assert cute_error < 5e-2, f"Kernel error {cute_error:.2e} too large"


@pytest.mark.skipif(
    not torch.cuda.is_available() or COMPUTE_CAPABILITY < 9,
    reason="Requires SM90+ GPU",
)
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(256, 512)])
@pytest.mark.parametrize("nheads,nheads_kv", [(8, 2), (4, 1)])
def test_topk_mask_gqa(seqlen_q, seqlen_k, nheads, nheads_kv):
    """Verify top-k mask works with GQA (mask broadcasts across heads)."""
    B, headdim, topk = 2, 128, 64
    tile_m, tile_n = SPARSE_TILE_M, SPARSE_TILE_N
    dtype, device = torch.bfloat16, "cuda"
    scale = 1.0 / math.sqrt(headdim)

    torch.manual_seed(123)
    q = torch.randn(B, seqlen_q, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(B, seqlen_k, nheads_kv, headdim, dtype=dtype, device=device)
    v = torch.randn(B, seqlen_k, nheads_kv, headdim, dtype=dtype, device=device)
    topk_indices = generate_topk_indices(B, seqlen_q, seqlen_k, topk, device=device)

    dense_mask = topk_to_dense_mask(topk_indices, seqlen_k)
    block_sparse = dense_mask_to_block_sparse(dense_mask, seqlen_q, seqlen_k, tile_m, tile_n)

    out_tuple = _flash_attn_fwd(
        q, k, v,
        softmax_scale=scale,
        causal=False,
        mask_mod=topk_mask_mod,
        aux_tensors=[dense_mask],
        block_sparse_tensors=block_sparse,
        m_block_size=M_BLOCK_SIZE,
        n_block_size=N_BLOCK_SIZE,
    )
    out_kernel = out_tuple[0]
    out_ref = compute_reference(q, k, v, topk_indices, scale)

    cute_error = (out_kernel.float() - out_ref).abs().max().item()
    print(f"\nGQA top-k={topk} @ Q={seqlen_q}, K={seqlen_k}, H={nheads}/{nheads_kv}")
    print(f"  Kernel error vs FP32 ref: {cute_error:.2e}")

    assert not torch.isnan(out_kernel).any()
    assert cute_error < 5e-2, f"GQA kernel error {cute_error:.2e} too large"


@pytest.mark.skipif(
    not torch.cuda.is_available() or COMPUTE_CAPABILITY < 9,
    reason="Requires SM90+ GPU",
)
def test_topk_mask_non_aligned_seqlen():
    """Test with sequence lengths not divisible by tile sizes."""
    B, seqlen_q, seqlen_k = 1, 300, 500
    nheads = nheads_kv = 4
    headdim, topk = 128, 32
    tile_m, tile_n = SPARSE_TILE_M, SPARSE_TILE_N
    dtype, device = torch.bfloat16, "cuda"
    scale = 1.0 / math.sqrt(headdim)

    torch.manual_seed(7)
    q = torch.randn(B, seqlen_q, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(B, seqlen_k, nheads_kv, headdim, dtype=dtype, device=device)
    v = torch.randn(B, seqlen_k, nheads_kv, headdim, dtype=dtype, device=device)
    topk_indices = generate_topk_indices(B, seqlen_q, seqlen_k, topk, device=device)

    dense_mask = topk_to_dense_mask(topk_indices, seqlen_k)
    block_sparse = dense_mask_to_block_sparse(dense_mask, seqlen_q, seqlen_k, tile_m, tile_n)

    out_tuple = _flash_attn_fwd(
        q, k, v,
        softmax_scale=scale,
        causal=False,
        mask_mod=topk_mask_mod,
        aux_tensors=[dense_mask],
        block_sparse_tensors=block_sparse,
        m_block_size=M_BLOCK_SIZE,
        n_block_size=N_BLOCK_SIZE,
    )
    out_kernel = out_tuple[0]
    out_ref = compute_reference(q, k, v, topk_indices, scale)

    cute_error = (out_kernel.float() - out_ref).abs().max().item()
    print(f"\nNon-aligned Q={seqlen_q}, K={seqlen_k}, top-k={topk}")
    print(f"  Kernel error vs FP32 ref: {cute_error:.2e}")

    assert not torch.isnan(out_kernel).any()
    assert cute_error < 5e-2, f"Non-aligned kernel error {cute_error:.2e} too large"


# ---------------------------------------------------------------------------
# Varlen tests (mask_mod + block_sparse with cu_seqlens)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available() or COMPUTE_CAPABILITY < 9,
    reason="Requires SM90+ GPU",
)
@pytest.mark.parametrize("seqlens_q,seqlens_k", [
    ([256, 512], [512, 1024]),
    ([512, 256, 256], [1024, 512, 512]),
    ([300, 400], [500, 800]),
])
@pytest.mark.parametrize("topk", [32, 64])
def test_topk_mask_varlen(seqlens_q, seqlens_k, topk):
    """Verify mask_mod + block_sparse works with varlen (cu_seqlens)."""
    B = len(seqlens_q)
    nheads = nheads_kv = 4
    headdim = 128
    tile_m, tile_n = SPARSE_TILE_M, SPARSE_TILE_N
    dtype, device = torch.bfloat16, "cuda"
    scale = 1.0 / math.sqrt(headdim)

    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)

    torch.manual_seed(42)

    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)
    q_packed = torch.randn(total_q, nheads, headdim, dtype=dtype, device=device)
    k_packed = torch.randn(total_k, nheads_kv, headdim, dtype=dtype, device=device)
    v_packed = torch.randn(total_k, nheads_kv, headdim, dtype=dtype, device=device)

    cu_seqlens_q = torch.tensor([0] + list(torch.cumsum(torch.tensor(seqlens_q), 0)),
                                dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0] + list(torch.cumsum(torch.tensor(seqlens_k), 0)),
                                dtype=torch.int32, device=device)

    # Build per-sequence topk indices and padded dense mask
    topk_indices_list = []
    dense_mask_padded = torch.zeros(B, max_seqlen_q, max_seqlen_k,
                                    dtype=torch.int32, device=device)
    for b in range(B):
        sq, sk = seqlens_q[b], seqlens_k[b]
        actual_topk = min(topk, sk)
        idx = generate_topk_indices(1, sq, sk, actual_topk, device=device)[0]
        topk_indices_list.append(idx)
        seq_dense = topk_to_dense_mask(idx.unsqueeze(0), sk)
        dense_mask_padded[b, :sq, :sk] = seq_dense[0]

    block_sparse = dense_mask_to_block_sparse(dense_mask_padded, max_seqlen_q, max_seqlen_k,
                                              tile_m, tile_n)

    out_tuple = _flash_attn_fwd(
        q_packed, k_packed, v_packed,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=scale,
        causal=False,
        mask_mod=topk_mask_mod,
        aux_tensors=[dense_mask_padded],
        block_sparse_tensors=block_sparse,
        m_block_size=M_BLOCK_SIZE,
        n_block_size=N_BLOCK_SIZE,
    )
    out_kernel = out_tuple[0]

    out_ref = compute_reference_varlen(q_packed, k_packed, v_packed,
                                       cu_seqlens_q.tolist(), cu_seqlens_k.tolist(),
                                       topk_indices_list, scale)

    cute_error = (out_kernel.float() - out_ref).abs().max().item()
    print(f"\nVarlen topk={topk} seqlens_q={seqlens_q} seqlens_k={seqlens_k}")
    print(f"  Kernel error vs FP32 ref: {cute_error:.2e}")

    assert not torch.isnan(out_kernel).any(), "NaN in varlen kernel output"
    assert torch.isfinite(out_kernel).all(), "Inf in varlen kernel output"
    assert cute_error < 5e-2, f"Varlen kernel error {cute_error:.2e} too large"
