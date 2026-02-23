# Tests for top-k sparse mask utilities (flash_attn.cute.topk_mask)
#
# Verifies that the packed bitmask approach correctly implements token-level
# sparse attention on top of block sparsity.
#
# Usage:
#   pytest test_topk_mask.py -v

import math

import pytest
import torch

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.topk_mask import (
    topk_to_bitmask,
    bitmask_to_block_sparse,
    prepare_topk_mask,
)


COMPUTE_CAPABILITY = torch.cuda.get_device_capability()[0]


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
    # For each (batch, query), sample k unique indices from [0, seqlen_k)
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

    q_ref = q.transpose(1, 2).float()  # (B, H, Q, D)
    k_ref = k.transpose(1, 2).float()  # (B, Hkv, K, D)
    v_ref = v.transpose(1, 2).float()  # (B, Hkv, K, Dv)

    if nheads != nheads_kv:
        repeat_factor = nheads // nheads_kv
        k_ref = k_ref.repeat_interleave(repeat_factor, dim=1)
        v_ref = v_ref.repeat_interleave(repeat_factor, dim=1)

    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) * scale  # (B, H, Q, K)

    # Build dense mask from top-k indices: (B, 1, Q, K) -> broadcasts across heads
    dense_mask = torch.zeros(B, 1, seqlen_q, seqlen_k, dtype=torch.bool, device=q.device)
    dense_mask.scatter_(3, topk_indices.unsqueeze(1).expand(-1, 1, -1, -1).long(), True)

    scores.masked_fill_(~dense_mask, float("-inf"))

    # Handle rows where all positions are masked (all -inf)
    all_masked = scores.eq(float("-inf")).all(dim=-1, keepdim=True)
    probs = scores.softmax(dim=-1)
    probs = probs.masked_fill(all_masked, 0.0)

    out = torch.matmul(probs, v_ref)  # (B, H, Q, Dv)
    return out.transpose(1, 2).contiguous()  # (B, Q, H, Dv)


# ---------------------------------------------------------------------------
# Unit tests for bitmask conversion
# ---------------------------------------------------------------------------


class TestBitmaskConversion:
    def test_basic_roundtrip(self):
        """Verify every index in topk_indices sets exactly the right bit."""
        B, Q, k, seqlen_k = 2, 4, 8, 256
        topk = generate_topk_indices(B, Q, seqlen_k, k)
        bitmask = topk_to_bitmask(topk, seqlen_k)

        # Decode bitmask back to indices and compare
        for b in range(B):
            for q in range(Q):
                decoded = set()
                for w in range(bitmask.shape[2]):
                    word = bitmask[b, q, w].item()
                    for bit in range(32):
                        if word & (1 << bit):
                            decoded.add(w * 32 + bit)
                expected = set(topk[b, q].tolist())
                assert decoded == expected, f"Mismatch at b={b}, q={q}"

    def test_all_bits_set(self):
        """When k == seqlen_k, all bits should be set."""
        B, Q, seqlen_k = 1, 2, 64
        topk = generate_topk_indices(B, Q, seqlen_k, seqlen_k)
        bitmask = topk_to_bitmask(topk, seqlen_k)
        # All words should be all-ones (for the valid range)
        for w in range(seqlen_k // 32):
            assert (bitmask[:, :, w] == -1).all()  # -1 in int32 = 0xFFFFFFFF

    def test_non_aligned_seqlen(self):
        """seqlen_k not divisible by 32."""
        B, Q, k, seqlen_k = 1, 1, 5, 50
        topk = generate_topk_indices(B, Q, seqlen_k, k)
        bitmask = topk_to_bitmask(topk, seqlen_k)
        assert bitmask.shape == (1, 1, 2)  # ceil(50/32) = 2


class TestBlockSparseDerivation:
    def test_correct_tile_count(self):
        """Active tile count should match tiles with any selected tokens."""
        B, Q, k, seqlen_k = 1, 128, 16, 512
        tile_m, tile_n = 128, 128
        topk = generate_topk_indices(B, Q, seqlen_k, k)
        bitmask = topk_to_bitmask(topk, seqlen_k)
        bs = bitmask_to_block_sparse(bitmask, Q, seqlen_k, tile_m, tile_n)

        # Manually count active KV tiles for the single Q tile
        active_tiles = set()
        for idx in topk[0].flatten().tolist():
            active_tiles.add(idx // tile_n)

        assert bs.mask_block_cnt[0, 0, 0].item() == len(active_tiles)

    def test_no_full_blocks(self):
        """full_block_cnt and full_block_idx should be None."""
        topk = generate_topk_indices(1, 64, 256, 8)
        bitmask = topk_to_bitmask(topk, 256)
        bs = bitmask_to_block_sparse(bitmask, 64, 256, 64, 128)
        assert bs.full_block_cnt is None
        assert bs.full_block_idx is None

    def test_empty_mask(self):
        """If no tokens are selected, no tiles should be active."""
        B, Q, seqlen_k = 1, 128, 256
        # Zero-element top-k (all -1 or empty)
        bitmask = torch.zeros(B, Q, (seqlen_k + 31) // 32, dtype=torch.int32, device="cuda")
        bs = bitmask_to_block_sparse(bitmask, Q, seqlen_k, 128, 128)
        assert (bs.mask_block_cnt == 0).all()


# ---------------------------------------------------------------------------
# End-to-end kernel tests
# ---------------------------------------------------------------------------


SEQLEN_PAIRS = [
    (128, 256),
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
    """Verify FA4 with top-k bitmask matches dense PyTorch reference."""
    B = 2
    nheads = 4
    nheads_kv = 4
    headdim = 128
    headdim_v = 128
    tile_m = 128
    tile_n = 128

    topk = min(topk, seqlen_k)
    device = "cuda"
    scale = 1.0 / math.sqrt(headdim)

    torch.manual_seed(42)
    q = torch.randn(B, seqlen_q, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(B, seqlen_k, nheads_kv, headdim, dtype=dtype, device=device)
    v = torch.randn(B, seqlen_k, nheads_kv, headdim_v, dtype=dtype, device=device)
    topk_indices = generate_topk_indices(B, seqlen_q, seqlen_k, topk, device=device)

    # Prepare mask
    mask_mod, aux_tensors, block_sparse = prepare_topk_mask(
        topk_indices, seqlen_q, seqlen_k, tile_m, tile_n
    )

    # Run FA4
    out_tuple = _flash_attn_fwd(
        q, k, v,
        softmax_scale=scale,
        causal=False,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        block_sparse_tensors=block_sparse,
        m_block_size=tile_m,
        n_block_size=tile_n,
    )
    out_kernel = out_tuple[0]

    # Reference
    out_ref = compute_reference(q, k, v, topk_indices, scale)

    # Tolerance (matching flash attention test patterns)
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    rtol = 2

    cute_error = (out_kernel.float() - out_ref).abs().max().item()
    ref_self_error = (compute_reference(q, k, v, topk_indices, scale) - out_ref).abs().max().item()

    print(
        f"\ntop-k={topk} @ Q={seqlen_q}, K={seqlen_k}, H={nheads}, D={headdim}"
    )
    print(f"  Kernel error vs FP32 ref: {cute_error:.2e}")
    print(f"  Tolerance: rtol={rtol} * {ref_self_error:.2e} + {fwd_atol:.2e}")

    assert not torch.isnan(out_kernel).any(), "NaN in kernel output"
    assert torch.isfinite(out_kernel).all(), "Inf in kernel output"
    assert cute_error <= rtol * ref_self_error + fwd_atol + 1e-3, (
        f"Kernel error {cute_error:.2e} exceeds tolerance"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or COMPUTE_CAPABILITY < 9,
    reason="Requires SM90+ GPU",
)
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(256, 512)])
@pytest.mark.parametrize("nheads,nheads_kv", [(8, 2), (4, 1)])
def test_topk_mask_gqa(seqlen_q, seqlen_k, nheads, nheads_kv):
    """Verify top-k mask works with GQA (mask broadcasts across heads)."""
    B = 2
    headdim = 128
    topk = 64
    tile_m = 128
    tile_n = 128
    dtype = torch.bfloat16
    device = "cuda"
    scale = 1.0 / math.sqrt(headdim)

    torch.manual_seed(123)
    q = torch.randn(B, seqlen_q, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(B, seqlen_k, nheads_kv, headdim, dtype=dtype, device=device)
    v = torch.randn(B, seqlen_k, nheads_kv, headdim, dtype=dtype, device=device)
    topk_indices = generate_topk_indices(B, seqlen_q, seqlen_k, topk, device=device)

    mask_mod, aux_tensors, block_sparse = prepare_topk_mask(
        topk_indices, seqlen_q, seqlen_k, tile_m, tile_n
    )

    out_tuple = _flash_attn_fwd(
        q, k, v,
        softmax_scale=scale,
        causal=False,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        block_sparse_tensors=block_sparse,
        m_block_size=tile_m,
        n_block_size=tile_n,
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
    B, seqlen_q, seqlen_k = 1, 200, 300
    nheads = nheads_kv = 4
    headdim = 128
    topk = 32
    tile_m = 128
    tile_n = 128
    dtype = torch.bfloat16
    device = "cuda"
    scale = 1.0 / math.sqrt(headdim)

    torch.manual_seed(7)
    q = torch.randn(B, seqlen_q, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(B, seqlen_k, nheads_kv, headdim, dtype=dtype, device=device)
    v = torch.randn(B, seqlen_k, nheads_kv, headdim, dtype=dtype, device=device)
    topk_indices = generate_topk_indices(B, seqlen_q, seqlen_k, topk, device=device)

    mask_mod, aux_tensors, block_sparse = prepare_topk_mask(
        topk_indices, seqlen_q, seqlen_k, tile_m, tile_n
    )

    out_tuple = _flash_attn_fwd(
        q, k, v,
        softmax_scale=scale,
        causal=False,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        block_sparse_tensors=block_sparse,
        m_block_size=tile_m,
        n_block_size=tile_n,
    )
    out_kernel = out_tuple[0]
    out_ref = compute_reference(q, k, v, topk_indices, scale)

    cute_error = (out_kernel.float() - out_ref).abs().max().item()
    print(f"\nNon-aligned Q={seqlen_q}, K={seqlen_k}, top-k={topk}")
    print(f"  Kernel error vs FP32 ref: {cute_error:.2e}")

    assert not torch.isnan(out_kernel).any()
    assert cute_error < 5e-2, f"Non-aligned kernel error {cute_error:.2e} too large"
