# Regression tests for FA2 varlen forward with head dimensions below 32.
#
# HEADDIM_SWITCH only instantiates 32-wide-or-larger tiles, so head_dim 8/16/24
# run through the predicated "uneven-K" path: the out-of-bounds columns of the
# 32-wide shared-memory tiles must be zero-filled. A build that forces
# Is_even_K=true silently reads uninitialized tile columns and returns wrong
# numbers (fp16) or NaNs (bf16). These shapes have no coverage in the rest of
# the suite, where the smallest parametrized head size is 32.

import math

import pytest
import torch

from vllm_flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
    is_fa_version_supported,
)

pytestmark = pytest.mark.skipif(
    not is_fa_version_supported(2), reason="FA2 forward kernel is not available"
)

DTYPES = [torch.float16, torch.bfloat16]
SMALL_HEAD_DIMS = [8, 16, 24]


def naive_varlen_ref(q, k, v, cu_seqlens_q, cu_seqlens_k, scale, causal):
    """Per-sequence fp32 attention reference.

    q, k, v: (total_tokens, nheads/nheads_k, head_dim) on CUDA.
    Returns (total_q, nheads, head_dim) in fp32.
    """
    batch = cu_seqlens_q.numel() - 1
    nheads, nheads_k = q.shape[1], k.shape[1]
    group = nheads // nheads_k
    outs = []
    for i in range(batch):
        qb = q[cu_seqlens_q[i] : cu_seqlens_q[i + 1]].float()
        kb = k[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].float()
        vb = v[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].float()
        kb = kb.repeat_interleave(group, dim=1)
        vb = vb.repeat_interleave(group, dim=1)
        seqlen_q, seqlen_k = qb.shape[0], kb.shape[0]
        # (nheads, seqlen_q, seqlen_k)
        scores = torch.einsum("qhd,shd->hqs", qb, kb) * scale
        if causal:
            qpos = torch.arange(seqlen_q, device=q.device)
            kpos = torch.arange(seqlen_k, device=q.device)
            valid = kpos[None, :] <= (seqlen_k - seqlen_q) + qpos[:, None]
            scores = scores.masked_fill(~valid[None, :, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hqs,shd->hqd", probs, vb).permute(1, 0, 2))
    return torch.cat(outs, dim=0)


def _check_varlen(seq_lens_q, seq_lens_k, nheads, nheads_k, head_dim, dtype, causal):
    device = "cuda"
    torch.manual_seed(0)
    total_q, total_k = sum(seq_lens_q), sum(seq_lens_k)
    q = torch.randn(total_q, nheads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_k, nheads_k, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_k, nheads_k, head_dim, device=device, dtype=dtype)
    cu_seqlens_q = torch.tensor(
        [0] + torch.cumsum(torch.tensor(seq_lens_q), 0).tolist(),
        device=device,
        dtype=torch.int32,
    )
    cu_seqlens_k = torch.tensor(
        [0] + torch.cumsum(torch.tensor(seq_lens_k), 0).tolist(),
        device=device,
        dtype=torch.int32,
    )
    scale = 1.0 / math.sqrt(head_dim)

    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(seq_lens_q),
        max_seqlen_k=max(seq_lens_k),
        softmax_scale=scale,
        causal=causal,
        return_softmax_lse=True,
    )
    ref = naive_varlen_ref(q, k, v, cu_seqlens_q, cu_seqlens_k, scale, causal)

    assert out.shape == (total_q, nheads, head_dim)
    assert torch.isfinite(out).all(), (
        "non-finite output from small-headdim varlen forward"
    )
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", SMALL_HEAD_DIMS)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_varlen_small_headdim(head_dim, causal, dtype):
    # Multi-token prefill over several unpadded sequences, including an
    # unaligned seqlen so the last K/V tile is only partially in bounds.
    _check_varlen(
        seq_lens_q=[128, 64],
        seq_lens_k=[128, 64],
        nheads=8,
        nheads_k=8,
        head_dim=head_dim,
        dtype=dtype,
        causal=causal,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", SMALL_HEAD_DIMS)
def test_flash_attn_varlen_small_headdim_gqa(head_dim, dtype):
    # GQA with uneven sequence lengths exercises the predicated tile loads on
    # the K/V heads shared by groups of query heads.
    _check_varlen(
        seq_lens_q=[37, 64, 129, 11],
        seq_lens_k=[50, 64, 200, 17],
        nheads=8,
        nheads_k=2,
        head_dim=head_dim,
        dtype=dtype,
        causal=False,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", SMALL_HEAD_DIMS)
def test_flash_attn_varlen_small_headdim_decode(head_dim, dtype):
    # max_seqlen_q=1 with GQA takes the ngroups-swap split-KV decode path and
    # its combine kernel, which has its own uneven-K predicates.
    _check_varlen(
        seq_lens_q=[1, 1, 1, 1],
        seq_lens_k=[50, 64, 600, 17],
        nheads=8,
        nheads_k=2,
        head_dim=head_dim,
        dtype=dtype,
        causal=False,
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_flash_attn_varlen_headdim64_guard(dtype):
    # Regular, tile-aligned head dim as a guard against generic regressions.
    _check_varlen(
        seq_lens_q=[128, 64],
        seq_lens_k=[128, 64],
        nheads=8,
        nheads_k=2,
        head_dim=64,
        dtype=dtype,
        causal=False,
    )
