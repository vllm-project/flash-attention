# Copyright (c) 2026, Tri Dao.
"""Tests for the OnlyQv (NoPE MLA) forward path on Hopper.

OnlyQv is triggered when q/k have a zero-width head dim and the full query
content rides in qv: scores = softmax(scale * (qv @ v^T)). This is the
GLM-5-style NoPE sparse-MLA geometry (kv_lora_rank=512, qk_rope_head_dim=0).

Regression focus: with a zero-width last dim, no TMA descriptor may be created
for q/k -- cuTensorMapEncodeTiled rejects zero extents, and under NDEBUG the
cutlass failure prints a dump to stderr and hands the kernel a zeroed
descriptor. The tests therefore use *fresh* `torch.empty(..., 0)` tensors
(never views of a wider tensor, whose base pointer would stay 16B-aligned and
mask the failure) and assert that nothing lands on stderr.
"""

import math

import pytest
import torch

from flash_attn_interface import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)

DEVICE = "cuda"
DV = 512  # kv_lora_rank / v head dim


def _requires_sm90():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("OnlyQv requires Hopper (SM90)")


def _requires_hdimdiff_build():
    """Skip on builds compiled with FLASHATTENTION_DISABLE_HDIMDIFF64 (the
    default), which lack the hdim64_dv{256,512} kernels OnlyQv dispatches to."""
    try:
        q = torch.empty(1, 1, 0, device=DEVICE, dtype=torch.bfloat16)
        k = torch.empty(1, 1, 0, device=DEVICE, dtype=torch.bfloat16)
        v = torch.zeros(1, 1, DV, device=DEVICE, dtype=torch.bfloat16)
        qv = torch.zeros(1, 1, DV, device=DEVICE, dtype=torch.bfloat16)
        cu = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
        flash_attn_varlen_func(q, k, v, cu, cu, 1, 1, qv=qv, softmax_scale=1.0)
    except RuntimeError as e:
        if "hdim != hdim_v" in str(e):
            pytest.skip("build lacks hdimdiff64 kernels (FLASHATTENTION_DISABLE_HDIMDIFF64)")
        raise


@pytest.fixture(autouse=True)
def _check_only_qv_build():
    _requires_sm90()
    _requires_hdimdiff_build()


def _only_qv_ref(qv, v, softmax_scale, causal=False, q_pos_offset=0):
    """qv: (s_q, h, dv), v: (s_k, h_kv, dv) -> out (s_q, h, dv). MQA-only."""
    s_q, h, dv = qv.shape
    s_k = v.shape[0]
    assert v.shape[1] == 1, "these tests use MQA"
    v = v.expand(-1, h, -1)
    scores = torch.einsum("qhd,thd->hqt", qv.float(), v.float())
    scores *= softmax_scale
    if causal:
        # query i attends to keys j <= q_pos_offset + i
        qi = torch.arange(s_q, device=qv.device)[:, None] + q_pos_offset
        ki = torch.arange(s_k, device=qv.device)[None, :]
        scores = scores.masked_fill(ki > qi, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqt,thd->qhd", probs, v.float())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [True, False])
def test_only_qv_varlen(dtype, causal):
    _requires_sm90()
    torch.random.manual_seed(0)
    nheads = 16
    seqlens = [1, 3, 128, 257]
    cu_q = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0).tolist()), dtype=torch.int32, device=DEVICE)
    total_q = int(cu_q[-1])
    qv = torch.randn(total_q, nheads, DV, device=DEVICE, dtype=dtype)
    v = torch.randn(total_q, 1, DV, device=DEVICE, dtype=dtype)  # MQA, s_k == s_q here
    q = torch.empty(total_q, nheads, 0, device=DEVICE, dtype=dtype)  # fresh, not a view
    k = torch.empty(total_q, 1, 0, device=DEVICE, dtype=dtype)
    softmax_scale = 1.0 / math.sqrt(DV)

    out, _lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_q,
        max_seqlen_q=max(seqlens),
        max_seqlen_k=max(seqlens),
        qv=qv,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    assert out.shape == (total_q, nheads, DV) and out.dtype == dtype

    outs, refs = [], []
    for i, s in enumerate(seqlens):
        sl = slice(int(cu_q[i]), int(cu_q[i + 1]))
        outs.append(out[sl].float())
        refs.append(_only_qv_ref(qv[sl], v[sl], softmax_scale, causal=causal))
    out_f, ref = torch.cat(outs), torch.cat(refs)
    torch.testing.assert_close(out_f, ref.to(dtype).float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_only_qv_paged_kvcache(dtype):
    _requires_sm90()
    torch.random.manual_seed(0)
    nheads, page_size = 16, 64
    batch, max_ctx = 3, 300
    num_pages_per_req = math.ceil(max_ctx / page_size)
    num_blocks = batch * num_pages_per_req
    # q seqlen > 1 exercises the spec-decode-style path
    q_seqlen = 2
    cache_seqlens = torch.tensor([77, 200, 300], dtype=torch.int32, device=DEVICE)

    block_table = torch.arange(num_blocks, dtype=torch.int32, device=DEVICE).reshape(batch, num_pages_per_req)
    k_cache = torch.empty(num_blocks, page_size, 1, 0, device=DEVICE, dtype=dtype)  # fresh, not a view
    v_cache = torch.randn(num_blocks, page_size, 1, DV, device=DEVICE, dtype=dtype)
    q = torch.empty(batch, q_seqlen, nheads, 0, device=DEVICE, dtype=dtype)
    qv = torch.randn(batch, q_seqlen, nheads, DV, device=DEVICE, dtype=dtype)
    softmax_scale = 1.0 / math.sqrt(DV)

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        qv=qv,
        page_table=block_table,
        cache_seqlens=cache_seqlens,
        causal=True,
        softmax_scale=softmax_scale,
    )
    assert out.shape == (batch, q_seqlen, nheads, DV)

    for b in range(batch):
        s_k = int(cache_seqlens[b])
        v_full = v_cache[block_table[b]].reshape(-1, 1, DV)[:s_k]
        # query positions are the last q_seqlen positions of the sequence
        ref = _only_qv_ref(qv[b], v_full, softmax_scale, causal=True, q_pos_offset=s_k - q_seqlen)
        torch.testing.assert_close(out[b].float(), ref.to(dtype).float(), rtol=1e-2, atol=1e-2)


def test_only_qv_fresh_zero_width_tensors_no_stderr(capfd):
    """Fresh 0-wide q/k must not trigger a cuTensorMapEncodeTiled failure dump."""
    _requires_sm90()
    torch.random.manual_seed(0)
    nheads = 16
    q = torch.empty(3, nheads, 0, device=DEVICE, dtype=torch.bfloat16)  # fresh, not a view
    k = torch.empty(5, 1, 0, device=DEVICE, dtype=torch.bfloat16)  # fresh, not a view
    v = torch.randn(5, 1, DV, device=DEVICE, dtype=torch.bfloat16)
    qv = torch.randn(3, nheads, DV, device=DEVICE, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 1, 3], dtype=torch.int32, device=DEVICE)
    cu_k = torch.tensor([0, 2, 5], dtype=torch.int32, device=DEVICE)
    flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=2,
        max_seqlen_k=3,
        qv=qv,
        causal=True,
        softmax_scale=1.0 / math.sqrt(DV),
    )
    torch.cuda.synchronize()
    err = capfd.readouterr().err
    assert err == "", f"unexpected stderr (TMA descriptor failure dump?): {err!r}"


def test_only_qv_rejects_k_new():
    _requires_sm90()
    nheads = 16
    q = torch.empty(1, 1, nheads, 0, device=DEVICE, dtype=torch.bfloat16)
    qv = torch.randn(1, 1, nheads, DV, device=DEVICE, dtype=torch.bfloat16)
    k_cache = torch.empty(1, 256, 1, 0, device=DEVICE, dtype=torch.bfloat16)
    v_cache = torch.randn(1, 256, 1, DV, device=DEVICE, dtype=torch.bfloat16)
    k_new = torch.empty(1, 1, 1, 0, device=DEVICE, dtype=torch.bfloat16)
    v_new = torch.randn(1, 1, 1, DV, device=DEVICE, dtype=torch.bfloat16)
    cache_seqlens = torch.tensor([256], dtype=torch.int32, device=DEVICE)
    with pytest.raises(RuntimeError, match="does not support"):
        flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k_new,
            v=v_new,
            qv=qv,
            cache_seqlens=cache_seqlens,
            softmax_scale=1.0 / math.sqrt(DV),
        )
