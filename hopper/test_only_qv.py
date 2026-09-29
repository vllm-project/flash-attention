# Copyright (c) 2026, Tri Dao.
"""Tests for the OnlyQv (NoPE MLA) forward path on Hopper.

OnlyQv is triggered when q/k have a zero-width head dim and the full query
content rides in q_v: scores = softmax(scale * (q_v @ v^T)). This is the
GLM-5-style NoPE sparse-MLA geometry (kv_lora_rank=512, qk_rope_head_dim=0).

The FA3 extension is torch.ops-only (no pybind), so the tests call
torch.ops.<lib>.fwd directly; the op is auto-detected from either a standalone
build (flash_attn_3_cuda) or a vLLM build (_vllm_fa3_C).

Regression focus: with a zero-width last dim, no TMA descriptor may be created
for q/k -- cuTensorMapEncodeTiled rejects zero extents, and under NDEBUG the
cutlass failure prints a dump to stderr and hands the kernel a zeroed
descriptor. The tests therefore use *fresh* `torch.empty(..., 0)` tensors
(never views of a wider tensor, whose base pointer would stay 16B-aligned and
mask the failure) and assert that nothing lands on stderr.
"""

import importlib
import math

import pytest
import torch

DEVICE = "cuda"
DV = 512  # kv_lora_rank / v head dim


def _load_fa3():
    """Return the torch.ops namespace of the FA3 extension, or None.

    Probe order prefers a vLLM build (always torch.ops) over a standalone
    flash_attn_3_cuda build; legacy pybind builds (no registered ops) are
    ignored.
    """
    for mod_name, ns in (("vllm.vllm_flash_attn._vllm_fa3_C", "_vllm_fa3_C"),
                         ("flash_attn_3_cuda", "flash_attn_3_cuda")):
        try:
            importlib.import_module(mod_name)
        except ImportError:
            continue
        ops = getattr(torch.ops, ns, None)
        if ops is None:
            continue
        try:
            if ops.fwd is not None:
                return ops
        except AttributeError:
            continue
    return None


fa3 = None


def _fwd(q, k, v, *, qv=None, out=None, cu_seqlens_q=None, cu_seqlens_k=None,
         seqused_k=None, max_seqlen_q=None, max_seqlen_k=None, page_table=None,
         softmax_scale, causal, window_left=-1, window_right=-1, softcap=0.0,
         num_splits=0, pack_gqa=None):
    """Positional call matching the STABLE_TORCH_LIBRARY fwd schema."""
    return fa3.fwd(
        q, k, v,
        None, None, qv,                    # k_new, v_new, q_v
        out,
        cu_seqlens_q, cu_seqlens_k, None,  # cu_seqlens_q/k/k_new
        None, seqused_k,                   # seqused_q, seqused_k
        max_seqlen_q, max_seqlen_k,
        page_table, None, None,            # page_table, kv_batch_idx, leftpad_k
        None, None, None,                  # rotary_cos/sin, seqlens_rotary
        None, None, None,                  # q/k/v_descale
        softmax_scale, causal,
        window_left, window_right, softcap,
        True,                              # is_rotary_interleaved
        None, num_splits, pack_gqa, 0,     # scheduler_metadata, num_splits, pack_gqa, sm_margin
        None, 1, 0, None,                  # s_aux, cp_world_size, cp_rank, cp_tot_seqused_k
    )


def _only_qv_ref(qv, v, softmax_scale, causal=False, q_pos_offset=0):
    """qv: (s_q, h, dv), v: (s_k, h_kv, dv) -> out (s_q, h, dv). MQA-only."""
    s_q, h, dv = qv.shape
    s_k = v.shape[0]
    assert v.shape[1] == 1, "these tests use MQA"
    v = v.expand(-1, h, -1)
    scores = torch.einsum("qhd,thd->hqt", qv.float(), v.float()) * softmax_scale
    if causal:
        # query i attends to keys j <= q_pos_offset + i
        qi = torch.arange(s_q, device=qv.device)[:, None] + q_pos_offset
        ki = torch.arange(s_k, device=qv.device)[None, :]
        scores = scores.masked_fill(ki > qi, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqt,thd->qhd", probs, v.float())


@pytest.fixture(autouse=True)
def _check_only_qv_build():
    global fa3
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("OnlyQv requires Hopper (SM90)")
    fa3 = _load_fa3()
    if fa3 is None:
        pytest.skip("FA3 extension not available (flash_attn_3_cuda or vLLM _vllm_fa3_C)")
    try:
        q = torch.empty(1, 1, 0, device=DEVICE, dtype=torch.bfloat16)
        k = torch.empty(1, 1, 0, device=DEVICE, dtype=torch.bfloat16)
        v = torch.zeros(1, 1, DV, device=DEVICE, dtype=torch.bfloat16)
        qv = torch.zeros(1, 1, DV, device=DEVICE, dtype=torch.bfloat16)
        cu = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
        _fwd(q, k, v, qv=qv, cu_seqlens_q=cu, cu_seqlens_k=cu,
             max_seqlen_q=1, max_seqlen_k=1, softmax_scale=1.0, causal=True)
    except RuntimeError as e:
        if "hdim" in str(e) or "head" in str(e):
            pytest.skip(f"build lacks the OnlyQv (hdim64_dv512) kernels: {e}")
        raise


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [True, False])
def test_only_qv_varlen(dtype, causal):
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

    out, _lse, *_ = _fwd(q, k, v, qv=qv, cu_seqlens_q=cu_q, cu_seqlens_k=cu_q,
                         max_seqlen_q=max(seqlens), max_seqlen_k=max(seqlens),
                         softmax_scale=softmax_scale, causal=causal)
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

    # batch-mode kvcache call: no cu_seqlens_q, seqused_k = cache lengths
    out, _lse, *_ = _fwd(q, k_cache, v_cache, qv=qv, page_table=block_table,
                         seqused_k=cache_seqlens, max_seqlen_q=q_seqlen,
                         softmax_scale=softmax_scale, causal=True)
    assert out.shape == (batch, q_seqlen, nheads, DV)

    for b in range(batch):
        s_k = int(cache_seqlens[b])
        v_full = v_cache[block_table[b]].reshape(-1, 1, DV)[:s_k]
        # query positions are the last q_seqlen positions of the sequence
        ref = _only_qv_ref(qv[b], v_full, softmax_scale, causal=True, q_pos_offset=s_k - q_seqlen)
        torch.testing.assert_close(out[b].float(), ref.to(dtype).float(), rtol=1e-2, atol=1e-2)


def test_only_qv_fresh_zero_width_tensors_no_stderr(capfd):
    """Fresh 0-wide q/k must not trigger a cuTensorMapEncodeTiled failure dump."""
    torch.random.manual_seed(0)
    nheads = 16
    q = torch.empty(3, nheads, 0, device=DEVICE, dtype=torch.bfloat16)  # fresh, not a view
    k = torch.empty(5, 1, 0, device=DEVICE, dtype=torch.bfloat16)  # fresh, not a view
    v = torch.randn(5, 1, DV, device=DEVICE, dtype=torch.bfloat16)
    qv = torch.randn(3, nheads, DV, device=DEVICE, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 1, 3], dtype=torch.int32, device=DEVICE)
    cu_k = torch.tensor([0, 2, 5], dtype=torch.int32, device=DEVICE)
    _fwd(q, k, v, qv=qv, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
         max_seqlen_q=2, max_seqlen_k=3, softmax_scale=1.0 / math.sqrt(DV), causal=True)
    torch.cuda.synchronize()
    err = capfd.readouterr().err
    assert err == "", f"unexpected stderr (TMA descriptor failure dump?): {err!r}"


def test_only_qv_rejects_k_new():
    nheads = 16
    q = torch.empty(1, 1, nheads, 0, device=DEVICE, dtype=torch.bfloat16)
    qv = torch.randn(1, 1, nheads, DV, device=DEVICE, dtype=torch.bfloat16)
    k_cache = torch.empty(1, 256, 1, 0, device=DEVICE, dtype=torch.bfloat16)
    v_cache = torch.randn(1, 256, 1, DV, device=DEVICE, dtype=torch.bfloat16)
    k_new = torch.empty(1, 1, 1, 0, device=DEVICE, dtype=torch.bfloat16)
    v_new = torch.randn(1, 1, 1, DV, device=DEVICE, dtype=torch.bfloat16)
    cache_seqlens = torch.tensor([256], dtype=torch.int32, device=DEVICE)
    with pytest.raises(RuntimeError, match="does not support"):
        fa3.fwd(
            q, k_cache, v_cache,
            k_new, v_new, qv,
            None,
            None, None, None,
            None, cache_seqlens,
            None, None,
            None, None, None,
            None, None, None,
            None, None, None,
            1.0 / math.sqrt(DV), True,
            -1, -1, 0.0,
            True,
            None, 0, None, 0,
            None, 1, 0, None,
        )
