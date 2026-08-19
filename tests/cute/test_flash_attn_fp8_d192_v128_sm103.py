import inspect
import math
import sys
import types

import pytest
import torch

# flash_attn/__init__.py eagerly imports the optional FA2 extension, which is not
# needed by the CuTe tests in an FA4-only development environment.
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))

from flash_attn.cute import flash_attn_func, flash_attn_varlen_func


def test_public_varlen_api_exposes_only_the_precision_knob():
    assert (
        "_varlen_num_blocks" not in inspect.signature(flash_attn_varlen_func).parameters
    )
    assert (
        "k3_rescale_threshold"
        not in inspect.signature(flash_attn_varlen_func).parameters
    )
    threshold = inspect.signature(flash_attn_varlen_func).parameters[
        "fp8_rescale_threshold"
    ]
    assert threshold.default is None


def _is_sm103_available():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3)


def _is_sm10x_available():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def _cumulative(lengths, device):
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, dtype=torch.int32, device=device)


def _target_tensors(device, dtype=torch.float8_e4m3fn):
    q = torch.zeros(384, 2, 192, dtype=dtype, device=device)
    k = torch.zeros(512, 2, 192, dtype=dtype, device=device)
    v = torch.zeros(512, 2, 128, dtype=dtype, device=device)
    return q, k, v


@pytest.mark.skipif(not _is_sm103_available(), reason="requires an SM103 GPU")
def test_fp8_rescale_threshold_rejects_invalid_value():
    device = torch.device("cuda")
    q, k, v = _target_tensors(device)
    with pytest.raises(AssertionError, match="must be one of"):
        flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=_cumulative((384,), device),
            cu_seqlens_k=_cumulative((512,), device),
            max_seqlen_q=384,
            max_seqlen_k=512,
            causal=True,
            fp8_rescale_threshold=1.0,
        )


@pytest.mark.skipif(not _is_sm103_available(), reason="requires an SM103 GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e5m2])
def test_fp8_rescale_threshold_rejects_non_k3_configuration(dtype):
    device = torch.device("cuda")
    q, k, v = _target_tensors(device, dtype=dtype)
    with pytest.raises(AssertionError, match="only supported"):
        flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=_cumulative((384,), device),
            cu_seqlens_k=_cumulative((512,), device),
            max_seqlen_q=384,
            max_seqlen_k=512,
            causal=True,
            fp8_rescale_threshold=0.75,
        )


@pytest.mark.skipif(not _is_sm103_available(), reason="requires an SM103 GPU")
@pytest.mark.parametrize("aux_kind", ["tensor", "scalar"])
def test_fp8_rescale_threshold_rejects_k3_unsupported_aux(aux_kind):
    device = torch.device("cuda")
    q, k, v = _target_tensors(device)
    extra_kwargs = (
        {"aux_tensors": [torch.empty(1, device=device)]}
        if aux_kind == "tensor"
        else {"aux_scalars": (1.0,)}
    )
    with pytest.raises(AssertionError, match="only supported"):
        flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=_cumulative((384,), device),
            cu_seqlens_k=_cumulative((512,), device),
            max_seqlen_q=384,
            max_seqlen_k=512,
            causal=True,
            fp8_rescale_threshold=0.75,
            **extra_kwargs,
        )


def _reference(q, k, v, q_lengths, kv_lengths, softmax_scale):
    outputs = []
    lses = []
    q_offset = 0
    kv_offset = 0
    group_size = q.shape[1] // k.shape[1]
    for q_len, kv_len in zip(q_lengths, kv_lengths, strict=True):
        q_i = q[q_offset : q_offset + q_len].float()
        k_i = (
            k[kv_offset : kv_offset + kv_len]
            .float()
            .repeat_interleave(group_size, dim=1)
        )
        v_i = (
            v[kv_offset : kv_offset + kv_len]
            .float()
            .repeat_interleave(group_size, dim=1)
        )
        scores = torch.einsum("qhd,khd->qkh", q_i, k_i) * softmax_scale
        q_idx = torch.arange(q_len, device=q.device).view(-1, 1)
        kv_idx = torch.arange(kv_len, device=q.device).view(1, -1)
        scores.masked_fill_(
            (kv_idx > q_idx + kv_len - q_len).unsqueeze(-1), float("-inf")
        )
        probabilities = torch.nan_to_num(scores.softmax(dim=1))
        outputs.append(torch.einsum("qkh,khd->qhd", probabilities, v_i))
        lses.append(scores.logsumexp(dim=1))
        q_offset += q_len
        kv_offset += kv_len
    return torch.cat(outputs), torch.cat(lses)


@pytest.mark.skipif(not _is_sm103_available(), reason="requires an SM103 GPU")
@pytest.mark.parametrize("fp8_rescale_threshold", [0.0, 0.75, 8.0])
def test_fp8_d192_v128_k3_varlen_causal(fp8_rescale_threshold):
    device = torch.device("cuda")
    q_lengths = (1, 257, 384)
    # The final request crosses the T8 scheduler's eight-head L2 capacity
    # boundary and exercises the KV-aware swizzle rather than the short-KV fallback.
    kv_lengths = (64, 320, 20_500)
    num_q_heads = num_kv_heads = 8
    torch.manual_seed(0)
    q = (
        torch.randn(
            sum(q_lengths), num_q_heads, 192, device=device, dtype=torch.bfloat16
        )
        / 10
    ).to(torch.float8_e4m3fn)
    k = (
        torch.randn(
            sum(kv_lengths), num_kv_heads, 192, device=device, dtype=torch.bfloat16
        )
        / 10
    ).to(torch.float8_e4m3fn)
    v = (
        torch.randn(
            sum(kv_lengths), num_kv_heads, 128, device=device, dtype=torch.bfloat16
        )
        / 10
    ).to(torch.float8_e4m3fn)
    softmax_scale = 1.0 / math.sqrt(192)

    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=_cumulative(q_lengths, device),
        cu_seqlens_k=_cumulative(kv_lengths, device),
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(kv_lengths),
        softmax_scale=softmax_scale,
        causal=True,
        num_splits=1,
        return_lse=True,
        fp8_rescale_threshold=fp8_rescale_threshold,
    )
    if lse.shape == (num_q_heads, sum(q_lengths)):
        lse = lse.transpose(0, 1).contiguous()

    out_ref, lse_ref = _reference(q, k, v, q_lengths, kv_lengths, softmax_scale)
    torch.testing.assert_close(out.float(), out_ref, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(lse.float(), lse_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not _is_sm103_available(), reason="requires an SM103 GPU")
@pytest.mark.parametrize(
    "q_lengths,kv_lengths",
    [
        ((384,), (128,)),
        ((0, 384), (64, 512)),
        ((384, 257), (0, 320)),
    ],
)
def test_fp8_d192_v128_k3_varlen_sequence_boundaries(q_lengths, kv_lengths):
    device = torch.device("cuda")
    num_heads = 2
    torch.manual_seed(3)
    q = (
        torch.randn(sum(q_lengths), num_heads, 192, device=device, dtype=torch.bfloat16)
        / 10
    ).to(torch.float8_e4m3fn)
    k = (
        torch.randn(
            sum(kv_lengths), num_heads, 192, device=device, dtype=torch.bfloat16
        )
        / 10
    ).to(torch.float8_e4m3fn)
    v = (
        torch.randn(
            sum(kv_lengths), num_heads, 128, device=device, dtype=torch.bfloat16
        )
        / 10
    ).to(torch.float8_e4m3fn)
    softmax_scale = 1.0 / math.sqrt(192)

    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=_cumulative(q_lengths, device),
        cu_seqlens_k=_cumulative(kv_lengths, device),
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(kv_lengths),
        softmax_scale=softmax_scale,
        causal=True,
        return_lse=True,
    )
    if lse.shape == (num_heads, sum(q_lengths)):
        lse = lse.transpose(0, 1).contiguous()

    out_ref, lse_ref = _reference(q, k, v, q_lengths, kv_lengths, softmax_scale)
    torch.testing.assert_close(out.float(), out_ref, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(lse.float(), lse_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not _is_sm103_available(), reason="requires an SM103 GPU")
def test_fp8_d192_v128_t8_cuda_graph_capture():
    device = torch.device("cuda")
    q_lengths = (257, 384)
    kv_lengths = (320, 512)
    num_heads = 8
    torch.manual_seed(2)
    q = (
        torch.randn(sum(q_lengths), num_heads, 192, device=device, dtype=torch.bfloat16)
        / 10
    ).to(torch.float8_e4m3fn)
    k = (
        torch.randn(
            sum(kv_lengths), num_heads, 192, device=device, dtype=torch.bfloat16
        )
        / 10
    ).to(torch.float8_e4m3fn)
    v = (
        torch.randn(
            sum(kv_lengths), num_heads, 128, device=device, dtype=torch.bfloat16
        )
        / 10
    ).to(torch.float8_e4m3fn)
    cu_q = _cumulative(q_lengths, device)
    cu_k = _cumulative(kv_lengths, device)
    out = torch.empty((*q.shape[:-1], 128), device=device, dtype=torch.bfloat16)

    kwargs = {
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
        "max_seqlen_q": max(q_lengths),
        "max_seqlen_k": max(kv_lengths),
        "causal": True,
        "num_splits": 1,
        "out": out,
    }
    flash_attn_varlen_func(q, k, v, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flash_attn_varlen_func(q, k, v, **kwargs)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()


@pytest.mark.skipif(not _is_sm10x_available(), reason="requires an SM10x GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("varlen", [False, True])
def test_d192_v128_fp16_bf16_default_profile_regression(dtype, varlen):
    device = torch.device("cuda")
    q_lengths = (129, 256)
    kv_lengths = (192, 320)
    num_heads = 4
    torch.manual_seed(1)
    q_flat = (
        torch.randn(sum(q_lengths), num_heads, 192, device=device, dtype=dtype) / 10
    )
    k_flat = (
        torch.randn(sum(kv_lengths), num_heads, 192, device=device, dtype=dtype) / 10
    )
    v_flat = (
        torch.randn(sum(kv_lengths), num_heads, 128, device=device, dtype=dtype) / 10
    )
    softmax_scale = 1.0 / math.sqrt(192)

    if varlen:
        out, lse = flash_attn_varlen_func(
            q_flat,
            k_flat,
            v_flat,
            cu_seqlens_q=_cumulative(q_lengths, device),
            cu_seqlens_k=_cumulative(kv_lengths, device),
            max_seqlen_q=max(q_lengths),
            max_seqlen_k=max(kv_lengths),
            softmax_scale=softmax_scale,
            causal=True,
            num_splits=1,
            return_lse=True,
        )
        out_flat = out
    else:
        # Dense tensors require one common length. Use the second request's sizes.
        q_len, kv_len = q_lengths[1], kv_lengths[1]
        q_dense = q_flat[-q_len:].unsqueeze(0)
        k_dense = k_flat[-kv_len:].unsqueeze(0)
        v_dense = v_flat[-kv_len:].unsqueeze(0)
        out, lse = flash_attn_func(
            q_dense,
            k_dense,
            v_dense,
            softmax_scale=softmax_scale,
            causal=True,
            num_splits=1,
            return_lse=True,
        )
        q_flat, k_flat, v_flat = q_dense[0], k_dense[0], v_dense[0]
        q_lengths, kv_lengths = (q_len,), (kv_len,)
        out_flat = out[0]

    if lse.shape == (num_heads, sum(q_lengths)):
        lse = lse.transpose(0, 1).contiguous()
    elif not varlen and lse.shape == (1, num_heads, q_lengths[0]):
        lse = lse[0].transpose(0, 1).contiguous()

    out_ref, lse_ref = _reference(
        q_flat, k_flat, v_flat, q_lengths, kv_lengths, softmax_scale
    )
    torch.testing.assert_close(out_flat.float(), out_ref, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(lse.float(), lse_ref, rtol=2e-3, atol=2e-3)
