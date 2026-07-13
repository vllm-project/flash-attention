# Copyright (c) 2025, the FlashAttention authors.
"""Per-row dynamic FP8 (e4m3fn) fused output tests, SM100/SM110.

Accuracy bound mirrors ``test_flash_attn_fp8_output.py``: kernel error vs
FP32 reference is allowed to be at most ``rtol``x the eager-BF16-then-
per-group-FP8 path's error vs the same FP32 reference, plus a small ULP
atol. Per-row dynamic scales are auto-computed; this test exercises the
``output_scales`` API entry point.
"""

import math
import os
from typing import Tuple

import pytest
import torch

from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func
from flash_attn.cute.testing import (
    attention_ref,
    is_fake_mode,
    maybe_fake_tensor_mode,
)

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", 0)) == 1
IS_FP8_SM_SUPPORTED = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 10
)

skip_if_no_fp8_sm = pytest.mark.skipif(
    not IS_FP8_SM_SUPPORTED,
    reason="Fused per-group FP8 output requires SM100/SM110 (Blackwell).",
)


def _quantize_fp8_per_group(
    x: torch.Tensor,
    group_size: int,
    out_dtype: torch.dtype = torch.float8_e4m3fn,
    ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference per-row dynamic FP8 quantizer.

    Returns ``(fp8_codes, dequant_scales)`` where
        ``fp8_codes`` has the same shape as ``x`` (``out_dtype``) and
        ``dequant_scales`` has shape ``x.shape[:-1] + (head_dim // group_size,)``
        (fp32). Dequantization is ``fp8.float() * scales.repeat_interleave(group_size, -1)``.

    With ``ue8m0`` the dequant scale is rounded up to a power of two (clamped to 1e-10 before
    rounding), matching vLLM ``per_token_group_quant_fp8(scale_ue8m0=True)``.
    """
    finfo = torch.finfo(out_dtype)
    f8_max = float(finfo.max)
    head_dim = x.shape[-1]
    assert head_dim % group_size == 0, f"head_dim={head_dim} % group_size={group_size} != 0"
    num_groups = head_dim // group_size
    x_fp32 = x.float()
    x_grp = x_fp32.unflatten(-1, (num_groups, group_size))
    amax = x_grp.abs().amax(dim=-1)
    if ue8m0:
        dequant_scale = torch.exp2(torch.ceil(torch.log2((amax / f8_max).clamp(min=1e-10))))
    else:
        is_zero = amax == 0
        dequant_scale = torch.where(is_zero, torch.ones_like(amax), amax) / f8_max
    fp8_codes = (x_grp / dequant_scale.unsqueeze(-1)).clamp(finfo.min, finfo.max).to(out_dtype)
    fp8_codes = fp8_codes.flatten(-2)
    if not ue8m0:
        dequant_scale = torch.where(amax == 0, torch.zeros_like(dequant_scale), dequant_scale)
    return fp8_codes, dequant_scale


def _dequantize_fp8_per_group(
    fp8_codes: torch.Tensor, scales: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Inverse of `_quantize_fp8_per_group`."""
    head_dim = fp8_codes.shape[-1]
    assert head_dim == scales.shape[-1] * group_size
    return fp8_codes.float() * scales.repeat_interleave(group_size, dim=-1)


def _assert_pergroup_fp8_close(
    fused_fp8: torch.Tensor,
    fused_scales: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group_size: int,
    rtol: float = 2.0,
    out_dtype: torch.dtype = torch.float8_e4m3fn,
    **ref_kwargs,
) -> None:
    """kernel_err <= rtol * eager_BF16+per-group-quant_err + ULP_atol, vs FP32 ref."""
    out_ref_fp32, _ = attention_ref(q, k, v, None, None, upcast=True, **ref_kwargs)
    out_pt_bf16, _ = attention_ref(
        q, k, v, None, None, upcast=False, reorder_ops=True, **ref_kwargs,
    )

    ref_fp8, ref_scales = _quantize_fp8_per_group(out_ref_fp32, group_size, out_dtype)
    pt_fp8, pt_scales = _quantize_fp8_per_group(out_pt_bf16, group_size, out_dtype)

    fused_deq = _dequantize_fp8_per_group(fused_fp8, fused_scales, group_size)
    ref_deq = _dequantize_fp8_per_group(ref_fp8, ref_scales, group_size)
    pt_deq = _dequantize_fp8_per_group(pt_fp8, pt_scales, group_size)

    fwd_atol = 2 * (ref_deq + 0.3 - 0.3 - ref_deq).abs().max().item()
    kernel_err = (fused_deq - ref_deq).abs().max().item()
    eager_err = (pt_deq - ref_deq).abs().max().item()

    assert kernel_err <= rtol * eager_err + fwd_atol, (
        f"fused per-group FP8 kernel max-err vs FP32 ref ({kernel_err:.4f}) > "
        f"{rtol}x eager-BF16+per-group-quant max-err ({eager_err:.4f}) + "
        f"ULP atol ({fwd_atol:.4f})"
    )


@skip_if_no_fp8_sm
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "head_dim,head_dim_v,group_size",
    [
        (128, 128, 128),  # standard MHA, single group per row
        (192, 128, 128),  # DeepSeek MLA prefill, single group per row
        (128, 128, 64),   # two groups per row (forward multi-group cast)
    ],
)
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_fp8_pergroup_output_matches_post_quant(
    dtype: torch.dtype,
    causal: bool,
    head_dim: int,
    head_dim_v: int,
    group_size: int,
    mha_type: str,
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, seqlen, num_heads = 2, 512, 16
    if mha_type == "mha":
        num_kv_heads = num_heads
    elif mha_type == "mqa":
        num_kv_heads = 1
    else:
        num_kv_heads = 4

    q = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seqlen, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, seqlen, num_kv_heads, head_dim_v, dtype=dtype, device=device)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    out_buf = torch.empty(batch, seqlen, num_heads, head_dim_v, dtype=torch.float8_e4m3fn, device=device)
    scales_buf = torch.empty(
        batch, seqlen, num_heads, head_dim_v // group_size, dtype=torch.float32, device=device
    )
    scales = scales_buf  # written in place; the call returns only (out, lse)
    out, _ = flash_attn_func(
        q, k, v,
        softmax_scale=softmax_scale, causal=causal,
        out=out_buf, output_scales=scales_buf,
    )
    if is_fake_mode():
        return
    assert out.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.float32
    assert scales.shape == (batch, seqlen, num_heads, head_dim_v // group_size)

    _assert_pergroup_fp8_close(out, scales, q, k, v, group_size, causal=causal)


@skip_if_no_fp8_sm
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_fp8_pergroup_output_varlen_deepseek_mla():
    """DeepSeek-V3 MLA prefill shape (qk=192, v=128) via the varlen API."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    seqlens = [256, 384, 512, 192]
    total_q = sum(seqlens)
    num_heads, num_kv_heads = 16, 1
    head_dim, head_dim_v = 192, 128
    group_size = 128
    dtype = torch.bfloat16

    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_q, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_q, num_kv_heads, head_dim_v, dtype=dtype, device=device)
    cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(0)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    out_buf = torch.empty(total_q, num_heads, head_dim_v, dtype=torch.float8_e4m3fn, device=device)
    scales_buf = torch.empty(
        total_q, num_heads, head_dim_v // group_size, dtype=torch.float32, device=device
    )
    scales = scales_buf  # written in place; the call returns only (out, lse)
    out, _ = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(seqlens), max_seqlen_k=max(seqlens),
        softmax_scale=softmax_scale, causal=True,
        out=out_buf, output_scales=scales_buf,
    )
    if is_fake_mode():
        return
    assert out.dtype == torch.float8_e4m3fn
    assert scales.shape == (total_q, num_heads, head_dim_v // group_size)

    rtol = 2.0
    kernel_err = 0.0
    eager_err = 0.0
    fwd_atol = 0.0
    for i, sl in enumerate(seqlens):
        s, e = int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item())
        qi = q[s:e].unsqueeze(0)
        ki = k[s:e].unsqueeze(0)
        vi = v[s:e].unsqueeze(0)
        out_ref_fp32, _ = attention_ref(qi, ki, vi, None, None, causal=True, upcast=True)
        out_pt_bf16, _ = attention_ref(
            qi, ki, vi, None, None, causal=True, upcast=False, reorder_ops=True,
        )
        ref_fp8, ref_scales = _quantize_fp8_per_group(out_ref_fp32, group_size)
        pt_fp8, pt_scales = _quantize_fp8_per_group(out_pt_bf16, group_size)
        ref_deq = _dequantize_fp8_per_group(ref_fp8, ref_scales, group_size)
        pt_deq = _dequantize_fp8_per_group(pt_fp8, pt_scales, group_size)
        fused_deq = _dequantize_fp8_per_group(
            out[s:e].unsqueeze(0), scales[s:e].unsqueeze(0), group_size
        )
        fwd_atol = max(fwd_atol, 2 * (ref_deq + 0.3 - 0.3 - ref_deq).abs().max().item())
        kernel_err = max(kernel_err, (fused_deq - ref_deq).abs().max().item())
        eager_err = max(eager_err, (pt_deq - ref_deq).abs().max().item())

    assert kernel_err <= rtol * eager_err + fwd_atol, (
        f"varlen fused per-group FP8 max-err vs FP32 ref ({kernel_err:.4f}) > "
        f"{rtol}x eager max-err ({eager_err:.4f}) + ULP atol ({fwd_atol:.4f})"
    )


def _alloc_colmajor_tma_scales(total_q, num_heads, groups_per_head, device):
    """Allocate output_scales in the DeepGEMM column-major / TMA-aligned layout, i.e. what
    vLLM ``per_token_group_quant_fp8(column_major_scales=True, tma_aligned_scales=True)`` makes.
    Logical shape (total_q, num_heads, groups_per_head); token dim is contiguous."""
    num_groups = num_heads * groups_per_head
    tma_aligned_m = ((total_q + 3) // 4) * 4
    buf2d = torch.empty_strided(
        (total_q, num_groups), (1, tma_aligned_m), dtype=torch.float32, device=device
    )
    return buf2d.unflatten(-1, (num_heads, groups_per_head))


@skip_if_no_fp8_sm
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_fp8_pergroup_output_ue8m0_colmajor():
    """DeepSeek MLA prefill with the DeepGEMM scale layout: a column-major / TMA-aligned
    output_scales buffer makes FA emit UE8M0 (power-of-two) scales in that layout."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    seqlens = [256, 384, 512, 192]
    total_q = sum(seqlens)
    num_heads, num_kv_heads = 16, 1
    head_dim, head_dim_v, group_size = 192, 128, 128
    dtype = torch.bfloat16

    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_q, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_q, num_kv_heads, head_dim_v, dtype=dtype, device=device)
    cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(0)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    out_buf = torch.empty(total_q, num_heads, head_dim_v, dtype=torch.float8_e4m3fn, device=device)
    scales_buf = _alloc_colmajor_tma_scales(total_q, num_heads, head_dim_v // group_size, device)
    scales = scales_buf  # written in place; the call returns only (out, lse)
    out, _ = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(seqlens), max_seqlen_k=max(seqlens),
        softmax_scale=softmax_scale, causal=True,
        out=out_buf, output_scales=scales_buf,
    )
    if is_fake_mode():
        return
    assert out.dtype == torch.float8_e4m3fn
    assert scales.data_ptr() == scales_buf.data_ptr()  # written in place, col-major preserved
    assert scales.stride(0) == 1 and not scales.is_contiguous()

    # Every nonzero scale must be an exact power of two (UE8M0).
    nz = scales[scales > 0]
    assert torch.equal(torch.log2(nz), torch.log2(nz).round()), "scales are not powers of two"

    # Accuracy vs FP32 ref, with the eager baseline also using UE8M0 rounding (fair bound).
    rtol = 2.0
    kernel_err = eager_err = fwd_atol = 0.0
    for i, sl in enumerate(seqlens):
        s, e = int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item())
        qi, ki, vi = q[s:e].unsqueeze(0), k[s:e].unsqueeze(0), v[s:e].unsqueeze(0)
        out_ref_fp32, _ = attention_ref(qi, ki, vi, None, None, causal=True, upcast=True)
        out_pt_bf16, _ = attention_ref(
            qi, ki, vi, None, None, causal=True, upcast=False, reorder_ops=True,
        )
        ref_fp8, ref_scales = _quantize_fp8_per_group(out_ref_fp32, group_size, ue8m0=True)
        pt_fp8, pt_scales = _quantize_fp8_per_group(out_pt_bf16, group_size, ue8m0=True)
        ref_deq = _dequantize_fp8_per_group(ref_fp8, ref_scales, group_size)
        pt_deq = _dequantize_fp8_per_group(pt_fp8, pt_scales, group_size)
        fused_deq = _dequantize_fp8_per_group(
            out[s:e].unsqueeze(0), scales[s:e].unsqueeze(0), group_size
        )
        fwd_atol = max(fwd_atol, 2 * (ref_deq + 0.3 - 0.3 - ref_deq).abs().max().item())
        kernel_err = max(kernel_err, (fused_deq - ref_deq).abs().max().item())
        eager_err = max(eager_err, (pt_deq - ref_deq).abs().max().item())

    assert kernel_err <= rtol * eager_err + fwd_atol, (
        f"ue8m0 col-major fused per-group max-err vs FP32 ref ({kernel_err:.4f}) > "
        f"{rtol}x eager max-err ({eager_err:.4f}) + ULP atol ({fwd_atol:.4f})"
    )


@skip_if_no_fp8_sm
def test_fp8_pergroup_output_requires_out():
    """Fused output requires a pre-allocated fp8 `out` (its dtype selects the variant)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    q = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device=device)
    scales_buf = torch.empty(2, 256, 8, 1, dtype=torch.float32, device=device)

    # Omitting `out` for fused output is rejected (no dtype to assume).
    with pytest.raises(AssertionError):
        flash_attn_func(q, k, v, causal=True, output_scales=scales_buf)


@skip_if_no_fp8_sm
def test_fp8_pergroup_output_preallocated():
    """User pre-allocates `out` and `output_scales`; library infers group_size."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    q = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device=device)

    out_buf = torch.empty(2, 256, 8, 128, dtype=torch.float8_e4m3fn, device=device)
    scales_buf = torch.empty(2, 256, 8, 1, dtype=torch.float32, device=device)
    scales = scales_buf  # written in place; the call returns only (out, lse)
    out, _ = flash_attn_func(
        q, k, v, causal=True,
        out=out_buf,
        output_scales=scales_buf,
    )
    assert out.data_ptr() == out_buf.data_ptr()
    assert scales.data_ptr() == scales_buf.data_ptr()

    _assert_pergroup_fp8_close(out, scales, q, k, v, group_size=128, causal=True)


@skip_if_no_fp8_sm
def test_fp8_pergroup_output_e5m2():
    """Output dtype is taken from the pre-allocated fp8 `out` — here e5m2 (max 57344)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    q = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device=device)

    out_buf = torch.empty(2, 256, 8, 128, dtype=torch.float8_e5m2, device=device)
    scales_buf = torch.empty(2, 256, 8, 1, dtype=torch.float32, device=device)
    scales = scales_buf  # written in place; the call returns only (out, lse)
    out, _ = flash_attn_func(
        q, k, v, causal=True, out=out_buf, output_scales=scales_buf,
    )
    assert out.dtype == torch.float8_e5m2
    _assert_pergroup_fp8_close(
        out, scales, q, k, v, group_size=128, out_dtype=torch.float8_e5m2, causal=True,
    )


@skip_if_no_fp8_sm
def test_fp8_pergroup_output_split_kv():
    """SplitKV path: forward writes FP32 partials, combine does the per-group quant."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    # Use a decode-like shape so the split-KV heuristic actually splits.
    batch, seqlen_q, seqlen_k = 4, 1, 4096
    num_heads, num_kv_heads, head_dim = 16, 1, 128
    q = torch.randn(batch, seqlen_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(batch, seqlen_k, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(batch, seqlen_k, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

    out_buf = torch.empty(batch, seqlen_q, num_heads, head_dim, dtype=torch.float8_e4m3fn, device=device)
    scales_buf = torch.empty(batch, seqlen_q, num_heads, 1, dtype=torch.float32, device=device)
    scales = scales_buf  # written in place; the call returns only (out, lse)
    out, _ = flash_attn_func(
        q, k, v,
        softmax_scale=1.0 / math.sqrt(head_dim),
        num_splits=4,
        out=out_buf, output_scales=scales_buf,
    )
    assert out.dtype == torch.float8_e4m3fn
    assert scales.shape == (batch, seqlen_q, num_heads, 1)

    _assert_pergroup_fp8_close(out, scales, q, k, v, group_size=128)


@skip_if_no_fp8_sm
def test_fp8_pergroup_output_split_kv_group64():
    """SplitKV with group_size=64 < k_block_size=128: combine reduces 2 groups per k-block."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, seqlen_q, seqlen_k = 4, 1, 4096
    num_heads, num_kv_heads, head_dim = 16, 1, 128
    group_size = 64
    q = torch.randn(batch, seqlen_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(batch, seqlen_k, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(batch, seqlen_k, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

    out_buf = torch.empty(batch, seqlen_q, num_heads, head_dim, dtype=torch.float8_e4m3fn, device=device)
    scales_buf = torch.empty(
        batch, seqlen_q, num_heads, head_dim // group_size, dtype=torch.float32, device=device
    )
    scales = scales_buf  # written in place; the call returns only (out, lse)
    out, _ = flash_attn_func(
        q, k, v,
        softmax_scale=1.0 / math.sqrt(head_dim),
        num_splits=4,
        out=out_buf, output_scales=scales_buf,
    )
    assert out.dtype == torch.float8_e4m3fn
    assert scales.shape == (batch, seqlen_q, num_heads, head_dim // group_size)

    _assert_pergroup_fp8_close(out, scales, q, k, v, group_size=group_size)


@skip_if_no_fp8_sm
def test_fp8_pergroup_output_validation_errors():
    """Misuses of the per-group API should raise loudly."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    q = torch.randn(1, 128, 4, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, 128, 4, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, 128, 4, 128, dtype=torch.bfloat16, device=device)

    # output_scale + output_scales mutually exclusive
    scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    scales_buf = torch.empty(1, 128, 4, 1, dtype=torch.float32, device=device)
    with pytest.raises(ValueError):
        flash_attn_func(q, k, v, output_scale=scale, output_scales=scales_buf)

    # output_scales last dim must divide head_dim_v
    bad_scales = torch.empty(1, 128, 4, 3, dtype=torch.float32, device=device)  # 128 % 3 != 0
    with pytest.raises(AssertionError):
        flash_attn_func(q, k, v, output_scales=bad_scales)
