# Copyright (c) 2025, the FlashAttention authors.
"""Static per-tensor FP8 (e4m3fn) fused output tests, SM100/SM110.

Accuracy bound mirrors `test_flash_attn.py`: kernel error vs FP32 ref is
allowed to be at most `rtol`x the eager-BF16-then-FP8 path's error vs the
same FP32 ref, plus a small ULP atol.
"""

import math
import os

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
    and torch.cuda.get_device_capability()[0] in (10, 11)
)

skip_if_no_fp8_sm = pytest.mark.skipif(
    not IS_FP8_SM_SUPPORTED,
    reason="Fused FP8 output requires SM100/SM110 (Blackwell).",
)

# Data-independent so it works under FakeTensorMode (no `.amax()` allowed).
DEFAULT_OUT_SCALE = 0.005


def _scale_t(value: float, device: torch.device) -> torch.Tensor:
    """Build a 0-d FP32 device tensor for the `output_scale` API."""
    return torch.tensor(value, dtype=torch.float32, device=device)


def _quantize_fp8(x: torch.Tensor, out_scale: float) -> torch.Tensor:
    """Static-scaled FP8 cast — what `static_scaled_fp8_quant` would emit."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    return (x.float() / out_scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)


def _assert_fp8_close(
    fused_fp8: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out_scale: float,
    rtol: float = 2.0,
    **ref_kwargs,
) -> None:
    """kernel_err <= rtol * eager_BF16+quant_err + ULP_atol, vs FP32 ref."""
    out_ref_fp32, _ = attention_ref(q, k, v, None, None, upcast=True, **ref_kwargs)
    out_pt_bf16, _ = attention_ref(
        q, k, v, None, None, upcast=False, reorder_ops=True, **ref_kwargs,
    )

    ref_fp8 = _quantize_fp8(out_ref_fp32, out_scale)
    pt_fp8 = _quantize_fp8(out_pt_bf16, out_scale)

    fused_deq = fused_fp8.float() * out_scale
    ref_deq = ref_fp8.float() * out_scale
    pt_deq = pt_fp8.float() * out_scale

    fwd_atol = 2 * (ref_deq + 0.3 - 0.3 - ref_deq).abs().max().item()
    kernel_err = (fused_deq - ref_deq).abs().max().item()
    eager_err = (pt_deq - ref_deq).abs().max().item()

    assert kernel_err <= rtol * eager_err + fwd_atol, (
        f"fused FP8 kernel max-err vs FP32 ref ({kernel_err:.4f}) > "
        f"{rtol}x eager-BF16+post-quant max-err ({eager_err:.4f}) + "
        f"ULP atol ({fwd_atol:.4f})"
    )


@skip_if_no_fp8_sm
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "head_dim,head_dim_v",
    [(64, 64), (128, 128), (192, 128)],  # small MHA + standard + DeepSeek MLA prefill
)
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_fp8_output_matches_post_quant(
    dtype: torch.dtype,
    causal: bool,
    head_dim: int,
    head_dim_v: int,
    mha_type: str,
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, seqlen, num_heads = 2, 512, 16
    if mha_type == "mha":
        num_kv_heads = num_heads
    elif mha_type == "mqa":
        num_kv_heads = 1
    else:  # gqa
        num_kv_heads = 4

    q = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seqlen, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, seqlen, num_kv_heads, head_dim_v, dtype=dtype, device=device)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    out_scale = DEFAULT_OUT_SCALE

    fused_buffer = torch.empty(
        batch, seqlen, num_heads, head_dim_v, dtype=torch.float8_e4m3fn, device=device,
    )
    fused_out, _ = flash_attn_func(
        q, k, v,
        softmax_scale=softmax_scale, causal=causal,
        out=fused_buffer,
        output_scale=_scale_t(out_scale, device),
    )
    if is_fake_mode():
        return  # compile-only pass; skip data-dependent comparison
    assert fused_out.dtype == torch.float8_e4m3fn

    _assert_fp8_close(fused_out, q, k, v, out_scale, causal=causal)


@skip_if_no_fp8_sm
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_fp8_output_varlen_deepseek_mla():
    """DeepSeek-V3 MLA prefill shape (qk=192, v=128) via the varlen API."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    seqlens = [256, 384, 512, 192]
    total_q = sum(seqlens)
    num_heads, num_kv_heads = 16, 1
    head_dim, head_dim_v = 192, 128
    dtype = torch.bfloat16
    out_scale = DEFAULT_OUT_SCALE

    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_q, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_q, num_kv_heads, head_dim_v, dtype=dtype, device=device)
    cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(0)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    fused_buffer = torch.empty(
        total_q, num_heads, head_dim_v, dtype=torch.float8_e4m3fn, device=device,
    )
    fused_out, _ = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(seqlens), max_seqlen_k=max(seqlens),
        softmax_scale=softmax_scale, causal=True,
        out=fused_buffer,
        output_scale=_scale_t(out_scale, device),
    )
    if is_fake_mode():
        return
    assert fused_out.dtype == torch.float8_e4m3fn

    # attention_ref is 4D-only, so compare per-sequence and accumulate errors.
    fwd_atol = 0.0
    kernel_err = 0.0
    eager_err = 0.0
    for i, sl in enumerate(seqlens):
        s, e = int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item())
        qi = q[s:e].unsqueeze(0)
        ki = k[s:e].unsqueeze(0)
        vi = v[s:e].unsqueeze(0)
        out_ref_fp32, _ = attention_ref(qi, ki, vi, None, None, causal=True, upcast=True)
        out_pt_bf16, _ = attention_ref(
            qi, ki, vi, None, None, causal=True, upcast=False, reorder_ops=True,
        )
        ref_fp8 = _quantize_fp8(out_ref_fp32, out_scale)
        pt_fp8 = _quantize_fp8(out_pt_bf16, out_scale)
        ref_deq = ref_fp8.float() * out_scale
        pt_deq = pt_fp8.float() * out_scale
        fused_deq = fused_out[s:e].unsqueeze(0).float() * out_scale
        fwd_atol = max(fwd_atol, 2 * (ref_deq + 0.3 - 0.3 - ref_deq).abs().max().item())
        kernel_err = max(kernel_err, (fused_deq - ref_deq).abs().max().item())
        eager_err = max(eager_err, (pt_deq - ref_deq).abs().max().item())

    rtol = 2.0
    assert kernel_err <= rtol * eager_err + fwd_atol, (
        f"varlen fused FP8 max-err vs FP32 ref ({kernel_err:.4f}) > "
        f"{rtol}x eager max-err ({eager_err:.4f}) + ULP atol ({fwd_atol:.4f})"
    )


@skip_if_no_fp8_sm
def test_fp8_output_auto_allocate():
    """User passes output_scale without `out`; library allocates FP8 buffer."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    q = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device=device)

    fused_out, _ = flash_attn_func(
        q, k, v, causal=True,
        output_scale=_scale_t(0.05, device),
    )
    assert fused_out.dtype == torch.float8_e4m3fn
    assert fused_out.shape == (2, 256, 8, 128)


@skip_if_no_fp8_sm
def test_fp8_output_scale_shape_variants():
    """`output_scale` accepts both 0-d and 1-elem 1-D tensors (same bytes out)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    q = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device=device)

    scale_0d = torch.tensor(0.05, dtype=torch.float32, device=device)
    scale_1d = torch.tensor([0.05], dtype=torch.float32, device=device)
    out_a, _ = flash_attn_func(q, k, v, causal=True, output_scale=scale_0d)
    out_b, _ = flash_attn_func(q, k, v, causal=True, output_scale=scale_1d)
    # Both shapes go through the same `.reshape(1)` normalization on the
    # host, so the resulting FP8 bytes must be identical.
    assert torch.equal(out_a.view(torch.uint8), out_b.view(torch.uint8))


@skip_if_no_fp8_sm
@pytest.mark.parametrize(
    "window_size",
    [(128, 0), (64, 64), (-1, 0)],  # left-only causal local, symmetric local, full causal
    ids=["causal_local_left", "symmetric_local", "causal_full"],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_fp8_output_sliding_window(window_size):
    """Local / sliding-window masking exercises a different mask_mod path."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, seqlen, num_heads = 2, 512, 8
    head_dim = 128
    dtype = torch.bfloat16
    out_scale = DEFAULT_OUT_SCALE

    q = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    causal = window_size == (-1, 0)
    ws_kernel = (None, None) if causal else window_size
    ws_ref = ws_kernel

    fused_buffer = torch.empty(
        batch, seqlen, num_heads, head_dim, dtype=torch.float8_e4m3fn, device=device,
    )
    fused_out, _ = flash_attn_func(
        q, k, v, softmax_scale=softmax_scale, causal=causal, window_size=ws_kernel,
        out=fused_buffer,
        output_scale=_scale_t(out_scale, device),
    )
    if is_fake_mode():
        return

    _assert_fp8_close(
        fused_out, q, k, v, out_scale, causal=causal, window_size=ws_ref,
    )


@skip_if_no_fp8_sm
@pytest.mark.parametrize("softcap", [15.0, 30.0])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_fp8_output_softcap(softcap: float):
    """Softcap (Gemma/GLM) wraps logits through tanh before softmax."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, seqlen, num_heads = 2, 512, 8
    head_dim = 128
    dtype = torch.bfloat16
    out_scale = DEFAULT_OUT_SCALE

    q = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    fused_buffer = torch.empty(
        batch, seqlen, num_heads, head_dim, dtype=torch.float8_e4m3fn, device=device,
    )
    fused_out, _ = flash_attn_func(
        q, k, v, softmax_scale=softmax_scale, causal=True, softcap=softcap,
        out=fused_buffer,
        output_scale=_scale_t(out_scale, device),
    )
    if is_fake_mode():
        return

    # Softcap matches `rtol = 3 if softcap > 0 else 2` in test_flash_attn_output.
    _assert_fp8_close(
        fused_out, q, k, v, out_scale, rtol=3.0, causal=True, softcap=softcap,
    )


@skip_if_no_fp8_sm
@pytest.mark.parametrize(
    "scale_factor",
    [0.1, 1.0, 4.0],
    ids=["scale_underuses_range", "scale_matches_peak", "scale_overuses_range"],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_fp8_output_scale_extremes(scale_factor: float):
    """Sweep `output_scale` away from the matched-peak choice.

    - small scale: values divide to >fp8_max → clamp.
    - matched scale: roughly fills the FP8 range.
    - large scale: values divide to << 1 → mantissa truncation.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, seqlen, num_heads = 2, 256, 8
    head_dim = 128
    dtype = torch.bfloat16
    out_scale = DEFAULT_OUT_SCALE * scale_factor

    q = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    fused_buffer = torch.empty(
        batch, seqlen, num_heads, head_dim, dtype=torch.float8_e4m3fn, device=device,
    )
    fused_out, _ = flash_attn_func(
        q, k, v, softmax_scale=softmax_scale, causal=True,
        out=fused_buffer,
        output_scale=_scale_t(out_scale, device),
    )
    if is_fake_mode():
        return

    _assert_fp8_close(fused_out, q, k, v, out_scale, causal=True)


@skip_if_no_fp8_sm
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_fp8_output_split_kv():
    """Split-KV + FP8: forward writes FP32 partials, combine emits FP8.

    Triggered by short Q + long K (decode-style); we force num_splits=4 to
    guarantee the split-KV combine path even if the auto-heuristic wouldn't
    pick it.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, seqlen_q, seqlen_k, num_heads = 2, 1, 4096, 16
    head_dim = 128
    out_scale = DEFAULT_OUT_SCALE

    q = torch.randn(batch, seqlen_q, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(batch, seqlen_k, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(batch, seqlen_k, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    fused_buffer = torch.empty(
        batch, seqlen_q, num_heads, head_dim, dtype=torch.float8_e4m3fn, device=device,
    )
    fused_out, _ = flash_attn_func(
        q, k, v,
        softmax_scale=softmax_scale, causal=True, num_splits=4,
        out=fused_buffer,
        output_scale=_scale_t(out_scale, device),
    )
    if is_fake_mode():
        return
    assert fused_out.dtype == torch.float8_e4m3fn

    _assert_fp8_close(fused_out, q, k, v, out_scale, causal=True)


def test_fp8_output_validation_errors():
    """Validation paths fire on any GPU (no kernel launch needed)."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        pytest.skip("Validation calls into _flash_attn_fwd which expects CUDA tensors")
    from flash_attn.cute.interface import _flash_attn_fwd

    q = torch.randn(2, 64, 4, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(2, 64, 4, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(2, 64, 4, 128, dtype=torch.bfloat16, device=device)
    out_fp8 = torch.empty(2, 64, 4, 128, dtype=torch.float8_e4m3fn, device=device)
    scale = _scale_t(0.5, device)

    # FP8 output without output_scale -> AssertionError
    with pytest.raises(AssertionError, match="no output_scale was provided"):
        _flash_attn_fwd(q, k, v, out=out_fp8, _arch=100)

    # output_scale as a Python float -> AssertionError
    with pytest.raises(AssertionError, match="must be a torch.Tensor"):
        _flash_attn_fwd(q, k, v, out=out_fp8, output_scale=0.5, _arch=100)

    # output_scale wrong dtype -> AssertionError
    bad_dtype = torch.tensor(0.5, dtype=torch.float64, device=device)
    with pytest.raises(AssertionError, match="must be float32"):
        _flash_attn_fwd(q, k, v, out=out_fp8, output_scale=bad_dtype, _arch=100)

    # output_scale with multiple elements -> AssertionError
    bad_shape = torch.tensor([0.5, 0.6], dtype=torch.float32, device=device)
    with pytest.raises(AssertionError, match="must be a scalar"):
        _flash_attn_fwd(q, k, v, out=out_fp8, output_scale=bad_shape, _arch=100)

    # output_scale on a BF16 output buffer -> AssertionError (out dtype mismatch).
    bf16_out = torch.empty(2, 64, 4, 128, dtype=torch.bfloat16, device=device)
    with pytest.raises(AssertionError, match="torch.float8_e4m3fn"):
        _flash_attn_fwd(q, k, v, out=bf16_out, output_scale=scale, _arch=100)

    # SM80 / SM90 / SM120 reject FP8 output in each forward class's __init__.
    with pytest.raises(AssertionError, match="Fused quant output not implemented"):
        _flash_attn_fwd(q, k, v, out=out_fp8, output_scale=scale, _arch=80)
    with pytest.raises(AssertionError, match="Fused quant output not implemented"):
        _flash_attn_fwd(q, k, v, out=out_fp8, output_scale=scale, _arch=90)
    with pytest.raises(AssertionError, match="Fused quant output not implemented"):
        _flash_attn_fwd(q, k, v, out=out_fp8, output_scale=scale, _arch=120)
