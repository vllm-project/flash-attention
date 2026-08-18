# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Fast subset of test_flash_attn.py for quick iteration.
# Covers: causal/noncausal, varlen/not varlen, MHA/GQA, split/not split, fwd+bwd.

import os
import random

import pytest
import torch

from einops import rearrange

from flash_attn.cute.testing import (
    attention_ref,
    generate_random_padding_mask,
    generate_qkv,
    maybe_fake_tensor_mode,
    is_fake_mode,
)
from flash_attn.cute.interface import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_combine,
)

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", 0)) == 1
IS_SM90 = torch.cuda.get_device_capability()[0] == 9
IS_SM100 = torch.cuda.get_device_capability()[0] == 10
IS_SM120 = torch.cuda.get_device_capability()[0] == 12


# ---------------------------------------------------------------------------
# Forward + backward (non-varlen)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("num_splits", [1, 3])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (128, 128),
        (256, 256),
        (113, 203),
        (1024, 1024),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_output(seqlen_q, seqlen_k, d, causal, num_splits, mha_type, dtype):
    if (IS_SM90 or IS_SM120) and num_splits > 1:
        pytest.skip("SM90/SM120 fwd doesn't support num_splits > 1")
    device = "cuda"
    torch.random.manual_seed(0)
    random.seed(0)
    torch.cuda.empty_cache()
    batch_size = 4
    nheads = 6
    nheads_kv = nheads if mha_type == "mha" else (3 if mha_type == "gqa" else 1)

    q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype).to(dtype).requires_grad_()
    k_ref = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype).to(dtype).requires_grad_()
    v_ref = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype).to(dtype).requires_grad_()

    q = q_ref.detach().to(dtype).requires_grad_()
    k = k_ref.detach().to(dtype).requires_grad_()
    v = v_ref.detach().to(dtype).requires_grad_()

    out_ref, _ = attention_ref(q_ref, k_ref, v_ref, None, None, causal=causal)
    out_pt, _ = attention_ref(
        q_ref, k_ref, v_ref, None, None, causal=causal, upcast=False, reorder_ops=True,
    )

    out, lse = flash_attn_func(q, k, v, causal=causal, num_splits=num_splits)

    if is_fake_mode():
        return

    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + fwd_atol

    # Backward (only for non-split, matching d)
    can_bwd = (
        num_splits == 1
        and d <= 128
        and not (causal and seqlen_k < seqlen_q)
    )
    if IS_SM90 and d == 64 and not causal:
        can_bwd = False  # SM90 d=64 non-causal xfail
    if not can_bwd:
        return

    g = torch.randn_like(out)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)

    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)

    dq_atol = 2 * (dq_ref + 0.3 - 0.3 - dq_ref).abs().max().item()
    dk_atol = 2 * (dk_ref + 0.3 - 0.3 - dk_ref).abs().max().item()
    dv_atol = 2 * (dv_ref + 0.3 - 0.3 - dv_ref).abs().max().item()
    assert (dq - dq_ref).abs().max().item() <= 2 * (dq_pt - dq_ref).abs().max().item() + dq_atol
    assert (dk - dk_ref).abs().max().item() <= 2 * (dk_pt - dk_ref).abs().max().item() + dk_atol
    assert (dv - dv_ref).abs().max().item() <= 2 * (dv_pt - dv_ref).abs().max().item() + dv_atol


# ---------------------------------------------------------------------------
# Forward + backward (varlen with cu_seqlens)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("seqlen", [128, 256, 1024])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_varlen_output(seqlen, d, causal, mha_type, dtype):
    """Varlen test with cu_seqlens (packed): equal seqlens so we can compare with non-varlen ref."""
    device = "cuda"
    seed = seqlen + d + int(causal) * 2
    torch.random.manual_seed(seed)
    random.seed(seed)
    batch_size = 9
    nheads = 6
    nheads_kv = nheads if mha_type == "mha" else (3 if mha_type == "gqa" else 1)

    q_ref = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype).to(dtype).requires_grad_()
    k_ref = torch.randn(batch_size, seqlen, nheads_kv, d, device=device, dtype=dtype).to(dtype).requires_grad_()
    v_ref = torch.randn(batch_size, seqlen, nheads_kv, d, device=device, dtype=dtype).to(dtype).requires_grad_()

    out_ref, _ = attention_ref(q_ref, k_ref, v_ref, None, None, causal=causal)
    out_pt, _ = attention_ref(
        q_ref, k_ref, v_ref, None, None, causal=causal, upcast=False, reorder_ops=True,
    )

    cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, seqlen, device=device, dtype=torch.int32)
    q_varlen = rearrange(q_ref.detach(), "b s h d -> (b s) h d").requires_grad_()
    k_varlen = rearrange(k_ref.detach(), "b s h d -> (b s) h d").requires_grad_()
    v_varlen = rearrange(v_ref.detach(), "b s h d -> (b s) h d").requires_grad_()

    out_varlen, lse = flash_attn_varlen_func(
        q_varlen, k_varlen, v_varlen,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seqlen, max_seqlen_k=seqlen,
        causal=causal,
    )

    if is_fake_mode():
        return

    out_reshaped = rearrange(out_varlen, "(b s) h d -> b s h d", b=batch_size)
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    assert (out_reshaped - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + fwd_atol

    # Backward
    can_bwd = d <= 128
    if not can_bwd:
        return

    g = torch.randn_like(out_varlen)
    dq_varlen, dk_varlen, dv_varlen = torch.autograd.grad(out_varlen, (q_varlen, k_varlen, v_varlen), g)

    assert dq_varlen.isfinite().all(), "dq contains non-finite values"
    assert dk_varlen.isfinite().all(), "dk contains non-finite values"
    assert dv_varlen.isfinite().all(), "dv contains non-finite values"
    assert dq_varlen.abs().max().item() > 0, "dq is all zeros"
    assert dk_varlen.abs().max().item() > 0, "dk is all zeros"
    assert dv_varlen.abs().max().item() > 0, "dv is all zeros"


# ---------------------------------------------------------------------------
# Forward + backward (varlen with padding masks — all unpad combinations)
# Covers 4 compile-key-distinct paths:
#   (unpad_q, unpad_kv) = (T,T): cu_seqlens for both Q and K
#   (unpad_q, unpad_kv) = (F,F): seqused for both Q and K
#   (unpad_q, unpad_kv) = (T,F): cu_seqlens_q + seqused_k
#   (unpad_q, unpad_kv) = (F,T): seqused_q + cu_seqlens_k
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("seqlen", [128, 256])
@pytest.mark.parametrize(
    "unpad_q,unpad_kv",
    [(True, True), (False, False), (True, False), (False, True)],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_varlen_unpad_output(seqlen, d, causal, mha_type, unpad_q, unpad_kv, dtype):
    """Varlen test with all 4 (unpad_q, unpad_kv) combos: cu_seqlens vs seqused."""
    device = "cuda"
    seed = seqlen + d + int(causal) * 2 + int(unpad_q) * 7 + int(unpad_kv) * 13
    torch.random.manual_seed(seed)
    random.seed(seed)
    batch_size = 9
    nheads = 6
    nheads_kv = nheads if mha_type == "mha" else (3 if mha_type == "gqa" else 1)

    q = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads_kv, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads_kv, d, device=device, dtype=dtype)
    q_ref = q.detach().to(dtype).requires_grad_()
    k_ref = k.detach().to(dtype).requires_grad_()
    v_ref = v.detach().to(dtype).requires_grad_()

    query_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode="random")
    key_padding_mask = query_padding_mask if causal else generate_random_padding_mask(
        seqlen, batch_size, device, mode="random"
    )

    (
        q_unpad_t, k_unpad_t, v_unpad_t, _qv_unpad,
        cu_seqlens_q, cu_seqlens_k,
        seqused_q, seqused_k,
        max_seqlen_q, max_seqlen_k,
        q_padded, k_padded, v_padded, _qv_padded,
        output_pad_fn, dq_pad_fn, dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask)

    out_ref, _ = attention_ref(
        q_ref, k_ref, v_ref, query_padding_mask, key_padding_mask, causal=causal,
    )
    out_pt, _ = attention_ref(
        q_ref, k_ref, v_ref, query_padding_mask, key_padding_mask, causal=causal,
        upcast=False, reorder_ops=True,
    )

    # Select Q input: packed (unpad) or padded (seqused)
    if unpad_q:
        q_in = q_unpad_t.detach().to(dtype).requires_grad_()
    else:
        q_in = q.detach().to(dtype).requires_grad_()
    # Select KV input: packed (unpad) or padded (seqused)
    if unpad_kv:
        k_in = k_unpad_t.detach().to(dtype).requires_grad_()
        v_in = v_unpad_t.detach().to(dtype).requires_grad_()
    else:
        k_in = k.detach().to(dtype).requires_grad_()
        v_in = v.detach().to(dtype).requires_grad_()

    out_unpad, lse = flash_attn_varlen_func(
        q_in, k_in, v_in,
        cu_seqlens_q=cu_seqlens_q if unpad_q else None,
        cu_seqlens_k=cu_seqlens_k if unpad_kv else None,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        seqused_q=seqused_q if not unpad_q else None,
        seqused_k=seqused_k if not unpad_kv else None,
        causal=causal,
    )

    if is_fake_mode():
        return

    # Reshape output to (batch, seqlen, nheads, d) for comparison
    out = output_pad_fn(out_unpad) if unpad_q else out_unpad

    # Mask out padding positions — kernel output at padding positions is undefined
    q_mask = rearrange(query_padding_mask, "b s -> b s 1 1")
    out_masked = out.clone().masked_fill_(~q_mask, 0.0)
    out_ref_masked = out_ref.clone().masked_fill_(~q_mask, 0.0)
    out_pt_masked = out_pt.clone().masked_fill_(~q_mask, 0.0)

    fwd_atol = 2 * (out_ref_masked + 0.3 - 0.3 - out_ref_masked).abs().max().item()
    assert (out_masked - out_ref_masked).abs().max().item() <= 2 * (out_pt_masked - out_ref_masked).abs().max().item() + fwd_atol

    # Backward (original test skips all SM90 varlen backward)
    can_bwd = d <= 128 and not IS_SM90
    if not can_bwd:
        return

    g = torch.randn_like(out_unpad)
    dq_in, dk_in, dv_in = torch.autograd.grad(out_unpad, (q_in, k_in, v_in), g)

    # Mask out padding positions again
    k_mask = rearrange(key_padding_mask, "b s -> b s 1 1")
    if not unpad_q:
        dq_in = dq_in.clone().masked_fill_(~q_mask, 0.0)
    if not unpad_kv:
        dk_in = dk_in.clone().masked_fill_(~k_mask, 0.0)
        dv_in = dv_in.clone().masked_fill_(~k_mask, 0.0)

    assert dq_in.isfinite().all(), "dq contains non-finite values"
    assert dk_in.isfinite().all(), "dk contains non-finite values"
    assert dv_in.isfinite().all(), "dv contains non-finite values"
    assert dq_in.abs().max().item() > 0, "dq is all zeros"
    assert dk_in.abs().max().item() > 0, "dk is all zeros"
    assert dv_in.abs().max().item() > 0, "dv is all zeros"

# ---------------------------------------------------------------------------
# Forward-only SM100 2CTA seqused coverage (d=dv=256)
# ---------------------------------------------------------------------------

def _assert_fwd_matches_ref(actual, reference, pytorch_reference):
    finite = torch.isfinite(reference)
    assert torch.equal(actual[~finite], reference[~finite])
    if finite.any():
        actual, reference, pytorch_reference = (
            tensor[finite] for tensor in (actual, reference, pytorch_reference)
        )
        atol = 2 * (reference + 0.3 - 0.3 - reference).abs().max().item()
        assert (actual - reference).abs().max().item() <= 2 * (
            pytorch_reference - reference
        ).abs().max().item() + atol

_SM100_SEQUSED_CASES = [
    pytest.param("padded", "both", torch.bfloat16, "mha", "noncausal", id="padded-boundaries"),
    pytest.param("packed", "q", torch.float16, "gqa", "causal", id="packed-q"),
    pytest.param("packed", "k", torch.bfloat16, "mqa", "local", id="packed-k"),
    pytest.param("packed", "both", torch.float16, "mha", "noncausal", id="packed-both"),
]


@pytest.mark.skipif(not IS_SM100, reason="SM100 2CTA hdim256-only coverage")
@pytest.mark.parametrize(
    "storage_mode,metadata_mode,dtype,mha_type,attention_mode", _SM100_SEQUSED_CASES
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_sm100_hdim256_seqused_fwd(
    storage_mode, metadata_mode, dtype, mha_type, attention_mode
):
    """Exercise logical used lengths independently of physical storage lengths."""
    device = "cuda"
    d = seqlen = 256
    nheads = 4
    nheads_kv = nheads if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    torch.random.manual_seed(256)

    def lengths(values):
        return torch.tensor(values, device=device, dtype=torch.int32)

    if storage_mode == "padded":
        physical_lengths_q = physical_lengths_k = lengths([seqlen] * 7)
        logical_lengths_q = lengths([0, 1, 127, 128, 129, 255, 256])
        logical_lengths_k = lengths([256, 255, 129, 128, 127, 1, 0])
    else:
        physical_lengths_q = lengths([256, 255, 192, 160, 144, 96, 32])
        physical_lengths_k = lengths([32, 96, 144, 160, 192, 255, 256])
        logical_lengths_q = (
            lengths([255, 1, 127, 128, 129, 64, 0])
            if metadata_mode in ("q", "both")
            else physical_lengths_q
        )
        logical_lengths_k = (
            lengths([0, 1, 127, 128, 129, 254, 255])
            if metadata_mode in ("k", "both")
            else physical_lengths_k
        )

    batch_size = physical_lengths_q.numel()
    positions = torch.arange(seqlen, device=device)
    physical_mask_q = positions[None, :] < physical_lengths_q[:, None]
    physical_mask_k = positions[None, :] < physical_lengths_k[:, None]
    logical_mask_q = positions[None, :] < logical_lengths_q[:, None]
    logical_mask_k = positions[None, :] < logical_lengths_k[:, None]
    q = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads_kv, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads_kv, d, device=device, dtype=dtype)
    # Make physically present but logically unused KV rows conspicuous.
    k.masked_fill_(~logical_mask_k[:, :, None, None], 4.0)
    v.masked_fill_(~logical_mask_k[:, :, None, None], 32.0)

    (
        q_unpad, k_unpad, v_unpad, _,
        cu_seqlens_q, cu_seqlens_k, _, _, _, _,
        q_padded, k_padded, v_padded, _,
        output_pad_fn, _, _,
    ) = generate_qkv(q, k, v, physical_mask_q, physical_mask_k)

    packed = storage_mode == "packed"
    use_seqused_q = metadata_mode in ("q", "both")
    use_seqused_k = metadata_mode in ("k", "both")
    causal = attention_mode == "causal"
    window_size = (64, 32) if attention_mode == "local" else (None, None)
    out_buffer = (
        torch.empty(
            batch_size * seqlen * nheads * d + 1, device=device, dtype=dtype
        )[1:].view(batch_size, seqlen, nheads, d)
        if not packed
        else None
    )
    out_raw, lse = flash_attn_varlen_func(
        q_unpad if packed else q_padded,
        k_unpad if packed else k_padded,
        v_unpad if packed else v_padded,
        cu_seqlens_q=cu_seqlens_q if packed else None,
        cu_seqlens_k=cu_seqlens_k if packed else None,
        seqused_q=logical_lengths_q if use_seqused_q else None,
        seqused_k=logical_lengths_k if use_seqused_k else None,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        causal=causal,
        window_size=window_size,
        pack_gqa=False,
        out=out_buffer,
        return_lse=True,
    )
    if is_fake_mode():
        return

    out = output_pad_fn(out_raw) if packed else out_raw
    lse_padded = output_pad_fn(lse.transpose(0, 1)) if packed else lse.permute(0, 2, 1)
    ref_args = (q, k, v, logical_mask_q, logical_mask_k)
    ref_kwargs = dict(causal=causal, window_size=window_size, return_lse=True)
    out_ref, _, lse_ref = attention_ref(*ref_args, **ref_kwargs)
    out_pt, _, lse_pt = attention_ref(
        *ref_args, upcast=False, reorder_ops=True, **ref_kwargs
    )

    # Used-length tails are undefined for both padded and packed storage.
    active = logical_mask_q[:, :, None, None].expand_as(out)
    _assert_fwd_matches_ref(out[active], out_ref[active], out_pt[active])
    lse_mask = logical_mask_q[:, :, None].expand(-1, -1, nheads)
    _assert_fwd_matches_ref(
        lse_padded[lse_mask],
        lse_ref.permute(0, 2, 1)[lse_mask],
        lse_pt.permute(0, 2, 1)[lse_mask],
    )


@pytest.mark.skipif(not IS_SM100, reason="SM100 2CTA hdim256-only coverage")
@pytest.mark.parametrize(
    "use_seqused_k",
    [True, False],
    ids=["padded-seqused-k", "dense-no-metadata"],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_sm100_hdim256_causal_empty_prefix_fwd(use_seqused_k):
    """Cover bottom-right causal rows with no keys, with and without seqused."""
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen_q, seqlen_k_used = 1, 2048, 1024
    seqlen_k = seqlen_q if use_seqused_k else seqlen_k_used
    nheads, d = 1, 256
    torch.random.manual_seed(256)

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    seqused_k = (
        torch.full((batch_size,), seqlen_k_used, device=device, dtype=torch.int32)
        if use_seqused_k
        else None
    )
    key_padding_mask = torch.arange(seqlen_k, device=device)[None, :] < seqlen_k_used
    k.masked_fill_(~key_padding_mask[:, :, None, None], 4.0)
    v.masked_fill_(~key_padding_mask[:, :, None, None], 32.0)

    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        seqused_k=seqused_k,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        causal=True,
        pack_gqa=False,
        return_lse=True,
    )
    if is_fake_mode():
        return

    empty_rows = seqlen_q - seqlen_k_used
    assert torch.equal(out[:, :empty_rows], torch.zeros_like(out[:, :empty_rows]))
    assert torch.equal(
        lse[:, :, :empty_rows],
        torch.full_like(lse[:, :, :empty_rows], float("-inf")),
    )

    ref_args = (q, k, v, None, key_padding_mask)
    ref_kwargs = dict(causal=True, return_lse=True)
    out_ref, _, lse_ref = attention_ref(*ref_args, **ref_kwargs)
    out_pt, _, lse_pt = attention_ref(
        *ref_args, upcast=False, reorder_ops=True, **ref_kwargs
    )
    _assert_fwd_matches_ref(
        out[:, empty_rows:], out_ref[:, empty_rows:], out_pt[:, empty_rows:]
    )
    _assert_fwd_matches_ref(lse, lse_ref, lse_pt)


# ---------------------------------------------------------------------------
# Combine kernel
# ---------------------------------------------------------------------------

def attention_combine_ref(out_partial, lse_partial):
    lse = torch.logsumexp(lse_partial, dim=0)
    scale = torch.exp(lse_partial - lse)
    scale = torch.where(torch.isinf(scale) | torch.isnan(scale), torch.zeros_like(scale), scale)
    out = (scale.unsqueeze(-1) * out_partial).sum(0)
    return out, lse


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("seqlen", [32, 256])
@pytest.mark.parametrize("num_splits", [2, 5, 17])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_combine(num_splits, seqlen, d, dtype):
    device = "cuda"
    torch.random.manual_seed(1)
    batch_size = 3
    nheads = 8

    # out_partial: (num_splits, batch, seqlen, nheads, d) with stride(-1)==1
    # lse_partial: (num_splits, batch, seqlen, nheads) with stride(-2)==1 (seqlen contiguous)
    out_partial = torch.randn(
        num_splits, batch_size, seqlen, nheads, d, device=device, dtype=torch.float32,
    )
    lse_partial = torch.randn(
        num_splits, batch_size, nheads, seqlen, device=device, dtype=torch.float32,
    ).transpose(-1, -2)
    lse_partial[num_splits // 2 :, : batch_size // 3] = -float("inf")

    out, lse = flash_attn_combine(out_partial, lse_partial, out_dtype=dtype, return_lse=True)
    if is_fake_mode():
        return
    out_ref, lse_ref = attention_combine_ref(out_partial, lse_partial)
    out_pt = out_ref.to(dtype)

    assert torch.allclose(lse, lse_ref, atol=1e-5, rtol=1e-5)
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() or torch.allclose(out, out_pt, atol=1e-5, rtol=1e-5)
