"""Regression coverage for copy-free FA2 GQA decode query addressing."""

import math

import pytest
import torch

from vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_heads", [(8, 2), (24, 4), (8, 1)])
@pytest.mark.parametrize("strided_query", [False, True])
@torch.inference_mode()
def test_varlen_decode_query_strides(dtype, num_heads, strided_query):
    torch.manual_seed(0)
    batch, head_dim, kv_len = 8, 128, 257
    heads, kv_heads = num_heads
    factor = 2 if strided_query else 1
    storage = torch.randn(
        batch * factor, heads * factor, head_dim, device="cuda", dtype=dtype
    )
    query = storage[1::2, 1::2] if strided_query else storage
    original = query.clone()
    key = torch.randn(batch * kv_len, kv_heads, head_dim, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    cu_query = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    cu_key = cu_query * kv_len
    destination = torch.empty(batch, heads, head_dim, device="cuda", dtype=dtype)

    def run(out):
        return flash_attn_varlen_func(
            query,
            key,
            value,
            max_seqlen_q=1,
            cu_seqlens_q=cu_query,
            max_seqlen_k=kv_len,
            cu_seqlens_k=cu_key,
            causal=True,
            out=out,
            fa_version=2,
        )

    default = run(None).clone()
    run(destination)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
    ) as prof:
        returned = run(destination)
    query_copies = [
        event
        for event in prof.events()
        if event.name == "aten::clone"
        and event.input_shapes
        and math.prod(event.input_shapes[0]) == query.numel()
    ]
    assert not query_copies, "GQA decode materialized another query tensor"
    assert returned.data_ptr() == destination.data_ptr()
    assert torch.equal(returned, default)
    assert torch.equal(query, original)

    groups = heads // kv_heads
    expected = (
        torch.nn.functional.scaled_dot_product_attention(
            query.unsqueeze(2).float(),
            key.view(batch, kv_len, kv_heads, head_dim)
            .transpose(1, 2)
            .repeat_interleave(groups, dim=1)
            .float(),
            value.view(batch, kv_len, kv_heads, head_dim)
            .transpose(1, 2)
            .repeat_interleave(groups, dim=1)
            .float(),
        )
        .squeeze(2)
        .to(dtype)
    )
    torch.testing.assert_close(returned, expected, atol=2e-2, rtol=1e-2)
