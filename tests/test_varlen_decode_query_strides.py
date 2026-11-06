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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["head_padding", "batch_padding", "storage_offset"])
@torch.inference_mode()
def test_varlen_decode_unaligned_query(dtype, layout):
    torch.manual_seed(42)
    batch, heads, kv_heads, dim, kv_len = 4, 32, 8, 128, 128
    if layout == "head_padding":
        query = torch.randn(batch, heads, dim + 1, device="cuda", dtype=dtype)[
            ..., :dim
        ]
    elif layout == "batch_padding":
        stride = heads * dim + 1
        storage = torch.randn(batch * stride, device="cuda", dtype=dtype)
        query = torch.as_strided(storage, (batch, heads, dim), (stride, dim, 1))
    else:
        storage = torch.randn(batch * heads * dim + 1, device="cuda", dtype=dtype)
        query = storage[1:].view(batch, heads, dim)
        assert query.is_contiguous()
    original = query.clone(memory_format=torch.contiguous_format)
    key = torch.randn(batch * kv_len, kv_heads, dim, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    cu_query = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    cu_key = cu_query * kv_len

    def run(q):
        return flash_attn_varlen_func(
            q,
            key,
            value,
            max_seqlen_q=1,
            cu_seqlens_q=cu_query,
            max_seqlen_k=kv_len,
            cu_seqlens_k=cu_key,
            causal=True,
            return_softmax_lse=True,
            fa_version=2,
        )

    expected, expected_lse = run(original)
    result, lse = run(query)
    assert torch.equal(result, expected)
    assert torch.equal(lse, expected_lse)
    assert torch.equal(query, original)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, captured_lse = run(query)
    query.copy_(original * 0.5)
    expected, expected_lse = run(query.clone(memory_format=torch.contiguous_format))
    graph.replay()
    assert torch.equal(captured, expected)
    assert torch.equal(captured_lse, expected_lse)
