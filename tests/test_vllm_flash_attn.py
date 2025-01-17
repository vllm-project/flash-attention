#
# This file is copied verbatim from vLLM:
# https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_flash_attn.py
#

from typing import List, Optional, Tuple

import pytest
import torch
import itertools

import flash_attn_wrapper  # noqa: F401

NUM_HEADS = [(4, 4), (8, 2), (16, 2)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16, 32]
DTYPES = [torch.float16, torch.bfloat16]
# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [32768, 2048]


def ref_paged_attn_kvc(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        query_lens: List[int],
        kv_lens: List[int],
        block_tables: torch.Tensor,
        scale: float,
        sliding_window: Optional[int] = None,
        soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    num_kv_heads = block_tables.shape[1]
    num_heads = query.shape[1]
    _, block_size, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        q = query[start_idx:start_idx + query_len]
        q *= scale

        outs = []
        for j in range(num_kv_heads):
            queries_per_key = num_heads // num_kv_heads
            q_ = q[:,j*queries_per_key:(j+1)*queries_per_key]

            kv_len = kv_lens[i * num_kv_heads + j]
            num_kv_blocks = (kv_len + block_size - 1) // block_size
            block_indices = block_tables[i, j, :num_kv_blocks]

            k = key_cache[block_indices]
            k = k.view(-1, 1, head_size)
            k = k[:kv_len]
            v = value_cache[block_indices]
            v = v.view(-1, 1, head_size)
            v = v[:kv_len]

            k = torch.repeat_interleave(k, queries_per_key, dim=1)
            v = torch.repeat_interleave(v, queries_per_key, dim=1)
            attn = torch.einsum("qhd,khd->hqk", q_, k).float()
            empty_mask = torch.ones(query_len, kv_len)
            mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
            if sliding_window is not None:
                sliding_window_mask = torch.triu(empty_mask,
                                                diagonal=kv_len -
                                                        (query_len + sliding_window) +
                                                        1).bool().logical_not()
                mask |= sliding_window_mask
            if soft_cap is not None:
                attn = soft_cap * torch.tanh(attn / soft_cap)
            attn.masked_fill_(mask[None], float("-inf"))
            attn = torch.softmax(attn, dim=-1).to(v.dtype)
            out = torch.einsum("hqk,khd->qhd", attn, v)
            outs.append(out)

        outputs.append(torch.cat(outs, dim=1))

        start_idx += query_len

    return torch.cat(outputs, dim=0)


def ref_paged_attn(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        query_lens: List[int],
        kv_lens: List[int],
        block_tables: torch.Tensor,
        scale: float,
        sliding_window: Optional[int] = None,
        soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        # clone to avoid clobbering the query tensor
        q = query[start_idx:start_idx + query_len].clone()
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size

        block_indices = block_tables[i, :num_kv_blocks]
        k = key_cache[block_indices]
        k = k.view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices]
        v = v.view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = torch.triu(empty_mask,
                                             diagonal=kv_len -
                                                      (query_len + sliding_window) +
                                                      1).bool().logical_not()
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize("kv_lens", [[1328, 18, 463], [1, 54, 293, 70]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("kvc", [True, False])
@torch.inference_mode()
def test_flash_attn_with_paged_kv(
        kv_lens: List[int],
        num_heads: Tuple[int, int],
        head_size: int,
        dtype: torch.dtype,
        block_size: int,
        soft_cap: Optional[float],
        num_blocks: int,
        kvc: bool,
) -> None:
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(num_blocks,
                            block_size,
                            num_kv_heads,
                            head_size,
                            dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    if kvc:
        key_cache = (key_cache.transpose(1, 2)
                              .transpose(0, 1)
                              .reshape(-1, block_size, head_size)).contiguous()
        value_cache = (value_cache.transpose(1, 2)
                                  .transpose(0, 1)
                                  .reshape(-1, block_size, head_size)).contiguous()
        kv_lens = list(itertools.chain.from_iterable(
            [l // 2] + [l] * (num_kv_heads - 2) + [l // 3]
            for l in kv_lens))
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 num_blocks,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)
    if kvc:
        block_tables = (block_tables[:,None]
                        + torch.arange(num_kv_heads).type(torch.int)[None,:,None]
                        * num_blocks)

    output = torch.ops.vllm.flash_attn_with_kvcache(
        decode_query=query.unsqueeze(1),
        key_cache=key_cache,
        value_cache=value_cache,
        softmax_scale=scale,
        causal=True,
        block_table=block_tables,
        cache_seqlens=kv_lens_tensor,
        softcap=soft_cap if soft_cap is not None else 0,
    ).squeeze(1)

    if num_blocks <= 2048:
        test_utils = ["test_faketensor", "test_schema"]
    else:
        test_utils = ["test_faketensor"]

    torch.library.opcheck(torch.ops.vllm.flash_attn_with_kvcache,
                          args=tuple(),
                          kwargs=dict(
                              decode_query=query.unsqueeze(1),
                              key_cache=key_cache,
                              value_cache=value_cache,
                              softmax_scale=scale,
                              causal=True,
                              block_table=block_tables,
                              cache_seqlens=kv_lens_tensor,
                              softcap=soft_cap if soft_cap is not None else 0,
                          ),
                          test_utils=test_utils)

    ref_func = ref_paged_attn_kvc if kvc else ref_paged_attn
    ref_output = ref_func(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[1] * num_seqs,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        soft_cap=soft_cap,
    )
    torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(output - ref_output))}"


@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18), (129, 463)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("kvc", [True, False])
@torch.inference_mode()
def test_varlen_with_paged_kv(
        seq_lens: List[Tuple[int, int]],
        num_heads: Tuple[int, int],
        head_size: int,
        sliding_window: Optional[int],
        dtype: torch.dtype,
        block_size: int,
        soft_cap: Optional[float],
        num_blocks: int,
        kvc: bool,
) -> None:
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    if kvc:
        kv_lens = list(itertools.chain.from_iterable(
            [l // 2] + [l] * (num_kv_heads - 2) + [l // 3]
            for l in kv_lens))
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = ((sliding_window,
                    sliding_window) if sliding_window is not None else
                   (-1, -1))
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens),
                        num_query_heads,
                        head_size,
                        dtype=dtype)
    key_cache = torch.randn(num_blocks,
                            block_size,
                            num_kv_heads,
                            head_size,
                            dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    if kvc:
        key_cache = (key_cache.transpose(1, 2)
                              .transpose(0, 1)
                              .reshape(-1, block_size, head_size)).contiguous()
        value_cache = (value_cache.transpose(1, 2)
                                  .transpose(0, 1)
                                  .reshape(-1, block_size, head_size)).contiguous()
    cu_query_lens = torch.tensor([0] + query_lens,
                                 dtype=torch.int32).cumsum(dim=0,
                                                           dtype=torch.int32)
    kv_lens_ = kv_lens
    cu_kv_lens = torch.tensor([0] + kv_lens_,
                              dtype=torch.int32).cumsum(dim=0,
                                                        dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                num_blocks,
                                (num_seqs, max_num_blocks_per_seq),
                                dtype=torch.int32)
    if kvc:
        block_tables = (block_tables[:,None]
                        + torch.arange(num_kv_heads).type(torch.int)[None,:,None]
                        * num_blocks)

    output = torch.ops.vllm.flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
    )

    if num_blocks <= 2048:
        test_utils = ["test_faketensor", "test_schema"]
    else:
        test_utils = ["test_faketensor"]

    torch.library.opcheck(torch.ops.vllm.flash_attn_varlen_func,
                          args=tuple(),
                          kwargs=dict(
                              q=query,
                              k=key_cache,
                              v=value_cache,
                              cu_seqlens_q=cu_query_lens,
                              cu_seqlens_k=cu_kv_lens,
                              max_seqlen_q=max_query_len,
                              max_seqlen_k=max_kv_len,
                              softmax_scale=scale,
                              causal=True,
                              window_size=window_size,
                              block_table=block_tables,
                              softcap=soft_cap if soft_cap is not None else 0,
                          ),
                          test_utils=test_utils)

    ref_func = ref_paged_attn_kvc if kvc else ref_paged_attn
    ref_output = ref_func(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens_,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )

    torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(output - ref_output))}"

@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_lens", [(1, 1), (1, 1024), (1, 2048), (1023, 2049), (1023, 1023), (32, 32), (65, 65), (129, 129)])
@pytest.mark.parametrize("num_heads", [1, 2, 4])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("NNZ_S", [0, 1, 2, 3, 7, 15, 32])
@torch.inference_mode()
def test_sparse_attention(
        batch_size: int,
        seq_lens: Tuple[int, int],
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
        NNZ_S: int,
) -> None:
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    block_size_M = 64
    block_size_N = 64
    seqlen_q, seqlen_k = seq_lens
    q = torch.randn(
        batch_size, seqlen_q, num_heads, head_size, dtype=dtype, requires_grad=False
    )
    k = torch.randn(
        batch_size, seqlen_k, num_heads, head_size, dtype=dtype, requires_grad=False
    )
    v = torch.randn(
        batch_size, seqlen_k, num_heads, head_size, dtype=dtype, requires_grad=False
    )
    NUM_ROWS = (seqlen_q + block_size_M - 1) // block_size_M
    if NNZ_S * block_size_N > seqlen_k:
        return
    NNZ_V = seqlen_k - NNZ_S * block_size_N
    block_count = torch.tensor([NNZ_S] * batch_size * NUM_ROWS * num_heads, dtype=torch.int32).reshape(batch_size, num_heads, NUM_ROWS)
    column_count = torch.tensor([NNZ_V] * batch_size * NUM_ROWS * num_heads, dtype=torch.int32).reshape(batch_size, num_heads, NUM_ROWS)
    block_offset = torch.tensor([[i * block_size_N for i in range(NNZ_S)]] * batch_size * NUM_ROWS * num_heads, dtype=torch.int32).reshape(batch_size, num_heads, NUM_ROWS, NNZ_S)
    column_index = torch.tensor([[NNZ_S * block_size_N + i for i in range(NNZ_V)]] * batch_size * NUM_ROWS * num_heads, dtype=torch.int32).reshape(batch_size, num_heads, NUM_ROWS, NNZ_V)
    from vllm_flash_attn import sparse_attn_func, flash_attn_func
    out, lse = sparse_attn_func(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        return_softmax_lse=True,
    )

    ref_out, ref_lse = flash_attn_func(
        q,
        k,
        v,
        return_softmax_lse=True,
    )

    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(out - ref_out))}"
    torch.testing.assert_close(lse, ref_lse, atol=2e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(lse - ref_lse))}"

@pytest.mark.parametrize("seq_lens", [[(1024, 1328)],
                                      [(1024, 1328), (1, 2048)],
                                      [(1025, 1328), (2, 2048)],
                                      [(1025, 2049), (2, 1281)],
                                     ])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_sparse_attention_varlen(
        seq_lens: List[Tuple[int, int]],
        head_size: int,
        dtype: torch.dtype,
) -> None:
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    block_size_M = 64
    block_size_N = 64
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_heads = 1
    query = torch.randn(sum(query_lens),
                        num_heads,
                        head_size,
                        dtype=dtype)
    key = torch.randn(sum(kv_lens),
                      num_heads,
                      head_size,
                      dtype=dtype)
    value = torch.randn_like(key)
    cu_query_lens = torch.tensor([0] + query_lens,
                                 dtype=torch.int32).cumsum(dim=0,
                                                           dtype=torch.int32)
    cu_kv_lens = torch.tensor([0] + kv_lens,
                                 dtype=torch.int32).cumsum(dim=0,
                                                           dtype=torch.int32)
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)

    NUM_ROWS = (max_query_len + block_size_M - 1) // block_size_M
    NNZ_S = 20
    NNZ_V = 2048
    batch_size = len(query_lens)

    block_counts = []
    column_counts = []
    block_offsets = []
    column_indices = []
    for b in range(batch_size):
        block_counts.append(torch.tensor([NNZ_S] * NUM_ROWS * num_heads, dtype=torch.int32).reshape(num_heads, NUM_ROWS))
        columns = kv_lens[b] - NNZ_S * block_size_N
        column_counts.append(torch.tensor([columns] * NUM_ROWS * num_heads, dtype=torch.int32).reshape(num_heads, NUM_ROWS))
        block_offsets.append(torch.tensor([[i * block_size_N for i in range(NNZ_S)]] * NUM_ROWS * num_heads, dtype=torch.int32).reshape(num_heads, NUM_ROWS, NNZ_S))
        column_indices.append(torch.tensor([[NNZ_S * block_size_N + i for i in range(NNZ_V)]] * NUM_ROWS * num_heads, dtype=torch.int32).reshape(num_heads, NUM_ROWS, NNZ_V))
    block_count = torch.concat(block_counts).reshape(batch_size, num_heads, NUM_ROWS)
    column_count = torch.concat(column_counts).reshape(batch_size, num_heads, NUM_ROWS)
    block_offset = torch.concat(block_offsets).reshape(batch_size, num_heads, NUM_ROWS, NNZ_S)
    column_index = torch.concat(column_indices).reshape(batch_size, num_heads, NUM_ROWS, NNZ_V)
    from vllm_flash_attn import sparse_attn_varlen_func, flash_attn_varlen_func
    out, lse = sparse_attn_varlen_func(
        query,
        key,
        value,
        block_count,
        block_offset,
        column_count,
        column_index,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        return_softmax_lse=True,
    )

    ref_out, ref_lse = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        return_softmax_lse=True,
    )

    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(out - ref_out))}"
    torch.testing.assert_close(lse, ref_lse, atol=2e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(lse - ref_lse))}"
