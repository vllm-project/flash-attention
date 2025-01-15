# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

import torch
import triton
import triton.language as tl
# from vllm_flash_attn import flash_attn_varlen_func

# try:
#     from .vertical_slash_index_minf import get_kernel
# except:
#     from vllm.attention.backends.cuda.vertical_slash_index_minf import get_kernel

# convert_vertical_slash_indexes = get_kernel().convert_vertical_slash_indexes

# @triton.autotune(
#    configs=[
#        triton.Config({}, num_stages=1, num_warps=4),
#        triton.Config({}, num_stages=1, num_warps=8),
#        triton.Config({}, num_stages=2, num_warps=4),
#        triton.Config({}, num_stages=2, num_warps=8),
#        triton.Config({}, num_stages=3, num_warps=4),
#        triton.Config({}, num_stages=3, num_warps=8),
#        triton.Config({}, num_stages=4, num_warps=4),
#        triton.Config({}, num_stages=4, num_warps=8),
#        triton.Config({}, num_stages=5, num_warps=4),
#        triton.Config({}, num_stages=5, num_warps=8),
#    ],
#    key=['N_CTX'],
# )
@triton.jit
def _triton_mixed_sparse_attn_fwd_kernel(
    Q, K, V, q_seqlens, kv_seqlens, sm_scale,
    block_count, block_offset, column_count, column_index,
    Out, softmax_lse,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_sz, stride_sh, stride_sm, stride_sk,
    Z, H, N_CTX,
    NUM_ROWS, NNZ_S, NNZ_V,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
    causal: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    # tl.device_print("start_m", start_m)
    # tl.device_print("off_hz", off_hz)

    q_seqlen = tl.load(q_seqlens + off_hz // H)
    kv_seqlen = tl.load(kv_seqlens + off_hz // H)
    if start_m * BLOCK_M >= q_seqlen:
        return
    # tl.device_print("q_seqlen", q_seqlen)
    
    
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh
    s_offset = (off_hz // H) * stride_sz + (off_hz % H) * stride_sh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    s_ptrs = softmax_lse + s_offset + offs_m[:, None] * stride_sm

    num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m)
    blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S
    num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m)
    cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    # if start_m == 0 and off_hz == 0:
    #     tl.device_print("q", q)
    q = (q * qk_scale).to(dtype)
    # q = (q * qk_scale).to(tl.float32)
    # q = (q * qk_scale).to(tl.float32)

    # loop over k, v and update accumulator
    m_mask = offs_m[:, None] < q_seqlen
    # if start_m == 0 and off_hz == 0:
    #     tl.device_print("num_blks", num_blks)
    iter = 0
    for block_index in range(num_blks):
        iter += 1
        # if start_m == 0 and off_hz == 0:
        #     tl.device_print("iter", iter)
        # if start_m == 0 and off_hz == 0:
        #     tl.device_print("block_index", block_index)
        start_n = tl.load(blks_ptr + block_index)
        # if start_m == 0 and off_hz == 0:
        #     tl.device_print("start_n", start_n)
        cols = start_n + offs_n
        # if start_m == 0 and off_hz == 0:
        #     tl.device_print("cols", cols)
        n_mask = cols < kv_seqlen
        # if start_m == 0 and off_hz == 0:
        #     tl.device_print("n_mask", n_mask)
        # if (start_m == 24 and off_hz == 0) and block_index == 0:
        # #     tl.device_print("start_n", start_n)
        #     tl.device_print("block_index", block_index)
        #     tl.device_print("kv_seqlen", kv_seqlen)
        #     tl.device_print("n_mask", n_mask)
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        # if (start_m == 150 and off_hz == 0) and block_index == 0:
        #     tl.device_print("k", k)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        # if start_m == 0 and off_hz == 0:
        #     tl.device_print("k", k)
        # if start_m == 0 and off_hz == 0:
        #     tl.device_print("v", v)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if causal:
            causal_mask = cols[None, :] - (kv_seqlen - q_seqlen) <= offs_m[:, None]
            qk = tl.where(m_mask & causal_mask & n_mask, qk, float("-inf"))
        else:
            qk = tl.where(m_mask & n_mask , qk, float("-inf")) # 添加n_mask生效
        qk += tl.dot(q, k.to(dtype))
        # if start_m == 0 and off_hz == 0:
        #     tl.device_print("qk", qk)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        # if start_m == 0 and off_hz == 0:
        #     tl.device_print("m_i_new", m_i_new)
        m_i_new_clamp = tl.clamp(m_i_new, -100000, 100000)
        alpha = tl.math.exp2(m_i - m_i_new_clamp)
        # if start_m == 0 and off_hz == 0:
        #     tl.device_print("alpha", alpha)
        p = tl.math.exp2(qk - m_i_new_clamp[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v.to(dtype))
        # if start_m == 150 and off_hz == 0:
        #     tl.device_print("qk", qk)
        #     # tl.device_print("alpha", alpha)
        #     # tl.device_print("acc", acc)
        #     # tl.device_print("p", p)
            # tl.device_print("m_i_new_clamp", m_i_new_clamp)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # if start_m == 0 and off_hz == 0:
    #     tl.device_print("l_i", l_i)
    #     tl.device_print("acc", acc)
    #     tl.device_print("m_i", m_i)

    # if start_m == 0 and off_hz == 0:
    #     # tl.device_print("num_cols", num_cols)
    #     tl.device_print("offs_m", offs_m)
    for start_n in range(0, num_cols, BLOCK_N):
        iter += 1
        # if (start_m == 0 and off_hz == 0) and start_n == 0:
        # #     # tl.device_print("iter", iter)
        #     tl.device_print("start_n", start_n)
        n_mask = start_n + offs_n < num_cols
        # if start_m == 0 and off_hz == 0:
        #     tl.device_print("n_mask", n_mask)        
        cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0)
        # if start_m == 0 and off_hz == 0:
        #     tl.device_print("cols", cols)         
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        # if (start_m == 0 and off_hz == 0) and start_n == 0:
        #     tl.device_print("k", k)

        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if causal:
            causal_mask = cols[None, :] - (kv_seqlen - q_seqlen) <= offs_m[:, None]
            # if (start_m == 0 and off_hz == 0) and start_n == 64:
            #     # tl.device_print("v", v)
            #     tl.device_print("cols", cols)
            #     tl.device_print("offs_m", offs_m)
            #     tl.device_print("causal_mask", causal_mask)
            qk = tl.where(m_mask & n_mask & causal_mask, qk, float("-inf"))
        else:
            qk = tl.where(m_mask & n_mask, qk, float("-inf"))
        qk += tl.dot(q, k.to(dtype))
        # if start_m == 0 and off_hz == 0:
        #     tl.device_print("qk", qk)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        # if start_m == 0 and off_hz == 0:
        #     tl.device_print("m_i_new", m_i_new)
        m_i_new_clamp = tl.clamp(m_i_new, -100000, 100000)
        # if start_m == 0 and off_hz == 0:
        #     tl.device_print("m_i_new_clamp", m_i_new_clamp)
        alpha = tl.math.exp2(m_i - m_i_new_clamp)
        # if start_m == 0 and off_hz == 0:
        #     tl.device_print("alpha", alpha)        
        p = tl.math.exp2(qk - m_i_new_clamp[:, None])
        # if start_m == 0 and off_hz == 0:
        #     tl.device_print("p", p) 
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v.to(dtype))
        # if start_m == 150 and off_hz == 0:
        #     tl.device_print("m_i_new_clamp", m_i_new_clamp)
        #     tl.device_print("p", p)
        #     tl.device_print("v", v)
        #     tl.device_print("acc2", acc)
            # tl.device_print("p2", p)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # write back O
    acc /= l_i[:, None]
    # acc = tl.where(m_mask, acc / l_i[:, None], 0.0)
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)
    lse = (tl.log2(l_i) + m_i) / 1.44269504 # 为什么要加m_i
    # if start_m == 150 and off_hz == 0:
    #     tl.device_print("l_i", l_i)
    # #     # tl.device_print("acc", acc)
    #     tl.device_print("lse", lse)
    #     tl.device_print("m_i", m_i)
    # if lse == 0:
    #     tl.device_print("l_i", l_i)
    #     tl.device_print("m_i", m_i)
    # tl.store(s_ptrs, lse[:, None].to(dtype), mask=m_mask)
    tl.store(s_ptrs, lse[:, None], mask=m_mask)


def _triton_mixed_sparse_attention(
    q: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    q_seqlens: torch.Tensor,    # [BATCH, ]
    kv_seqlens: torch.Tensor,
    block_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    block_offset: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
    column_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    column_index: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
    sm_scale: float,
    block_size_M: int = 64,
    block_size_N: int = 64,
    causal: bool = True,
) -> torch.Tensor:
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.zeros_like(q)
    softmax_lse = torch.zeros(q.shape[0], q.shape[1], q.shape[2], 1, device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    # import pdb 
    # pdb.set_trace()
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    # dtype = tl.float16
    _triton_mixed_sparse_attn_fwd_kernel[grid](
        q, k, v, q_seqlens, kv_seqlens, sm_scale,
        block_count, block_offset, column_count, column_index,
        o, softmax_lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2), softmax_lse.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        block_count.shape[-1], block_offset.shape[-1], column_index.shape[-1],
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=4, num_stages=2,
        causal=causal,
    )

    return o, softmax_lse


def vertical_slash_sparse_attention(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,    # [BATCH, N_HEADS, N_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    s_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    softmax_scale: float,
    causal: bool = True,
    stage: str = "intra",
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    if stage == "intra":
        assert causal 
    else:
        assert not causal

    batch_size, num_heads, context_size, head_dim = query.shape
    _, _, kv_seq_len, _ = key.shape
    pad = block_size_M - (context_size & (block_size_M - 1))
    query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0]) # pad on the right of context_size dim
    key = torch.nn.functional.pad(key, [0, 0, 0, pad, 0, 0, 0, 0])
    value = torch.nn.functional.pad(value, [0, 0, 0, pad, 0, 0, 0, 0])

    if head_dim not in [16, 32, 64, 128, 256, 512]:
        target_dim = 2 ** math.ceil(math.log2(head_dim)) - head_dim
        query = torch.nn.functional.pad(query, [0, target_dim, 0, 0, 0, 0, 0, 0])
        key = torch.nn.functional.pad(key, [0, target_dim, 0, 0, 0, 0, 0, 0])
        value = torch.nn.functional.pad(value, [0, target_dim, 0, 0, 0, 0, 0, 0])

    v_idx = v_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=False)[0]
    s_idx = s_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=True)[0]
    q_seqlens = torch.tensor([context_size], dtype=torch.int32, device=query.device)
    kv_seqlens = torch.tensor([kv_seq_len], dtype=torch.int32, device=query.device)
    # sm_scale = head_dim ** -0.5
    sm_scale = softmax_scale if isinstance(softmax_scale, float) else softmax_scale.item()
    # import pdb 
    # pdb.set_trace()
    # if context_size < 32768:
    #     import pdb 
    #     pdb.set_trace()
    #     import pickle as pkl
    #     # pkl.dump((q_seqlens, kv_seqlens, v_idx, s_idx, context_size, block_size_M, block_size_N,), open("data.pkl", "wb"))
    # pkl.dump((output, softmax_lse), open("first_chunk.pkl", "wb"))
    # pkl.dump((flash_results[2][0][0], flash_results[2][0][1], flash_results[2][1][0], flash_results[2][1][1], flash_results[2][2][0], flash_results[2][2][1]), open("three_chunk.pkl", "wb"))
    #     pkl.dump((query, key, value), open("qkv.pkl", "wb"))
    block_count, block_offset, column_count, column_index = convert_vertical_slash_indexes(
        q_seqlens, kv_seqlens, v_idx, s_idx, context_size, block_size_M, block_size_N, causal,
    )
    out, softmax_lse = _triton_mixed_sparse_attention(
        query, key, value, q_seqlens, kv_seqlens,
        block_count, block_offset, column_count, column_index,
        sm_scale, block_size_M, block_size_N, causal
    )
    return out[..., :context_size, :head_dim], softmax_lse[..., :context_size, :]
