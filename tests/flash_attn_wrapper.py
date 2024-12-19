from typing import Optional, List, Tuple
import torch

from vllm_flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
from vllm_flash_attn import flash_attn_with_kvcache as _flash_attn_with_kvcache
from vllm_flash_attn import sparse_attn_func as _sparse_attn_func


@torch.library.custom_op("vllm::flash_attn_varlen_func", mutates_args=[])
def flash_attn_varlen_func(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Optional[List[int]] = None,
        softcap: float = 0.0,
        alibi_slopes: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # custom op does not support tuple input
    real_window_size: Tuple[int, int]
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])
    return _flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=real_window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        block_table=block_table,
    )


@flash_attn_varlen_func.register_fake  # type: ignore
def _(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Optional[List[int]] = None,
        softcap: float = 0.0,
        alibi_slopes: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op("vllm::flash_attn_with_kvcache", mutates_args=[])
def flash_attn_with_kvcache(
        decode_query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_seqlens: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        alibi_slopes: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
) -> torch.Tensor:
    return _flash_attn_with_kvcache(
        decode_query,
        key_cache,
        value_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=causal,
        alibi_slopes=alibi_slopes,
        softcap=softcap,
    )

@flash_attn_with_kvcache.register_fake  # type: ignore
def _(
        decode_query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_seqlens: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        alibi_slopes: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
) -> torch.Tensor:
    return torch.empty_like(decode_query)

@torch.library.custom_op("vllm::sparse_attn_func", mutates_args=[])
def sparse_attn_func(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_count: torch.Tensor,
        block_offset: torch.Tensor,
        column_count: torch.Tensor,
        column_index: torch.Tensor,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        softcap: float = 0.0,
        alibi_slopes: Optional[torch.Tensor] = None,
        return_softmax_lse: Optional[bool] = False,
) -> torch.Tensor:
    return _sparse_attn_func(
        q=q,
        k=k,
        v=v,
        block_count=block_count,
        block_offset=block_offset,
        column_count=column_count,
        column_index=column_index,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        return_softmax_lse=return_softmax_lse,
    )
