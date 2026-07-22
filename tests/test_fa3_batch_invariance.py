import pytest
import torch

from vllm_flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
    get_scheduler_metadata,
    is_fa_version_supported,
)


@pytest.mark.skipif(
    not is_fa_version_supported(3), reason="FlashAttention 3 is not supported"
)
@pytest.mark.parametrize("aot_schedule", [False, True])
@torch.inference_mode()
def test_paged_kv_batch_invariant(aot_schedule: bool) -> None:
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    dtype = torch.bfloat16
    block_size = 16
    num_q_heads, num_kv_heads, head_size = 14, 2, 64
    target_q_len, neighbor_q_len = 1, 129
    target_kv_len, neighbor_kv_len = 1024, 1536
    blocks_per_row = neighbor_kv_len // block_size

    target_q = torch.randn(target_q_len, num_q_heads, head_size, dtype=dtype)
    neighbor_q = torch.randn(neighbor_q_len, num_q_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        blocks_per_row * 2,
        block_size,
        num_kv_heads,
        head_size,
        dtype=dtype,
    )
    value_cache = torch.randn_like(key_cache)
    target_blocks = torch.arange(blocks_per_row, dtype=torch.int32)
    neighbor_blocks = torch.arange(
        blocks_per_row, blocks_per_row * 2, dtype=torch.int32
    )

    def run(include_neighbor: bool) -> torch.Tensor:
        if include_neighbor:
            query = torch.cat((target_q, neighbor_q))
            cu_seqlens_q = torch.tensor(
                [0, target_q_len, target_q_len + neighbor_q_len], dtype=torch.int32
            )
            seqused_k = torch.tensor(
                [target_kv_len, neighbor_kv_len], dtype=torch.int32
            )
            block_table = torch.stack((target_blocks, neighbor_blocks))
            max_seqlen_q, max_seqlen_k = neighbor_q_len, neighbor_kv_len
        else:
            query = target_q
            cu_seqlens_q = torch.tensor([0, target_q_len], dtype=torch.int32)
            seqused_k = torch.tensor([target_kv_len], dtype=torch.int32)
            block_table = target_blocks.unsqueeze(0).contiguous()
            max_seqlen_q, max_seqlen_k = target_q_len, target_kv_len

        scheduler_metadata = None
        if aot_schedule:
            scheduler_metadata = get_scheduler_metadata(
                batch_size=seqused_k.numel(),
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                num_heads_q=num_q_heads,
                num_heads_kv=num_kv_heads,
                headdim=head_size,
                cache_seqlens=seqused_k,
                qkv_dtype=dtype,
                cu_seqlens_q=cu_seqlens_q,
                page_size=block_size,
                causal=True,
                batch_invariant=True,
            )
        output = flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=head_size**-0.5,
            causal=True,
            block_table=block_table,
            scheduler_metadata=scheduler_metadata,
            fa_version=3,
            batch_invariant=True,
        )
        return output[:target_q_len]

    torch.testing.assert_close(run(False), run(True), rtol=0, atol=0)
