"""Regression for logical-to-physical page mapping in SM90 sparse TMA loads."""

import pytest
import torch

from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from flash_attn.cute.interface import _flash_attn_fwd


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("qheads_per_kvhead", [1, 4])
@pytest.mark.parametrize("block_lists", ["full", "masked", "mixed"])
def test_sm90_paged_sparse_tma(batch_size, qheads_per_kvhead, block_lists):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("SM90 runtime regression")
    torch.manual_seed(42)
    device, dtype = "cuda", torch.bfloat16
    page_size, seqlen_k, seqlen_q, kvheads, dim = 128, 450, 17, 2, 128
    qheads = kvheads * qheads_per_kvhead
    # Reverse physical pages and leave extra pages in the pool: logical IDs
    # cannot accidentally address the correct data, including for batch zero.
    num_pages = batch_size * 4 + 3
    page_table = torch.arange(num_pages - 1, 2, -1, device=device, dtype=torch.int32)
    page_table = page_table.reshape(batch_size, 4)
    q = torch.randn(batch_size, seqlen_q, qheads, dim, device=device, dtype=dtype)
    k = torch.randn(num_pages, page_size, kvheads, dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    full = [0, 2] if block_lists != "masked" else []
    masked = [0, 2, 3] if block_lists == "masked" else ([3] if block_lists == "mixed" else [])

    def counts_and_indices(blocks):
        counts = torch.full((batch_size, 1, 1), len(blocks), device=device, dtype=torch.int32)
        indices = torch.zeros((batch_size, 1, 1, 4), device=device, dtype=torch.int32)
        if blocks:
            indices[..., :len(blocks)] = torch.tensor(blocks, device=device, dtype=torch.int32)
        return counts, indices

    mask_cnt, mask_idx = counts_and_indices(masked)
    full_cnt, full_idx = counts_and_indices(full)
    sparse = BlockSparseTensorsTorch(mask_cnt, mask_idx, full_cnt, full_idx, block_size=(128, 128))
    actual = _flash_attn_fwd(
        q, k, v,
        page_table=page_table,
        seqused_k=torch.full((batch_size,), seqlen_k, device=device, dtype=torch.int32),
        block_sparse_tensors=sparse,
        tile_mn=(128, 128),
        num_splits=1,
        pack_gqa=True,
    )[0]
    positions = torch.cat([
        torch.arange(b * page_size, min((b + 1) * page_size, seqlen_k), device=device)
        for b in sorted(full + masked)
    ])
    k_ref = k[page_table.long()].reshape(batch_size, -1, kvheads, dim)[:, positions]
    v_ref = v[page_table.long()].reshape(batch_size, -1, kvheads, dim)[:, positions]
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(1, 2),
        k_ref.float().repeat_interleave(qheads_per_kvhead, dim=2).transpose(1, 2),
        v_ref.float().repeat_interleave(qheads_per_kvhead, dim=2).transpose(1, 2),
    ).transpose(1, 2)
    torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
