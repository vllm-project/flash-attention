"""hd512 GQA on SM100, served by the MLA kernel with a separate PV operand.

Shapes follow DiffusionGemma's global layers: 16 query heads, 2 KV heads,
d = dv = 512, a 256-token query block over a long paged prefix.
"""
import pytest
import torch

from flash_attn.cute.interface import _flash_attn_fwd

DEV = "cuda"
DT = torch.bfloat16
HQ, HKV, D = 16, 2, 512
FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (10, 11)


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="SM100-family only")


def _paged_inputs(batch, seqlen_q, seqlen_k, page, hq=HQ, hkv=HKV):
    pages = (seqlen_k + page - 1) // page
    num_blocks = batch * pages + 3
    kc = torch.randn(num_blocks, page, hkv, D, device=DEV, dtype=DT)
    vc = torch.randn(num_blocks, page, hkv, D, device=DEV, dtype=DT)
    block_table = torch.randperm(num_blocks, device=DEV)[: batch * pages]
    block_table = block_table.view(batch, pages).to(torch.int32)
    q = torch.randn(batch * seqlen_q, hq, D, device=DEV, dtype=DT)
    cu_q = torch.arange(batch + 1, device=DEV, dtype=torch.int32) * seqlen_q
    seqused_k = torch.full((batch,), seqlen_k, device=DEV, dtype=torch.int32)
    return q, kc, vc, block_table, cu_q, seqused_k


def _reference(q, kc, vc, block_table, seqlen_q, seqlen_k, scale, causal):
    outs, lses = [], []
    for b, is_causal in enumerate(causal):
        k = kc[block_table[b].long()].flatten(0, 1)[:seqlen_k].float()
        v = vc[block_table[b].long()].flatten(0, 1)[:seqlen_k].float()
        k, v = (t.repeat_interleave(q.shape[1] // kc.shape[2], 1) for t in (k, v))
        s = torch.einsum("qhd,khd->hqk", q[b * seqlen_q : (b + 1) * seqlen_q].float(), k)
        s = s * scale
        if is_causal:
            qi = torch.arange(seqlen_q, device=DEV)[:, None] + seqlen_k - seqlen_q
            s = s.masked_fill(torch.arange(seqlen_k, device=DEV) > qi, float("-inf"))
        outs.append(torch.einsum("hqk,khd->qhd", s.softmax(-1), v))
        lses.append(s.logsumexp(-1))
    return torch.cat(outs), torch.cat(lses, dim=1)


def _rel_err(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("page", [16, 128])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(256, 256), (256, 1280), (256, 4352), (1, 1024)])
@pytest.mark.parametrize("causal", [False, True])
def test_hd512_gqa_paged(page, seqlen_q, seqlen_k, causal):
    torch.manual_seed(0)
    q, kc, vc, block_table, cu_q, seqused_k = _paged_inputs(2, seqlen_q, seqlen_k, page)
    scale = D**-0.5
    out = _flash_attn_fwd(
        q, kc, vc,
        cu_seqlens_q=cu_q,
        seqused_k=seqused_k,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        page_table=block_table,
        softmax_scale=scale,
        causal=causal,
    )[0]
    ref, _ = _reference(q, kc, vc, block_table, seqlen_q, seqlen_k, scale, [causal] * 2)
    assert _rel_err(out, ref) < 1e-2


# Two KV-head counts sharing a group size must not reuse each other's compiled kernel.
@pytest.mark.parametrize("hq,hkv", [(16, 2), (32, 4)])
def test_hd512_gqa_lse(hq, hkv):
    torch.manual_seed(0)
    seqlen_q, seqlen_k = 256, 1280
    q, kc, vc, block_table, cu_q, seqused_k = _paged_inputs(2, seqlen_q, seqlen_k, 16, hq, hkv)
    scale = D**-0.5
    kwargs = dict(
        cu_seqlens_q=cu_q,
        seqused_k=seqused_k,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        page_table=block_table,
        softmax_scale=scale,
        return_lse=True,
    )
    out, lse = _flash_attn_fwd(q, kc, vc, **kwargs)[:2]
    ref, ref_lse = _reference(q, kc, vc, block_table, seqlen_q, seqlen_k, scale, [False] * 2)
    assert _rel_err(out, ref) < 1e-2
    assert lse.shape == (hq, q.shape[0])
    torch.testing.assert_close(lse, ref_lse, atol=1e-2, rtol=1e-3)

    lse_buf = torch.empty(q.shape[0], hq, device=DEV, dtype=torch.float32).mT
    _flash_attn_fwd(q, kc, vc, lse=lse_buf, **kwargs)
    torch.testing.assert_close(lse_buf, ref_lse, atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("fp8_dtype", FP8)
def test_hd512_gqa_rejects_fp8(fp8_dtype):
    q, kc, vc, block_table, cu_q, seqused_k = _paged_inputs(1, 16, 64, 16)
    with pytest.raises(NotImplementedError, match="FP8"):
        _flash_attn_fwd(
            q, kc.to(fp8_dtype), vc.to(fp8_dtype),
            cu_seqlens_q=cu_q,
            seqused_k=seqused_k,
            max_seqlen_q=16,
            max_seqlen_k=64,
            page_table=block_table,
        )
