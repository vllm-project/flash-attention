import pytest
import torch

from flash_attn_interface import flash_attn_varlen_func


@pytest.mark.skipif(
    torch.cuda.get_device_capability("cuda")[0] < 9,
    reason="FP8 FlashAttention-3 requires SM90 or later",
)
@pytest.mark.parametrize("seqlen_k", [96, 97, 128, 192, 256])
def test_fp8_decode_prefill_final_row_matches(seqlen_k):
    """The final causal row must not depend on decode vs. prefill tiling."""
    torch.manual_seed(42)
    device = "cuda"
    nheads_q, nheads_kv, headdim = 12, 2, 128

    q = torch.randn(seqlen_k, nheads_q, headdim, device=device).to(
        torch.float8_e4m3fn
    )
    k = torch.randn(seqlen_k, nheads_kv, headdim, device=device).to(
        torch.float8_e4m3fn
    )
    v = torch.randn(seqlen_k, nheads_kv, headdim, device=device).to(
        torch.float8_e4m3fn
    )
    descale = torch.ones(1, nheads_kv, device=device)

    def run(query):
        seqlen_q = query.shape[0]
        out, _ = flash_attn_varlen_func(
            query,
            k,
            v,
            cu_seqlens_q=torch.tensor([0, seqlen_q], device=device, dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, seqlen_k], device=device, dtype=torch.int32),
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            softmax_scale=headdim**-0.5,
            causal=True,
            q_descale=descale,
            k_descale=descale,
            v_descale=descale,
        )
        return out

    prefill = run(q)[-1]
    decode = run(q[-1:])[0]
    assert torch.equal(prefill, decode)
