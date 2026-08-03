import pytest
import torch

import flash_attn.cute.interface as interface
from flash_attn.cute.flash_fwd_sm90 import _use_paged_kv_overlap_sm90
from flash_attn.cute.interface import _flash_attn_fwd, _tile_size_fwd_sm90
from flash_attn.cute.testing import attention_ref


IS_SM90 = (
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
)


def test_sm90_d96_paged_non_tma_tile_policy():
    assert _tile_size_fwd_sm90(
        96,
        96,
        False,
        False,
        sparse_block_size_q=None,
        paged_kv_non_tma=True,
    ) == interface.FwdConfig(192, 128, False, True)


def test_sm90_paged_overlap_keeps_tile_n80():
    assert _use_paged_kv_overlap_sm90(True, True, 80)
    assert not _use_paged_kv_overlap_sm90(True, True, 64)


@pytest.mark.skipif(not IS_SM90, reason="SM90 dynamic scheduler regression")
def test_dynamic_scheduler_uses_internal_state(monkeypatch):
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen_q, seqlen_k = 2, 1, 256
    num_heads, head_dim = 4, 64
    q_padded = torch.randn(
        batch_size,
        seqlen_q,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    k_padded = torch.randn(
        batch_size,
        seqlen_k,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v_padded = torch.randn_like(k_padded)
    q, k, v = [tensor.flatten(0, 1) for tensor in (q_padded, k_padded, v_padded)]
    cu_seqlens_q = torch.arange(
        batch_size + 1, device=device, dtype=torch.int32
    )
    cu_seqlens_k = torch.arange(
        0,
        (batch_size + 1) * seqlen_k,
        seqlen_k,
        device=device,
        dtype=torch.int32,
    )
    selected = []
    original_init = interface.FlashAttentionForwardSm90.__init__

    def record_init(self, *args, **kwargs):
        selected.append(kwargs["use_dynamic_varlen"])
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(interface.FlashAttentionForwardSm90, "__init__", record_init)
    _flash_attn_fwd.compile_cache.clear()
    try:
        out, *_ = _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            num_splits=1,
            causal=True,
        )
        torch.cuda.synchronize()
        assert selected == [True]
        out_ref, _ = attention_ref(q_padded, k_padded, v_padded, causal=True)
        torch.testing.assert_close(
            out.view_as(q_padded), out_ref, atol=1e-2, rtol=1e-2
        )
    finally:
        _flash_attn_fwd.compile_cache.clear()


@pytest.mark.parametrize("use_mla", [False, True], ids=("mha", "mla-dv256"))
@pytest.mark.skipif(not IS_SM90, reason="SM90 dynamic SplitKV output")
def test_dynamic_split_mixed_counts_write_correct_output(use_mla):
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen_q, seqlen_k = 4, 1, 512
    num_heads, num_heads_kv = 4, 1
    head_dim, head_dim_v = (64, 256) if use_mla else (256, 256)
    q_padded = torch.randn(
        batch_size,
        seqlen_q,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        num_heads_kv,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn(
        batch_size,
        seqlen_k,
        num_heads_kv,
        head_dim_v,
        device=device,
        dtype=dtype,
    )
    qv_padded = (
        torch.randn(
            batch_size,
            seqlen_q,
            num_heads,
            head_dim_v,
            device=device,
            dtype=dtype,
        )
        if use_mla
        else None
    )
    cu_seqlens_q = torch.arange(
        batch_size + 1, device=device, dtype=torch.int32
    )
    dynamic_splits = torch.tensor(
        [1, 3, 1, 7], device=device, dtype=torch.int32
    )

    out, lse, *_ = _flash_attn_fwd(
        q_padded.flatten(0, 1),
        k,
        v,
        qv=qv_padded.flatten(0, 1) if qv_padded is not None else None,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        num_splits=7,
        num_splits_dynamic_ptr=dynamic_splits,
        return_lse=True,
    )
    out_ref, _, lse_ref = attention_ref(
        q_padded, k, v, qv=qv_padded, return_lse=True
    )
    torch.testing.assert_close(
        out.view_as(out_ref).float(),
        out_ref.float(),
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        lse,
        lse_ref.permute(1, 0, 2).flatten(1).float(),
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.skipif(not IS_SM90, reason="SM90 dynamic SplitKV CUDA graph")
def test_dynamic_split_scheduler_resets_counter_on_cuda_graph_replay(monkeypatch):
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen_q, seqlen_k = 4, 1, 512
    num_heads, num_heads_kv, head_dim = 4, 1, 256
    num_splits = 7

    q = torch.randn(
        batch_size * seqlen_q,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        num_heads_kv,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn_like(k)
    cu_seqlens_q = torch.arange(
        batch_size + 1, device=device, dtype=torch.int32
    )
    dynamic_splits = torch.tensor(
        [1, 3, 5, 7], device=device, dtype=torch.int32
    )
    scheduler_counter = torch.zeros(1, device=device, dtype=torch.int32)
    torch_zeros = torch.zeros

    def use_external_scheduler_counter(shape, *args, **kwargs):
        if (
            shape == (1,)
            and kwargs.get("dtype") == torch.int32
            and kwargs.get("device") == scheduler_counter.device
        ):
            return scheduler_counter
        return torch_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(torch, "zeros", use_external_scheduler_counter)

    def run():
        return _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            num_splits=num_splits,
            num_splits_dynamic_ptr=dynamic_splits,
        )[0]

    def reference():
        q_heads = q.view(
            batch_size, seqlen_q, num_heads, head_dim
        ).transpose(1, 2)
        k_heads = k.repeat_interleave(
            num_heads // num_heads_kv, dim=2
        ).transpose(1, 2)
        v_heads = v.repeat_interleave(
            num_heads // num_heads_kv, dim=2
        ).transpose(1, 2)
        scores = (
            q_heads.float() @ k_heads.float().transpose(-2, -1)
        ) / head_dim**0.5
        return (
            (scores.softmax(dim=-1) @ v_heads.float())
            .transpose(1, 2)
            .flatten(0, 1)
        )

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        run()
    torch.cuda.current_stream().wait_stream(side_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = run()
    assert scheduler_counter.item() == 0

    for replay in range(3):
        q.copy_(torch.randn_like(q))
        graph.replay()
        torch.cuda.synchronize()
        assert scheduler_counter.item() == 0
        torch.testing.assert_close(
            graph_out.float(),
            reference(),
            atol=2e-2,
            rtol=2e-2,
            msg=lambda msg: f"CUDA graph replay {replay}: {msg}",
        )
