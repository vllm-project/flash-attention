import pytest
import torch

import flash_attn.cute.interface as interface
import flash_attn.cute.tile_scheduler as tile_scheduler
from flash_attn.cute.flash_fwd_sm90 import _use_paged_kv_overlap_sm90
from flash_attn.cute.interface import (
    _combine_max_seqlen_q_sm90,
    _flash_attn_fwd,
    _num_splits_sm90,
    _use_dynamic_varlen_scheduler_sm90,
    num_splits_heuristic,
)
from flash_attn.cute.tile_scheduler import (
    DynamicPersistentVarlenTileScheduler,
    SingleTileVarlenScheduler,
    StaticPersistentVarlenTileScheduler,
    _sm90_gqa_l2_divisor,
)
from flash_attn.cute.testing import maybe_fake_tensor_mode


@pytest.mark.parametrize(
    "total_mblocks,num_n_blocks,batch_size,qhead_per_kvhead,paged_kv,"
    "head_dim,head_dim_v,expected",
    [
        (1, 3, 2, 4, False, 128, 128, 1),
        (26, 4, 2, 4, False, 128, 128, 4),
        (27, 4, 2, 4, False, 128, 128, 1),
        (27, 5, 2, 4, False, 128, 128, 4),
        (2, 63, 1, 4, False, 128, 128, 63),
        (2, 64, 1, 4, False, 128, 128, 16),
        (2, 191, 1, 4, False, 128, 128, 47),
        (2, 192, 1, 4, False, 128, 128, 24),
        (66, 5, 1, 8, True, 64, 64, 2),
        (67, 5, 1, 8, True, 64, 64, 2),
        (131, 5, 1, 8, True, 64, 64, 2),
        (132, 5, 1, 8, True, 64, 64, 1),
        (67, 5, 1, 8, False, 64, 64, 1),
        (67, 5, 1, 8, True, 128, 128, 1),
        (67, 5, 2, 8, True, 64, 64, 1),
        (133, 5, 2, 8, False, 128, 128, 1),
    ],
)
def test_num_splits_sm90_boundaries(
    total_mblocks,
    num_n_blocks,
    batch_size,
    qhead_per_kvhead,
    paged_kv,
    head_dim,
    head_dim_v,
    expected,
):
    assert (
        _num_splits_sm90(
            total_mblocks,
            132,
            num_n_blocks,
            128,
            batch_size,
            qhead_per_kvhead,
            paged_kv,
            head_dim,
            head_dim_v,
        )
        == expected
    )


def test_generic_num_splits_never_returns_zero():
    assert num_splits_heuristic(133, 132, 5, 128) == 1


@pytest.mark.parametrize(
    "scheduler,sm_count,expected_grid_x",
    [
        (StaticPersistentVarlenTileScheduler, 114, 228),
        (DynamicPersistentVarlenTileScheduler, 132, 132),
    ],
)
def test_persistent_grid_uses_forward_sm_count(
    monkeypatch, scheduler, sm_count, expected_grid_x
):
    monkeypatch.setattr(
        SingleTileVarlenScheduler,
        "get_grid_shape",
        staticmethod(lambda params: (1000, 1, 1)),
    )

    class WrongDeviceHardwareInfo:
        def get_device_multiprocessor_count(self):
            raise AssertionError("persistent scheduler queried another device")

    monkeypatch.setattr(
        tile_scheduler, "HardwareInfo", WrongDeviceHardwareInfo
    )
    grid = scheduler.get_grid_shape(object(), sm_count=sm_count)
    assert grid[0] == expected_grid_x


@pytest.mark.parametrize(
    "arch,is_packed_varlen,expected",
    [
        (80, True, None),
        (90, True, 17),
        (90, False, None),
        (100, True, None),
        (110, True, None),
        (120, True, None),
    ],
)
def test_combine_max_seqlen_q_is_sm90_packed_only(
    arch, is_packed_varlen, expected
):
    assert (
        _combine_max_seqlen_q_sm90(arch, 17, is_packed_varlen)
        == expected
    )


@maybe_fake_tensor_mode()
def test_combine_max_seqlen_q_compile_specialization(monkeypatch):
    compile_calls = []

    def record_compile(*args):
        compile_calls.append(args)
        return object()

    monkeypatch.setattr(interface, "_compile_fwd_combine", record_compile)
    interface._flash_attn_fwd_combine.compile_cache.clear()

    out_partial = torch.empty(
        (2, 8, 4, 64), device="cuda", dtype=torch.bfloat16
    )
    lse_partial = torch.empty(
        (2, 8, 4), device="cuda", dtype=torch.float32
    )
    out = torch.empty((8, 4, 64), device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.empty((3,), device="cuda", dtype=torch.int32)

    try:
        interface._flash_attn_fwd_combine(
            out_partial, lse_partial, out, cu_seqlens=cu_seqlens
        )
        assert compile_calls[-1][-1] is False

        interface._flash_attn_fwd_combine(
            out_partial,
            lse_partial,
            out,
            cu_seqlens=cu_seqlens,
            max_seqlen_q=4,
        )
        assert compile_calls[-1][-1] is True
    finally:
        interface._flash_attn_fwd_combine.compile_cache.clear()


def test_combine_compile_argument_binding(monkeypatch):
    compile_calls = []

    def record_compile(*args, **kwargs):
        compile_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(interface.cute, "compile", record_compile)
    common_args = (
        interface.torch2cute_dtype_map[torch.bfloat16],
        interface.Float32,
        64,
        16,
        64,
        4,
        True,
        False,
        False,
        False,
        None,
    )

    interface._compile_fwd_combine(*common_args, False)
    args, _ = compile_calls[-1]
    assert args[-2] is None

    interface._compile_fwd_combine(*common_args, True)
    args, _ = compile_calls[-1]
    assert isinstance(args[-2], interface.Int32)


@pytest.mark.parametrize(
    "ratio,expected",
    [(1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (8, 8), (9, 16), (16, 16), (32, 16)],
)
def test_sm90_gqa_l2_divisor(ratio, expected):
    assert _sm90_gqa_l2_divisor(ratio) == expected


@pytest.mark.parametrize(
    "requested,paged_non_tma,tile_n,expected",
    [
        (True, True, 128, True),
        (True, True, 64, False),
        (True, False, 128, False),
        (False, True, 128, False),
    ],
)
def test_sm90_paged_overlap_uses_measured_tile(
    requested, paged_non_tma, tile_n, expected
):
    assert (
        _use_paged_kv_overlap_sm90(requested, paged_non_tma, tile_n)
        is expected
    )


def _dynamic_selector(**overrides):
    args = {
        "arch": 90,
        "batch_size": 1,
        "num_head": 32,
        "num_head_kv": 4,
        "head_dim": 128,
        "max_seqlen_q": 128,
        "max_seqlen_k": 8192,
        "no_explicit_window": True,
        "local": False,
        "mask_mod": None,
        "aux_tensors": None,
    }
    args.update(overrides)
    return _use_dynamic_varlen_scheduler_sm90(**args)


@pytest.mark.parametrize(
    "overrides,expected",
    [
        (
            {
                "batch_size": 2,
                "head_dim": 64,
                "no_explicit_window": False,
                "local": True,
            },
            True,
        ),
        (
            {
                "num_head_kv": 32,
                "head_dim": 64,
                "no_explicit_window": False,
                "local": True,
            },
            True,
        ),
        ({"max_seqlen_k": 16 * 1024 - 1}, False),
        ({"num_head": 64, "max_seqlen_q": 8192, "max_seqlen_k": 8192}, True),
        ({"num_head": 60, "max_seqlen_q": 8192, "max_seqlen_k": 8192}, False),
        ({"num_head": 64, "max_seqlen_q": 8191, "max_seqlen_k": 8192}, False),
        ({"num_head": 64, "max_seqlen_q": 8192, "max_seqlen_k": 8191}, False),
        ({"max_seqlen_k": 16 * 1024, "mask_mod": object()}, False),
        ({"max_seqlen_k": 16 * 1024, "aux_tensors": []}, False),
        ({"max_seqlen_k": 16 * 1024, "head_dim": 64}, False),
        ({}, False),
    ],
)
def test_dynamic_varlen_selector(overrides, expected):
    assert _dynamic_selector(**overrides) is expected


@pytest.mark.parametrize(
    "no_explicit_window,local,expected",
    [
        (False, False, False),  # Explicit right=0 canonicalizes to non-local.
        (True, False, True),  # Ordinary causal attention has no raw window.
        (False, True, False),  # Explicit local window.
    ],
)
def test_dynamic_gqa_window_boundaries(
    no_explicit_window, local, expected
):
    assert _dynamic_selector(
        max_seqlen_k=16 * 1024,
        no_explicit_window=no_explicit_window,
        local=local,
    ) is expected


@pytest.mark.parametrize("arch", [80, 100, 110, 120])
def test_dynamic_varlen_selector_rejects_non_sm90(arch):
    assert not _dynamic_selector(
        arch=arch,
        batch_size=2,
        num_head=32,
        num_head_kv=32,
    )


@pytest.mark.parametrize("arch", [80, 100, 110, 120])
@maybe_fake_tensor_mode()
def test_dynamic_workspace_rejects_non_sm90(arch):
    dtype = torch.bfloat16
    q = torch.empty((1, 1, 64), device="cuda", dtype=dtype)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    cu_seqlens = torch.empty(2, device="cuda", dtype=torch.int32)
    workspace = torch.zeros(2, device="cuda", dtype=torch.int32)

    with pytest.raises(
        AssertionError,
        match="dynamic varlen scheduler is only supported on SM90",
    ):
        _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=1,
            max_seqlen_k=1,
            dynamic_scheduler_workspace=workspace,
            _arch=arch,
            compile_only=True,
        )


IS_SM90 = (
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
)


@pytest.mark.skipif(not IS_SM90, reason="SM90 dynamic scheduler regression")
def test_dynamic_scheduler_workspace_resets_eager_and_graph():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen_q, seqlen_k = 2, 1, 256
    num_heads, head_dim = 4, 64
    q = torch.randn(
        batch_size * seqlen_q,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        batch_size * seqlen_k,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn_like(k)
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
    workspace = torch.zeros(2, device=device, dtype=torch.int32)
    out = torch.empty_like(q)

    def run():
        _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            num_splits=1,
            out=out,
            dynamic_scheduler_workspace=workspace,
        )

    run()
    torch.cuda.synchronize()
    assert torch.count_nonzero(workspace).item() == 0

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(workspace).item() == 0


@pytest.mark.skipif(not IS_SM90, reason="SM90 dynamic scheduler regression")
@pytest.mark.parametrize(
    "batch_size,num_heads,num_heads_kv,num_splits,seqlen_k,head_dim,"
    "causal,window_size_right",
    [
        (1, 8, 1, 1, 640, 64, False, None),  # Short GQA is not selected.
        (2, 4, 4, 2, 640, 64, False, None),  # Split-K disables dynamic.
        (1, 8, 1, 1, 16 * 1024, 128, False, 0),  # Explicit right=0.
    ],
)
def test_ineffective_workspace_reuses_compile_key(
    batch_size,
    num_heads,
    num_heads_kv,
    num_splits,
    seqlen_k,
    head_dim,
    causal,
    window_size_right,
):
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    seqlen_q = 1
    q = torch.randn(
        batch_size * seqlen_q,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        batch_size * seqlen_k,
        num_heads_kv,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn_like(k)
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
    workspace = torch.zeros(2, device=device, dtype=torch.int32)

    def run(dynamic_scheduler_workspace=None):
        _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            num_splits=num_splits,
            causal=causal,
            window_size_right=window_size_right,
            dynamic_scheduler_workspace=dynamic_scheduler_workspace,
        )

    _flash_attn_fwd.compile_cache.clear()
    run()
    keys_without_workspace = set(_flash_attn_fwd.compile_cache.cache)
    run(workspace)
    torch.cuda.synchronize()
    assert set(_flash_attn_fwd.compile_cache.cache) == keys_without_workspace
    assert torch.count_nonzero(workspace).item() == 0
