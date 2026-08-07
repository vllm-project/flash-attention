"""Focused tests for the bounded SM120 BF16 D256 paged-decode path."""

import math
import gc
from unittest.mock import MagicMock
import weakref

import pytest
import torch
import torch.nn.functional as F

from flash_attn.cute import flash_attn_varlen_func
from flash_attn.cute import interface as cute_interface
from flash_attn.cute.sm120_paged_decode_cache import Sm120PagedDecodeRuntimeCache
from flash_attn.cute.sm120_paged_decode import (
    Sm120PagedDecodeD256Metadata,
    build_sm120_paged_decode_d256_compile_plan,
    build_sm120_paged_decode_d256_warmup_plan,
    select_sm120_paged_decode_d256_num_splits,
    sm120_paged_decode_d256_available,
    sm120_paged_decode_d256_enabled,
)


def _metadata(**overrides):
    values = dict(
        capability=120,
        dtype=torch.bfloat16,
        batch_size=1,
        query_tokens=1,
        query_heads=24,
        kv_heads=4,
        head_dim=256,
        page_size=784,
        kv_tokens=4096,
        paged_kv=True,
        has_cu_seqlens_q=True,
        has_seqused_k=True,
    )
    values.update(overrides)
    return Sm120PagedDecodeD256Metadata(**values)


@pytest.mark.parametrize(
    ("kv_tokens", "expected"),
    [(256, 1), (257, 16), (512, 16), (513, 32), (1024, 32), (1025, 48)],
)
def test_split_selector_boundaries(kv_tokens, expected):
    assert select_sm120_paged_decode_d256_num_splits(kv_tokens) == expected


@pytest.mark.parametrize(
    "override",
    [
        {"dtype": torch.float16},
        {"head_dim": 128},
        {"query_heads": 32},
        {"kv_heads": 8},
        {"page_size": 128},
        {"has_unsupported_feature": True},
    ],
)
def test_eligibility_fails_closed(override):
    assert not sm120_paged_decode_d256_available(_metadata(**override))


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no", "invalid", "2"])
def test_opt_in_invalid_or_disabled_values_fail_closed(monkeypatch, value):
    monkeypatch.setenv("FLASH_ATTENTION_SM120_DECODE_KERNEL", value)
    assert not sm120_paged_decode_d256_enabled()


def test_public_launcher_does_not_call_varlen_for_unsupported_metadata(monkeypatch):
    monkeypatch.setenv("FLASH_ATTENTION_SM120_DECODE_KERNEL", "1")
    varlen = MagicMock()
    monkeypatch.setattr(cute_interface, "flash_attn_varlen_func", varlen)

    launched = cute_interface.try_launch_sm120_paged_decode_d256(
        _metadata(dtype=torch.float16),
        q=object(),
        k=object(),
        v=object(),
    )

    assert not launched
    varlen.assert_not_called()


def _hot_plan_tensors():
    return tuple(torch.empty((2, 2)) for _ in range(7))


def test_hot_plan_requires_object_storage_stream_launcher_and_configuration_match():
    cache = Sm120PagedDecodeRuntimeCache()
    tensors = _hot_plan_tensors()
    stream = ("cuda", 0, 11)
    launcher = (101, 102)
    key = ("decode", "combine")
    cache.record_hot_plan(key, tensors, stream, launcher, 16, 16)

    assert cache.get_hot_plan(key, tensors, stream, launcher, 16, 16) is not None
    tensors[0].set_(torch.empty_like(tensors[0]))
    assert cache.get_hot_plan(key, tensors, stream, launcher, 16, 16) is None

    tensors = _hot_plan_tensors()
    cache.record_hot_plan(key, tensors, stream, launcher, 16, 16)
    replaced_output = (*tensors[:3], torch.empty_like(tensors[3]), *tensors[4:])
    assert cache.get_hot_plan(key, replaced_output, stream, launcher, 16, 16) is None
    assert cache.get_hot_plan(key, tensors, ("cuda", 0, 12), launcher, 16, 16) is None
    assert cache.get_hot_plan(key, tensors, stream, (103, 102), 16, 16) is None
    assert cache.get_hot_plan(key, tensors, stream, launcher, 784, 16) is None
    assert cache.get_hot_plan(key, tensors, stream, launcher, 16, 32) is None


def test_hot_plan_rejects_shape_stride_and_dtype_mutation():
    cache = Sm120PagedDecodeRuntimeCache()
    stream = ("cuda", 0, 11)
    launcher = (101, 102)
    key = ("decode", "combine")

    tensors = _hot_plan_tensors()
    cache.record_hot_plan(key, tensors, stream, launcher, 16, 16)
    tensors[0].resize_(4, 1)
    assert cache.get_hot_plan(key, tensors, stream, launcher, 16, 16) is None

    tensors = _hot_plan_tensors()
    cache.record_hot_plan(key, tensors, stream, launcher, 16, 16)
    tensors[0].as_strided_((2, 2), (1, 2))
    assert cache.get_hot_plan(key, tensors, stream, launcher, 16, 16) is None

    tensors = _hot_plan_tensors()
    cache.record_hot_plan(key, tensors, stream, launcher, 16, 16)
    tensors[0].data = tensors[0].to(torch.bfloat16)
    assert cache.get_hot_plan(key, tensors, stream, launcher, 16, 16) is None


def test_hot_plan_uses_weak_input_references_and_allows_page_content_updates():
    cache = Sm120PagedDecodeRuntimeCache()
    tensors = list(_hot_plan_tensors())
    stream = ("cuda", 0, 11)
    launcher = (101, 102)
    cache.record_hot_plan(
        ("decode", "combine"), tuple(tensors), stream, launcher, 16, 16
    )
    tensors[5].fill_(1)
    tensors[6].fill_(2)
    assert (
        cache.get_hot_plan(
            ("decode", "combine"), tuple(tensors), stream, launcher, 16, 16
        )
        is not None
    )

    input_ref = weakref.ref(tensors[0])
    tensors[0] = torch.empty_like(tensors[0])
    gc.collect()
    assert input_ref() is None


def test_workspace_cache_is_bounded_per_stream_and_capture_falls_back():
    cache = Sm120PagedDecodeRuntimeCache()
    stream = ("cuda", 0, 11)
    created = []

    def make_workspace():
        workspace = object()
        created.append(workspace)
        return workspace

    first = cache.get_workspace((0,), stream, make_workspace, allow_cache=True)
    for index in range(1, 9):
        cache.get_workspace((index,), stream, make_workspace, allow_cache=True)
    assert cache.workspace_count(stream) == 8
    assert (
        cache.get_workspace((0,), stream, make_workspace, allow_cache=True) is not first
    )

    capture_stream = ("cuda", 0, 12)
    first_capture = cache.get_workspace(
        (0,), capture_stream, make_workspace, allow_cache=False
    )
    second_capture = cache.get_workspace(
        (0,), capture_stream, make_workspace, allow_cache=False
    )
    assert first_capture is not second_capture
    assert cache.workspace_count(capture_stream) == 0

    cache.clear()
    assert cache.workspace_count(stream) == 0
    assert cache.hot_plan_count(stream) == 0


def test_compile_plan_contains_exact_forward_and_combine_witness():
    plan = build_sm120_paged_decode_d256_compile_plan(_metadata(kv_tokens=513))
    assert [(unit.kernel, unit.num_splits) for unit in plan] == [
        ("sm120_paged_decode_d256_forward", 32),
        ("sm120_splitkv_combine", 32),
    ]


def test_warmup_plan_is_bounded_and_deduplicated_by_selector_witness():
    plan = build_sm120_paged_decode_d256_warmup_plan(_metadata(), 16_384)
    forward = [unit for unit in plan if unit.kernel.endswith("forward")]
    combine = [unit for unit in plan if unit.kernel.endswith("combine")]
    assert [unit.num_splits for unit in forward] == [16, 32, 48]
    assert [unit.num_splits for unit in combine] == [16, 32, 48]
    assert [unit.kv_tokens_witness for unit in forward] == [257, 513, 1025]


def _require_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120-only specialization")


@pytest.mark.parametrize("page_size,kv_tokens", [(16, 513), (784, 1025)])
def test_paged_decode_matches_sdpa_and_cuda_graph_replay(
    monkeypatch, page_size, kv_tokens
):
    _require_sm120()
    monkeypatch.setenv("FLASH_ATTENTION_SM120_DECODE_KERNEL", "1")
    torch.manual_seed(17)
    device = torch.device("cuda")
    pages = math.ceil(kv_tokens / page_size)
    q = torch.randn(1, 24, 256, dtype=torch.bfloat16, device=device)
    k = torch.randn(pages, page_size, 4, 256, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    page_table = torch.arange(pages, dtype=torch.int32, device=device)[None]
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    seqused_k = torch.tensor([kv_tokens], dtype=torch.int32, device=device)
    out = torch.empty_like(q)
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_q,
        seqused_k=seqused_k,
        max_seqlen_q=1,
        max_seqlen_k=kv_tokens,
        page_table=page_table,
        causal=True,
        num_splits=0,
    )
    flash_attn_varlen_func(**kwargs)
    logical_k = k[page_table[0]].flatten(0, 1)[:kv_tokens]
    logical_v = v[page_table[0]].flatten(0, 1)[:kv_tokens]
    reference = (
        F.scaled_dot_product_attention(
            q.transpose(0, 1)[None].float(),
            logical_k.transpose(0, 1)[None].repeat_interleave(6, dim=1).float(),
            logical_v.transpose(0, 1)[None].repeat_interleave(6, dim=1).float(),
            is_causal=False,
        )
        .squeeze(0)
        .transpose(0, 1)
        .to(torch.bfloat16)
    )
    torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flash_attn_varlen_func(**kwargs)
    page_table.copy_(torch.flip(page_table, dims=(1,)))
    seqused_k.fill_(max(257, kv_tokens - 1))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
