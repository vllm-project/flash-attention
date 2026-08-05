# Copyright (c) 2025, FlashAttention contributors.
"""Binding SM90 E4M3 attention tests with static per-tensor descales."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest
import torch


HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 16
FP8_SEEDS = (17, 2027, 65537)
SEED_LAYERS = {17: 0, 2027: 29, 65537: 31}


@dataclass(frozen=True)
class AttentionCase:
    name: str
    seed: int
    q_lens: tuple[int, ...]
    k_lens: tuple[int, ...]
    block_rows: tuple[tuple[int, ...], ...]
    num_pages: int
    tile_m: int
    num_splits: int
    input_causal: bool

    @property
    def canonical_causal(self) -> bool:
        return self.input_causal and max(self.q_lens) != 1


def _decode_rows() -> tuple[tuple[int, ...], ...]:
    return (
        (3,),
        (17, 11),
        (43, 29, 37, 23, 41, 31, 47, 19),
        tuple(100 + ((17 * index + 7) % 65) for index in range(65)),
    )


PREFILL_ROWS = (
    (19, 3),
    (17, 5, 31, 2, 23, 11, 29, 7, 13, 37, 41, 43, 47, 53, 59, 61, 67),
)
DECODE_ROWS = _decode_rows()


def _arithmetic_cases() -> tuple[AttentionCase, ...]:
    cases = [
        AttentionCase(
            f"paged-prefill-seed-{seed}",
            seed,
            (17, 257),
            (17, 257),
            PREFILL_ROWS,
            96,
            128,
            1,
            True,
        )
        for seed in FP8_SEEDS
    ]
    cases.extend(
        AttentionCase(
            f"paged-decode-seed-{seed}",
            seed,
            (1, 1, 1, 1),
            (16, 17, 128, 1025),
            DECODE_ROWS,
            192,
            64,
            0,
            True,
        )
        for seed in FP8_SEEDS
    )
    cases.extend(
        (
            AttentionCase(
                "dynamic-decode-batch-16",
                17,
                (1,) * 16,
                (1,) * 16,
                tuple((2 * index + 1,) for index in range(16)),
                64,
                64,
                1,
                True,
            ),
            AttentionCase(
                "persistent-short-prefill",
                17,
                (16,),
                (16,),
                ((7,),),
                32,
                64,
                1,
                True,
            ),
            AttentionCase(
                "persistent-long-prefill",
                17,
                (17,),
                (17,),
                ((7, 19),),
                32,
                128,
                1,
                True,
            ),
            AttentionCase(
                "dynamic-mixed-prefill",
                17,
                (2, 16),
                (2, 16),
                ((7,), (19,)),
                32,
                64,
                1,
                True,
            ),
            AttentionCase(
                "static-split-decode",
                17,
                (1, 1, 1, 1),
                (16, 17, 128, 1025),
                DECODE_ROWS,
                192,
                64,
                32,
                True,
            ),
            AttentionCase(
                "static-split-short-prefill",
                17,
                (1, 15),
                (16, 15),
                ((3,), (17,)),
                64,
                64,
                32,
                True,
            ),
            AttentionCase(
                "static-split-long-prefill",
                17,
                (1, 23),
                (16, 23),
                ((3,), (17, 11)),
                64,
                128,
                32,
                True,
            ),
            AttentionCase(
                "persistent-single-token-decode",
                17,
                (1,),
                (16,),
                ((7,),),
                32,
                64,
                1,
                True,
            ),
        )
    )
    assert len(cases) == 14
    assert all(case.input_causal for case in cases)
    return tuple(cases)


ARITHMETIC_CASES = _arithmetic_cases()
PERSISTENT_DECODE_CASE = ARITHMETIC_CASES[13]


def _scale_pair(manifest: dict[str, Any], seed: int) -> tuple[float, float]:
    layers = manifest.get("layers")
    assert isinstance(layers, list) and len(layers) == 32
    layer_index = SEED_LAYERS[seed]
    layer = layers[layer_index]
    assert layer["layer"] == layer_index
    k_scale = float(layer["k_scale"])
    v_scale = float(layer["v_scale"])
    assert math.isfinite(k_scale) and k_scale > 0.0 and k_scale != 1.0
    assert math.isfinite(v_scale) and v_scale > 0.0 and v_scale != 1.0
    return k_scale, v_scale


def _cpu_values(
    generator: torch.Generator,
    shape: tuple[int, ...],
    scale: float,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    values = torch.randn(shape, dtype=torch.float32, generator=generator)
    return values.mul_(64.0 * scale).clamp_(-192.0 * scale, 192.0 * scale).to(dtype)


def _non_fp8_values(
    generator: torch.Generator,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    return (
        torch.randn(shape, dtype=torch.float32, generator=generator)
        .mul_(2)
        .clamp_(-6, 6)
        .to(dtype)
    )


def _quantize_expected(values: torch.Tensor, scale: float) -> torch.Tensor:
    limit = torch.finfo(torch.float8_e4m3fn).max
    return values.float().div(scale).clamp(-limit, limit).to(torch.float8_e4m3fn)


def _quantize_query(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    with set_current_vllm_config(VllmConfig()):
        quantizer = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)
        quantized, returned_scale = quantizer(
            values.reshape(-1, HEAD_DIM).cuda(), scale.reshape(1)
        )
    assert returned_scale.numel() == 1
    assert returned_scale.data_ptr() == scale.data_ptr()
    return quantized.reshape(values.shape)


def _padded_table(case: AttentionCase) -> torch.Tensor:
    width = max(len(row) for row in case.block_rows)
    table = torch.full((len(case.block_rows), width), -1, dtype=torch.int32)
    for batch, row in enumerate(case.block_rows):
        needed = math.ceil(case.k_lens[batch] / PAGE_SIZE)
        assert len(row) == needed
        table[batch, :needed] = torch.tensor(row, dtype=torch.int32)
    return table


def _slots(case: AttentionCase) -> tuple[torch.Tensor, tuple[tuple[int, ...], ...]]:
    flattened: list[int] = []
    per_sequence: list[tuple[int, ...]] = []
    for length, row in zip(case.k_lens, case.block_rows, strict=True):
        current = tuple(
            row[pos // PAGE_SIZE] * PAGE_SIZE + pos % PAGE_SIZE
            for pos in range(length)
        )
        per_sequence.append(current)
        flattened.extend(current)
    return torch.tensor(flattened, dtype=torch.int64), tuple(per_sequence)


def _write_fp8_cache(
    case: AttentionCase,
    key: torch.Tensor,
    value: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[tuple[int, ...], ...],
]:
    from vllm import _custom_ops as ops

    cache = torch.zeros(
        case.num_pages,
        KV_HEADS,
        PAGE_SIZE,
        2 * HEAD_DIM,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    key_cache, value_cache = cache.transpose(1, 2).split(HEAD_DIM, dim=-1)
    slots, per_sequence = _slots(case)
    slots = slots.cuda()

    q1_final: list[int] = []
    ordinary: list[int] = []
    base = 0
    for q_len, k_len in zip(case.q_lens, case.k_lens, strict=True):
        indices = list(range(base, base + k_len))
        if q_len == 1 and k_len > 1:
            ordinary.extend(indices[:-1])
            q1_final.append(indices[-1])
        else:
            ordinary.extend(indices)
        base += k_len

    for indices in (ordinary, q1_final):
        if not indices:
            continue
        index = torch.tensor(indices, dtype=torch.int64, device="cuda")
        ops.reshape_and_cache_flash(
            key.index_select(0, index),
            value.index_select(0, index),
            key_cache,
            value_cache,
            slots.index_select(0, index),
            "fp8",
            k_scale,
            v_scale,
        )
    return key_cache, value_cache, cache, per_sequence


def _expected_cache(
    case: AttentionCase,
    logical: torch.Tensor,
    scale: float,
    per_sequence: tuple[tuple[int, ...], ...],
) -> torch.Tensor:
    expected = torch.zeros(
        case.num_pages * PAGE_SIZE,
        KV_HEADS,
        HEAD_DIM,
        dtype=torch.float8_e4m3fn,
    )
    quantized = _quantize_expected(logical, scale)
    logical_base = 0
    for length, slots in zip(case.k_lens, per_sequence, strict=True):
        expected[list(slots)] = quantized[logical_base : logical_base + length]
        logical_base += length
    return expected.reshape(case.num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM)


def _descale(scale: torch.Tensor, batch: int) -> torch.Tensor:
    view = scale.as_strided((batch, KV_HEADS), (0, 0))
    assert view.dtype == torch.float32 and view.is_cuda
    assert view.stride() == (0, 0)
    assert view.data_ptr() == scale.data_ptr()
    return view


def _lse_rows(lse: torch.Tensor, total_q: int) -> torch.Tensor:
    if lse.shape == (HEADS, total_q):
        return lse.transpose(0, 1).contiguous()
    assert lse.shape == (total_q, HEADS)
    return lse


def _reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    case: AttentionCase,
    per_sequence: tuple[tuple[int, ...], ...],
    q_scale: float,
    k_scale: float,
    v_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = query.float().double().cpu().mul(q_scale)
    k_flat = key_cache.float().double().cpu().reshape(-1, KV_HEADS, HEAD_DIM).mul(k_scale)
    v_flat = value_cache.float().double().cpu().reshape(-1, KV_HEADS, HEAD_DIM).mul(v_scale)
    output = torch.zeros_like(q, dtype=torch.float64)
    lse = torch.full((q.shape[0], HEADS), float("-inf"), dtype=torch.float64)
    q_base = 0
    for q_len, k_len, slots in zip(case.q_lens, case.k_lens, per_sequence, strict=True):
        keys = k_flat[list(slots)]
        values = v_flat[list(slots)]
        for q_row in range(q_len):
            last_key = k_len - q_len + q_row if case.input_causal else k_len - 1
            if last_key < 0:
                continue
            for q_head in range(HEADS):
                kv_head = q_head // (HEADS // KV_HEADS)
                scores = torch.mv(
                    keys[: last_key + 1, kv_head], q[q_base + q_row, q_head]
                ) / math.sqrt(HEAD_DIM)
                lse[q_base + q_row, q_head] = torch.logsumexp(scores, dim=0)
                probs = torch.softmax(scores, dim=0)
                output[q_base + q_row, q_head] = probs @ values[: last_key + 1, kv_head]
        q_base += q_len
    return output, lse


def _metrics(
    output: torch.Tensor,
    lse: torch.Tensor,
    reference_output: torch.Tensor,
    reference_lse: torch.Tensor,
    q_lens: tuple[int, ...],
    bounds: tuple[float, float, float, float, float],
) -> list[dict[str, Any]]:
    maxabs_bound, nrms_bound, lse_bound, seq_nrms_bound, seq_lse_bound = bounds
    got = output.float().double().cpu()
    got_lse = _lse_rows(lse.float().cpu(), output.shape[0]).double()
    records: list[dict[str, Any]] = []
    q_base = 0
    for sequence, q_len in enumerate(q_lens):
        seq_error = got[q_base : q_base + q_len] - reference_output[q_base : q_base + q_len]
        seq_ref = reference_output[q_base : q_base + q_len]
        seq_nrms = float(
            seq_error.square().mean().sqrt()
            / max(float(seq_ref.square().mean().sqrt()), 0.001)
        )
        seq_lse = float(
            (
                got_lse[q_base : q_base + q_len]
                - reference_lse[q_base : q_base + q_len]
            )
            .square()
            .mean()
            .sqrt()
        )
        assert seq_nrms <= seq_nrms_bound
        assert seq_lse <= seq_lse_bound
        for row in range(q_len):
            for head in range(HEADS):
                error = got[q_base + row, head] - reference_output[q_base + row, head]
                maxabs = float(error.abs().max())
                nrms = float(
                    error.square().mean().sqrt()
                    / max(
                        float(
                            reference_output[q_base + row, head]
                            .square()
                            .mean()
                            .sqrt()
                        ),
                        0.001,
                    )
                )
                lse_abs = float(
                    abs(
                        got_lse[q_base + row, head]
                        - reference_lse[q_base + row, head]
                    )
                )
                assert math.isfinite(maxabs) and math.isfinite(nrms) and math.isfinite(lse_abs)
                assert maxabs <= maxabs_bound
                assert nrms <= nrms_bound
                assert lse_abs <= lse_bound
                records.append(
                    {
                        "sequence": sequence,
                        "query_row": row,
                        "query_head": head,
                        "output_maxabs": maxabs,
                        "output_nrms": nrms,
                        "lse_abs": lse_abs,
                        "sequence_output_nrms": seq_nrms,
                        "sequence_lse_rms": seq_lse,
                    }
                )
        q_base += q_len
    return records


def _run_fp8_case(
    case: AttentionCase,
    manifest: dict[str, Any],
    *,
    return_lse: bool = True,
) -> dict[str, Any]:
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

    k_value, v_value = _scale_pair(manifest, case.seed)
    generator = torch.Generator().manual_seed(case.seed)
    query_cpu = _cpu_values(generator, (sum(case.q_lens), HEADS, HEAD_DIM), k_value)
    key_cpu = _cpu_values(generator, (sum(case.k_lens), KV_HEADS, HEAD_DIM), k_value)
    value_cpu = _cpu_values(generator, (sum(case.k_lens), KV_HEADS, HEAD_DIM), v_value)
    k_scale = torch.tensor(k_value, dtype=torch.float32, device="cuda")
    v_scale = torch.tensor(v_value, dtype=torch.float32, device="cuda")
    query = _quantize_query(query_cpu, k_scale)
    expected_q = _quantize_expected(query_cpu, k_value)
    assert torch.equal(query.cpu().view(torch.uint8), expected_q.view(torch.uint8))
    key = key_cpu.cuda()
    value = value_cpu.cuda()
    key_cache, value_cache, cache, per_sequence = _write_fp8_cache(
        case, key, value, k_scale, v_scale
    )
    expected_k = _expected_cache(case, key_cpu, k_value, per_sequence)
    expected_v = _expected_cache(case, value_cpu, v_value, per_sequence)
    assert torch.equal(key_cache.cpu().view(torch.uint8), expected_k.view(torch.uint8))
    assert torch.equal(value_cache.cpu().view(torch.uint8), expected_v.view(torch.uint8))

    table = _padded_table(case).cuda()
    cu_q = torch.tensor(
        (0, *torch.tensor(case.q_lens).cumsum(0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    used_k = torch.tensor(case.k_lens, dtype=torch.int32, device="cuda")
    cache_before = (key_cache.clone(), value_cache.clone())
    batch = len(case.q_lens)
    output, lse, _, _ = _flash_attn_fwd(
        query,
        key_cache,
        value_cache,
        cu_seqlens_q=cu_q,
        seqused_k=used_k,
        max_seqlen_q=max(case.q_lens),
        max_seqlen_k=max(case.k_lens),
        page_table=table,
        softmax_scale=1.0 / math.sqrt(HEAD_DIM),
        causal=case.input_causal,
        tile_mn=(case.tile_m, 128),
        num_splits=case.num_splits,
        pack_gqa=True,
        return_lse=return_lse,
        q_descale=_descale(k_scale, batch),
        k_descale=_descale(k_scale, batch),
        v_descale=_descale(v_scale, batch),
        _arch=90,
    )
    assert output.dtype == torch.bfloat16
    assert torch.equal(key_cache.view(torch.uint8), cache_before[0].view(torch.uint8))
    assert torch.equal(value_cache.view(torch.uint8), cache_before[1].view(torch.uint8))
    result: dict[str, Any] = {
        "output": output,
        "lse": lse,
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": table,
        "slot_mapping": per_sequence,
        "logical_hashes": {
            "q_bf16": _tensor_hash(query_cpu),
            "k_bf16": _tensor_hash(key_cpu),
            "v_bf16": _tensor_hash(value_cpu),
        },
        "byte_hashes": {
            "q": _tensor_hash(query),
            "k": _tensor_hash(key_cache),
            "v": _tensor_hash(value_cache),
            "full_cache": _tensor_hash(cache),
        },
        "expected_byte_hashes": {
            "q": _tensor_hash(expected_q),
            "logical_k": _tensor_hash(_quantize_expected(key_cpu, k_value)),
            "logical_v": _tensor_hash(_quantize_expected(value_cpu, v_value)),
            "k_cache": _tensor_hash(expected_k),
            "v_cache": _tensor_hash(expected_v),
        },
    }
    if return_lse:
        reference_output, reference_lse = _reference(
            query,
            key_cache,
            value_cache,
            case,
            per_sequence,
            k_value,
            k_value,
            v_value,
        )
        result["records"] = _metrics(
            output,
            lse,
            reference_output,
            reference_lse,
            case.q_lens,
            (0.16, 0.08, 0.10, 0.025, 0.025),
        )
    return result


def _tensor_hash(tensor: torch.Tensor) -> str:
    import hashlib

    data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("case", ARITHMETIC_CASES, ids=lambda case: case.name)
def test_fp8_direct_arithmetic(case, fp8_scale_manifest, fp8_record):
    result = _run_fp8_case(case, fp8_scale_manifest)
    expected = sum(case.q_lens) * HEADS
    assert len(result["records"]) == expected
    fp8_record(
        {
            "id": case.name,
            "plane": "flash-attention-direct",
            "seed": case.seed,
            "scale_layer": SEED_LAYERS[case.seed],
            "q_lengths": case.q_lens,
            "k_lengths": case.k_lens,
            "input_causal": case.input_causal,
            "canonical_key_causal": case.canonical_causal,
            "block_table": _padded_table(case).tolist(),
            "slot_mapping": result["slot_mapping"],
            "logical_hashes": result["logical_hashes"],
            "byte_hashes": result["byte_hashes"],
            "expected_byte_hashes": result["expected_byte_hashes"],
            "expected_row_head_records": expected,
            "observed_row_head_records": len(result["records"]),
            "row_head_records": result["records"],
        }
    )


def _run_masked_semantics(manifest):
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

    seed = 17
    k_value, v_value = _scale_pair(manifest, seed)
    generator = torch.Generator().manual_seed(seed)
    query_cpu = _cpu_values(generator, (2, HEADS, HEAD_DIM), k_value)
    key_cpu = _cpu_values(generator, (1, KV_HEADS, HEAD_DIM), k_value)
    value_cpu = _cpu_values(generator, (1, KV_HEADS, HEAD_DIM), v_value)
    k_scale = torch.tensor(k_value, dtype=torch.float32, device="cuda")
    v_scale = torch.tensor(v_value, dtype=torch.float32, device="cuda")
    query = _quantize_query(query_cpu, k_scale)
    expected_q = _quantize_expected(query_cpu, k_value)
    assert torch.equal(query.cpu().view(torch.uint8), expected_q.view(torch.uint8))
    key = _quantize_expected(key_cpu, k_value).cuda()
    value = _quantize_expected(value_cpu, v_value).cuda()
    cu_q = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    output, lse, _, _ = _flash_attn_fwd(
        query,
        key,
        value,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=2,
        max_seqlen_k=1,
        causal=True,
        tile_mn=(64, 128),
        pack_gqa=True,
        return_lse=True,
        q_descale=_descale(k_scale, 1),
        k_descale=_descale(k_scale, 1),
        v_descale=_descale(v_scale, 1),
        _arch=90,
    )
    return {
        "seed": seed,
        "k_value": k_value,
        "v_value": v_value,
        "query_cpu": query_cpu,
        "key_cpu": key_cpu,
        "value_cpu": value_cpu,
        "query": query,
        "expected_q": expected_q,
        "key": key,
        "value": value,
        "output": output,
        "lse": lse,
    }


def test_fp8_fully_masked_semantics(fp8_scale_manifest, fp8_record):
    result = _run_masked_semantics(fp8_scale_manifest)
    seed = result["seed"]
    k_value = result["k_value"]
    v_value = result["v_value"]
    query_cpu = result["query_cpu"]
    key_cpu = result["key_cpu"]
    value_cpu = result["value_cpu"]
    query = result["query"]
    expected_q = result["expected_q"]
    key = result["key"]
    value = result["value"]
    output = result["output"]
    lse = result["lse"]
    lse_rows = _lse_rows(lse, 2)
    assert torch.equal(output[0].view(torch.int16), torch.zeros_like(output[0].view(torch.int16)))
    assert torch.isneginf(lse_rows[0]).all()
    assert torch.isfinite(output[1]).all() and torch.isfinite(lse_rows[1]).all()
    dense_case = AttentionCase("fully-masked-semantics", seed, (2,), (1,), ((0,),), 1, 64, 1, True)
    reference_output, reference_lse = _reference(
        query,
        key.reshape(1, 1, KV_HEADS, HEAD_DIM),
        value.reshape(1, 1, KV_HEADS, HEAD_DIM),
        dense_case,
        ((0,),),
        k_value,
        k_value,
        v_value,
    )
    records = _metrics(
        output[1:],
        lse_rows[1:],
        reference_output[1:],
        reference_lse[1:],
        (1,),
        (0.16, 0.08, 0.10, 0.025, 0.025),
    )
    masked = [
        {"sequence": 0, "query_row": 0, "query_head": head, "fully_masked": True}
        for head in range(HEADS)
    ]
    assert len(masked) + len(records) == 64
    fp8_record(
        {
            "id": dense_case.name,
            "plane": "flash-attention-direct",
            "seed": seed,
            "scale_layer": SEED_LAYERS[seed],
            "logical_hashes": {
                "q_bf16": _tensor_hash(query_cpu),
                "k_bf16": _tensor_hash(key_cpu),
                "v_bf16": _tensor_hash(value_cpu),
            },
            "byte_hashes": {
                "q": _tensor_hash(query),
                "k": _tensor_hash(key),
                "v": _tensor_hash(value),
            },
            "expected_byte_hashes": {
                "q": _tensor_hash(expected_q),
                "k": _tensor_hash(_quantize_expected(key_cpu, k_value)),
                "v": _tensor_hash(_quantize_expected(value_cpu, v_value)),
            },
            "expected_row_head_records": 64,
            "observed_row_head_records": 64,
            "row_head_records": masked + records,
        }
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("form", ["prefill", "decode"])
@pytest.mark.parametrize("tile_m", [64, 128], ids=["tile-64", "tile-128"])
def test_non_fp8_direct(dtype, form, tile_m):
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

    if form == "prefill":
        case = AttentionCase(
            "non-fp8-prefill",
            424242,
            (17, 257),
            (17, 257),
            PREFILL_ROWS,
            96,
            tile_m,
            1,
            True,
        )
    else:
        case = AttentionCase(
            "non-fp8-decode",
            424242,
            (1, 1, 1, 1),
            (16, 17, 128, 1025),
            DECODE_ROWS,
            192,
            tile_m,
            1,
            True,
        )
    generator = torch.Generator().manual_seed(case.seed)
    query = _non_fp8_values(generator, (sum(case.q_lens), HEADS, HEAD_DIM), dtype).cuda()
    logical_k = _non_fp8_values(generator, (sum(case.k_lens), KV_HEADS, HEAD_DIM), dtype)
    logical_v = _non_fp8_values(generator, (sum(case.k_lens), KV_HEADS, HEAD_DIM), dtype)
    _, per_sequence = _slots(case)
    expected_key = torch.zeros(
        case.num_pages * PAGE_SIZE, KV_HEADS, HEAD_DIM, dtype=dtype
    )
    expected_value = torch.zeros_like(expected_key)
    offset = 0
    for length, slots in zip(case.k_lens, per_sequence, strict=True):
        expected_key[list(slots)] = logical_k[offset : offset + length]
        expected_value[list(slots)] = logical_v[offset : offset + length]
        offset += length
    expected_key = expected_key.reshape(
        case.num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM
    )
    expected_value = expected_value.reshape_as(expected_key)
    cache = torch.zeros(
        case.num_pages,
        KV_HEADS,
        PAGE_SIZE,
        2 * HEAD_DIM,
        dtype=dtype,
        device="cuda",
    )
    key_cache, value_cache = cache.transpose(1, 2).split(HEAD_DIM, dim=-1)
    key_cache.copy_(expected_key.cuda())
    value_cache.copy_(expected_value.cuda())
    table = _padded_table(case).cuda()
    cu_q = torch.tensor(
        (0, *torch.tensor(case.q_lens).cumsum(0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    used_k = torch.tensor(case.k_lens, dtype=torch.int32, device="cuda")
    before = (key_cache.clone(), value_cache.clone())
    output, lse, _, _ = _flash_attn_fwd(
        query,
        key_cache,
        value_cache,
        cu_seqlens_q=cu_q,
        seqused_k=used_k,
        max_seqlen_q=max(case.q_lens),
        max_seqlen_k=max(case.k_lens),
        page_table=table,
        causal=case.input_causal,
        tile_mn=(tile_m, 128),
        num_splits=1,
        pack_gqa=True,
        return_lse=True,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        _arch=90,
    )
    assert output.dtype == dtype
    assert torch.equal(key_cache, before[0]) and torch.equal(value_cache, before[1])
    reference_output, reference_lse = _reference(
        query, key_cache, value_cache, case, per_sequence, 1.0, 1.0, 1.0
    )
    bounds = (
        (0.03, 0.02, 0.03, 0.01, 0.01)
        if dtype == torch.bfloat16
        else (0.015, 0.01, 0.02, 0.005, 0.008)
    )
    records = _metrics(output, lse, reference_output, reference_lse, case.q_lens, bounds)
    assert len(records) == sum(case.q_lens) * HEADS


def test_fp8_precompile(fp8_scale_manifest, fp8_record):
    from vllm.vllm_flash_attn.cute.interface import (
        _flash_attn_fwd,
        _flash_attn_fwd_combine,
    )

    def compile_cache_keys():
        return {
            "forward": tuple(
                sorted(repr(key) for key in _flash_attn_fwd.compile_cache.cache)
            ),
            "combine": tuple(
                sorted(
                    repr(key) for key in _flash_attn_fwd_combine.compile_cache.cache
                )
            ),
        }

    representatives = (
        ARITHMETIC_CASES[3],
        ARITHMETIC_CASES[6],
        PERSISTENT_DECODE_CASE,
        ARITHMETIC_CASES[7],
        ARITHMETIC_CASES[8],
        ARITHMETIC_CASES[9],
        ARITHMETIC_CASES[0],
        ARITHMETIC_CASES[11],
        ARITHMETIC_CASES[12],
    )
    for case in representatives:
        _run_fp8_case(case, fp8_scale_manifest, return_lse=False)
    production = compile_cache_keys()
    for case in representatives:
        _run_fp8_case(case, fp8_scale_manifest, return_lse=True)
    _run_masked_semantics(fp8_scale_manifest)
    test_only = compile_cache_keys()
    assert len(production["forward"]) == 9
    assert len(production["combine"]) == 1
    assert len(test_only["forward"]) == 19
    assert len(test_only["combine"]) == 2
    fp8_record(
        {
            "id": "precompile",
            "plane": "flash-attention-precompile",
            "production": production,
            "test_only": test_only,
        }
    )


def test_fp8_sanitizer_decode_seed_2027(fp8_scale_manifest):
    case = next(case for case in ARITHMETIC_CASES if case.name == "paged-decode-seed-2027")
    result = _run_fp8_case(case, fp8_scale_manifest)
    assert len(result["records"]) == 128
