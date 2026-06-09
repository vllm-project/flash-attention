# Copyright (c) 2026, Tri Dao.

import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from einops import rearrange

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_VLLM_ROOT = _WORKSPACE_ROOT / "vllm"
if _VLLM_ROOT.exists() and str(_VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VLLM_ROOT))

try:
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode
except ImportError:
    pytest.skip(
        "Sibling vLLM repo is required for Triton reference",
        allow_module_level=True,
    )

try:
    from vllm import _custom_ops as vllm_ops
except Exception:
    vllm_ops = None

# FA4 CuTe fp16-Q + fp8-KV-cache dequant SM90 forward (the kernel under test).
# flash_attn/__init__ eagerly imports the compiled FA2 extension (flash_attn_2_cuda);
# stub it so the pure-Python cute path imports without a built FA2 wheel.
import types as _types

sys.modules.setdefault("flash_attn_2_cuda", _types.ModuleType("flash_attn_2_cuda"))
try:
    from flash_attn.cute.interface import _flash_attn_fwd as _cute_flash_attn_fwd
    from flash_attn.cute.testing import attention_ref
except Exception:  # pragma: no cover - exercised only when the cute path is unavailable
    _cute_flash_attn_fwd = None
    attention_ref = None


# Gemma4 uses softmax scale = 1.0 (gemma4.py self.scaling=1.0; learnable QK-norm handles
# scaling implicitly), NOT head_dim**-0.5. This is the scale the kernels actually receive
# from vLLM. The earlier hardcoded head_dim**-0.5 (~0.044 for hd512) under-represented the
# real score magnitude by ~22x, so the op-level test never exercised the production regime.
GEMMA4_SOFTMAX_SCALE = 1.0
# The legacy scale the test previously hardcoded; kept for the scale-contrast diagnostic.
LEGACY_INVSQRT_SOFTMAX_SCALE = 512 ** -0.5


def _cute_dequant_paged_attention(
    query: torch.Tensor,
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    page_table: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    softmax_scale: float = GEMMA4_SOFTMAX_SCALE,
) -> torch.Tensor:
    """Run the SM90 fp16-Q + fp8-KV-cache dequant forward (fp16 Q + fp8 paged K/V -> fp16 O).

    Q is cast to fp16 (the compute dtype; no q_descale). The fp8 K/V are dequantized
    in-kernel to fp16; per-tensor k/v scales are passed as the descales the kernel
    folds (k into the score scale, v into the output). This matches the fp16-Q Triton
    path, which dequantizes the fp8 cache and runs the matmuls in fp16.
    """
    query = query.half()
    num_seqs = len(query_lens)
    num_kv_heads = key_cache_fp8.shape[2]
    cu_query_lens = torch.tensor(
        [0] + query_lens, device=query.device, dtype=torch.int32
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens_t = torch.tensor(kv_lens, device=query.device, dtype=torch.int32)
    scale_shape = (num_seqs, num_kv_heads)
    k_descale = torch.full(scale_shape, k_scale.item(), device=query.device, dtype=torch.float32)
    v_descale = torch.full(scale_shape, v_scale.item(), device=query.device, dtype=torch.float32)
    out = _cute_flash_attn_fwd(
        q=query,
        k=key_cache_fp8,
        v=value_cache_fp8,
        softmax_scale=softmax_scale,
        causal=True,
        q_descale=None,
        k_descale=k_descale,
        v_descale=v_descale,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(kv_lens),
        page_table=page_table,
        fp8_kv_dequant=True,
    )
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


@dataclass(frozen=True)
class AttentionCase:
    id: str
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    block_size: int
    query_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]


@dataclass(frozen=True)
class ScaleCase:
    id: str
    q: float
    k: float
    v: float


ATTENTION_CASES = (
    AttentionCase(
        id="mini_gemma4",
        num_query_heads=8, # tp4
        num_kv_heads=4, # tp4
        head_size=512,
        block_size=64,
        query_lens=(1, 5),
        kv_lens=(33, 47),
    ),
    AttentionCase(
        id="gemma4_offline_lockstep_decode",
        num_query_heads=8, # tp4
        num_kv_heads=4, # tp4
        head_size=512,
        block_size=64,
        query_lens=(1,) * 4,
        kv_lens=(28000,) * 4,
    ),
    AttentionCase(
        id="gemma4_offline_lockstep_prefill",
        num_query_heads=8, # tp4
        num_kv_heads=4, # tp4
        head_size=512,
        block_size=64,
        query_lens=(3000,) * 4,
        kv_lens=(28000,) * 4,
    ),
)

SCALE_CASES = (
    ScaleCase(id="default_unity_scales", q=1.0, k=1.0, v=1.0),
    ScaleCase(id="non_default_scales", q=0.5, k=0.5, v=0.25),
)
# ---------------------------------------------------------------------------
# Committed accuracy bars, named so they cannot silently drift (this replaces the
# old source-reading `test_committed_tolerances_unchanged` guard):
#   * FP8_KV_*:     the vLLM Triton FP8 bar shared by every FP8 comparison, see
#     vllm/tests/kernels/attention/test_triton_unified_attention.py.
#   * REF_XCHECK_*: the tight bar for the dense-vs-paged reference cross-check
#     (both formulations describe identical K/V, so this is near-exact).
# ---------------------------------------------------------------------------
FP8_KV_ATOL = FP8_KV_RTOL = 1.5e-1
REF_XCHECK_ATOL = REF_XCHECK_RTOL = 1e-2


def _quantize_per_tensor_fp8(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Static per-tensor FP8 quantization, matching vLLM's Q/cache convention."""
    if vllm_ops is not None:
        try:
            flat = tensor.contiguous().reshape(-1, tensor.shape[-1])
            quantized, _ = vllm_ops.scaled_fp8_quant(
                flat,
                scale,
                group_shape=(-1, -1),
            )
            return quantized.reshape(tensor.shape)
        except Exception:
            pass

    return (tensor / scale).to(torch.float8_e4m3fn)


# Local copy of the paged-KV builder used by tests/cute/test_flash_attn.py
# (its module-level `_generate_block_kvcache`; it is NOT exported from
# flash_attn.cute.testing). Inlined so this standalone vLLM/Triton-parity +
# benchmark harness imports no other flash-attention test module. The first-class
# FP8 paged correctness test now lives in test_flash_attn.py::test_flash_attn_kvcache.
def _generate_block_kvcache(
    seqlen_k, page_size, batch_size, nheads_k, d, dv, device, dtype, dtype_ref
):
    num_blocks = math.ceil(seqlen_k / page_size) * batch_size * 3
    k_cache_paged = (
        torch.randn(num_blocks, page_size, nheads_k, d, device=device, dtype=dtype_ref)
        .to(dtype)
        .to(dtype_ref)
    )
    v_cache_paged = (
        torch.randn(num_blocks, page_size, nheads_k, dv, device=device, dtype=dtype_ref)
        .to(dtype)
        .to(dtype_ref)
    )
    page_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=batch_size,
    )
    k_cache = rearrange(
        k_cache_paged[page_table.flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    v_cache = rearrange(
        v_cache_paged[page_table.flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    return k_cache, v_cache, page_table, k_cache_paged, v_cache_paged, num_blocks


def _build_case_inputs(attention_case: "AttentionCase", scale_case: "ScaleCase") -> dict:
    """Single source of truth for case tensors, shared by the pytest test and the
    __main__ benchmark.

    Builds the bf16 originals (dense + paged via the shared
    ``flash_attn.cute.testing._generate_block_kvcache``) plus the per-tensor FP8
    quantized variants the kernel / Triton / reference consume. Per-sequence
    lengths are carried as ``kv_lens`` (the kernel's ``seqused_k``); the cache is
    allocated uniformly at ``max(kv_lens)``.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.random.manual_seed(0)

    num_query_heads = attention_case.num_query_heads
    num_kv_heads = attention_case.num_kv_heads
    head_size = attention_case.head_size
    block_size = attention_case.block_size
    query_lens = list(attention_case.query_lens)
    kv_lens = list(attention_case.kv_lens)
    seqlen_k = max(kv_lens)

    k_cache, v_cache, page_table, k_cache_paged, v_cache_paged, _num_blocks = (
        _generate_block_kvcache(
            seqlen_k, block_size, len(kv_lens), num_kv_heads, head_size, head_size,
            device, dtype, dtype,
        )
    )

    query = torch.randn(
        sum(query_lens), num_query_heads, head_size, device=device, dtype=dtype
    )

    q_scale = torch.tensor(scale_case.q, device=device, dtype=torch.float32)
    k_scale = torch.tensor(scale_case.k, device=device, dtype=torch.float32)
    v_scale = torch.tensor(scale_case.v, device=device, dtype=torch.float32)

    query_fp8 = _quantize_per_tensor_fp8(query, q_scale)
    # Paged FP8 (kernel + Triton index these through the page table).
    key_cache_fp8 = _quantize_per_tensor_fp8(k_cache_paged, k_scale)
    value_cache_fp8 = _quantize_per_tensor_fp8(v_cache_paged, v_scale)
    # Dense FP8 (the reference reads these directly); identical bytes to the paged
    # buffers gathered through the page table, since quantization is elementwise.
    key_dense_fp8 = _quantize_per_tensor_fp8(k_cache, k_scale)
    value_dense_fp8 = _quantize_per_tensor_fp8(v_cache, v_scale)

    # Dense bf16 caches were only needed to derive the dense FP8 reference inputs.
    del k_cache, v_cache
    torch.cuda.empty_cache()

    return {
        "query_lens": query_lens,
        "kv_lens": kv_lens,
        "page_table": page_table,
        "head_size": head_size,
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "query": query,
        "key_cache": k_cache_paged,
        "value_cache": v_cache_paged,
        "query_fp8": query_fp8,
        "key_cache_fp8": key_cache_fp8,
        "value_cache_fp8": value_cache_fp8,
        "key_dense_fp8": key_dense_fp8,
        "value_dense_fp8": value_dense_fp8,
        "q_scale": q_scale,
        "k_scale": k_scale,
        "v_scale": v_scale,
    }


def _reference_paged_fp8_attention(
    query_fp8: torch.Tensor,
    key_dense_fp8: torch.Tensor,
    value_dense_fp8: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    softmax_scale: float = GEMMA4_SOFTMAX_SCALE,
) -> torch.Tensor:
    """Per-sequence FP8 reference via the shared ``attention_ref``.

    The FP8 inputs are handed over as a bf16 *container* (so attention_ref's final
    ``.to(dtype_og)`` does not re-quantize the output) together with per-tensor
    descales; attention_ref upcasts to fp32 and applies the descales, i.e.
    dequantize-then-attend. This replaces the hand-rolled causal/paged references.
    """
    num_kv_heads = key_dense_fp8.shape[2]
    device = query_fp8.device
    # attention_ref applies 1/sqrt(d) internally (after q_descale). Fold the target
    # softmax_scale into q_descale so the reference uses the SAME effective scale as the
    # kernels under test: q_descale_eff = q_scale * softmax_scale * sqrt(d). At the legacy
    # softmax_scale = 1/sqrt(d) this reduces to q_scale (back-compatible).
    _scale_fold = softmax_scale * (query_fp8.shape[-1] ** 0.5)
    q_desc = torch.full(
        (1, num_kv_heads), float(q_scale) * _scale_fold, device=device, dtype=torch.float32
    )
    k_desc = torch.full((1, num_kv_heads), float(k_scale), device=device, dtype=torch.float32)
    v_desc = torch.full((1, num_kv_heads), float(v_scale), device=device, dtype=torch.float32)

    outputs: list[torch.Tensor] = []
    q_start = 0
    for batch_idx, (query_len, kv_len) in enumerate(zip(query_lens, kv_lens)):
        out_i, _ = attention_ref(
            query_fp8[q_start : q_start + query_len][None].to(torch.bfloat16),
            key_dense_fp8[batch_idx, :kv_len][None].to(torch.bfloat16),
            value_dense_fp8[batch_idx, :kv_len][None].to(torch.bfloat16),
            causal=True,
            upcast=True,
            q_descale=q_desc,
            k_descale=k_desc,
            v_descale=v_desc,
        )
        outputs.append(out_i[0])
        q_start += query_len
    return torch.cat(outputs, dim=0)


def _gather_paged_to_dense_fp8(
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    page_table: torch.Tensor,
    kv_lens: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the paged FP8 buffers the kernel indexes back into dense
    ``(batch, max(kv_lens), num_kv_heads, head_size)`` views.

    Lets the reference run on the EXACT bytes the kernel reads, so the dense-vs-
    paged cross-check validates the page table / paged construction itself.
    """
    block_size = key_cache_fp8.shape[1]
    num_kv_heads = key_cache_fp8.shape[2]
    head_size = key_cache_fp8.shape[3]
    head_size_v = value_cache_fp8.shape[3]
    max_kv_len = max(kv_lens)
    key_dense = key_cache_fp8.new_zeros((len(kv_lens), max_kv_len, num_kv_heads, head_size))
    value_dense = value_cache_fp8.new_zeros(
        (len(kv_lens), max_kv_len, num_kv_heads, head_size_v)
    )
    for batch_idx, kv_len in enumerate(kv_lens):
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        blocks = page_table[batch_idx, :num_kv_blocks].long()
        key_dense[batch_idx, :kv_len] = key_cache_fp8[blocks].reshape(
            -1, num_kv_heads, head_size
        )[:kv_len]
        value_dense[batch_idx, :kv_len] = value_cache_fp8[blocks].reshape(
            -1, num_kv_heads, head_size_v
        )[:kv_len]
    return key_dense, value_dense


def _divergence(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    """(max_abs, max_rel, mean_abs) of actual vs expected, computed in fp32."""
    a, e = actual.float(), expected.float()
    diff = (a - e).abs()
    max_abs = diff.max().item()
    max_rel = (diff / e.abs().clamp_min(1e-6)).max().item()
    mean_abs = diff.mean().item()
    return max_abs, max_rel, mean_abs


def _triton_unified_attention_ref(
    query_fp8: torch.Tensor,
    key_cache_fp8: torch.Tensor,
    value_cache_fp8: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    page_table: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output_dtype: torch.dtype,
    softmax_scale: float = GEMMA4_SOFTMAX_SCALE,
) -> torch.Tensor:
    """Call vLLM's Triton FP8 Q + FP8 KV-cache path with expected shapes.

    Shapes:
      query_fp8: (sum(query_lens), num_query_heads, head_size)
      key_cache_fp8/value_cache_fp8:
        (num_blocks, block_size, num_kv_heads, head_size)
      page_table: (num_seqs, max_num_blocks_per_seq)
      q_scale/k_scale/v_scale: scalar per-tensor FP32 scales in this test
      q_descale: scalar
      k_descale/v_descale: (num_seqs, num_kv_heads)

    vLLM's TritonAttentionImpl gets key/value cache by unbinding an outer
    kv_cache shaped (num_blocks, 2, block_size, num_kv_heads, head_size).
    For FP8_PER_TENSOR on CUDA, vLLM also statically quantizes Q and passes
    scalar layer._q_scale plus expanded layer._k_scale/layer._v_scale.
    """
    num_seqs = len(query_lens)
    num_query_heads = query_fp8.shape[1]
    num_kv_heads = key_cache_fp8.shape[2]
    cu_query_lens = torch.tensor(
        [0] + query_lens, device=query_fp8.device, dtype=torch.int32
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens_t = torch.tensor(kv_lens, device=query_fp8.device, dtype=torch.int32)
    scale_shape = (num_seqs, num_kv_heads)
    k_descale = torch.full(
        scale_shape, k_scale.item(), device=query_fp8.device, dtype=torch.float32
    )
    v_descale = torch.full(
        scale_shape, v_scale.item(), device=query_fp8.device, dtype=torch.float32
    )
    out = torch.empty(
        query_fp8.shape,
        device=query_fp8.device,
        dtype=output_dtype,
    )

    unified_attention(
        q=query_fp8,
        k=key_cache_fp8,
        v=value_cache_fp8,
        out=out,
        cu_seqlens_q=cu_query_lens,
        max_seqlen_q=max(query_lens),
        seqused_k=kv_lens_t,
        max_seqlen_k=max(kv_lens),
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(-1, -1),
        block_table=page_table,
        softcap=0,
        q_descale=q_scale,
        k_descale=k_descale,
        v_descale=v_descale,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )
    assert out.shape == (sum(query_lens), num_query_heads, query_fp8.shape[-1])
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "attention_case", ATTENTION_CASES, ids=lambda attention_case: attention_case.id
)
@pytest.mark.parametrize(
    "scale_case", SCALE_CASES, ids=lambda scale_case: scale_case.id
)
@torch.inference_mode()
def test_gemma4_fp16q_fp8kv_dequant_matches_triton_unified_attention(
    attention_case: AttentionCase,
    scale_case: ScaleCase,
):
    """Gemma4 full-attention cases: fp16-Q + fp8-KV-cache dequant forward, fp16 Q
    with per-tensor FP8 paged KV cache dequantized in-kernel to fp16 (the consumer
    MMA warpgroups do the fp8->fp16 dequant; O is fp16).

    Q is true fp16 (no q_descale); only K/V are fp8. This is the production
    fp16-Q + fp8-KV regime, and the fp16-Q Triton path dequantizes the cache and
    runs the matmuls in fp16 -- a faithful, apples-to-apples oracle (unlike fp8-Q
    which would run fp8 MMA). Reports three divergences each run: CuteDSL-vs-
    Reference, Triton-vs-Reference, and CuteDSL-vs-Triton. The two CuteDSL
    comparisons are GATED at the vLLM bar (FP8_KV_*); Triton-vs-Reference is info.
    """
    if _cute_flash_attn_fwd is None:
        pytest.skip("FA4 CuTe interface is not importable in this environment")

    # The dequant path now folds the per-(batch, kv_head) k/v descales: q*k into the
    # per-tile softmax score scale and v into the final O scale (the dequant cast
    # stays a pure fp8->fp16 cast). So non-unity k/v scales match too.

    ins = _build_case_inputs(attention_case, scale_case)
    query_lens = ins["query_lens"]
    kv_lens = ins["kv_lens"]
    q_scale, k_scale, v_scale = ins["q_scale"], ins["k_scale"], ins["v_scale"]
    # Compute dtype is fp16: feed fp16 Q to the kernel, the Triton oracle, and the
    # reference (the reference upcasts to f32, so its result is ~unchanged). Using the
    # same fp16 tensor for both reference calls keeps the paging cross-check exact.
    q16 = ins["query"].half()
    # fp16 Q is true precision -> q_descale is identity (1.0) in the reference.
    q_scale_one = torch.tensor(1.0, device=ins["query"].device, dtype=torch.float32)

    # Reference on fp16 Q + dense FP8 K/V (attention_ref dequantizes K/V via descales).
    reference_output = _reference_paged_fp8_attention(
        q16, ins["key_dense_fp8"], ins["value_dense_fp8"],
        query_lens, kv_lens, q_scale_one, k_scale, v_scale,
    )
    # Cross-check: the paged buffers the kernel indexes gather back to the same
    # dense K/V (guards the page table / paged construction). Near-exact bar.
    key_gathered, value_gathered = _gather_paged_to_dense_fp8(
        ins["key_cache_fp8"], ins["value_cache_fp8"], ins["page_table"], kv_lens,
    )
    paged_reference_output = _reference_paged_fp8_attention(
        q16, key_gathered, value_gathered,
        query_lens, kv_lens, q_scale_one, k_scale, v_scale,
    )
    torch.testing.assert_close(
        reference_output, paged_reference_output,
        atol=REF_XCHECK_ATOL, rtol=REF_XCHECK_RTOL,
    )

    # Sanity: per-tensor FP8 quantization actually perturbed the KV cache.
    key_dequant = ins["key_cache_fp8"].to(torch.bfloat16).float() * k_scale
    assert (key_dequant - ins["key_cache"].float()).abs().max() > 0

    # fp16-Q Triton: Q_IS_FP8 is False, so it dequantizes the fp8 cache to fp16 and
    # runs fp16 matmuls (the same computation as the kernel under test).
    triton_output = _triton_unified_attention_ref(
        q16, ins["key_cache_fp8"], ins["value_cache_fp8"],
        query_lens, kv_lens, ins["page_table"], q_scale, k_scale, v_scale,
        torch.bfloat16,
    )
    # ---- FA4 CuTe fp16-Q + fp8-KV dequant SM90 forward (the kernel under test) ----
    cute_output = _cute_dequant_paged_attention(
        q16, ins["key_cache_fp8"], ins["value_cache_fp8"],
        query_lens, kv_lens, ins["page_table"], k_scale, v_scale,
    )

    # Report all three divergences (pytest warnings summary; no -s needed).
    # Emitted BEFORE the gates so the numbers show even when a gate fails.
    cr, tr, ct = (
        _divergence(cute_output, reference_output),
        _divergence(triton_output, reference_output),
        _divergence(cute_output, triton_output),
    )
    case_label = f"{attention_case.id}/{scale_case.id}"
    warnings.warn(
        f"[fp8-kv {case_label}] divergence (FP8 bar atol=rtol={FP8_KV_ATOL}):\n"
        f"    CuteDSL vs Reference [GATED]: max_abs={cr[0]:.4f} max_rel={cr[1]:.3f} mean_abs={cr[2]:.4f}\n"
        f"    Triton  vs Reference [info ]: max_abs={tr[0]:.4f} max_rel={tr[1]:.3f} mean_abs={tr[2]:.4f}\n"
        f"    CuteDSL vs Triton    [GATED]: max_abs={ct[0]:.4f} max_rel={ct[1]:.3f} mean_abs={ct[2]:.4f}",
        stacklevel=2,
    )

    # Gates: the kernel must match the fp32 reference AND the production Triton
    # path (vLLM FP8 bar). Triton-vs-Reference is intentionally NOT gated.
    # check_dtype=False: the dequant kernel's O is fp16 (compute dtype) while the
    # references are bf16; the values are compared (upcast to fp32 by assert_close),
    # not the storage dtype (the fp32 divergence above is the real accuracy gate).
    torch.testing.assert_close(
        cute_output, reference_output, atol=FP8_KV_ATOL, rtol=FP8_KV_RTOL, check_dtype=False
    )
    torch.testing.assert_close(
        cute_output, triton_output, atol=FP8_KV_ATOL, rtol=FP8_KV_RTOL, check_dtype=False
    )


# ===========================================================================
# Lightweight perf benchmark (NOT collected by pytest: not a test_* function,
# lives behind __main__). Shares `_build_case_inputs` with the pytest test above
# so the benchmark and the test exercise identical case construction.
#
# Compares these backends on the gemma4 decode + prefill cases:
#   * FA4-dequant  : our SM90 fp16-Q + fp8-KV in-kernel-dequant forward
#   * Triton-fp16q : vLLM unified_attention, fp16 Q + fp8 KV (the apples-to-apples oracle)
#   * Triton-fp8   : vLLM unified_attention FP8_PER_TENSOR (the customer backend)
#   * FA4-bf16     : the existing SM90 bf16 paged forward (the bf16 baseline)
#
# Run:
#   FLASH_ATTENTION_FAKE_TENSOR=0 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
#       python tests/cute/test_flash_attn_fp8_kv_cache.py
# ===========================================================================


def _bench_fa4_bf16_paged(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    page_table: torch.Tensor,
) -> torch.Tensor:
    """Run the EXISTING SM90 bf16 paged forward on the un-quantized bf16 inputs.

    Mirrors the arg construction in `_cute_dequant_paged_attention` but with the bf16
    originals and NO descales -- this is the existing bf16 SM90 paged forward, the
    baseline that the FP8 path is meant to beat.
    """
    cu_query_lens = torch.tensor(
        [0] + query_lens, device=query.device, dtype=torch.int32
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens_t = torch.tensor(kv_lens, device=query.device, dtype=torch.int32)
    out = _cute_flash_attn_fwd(
        q=query,
        k=key_cache,
        v=value_cache,
        softmax_scale=query.shape[-1] ** -0.5,
        causal=True,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(kv_lens),
        page_table=page_table,
    )
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _bench_make_fn(backend: str, ins: dict):
    """Return a zero-arg callable that runs one backend on the prebuilt inputs."""
    ql = ins["query_lens"]
    kl = ins["kv_lens"]
    pt = ins["page_table"]
    if backend == "fa4_dequant":
        return lambda: _cute_dequant_paged_attention(
            ins["query"], ins["key_cache_fp8"], ins["value_cache_fp8"],
            ql, kl, pt, ins["k_scale"], ins["v_scale"],
        )
    if backend == "triton_fp16q":
        # fp16-Q Triton: dequantizes the fp8 cache to fp16 and runs fp16 matmuls (the
        # same computation as fa4_dequant) -- the apples-to-apples production oracle.
        return lambda: _triton_unified_attention_ref(
            ins["query"].half(), ins["key_cache_fp8"], ins["value_cache_fp8"],
            ql, kl, pt, ins["q_scale"], ins["k_scale"], ins["v_scale"],
            output_dtype=torch.bfloat16,
        )
    if backend == "triton_fp8":
        return lambda: _triton_unified_attention_ref(
            ins["query_fp8"], ins["key_cache_fp8"], ins["value_cache_fp8"],
            ql, kl, pt, ins["q_scale"], ins["k_scale"], ins["v_scale"],
            output_dtype=torch.bfloat16,
        )
    if backend == "fa4_bf16":
        return lambda: _bench_fa4_bf16_paged(
            ins["query"], ins["key_cache"], ins["value_cache"], ql, kl, pt,
        )
    raise ValueError(f"unknown backend {backend!r}")


def _bench_correctness_guard(out: torch.Tensor, ins: dict) -> None:
    """Correctness-before-timing: assert shape/dtype/finiteness so we never
    benchmark a NaN or a broken config (kernels are already test-validated)."""
    expected_shape = (sum(ins["query_lens"]), ins["num_query_heads"], ins["head_size"])
    assert tuple(out.shape) == expected_shape, (
        f"unexpected output shape {tuple(out.shape)} != {expected_shape}"
    )
    assert out.dtype in (torch.bfloat16, torch.float16), (
        f"expected bf16/fp16 output, got {out.dtype}"
    )
    assert torch.isfinite(out.float()).all(), "non-finite values in output"


def _bench_flops_bytes(ins: dict):
    """FLOPs and HBM bytes summed per-sequence over the uniform case (batch=1 per
    seq). Uses bench_utils.flops / bandwidth_fwd_bytes. fp8 K/V => 1 byte; the
    FA4-bf16 baseline reads bf16 K/V => 2 bytes. Q/O are bf16-ish either way but
    bandwidth_fwd_bytes uses a single dtype_bytes; we report decode bytes at the
    backend's KV dtype below by passing the right dtype_bytes per backend.

    Returns (total_flops, fp8_bytes, bf16_bytes).
    """
    from flash_attn.cute.bench_utils import flops, bandwidth_fwd_bytes

    nheads = ins["num_query_heads"]
    nheads_kv = ins["num_kv_heads"]
    hd = ins["head_size"]
    total_flops = 0.0
    fp8_bytes = 0.0
    bf16_bytes = 0.0
    for sq, sk in zip(ins["query_lens"], ins["kv_lens"]):
        total_flops += flops(
            batch=1, nheads=nheads, seqlen_q=sq, seqlen_k=sk,
            headdim=hd, headdim_v=hd, causal=True,
        )
        fp8_bytes += bandwidth_fwd_bytes(
            batch=1, nheads=nheads, nheads_kv=nheads_kv, seqlen_q=sq, seqlen_k=sk,
            headdim=hd, headdim_v=hd, dtype_bytes=1,
        )
        bf16_bytes += bandwidth_fwd_bytes(
            batch=1, nheads=nheads, nheads_kv=nheads_kv, seqlen_q=sq, seqlen_k=sk,
            headdim=hd, headdim_v=hd, dtype_bytes=2,
        )
    return total_flops, fp8_bytes, bf16_bytes


def _bench_time_one(fn, *, is_decode: bool, cudagraph: str, warmup: int, rep: int):
    """Time a single backend closure. Returns (median_ms, method).

    decode + cudagraph in {auto,on}: try do_bench_cudagraph on a fresh stream
    (hopper/benchmark_mla_decode.py idiom); on ANY exception fall back to
    do_bench. prefill (or cudagraph==off): plain do_bench.

    NOTE on units: triton.do_bench / do_bench_cudagraph build `times` from
    cuda.Event.elapsed_time(), which returns MILLISECONDS. (The repo's
    benchmark_mla_decode.py multiplies the result by 1e3 to print microseconds,
    confirming the return value is ms.) We therefore report the median in ms
    directly with NO 1e-3 rescale. We pass return_mode="median" for the median.
    """
    from triton.testing import do_bench, do_bench_cudagraph

    use_cg = is_decode and cudagraph in ("auto", "on")
    if use_cg:
        try:
            torch.cuda.synchronize()
            with torch.cuda.stream(torch.cuda.Stream()):
                med_ms = do_bench_cudagraph(fn, rep=rep, return_mode="median")
            return float(med_ms), "cudagraph"
        except Exception as exc:  # noqa: BLE001 -- graceful fallback by design
            if cudagraph == "on":
                # Explicit on: surface why it could not capture, but still fall back.
                print(f"        [cudagraph capture failed: {type(exc).__name__}: {exc}]")
    med_ms = do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
    return float(med_ms), "do_bench"


# ---------------------------------------------------------------------------
# CUDA-graph-capturable decode timing (additive; decode-cudagraph path only).
#
# The existing decode closures (`_cute_dequant_paged_attention`,
# `_triton_unified_attention_ref`, `_bench_fa4_bf16_paged`) rebuild device args on
# EVERY call -- `torch.tensor([0]+query_lens).cumsum(...)`, `torch.tensor(kv_lens)`,
# `torch.full(...).item()` k/v descales (a host-device sync!), and Triton also
# `torch.empty(...)` for `out` per call. Host syncs + per-call allocations abort
# the capture, so `do_bench_cudagraph` recorded an EMPTY graph and silently fell
# back to launch-overhead-inclusive `do_bench` timing.
#
# `_bench_capturable_fn` hoists ALL device-arg construction (the `.item()`,
# `torch.tensor`, `torch.full`, `torch.empty`, and the q_descale expansion) OUT of
# the timed closure so the closure is pure-launch (in-place into a pre-allocated
# `out_buf`). `_bench_time_cudagraph` then captures + times via `do_bench_cudagraph`.
# These are used ONLY for the decode case; prefill keeps plain `do_bench`.
# ---------------------------------------------------------------------------


def _bench_capturable_fn(backend: str, ins: dict):
    """Build ALL device args ONCE and return a pure-launch zero-arg closure.

    Every host-device sync (`.item()`) and every allocation (`torch.tensor`,
    `torch.full`, `torch.empty`, the `q_descale` expand) happens HERE, before
    capture -- never inside the returned closure. The closure runs in-place into
    a pre-allocated `out_buf` so the output pointer is stable across replays.

    Returns (fn, out_buf). `out_buf` is the static output the caller clones.
    """
    device = torch.device("cuda")
    query_lens = ins["query_lens"]
    kv_lens = ins["kv_lens"]
    pt = ins["page_table"]
    num_seqs = len(query_lens)
    num_kv_heads = ins["num_kv_heads"]
    num_query_heads = ins["num_query_heads"]
    head_size = ins["head_size"]
    total_q = sum(query_lens)

    # --- one-time device-arg construction (NO per-call host work) ---
    cu = torch.tensor(
        [0] + list(query_lens), device=device, dtype=torch.int32
    ).cumsum(dim=0, dtype=torch.int32)
    seqused = torch.tensor(list(kv_lens), device=device, dtype=torch.int32)
    mq = int(max(query_lens))
    mk = int(max(kv_lens))

    q_scale = ins["q_scale"]
    k_scale = ins["k_scale"]
    v_scale = ins["v_scale"]
    # k/v descales as (num_seqs, num_kv_heads) f32; do the .item() ONCE here.
    k_descale = torch.full(
        (num_seqs, num_kv_heads), k_scale.item(), device=device, dtype=torch.float32
    )
    v_descale = torch.full(
        (num_seqs, num_kv_heads), v_scale.item(), device=device, dtype=torch.float32
    )
    # Pre-expand q_descale to (num_seqs, num_kv_heads) f32 so interface.py's per-call
    # `q_descale.reshape(1).expand(...).contiguous()` (only fires when numel == 1) is
    # SKIPPED -- otherwise a fresh GPU tensor is allocated inside every replay.
    q_descale_exp = (
        q_scale.reshape(1).expand(num_seqs, num_kv_heads).contiguous().float()
    )

    # Static, pre-allocated output buffer (stable pointer across replays). The
    # fp8-dequant path computes in fp16 and writes fp16 O (== compute dtype), so its
    # buffer must be fp16; all other backends write bf16.
    out_buf = torch.empty(
        (total_q, num_query_heads, head_size),
        dtype=torch.float16 if backend == "fa4_dequant" else torch.bfloat16,
        device=device,
    )
    softmax_scale = head_size ** -0.5

    if backend == "fa4_bf16":
        q = ins["query"]
        k = ins["key_cache"]
        v = ins["value_cache"]

        def fn():
            _cute_flash_attn_fwd(
                q=q, k=k, v=v,
                softmax_scale=softmax_scale, causal=True,
                cu_seqlens_q=cu, seqused_k=seqused,
                max_seqlen_q=mq, max_seqlen_k=mk,
                page_table=pt, out=out_buf,
            )

        return fn, out_buf

    if backend == "triton_fp8":
        q8 = ins["query_fp8"]
        k8 = ins["key_cache_fp8"]
        v8 = ins["value_cache_fp8"]

        def fn():
            unified_attention(
                q=q8, k=k8, v=v8, out=out_buf,
                cu_seqlens_q=cu, max_seqlen_q=mq,
                seqused_k=seqused, max_seqlen_k=mk,
                softmax_scale=softmax_scale, causal=True,
                window_size=(-1, -1), block_table=pt, softcap=0,
                q_descale=q_descale_exp, k_descale=k_descale, v_descale=v_descale,
                kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
            )

        return fn, out_buf

    if backend == "fa4_dequant":
        # fp16-Q + fp8-KV dequant forward: fp16 Q + fp8 paged K/V dequantized in-kernel.
        # Cast to fp16 ONCE here (one-time alloc, stable pointer) -- this capturable
        # path calls the kernel directly and the dequant path requires fp16 Q.
        q = ins["query"].half()  # fp16 (true precision; not query_fp8)
        k8 = ins["key_cache_fp8"]
        v8 = ins["value_cache_fp8"]
        # Pre-built identity q_descale: fp16 Q needs no q-descale, and a STATIC tensor
        # avoids interface.py materializing torch.ones(...) inside every graph replay
        # (which would break cudagraph capture).
        q_descale_id = torch.ones(
            (num_seqs, num_kv_heads), device=device, dtype=torch.float32
        )

        def fn():
            _cute_flash_attn_fwd(
                q=q, k=k8, v=v8,
                softmax_scale=softmax_scale, causal=True,
                q_descale=q_descale_id, k_descale=k_descale, v_descale=v_descale,
                cu_seqlens_q=cu, seqused_k=seqused,
                max_seqlen_q=mq, max_seqlen_k=mk,
                page_table=pt, out=out_buf, fp8_kv_dequant=True,
            )

        return fn, out_buf

    if backend == "triton_fp16q":
        # fp16-Q Triton: Q_IS_FP8 is False -> dequantizes the fp8 cache to fp16 and runs
        # fp16 matmuls (the apples-to-apples oracle for fa4_dequant). q_descale is unused here.
        q = ins["query"].half()  # fp16
        k8 = ins["key_cache_fp8"]
        v8 = ins["value_cache_fp8"]

        def fn():
            unified_attention(
                q=q, k=k8, v=v8, out=out_buf,
                cu_seqlens_q=cu, max_seqlen_q=mq,
                seqused_k=seqused, max_seqlen_k=mk,
                softmax_scale=softmax_scale, causal=True,
                window_size=(-1, -1), block_table=pt, softcap=0,
                q_descale=q_descale_exp, k_descale=k_descale, v_descale=v_descale,
                kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
            )

        return fn, out_buf

    raise ValueError(f"unknown backend {backend!r}")


def _bench_time_cudagraph(fn, *, rep: int):
    """CUDA-graph capture + replay timing via triton's `do_bench_cudagraph`.

    Returns (median_ms, method). The closure must be pure-launch (every device-arg
    allocation + `.item()` sync hoisted out by `_bench_capturable_fn`) so the
    capture is NON-empty. We warm on a side stream first (FA4 JIT-compile + Triton
    autotune settle OFF the capture path), then let `do_bench_cudagraph` capture +
    time on a fresh stream (the hopper/benchmark_mla_decode.py idiom). On ANY
    capture failure we fall back to `do_bench` and tag the method, so the table
    never silently mixes capture results with launch-overhead-inclusive numbers.
    """
    from triton.testing import do_bench, do_bench_cudagraph

    try:
        torch.cuda.synchronize()
        # Warm on a side stream -- triggers FA4 JIT + Triton autotune off the
        # capture stream (so neither happens DURING capture).
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.cuda.stream(torch.cuda.Stream()):
            med_ms = do_bench_cudagraph(fn, rep=rep, return_mode="median")
        return float(med_ms), "cudagraph"
    except Exception as exc:  # noqa: BLE001 -- graceful fallback by design
        print(f"        [cudagraph capture failed: {type(exc).__name__}: {exc}]")
        med_ms = do_bench(fn, rep=rep, return_mode="median")
        return float(med_ms), "do_bench"


def _bench_run(args) -> None:
    import time

    if _cute_flash_attn_fwd is None:
        print("FA4 CuTe interface is not importable; cannot benchmark. Aborting.")
        return

    # Pick a free GPU if the caller did not pin one.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used",
                 "--format=csv,noheader,nounits"],
                text=True,
            )
            rows = [r.split(",") for r in out.strip().splitlines()]
            free = sorted(rows, key=lambda r: int(r[1]))[0][0].strip()
            os.environ["CUDA_VISIBLE_DEVICES"] = free
            print(f"[auto] CUDA_VISIBLE_DEVICES={free} (lowest memory.used)")
        except Exception as exc:  # noqa: BLE001
            print(f"[auto] GPU auto-pick failed ({exc}); using default device")

    case_by_id = {c.id: c for c in ATTENTION_CASES}
    decode_case = case_by_id["gemma4_offline_lockstep_decode"]
    prefill_case = case_by_id["gemma4_offline_lockstep_prefill"]
    scale_by_key = {"unity": SCALE_CASES[0], "non_default": SCALE_CASES[1]}

    want_cases = []
    if "decode" in args.cases:
        want_cases.append(("decode", decode_case, True))
    if "prefill" in args.cases:
        want_cases.append(("prefill", prefill_case, False))

    backend_label = {
        "fa4_dequant": "FA4-dequant",
        "triton_fp8": "Triton-fp8",
        "triton_fp16q": "Triton-fp16q",
        "fa4_bf16": "FA4-bf16",
    }

    for case_key, case, is_decode in want_cases:
        for scale_key in args.scales:
            scale_case = scale_by_key[scale_key]
            print()
            print("=" * 78)
            print(f"CASE: {case_key}  ({case.id})   SCALE: {scale_key} "
                  f"(q={scale_case.q}, k={scale_case.k}, v={scale_case.v})")
            print(f"  query_lens={case.query_lens}  kv_lens={case.kv_lens}  "
                  f"nheads={case.num_query_heads}/{case.num_kv_heads}  "
                  f"head_size={case.head_size}  block_size={case.block_size}")
            print("=" * 78)

            ins = _build_case_inputs(case, scale_case)
            total_flops, fp8_bytes, bf16_bytes = _bench_flops_bytes(ins)

            results = []  # (backend, method, median_ms, kv_bytes)
            for backend in args.backends:
                fn = _bench_make_fn(backend, ins)
                # --- mandatory prewarm (JIT-compile + autotune OUT of timing,
                #     and required before cudagraph capture) + correctness guard.
                ok = True
                last_out = None
                try:
                    for _ in range(args.prewarm_iters):
                        last_out = fn()
                    torch.cuda.synchronize()
                    _bench_correctness_guard(last_out, ins)
                except Exception as exc:  # noqa: BLE001 -- graceful per-backend skip
                    ok = False
                    print(f"  [{backend_label[backend]}] prewarm/guard FAILED -> "
                          f"skipping: {type(exc).__name__}: {exc}")
                if not ok:
                    results.append((backend, "—", None, None))
                    continue

                kv_bytes = bf16_bytes if backend == "fa4_bf16" else fp8_bytes
                time.sleep(1)  # throttling settle before the timed measurement
                if is_decode and args.cudagraph in ("auto", "on"):
                    # Decode CUDA-graph path: time a pure-launch closure (all device
                    # args hoisted out -> capture is NON-empty) via do_bench_cudagraph.
                    # `_bench_time_cudagraph` tags "cudagraph" on a real capture and
                    # "do_bench" only if capture genuinely fails.
                    cap_fn, _out_buf = _bench_capturable_fn(backend, ins)
                    med_ms, method = _bench_time_cudagraph(cap_fn, rep=args.rep)
                else:
                    med_ms, method = _bench_time_one(
                        fn, is_decode=is_decode, cudagraph=args.cudagraph,
                        warmup=args.warmup, rep=args.rep,
                    )
                results.append((backend, method, med_ms, kv_bytes))

            # FA4-bf16 baseline for the speedup column.
            fa4_bf16_ms = next(
                (ms for (b, m, ms, kb) in results if b == "fa4_bf16" and ms is not None),
                None,
            )

            metric_name = "TFLOPS" if not is_decode else "GB/s"
            print()
            print(f"  {'backend':<11} | {'method':<10} | {'median_ms':>10} | "
                  f"{'speedup_vs_FA4bf16':>18} | {metric_name:>10}")
            print("  " + "-" * 72)
            for backend, method, med_ms, kv_bytes in results:
                label = backend_label[backend]
                if med_ms is None:
                    print(f"  {label:<11} | {'—':<10} | {'—':>10} | "
                          f"{'—':>18} | {'—':>10}")
                    continue
                med_s = med_ms * 1e-3
                if not is_decode:
                    metric = total_flops * 1e-12 / med_s  # TFLOPS
                else:
                    metric = kv_bytes * 1e-9 / med_s  # GB/s
                if fa4_bf16_ms is not None and med_ms > 0:
                    speedup = f"{fa4_bf16_ms / med_ms:.3f}x"
                else:
                    speedup = "—"
                print(f"  {label:<11} | {method:<10} | {med_ms:>10.4f} | "
                      f"{speedup:>18} | {metric:>10.1f}")

            # Tidy up before the next (case, scale): drop everything and reclaim.
            del ins, results
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# nsys capture mode (additive; __main__-only, gated behind --nsys).
#
# Goal: collect a CONCISE, cudaProfilerApi-bounded kernel-level trace so we can
# read true per-kernel GPU durations + kernel names/grids and attribute them to
# (case, backend) via NVTX ranges. We deliberately split into two phases:
#
#   Phase 1 (OUTSIDE the profiled window): build inputs and the PURE-LAUNCH
#     closures (reusing `_bench_capturable_fn` -- works for decode AND prefill),
#     then prewarm each closure so FA4 JIT-compile and Triton autotune settle
#     here, NOT inside the capture. Closures (and their `ins`) stay resident.
#
#   Phase 2 (the capture window): cudaProfilerStart() ... per (case, backend,
#     scale) wrap `args.nsys_iters` pure-launch replays in an NVTX range, sync,
#     then cudaProfilerStop(). The trace therefore contains ONLY steady-state
#     kernel launches, each attributable to a labeled (case, backend, scale).
# --------------------------------------------------------------------------- #
def _bench_run_nsys(args) -> None:
    """Concise cudaProfilerApi-bounded nsys driver. Phase 1 prewarms outside the
    capture; Phase 2 replays the pure-launch closures inside the profiler window
    under NVTX ranges so kernels attribute to (case, backend, scale)."""
    if _cute_flash_attn_fwd is None:
        print("FA4 CuTe interface is not importable; cannot benchmark. Aborting.")
        return

    # Pick a free GPU if the caller did not pin one (mirrors `_bench_run`).
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used",
                 "--format=csv,noheader,nounits"],
                text=True,
            )
            rows = [r.split(",") for r in out.strip().splitlines()]
            free = sorted(rows, key=lambda r: int(r[1]))[0][0].strip()
            os.environ["CUDA_VISIBLE_DEVICES"] = free
            print(f"[auto] CUDA_VISIBLE_DEVICES={free} (lowest memory.used)")
        except Exception as exc:  # noqa: BLE001
            print(f"[auto] GPU auto-pick failed ({exc}); using default device")

    case_by_id = {c.id: c for c in ATTENTION_CASES}
    decode_case = case_by_id["gemma4_offline_lockstep_decode"]
    prefill_case = case_by_id["gemma4_offline_lockstep_prefill"]
    scale_by_key = {"unity": SCALE_CASES[0], "non_default": SCALE_CASES[1]}

    want_cases = []
    if "decode" in args.cases:
        want_cases.append(("decode", decode_case))
    if "prefill" in args.cases:
        want_cases.append(("prefill", prefill_case))

    backend_label = {
        "fa4_dequant": "FA4-dequant",
        "triton_fp8": "Triton-fp8",
        "triton_fp16q": "Triton-fp16q",
        "fa4_bf16": "FA4-bf16",
    }

    # --- Phase 1: build inputs + pure-launch closures, prewarm OUTSIDE capture.
    # Each entry: (label, case_key, backend, scale_key, cap_fn). We keep `ins`
    # and `cap_fn` resident (held by `keepalive`) so device pointers stay valid
    # through the capture window.
    plan = []          # list of (label, case_key, backend, scale_key, cap_fn)
    keepalive = []     # holds `ins` dicts + out_bufs so nothing is GC'd/freed
    print()
    print("=" * 78)
    print("nsys mode: Phase 1 -- build inputs + prewarm (OUTSIDE capture window)")
    print("=" * 78)
    for case_key, case in want_cases:
        for scale_key in args.scales:
            scale_case = scale_by_key[scale_key]
            print(f"  building {case_key}/{scale_key}  "
                  f"query_lens={case.query_lens}  kv_lens={case.kv_lens}")
            ins = _build_case_inputs(case, scale_case)
            keepalive.append(ins)
            for backend in args.backends:
                # Reuse the PURE-LAUNCH closure (all device-arg construction +
                # `.item()` syncs happen here, before any timed/captured window).
                cap_fn, out_buf = _bench_capturable_fn(backend, ins)
                keepalive.append(out_buf)
                # Prewarm: settle FA4 JIT-compile + Triton autotune OFF the
                # capture path. Skip a backend that fails (e.g. unsupported arch).
                try:
                    for _ in range(args.prewarm_iters):
                        cap_fn()
                    torch.cuda.synchronize()
                except Exception as exc:  # noqa: BLE001 -- graceful per-backend skip
                    print(f"    [{backend_label[backend]}] {case_key}/{scale_key} "
                          f"prewarm FAILED -> skipping: "
                          f"{type(exc).__name__}: {exc}")
                    continue
                label = f"{case_key}/{backend}/{scale_key}"
                plan.append((label, case_key, backend, scale_key, cap_fn))
                print(f"    [{backend_label[backend]}] prewarmed -> "
                      f"NVTX label '{label}'")

    if not plan:
        print("nsys mode: no backend prewarmed successfully; nothing to capture.")
        return

    # --- Phase 2: the capture window. Lazy-import the profiler/nvtx hooks here.
    import torch.cuda.profiler as cuda_profiler
    import torch.cuda.nvtx as nvtx

    print()
    print("=" * 78)
    print(f"nsys mode: Phase 2 -- capture window "
          f"({args.nsys_iters} iters/range, {len(plan)} ranges)")
    print("=" * 78)
    torch.cuda.synchronize()
    cuda_profiler.start()
    try:
        for label, _case_key, _backend, _scale_key, cap_fn in plan:
            with nvtx.range(label):
                for _ in range(args.nsys_iters):
                    cap_fn()
            torch.cuda.synchronize()
    finally:
        cuda_profiler.stop()
    torch.cuda.synchronize()

    # --- Report the NVTX label -> (case, backend, scale) map + stats hint.
    print()
    print("=" * 78)
    print("nsys mode: NVTX label -> (case, backend, scale) map")
    print("=" * 78)
    for label, case_key, backend, scale_key, _cap_fn in plan:
        print(f"  {label:<40} -> case={case_key:<8} "
              f"backend={backend_label[backend]:<11} scale={scale_key}")
    print()
    print("Suggested analysis (point at the .nsys-rep you wrote with -o):")
    print("  nsys stats --report nvtx_gpu_proj_sum --report cuda_gpu_kern_sum \\")
    print("    --format table <out>.nsys-rep")


def _bench_main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Perf benchmark: FA4-dequant vs Triton vs FA4-bf16 on "
                    "gemma4 decode + prefill (additive to the test file)."
    )

    def _csv(allowed):
        def parse(s):
            vals = [x.strip() for x in s.split(",") if x.strip()]
            bad = [v for v in vals if v not in allowed]
            if bad:
                raise argparse.ArgumentTypeError(
                    f"invalid value(s) {bad}; allowed: {sorted(allowed)}"
                )
            return vals
        return parse

    parser.add_argument("--cases", type=_csv({"decode", "prefill"}),
                        default=["decode", "prefill"])
    parser.add_argument("--scales", type=_csv({"unity", "non_default"}),
                        default=["unity", "non_default"])
    parser.add_argument("--backends",
                        type=_csv({"fa4_dequant", "triton_fp16q", "triton_fp8", "fa4_bf16"}),
                        default=["fa4_dequant", "triton_fp16q", "fa4_bf16", "triton_fp8"])
    parser.add_argument("--cudagraph", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=10)
    parser.add_argument("--prewarm-iters", type=int, default=3, dest="prewarm_iters")
    # nsys capture mode (additive): replay pure-launch closures inside a
    # cudaProfilerApi window under NVTX ranges for a concise kernel-level trace.
    parser.add_argument("--nsys", action="store_true", default=False,
                        help="cudaProfilerApi-bounded nsys capture mode "
                             "(skips the normal do_bench/cudagraph timing run).")
    parser.add_argument("--nsys-iters", type=int, default=20, dest="nsys_iters",
                        help="pure-launch replays per NVTX range in --nsys mode.")
    args = parser.parse_args(argv)
    if args.nsys:
        _bench_run_nsys(args)
        return
    _bench_run(args)


if __name__ == "__main__":
    import os  # noqa: E402 -- local to the __main__ entry; tests don't need it.

    # When launched as `python tests/cute/<file>.py`, sys.path[0] is this script's
    # dir (tests/cute), not the repo root, so the cwd-resolved (not pip-installed)
    # `flash_attn` package is not importable. pytest avoids this by running from the
    # repo root. Re-add the FA repo root here and re-import the cute interface if the
    # module-level import was swallowed -- additive, __main__-only, mirrors the
    # existing module-level vLLM-path guard. Tests are unaffected.
    _FA_REPO_ROOT = str(Path(__file__).resolve().parents[2])
    if _FA_REPO_ROOT not in sys.path:
        sys.path.insert(0, _FA_REPO_ROOT)
    if _cute_flash_attn_fwd is None:
        try:
            from flash_attn.cute.interface import _flash_attn_fwd as _cute_flash_attn_fwd
            from flash_attn.cute.testing import attention_ref
        except Exception:  # pragma: no cover
            _cute_flash_attn_fwd = None

    _bench_main()
