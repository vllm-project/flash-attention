import argparse
import math
import os
import random
from typing import List, Tuple

import torch

# We benchmark the vLLM FA2 varlen path with and without returning auxiliary tensors (e.g., abs_s).
from vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

# Helper: robust call wrappers that always try FA2 first, and gracefully
# drop unknown kwargs (like return_aux or fa_version) for older installs.
def _call_varlen_linear(
    q, k, v,
    max_q, cu_q, max_k, cu_k,
    *, causal=True, softcap=0.0,
    want_aux=False, want_lse=False,
):
    # Old environments don't accept `seqused_k` at all for linear (non-paged) path.
    kwargs_common = dict(
        q=q, k=k, v=v,
        max_seqlen_q=max_q,
        cu_seqlens_q=cu_q,
        max_seqlen_k=max_k,
        cu_seqlens_k=cu_k,
        causal=causal,
        softcap=softcap,
        return_softmax_lse=want_lse,
    )
    # Try with FA2 flags first
    try:
        return flash_attn_varlen_func(
            **kwargs_common,
            return_aux=want_aux,
            fa_version=2,
        )
    except TypeError as e:
        msg = str(e)
        # Drop unknown kwargs and retry progressively
        try_kwargs = dict(kwargs_common)
        if "return_aux" in msg:
            try_kwargs.pop("return_aux", None)
        else:
            try_kwargs["return_aux"] = want_aux
        if "fa_version" in msg:
            # no fa_version in older builds
            pass
        else:
            try_kwargs["fa_version"] = 2
        # Final minimal retry without both (and without return_softmax_lse if needed)
        try:
            return flash_attn_varlen_func(**{k:v for k,v in try_kwargs.items() if k not in ("return_aux", "fa_version")})
        except TypeError:
            # Last resort: drop return_softmax_lse too
            try_kwargs.pop("return_softmax_lse", None)
            return flash_attn_varlen_func(**{k:v for k,v in try_kwargs.items() if k not in ("return_aux", "fa_version")})


def _call_varlen_paged(
    q, k_paged, v_paged,
    max_q, cu_q, max_k,
    block_table, seqused_k,
    *, causal=True, softcap=0.0,
    want_aux=False, want_lse=False,
):
    kwargs_common = dict(
        q=q, k=k_paged, v=v_paged,
        max_seqlen_q=max_q,
        cu_seqlens_q=cu_q,
        max_seqlen_k=max_k,
        cu_seqlens_k=None,
        seqused_k=seqused_k,
        block_table=block_table,
        causal=causal,
        softcap=softcap,
        return_softmax_lse=want_lse,
    )
    try:
        return flash_attn_varlen_func(
            **kwargs_common,
            return_aux=want_aux,
            fa_version=2,
        )
    except TypeError as e:
        msg = str(e)
        try_kwargs = dict(kwargs_common)
        if "return_aux" in msg:
            try_kwargs.pop("return_aux", None)
        else:
            try_kwargs["return_aux"] = want_aux
        if "fa_version" in msg:
            pass
        else:
            try_kwargs["fa_version"] = 2
        # First retry: drop aux/fa_version
        try:
            return flash_attn_varlen_func(**{k:v for k,v in try_kwargs.items() if k not in ("return_aux", "fa_version")})
        except TypeError as e2:
            # Second retry: also drop return_softmax_lse
            try_kwargs.pop("return_softmax_lse", None)
            try:
                return flash_attn_varlen_func(**{k:v for k,v in try_kwargs.items() if k not in ("return_aux", "fa_version")})
            except TypeError as e3:
                # Likely the environment doesn't support paged args (seqused_k/block_table)
                raise NotImplementedError(f"Paged-KV path unsupported in this environment: {e3}")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_varlen_qkv(
    B: int,
    Hq: int,
    Hkv: int,
    D: int,
    seqlens_q: List[int],
    seqlens_k: List[int],
    dtype: torch.dtype,
    device: str,
):
    assert len(seqlens_q) == B and len(seqlens_k) == B
    total_q = int(sum(seqlens_q))
    total_k = int(sum(seqlens_k))
    q = torch.randn(total_q, Hq, D, device=device, dtype=dtype)
    k = torch.randn(total_k, Hkv, D, device=device, dtype=dtype)
    v = torch.randn_like(k)

    cu_q = (
        torch.tensor([0] + list(seqlens_q), device=device, dtype=torch.int32)
        .cumsum(0, dtype=torch.int32)
        .contiguous()
    )
    cu_k = (
        torch.tensor([0] + list(seqlens_k), device=device, dtype=torch.int32)
        .cumsum(0, dtype=torch.int32)
        .contiguous()
    )
    return q, k, v, cu_q, cu_k, max(seqlens_q), max(seqlens_k)


def _linear_to_paged_kv(
    k_lin: torch.Tensor,
    v_lin: torch.Tensor,
    seqlens_k: List[int],
    Hkv: int,
    D: int,
    page_block_size: int = 16,
):
    """Convert linear KV (total_k, Hkv, D) to paged layout.

    Returns:
      k_paged, v_paged: (num_blocks, page_block_size, Hkv, D)
      block_table: (B, max_blocks) int32
      seqused_k: (B,) int32
    """
    assert k_lin.dim() == 3 and k_lin.shape[1] == Hkv and k_lin.shape[2] == D
    assert v_lin.shape == k_lin.shape
    assert page_block_size % 16 == 0
    device = k_lin.device
    dtype = k_lin.dtype
    B = len(seqlens_k)
    # blocks per seq
    nblocks = [int((s + page_block_size - 1) // page_block_size) for s in seqlens_k]
    max_blocks = max(nblocks) if B > 0 else 0
    total_blocks = sum(nblocks)

    k_paged = torch.zeros(total_blocks, page_block_size, Hkv, D, device=device, dtype=dtype)
    v_paged = torch.zeros_like(k_paged)
    block_table = torch.zeros(B, max_blocks, device=device, dtype=torch.int32)

    k_offsets = [0]
    for s in seqlens_k:
        k_offsets.append(k_offsets[-1] + s)
    blk = 0
    for b in range(B):
        start = k_offsets[b]
        Sk = seqlens_k[b]
        for j in range(nblocks[b]):
            block_table[b, j] = blk
            pos0 = start + j * page_block_size
            pos1 = min(start + (j + 1) * page_block_size, start + Sk)
            span = pos1 - pos0
            if span > 0:
                k_paged[blk, :span].copy_(k_lin[pos0:pos1])
                v_paged[blk, :span].copy_(v_lin[pos0:pos1])
            blk += 1

    seqused_k = torch.tensor(seqlens_k, device=device, dtype=torch.int32)
    return k_paged.contiguous(), v_paged.contiguous(), block_table.contiguous(), seqused_k.contiguous()


def _time_cuda_callable(fn, warmup: int, iters: int) -> float:
    # Return average milliseconds elapsed over iters
    torch.cuda.synchronize()
    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    return ms / max(1, iters)


def _maybe_to_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str.lower() in ["fp16", "float16", "half"]:
        return torch.float16
    if dtype_str.lower() in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if dtype_str.lower() in ["fp32", "float", "float32"]:
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_str}")


def _gen_seqlens(B: int, min_q: int, max_q: int, min_k: int, max_k: int) -> Tuple[List[int], List[int]]:
    sq = [random.randint(min_q, max_q) for _ in range(B)]
    sk = [random.randint(min_k, max_k) for _ in range(B)]
    return sq, sk


def benchmark_linear(
    B: int,
    Hq: int,
    Hkv: int,
    D: int,
    seqlens_q: List[int],
    seqlens_k: List[int],
    dtype: torch.dtype,
    device: str,
    iters: int,
    warmup: int,
):
    q, k, v, cu_q, cu_k, max_q, max_k = _make_varlen_qkv(
        B, Hq, Hkv, D, seqlens_q, seqlens_k, dtype, device
    )

    # Baseline: return_aux=False
    def _baseline():
        _call_varlen_linear(q, k, v, max_q, cu_q, max_k, cu_k, want_aux=False, want_lse=False)

    # With abs_s (if available): return_aux=True
    def _with_aux():
        _call_varlen_linear(q, k, v, max_q, cu_q, max_k, cu_k, want_aux=True, want_lse=False)

    t_base = _time_cuda_callable(_baseline, warmup, iters)
    t_aux = _time_cuda_callable(_with_aux, warmup, iters)

    # Try one call to inspect extras presence and shape
    outs = _call_varlen_linear(q, k, v, max_q, cu_q, max_k, cu_k, want_aux=True, want_lse=True)
    extras = outs[2:] if isinstance(outs, (tuple, list)) and len(outs) > 2 else []

    return t_base, t_aux, extras


def benchmark_paged(
    B: int,
    Hq: int,
    Hkv: int,
    D: int,
    seqlens_q: List[int],
    seqlens_k: List[int],
    page_block_size: int,
    dtype: torch.dtype,
    device: str,
    iters: int,
    warmup: int,
):
    total_q = sum(seqlens_q)
    q = torch.randn(total_q, Hq, D, device=device, dtype=dtype)
    total_k = sum(seqlens_k)
    k_lin = torch.randn(total_k, Hkv, D, device=device, dtype=dtype)
    v_lin = torch.randn_like(k_lin)

    cu_q = (
        torch.tensor([0] + list(seqlens_q), device=device, dtype=torch.int32)
        .cumsum(0, dtype=torch.int32)
        .contiguous()
    )

    k_paged, v_paged, block_table, seqused_k = _linear_to_paged_kv(
        k_lin, v_lin, seqlens_k, Hkv, D, page_block_size
    )
    max_q = max(seqlens_q)
    max_k = max(seqlens_k)

    def _baseline():
        _call_varlen_paged(q, k_paged, v_paged, max_q, cu_q, max_k, block_table, seqused_k, want_aux=False, want_lse=False)

    def _with_aux():
        _call_varlen_paged(q, k_paged, v_paged, max_q, cu_q, max_k, block_table, seqused_k, want_aux=True, want_lse=False)

    try:
        t_base = _time_cuda_callable(_baseline, warmup, iters)
        t_aux = _time_cuda_callable(_with_aux, warmup, iters)
        outs = _call_varlen_paged(q, k_paged, v_paged, max_q, cu_q, max_k, block_table, seqused_k, want_aux=True, want_lse=True)
        extras = outs[2:] if isinstance(outs, (tuple, list)) and len(outs) > 2 else []
        return t_base, t_aux, extras
    except NotImplementedError as e:
        # Gracefully skip paged path on old builds
        print(f"[Skip] Paged-KV benchmark is not supported by this build: {e}")
        return float('nan'), float('nan'), []


def main():
    parser = argparse.ArgumentParser(description="Benchmark overhead of abs_s (FA2 varlen)")
    parser.add_argument("--mode", choices=["linear", "paged", "both"], default="both",
                        help="Benchmark non-paged (linear) KV, paged KV, or both")
    parser.add_argument("--B", type=int, default=2, help="Batch size (number of sequences)")
    parser.add_argument("--Hq", type=int, default=8, help="Number of query heads")
    parser.add_argument("--Hkv", type=int, default=2, help="Number of KV heads (GQA)")
    parser.add_argument("--D", type=int, default=128, help="Head dimension")
    parser.add_argument("--min_q", type=int, default=256, help="Min seqlen for Q")
    parser.add_argument("--max_q", type=int, default=512, help="Max seqlen for Q")
    parser.add_argument("--min_k", type=int, default=256, help="Min seqlen for K/V")
    parser.add_argument("--max_k", type=int, default=512, help="Max seqlen for K/V")
    parser.add_argument("--page", type=int, default=16, help="Page block size for paged KV (multiple of 16)")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--iters", type=int, default=50, help="Timing iterations")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device")
    # FA version is fixed to FA2 in this benchmark; no flag needed.
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required for this benchmark"

    set_seed(args.seed)
    device = args.device
    dtype = _maybe_to_dtype(args.dtype)
    assert dtype in (torch.float16, torch.bfloat16)

    # Generate variable sequence lengths
    seqlens_q, seqlens_k = _gen_seqlens(args.B, args.min_q, args.max_q, args.min_k, args.max_k)

    print("==== Benchmark Config ====")
    print(f"device={device}, dtype={dtype}, B={args.B}, Hq={args.Hq}, Hkv={args.Hkv}, D={args.D}")
    print(f"seqlens_q={seqlens_q}")
    print(f"seqlens_k={seqlens_k}")
    if args.mode in ("paged", "both"):
        print(f"page_block_size={args.page}")
    print(f"iters={args.iters}, warmup={args.warmup}, fa_version=2 (fixed)")

    def _report(title: str, t_base: float, t_aux: float, extras):
        overhead = (t_aux - t_base) / t_base * 100.0 if t_base > 0 else float("nan")
        print(f"\n[{title}] avg_ms baseline={t_base:.3f} | with_abs_s={t_aux:.3f} | overhead={overhead:.2f}%")
        if extras:
            for i, ex in enumerate(extras):
                if isinstance(ex, torch.Tensor):
                    print(f"  extra[{i}] shape={tuple(ex.shape)} dtype={ex.dtype}")
                else:
                    print(f"  extra[{i}] type={type(ex)}")
        else:
            print("  (No extras returned; abs_s path likely unavailable in this build)")

    if args.mode in ("linear", "both"):
        t0, t1, extras = benchmark_linear(
            args.B, args.Hq, args.Hkv, args.D,
            seqlens_q, seqlens_k, dtype, device,
            args.iters, args.warmup,
        )
        _report("Linear (non-paged) KV", t0, t1, extras)

    if args.mode in ("paged", "both"):
        t0, t1, extras = benchmark_paged(
            args.B, args.Hq, args.Hkv, args.D, seqlens_q, seqlens_k,
            args.page, dtype, device, args.iters, args.warmup,
        )
        _report("Paged KV", t0, t1, extras)


if __name__ == "__main__":
    main()
