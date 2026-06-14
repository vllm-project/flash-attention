"""FA4 fused FP8 attention-output quantization vs. BF16 attn + a separate post-quant cast.

Covers both fused-quant-output schemes via ``--quant``:
  - ``static``   : per-tensor static FP8 (``output_scale=``), like vLLM ``static_scaled_fp8_quant``.
  - ``pergroup`` : per-row dynamic FP8 with head_dim_v grouped in 128-wide chunks
                   (``output_scales=``), vLLM-style per-group dynamic quant.

FA4 ships no FP8 quant op, so each unfused baseline uses ``torch.compile`` to fuse the
post-quant amax/divide/clamp/cast into one kernel. Requires SM100/SM110 (Blackwell).

    python benchmarks/benchmark_attn_quant_output_fusion.py [--quant both] [--shape ...] [--rep N]
"""

import argparse
import time
from typing import Tuple

import torch
from triton.testing import do_bench

from flash_attn.cute.bench_utils import flops
from flash_attn.cute.interface import flash_attn_func


# value = (batch, seqlen_q, seqlen_k, num_heads, num_kv_heads, head_dim, head_dim_v, causal, num_splits)
# num_splits > 1 exercises the SplitKV path (combine kernel does the quant).
SHAPES = {
    "prefill_mla_4k":     (2,  4096, 4096, 16,   1, 192, 128, True, 1),  # DeepSeek-V3 MLA prefill
    "prefill_mha_4k":     (2,  4096, 4096, 32,  32, 128, 128, True, 1),  # standard MHA prefill
    "prefill_gqa_8k":     (2,  8192, 8192, 32,   4, 128, 128, True, 1),  # Llama-style GQA (8:1)
    "decode_gqa_8k":      (16,    1, 8192, 16,   1, 128, 128, True, 1),  # GQA decode
    "decode_mha_8k":      (16,    1, 8192, 16,  16, 128, 128, True, 1),  # MHA decode
    "decode_gqa_8k_split": (16,   1, 8192, 16,   1, 128, 128, True, 8),  # GQA decode, SplitKV (combine quant)
    "decode_mha_8k_split": (16,   1, 8192, 16,  16, 128, 128, True, 8),  # MHA decode, SplitKV (combine quant)
}

GROUP_SIZE = 128


def _static_fp8_quant_eager(out_bf16: torch.Tensor, inv_scale: float) -> torch.Tensor:
    """Stand-in for vLLM's ``static_scaled_fp8_quant``."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    return out_bf16.float().mul(inv_scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)


def _per_group_fp8_quant_eager(
    out_bf16: torch.Tensor, group_size: int = GROUP_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stand-in for a vLLM-style per-group ``per_token_group_quant_fp8`` op."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    f8_max = float(finfo.max)
    x = out_bf16.float()
    head_dim = x.shape[-1]
    x_grp = x.unflatten(-1, (head_dim // group_size, group_size))
    amax = x_grp.abs().amax(dim=-1)
    safe_amax = torch.where(amax == 0, torch.ones_like(amax), amax)
    dequant_scale = safe_amax / f8_max
    fp8 = (x_grp / dequant_scale.unsqueeze(-1)).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
    dequant_scale = torch.where(amax == 0, torch.zeros_like(dequant_scale), dequant_scale)
    return fp8.flatten(-2), dequant_scale


_static_fp8_quant_compiled = torch.compile(_static_fp8_quant_eager, mode="reduce-overhead")
_per_group_fp8_quant_compiled = torch.compile(_per_group_fp8_quant_eager, mode="reduce-overhead")


def _setup_static(q, k, v, causal, num_splits, batch, sq, nh, dv, device):
    """Return (unfused_baseline_fn, fused_fn) for the per-tensor static FP8 path."""
    # Pick a representative scale (peak of one BF16 forward).
    ref_out, _ = flash_attn_func(q, k, v, causal=causal)
    finfo = torch.finfo(torch.float8_e4m3fn)
    out_scale = max(float(ref_out.float().abs().amax().item()) / finfo.max, 1e-4)
    inv_scale = 1.0 / out_scale
    out_scale_t = torch.tensor(out_scale, dtype=torch.float32, device=device)
    fp8_buf = torch.empty(batch, sq, nh, dv, dtype=torch.float8_e4m3fn, device=device)

    def unfused():
        out, _ = flash_attn_func(q, k, v, causal=causal, num_splits=num_splits)
        return _static_fp8_quant_compiled(out, inv_scale)

    def fused():
        return flash_attn_func(
            q, k, v, causal=causal, num_splits=num_splits, out=fp8_buf, output_scale=out_scale_t
        )

    return unfused, fused


def _colmajor_tma_scales(batch, sq, nh, num_groups, device):
    """Column-major / TMA-aligned scales buffer (DeepGEMM layout, what real DeepSeek o_proj
    consumes): token dim contiguous. Logical shape (batch, sq, nh, num_groups)."""
    m = batch * sq
    tma_aligned_m = ((m + 3) // 4) * 4
    buf2d = torch.empty_strided(
        (m, nh * num_groups), (1, tma_aligned_m), dtype=torch.float32, device=device
    )
    return buf2d.unflatten(-1, (nh, num_groups)).unflatten(0, (batch, sq))


def _setup_pergroup(q, k, v, causal, num_splits, batch, sq, nh, dv, device, ue8m0=False):
    """Return (unfused_baseline_fn, fused_fn) for the per-group dynamic FP8 path. ``ue8m0`` uses
    the column-major / TMA-aligned scales layout (DeepGEMM), else plain row-major fp32."""
    fp8_buf = torch.empty(batch, sq, nh, dv, dtype=torch.float8_e4m3fn, device=device)
    if ue8m0:
        scales_buf = _colmajor_tma_scales(batch, sq, nh, dv // GROUP_SIZE, device)
    else:
        scales_buf = torch.empty(batch, sq, nh, dv // GROUP_SIZE, dtype=torch.float32, device=device)

    def unfused():
        out, _ = flash_attn_func(q, k, v, causal=causal, num_splits=num_splits)
        return _per_group_fp8_quant_compiled(out, GROUP_SIZE)

    def fused():
        return flash_attn_func(
            q, k, v, causal=causal, num_splits=num_splits, out=fp8_buf, output_scales=scales_buf
        )

    return unfused, fused


def _setup_pergroup_ue8m0(*args, **kwargs):
    return _setup_pergroup(*args, ue8m0=True, **kwargs)


QUANT_MODES = {
    "static":         {"setup": _setup_static,         "label": "fused-fp8"},
    "pergroup":       {"setup": _setup_pergroup,       "label": "fused-pg-fp8"},
    "pergroup-ue8m0": {"setup": _setup_pergroup_ue8m0, "label": "fused-pg-ue8m0"},
}


def bench_one(name, shape, quant, warmup, rep):
    batch, sq, sk, nh, nkv, dq, dv, causal, num_splits = shape
    device = torch.device("cuda")
    dtype = torch.bfloat16

    q = torch.randn(batch, sq, nh, dq, dtype=dtype, device=device)
    k = torch.randn(batch, sk, nkv, dq, dtype=dtype, device=device)
    v = torch.randn(batch, sk, nkv, dv, dtype=dtype, device=device)

    def fwd_bf16():
        return flash_attn_func(q, k, v, causal=causal, num_splits=num_splits)

    unfused, fused = QUANT_MODES[quant]["setup"](q, k, v, causal, num_splits, batch, sq, nh, dv, device)

    time.sleep(1.0)
    ms_bf16 = do_bench(fwd_bf16, warmup=warmup, rep=rep) * 1e-3
    time.sleep(1.0)
    ms_unfused = do_bench(unfused, warmup=warmup, rep=rep) * 1e-3
    time.sleep(1.0)
    ms_fused = do_bench(fused, warmup=warmup, rep=rep) * 1e-3

    n_flops = flops(batch, nh, sq, sk, dq, dv, causal=causal)
    def tflops(s): return n_flops / s * 1e-12

    # Two baselines: the realistic alternative (bf16 attn + separate quant kernel) and
    # plain bf16 attention with no quant (lower bound). vs-bf16 ~1.0x => the fused quant
    # is essentially free on top of attention.
    speedup_vs_quant = ms_unfused / ms_fused
    speedup_vs_bf16 = ms_bf16 / ms_fused
    fused_label = QUANT_MODES[quant]["label"]

    print(
        f"[{quant:<8}] {name:<20} b={batch} sq={sq:>5} sk={sk:>5} h={nh:>3}/{nkv:<3} d={dq}-{dv:<3} ns={num_splits} "
        f"bf16={ms_bf16*1e6:>7.1f}us/{tflops(ms_bf16):>4.0f}TF  "
        f"bf16+quant={ms_unfused*1e6:>7.1f}us/{tflops(ms_unfused):>4.0f}TF  "
        f"{fused_label}={ms_fused*1e6:>7.1f}us/{tflops(ms_fused):>4.0f}TF  "
        f"vs-bf16+quant={speedup_vs_quant:.2f}x  vs-bf16={speedup_vs_bf16:.2f}x"
    )


def main():
    parser = argparse.ArgumentParser(description="FA4 fused FP8 attention-output quant benchmark")
    parser.add_argument("--quant", choices=list(QUANT_MODES) + ["both"], default="both",
                        help="Quant scheme to benchmark. Default: both.")
    parser.add_argument("--shape", action="append", choices=list(SHAPES) + ["all"],
                        default=None, help="Shape preset to run (repeatable). Default: all.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=10)
    args = parser.parse_args()

    cap = torch.cuda.get_device_capability()
    if cap[0] not in (10, 11):
        raise SystemExit(
            f"Fused FP8 output requires SM100/SM110 (Blackwell). "
            f"Detected sm{cap[0]}{cap[1]}; aborting."
        )

    shapes = list(SHAPES) if not args.shape or "all" in args.shape else args.shape
    quants = list(QUANT_MODES) if args.quant == "both" else [args.quant]
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Warmup={args.warmup}, rep={args.rep}\n")

    for quant in quants:
        for name in shapes:
            torch.cuda.empty_cache()
            bench_one(name, SHAPES[name], quant, args.warmup, args.rep)
        print()


if __name__ == "__main__":
    main()
