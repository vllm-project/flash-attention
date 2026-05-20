"""FA4 fused FP8 output vs. BF16 attn + torch.compile'd post-quant cast.

Requires SM100/SM110 (Blackwell). FA4 ships no FP8 quant op, so the unfused
baseline uses torch.compile to fuse the divide+clamp+cast into one kernel.

    python benchmarks/bench_flash_attention_fp8_output.py [--shape ...] [--rep N]
"""

import argparse
import time

import torch
from triton.testing import do_bench

from flash_attn.cute.bench_utils import flops
from flash_attn.cute.interface import flash_attn_func


# (name, batch, seqlen_q, seqlen_k, num_heads, num_kv_heads, head_dim, head_dim_v, causal)
# Naming convention: <mode>_<attn>_<seqlen>, where:
#   mode = prefill (sq == sk) | decode (sq == 1, sk large)
#   attn = mla (qk=192, v=128) | mha (h_q == h_kv) | gqa (h_q > h_kv)
#   seqlen = the K-side context length
SHAPES = {
    # DeepSeek-V3 MLA prefill — the primary target of this PR.
    "prefill_mla_4k":  (2,  4096, 4096, 16,   1, 192, 128, True),
    # Standard MHA prefill, 4K context.
    "prefill_mha_4k":  (2,  4096, 4096, 32,  32, 128, 128, True),
    # Llama-style GQA prefill (8:1 ratio), 8K context.
    "prefill_gqa_8k":  (2,  8192, 8192, 32,   4, 128, 128, True),
    # GQA decode (sq=1, h=16/1), 8K context — common decode shape.
    "decode_gqa_8k":   (16,    1, 8192, 16,   1, 128, 128, True),
    # MHA decode, 8K context.
    "decode_mha_8k":   (16,    1, 8192, 16,  16, 128, 128, True),
}


def static_fp8_quant_eager(out_bf16: torch.Tensor, inv_scale: float) -> torch.Tensor:
    """Stand-in for vLLM's `static_scaled_fp8_quant`."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    return out_bf16.float().mul(inv_scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)



_static_fp8_quant_compiled = torch.compile(static_fp8_quant_eager, mode="reduce-overhead")


def bench_one(name, shape, warmup, rep):
    batch, sq, sk, nh, nkv, dq, dv, causal = shape
    device = torch.device("cuda")
    dtype = torch.bfloat16

    q = torch.randn(batch, sq, nh, dq, dtype=dtype, device=device)
    k = torch.randn(batch, sk, nkv, dq, dtype=dtype, device=device)
    v = torch.randn(batch, sk, nkv, dv, dtype=dtype, device=device)

    # Pick a representative scale (peak of one BF16 forward).
    ref_out, _ = flash_attn_func(q, k, v, causal=causal)
    finfo = torch.finfo(torch.float8_e4m3fn)
    out_scale = max(float(ref_out.float().abs().amax().item()) / finfo.max, 1e-4)
    inv_scale = 1.0 / out_scale
    out_scale_t = torch.tensor(out_scale, dtype=torch.float32, device=device)

    fp8_buf = torch.empty(batch, sq, nh, dv, dtype=torch.float8_e4m3fn, device=device)

    def fwd_bf16():
        return flash_attn_func(q, k, v, causal=causal)

    def fwd_bf16_then_quant():
        out, _ = flash_attn_func(q, k, v, causal=causal)
        return _static_fp8_quant_compiled(out, inv_scale)

    def fwd_fp8_fused():
        return flash_attn_func(
            q, k, v, causal=causal,
            out=fp8_buf,
            output_scale=out_scale_t,
        )

    time.sleep(1.0)
    ms_bf16 = do_bench(fwd_bf16, warmup=warmup, rep=rep) * 1e-3
    time.sleep(1.0)
    ms_unfused = do_bench(fwd_bf16_then_quant, warmup=warmup, rep=rep) * 1e-3
    time.sleep(1.0)
    ms_fused = do_bench(fwd_fp8_fused, warmup=warmup, rep=rep) * 1e-3

    n_flops = flops(batch, nh, sq, sk, dq, dv, causal=causal)
    def tflops(s): return n_flops / s * 1e-12

    saved = ms_unfused - ms_fused
    speedup = ms_unfused / ms_fused

    print(
        f"{name:<14} b={batch} sq={sq:>5} sk={sk:>5} h={nh:>3}/{nkv:<3} d={dq}-{dv:<3}  "
        f"bf16={ms_bf16*1e6:>7.1f}us/{tflops(ms_bf16):>4.0f}TF  "
        f"bf16+quant={ms_unfused*1e6:>7.1f}us/{tflops(ms_unfused):>4.0f}TF  "
        f"fused-fp8={ms_fused*1e6:>7.1f}us/{tflops(ms_fused):>4.0f}TF  "
        f"saved={saved*1e6:>+6.1f}us ({speedup:.2f}x)"
    )


def main():
    parser = argparse.ArgumentParser(description="FA4 fused FP8 output benchmark")
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
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Warmup={args.warmup}, rep={args.rep}\n")

    for name in shapes:
        torch.cuda.empty_cache()
        bench_one(name, SHAPES[name], args.warmup, args.rep)


if __name__ == "__main__":
    main()
