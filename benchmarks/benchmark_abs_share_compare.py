import argparse
import math
import time
from typing import List

import torch

try:
    from flash_attn.utils.cpu_block_attn import one_pass_abs_share
except Exception:
    one_pass_abs_share = None  # type: ignore

try:
    # vLLM style FA2 interface (already used in other benchmark script)
    from vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
except Exception:
    flash_attn_varlen_func = None  # type: ignore


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark abs_s overhead: baseline QK^T vs FA2 kernel return_aux vs CPU reconstruction")
    p.add_argument('--d', type=int, default=128, help='Head dimension')
    p.add_argument('--N', type=int, nargs='+', default=[512, 1024, 2048], help='Sequence lengths')
    p.add_argument('--dtype', type=str, default='float16', choices=['float16','bfloat16','float32'], help='GPU dtype')
    p.add_argument('--causal', action='store_true', help='Apply causal mask (j>i zeroed)')
    p.add_argument('--iters', type=int, default=100, help='Timing iterations')
    p.add_argument('--warmup', type=int, default=20, help='Warmup iterations')
    p.add_argument('--device', type=str, default='cuda', help='GPU device')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--compare-values', action='store_true', help='Compare kernel abs_s (row |S| sums) with CPU reference sums')
    p.add_argument('--no-cpu', action='store_true', help='Skip CPU abs share path')
    p.add_argument('--no-gpu', action='store_true', help='Skip GPU abs share path')
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()


def format_time(t):
    if t < 1e-6:
        return f"{t*1e9:.2f} ns"
    if t < 1e-3:
        return f"{t*1e6:.2f} us"
    if t < 1:
        return f"{t*1e3:.2f} ms"
    return f"{t:.2f} s"


def _call_kernel_abs_s(q: torch.Tensor, k: torch.Tensor, causal: bool, want_aux: bool):
    """Call FA2 varlen kernel in the simplest (single sequence) setting.

    We simulate varlen with B=1 sequence of length N.
    Expected returns:
      - Without aux: (out, lse)
      - With aux (if supported): (out, lse, abs_s, ...)
    abs_s (if present) expected shape (Hq, total_q) or compatible.
    """
    if flash_attn_varlen_func is None:
        raise RuntimeError("flash_attn_varlen_func not available")
    N, Hq, D = q.shape  # (N, Hq, D)
    # Flatten batch/time for varlen format: total_q = N
    q_lin = q.reshape(N * Hq, D)  # not actually correct format (expected (total_q, Hq, D)); replicate existing usage pattern
    # Actually per earlier benchmark, layout is (total_q, Hq, D). We'll allocate accordingly:
    q_lin = q  # treat input already shaped (N, Hq, D)
    k_lin = k  # (N, Hkv, D) but we'll assume Hkv == Hq for simplicity here.
    v_lin = k_lin
    cu = torch.tensor([0, N], device=q.device, dtype=torch.int32)
    max_seqlen = N
    try:
        return flash_attn_varlen_func(
            q=q_lin.reshape(N, Hq, D),
            k=k_lin.reshape(N, Hq, D),
            v=v_lin.reshape(N, Hq, D),
            max_seqlen_q=max_seqlen,
            cu_seqlens_q=cu,
            max_seqlen_k=max_seqlen,
            cu_seqlens_k=cu,
            causal=causal,
            softcap=0.0,
            return_aux=want_aux,
            fa_version=2,
            return_softmax_lse=True,
        )
    except TypeError as e:
        # Fallback removal of fa_version / return_aux like other benchmark
        kwargs = dict(
            q=q_lin.reshape(N, Hq, D),
            k=k_lin.reshape(N, Hq, D),
            v=v_lin.reshape(N, Hq, D),
            max_seqlen_q=max_seqlen,
            cu_seqlens_q=cu,
            max_seqlen_k=max_seqlen,
            cu_seqlens_k=cu,
            causal=causal,
            softcap=0.0,
            return_softmax_lse=True,
        )
        msg = str(e)
        if "return_aux" not in msg:
            kwargs["return_aux"] = want_aux
        if "fa_version" not in msg:
            kwargs["fa_version"] = 2
        try:
            return flash_attn_varlen_func(**{k:v for k,v in kwargs.items() if k not in ("return_aux","fa_version")})
        except TypeError:
            kwargs.pop("return_softmax_lse", None)
            return flash_attn_varlen_func(**{k:v for k,v in kwargs.items() if k not in ("return_aux","fa_version")})


def baseline_matmul(q: torch.Tensor, k: torch.Tensor, causal: bool):
    # Baseline cost approximated by calling kernel without aux and ignoring output values.
    outs = _call_kernel_abs_s(q, k, causal, want_aux=False)
    return outs[0]


def time_fn(fn, *args, iters: int, warmup: int):
    # GPU sync aware timer.
    for _ in range(warmup):
        out = fn(*args)
        if torch.is_tensor(out) and out.is_cuda:
            torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    if torch.is_tensor(out) and out.is_cuda:
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / iters


def to_dtype(t: torch.Tensor, dtype: torch.dtype):
    return t.to(dtype=dtype)


def estimate_flops(N: int, d: int):
    # Matmul Q(N,d) x K^T(d,N): 2*N*N*d (mul+add). + abs (N^2) + reduce+div ~ 3*N^2.
    # We separate baseline vs abs share extra.
    matmul_flops = 2 * N * N * d
    abs_norm_flops = 3 * N * N  # rough
    return matmul_flops, abs_norm_flops


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.no_gpu:
        torch.cuda.set_device(args.device)
    else:
        if not torch.cuda.is_available() and not args.no_gpu:
            print("[WARN] CUDA 不可用，GPU 路径将被跳过")

    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    gpu_dtype = dtype_map[args.dtype]

    header = f"{'N':>6} | {'Baseline(ms)':>12} | {'Kernel_abs(ms)':>14} | {'+%':>6} | {'CPU_abs(ms)':>11} | {'+%':>6} | {'Matmul GFLOPs':>13} | {'Abs Extra GFLOPs':>16}"
    print(header)
    print('-'*len(header))

    for N in args.N:
        d = args.d
        H = 1
        Q_cpu = torch.randn(N, H, d, dtype=torch.float32)
        K_cpu = torch.randn(N, H, d, dtype=torch.float32)

        if torch.cuda.is_available() and not args.no_gpu:
            Q_gpu = to_dtype(Q_cpu.cuda(non_blocking=True), gpu_dtype)
            K_gpu = to_dtype(K_cpu.cuda(non_blocking=True), gpu_dtype)
        else:
            Q_gpu = K_gpu = None

        if Q_gpu is not None and flash_attn_varlen_func is not None and not args.no_gpu:
            baseline_t = time_fn(lambda a,b: _call_kernel_abs_s(a,b,args.causal, want_aux=False), Q_gpu, K_gpu, iters=args.iters, warmup=args.warmup)
        else:
            baseline_t = float('nan')

        if Q_gpu is not None and flash_attn_varlen_func is not None and not args.no_gpu:
            gpu_abs_t = time_fn(lambda a,b: _call_kernel_abs_s(a,b,args.causal, want_aux=True), Q_gpu, K_gpu, iters=args.iters, warmup=args.warmup)
        else:
            gpu_abs_t = float('nan')

        if not args.no_cpu and one_pass_abs_share is not None:
            cpu_abs_t = time_fn(lambda a,b: one_pass_abs_share(a.squeeze(1), b.squeeze(1), causal=args.causal), Q_cpu, K_cpu, iters=max(5, args.iters//10), warmup=max(2, args.warmup//10))
        else:
            cpu_abs_t = float('nan')

        gpu_overhead = (gpu_abs_t / baseline_t - 1.0)*100 if (not math.isnan(gpu_abs_t) and not math.isnan(baseline_t)) else float('nan')
        cpu_overhead = (cpu_abs_t*1e3 / (baseline_t*1e3) - 1.0)*100 if (not math.isnan(cpu_abs_t) and not math.isnan(baseline_t)) else float('nan')

        matmul_flops, abs_extra_flops = estimate_flops(N, d)
        gflops = matmul_flops / 1e9
        abs_extra_gflops = abs_extra_flops / 1e9

        print(f"{N:6d} | {baseline_t*1e3:12.3f} | {gpu_abs_t*1e3:14.3f} | {gpu_overhead:6.1f} | {cpu_abs_t*1e3:11.3f} | {cpu_overhead:6.1f} | {gflops:13.2f} | {abs_extra_gflops:16.3f}")

        if args.compare_values and Q_gpu is not None and one_pass_abs_share is not None and not args.no_gpu and not args.no_cpu and flash_attn_varlen_func is not None:
            try:
                outs = _call_kernel_abs_s(Q_gpu, K_gpu, args.causal, want_aux=True)
                extras = outs[2:] if isinstance(outs,(tuple,list)) else []
                abs_s = None
                for ex in extras:
                    if isinstance(ex, torch.Tensor):
                        abs_s = ex
                        break
                if abs_s is None:
                    print("    (No abs_s returned by kernel)")
                else:
                    Qf = Q_cpu.squeeze(1)
                    Kf = K_cpu.squeeze(1)
                    S_cpu = Qf @ Kf.T
                    if args.causal:
                        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
                        S_cpu = S_cpu.masked_fill(mask, 0.0)
                    ref_sums = S_cpu.abs().sum(dim=-1)
                    abs_s_view = abs_s.reshape(-1, N)[0].float().cpu()
                    max_diff = (abs_s_view - ref_sums).abs().max().item()
                    rel_err = max_diff / (ref_sums.abs().max().item() + 1e-9)
                    print(f"    abs_s row-sum max_diff={max_diff:.3e} rel_err={rel_err:.3e}")
            except Exception as e:
                print(f"    (compare failed: {e})")

    print("Done.")


if __name__ == '__main__':
    main()
