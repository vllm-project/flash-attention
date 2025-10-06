import argparse
import math
import time
from typing import List

import torch
import importlib.util
import os

# Load CPU helpers (module file directly), whether or not certain symbols exist.
def _load_cpu_helpers():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, '..'))  # benchmarks/.. -> repo root (approx)
    candidate = os.path.join(repo_root, 'flash_attn', 'utils', 'cpu_block_attn.py')
    if not os.path.isfile(candidate):
        return None, None, None
    spec = importlib.util.spec_from_file_location("_cpu_block_attn_mod", candidate)
    if spec is None or spec.loader is None:
        return None, None, None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore
    except Exception:
        return None, None, None
    one_pass = getattr(mod, 'one_pass_abs_share', None)
    kernel_abs = getattr(mod, 'kernel_abs_s_cpu', None)
    blockwise = getattr(mod, 'blockwise_softmax_block_share', None)
    return one_pass, kernel_abs, blockwise

one_pass_abs_share, kernel_abs_s_cpu, blockwise_softmax_block_share = _load_cpu_helpers()

try:
    # vLLM style FA2 interface (already used in other benchmark script)
    from vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
except Exception:
    flash_attn_varlen_func = None  # type: ignore


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark abs_s overhead: baseline vs FA2 (vllm_flash_attn) return_aux vs optional CPU reconstruction")
    p.add_argument('--d', type=int, default=128, help='Head dimension')
    p.add_argument('--N', type=int, nargs='+', default=[512, 1024, 2048], help='Sequence lengths')
    p.add_argument('--dtype', type=str, default='float16', choices=['float16','bfloat16','float32'], help='GPU dtype')
    p.add_argument('--causal', action='store_true', help='Apply causal mask (j>i zeroed)')
    p.add_argument('--iters', type=int, default=100, help='Timing iterations')
    p.add_argument('--warmup', type=int, default=20, help='Warmup iterations')
    p.add_argument('--device', type=str, default='cuda', help='Device: cuda | cuda:IDX | IDX | cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--compare-values', action='store_true', help='Compare kernel abs_s vs CPU reference (1/sqrt(D), no causal); uses inline CPU ref if utils export is absent')
    p.add_argument('--no-cpu', action='store_true', help='Skip CPU abs share path (or forced if cpu_block_attn not found)')
    p.add_argument('--no-gpu', action='store_true', help='Skip GPU abs share path')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--block-size', type=int, default=0, help='Block size for CPU blockwise softmax mass W; if 0, a default will be chosen')
    p.add_argument('--report-block', action='store_true', help='Print diagnostics for blockwise W (row sum ~1, sample)')
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


def _cpu_abs_s_inline(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Inline CPU reference for abs_s (scale 1/sqrt(D), no causal).

    Accepts q, k as (N, D) or (N, 1, D). Returns (1, N).
    Only used for --compare-values when utils do not export kernel_abs_s_cpu.
    """
    if q.dim() == 3:
        N, H, D = q.shape
        assert H == 1, "inline abs_s expects H=1"
        q2 = q[:, 0, :].contiguous()
    else:
        N, D = q.shape
        q2 = q.contiguous()
    if k.dim() == 3:
        Nk, Hk, Dk = k.shape
        assert Hk == 1 and Nk == N and Dk == D
        k2 = k[:, 0, :].contiguous()
    else:
        Nk, Dk = k.shape
        assert Nk == N and Dk == D
        k2 = k.contiguous()
    scale = (1.0 / (D ** 0.5))
    S = (q2 @ k2.T) * scale  # (N, N)
    return S.abs().sum(dim=-1, keepdim=True).transpose(0, 1)  # (1, N)


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


def _parse_device(dev_str: str):
    s = dev_str.strip().lower()
    if s == 'cpu':
        return 'cpu'
    if s == 'cuda':
        return 'cuda:0'
    if s.startswith('cuda:'):
        return s
    # pure integer -> cuda:index
    if s.isdigit():
        return f'cuda:{s}'
    return s  # fallback as-is

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    parsed_device = _parse_device(args.device)
    use_gpu = torch.cuda.is_available() and not args.no_gpu and parsed_device != 'cpu'
    if use_gpu:
        try:
            torch.cuda.set_device(parsed_device)
        except Exception as e:
            print(f"[WARN] 无法设置 GPU 设备 '{parsed_device}': {e}. 将跳过 GPU 基准。")
            use_gpu = False
    else:
        if not torch.cuda.is_available() and not args.no_gpu:
            print("[WARN] CUDA 不可用，GPU 路径将被跳过")

    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    gpu_dtype = dtype_map[args.dtype]

    header = f"{'N':>6} | {'Baseline(ms)':>12} | {'Kernel_abs(ms)':>14} | {'+%':>6} | {'CPU_block(ms)':>13} | {'+%':>6} | {'Matmul GFLOPs':>13} | {'Abs Extra GFLOPs':>16}"
    print(header)
    print('-'*len(header))

    for N in args.N:
        d = args.d
        H = 1
        Q_cpu = torch.randn(N, H, d, dtype=torch.float32)
        K_cpu = torch.randn(N, H, d, dtype=torch.float32)

        if use_gpu:
            Q_gpu = to_dtype(Q_cpu.cuda(non_blocking=True), gpu_dtype)
            K_gpu = to_dtype(K_cpu.cuda(non_blocking=True), gpu_dtype)
        else:
            Q_gpu = K_gpu = None

        if Q_gpu is not None and flash_attn_varlen_func is not None and use_gpu:
            baseline_t = time_fn(lambda a,b: _call_kernel_abs_s(a,b,args.causal, want_aux=False), Q_gpu, K_gpu, iters=args.iters, warmup=args.warmup)
        else:
            baseline_t = float('nan')

        if Q_gpu is not None and flash_attn_varlen_func is not None and use_gpu:
            gpu_abs_t = time_fn(lambda a,b: _call_kernel_abs_s(a,b,args.causal, want_aux=True), Q_gpu, K_gpu, iters=args.iters, warmup=args.warmup)
        else:
            gpu_abs_t = float('nan')

        # CPU blockwise timing (uses blockwise_softmax_block_share if available)
        if not args.no_cpu and blockwise_softmax_block_share is not None:
            bs_used = args.block_size if (args.block_size and args.block_size > 0) else min(128, N)
            cpu_abs_t = time_fn(
                lambda a,b: blockwise_softmax_block_share(a.squeeze(1), b.squeeze(1), bs_used, causal=args.causal),
                Q_cpu, K_cpu,
                iters=max(5, args.iters//10), warmup=max(2, args.warmup//10)
            )
        else:
            cpu_abs_t = float('nan')

        gpu_overhead = (gpu_abs_t / baseline_t - 1.0)*100 if (not math.isnan(gpu_abs_t) and not math.isnan(baseline_t)) else float('nan')
        cpu_overhead = (cpu_abs_t*1e3 / (baseline_t*1e3) - 1.0)*100 if (not math.isnan(cpu_abs_t) and not math.isnan(baseline_t)) else float('nan')

        matmul_flops, abs_extra_flops = estimate_flops(N, d)
        gflops = matmul_flops / 1e9
        abs_extra_gflops = abs_extra_flops / 1e9

        print(f"{N:6d} | {baseline_t*1e3:12.3f} | {gpu_abs_t*1e3:14.3f} | {gpu_overhead:6.1f} | {cpu_abs_t*1e3:11.3f} | {cpu_overhead:6.1f} | {gflops:13.2f} | {abs_extra_gflops:16.3f}")

        # Diagnostics for blockwise W
        if args.report_block and blockwise_softmax_block_share is not None and not math.isnan(cpu_abs_t):
            try:
                bs_used = args.block_size if (args.block_size and args.block_size > 0) else min(128, N)
                Wblk = blockwise_softmax_block_share(Q_cpu.squeeze(1), K_cpu.squeeze(1), bs_used, causal=args.causal)
                rowsum = Wblk.sum(dim=1)
                print(f"    blockW rowsum min={rowsum.min().item():.6f} max={rowsum.max().item():.6f} mean={rowsum.mean().item():.6f} Tc={Wblk.shape[1]}")
                if args.verbose:
                    i = min(3, Wblk.shape[0]-1)
                    print(f"    sample W[i,:5]={Wblk[i,:min(5,Wblk.shape[1])].tolist()}")
            except Exception as e:
                print(f"    [block] failed: {e}")

        # -------- value compare (per N) --------
        if args.compare_values:
            if not use_gpu:
                print("    [compare] skip: GPU disabled or unavailable")
                continue
            if flash_attn_varlen_func is None:
                print("    [compare] skip: flash_attn_varlen_func not imported")
                continue
            try:
                outs = _call_kernel_abs_s(Q_gpu, K_gpu, args.causal, want_aux=True)
            except Exception as e:
                print(f"    [compare] kernel call failed: {e}")
                continue
            extras = outs[2:] if isinstance(outs,(tuple,list)) else []
            abs_s = None
            for ex in extras:
                if isinstance(ex, torch.Tensor):
                    abs_s = ex
                    break
            if abs_s is None:
                print("    [compare] skip: kernel returned no aux tensor (abs_s)")
                continue
            # CPU reference (kernel semantics: scale 1/sqrt(d), no causal)
            Qf = Q_cpu.float()
            Kf = K_cpu.float()
            if kernel_abs_s_cpu is not None:
                ref = kernel_abs_s_cpu(Qf, Kf)[0].cpu()  # H=1
            else:
                ref = _cpu_abs_s_inline(Qf.squeeze(1), Kf.squeeze(1))[0].cpu()
            abs_s_view = abs_s.reshape(-1, N)[0].float().cpu()
            max_diff = (abs_s_view - ref).abs().max().item()
            rel_err = max_diff / (ref.abs().max().item() + 1e-9)
            print(f"    abs_s max_diff={max_diff:.3e} rel_err={rel_err:.3e} (kernel_vs_cpu)")
            if args.verbose:
                print(f"    shapes: abs_s={tuple(abs_s.shape)} ref={tuple(ref.shape)}")

    print("Done.")


if __name__ == '__main__':
    main()
