"""
Single-file FlashAttention2 with attention sinks benchmarks.

Usage:
    python3 benchmark_fa2_sinks.py

Parameters:
    so_path  - path to the shared library (.so)
    path_new - path to save results
"""
import math
import torch
from einops import rearrange
import torch.utils.benchmark as benchmark
import csv
import vllm
import os

pkg = os.path.dirname(vllm.__file__)
so_path = os.path.join(pkg, "vllm_flash_attn", "_vllm_fa2_C.abi3.so")
path_new = os.path.join(".", "benchmark_fa2_sinks.csv")

csv_rows = []

def benchmark_forward(
    fn, inputs, repeats=10, desc="", verbose=False, amp=False, amp_dtype=torch.float16, **kwinputs
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs)

    t = benchmark.Timer(
        stmt="fn_amp(inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m

def benchmark_fwd_bwd(
    fn,
    inputs,
    grad=None,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the forward+backward pass of an arbitrary function."""
    return benchmark_forward(
            fn,
            inputs,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        )

attention_triton = None
xops = None

def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    if causal:
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    return output.to(dtype=qkv.dtype)


def time_fwd_bwd(func, *args, **kwargs):
    time_f = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean


repeats = 30
device = 'cuda'
dtype = torch.float16

bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
causal_vals = [False, True]
headdim_vals = [64, 128]
dim = 2048
dropout_p = 0.0

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}
def test_time(path, func_name, old_or_new):
    for causal in causal_vals:
        for headdim in headdim_vals:
            for batch_size, seqlen in bs_seqlen_vals:
                config = (causal, headdim, batch_size, seqlen)
                nheads = dim // headdim

                qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                                requires_grad=True)
                q = qkv[:, :, 0]                           # (B, S, H, D)
                k = qkv[:, :, 1]                           # (B, S, H, D)
                v = qkv[:, :, 2]                           # (B, S, H, D)

                q = q.reshape(-1, nheads, headdim)
                k = k.reshape(-1, nheads, headdim)
                v = v.reshape(-1, nheads, headdim)
                s_aux = torch.randn(nheads,device=device, dtype=dtype,requires_grad=True)
                out_buf = torch.empty_like(q)
                fa2_fwd_closure = [q, k, v,
                        out_buf,
                        torch.tensor([(seqlen)*i for i in range(batch_size+1)], device=device, dtype = torch.int32),
                        torch.tensor([(seqlen)*i for i in range(batch_size+1)], device=device, dtype = torch.int32),
                        None,
                        None,
                        None,
                        None,
                        seqlen,
                        seqlen,
                        dropout_p,
                        torch.tensor(1.0 / (headdim ** 0.5), device=device),
                        False,
                        causal,
                        -1,
                        -1,
                        0.0,
                        dropout_p > 0,
                        None,
                        s_aux
                    ]
                f = time_fwd_bwd(func_name, fa2_fwd_closure, repeats=repeats, verbose=False)
                time_f[config, "Flash2"] = f
                print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")
                speed_f[config, "Flash2"] = efficiency(
                flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
                time_f[config, "Flash2"])
                print(
                    f"{"Flash2"} fwd: {speed_f[config, "Flash2"]:.2f} TFLOPs/s, "
                )
                csv_rows.append([causal, headdim, batch_size, seqlen, f"{speed_f[config, "Flash2"]:.2f}"])
    with open(path, "a", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["causal", "headdim", "batch_size", "seqlen", "TFLOPs/s"])
        writer.writerows(csv_rows)

    print(f"已写入{path}")

torch.ops.load_library(so_path)
func_name = torch.ops._vllm_fa2_C.varlen_fwd
test_time(path_new, func_name, "new")
