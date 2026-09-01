"""Benchmark the SM103 FP8 d192/v128 packed-varlen causal profile.

Usage:
    python benchmarks/benchmark_sm103_fp8_d192_v128.py
    python benchmarks/benchmark_sm103_fp8_d192_v128.py --seed 17 \
        --q-lengths 4096,6144,8192,12288 \
        --kv-lengths 12288,14336,18432,20480

The optimized T8 profile is selected automatically. No tuning environment
variables or private API arguments are required. Pass
``--fp8-rescale-threshold`` to benchmark the T0, T0.75, or T8 precision profile.
"""

from __future__ import annotations

import argparse
import math
import sys
import types

import torch

# The CuTe implementation does not require the optional FA2 extension.
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))

from flash_attn.cute import flash_attn_varlen_func

DEFAULT_Q_LENGTHS = (5090, 8336, 10196, 9146)
DEFAULT_KV_LENGTHS = (17269, 16990, 15900, 15377)
NUM_HEADS = 12
QK_DIM = 192
VALUE_DIM = 128


def _lengths(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def _cumulative(lengths: tuple[int, ...], device: torch.device) -> torch.Tensor:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, dtype=torch.int32, device=device)


def _causal_flops(q_lengths: tuple[int, ...], kv_lengths: tuple[int, ...]) -> int:
    twice_pairs = sum(
        q_len * (2 * kv_len - q_len)
        for q_len, kv_len in zip(q_lengths, kv_lengths, strict=True)
    )
    return NUM_HEADS * twice_pairs * (QK_DIM + VALUE_DIM)


def _replay_for_seconds(
    graph: torch.cuda.CUDAGraph, seconds: float, chunk_replays: int
) -> tuple[float, int]:
    elapsed_ms = 0.0
    replays = 0
    while elapsed_ms < seconds * 1000:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(chunk_replays):
            graph.replay()
        end.record()
        end.synchronize()
        elapsed_ms += start.elapsed_time(end)
        replays += chunk_replays
    return elapsed_ms, replays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-lengths", type=_lengths, default=DEFAULT_Q_LENGTHS)
    parser.add_argument("--kv-lengths", type=_lengths, default=DEFAULT_KV_LENGTHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--measure-seconds", type=float, default=3.0)
    parser.add_argument("--chunk-replays", type=int, default=200)
    parser.add_argument(
        "--fp8-rescale-threshold",
        type=float,
        choices=(0.0, 0.75, 8.0),
        default=None,
        help="FP8 rescale threshold; defaults to the production T8 profile",
    )
    args = parser.parse_args()

    q_lengths = args.q_lengths
    kv_lengths = args.kv_lengths
    if len(q_lengths) != len(kv_lengths) or not q_lengths:
        raise ValueError("Q and KV lengths must have the same nonzero batch size")
    if any(
        q_len <= 0 or q_len > kv_len
        for q_len, kv_len in zip(q_lengths, kv_lengths, strict=True)
    ):
        raise ValueError("each request must satisfy 0 < q_len <= kv_len")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("this benchmark requires an SM103 GPU")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    q = torch.randn(
        sum(q_lengths), NUM_HEADS, QK_DIM, dtype=torch.bfloat16, device=device
    ).to(torch.float8_e4m3fn)
    k = torch.randn(
        sum(kv_lengths), NUM_HEADS, QK_DIM, dtype=torch.bfloat16, device=device
    ).to(torch.float8_e4m3fn)
    v = torch.randn(
        sum(kv_lengths), NUM_HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device
    ).to(torch.float8_e4m3fn)
    out = torch.empty(
        sum(q_lengths), NUM_HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device
    )
    cu_q = _cumulative(q_lengths, device)
    cu_k = _cumulative(kv_lengths, device)

    def run() -> None:
        flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(q_lengths),
            max_seqlen_k=max(kv_lengths),
            softmax_scale=1.0 / math.sqrt(QK_DIM),
            causal=True,
            num_splits=1,
            out=out,
            fp8_rescale_threshold=args.fp8_rescale_threshold,
        )

    side_stream = torch.cuda.Stream()
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            run()
    side_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    warmup_ms, warmup_replays = _replay_for_seconds(
        graph, args.warmup_seconds, args.chunk_replays
    )
    elapsed_ms, replays = _replay_for_seconds(
        graph, args.measure_seconds, args.chunk_replays
    )
    milliseconds = elapsed_ms / replays
    tflops = _causal_flops(q_lengths, kv_lengths) / milliseconds / 1e9

    print(
        f"shape q_lengths={q_lengths} kv_lengths={kv_lengths} "
        f"heads={NUM_HEADS} qk_dim={QK_DIM} value_dim={VALUE_DIM} seed={args.seed} "
        "fp8_rescale_threshold="
        f"{8.0 if args.fp8_rescale_threshold is None else args.fp8_rescale_threshold}"
    )
    print(
        f"warmup_gpu_ms={warmup_ms:.3f} warmup_replays={warmup_replays} "
        f"measurement_gpu_ms={elapsed_ms:.3f} measurement_replays={replays}"
    )
    print(f"result ms={milliseconds:.6f} tflops={tflops:.3f}")


if __name__ == "__main__":
    main()
