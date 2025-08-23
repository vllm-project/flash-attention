import argparse
import math
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen_q", type=int, default=16)
    parser.add_argument("--seqlen_k", type=int, default=4096)
    parser.add_argument("--nheads_q", type=int, default=8)
    parser.add_argument("--nheads_k", type=int, default=2)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    # Shapes
    B = args.batch
    Sq = args.seqlen_q
    Sk = args.seqlen_k
    Hq = args.nheads_q
    Hk = args.nheads_k
    D = args.headdim
    BS = args.block_size

    assert Sk > 0 and Sq > 0 and BS > 0, "sequence lengths and block size must be > 0"
    nblocks_per_seq = math.ceil(Sk / BS)
    num_blocks = B * nblocks_per_seq

    # Allocate paged KV cache and block table
    k_cache_paged = torch.randn(num_blocks, BS, Hk, D, device=device, dtype=dtype)
    v_cache_paged = torch.randn(num_blocks, BS, Hk, D, device=device, dtype=dtype)
    block_table = (
        torch.arange(num_blocks, dtype=torch.int32, device=device)
        .view(B, nblocks_per_seq)
    )

    # Random queries and cache seqlens
    q = torch.randn(B, Sq, Hq, D, device=device, dtype=dtype)
    cache_seqlens = torch.randint(
        low=max(1, Sq // 2), high=Sk + 1, size=(B,), dtype=torch.int32, device=device
    )

    # Call paged KV kernel
    from flash_attn import flash_attn_with_kvcache

    out = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache_paged,
        v_cache=v_cache_paged,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=bool(args.causal),
    )

    print("OK: out shape:", tuple(out.shape))
    print("dtype:", out.dtype, "mean(abs):", out.abs().mean().item())


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemError("CUDA is not available. Please install a CUDA-enabled PyTorch.")
    main()
