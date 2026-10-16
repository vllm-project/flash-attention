"""SM75 numerics check: FA2 varlen+paged-KV vs. pure-PyTorch fp32 reference.

Covers the vLLM decode/spec-verify geometry (GQA, hdim 128, causal,
bottom-right aligned mask, q_len 1..8, long contexts). Run on a Turing GPU:

    CUDA_VISIBLE_DEVICES=<rtx8000> python tests/sm75_paged_numerics.py

Imports vllm_flash_attn from the repo root, so the freshly built
_vllm_fa2_C.abi3.so must be copied into ./vllm_flash_attn/ first.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_flash_attn import flash_attn_varlen_func  # noqa: E402

torch.manual_seed(1234)
dev = "cuda:0"

NUM_Q_HEADS, NUM_KV_HEADS, HEAD = 8, 2, 128
BLOCK_SIZE = 16


def ref_attention(q, k_seq, v_seq, scale):
    """fp32 reference, causal mask aligned to the bottom-right corner."""
    q_len, seq_len = q.shape[0], k_seq.shape[0]
    group = NUM_Q_HEADS // NUM_KV_HEADS
    out = torch.empty(q_len, NUM_Q_HEADS, HEAD, dtype=torch.float32)
    for h in range(NUM_Q_HEADS):
        kh = h // group
        scores = (q[:, h].float() @ k_seq[:, kh].float().T) * scale
        col = torch.arange(seq_len)[None, :]
        row = torch.arange(q_len)[:, None]
        scores.masked_fill_(col > row + (seq_len - q_len), float("-inf"))
        out[:, h] = scores.softmax(dim=-1) @ v_seq[:, kh].float()
    return out


def main():
    cap = torch.cuda.get_device_capability(0)
    print(f"device: {torch.cuda.get_device_name(0)} (sm{cap[0]}{cap[1]})")

    worst = 0.0
    for q_len, seq_lens in [(1, [3000, 5000]), (2, [3000, 5000]),
                            (4, [3000, 5000]), (7, [1000, 31000]),
                            (8, [128, 31000])]:
        num_seqs = len(seq_lens)
        max_blocks = (max(seq_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
        num_blocks = sum((s + BLOCK_SIZE - 1) // BLOCK_SIZE
                         for s in seq_lens) + 1

        g = torch.Generator(device="cpu").manual_seed(42 + q_len)
        q = torch.randn(q_len * num_seqs, NUM_Q_HEADS, HEAD,
                        generator=g).half().to(dev)
        k = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD,
                        generator=g).half().to(dev)
        v = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD,
                        generator=g).half().to(dev)

        cu_q = torch.tensor([0] + [q_len * (i + 1) for i in range(num_seqs)],
                            dtype=torch.int32, device=dev)
        seqused = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
        bt = torch.zeros(num_seqs, max_blocks, dtype=torch.int32, device=dev)
        nxt = 1
        for i, s in enumerate(seq_lens):
            n = (s + BLOCK_SIZE - 1) // BLOCK_SIZE
            bt[i, :n] = torch.arange(nxt, nxt + n, dtype=torch.int32)
            nxt += n

        scale = HEAD ** -0.5
        out = flash_attn_varlen_func(
            q, k, v,
            max_seqlen_q=q_len,
            cu_seqlens_q=cu_q,
            max_seqlen_k=max(seq_lens),
            seqused_k=seqused,
            softmax_scale=scale,
            causal=True,
            block_table=bt,
            fa_version=2,
        )
        torch.cuda.synchronize()

        k_cpu, v_cpu = k.cpu(), v.cpu()
        for i, s in enumerate(seq_lens):
            n = (s + BLOCK_SIZE - 1) // BLOCK_SIZE
            blocks = bt[i, :n].cpu().long()
            k_seq = k_cpu[blocks].reshape(-1, NUM_KV_HEADS, HEAD)[:s]
            v_seq = v_cpu[blocks].reshape(-1, NUM_KV_HEADS, HEAD)[:s]
            q_seq = q[i * q_len:(i + 1) * q_len].cpu()
            ref = ref_attention(q_seq, k_seq, v_seq, scale)
            got = out[i * q_len:(i + 1) * q_len].cpu().float()
            abse = (got - ref).abs().max().item()
            rel = ((got - ref).abs() / (ref.abs() + 1e-2)).max().item()
            worst = max(worst, abse)
            print(f"q_len={q_len} seq={s:>6}: max abs err {abse:.2e}  "
                  f"max rel err {rel:.2e}")

    # fp16 inputs with fp32 accumulation: a few 1e-3 absolute is expected.
    print(f"worst abs: {worst:.2e} -> {'PASS' if worst < 5e-3 else 'FAIL'}")
    return 0 if worst < 5e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
