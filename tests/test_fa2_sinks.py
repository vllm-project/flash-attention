"""
Single-file FlashAttention2 with attention sinks accuracy test.

Usage:
    pytest test_fa2_accuracy_single.py

Parameters:
    so_path  - path to the shared library (.so)
    atol     - absolute tolerance
    rtol     - relative tolerance
    csv_path - path to save accuracy
"""
import pytest
import torch
import csv
import os
import vllm
from typing import List, Tuple

pkg = os.path.dirname(vllm.__file__)
so_path = os.path.join(pkg, "vllm_flash_attn", "_vllm_fa2_C.abi3.so")
torch.ops.load_library(so_path)
fa2_op = torch.ops._vllm_fa2_C.varlen_fwd

atol=1e-3
rtol=1e-3
csv_path = os.path.join(".", "test_fa2_sinks_accuracy.csv")

_csv_file = open(csv_path, "a", newline="")
_writer = csv.writer(_csv_file)
if os.stat(csv_path).st_size == 0:          # 空文件才写表头
    _writer.writerow(["causal", "headdim", "batch_size", "seqlen",
                      "nheads_q", "q_heads_per_k_heads",
                      "rtol", "atol", "bad_ratio"])

bs_seqlen_vals = [(32, 1), (32, 512), (16, 1024), (8, 2048),
                  (4, 4096), (2, 8192), (1, 16384)]
causal_vals = [False, True]
headdim_vals = [64, 128]
nheads_q_vals = [4, 8, 16, 32, 64]
q_heads_per_k_heads_vals = [4, 2, 1]


def _parametrize() -> List[Tuple]:
    cases = []
    for causal in causal_vals:
        for headdim in headdim_vals:
            for (bs, sq) in bs_seqlen_vals:
                for nhq in nheads_q_vals:
                    for qhk in q_heads_per_k_heads_vals:
                        cases.append((causal, headdim, bs, sq, nhq, qhk))
    return cases


@pytest.mark.parametrize("causal,headdim,batch_size,seqlen,nheads_q,q_heads_per_k_heads",
                         _parametrize())
def test_accuracy(causal, headdim, batch_size, seqlen, nheads_q,
                  q_heads_per_k_heads):
    device, dtype = "cuda", torch.float16
    nheads_k = nheads_q // q_heads_per_k_heads

   
    q_o = torch.randn(batch_size, seqlen, nheads_q, headdim,
                      device=device, dtype=dtype, requires_grad=False)
    k_o = torch.randn(batch_size, seqlen, nheads_k, headdim,
                      device=device, dtype=dtype, requires_grad=False)
    v_o = torch.randn(batch_size, seqlen, nheads_k, headdim,
                      device=device, dtype=dtype, requires_grad=False)

    q = q_o.reshape(-1, nheads_q, headdim)
    k = k_o.reshape(-1, nheads_k, headdim)
    v = v_o.reshape(-1, nheads_k, headdim)
    out_buf = torch.empty_like(q)
    cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, seqlen,
                              device=device, dtype=torch.int32)
    s_aux = torch.randn(nheads_q, device=device, dtype=dtype)

    fa2_op(q, k, v, out_buf,
           cu_seqlens, cu_seqlens,
           None, None, None, None,
           seqlen, seqlen,
           0.0,
           torch.tensor(headdim ** -0.5, device=device),
           False, causal,
           -1, -1, 0.0,
           False, None, s_aux)

    out_buf = out_buf.reshape(batch_size, seqlen, nheads_q, headdim)

    bad_count = 0
    total_count = 0
    for b in range(batch_size):
        for h in range(nheads_q):
            q1 = q_o[b, :, h, :].float()
            k1 = k_o[b, :, h // q_heads_per_k_heads, :].float()
            v1 = v_o[b, :, h // q_heads_per_k_heads, :].float()

            scores = (q1 @ k1.T) * (headdim ** -0.5)
            if causal:
                causal_mask = torch.triu(
                    torch.full((seqlen, seqlen), float("-inf"),
                               device=device, dtype=torch.float32), diagonal=1)
                scores = scores + causal_mask

            sink_col = s_aux[h].float().view(1, 1).expand(seqlen, 1)
            scores_ext = torch.cat([sink_col, scores], dim=1)
            attn = torch.softmax(scores_ext, dim=1)
            v_ext = torch.cat([torch.zeros(1, v1.shape[1], device=device, dtype=torch.float32),
                               v1], dim=0)
            ref = (attn @ v_ext).to(dtype)

            bad = ~torch.isclose(ref, out_buf[b, :, h, :], atol=atol, rtol=rtol)
            bad_count += bad.sum().item()
            total_count += ref.numel()

    ratio = bad_count / total_count * 100

    _writer.writerow([causal, headdim, batch_size, seqlen, nheads_q,
                      q_heads_per_k_heads, rtol, atol, f"{ratio:.2f}%"])
    _csv_file.flush()
    assert ratio == 0, f"Bad ratio {ratio:.2f}% ({bad_count}/{total_count}) > 0"


def pytest_sessionfinish(session, exitstatus):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["causal", "headdim", "batch_size", "seqlen",
                             "nheads_q", "q_heads_per_k_heads",
                             "rtol", "atol", "bad_ratio"])
        writer.writerows(_ROWS)