import math
import os
import pytest
import torch

'''该测试针对 vllm_flash_attn 的 FA2 路径，假设自定义内核会在 varlen 前向
额外返回一个表示累计 |s| 的张量（例如 abs_s）。不假定返回参数个数，
如果检测不到该额外张量，就跳过测试。'''

from vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func


def _make_varlen_inputs(
    B=2, Hq=8, Hkv=2, D=128, seqlens_q=(17, 23), seqlens_k=(29, 31),
    dtype=torch.bfloat16, device="cuda",
):
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)
    q = torch.randn(total_q, Hq, D, device=device, dtype=dtype)
    k = torch.randn(total_k, Hkv, D, device=device, dtype=dtype)
    v = torch.randn_like(k)

    # 输出为 int32
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


def _ref_abs_s(q, k, seqlens_q, seqlens_k, Hq, Hkv):
    # Reference computes sum over abs(scores) before softmax for each (b, h, t)
    # We expand K to Hq heads to match grouped-query pattern.
    B = len(seqlens_q)
    dq = q.float()  # (total_q, Hq, D)
    dk = k.float()  # (total_k, Hkv, D)

    q_offsets = [0]
    for s in seqlens_q:
        q_offsets.append(q_offsets[-1] + s)
    k_offsets = [0]
    for s in seqlens_k:
        k_offsets.append(k_offsets[-1] + s)

    abs_sums = []  # per batch, per head, per t
    D = dq.shape[-1]
    scale = 1.0 / math.sqrt(D)
    for b in range(B):
        qb = dq[q_offsets[b]:q_offsets[b+1]]    # (Sq, Hq, D)
        kb = dk[k_offsets[b]:k_offsets[b+1]]    # (Sk, Hkv, D)
        # repeat kv heads
        rep = Hq // Hkv
        kb_r = kb.repeat(1, rep, 1)            # (Sk, Hq, D)
        # scores: (Hq, Sq, Sk)
        scores = torch.einsum("thd,shd->hts", qb*scale, kb_r)
        # abs sum along key dim
        abs_sums.append(scores.abs().sum(dim=-1))  # (Hq, Sq)
    # pack into (Hq, total_q)
    total_q = sum(seqlens_q)
    out = torch.zeros((Hq, total_q), dtype=torch.float32, device=q.device)
    base = 0
    for b in range(B):
        Sq = seqlens_q[b]
        out[:, base:base+Sq] = abs_sums[b]
        base += Sq
    return out


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("Hq,Hkv,D", [(8, 2, 128), (4, 4, 128)])
def test_abs_s_present_and_reasonable(dtype, Hq, Hkv, D):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    q, k, v, cu_q, cu_k, max_q, max_k = _make_varlen_inputs(
        B=2, Hq=Hq, Hkv=Hkv, D=D, dtype=dtype, device="cuda"
    )

    # Call the raw varlen_fwd; detect extra outputs
    # 调用 vllm 接口（fa_version=2），并要求返回辅助张量
    # 返回格式：
    # - 若 return_aux=False：out 或 (out, lse)
    # - 若 return_aux=True：会在末尾追加任意数量的 extras（例如 abs_s）
    outputs = flash_attn_varlen_func(
        q, k, v,
        max_seqlen_q=max_q,
        cu_seqlens_q=cu_q,
        max_seqlen_k=max_k,
        cu_seqlens_k=cu_k,
        seqused_k=None,
        causal=True,
        softcap=0.0,
        return_softmax_lse=True,
        return_aux=True,
        fa_version=2,
    )

    # 兼容返回：有 lse 时，前两个是 out, lse，其余 extras；没有 lse 则第一个是 out
    assert isinstance(outputs, (tuple, list))

    # 打印返回项和基础信息（需要用 -s 运行 pytest 才会显示）
    print(f"[fa2] Hq={Hq}, Hkv={Hkv}, D={D}, dtype={dtype}, len(outputs)={len(outputs)}")
    def _tinfo(x):
        return f"Tensor(shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device})" if isinstance(x, torch.Tensor) else str(type(x))
    if len(outputs) >= 1:
        print(f"[fa2] out: {_tinfo(outputs[0])}")
    if len(outputs) >= 2:
        print(f"[fa2] lse: {_tinfo(outputs[1])}")
    if len(outputs) > 2:
        for i, ex in enumerate(outputs[2:], start=2):
            print(f"[fa2] extra[{i}]: {_tinfo(ex)}")

    if len(outputs) < 3:
        # 没有返回任何辅助张量（如 abs_s）。为了便于观察，打印 out/lse 的统计后再跳过。
        out0 = outputs[0]
        print(f"[fa2] out stats: min={out0.min().item():.6f}, max={out0.max().item():.6f}, mean={out0.float().mean().item():.6f}")
        if len(outputs) >= 2 and isinstance(outputs[1], torch.Tensor) and outputs[1].numel() > 0:
            lse0 = outputs[1]
            print(f"[fa2] lse stats: min={lse0.min().item():.6f}, max={lse0.max().item():.6f}, mean={lse0.float().mean().item():.6f}")
        pytest.skip("No extra aux tensors returned by FA2 build (abs_s unavailable)")

    # 经验：第 3 个返回开始是辅助张量，取第一个作为 abs_s
    abs_s = outputs[2]
    assert isinstance(abs_s, torch.Tensor)

    # Basic shape checks: accept either (Hq, total_q) or broadcastable variants
    total_q = q.shape[0]
    assert total_q > 0
    assert abs_s.numel() > 0

    # Compute a simple reference in fp32 and compare up to tolerance
    ref = _ref_abs_s(q, k, [int((cu_q[i+1]-cu_q[i]).item()) for i in range(cu_q.numel()-1)],
                     [int((cu_k[i+1]-cu_k[i]).item()) for i in range(cu_k.numel()-1)], Hq, Hkv)

    # If shapes mismatch, try to reshape to (Hq, total_q) if compatible
    if abs_s.shape != ref.shape:
        try:
            abs_s_view = abs_s.reshape(ref.shape)
        except Exception:
            # last resort: fail with debug info
            raise AssertionError(f"Unexpected abs_s shape {abs_s.shape}, expected {ref.shape}")
    else:
        abs_s_view = abs_s

    # Allow loose tolerance since kernels may reorder floating ops
    rel_tol = 1e-2
    abs_tol = 1e-2
    max_diff = (abs_s_view.float() - ref).abs().max().item()
    denom = max(ref.abs().max().item(), 1.0)
    rel_err = max_diff / denom

    # 打印关键信息，便于人工查看 abs_s 的数值特征
    print(f"abs_s shape: {tuple(abs_s.shape)}")
    print(f"ref shape:   {tuple(ref.shape)}")
    print(f"abs_s stats: min={abs_s_view.min().item():.6f}, max={abs_s_view.max().item():.6f}, mean={abs_s_view.mean().item():.6f}")
    print(f"ref stats:   min={ref.min().item():.6f}, max={ref.max().item():.6f}, mean={ref.mean().item():.6f}")
    print(f"diff: max_abs={max_diff:.6f}, rel_err={rel_err:.6f}")
    assert rel_err < 0.1 or max_diff < 0.5, f"abs_s differs too much: rel={rel_err}, abs={max_diff}"


if __name__ == "__main__":
    # Allow running as a script
    from pathlib import Path
    try:
        test_abs_s_present_and_reasonable(torch.bfloat16, 8, 2, 128)
        print("OK")
    except SystemExit:
        pass
