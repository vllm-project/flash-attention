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


def _make_paged_kv_from_linear(
    k_lin: torch.Tensor,
    v_lin: torch.Tensor,
    seqlens_k,
    Hkv: int,
    D: int,
    page_block_size: int = 16,
):
    """将线性 KV（total_k, Hkv, D）转换为分页布局：
    - 返回 k_paged, v_paged: (num_blocks, page_block_size, Hkv, D)
    - block_table: (B, max_num_blocks_per_seq) int32
    - seqused_k: (B,) int32（每个序列真实使用的 key 数）
    - cu_seqlens_k: (B+1,) int32
    要求 page_block_size 可被 16 整除。
    """
    assert k_lin.dim() == 3 and k_lin.shape[1] == Hkv and k_lin.shape[2] == D
    assert v_lin.shape == k_lin.shape
    assert page_block_size % 16 == 0

    B = len(seqlens_k)
    # 计算每序列的块数
    nblocks_per_seq = [int((s + page_block_size - 1) // page_block_size) for s in seqlens_k]
    max_blocks = max(nblocks_per_seq) if B > 0 else 0
    total_blocks = sum(nblocks_per_seq)

    # 分配分页 KV
    device = k_lin.device
    dtype = k_lin.dtype
    k_paged = torch.zeros(total_blocks, page_block_size, Hkv, D, device=device, dtype=dtype)
    v_paged = torch.zeros_like(k_paged)

    # 构建 block_table（为未使用位置填 0，占位）
    block_table = torch.zeros(B, max_blocks, device=device, dtype=torch.int32)

    # 将线性 KV 拷入分页缓存
    k_offsets = [0]
    for s in seqlens_k:
        k_offsets.append(k_offsets[-1] + s)
    blk_cursor = 0
    for b in range(B):
        Sk = seqlens_k[b]
        start = k_offsets[b]
        for j in range(nblocks_per_seq[b]):
            block_table[b, j] = blk_cursor
            pos0 = start + j * page_block_size
            pos1 = min(start + (j + 1) * page_block_size, k_offsets[b + 1])
            span = pos1 - pos0
            if span > 0:
                k_paged[blk_cursor, :span].copy_(k_lin[pos0:pos1])
                v_paged[blk_cursor, :span].copy_(v_lin[pos0:pos1])
            blk_cursor += 1

    seqused_k = torch.tensor(seqlens_k, device=device, dtype=torch.int32)
    cu_seqlens_k = (
        torch.tensor([0] + list(seqlens_k), device=device, dtype=torch.int32)
        .cumsum(0, dtype=torch.int32)
        .contiguous()
    )
    return k_paged, v_paged, block_table.contiguous(), seqused_k.contiguous(), cu_seqlens_k


def _ref_abs_s_per_page(
    q: torch.Tensor,
    k_lin: torch.Tensor,
    seqlens_q,
    seqlens_k,
    Hq: int,
    Hkv: int,
    page_block_size: int,
):
    """参考实现：计算每序列、每头、每页（按 page_block_size 切分）的 |scores| 之和。
    返回 (B, Hq, max_blocks) 的张量（fp32）。
    """
    B = len(seqlens_q)
    dq = q.float()
    dk = k_lin.float()
    q_offsets = [0]
    for s in seqlens_q:
        q_offsets.append(q_offsets[-1] + s)
    k_offsets = [0]
    for s in seqlens_k:
        k_offsets.append(k_offsets[-1] + s)
    max_blocks = max([(sk + page_block_size - 1) // page_block_size for sk in seqlens_k]) if B > 0 else 0
    out = torch.zeros((B, Hq, max_blocks), dtype=torch.float32, device=q.device)
    D = dq.shape[-1]
    scale = 1.0 / math.sqrt(D)
    rep = Hq // Hkv
    for b in range(B):
        Sq = seqlens_q[b]; Sk = seqlens_k[b]
        qb = dq[q_offsets[b]:q_offsets[b+1]]    # (Sq, Hq, D)
        kb = dk[k_offsets[b]:k_offsets[b+1]]    # (Sk, Hkv, D)
        kb_r = kb.repeat(1, rep, 1)            # (Sk, Hq, D)
        scores = torch.einsum("thd,shd->hts", qb*scale, kb_r)  # (Hq, Sq, Sk)
        nblk = (Sk + page_block_size - 1) // page_block_size
        for j in range(nblk):
            ks = j * page_block_size
            ke = min((j + 1) * page_block_size, Sk)
            out[b, :, j] = scores[..., ks:ke].abs().sum(dim=(-1, -2))  # sum over (Sq, page_len)
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


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("Hq,Hkv,D", [(8, 2, 128), (4, 4, 128)])
def test_abs_s_total_sum_accuracy(dtype, Hq, Hkv, D):
    """验证 abs_s 的元素总和（sum）是否与参考实现一致。

    - 若内核没有返回任何辅助张量（extras），则跳过测试。
    - 参考实现与 test_abs_s_present_and_reasonable 中的一致。
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    q, k, v, cu_q, cu_k, max_q, max_k = _make_varlen_inputs(
        B=2, Hq=Hq, Hkv=Hkv, D=D, dtype=dtype, device="cuda"
    )

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

    if len(outputs) < 3:
        pytest.skip("No extra aux tensors returned by FA2 build (abs_s unavailable)")

    abs_s = outputs[2]
    assert isinstance(abs_s, torch.Tensor)

    ref = _ref_abs_s(q, k, [int((cu_q[i+1]-cu_q[i]).item()) for i in range(cu_q.numel()-1)],
                     [int((cu_k[i+1]-cu_k[i]).item()) for i in range(cu_k.numel()-1)], Hq, Hkv)

    # 若形状不同，尝试 reshape
    if abs_s.shape != ref.shape:
        try:
            abs_s_view = abs_s.reshape(ref.shape)
        except Exception:
            raise AssertionError(f"Unexpected abs_s shape {abs_s.shape}, expected {ref.shape}")
    else:
        abs_s_view = abs_s

    # 仅比较总和（sum），容差较宽松以容纳不同的浮点序
    sum_abs_s = abs_s_view.float().sum().item()
    sum_ref = ref.sum().item()
    diff = abs(sum_abs_s - sum_ref)
    denom = max(abs(sum_ref), 1.0)
    rel_err = diff / denom

    print(f"sum(abs_s)={sum_abs_s:.6f}, sum(ref)={sum_ref:.6f}, diff={diff:.6f}, rel_err={rel_err:.6f}")
    assert rel_err < 1e-2 or diff < 1e-1, f"Total sum mismatch: rel={rel_err}, abs={diff}"


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("Hq,Hkv,D,page_sz", [(8, 2, 128, 16), (4, 4, 128, 32)])
def test_abs_s_per_page_sum(dtype, Hq, Hkv, D, page_sz):
    """在 paged-KV 场景中，验证每页 abs_s 之和。
    - 构造分页 KV（block_table + (num_blocks, page_sz, Hkv, D) 的 KV）。
    - 通过 vLLM 接口调用 FA2 varlen 路径，要求 return_aux=True。
    - 如果 extras 中包含每页统计（例如形状 (B, Hq, max_blocks) 或 (B, max_blocks)），则与参考实现对比；否则跳过。
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # 构造变长序列
    B = 2
    seqlens_q = (17, 23)
    seqlens_k = (29, 31)
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)
    device = "cuda"

    q = torch.randn(total_q, Hq, D, device=device, dtype=dtype)
    k_lin = torch.randn(total_k, Hkv, D, device=device, dtype=dtype)
    v_lin = torch.randn_like(k_lin)

    # cu_seqlens（int32）
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

    # 构造 paged KV 与 block_table
    k_paged, v_paged, block_table, seqused_k, cu_k_paged = _make_paged_kv_from_linear(
        k_lin, v_lin, seqlens_k, Hkv, D, page_block_size=page_sz
    )

    # 参考的每页 abs_s 之和（(B, Hq, max_blocks)）
    ref_pages = _ref_abs_s_per_page(q, k_lin, seqlens_q, seqlens_k, Hq, Hkv, page_sz)
    B2, Hq2, max_blocks = ref_pages.shape
    assert B2 == B and Hq2 == Hq

    # 走 vLLM 接口（FA2）
    outputs = flash_attn_varlen_func(
        q, k_paged, v_paged,
        max_seqlen_q=max(seqlens_q),
        cu_seqlens_q=cu_q,
        max_seqlen_k=max(seqlens_k),
        cu_seqlens_k=None,  # paged-KV 路径禁止与 seqused_k 同时提供；此处仅传 seqused_k
        seqused_k=seqused_k,
        block_table=block_table,
        causal=True,
        softcap=0.0,
        return_softmax_lse=True,
        return_aux=True,
        fa_version=2,
    )

    if len(outputs) <= 2:
        pytest.skip("No extras returned by FA2 build; per-page abs_s is unavailable.")

    extras = outputs[2:]
    # 试着在 extras 中寻找候选的每页统计：优先 (B, Hq, max_blocks)，其次 (B, max_blocks)
    candidate = None
    for t in extras:
        if not isinstance(t, torch.Tensor):
            continue
        if t.shape == (B, Hq, max_blocks):
            candidate = t
            break
        if t.shape == (B, max_blocks):
            candidate = t.unsqueeze(1).expand(B, Hq, max_blocks).contiguous()
            break
        if t.dim() == 3 and sorted(t.shape) == sorted((B, Hq, max_blocks)):
            # 允许不同的维度顺序（例如 (Hq, B, max_blocks)）
            # 简单尝试几种常见转置
            for perm in [(1,0,2),(0,2,1),(2,0,1),(1,2,0),(2,1,0)]:
                if t.permute(perm).contiguous().shape == (B, Hq, max_blocks):
                    candidate = t.permute(perm).contiguous()
                    break
            if candidate is not None:
                break
    if candidate is None:
        pytest.skip("No per-page abs_s candidate found in extras; please return a tensor with shape (B, Hq, max_blocks) or (B, max_blocks).")

    # 数值对比
    diff = (candidate.float() - ref_pages).abs()
    max_diff = diff.max().item()
    denom = max(ref_pages.abs().max().item(), 1.0)
    rel_err = max_diff / denom
    print(f"per-page max_abs_diff={max_diff:.6f}, rel_err={rel_err:.6f}")
    assert rel_err < 1e-2 or max_diff < 1e-1
