"""Sparse-MLA (DSA) forward against a flat, request-shared KV cache, as vLLM calls it."""

import importlib.machinery
import math
import os
import sys
import types

import pytest
import torch

try:
    from flash_attn.cute import flash_attn_varlen_func
except ImportError:
    # no FA2 extension: register a virtual flash_attn package over the source tree
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _mod = types.ModuleType("flash_attn")
    _mod.__path__ = [os.path.join(_root, "flash_attn")]
    _mod.__package__ = "flash_attn"
    _mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None, is_package=True)
    _mod.__spec__.submodule_search_locations = _mod.__path__
    sys.modules["flash_attn"] = _mod
    from flash_attn.cute import flash_attn_varlen_func

IS_SM100 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM
NUM_HEADS = 128
TOPK = 2048
TILE_N = 128
SOFTMAX_SCALE = HEAD_SIZE**-0.5

pytestmark = pytest.mark.skipif(not IS_SM100, reason="sparse MLA forward is SM100 only")


def make_flat_kv(num_rows, device, dtype=torch.bfloat16, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    kv = torch.randn(
        num_rows, 1, HEAD_SIZE, device=device, dtype=torch.float32, generator=gen
    ).to(dtype)
    # Last-dim-contiguous views; maybe_contiguous() must not copy them.
    v = kv[:, :, :KV_LORA_RANK]
    k = kv[:, :, KV_LORA_RANK:]
    assert v.stride(-1) == 1 and k.stride(-1) == 1
    return kv, k, v


def make_q(total_q, device, dtype=torch.bfloat16, seed=1, split_views=True, num_heads=NUM_HEADS):
    gen = torch.Generator(device=device).manual_seed(seed)
    q_full = torch.randn(
        total_q, num_heads, HEAD_SIZE, device=device, dtype=torch.float32, generator=gen
    ).to(dtype)
    qv = q_full[:, :, :KV_LORA_RANK]
    q = q_full[:, :, KV_LORA_RANK:]
    if not split_views:
        qv, q = qv.contiguous(), q.contiguous()
    return q, qv


def make_indices(valid_counts, num_rows, device, seed=2, topk=TOPK, pad=-1):
    """Per-row top-k lists: `valid_counts[m]` real flat row ids, then `pad` sentinels."""
    gen = torch.Generator(device=device).manual_seed(seed)
    total_q = len(valid_counts)
    idx = torch.full((total_q, topk), pad, device=device, dtype=torch.int32)
    for m, n in enumerate(valid_counts):
        if n == 0:
            continue
        perm = torch.randperm(num_rows, device=device, generator=gen)[:n]
        idx[m, :n] = perm.to(torch.int32)
    valid_len = torch.tensor(valid_counts, device=device, dtype=torch.int32)
    return idx, valid_len


def ref_sparse_mla(q, qv, kv, idx, valid_counts, softmax_scale=SOFTMAX_SCALE, upcast=True):
    """Pure-torch attention over the gathered rows. Returns (out, lse) with lse in natural log."""
    total_q = q.shape[0]
    compute_dtype = torch.float32 if upcast else q.dtype
    out = torch.zeros(total_q, q.shape[1], KV_LORA_RANK, device=q.device, dtype=q.dtype)
    lse = torch.full((total_q, q.shape[1]), -math.inf, device=q.device, dtype=torch.float32)
    for m in range(total_q):
        n = int(valid_counts[m])
        if n == 0:
            continue
        rows = idx[m, :n].long()
        assert (rows >= 0).all() and (rows < kv.shape[0]).all()
        kv_g = kv[rows, 0, :].to(compute_dtype)  # (n, HEAD_SIZE)
        v_g, k_g = kv_g[:, :KV_LORA_RANK], kv_g[:, KV_LORA_RANK:]
        scores = (
            q[m].to(compute_dtype) @ k_g.transpose(0, 1)
            + qv[m].to(compute_dtype) @ v_g.transpose(0, 1)
        ) * softmax_scale
        scores = scores.float()
        lse[m] = torch.logsumexp(scores, dim=-1)
        p = torch.softmax(scores, dim=-1).to(compute_dtype)
        out[m] = (p @ v_g).to(q.dtype)
    return out, lse


def run_kernel(
    q, qv, k, v, cu_seqlens_q, num_kv_rows, idx, valid_len,
    softmax_scale=SOFTMAX_SCALE, return_lse=True,
):
    batch = cu_seqlens_q.numel() - 1
    device = q.device
    return flash_attn_varlen_func(
        q,
        k,
        v,
        qv=qv,
        cu_seqlens_q=cu_seqlens_q,
        # Every request shares the whole flat cache: offset 0, length = flat row count.
        cu_seqlens_k=torch.zeros(batch + 1, device=device, dtype=torch.int32),
        seqused_k=torch.full((batch,), num_kv_rows, device=device, dtype=torch.int32),
        max_seqlen_q=int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max()),
        max_seqlen_k=num_kv_rows,
        gather_kv_indices=idx,
        gather_kv_valid_length=valid_len,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
    )


def cu_seqlens_from(q_lens, device):
    out = torch.zeros(len(q_lens) + 1, device=device, dtype=torch.int32)
    out[1:] = torch.tensor(q_lens, device=device, dtype=torch.int32).cumsum(0)
    return out


def assert_close(out, lse, out_ref, lse_ref, out_pt, valid_counts):
    zero_rows = [m for m, n in enumerate(valid_counts) if n == 0]
    nonzero = [m for m, n in enumerate(valid_counts) if n > 0]
    for m in zero_rows:
        assert (out[m] == 0).all(), f"row {m}: expected all-zero output for 0 valid indices"
        assert (lse[m] == -math.inf).all(), f"row {m}: expected -inf lse for 0 valid indices"
    if not nonzero:
        return
    sel = torch.tensor(nonzero, device=out.device)
    o, o_ref, o_pt = out[sel].float(), out_ref[sel].float(), out_pt[sel].float()
    pt_err = (o_pt - o_ref).abs().max().item()
    err = (o - o_ref).abs().max().item()
    assert err <= 2 * pt_err + 1e-3, f"out max diff {err} vs torch-bf16 {pt_err}"
    lse_err = (lse[sel] - lse_ref[sel]).abs().max().item()
    assert lse_err <= 5e-3, f"lse max diff {lse_err}"


@pytest.mark.parametrize("split_views", [True, False])
@pytest.mark.parametrize("batch", [8, 13])
def test_decode_ragged_valid_counts(batch, split_views):
    """q_len == 1 per request; ragged valid counts including zero-valid rows."""
    device = "cuda"
    num_kv_rows = 4096
    q_lens = [1] * batch
    total_q = batch
    cu_seqlens_q = cu_seqlens_from(q_lens, device)

    counts = [TOPK, 0, 1, TILE_N, TILE_N + 1, TOPK - 1, 777, 0]
    counts = (counts * ((total_q // len(counts)) + 1))[:total_q]

    kv, k, v = make_flat_kv(num_kv_rows, device)
    q, qv = make_q(total_q, device, split_views=split_views)
    idx, valid_len = make_indices(counts, num_kv_rows, device)

    out, lse = run_kernel(q, qv, k, v, cu_seqlens_q, num_kv_rows, idx, valid_len)
    out_ref, lse_ref = ref_sparse_mla(q, qv, kv, idx, counts)
    out_pt, _ = ref_sparse_mla(q, qv, kv, idx, counts, upcast=False)
    assert_close(out, lse, out_ref, lse_ref, out_pt, counts)


def test_prefill_varlen_mixed_q_lens():
    """Mixed q_lens, per-token top-k, ragged valid counts including zero-valid rows."""
    device = "cuda"
    num_kv_rows = 3072
    q_lens = [1, 5, 17, 2, 33, 1, 8, 64]
    total_q = sum(q_lens)
    cu_seqlens_q = cu_seqlens_from(q_lens, device)

    gen = torch.Generator(device="cpu").manual_seed(7)
    counts = torch.randint(0, TOPK + 1, (total_q,), generator=gen).tolist()
    counts[0] = 0
    counts[1] = TOPK
    counts[-1] = 0
    counts[total_q // 2] = TILE_N

    kv, k, v = make_flat_kv(num_kv_rows, device, seed=3)
    q, qv = make_q(total_q, device, seed=4)
    idx, valid_len = make_indices(counts, num_kv_rows, device, seed=5)

    out, lse = run_kernel(q, qv, k, v, cu_seqlens_q, num_kv_rows, idx, valid_len)
    out_ref, lse_ref = ref_sparse_mla(q, qv, kv, idx, counts)
    out_pt, _ = ref_sparse_mla(q, qv, kv, idx, counts, upcast=False)
    assert_close(out, lse, out_ref, lse_ref, out_pt, counts)


def test_all_rows_full_is_bit_identical_to_no_valid_length():
    """Passing an all-full valid_length must not change a single bit of the result."""
    device = "cuda"
    num_kv_rows = 4096
    batch = 8
    cu_seqlens_q = cu_seqlens_from([1] * batch, device)
    kv, k, v = make_flat_kv(num_kv_rows, device, seed=6)
    q, qv = make_q(batch, device, seed=7)
    idx, valid_len = make_indices([TOPK] * batch, num_kv_rows, device, seed=8)

    out_dyn, lse_dyn = run_kernel(q, qv, k, v, cu_seqlens_q, num_kv_rows, idx, valid_len)
    out_static, lse_static = run_kernel(q, qv, k, v, cu_seqlens_q, num_kv_rows, idx, None)
    assert torch.equal(out_dyn, out_static)
    assert torch.equal(lse_dyn, lse_static)


def test_negative_one_padding_without_valid_length():
    """Literal -1 padding must be filtered by the bitmask even with no valid_length tensor."""
    device = "cuda"
    num_kv_rows = 2048
    batch = 8
    cu_seqlens_q = cu_seqlens_from([1] * batch, device)
    counts = [TOPK, 0, 3, 512, 1024, 2047, 1, 0]
    kv, k, v = make_flat_kv(num_kv_rows, device, seed=9)
    q, qv = make_q(batch, device, seed=10)
    idx, _ = make_indices(counts, num_kv_rows, device, seed=11)

    out, lse = run_kernel(q, qv, k, v, cu_seqlens_q, num_kv_rows, idx, None)
    out_ref, lse_ref = ref_sparse_mla(q, qv, kv, idx, counts)
    out_pt, _ = ref_sparse_mla(q, qv, kv, idx, counts, upcast=False)
    assert_close(out, lse, out_ref, lse_ref, out_pt, counts)


@pytest.mark.parametrize("pad", [-1, -2147483648, 1 << 20])
def test_out_of_range_padding_sentinels(pad):
    """The validity predicate is `0 <= idx < seqused_k`; any sentinel outside it works."""
    device = "cuda"
    num_kv_rows = 2048
    batch = 8
    cu_seqlens_q = cu_seqlens_from([1] * batch, device)
    counts = [TOPK, 0, 300, 1024, 2047, 1, 1500, 0]
    kv, k, v = make_flat_kv(num_kv_rows, device, seed=21)
    q, qv = make_q(batch, device, seed=22)
    idx, valid_len = make_indices(counts, num_kv_rows, device, seed=23, pad=pad)

    out, lse = run_kernel(q, qv, k, v, cu_seqlens_q, num_kv_rows, idx, valid_len)
    out_ref, lse_ref = ref_sparse_mla(q, qv, kv, idx, counts)
    out_pt, _ = ref_sparse_mla(q, qv, kv, idx, counts, upcast=False)
    assert_close(out, lse, out_ref, lse_ref, out_pt, counts)


@pytest.mark.parametrize("valid_len,attended", [(128, 128), (1024, 1024), (200, 256)])
def test_early_exit_ignores_entries_beyond_valid_length(valid_len, attended):
    """attended = round_up(valid_len, 128): the tail of the last block is still attended."""
    device = "cuda"
    num_kv_rows = 4096
    batch = 4
    cu_seqlens_q = cu_seqlens_from([1] * batch, device)
    kv, k, v = make_flat_kv(num_kv_rows, device, seed=15)
    q, qv = make_q(batch, device, seed=16)
    idx, _ = make_indices([TOPK] * batch, num_kv_rows, device, seed=17)
    valid = torch.full((batch,), valid_len, device=device, dtype=torch.int32)

    out, lse = run_kernel(q, qv, k, v, cu_seqlens_q, num_kv_rows, idx, valid)
    counts = [attended] * batch
    out_ref, lse_ref = ref_sparse_mla(q, qv, kv, idx, counts)
    out_pt, _ = ref_sparse_mla(q, qv, kv, idx, counts, upcast=False)
    assert_close(out, lse, out_ref, lse_ref, out_pt, counts)

    out_full, _ = ref_sparse_mla(q, qv, kv, idx, [TOPK] * batch)
    assert (out.float() - out_full.float()).abs().max().item() > 1e-2, (
        "early exit had no effect: result matches attending the full top-k"
    )


def test_dense_batch_seqlen_layout():
    """Non-varlen (batch, seqlen_q) index and valid-length layout."""
    from flash_attn.cute import flash_attn_func

    device = "cuda"
    batch, seqlen_q, seqlen_k = 4, 2, 4096
    kv_flat, _, _ = make_flat_kv(batch * seqlen_k, device, seed=18)
    kv = kv_flat.reshape(batch, seqlen_k, 1, HEAD_SIZE)
    v, k = kv[..., :KV_LORA_RANK], kv[..., KV_LORA_RANK:]
    q, qv = make_q(batch * seqlen_q, device, seed=19)
    q = q.reshape(batch, seqlen_q, NUM_HEADS, QK_ROPE_HEAD_DIM)
    qv = qv.reshape(batch, seqlen_q, NUM_HEADS, KV_LORA_RANK)

    counts = [TOPK, 0, TILE_N, 999, 1, TOPK - 1, 512, 0]
    idx, valid_len = make_indices(counts, seqlen_k, device, seed=20)
    idx = idx.reshape(batch, seqlen_q, TOPK)
    valid_len = valid_len.reshape(batch, seqlen_q)

    out, lse = flash_attn_func(
        q,
        k,
        v,
        qv=qv,
        gather_kv_indices=idx,
        gather_kv_valid_length=valid_len,
        softmax_scale=SOFTMAX_SCALE,
        return_lse=True,
    )
    for b in range(batch):
        kv_b = kv[b].reshape(seqlen_k, 1, HEAD_SIZE)
        counts_b = counts[b * seqlen_q : (b + 1) * seqlen_q]
        out_ref, lse_ref = ref_sparse_mla(q[b], qv[b], kv_b, idx[b], counts_b)
        out_pt, _ = ref_sparse_mla(q[b], qv[b], kv_b, idx[b], counts_b, upcast=False)
        assert_close(out[b], lse[b], out_ref, lse_ref, out_pt, counts_b)


def test_lse_is_natural_log():
    device = "cuda"
    num_kv_rows = 2048
    batch = 4
    cu_seqlens_q = cu_seqlens_from([1] * batch, device)
    counts = [TOPK, 256, 1153, 128]
    kv, k, v = make_flat_kv(num_kv_rows, device, seed=12)
    q, qv = make_q(batch, device, seed=13)
    idx, valid_len = make_indices(counts, num_kv_rows, device, seed=14)

    _, lse = run_kernel(q, qv, k, v, cu_seqlens_q, num_kv_rows, idx, valid_len)
    _, lse_ref_e = ref_sparse_mla(q, qv, kv, idx, counts)
    lse_ref_2 = lse_ref_e / math.log(2.0)

    err_e = (lse - lse_ref_e).abs().max().item()
    err_2 = (lse - lse_ref_2).abs().max().item()
    assert err_e <= 5e-3, f"lse is not natural log: max diff vs ln {err_e}, vs log2 {err_2}"
    assert err_2 > 1.0, "lse could not be distinguished from log2; test inputs are degenerate"


def test_valid_length_broadcast_cache_reuse():
    """A static zero batch stride must not leak into a contiguous-length call."""
    from flash_attn.cute import flash_attn_func
    from flash_attn.cute.interface import _flash_attn_fwd

    _flash_attn_fwd.compile_cache.clear()
    batch, rows = 2, 256
    kv, _, _ = make_flat_kv(batch * rows, "cuda")
    kv = kv.reshape(batch, rows, 1, HEAD_SIZE)
    q, qv = make_q(batch, "cuda")
    q, qv = q.unsqueeze(1), qv.unsqueeze(1)
    idx, _ = make_indices([rows] * batch, rows, "cuda", topk=rows)
    idx = idx.unsqueeze(1)
    broadcast = torch.tensor([[128]], device="cuda", dtype=torch.int32).expand(batch, 1)
    contiguous = torch.tensor([[128], [256]], device="cuda", dtype=torch.int32)
    for lengths in [broadcast, contiguous, broadcast]:
        out, _ = flash_attn_func(
            q, kv[..., KV_LORA_RANK:], kv[..., :KV_LORA_RANK], qv=qv,
            gather_kv_indices=idx, gather_kv_valid_length=lengths,
            softmax_scale=SOFTMAX_SCALE,
        )
        for b in range(batch):
            ref, _ = ref_sparse_mla(q[b], qv[b], kv[b], idx[b], lengths[b].tolist())
            torch.testing.assert_close(out[b], ref, atol=0.015, rtol=0.015)
