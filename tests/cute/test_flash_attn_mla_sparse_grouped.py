"""Sparse-MLA (DSA) forward with several queries packed into each 128-row tile.

With fewer than 128 query heads per KV head (e.g. 16 per rank under TP), the MLA kernel packs
`group` consecutive queries of 128 // group heads into one 128-head row of qv and gathers the
union of their top-k lists; per-query membership bits mask each query's rows.
"""

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

from flash_attn.cute.topk_gather_kv import GROUPED_ROW_BITS

IS_SM100 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM
TOPK = 2048
TILE_N = 128
SOFTMAX_SCALE = HEAD_SIZE**-0.5

pytestmark = pytest.mark.skipif(not IS_SM100, reason="sparse MLA forward is SM100 only")


def make_flat_kv(num_rows, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    kv = torch.randn(num_rows, 1, HEAD_SIZE, device=device, generator=gen).to(torch.bfloat16)
    return kv, kv[:, :, KV_LORA_RANK:], kv[:, :, :KV_LORA_RANK]


def make_q(total_q, num_heads, device, seed=1):
    gen = torch.Generator(device=device).manual_seed(seed)
    q_full = torch.randn(total_q, num_heads, HEAD_SIZE, device=device, generator=gen)
    q_full = q_full.to(torch.bfloat16)
    return q_full[:, :, KV_LORA_RANK:], q_full[:, :, :KV_LORA_RANK]


def make_topk(total_q, num_rows, counts, overlap, device, seed=2, group=8):
    """Per-query top-k lists (-1 padded). A share `overlap` of each list comes from a pool shared
    by the group; the rest is private to the query, so overlap=0 gives disjoint lists."""
    assert num_rows >= TOPK * (group + 1)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.full((total_q, TOPK), -1, dtype=torch.int32)
    for g in range(0, total_q, group):
        pool = torch.randperm(num_rows, generator=gen)
        shared = pool[:TOPK]
        for t, m in enumerate(range(g, min(g + group, total_q))):
            n = counts[m]
            n_shared = int(n * overlap)
            own = pool[TOPK * (t + 1) : TOPK * (t + 1) + n - n_shared]
            rows = torch.cat([shared[torch.randperm(TOPK, generator=gen)[:n_shared]], own])
            idx[m, :n] = rows.to(torch.int32)
    return idx.to(device)


def build_groups(idx, counts, group):
    """Reference union builder: entries row | (membership << 24), -1 padded to a multiple of 128."""
    total_q = idx.shape[0]
    num_groups = total_q // group
    unions = []
    for g in range(num_groups):
        member = {}
        for t in range(group):
            m = g * group + t
            for r in idx[m, : counts[m]].tolist():
                member[r] = member.get(r, 0) | (1 << t)
        unions.append(sorted(member.items()))
    width = max(TILE_N, -(-max(len(u) for u in unions) // TILE_N) * TILE_N)
    enc = torch.full((num_groups, width), -1, dtype=torch.int64)
    for g, u in enumerate(unions):
        if u:
            rows = torch.tensor([r for r, _ in u], dtype=torch.int64)
            bits = torch.tensor([b for _, b in u], dtype=torch.int64)
            enc[g, : len(u)] = rows | (bits << GROUPED_ROW_BITS)
    enc = torch.where(enc >= 2**31, enc - 2**32, enc).to(torch.int32)  # wrap bit 31 like int32
    lens = torch.tensor([len(u) for u in unions], dtype=torch.int32)
    return enc.to(idx.device), lens.to(idx.device)


def ref_sparse_mla(q, qv, kv, idx, counts, upcast=True):
    """Per-query torch attention over each query's own top-k; lse in natural log."""
    compute_dtype = torch.float32 if upcast else q.dtype
    out = torch.zeros(q.shape[0], q.shape[1], KV_LORA_RANK, device=q.device, dtype=q.dtype)
    lse = torch.full((q.shape[0], q.shape[1]), -math.inf, device=q.device)
    for m in range(q.shape[0]):
        if counts[m] == 0:
            continue
        kv_g = kv[idx[m, : counts[m]].long(), 0, :].to(compute_dtype)
        v_g, k_g = kv_g[:, :KV_LORA_RANK], kv_g[:, KV_LORA_RANK:]
        scores = (q[m].to(compute_dtype) @ k_g.T + qv[m].to(compute_dtype) @ v_g.T) * SOFTMAX_SCALE
        scores = scores.float()
        lse[m] = torch.logsumexp(scores, dim=-1)
        out[m] = (torch.softmax(scores, dim=-1).to(compute_dtype) @ v_g).to(q.dtype)
    return out, lse


def run_grouped(q, qv, k, v, num_kv_rows, enc, lens, group):
    total_q, heads = q.shape[:2]
    num_groups = total_q // group
    device = q.device
    out, lse = flash_attn_varlen_func(
        q.reshape(num_groups, group * heads, QK_ROPE_HEAD_DIM),
        k,
        v,
        qv=qv.reshape(num_groups, group * heads, KV_LORA_RANK),
        cu_seqlens_q=torch.arange(num_groups + 1, device=device, dtype=torch.int32),
        cu_seqlens_k=torch.zeros(num_groups + 1, device=device, dtype=torch.int32),
        seqused_k=torch.full((num_groups,), num_kv_rows, device=device, dtype=torch.int32),
        max_seqlen_q=1,
        max_seqlen_k=num_kv_rows,
        gather_kv_indices=enc,
        gather_kv_valid_length=lens,
        gather_kv_query_group_size=group,
        softmax_scale=SOFTMAX_SCALE,
        return_lse=True,
    )
    return out.reshape(total_q, heads, KV_LORA_RANK), lse.reshape(total_q, heads)


def check(group, overlap, counts, num_kv_rows=32768, seed=0):
    heads = 128 // group
    total_q = len(counts)
    kv, k, v = make_flat_kv(num_kv_rows, "cuda", seed=seed)
    q, qv = make_q(total_q, heads, "cuda", seed=seed + 1)
    idx = make_topk(total_q, num_kv_rows, counts, overlap, "cuda", seed=seed + 2, group=group)
    enc, lens = build_groups(idx, counts, group)

    out, lse = run_grouped(q, qv, k, v, num_kv_rows, enc, lens, group)
    out_ref, lse_ref = ref_sparse_mla(q, qv, kv, idx, counts)
    out_pt, _ = ref_sparse_mla(q, qv, kv, idx, counts, upcast=False)

    empty = [m for m, n in enumerate(counts) if n == 0]
    for m in empty:
        assert (out[m] == 0).all(), f"query {m}: expected zero output without top-k entries"
        assert (lse[m] == -math.inf).all(), f"query {m}: expected -inf lse without top-k entries"
    sel = torch.tensor([m for m, n in enumerate(counts) if n > 0], device="cuda")
    pt_err = (out_pt[sel].float() - out_ref[sel].float()).abs().max().item()
    err = (out[sel].float() - out_ref[sel].float()).abs().max().item()
    assert err <= 2 * pt_err + 1e-3, f"out max diff {err} vs torch-bf16 {pt_err}"
    lse_err = (lse[sel] - lse_ref[sel]).abs().max().item()
    assert lse_err <= 5e-3, f"lse max diff {lse_err}"
    return lens


@pytest.mark.parametrize("group", [2, 4, 8])
@pytest.mark.parametrize("overlap", [0.9, 0.0, 1.0])
def test_grouped_matches_per_query_reference(group, overlap):
    counts = [TOPK] * (4 * group)
    lens = check(group, overlap, counts)
    if overlap == 0.0:
        assert int(lens.max()) == group * TOPK, "disjoint top-k must fill the whole union width"
    if overlap == 1.0:
        assert int(lens.max()) == TOPK, "identical top-k must collapse to one list"


@pytest.mark.parametrize("group", [2, 8])
def test_grouped_ragged_counts_and_empty_queries(group):
    """Queries with short or empty top-k; union lengths that are not multiples of 128."""
    pattern = [TOPK, 0, 1, TILE_N + 1, 777, TOPK - 1, 0, 300]
    counts = (pattern * (2 * group))[: 4 * group]
    counts[group : 2 * group] = [0] * group  # a group with no entries at all
    lens = check(group, 0.8, counts, seed=10)
    assert int(lens[1]) == 0
    assert any(int(n) % TILE_N for n in lens), "expected a union length off the 128 grid"


def test_grouped_without_valid_length_is_bit_identical():
    """Without per-group lengths the kernel walks the full width; -1 padding must mask it all."""
    group, heads = 8, 16
    counts = [TOPK, 300, 0, 1500, TOPK, 7, 900, 1] * 2
    kv, k, v = make_flat_kv(32768, "cuda", seed=30)
    q, qv = make_q(len(counts), heads, "cuda", seed=31)
    idx = make_topk(len(counts), 32768, counts, 0.7, "cuda", seed=32, group=group)
    enc, lens = build_groups(idx, counts, group)
    out_len, lse_len = run_grouped(q, qv, k, v, 32768, enc, lens, group)
    out_none, lse_none = run_grouped(q, qv, k, v, 32768, enc, None, group)
    assert torch.equal(out_len, out_none)
    assert torch.equal(lse_len, lse_none)


def test_grouped_rejects_causal():
    """Causality lives in each query's top-k; a per-tile causal limit would be wrong."""
    group, heads = 8, 16
    kv, k, v = make_flat_kv(4096, "cuda")
    q, qv = make_q(group, heads, "cuda")
    enc = torch.full((1, TILE_N), -1, device="cuda", dtype=torch.int32)
    with pytest.raises(AssertionError, match="causality"):
        flash_attn_varlen_func(
            q.reshape(1, 128, QK_ROPE_HEAD_DIM), k, v, qv=qv.reshape(1, 128, KV_LORA_RANK),
            cu_seqlens_q=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            cu_seqlens_k=torch.zeros(2, device="cuda", dtype=torch.int32),
            seqused_k=torch.full((1,), 4096, device="cuda", dtype=torch.int32),
            max_seqlen_q=1, max_seqlen_k=4096, gather_kv_indices=enc,
            gather_kv_query_group_size=group, softmax_scale=SOFTMAX_SCALE, causal=True,
        )


def test_grouped_rows_with_high_membership_bit():
    """Bit 31 (query 7 of a group of 8) makes entries negative; only -1 is padding."""
    group = 8
    counts = [0] * 7 + [TOPK]
    counts = counts * 2
    check(group, 0.5, counts, seed=20)
