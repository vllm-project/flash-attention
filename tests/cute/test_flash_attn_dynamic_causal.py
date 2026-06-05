"""Regression tests for per-sequence ``dynamic_causal`` masking on SM90.

These cover the split-KV bug fixed in ``flash_fwd_sm90.py``: for a
*bidirectional* sequence (``psc == 0``) with split-KV active (``num_splits > 1``)
and ``seqlen_q < seqlen_k`` the forward kernel used to

  * leave ``n_block_min`` at its (possibly causal) split offset while only
    resetting ``n_block_max`` to the global max, so the splits overlapped and
    keys were double-counted -> corrupted softmax (relative error ~0.33), and
  * compute different block ranges on the producer (K/V load) and consumer
    sides, so the pipeline block counts disagreed and the kernel deadlocked
    (the GPU spun indefinitely; observed during CUDA-graph capture).

The fix recomputes ``[n_block_min, n_block_max)`` over the *full* key range and
partitions it into disjoint per-split slices identically on both sides, gated
only on ``dynamic_causal is not None`` (NOT on the kernel's ``causal`` compile
flag -- vLLM compiles the kernel with ``causal=False`` and passes a
``dynamic_causal`` tensor, which is exactly the path that used to hang).

A regression here may hang rather than fail; rely on the CI per-test timeout.
"""
import pytest
import torch

from flash_attn.cute.interface import _flash_attn_fwd

DEV = "cuda"
DT = torch.bfloat16
PAGE = 16


def _is_sm90() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 9


pytestmark = pytest.mark.skipif(
    not _is_sm90(), reason="dynamic_causal split-KV fix is SM90-specific"
)


def _rep(t, hq):
    """Repeat KV heads to match the number of query heads (GQA/MQA)."""
    return t.repeat_interleave(hq // t.shape[1], 1)


def _ref_bidirectional(q, k, v, scale):
    sq, hq, d = q.shape
    qf = q.float().transpose(0, 1)
    kf = _rep(k.float(), hq).transpose(0, 1)
    vf = _rep(v.float(), hq).transpose(0, 1)
    p = (torch.matmul(qf, kf.transpose(-1, -2)) * scale).softmax(-1)
    return torch.matmul(p, vf).transpose(0, 1).to(DT)


def _ref_causal(q, k, v, scale):
    """Bottom-right aligned causal reference (query i attends to keys <= i + (Sk - Sq))."""
    sq, hq, d = q.shape
    sk = k.shape[0]
    qf = q.float().transpose(0, 1)
    kf = _rep(k.float(), hq).transpose(0, 1)
    vf = _rep(v.float(), hq).transpose(0, 1)
    s = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    qi = torch.arange(sq, device=DEV).view(1, sq, 1)
    ki = torch.arange(sk, device=DEV).view(1, 1, sk)
    s = s.masked_fill(ki > (qi + (sk - sq)), float("-inf"))
    p = s.softmax(-1)
    return torch.matmul(p, vf).transpose(0, 1).to(DT)


def _run_fa(q, k, v, scale, sk, psc, num_splits, causal):
    """Single-sequence paged forward through the low-level cute entrypoint."""
    sq, hq, d = q.shape
    hkv = k.shape[1]
    num_blocks = (sk + PAGE - 1) // PAGE
    # block 0 left unused so block_table indices start at 1 (mirrors vLLM layout)
    kc = torch.zeros(num_blocks + 1, PAGE, hkv, d, device=DEV, dtype=DT)
    vc = torch.zeros(num_blocks + 1, PAGE, hkv, d, device=DEV, dtype=DT)
    for t in range(sk):
        kc[1 + t // PAGE, t % PAGE] = k[t]
        vc[1 + t // PAGE, t % PAGE] = v[t]
    cu_q = torch.tensor([0, sq], device=DEV, dtype=torch.int32)
    seqused_k = torch.tensor([sk], device=DEV, dtype=torch.int32)
    block_table = torch.arange(1, 1 + num_blocks, device=DEV, dtype=torch.int32).reshape(1, -1)
    dynamic_causal = torch.tensor([psc], device=DEV, dtype=torch.int32)
    out, _ = _flash_attn_fwd(
        q, kc, vc,
        cu_seqlens_q=cu_q,
        seqused_k=seqused_k,
        max_seqlen_q=sq,
        max_seqlen_k=sk,
        page_table=block_table,
        softmax_scale=scale,
        causal=causal,
        dynamic_causal=dynamic_causal,
        num_splits=num_splits,
        return_lse=True,
    )
    return out.reshape(sq, hq, d)


def _rel_err(out, ref):
    return ((out.float() - ref.float()).norm() / ref.float().norm()).item()


# Sq < Sk is the regression-triggering shape; include num_splits > 1 for split-KV.
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(256, 512), (256, 2048), (128, 1024), (1, 512)])
@pytest.mark.parametrize("num_splits", [1, 2, 4])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal_compile", [False, True])
def test_dynamic_causal_bidirectional_splitkv(
    seqlen_q, seqlen_k, num_splits, head_dim, causal_compile
):
    """psc == 0 must match a full bidirectional reference for any num_splits,
    regardless of the kernel's causal compile flag (causal_compile=False is the
    path vLLM uses and that previously deadlocked under split-KV)."""
    torch.manual_seed(0)
    q = torch.randn(seqlen_q, 16, head_dim, device=DEV, dtype=DT)
    k = torch.randn(seqlen_k, 2, head_dim, device=DEV, dtype=DT)
    v = torch.randn(seqlen_k, 2, head_dim, device=DEV, dtype=DT)
    scale = head_dim ** -0.5

    out = _run_fa(q, k, v, scale, seqlen_k, psc=0, num_splits=num_splits, causal=causal_compile)
    torch.cuda.synchronize()
    err = _rel_err(out, _ref_bidirectional(q, k, v, scale))
    assert err < 2e-2, f"bidirectional psc=0 rel_err={err:.3e} (split double-counting regression)"


@pytest.mark.parametrize("seqlen_q,seqlen_k", [(256, 512), (256, 2048), (128, 1024)])
@pytest.mark.parametrize("num_splits", [1, 2, 4])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_dynamic_causal_causal_splitkv(seqlen_q, seqlen_k, num_splits, head_dim):
    """psc != 0 with the kernel compiled causal must match a bottom-right causal
    reference and stay consistent across split counts (no regression on the
    causal path from the bidirectional fix)."""
    torch.manual_seed(0)
    q = torch.randn(seqlen_q, 16, head_dim, device=DEV, dtype=DT)
    k = torch.randn(seqlen_k, 2, head_dim, device=DEV, dtype=DT)
    v = torch.randn(seqlen_k, 2, head_dim, device=DEV, dtype=DT)
    scale = head_dim ** -0.5

    out = _run_fa(q, k, v, scale, seqlen_k, psc=1, num_splits=num_splits, causal=True)
    torch.cuda.synchronize()
    err = _rel_err(out, _ref_causal(q, k, v, scale))
    assert err < 2e-2, f"causal psc=1 rel_err={err:.3e}"
