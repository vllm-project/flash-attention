# swap_AB Layout Design

## Overview

`swap_AB=True` transposes the flash attention computation so that queries go on
the MMA N dimension (variable, can be small) instead of M (fixed at 128). This
reduces wasted compute for decode (Q=1).

```
Standard:  S[M=q, N=kv]    = Q[A] x K^T[B]  ->  O[M=q, N=hdim]   = P[A,tmem] x V[B]
swap_AB:   S^T[M=kv, N=q]  = K[A] x Q^T[B]  ->  O^T[M=hdim, N=q] = V^T[A] x P[B,smem]
```

## Layout Flow

### 1. tiled_mma construction

```python
# Standard:                                    swap_AB:
make_trivial_tiled_mma(                        make_trivial_tiled_mma(
    q_dtype, K, K, acc, ONE, (128,128))            k_dtype, K, K, acc, ONE, (128,128))
#   A=Q(K-major), B=K(K-major)                #   A=K(K-major), B=Q(K-major)

make_trivial_tiled_mma(                        make_trivial_tiled_mma(
    v_dtype, K, MN, acc, ONE, (128,128),           v_dtype, MN, K, acc, ONE, (128,128))
    a_source=TMEM)                             #   A=V^T(MN-major), B=P(K-major), both SMEM
#   A=P(K-major,TMEM), B=V(MN-major)
```

Both use mma_tiler = (128, 128, 128). N-override in the instruction descriptor
reduces actual compute to (128, q_padded, 128).

### 2. smem layouts

```python
# Standard:                                    swap_AB:
sQ = make_smem_layout_a(qk, ..., q_stage)      sK = make_smem_layout_a(qk, ..., kv_stage)
sK = make_smem_layout_b(qk, ..., kv_stage)     sQ = make_smem_layout_b(qk, ..., q_stage=1)
tP = make_smem_layout_a(pv, ..., s_stage)       sV = make_smem_layout_a(pv, ..., kv_stage)
sV = make_smem_layout_b(pv, ..., kv_stage)      sP = make_smem_layout_b(pv, ..., 1)
```

When M=N and both K-major, `_thrfrg_A == _thrfrg_B` (symmetric). When M≠N,
the layouts differ: A-layout has M rows, B-layout has N rows. For swap_AB with
N=q_padded<128, the B-layout (Q) is smaller: `((N,16),1,(4,2),STAGE)` vs
A-layout (K) `((128,16),1,(4,2),STAGE)`.

With native N=q_padded (no N-override):
- sQ (B): `((q_padded,16),1,(4,2),1)` — only q_padded rows, saves smem
- sK (A): `((128,16),1,(4,2),kv_stage)` — standard 128 rows
- tmem acc: `((128, q_padded),1,1,1)` — only q_padded columns

Current approach uses N-override via `declare_ptx_idesc(n_override=q_padded)`:
- All layouts use (128,128). N-override in idesc reduces MMA compute to q_padded.
- Wastes smem and tmem columns but avoids layout asymmetry complexity.

Native N=q_padded tiler also works (verified) but gives same perf since
smem/tmem savings are small relative to kernel launch overhead.

### 3. TMA atoms

```python
# swap_AB swaps make_tiled_tma_atom_A <-> _B
make_tma_Q = make_tiled_tma_atom_B  # Q is B in swap_AB
make_tma_KV = make_tiled_tma_atom_A  # K,V are A in swap_AB
```

### 4. MMA fragments (kernel)

```python
# Standard:                                    swap_AB:
tSrQ = make_fragment_A(sQ)                     tSrK = make_fragment_A(sK)
tSrK = make_fragment_B(sK)                     tSrQ = make_fragment_B(sQ)
tOrV = make_fragment_B(sV)                     tOrV = make_fragment_A(sV)

# P fragment:
tOrP = make_fragment_A(tP)  # from tmem        tOrP = make_fragment_B(sP)[_,_,_,0]  # from smem
```

### 5. GEMM callables (mma function)

```python
# swap_AB: bind B (static), pass A (runtime)
gemm_Si: bind B=Q, runtime A=K               (standard: bind A=Q, runtime B=K)
gemm_Pi: bind B=P(sP), runtime A=V^T         (standard: bind A=P(tmem), runtime B=V)

# N-override via declare_ptx_idesc(op, n_override=q_padded)
# Reduces QK compute from 128x128 to 128x16
# Reduces PV compute from 128x128 to 128x16
```

### 6. tmem accumulator layout

Both standard and swap_AB use tmem ((128, 128), 1, 1, STAGE) : ((65536, 1), ...).

```
Standard:  tStS: M=q rows, N=kv cols       tOtO: M=q rows, N=hdim cols
swap_AB:   tStS: M=kv rows, N=q cols       tOtO: M=hdim rows, N=q cols
```

With N-override=16, only columns 0..15 of the 128 are computed. Columns 16..127
remain zero (from zero_init).

### 7. Softmax (softmax_step)

Thread mapping from Ld32x32bOp(Rep(32)) on tmem ((128, 128)):((65536, 1)):
- N=q is stride-1 (contiguous), M=kv is stride-65536
- Each thread gets values along N (queries) for one M position (kv)
- thread_idx → kv position, tSrS_t2r[j] → S^T[kv=tidx, q=j]

Reduction for query j: reduce tSrS_t2r[j] across all 128 threads.
- Intra-warp: warp_reduce(tSrS_t2r[0], fmax/add) across 32 lanes
- Cross-warp: smem scratch[warp_idx] + barrier + read all 4

Only 2 barriers per kv-block (optimized from 5):
- Barrier 1: after max smem write → all threads read max, compute exp2, write P + sum
- Barrier 2: fence sP writes + sum smem write → signal MMA, read sum

P write: sP_2d[q=0, kv=thread_idx] = v_dtype(exp2_val)
- Each thread writes one P value for query 0 at its kv position

### 8. Correction (correction_loop)

sScale[tidx] stores per-thread acc_scale. In swap_AB, all 128 threads have
the same acc_scale (reduced to a single value per query). The correction warp
reads sScale[tidx] and applies it. Since all threads wrote the same value, the
correction applies uniformly. No code change needed.

### 9. Epilogue (correction_epilogue)

tmem O^T has M=hdim on rows, N=q on columns.
Standard epilogue reads tmem row-by-row -> sO -> gmem.
For swap_AB: row i of tmem = hdim position i, NOT query i.

Fix: in correction_epilogue, use coordinate tensor to get (M=hdim, N=q) indices,
then write to sO[q, hdim] (swapped) instead of sO[hdim, q]:
```python
for j in range(frg_size):
    hdim_idx = tOcO_t2r_i[j][0]  # M = hdim
    q_idx = tOcO_t2r_i[j][1]      # N = query
    sO[q_idx, hdim_idx] = o_dtype(tOrO_frg[j])
```

### 10. GQA (pack_gqa)

**TODO**: pack_gqa reorganizes Q heads to share KV heads. With swap_AB, Q is the
B operand. pack_gqa modifies mQ's layout before TMA. Need to verify that
pack_gqa_layout works correctly when Q is B instead of A.

The GQA diff (0.36) suggests the pack_gqa M-tiling conflicts with the swap_AB
N-mapping. Investigation needed.

### 11. Causal masking

Causal masking applies `mask_fn(tSrS_t2r, n_block=n_block)` which masks based
on position. In swap_AB, M=kv and N=q — the mask needs to check (kv_pos, q_pos)
instead of (q_pos, kv_pos). The mask function uses coordinate tensors from the
QK MMA partition, which already reflect the swapped M/N. Should work without
changes if the mask function uses the coordinate values correctly.

### 12. Split KV

Split KV partitions the KV sequence across multiple CTAs. In swap_AB, KV is the
M dimension. The tile scheduler assigns m_blocks for queries. With swap_AB, the
meaning of m_block shifts. Need to verify the scheduler still works correctly.
