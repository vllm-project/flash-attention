# FA4 CuTe Tensor Naming & Layout Annotation Guide

## Naming Convention

FA4 (Flash Attention 4, Blackwell/SM100) uses a systematic prefix-based naming
scheme for CuTe tensors.

### Unpartitioned tensors: `{Loc}{Name}`

Single-letter prefix, no thread partitioning applied.

| Loc | Memory | Example | Decode                     |
|-----|--------|---------|----------------------------|
| `m` | gmem   | `mQ`    | Q in global memory         |
| `s` | smem   | `sQ`    | Q in shared memory         |
| `g` | gmem   | `gO`    | O tile in global memory    |

### Partitioned tensors: `t{Part}{Loc}{Name}`

Partitioned by an MMA or copy operation.

```
t  {Part}  {Loc}  {Name}
│    │       │      │
│    │       │      └─ tensor name: Q, K, S, P, V, O, Scale, ...
│    │       └─ location: r=register, t=tmem, s=smem, c=coordinate(identity)
│    └─ partitioner: S=QK MMA, O=PV MMA
└─ "threaded" — partitioned by some tiled operation
```

| Loc | Location                | Example  | Decode                                |
|-----|-------------------------|----------|---------------------------------------|
| `r` | registers               | `tSrQ`   | Q register fragment, QK MMA partition |
| `t` | tmem                    | `tStS`   | S in tmem, QK MMA C-output partition  |
| `s` | smem                    | `tOsO`   | O in smem, PV MMA partition           |
| `c` | coordinate (identity)   | `tScS`   | S index-space coords, QK MMA partition|

| Part | Partitioner  | Example  | Decode                         |
|------|-------------|----------|--------------------------------|
| `S`  | QK MMA      | `tSrK`   | K fragment from QK MMA         |
| `O`  | PV MMA      | `tOrV`   | V fragment from PV MMA         |

### Suffix: copy direction

Appended when a tensor is a source or destination of a copy operation.

| Suffix | Direction          | Example      | Decode                            |
|--------|--------------------|--------------|-----------------------------------|
| `_t2r` | tmem → register    | `tStS_t2r`   | tmem S, source side of t→r copy   |
| `_r2t` | register → tmem    | `tStP_r2t`   | tmem P, dest side of r→t copy     |
| `_r2s` | register → smem    | `tOsO_r2s`   | smem O, dest side of r→s copy     |

### Quick reference

| Name       | Shape                               | Codomain      |
|------------|-------------------------------------|---------------|
| `tSrQ`     | `(MMA, MMA_M, MMA_K, STAGE)`       | reg_idx       |
| `tSrK`     | `(MMA, MMA_N, MMA_K, STAGE)`       | reg_idx       |
| `tStS`     | `((M, N), 1, 1, STAGE)`            | tmem_off      |
| `tScS`     | `((M, N), 1, 1)`                   | (m_idx,n_idx) |
| `tOrP`     | `(MMA, MMA_M, MMA_K, STAGE)`       | reg_idx       |
| `tOrV`     | `(MMA, MMA_N, MMA_K, STAGE)`       | reg_idx       |
| `tOtO`     | `((M, N), 1, 1, STAGE)`            | tmem_off      |
| `tStS_t2r` | `(CPY, CPY_M, CPY_K)`              | tmem_off      |
| `tSrS_t2r` | `(CPY, CPY_M, CPY_K)`              | reg_idx       |

## Concrete Layouts (head_dim=128, BF16, S\<3,4,3\> swizzle)

### QK MMA: S = Q × K^T (standard)

```python
# -- smem layouts --
sQ_layout = make_smem_layout_a(mma_qk, (128,128,128), bf16, 2)    # K-major S<3,4,3>
sK_layout = make_smem_layout_b(mma_qk, (128,128,128), bf16, 4)    # K-major S<3,4,3>
# concrete sQ: ((128,16),1,(4,2),2):((64,1),0,(16,8192),16384)
# concrete sK: ((128,16),1,(4,2),4):((64,1),0,(16,8192),16384)

# -- smem tensors --
sQ = storage.sQ.get_tensor(sQ_layout)                             # (MMA, MMA_M, MMA_K, STAGE)
sK = storage.sK.get_tensor(sK_layout)                             # (MMA, MMA_N, MMA_K, STAGE)

# -- TMA descriptors --
tma_Q = make_tiled_tma_atom_A(mQ, sQ_layout, ...)
tma_K = make_tiled_tma_atom_B(mK, sK_layout, ...)

# -- MMA fragments --
thr_mma_qk = tiled_mma_qk.get_slice(tidx)
tSrQ = thr_mma_qk.make_fragment_A(sQ)                             # (MMA, MMA_M, MMA_K, STAGE)
tSrK = thr_mma_qk.make_fragment_B(sK)                             # (MMA, MMA_N, MMA_K, STAGE)

# -- tmem accumulator --
tStS = thr_mma_qk.make_fragment_C(...)                             # (MMA, MMA_M, MMA_N, STAGE)
# concrete: ((128,128),1,1,1):((65536,1),0,0,0)
# M=q stride-65536 (tmem rows), N=kv stride-1 (tmem cols)
```

### PV MMA: O = P × V (standard, P from tmem)

```python
# -- smem layout (V only — P comes from tmem in standard path) --
sV_layout = make_smem_layout_b(mma_pv, (128,128,128), bf16, 4)    # K-major S<3,4,3>

# -- MMA fragments --
thr_mma_pv = tiled_mma_pv.get_slice(tidx)
tOrP = thr_mma_pv.make_fragment_A(tP)                             # (MMA, MMA_M, MMA_K, STAGE)
tOrV = thr_mma_pv.make_fragment_B(sV)                             # (MMA, MMA_N, MMA_K, STAGE)

# -- tmem accumulator --
tOtO = thr_mma_pv.make_fragment_C(...)                             # (MMA, MMA_M, MMA_N, STAGE)
# concrete: ((128,128),1,1,1):((65536,1),0,0,0)
# M=q stride-65536 (tmem rows), N=hdim stride-1 (tmem cols)
```

### Softmax tmem partition (128 threads = 4 warps × 32 lanes)

```python
tmem_load_atom = make_copy_atom(Ld32x32bOp(Rep(32)), F32)
thr_tmem_load  = make_tmem_copy(tmem_load_atom, tSAcc).get_slice(tidx)

tStS_t2r = thr_tmem_load.partition_S(tSAcc)                       # (CPY, CPY_M, CPY_K)
tSrS_t2r = make_fragment(thr_tmem_load.partition_D(...).shape, F32)  # (CPY, CPY_M, CPY_K)
# concrete src: (((32,32),1),1,4):(((65536,1),0),0,32)
# concrete dst: ((32,1),1,4) — 128 F32 per thread
#
# Thread mapping (lane l, warp w):
#   lane l  → row l  (each lane owns one M-row)
#   4 reps  → cover all 128 N-columns
#   In standard kernel: each thread has 1 query's full row → row-wise softmax

tStP_r2t = thr_tmem_store.partition_D(...)                         # (CPY, CPY_M, CPY_K)
# concrete: (((16,32),1),1,4):(((1,65536),0),0,16)
```

## CuTe API Reference

### A/B operand pipeline: smem layout → TMA → fragment

The A and B operands each follow the same pipeline. Everything is wired through
a shared `_thrfrg` permutation that the MMA atom defines.

### Constructing tiled_mma and mma_tiler

```python
# ── tiled_mma: wraps an MMA atom with thread tiling ─────────────────────
tiled_mma = make_trivial_tiled_mma(
    ab_dtype,          # e.g. BFloat16 — determines atom K (16 for BF16, 32 for FP8)
    a_major_mode,      # OperandMajorMode.K or .MN — which dim is stride-1 in smem
    b_major_mode,      # OperandMajorMode.K or .MN
    acc_dtype,         # e.g. Float32
    cta_group,         # CtaGroup.ONE or .TWO (1-CTA or 2-CTA instructions)
    mma_tiler_mn,      # (M, N) — instruction tile shape, e.g. (128, 128)
)
# Internally creates MmaOp with shape (M, N, K) where K is derived from dtype.

# ── mma_tiler: CTA tile shape for partitioning ──────────────────────────
mma_inst_k = cute.size(tiled_mma.shape_mnk, mode=[2])    # atom K, e.g. 16
mma_tiler = (BLK_M, BLK_N, mma_inst_k * num_k_tiles)    # e.g. (128, 128, 64)
#             ^^^^   ^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^
#             M-dim  N-dim  K-dim = atom_K × num_k_tiles_per_stage
# mma_tiler controls how much work each CTA does per GEMM iteration.
# make_smem_layout_a/b use it to determine the smem tile shape.
# local_tile uses it to partition gmem into CTA-sized tiles.
```

### A/B operand pipeline

```python
# ── Step 1: smem layout ─────────────────────────────────────────────────
# (MMA, MMA_M, MMA_K, STAGE)
sA_layout = make_smem_layout_a(tiled_mma, mma_tiler, a_dtype, num_stages)
# Returns ComposedLayout: .outer = tiled shape+stride, .inner = swizzle
# Internally calls partition_shape_A to tile (BLK_M, BLK_K) into MMA-atom-sized
# pieces, then applies the swizzle atom and appends the pipeline stage dimension.

# ── Step 2: smem tensor ─────────────────────────────────────────────────
# (MMA, MMA_M, MMA_K, STAGE)
sA = smem.allocate_tensor(a_dtype, sA_layout.outer, swizzle=sA_layout.inner)

# ── Step 3: TMA descriptor (gmem → smem) ────────────────────────────────
tma_A = make_tiled_tma_atom_A(mA, sA_layout, mma_tiler, tiled_mma, ...)
# Uses _thrfrg_A internally to match smem layout.

# ── Step 4: MMA fragment (smem → registers) ─────────────────────────────
# (MMA, MMA_M, MMA_K, STAGE)
tCrA = tiled_mma.make_fragment_A(sA)

# Select a stage for the GEMM:
tCrA_i = tCrA[_,_,_,stage_idx]                           # (MMA, MMA_M, MMA_K)
```

The same pattern applies for B with `make_smem_layout_b`, `make_tiled_tma_atom_B`,
`make_fragment_B`. All use `_thrfrg_B` internally.

**Critical invariant:** `make_smem_layout_a` + `make_tiled_tma_atom_A` +
`make_fragment_A` all use the same `_thrfrg_A` permutation. Never mix A/B
helpers (e.g., A-layout with B-TMA) — the data will be wrong.

`_thrfrg_A` and `_thrfrg_B` are **symmetric** when M=N and both operands have
the same major mode. This is why A/B operands can be swapped with a symmetric
tiler (e.g., (128,128,128) with both K-major).

### C accumulator pipeline

```python
# (MMA, MMA_M, MMA_N)
acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
# (MMA, MMA_M, MMA_N, STAGE)
tCtAcc = tiled_mma.make_fragment_C(cute.append(acc_shape, num_stages))
# STAGE is optional — cute.append omitted for stageless accumulators.
```

### Partitioning gmem tiles (from dense_gemm_persistent.py)

```python
# (bM, bK, RestM, RestK, RestL)
gA_mkl = cute.local_tile(mA_mkl, cute.slice_(mma_tiler, (None, 0, None)), ...)
# (bN, bK, RestN, RestK, RestL)
gB_nkl = cute.local_tile(mB_nkl, cute.slice_(mma_tiler, (0, None, None)), ...)
# (bM, bN, RestM, RestN, RestL)
gC_mnl = cute.local_tile(mC_mnl, cute.slice_(mma_tiler, (None, None, 0)), ...)

thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
# (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
tCgA = thr_mma.partition_A(gA_mkl)
# (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
tCgB = thr_mma.partition_B(gB_nkl)
# (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
tCgC = thr_mma.partition_C(gC_mnl)
```

### Copy partition functions

```python
thr_copy = tiled_copy.get_slice(tidx)
src_part = thr_copy.partition_S(src_tensor)           # (CPY, CPY_M, CPY_K, ...)
dst_part = thr_copy.partition_D(dst_tensor)           # (CPY, CPY_M, CPY_K, ...)
# Shapes depend on the copy atom.
```

Note: CUTLASS examples use the naming convention `tAsA` (copy-A partitioner,
smem, tensor-A) for copy partitions. FA4 uses `tStS_t2r` style instead — see
the naming section above.

### Blackwell tmem copy: Ld32x32bOp(Repetition(R))

Loads an R×32 block from tmem. With 128 threads (4 warps × 32 lanes):
- Each warp of 32 threads reads 32 columns (one per lane)
- `Repetition(R)` → each thread reads R rows from its column
- With R=32 and tiling reps: covers the full (128, 128) tmem accumulator

### make_smem_layout_a vs make_smem_layout_b

| Helper               | Operand | Permutation  |
|----------------------|---------|--------------|
| `make_smem_layout_a` | A (M×K) | `_thrfrg_A`  |
| `make_smem_layout_b` | B (N×K) | `_thrfrg_B`  |

Both return a `ComposedLayout`:
- `.outer` = tiled shape+stride
- `.inner` = swizzle function (e.g., S<3,4,3> = 128B swizzle, 32B atomicity)

### Major mode

| Major mode | Stride-1 dim | Smem layout effect                    | Use case              |
|------------|-------------|---------------------------------------|-----------------------|
| K-major    | K           | K_mma dimension contiguous in smem     | Q, K, P               |
| MN-major   | M or N      | Spatial dimension contiguous; interleaved layout | V^T (hdim stride-1) |

K-major is the default for attention operands. MN-major produces a different
smem layout shape — note the `(64,2)` interleave in sVt's M-mode:
```
K-major:  ((128,16),1,(4,2),STAGE)   — M=128 contiguous block
MN-major: (((64,2),16),1,8,STAGE)    — M=(64,2) interleaved
```

## Blackwell (SM100) Tensor Core Reference

### tcgen05.mma instruction shapes

The MMA atom shape is `(M, N, K)` where M and K are fixed per dtype, N is variable.

| Dtype           | Op class         | Atom K | Acc type | M           | N                  |
|-----------------|------------------|--------|----------|-------------|--------------------|
| FP16 / BF16     | `MmaF16BF16Op`   | 16     | FP16/FP32| 64, 128, 256| 8–256 (mult of 8)  |
| TF32            | `MmaTF32Op`      | 8      | FP32     | 64, 128, 256| 8–256 (mult of 8)  |
| INT8 / UINT8    | `MmaI8Op`        | 32     | INT32    | 64, 128, 256| 8–256 (mult of 8)  |
| FP8 (E4M3/E5M2) | `MmaFP8Op`      | 32     | FP16/FP32| 64, 128, 256| 8–256 (mult of 8)  |
| MX FP8 (block-scaled)| `MmaMXF8Op` | 32     | FP32     | 64, 128, 256| 8–256 (mult of 8)  |
| MX FP4 (block-scaled)| `MmaMXF4Op` | 64     | FP32     | 64, 128, 256| 8–256 (mult of 8)  |

- **M** is the fixed dimension (64, 128, or 256 only). Accumulator C has M rows in tmem.
- **N** is the variable dimension (any multiple of 8 from 8 to 256). Can be overridden in
  the instruction descriptor to reduce compute for small tiles.
- **K** is the reduction dimension per MMA instruction. Determined by dtype.
- `make_trivial_tiled_mma(ab_dtype, ..., (M, N))` sets K automatically from dtype.

### Operand sources

| Source | Where A comes from | Notes |
|--------|-------------------|-------|
| `SMEM` | Shared memory     | Default. Both A and B read from smem via TMA. |
| `TMEM` | Tensor memory     | A can come from tmem (e.g., P in standard PV MMA). B is always smem. |

### CTA groups

| Group | Threads | Use case |
|-------|---------|----------|
| `ONE` | 1 CTA   | Standard. 128-thread warpgroup. |
| `TWO` | 2 CTAs  | Two SMs cooperate on one tile. M=64 uses this. |

## Key Relationships

```
Standard kernel:
  QK MMA:  S[q,kv]    = Q[A,smem] × K^T[B,smem]  → tStS in tmem
  Softmax: tStS ──tmem load──► registers ──row-wise max/sum──► exp2
           ──convert──► P ──tmem store──► tStP in tmem (A-operand for PV)
  PV MMA:  O[q,hdim]  = P[A,tmem] × V[B,smem]    → tOtO in tmem

Pipeline (all variants):
  MMA produces S → softmax consumes S, produces P →
  MMA consumes P, produces O
  Sync: pipeline_s_p_o (mbarrier with acquire/release semantics)
```
