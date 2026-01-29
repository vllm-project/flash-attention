# FlashAttention Upstream Sync - Continuation Guide

## Task Overview

Sync the vLLM fork of flash-attention (`origin/main`) with the upstream Dao-AILab/flash-attention (`upstream/main`), preserving downstream-specific features and performance optimizations.

## Repository Layout

- **Workspace**: `/home/LucasWilkinson/code/flash-attention`
- **Downstream (vLLM fork)**: `origin/main` 
- **Upstream (Dao-AILab)**: `upstream/main`
- **Last upstream sync base**: `d836a6bf09bf3838c6e71c9cf675b3708fea0d71`
- **Current sync branch**: `sync/upstream-main-20260121`

## Key Downstream Features to Preserve

1. **PR #78 - Attention Sinks Performance Boost**: 
   - Introduces `n_offset` approach for local attention optimization
   - Modifies `block.h` to return `cute::tuple<int, int, int>` (includes `n_offset`)
   - Shifts KV pointers directly instead of computing `n_block_min`

2. **PR #70 - Varlen Combine Scheduler**:
   - Changes to `flash_fwd_combine_kernel.h` and `flash_fwd_combine_launch_template.h`
   - Adds `int b` to scheduler structs

3. **Context Parallelism (CP)**:
   - `cp_world_size`, `cp_rank`, `cp_tot_seqused_k` parameters
   - `tot_seqlen_k` in `SeqlenInfo_t`

4. **PyTorch TORCH_LIBRARY bindings**:
   - `csrc/common/pytorch_shim.h` and `csrc/common/registration.h`
   - `mha_fwd_tuple` wrapper for correct return type

5. **vLLM-specific interface**:
   - `vllm_flash_attn/flash_attn_interface.py`
   - `hopper/flash_api_torch_lib.cpp` (now merged into `flash_api.cpp`)

## Current State

### Branch Status
```
sync/upstream-main-20260121 (tracking origin/sync/upstream-main-20260121)
```

### Recent Commits
```
997fc13 fix compile error
35756f5 restore changes  
2cf8a1f Merge remote-tracking branch 'upstream/main' into sync/upstream-main-20260121
```

### Files Modified vs origin/main (key hopper files)
- `hopper/block.h` - Reset to origin/main (has n_offset)
- `hopper/flash.h` - Modified
- `hopper/flash_api.cpp` - Modified (removed static_cast, aligned attention_chunk logic)
- `hopper/flash_api_torch_lib.cpp` - DELETED (merged into flash_api.cpp)
- `hopper/flash_fwd_combine_kernel.h` - Should match origin/main
- `hopper/flash_fwd_combine_launch_template.h` - Modified
- `hopper/flash_fwd_launch_template.h` - Modified (removed attention_chunk_divmod)
- `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` - Modified (removed attention_chunk_divmod)
- `hopper/mask.h` - Modified (removed attention_chunk_divmod)
- `hopper/tile_scheduler.hpp` - Modified

## Key Reconciliation Issue: `attention_chunk` vs `n_offset`

### The Conflict
- **Upstream** uses `attention_chunk_divmod` for attention chunking feature
- **Downstream (PR #78)** uses `n_offset` for local attention performance optimization
- These approaches are mutually exclusive in the current implementation

### Resolution Strategy (Current)
Remove `attention_chunk_divmod` from downstream and use the `n_offset` approach from PR #78:
- `block.h::get_n_block_min_max` returns `n_offset` instead of computing `n_block_min` for local
- `mask.h` - remove `attention_chunk_divmod` from constructor and masking logic
- `mainloop_fwd_sm90_tma_gmma_ws.hpp` - remove `attention_chunk` from `Params` and `Arguments`
- `flash_fwd_launch_template.h` - remove `attention_chunk` from mainloop_args

### Files Already Updated (User Changes)
1. `hopper/mask.h` - Removed `attention_chunk_divmod` member and constructor param
2. `hopper/flash_fwd_launch_template.h` - Removed `attention_chunk` from args
3. `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` - Removed `attention_chunk` from Params/Args

## Build & Test Commands

### Build vLLM with flash-attention
```bash
cd /home/LucasWilkinson/code/vllm
source /mnt/data/engine/lwilkinson/vllm/.venv/bin/activate
export VLLM_FLASH_ATTN_SRC_DIR=/home/LucasWilkinson/code/flash-attention
VLLM_DISABLE_SCCACHE=1 python setup.py build_ext --inplace
```

### Run Attention Backend Tests
```bash
cd /home/LucasWilkinson/code/vllm
source /mnt/data/engine/lwilkinson/vllm/.venv/bin/activate
VLLM_DISABLED_BACKENDS=flashinfer chg run -g 1 -- \
  /mnt/data/engine/lwilkinson/vllm/.venv/bin/python -m pytest \
  tests/v1/attention/test_attention_backends.py -v
```

### Run MLA Backend Tests
```bash
VLLM_DISABLED_BACKENDS=flashinfer chg run -g 1 -- \
  /mnt/data/engine/lwilkinson/vllm/.venv/bin/python -m pytest \
  tests/v1/attention/test_mla_backends.py -v
```

## Remaining Work

### 1. Verify Build Compiles
After removing `attention_chunk_divmod`, rebuild and fix any remaining compile errors.

**Files still referencing `attention_chunk`** (may need review):
- `hopper/flash_api.cpp` - API parameter (keep, but don't use in divmod)
- `hopper/flash.h` - `Flash_fwd_params.attention_chunk` member
- `hopper/mainloop_bwd_sm80.hpp` - backward pass
- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp` - backward pass  
- `hopper/flash_bwd_launch_template.h` - backward pass
- `hopper/flash_api_stable.cpp` - stable API
- `hopper/flash_attn_interface.py` - Python interface
- `hopper/test_*.py` - tests

### 2. Ensure Key Files Match origin/main
These should match downstream exactly:
- `hopper/block.h`
- `hopper/flash_fwd_combine_kernel.h`  
- `hopper/flash_fwd_combine_launch_template.h`
- `csrc/common/pytorch_shim.h`
- `csrc/common/registration.h`

### 3. Verify flash_api.cpp Alignment
- Function signatures should use `int64_t` (PyTorch requirement)
- No unnecessary `static_cast<int>` for standard parameters
- `attention_chunk` should be passed through but not used in `attention_chunk_divmod`

### 4. Test Local Attention
The key test case that validates PR #78 fix:
```bash
pytest tests/entrypoints/llm/test_accuracy.py::test_lm_eval_accuracy_v1_engine[google/gemma-3-1b-it]
```

### 5. Generate Final Patch
Once validated:
```bash
cd /home/LucasWilkinson/code/flash-attention
git diff upstream/main > /tmp/flash-attn-upstream.patch
```

## Reference PRs

- **PR #78**: [Attention Sinks Perf Boost](https://github.com/vllm-project/flash-attention/commit/2d3b7508f67ad976f781e2042ace676419dd78dd)
- **PR #70**: Varlen combine scheduler
- **PR #93**: Context parallelism (broke local attention, fixed by reverting to n_offset approach)

## Summary of `attention_chunk` Handling

The `attention_chunk` parameter exists in:
- `Flash_fwd_params.attention_chunk` (in `flash.h`) - **KEEP** as API field
- `flash_api.cpp` - **KEEP** as function parameter, assigned to `params.attention_chunk`
- `flash_attn_interface.py` - **KEEP** in Python interface

But `attention_chunk_divmod` logic is **REMOVED** from:
- `block.h` - No `attention_chunk_divmod` parameter
- `mask.h` - No `attention_chunk_divmod` member or masking logic
- `mainloop_fwd_sm90_tma_gmma_ws.hpp` - No `attention_chunk` in Params/Arguments
- `flash_fwd_launch_template.h` - No `attention_chunk` in mainloop_args

This means `attention_chunk` is accepted by the API but has no effect in the forward kernel. 
The `n_offset` approach from PR #78 handles local attention instead.

## Notes

- FlashInfer tests fail due to separate environment issue (`PagedParams` missing `k_page_stride`/`v_page_stride`) - use `VLLM_DISABLED_BACKENDS=flashinfer` to skip
- The `attention_chunk` feature from upstream is NOT used by vLLM currently
- Port 3333 preferred over 8000 for server tests
- Always check for zombie processes before GPU tests

## Quick Resume Commands

```bash
# Navigate to workspace
cd /home/LucasWilkinson/code/flash-attention

# Check current branch
git branch -v

# See modified files  
git status

# Diff with downstream baseline
git diff origin/main -- hopper/

# Diff with upstream
git diff upstream/main -- hopper/
```
