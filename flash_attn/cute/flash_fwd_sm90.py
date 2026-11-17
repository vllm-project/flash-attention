# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM90 (Hopper) forward pass for flash attention, extracted from flash_fwd.py.

import math
from types import SimpleNamespace
from typing import Callable, Literal, NamedTuple, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Float16, BFloat16, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.base_dsl.arch import Arch

from quack import copy_utils
from quack import layout_utils
from quack import sm90_utils

from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from cutlass.utils import TensorMapManager, TensorMapUpdateMode
from flash_attn.cute import utils
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.softmax import Softmax, apply_score_mod_inner, load_learnable_sink
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.block_sparse_utils import (
    produce_block_sparse_loads,
    consume_block_sparse_loads,
)
from flash_attn.cute import pipeline as pipeline_custom
from flash_attn.cute.pack_gqa import PackGQA, pack_gqa_layout, make_packgqa_tiled_tma_atom
from flash_attn.cute.paged_kv import PagedKVManager
from flash_attn.cute.named_barrier import NamedBarrierFwd, NamedBarrierFwdSm90Fp8KV
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.tile_scheduler import (
    SchedulingMode,
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
)
from cutlass.cute import FastDivmodDivisorV2

from flash_attn.cute.flash_fwd import FlashAttentionForwardBase
from flash_attn.cute.utils import AuxData
from flash_attn.cute.config import Sm90Registers


class FlashAttentionForwardSm90(FlashAttentionForwardBase):
    def __init__(
        self,
        *args,
        intra_wg_overlap: bool = True,
        mma_pv_is_rs: bool = True,
        paged_kv_non_tma: bool = False,
        is_split_kv: bool = False,
        kv_dtype=None,
        fp8_kv_dequant: bool = False,
        registers: Sm90Registers,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert self.output_quant_key is None, (
            f"Fused quant output not implemented for {type(self).__name__}"
        )
        self.num_mma_regs, self.num_producer_regs = registers.mma, registers.producer
        self.intra_wg_overlap = intra_wg_overlap
        # Match the FA3 FP8 path: keep softmax probabilities in registers for PV.
        # SM90 FP8 needs a different accumulator-to-A-fragment mapping than FP16.
        self.mma_pv_is_rs = mma_pv_is_rs
        self.buffer_align_bytes = 1024
        self.use_tma_KV = not paged_kv_non_tma
        self.is_split_kv = is_split_kv
        # SM90 FP8-MMA path: FP8 WGMMA sums below FP32, so PV is promoted per KV block.
        self.fp8_kv_dequant = fp8_kv_dequant
        # Split the FP8 QK sum in two; non-causal builds would serialize WGMMAs (ptxas C7511).
        self.qk_split = fp8_kv_dequant and (self.is_causal or self.is_local)
        assert not self.fp8_kv_dequant or self.mma_pv_is_rs, (
            "SM90 FP8 KV dequant requires RS-mode PV because its shared storage "
            "does not allocate sP"
        )
        self.kv_dtype = kv_dtype if kv_dtype is not None else self.dtype
        # Hopper FP8 WGMMA needs FP8 on both inputs. Q is quantized once per output
        # tile. P is scaled before its FP8 cast to preserve long-context softmax
        # values; the reciprocal is folded into final output normalization.
        self.fp8_q_scale = 1.0
        self.fp8_p_scale = 256.0
        # The P residual gets its own scale and PV pass, so a probability down to
        # 2^-23 of the row max survives instead of rounding to 0 below 2^-18.
        # |residual| <= 8 for p <= 1, so 8 * 32 = 256 stays below E4M3's 448.
        self.fp8_p_res_scale = 32.0
        # FP8 K is TMA'd straight into two sK slots; sStage holds only V.
        self.num_k_slots_fp8 = 2
        assert self.use_tma_KV or not (self.check_hdim_oob or self.check_hdim_v_oob), (
            "Paged KV does not support irregular head dim"
        )
        self.cluster_shape_mn = (1, 1)
        assert self.arch.is_family_of(Arch.sm_90a), "Only SM 9.x is supported"

    def _get_smem_layout_atom(self):
        sQ_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdim),
            self.dtype,
        )
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdimv
            ),
            self.dtype,
        )
        sO_layout_atom = sV_layout_atom
        if not self.mma_pv_is_rs:
            sP_layout_atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(
                    LayoutEnum.ROW_MAJOR, self.dtype, self.tile_n
                ),
                self.dtype,
            )
        else:
            sP_layout_atom = None
        return sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, sP_layout_atom

    def _get_tiled_mma(self):
        self.atom_layout_n = 2 if self.tile_hdim > 256 or self.tile_hdimv > 256 else 1
        mma_dtype = self.kv_dtype if self.fp8_kv_dequant else self.dtype
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            mma_dtype,
            mma_dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, self.atom_layout_n, 1),
            tiler_mn=(64, self.tile_n),
        )
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            mma_dtype,
            mma_dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K if self.fp8_kv_dequant else warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(
                self.tile_m // 64,
                self.atom_layout_n,
                1,
            ),  # Might need (1, 2, 1) for hdim 512
            tiler_mn=(64, min(256, self.tile_hdimv)),
            a_source=warpgroup.OperandSource.RMEM
            if self.mma_pv_is_rs
            else warpgroup.OperandSource.SMEM,
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)], self.buffer_align_bytes
            ]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sQV], 1024]
        cosize_sP = cute.cosize(self.sP_layout) if const_expr(self.sP_layout is not None) else 0
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
        # 1 stage * 2 for Q pipeline (full + empty), self.num_stages*2 for K, self.num_stages*2 for V,
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        if const_expr(self.fp8_kv_dequant):
            sQ_input_struct = cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(self.sQ_layout)],
                self.buffer_align_bytes,
            ]
            sQ_mma_struct = cute.struct.Align[
                cute.struct.MemRange[self.kv_dtype, cute.cosize(self.sQ_mma_layout)],
                self.buffer_align_bytes,
            ]
            sK_mma_struct = cute.struct.Align[
                cute.struct.MemRange[self.kv_dtype, cute.cosize(self.sK_layout)],
                self.buffer_align_bytes,
            ]
            sV_mma_struct = cute.struct.Align[
                cute.struct.MemRange[self.kv_dtype, cute.cosize(self.sV_mma_layout)],
                self.buffer_align_bytes,
            ]
            sStage_struct = cute.struct.Align[
                cute.struct.MemRange[self.kv_dtype, cute.cosize(self.sStage_layout)],
                self.buffer_align_bytes,
            ]
            mbar_ptr_Stage_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
            mbar_ptr_K_fp8_struct = cute.struct.MemRange[cutlass.Int64, self.num_k_slots_fp8 * 2]

            @cute.struct
            class SharedStorageQKVFp8Mma:
                mbar_ptr_Q: mbar_ptr_Q_struct
                mbar_ptr_Stage: mbar_ptr_Stage_struct
                mbar_ptr_K: mbar_ptr_K_fp8_struct
                sQ: sQ_input_struct
                sQ_mma: sQ_mma_struct
                sK: sK_mma_struct
                sV_mma: sV_mma_struct
                sStage: sStage_struct

            return SharedStorageQKVFp8Mma

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct

        @cute.struct
        class SharedStorageSharedQV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sQ: sQV_struct
            sK: sK_struct
            sP: sP_struct

        return SharedStorageQKV if const_expr(not self.Q_in_regs) else SharedStorageSharedQV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mDynamicCausal: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,  # (b_k, max_num_pages_per_seq)
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_data: AuxData = AuxData(),
        # FP8-KV: per-(batch, kv_head) f32 q/k/v descales (any may be None).
        descale_tensors=None,
        output_scale: Optional[cute.Tensor] = None,
        mCuTotalMBlocks: Optional[cute.Tensor] = None,
        mCuTotalSplitsMBlocks: Optional[cute.Tensor] = None,
        tensormap_workspace: Optional[
            cute.Tensor
        ] = None,  # (num_ctas, 16) Int64, one V descriptor per CTA
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout
        (except FP8 MMA: native fp16/bf16 Q, E4M3 K/V, native fp16/bf16 O):
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        """
        if const_expr(not self.fp8_kv_dequant):
            self._check_type(
                *(
                    t.element_type if t is not None else None
                    for t in (mQ, mK, mV, mO, mLSE, mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK)
                ),
                is_split_kv=self.is_split_kv,
            )
        else:
            # FP8-KV: K/V are fp8 (rejected by the symmetric _check_type). Q/O may each be
            # fp16 or bf16. Q and softmax P use compensated pairs of E4M3 values;
            # QK and PV run as FP8 WGMMA, which sums below FP32; PV is promoted per block.
            assert self.dtype == Float16, (
                "fp8_kv_dequant requires fp16 auxiliary conversion layouts"
            )
            assert mQ.element_type in (Float16, BFloat16), "fp8_kv_dequant expects fp16 or bf16 Q"
            if const_expr(self.is_split_kv):
                assert mO.element_type == Float32, (
                    "fp8_kv_dequant SplitKV expects Float32 O (partial accumulator)"
                )
            else:
                assert mO.element_type in (Float16, BFloat16), (
                    "fp8_kv_dequant expects fp16 or bf16 O"
                )

        # Q input / O output dtypes. q_dtype drives Q quantization and o_dtype drives the
        # epilogue cast. SplitKV writes fp32 partials via acc_O directly, so o_dtype stays
        # the compute dtype there (the combine kernel produces the final bf16/fp16 O).
        self.q_dtype = mQ.element_type
        if const_expr(self.fp8_kv_dequant):
            self.o_dtype = mO.element_type if const_expr(not self.is_split_kv) else self.dtype

        self.varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None

        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        Q_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ = layout_utils.select(mQ, Q_layout_transpose)
        num_splits = Int32(1)
        if const_expr(not self.is_split_kv):
            O_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
            LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        else:
            O_layout_transpose = (
                [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            )
            LSE_layout_transpose = [3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0]
            num_splits = mO.shape[0]
        mO = layout_utils.select(mO, O_layout_transpose)
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [layout_utils.select(t, KV_layout_transpose) for t in (mK, mV)]
        mLSE = (
            layout_utils.select(mLSE, LSE_layout_transpose)
            if const_expr(mLSE is not None)
            else None
        )

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_threads_per_warp_group = 128
        self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
        if const_expr(self.fp8_kv_dequant):
            assert self.num_wg_mma == 2, "SM90 d512 FP8 MMA requires exactly two MMA warp groups"
        assert self.num_wg_mma in [1, 2, 3]
        self.num_threads = self.num_threads_per_warp_group * (self.num_wg_mma + 1)
        self.num_producer_threads = 32
        self.num_Q_load_threads = self.num_threads_per_warp_group  # If not TMA_Q
        self.num_epilogue_threads = self.num_mma_threads
        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)

        self.use_scheduler_barrier = (
            (self.num_wg_mma >= 2 and self.tile_hdim <= 128)
            if const_expr(self.intra_wg_overlap)
            else (self.num_wg_mma == 2)
        )
        self.use_tma_Q = self.arch >= Arch.sm_90 and not (
            self.pack_gqa and self.tile_m % self.qhead_per_kvhead != 0
        )
        self.use_tma_O = self.use_tma_Q and not self.is_split_kv
        self.rescale_O_before_gemm = self.tile_hdimv > 128 and self.intra_wg_overlap
        # hdim > 256: two MMA warpgroups each read the whole sP tile in the PV GEMM.
        self.p_shared_across_wg = self.atom_layout_n == 2
        self._setup_attributes()
        # TODO: we prob don't need most of what's in _setup_attributes
        sK_dtype = self.kv_dtype if const_expr(self.fp8_kv_dequant) else mK.element_type
        sV_dtype = self.kv_dtype if const_expr(self.fp8_kv_dequant) else mV.element_type
        self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
            sm90_utils.make_smem_layout(dtype, LayoutEnum.ROW_MAJOR, shape, stage)
            for dtype, shape, stage in [
                # sQ holds the (possibly bf16) Q tensor as TMA'd; the fp16 compute view
                # is an in-place recast (both are 2 bytes) -- see mma().
                (self.q_dtype, (self.tile_m, self.tile_hdim), None),
                (
                    sK_dtype,
                    (self.tile_n, self.tile_hdim),
                    self.num_k_slots_fp8 if const_expr(self.fp8_kv_dequant) else self.num_stages,
                ),
                (sV_dtype, (self.tile_n, self.tile_hdimv), self.num_stages),
                # sO uses the output dtype (o_dtype); for splitkv o_dtype == compute dtype
                # (the fp32 partials are written from acc_O directly, not via sO).
                (self.o_dtype, (self.tile_m, self.tile_hdimv), None),
            ]
        ]
        self.sQ_mma_layout = None
        self.sV_mma_layout = None
        self.sStage_layout = None
        if const_expr(self.fp8_kv_dequant):
            self.sQ_mma_layout = sm90_utils.make_smem_layout(
                self.kv_dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_hdim), None
            )
            # Construct WGMMA-B directly as (output_dim, reduction_dim), K-major.
            self.sV_mma_layout = sm90_utils.make_smem_layout(
                self.kv_dtype,
                LayoutEnum.ROW_MAJOR,
                (self.tile_hdimv, self.tile_n),
                self.num_stages,
            )
            # One row-major staging tile for V TMA loads; K is TMA'd into sK.
            self.sStage_layout = sm90_utils.make_smem_layout(
                self.kv_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdim), 1
            )
        self.sP_layout = None
        if const_expr(not self.mma_pv_is_rs):
            self.sP_layout = sm90_utils.make_smem_layout(
                mV.element_type, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_n)
            )

        # Bound V loads by the logical sequence length (see below); decided here because the
        # shared storage needs a descriptor slot for it.
        self.ragged_tma_V = (
            self.use_tma_KV
            and mPageTable is None
            and (mCuSeqlensK is not None or mSeqUsedK is not None)
            and not (isinstance(mV.stride[0], int) and mV.stride[0] == 0)
        )
        if const_expr(self.ragged_tma_V):
            assert tensormap_workspace is not None, "ragged V TMA loads need a tensormap workspace"
        SharedStorage = self._get_shared_storage_cls()

        mQ_og, mO_og = mQ, mO
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)

        # TMA
        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()  # Might multicast
        gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
            ]
        }
        make_tiled_tma_atom_fn = (
            partial(make_packgqa_tiled_tma_atom, qhead_per_kvhead=self.qhead_per_kvhead, head_idx=2)
            if const_expr(self.pack_gqa)
            else cpasync.make_tiled_tma_atom
        )
        tma_atom_Q, tma_tensor_Q = None, None
        if const_expr(self.use_tma_Q):
            tma_atom_Q, tma_tensor_Q = make_tiled_tma_atom_fn(
                gmem_tiled_copy_Q,
                mQ_og if const_expr(self.pack_gqa) else mQ,
                self.sQ_layout,
                (self.tile_m, self.tile_hdim),  # No mcast
            )
        tma_atom_K, tma_tensor_K = None, None
        tma_atom_V, tma_tensor_V = None, None
        # Bound the V TMA load by the logical seqlen_k (cu_seqlens_k and/or seqused_k) so rows
        # in [seqlen_k, tile_n) arrive zero-filled instead of as whatever the buffer holds. P is
        # exactly zero there, but 0 * NaN = NaN in O = P @ V. K needs no bound: masked scores
        # become -inf before exp. The descriptor is the load-safe form of
        # flash_attn.cute.copy_utils (quack's ptr_shift form is store-only, and its wraparound
        # form needs rank <= 3, i.e. only the packed (t, d, h) layout); it covers both the packed
        # and the dense (s, d, h, b) layout. Paged KV is excluded (finite rows past seqused_k in
        # a page are a caller contract), as are broadcast (seq-stride 0) V views.
        # The load warp rewrites its CTA's copy of the V descriptor per work tile with the
        # sequence's row extent (tensormap.replace), so rows at or past seqlen_k are out of
        # bounds for TMA and arrive zero-filled; a descriptor with a per-launch extent cannot
        # do this (TMA prefetches past the footprint it is given, #3068).
        mV_gmem = mV if const_expr(self.ragged_tma_V) else None
        mV_tma = mV
        if const_expr(self.use_tma_KV):
            # FP8 K lands in sK and V in sStage (same layout); neither is widened.
            sK_tma_box = cute.select(
                self.sStage_layout if const_expr(self.fp8_kv_dequant) else self.sK_layout,
                mode=[0, 1],
            )
            sV_tma_box = cute.select(
                self.sStage_layout if const_expr(self.fp8_kv_dequant) else self.sV_layout,
                mode=[0, 1],
            )
            tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mK,
                sK_tma_box,
                (self.tile_n, self.tile_hdim),
                1,  # No mcast for now
            )
            tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mV_tma,
                sV_tma_box,
                (self.tile_n, self.tile_hdimv),
                1,  # No mcast for now
            )
        tma_atom_O, tma_tensor_O = None, None
        if const_expr(self.use_tma_O):
            mO_tma = mO_og if const_expr(self.pack_gqa) else mO
            if const_expr(self.varlen_q):
                mO_tma = copy_utils.create_ragged_tensor_for_tma(
                    mO_tma, ragged_dim=0, ptr_shift=True
                )
            tma_atom_O, tma_tensor_O = make_tiled_tma_atom_fn(
                gmem_tiled_copy_O,
                mO_tma,
                self.sO_layout,
                (self.tile_m, self.tile_hdimv),  # No mcast
            )
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            # TODO: dispatch to DynamicPersistentVarlenScheduler when appropriate
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = (
                SingleTileScheduler
                if const_expr(not self.is_causal or self.is_local)
                else SingleTileLPTScheduler
            )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            num_splits,
            cute.size(mK.shape[0])
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
            cu_total_m_blocks_ptr=mCuTotalMBlocks,
            cu_total_splits_m_blocks_ptr=mCuTotalSplitsMBlocks,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        if const_expr(self.fp8_kv_dequant):
            assert not tile_sched_args.is_persistent and (
                getattr(tile_sched_params, "scheduling_mode", SchedulingMode.STATIC)
                == SchedulingMode.STATIC
            ), "FP8 sK slot parity (_kslot) restarts per tile: one tile per CTA only"
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
            softmax_scale, self.score_mod
        )
        window_size_left = Int32(window_size_left) if window_size_left is not None else None
        window_size_right = Int32(window_size_right) if window_size_right is not None else None
        fastdiv_mods = utils.compute_fastdiv_mods(
            mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_data.tensors, mPageTable
        )

        self.kernel(
            tma_tensor_Q if const_expr(self.use_tma_Q) else mQ,
            tma_tensor_K if const_expr(self.use_tma_KV) else mK,
            tma_tensor_V if const_expr(self.use_tma_KV) else mV,
            mV_gmem,
            tensormap_workspace if const_expr(self.ragged_tma_V) else None,
            tma_tensor_O if const_expr(self.use_tma_O) else mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mDynamicCausal,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            blocksparse_tensors,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sP_layout,
            self.sQ_mma_layout,
            self.sV_mma_layout,
            self.sStage_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            num_splits,
            aux_data,
            fastdiv_mods,
            output_scale,
            descale_tensors,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mV_gmem: Optional[cute.Tensor],  # gmem V behind tma_atom_V (ragged V only)
        tensormap_workspace: Optional[cute.Tensor],  # (num_ctas, 16) Int64 (ragged V only)
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mDynamicCausal: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        sQ_mma_layout: cute.ComposedLayout | None,
        sV_mma_layout: cute.ComposedLayout | None,
        sStage_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        num_splits: Int32 = Int32(1),
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        output_scale: Optional[cute.Tensor] = None,
        descale_tensors=None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Mbarrier / pipeline init
        mbar_ptr_Q = storage.mbar_ptr_Q.data_ptr()

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        tma_warp = ThreadCooperativeGroup(1)
        load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
        load_warps = ThreadCooperativeGroup(self.num_threads_per_warp_group // cute.arch.WARP_SIZE)
        mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
        if const_expr(self.use_tma_Q):
            pipeline_q = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["Q"],
                defer_sync=True,
            )
        else:
            pipeline_q = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )

        pipeline_stage = None
        if const_expr(self.fp8_kv_dequant):
            pipeline_k = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_k_slots_fp8,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["K"],
                defer_sync=True,
            )
            pipeline_v = None
            pipeline_stage = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_Stage.data_ptr(),
                num_stages=1,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["K"],
                defer_sync=True,
            )
        elif const_expr(self.use_tma_KV):
            pipeline_k = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["K"],
                defer_sync=True,
            )
            pipeline_v = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["V"],
                defer_sync=True,
            )
        else:
            pipeline_k = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )
            pipeline_v = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        # sQ is the input-dtype (q_dtype) view of the storage: the storage MemRange is
        # typed with the compute dtype, so the TMA'd bf16 Q must be read back as q_dtype
        # (passing dtype here, like sO below) -- otherwise bf16 bytes get read as fp16.
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.q_dtype)
        sQ_mma = sQ
        sQ_res = sQ
        if const_expr(self.fp8_kv_dequant):
            sQ_mma = storage.sQ_mma.get_tensor(sQ_mma_layout.outer, swizzle=sQ_mma_layout.inner)
            # Reuse the now-dead FP16 Q input bytes for the FP8 residual term.
            sQ_res = storage.sQ.get_tensor(
                sQ_mma_layout.outer, swizzle=sQ_mma_layout.inner, dtype=self.kv_dtype
            )
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if const_expr(self.fp8_kv_dequant):
            sVt = storage.sV_mma.get_tensor(sV_mma_layout.outer, swizzle=sV_mma_layout.inner)
            sV = layout_utils.transpose_view(sVt)
        elif const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
            sVt = layout_utils.transpose_view(sV)
        else:
            sV = storage.sQ.get_tensor(
                sV_layout.outer, swizzle=sV_layout.inner, dtype=mV.element_type
            )
            sVt = layout_utils.transpose_view(sV)
        sP = None
        if const_expr(sP_layout is not None):
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        # FP8 staging buffer: TMA destination and WGMMA-layout rearrangement source.
        sStage = None
        if const_expr(self.fp8_kv_dequant):
            sStage = storage.sStage.get_tensor(sStage_layout.outer, swizzle=sStage_layout.inner)
        # reuse sQ's data iterator (sO uses the output dtype, both 2 bytes so they alias)
        sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.o_dtype)

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            num_splits=num_splits,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0]
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            mCuTotalMBlocks=(
                blocksparse_tensors.cu_total_m_blocks if blocksparse_tensors is not None else None
            ),
            mCuBlockIdxOffsets=(
                blocksparse_tensors.cu_block_idx_offsets
                if blocksparse_tensors is not None
                else None
            ),
            # Don't need to pass in tile_mn because we won't access offset_padded
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        self._mDynamicCausal = mDynamicCausal
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        # Cluster wait before starting
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx < 4:  # Producer
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            self.load(
                mQ,
                mK,
                mV,
                mV_gmem,
                tensormap_workspace,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                gmem_tiled_copy_Q,
                mPageTable,
                blocksparse_tensors,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                num_splits,
                sStage,
                pipeline_stage,
            )

        else:  # Consumer
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            # ///////////////////////////////////////////////////////////////////////////////
            # Tile MMA compute thread partitions and allocate accumulators
            # ///////////////////////////////////////////////////////////////////////////////
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                mO,
                mLSE,
                sQ,
                sK,
                sVt,
                sP,
                sO,
                sQ_mma,
                sQ_res,
                learnable_sink,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                gmem_tiled_copy_O,
                tma_atom_O,
                tidx,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                blocksparse_tensors,
                aux_data,
                fastdiv_mods,
                num_splits,
                # FP8-KV cast inputs (sV/sStage/pipeline_stage; None otherwise).
                sV=sV if const_expr(self.fp8_kv_dequant) else None,
                sStage=sStage,
                pipeline_stage=pipeline_stage,
                descale_tensors=descale_tensors,
            )

    @cute.jit
    def load_fp8(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sStage: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        pipeline_q: pipeline.PipelineAsync,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_stage: pipeline.PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        mPageTable: Optional[cute.Tensor],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        num_splits: Int32 = Int32(1),
    ):
        """FP8-KV producer: warp 0 TMAs fp8 K tiles into sK slots and V tiles into sStage."""
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tidx, _, _ = cute.arch.thread_idx()
        stage_full_bar = pipeline_stage.sync_object_full.get_barrier(0)
        q_producer_phase = Int32(1)
        # Staging producer phase (num_stages=1): toggles once per staged tile.
        stage_producer_phase = Int32(1)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            head_idx_kv = (
                head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
            )

            # Q remains in its native fp16/bf16 source dtype until compensated
            # E4M3 preparation in the MMA warp group.
            load_Q = None
            pack_gqa = None
            if const_expr(self.use_tma_Q):
                gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
                load_Q, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
                )
            else:
                pack_gqa = PackGQA(
                    self.tile_m, self.tile_hdim, self.check_hdim_oob, self.qhead_per_kvhead
                )

            # Paged TMA tiles, keeping the page dimension indexable (page_idx).
            mK_cur = mK[None, None, head_idx_kv, None]
            mV_cur = mV[None, None, head_idx_kv, None]
            gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (0, 0, None))
            gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (0, 0, None))

            load_K, _, _ = copy_utils.tma_get_copy_fn(tma_atom_K, 0, cute.make_layout(1), gK, sK)
            stage_copy_V, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V, 0, cute.make_layout(1), gV, sStage
            )

            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
            # A bidirectional entry in a dynamic-causal batch owns a slice of
            # the full KV range, not the causal range.  This must mirror the
            # consumer's range calculation exactly: each emitted K/V stage has
            # a matching consumer wait.
            if const_expr(self._mDynamicCausal is not None):
                psc = self._mDynamicCausal[batch_idx]
                if not psc:
                    n_block_max_full = cute.ceil_div(seqlen.seqlen_k, self.tile_n)
                    if const_expr(self.is_split_kv):
                        num_n_blocks_per_split = cute.ceil_div(n_block_max_full, num_splits)
                        n_block_min = split_idx * num_n_blocks_per_split
                        n_block_max = cutlass.min(
                            n_block_min + num_n_blocks_per_split, n_block_max_full
                        )
                    else:
                        n_block_min = Int32(0)
                        n_block_max = n_block_max_full

            # An empty SplitKV partition still needs the same one K and one V
            # pipeline transaction as the consumer's first/last-half sequence.
            # Use physical page 0 as a known-valid dummy source; its result is
            # discarded by the matching consumer-side empty-partition path.
            is_empty_split = n_block_min >= n_block_max
            if is_empty_split:
                n_block_min = Int32(0)
                n_block_max = Int32(1)

            # ---- Prologue emit: load K[n_block_max - 1] (warp 0 only) ----
            page_idx = Int32(0)
            if warp_idx_in_wg == 0:
                if is_empty_split:
                    page_idx = Int32(0)
                else:
                    page_idx = mPageTable[batch_idx, n_block_max - 1]
                ks, kp = self._kslot(n_block_max - 1, n_block_max)
                pipeline_k.producer_acquire_w_index_phase(ks, kp ^ 1)
                load_K(
                    src_idx=page_idx,
                    dst_idx=ks,
                    tma_bar_ptr=pipeline_k.sync_object_full.get_barrier(ks),
                )

            # Q load (warp 0 for TMA, all 128 for pack_gqa cp.async).
            if const_expr(self.use_tma_Q):
                if warp_idx_in_wg == 0:
                    pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                    load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
                    q_producer_phase ^= 1
            else:
                pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                pack_gqa.load_Q(mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q)
                cute.arch.cp_async_commit_group()
                pipeline_q.producer_commit_w_index(0)
                q_producer_phase ^= 1

            # ---- Mainloop + epilogue: load K[n] then V[n_prev], then V[min]. All TMA
            # staging is warp-0 only; the other producer warps just advance the tile
            # scheduler in lockstep below. ----
            if warp_idx_in_wg == 0:
                for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                    n_block_prev = n_block_max - i - 1
                    n_block = n_block_prev - 1
                    page_idx = mPageTable[batch_idx, n_block]
                    page_idx_prev = mPageTable[batch_idx, n_block_prev]
                    # load K[n] into its sK slot
                    ks, kp = self._kslot(n_block, n_block_max)
                    pipeline_k.producer_acquire_w_index_phase(ks, kp ^ 1)
                    load_K(
                        src_idx=page_idx,
                        dst_idx=ks,
                        tma_bar_ptr=pipeline_k.sync_object_full.get_barrier(ks),
                    )
                    # stage V[n_prev]
                    pipeline_stage.producer_acquire_w_index_phase(0, stage_producer_phase)
                    stage_copy_V(src_idx=page_idx_prev, dst_idx=0, tma_bar_ptr=stage_full_bar)
                    stage_producer_phase ^= 1

                # ---- Epilogue emit: stage V[n_block_min] ----
                page_idx = Int32(0)
                if is_empty_split:
                    page_idx = Int32(0)
                else:
                    page_idx = mPageTable[batch_idx, n_block_min]
                pipeline_stage.producer_acquire_w_index_phase(0, stage_producer_phase)
                stage_copy_V(src_idx=page_idx, dst_idx=0, tma_bar_ptr=stage_full_bar)
                stage_producer_phase ^= 1

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mV_gmem: Optional[cute.Tensor],
        tensormap_workspace: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        mPageTable: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        num_splits: Int32 = Int32(1),
        sStage: Optional[cute.Tensor] = None,
        pipeline_stage: Optional[pipeline.PipelineAsync] = None,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tidx, _, _ = cute.arch.thread_idx()

        if const_expr(self.fp8_kv_dequant):
            # fp8_kv_dequant path uses different load logic, separate from the non-fp8 path.
            self.load_fp8(
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sStage,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_q,
                pipeline_k,
                pipeline_stage,
                gmem_tiled_copy_Q,
                mPageTable,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                num_splits,
            )
            return

        # TMA: only warp 0 loads. cp_async: all warps load.
        # When not use_tma_Q, all 128 producer threads participate in Q loading.
        is_load_warp = warp_idx_in_wg == 0 or const_expr(not self.use_tma_KV or not self.use_tma_Q)
        # KV loading restricted to warp 0 for TMA, all warps for non-TMA KV
        is_kv_load_warp = warp_idx_in_wg == 0 or const_expr(not self.use_tma_KV)

        if is_load_warp:
            q_producer_phase = Int32(1)
            kv_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            tensormap_manager = None
            tensormap_V_gmem = None
            v_desc_ptr = None
            if const_expr(self.ragged_tma_V):
                tensormap_manager = TensorMapManager(TensorMapUpdateMode.GMEM, 128)
                bidx, bidy, bidz = cute.arch.block_idx()
                gdx, gdy, _ = cute.arch.grid_dim()
                cta_linear_idx = bidx + bidy * gdx + bidz * gdx * gdy
                if cta_linear_idx >= cute.size(tensormap_workspace, mode=[0]):
                    # An undersized workspace would silently corrupt memory: fail loudly instead.
                    cute.arch.inline_ptx("trap;", read_only_args=[])
                tensormap_V_gmem = tensormap_manager.get_tensormap_ptr(
                    tensormap_workspace[(cta_linear_idx, None)].iterator
                )
                v_desc_ptr = tensormap_manager.get_tensormap_ptr(
                    tensormap_V_gmem, cute.AddressSpace.generic
                )
                tensormap_manager.init_tensormap_from_atom(tma_atom_V, tensormap_V_gmem, 0)
                tensormap_manager.fence_tensormap_initialization()
            # The V extent depends on the batch only: rewrite the descriptor when it changes.
            v_desc_batch_idx = Int32(-1)
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                # if work_tile.is_valid_tile:
                m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                if const_expr(self.ragged_tma_V) and batch_idx != v_desc_batch_idx:
                    # Shrink the rows extent (mode 0) of V to this sequence's end: packed
                    # layouts count absolute rows, dense layouts per-batch rows. Extents >= 1.
                    v_desc_batch_idx = batch_idx
                    v_rows_end = (
                        seqlen.offset_k + seqlen.seqlen_k
                        if const_expr(seqlen.has_cu_seqlens_k)
                        else seqlen.seqlen_k
                    )
                    v_rows_end = cutlass.max(v_rows_end, 1)
                    v_shape = list(mV_gmem.shape)
                    v_shape[0] = v_rows_end
                    mV_bounded = cute.make_tensor(
                        mV_gmem.iterator, cute.make_layout(tuple(v_shape), stride=mV_gmem.stride)
                    )
                    tensormap_manager.update_tensormap(
                        (mV_bounded,), (tma_atom_V,), (tensormap_V_gmem,), 0, (None,)
                    )
                    tensormap_manager.fence_tensormap_update(tensormap_V_gmem)
                    cute.arch.inline_ptx(
                        "prefetch.tensormap [{$r0}];", read_only_args=[tensormap_V_gmem.toint()]
                    )
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                head_idx_kv = (
                    head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
                )

                load_Q = None
                if const_expr(self.use_tma_Q):
                    gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
                    load_Q, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
                    )

                paged_kv_manager = None
                tma_load_K_fn = None
                tma_load_V_fn = None
                if const_expr(self.use_tma_KV):
                    # === TMA path (non-paged and paged with page_size == n_block_size) ===
                    if const_expr(mPageTable is not None):
                        # Paged TMA: keep page dimension indexable
                        mK_cur = mK[None, None, head_idx_kv, None]
                        mV_cur = mV[None, None, head_idx_kv, None]
                        gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (0, 0, None))
                        gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (0, 0, None))
                    else:
                        # Non-paged TMA
                        mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[
                            None, None, head_idx_kv
                        ]
                        mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[
                            None, None, head_idx_kv
                        ]
                        gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
                        gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (None, 0))
                    # TODO: mcast
                    tma_load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_K, 0, cute.make_layout(1), gK, sK
                    )
                    tma_load_K_fn = copy_utils.tma_producer_copy_fn(tma_load_K_fn, pipeline_k)
                    tma_load_V_fn, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_V,
                        0,
                        cute.make_layout(1),
                        gV,
                        sV,
                        **({"tma_desc_ptr": v_desc_ptr} if const_expr(self.ragged_tma_V) else {}),
                    )
                    tma_load_V_fn = copy_utils.tma_producer_copy_fn(tma_load_V_fn, pipeline_v)
                else:
                    # === cp_async path (paged KV with page_size != n_block_size) ===
                    paged_kv_manager = PagedKVManager.create(
                        mPageTable,
                        mK,
                        mV,
                        FastDivmodDivisorV2(mK.shape[0]),
                        batch_idx,
                        head_idx_kv,
                        tidx,
                        seqlen.seqlen_k,
                        0,  # leftpad_k
                        self.tile_n,
                        self.tile_hdim,
                        self.tile_hdimv,
                        self.num_threads_per_warp_group,
                        mK.element_type,
                        arch=self.arch.major * 10 + self.arch.minor,
                    )

                load_K = partial(
                    self.load_KV,
                    tma_load_K_fn,
                    paged_kv_manager,
                    sK,
                    pipeline_kv=pipeline_k,
                    K_or_V="K",
                )
                load_V = partial(
                    self.load_KV,
                    tma_load_V_fn,
                    paged_kv_manager,
                    sV,
                    pipeline_kv=pipeline_v,
                    K_or_V="V",
                )

                pack_gqa = None
                if const_expr(not self.use_tma_Q):
                    pack_gqa = PackGQA(
                        self.tile_m, self.tile_hdim, self.check_hdim_oob, self.qhead_per_kvhead
                    )

                if const_expr(not self.use_block_sparsity):
                    n_block_min, n_block_max = block_info.get_n_block_min_max(
                        seqlen, m_block, split_idx, num_splits
                    )
                    if const_expr(self._mDynamicCausal is not None):
                        psc_producer = self._mDynamicCausal[batch_idx]
                        if not psc_producer:
                            # Mirror the consumer's bidirectional split range so the
                            # producer loads exactly the K/V blocks the consumer
                            # processes. Any divergence here deadlocks the pipeline.
                            n_block_max_full = cute.ceil_div(seqlen.seqlen_k, self.tile_n)
                            if const_expr(self.is_split_kv):
                                num_n_blocks_per_split = cute.ceil_div(n_block_max_full, num_splits)
                                n_block_min = split_idx * num_n_blocks_per_split
                                n_block_max = cutlass.min(
                                    n_block_min + num_n_blocks_per_split, n_block_max_full
                                )
                            else:
                                n_block_min = Int32(0)
                                n_block_max = n_block_max_full
                    # Clamp n_block to 0 when n_block_max == 0 (can happen with causal
                    # + pack_gqa when seqlen_k < tile_n). TMA handles n_block=-1
                    # gracefully (fills zeros), but cp.async would crash on
                    # out-of-bounds page table access.
                    n_block = (
                        n_block_max - 1
                        if const_expr(self.use_tma_KV)
                        else cutlass.max(n_block_max - 1, 0)
                    )
                    page_idx = (
                        mPageTable[batch_idx, n_block]
                        if const_expr(mPageTable is not None and self.use_tma_KV)
                        else None
                    )

                    # First iteration: load K on pipeline_k, Q on pipeline_q
                    if is_kv_load_warp:
                        pipeline_k.producer_acquire(kv_producer_state)
                        if const_expr(not self.use_tma_KV):
                            paged_kv_manager.load_page_table(n_block)
                        load_K(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)
                    if const_expr(self.use_tma_Q):
                        if warp_idx_in_wg == 0:
                            pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                            load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
                            q_producer_phase ^= 1
                    else:
                        pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                        pack_gqa.load_Q(
                            mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q
                        )
                        cute.arch.cp_async_commit_group()
                        pipeline_q.producer_commit_w_index(0)
                        q_producer_phase ^= 1

                    if is_kv_load_warp:
                        if const_expr(not self.intra_wg_overlap or not self.use_tma_KV):
                            pipeline_v.producer_acquire(kv_producer_state)
                            load_V(
                                block=n_block, producer_state=kv_producer_state, page_idx=page_idx
                            )
                            kv_producer_state.advance()
                            for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                                n_block = n_block_max - 1 - i - 1
                                page_idx = (
                                    mPageTable[batch_idx, n_block]
                                    if const_expr(mPageTable is not None and self.use_tma_KV)
                                    else None
                                )
                                if const_expr(not self.use_tma_KV):
                                    paged_kv_manager.load_page_table(n_block)
                                pipeline_k.producer_acquire(kv_producer_state)
                                load_K(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )
                                pipeline_v.producer_acquire(kv_producer_state)
                                load_V(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )
                                kv_producer_state.advance()
                        else:
                            for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                                n_block_prev = n_block_max - i - 1
                                n_block = n_block_prev - 1
                                page_idx = (
                                    mPageTable[batch_idx, n_block]
                                    if const_expr(mPageTable is not None)
                                    else None
                                )
                                page_idx_prev = (
                                    mPageTable[batch_idx, n_block_prev]
                                    if const_expr(mPageTable is not None)
                                    else None
                                )
                                kv_producer_state_prev = kv_producer_state.clone()
                                kv_producer_state.advance()
                                pipeline_k.producer_acquire(kv_producer_state)
                                load_K(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )
                                pipeline_v.producer_acquire(kv_producer_state_prev)
                                load_V(
                                    block=n_block_prev,
                                    producer_state=kv_producer_state_prev,
                                    page_idx=page_idx_prev,
                                )
                            n_block = n_block_min
                            page_idx = (
                                mPageTable[batch_idx, n_block]
                                if const_expr(mPageTable is not None)
                                else None
                            )
                            pipeline_v.producer_acquire(kv_producer_state)
                            load_V(
                                block=n_block, producer_state=kv_producer_state, page_idx=page_idx
                            )
                            kv_producer_state.advance()
                else:
                    # Block sparsity: use TMA closures directly (not paged)
                    # Load Q on pipeline_q, separate from K/V pipeline
                    if const_expr(self.use_tma_Q):
                        if warp_idx_in_wg == 0:
                            pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                            load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
                            q_producer_phase ^= 1
                    else:
                        pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                        pack_gqa.load_Q(
                            mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q
                        )
                        cute.arch.cp_async_commit_group()
                        pipeline_q.producer_commit_w_index(0)
                        q_producer_phase ^= 1
                    if is_kv_load_warp:
                        kv_producer_state = produce_block_sparse_loads(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            m_block,
                            seqlen,
                            kv_producer_state,
                            tma_load_K_fn,
                            tma_load_V_fn,
                            pipeline_k,
                            pipeline_v,
                            self.intra_wg_overlap,
                            self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                            self.q_subtile_factor,
                        )

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
                # End of persistent scheduler loop

            # Producer tail is only useful for cluster to avoid early exit of blocks.
            # We only need producer_tail on V since that's the last that's loaded, we don't
            # need it for Q (no cluster) and K.
            if is_kv_load_warp:
                pipeline_v.producer_tail(kv_producer_state)

    @cute.jit
    def load_KV(
        self,
        tma_load_fn: Optional[Callable],
        paged_kv_manager: Optional[PagedKVManager],
        sX: cute.Tensor,
        block: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        page_idx: Optional[Int32] = None,
    ):
        if const_expr(self.use_tma_KV):
            src_idx = block if const_expr(page_idx is None) else page_idx
            tma_load_fn(src_idx=src_idx, producer_state=producer_state)
        else:
            paged_kv_manager.load_KV(block, sX[None, None, producer_state.index], K_or_V)
            cute.arch.cp_async_commit_group()
        pipeline_kv.producer_commit(producer_state)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sP: Optional[cute.Tensor],
        sO: cute.Tensor,
        # FP8 high and residual Q terms used by the compensated QK WGMMA.
        sQ_mma: cute.Tensor,
        sQ_res: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        num_splits: Int32 = Int32(1),
        # FP8-KV staging inputs (None on the non-fp8 path).
        sV: Optional[cute.Tensor] = None,
        sStage: Optional[cute.Tensor] = None,
        pipeline_stage: Optional[pipeline.PipelineAsync] = None,
        descale_tensors=None,
    ):
        aux_tensors = aux_data.tensors
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        warp_group_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_warp_group
        )
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))
        quantize_q = const_expr(self.fp8_kv_dequant)
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ_mma, sK
        )
        if const_expr(self.fp8_kv_dequant):
            _, tSrQ_res, _ = sm90_utils.partition_fragment_ABC(
                wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ_res, sK
            )
            mma_qk_fn = partial(
                self._gemm_qk_fp8_compensated,
                tiled_mma_qk,
                (self.tile_m, self.tile_n),
                # Residual terms first, while the running sum is still small.
                tSrQ_res,
                tSrQ,
                tSrK,
            )
        else:
            mma_qk_fn = partial(
                sm90_utils.gemm_zero_init,
                tiled_mma_qk,
                (self.tile_m, self.tile_n),
                tSrQ,
                tSrK,
            )
        acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
            wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), sP, sVt
        )
        tOrP_res = None
        pv_scratch = None
        if const_expr(self.fp8_kv_dequant):
            # FP8 WGMMA sums below FP32 (PTX 9.7.16.5.2): promote PV every block.
            assert self.rescale_O_before_gemm
            tOrP_res = cute.make_rmem_tensor_like(tOrP)
            pv_scratch = self._make_pv_scratch(
                sVt, warp_group_thread_layout(warp_group_idx), pv_cols=64
            )
            mma_pv_fn = partial(self._gemm_pv_fp8_promoted, pv_scratch, acc_O, tOrP, tOrP_res)
        else:
            mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom_P = utils.get_smem_store_atom(
            self.arch.major * 10 + self.arch.minor, self.dtype
        )
        smem_thr_copy_P = cute.make_tiled_copy_C(smem_copy_atom_P, tiled_mma_qk).get_slice(tidx)
        tPsP = smem_thr_copy_P.partition_D(sP) if const_expr(sP is not None) else None
        smem_copy_params = SimpleNamespace(
            smem_thr_copy_P=smem_thr_copy_P, tPsP=tPsP, tOrP_res=tOrP_res
        )

        self.mma_init()

        q_consumer_phase = Int32(0)
        kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )

        # FP8 staging -> FP8 WGMMA buffers. K keeps its orientation; V is explicitly
        # transposed so the PV B operand is K-major as required by Hopper FP8 WGMMA.
        dequant_params = None
        if const_expr(self.fp8_kv_dequant):
            sStage2 = sStage[None, None, 0]
            sStageVt2 = layout_utils.transpose_view(sStage2)
            sVt2 = sVt[None, None, 0]
            half = const_expr(min(256, self.tile_hdimv))
            # Build both WG-local output-dimension halves with static coordinates.
            # A dynamic local_tile coordinate loses the 128-bit alignment proof
            # required by STSM even though both 256-column offsets are aligned.
            gStage_V0 = cute.local_tile(sStageVt2, (half, self.tile_n), (0, 0))
            gStage_V1 = cute.local_tile(sStageVt2, (half, self.tile_n), (1, 0))
            gV0 = cute.local_tile(sVt2, (half, self.tile_n), (0, 0))
            gV1 = cute.local_tile(sVt2, (half, self.tile_n), (1, 0))

            # Port of FA3's FP8 V transpose. LDSM.T reads 64x8 byte tiles as
            # 16-bit matrices, PRMT interleaves their bytes, and STSM writes the
            # K-major WGMMA layout without scalar shared-memory traffic.
            ldsm_atom = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4), self.kv_dtype
            )
            stsm_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(False, 4), self.kv_dtype
            )
            ldsm_copy = cute.make_tiled_copy_tv(
                ldsm_atom,
                cute.make_layout((32, 4, 1, 1), stride=(4, 1, 0, 0)),
                cute.make_layout((2, 2, 1, 4), stride=(1, 2, 16, 4)),
            )
            stsm_copy = cute.make_tiled_copy_tv(
                stsm_atom,
                cute.make_layout((8, 4, 4, 1), stride=(4, 1, 32, 0)),
                # FA3's faster STSM layout avoids shared-memory bank conflicts.
                # It permutes V's output columns, so each completed PV must apply
                # the inverse permutation to its accumulator registers.
                cute.make_layout((1, 4, 2, 2), stride=(0, 1, 4, 8)),
            )
            transpose_tid = tidx % self.num_threads_per_warp_group
            ldsm_thr = ldsm_copy.get_slice(transpose_tid)
            stsm_thr = stsm_copy.get_slice(transpose_tid)
            tStage_V0 = ldsm_thr.partition_S(cute.flat_divide(gStage_V0, (64, 8)))
            tStage_V1 = ldsm_thr.partition_S(cute.flat_divide(gStage_V1, (64, 8)))
            tsV0 = stsm_thr.partition_D(cute.flat_divide(gV0, (8, 16)))
            tsV1 = stsm_thr.partition_D(cute.flat_divide(gV1, (8, 16)))
            tStage_V0 = cute.group_modes(tStage_V0, 1, cute.rank(tStage_V0))
            tStage_V1 = cute.group_modes(tStage_V1, 1, cute.rank(tStage_V1))
            tsV0 = cute.group_modes(tsV0, 1, cute.rank(tsV0))
            tsV1 = cute.group_modes(tsV1, 1, cute.rank(tsV1))
            dequant_params = SimpleNamespace(
                pipeline_stage=pipeline_stage,
                warp_group_idx=warp_group_idx,
                nmax=Int32(0),
                tStage_V0=tStage_V0,
                tStage_V1=tStage_V1,
                tsV0=tsV0,
                tsV1=tsV1,
                ldsm_copy=ldsm_copy,
                stsm_copy=stsm_copy,
            )

        q_narrow_params = None
        if const_expr(quantize_q):
            VEC_Q = const_expr(8)
            tiled_copy_Q = copy_utils.tiled_copy_2d(
                self.kv_dtype,
                self.tile_hdim // VEC_Q,
                self.num_mma_threads,
                num_copy_elems=VEC_Q,
            )
            thr_copy_Q = tiled_copy_Q.get_slice(tidx)
            q_narrow_params = SimpleNamespace(
                tQsrc=thr_copy_Q.partition_S(sQ),
                tQdst=thr_copy_Q.partition_D(sQ_mma),
                tQres=thr_copy_Q.partition_D(sQ_res),
            )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        num_softmax_rows = const_expr(acc_O.shape[0][0] * acc_O.shape[1])
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=num_softmax_rows,
            softmax_scale=softmax_scale,
        )

        # For RescaleOBeforeGemm: persistent scores_scale across iterations
        scores_scale = None
        if const_expr(self.rescale_O_before_gemm):
            scores_scale = cute.make_rmem_tensor_like(softmax.row_max, Float32)

        # FP8-KV uses separate *_fp8 consumer functions (selected via const_expr) so the
        # generic functions stay unchanged.
        mma_one_n_block_all = partial(
            self.mma_one_n_block_intrawg_overlap_fp8
            if const_expr(self.fp8_kv_dequant)
            else (
                self.mma_one_n_block_intrawg_overlap
                if const_expr(self.intra_wg_overlap)
                else self.mma_one_n_block
            ),
            mma_qk_fn=mma_qk_fn,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            acc_O=acc_O,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            check_inf=True,
            scores_scale=scores_scale,
            **(dict(dequant_params=dequant_params) if const_expr(self.fp8_kv_dequant) else {}),
            **(dict(qk_split_acc=pv_scratch.acc) if const_expr(self.qk_split) else {}),
        )

        process_first_half_block = partial(
            self.first_half_block_overlap_fp8
            if const_expr(self.fp8_kv_dequant)
            else self.first_half_block_overlap,
            mma_qk_fn=mma_qk_fn,
            pipeline_k=pipeline_k,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            scores_scale=scores_scale,
            acc_O=acc_O,
            **(dict(dequant_params=dequant_params) if const_expr(self.fp8_kv_dequant) else {}),
        )
        process_last_half_block = partial(
            self.last_half_block_overlap_fp8
            if const_expr(self.fp8_kv_dequant)
            else self.last_half_block_overlap,
            pipeline_v=pipeline_v,
            mma_pv_fn=mma_pv_fn,
            scores_scale=scores_scale,
            acc_O=acc_O,
            **(dict(dequant_params=dequant_params) if const_expr(self.fp8_kv_dequant) else {}),
        )
        while work_tile.is_valid_tile:
            # if work_tile.is_valid_tile:

            # shape: (atom_v_m * rest_m)
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            softmax_tile = softmax
            softmax_scale_eff = softmax_scale
            v_descale_tile = Float32(1.0)
            # FP8-KV per-(batch, kv_head) descales: q*k folds into the score scale
            # while v folds into final output normalization.
            if const_expr(self.fp8_kv_dequant):
                head_idx_kv = (
                    head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
                )
                qk_descale, v_descale_tile = self._load_effective_descales(
                    descale_tensors, batch_idx, head_idx_kv
                )
                qk_descale = qk_descale / Float32(self.fp8_q_scale)
                v_descale_tile = v_descale_tile / Float32(self.fp8_p_scale)
                if const_expr(self.score_mod is None):
                    softmax_scale_log2_eff = softmax_scale_log2 * qk_descale
                    softmax_scale_eff = None
                else:
                    softmax_scale_log2_eff = softmax_scale_log2
                    softmax_scale_eff = softmax_scale * qk_descale
                softmax_tile = Softmax.create(
                    softmax_scale_log2_eff,
                    num_rows=num_softmax_rows,
                    softmax_scale=softmax_scale_eff,
                )

            # Recompute fastdiv_mods if necessary for varlen with aux_tensors
            recompute_fastdiv_mods_q = cutlass.const_expr(
                aux_tensors is not None and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
            )
            recompute_fastdiv_mods_k = cutlass.const_expr(
                aux_tensors is not None and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
            )
            if cutlass.const_expr(fastdiv_mods is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    seqlen_q_divmod
                    if not recompute_fastdiv_mods_q
                    else FastDivmodDivisorV2(seqlen.seqlen_q),
                    seqlen_k_divmod
                    if not recompute_fastdiv_mods_k
                    else FastDivmodDivisorV2(seqlen.seqlen_k),
                )

            psc = (
                self._mDynamicCausal[batch_idx]
                if const_expr(self._mDynamicCausal is not None)
                else None
            )
            mask = AttentionMaskCls(seqlen, dynamic_causal=psc)
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )
            score_mod_fn = None
            if const_expr(self.score_mod is not None):
                score_mod_fn = partial(
                    self.apply_score_mod,
                    thr_mma_qk,
                    batch_idx,
                    head_idx,
                    m_block,
                    softmax_scale=softmax_scale_eff,
                    aux_data=aux_data,
                    fastdiv_mods=fastdiv_mods,
                )
            process_first_half_block_tile = partial(process_first_half_block, softmax=softmax_tile)
            process_last_half_block_tile = partial(process_last_half_block, softmax=softmax_tile)
            mma_one_n_block = partial(
                mma_one_n_block_all,
                seqlen=seqlen,
                softmax=softmax_tile,
                score_mod_fn=score_mod_fn,
            )
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
            if const_expr(self._mDynamicCausal is not None):
                # Per-sequence causal: psc == 0 means this sequence is processed
                # bidirectionally. get_n_block_min_max may have applied a causal
                # upper bound (when the kernel is compiled causal) and, for
                # split-KV, partitioned that (possibly causal) range. For a
                # bidirectional sequence each split must instead own a DISJOINT
                # slice of the FULL key range. Recompute [n_block_min, n_block_max)
                # over the full range here, and IDENTICALLY on the producer side
                # (see the K/V load loop), so the pipeline block counts agree -- a
                # producer/consumer mismatch deadlocks the kernel (GPU spins).
                # The previous code only reset n_block_max to the global max while
                # leaving n_block_min at its split offset, so splits overlapped and
                # keys were double-counted -> corrupted softmax (rel_err ~0.33).
                if not psc:
                    n_block_max_full = cute.ceil_div(seqlen.seqlen_k, self.tile_n)
                    if const_expr(self.is_split_kv):
                        num_n_blocks_per_split = cute.ceil_div(n_block_max_full, num_splits)
                        n_block_min = split_idx * num_n_blocks_per_split
                        n_block_max = cutlass.min(
                            n_block_min + num_n_blocks_per_split, n_block_max_full
                        )
                    else:
                        n_block_min = Int32(0)
                        n_block_max = n_block_max_full
            # Mirror load_fp8: empty partitions consume a known-valid dummy
            # K/V pair so the pipeline transaction count stays matched.  Their
            # partial result is suppressed before SplitKV combine below.
            is_empty_split = n_block_min >= n_block_max
            if const_expr(self.fp8_kv_dequant):
                if is_empty_split:
                    n_block_min = Int32(0)
                    n_block_max = Int32(1)
                dequant_params.nmax = n_block_max
            pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)
            if const_expr(quantize_q):
                self._quantize_q_to_fp8(q_narrow_params)
            # For performance reason, we separate out two kinds of iterations:
            # those that need masking on S, and those that don't.
            # We need masking on S for the very last block when K and V has length not multiple of tile_n.
            # We also need masking on S if it's causal, for the last several blocks.
            # softmax.reset()  # Don't need reset as we explicitly call softmax w is_first=True
            O_should_accumulate = False

            # ==========================================
            # MAINLOOP
            # ==========================================
            if const_expr(not self.use_block_sparsity):
                # ==========================================
                # No block-sparsity (original path)
                # ==========================================
                # First iteration with seqlen masking
                if const_expr(self.intra_wg_overlap):
                    kv_consumer_state = process_first_half_block_tile(
                        n_block=n_block_max - 1,
                        seqlen=seqlen,
                        kv_consumer_state=kv_consumer_state,
                        mask_fn=partial(mask_fn, mask_mod=self.mask_mod),
                        score_mod_fn=score_mod_fn,
                        is_first_block=True,
                    )
                else:
                    self.warp_scheduler_barrier_sync()
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=n_block_max - 1,
                        seqlen=seqlen,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=True),
                        is_first_n_block=True,
                        mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True),
                    )
                    O_should_accumulate = True
                # if cute.arch.thread_idx()[0] == 128: cute.printf("m_block = {}, n_block_max = {}, n_block_min = {}", m_block, n_block_max, n_block_min)
                n_block_max -= 1
                # Next couple of iterations with causal masking
                if const_expr(self.is_causal or self.is_local):
                    n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                        seqlen, m_block, n_block_min
                    )
                    if const_expr(self._mDynamicCausal is not None):
                        if not psc:
                            n_block_min_causal_local_mask = n_block_min
                    # if cute.arch.thread_idx()[0] == 128: cute.printf("n_block_min_causal_local_mask = {}", n_block_min_causal_local_mask)
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_causal_local_mask, unroll=1
                    ):
                        kv_consumer_state = mma_one_n_block(
                            kv_consumer_state,
                            n_block=n_block_max - 1 - n_tile,
                            seqlen=seqlen,
                            mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                            mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=False),
                        )
                        O_should_accumulate = True
                    n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
                # Interior blocks need no causal/local masking, but custom mask_mod still applies.
                n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                    seqlen, m_block, n_block_min
                )
                # if cute.arch.thread_idx()[0] == 128: cute.printf("n_block_min_before_local_mask = {}, n_block_min = {}", n_block_min_before_local_mask, n_block_min)
                for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=n_block_max - 1 - n_tile,
                        seqlen=seqlen,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=False)
                        if const_expr(self.mask_mod is not None)
                        else None,
                    )
                    O_should_accumulate = True
                # Separate iterations with local masking on the left
                if const_expr(self.is_local and block_info.window_size_left is not None):
                    n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                    for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                        kv_consumer_state = mma_one_n_block(
                            kv_consumer_state,
                            n_block=n_block_max - 1 - n_tile,
                            seqlen=seqlen,
                            mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                            mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=False),
                        )
                        O_should_accumulate = True
                # Release Q pipeline so the producer can load the next tile's Q
                pipeline_q.consumer_release_w_index(0)
                # Last "half" iteration
                if const_expr(self.intra_wg_overlap):
                    kv_consumer_state = process_last_half_block_tile(
                        kv_consumer_state=kv_consumer_state,
                        zero_init=not O_should_accumulate,
                    )
                    O_should_accumulate = True
                else:
                    self.warp_scheduler_barrier_arrive()

            else:
                # ==========================================
                # Block sparsity
                # ==========================================
                kv_consumer_state, O_should_accumulate, processed_any = consume_block_sparse_loads(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    seqlen,
                    kv_consumer_state,
                    mma_pv_fn,
                    mma_one_n_block,
                    process_first_half_block_tile,
                    process_last_half_block_tile,
                    mask_fn,
                    score_mod_fn,
                    O_should_accumulate,
                    self.mask_mod,
                    fastdiv_mods,
                    self.intra_wg_overlap,
                    self.warp_scheduler_barrier_sync,
                    self.warp_scheduler_barrier_arrive,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor,
                )

                # Release Q pipeline so the producer can load the next tile's Q
                pipeline_q.consumer_release_w_index(0)

                # Handle empty case (when no blocks to process)
                if not processed_any:
                    softmax_tile.reset()
                    acc_O.fill(0.0)

            q_consumer_phase ^= 1

            sink_val = None
            if const_expr(learnable_sink is not None):
                if const_expr(not self.pack_gqa):
                    sink_val = Float32(learnable_sink[head_idx])
                else:  # Each thread might have a different sink value due to different q_head
                    sink_val = cute.make_rmem_tensor_like(softmax_tile.row_max, Float32)
                    cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
                    tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
                    for r in cutlass.range(cute.size(sink_val), unroll_full=True):
                        row = m_block * self.tile_m + tScS_mn[r][0]
                        q_head_idx = row % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                        sink_val[r] = Float32(learnable_sink[q_head_idx])
                if const_expr(self.is_split_kv):
                    if split_idx > 0:
                        if const_expr(not self.pack_gqa):
                            sink_val = -Float32.inf
                        else:
                            sink_val.fill(-Float32.inf)

            # normalize acc_O by row_sum and calculate the lse
            row_scale = softmax_tile.finalize(final_scale=v_descale_tile, sink_val=sink_val)
            softmax_tile.rescale_O(acc_O, row_scale)

            # Make an empty partition contribute zero weight to SplitKV combine.
            # This also defines an all-empty sequence as a zero output.
            if is_empty_split:
                acc_O.fill(Float32(0.0))
                # A non-split launch keeps the learnable sink's LSE, as before;
                # FP8 consumed a dummy K/V block, so its LSE is not meaningful.
                if const_expr(self.is_split_kv or self.fp8_kv_dequant):
                    softmax_tile.row_sum.fill(-Float32.inf)

            # ///////////////////////////////////////////////////////////////////////////////
            # Epilogue
            # ///////////////////////////////////////////////////////////////////////////////
            self.epilogue(
                acc_O,
                softmax_tile.row_sum,
                mO,
                mLSE,
                sO,
                seqlen,
                gmem_tiled_copy_O,
                tma_atom_O,
                tiled_mma_pv,
                tidx,
                m_block,
                head_idx,
                batch_idx,
                split_idx,
            )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def first_half_block_overlap(
        self,
        n_block: Int32,
        mma_qk_fn: Callable,
        kv_consumer_state,
        pipeline_k,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,
        acc_O: Optional[cute.Tensor] = None,
        mask_fn: Callable = None,
        score_mod_fn: Optional[Callable] = None,
        is_first_block: bool = False,
    ):
        """Processes the first half block when using intra-warpgroup-overlap"""

        pipeline_k.consumer_wait(kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state))
        acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)
        pipeline_k.consumer_release(kv_consumer_state)

        # Apply score modification if present
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)

        # Apply mask; mask_seqlen always True for first block
        # Caveat: if full block further right than mask block, seqlen masking is redundant;
        # however, masking is being applied anyway, so essentially no perf hit
        mask_fn(acc_S, n_block=n_block, mask_seqlen=True)

        row_scale = softmax.online_softmax(acc_S, is_first=is_first_block)

        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        tOrP_cur.store(tOrP_acc.load().to(self.dtype))

        if const_expr(not self.mma_pv_is_rs):
            self.cross_wg_p_barrier()
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
            # Fence and barrier to make smem store visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            self.cross_wg_p_raw_barrier()

        # For RescaleOBeforeGemm: initialize acc_O
        if const_expr(self.rescale_O_before_gemm):
            acc_O.fill(0.0)
            scores_scale.store(row_scale.load())

        return kv_consumer_state

    @cute.jit
    def _load_effective_descales(self, descale_tensors, batch_idx: Int32, kv_head_idx: Int32):
        """Load effective QK score and V output descales, defaulting to identity."""
        qk_descale = Float32(1.0)
        v_descale = Float32(1.0)
        if const_expr(descale_tensors is not None):
            if const_expr(descale_tensors.q_descale is not None):
                qk_descale = qk_descale * Float32(descale_tensors.q_descale[batch_idx, kv_head_idx])
            if const_expr(descale_tensors.k_descale is not None):
                qk_descale = qk_descale * Float32(descale_tensors.k_descale[batch_idx, kv_head_idx])
            if const_expr(descale_tensors.v_descale is not None):
                v_descale = Float32(descale_tensors.v_descale[batch_idx, kv_head_idx])
        return qk_descale, v_descale

    @cute.jit
    def _kslot(self, n_block, n_block_max):
        """sK slot and its fill parity for K[n_block]; fills run n_block_max - 1 down."""
        slot = n_block % 2
        first = n_block_max - 1
        if first % 2 != slot:
            first = first - 1
        return slot, ((first - n_block) // 2) % 2

    @cute.jit
    def _gemm_two_fp8_a(
        self,
        tiled_mma,
        acc,
        tA_first,
        tA_second,
        tB,
        zero_init: cutlass.Constexpr[bool],
        wg_wait: cutlass.Constexpr[int],
        mid=None,
    ):
        """Issue two FP8 A terms in one WGMMA commit group; mid() runs between them."""
        warpgroup.fence()
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
        for k in cutlass.range_constexpr(cute.size(tA_first.shape[2])):
            cute.gemm(mma_atom, acc, tA_first[None, None, k], tB[None, None, k], acc)
            mma_atom.set(warpgroup.Field.ACCUMULATE, True)
        if const_expr(mid is not None):
            mid()
            warpgroup.fence()
        for k in cutlass.range_constexpr(cute.size(tA_second.shape[2])):
            cute.gemm(mma_atom, acc, tA_second[None, None, k], tB[None, None, k], acc)
        warpgroup.commit_group()
        if const_expr(wg_wait >= 0):
            warpgroup.wait_group(wg_wait)

    @cute.jit
    def _gemm_qk_fp8_compensated(
        self,
        tiled_mma,
        shape,
        tQ_first,
        tQ_second,
        tK,
        B_idx=None,
        wg_wait: int = -1,
        mid=None,
        split_acc=None,
    ):
        acc = cute.make_rmem_tensor(tiled_mma.partition_shape_C(shape), Float32)
        rK = tK if const_expr(B_idx is None) else tK[None, None, None, B_idx]
        if const_expr(not self.qk_split):
            self._gemm_two_fp8_a(tiled_mma, acc, tQ_first, tQ_second, rK, True, wg_wait, mid=mid)
        elif const_expr(split_acc is None):
            # First block: own second accumulator, always waited for and folded here.
            acc2 = cute.make_rmem_tensor(tiled_mma.partition_shape_C(shape), Float32)
            self._gemm_qk_fp8_split(tiled_mma, acc, acc2, tQ_first, tQ_second, rK, None)
            warpgroup.wait_group(0)
            self._fold_qk_split(acc, acc2)
        else:
            # wg_wait is unused: the caller waits and folds split_acc before reusing it.
            self._gemm_qk_fp8_split(tiled_mma, acc, split_acc, tQ_first, tQ_second, rK, mid)
        return acc

    @cute.jit
    def _gemm_qk_fp8_split(
        self,
        tiled_mma,
        acc,
        acc2,
        tA_first,
        tA_second,
        rK,
        mid,
    ):
        """One commit group: k-steps in the first half sum into acc, the rest into acc2."""
        nk = const_expr(cute.size(tA_first.shape[2]))
        warpgroup.fence()
        a1 = cute.make_mma_atom(tiled_mma.op)
        a2 = cute.make_mma_atom(tiled_mma.op)
        a1.set(warpgroup.Field.ACCUMULATE, False)
        a2.set(warpgroup.Field.ACCUMULATE, False)
        for t in cutlass.range_constexpr(2):
            tA = tA_first if const_expr(t == 0) else tA_second
            if const_expr(t == 1 and mid is not None):
                mid()
                warpgroup.fence()
            for k in cutlass.range_constexpr(nk):
                if const_expr(k < nk // 2):
                    cute.gemm(a1, acc, tA[None, None, k], rK[None, None, k], acc)
                    a1.set(warpgroup.Field.ACCUMULATE, True)
                else:
                    cute.gemm(a2, acc2, tA[None, None, k], rK[None, None, k], acc2)
                    a2.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()

    @cute.jit
    def _fold_qk_split(self, acc, acc2):
        for i in cutlass.range_constexpr(cute.size(acc)):
            acc[i] = acc[i] + acc2[i]

    def _make_pv_scratch(self, sVt, wg_thr_layout, pv_cols: int):
        """PV column chunks: one 64 x pv_cols WGMMA per WG into a shared FP32 scratch."""
        half = min(256, self.tile_hdimv)
        assert half == 256 and self.tile_hdimv == 2 * half and half % pv_cols == 0
        tiled_mma = sm90_utils_basic.make_trivial_tiled_mma(
            self.kv_dtype,
            self.kv_dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 2, 1),
            tiler_mn=(64, pv_cols),
            a_source=warpgroup.OperandSource.RMEM,
        )
        wg_mma = tiled_mma.get_slice(wg_thr_layout)
        # Rows [c * pv_cols, (c + 1) * pv_cols) of each WG's half of sVt, K-major.
        chunk_layout = cute.make_layout(
            ((8, pv_cols // 8, 2), (self.tile_n, 1), (1, 1)),
            stride=((self.tile_n, 8 * self.tile_n, half * self.tile_n), (1, 0), (0, 0)),
        )
        tVs = []
        acc = None
        for c in range(half // pv_cols):
            sVt_c = cute.make_tensor(sVt.iterator + c * pv_cols * self.tile_n, chunk_layout)
            acc, _, tV_c = sm90_utils.partition_fragment_ABC(
                wg_mma, (self.tile_m, 2 * pv_cols, self.tile_n), None, sVt_c
            )
            tVs.append(tV_c)
        return SimpleNamespace(tiled_mma=tiled_mma, acc=acc, tVs=tVs, n8=pv_cols // 8)

    @cute.jit
    def _gemm_pv_fp8_promoted(
        self, pv, acc_O, tP_hi, tP_res, zero_init=False, B_idx=None, wg_wait: int = -1
    ):
        """O += P @ V per 64-key block, added in FP32; zero_init and wg_wait are unused.

        P_hi and the scaled P residual run as separate passes through the same
        scratch, so the residual is unscaled in FP32 when it is folded into O.
        """
        for c in cutlass.range_constexpr(len(pv.tVs)):
            rV = pv.tVs[c] if const_expr(B_idx is None) else pv.tVs[c][None, None, None, B_idx]
            self._gemm_one_fp8_a(pv.tiled_mma, pv.acc, tP_hi, rV)
            self._pv_fold(pv, acc_O, c, 1.0)
            self._gemm_one_fp8_a(pv.tiled_mma, pv.acc, tP_res, rV)
            self._pv_fold(pv, acc_O, c, 1.0 / self.fp8_p_res_scale)

    @cute.jit
    def _gemm_one_fp8_a(self, tiled_mma, acc, tA, tB):
        """acc = tA @ tB as one synchronous FP8 WGMMA commit group."""
        warpgroup.fence()
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        mma_atom.set(warpgroup.Field.ACCUMULATE, False)
        for k in cutlass.range_constexpr(cute.size(tA.shape[2])):
            cute.gemm(mma_atom, acc, tA[None, None, k], tB[None, None, k], acc)
            mma_atom.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()
        warpgroup.wait_group(0)

    @cute.jit
    def _pv_fold(self, pv, acc_O, c: cutlass.Constexpr[int], scale: cutlass.Constexpr[float]):
        fo = cute.group_modes(acc_O, 1, cute.rank(acc_O))
        fc = cute.group_modes(pv.acc, 1, cute.rank(pv.acc))
        for mi in cutlass.range_constexpr(cute.size(fc.shape[1])):
            for e in cutlass.range_constexpr(2):
                for r in cutlass.range_constexpr(2):
                    for j in cutlass.range_constexpr(pv.n8):
                        idx = ((e, r, c * pv.n8 + j), mi)
                        if const_expr(scale == 1.0):
                            fo[idx] = fo[idx] + fc[((e, r, j), mi)]
                        else:
                            fo[idx] = fo[idx] + fc[((e, r, j), mi)] * Float32(scale)

    @cute.jit
    def _quantize_q_to_fp8(self, q_narrow_params):
        """Quantize the TMA-loaded FP16/BF16 Q tile into FP8 shared memory once."""
        qp = q_narrow_params
        rS = cute.make_rmem_tensor_like(qp.tQsrc)
        rD = cute.make_rmem_tensor_like(qp.tQdst)
        rR = cute.make_rmem_tensor_like(qp.tQres)
        cute.autovec_copy(qp.tQsrc, rS)
        q_scaled = rS.load().to(Float32) * Float32(self.fp8_q_scale)
        q_hi = q_scaled.to(self.kv_dtype)
        rD.store(q_hi)
        rR.store((q_scaled - q_hi.to(Float32)).to(self.kv_dtype))
        cute.autovec_copy(rD, qp.tQdst)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.NarrowQ), number_of_threads=self.num_mma_threads
        )
        # All input FP16/BF16 values have been captured in registers, so the Q input
        # allocation can now safely hold the compact residual FP8 tile.
        cute.autovec_copy(rR, qp.tQres)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.NarrowQ), number_of_threads=self.num_mma_threads
        )

    @cute.jit
    def _transpose_fp8_v_tile(self, dp, tStage_V, tsV):
        for i in cutlass.range_constexpr(cute.size(tStage_V.shape[1])):
            dst = tsV[None, i]
            # The STSM TV layout guarantees every lane's address is 16-byte
            # aligned; retain that fact after partition/slicing in the DSL IR.
            dst = cute.make_tensor(dst.iterator.align(16), dst.layout)
            rV = cute.make_rmem_tensor_like(dst)
            cute.copy(dp.ldsm_copy, tStage_V[None, i], rV)
            rV_u32 = cute.recast_tensor(rV, Int32)
            for j in cutlass.range_constexpr(cute.size(rV_u32) // 2):
                upper = rV_u32[2 * j]
                lower = rV_u32[2 * j + 1]
                rV_u32[2 * j] = cute.arch.prmt(upper, lower, Int32(0x6420))
                rV_u32[2 * j + 1] = cute.arch.prmt(upper, lower, Int32(0x7531))
            cute.copy(dp.stsm_copy, rV, dst)

    @cute.jit
    def _transpose_staged_fp8_v(self, dp, phase):
        """FA3-style LDSM.T -> PRMT -> STSM transpose for one WG's V half."""
        dp.pipeline_stage.consumer_wait_w_index_phase(0, phase)
        if dp.warp_group_idx == 0:
            self._transpose_fp8_v_tile(dp, dp.tStage_V0, dp.tsV0)
        else:
            self._transpose_fp8_v_tile(dp, dp.tStage_V1, dp.tsV1)
        # Order the LDSM reads before the next V TMA; lane 0 of each warp releases.
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        dp.pipeline_stage.consumer_release_w_index(0)

    @cute.jit
    def _permute_Cregs_fp8(self, acc):
        """Match FA3's QK-C to FP8 PV-A register permutation."""
        acc_u64 = cute.recast_tensor(acc, cutlass.Int64)
        flat = cute.make_tensor(acc_u64.iterator, cute.make_layout(cute.size(acc_u64)))
        for i in cutlass.range_constexpr(cute.size(flat) // 4):
            tmp = flat[4 * i + 1]
            flat[4 * i + 1] = flat[4 * i + 2]
            flat[4 * i + 2] = tmp

    @cute.jit
    def _permute_output_fp8(self, acc):
        """Undo the output-column permutation from FA3's conflict-free V STSM."""
        # The hdim-512 Python kernel's PV accumulator carries the local FP8
        # output permutation as well. Undo that first, then apply FA3's
        # cross-lane V-column permutation below. The order is significant.
        frag32 = cute.group_modes(acc, 1, cute.rank(acc))
        for mi in cutlass.range_constexpr(cute.size(frag32.shape[1])):
            for j in cutlass.range_constexpr(cute.size(frag32.shape[0][1])):
                for i in cutlass.range_constexpr(frag32.shape[0][2] // 2):
                    tmp = frag32[((1, j, 2 * i), mi)]
                    frag32[((1, j, 2 * i), mi)] = frag32[((0, j, 2 * i + 1), mi)]
                    frag32[((0, j, 2 * i + 1), mi)] = tmp

        frag = cute.group_modes(cute.recast_tensor(acc, cutlass.Int64), 1, cute.rank(acc))
        n8 = frag.shape[0][2]
        assert n8 % 2 == 0
        quad_idx = cute.arch.lane_idx() % 4
        lane_03 = quad_idx == 0 or quad_idx == 3
        upper_src = Int32(0)
        if quad_idx == 1:
            upper_src = Int32(2)
        elif quad_idx == 2:
            upper_src = Int32(3)
        elif quad_idx == 3:
            upper_src = Int32(1)
        lower_src = upper_src ^ Int32(2)
        for mi in cutlass.range_constexpr(cute.size(frag.shape[1])):
            for j in cutlass.range_constexpr(cute.size(frag.shape[0][1])):
                for i in cutlass.range_constexpr(n8 // 2):
                    upper = frag[((0, j, 2 * i), mi)]
                    lower = frag[((0, j, 2 * i + 1), mi)]
                    upper0, lower0 = upper, lower
                    if not lane_03:
                        upper0, lower0 = lower, upper
                    upper0 = utils.shuffle_sync(upper0, upper_src, width=4)
                    lower0 = utils.shuffle_sync(lower0, lower_src, width=4)
                    if lane_03:
                        frag[((0, j, 2 * i), mi)] = upper0
                        frag[((0, j, 2 * i + 1), mi)] = lower0
                    else:
                        frag[((0, j, 2 * i), mi)] = lower0
                        frag[((0, j, 2 * i + 1), mi)] = upper0

    @cute.jit
    def _reshape_acc_to_fp8_frgA(self, acc):
        """SM90 FP8 form of FA3 convert_layout_acc_Aregs."""
        layout = acc.layout
        n8 = layout.shape[0][2]
        assert n8 % 4 == 0
        frg_layout = cute.make_layout(
            ((4, 2, 2), layout.shape[1], (n8 // 4, layout.shape[2])),
            stride=(
                (1, 4, 8),
                layout.stride[1],
                (16, layout.stride[2]),
            ),
        )
        return cute.make_tensor(acc.iterator, frg_layout)

    @cute.jit
    def _prepare_fp8_probabilities(self, acc_S, tOrP, tOrP_res):
        """Convert softmax probabilities into compensated FP8 PV A fragments."""
        self._permute_Cregs_fp8(acc_S)
        tP_acc = self._reshape_acc_to_fp8_frgA(acc_S)
        p_scaled = tP_acc.load() * Float32(self.fp8_p_scale)
        p_hi = p_scaled.to(self.kv_dtype)
        tOrP.store(p_hi)
        p_res = (p_scaled - p_hi.to(Float32)) * Float32(self.fp8_p_res_scale)
        tOrP_res.store(p_res.to(self.kv_dtype))

    @cute.jit
    def first_half_block_overlap_fp8(
        self,
        n_block: Int32,
        mma_qk_fn: Callable,
        kv_consumer_state,
        pipeline_k,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,
        acc_O: Optional[cute.Tensor] = None,
        mask_fn: Callable = None,
        score_mod_fn: Optional[Callable] = None,
        is_first_block: bool = False,
        dequant_params: Optional[SimpleNamespace] = None,
    ):
        """FP8-KV first half: wait for K in its sK slot, issue QK, release the slot."""
        dp = dequant_params
        ks, kp = self._kslot(n_block, dp.nmax)
        pipeline_k.consumer_wait_w_index_phase(ks, kp)
        acc_S = mma_qk_fn(B_idx=ks, wg_wait=0)
        pipeline_k.consumer_release_w_index(ks)

        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        mask_fn(acc_S, n_block=n_block, mask_seqlen=True)
        row_scale = softmax.online_softmax(acc_S, is_first=is_first_block)
        self._prepare_fp8_probabilities(acc_S, tOrP, smem_copy_params.tOrP_res)
        if const_expr(self.rescale_O_before_gemm):
            acc_O.fill(0.0)
            scores_scale.store(row_scale.load())
        return kv_consumer_state

    @cute.jit
    def last_half_block_overlap(
        self,
        kv_consumer_state,
        pipeline_v,
        mma_pv_fn: Callable,
        zero_init: bool,
        scores_scale: Optional[cute.Tensor] = None,
        softmax: Optional[Softmax] = None,
        acc_O: Optional[cute.Tensor] = None,
    ):
        """Processes the final PV GEMM when using intra-warpgroup-overlap"""

        # For RescaleOBeforeGemm: rescale O before the final PV GEMM
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)

        pipeline_v.consumer_wait(kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state))
        mma_pv_fn(B_idx=kv_consumer_state.index, zero_init=zero_init, wg_wait=0)
        pipeline_v.consumer_release(kv_consumer_state)
        kv_consumer_state.advance()
        return kv_consumer_state

    @cute.jit
    def last_half_block_overlap_fp8(
        self,
        kv_consumer_state,
        pipeline_v,
        mma_pv_fn: Callable,
        zero_init: bool,
        scores_scale: Optional[cute.Tensor] = None,
        softmax: Optional[Softmax] = None,
        acc_O: Optional[cute.Tensor] = None,
        dequant_params: Optional[SimpleNamespace] = None,
    ):
        """FP8-KV final PV: each WG rearranges its V half before the synchronous PV."""
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)
        dp = dequant_params
        nthr_wg = const_expr(self.num_threads_per_warp_group)
        self._transpose_staged_fp8_v(dp, kv_consumer_state.phase)
        if dp.warp_group_idx == 0:
            cute.arch.barrier(barrier_id=int(NamedBarrierFwd.DequantV0), number_of_threads=nthr_wg)
        else:
            cute.arch.barrier(barrier_id=int(NamedBarrierFwd.DequantV1), number_of_threads=nthr_wg)
        mma_pv_fn(B_idx=kv_consumer_state.index, zero_init=zero_init, wg_wait=0)
        self._permute_output_fp8(acc_O)
        kv_consumer_state.advance()
        return kv_consumer_state

    @cute.jit
    def mma_one_n_block(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,  # not used
        score_mod_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ):
        pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
        # S = Q @ K.T
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(smem_pipe_read)

        # handle score mods and masking
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)

        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)
        # if cute.arch.thread_idx()[0] == 0: cute.print_tensor(layout_utils.reshape_acc_to_mn(acc_S))
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        # tOrP.store(tOrP_acc.load().to(self.dtype))
        # the "to(self.dtype)" conversion fails to vectorize for block sizes other
        # than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
        # 2 elements. So we just call ptx directly.
        utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            self.cross_wg_p_barrier()
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        softmax.rescale_O(acc_O, row_scale)
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
            self.cross_wg_p_raw_barrier()
        pipeline_v.consumer_wait(smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read))
        self.warp_scheduler_barrier_sync()
        # O += P @ V
        mma_pv_fn(B_idx=smem_pipe_read.index, wg_wait=0)
        pipeline_v.consumer_release(smem_pipe_read)
        smem_pipe_read.advance()
        return smem_pipe_read

    @cute.jit
    def mma_one_n_block_intrawg_overlap(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,
        score_mod_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
        check_inf: cutlass.Constexpr = True,
    ):
        smem_pipe_read_v = smem_pipe_read.clone()
        smem_pipe_read.advance()
        pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
        self.warp_scheduler_barrier_sync()
        # S = Q @ K.T
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        # RescaleOBeforeGemm: rescale O while QK GEMM is in flight, before PV GEMM
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)
        pipeline_v.consumer_wait(smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v))
        # O += P @ V
        mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(1)
        pipeline_k.consumer_release(smem_pipe_read)

        # handle score mods and masking
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(layout_utils.reshape_acc_to_mn(acc_S))

        row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
        warpgroup.wait_group(0)
        pipeline_v.consumer_release(smem_pipe_read_v)
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        # tOrP_cur.store(tOrP_acc.load().to(self.dtype))
        # the "to(self.dtype)" conversion fails to vectorize for block sizes other
        # than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
        # 2 elements. So we just call ptx directly.
        utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            self.cross_wg_p_barrier()
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        if const_expr(not self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, row_scale)
        if const_expr(self.rescale_O_before_gemm):
            scores_scale.store(row_scale.load())
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
            self.cross_wg_p_raw_barrier()
        return smem_pipe_read

    @cute.jit
    def mma_one_n_block_intrawg_overlap_fp8(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,
        score_mod_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
        check_inf: cutlass.Constexpr = True,
        dequant_params: Optional[SimpleNamespace] = None,
        qk_split_acc: Optional[cute.Tensor] = None,
    ):
        # FP8-KV intra-WG overlap: wait for K[n] for QK[n], then cast V[n_prev]
        # between the two Q terms of QK[n] for PV[n_prev]. Separate from the generic overlap
        # to keep that path unchanged.
        smem_pipe_read_v = smem_pipe_read.clone()
        smem_pipe_read.advance()
        dp = dequant_params
        nthr_wg = const_expr(self.num_threads_per_warp_group)
        # ---- 1. wait for K[n] in its sK slot ----
        ks, kp = self._kslot(n_block, dp.nmax)
        pipeline_k.consumer_wait_w_index_phase(ks, kp)

        # ---- 3. cast V[n_prev] -> sV (WG-split) while the first Q term's QGMMAs run ----
        def _vstep():
            self._transpose_staged_fp8_v(dp, smem_pipe_read_v.phase)
            if dp.warp_group_idx == 0:
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.DequantV0), number_of_threads=nthr_wg
                )
            else:
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.DequantV1), number_of_threads=nthr_wg
                )

        # ---- 2. S = Q @ K.T (async), with step 3 between its two Q terms ----
        self.warp_scheduler_barrier_sync()
        if const_expr(self.qk_split):
            acc_S = mma_qk_fn(B_idx=ks, wg_wait=-1, mid=_vstep, split_acc=qk_split_acc)
        else:
            acc_S = mma_qk_fn(B_idx=ks, wg_wait=-1, mid=_vstep)
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)
        # ---- 4. O += P @ V; the promoted passes wait for QK and PV ----
        if const_expr(self.qk_split):
            # The PV passes reuse the scratch: fold QK and finish S before they start.
            warpgroup.wait_group(0)
            self._fold_qk_split(acc_S, qk_split_acc)
        else:
            mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)
            self.warp_scheduler_barrier_arrive()
            warpgroup.wait_group(1)
        pipeline_k.consumer_release_w_index(ks)
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
        if const_expr(self.qk_split):
            mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)
            self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(0)
        self._prepare_fp8_probabilities(acc_S, tOrP, smem_copy_params.tOrP_res)
        if const_expr(not self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, row_scale)
        if const_expr(self.rescale_O_before_gemm):
            scores_scale.store(row_scale.load())
        return smem_pipe_read

    @cute.jit
    def cross_wg_p_barrier(self):
        """WAR barrier on sP for hdim > 256: both MMA warpgroups read the whole sP in the PV
        GEMM, so neither may overwrite it with the next P before the other is done reading."""
        if const_expr(self.p_shared_across_wg):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.PSync), number_of_threads=self.num_mma_threads
            )

    @cute.jit
    def cross_wg_p_raw_barrier(self):
        """RAW barrier on sP for hdim > 256: after the P store and proxy fence, before the PV
        GEMM, so neither warpgroup's wgmma reads sP before the other's store is fenced."""
        if const_expr(self.p_shared_across_wg):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.PSyncRAW), number_of_threads=self.num_mma_threads
            )

    @cute.jit
    def mma_init(self):
        warp_group_idx = utils.canonical_warp_group_idx(sync=False)
        if const_expr(self.use_scheduler_barrier):
            if warp_group_idx == 1:
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1),
                    number_of_threads=2 * self.num_threads_per_warp_group,
                )

    @cute.jit
    def apply_score_mod(
        self,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        acc_S,
        n_block,
        softmax_scale,
        seqlen,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
    ):
        # Prepare index tensor
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        cS = cute.domain_offset((m_block * self.tile_m, n_block * self.tile_n), cS)
        tScS = thr_mma_qk.partition_C(cS)

        apply_score_mod_inner(
            acc_S,
            tScS,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            self.score_vec_size,
            self.qk_acc_dtype,
            aux_data,
            fastdiv_mods,
            seqlen_info=seqlen,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )

    def warp_scheduler_barrier_sync(self):
        if const_expr(self.use_scheduler_barrier):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1)
                - 1
                + utils.canonical_warp_group_idx(sync=False),
                number_of_threads=2 * self.num_threads_per_warp_group,
            )

    def warp_scheduler_barrier_arrive(self):
        if const_expr(self.use_scheduler_barrier):
            assert self.num_wg_mma in [2, 3]
            cur_wg = utils.canonical_warp_group_idx(sync=False) - 1
            if const_expr(self.num_wg_mma == 2):
                next_wg = 1 - cur_wg
            else:
                t = cur_wg + 1
                next_wg = t % self.num_wg_mma
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg,
                number_of_threads=2 * self.num_threads_per_warp_group,
            )


# FlashAttentionForwardSm90Fp8KV: SM90 forward for FP8 (E4M3) paged K/V with head_dim 512.
#
# One CTA handles one (tile_m = 64)-row Q tile with four warp groups:
#   - warp group 0, producer: TMA (or cp.async) Q, TMA K and V pages into two slots each.
#   - warp groups 1-2, PV: quantize Q into two E4M3 terms, then compute O^T = V^T P^T with FP16
#     WGMMA. Each owns half of head_dim. A = V^T is read from the E4M3 V tile with
#     ldmatrix.trans and widened to FP16, which is exact; B = P comes from shared memory.
#   - warp group 3, softmax: S = Q K^T as FP8 WGMMA of both Q terms, online softmax with a lazy
#     row max, then FP16 P and its row rescales into one of two shared memory slots.
# O is written straight from the PV registers: the output dtype, or FP32 partials with SplitKV.


class Fp8KVDescaleTensors(NamedTuple):
    """Per-(batch, kv_head) FP32 descales; any slot may be None, which means 1.

    Rebuilds each present slot in place, so a missing slot does not shift the later ones.
    """

    q_descale: Optional[cute.Tensor] = None
    k_descale: Optional[cute.Tensor] = None
    v_descale: Optional[cute.Tensor] = None

    def __new_from_mlir_values__(self, values):
        values = iter(values)
        return Fp8KVDescaleTensors(*(next(values) if t is not None else None for t in self))


class FlashAttentionForwardSm90Fp8KV(FlashAttentionForwardSm90):
    """SM90 forward for BF16/FP16 Q with paged E4M3 K/V, head_dim = head_dim_v = 512.

    Supported: varlen Q (cu_seqlens_q, no seqused_q), paged K/V with page_size == tile_n == 64
    and seqused_k, causal, local, softcap (as score_mod), learnable sink, SplitKV, pack_gqa,
    and q/k/v descales of shape (batch, num_heads_kv). Not supported: mask_mod, block
    sparsity, dynamic causal, aux tensors, output_scale.

    Accuracy: Q becomes hi + res, two E4M3 terms at a fixed scale of 1 (see
    utils.e4m3_hi_res_x8 for the exact range). QK runs as FP8 WGMMA, which sums below FP32.
    P is FP16, scaled by 2^max_offset, and V is widened exactly from E4M3 to FP16.
    """

    # Lazy row max: the reference max is raised only when a row max grows by more than
    # 2^rescale_threshold, so most blocks skip the O rescale; P then stays below
    # 2^rescale_threshold. P is stored as P * 2^max_offset so that small probabilities stay
    # FP16 normal; see Note [Low Precision Scaling] in flash_fwd_sm100.py.
    rescale_threshold = 8.0
    max_offset = 7

    def __init__(
        self,
        dtype,
        head_dim: int,
        head_dim_v: int,
        qhead_per_kvhead: int,
        *,
        is_causal: bool = False,
        is_local: bool = False,
        is_split_kv: bool = False,
        pack_gqa: bool = True,
        tile_m: int = 64,
        tile_n: int = 64,
        score_mod: Optional[cutlass.Constexpr] = None,
        kv_dtype=cutlass.Float8E4M3FN,
    ):
        self.use_tma_Q = not (pack_gqa and tile_m % qhead_per_kvhead != 0)
        # Register split per thread (4 warp groups x 128 threads share 64K registers). A
        # cp.async Q load runs on all 128 producer threads and needs more producer registers.
        if self.use_tma_Q:
            producer_regs, self.num_softmax_regs = 24, 152
        else:
            producer_regs, self.num_softmax_regs = 40, 136
        pv_regs = 168
        super().__init__(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            is_causal=is_causal,
            is_local=is_local,
            pack_gqa=pack_gqa,
            tile_m=tile_m,
            tile_n=tile_n,
            num_stages=1,
            Q_in_regs=False,
            score_mod=score_mod,
            intra_wg_overlap=True,
            mma_pv_is_rs=True,
            is_split_kv=is_split_kv,
            kv_dtype=kv_dtype,
            fp8_kv_dequant=True,
            registers=Sm90Registers(mma=pv_regs, producer=producer_regs),
        )
        assert dtype == Float16, "P and the PV WGMMA use FP16"
        assert kv_dtype == cutlass.Float8E4M3FN, "K/V must be E4M3"
        assert self.tile_hdim == self.tile_hdimv == 512 and not self.check_hdim_oob
        assert tile_m == 64 and tile_n == 64, "one 64 x 64 WGMMA tile per QK and per P slot"
        assert self.max_offset + self.rescale_threshold < math.log2(65504.0)

        self.num_threads_per_warp_group = 128
        self.num_pv_warp_groups = 2
        self.num_pv_threads = self.num_pv_warp_groups * self.num_threads_per_warp_group
        self.num_softmax_threads = self.num_threads_per_warp_group
        self.num_threads = (
            self.num_threads_per_warp_group + self.num_pv_threads + (self.num_softmax_threads)
        )
        assert producer_regs + self.num_pv_warp_groups * pv_regs + self.num_softmax_regs <= (
            65536 // self.num_threads_per_warp_group
        )
        # Threads in the hand-off barriers between the softmax and the PV warp groups.
        self.num_handoff_threads = self.num_pv_threads + self.num_softmax_threads
        self.num_producer_threads = 32
        self.num_Q_load_threads = self.num_threads_per_warp_group
        self.num_epilogue_threads = self.num_pv_threads
        # Head-dim columns per PV warp group and 64-row WGMMA M tiles in each.
        self.hdim_per_pv_wg = self.tile_hdimv // self.num_pv_warp_groups
        self.num_pv_m_tiles = self.hdim_per_pv_wg // 64
        # PV WGMMA groups kept in flight while the next V^T fragment is converted.
        self.pv_wgmma_window = 2
        # K and V each use two shared memory slots.
        self.num_kv_slots = 2

    def _get_tiled_mma(self):
        # QK: the two K-mode slices of this tiled MMA partition interleaved head_dim halves.
        # The softmax warp group issues both halves itself into separate accumulators.
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.kv_dtype,
            self.kv_dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 2),
            tiler_mn=(64, self.tile_n),
        )
        # PV, per warp group and 64 head_dim rows: O^T (hd, q) += V^T (hd, key) P^T (key, q).
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(64, self.tile_m),
            a_source=warpgroup.OperandSource.RMEM,
        )
        # Ownership of the (tile_m, head_dim_v) output: one head_dim half per PV warp group.
        # Not issued; the epilogue writes LSE through its C partition and each PV thread reads
        # its LSE rows through the same partition.
        tiled_mma_o = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, self.num_pv_warp_groups, 1),
            tiler_mn=(64, self.hdim_per_pv_wg),
            a_source=warpgroup.OperandSource.RMEM,
        )
        return tiled_mma_qk, tiled_mma_pv, tiled_mma_o

    def _get_shared_storage_cls(self):
        align = self.buffer_align_bytes

        @cute.struct
        class SharedStorage:
            # The Q input; after quantization it holds Q_res, then Q_hi.
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(self.sQ_layout)], align
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.kv_dtype, cute.cosize(self.sK_layout)], align
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.kv_dtype, cute.cosize(self.sV_layout)], align
            ]
            sP: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(self.sP_layout)], align
            ]
            # Per P slot: the online-softmax row rescales and one changed flag per softmax warp.
            sRowScale: cute.struct.Align[
                cute.struct.MemRange[Float32, self.tile_m * self.num_kv_slots], 128
            ]
            sRowScaleChanged: cute.struct.MemRange[Int32, 4 * self.num_kv_slots]
            # Final per-row output scales and LSE.
            sFinalScale: cute.struct.MemRange[Float32, self.tile_m]
            sLSE: cute.struct.MemRange[Float32, self.tile_m]
            mbar_ptr_Q: cute.struct.MemRange[cutlass.Int64, 2]
            mbar_ptr_K: cute.struct.MemRange[cutlass.Int64, self.num_kv_slots * 2]
            mbar_ptr_V: cute.struct.MemRange[cutlass.Int64, self.num_kv_slots * 2]

        return SharedStorage

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (total_q, h, d)
        mK: cute.Tensor,  # (num_pages, page_size, h_k, d)
        mV: cute.Tensor,  # (num_pages, page_size, h_k, dv)
        mO: cute.Tensor,  # (total_q, h, dv), or (num_splits, total_q, h, dv) FP32 with SplitKV
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mDynamicCausal: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,  # (b, max_num_pages_per_seq)
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_data: AuxData = AuxData(),
        descale_tensors: Optional[Fp8KVDescaleTensors] = None,
        output_scale: Optional[cute.Tensor] = None,
        mCuTotalMBlocks: Optional[cute.Tensor] = None,
        mCuTotalSplitsMBlocks: Optional[cute.Tensor] = None,
        tensormap_workspace: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Launch the kernel. Same positional signature as FlashAttentionForwardSm90."""
        assert mCuSeqlensQ is not None and mSeqUsedQ is None, "varlen Q through cu_seqlens_q"
        assert mPageTable is not None and mCuSeqlensK is None, "paged K/V only"
        assert mDynamicCausal is None, "dynamic causal is not supported"
        assert blocksparse_tensors is None, "block sparsity is not supported"
        assert self.mask_mod is None, "mask_mod is not supported"
        assert aux_data.tensors is None, "aux tensors are not supported"
        assert output_scale is None, "fused output quantization is not supported"
        assert mQ.element_type in (Float16, BFloat16), "Q must be fp16 or bf16"
        assert mK.element_type == self.kv_dtype and mV.element_type == self.kv_dtype
        if const_expr(self.is_split_kv):
            assert mO.element_type == Float32, "SplitKV writes FP32 partials"
        else:
            assert mO.element_type in (Float16, BFloat16), "O must be fp16 or bf16"
        self.q_dtype = mQ.element_type
        self.o_dtype = mO.element_type
        self.varlen_q = True
        self.use_tma_O = False  # O is stored from registers
        self.use_block_sparsity = False

        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        # (total_q, h, d) -> (total_q, d, h); paged K/V -> (page_size, d, h_k, num_pages).
        mQ = layout_utils.select(mQ, [0, 2, 1])
        num_splits = Int32(1)
        if const_expr(not self.is_split_kv):
            mO = layout_utils.select(mO, [0, 2, 1])
            mLSE = layout_utils.select(mLSE, [1, 0]) if const_expr(mLSE is not None) else None
        else:
            num_splits = mO.shape[0]
            mO = layout_utils.select(mO, [1, 3, 2, 0])
            mLSE = layout_utils.select(mLSE, [2, 1, 0]) if const_expr(mLSE is not None) else None
        mK, mV = [layout_utils.select(t, [1, 3, 2, 0]) for t in (mK, mV)]

        tiled_mma_qk, tiled_mma_pv, tiled_mma_o = self._get_tiled_mma()
        self._setup_attributes()
        self.sQ_layout = sm90_utils.make_smem_layout(
            self.q_dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_hdim), None
        )
        # Q_res and Q_hi: two E4M3 tiles that reuse the Q input bytes.
        self.sQ_fp8_layout = sm90_utils.make_smem_layout(
            self.kv_dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_hdim), None
        )
        self.sK_layout = sm90_utils.make_smem_layout(
            self.kv_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdim), self.num_kv_slots
        )
        self.sV_layout = sm90_utils.make_smem_layout(
            self.kv_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdimv), self.num_kv_slots
        )
        # FP16 P (query, key), one tile per slot: the K-major B operand of the PV WGMMA.
        self.sP_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_n), self.num_kv_slots
        )
        q_fp8_bytes = cute.cosize(self.sQ_fp8_layout) * self.kv_dtype.width // 8
        assert 2 * q_fp8_bytes <= cute.cosize(self.sQ_layout) * self.q_dtype.width // 8
        # Q_hi starts q_fp8_bytes after Q_res; a whole number of swizzle periods keeps its
        # swizzle phase equal to that of its own layout.
        assert q_fp8_bytes % self.buffer_align_bytes == 0
        SharedStorage = self._get_shared_storage_cls()

        mQ_og = mQ
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(dtype, cute.select(layout, mode=[0, 1]))
            for name, dtype, layout in [
                ("Q", self.q_dtype, self.sQ_layout),
                ("K", self.kv_dtype, self.sK_layout),
                ("V", self.kv_dtype, self.sV_layout),
            ]
        }
        tma_atom_Q = None
        mQ_load = mQ
        if const_expr(self.use_tma_Q):
            make_tiled_tma_atom_fn = (
                partial(
                    make_packgqa_tiled_tma_atom, qhead_per_kvhead=self.qhead_per_kvhead, head_idx=2
                )
                if const_expr(self.pack_gqa)
                else cpasync.make_tiled_tma_atom
            )
            tma_atom_Q, mQ_load = make_tiled_tma_atom_fn(
                cpasync.CopyBulkTensorTileG2SOp(),
                mQ_og if const_expr(self.pack_gqa) else mQ,
                self.sQ_layout,
                (self.tile_m, self.tile_hdim),
            )
        tma_atom_K, mK_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
            1,
        )
        tma_atom_V, mV_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
            1,
        )

        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mCuSeqlensQ.shape[0] - 1),
            num_splits,
            mK.shape[0] * mPageTable.shape[1],
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=None,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
            cu_total_m_blocks_ptr=mCuTotalMBlocks,
            cu_total_splits_m_blocks_ptr=mCuTotalSplitsMBlocks,
        )
        tile_sched_params = SingleTileVarlenScheduler.to_underlying_arguments(tile_sched_args)
        # The K/V slot parity (_kslot) restarts per tile: one tile per CTA.
        assert tile_sched_params.scheduling_mode == SchedulingMode.STATIC
        grid_dim = SingleTileVarlenScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
            softmax_scale, self.score_mod
        )
        window_size_left = Int32(window_size_left) if window_size_left is not None else None
        window_size_right = Int32(window_size_right) if window_size_right is not None else None

        self.kernel(
            mQ_load,
            mK_tma,
            mV_tma,
            mO,
            mLSE,
            mCuSeqlensQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            self.sQ_layout,
            self.sQ_fp8_layout,
            self.sK_layout,
            self.sV_layout,
            self.sP_layout,
            self.gmem_tiled_copy_Q,
            tiled_mma_qk,
            tiled_mma_pv,
            tiled_mma_o,
            tile_sched_params,
            SingleTileVarlenScheduler,
            SharedStorage,
            num_splits,
            aux_data,
            descale_tensors,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: cute.Tensor,
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        sQ_layout: cute.ComposedLayout,
        sQ_fp8_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout,
        gmem_tiled_copy_Q: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_o: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        num_splits: Int32,
        aux_data: AuxData,
        descale_tensors: Optional[Fp8KVDescaleTensors],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        warp_size = cute.arch.WARP_SIZE
        tma_warp = ThreadCooperativeGroup(1)
        load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
        pv_warps = ThreadCooperativeGroup(self.num_pv_threads // warp_size)
        softmax_warps = ThreadCooperativeGroup(self.num_softmax_threads // warp_size)
        # Q is consumed (quantized) by the PV warp groups.
        if const_expr(self.use_tma_Q):
            pipeline_q = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_Q.data_ptr(),
                num_stages=1,
                producer_group=tma_warp,
                consumer_group=pv_warps,
                tx_count=self.tma_copy_bytes["Q"],
                defer_sync=True,
            )
        else:
            pipeline_q = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_Q.data_ptr(),
                num_stages=1,
                producer_group=load_threads,
                consumer_group=pv_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )
        pipeline_k = pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=self.num_kv_slots,
            producer_group=tma_warp,
            consumer_group=softmax_warps,
            tx_count=self.tma_copy_bytes["K"],
            defer_sync=True,
        )
        pipeline_v = pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_V.data_ptr(),
            num_stages=self.num_kv_slots,
            producer_group=tma_warp,
            consumer_group=pv_warps,
            tx_count=self.tma_copy_bytes["V"],
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sQ_res = storage.sQ.get_tensor(
            sQ_fp8_layout.outer, swizzle=sQ_fp8_layout.inner, dtype=self.kv_dtype
        )
        sQ_hi = cute.make_tensor(sQ_res.iterator + cute.cosize(sQ_fp8_layout), sQ_res.layout)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        row_stats = (
            storage.sRowScale.get_tensor(cute.make_layout((self.tile_m, self.num_kv_slots))),
            storage.sRowScaleChanged.get_tensor(cute.make_layout((4, self.num_kv_slots))),
            storage.sFinalScale.get_tensor(cute.make_layout(self.tile_m)),
            storage.sLSE.get_tensor(cute.make_layout(self.tile_m)),
        )

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            num_splits=num_splits,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=None,
            mSeqUsedQ=None,
            mSeqUsedK=mSeqUsedK,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        warps_per_wg = self.num_threads_per_warp_group // warp_size
        first_softmax_warp = warps_per_wg * (1 + self.num_pv_warp_groups)
        tidx, _, _ = cute.arch.thread_idx()
        if warp_idx < warps_per_wg:
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            self.load(
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_q,
                pipeline_k,
                pipeline_v,
                gmem_tiled_copy_Q,
                mPageTable,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                num_splits,
            )
        elif warp_idx >= first_softmax_warp:
            cute.arch.setmaxregister_increase(self.num_softmax_regs)
            self.softmax_wg(
                tiled_mma_qk,
                sQ_hi,
                sQ_res,
                sK,
                sP,
                row_stats,
                learnable_sink,
                pipeline_k,
                tidx - first_softmax_warp * warp_size,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                aux_data,
                num_splits,
                descale_tensors,
            )
        else:
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            self.pv_wg(
                tiled_mma_pv,
                tiled_mma_o,
                mO,
                mLSE,
                sQ,
                sQ_hi,
                sQ_res,
                sV,
                sP,
                row_stats,
                pipeline_q,
                pipeline_v,
                tidx - self.num_threads_per_warp_group,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                num_splits,
            )

    @cute.jit
    def _kslot(self, n_block, n_block_max):
        """K (and V) slot of block n_block and its fill parity; fills run n_block_max - 1 down.

        Requires 0 <= n_block < n_block_max.
        """
        return n_block & 1, ((n_block_max - 1 - n_block) >> 1) & 1

    @cute.jit
    def _block_range(self, block_info, seqlen, m_block, split_idx, num_splits):
        """[n_block_min, n_block_max) of this tile, and whether it is empty. An empty range
        becomes the single block 0 so that every role keeps one K and one V transaction; its
        result is discarded."""
        n_block_min, n_block_max = block_info.get_n_block_min_max(
            seqlen, m_block, split_idx, num_splits
        )
        is_empty = n_block_min >= n_block_max
        if is_empty:
            n_block_min = Int32(0)
            n_block_max = Int32(1)
        return n_block_min, n_block_max, is_empty

    # ///////////////////////////////////////////////////////////////////////////////////////
    # Producer
    # ///////////////////////////////////////////////////////////////////////////////////////

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        mPageTable: cute.Tensor,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        num_splits: Int32,
    ):
        """Q first (it is quantized before the first QK), then K[n_max - 1], then K[n] and
        V[n + 1] for n descending, then V[n_min]. Each page index is read once."""
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tidx, _, _ = cute.arch.thread_idx()
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        if work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            head_idx_kv = (
                head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
            )
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            if const_expr(self.use_tma_Q):
                if warp_idx_in_wg == 0:
                    gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
                    load_Q, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
                    )
                    pipeline_q.producer_acquire_w_index_phase(0, Int32(1))
                    load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
            else:
                # All 128 producer threads load Q with cp.async.
                pack_gqa = PackGQA(
                    self.tile_m, self.tile_hdim, self.check_hdim_oob, self.qhead_per_kvhead
                )
                pipeline_q.producer_acquire_w_index_phase(0, Int32(1))
                pack_gqa.load_Q(mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q)
                cute.arch.cp_async_commit_group()
                pipeline_q.producer_commit_w_index(0)

            n_block_min, n_block_max, is_empty = self._block_range(
                block_info, seqlen, m_block, split_idx, num_splits
            )
            if warp_idx_in_wg == 0:
                # Paged TMA tiles, page dimension indexable.
                gK = cute.local_tile(
                    mK[None, None, head_idx_kv, None], (self.tile_n, self.tile_hdim), (0, 0, None)
                )
                gV = cute.local_tile(
                    mV[None, None, head_idx_kv, None], (self.tile_n, self.tile_hdimv), (0, 0, None)
                )
                load_K, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_K, 0, cute.make_layout(1), gK, sK
                )
                load_V, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_V, 0, cute.make_layout(1), gV, sV
                )
                issue = partial(self._issue_kv, n_block_max=n_block_max)
                # An empty range reads physical page 0, a valid dummy source.
                page = Int32(0)
                if not is_empty:
                    page = mPageTable[batch_idx, n_block_max - 1]
                issue(pipeline_k, load_K, n_block_max - 1, page)
                for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                    n_block = n_block_max - 2 - i
                    page_next = page
                    page = mPageTable[batch_idx, n_block]
                    issue(pipeline_k, load_K, n_block, page)
                    issue(pipeline_v, load_V, n_block + 1, page_next)
                issue(pipeline_v, load_V, n_block_min, page)

    @cute.jit
    def _issue_kv(self, pipe, copy_fn, n_block, page, n_block_max):
        """TMA K or V page `page` of block n_block into its slot."""
        slot, phase = self._kslot(n_block, n_block_max)
        pipe.producer_acquire_w_index_phase(slot, phase ^ 1)
        copy_fn(src_idx=page, dst_idx=slot, tma_bar_ptr=pipe.sync_object_full.get_barrier(slot))

    # ///////////////////////////////////////////////////////////////////////////////////////
    # Softmax warp group
    # ///////////////////////////////////////////////////////////////////////////////////////

    @cute.jit
    def softmax_wg(
        self,
        tiled_mma_qk: cute.TiledMma,
        sQ_hi: cute.Tensor,
        sQ_res: cute.Tensor,
        sK: cute.Tensor,
        sP: cute.Tensor,
        row_stats: tuple,
        learnable_sink: Optional[cute.Tensor],
        pipeline_k: pipeline.PipelineAsync,
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        aux_data: AuxData,
        num_splits: Int32,
        descale_tensors: Optional[Fp8KVDescaleTensors],
    ):
        """QK and online softmax over all KV blocks of the tile; hands P and its row rescales
        to the PV warp groups block by block, then the final row scales and LSE."""
        sRowScale, sRowScaleChanged, sFinalScale, sLSE = row_stats
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        # Q_res, Q_hi and K fragments of each head_dim half (K-mode slice of tiled_mma_qk).
        qk_halves = []
        for h in cutlass.range_constexpr(2):
            wg_slice = tiled_mma_qk.get_slice(h * self.num_threads_per_warp_group)
            _, tQhi, tK = sm90_utils.partition_fragment_ABC(
                wg_slice, (self.tile_m, self.tile_n, self.tile_hdim), sQ_hi, sK
            )
            _, tQres, _ = sm90_utils.partition_fragment_ABC(
                wg_slice, (self.tile_m, self.tile_n, self.tile_hdim), sQ_res, sK
            )
            qk_halves.append((tQres, tQhi, tK))
        smem_thr_copy_P = cute.make_tiled_copy_C(
            utils.get_smem_store_atom(self.arch.major * 10 + self.arch.minor, self.dtype),
            tiled_mma_qk,
        ).get_slice(tidx)
        # (row, col) coordinate of each S element.
        tScS = layout_utils.reshape_acc_to_mn(
            thr_mma_qk.partition_C(cute.make_identity_tensor((self.tile_m, self.tile_n)))
        )
        sm = SimpleNamespace(
            thr_mma_qk=thr_mma_qk,
            qk_halves=qk_halves,
            # Masked S of the block whose softmax is next.
            S=cute.make_rmem_tensor(
                tiled_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
            ),
            smem_thr_copy_P=smem_thr_copy_P,
            tPsP=smem_thr_copy_P.partition_D(sP),
            is_col0=tScS[0][1] == 0,
            sRowScale=sRowScale,
            sRowScaleChanged=sRowScaleChanged,
            warp=tidx // cute.arch.WARP_SIZE,
            lane=tidx % cute.arch.WARP_SIZE,
        )
        num_rows = cute.size(tScS, mode=[0])

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        if work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            head_idx_kv = (
                head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
            )
            # q * k descales fold into the score scale, v into the final output scale.
            qk_descale, v_descale = self._load_effective_descales(
                descale_tensors, batch_idx, head_idx_kv
            )
            v_descale = v_descale / Float32(2**self.max_offset)
            if const_expr(self.score_mod is None):
                scale_log2_eff = softmax_scale_log2 * qk_descale
                softmax_scale_eff = None
            else:
                scale_log2_eff = softmax_scale_log2
                softmax_scale_eff = softmax_scale * qk_descale
            softmax = Softmax.create(
                scale_log2_eff, num_rows=num_rows, softmax_scale=softmax_scale_eff
            )
            mask = AttentionMaskCls(seqlen)
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                aux_data=aux_data,
            )
            score_mod_fn = None
            if const_expr(self.score_mod is not None):
                score_mod_fn = partial(
                    self.apply_score_mod,
                    thr_mma_qk,
                    batch_idx,
                    head_idx,
                    m_block,
                    softmax_scale=softmax_scale_eff,
                    aux_data=aux_data,
                )
            n_block_min, n_block_max, is_empty = self._block_range(
                block_info, seqlen, m_block, split_idx, num_splits
            )
            sink_val = None
            if const_expr(learnable_sink is not None):
                sink_val = self._load_sink(learnable_sink, tScS, m_block, head_idx, split_idx)
            # Wait until the PV warp groups have written Q_hi and Q_res.
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwdSm90Fp8KV.QReady),
                number_of_threads=self.num_handoff_threads,
            )
            nmax = n_block_max
            # Block nmax - 1: both QK halves, then seed the reference max (the softmax of
            # each block runs one step after its QK).
            slot, phase = self._kslot(nmax - 1, nmax)
            pipeline_k.consumer_wait_w_index_phase(slot, phase)
            accs = []
            for h in cutlass.range_constexpr(2):
                accs.append(self._qk_half(tiled_mma_qk, sm, h, slot))
            warpgroup.wait_group(0)
            self._fold_s(
                sm,
                accs,
                pipeline_k,
                nmax - 1,
                nmax,
                seqlen,
                partial(mask_fn, mask_mod=None, mask_seqlen=True),
                score_mod_fn,
            )
            self._init_row_max(softmax, sm.S)
            n_block_max -= 1
            step = partial(
                self._softmax_step,
                tiled_mma_qk,
                sm,
                pipeline_k,
                softmax,
                seqlen,
                score_mod_fn,
                nmax,
            )
            if const_expr(self.is_causal or self.is_local):
                n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                    seqlen, m_block, n_block_min
                )
                for n_tile in cutlass.range(n_block_max - n_block_min_causal_local_mask, unroll=1):
                    step(
                        n_block_max - 1 - n_tile,
                        mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=False),
                    )
                n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
            n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                seqlen, m_block, n_block_min
            )
            for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                step(n_block_max - 1 - n_tile, mask_fn=None)
            if const_expr(self.is_local and block_info.window_size_left is not None):
                n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                    step(
                        n_block_max - 1 - n_tile,
                        mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=False),
                    )
            # The last step stored P[n_block_min + 1] without signalling it.
            if n_block_min + 1 < nmax:
                self._signal_p(n_block_min + 1)
            row_scale = self._online_softmax_lazy_max(softmax, sm.S)
            self._publish_p(sm, row_scale, n_block_min, n_block_min < nmax - 2)
            self._signal_p(n_block_min)
            # The row sums hold sum(P * 2^max_offset): undo the power of two exactly.
            softmax.row_sum.store(softmax.row_sum.load() * Float32(2.0**-self.max_offset))
            final_scale = softmax.finalize(final_scale=v_descale, sink_val=sink_val)
            if is_empty:
                # O = 0 and no weight in the SplitKV combine.
                final_scale.fill(0.0)
                softmax.row_sum.fill(-Float32.inf)
            tSsFinalScale = self._row_view(thr_mma_qk, sFinalScale)
            tSsLSE = self._row_view(thr_mma_qk, sLSE)
            # One writer per row: the thread holding its column 0.
            if sm.is_col0:
                for r in cutlass.range_constexpr(num_rows):
                    tSsFinalScale[r, 0] = final_scale[r]
                    tSsLSE[r, 0] = softmax.row_sum[r]
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwdSm90Fp8KV.RowStats),
                number_of_threads=self.num_handoff_threads,
            )

    @cute.jit
    def _load_sink(self, learnable_sink, tScS, m_block, head_idx, split_idx):
        """Sink logit of this thread's rows (per row with pack_gqa); -inf for split_idx > 0 so
        that a SplitKV launch counts the sink once."""
        if const_expr(not self.pack_gqa):
            sink_val = load_learnable_sink(
                learnable_sink, head_idx, 0, self.qhead_per_kvhead, False
            )
            if const_expr(self.is_split_kv):
                if split_idx > 0:
                    sink_val = -Float32.inf
        else:
            sink_val = cute.make_rmem_tensor(cute.size(tScS, mode=[0]), Float32)
            for r in cutlass.range_constexpr(cute.size(sink_val)):
                sink_val[r] = load_learnable_sink(
                    learnable_sink,
                    head_idx,
                    m_block * self.tile_m + tScS[r, 0][0],
                    self.qhead_per_kvhead,
                    True,
                )
            if const_expr(self.is_split_kv):
                if split_idx > 0:
                    sink_val.fill(-Float32.inf)
        return sink_val

    def _row_view(self, thr_mma, vec):
        """A (tile_m,) per-row vector seen through thr_mma's C partition in (row, col) form:
        [r, 0] is the entry of this thread's row r."""
        v = cute.make_tensor(
            vec.iterator, cute.make_layout((self.tile_m, self.tile_n), stride=(1, 0))
        )
        return layout_utils.reshape_acc_to_mn(thr_mma.partition_C(v))

    @cute.jit
    def _qk_half(self, tiled_mma_qk, sm, h: cutlass.Constexpr[int], k_slot):
        """Issue, without waiting, head_dim half h of S = Q_res K^T + Q_hi K^T into its own
        accumulator. The residual term goes first, while the running sum is small."""
        tQres, tQhi, tK = sm.qk_halves[h]
        acc = cute.make_rmem_tensor(
            tiled_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
        )
        self._gemm_two_fp8_a(tiled_mma_qk, acc, tQres, tQhi, tK[None, None, None, k_slot], True, -1)
        return acc

    @cute.jit
    def _fold_s(self, sm, accs, pipeline_k, n_block, nmax, seqlen, mask_fn, score_mod_fn):
        """Release K[n_block] and fold the two QK halves into sm.S, then score_mod and mask."""
        slot, _ = self._kslot(n_block, nmax)
        pipeline_k.consumer_release_w_index(slot)
        sm.S.store(accs[0].load() + accs[1].load())
        if const_expr(score_mod_fn is not None):
            score_mod_fn(sm.S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=sm.S, n_block=n_block)

    @cute.jit
    def _init_row_max(self, softmax, acc_S):
        """Seed the reference row max with the first block's row max; row sums start at 0."""
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        for r in cutlass.range(cute.size(softmax.row_max), unroll_full=True):
            row_max = utils.fmax_reduce(acc_S_mn[r, None].load(), init_val=None, arch=softmax.arch)
            softmax.row_max[r] = cute.arch.warp_reduction_max(row_max, threads_in_group=4)
            softmax.row_sum[r] = Float32(0.0)

    @cute.jit
    def _online_softmax_lazy_max(self, softmax, acc_S):
        """Online softmax that raises the reference row max only when it grows by more than
        2^rescale_threshold. Writes P * 2^max_offset into acc_S and returns the row rescales,
        which are exactly 1 for rows whose reference max did not change."""
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        row_scale = cute.make_fragment_like(softmax.row_max, Float32)
        row_max = softmax.row_max
        row_sum = softmax.row_sum
        scale_log2 = softmax.scale_log2
        for r in cutlass.range(cute.size(row_max), unroll_full=True):
            acc_S_row = acc_S_mn[r, None].load()
            row_max_cur = utils.fmax_reduce(acc_S_row, init_val=row_max[r], arch=softmax.arch)
            row_max_cur = cute.arch.warp_reduction_max(row_max_cur, threads_in_group=4)
            row_max_ref = row_max[r]
            if (row_max_cur - row_max_ref) * scale_log2 > Float32(self.rescale_threshold):
                row_scale[r] = cute.math.exp2(
                    (row_max_ref - row_max_cur) * scale_log2, fastmath=True
                )
                row_max[r] = row_max_cur
            else:
                row_scale[r] = 1.0
            row_max_safe = row_max[r]
            row_max_safe = 0.0 if row_max_safe == -Float32.inf else row_max_safe
            bias = row_max_safe * scale_log2 - Float32(self.max_offset)
            acc_S_row_exp = cute.math.exp2(acc_S_row * scale_log2 - bias, fastmath=True)
            row_sum[r] = utils.fadd_reduce(
                acc_S_row_exp, init_val=row_sum[r] * row_scale[r], arch=softmax.arch
            )
            acc_S_mn[r, None].store(acc_S_row_exp)
        return row_scale

    @cute.jit
    def _publish_p(self, sm, row_scale, n_block, wait_empty):
        """Store FP16 P[n_block] (from sm.S, already scaled by 2^max_offset) and its row
        rescales into slot n_block % 2, without signalling. wait_empty: first wait until the
        PV warp groups have consumed P[n_block + 2] from that slot."""
        slot = n_block % 2
        tP_acc = layout_utils.reshape_acc_to_frgA(sm.S)
        tP = cute.make_rmem_tensor_like(tP_acc, self.dtype)
        utils.cvt_f16(tP_acc, tP)
        if wait_empty:
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwdSm90Fp8KV.PEmpty0) + slot,
                number_of_threads=self.num_handoff_threads,
            )
        cute.copy(
            sm.smem_thr_copy_P, sm.smem_thr_copy_P.retile(tP), sm.tPsP[None, None, None, slot]
        )
        # Locals rather than attribute reads inside the dynamic branches below.
        sRowScaleChanged = sm.sRowScaleChanged
        tSsRowScale = self._row_view(sm.thr_mma_qk, sm.sRowScale[None, slot])
        lane = sm.lane
        warp = sm.warp
        # The rescales are <= 1 or NaN, so their product is 1 only if all of them are 1.
        prod = Float32(1.0)
        for r in cutlass.range_constexpr(cute.size(row_scale)):
            prod = prod * row_scale[r]
        if sm.is_col0:
            for r in cutlass.range_constexpr(cute.size(row_scale)):
                tSsRowScale[r, 0] = row_scale[r]
        changed = Int32(0)
        if cute.arch.vote_any_sync(prod != Float32(1.0)):
            changed = Int32(1)
        if lane == 0:
            sRowScaleChanged[warp, slot] = changed
        # P is read by WGMMA (async proxy) in both PV warp groups.
        cute.arch.fence_view_async_shared()

    @cute.jit
    def _signal_p(self, n_block):
        """P[n_block] and its row rescales are in slot n_block % 2 and fenced."""
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierFwdSm90Fp8KV.PFull0) + n_block % 2,
            number_of_threads=self.num_handoff_threads,
        )

    @cute.jit
    def _softmax_step(
        self, tiled_mma_qk, sm, pipeline_k, softmax, seqlen, score_mod_fn, nmax, n_block, mask_fn
    ):
        """Issue QK[n] half 0, signal P[n + 2], softmax of S[n + 1], issue QK[n] half 1,
        store P[n + 1], wait for QK[n], then fold and mask S[n]. P is signalled after a QK
        half is issued so that the PV warp groups do not delay it."""
        slot, phase = self._kslot(n_block, nmax)
        pipeline_k.consumer_wait_w_index_phase(slot, phase)
        acc0 = self._qk_half(tiled_mma_qk, sm, 0, slot)
        if n_block + 2 < nmax:
            self._signal_p(n_block + 2)
        row_scale = self._online_softmax_lazy_max(softmax, sm.S)
        acc1 = self._qk_half(tiled_mma_qk, sm, 1, slot)
        # Two P slots: P[n + 1] reuses the slot of P[n + 3].
        self._publish_p(sm, row_scale, n_block + 1, n_block + 1 < nmax - 2)
        warpgroup.wait_group(0)
        self._fold_s(sm, [acc0, acc1], pipeline_k, n_block, nmax, seqlen, mask_fn, score_mod_fn)

    # ///////////////////////////////////////////////////////////////////////////////////////
    # PV warp groups
    # ///////////////////////////////////////////////////////////////////////////////////////

    @cute.jit
    def pv_wg(
        self,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_o: cute.TiledMma,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sQ_hi: cute.Tensor,
        sQ_res: cute.Tensor,
        sV: cute.Tensor,
        sP: cute.Tensor,
        row_stats: tuple,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        tidx: Int32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        num_splits: Int32,
    ):
        """Quantize Q, then per KV block: wait for P, apply the row rescales, O^T += V^T P^T,
        release the P slot. Finally apply the row scales and write LSE and O."""
        sRowScale, sRowScaleChanged, sFinalScale, sLSE = row_stats
        pv = self._pv_setup(tiled_mma_pv, sV, sP, tidx)
        tiled_copy_Q = copy_utils.tiled_copy_2d(
            self.kv_dtype, self.tile_hdim // 8, self.num_pv_threads, num_copy_elems=8
        )
        thr_copy_Q = tiled_copy_Q.get_slice(tidx)
        tQsQ = thr_copy_Q.partition_S(sQ)
        tQsQ_hi = thr_copy_Q.partition_D(sQ_hi)
        tQsQ_res = thr_copy_Q.partition_D(sQ_res)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        if work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max, is_empty = self._block_range(
                block_info, seqlen, m_block, split_idx, num_splits
            )
            pipeline_q.consumer_wait_w_index_phase(0, Int32(0))
            self._quantize_q(tQsQ, tQsQ_hi, tQsQ_res)
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwdSm90Fp8KV.QReady),
                number_of_threads=self.num_handoff_threads,
            )
            for mt in cutlass.range_constexpr(self.num_pv_m_tiles):
                pv.acc[mt].fill(0.0)
            for i in cutlass.range(n_block_max - n_block_min, unroll=1):
                n_block = n_block_max - 1 - i
                slot = n_block % 2
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwdSm90Fp8KV.PFull0) + slot,
                    number_of_threads=self.num_handoff_threads,
                )
                # Skip the rescale when the softmax flagged that every row scale is 1.
                changed = Int32(0)
                for w in cutlass.range_constexpr(cute.size(sRowScaleChanged, mode=[0])):
                    changed = changed | sRowScaleChanged[w, slot]
                if changed != 0:
                    self._scale_cols(pv, sRowScale[None, slot])
                self._pv_block(pv, pipeline_v, n_block, n_block_max)
                warpgroup.wait_group(0)
                # This slot is next written with P[n_block - 2].
                if n_block > n_block_min + 1:
                    cute.arch.barrier_arrive(
                        barrier_id=int(NamedBarrierFwdSm90Fp8KV.PEmpty0) + slot,
                        number_of_threads=self.num_handoff_threads,
                    )
            pipeline_q.consumer_release_w_index(0)
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwdSm90Fp8KV.RowStats),
                number_of_threads=self.num_handoff_threads,
            )
            self._scale_cols(pv, sFinalScale)
            if is_empty:
                # Explicit zeros: 0 * acc is NaN if the dummy page holds NaN.
                for mt in cutlass.range_constexpr(self.num_pv_m_tiles):
                    pv.acc[mt].fill(0.0)
            thr_mma_o = tiled_mma_o.get_slice(tidx)
            tOsLSE = layout_utils.reshape_acc_to_mn(
                thr_mma_o.partition_C(
                    cute.make_tensor(
                        sLSE.iterator,
                        cute.make_layout((self.tile_m, self.tile_hdimv), stride=(1, 0)),
                    )
                )
            )
            lse = cute.make_rmem_tensor(cute.size(tOsLSE, mode=[0]), Float32)
            for r in cutlass.range_constexpr(cute.size(lse)):
                lse[r] = tOsLSE[r, 0]
            self.epilogue(
                None,
                lse,
                mO,
                mLSE,
                None,
                seqlen,
                None,
                None,
                tiled_mma_o,
                tidx,
                m_block,
                head_idx,
                batch_idx,
                split_idx,
                store_O_fn=partial(self._store_O, pv, m_block, seqlen.seqlen_q),
            )

    @cute.jit
    def _quantize_q(self, tQsQ, tQsQ_hi, tQsQ_res):
        """Split the BF16/FP16 Q tile into Q_hi and Q_res (utils.e4m3_hi_res_x8) in place:
        both alias the Q input, so they are written only after every PV thread has read Q."""
        rQ = cute.make_rmem_tensor_like(tQsQ)
        rQ_hi = cute.make_rmem_tensor_like(tQsQ_hi)
        rQ_res = cute.make_rmem_tensor_like(tQsQ_res)
        cute.autovec_copy(tQsQ, rQ)
        n = const_expr(cute.size(rQ))
        assert n % 8 == 0 and cute.size(rQ_hi) == n and cute.size(rQ_res) == n
        q32 = cute.make_tensor(cute.recast_tensor(rQ, Int32).iterator, cute.make_layout(n // 2))
        hi32 = cute.make_tensor(cute.recast_tensor(rQ_hi, Int32).iterator, cute.make_layout(n // 4))
        res32 = cute.make_tensor(
            cute.recast_tensor(rQ_res, Int32).iterator, cute.make_layout(n // 4)
        )
        for g in cutlass.range_constexpr(n // 8):
            words = utils.e4m3_hi_res_x8(
                q32[4 * g],
                q32[4 * g + 1],
                q32[4 * g + 2],
                q32[4 * g + 3],
                self.q_dtype == BFloat16,
            )
            hi32[2 * g] = words[0]
            hi32[2 * g + 1] = words[1]
            res32[2 * g] = words[2]
            res32[2 * g + 1] = words[3]
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.NarrowQ), number_of_threads=self.num_pv_threads
        )
        cute.autovec_copy(rQ_hi, tQsQ_hi)
        cute.autovec_copy(rQ_res, tQsQ_res)
        # Q_hi and Q_res are read by WGMMA (async proxy).
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.NarrowQ), number_of_threads=self.num_pv_threads
        )

    @cute.jit
    def _pv_setup(self, tiled_mma_pv, sV, sP, tidx):
        """Per-thread PV state: accumulators and A fragments of the warp group's 64-row head_dim
        tiles, the P operand, the ldmatrix view of the warp group's V^T half, and the
        coordinate of each accumulator element.

        The A fragment of tile mt is built from E4M3 V^T with ldmatrix.trans (16-bit), which
        gives each thread 2 x 2 (head_dim, key) byte blocks; e4m3_2x2_to_f16x2_transposed
        widens them to FP16. Accumulator row rho of a tile then holds head_dim
        pv_row_to_hdim(rho) of that tile: rows rho and rho + 8 are adjacent head_dim columns.
        """
        wg = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        wg_tidx = tidx % self.num_threads_per_warp_group
        thr_mma = tiled_mma_pv.get_slice(wg_tidx)
        acc = [
            cute.make_rmem_tensor(thr_mma.partition_shape_C((64, self.tile_m)), Float32)
            for _ in range(self.num_pv_m_tiles)
        ]
        frag = [
            thr_mma.make_fragment_A(thr_mma.partition_shape_A((64, self.tile_n)))
            for _ in range(self.num_pv_m_tiles)
        ]
        ldsm_copy = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4), self.kv_dtype),
            cute.make_layout(((8, 4), 4, 1), stride=((4, 32), 1, 0)),
            cute.make_layout((2, 2, 8), stride=(1, 2, 4)),
        )
        # (head_dim, key, slot) -> this warp group's head_dim half -> (hd 64, key 8, m tile,
        # key block, slot).
        sVt_wg = cute.local_tile(
            layout_utils.transpose_view(sV), (self.hdim_per_pv_wg, self.tile_n), (wg, 0, None)
        )
        sVt_tiles = cute.flat_divide(sVt_wg, (64, 8))
        # (row, col) = (accumulator row, query) coordinate of each accumulator element.
        tAccC = layout_utils.reshape_acc_to_mn(
            thr_mma.partition_C(cute.make_identity_tensor((64, self.tile_m)))
        )
        return SimpleNamespace(
            tiled_mma=tiled_mma_pv,
            thr_mma=thr_mma,
            acc=acc,
            frag=frag,
            tOrP=thr_mma.make_fragment_B(thr_mma.partition_B(sP)),
            ldsm_copy=ldsm_copy,
            ldsm_thr=ldsm_copy.get_slice(wg_tidx),
            sVt_tiles=sVt_tiles,
            tAccC=tAccC,
            hdim_offset=wg * self.hdim_per_pv_wg,
            # Accumulator row -> head_dim column within a 64-row tile.
            pv_row_to_hdim=cute.make_layout((8, 2, 4), stride=(2, 1, 16)),
        )

    @cute.jit
    def _load_vt(self, pv, v_slot, mt: cutlass.Constexpr[int]):
        """E4M3 bytes of the V^T A fragment of tile mt, as 32-bit words of 2 x 2 blocks."""
        tVsV = pv.ldsm_thr.partition_S(pv.sVt_tiles[None, None, mt, None, v_slot])
        tVrV = cute.make_rmem_tensor_like(tVsV)
        cute.copy(pv.ldsm_copy, tVsV, tVrV)
        return cute.recast_tensor(tVrV, Int32)

    @cute.jit
    def _pv_block(self, pv, pipeline_v, n_block, n_block_max):
        """O^T += V[n_block]^T P[n_block]^T over this warp group's tiles, keeping up to
        pv_wgmma_window WGMMA groups in flight while the next V^T fragment is converted."""
        v_slot, v_phase = self._kslot(n_block, n_block_max)
        p_slot = n_block % 2
        num_k = const_expr(self.tile_n // 16)
        pipeline_v.consumer_wait_w_index_phase(v_slot, v_phase)
        raw = self._load_vt(pv, v_slot, 0)
        # Each 32-bit word of 2 x 2 E4M3 values becomes two f16x2 A-fragment words; a k step
        # takes four A-fragment words.
        assert cute.size(raw) == 2 * num_k
        assert (
            cute.size(pv.frag[0], mode=[2]) == num_k and cute.size(pv.frag[0][None, None, 0]) == 8
        )
        mma_atom = cute.make_mma_atom(pv.tiled_mma.op)
        mma_atom.set(warpgroup.Field.ACCUMULATE, True)
        for mt in cutlass.range_constexpr(self.num_pv_m_tiles):
            frag32 = cute.recast_tensor(pv.frag[mt], Int32)
            frag32 = cute.make_tensor(frag32.iterator, cute.make_layout(cute.size(frag32)))
            raw_next = None
            for k in cutlass.range_constexpr(num_k):
                group = const_expr(mt * num_k + k)
                if const_expr(group >= self.pv_wgmma_window):
                    warpgroup.wait_group(self.pv_wgmma_window - 1)
                for kh in cutlass.range_constexpr(2):
                    cols = utils.e4m3_2x2_to_f16x2_transposed(raw[2 * k + kh])
                    frag32[4 * k + 2 * kh] = cols[0]
                    frag32[4 * k + 2 * kh + 1] = cols[1]
                if const_expr(k == num_k // 2 - 1 and mt + 1 < self.num_pv_m_tiles):
                    raw_next = self._load_vt(pv, v_slot, mt + 1)
                warpgroup.fence()
                cute.gemm(
                    mma_atom,
                    pv.acc[mt],
                    pv.frag[mt][None, None, k],
                    pv.tOrP[None, None, k, p_slot],
                    pv.acc[mt],
                )
                warpgroup.commit_group()
            raw = raw_next
        pipeline_v.consumer_release_w_index(v_slot)

    @cute.jit
    def _scale_cols(self, pv, scales):
        """O^T[:, q] *= scales[q] for this thread's query columns q."""
        num_cols = const_expr(cute.size(pv.tAccC, mode=[1]))
        col_scale = cute.make_rmem_tensor(num_cols, Float32)
        for k in cutlass.range_constexpr(num_cols):
            col_scale[k] = scales[pv.tAccC[0, k][1]]
        for mt in cutlass.range_constexpr(self.num_pv_m_tiles):
            acc_mn = layout_utils.reshape_acc_to_mn(pv.acc[mt])
            for k in cutlass.range_constexpr(num_cols):
                for i in cutlass.range_constexpr(cute.size(acc_mn, mode=[0])):
                    acc_mn[i, k] = acc_mn[i, k] * col_scale[k]

    @cute.jit
    def _store_O(self, pv, m_block, seqlen_q, mO_cur):
        """store_O_fn of the epilogue: write this warp group's O^T columns from registers,
        query rows below seqlen_q only. mO_cur is (seqlen_q, head_dim_v), or
        ((qhead_per_kvhead, seqlen_q), head_dim_v) with pack_gqa; FP32 with SplitKV."""
        dtype = mO_cur.element_type
        qh = const_expr(self.qhead_per_kvhead if self.pack_gqa else 1)
        # Rows i = 0, 1 of a thread's accumulator are accumulator rows rho and rho + 8, which
        # hold adjacent head_dim columns: store them as one pair.
        assert cute.size(pv.tAccC, mode=[0]) == 2
        hdim_pair = pv.hdim_offset + pv.pv_row_to_hdim(pv.tAccC[0, 0][0])
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dtype,
            num_bits_per_copy=2 * dtype.width,
        )
        if const_expr(dtype is not Float32):
            # Streaming (evict-first) stores keep the K/V pages in L2.
            store_atom = cute.make_copy_atom(
                cute.nvgpu.CopyR2GOp(),
                dtype,
                num_bits_per_copy=2 * dtype.width,
                store_cache_mode=cute.nvgpu.StoreCacheMode.STREAMING,
            )
        for k in cutlass.range_constexpr(cute.size(pv.tAccC, mode=[1])):
            row = m_block * self.tile_m + pv.tAccC[0, k][1]
            if row < seqlen_q * qh:
                if const_expr(self.pack_gqa):
                    m_idx = row // qh
                    mO_row = mO_cur[(row - m_idx * qh, m_idx), None]
                else:
                    mO_row = mO_cur[row, None]
                for mt in cutlass.range_constexpr(self.num_pv_m_tiles):
                    acc_mn = layout_utils.reshape_acc_to_mn(pv.acc[mt])
                    src = cute.make_rmem_tensor(2, Float32)
                    src[0] = acc_mn[0, k]
                    src[1] = acc_mn[1, k]
                    dst = cute.local_tile(mO_row, (2,), ((hdim_pair + 64 * mt) // 2,))
                    if const_expr(dtype is Float32):
                        cute.copy(store_atom, src, dst)
                    else:
                        cute.copy(store_atom, utils.cvt_f16(src, dtype), dst)
