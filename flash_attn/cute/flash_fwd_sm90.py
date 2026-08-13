# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM90 (Hopper) forward pass for flash attention, extracted from flash_fwd.py.

from types import SimpleNamespace
from typing import Callable, Literal, Optional
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
from flash_attn.cute import utils
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.softmax import Softmax, apply_score_mod_inner
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
from flash_attn.cute.named_barrier import NamedBarrierFwd
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    WorkTileInfo,
    SingleTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
    BatchOneDynamicSplitVarlenTileScheduler,
    StaticPersistentVarlenTileScheduler,
    DynamicPersistentVarlenTileScheduler,
)
from cutlass.cute import FastDivmodDivisor

from flash_attn.cute.flash_fwd import FlashAttentionForwardBase
from flash_attn.cute.utils import AuxData


def _use_paged_kv_overlap_sm90(intra_wg_overlap, paged_kv_non_tma, tile_n):
    return intra_wg_overlap and paged_kv_non_tma and tile_n in (80, 128, 240)


def _use_direct_single_split_sm90(
    use_persistent_varlen, use_dynamic_varlen, use_dynamic_splits
):
    # Runtime count one writes final output for both dynamic multi-request
    # scheduling and the persistent batch-one direct scheduler.
    return use_dynamic_splits and (use_persistent_varlen or use_dynamic_varlen)


class FlashAttentionForwardSm90(FlashAttentionForwardBase):
    def __init__(
        self,
        *args,
        intra_wg_overlap: bool = True,
        mma_pv_is_rs: bool = True,
        paged_kv_non_tma: bool = False,
        paged_kv_aligned_page_size: int = 0,
        is_split_kv: bool = False,
        use_persistent_varlen: bool = False,
        use_dynamic_varlen: bool = False,
        use_dynamic_splits: bool = False,
        persistent_scheduler_sm_count: Optional[int] = None,
        has_qv: bool = False,
        cp_world_size: int = 1,
        cp_rank: int = 0,
        kv_dtype=None,
        fp8_kv_dequant: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert self.output_quant_key is None, (
            f"Fused quant output not implemented for {type(self).__name__}"
        )
        self.has_qv = has_qv
        assert cp_world_size >= 1
        assert 0 <= cp_rank < cp_world_size
        assert cp_world_size == 1 or self.has_qv
        self.cp_world_size = cp_world_size
        self.cp_rank = cp_rank
        # TODO: Support MLA intra-WG overlap. QK and QvV share acc_S, and V
        # pipeline ownership must span both QvV and PV.
        self.intra_wg_overlap = intra_wg_overlap and not self.has_qv
        self.use_paged_kv_overlap = _use_paged_kv_overlap_sm90(
            self.intra_wg_overlap,
            paged_kv_non_tma,
            self.tile_n,
        )
        self.mma_pv_is_rs = mma_pv_is_rs
        self.buffer_align_bytes = 1024
        self.use_tma_KV = not paged_kv_non_tma
        self.paged_kv_aligned_page_size = paged_kv_aligned_page_size
        self.is_split_kv = is_split_kv
        self.use_persistent_varlen = use_persistent_varlen
        self.use_dynamic_varlen = use_dynamic_varlen
        self.use_dynamic_splits = use_dynamic_splits
        assert not self.use_dynamic_splits or self.is_split_kv
        assert not (self.use_persistent_varlen and self.use_dynamic_varlen)
        self.persistent_scheduler_sm_count = persistent_scheduler_sm_count
        assert (
            not (self.use_persistent_varlen or self.use_dynamic_varlen)
            or self.persistent_scheduler_sm_count is not None
        )
        # FP8-KV scale-fold path (SM90: fp16 Q + (fp8-e4m3 -> fp16) paged K/V -> fp16 O).
        self.fp8_kv_dequant = fp8_kv_dequant
        self.kv_dtype = kv_dtype if kv_dtype is not None else self.dtype
        assert not self.fp8_kv_dequant or not (
            self.has_qv
            or self.cp_world_size != 1
            or self.use_persistent_varlen
            or self.use_dynamic_varlen
            or self.use_dynamic_splits
        ), "fp8_kv_dequant does not support MLA, context parallelism, or dynamic scheduling"
        assert not self.fp8_kv_dequant or (self.use_tma_KV and self.intra_wg_overlap), (
            "fp8_kv_dequant requires the TMA-KV intra-warpgroup-overlap path"
        )
        assert self.use_tma_KV or not (self.check_hdim_oob or self.check_hdim_v_oob), (
            "Paged KV does not support irregular head dim"
        )
        self.cluster_shape_mn = (1, 1)
        assert self.arch.is_family_of(Arch.sm_90a), "Only SM 9.x is supported"

    @cute.jit
    def resolve_num_splits(
        self, split_idx: Int32, num_splits: Int32
    ) -> tuple[Int32, Int32]:
        if const_expr(self.use_dynamic_splits):
            num_splits = split_idx >> 16
            split_idx &= 0xFFFF
        return split_idx, num_splits

    @cute.jit
    def paged_tma_page_idx(
        self,
        mPageTable: cute.Tensor,
        batch_idx: Int32,
        n_block: Int32,
        blocks_per_page: Int32,
    ) -> Int32:
        n_block = cutlass.max(n_block, 0)
        return (
            mPageTable[batch_idx, n_block // blocks_per_page] * blocks_per_page
            + n_block % blocks_per_page
        )

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
        atom_layout_n = 2 if self.tile_hdim > 256 or self.tile_hdimv > 256 else 1
        score_atom_layout_n = 1 if self.use_asym_dv512 else atom_layout_n
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, score_atom_layout_n, 1),
            tiler_mn=(64, self.tile_n),
        )
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(
                self.tile_m // 64,
                atom_layout_n,
                1,
            ),  # Might need (1, 2, 1) for hdim 512
            tiler_mn=(64, min(256, self.tile_hdimv)),
            a_source=warpgroup.OperandSource.RMEM
            if self.mma_pv_is_rs
            else warpgroup.OperandSource.SMEM,
        )
        return tiled_mma_qk, tiled_mma_pv, tiled_mma_qk

    @cute.jit
    def _gemm_two_stage(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        B_idx: Int32,
        zero_init: cutlass.Constexpr[bool] = False,
        wg_wait: cutlass.Constexpr[int] = -1,
    ) -> None:
        """Issue WGMMA using the current two-stage pipeline slot."""
        assert self.num_stages == 2
        sm90_utils.gemm(
            tiled_mma,
            acc,
            tCrA,
            tCrB[None, None, None, B_idx],
            zero_init=zero_init,
            wg_wait=wg_wait,
        )

    def _get_shared_storage_cls(self):
        sQ_struct = cute.struct.Align[
            cute.struct.MemRange[self.q_dtype, cute.cosize(self.sQ_layout)],
            self.buffer_align_bytes,
        ]
        sK_struct, sV_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)], self.buffer_align_bytes
            ]
            for layout in (self.sK_layout, self.sV_layout)
        ]
        cosize_sQv = cute.cosize(self.sQv_layout) if const_expr(self.has_qv) else 0
        sQv_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cosize_sQv], self.buffer_align_bytes
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sQV], 1024]
        cosize_sP = cute.cosize(self.sP_layout) if const_expr(self.sP_layout is not None) else 0
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
        sScale_struct = (
            cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.sScale_layout)], 128
            ]
            if const_expr(self.use_asym_dv512)
            else cute.struct.MemRange[Float32, 0]
        )
        # 1 stage * 2 for Q pipeline (full + empty), self.num_stages*2 for K, self.num_stages*2 for V,
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        num_work_info_buffers = 2
        # m_block, head_idx, batch_idx, split_idx, is_valid_tile
        num_work_info_fields = 5
        work_info_struct = cute.struct.MemRange[
            cutlass.Int32,
            num_work_info_buffers * num_work_info_fields
            if const_expr(self.use_dynamic_varlen)
            else 0,
        ]

        if const_expr(self.fp8_kv_dequant):
            # sStage: the fp8 staging buffer; its num_stages=1 TMA pipeline gates GMEM->sStage.
            sStage_struct = cute.struct.Align[
                cute.struct.MemRange[self.kv_dtype, cute.cosize(self.sStage_layout)],
                self.buffer_align_bytes,
            ]
            mbar_ptr_Stage_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]

            @cute.struct
            class SharedStorageQKVDequant:
                mbar_ptr_Q: mbar_ptr_Q_struct
                mbar_ptr_K: mbar_ptr_K_struct
                mbar_ptr_V: mbar_ptr_V_struct
                mbar_ptr_Stage: mbar_ptr_Stage_struct
                work_info: work_info_struct
                sV: sV_struct
                sQ: sQ_struct
                sK: sK_struct
                sQv: sQv_struct
                sP: sP_struct
                sScale: sScale_struct
                sStage: sStage_struct

            return SharedStorageQKVDequant

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            work_info: work_info_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sQv: sQv_struct
            sP: sP_struct
            sScale: sScale_struct

        @cute.struct
        class SharedStorageSharedQV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            work_info: work_info_struct
            sQ: sQV_struct
            sK: sK_struct
            sQv: sQv_struct
            sP: sP_struct
            sScale: sScale_struct

        return SharedStorageQKV if const_expr(not self.Q_in_regs) else SharedStorageSharedQV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        mQv: Optional[cute.Tensor],  # (b, s_q, h, dv) or (total_q, h, dv)
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mCpTotSeqUsedK: Optional[cute.Tensor] = None,
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
        mWorkCounter: Optional[cute.Tensor] = None,
        num_splits_dynamic_ptr: Optional[cute.Tensor] = None,
        mOFinal: Optional[cute.Tensor] = None,
        mLSEFinal: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mQv/mK/mV/mO normally have the same fp16 or bf16 data type and
        layout. The fp8-KV path instead accepts fp16/bf16 Q and O with fp8
        e4m3 K/V, and does not support Qv/MLA.

        Tensor layout is (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1).
        """
        assert (mQv is not None) == self.has_qv
        if const_expr(mCpTotSeqUsedK is not None):
            assert mCpTotSeqUsedK.element_type == Int32
        if const_expr(self.cp_world_size > 1):
            assert mCpTotSeqUsedK is not None
        if const_expr(self.has_qv):
            assert mQv.element_type == self.dtype, "Qv must have the same dtype as Q"
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
            # fp16 OR bf16 -- compute is always fp16 (the single-instruction fp8->fp16 fast
            # path). Q is narrowed bf16->fp16 in-kernel and the fp32 accumulator is cast to
            # mO's dtype in the epilogue, so the Q/O tensor dtypes are independent of compute.
            assert self.dtype == Float16, "fp8_kv_dequant computes in fp16"
            assert mQ.element_type in (Float16, BFloat16), "fp8_kv_dequant expects fp16 or bf16 Q"
            if const_expr(self.is_split_kv):
                assert mO.element_type == Float32, (
                    "fp8_kv_dequant SplitKV expects Float32 O (partial accumulator)"
                )
            else:
                assert mO.element_type in (Float16, BFloat16), (
                    "fp8_kv_dequant expects fp16 or bf16 O"
                )

        # Q input / O output dtypes (may differ from the fp16 compute dtype on the fp8-KV
        # path). q_dtype gates the in-kernel bf16->fp16 Q narrow; o_dtype drives the
        # epilogue cast. SplitKV writes fp32 partials via acc_O directly, so o_dtype stays
        # the compute dtype there (the combine kernel produces the final bf16/fp16 O).
        self.q_dtype = mQ.element_type
        if const_expr(self.fp8_kv_dequant):
            self.o_dtype = mO.element_type if const_expr(not self.is_split_kv) else self.dtype

        direct_single_split = _use_direct_single_split_sm90(
            self.use_persistent_varlen,
            self.use_dynamic_varlen,
            self.use_dynamic_splits,
        )
        assert (mOFinal is not None) == direct_single_split
        assert not direct_single_split or mOFinal.element_type == self.dtype

        self.varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None

        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        mOFinal = (
            assume_tensor_aligned(mOFinal)
            if const_expr(direct_single_split)
            else None
        )
        mQv = assume_tensor_aligned(mQv) if const_expr(self.has_qv) else None
        Q_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ = layout_utils.select(mQ, Q_layout_transpose)
        if const_expr(self.has_qv):
            mQv = layout_utils.select(mQv, Q_layout_transpose)
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
        if const_expr(direct_single_split):
            final_O_layout_transpose = (
                [1, 3, 2, 0]
                if const_expr(mCuSeqlensQ is None)
                else [0, 2, 1]
            )
            final_LSE_layout_transpose = (
                [2, 1, 0]
                if const_expr(mCuSeqlensQ is None)
                else [1, 0]
            )
            mOFinal = layout_utils.select(mOFinal, final_O_layout_transpose)
            mLSEFinal = (
                layout_utils.select(mLSEFinal, final_LSE_layout_transpose)
                if const_expr(mLSEFinal is not None)
                else None
            )

        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
        self.use_asym_dv512 = cutlass.const_expr(
            self.has_qv
            and self.tile_hdimv == 512
            and self.tile_m == 64
            and self.tile_n == 64
            and not self.mma_pv_is_rs
            and not self.use_block_sparsity
        )
        tiled_mma_qk, tiled_mma_pv, tiled_mma_qv = self._get_tiled_mma()
        self.num_score_threads = tiled_mma_qk.size
        self.num_mma_threads = tiled_mma_pv.size
        self.num_threads_per_warp_group = 128
        self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
        assert self.num_wg_mma in [1, 2, 3]
        self.num_threads = self.num_threads_per_warp_group * (self.num_wg_mma + 1)
        self.num_producer_threads = 32
        self.num_Q_load_threads = self.num_threads_per_warp_group  # If not TMA_Q
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs, self.num_producer_regs = {1: (256, 56), 2: (240, 24), 3: (160, 32)}[
            self.num_wg_mma
        ]
        self.use_scheduler_barrier = (
            (self.num_wg_mma >= 2 and self.tile_hdim <= 128)
            if const_expr(self.intra_wg_overlap)
            else (self.num_wg_mma == 2)
        )
        if const_expr(self.has_qv):
            # MLA either gives each WG a complete score path or publishes P
            # explicitly. Neither follows the alternating QK/PV scheduler.
            self.use_scheduler_barrier = False
        self.use_tma_Q = self.arch >= Arch.sm_90 and not (
            self.pack_gqa and self.tile_m % self.qhead_per_kvhead != 0
        )
        self.use_tma_O = self.use_tma_Q and not self.is_split_kv
        # Producer needs more registers when doing cp.async Q or KV loads
        if const_expr(self.num_wg_mma == 2 and (not self.use_tma_Q or not self.use_tma_KV)):
            self.num_mma_regs, self.num_producer_regs = (
                (232, 40) if const_expr(self.has_qv) else (224, 40)
            )
        self.rescale_O_before_gemm = self.tile_hdimv > 128 and self.intra_wg_overlap
        self._setup_attributes()
        # TODO: we prob don't need most of what's in _setup_attributes
        # Per-tensor smem dtypes. FP8-KV is the one special case: sK/sV hold the fp16
        # dequant target (self.dtype), not mK/mV's fp8 source.
        sK_dtype = self.dtype if const_expr(self.fp8_kv_dequant) else mK.element_type
        sV_dtype = self.dtype if const_expr(self.fp8_kv_dequant) else mV.element_type
        sQv_dtype = mQv.element_type if const_expr(self.has_qv) else self.q_dtype
        sO_dtype = self.o_dtype if const_expr(self.fp8_kv_dequant) else sQv_dtype
        self.sQ_layout, self.sK_layout, self.sV_layout, self.sQv_layout, self.sO_layout = [
            sm90_utils.make_smem_layout(dtype, LayoutEnum.ROW_MAJOR, shape, stage)
            for dtype, shape, stage in [
                # sQ holds the (possibly bf16) Q tensor as TMA'd; the fp16 compute view
                # is an in-place recast (both are 2 bytes) -- see mma().
                (self.q_dtype, (self.tile_m, self.tile_hdim), None),
                (sK_dtype, (self.tile_n, self.tile_hdim), self.num_stages),
                (sV_dtype, (self.tile_n, self.tile_hdimv), self.num_stages),
                (sQv_dtype, (self.tile_m, self.tile_hdimv), None),
                # sO uses the output dtype (o_dtype); for splitkv o_dtype == compute dtype
                # (the fp32 partials are written from acc_O directly, not via sO).
                (sO_dtype, (self.tile_m, self.tile_hdimv), None),
            ]
        ]
        # fp8 staging tile: ONE shared buffer (tile_hdim == tile_hdimv so it serves K and V).
        self.sStage_layout = None
        if const_expr(self.fp8_kv_dequant):
            self.sStage_layout = sm90_utils.make_smem_layout(
                self.kv_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdim), 1
            )
        self.sP_layout = None
        if const_expr(not self.mma_pv_is_rs):
            self.sP_layout = sm90_utils.make_smem_layout(
                mV.element_type, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_n)
            )
        self.sScale_layout = cute.make_layout(
            (self.tile_m, self.num_stages), stride=(1, self.tile_m)
        )

        SharedStorage = self._get_shared_storage_cls()

        mQ_og, mQv_og, mO_og = mQ, mQv, mO
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(self.has_qv):
                mQv = pack_gqa_layout(mQv, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)
            if const_expr(direct_single_split):
                mOFinal = pack_gqa_layout(
                    mOFinal, self.qhead_per_kvhead, nheads_kv, head_idx=2
                )
                if const_expr(mLSEFinal is not None):
                    mLSEFinal = pack_gqa_layout(
                        mLSEFinal, self.qhead_per_kvhead, nheads_kv, head_idx=1
                    )

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
        if const_expr(self.has_qv):
            self.tma_copy_bytes["Qv"] = cute.size_in_bytes(
                mQv.element_type, cute.select(self.sQv_layout, mode=[0, 1])
            )
        make_tiled_tma_atom_fn = (
            partial(make_packgqa_tiled_tma_atom, qhead_per_kvhead=self.qhead_per_kvhead, head_idx=2)
            if const_expr(self.pack_gqa)
            else cpasync.make_tiled_tma_atom
        )
        tma_atom_Q, tma_tensor_Q = None, None
        tma_atom_Qv, tma_tensor_Qv = None, None
        if const_expr(self.use_tma_Q):
            tma_atom_Q, tma_tensor_Q = make_tiled_tma_atom_fn(
                gmem_tiled_copy_Q,
                mQ_og if const_expr(self.pack_gqa) else mQ,
                self.sQ_layout,
                (self.tile_m, self.tile_hdim),  # No mcast
            )
            if const_expr(self.has_qv):
                tma_atom_Qv, tma_tensor_Qv = make_tiled_tma_atom_fn(
                    gmem_tiled_copy_Q,
                    mQv_og if const_expr(self.pack_gqa) else mQv,
                    self.sQv_layout,
                    (self.tile_m, self.tile_hdimv),  # No mcast
                )
        tma_atom_K, tma_tensor_K = None, None
        tma_atom_V, tma_tensor_V = None, None
        if const_expr(self.use_tma_KV):
            # FP8-KV: K and V from GMEM land in sStage for the consumer to dequant(instead of sK/sV)
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
                mV,
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
            TileScheduler = (
                BatchOneDynamicSplitVarlenTileScheduler
                if const_expr(
                    self.use_persistent_varlen and self.use_dynamic_splits
                )
                else (
                    DynamicPersistentVarlenTileScheduler
                    if const_expr(self.use_dynamic_varlen)
                    else (
                        StaticPersistentVarlenTileScheduler
                        if const_expr(self.use_persistent_varlen)
                        else SingleTileVarlenScheduler
                    )
                )
            )
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
            num_splits_dynamic_ptr=num_splits_dynamic_ptr,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = (
            TileScheduler.get_grid_shape(
                tile_sched_params,
                sm_count=self.persistent_scheduler_sm_count,
            )
            if const_expr(
                self.use_persistent_varlen or self.use_dynamic_varlen
            )
            else TileScheduler.get_grid_shape(tile_sched_params)
        )
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
            (
                tma_tensor_Qv if const_expr(self.use_tma_Q) else mQv
            )
            if const_expr(self.has_qv)
            else None,
            tma_tensor_K if const_expr(self.use_tma_KV) else mK,
            tma_tensor_V if const_expr(self.use_tma_KV) else mV,
            tma_tensor_O if const_expr(self.use_tma_O) else mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mCpTotSeqUsedK,
            mDynamicCausal,
            mPageTable,
            tma_atom_Q,
            tma_atom_Qv,
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
            self.sQv_layout,
            self.sO_layout,
            self.sP_layout,
            self.sScale_layout,
            self.sStage_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tiled_mma_qv,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            num_splits,
            aux_data,
            fastdiv_mods,
            output_scale,
            descale_tensors,
            mWorkCounter,
            mOFinal,
            mLSEFinal,
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
        mQv: Optional[cute.Tensor],
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mCpTotSeqUsedK: Optional[cute.Tensor],
        mDynamicCausal: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_Qv: Optional[cute.CopyAtom],
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
        sQv_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        sScale_layout: cute.Layout,
        sStage_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_qv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        num_splits: Int32 = Int32(1),
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        output_scale: Optional[cute.Tensor] = None,
        descale_tensors=None,
        mWorkCounter: Optional[cute.Tensor] = None,
        mOFinal: Optional[cute.Tensor] = None,
        mLSEFinal: Optional[cute.Tensor] = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_Qv, tma_atom_K, tma_atom_V, tma_atom_O):
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
        score_warps = ThreadCooperativeGroup(self.num_score_threads // cute.arch.WARP_SIZE)
        if const_expr(self.use_tma_Q):
            pipeline_q = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["Q"]
                + (self.tma_copy_bytes["Qv"] if const_expr(self.has_qv) else 0),
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
            pipeline_k = None
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
                consumer_group=score_warps,
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
                consumer_group=score_warps,
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
        # sQ is the input-dtype view; FP8-KV may TMA a bf16 Q even though its
        # compute dtype is fp16.
        sQ = storage.sQ.get_tensor(
            sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.q_dtype
        )
        sQv = (
            storage.sQv.get_tensor(sQv_layout.outer, swizzle=sQv_layout.inner)
            if const_expr(self.has_qv)
            else sQ
        )
        # FP8-KV bf16 Q: fp16 compute view aliasing sQ's smem (both 2 bytes, so the
        # element layout/swizzle is identical -- reinterpret the same storage as fp16, the
        # same way sO reuses sQ's storage below). _narrow_q_to_compute() casts sQ bf16->fp16
        # in place per Q tile; the QK WGMMA reads sQ_mma. When Q is already fp16, sQ_mma is sQ.
        sQ_mma = sQ
        if const_expr(self.fp8_kv_dequant and self.q_dtype != self.dtype):
            sQ_mma = storage.sQ.get_tensor(
                sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.dtype
            )
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        else:
            sV = storage.sQ.get_tensor(
                sV_layout.outer, swizzle=sV_layout.inner, dtype=mV.element_type
            )
        # Transpose view of V to tensor with layout (head_dim_v, tile_n) for tiled mma
        sVt = layout_utils.transpose_view(sV)
        sP = None
        if const_expr(sP_layout is not None):
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        sScale = (
            storage.sScale.get_tensor(sScale_layout)
            if const_expr(self.use_asym_dv512)
            else None
        )
        # FP8 staging buffer (TMA destination, dequant source).
        sStage = (
            storage.sStage.get_tensor(sStage_layout.outer, swizzle=sStage_layout.inner)
            if const_expr(self.fp8_kv_dequant)
            else None
        )
        # With MLA, Qv is dead after the score mainloop, so the epilogue can
        # safely reuse its equally-shaped tile. The Q pipeline is released only
        # after the epilogue, preventing the producer from overwriting sO.
        sO_storage = storage.sQv if const_expr(self.has_qv) else storage.sQ
        sO_dtype = self.o_dtype if const_expr(self.fp8_kv_dequant) else self.dtype
        sO = sO_storage.get_tensor(
            sO_layout.outer, swizzle=sO_layout.inner, dtype=sO_dtype
        )
        sWorkInfo = (
            storage.work_info.get_tensor(
                cute.make_layout((5, 2), stride=(1, 5))
            )
            if const_expr(self.use_dynamic_varlen)
            else None
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
            mCpTotSeqUsedK=mCpTotSeqUsedK,
            cp_world_size=self.cp_world_size,
            cp_rank=self.cp_rank,
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
                mQv,
                mK,
                mV,
                sQ,
                sQv,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_Qv,
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
                sWorkInfo,
                mWorkCounter,
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
            if const_expr(self.use_asym_dv512):
                if tidx < self.num_score_threads:
                    self.mma(
                        tiled_mma_qk,
                        tiled_mma_pv,
                        tiled_mma_qv,
                        mO,
                        mLSE,
                        sQ,
                        sQv,
                        sK,
                        sV,
                        sVt,
                        sP,
                        sScale,
                        sO,
                        sQ_mma,
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
                        sWorkInfo,
                        blocksparse_tensors,
                        aux_data,
                        fastdiv_mods,
                        num_splits,
                        mOFinal,
                        mLSEFinal,
                        sStage,
                        pipeline_stage,
                        descale_tensors,
                    )
                else:
                    self.mma_pv_only(
                        tiled_mma_pv,
                        mO,
                        sVt,
                        sP,
                        sScale,
                        sO,
                        pipeline_v,
                        pipeline_q,
                        gmem_tiled_copy_O,
                        tma_atom_O,
                        tidx,
                        softmax_scale_log2,
                        softmax_scale,
                        block_info,
                        SeqlenInfoCls,
                        TileSchedulerCls,
                        sWorkInfo,
                        num_splits,
                        mOFinal,
                    )
            else:
                self.mma(
                    tiled_mma_qk,
                    tiled_mma_pv,
                    tiled_mma_qv,
                    mO,
                    mLSE,
                    sQ,
                    sQv,
                    sK,
                    sV,
                    sVt,
                    sP,
                    sScale,
                    sO,
                    sQ_mma,
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
                    sWorkInfo,
                    blocksparse_tensors,
                    aux_data,
                    fastdiv_mods,
                    num_splits,
                    mOFinal,
                    mLSEFinal,
                    sStage,
                    pipeline_stage,
                    descale_tensors,
                )

    @cute.jit
    def load_fp8(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sStage: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        pipeline_q: pipeline.PipelineAsync,
        pipeline_stage: pipeline.PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        mPageTable: Optional[cute.Tensor],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        num_splits: Int32 = Int32(1),
    ):
        """FP8-KV producer: warp 0 TMAs fp8 K/V tiles from GMEM -> the single staging buffer
        sStage, staggered to match the consumer's intra-WG overlap."""
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

            # Q stays fp16 (no dequant)
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

            stage_copy_K, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K, 0, cute.make_layout(1), gK, sStage
            )
            stage_copy_V, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V, 0, cute.make_layout(1), gV, sStage
            )

            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )

            # ---- Prologue emit: stage K[n_block_max - 1] (warp 0 only) ----
            if warp_idx_in_wg == 0:
                page_idx = mPageTable[batch_idx, n_block_max - 1]
                pipeline_stage.producer_acquire_w_index_phase(0, stage_producer_phase)
                stage_copy_K(src_idx=page_idx, dst_idx=0, tma_bar_ptr=stage_full_bar)
                stage_producer_phase ^= 1

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

            # ---- Mainloop + epilogue: stage K[n] then V[n_prev], then V[min]. All TMA
            # staging is warp-0 only; the other producer warps just advance the tile
            # scheduler in lockstep below. ----
            if warp_idx_in_wg == 0:
                for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                    n_block_prev = n_block_max - i - 1
                    n_block = n_block_prev - 1
                    page_idx = mPageTable[batch_idx, n_block]
                    page_idx_prev = mPageTable[batch_idx, n_block_prev]
                    # stage K[n]
                    pipeline_stage.producer_acquire_w_index_phase(0, stage_producer_phase)
                    stage_copy_K(src_idx=page_idx, dst_idx=0, tma_bar_ptr=stage_full_bar)
                    stage_producer_phase ^= 1
                    # stage V[n_prev]
                    pipeline_stage.producer_acquire_w_index_phase(0, stage_producer_phase)
                    stage_copy_V(src_idx=page_idx_prev, dst_idx=0, tma_bar_ptr=stage_full_bar)
                    stage_producer_phase ^= 1

                # ---- Epilogue emit: stage V[n_block_min] ----
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
        mQv: Optional[cute.Tensor],
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sQv: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_Qv: Optional[cute.CopyAtom],
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
        sWorkInfo: Optional[cute.Tensor],
        mWorkCounter: Optional[cute.Tensor],
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
                sStage,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_q,
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

        if is_load_warp or const_expr(self.use_dynamic_varlen):
            q_producer_phase = Int32(1)
            kv_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            tile_scheduler = TileSchedulerCls()
            work_info_phase = Int32(0)
            work_tile = (
                self.publish_dynamic_work(
                    tile_scheduler,
                    mWorkCounter,
                    sWorkInfo,
                    work_info_phase,
                    warp_idx_in_wg,
                )
                if const_expr(self.use_dynamic_varlen)
                else tile_scheduler.initial_work_tile_info()
            )
            if const_expr(self.use_dynamic_varlen):
                work_info_phase ^= 1
            while work_tile.is_valid_tile:
                # if work_tile.is_valid_tile:
                m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
                split_idx, num_splits_cur = self.resolve_num_splits(
                    split_idx, num_splits
                )
                seqlen = SeqlenInfoCls(batch_idx)
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                mQv_cur = (
                    seqlen.offset_batch_Q(mQv, batch_idx, dim=3)[None, None, head_idx]
                    if const_expr(self.has_qv)
                    else None
                )
                head_idx_kv = (
                    head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
                )

                load_Q = None
                load_Qv = None
                if const_expr(self.use_tma_Q):
                    gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
                    load_Q, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
                    )
                    if const_expr(self.has_qv):
                        gQv = cute.local_tile(
                            mQv_cur, (self.tile_m, self.tile_hdimv), (m_block, 0)
                        )
                        load_Qv, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_Qv,
                            0,
                            cute.make_layout(1),
                            gQv,
                            sQv,
                            single_stage=True,
                        )

                paged_kv_manager = None
                tma_load_K_fn = None
                tma_load_V_fn = None
                if const_expr(self.use_tma_KV):
                    # === TMA path (non-paged and paged with page_size divisible by tile_n) ===
                    if const_expr(mPageTable is not None):
                        # Keep Rest[0] (tile within page) and Rest[2] (physical page).
                        mK_cur = mK[None, None, head_idx_kv, None]
                        mV_cur = mV[None, None, head_idx_kv, None]
                        gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0, None))
                        gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (None, 0, None))
                        # Flatten both retained Rest modes into the single residual
                        # coordinate expected by the TMA copy.
                        gK = cute.group_modes(gK, 2, 4)
                        gV = cute.group_modes(gV, 2, 4)
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
                        tma_atom_V, 0, cute.make_layout(1), gV, sV
                    )
                    tma_load_V_fn = copy_utils.tma_producer_copy_fn(tma_load_V_fn, pipeline_v)
                else:
                    # === cp_async path (paged KV with page_size != n_block_size) ===
                    paged_kv_manager = PagedKVManager.create(
                        mPageTable,
                        mK,
                        mV,
                        FastDivmodDivisor(mK.shape[0]),
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
                        cache_v_ptr=self.use_paged_kv_overlap,
                        aligned_page_size=self.paged_kv_aligned_page_size,
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
                pack_gqa_qv = None
                if const_expr(not self.use_tma_Q):
                    pack_gqa = PackGQA(
                        self.tile_m, self.tile_hdim, self.check_hdim_oob, self.qhead_per_kvhead
                    )
                    if const_expr(self.has_qv):
                        pack_gqa_qv = PackGQA(
                            self.tile_m,
                            self.tile_hdimv,
                            self.check_hdim_v_oob,
                            self.qhead_per_kvhead,
                        )

                if const_expr(not self.use_block_sparsity):
                    n_block_min, n_block_max = block_info.get_n_block_min_max(
                        seqlen, m_block, split_idx, num_splits_cur
                    )
                    if const_expr(self._mDynamicCausal is not None):
                        psc_producer = self._mDynamicCausal[batch_idx]
                        if not psc_producer:
                            # Mirror the consumer's bidirectional split range so the
                            # producer loads exactly the K/V blocks the consumer
                            # processes. Any divergence here deadlocks the pipeline.
                            n_block_max_full = cute.ceil_div(seqlen.seqlen_k, self.tile_n)
                            if const_expr(self.is_split_kv):
                                num_n_blocks_per_split = cute.ceil_div(
                                    n_block_max_full, num_splits_cur
                                )
                                n_block_min = split_idx * num_n_blocks_per_split
                                n_block_max = cutlass.min(
                                    n_block_min + num_n_blocks_per_split, n_block_max_full
                                )
                            else:
                                n_block_min = Int32(0)
                                n_block_max = n_block_max_full
                    # Keep the dummy pipeline transaction when the range is empty.
                    # TMA handles block=-1 for dense KV, while paged TMA needs a
                    # valid page-table lookup; the consumer fully masks page 0.
                    n_block = (
                        n_block_max - 1
                        if const_expr(self.use_tma_KV)
                        else cutlass.max(n_block_max - 1, 0)
                    )
                    paged_tma_blocks_per_page = (
                        mK.shape[0] // self.tile_n
                        if const_expr(mPageTable is not None and self.use_tma_KV)
                        else 1
                    )
                    page_idx = (
                        self.paged_tma_page_idx(
                            mPageTable,
                            batch_idx,
                            n_block,
                            paged_tma_blocks_per_page,
                        )
                        if const_expr(mPageTable is not None and self.use_tma_KV)
                        else None
                    )

                    # First iteration: load K on pipeline_k, Q on pipeline_q
                    if is_kv_load_warp:
                        pipeline_k.producer_acquire(kv_producer_state)
                        if const_expr(not self.use_tma_KV):
                            paged_kv_manager.load_page_table(n_block, mask_seqlen=True)
                            if const_expr(self.use_paged_kv_overlap):
                                paged_kv_manager.update_V_ptr()
                        load_K(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)
                    if const_expr(self.use_tma_Q):
                        if warp_idx_in_wg == 0:
                            pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                            q_barrier = pipeline_q.sync_object_full.get_barrier(0)
                            load_Q(tma_bar_ptr=q_barrier)
                            if const_expr(self.has_qv):
                                load_Qv(tma_bar_ptr=q_barrier)
                            q_producer_phase ^= 1
                    else:
                        pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                        pack_gqa.load_Q(
                            mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q
                        )
                        if const_expr(self.has_qv):
                            pack_gqa_qv.load_Q(
                                mQv_cur,
                                sQv,
                                gmem_tiled_copy_Q,
                                tidx,
                                m_block,
                                seqlen.seqlen_q,
                            )
                        cute.arch.cp_async_commit_group()
                        pipeline_q.producer_commit_w_index(0)
                        q_producer_phase ^= 1

                    if is_kv_load_warp:
                        if const_expr(
                            not self.intra_wg_overlap
                            or (not self.use_tma_KV and not self.use_paged_kv_overlap)
                        ):
                            pipeline_v.producer_acquire(kv_producer_state)
                            load_V(
                                block=n_block, producer_state=kv_producer_state, page_idx=page_idx
                            )
                            kv_producer_state.advance()
                            for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                                n_block = n_block_max - 1 - i - 1
                                page_idx = (
                                    self.paged_tma_page_idx(
                                        mPageTable,
                                        batch_idx,
                                        n_block,
                                        paged_tma_blocks_per_page,
                                    )
                                    if const_expr(mPageTable is not None and self.use_tma_KV)
                                    else None
                                )
                                if const_expr(not self.use_tma_KV):
                                    paged_kv_manager.load_page_table(
                                        n_block,
                                        mask_seqlen=False,
                                    )
                                pipeline_k.producer_acquire(kv_producer_state)
                                load_K(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                    mask_seqlen=False,
                                )
                                pipeline_v.producer_acquire(kv_producer_state)
                                load_V(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                    mask_seqlen=False,
                                )
                                kv_producer_state.advance()
                        else:
                            for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                                n_block_prev = n_block_max - i - 1
                                n_block = n_block_prev - 1
                                page_idx = (
                                    self.paged_tma_page_idx(
                                        mPageTable,
                                        batch_idx,
                                        n_block,
                                        paged_tma_blocks_per_page,
                                    )
                                    if const_expr(mPageTable is not None and self.use_tma_KV)
                                    else None
                                )
                                page_idx_prev = (
                                    self.paged_tma_page_idx(
                                        mPageTable,
                                        batch_idx,
                                        n_block_prev,
                                        paged_tma_blocks_per_page,
                                    )
                                    if const_expr(mPageTable is not None and self.use_tma_KV)
                                    else None
                                )
                                kv_producer_state_prev = kv_producer_state.clone()
                                kv_producer_state.advance()
                                if const_expr(not self.use_tma_KV):
                                    paged_kv_manager.load_page_table(
                                        n_block,
                                        mask_seqlen=False,
                                    )
                                pipeline_k.producer_acquire(kv_producer_state)
                                load_K(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                    mask_seqlen=False,
                                )
                                pipeline_v.producer_acquire(kv_producer_state_prev)
                                if const_expr(self.use_tma_KV):
                                    load_V(
                                        block=n_block_prev,
                                        producer_state=kv_producer_state_prev,
                                        page_idx=page_idx_prev,
                                    )
                                else:
                                    self.load_paged_V_cached(
                                        paged_kv_manager,
                                        sV,
                                        n_block_prev,
                                        pipeline_v,
                                        kv_producer_state_prev,
                                    )
                            n_block = n_block_min
                            page_idx = (
                                self.paged_tma_page_idx(
                                    mPageTable,
                                    batch_idx,
                                    n_block,
                                    paged_tma_blocks_per_page,
                                )
                                if const_expr(mPageTable is not None and self.use_tma_KV)
                                else None
                            )
                            pipeline_v.producer_acquire(kv_producer_state)
                            if const_expr(self.use_tma_KV):
                                load_V(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )
                            else:
                                self.load_paged_V_cached(
                                    paged_kv_manager,
                                    sV,
                                    n_block,
                                    pipeline_v,
                                    kv_producer_state,
                                    update_cache=False,
                                )
                            kv_producer_state.advance()
                else:
                    # Block sparsity: use TMA closures directly (not paged)
                    # Load Q on pipeline_q, separate from K/V pipeline
                    if const_expr(self.use_tma_Q):
                        if warp_idx_in_wg == 0:
                            pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                            q_barrier = pipeline_q.sync_object_full.get_barrier(0)
                            load_Q(tma_bar_ptr=q_barrier)
                            if const_expr(self.has_qv):
                                load_Qv(tma_bar_ptr=q_barrier)
                            q_producer_phase ^= 1
                    else:
                        pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                        pack_gqa.load_Q(
                            mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q
                        )
                        if const_expr(self.has_qv):
                            pack_gqa_qv.load_Q(
                                mQv_cur,
                                sQv,
                                gmem_tiled_copy_Q,
                                tidx,
                                m_block,
                                seqlen.seqlen_q,
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

                if const_expr(self.use_dynamic_varlen):
                    work_tile = self.publish_dynamic_work(
                        tile_scheduler,
                        mWorkCounter,
                        sWorkInfo,
                        work_info_phase,
                        warp_idx_in_wg,
                    )
                    work_info_phase ^= 1
                else:
                    tile_scheduler.prefetch_next_work()
                    work_tile = tile_scheduler.advance_to_next_work()
                # End of persistent scheduler loop

            # Producer tail is only useful for cluster to avoid early exit of blocks.
            # We only need producer_tail on V since that's the last that's loaded, we don't
            # need it for Q (no cluster) and K.
            if is_kv_load_warp:
                pipeline_v.producer_tail(kv_producer_state)

    @cute.jit
    def read_dynamic_work(
        self, sWorkInfo: cute.Tensor, work_info_phase: Int32
    ) -> WorkTileInfo:
        return WorkTileInfo(
            (
                Int32(sWorkInfo[0, work_info_phase]),
                Int32(sWorkInfo[1, work_info_phase]),
                Int32(sWorkInfo[2, work_info_phase]),
                Int32(sWorkInfo[3, work_info_phase]),
            ),
            sWorkInfo[4, work_info_phase] != 0,
        )

    @cute.jit
    def publish_dynamic_work(
        self,
        tile_scheduler: DynamicPersistentVarlenTileScheduler,
        mWorkCounter: cute.Tensor,
        sWorkInfo: Optional[cute.Tensor],
        work_info_phase: Int32,
        warp_idx_in_wg: Int32,
    ) -> WorkTileInfo:
        if warp_idx_in_wg == 0:
            claimed_work = tile_scheduler.claim_next_work(mWorkCounter)
            if cute.arch.lane_idx() == 0:
                m_block, head_idx, batch_idx, split_idx = claimed_work.tile_idx
                sWorkInfo[0, work_info_phase] = m_block
                sWorkInfo[1, work_info_phase] = head_idx
                sWorkInfo[2, work_info_phase] = batch_idx
                sWorkInfo[3, work_info_phase] = split_idx
                sWorkInfo[4, work_info_phase] = (
                    Int32(1) if claimed_work.is_valid_tile else Int32(0)
                )
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.DynamicWork),
            number_of_threads=self.num_threads,
        )
        return self.read_dynamic_work(sWorkInfo, work_info_phase)

    @cute.jit
    def wait_dynamic_work(
        self, sWorkInfo: cute.Tensor, work_info_phase: Int32
    ) -> WorkTileInfo:
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.DynamicWork),
            number_of_threads=self.num_threads,
        )
        return self.read_dynamic_work(sWorkInfo, work_info_phase)

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
        mask_seqlen: cutlass.Constexpr[bool] = True,
    ):
        if const_expr(self.use_tma_KV):
            src_idx = block if const_expr(page_idx is None) else page_idx
            tma_load_fn(src_idx=src_idx, producer_state=producer_state)
        else:
            paged_kv_manager.load_KV(
                block,
                sX[None, None, producer_state.index],
                K_or_V,
                mask_seqlen=mask_seqlen,
            )
            cute.arch.cp_async_commit_group()
        pipeline_kv.producer_commit(producer_state)

    @cute.jit
    def load_paged_V_cached(
        self,
        paged_kv_manager: PagedKVManager,
        sV: cute.Tensor,
        block: Int32,
        pipeline_v: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        update_cache: cutlass.Constexpr[bool] = True,
    ):
        paged_kv_manager.load_V_cached(
            block,
            sV[None, None, producer_state.index],
            update_cache=update_cache,
        )
        cute.arch.cp_async_commit_group()
        pipeline_v.producer_commit(producer_state)

    @cute.jit
    def store_asym_scale(
        self,
        sScale: cute.Tensor,
        taccOcO_row: cute.Tensor,
        row_scale: cute.Tensor,
        stage: Int32,
    ):
        assert cute.size(row_scale) == cute.size(taccOcO_row)
        if taccOcO_row[0][1] == 0:
            for r in cutlass.range_constexpr(cute.size(row_scale)):
                sScale[taccOcO_row[r][0], stage] = row_scale[r]

    @cute.jit
    def load_asym_scale(
        self,
        sScale: cute.Tensor,
        taccOcO_row: cute.Tensor,
        row_scale: cute.Tensor,
        stage: Int32,
    ):
        assert cute.size(row_scale) == cute.size(taccOcO_row)
        for r in cutlass.range_constexpr(cute.size(row_scale)):
            row_scale[r] = sScale[taccOcO_row[r][0], stage]

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_qv: cute.TiledMma,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sQv: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sVt: cute.Tensor,
        sP: Optional[cute.Tensor],
        sScale: Optional[cute.Tensor],
        sO: cute.Tensor,
        # FP8-KV bf16 Q: fp16 compute view aliasing sQ (built in kernel() from the
        # region-local sQ_layout). Equals sQ when Q is already fp16.
        sQ_mma: cute.Tensor,
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
        sWorkInfo: cute.Tensor,
        blocksparse_tensors: Optional[BlockSparseTensors],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        num_splits: Int32 = Int32(1),
        mOFinal: Optional[cute.Tensor] = None,
        mLSEFinal: Optional[cute.Tensor] = None,
        # FP8-KV cast inputs (None on the non-fp8 path).
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
        # FP8-KV bf16 Q: the QK WGMMA needs fp16 operands. sQ holds the bf16 Q as TMA'd;
        # sQ_mma is the fp16 compute view (same smem) built in kernel(). _narrow_q_to_compute()
        # casts sQ bf16->fp16 in place per tile before the QK reads sQ_mma.
        narrow_q = const_expr(self.fp8_kv_dequant and self.q_dtype != self.dtype)
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ_mma, sK
        )
        mma_qk_fn = partial(
            sm90_utils.gemm_zero_init, tiled_mma_qk, (self.tile_m, self.tile_n), tSrQ, tSrK
        )
        tSrQv, tSrV = None, None
        if const_expr(self.has_qv):
            wg_mma_qv = tiled_mma_qv.get_slice(warp_group_thread_layout(warp_group_idx))
            _, tSrQv, tSrV = sm90_utils.partition_fragment_ABC(
                wg_mma_qv, (self.tile_m, self.tile_n, self.tile_hdimv), sQv, sV
            )
        acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
            wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), sP, sVt
        )
        mma_pv_fn = (
            partial(
                self._gemm_two_stage,
                tiled_mma_pv,
                acc_O,
                tOrP,
                tOrVt,
            )
            if const_expr(self.has_qv)
            else partial(
                sm90_utils.gemm_w_idx,
                tiled_mma_pv,
                acc_O,
                tOrP,
                tOrVt,
            )
        )
        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        taccOcO_row = layout_utils.reshape_acc_to_mn(
            tiled_mma_pv.get_slice(tidx).partition_C(cO)
        )[None, 0]

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom_P = utils.get_smem_store_atom(
            self.arch.major * 10 + self.arch.minor, self.dtype
        )
        smem_thr_copy_P = cute.make_tiled_copy_C(smem_copy_atom_P, tiled_mma_qk).get_slice(tidx)
        tPsP = smem_thr_copy_P.partition_D(sP) if const_expr(sP is not None) else None
        smem_copy_params = SimpleNamespace(smem_thr_copy_P=smem_thr_copy_P, tPsP=tPsP)

        self.mma_init()

        q_consumer_phase = Int32(0)
        kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )

        # FP8-KV MMA WGs side dequant setup:
        # K: cooperative 256-thread(2 WGs) copy of the full tile;
        # V: each WG copies only its own hdimv-half.
        dequant_params = None
        if const_expr(self.fp8_kv_dequant):
            sStage2 = sStage[None, None, 0]
            sK2 = sK[None, None, 0]
            sV2 = sV[None, None, 0]
            half = const_expr(min(256, self.tile_hdimv))
            VEC = const_expr(8)  # 8 fp16 == 128-bit store; 8 fp8 == 64-bit load
            tiled_copy_K = copy_utils.tiled_copy_2d(
                self.dtype, self.tile_hdim // VEC, self.num_mma_threads, num_copy_elems=VEC
            )
            tiled_copy_V = copy_utils.tiled_copy_2d(
                self.dtype, half // VEC, self.num_threads_per_warp_group, num_copy_elems=VEC
            )
            thr_copy_K = tiled_copy_K.get_slice(tidx)
            thr_copy_V = tiled_copy_V.get_slice(tidx % self.num_threads_per_warp_group)
            # WG-local hdimv-half sub-views (tile coord (0, warp_group_idx)).
            gStage_V = cute.local_tile(sStage2, (self.tile_n, half), (0, warp_group_idx))
            gV = cute.local_tile(sV2, (self.tile_n, half), (0, warp_group_idx))
            dequant_params = SimpleNamespace(
                pipeline_stage=pipeline_stage,
                warp_group_idx=warp_group_idx,
                tStage_K=thr_copy_K.partition_S(sStage2),
                tsK=thr_copy_K.partition_D(sK2),
                tStage_V=thr_copy_V.partition_S(gStage_V),
                tsV=thr_copy_V.partition_D(gV),
            )

        # FP8-KV bf16 Q in-place narrow partitions (256-thread cooperative copy of the
        # full Q tile; src is the bf16 sQ, dst the fp16 compute view over the same smem).
        q_narrow_params = None
        if const_expr(narrow_q):
            VEC_Q = const_expr(8)
            tiled_copy_Q = copy_utils.tiled_copy_2d(
                self.dtype, self.tile_hdim // VEC_Q, self.num_mma_threads, num_copy_elems=VEC_Q
            )
            thr_copy_Q = tiled_copy_Q.get_slice(tidx)
            q_narrow_params = SimpleNamespace(
                tQsrc=thr_copy_Q.partition_S(sQ),
                tQdst=thr_copy_Q.partition_D(sQ_mma),
            )

        tile_scheduler = TileSchedulerCls()
        work_info_phase = Int32(0)
        work_tile = (
            self.wait_dynamic_work(sWorkInfo, work_info_phase)
            if const_expr(self.use_dynamic_varlen)
            else tile_scheduler.initial_work_tile_info()
        )
        if const_expr(self.use_dynamic_varlen):
            work_info_phase ^= 1
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

        # Keep the FP8 and Dense/MLA partials in distinct constexpr branches:
        # their call signatures intentionally differ.
        if const_expr(self.fp8_kv_dequant):
            mma_one_n_block_all = partial(
                self.mma_one_n_block_intrawg_overlap_fp8,
                mma_qk_fn=mma_qk_fn,
                pipeline_k=pipeline_k,
                pipeline_v=pipeline_v,
                acc_O=acc_O,
                tOrP=tOrP,
                smem_copy_params=smem_copy_params,
                check_inf=True,
                scores_scale=scores_scale,
                dequant_params=dequant_params,
            )
            process_first_half_block = partial(
                self.first_half_block_overlap_fp8,
                mma_qk_fn=mma_qk_fn,
                pipeline_k=pipeline_k,
                tOrP=tOrP,
                smem_copy_params=smem_copy_params,
                scores_scale=scores_scale,
                acc_O=acc_O,
                dequant_params=dequant_params,
            )
            process_last_half_block = partial(
                self.last_half_block_overlap_fp8,
                pipeline_v=pipeline_v,
                mma_pv_fn=mma_pv_fn,
                scores_scale=scores_scale,
                acc_O=acc_O,
                dequant_params=dequant_params,
            )
        else:
            mma_one_n_block_all = partial(
                self.mma_one_n_block_intrawg_overlap
                if const_expr(self.intra_wg_overlap)
                else self.mma_one_n_block,
                mma_qk_fn=mma_qk_fn,
                tiled_mma_qv=tiled_mma_qv,
                tSrQv=tSrQv,
                tSrV=tSrV,
                pipeline_k=pipeline_k,
                pipeline_v=pipeline_v,
                acc_O=acc_O,
                tOrP=tOrP,
                smem_copy_params=smem_copy_params,
                sScale=sScale,
                taccOcO_row=taccOcO_row,
                check_inf=True,
                scores_scale=scores_scale,
            )
            process_first_half_block = partial(
                self.first_half_block_overlap,
                mma_qk_fn=mma_qk_fn,
                pipeline_k=pipeline_k,
                tOrP=tOrP,
                smem_copy_params=smem_copy_params,
                scores_scale=scores_scale,
                acc_O=acc_O,
            )
            process_last_half_block = partial(
                self.last_half_block_overlap,
                pipeline_v=pipeline_v,
                mma_pv_fn=mma_pv_fn,
                scores_scale=scores_scale,
                acc_O=acc_O,
            )
        while work_tile.is_valid_tile:
            # if work_tile.is_valid_tile:

            # shape: (atom_v_m * rest_m)
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            split_idx, num_splits_cur = self.resolve_num_splits(
                split_idx, num_splits
            )
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
                    else FastDivmodDivisor(seqlen.seqlen_q),
                    seqlen_k_divmod
                    if not recompute_fastdiv_mods_k
                    else FastDivmodDivisor(seqlen.seqlen_k),
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
                seqlen, m_block, split_idx, num_splits_cur
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
                        num_n_blocks_per_split = cute.ceil_div(
                            n_block_max_full, num_splits_cur
                        )
                        n_block_min = split_idx * num_n_blocks_per_split
                        n_block_max = cutlass.min(
                            n_block_min + num_n_blocks_per_split, n_block_max_full
                        )
                    else:
                        n_block_min = Int32(0)
                        n_block_max = n_block_max_full
            n_block_max_orig = n_block_max
            pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)
            # FP8-KV bf16 Q: cast the just-arrived bf16 Q tile -> fp16 in place (with a
            # 256-thread barrier) before any QK WGMMA reads it. No-op when Q is already fp16.
            if const_expr(narrow_q):
                self._narrow_q_to_compute(q_narrow_params)
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
                # The remaining iterations have no masking
                n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                    seqlen, m_block, n_block_min
                )
                interior_mask_fn = (
                    partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=False)
                    if const_expr(self.mask_mod is not None)
                    else None
                )
                # if cute.arch.thread_idx()[0] == 128: cute.printf("n_block_min_before_local_mask = {}, n_block_min = {}", n_block_min_before_local_mask, n_block_min)
                for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=n_block_max - 1 - n_tile,
                        seqlen=seqlen,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=interior_mask_fn,
                        check_inf=(
                            self.mask_mod is not None or self.score_mod is not None
                        ),
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
                # Handle empty case (when no blocks to process)
                if not processed_any:
                    softmax_tile.reset()
                    acc_O.fill(0.0)

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
            row_scale = softmax_tile.finalize(
                final_scale=v_descale_tile, sink_val=sink_val
            )
            if const_expr(self.use_asym_dv512):
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.PEmpty),
                    number_of_threads=self.num_mma_threads,
                )
                self.store_asym_scale(
                    sScale, taccOcO_row, row_scale, kv_consumer_state.index
                )
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierFwd.PFull),
                    number_of_threads=self.num_mma_threads,
                )
            softmax_tile.rescale_O(acc_O, row_scale)

            # Override empty splits so combine kernel gives zero weight
            if const_expr(self.is_split_kv):
                if n_block_min >= n_block_max_orig:
                    acc_O.fill(Float32(0.0))
                    softmax_tile.row_sum.fill(-Float32.inf)

            # ///////////////////////////////////////////////////////////////////////////////
            # Epilogue
            # ///////////////////////////////////////////////////////////////////////////////
            self.epilogue_split_or_final(
                acc_O,
                softmax_tile.row_sum,
                mO,
                mLSE,
                mOFinal,
                mLSEFinal,
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
                num_splits_cur,
            )

            # sO aliases the query tile (Qv for MLA, Q otherwise), so release
            # the Q pipeline only after the epilogue is finished with it.
            pipeline_q.consumer_release_w_index(0)
            q_consumer_phase ^= 1

            if const_expr(self.use_dynamic_varlen):
                work_tile = self.wait_dynamic_work(sWorkInfo, work_info_phase)
                work_info_phase ^= 1
            else:
                work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def epilogue_split_or_final(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mOFinal: Optional[cute.Tensor],
        mLSEFinal: Optional[cute.Tensor],
        sO: cute.Tensor,
        seqlen: SeqlenInfoQK,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        split_idx: Int32,
        num_splits: Int32,
    ):
        if const_expr(
            _use_direct_single_split_sm90(
                self.use_persistent_varlen,
                self.use_dynamic_varlen,
                self.use_dynamic_splits,
            )
        ):
            if num_splits == 1:
                self.epilogue_single_split(
                    acc_O,
                    lse,
                    mOFinal,
                    mLSEFinal,
                    sO,
                    seqlen,
                    gmem_tiled_copy_O,
                    tiled_mma,
                    tidx,
                    m_block,
                    head_idx,
                    batch_idx,
                )
            else:
                self.epilogue(
                    acc_O,
                    lse,
                    mO,
                    mLSE,
                    sO,
                    seqlen,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    tiled_mma,
                    tidx,
                    m_block,
                    head_idx,
                    batch_idx,
                    split_idx,
                )
        else:
            self.epilogue(
                acc_O,
                lse,
                mO,
                mLSE,
                sO,
                seqlen,
                gmem_tiled_copy_O,
                tma_atom_O,
                tiled_mma,
                tidx,
                m_block,
                head_idx,
                batch_idx,
                split_idx,
            )

    @cute.jit
    def epilogue_single_split(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sO: cute.Tensor,
        seqlen: SeqlenInfoQK,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        """Store a dynamic one-split tile directly to the final SM90 output."""
        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        pack_gqa = PackGQA(
            self.tile_m,
            self.tile_hdimv,
            self.check_hdim_v_oob,
            self.qhead_per_kvhead,
        )
        rO = cute.make_fragment_like(acc_O, self.dtype)
        rO.store(acc_O.load().to(self.dtype))
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_epilogue_threads,
        )
        smem_copy_atom_O = utils.get_smem_store_atom(
            self.arch.major * 10 + self.arch.minor, self.dtype
        )
        smem_thr_copy_O = cute.make_tiled_copy_C(
            smem_copy_atom_O, tiled_mma
        ).get_slice(tidx)
        cute.copy(
            smem_copy_atom_O,
            smem_thr_copy_O.retile(rO),
            smem_thr_copy_O.partition_D(sO),
        )

        if const_expr(mLSE is not None):
            mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[
                None, head_idx
            ]
            if const_expr(not self.pack_gqa):
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
                gLSE_expanded = cute.make_tensor(
                    gLSE.iterator,
                    cute.append(
                        gLSE.layout,
                        cute.make_layout((self.tile_hdimv,), stride=(0,)),
                    ),
                )
                thr_mma = tiled_mma.get_slice(tidx)
                taccOgLSE = layout_utils.reshape_acc_to_mn(
                    thr_mma.partition_C(gLSE_expanded)
                )
                taccOcO = layout_utils.reshape_acc_to_mn(
                    thr_mma.partition_C(cO)
                )
                t0accOcO = layout_utils.reshape_acc_to_mn(
                    thr_mma.get_slice(0).partition_C(cO)
                )
                if taccOcO[0][1] == 0:
                    for m in cutlass.range(
                        cute.size(taccOgLSE.shape[1]), unroll_full=True
                    ):
                        if (
                            t0accOcO[m, 0][0]
                            < seqlen.seqlen_q
                            - m_block * self.tile_m
                            - taccOcO[0][0]
                        ):
                            taccOgLSE[m, 0] = lse[m]
            else:
                pack_gqa.store_LSE(
                    mLSE_cur,
                    lse,
                    tiled_mma,
                    tidx,
                    m_block,
                    seqlen.seqlen_q,
                )

        mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[
            None, None, head_idx
        ]
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_epilogue_threads,
        )
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOrO = cute.make_fragment_like(tOsO, self.dtype)
        cute.autovec_copy(tOsO, tOrO)
        if const_expr(not self.pack_gqa):
            gO = cute.local_tile(
                mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0)
            )
            tOgO = gmem_thr_copy_O.partition_D(gO)
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
            tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
            for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                if (
                    t0OcO[0, rest_m, 0][0]
                    < seqlen.seqlen_q
                    - m_block * self.tile_m
                    - tOcO[0][0]
                ):
                    cute.copy(
                        gmem_tiled_copy_O,
                        tOrO[None, rest_m, None],
                        tOgO[None, rest_m, None],
                        pred=(
                            tOpO[None, rest_m, None]
                            if const_expr(self.check_hdim_v_oob)
                            else None
                        ),
                    )
        else:
            pack_gqa.store_O(
                mO_cur,
                tOrO,
                gmem_tiled_copy_O,
                tidx,
                m_block,
                seqlen.seqlen_q,
            )

    @cute.jit
    def mma_pv_only(
        self,
        tiled_mma_pv: cute.TiledMma,
        mO: cute.Tensor,
        sVt: cute.Tensor,
        sP: cute.Tensor,
        sScale: cute.Tensor,
        sO: cute.Tensor,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        sWorkInfo: Optional[cute.Tensor],
        num_splits: Int32 = Int32(1),
        mOFinal: Optional[cute.Tensor] = None,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        warp_group_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_warp_group
        )
        wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))
        acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
            wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), sP, sVt
        )
        mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)

        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        taccOcO_row = layout_utils.reshape_acc_to_mn(
            tiled_mma_pv.get_slice(tidx).partition_C(cO)
        )[None, 0]
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=softmax_scale,
        )
        row_scale = cute.make_rmem_tensor_like(softmax.row_max, Float32)

        q_consumer_phase = Int32(0)
        kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )
        tile_scheduler = TileSchedulerCls()
        work_info_phase = Int32(0)
        work_tile = (
            self.wait_dynamic_work(sWorkInfo, work_info_phase)
            if const_expr(self.use_dynamic_varlen)
            else tile_scheduler.initial_work_tile_info()
        )
        if const_expr(self.use_dynamic_varlen):
            work_info_phase ^= 1

        # Supply the PV half of the initial PEmpty phase. Every later phase is
        # completed by the score WG before it overwrites P.
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierFwd.PEmpty),
            number_of_threads=self.num_mma_threads,
        )

        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            split_idx, num_splits_cur = self.resolve_num_splits(
                split_idx, num_splits
            )
            seqlen = SeqlenInfoCls(batch_idx)
            psc = (
                self._mDynamicCausal[batch_idx]
                if const_expr(self._mDynamicCausal is not None)
                else None
            )
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits_cur
            )
            if const_expr(self._mDynamicCausal is not None):
                if not psc:
                    n_block_max_full = cute.ceil_div(seqlen.seqlen_k, self.tile_n)
                    if const_expr(self.is_split_kv):
                        num_n_blocks_per_split = cute.ceil_div(
                            n_block_max_full, num_splits_cur
                        )
                        n_block_min = split_idx * num_n_blocks_per_split
                        n_block_max = cutlass.min(
                            n_block_min + num_n_blocks_per_split, n_block_max_full
                        )
                    else:
                        n_block_min = Int32(0)
                        n_block_max = n_block_max_full
            n_block_max_orig = n_block_max
            num_n_blocks = cutlass.max(n_block_max - n_block_min, 1)

            pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)

            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.PFull),
                number_of_threads=self.num_mma_threads,
            )
            mma_pv_fn(B_idx=kv_consumer_state.index, zero_init=True, wg_wait=0)
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.PEmpty),
                number_of_threads=self.num_mma_threads,
            )
            pipeline_v.consumer_release(kv_consumer_state)
            kv_consumer_state.advance()

            for _ in cutlass.range(num_n_blocks - 1, unroll=1):
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.PFull),
                    number_of_threads=self.num_mma_threads,
                )
                self.load_asym_scale(
                    sScale, taccOcO_row, row_scale, kv_consumer_state.index
                )
                softmax.rescale_O(acc_O, row_scale)
                mma_pv_fn(B_idx=kv_consumer_state.index, zero_init=False, wg_wait=0)
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierFwd.PEmpty),
                    number_of_threads=self.num_mma_threads,
                )
                pipeline_v.consumer_release(kv_consumer_state)
                kv_consumer_state.advance()

            # The final PFull phase carries normalization scales rather than P.
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.PFull),
                number_of_threads=self.num_mma_threads,
            )
            self.load_asym_scale(
                sScale, taccOcO_row, row_scale, kv_consumer_state.index
            )
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.PEmpty),
                number_of_threads=self.num_mma_threads,
            )
            softmax.rescale_O(acc_O, row_scale)

            if const_expr(self.is_split_kv):
                if n_block_min >= n_block_max_orig:
                    acc_O.fill(Float32(0.0))

            self.epilogue_split_or_final(
                acc_O,
                softmax.row_sum,
                mO,
                None,
                mOFinal,
                None,
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
                num_splits_cur,
            )

            pipeline_q.consumer_release_w_index(0)
            q_consumer_phase ^= 1
            work_tile = (
                self.wait_dynamic_work(sWorkInfo, work_info_phase)
                if const_expr(self.use_dynamic_varlen)
                else tile_scheduler.advance_to_next_work()
            )
            if const_expr(self.use_dynamic_varlen):
                work_info_phase ^= 1

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
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
            # Fence and barrier to make smem store visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()

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
    def _narrow_q_to_compute(self, q_narrow_params):
        """In-place narrow the bf16 Q tile -> fp16 compute dtype. src/dst alias the same
        smem (both 2 bytes), so each thread reads bf16 into registers, converts, and writes
        fp16 back. A 256-thread NarrowQ barrier publishes the fp16 Q to both MMA warpgroups
        before the QK WGMMA. Called once per Q tile (Q is reused across all KV blocks)."""
        # The in-place aliasing (sQ_mma reinterprets sQ's storage) is only valid when the
        # Q-input and compute dtypes have the same width -- bf16<->fp16 are both 16-bit. A
        # different width would make sQ_mma's elements overlap sQ's incorrectly.
        assert self.q_dtype.width == self.dtype.width, (
            "in-place Q narrow requires q_dtype and compute dtype of equal width"
        )
        qp = q_narrow_params
        rS = cute.make_rmem_tensor_like(qp.tQsrc)
        rD = cute.make_rmem_tensor_like(qp.tQdst)
        cute.autovec_copy(qp.tQsrc, rS)
        rD.store(rS.load().to(self.dtype))
        cute.autovec_copy(rD, qp.tQdst)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.NarrowQ), number_of_threads=self.num_mma_threads
        )

    @cute.jit
    def _cast_tile_fp8_to_f16(self, pipeline_stage, tStage, tsDst, phase):
        """Cast one staged fp8 tile -> fp16 smem without applying a descale."""
        rS = cute.make_rmem_tensor_like(tStage)
        rD = cute.make_rmem_tensor_like(tsDst)
        pipeline_stage.consumer_wait_w_index_phase(0, phase)
        cute.autovec_copy(tStage, rS)
        rD.store(utils.cvt_fp8_to_f16_packed(rS.load(), self.dtype))
        cute.autovec_copy(rD, tsDst)
        pipeline_stage.consumer_release_w_index(0)
        cute.arch.fence_view_async_shared()

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
        """FP8-KV first-half block: cooperative 256-thread cast of the first K tile
        + DequantK barrier, then the synchronous QK."""
        dp = dequant_params
        nthr = const_expr(self.num_mma_threads)
        self._cast_tile_fp8_to_f16(dp.pipeline_stage, dp.tStage_K, dp.tsK, Int32(0))
        cute.arch.barrier(barrier_id=int(NamedBarrierFwd.DequantK), number_of_threads=nthr)
        acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)

        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        mask_fn(acc_S, n_block=n_block, mask_seqlen=True)
        row_scale = softmax.online_softmax(acc_S, is_first=is_first_block)
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = tOrP
        tOrP_cur.store(tOrP_acc.load().to(self.dtype))
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
        """FP8-KV final PV: each WG casts its own hdimv-half of the last V tile
        + WG-local DequantV barrier, then the synchronous PV."""
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)
        dp = dequant_params
        nthr_wg = const_expr(self.num_threads_per_warp_group)
        self._cast_tile_fp8_to_f16(dp.pipeline_stage, dp.tStage_V, dp.tsV, Int32(1))
        if dp.warp_group_idx == 0:
            cute.arch.barrier(barrier_id=int(NamedBarrierFwd.DequantV0), number_of_threads=nthr_wg)
        else:
            cute.arch.barrier(barrier_id=int(NamedBarrierFwd.DequantV1), number_of_threads=nthr_wg)
        mma_pv_fn(B_idx=kv_consumer_state.index, zero_init=zero_init, wg_wait=0)
        kv_consumer_state.advance()
        return kv_consumer_state

    @cute.jit
    def mma_one_n_block(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        tiled_mma_qv: cute.TiledMma,
        tSrQv: Optional[cute.Tensor],
        tSrV: Optional[cute.Tensor],
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        sScale: Optional[cute.Tensor],
        taccOcO_row: cute.Tensor,
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
        if const_expr(self.has_qv):
            # V participates in both score formation and the later PV GEMM.
            # Keep its pipeline stage owned until both operations complete.
            pipeline_v.consumer_wait(
                smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read)
            )
            self._gemm_two_stage(
                tiled_mma_qv,
                acc_S,
                tSrQv,
                tSrV,
                smem_pipe_read.index,
                zero_init=False,
                wg_wait=-1,
            )
        self.warp_scheduler_barrier_arrive()
        if const_expr(self.has_qv):
            # QK is the older of the two committed WGMMA groups.
            warpgroup.wait_group(1)
        else:
            warpgroup.wait_group(0)
        pipeline_k.consumer_release(smem_pipe_read)
        if const_expr(self.has_qv):
            warpgroup.wait_group(0)

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
        if const_expr(self.use_asym_dv512):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.PEmpty),
                number_of_threads=self.num_mma_threads,
            )
            if const_expr(not is_first_n_block):
                self.store_asym_scale(
                    sScale, taccOcO_row, row_scale, smem_pipe_read.index
                )
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        softmax.rescale_O(acc_O, row_scale)
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
            if const_expr(self.use_asym_dv512):
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierFwd.PFull),
                    number_of_threads=self.num_mma_threads,
                )
        if const_expr(not self.has_qv):
            pipeline_v.consumer_wait(
                smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read)
            )
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
        tiled_mma_qv: cute.TiledMma,
        tSrQv: Optional[cute.Tensor],
        tSrV: Optional[cute.Tensor],
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        sScale: Optional[cute.Tensor],
        taccOcO_row: cute.Tensor,
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
    ):
        # FP8-KV intra-WG overlap: cast K[n] for QK[n], then cast V[n_prev]
        # under the QK WGMMA for PV[n_prev]. Separate from the generic overlap
        # to keep that path unchanged.
        smem_pipe_read_v = smem_pipe_read.clone()
        smem_pipe_read.advance()
        dp = dequant_params
        nthr = const_expr(self.num_mma_threads)
        nthr_wg = const_expr(self.num_threads_per_warp_group)
        # ---- 1. cast K[n] -> sK (cooperative 256-thread, staging phase 1) ----
        self._cast_tile_fp8_to_f16(dp.pipeline_stage, dp.tStage_K, dp.tsK, Int32(1))
        cute.arch.barrier(barrier_id=int(NamedBarrierFwd.DequantK), number_of_threads=nthr)
        # ---- 2. S = Q @ K.T (async) ----
        self.warp_scheduler_barrier_sync()
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)
        # ---- 3. cast V[n_prev] -> sV (WG-split, phase 0) WHILE QK[n] runs ----
        self._cast_tile_fp8_to_f16(dp.pipeline_stage, dp.tStage_V, dp.tsV, Int32(0))
        if dp.warp_group_idx == 0:
            cute.arch.barrier(barrier_id=int(NamedBarrierFwd.DequantV0), number_of_threads=nthr_wg)
        else:
            cute.arch.barrier(barrier_id=int(NamedBarrierFwd.DequantV1), number_of_threads=nthr_wg)
        # ---- 4. O += P @ V (async), then wait QK then PV ----
        mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(1)
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
        warpgroup.wait_group(0)
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = tOrP
        utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, row_scale)
        if const_expr(self.rescale_O_before_gemm):
            scores_scale.store(row_scale.load())
        return smem_pipe_read

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
