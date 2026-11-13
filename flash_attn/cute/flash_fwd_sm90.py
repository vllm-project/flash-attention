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
from cutlass.utils import TensorMapManager, TensorMapUpdateMode
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
        kv_cluster_m: int = 1,
        single_qk_wg: bool = False,
        kv_split_hdim: bool = False,
        k_chunks: int = 4,
        split_kv_producer: bool = False,
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
        # kv_cluster_m > 1: a cluster of kv_cluster_m CTAs along M (same batch / kv head) runs
        # the same n_block range and TMA-multicasts every K/V tile, so each CTA fetches only
        # 1 / kv_cluster_m of it from L2.
        self.kv_cluster_m = kv_cluster_m
        assert kv_cluster_m == 1 or (self.use_tma_KV and not fp8_kv_dequant), (
            "K/V multicast requires TMA K/V"
        )
        self.cluster_shape_mn = (kv_cluster_m, 1)
        # single_qk_wg: when the PV GEMM is split along hdimv across 2 MMA WGs (hdim > 256), only
        # WG 0 computes S = Q K^T and the softmax; it hands P and the per-row O scale to WG 1
        # (which only runs its half of the PV GEMM) through a 2-stage smem pipeline.
        self.single_qk_wg = single_qk_wg
        assert not single_qk_wg or (intra_wg_overlap and not (mma_pv_is_rs or fp8_kv_dequant)), (
            "single_qk_wg shares P through smem and needs intra_wg_overlap"
        )
        # kv_split_hdim (single_qk_wg, TMA K/V): the K tile is loaded and released as k_chunks hdim
        # chunks (k_chunks K pipeline stages over one buffer), each released once its QK WGMMA
        # group is done, so the next K tile's chunks load while the QK GEMM still reads the later
        # chunks. Likewise V, as 2 hdim chunks (halves) read by one MMA WG each, so each WG
        # releases its chunk of V independently.
        self.kv_split_hdim = kv_split_hdim
        # Number of hdim chunks K is split into (each a multiple of the 64-column smem atom).
        self.k_chunks = k_chunks if kv_split_hdim else 1
        assert not kv_split_hdim or (single_qk_wg and self.use_tma_KV)
        # split_kv_producer (K and V both split; prefill): producer warp 0 issues the K loads (and
        # Q) and warp 1 the V loads, so the next K tile is not held behind the previous V tile's
        # release.
        self.split_kv_producer = split_kv_producer
        assert not split_kv_producer or kv_split_hdim
        # sP, the P pipeline and the K / V chunk state follow the K / V pipeline phase.
        assert not (single_qk_wg or kv_split_hdim) or self.num_stages == 1, (
            "single_qk_wg and kv_split_hdim need num_stages == 1"
        )
        # The cluster tail waits for the V releases only from producer warp 1.
        assert kv_cluster_m == 1 or split_kv_producer, "kv_cluster_m > 1 needs split_kv_producer"
        # Each K chunk is a whole number of 64-column smem atoms.
        assert not kv_split_hdim or self.tile_hdim % (64 * self.k_chunks) == 0, (
            "kv_split_hdim needs tile_hdim to be a multiple of 64 * k_chunks"
        )
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
        assert not self.single_qk_wg or (self.atom_layout_n == 2 and self.tile_m == 64)
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            mma_dtype,
            mma_dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(
                self.tile_m // 64,
                1 if self.single_qk_wg else self.atom_layout_n,
                1,
            ),
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
        mbar_ptr_K_struct = cute.struct.MemRange[
            cutlass.Int64,
            (self.k_chunks if const_expr(self.kv_split_hdim) else self.num_stages) * 2,
        ]
        mbar_ptr_V_struct = cute.struct.MemRange[
            cutlass.Int64, (2 if const_expr(self.kv_split_hdim) else self.num_stages) * 2
        ]

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

        if const_expr(self.single_qk_wg):
            # Per-row softmax stats exchanged between the MMA WGs: 2 slots alternating by n_block
            # (O scale of WG 0's P) and 1 for the final O scale.
            sRowStat_struct = cute.struct.Align[cute.struct.MemRange[Float32, self.tile_m * 3], 16]
            # single_qk_wg: full / empty barriers of the 2-stage P pipeline from WG 0 to WG 1.
            mbar_ptr_P_struct = cute.struct.MemRange[cutlass.Int64, 2 * 2]

            @cute.struct
            class SharedStorageQKVSplitQK:
                mbar_ptr_Q: mbar_ptr_Q_struct
                mbar_ptr_K: mbar_ptr_K_struct
                mbar_ptr_V: mbar_ptr_V_struct
                mbar_ptr_P: mbar_ptr_P_struct
                sV: sV_struct
                sQ: sQ_struct
                sK: sK_struct
                sP: sP_struct
                sRowStat: sRowStat_struct

            assert not self.Q_in_regs
            return SharedStorageQKVSplitQK

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
        # single_qk_wg: the QK MMA uses only WG 0; the PV MMA spans all MMA WGs.
        self.num_mma_threads = (
            tiled_mma_pv if const_expr(self.single_qk_wg) else tiled_mma_qk
        ).size
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
        assert not (self.single_qk_wg and self.use_block_sparsity)

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
                mV.element_type,
                LayoutEnum.ROW_MAJOR,
                (self.tile_m, self.tile_n),
                # single_qk_wg: 2 sP buffers, alternating by n_block
                2 if const_expr(self.single_qk_wg) else None,
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
        self.sK_chunk_layout = None
        if const_expr(self.kv_split_hdim):
            # (tile_n, hdim / k_chunks) x k_chunks; aliases sK_layout byte for byte.
            self.sK_chunk_layout = sm90_utils.make_smem_layout(
                mK.element_type,
                LayoutEnum.ROW_MAJOR,
                (self.tile_n, self.tile_hdim // self.k_chunks),
                self.k_chunks,
            )
            self.tma_copy_bytes["K"] = cute.size_in_bytes(
                mK.element_type, cute.select(self.sK_chunk_layout, mode=[0, 1])
            )
        self.sV_chunk_layout = None
        if const_expr(self.kv_split_hdim):
            # (tile_n, hdimv / 2) x 2; aliases sV_layout byte for byte.
            self.sV_chunk_layout = sm90_utils.make_smem_layout(
                mV.element_type, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdimv // 2), 2
            )
            self.tma_copy_bytes["V"] = cute.size_in_bytes(
                mV.element_type, cute.select(self.sV_chunk_layout, mode=[0, 1])
            )
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
            if const_expr(self.kv_cluster_m > 1):
                gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SMulticastOp()
            tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mK,
                cute.select(self.sK_chunk_layout, mode=[0, 1])
                if const_expr(self.kv_split_hdim)
                else sK_tma_box,
                (self.tile_n, self.tile_hdim // self.k_chunks),
                self.kv_cluster_m,
            )
            tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mV_tma,
                cute.select(self.sV_chunk_layout, mode=[0, 1])
                if const_expr(self.kv_split_hdim)
                else sV_tma_box,
                (
                    self.tile_n,
                    self.tile_hdimv // 2 if const_expr(self.kv_split_hdim) else self.tile_hdimv,
                ),
                self.kv_cluster_m,
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
            cluster_shape_mn=self.cluster_shape_mn,
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
            self.sK_chunk_layout,
            self.sV_chunk_layout,
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
            cluster=[*self.cluster_shape_mn, 1] if const_expr(self.kv_cluster_m > 1) else None,
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
        sK_chunk_layout: cute.ComposedLayout | None,
        sV_chunk_layout: cute.ComposedLayout | None,
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
        # single_qk_wg: only WG 0 reads Q and K.
        qk_warps = mma_warps
        if const_expr(self.single_qk_wg):
            qk_warps = ThreadCooperativeGroup(
                self.num_threads_per_warp_group // cute.arch.WARP_SIZE
            )
        pipeline_p = None
        if const_expr(self.single_qk_wg):
            # Every thread of WG 0 (producer) / WG 1 (consumer) arrives.
            wg_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
            pipeline_p = pipeline.PipelineAsync.create(
                barrier_storage=storage.mbar_ptr_P.data_ptr(),
                num_stages=2,
                producer_group=wg_threads,
                consumer_group=wg_threads,
                defer_sync=True,
            )
        if const_expr(self.use_tma_Q):
            pipeline_q = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=tma_warp,
                consumer_group=qk_warps,
                tx_count=self.tma_copy_bytes["Q"],
                defer_sync=True,
            )
        else:
            pipeline_q = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=load_threads,
                consumer_group=qk_warps,
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
            k_consumer_group = qk_warps
            # V is split in half: WG 0 reads the first half, WG 1 the second.
            # Each WG releases only its half (qk_warps is used here as a 1-WG-sized group).
            v_consumer_group = qk_warps if const_expr(self.kv_split_hdim) else mma_warps
            kv_cta_layout_vmnk = None
            if const_expr(self.kv_cluster_m > 1):
                # Every consumer warp releases each K/V stage to all CTAs that multicast into it.
                k_consumer_group, v_consumer_group = [
                    ThreadCooperativeGroup(g.size * self.kv_cluster_m)
                    for g in (k_consumer_group, v_consumer_group)
                ]
                kv_cta_layout_vmnk = cute.make_layout((1, *self.cluster_shape_mn, 1))
            pipeline_k = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.k_chunks if const_expr(self.kv_split_hdim) else self.num_stages,
                producer_group=tma_warp,
                consumer_group=k_consumer_group,
                tx_count=self.tma_copy_bytes["K"],
                cta_layout_vmnk=kv_cta_layout_vmnk,
                defer_sync=True,
            )
            pipeline_v = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=2 if const_expr(self.kv_split_hdim) else self.num_stages,
                producer_group=tma_warp,
                consumer_group=v_consumer_group,
                tx_count=self.tma_copy_bytes["V"],
                cta_layout_vmnk=kv_cta_layout_vmnk,
                defer_sync=True,
            )
        else:
            pipeline_k = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=load_threads,
                consumer_group=qk_warps,
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
        sRowStat = None
        if const_expr(self.single_qk_wg):
            sRowStat = storage.sRowStat.get_tensor(cute.make_layout((self.tile_m, 3)))

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
        # The n_block range is shared by all CTAs of a K/V multicast cluster: it is the range
        # of the cluster's combined (tile_m * kv_cluster_m)-row block.
        block_info_kv = block_info
        if const_expr(self.kv_cluster_m > 1):
            block_info_kv = BlockInfo(
                self.tile_m * self.kv_cluster_m,
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
            sV_load = sV
            sK_load = sK
            if const_expr(self.kv_split_hdim):
                sV_load = storage.sV.get_tensor(
                    sV_chunk_layout.outer, swizzle=sV_chunk_layout.inner
                )
                sK_load = storage.sK.get_tensor(
                    sK_chunk_layout.outer, swizzle=sK_chunk_layout.inner
                )
            self.load(
                mQ,
                mK,
                mV,
                mV_gmem,
                tensormap_workspace,
                sQ,
                sK_load,
                sV_load,
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
                block_info_kv=block_info_kv,
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
                block_info_kv=block_info_kv,
                sRowStat=sRowStat,
                pipeline_p=pipeline_p,
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
        block_info_kv: Optional[BlockInfo] = None,
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
        is_load_warp = (
            warp_idx_in_wg == 0
            or const_expr(not self.use_tma_KV or not self.use_tma_Q)
            or (const_expr(self.split_kv_producer) and warp_idx_in_wg == 1)
        )
        # KV loading restricted to warp 0 for TMA, all warps for non-TMA KV
        is_kv_load_warp = (
            warp_idx_in_wg == 0
            or const_expr(not self.use_tma_KV)
            or (const_expr(self.split_kv_producer) and warp_idx_in_wg == 1)
        )

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
                # The warp that issues the V loads also writes the V descriptor.
                v_desc_warp = 1 if const_expr(self.split_kv_producer) else 0
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_V, tensormap_V_gmem, v_desc_warp
                )
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
                        (mV_bounded,), (tma_atom_V,), (tensormap_V_gmem,), v_desc_warp, (None,)
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
                    v_tma_kwargs = (
                        {"tma_desc_ptr": v_desc_ptr} if const_expr(self.ragged_tma_V) else {}
                    )
                    if const_expr(self.kv_split_hdim):
                        tma_load_K_chunks = self._make_kv_chunk_copies(
                            tma_atom_K,
                            mK_cur,
                            sK,
                            self.tile_hdim,
                            paged=mPageTable is not None,
                            num_chunks=self.k_chunks,
                        )
                        tma_load_V_chunks = self._make_kv_chunk_copies(
                            tma_atom_V,
                            mV_cur,
                            sV,
                            self.tile_hdimv,
                            paged=mPageTable is not None,
                            **v_tma_kwargs,
                        )
                    elif const_expr(self.kv_cluster_m == 1):
                        tma_load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_K, 0, cute.make_layout(1), gK, sK
                        )
                        tma_load_V_fn, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_V,
                            0,
                            cute.make_layout(1),
                            gV,
                            sV,
                            **v_tma_kwargs,
                        )
                    else:
                        # Each CTA loads its 1 / kv_cluster_m slice of the tile and multicasts
                        # it to every CTA of the cluster.
                        kv_cta_coord, kv_cta_layout, kv_mcast_mask = self._kv_mcast_params()
                        tma_load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_K,
                            kv_cta_coord,
                            kv_cta_layout,
                            gK,
                            sK,
                            mcast_mask=kv_mcast_mask,
                        )
                        tma_load_V_fn, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_V,
                            kv_cta_coord,
                            kv_cta_layout,
                            gV,
                            sV,
                            mcast_mask=kv_mcast_mask,
                            **v_tma_kwargs,
                        )
                    if const_expr(not self.kv_split_hdim):
                        tma_load_K_fn = copy_utils.tma_producer_copy_fn(tma_load_K_fn, pipeline_k)
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

                if const_expr(self.kv_split_hdim):
                    load_K = partial(self._load_kv_chunks, tma_load_K_chunks, pipeline_k)
                    load_V = partial(self._load_kv_chunks, tma_load_V_chunks, pipeline_v)
                else:
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
                    if const_expr(self.kv_cluster_m == 1):
                        n_block_min, n_block_max = block_info.get_n_block_min_max(
                            seqlen, m_block, split_idx, num_splits
                        )
                    else:
                        n_block_min, n_block_max = block_info_kv.get_n_block_min_max(
                            seqlen, m_block // self.kv_cluster_m, split_idx, num_splits
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
                    # single_qk_wg: issue the Q TMA first, so Q streams in while the first
                    # page-table entry (needed for the K TMA) is still being read.
                    q_first = const_expr(self.single_qk_wg and self.use_tma_Q)
                    if const_expr(q_first):
                        if warp_idx_in_wg == 0:
                            pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                            load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
                            q_producer_phase ^= 1
                    page_idx = None
                    if const_expr(mPageTable is not None and self.use_tma_KV):
                        page_idx = mPageTable[batch_idx, n_block]

                    # First iteration: load K on pipeline_k, Q on pipeline_q
                    if is_kv_load_warp:
                        if const_expr(not self.kv_split_hdim):
                            pipeline_k.producer_acquire(kv_producer_state)
                        if const_expr(not self.use_tma_KV):
                            paged_kv_manager.load_page_table(n_block)
                        if const_expr(self.split_kv_producer):
                            if warp_idx_in_wg == 0:
                                load_K(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )
                        else:
                            load_K(
                                block=n_block, producer_state=kv_producer_state, page_idx=page_idx
                            )
                    if const_expr(not q_first and self.use_tma_Q):
                        if warp_idx_in_wg == 0:
                            pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                            load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
                            q_producer_phase ^= 1
                    elif const_expr(not q_first):
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
                                if const_expr(self.split_kv_producer):
                                    if warp_idx_in_wg == 0:
                                        load_K(
                                            block=n_block,
                                            producer_state=kv_producer_state,
                                            page_idx=page_idx,
                                        )
                                    else:
                                        load_V(
                                            block=n_block_prev,
                                            producer_state=kv_producer_state_prev,
                                            page_idx=page_idx_prev,
                                        )
                                else:
                                    if const_expr(not self.kv_split_hdim):
                                        pipeline_k.producer_acquire(kv_producer_state)
                                    load_K(
                                        block=n_block,
                                        producer_state=kv_producer_state,
                                        page_idx=page_idx,
                                    )
                                    if const_expr(not self.kv_split_hdim):
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
                            if const_expr(self.split_kv_producer):
                                if warp_idx_in_wg == 1:
                                    load_V(
                                        block=n_block,
                                        producer_state=kv_producer_state,
                                        page_idx=page_idx,
                                    )
                            else:
                                if const_expr(not self.kv_split_hdim):
                                    pipeline_v.producer_acquire(kv_producer_state)
                                load_V(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
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
                if const_expr(self.kv_cluster_m > 1):
                    # Peers arrive on our K / V empty barriers remotely; wait for them before exit.
                    if const_expr(self.kv_split_hdim):
                        if warp_idx_in_wg == 0:
                            for h in cutlass.range_constexpr(self.k_chunks):
                                pipeline_k.producer_acquire(
                                    self._kv_chunk_state(kv_producer_state, h)
                                )
                        if warp_idx_in_wg == 1:
                            for h in cutlass.range_constexpr(2):
                                pipeline_v.producer_acquire(
                                    self._v_chunk_state(kv_producer_state, h)
                                )
                    else:
                        pipeline_k.producer_tail(kv_producer_state.clone())
                if const_expr(not self.kv_split_hdim):
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
        block_info_kv: Optional[BlockInfo] = None,
        sRowStat: Optional[cute.Tensor] = None,
        pipeline_p: Optional[pipeline.PipelineAsync] = None,
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
        elif const_expr(self.kv_split_hdim):
            mma_qk_fn = partial(
                self._gemm_qk_k_split,
                tiled_mma_qk,
                (self.tile_m, self.tile_n),
                tSrQ,
                tSrK,
                pipeline_k,
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
        # single_qk_wg: rows of this thread's S (WG 0) and acc_O fragments, for the O scales
        # passed through sRowStat.
        row_idx_qk, row_idx_pv = None, None
        if const_expr(self.single_qk_wg):
            tScS_qk_mn = layout_utils.reshape_acc_to_mn(
                thr_mma_qk.partition_C(cute.make_identity_tensor((self.tile_m, self.tile_n)))
            )
            tOcO_mn = layout_utils.reshape_acc_to_mn(
                tiled_mma_pv.get_slice(tidx).partition_C(
                    cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
                )
            )
            row_idx_qk = cute.make_rmem_tensor(cute.size(tScS_qk_mn, mode=[0]), Int32)
            row_idx_pv = cute.make_rmem_tensor(cute.size(tOcO_mn, mode=[0]), Int32)
            for r in cutlass.range_constexpr(cute.size(row_idx_qk)):
                row_idx_qk[r] = tScS_qk_mn[r, 0][0]
            for r in cutlass.range_constexpr(cute.size(row_idx_pv)):
                row_idx_pv[r] = tOcO_mn[r, 0][0]
        smem_copy_params = SimpleNamespace(
            smem_thr_copy_P=smem_thr_copy_P,
            tPsP=tPsP,
            tOrP_res=tOrP_res,
            pipeline_p=pipeline_p,
            sRowStat=sRowStat,
            row_idx_qk=row_idx_qk,
            row_idx_pv=row_idx_pv,
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
            **(dict(smem_copy_params=smem_copy_params) if const_expr(self.single_qk_wg) else {}),
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
            if const_expr(self.kv_cluster_m == 1):
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block, split_idx, num_splits
                )
            else:
                n_block_min, n_block_max = block_info_kv.get_n_block_min_max(
                    seqlen, m_block // self.kv_cluster_m, split_idx, num_splits
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
                if const_expr(self.single_qk_wg):
                    if self._is_qk_wg():
                        pipeline_q.consumer_release_w_index(0)
                else:
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
            if const_expr(not self.single_qk_wg):
                row_scale = softmax_tile.finalize(final_scale=v_descale_tile, sink_val=sink_val)
            else:
                # WG 0 owns the softmax state; WG 1 gets its final O scale through smem.
                row_scale = cute.make_fragment_like(softmax_tile.row_max, Float32)
                if self._is_qk_wg():
                    row_scale.store(
                        softmax_tile.finalize(final_scale=v_descale_tile, sink_val=sink_val).load()
                    )
                    self._store_row_stat(smem_copy_params, row_scale, Int32(2))
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.RowStat), number_of_threads=self.num_mma_threads
                )
                if not self._is_qk_wg():
                    row_scale.store(self._load_row_stat(smem_copy_params, Int32(2)).load())
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
        block_args = dict(
            n_block=n_block,
            mma_qk_fn=mma_qk_fn,
            kv_consumer_state=kv_consumer_state,
            pipeline_k=pipeline_k,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            softmax=softmax,
            seqlen=seqlen,
            scores_scale=scores_scale,
            acc_O=acc_O,
            mask_fn=mask_fn,
            score_mod_fn=score_mod_fn,
            is_first_block=is_first_block,
        )
        if const_expr(not self.single_qk_wg):
            self._first_half_block_overlap_body(**block_args)
        else:
            if self._is_qk_wg():
                self._first_half_block_overlap_body(**block_args)
            else:
                acc_O.fill(0.0)
        return kv_consumer_state

    def _first_half_block_overlap_body(
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
        if const_expr(self.kv_split_hdim):
            acc_S = mma_qk_fn(kv_state=kv_consumer_state)
            self._k_last_chunk_wait_release(pipeline_k, kv_consumer_state, num_younger_groups=0)
        else:
            pipeline_k.consumer_wait(
                kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state)
            )
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
            self._p_producer_acquire(smem_copy_params, kv_consumer_state)
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(
                smem_copy_params.smem_thr_copy_P,
                tPrP,
                self._p_stage(smem_copy_params.tPsP, kv_consumer_state),
            )
            if const_expr(self.single_qk_wg):
                self._store_row_stat(smem_copy_params, row_scale, kv_consumer_state.phase)
            # Fence and barrier to make smem store visible to WGMMA
            # (single_qk_wg: done when P is published, see _p_producer_commit)
            if const_expr(not self.single_qk_wg):
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
            self.cross_wg_p_raw_barrier()

        # For RescaleOBeforeGemm: initialize acc_O
        if const_expr(self.rescale_O_before_gemm):
            acc_O.fill(0.0)
            scores_scale.store(row_scale.load())

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
        smem_copy_params: Optional[SimpleNamespace] = None,
    ):
        """Processes the final PV GEMM when using intra-warpgroup-overlap"""
        if const_expr(not self.single_qk_wg):
            self._last_half_block_overlap_body(
                kv_consumer_state, pipeline_v, mma_pv_fn, zero_init, scores_scale, softmax, acc_O
            )
        else:
            if self._is_qk_wg():
                self._p_producer_commit(smem_copy_params, kv_consumer_state)
                self._last_half_block_overlap_body(
                    kv_consumer_state,
                    pipeline_v,
                    mma_pv_fn,
                    zero_init,
                    scores_scale,
                    softmax,
                    acc_O,
                )
            else:
                self._pv_only_block(
                    kv_consumer_state,
                    pipeline_v,
                    partial(mma_pv_fn, zero_init=zero_init),
                    acc_O,
                    softmax,
                    smem_copy_params,
                )
        kv_consumer_state.advance()
        return kv_consumer_state

    def _last_half_block_overlap_body(
        self,
        kv_consumer_state,
        pipeline_v,
        mma_pv_fn: Callable,
        zero_init: bool,
        scores_scale: Optional[cute.Tensor] = None,
        softmax: Optional[Softmax] = None,
        acc_O: Optional[cute.Tensor] = None,
    ):
        # For RescaleOBeforeGemm: rescale O before the final PV GEMM
        if const_expr(self.rescale_O_before_gemm):
            if const_expr(self.single_qk_wg):
                softmax.rescale_O_if_needed(acc_O, scores_scale)
            else:
                softmax.rescale_O(acc_O, scores_scale)

        v_state = self._v_chunk_state(kv_consumer_state, 0)
        pipeline_v.consumer_wait(v_state, pipeline_v.consumer_try_wait(v_state))
        mma_pv_fn(
            B_idx=kv_consumer_state.index,
            zero_init=zero_init,
            wg_wait=0,
            **self._p_idx(kv_consumer_state),
        )
        pipeline_v.consumer_release(v_state)

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
        block_args = dict(
            n_block=n_block,
            mma_qk_fn=mma_qk_fn,
            mma_pv_fn=mma_pv_fn,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            acc_O=acc_O,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            softmax=softmax,
            seqlen=seqlen,
            scores_scale=scores_scale,
            score_mod_fn=score_mod_fn,
            mask_fn=mask_fn,
            check_inf=check_inf,
        )
        if const_expr(not self.single_qk_wg):
            self._n_block_intrawg_overlap_body(smem_pipe_read, smem_pipe_read_v, **block_args)
        else:
            if self._is_qk_wg():
                self._n_block_intrawg_overlap_body(smem_pipe_read, smem_pipe_read_v, **block_args)
            else:
                # PV GEMM of the previous n_block with the P / O scale published by WG 0.
                self._pv_only_block(
                    smem_pipe_read_v, pipeline_v, mma_pv_fn, acc_O, softmax, smem_copy_params
                )
        return smem_pipe_read

    def _n_block_intrawg_overlap_body(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        smem_pipe_read_v: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
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
        """S = Q K^T of the current n_block (smem_pipe_read) overlapped with O += P V of the
        previous one (smem_pipe_read_v), then the online softmax of S."""
        if const_expr(not self.kv_split_hdim):
            pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
        self.warp_scheduler_barrier_sync()
        # S = Q @ K.T
        if const_expr(self.kv_split_hdim):
            acc_S = mma_qk_fn(kv_state=smem_pipe_read)
        else:
            acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        self._p_producer_commit(smem_copy_params, smem_pipe_read_v)
        # RescaleOBeforeGemm: rescale O while QK GEMM is in flight, before PV GEMM
        if const_expr(self.rescale_O_before_gemm):
            if const_expr(self.single_qk_wg):
                softmax.rescale_O_if_needed(acc_O, scores_scale)
            else:
                softmax.rescale_O(acc_O, scores_scale)
        v_state = self._v_chunk_state(smem_pipe_read_v, 0)
        pipeline_v.consumer_wait(v_state, pipeline_v.consumer_try_wait(v_state))
        # O += P @ V
        mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1, **self._p_idx(smem_pipe_read_v))
        self.warp_scheduler_barrier_arrive()
        if const_expr(self.kv_split_hdim):
            # The PV GEMM was committed after all QK chunks.
            self._k_last_chunk_wait_release(pipeline_k, smem_pipe_read, num_younger_groups=1)
        else:
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
        pipeline_v.consumer_release(v_state)
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
            self._p_producer_acquire(smem_copy_params, smem_pipe_read)
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(
                smem_copy_params.smem_thr_copy_P,
                tPrP,
                self._p_stage(smem_copy_params.tPsP, smem_pipe_read),
            )
            if const_expr(self.single_qk_wg):
                self._store_row_stat(smem_copy_params, row_scale, smem_pipe_read.phase)
        if const_expr(not self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, row_scale)
        if const_expr(self.rescale_O_before_gemm):
            scores_scale.store(row_scale.load())
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            # (single_qk_wg: done when P is published, see _p_producer_commit)
            if const_expr(not self.single_qk_wg):
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
            self.cross_wg_p_raw_barrier()

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

    def _p_stage(self, tPsP: cute.Tensor, kv_state) -> cute.Tensor:
        """The sP buffer of the n_block whose K/V pipeline state is kv_state."""
        if const_expr(not self.single_qk_wg):
            return tPsP
        return tPsP[None, None, None, kv_state.phase]

    def _p_idx(self, kv_state) -> dict:
        """sP buffer index (A_idx) for the PV GEMM of the n_block whose K/V state is kv_state."""
        return dict(A_idx=kv_state.phase) if const_expr(self.single_qk_wg) else {}

    @cute.jit
    def cross_wg_p_barrier(self):
        """WAR barrier on sP for hdim > 256: both MMA warpgroups read the whole sP in the PV
        GEMM, so neither may overwrite it with the next P before the other is done reading.
        Not needed with single_qk_wg, which double-buffers sP."""
        if const_expr(self.p_shared_across_wg and not self.single_qk_wg):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.PSync), number_of_threads=self.num_mma_threads
            )

    def _is_qk_wg(self):
        """single_qk_wg: whether this thread is in MMA WG 0 (warp group 1 of the CTA)."""
        return utils.canonical_warp_group_idx(sync=True) == 1

    def _p_producer_acquire(self, smem_copy_params: SimpleNamespace, kv_state):
        """single_qk_wg: wait until WG 1 is done with the P buffer of kv_state's n_block."""
        if const_expr(self.single_qk_wg):
            smem_copy_params.pipeline_p.producer_acquire(
                self._p_pipe_state(kv_state, producer=True)
            )

    def _p_producer_commit(self, smem_copy_params: SimpleNamespace, kv_state):
        """Fence the P and row-scale stores and publish them to WG 1.
        Called after the next QK GEMM is issued, so WG 1's PV GEMM queues behind it."""
        if const_expr(self.single_qk_wg):
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            smem_copy_params.pipeline_p.producer_commit(self._p_pipe_state(kv_state, producer=True))

    def _store_row_stat(self, smem_copy_params: SimpleNamespace, vals: cute.Tensor, slot: Int32):
        """Store per-row values of the softmax rows (replicated across a quad) to sRowStat."""
        for r in range(cute.size(vals)):
            smem_copy_params.sRowStat[smem_copy_params.row_idx_qk[r], slot] = vals[r]

    def _load_row_stat(self, smem_copy_params: SimpleNamespace, slot: Int32) -> cute.Tensor:
        """Load the per-row values stored by _store_row_stat for the rows of this thread's acc_O."""
        vals = cute.make_rmem_tensor(cute.size(smem_copy_params.row_idx_pv), Float32)
        for r in range(cute.size(vals)):
            vals[r] = smem_copy_params.sRowStat[smem_copy_params.row_idx_pv[r], slot]
        return vals

    def _pv_only_block(
        self,
        kv_state,
        pipeline_v,
        mma_pv_fn: Callable,
        acc_O: cute.Tensor,
        softmax: Softmax,
        smem_copy_params: SimpleNamespace,
    ):
        """single_qk_wg, WG 1: O = O * scale + P V for the n_block of kv_state."""
        p_state = self._p_pipe_state(kv_state, producer=False)
        smem_copy_params.pipeline_p.consumer_wait(p_state)
        softmax.rescale_O_if_needed(acc_O, self._load_row_stat(smem_copy_params, kv_state.phase))
        v_state = self._v_chunk_state(kv_state, 1)
        pipeline_v.consumer_wait(v_state, pipeline_v.consumer_try_wait(v_state))
        mma_pv_fn(B_idx=kv_state.index, wg_wait=0, **self._p_idx(kv_state))
        pipeline_v.consumer_release(v_state)
        smem_copy_params.pipeline_p.consumer_release(p_state)

    def _kv_chunk_state(self, kv_state, chunk: int, num_chunks: Optional[int] = None):
        """Pipeline state of hdim chunk `chunk` for the n_block of kv_state.
        Each chunk is used once per n_block, so it shares kv_state's phase."""
        num_chunks = self.k_chunks if num_chunks is None else num_chunks
        return pipeline.PipelineState(num_chunks, kv_state.count, Int32(chunk), kv_state.phase)

    def _make_kv_chunk_copies(
        self, tma_atom_K, mK_cur, sK, hdim: int, paged: bool, num_chunks=2, **copy_kwargs
    ):
        """One TMA copy per hdim chunk of K or V. sK is the per-chunk view."""
        copies = []
        for h in range(num_chunks):
            gK_h = cute.local_tile(
                mK_cur,
                (self.tile_n, hdim // num_chunks),
                (0, h, None) if paged else (None, h),
            )
            if const_expr(self.kv_cluster_m == 1):
                copy_h, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_K, 0, cute.make_layout(1), gK_h, sK, **copy_kwargs
                )
            else:
                # Each CTA loads its 1 / kv_cluster_m slice and multicasts it to the cluster.
                cta_coord, cta_layout, mcast_mask = self._kv_mcast_params()
                copy_h, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_K,
                    cta_coord,
                    cta_layout,
                    gK_h,
                    sK,
                    mcast_mask=mcast_mask,
                    **copy_kwargs,
                )
            copies.append(copy_h)
        return copies

    def _kv_mcast_params(self):
        """Multicast arguments for K / V TMA loads: CTA coordinate, cluster layout and mask."""
        cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(
            cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        )
        mcast_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=0)
        return cluster_coord_mnk[0], cute.make_layout(self.kv_cluster_m), mcast_mask

    def _v_chunk_state(self, kv_state, chunk: int):
        """Pipeline state of the V half read by MMA WG `chunk`."""
        if const_expr(not self.kv_split_hdim):
            return kv_state
        return self._kv_chunk_state(kv_state, chunk, num_chunks=2)

    def _load_kv_chunks(self, tma_load_chunks, pipeline_k, block, producer_state, page_idx=None):
        """TMA each hdim chunk of K or V for tile `block` into its pipeline stage."""
        src_idx = block if const_expr(page_idx is None) else page_idx
        num_chunks = len(tma_load_chunks)
        for h in range(num_chunks):
            k_state = self._kv_chunk_state(producer_state, h, num_chunks=num_chunks)
            pipeline_k.producer_acquire(k_state)
            tma_load_chunks[h](
                src_idx=src_idx, dst_idx=h, tma_bar_ptr=pipeline_k.producer_get_barrier(k_state)
            )

    def _gemm_qk_k_split(self, tiled_mma, shape, tCrA, tCrB, pipeline_k, kv_state):
        """S = Q K^T with one WGMMA group per K chunk.
        Each chunk is released once its group is done.
        The last chunk is released by _k_last_chunk_wait_release."""
        acc = cute.make_rmem_tensor(tiled_mma.partition_shape_C(shape), Float32)
        num_k = cute.size(tCrA.shape[2])
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        mma_atom.set(warpgroup.Field.ACCUMULATE, False)
        for h in range(self.k_chunks):
            k_state = self._kv_chunk_state(kv_state, h)
            pipeline_k.consumer_wait(k_state, pipeline_k.consumer_try_wait(k_state))
            warpgroup.fence()
            for k in range(h * num_k // self.k_chunks, (h + 1) * num_k // self.k_chunks):
                cute.gemm(mma_atom, acc, tCrA[None, None, k], tCrB[None, None, k, 0], acc)
                mma_atom.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            if const_expr(h >= 1):
                # Chunk h is queued behind chunk h - 1: wait for the latter and release it, so the
                # next K tile's chunk h - 1 loads while this QK GEMM still runs.
                warpgroup.wait_group(1)
                pipeline_k.consumer_release(self._kv_chunk_state(kv_state, h - 1))
        return acc

    def _k_last_chunk_wait_release(self, pipeline_k, kv_state, num_younger_groups: int):
        """Release the last K chunk once its QK group is done."""
        warpgroup.wait_group(num_younger_groups)
        pipeline_k.consumer_release(self._kv_chunk_state(kv_state, self.k_chunks - 1))

    def _p_pipe_state(self, kv_state, producer: bool):
        """P pipeline state for the n_block of kv_state.
        The sP buffer is kv_state.phase and is reused every 2 n_blocks.
        Producers start on the opposite barrier phase."""
        p_phase = (kv_state.count >> 1) & 1
        if const_expr(producer):
            p_phase = p_phase ^ 1
        return pipeline.PipelineState(2, kv_state.count, kv_state.phase, p_phase)

    @cute.jit
    def cross_wg_p_raw_barrier(self):
        """RAW barrier on sP for hdim > 256: after the P store and proxy fence, before the PV
        GEMM, so neither warpgroup's wgmma reads sP before the other's store is fenced.
        single_qk_wg publishes sP through pipeline_p instead."""
        if const_expr(self.p_shared_across_wg and not self.single_qk_wg):
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
