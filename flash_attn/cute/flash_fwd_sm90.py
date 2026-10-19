# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM90 (Hopper) forward pass for flash attention, extracted from flash_fwd.py.

from types import SimpleNamespace
from typing import Callable, Literal, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Float16, BFloat16, Float8E4M3FN, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

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
    SingleTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
    WorkTileInfo,
)
from cutlass.cute import FastDivmodDivisor

from flash_attn.cute.flash_fwd import FlashAttentionForwardBase
from flash_attn.cute.utils import AuxData


def _use_intra_wg_overlap_sm90(intra_wg_overlap, native_fp8):
    # Native FP8 runs the non-overlapped mainloop: its two-level PV accumulation keeps a second
    # O accumulator in registers, which leaves no room for the next tile's scores.
    return intra_wg_overlap and not native_fp8


def _convert_layout_acc_frgA_fp8(acc_layout: cute.Layout) -> cute.Layout:
    """Map an SM90 score accumulator to the FP8 WGMMA A-register layout.

    Hopper's FP8 RS WGMMA consumes 32 K values per register group rather than
    the 16 consumed by the f16/bf16 instruction.  This is the CuTeDSL form of
    FA3's ``convert_layout_acc_Aregs`` FP8 specialization.
    """
    assert cute.rank(acc_layout) == 3 and cute.rank(acc_layout.shape[0]) == 3
    assert acc_layout.shape[0][0] == 2 and acc_layout.shape[0][1] == 2
    assert acc_layout.stride[0][0] == 1 and acc_layout.stride[0][1] == 2
    n8 = acc_layout.shape[0][2]
    assert n8 % 4 == 0, "native FP8 PV requires the score tile N to be divisible by 32"
    n8_stride = acc_layout.stride[0][2]
    return cute.make_layout(
        (
            (4, 2, 2),
            acc_layout.shape[1],
            (n8 // 4, acc_layout.shape[2]),
        ),
        stride=(
            (1, n8_stride, 2 * n8_stride),
            acc_layout.stride[1],
            (4 * n8_stride, acc_layout.stride[2]),
        ),
    )


@cute.jit
def _permute_Aregs_fp8(frag: cute.Tensor) -> None:
    """Pack converted E4M3 probabilities into FP8 PV A registers.

    The producer writes V^T with LDSM.T/STSM in the same permuted K order as the score
    accumulator, so each thread only regroups bytes within its own registers.
    """
    assert frag.element_type == Float8E4M3FN
    assert frag.shape[0][0] == 4 and frag.shape[0][1] == 2
    frag_i32 = cute.recast_tensor(frag, Int32)
    for i in cutlass.range(cute.size(frag_i32) // 2, unroll_full=True):
        cols_lo, cols_hi = (frag_i32[2 * i], frag_i32[2 * i + 1])
        frag_i32[2 * i] = cute.arch.prmt(cols_lo, cols_hi, Int32(0x5410))
        frag_i32[2 * i + 1] = cute.arch.prmt(cols_lo, cols_hi, Int32(0x7632))


@dsl_user_op
def _ldmatrix_x4_trans_b16(smem_addr: Int32, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple((Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)))


@dsl_user_op
def _stmatrix_x4_b16(
    smem_addr: Int32, r0: Int32, r1: Int32, r2: Int32, r3: Int32, *, loc=None, ip=None
) -> None:
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int32(r0).ir_value(loc=loc, ip=ip),
            Int32(r1).ir_value(loc=loc, ip=ip),
            Int32(r2).ir_value(loc=loc, ip=ip),
            Int32(r3).ir_value(loc=loc, ip=ip),
        ],
        "stmatrix.sync.aligned.m8n8.x4.shared.b16 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class SingleMBlockVarlenScheduler(SingleTileVarlenScheduler):
    """Varlen scheduler for launches where every batch has exactly one M block (decode).

    The grid is (head, split, batch), so each CTA reads its tile coordinates directly from
    blockIdx instead of searching the cumulative M-block counts in global memory.
    """

    def __init__(
        self, params, head_idx: Int32, batch_idx: Int32, split_idx: Int32, *, loc=None, ip=None
    ):
        self.params = params
        self._head_idx = head_idx
        self._batch_idx = batch_idx
        self._split_idx = split_idx
        self._is_first_block = True
        self._ctx = None
        self._loc = loc
        self._ip = ip

    @staticmethod
    @cute.jit
    def create(params, ctx=None, *, loc=None, ip=None) -> "SingleMBlockVarlenScheduler":
        head_idx, split_idx, batch_idx = cute.arch.block_idx()
        return SingleMBlockVarlenScheduler(params, head_idx, batch_idx, split_idx, loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(params, *, loc=None, ip=None):
        d = params.decoder
        return (d.num_head, d.num_splits, d.num_batch)

    @cute.jit
    def _decode_work_tile(self) -> WorkTileInfo:
        d = self.params.decoder
        split_idx = self._split_idx if const_expr(d.is_split_kv) else Int32(0)
        return WorkTileInfo(
            (Int32(0), Int32(self._head_idx), Int32(self._batch_idx), Int32(split_idx)),
            self._is_first_block,
        )

    def __extract_mlir_values__(self):
        values, self._values_pos = ([], [])
        for obj in (self.params, self._head_idx, self._batch_idx, self._split_idx):
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            (self.params, self._head_idx, self._batch_idx, self._split_idx), self._values_pos
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        scheduler = SingleMBlockVarlenScheduler(*obj_list, loc=self._loc)
        scheduler._is_first_block = self._is_first_block
        return scheduler


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
        native_fp8: bool = False,
        native_fp8_o_dtype=None,
        native_fp8_tma_box_n: Optional[int] = None,
        native_fp8_direct_3d_grid: bool = False,
        native_fp8_ctas_per_sm: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert self.output_quant_key is None, (
            f"Fused quant output not implemented for {type(self).__name__}"
        )
        self.native_fp8 = native_fp8
        self.direct_3d_grid = native_fp8 and native_fp8_direct_3d_grid
        self.ctas_per_sm = native_fp8_ctas_per_sm if native_fp8 else 1
        assert (native_fp8_o_dtype is not None) == self.native_fp8, (
            "native_fp8_o_dtype must be supplied exactly when native_fp8 is enabled"
        )
        if self.native_fp8:
            assert native_fp8_o_dtype in (Float16, BFloat16), (
                "native_fp8_o_dtype must be Float16 or BFloat16"
            )
            self.o_dtype = native_fp8_o_dtype
        self.intra_wg_overlap = _use_intra_wg_overlap_sm90(intra_wg_overlap, self.native_fp8)
        self.mma_pv_is_rs = mma_pv_is_rs
        self.buffer_align_bytes = 1024
        self.use_tma_KV = not paged_kv_non_tma
        self.tma_box_n = native_fp8_tma_box_n
        assert self.tma_box_n is None or (
            self.native_fp8
            and self.use_tma_KV
            and self.tile_n % self.tma_box_n == 0
            and self.tile_n // self.tma_box_n <= cute.arch.WARP_SIZE
        )
        self.kv_boxes_per_tile = 1 if self.tma_box_n is None else self.tile_n // self.tma_box_n
        self.is_split_kv = is_split_kv
        # FP8-KV scale-fold path (SM90: fp16 Q + (fp8-e4m3 -> fp16) paged K/V -> fp16 O).
        self.fp8_kv_dequant = fp8_kv_dequant
        # Native Hopper FP8 is a separate instruction dataflow: Q/K/V and both
        # WGMMA operands remain E4M3.  Never conflate it with fp8_kv_dequant,
        # which deliberately widens paged K/V into the existing f16 mainloop.
        assert not (self.native_fp8 and self.fp8_kv_dequant), (
            "native_fp8 and fp8_kv_dequant are mutually exclusive"
        )
        assert not self.native_fp8 or self.dtype == Float8E4M3FN, (
            "native_fp8 requires E4M3 Q/K/V compute operands"
        )
        assert not self.native_fp8 or (
            self.mma_pv_is_rs
            and not self.Q_in_regs
            and (self.tile_hdim, self.tile_hdimv)
            in ((64, 64), (96, 96), (128, 128), (192, 192), (256, 256), (192, 128))
            and self.tile_n % 32 == 0
        ), (
            "native_fp8 supports RS PV, separate Q/V shared storage, head dimensions "
            "{64,96,128,192,256} equal or (192,128), and tile_n divisible by 32"
        )
        self.kv_dtype = kv_dtype if kv_dtype is not None else self.dtype
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
        # O is written in o_dtype, which is wider than the E4M3 inputs for native FP8.
        sO_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.o_dtype, self.tile_hdimv
            ),
            self.o_dtype,
        )
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
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, self.atom_layout_n, 1),
            tiler_mn=(64, self.tile_n),
        )
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            # FP8 WGMMA requires K-major B, so native FP8 reads a K-major copy of V.
            warpgroup.OperandMajorMode.K if self.native_fp8 else warpgroup.OperandMajorMode.MN,
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

    @cute.jit
    def _scores_to_pv_fragment(self, acc_S: cute.Tensor) -> cute.Tensor:
        """Return the probability view required by the selected PV WGMMA."""
        if const_expr(self.native_fp8):
            return cute.make_tensor(acc_S.iterator, _convert_layout_acc_frgA_fp8(acc_S.layout))
        return layout_utils.reshape_acc_to_frgA(acc_S)

    def _setup_native_fp8_output_copy(self) -> None:
        """Retype the non-TMA epilogue copy for native FP8's 16-bit output."""
        if not self.native_fp8:
            return
        _, _, _, sO_layout_atom, _ = self._get_smem_layout_atom()
        copy_elems = 128 // self.o_dtype.width
        tO_shape_dim_1 = sO_layout_atom.outer.shape[1] // copy_elems
        tO_layout = cute.make_ordered_layout(
            (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),
            order=(1, 0),
        )
        assert self.tile_m % tO_layout.shape[0] == 0
        output_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.o_dtype, num_bits_per_copy=128
        )
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            output_copy_atom, tO_layout, cute.make_layout((1, copy_elems))
        )

    @cute.jit
    def _transpose_v_ldsm_native_fp8(
        self, sVLoad: cute.Tensor, sV: cute.Tensor, tidx: Int32, stage_idx: Int32
    ) -> None:
        """Producer-side V transpose (sVLoad row-major -> sV K-major) with LDSM.T / STSM.

        Runs on the producer warpgroup's transpose warps (the last v_transpose_warps warps),
        so the MMA warpgroups never wait on it. Each warp moves 16 x 32 units of V, issuing
        two LDSM.T before the matching STSMs to keep two shared-memory loads in flight.
        """
        assert self.tile_n % 32 == 0 and self.tile_hdimv % 32 == 0
        num_warps = const_expr(self.v_transpose_warps)
        units_d = const_expr(self.tile_hdimv // 32)
        num_units = const_expr(self.tile_n // 16 * units_d)
        first_warp = const_expr(self.num_threads_per_warp_group // cute.arch.WARP_SIZE - num_warps)
        warp = cute.arch.make_warp_uniform(tidx // cute.arch.WARP_SIZE) - first_warp
        lane = cute.arch.lane_idx()
        mat = lane // 8
        row = lane % 8
        src_lane = cute.crd2idx((mat % 2 * 8 + row, mat // 2 * 16, 0), sVLoad.layout)
        dst_lane = cute.crd2idx((0, mat // 2 * 16 + 2 * row + mat % 2, 0), sV.layout)
        src_lane = src_lane + cute.crd2idx((0, 0, stage_idx), sVLoad.layout)
        dst_lane = dst_lane + cute.crd2idx((0, 0, stage_idx), sV.layout)
        src_base = Int32(sVLoad.iterator.toint())
        dst_base = Int32(sV.iterator.toint())
        swz_src = sVLoad.iterator.type.swizzle_type
        swz_dst = sV.iterator.type.swizzle_type

        def unit_offsets(unit):
            n_chunk, d_chunk = (unit // units_d, unit % units_d)
            src = cute.crd2idx((n_chunk * 16, d_chunk * 32, 0), sVLoad.layout)
            dst = cute.crd2idx((n_chunk * 16, d_chunk * 32, 0), sV.layout)
            return (src, dst)

        for i in cutlass.range_constexpr(cute.ceil_div(num_units, 2 * num_warps)):
            for w in cutlass.range_constexpr(num_warps):
                unit_a = 2 * i * num_warps + w
                unit_b = unit_a + num_warps
                if const_expr(unit_a < num_units):
                    src_a, dst_a = unit_offsets(unit_a)
                    src_b, dst_b = unit_offsets(min(unit_b, num_units - 1))
                    if warp == w:
                        r_a = _ldmatrix_x4_trans_b16(
                            src_base + self._swizzle_offset(src_lane + src_a, swz_src)
                        )
                        if const_expr(unit_b < num_units):
                            r_b = _ldmatrix_x4_trans_b16(
                                src_base + self._swizzle_offset(src_lane + src_b, swz_src)
                            )
                            self._stsm_vt_native_fp8(
                                dst_base + self._swizzle_offset(dst_lane + dst_a, swz_dst), *r_a
                            )
                            self._stsm_vt_native_fp8(
                                dst_base + self._swizzle_offset(dst_lane + dst_b, swz_dst), *r_b
                            )
                        else:
                            self._stsm_vt_native_fp8(
                                dst_base + self._swizzle_offset(dst_lane + dst_a, swz_dst), *r_a
                            )

    @staticmethod
    def _swizzle_offset(offset: Int32, swizzle_type) -> Int32:
        return copy_utils.swizzle_int(
            Int32(offset), swizzle_type.num_bits, swizzle_type.num_base, swizzle_type.num_shift
        )

    @cute.jit
    def _stsm_vt_native_fp8(
        self, dst_addr: Int32, r0: Int32, r1: Int32, r2: Int32, r3: Int32
    ) -> None:
        # LDSM.T transposed 16-bit pairs of FP8 values; split each pair into its even and odd
        # bytes so STSM writes whole K-major rows of V^T.
        x0 = cute.arch.prmt(r0, r1, Int32(0x6420))
        y0 = cute.arch.prmt(r0, r1, Int32(0x7531))
        x1 = cute.arch.prmt(r2, r3, Int32(0x6420))
        y1 = cute.arch.prmt(r2, r3, Int32(0x7531))
        _stmatrix_x4_b16(dst_addr, x0, y0, x1, y1)

    def _get_shared_storage_cls(self):
        # sO aliases sQ after the mainloop. Native FP8 needs extra sQ room to hold a
        # 16-bit output tile of the same shape as the E4M3 Q tile.
        q_storage_elems = cute.cosize(self.sQ_layout)
        if const_expr(self.native_fp8):
            q_storage_elems = max(
                q_storage_elems,
                (cute.cosize(self.sO_layout) * self.o_dtype.width + self.dtype.width - 1)
                // self.dtype.width,
            )
        sQ_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, q_storage_elems], self.buffer_align_bytes
        ]
        sK_struct, sV_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)], self.buffer_align_bytes
            ]
            for layout in (self.sK_layout, self.sV_layout)
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
                sV: sV_struct
                sQ: sQ_struct
                sK: sK_struct
                sStage: sStage_struct

            return SharedStorageQKVDequant

        if const_expr(self.native_fp8):
            # TMA / cp.async land V row-major in sVLoad; sV holds the K-major copy for PV.
            sVLoad_struct = cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(self.sV_load_layout)],
                self.buffer_align_bytes,
            ]
            mbar_ptr_Vt_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

            @cute.struct
            class SharedStorageQKVNativeFp8:
                mbar_ptr_Q: mbar_ptr_Q_struct
                mbar_ptr_K: mbar_ptr_K_struct
                mbar_ptr_V: mbar_ptr_V_struct
                mbar_ptr_Vt: mbar_ptr_Vt_struct
                sV: sV_struct
                sVLoad: sVLoad_struct
                sQ: sQ_struct
                sK: sK_struct
                sP: sP_struct

            return SharedStorageQKVNativeFp8

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
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout
        (except the fp8-KV scale-fold path: fp16 Q + fp8 e4m3 K/V -> fp16 O, and native
        FP8: e4m3 Q/K/V -> fp16/bf16 O):
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        """
        if const_expr(self.native_fp8):
            assert all(t.element_type == Float8E4M3FN for t in (mQ, mK, mV)), (
                "native_fp8 requires E4M3 Q, K, and V"
            )
            assert mO.element_type == (Float32 if self.is_split_kv else self.o_dtype), (
                "native_fp8 writes the configured output dtype, or FP32 SplitKV partials"
            )
            assert mLSE is None or mLSE.element_type == Float32
            assert all(
                t is None or t.element_type == Int32
                for t in (mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK)
            )
        elif const_expr(not self.fp8_kv_dequant):
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
        assert self.num_wg_mma in [1, 2, 3]
        self.num_threads = self.num_threads_per_warp_group * (self.num_wg_mma + 1)
        self.num_producer_threads = 32
        self.num_Q_load_threads = self.num_threads_per_warp_group  # If not TMA_Q
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs, self.num_producer_regs = {1: (256, 56), 2: (240, 24), 3: (160, 32)}[
            self.num_wg_mma
        ]
        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)

        assert not (self.native_fp8 and self.use_block_sparsity), (
            "SM90 native FP8 does not support block sparsity"
        )
        # Native FP8 transposes V (row-major sVLoad -> K-major sV) in the producer warpgroup, as
        # FA3's Hopper FP8 kernel does (Transpose_V), off the MMA warpgroups' critical path.
        self.transpose_v_in_producer = self.native_fp8
        self.dedicated_kv_load_warp = self.transpose_v_in_producer and self.use_tma_KV
        self.v_transpose_warps = 3 if self.dedicated_kv_load_warp else 4
        self.use_scheduler_barrier = (
            (self.num_wg_mma >= 2 and self.tile_hdim <= 128)
            if const_expr(self.intra_wg_overlap)
            else (self.num_wg_mma == 2)
        )
        self.use_tma_Q = self.arch >= Arch.sm_90 and not (
            self.pack_gqa and self.tile_m % self.qhead_per_kvhead != 0
        )
        self.use_tma_O = self.use_tma_Q and not self.is_split_kv
        # Producer needs more registers when doing cp.async Q or KV loads
        if const_expr(self.num_wg_mma == 2 and (not self.use_tma_Q or not self.use_tma_KV)):
            self.num_mma_regs, self.num_producer_regs = 224, 40
        if const_expr(self.transpose_v_in_producer and self.num_wg_mma == 2):
            self.num_producer_regs = 40
            self.num_mma_regs = 232 if const_expr(self.use_tma_Q and self.use_tma_KV) else 224
        if const_expr(self.transpose_v_in_producer and self.num_wg_mma == 3):
            self.num_mma_regs, self.num_producer_regs = (152, 56)
        if const_expr(self.ctas_per_sm == 2):
            assert self.num_wg_mma == 1, "two CTAs per SM need a single MMA warpgroup"
            self.num_mma_regs, self.num_producer_regs = (200, 56)
        self.rescale_O_before_gemm = self.tile_hdimv > 128 and self.intra_wg_overlap
        # hdim > 256: two MMA warpgroups each read the whole sP tile in the PV GEMM.
        self.p_shared_across_wg = self.atom_layout_n == 2
        self._setup_attributes()
        self._setup_native_fp8_output_copy()
        # TODO: we prob don't need most of what's in _setup_attributes
        # Per-tensor smem dtypes. FP8-KV is the one special case: sK/sV hold the fp16
        # dequant target (self.dtype), not mK/mV's fp8 source.
        sK_dtype = self.dtype if const_expr(self.fp8_kv_dequant) else mK.element_type
        sV_dtype = self.dtype if const_expr(self.fp8_kv_dequant) else mV.element_type
        self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
            sm90_utils.make_smem_layout(dtype, layout_enum, shape, stage)
            for dtype, layout_enum, shape, stage in [
                # sQ holds the (possibly bf16) Q tensor as TMA'd; the fp16 compute view
                # is an in-place recast (both are 2 bytes) -- see mma().
                (self.q_dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_hdim), None),
                (sK_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdim), self.num_stages),
                # Native FP8 keeps the PV operand K-major; V is loaded into sVLoad first.
                (
                    sV_dtype,
                    LayoutEnum.COL_MAJOR if self.native_fp8 else LayoutEnum.ROW_MAJOR,
                    (self.tile_n, self.tile_hdimv),
                    self.num_stages,
                ),
                # sO uses the output dtype (o_dtype); for splitkv o_dtype == compute dtype
                # (the fp32 partials are written from acc_O directly, not via sO).
                (self.o_dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_hdimv), None),
            ]
        ]
        # Row-major V tile the producer loads into; distinct from sV only for native FP8.
        self.sV_load_layout = (
            sm90_utils.make_smem_layout(
                sV_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdimv), self.num_stages
            )
            if const_expr(self.native_fp8)
            else self.sV_layout
        )
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
                ("V", mV, self.sV_load_layout),
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
        if const_expr(self.use_tma_KV):
            # FP8-KV: K and V from GMEM land in sStage for the consumer to dequant(instead of sK/sV)
            sK_tma_box = cute.select(
                self.sStage_layout if const_expr(self.fp8_kv_dequant) else self.sK_layout,
                mode=[0, 1],
            )
            sV_tma_box = cute.select(
                self.sStage_layout if const_expr(self.fp8_kv_dequant) else self.sV_load_layout,
                mode=[0, 1],
            )
            tma_box_n = self.tile_n
            if const_expr(self.tma_box_n is not None):
                tma_box_n = self.tma_box_n
                sK_tma_box = cute.composition(sK_tma_box, (tma_box_n, self.tile_hdim))
                sV_tma_box = cute.composition(sV_tma_box, (tma_box_n, self.tile_hdimv))
            tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mK,
                sK_tma_box,
                (tma_box_n, self.tile_hdim),
                1,  # No mcast for now
            )
            tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mV,
                sV_tma_box,
                (tma_box_n, self.tile_hdimv),
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
            TileScheduler = (
                SingleMBlockVarlenScheduler
                if const_expr(self.direct_3d_grid and mSeqUsedQ is None)
                else SingleTileVarlenScheduler
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
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
            cu_total_m_blocks_ptr=mCuTotalMBlocks,
            cu_total_splits_m_blocks_ptr=mCuTotalSplitsMBlocks,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
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
            self.sV_load_layout,
            self.sO_layout,
            self.sP_layout,
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
            min_blocks_per_mp=self.ctas_per_sm,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
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
        sV_load_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
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
        # Decode with a direct (head, split, batch) grid: read the first tile's seqlens before
        # the barrier setup so their global-memory latency overlaps it.
        self._seqlen_first = None
        if const_expr(self.native_fp8 and TileScheduler is SingleMBlockVarlenScheduler):
            _, _, first_batch, _ = (
                TileScheduler.create(tile_sched_params).initial_work_tile_info().tile_idx
            )
            self._seqlen_first = SeqlenInfoCls(first_batch)
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
        pipeline_v = None
        if const_expr(self.fp8_kv_dequant):
            pipeline_k = None
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
            if const_expr(not self.transpose_v_in_producer):
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
            if const_expr(not self.transpose_v_in_producer):
                pipeline_v = pipeline_custom.PipelineCpAsync.create(
                    barrier_storage=storage.mbar_ptr_V.data_ptr(),
                    num_stages=self.num_stages,
                    producer_group=load_threads,
                    consumer_group=mma_warps,
                    defer_sync=True,
                    elect_one_release=True,
                    syncwarp_before_release=False,
                )
        pipeline_vt = None
        if const_expr(self.transpose_v_in_producer):
            transpose_warps = ThreadCooperativeGroup(self.v_transpose_warps)
            if const_expr(self.use_tma_KV):
                pipeline_vt = pipeline_custom.PipelineTmaAsync.create(
                    barrier_storage=storage.mbar_ptr_Vt.data_ptr(),
                    num_stages=self.num_stages,
                    producer_group=tma_warp,
                    consumer_group=transpose_warps,
                    tx_count=self.tma_copy_bytes["V"],
                    defer_sync=True,
                )
            else:
                pipeline_vt = pipeline_custom.PipelineCpAsync.create(
                    barrier_storage=storage.mbar_ptr_Vt.data_ptr(),
                    num_stages=self.num_stages,
                    producer_group=load_threads,
                    consumer_group=transpose_warps,
                    defer_sync=True,
                    elect_one_release=True,
                    syncwarp_before_release=False,
                )
            pipeline_v = pipeline_custom.PipelineAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=transpose_warps,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_commit=True,
                elect_one_release=True,
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
        # Native FP8: the producer fills the row-major sVLoad, and the consumer copies it
        # into the K-major sV read by the FP8 PV WGMMA. Otherwise sVLoad is sV.
        sVLoad = sV
        if const_expr(self.native_fp8):
            sVLoad = storage.sVLoad.get_tensor(sV_load_layout.outer, swizzle=sV_load_layout.inner)
        # Transpose view of V to tensor with layout (head_dim_v, tile_n) for tiled mma
        sVt = layout_utils.transpose_view(sV)
        sP = None
        if const_expr(sP_layout is not None):
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        # fp8 staging buffer (TMA dst, dequant src).
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
            if const_expr(self.transpose_v_in_producer):
                self.load_native_fp8(
                    mQ,
                    mK,
                    mV,
                    sQ,
                    sK,
                    sVLoad,
                    sV,
                    tma_atom_Q,
                    tma_atom_K,
                    tma_atom_V,
                    pipeline_k,
                    pipeline_vt,
                    pipeline_v,
                    pipeline_q,
                    gmem_tiled_copy_Q,
                    mPageTable,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    num_splits,
                )
            else:
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
        mK: cute.Tensor,
        mV: cute.Tensor,
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

        if is_load_warp:
            q_producer_phase = Int32(1)
            kv_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                # if work_tile.is_valid_tile:
                m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
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
    def load_native_fp8(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVLoad: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        pipeline_k: pipeline.PipelineAsync,
        pipeline_vt: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        mPageTable: Optional[cute.Tensor],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        num_splits: Int32 = Int32(1),
    ):
        """Native-FP8 producer warpgroup: loads Q/K/V and transposes V off the MMA critical path.

        V lands row-major in sVLoad (pipeline_vt); the transpose warps write the K-major PV
        operand into sV and hand it to the MMA warpgroups through pipeline_v (FA3's
        Transpose_V scheme). With TMA K/V, warp 0 only issues loads while warps 1-3 transpose.
        Paged K/V whose page size differs from tile_n is loaded as several TMA boxes per tile.
        """
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tidx, _, _ = cute.arch.thread_idx()
        is_kv_load_warp = warp_idx_in_wg == 0 or const_expr(not self.use_tma_KV)
        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_stages
        )
        transpose_v = partial(
            self._transpose_v_stage_native_fp8, pipeline_vt, pipeline_v, sVLoad, sV, tidx
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = (
                SeqlenInfoCls(batch_idx)
                if const_expr(self._seqlen_first is None)
                else self._seqlen_first
            )
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            head_idx_kv = (
                head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
            )
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
            paged_kv_manager = None
            tma_load_K_fn = None
            tma_load_V_fn = None
            load_K = None
            load_V = None
            if const_expr(self.use_tma_KV):
                if const_expr(mPageTable is not None):
                    mK_cur = mK[None, None, head_idx_kv, None]
                    mV_cur = mV[None, None, head_idx_kv, None]
                    if const_expr(self.tma_box_n is not None):
                        ps = const_expr(self.tma_box_n)
                        gK = cute.zipped_divide(mK_cur, (ps, self.tile_hdim))
                        gV = cute.zipped_divide(mV_cur, (ps, self.tile_hdimv))
                    else:
                        gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (0, 0, None))
                        gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (0, 0, None))
                else:
                    mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
                    mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]
                    gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
                    gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (None, 0))
                if const_expr(self.tma_box_n is not None):
                    ps = const_expr(self.tma_box_n)
                    load_K, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_K,
                        0,
                        cute.make_layout(1),
                        gK,
                        cute.zipped_divide(sK, (ps, self.tile_hdim)),
                    )
                    load_V, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_V,
                        0,
                        cute.make_layout(1),
                        gV,
                        cute.zipped_divide(sVLoad, (ps, self.tile_hdimv)),
                    )
                else:
                    tma_load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_K, 0, cute.make_layout(1), gK, sK
                    )
                    tma_load_K_fn = copy_utils.tma_producer_copy_fn(tma_load_K_fn, pipeline_k)
                    tma_load_V_fn, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_V, 0, cute.make_layout(1), gV, sVLoad
                    )
                    tma_load_V_fn = copy_utils.tma_producer_copy_fn(tma_load_V_fn, pipeline_vt)
            else:
                paged_kv_manager = PagedKVManager.create(
                    mPageTable,
                    mK,
                    mV,
                    FastDivmodDivisor(mK.shape[0]),
                    batch_idx,
                    head_idx_kv,
                    tidx,
                    seqlen.seqlen_k,
                    0,
                    self.tile_n,
                    self.tile_hdim,
                    self.tile_hdimv,
                    self.num_threads_per_warp_group,
                    mK.element_type,
                    arch=self.arch.major * 10 + self.arch.minor,
                )
            if const_expr(self.tma_box_n is None):
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
                    sVLoad,
                    pipeline_kv=pipeline_vt,
                    K_or_V="V",
                )
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
            if const_expr(self._mDynamicCausal is not None):
                psc_producer = self._mDynamicCausal[batch_idx]
                if not psc_producer:
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
            n_block = (
                n_block_max - 1 if const_expr(self.use_tma_KV) else cutlass.max(n_block_max - 1, 0)
            )
            num_blocks = cutlass.max(n_block_max - n_block_min, 1)
            issue_kv = partial(
                self._issue_kv_block_native_fp8,
                load_K,
                load_V,
                pipeline_k,
                pipeline_vt,
                mPageTable=mPageTable,
                batch_idx=batch_idx,
                seqlen_k=seqlen.seqlen_k,
                page_size=mK.shape[0] if const_expr(mPageTable is not None) else None,
            )
            if const_expr(self.dedicated_kv_load_warp):
                load_Q_cpasync = None
                if const_expr(not self.use_tma_Q):
                    load_Q_cpasync = partial(
                        self._load_q_cpasync_native_fp8,
                        pipeline_q,
                        q_producer_phase,
                        pack_gqa,
                        mQ_cur,
                        sQ,
                        gmem_tiled_copy_Q,
                        tidx,
                        m_block,
                        seqlen.seqlen_q,
                    )
                kv_producer_state = self._load_tma_kv_native_fp8(
                    issue_kv,
                    transpose_v,
                    load_Q,
                    pipeline_q,
                    q_producer_phase,
                    mPageTable,
                    batch_idx,
                    n_block,
                    num_blocks,
                    kv_producer_state,
                    load_Q_cpasync,
                )
                q_producer_phase ^= 1
            else:
                kv_producer_state = self._load_kv_transpose_native_fp8(
                    issue_kv,
                    transpose_v,
                    load_Q,
                    pack_gqa,
                    pipeline_q,
                    pipeline_vt,
                    q_producer_phase,
                    paged_kv_manager,
                    mPageTable,
                    gmem_tiled_copy_Q,
                    mQ_cur,
                    sQ,
                    tidx,
                    m_block,
                    seqlen,
                    batch_idx,
                    n_block,
                    num_blocks,
                    kv_producer_state,
                    is_kv_load_warp,
                )
                q_producer_phase ^= 1
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def _load_kv_transpose_native_fp8(
        self,
        issue_kv: Callable,
        transpose_v: Callable,
        load_Q: Optional[Callable],
        pack_gqa: Optional[PackGQA],
        pipeline_q: pipeline.PipelineAsync,
        pipeline_vt: pipeline.PipelineAsync,
        q_producer_phase: Int32,
        paged_kv_manager: Optional[PagedKVManager],
        mPageTable: Optional[cute.Tensor],
        gmem_tiled_copy_Q: cute.TiledCopy,
        mQ_cur: cute.Tensor,
        sQ: cute.Tensor,
        tidx: Int32,
        m_block: Int32,
        seqlen: SeqlenInfoQK,
        batch_idx: Int32,
        n_block: Int32,
        num_blocks: Int32,
        kv_producer_state: pipeline.PipelineState,
        is_kv_load_warp,
    ):
        """Producer loop when every producer warp both loads (cp.async) and transposes.

        Transposes lag the loads by up to num_stages - 1 blocks so a transpose waiting on V
        does not hold back the next block's load.
        """
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        page_idx = None
        if const_expr(not self.use_tma_KV):
            paged_kv_manager.load_page_table(n_block)
        elif const_expr(mPageTable is not None):
            page_idx = mPageTable[batch_idx, n_block]
        if is_kv_load_warp:
            issue_kv(n_block, kv_producer_state, page_idx)
        if const_expr(not self.use_tma_KV):
            paged_kv_manager.load_page_table(cutlass.max(n_block - 1, 0))
        elif const_expr(mPageTable is not None):
            page_idx = mPageTable[batch_idx, cutlass.max(n_block - 1, 0)]
        if const_expr(self.use_tma_Q):
            if warp_idx_in_wg == 0:
                pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
        else:
            pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
            pack_gqa.load_Q(mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q)
            cute.arch.cp_async_commit_group()
            pipeline_q.producer_commit_w_index(0)
        q_producer_phase ^= 1
        transpose_lag = const_expr(self.num_stages - 1)
        v_transpose_state = kv_producer_state.clone()
        num_pending = Int32(1)
        for i in cutlass.range(num_blocks - 1, unroll=1):
            kv_producer_state.advance()
            n_block -= 1
            if is_kv_load_warp:
                issue_kv(n_block, kv_producer_state, page_idx)
            if const_expr(not self.use_tma_KV):
                paged_kv_manager.load_page_table(cutlass.max(n_block - 1, 0))
            elif const_expr(mPageTable is not None):
                page_idx = mPageTable[batch_idx, cutlass.max(n_block - 1, 0)]
            num_pending += 1
            transpose_ready = num_pending > transpose_lag
            if const_expr(self.num_stages > 2):
                transpose_ready = transpose_ready or cute.arch.vote_all_sync(
                    pipeline_vt.sync_object_full.test_wait(
                        v_transpose_state.index, v_transpose_state.phase ^ 1
                    )
                )
            if transpose_ready:
                transpose_v(v_transpose_state)
                v_transpose_state.advance()
                num_pending -= 1
        for i in cutlass.range(num_pending, unroll=1):
            transpose_v(v_transpose_state)
            v_transpose_state.advance()
        kv_producer_state.advance()
        return kv_producer_state

    @cute.jit
    def _load_tma_kv_native_fp8(
        self,
        issue_kv: Callable,
        transpose_v: Callable,
        load_Q: Optional[Callable],
        pipeline_q: pipeline.PipelineAsync,
        q_producer_phase: Int32,
        mPageTable: Optional[cute.Tensor],
        batch_idx: Int32,
        n_block: Int32,
        num_blocks: Int32,
        kv_producer_state: pipeline.PipelineState,
        load_Q_cpasync: Optional[Callable] = None,
    ):
        """Producer loop with a dedicated TMA warp: warp 0 issues Q/K/V loads as far ahead as
        the stages allow, warps 1-3 transpose each block's V in order."""
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        if warp_idx_in_wg == 0:
            if const_expr(self.use_tma_Q):
                pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
            page_idx = None
            if const_expr(mPageTable is not None):
                page_idx = mPageTable[batch_idx, n_block]
            if const_expr(load_Q_cpasync is not None):
                load_Q_cpasync()
            issue_kv(n_block, kv_producer_state, page_idx)
            if const_expr(mPageTable is not None):
                page_idx = mPageTable[batch_idx, cutlass.max(n_block - 1, 0)]
            for i in cutlass.range(num_blocks - 1, unroll=1):
                kv_producer_state.advance()
                n_block -= 1
                issue_kv(n_block, kv_producer_state, page_idx)
                if const_expr(mPageTable is not None):
                    page_idx = mPageTable[batch_idx, cutlass.max(n_block - 1, 0)]
            kv_producer_state.advance()
        else:
            if const_expr(load_Q_cpasync is not None):
                load_Q_cpasync()
            for i in cutlass.range(num_blocks, unroll=1):
                transpose_v(kv_producer_state)
                kv_producer_state.advance()
        return kv_producer_state

    @cute.jit
    def _load_q_cpasync_native_fp8(
        self,
        pipeline_q: pipeline.PipelineAsync,
        q_producer_phase: Int32,
        pack_gqa: PackGQA,
        mQ_cur: cute.Tensor,
        sQ: cute.Tensor,
        gmem_tiled_copy_Q: cute.TiledCopy,
        tidx: Int32,
        m_block: Int32,
        seqlen_q: Int32,
    ):
        pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
        pack_gqa.load_Q(mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen_q)
        cute.arch.cp_async_commit_group()
        pipeline_q.producer_commit_w_index(0)

    @cute.jit
    def _issue_kv_block_native_fp8(
        self,
        load_K: Callable,
        load_V: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_vt: pipeline.PipelineAsync,
        n_block: Int32,
        producer_state: pipeline.PipelineState,
        page_idx: Optional[Int32],
        mPageTable: Optional[cute.Tensor] = None,
        batch_idx: Int32 = Int32(0),
        seqlen_k: Int32 = Int32(0),
        page_size: Optional[Int32] = None,
    ):
        """Issue one K/V block. With tma_box_n, the tile is split into page-aligned TMA boxes;
        lane b looks up the page of box b and the boxes are issued from lane-shuffled indices."""
        if const_expr(self.tma_box_n is not None):
            nbox = const_expr(self.kv_boxes_per_tile)
            lane = cute.arch.lane_idx()
            key0 = n_block * self.tile_n + lane * self.tma_box_n
            box_valid = lane < nbox and key0 >= 0 and key0 < seqlen_k
            my_page = Int32(-1)
            my_row_box = Int32(0)
            if box_valid:
                my_page = mPageTable[batch_idx, key0 // page_size]
                my_row_box = key0 % page_size // self.tma_box_n
            pipeline_k.producer_acquire(producer_state)
            for box in cutlass.range_constexpr(nbox):
                load_K(
                    src_idx=(
                        cute.arch.shuffle_sync(my_row_box, box),
                        0,
                        cute.arch.shuffle_sync(my_page, box),
                    ),
                    dst_idx=box + nbox * producer_state.index,
                    tma_bar_ptr=pipeline_k.producer_get_barrier(producer_state),
                )
            pipeline_vt.producer_acquire(producer_state)
            for box in cutlass.range_constexpr(nbox):
                load_V(
                    src_idx=(
                        cute.arch.shuffle_sync(my_row_box, box),
                        0,
                        cute.arch.shuffle_sync(my_page, box),
                    ),
                    dst_idx=box + nbox * producer_state.index,
                    tma_bar_ptr=pipeline_vt.producer_get_barrier(producer_state),
                )
        else:
            pipeline_k.producer_acquire(producer_state)
            load_K(block=n_block, producer_state=producer_state, page_idx=page_idx)
            pipeline_vt.producer_acquire(producer_state)
            load_V(block=n_block, producer_state=producer_state, page_idx=page_idx)

    @cute.jit
    def _transpose_v_stage_native_fp8(
        self,
        pipeline_vt: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        sVLoad: cute.Tensor,
        sV: cute.Tensor,
        tidx: Int32,
        producer_state: pipeline.PipelineState,
    ):
        """Wait for one loaded V stage, transpose it into sV, publish it to the MMA warps."""
        pipeline_vt.consumer_wait_w_index_phase(producer_state.index, producer_state.phase ^ 1)
        pipeline_v.producer_acquire(producer_state)
        self._transpose_v_ldsm_native_fp8(sVLoad, sV, tidx, producer_state.index)
        cute.arch.fence_view_async_shared()
        pipeline_v.producer_commit(producer_state)
        pipeline_vt.consumer_release_w_index(producer_state.index)

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
        blocksparse_tensors: Optional[BlockSparseTensors],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        num_splits: Int32 = Int32(1),
        # FP8-KV cast inputs (None on the non-fp8 path): sV (fp16 V write target),
        # sStage, pipeline_stage.
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
        acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
            wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), sP, sVt
        )
        mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)
        # Native FP8: FP8 WGMMA does not accumulate in full fp32, so each PV GEMM accumulates
        # one KV tile from zero and the result is added to the fp32 acc_O (two-level
        # accumulation, as in FA3's FP8 forward).
        acc_O_pv = None
        if const_expr(self.native_fp8 and self.tile_hdimv <= 192):
            acc_O_pv = cute.make_rmem_tensor_like(acc_O, Float32)
            mma_pv_fn = partial(
                self._mma_pv_tile_native_fp8,
                partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O_pv, tOrP, tOrVt),
            )
        elif const_expr(self.native_fp8):
            # hdim_v 256: O and a second full-size O do not fit in registers, so the PV GEMM
            # runs as two hdim_v halves through one half-size tile accumulator.
            half = self.tile_hdimv // 2
            tiled_mma_pv_half = sm90_utils_basic.make_trivial_tiled_mma(
                self.dtype,
                self.dtype,
                warpgroup.OperandMajorMode.K,
                warpgroup.OperandMajorMode.K,
                Float32,
                atom_layout_mnk=(self.tile_m // 64, self.atom_layout_n, 1),
                tiler_mn=(64, half),
                a_source=warpgroup.OperandSource.RMEM,
            )
            wg_mma_pv_half = tiled_mma_pv_half.get_slice(warp_group_thread_layout(warp_group_idx))
            acc_O_half = cute.make_rmem_tensor(
                wg_mma_pv_half.partition_shape_C((self.tile_m, half)), Float32
            )
            tOrVt_halves = [
                wg_mma_pv_half.make_fragment_B(
                    wg_mma_pv_half.partition_B(
                        cute.local_tile(sVt, (half, self.tile_n), (h, 0, None))
                    )
                )
                for h in range(2)
            ]
            mma_pv_fn = partial(
                self._mma_pv_halves_native_fp8,
                tiled_mma_pv_half,
                acc_O,
                acc_O_half,
                tOrP,
                tOrVt_halves,
            )

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
        work_tile = tile_scheduler.initial_work_tile_info()
        num_softmax_rows = const_expr(acc_O.shape[0][0] * acc_O.shape[1])
        # Native FP8 biases P by 2**8 so its E4M3 cast keeps mantissa (see Softmax).
        native_fp8_max_offset = const_expr(8 if self.native_fp8 else 0)
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=num_softmax_rows,
            softmax_scale=softmax_scale,
            max_offset=native_fp8_max_offset,
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
            **(dict(acc_O_pv=acc_O_pv) if const_expr(self.native_fp8) else {}),
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
            seqlen = (
                SeqlenInfoCls(batch_idx)
                if const_expr(self._seqlen_first is None)
                else self._seqlen_first
            )

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
            elif const_expr(self.native_fp8 and descale_tensors is not None):
                # Native FP8 descales may be scalars or (B, Hkv), possibly broadcast.
                head_idx_kv = (
                    head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
                )
                qk_descale, v_descale_tile = self._load_native_fp8_descales(
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
                    max_offset=native_fp8_max_offset,
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
            if const_expr(self.native_fp8):
                acc_O.fill(0.0)

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
                interior_mask_fn = partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=False)
                # Native FP8 without mask_mod: blocks between the boundary blocks need no mask.
                if const_expr(
                    self.native_fp8 and self.mask_mod is None and self._mDynamicCausal is None
                ):
                    interior_mask_fn = None
                # if cute.arch.thread_idx()[0] == 128: cute.printf("n_block_min_before_local_mask = {}, n_block_min = {}", n_block_min_before_local_mask, n_block_min)
                for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=n_block_max - 1 - n_tile,
                        seqlen=seqlen,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=interior_mask_fn,
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

            # Override empty splits so combine kernel gives zero weight
            if const_expr(self.is_split_kv):
                if n_block_min >= n_block_max_orig:
                    acc_O.fill(Float32(0.0))
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

    def _mma_pv_tile_native_fp8(
        self, mma_pv_tile_fn: Callable, B_idx: Int32, zero_init=False, wg_wait: int = -1
    ):
        """PV GEMM of one KV tile into the zeroed tile accumulator (two-level accumulation)."""
        mma_pv_tile_fn(B_idx=B_idx, zero_init=True, wg_wait=wg_wait)

    def _mma_pv_halves_native_fp8(
        self,
        tiled_mma_half: cute.TiledMma,
        acc_O: cute.Tensor,
        acc_O_half: cute.Tensor,
        tOrP: cute.Tensor,
        tOrVt_halves: list,
        B_idx: Int32,
        zero_init=False,
        wg_wait: int = -1,
    ):
        """PV GEMM of one KV tile in two hdim_v halves: each half accumulates from zero in the
        tensor cores and is added to its half of the fp32 acc_O (two-level accumulation)."""
        n = cute.size(acc_O_half)
        for h in range(2):
            sm90_utils.gemm_w_idx(
                tiled_mma_half, acc_O_half, tOrP, tOrVt_halves[h], True, B_idx=B_idx, wg_wait=0
            )
            for i in range(n):
                acc_O[h * n + i] = acc_O[h * n + i] + acc_O_half[i]

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

        tOrP_acc = self._scores_to_pv_fragment(acc_S)
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
    def _load_native_fp8_descales(self, descale_tensors, batch_idx: Int32, kv_head_idx: Int32):
        """Load native FP8 QK score and V output descales from scalar or (B, Hkv) storage."""
        qk_descale = Float32(1.0)
        v_descale = Float32(1.0)
        if const_expr(descale_tensors is not None):
            if const_expr(descale_tensors.q_descale is not None):
                q_descale = descale_tensors.q_descale
                q_scale = (
                    q_descale[()]
                    if const_expr(len(q_descale.shape) == 0)
                    else q_descale[batch_idx, kv_head_idx]
                )
                qk_descale = qk_descale * Float32(q_scale)
            if const_expr(descale_tensors.k_descale is not None):
                k_descale = descale_tensors.k_descale
                k_scale = (
                    k_descale[()]
                    if const_expr(len(k_descale.shape) == 0)
                    else k_descale[batch_idx, kv_head_idx]
                )
                qk_descale = qk_descale * Float32(k_scale)
            if const_expr(descale_tensors.v_descale is not None):
                v_descale_tensor = descale_tensors.v_descale
                v_scale = (
                    v_descale_tensor[()]
                    if const_expr(len(v_descale_tensor.shape) == 0)
                    else v_descale_tensor[batch_idx, kv_head_idx]
                )
                v_descale = Float32(v_scale)
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
        acc_O_pv: Optional[cute.Tensor] = None,
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
        tOrP_acc = self._scores_to_pv_fragment(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        if const_expr(self.native_fp8):
            # The packed f16 conversion below is not valid for E4M3.
            tOrP_cur.store(tOrP_acc.load().to(self.dtype))
            _permute_Aregs_fp8(tOrP_cur)
        else:
            # tOrP.store(tOrP_acc.load().to(self.dtype))
            # the "to(self.dtype)" conversion fails to vectorize for block sizes other
            # than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
            # 2 elements. So we just call ptx directly.
            utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            self.cross_wg_p_barrier()
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        if const_expr(acc_O_pv is None):
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
        if const_expr(acc_O_pv is not None):
            # O = O * row_scale + this tile's P @ V (rescale and two-level add in one FFMA)
            acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
            acc_O_pv_mn = layout_utils.reshape_acc_to_mn(acc_O_pv)
            for r in cutlass.range(cute.size(row_scale), unroll_full=True):
                acc_O_mn[r, None].store(
                    acc_O_mn[r, None].load() * row_scale[r] + acc_O_pv_mn[r, None].load()
                )
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
        tOrP_acc = self._scores_to_pv_fragment(acc_S)
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
