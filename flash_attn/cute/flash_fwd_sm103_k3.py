# Dedicated SM103 K3 forward kernel for FP8 E4M3FN d192/v128 packed-varlen
# causal MHA. The interface dispatches all other configurations to the generic
# SM100 forward kernel. T8 uses two independent 1CTA MMA groups with K/V
# multicast; T0 and T0.75 use the tuned single-CTA schedule.

import math
from dataclasses import dataclass
from typing import Tuple, Callable, Optional, Literal
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Boolean, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL

from quack import copy_utils, layout_utils

from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute import utils
import flash_attn.cute.pipeline as pipeline_custom
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.softmax import SoftmaxSm100
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute import mma_sm100_desc as sm100_desc
from flash_attn.cute import blackwell_helpers as sm100_utils
from flash_attn.cute.named_barrier import NamedBarrierFwdSm100
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.tile_scheduler import (
    SchedulingMode,
    TileSchedulerArguments,
    TileSchedulerProtocol,
    WorkTileInfo,
)
from flash_attn.cute.fa_logging import fa_log
from flash_attn.cute.utils import AuxData
from flash_attn.cute.flash_fwd_sm100 import DescaleTensors


@dataclass
class K3TileSchedulerArguments(TileSchedulerArguments):
    """K3 scheduler inputs that additionally carry exact packed KV lengths."""

    mCuSeqlensK: Optional[cute.Tensor] = None
    mSeqUsedK: Optional[cute.Tensor] = None


class K3SingleTileVarlenScheduler:
    @dataclass
    class Params(ParamsBase):
        num_head: Int32
        num_batch: Int32
        total_q: Int32
        max_kvblock_in_l2: Int32
        tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
        mCuSeqlensQ: Optional[cute.Tensor] = None
        mCuSeqlensK: Optional[cute.Tensor] = None
        cluster_shape_m: cutlass.Constexpr[int] = 1

        @staticmethod
        @cute.jit
        def create(
            args: K3TileSchedulerArguments,
            *,
            scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
            loc=None,
            ip=None,
        ) -> "K3SingleTileVarlenScheduler.Params":
            assert scheduling_mode == SchedulingMode.STATIC
            size_l2 = 50 * 1024 * 1024  # 50 MB for K & V
            # if backward, this is qdo block size
            kv_block_size = (
                (args.headdim + args.headdim_v) * args.element_size * args.tile_shape_mn[1]
            )
            max_kvblock_in_l2 = size_l2 // kv_block_size
            assert args.mCuSeqlensQ is not None and args.mCuSeqlensK is not None
            assert args.cluster_shape_mn[1] == 1, "Only cluster_shape_mn[1] == 1 is supported"
            return K3SingleTileVarlenScheduler.Params(
                num_head=args.num_head,
                num_batch=args.num_batch,
                total_q=args.total_q,
                max_kvblock_in_l2=max_kvblock_in_l2,
                tile_shape_mn=args.tile_shape_mn,
                mCuSeqlensQ=args.mCuSeqlensQ,
                mCuSeqlensK=args.mCuSeqlensK,
                cluster_shape_m=args.cluster_shape_mn[0],
            )

    def __init__(
        self,
        params: Params,
        tile_idx: Int32,
        work_tile_cache: cute.Tensor | None = None,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._tile_idx = tile_idx
        self._is_first_block = True
        self.work_tile_cache = work_tile_cache
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: K3TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        return K3SingleTileVarlenScheduler.Params.create(
            args, scheduling_mode=scheduling_mode, loc=loc, ip=ip
        )

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        work_tile_cache: cute.Tensor | None = None,
        *,
        loc=None,
        ip=None,
    ) -> "K3SingleTileVarlenScheduler":
        tile_idx = cute.arch.block_idx()[0]
        return K3SingleTileVarlenScheduler(
            params,
            tile_idx,
            work_tile_cache=work_tile_cache,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        total_blocks_max = (
            params.total_q
            + params.num_batch * (params.cluster_shape_m * params.tile_shape_mn[0] - 1)
        ) // params.tile_shape_mn[0]
        # Round down to nearest multiple of cluster since odd excess is always padding.
        total_blocks_max = total_blocks_max // params.cluster_shape_m * params.cluster_shape_m
        return (total_blocks_max * params.num_head, Int32(1), Int32(1))

    @cute.jit
    def _get_num_m_blocks(self, lane: Int32, bidb_start: Int32) -> Int32:
        params = self.params
        batch_idx = lane + bidb_start
        cur_cu_seqlen = Int32(0)
        if batch_idx <= params.num_batch:
            cur_cu_seqlen = params.mCuSeqlensQ[batch_idx]
        next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
        seqlen = next_cu_seqlen - cur_cu_seqlen
        return (
            cute.ceil_div(cute.ceil_div(seqlen, params.tile_shape_mn[0]), params.cluster_shape_m)
            if batch_idx < params.num_batch and lane < cute.arch.WARP_SIZE - 1
            else Int32(0)
        )

    @cute.jit
    def _get_num_n_blocks(self, lane: Int32, bidb_start: Int32) -> Int32:
        """Return this request's KV tile count for the T8 L2 swizzle."""
        params = self.params
        batch_idx = lane + bidb_start
        seqlen = Int32(0)
        cur_cu_seqlen = Int32(0)
        if batch_idx <= params.num_batch:
            cur_cu_seqlen = params.mCuSeqlensK[batch_idx]
        next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
        seqlen = next_cu_seqlen - cur_cu_seqlen
        return (
            cute.ceil_div(seqlen, params.tile_shape_mn[1])
            if batch_idx < params.num_batch and lane < cute.arch.WARP_SIZE - 1
            else Int32(0)
        )

    @cute.jit
    def _varlen_coord_map(self) -> WorkTileInfo:
        """Map self._tile_idx to (block, head, batch) via warp-level prefix sums."""
        params = self.params
        lane_idx = cute.arch.lane_idx()
        num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=0)
        num_n_blocks = self._get_num_n_blocks(lane_idx, bidb_start=0)
        num_m_blocks_cumulative = utils.warp_prefix_sum(num_m_blocks, lane_idx)
        # Total number of blocks for the next 31 batches
        m_blocks_in_group = cute.arch.shuffle_sync(num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1)
        # Same for all lanes
        group_end_tile = m_blocks_in_group * params.num_head
        # if cute.arch.thread_idx()[0] == 128 + 31: cute.printf("K3SingleTileVarlenScheduler: tile_idx=%d, group_end_tile = %d, num_m_blocks=%d, num_m_blocks_cumulative = %d, m_blocks_in_group = %d", self._tile_idx, group_end_tile, num_m_blocks, num_m_blocks_cumulative, m_blocks_in_group)
        block, head_idx, batch_idx = Int32(0), Int32(0), Int32(0)
        next_tile_idx = self._tile_idx // params.cluster_shape_m
        while group_end_tile <= next_tile_idx:
            batch_idx += cute.arch.WARP_SIZE - 1
            if batch_idx >= params.num_batch:
                batch_idx = Int32(params.num_batch)
                group_end_tile = next_tile_idx + 1
            else:
                num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=batch_idx)
                num_n_blocks = self._get_num_n_blocks(lane_idx, bidb_start=batch_idx)
                num_m_blocks_cumulative = utils.warp_prefix_sum(num_m_blocks, lane_idx)
                m_blocks_in_group = cute.arch.shuffle_sync(
                    num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1
                )
                group_end_tile += m_blocks_in_group * params.num_head
        is_valid = False
        if batch_idx >= params.num_batch:
            block, head_idx, batch_idx = Int32(0), Int32(0), Int32(params.num_batch)
        else:
            group_start_tile = group_end_tile - m_blocks_in_group * params.num_head
            # if cute.arch.thread_idx()[0] == 128 + 31: cute.printf("K3SingleTileVarlenScheduler: tile_idx=%d, group_end_tile = %d, num_m_blocks=%d, batch_idx = %d", self._tile_idx, group_end_tile, num_m_blocks, batch_idx)
            # The next problem to process is the first one that does not have ending tile position
            # that is greater than or equal to tile index.
            batch_idx_in_group = cute.arch.popc(
                cute.arch.vote_ballot_sync(
                    group_start_tile + num_m_blocks_cumulative * params.num_head <= next_tile_idx
                )
            )
            batch_idx += batch_idx_in_group
            num_m_blocks_prev_lane = (
                0
                if batch_idx_in_group == 0
                else cute.arch.shuffle_sync(num_m_blocks_cumulative, batch_idx_in_group - 1)
            )
            num_m_blocks = cute.arch.shuffle_sync(num_m_blocks, batch_idx_in_group)
            num_n_blocks = cute.arch.shuffle_sync(num_n_blocks, batch_idx_in_group)
            mh_block = next_tile_idx - group_start_tile - num_m_blocks_prev_lane * params.num_head
            # LPT order with KV-length-aware L2 head grouping.
            if num_n_blocks <= params.max_kvblock_in_l2 // 8:
                num_n_blocks = (
                    num_m_blocks
                    * params.tile_shape_mn[0]
                    * params.cluster_shape_m
                    // params.tile_shape_mn[1]
                )
            nheads_in_l2 = (
                16
                if num_n_blocks * 16 <= params.max_kvblock_in_l2
                else (
                    8
                    if num_n_blocks * 8 <= params.max_kvblock_in_l2
                    else (
                        4
                        if num_n_blocks * 4 <= params.max_kvblock_in_l2
                        else (2 if num_n_blocks * 2 <= params.max_kvblock_in_l2 else 1)
                    )
                )
            )
            nheads_in_l2 = min(nheads_in_l2, params.num_head)
            mh_in_l2 = nheads_in_l2 * num_m_blocks
            section_idx = mh_block // mh_in_l2
            l2_mod = mh_block - section_idx * mh_in_l2
            nheads_in_this_section = (
                nheads_in_l2
                if nheads_in_l2 * (section_idx + 1) <= params.num_head
                else params.num_head - section_idx * nheads_in_l2
            )
            block = l2_mod // nheads_in_this_section
            head_idx_residual = l2_mod - block * nheads_in_this_section
            head_idx = section_idx * nheads_in_l2 + head_idx_residual
            block = num_m_blocks - 1 - block
            is_valid = self._is_first_block and batch_idx < params.num_batch
            if cutlass.const_expr(params.cluster_shape_m > 1):
                bidx_in_cluster = cute.arch.block_in_cluster_idx()
                block = block * params.cluster_shape_m + bidx_in_cluster[0]
        # if cute.arch.thread_idx()[0] == 128: cute.printf("K3SingleTileVarlenScheduler: tile_idx=%d, batch_idx=%d, head_idx=%d, block=%d, is_valid = %d", self._tile_idx, batch_idx, head_idx, block, is_valid)
        return WorkTileInfo(
            (Int32(block), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid
        )

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        if const_expr(
            self.work_tile_cache is not None and not self._is_first_block
        ):
            return WorkTileInfo((Int32(0), Int32(0), Int32(self.params.num_batch), Int32(0)), False)
        return self._varlen_coord_map()

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None):
        if const_expr(self.work_tile_cache is not None):
            return WorkTileInfo(
                (
                    Int32(self.work_tile_cache[0]),
                    Int32(self.work_tile_cache[1]),
                    Int32(self.work_tile_cache[2]),
                    Int32(self.work_tile_cache[3]),
                ),
                self.work_tile_cache[4] != 0,
            )
        return self._varlen_coord_map()

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._is_first_block = False
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        pass

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        objs = [self.params, self._tile_idx]
        if const_expr(self.work_tile_cache is not None):
            objs += [self.work_tile_cache]
        for obj in objs:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        objs = [self.params, self._tile_idx]
        if const_expr(self.work_tile_cache is not None):
            objs += [self.work_tile_cache]
        for obj, n_items in zip(objs, self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        params, tile_idx = obj_list[:2]
        next_obj = 2
        work_tile_cache = None
        if const_expr(self.work_tile_cache is not None):
            work_tile_cache = obj_list[next_obj]
        return self.__class__(
            params,
            tile_idx,
            work_tile_cache,
            loc=self._loc,
        )


@dataclass
class SoftmaxSm103K3(SoftmaxSm100):
    """SM103 K3 softmax with the fast exp2 used by the tuned profile."""

    @staticmethod
    def create(
        scale_log2: Float32,
        rescale_threshold: cutlass.Constexpr[float] = 0.0,
        softmax_scale: Float32 | None = None,
        max_offset: cutlass.Constexpr[int] = 0,
    ):
        num_rows = 1
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return SoftmaxSm103K3(
            scale_log2,
            num_rows,
            row_max,
            row_sum,
            100,
            softmax_scale,
            rescale_threshold=rescale_threshold,
            max_offset=max_offset,
        )

    @cute.jit
    def update_row_max_from_local(
        self,
        row_max_new: Float32,
        is_first: Boolean,
    ) -> Tuple[Float32, Float32]:
        if cutlass.const_expr(is_first):
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale = 0.0
        else:
            row_max_old = self.row_max[0]
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2
            acc_scale = cute.math.exp2(acc_scale_, fastmath=True)
            if cutlass.const_expr(self.rescale_threshold > 0.0):
                if acc_scale_ >= -self.rescale_threshold:
                    row_max_new = row_max_old
                    row_max_safe = row_max_old
                    acc_scale = 1.0
        self.row_max[0] = row_max_new
        return row_max_safe, acc_scale


class FlashAttentionForwardSm103K3:
    """Dedicated SM103 FP8 d192/v128 packed-varlen causal forward kernel."""

    def __init__(
        self,
        k3_rescale_threshold: float = 8.0,
    ):
        assert k3_rescale_threshold in (0.0, 0.75, 8.0)

        self.arch = BaseDSL._get_dsl().get_arch_enum()
        assert self.arch.is_family_of(Arch.sm_103f), "K3 requires SM103"
        self.head_dim_padded = 192
        self.head_dim_v_padded = 128
        self.same_hdim_kv = False
        self.same_hdim_kv_padded = False
        self.check_hdim_oob = False
        self.check_hdim_v_oob = False
        self.m_block_size = 128
        self.n_block_size = 128
        self.q_stage = 2
        self.fp8_rescale_threshold = k3_rescale_threshold
        self.use_causal_kv_multicast = k3_rescale_threshold == 8.0
        self.use_2cta_instrs = False
        self.cta_group_size = 1
        self.cluster_shape_mn = (2, 1) if self.use_causal_kv_multicast else (1, 1)
        self.cta_tiler = (256, 128, 192)
        self.mma_tiler_qk = (128, 128, 192)
        self.mma_tiler_pv = (128, 128, 128)
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.is_persistent = False
        self.is_causal = True
        self.is_local = False
        self.is_varlen_q = True
        self.qhead_per_kvhead = 1
        self.is_split_kv = False
        self.pack_gqa = False
        self.use_tma_KV = True
        self.use_tma_Q = True
        self.split_P_arrive = 0
        # Varlen can use TMA to store O, but only through a "ragged" descriptor: the O rows of a
        # request sit at the tail of a BIG_INT-sized ragged dim whose base pointer is shifted back
        # by BIG_INT rows, so TMA's own bound check drops the tile rows past seqlen_q. Without that
        # clipping a plain (total_q, h, dv) descriptor would let the last m_block of a request
        # clobber the first rows of the next request. See copy_utils.create_ragged_tensor_for_tma
        # and the existing users in flash_fwd_sm90.py / flash_fwd_mla_sm100.py.
        self.ragged_tma_O = True
        self.use_tma_O = True
        self.use_correction_warps_for_epi = False
        self.q_subtile_factor = 1
        self.score_mod = None
        self.mask_mod = None
        self.output_quant_key = None
        self.score_vec_size = 2
        self.mask_vec_size = 1
        self.is_sm103 = True
        self.tmem_ldred_max = True
        self.fp8_p_prescale_log2 = max(
            0, math.floor(math.log2(448) - self.fp8_rescale_threshold)
        )
        self.enable_ex2_emu = False
        self.s0_s1_barrier = False
        self.overlap_sO_sQ = True
        self.use_clc_scheduler = False
        self.sched_stages = 1
        self.scheduling_mode = SchedulingMode.STATIC
        self.cache_scheduler_coord = True
        self.cache_scheduler_warp = 14
        self.TileScheduler = K3SingleTileVarlenScheduler

        fa_log(
            1,
            f"TileScheduler={self.TileScheduler.__name__}, "
            f"scheduling_mode={self.scheduling_mode.name}, USE_2CTA={self.use_2cta_instrs}, "
            f"KV_MCAST={self.use_causal_kv_multicast}",
        )

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.epilogue_warp_ids = (13,)
        self.load_warp_ids = (14,)
        self.empty_warp_ids = (15,)
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.epilogue_warp_ids,
                *self.empty_warp_ids,
            )
        )
        assert 0 <= self.cache_scheduler_warp < self.threads_per_cta // cute.arch.WARP_SIZE

        self.clc_scheduler_warp_id = None

        self.tmem_s_offset = [0, self.n_block_size]  # e.g., 0, 128
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
            for i in range(self.q_stage)
        ]  # e.g., 256, 384
        self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded
        assert self.tmem_total <= self.tmem_alloc_cols
        self.tmem_s_to_p_offset = self.n_block_size // 2
        self.tmem_p_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]  # 0, 128

        # vec buffer for row_max & row_sum
        self.tmem_vec_offset = self.tmem_s_offset

        self._tune = {"ex2_emu_freq": 0, "ex2_emu_start_frg": 0}
        self.num_regs_softmax = 176
        self.num_regs_correction = 72
        self.num_regs_other = 88
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        self.stats_late_acquire = True
        self.s_stage = 2
        assert self.s_stage >= self.q_stage

        smem_size_q = self.q_stage * self.m_block_size * self.head_dim_padded * self.q_dtype.width // 8
        smem_size_o = self.q_stage * self.m_block_size * self.head_dim_v_padded * self.o_dtype.width // 8
        smem_size_q_o = smem_size_q + smem_size_o if not self.overlap_sO_sQ else max(smem_size_q, smem_size_o)
        smem_size_p = (
            self.s_stage * self.m_block_size * self.n_block_size * self.q_dtype.width // 8
        )
        smem_size_k_per_stage = self.n_block_size * self.head_dim_padded * self.k_dtype.width // 8
        smem_size_v_per_stage = self.n_block_size * self.head_dim_v_padded * self.v_dtype.width // 8
        smem_size_kv_per_stage = max(smem_size_k_per_stage, smem_size_v_per_stage) // self.cta_group_size
        # Cap small head_dim from over-staging: the 224*1024 budget undercounts
        # per-stage state, so at hd_padded=16 the unbounded formula picks 52 stages
        # and overflows the 227 KB SMEM cap. No-op for hd_padded >= 32 (max 26).
        kv_stage = min(
            (224 * 1024 - smem_size_q_o - smem_size_p) // smem_size_kv_per_stage,
            32,
        )
        if self.head_dim_padded == 192 and self.head_dim_v_padded == 128 and kv_stage == 2:
            # For hdim 192,128, we can fit 3 stages if we use uneven_kv_smem
             kv_stage = 3
        self.kv_stage = kv_stage
        # print("kv_stage", self.kv_stage)
        # For hdim 192,128 1CTA, we don't have enough smem to store all 3 stages of KV:
        # 128 x 192 x 2 bytes x 3 stages = 144KB, and we need 96KB for Q.
        # Instead we store smem as [smem_large, smem_small, smem_large], where smem_large is
        # 128 x 192 and smem_small is 128 x 128. We set the stride between the stages to be
        # 128 * 160, so that indexing the 0th and 2nd stages will get the right address,
        # but for the 1st stage we need to add or subtract (depending on phase) 128 x 64.
        self.uneven_kv_smem = (
            self.head_dim_padded == 192 and self.head_dim_v_padded == 128 and self.kv_stage == 3
        )
        self.uneven_kv_smem_offset = (
            self.n_block_size * (self.head_dim_padded - self.head_dim_v_padded) // 2
            if self.uneven_kv_smem
            else 0
        )
        assert self.uneven_kv_smem_offset % 1024 == 0

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
        descale_tensors: Optional[DescaleTensors] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_data: AuxData = AuxData(),
        output_scale: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        This method prepares the input tensors for processing, validates their shapes and types,
        configures the computation parameters, and launches the CUDA kernel.

        The method handles:
        1. Tensor layout transformations for specific memory access patterns
        2. Validation of tensor shapes and data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch with appropriate parameters
        """
        # setup static attributes before smem/grid/tma computation
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        assert self.q_dtype == cutlass.Float8E4M3FN, "K3 requires FP8 E4M3FN Q"
        assert self.k_dtype == cutlass.Float8E4M3FN, "K3 requires FP8 E4M3FN K"
        assert self.v_dtype == cutlass.Float8E4M3FN, "K3 requires FP8 E4M3FN V"
        assert self.o_dtype == cutlass.BFloat16, "K3 requires BF16 output"
        assert mCuSeqlensQ is not None and mCuSeqlensK is not None
        assert mSeqUsedQ is None and mSeqUsedK is None
        assert mDynamicCausal is None and mPageTable is None
        assert learnable_sink is None and blocksparse_tensors is None
        assert aux_data.tensors is None and aux_data.scalars is None
        assert output_scale is None
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        # Packed-varlen layouts: (total, heads, dim) -> (total, dim, heads).
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[0, 2, 1]))
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=[0, 2, 1]))
            for t in (mK, mV)
        ]
        num_splits = Int32(1)
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=[0, 2, 1]))
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=[1, 0]))
            if const_expr(mLSE is not None)
            else None
        )
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=[1, 0, 2]))

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()
        self.ex2_emu_freq = 0
        self.ex2_emu_start_frg = 0
        cta_group = tcgen05.CtaGroup.ONE
        q_major_mode = tcgen05.OperandMajorMode.K
        k_major_mode = tcgen05.OperandMajorMode.K
        v_major_mode = tcgen05.OperandMajorMode.MN
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)
        # Keep the default TS path intact. The experimental full-P path writes P
        # to shared memory and feeds PV through an SS MMA operand.
        p_source = tcgen05.OperandSource.SMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            q_major_mode,
            k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.mma_tiler_pv[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )
        local_cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))

        # epi_tile is per-CTA (not full 2CTA) since each CTA writes its own O portion
        self.epi_tile = (self.m_block_size, self.head_dim_v_padded)

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.s_stage
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.v_dtype, self.kv_stage
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.q_stage
        )
        if const_expr(not self.same_hdim_kv_padded):
            # sK and sV are using the same physical smem so we need to adjust the stride so that they line up
            stride_sK = const_expr(
                max(sK_layout.outer.stride[-1], 0)
            )  # take max to turn tuple to Int32
            stride_sV = const_expr(max(sV_layout.outer.stride[-1], 0))
            stage_stride = const_expr(
                max(stride_sK, stride_sV)
                if not self.uneven_kv_smem
                else (stride_sK + stride_sV) // 2
            )
            sK_layout = cute.make_composed_layout(
                sK_layout.inner,
                0,
                cute.make_layout(
                    (*sK_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sK_layout.outer.stride[:-1], stage_stride),
                ),
            )
            sV_layout = cute.make_composed_layout(
                sV_layout.inner,
                0,
                cute.make_layout(
                    (*sV_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sV_layout.outer.stride[:-1], stage_stride),
                ),
            )

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
            ]
        }
        for name in ("Q", "K", "V"):
            self.tma_copy_bytes[name] *= self.cta_group_size

        # TMA load for Q.  In the KV-multicast schedule Q remains CTA-local;
        # only the B operands (K/V) are multicast across cluster M.
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_load_op_kv = (
            sm100_utils_basic.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mnk, tiled_mma_qk.thr_id
            )
            if const_expr(self.use_causal_kv_multicast)
            else tma_load_op
        )
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            (
                local_cta_layout_vmnk.shape
                if const_expr(self.use_causal_kv_multicast)
                else cta_layout_vmnk.shape
            ),
        )
        gmem_tiled_copy_Q = None

        tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op_kv,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            cta_layout_vmnk.shape,
        )
        tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op_kv,
            mV,
            cute.select(sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_pv,
            tiled_mma_pv,
            cta_layout_vmnk.shape,
        )

        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        mO_tma = copy_utils.create_ragged_tensor_for_tma(mO, ragged_dim=0, ptr_shift=True)
        tma_atom_O, mO = cpasync.make_tiled_tma_atom(
            tma_store_op, mO_tma, cute.select(sO_layout, mode=[0, 1]), self.epi_tile
        )
        gmem_tiled_copy_O = None

        TileScheduler = self.TileScheduler
        _num_block_divisor = self.cta_tiler[0]
        tile_sched_args = K3TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), _num_block_divisor),
            cute.size(mQ.shape[2]),
            cute.size(mCuSeqlensQ.shape[0] - 1),
            num_splits,
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[0],  # Note that this is different from Sm90 since we transpose mV in Sm100
            total_q=cute.size(mQ.shape[0]),
            tile_shape_mn=self.cta_tiler[:2],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            qhead_per_kvhead_packgqa=1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=True,
            is_split_kv=False,
            cluster_shape_mn=self.cluster_shape_mn,
            use_cluster_idx=False,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args, scheduling_mode=self.scheduling_mode
        )
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        sO_size = cute.cosize(sO_layout) if const_expr(not self.overlap_sO_sQ) else 0
        sP_size = cute.cosize(tP_layout)
        sQ_size = (
            cute.cosize(sQ_layout) if const_expr(not self.overlap_sO_sQ) else
            cutlass.max(cute.cosize(sQ_layout), cute.cosize(sO_layout) * self.o_dtype.width // self.q_dtype.width)
        )

        @cute.struct
        class SharedStorage:
            # m_barriers for pipelines
            mbar_load_Q: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_load_KV: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_S_full_P_full_O_rescaled: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_P_full_lastsplit: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_O_full: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_softmax_stats: cute.struct.MemRange[Int64, self.q_stage * 2]
            # mbar_softmax_stats: cute.struct.MemRange[Int64, self.q_stage * 4 * 2]
            mbar_O_epi: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_s0_s1_sequence: cute.struct.MemRange[Int64, 2 * 2]
            # Tmem dealloc cluster barrier
            tmem_dealloc_mbar: Int64
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 2]
            # One CTA-wide copy of (m_block, head, batch, split, valid). It sits before
            # sO's 1-KiB alignment, so this benchmark path normally consumes padding.
            scheduler_coord: cute.struct.MemRange[Int32, 5]
            # Large TMA buffers with 1024-byte alignment
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size], self.buffer_align_bytes
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes
            ]
            sP: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sP_size], self.buffer_align_bytes
            ]
            sK: cute.struct.Align[
                # cute.cosize(sK_layout) is correct even in the case of self.uneven_kv_smem
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale, self.score_mod)
        window_size_left = Int32(window_size_left) if window_size_left is not None else None
        window_size_right = Int32(window_size_right) if window_size_right is not None else None
        fastdiv_mods = None
        head_divmod = None
        self.use_block_sparsity = False

        # Launch the kernel synchronously
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
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
            descale_tensors,
            blocksparse_tensors,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_Q,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            num_splits,
            aux_data,
            fastdiv_mods,
            head_divmod,
            output_scale,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk if cute.size(self.cluster_shape_mnk) > 1 else None,
            stream=stream,
            min_blocks_per_mp=1,
        )

    def _generate_attention_mask_cls(self, window_size_left, window_size_right):
        return partial(
            AttentionMask,
            self.m_block_size,
            self.n_block_size,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b) or (total_q, d, h) if there is cu_seqlens_q
        mK: cute.Tensor,  # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there is cu_seqlens_k or (page_size, d, h_k, num_pages) if there is page_table
        mV: cute.Tensor,  # (d, s_k, h_k, b_k) or (d, total_k, h_k) if there is cu_seqlens_k or (d, page_size, h_k, num_pages) if there is page_table
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        descale_tensors: Optional[DescaleTensors],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_Q: Optional[cute.TiledCopy],
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        num_splits: Int32,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
        output_scale: Optional[cute.Tensor] = None,
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )
        # Most pipelines remain CTA-local in the KV-multicast schedule.  The
        # physical cluster layout is used only by the K/V TMA pipeline and by
        # cluster initialization.
        pipeline_cta_layout_vmnk = (
            cute.make_layout((1, 1, 1, 1))
            if const_expr(self.use_causal_kv_multicast)
            else cta_layout_vmnk
        )
        # Setup cta/thread coordinates
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(cute.size(tiled_mma_qk.thr_id.shape) == 1):
            mma_tile_coord_v = 0
        else:
            mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE * len(
                (self.mma_warp_id,
                 *self.softmax0_warp_ids,
                 *self.softmax1_warp_ids,
                 *self.correction_warp_ids)
            ),
        )
        # Tensor memory dealloc barrier init
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        mma_warp = ThreadCooperativeGroup(len([self.mma_warp_id]))
        mma_warp_kv = ThreadCooperativeGroup(
            len([self.mma_warp_id])
            * (self.cluster_shape_mn[0] if self.use_causal_kv_multicast else 1)
        )
        tma_warp = ThreadCooperativeGroup(1)
        load_threads = ThreadCooperativeGroup(len(self.load_warp_ids) * cute.arch.WARP_SIZE)
        softmax_warps = ThreadCooperativeGroup(len(self.softmax0_warp_ids))
        softmax_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.softmax0_warp_ids))
        # softmax_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE)
        correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        # correction_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE)
        softmax_correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids + self.correction_warp_ids)
        )
        epilogue_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.epilogue_warp_ids))
        # For UMMA-bridging pipelines: the non-MMA side spans both CTAs in the cluster,
        # so the thread count must include warps from both CTAs.
        softmax_warps_cluster = ThreadCooperativeGroup(
            len(self.softmax0_warp_ids) * self.cta_group_size
        )
        correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids) * self.cta_group_size
        )
        softmax_correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids + self.correction_warp_ids) * self.cta_group_size
        )
        pipeline_q = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_Q.data_ptr(),
            num_stages=self.q_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=pipeline_cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_kv = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_KV.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp_kv,
            tx_count=self.tma_copy_bytes["K"],
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # This pipeline is not the typical producer-consumer pipeline. The "producer" mma warp
        # uses it to signal that S is ready, and the softmax threads wait for S to be ready.
        # On the QK-first path this pipeline tracks only S-ready/S-consumed, so
        # QK can reuse a tmem slot as soon as softmax has copied S to registers.
        # Other paths retain the original combined P-ready/O-rescaled empty barrier.
        pipeline_s_p_o = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_S_full_P_full_O_rescaled.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=softmax_threads,
            cta_layout_vmnk=pipeline_cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_p_lastsplit = pipeline_custom.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_P_full_lastsplit.data_ptr(),
            num_stages=self.q_stage,
            producer_group=softmax_correction_threads_cluster,
            consumer_group=mma_warp,
            cta_layout_vmnk=pipeline_cta_layout_vmnk,
            defer_sync=True,
        )
        # MMA warp uses this to signal to the correction warps that O is ready.
        pipeline_o_acc = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=correction_threads_cluster,
            cta_layout_vmnk=pipeline_cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_s0_s1_sequence = None
        if const_expr(self.s0_s1_barrier and self.q_stage > 1):
            # This is not a typical producer-consumer pipeline. We will directly use
            # pipeline_s0_s1_sequence.sync_object_full and will not use
            # pipeline_s0_s1_sequence.sync_object_empty.
            pipeline_s0_s1_sequence = pipeline_custom.PipelineAsync.create(
                barrier_storage=storage.mbar_s0_s1_sequence.data_ptr(),
                num_stages=2,
                producer_group=softmax_threads,
                consumer_group=softmax_threads,
                defer_sync=True,
            )
        pipeline_sm_stats = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_softmax_stats.data_ptr(),
            num_stages=self.q_stage,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        # Should put the NamedBarrier inside the pipeline class so we'll just have pipeline_sm_stats
        sm_stats_barrier = pipeline_custom.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0), num_threads=cute.arch.WARP_SIZE * 2
        )
        pipeline_o_epi = None
        if const_expr(not self.use_correction_warps_for_epi):
            pipeline_o_epi = pipeline_custom.PipelineAsync.create(
                barrier_storage=storage.mbar_O_epi.data_ptr(),
                num_stages=self.q_stage,
                producer_group=correction_threads,
                consumer_group=epilogue_threads,
                defer_sync=True,
            )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sP = storage.sP.get_tensor(tP_layout.outer, swizzle=tP_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
        if const_expr(not self.overlap_sO_sQ):
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        else:
            sO = cute.make_tensor(cute.recast_ptr(sQ.iterator, sO_layout.inner, self.o_dtype), sO_layout.outer)
        sScale = storage.sScale.get_tensor(cute.make_layout(self.q_stage * self.m_block_size * 2))

        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
        thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        # This is a fake tensor, by right we need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))
        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(cute.append(pv_acc_shape, self.q_stage))
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset[0], tOtO.layout)
        tOrP = thr_mma_pv.make_fragment_A(sP)

        block_info = BlockInfo(
            # This is cta_tiler, not mma_tiler_qk, since we move by block by (2 * mma_tiler[0], mma_tiler[1])
            self.cta_tiler[0],
            self.cta_tiler[1],
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            mCuTotalMBlocks=None,
            mCuBlockIdxOffsets=None,
        )
        AttentionMaskCls = self._generate_attention_mask_cls(
            window_size_left, window_size_right
        )
        # Cluster wait before tensor memory alloc
        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        scheduler_coord = storage.scheduler_coord.get_tensor(cute.make_layout(5))
        bootstrap_scheduler = self.tile_scheduler_cls.create(tile_sched_params)
        if warp_idx == self.cache_scheduler_warp:
            bootstrap_work = bootstrap_scheduler.initial_work_tile_info()
            if cute.arch.lane_idx() == 0:
                scheduler_coord[0] = bootstrap_work.tile_idx[0]
                scheduler_coord[1] = bootstrap_work.tile_idx[1]
                scheduler_coord[2] = bootstrap_work.tile_idx[2]
                scheduler_coord[3] = bootstrap_work.tile_idx[3]
                valid = Int32(0)
                if bootstrap_work.is_valid_tile:
                    valid = Int32(1)
                scheduler_coord[4] = valid
        cute.arch.sync_threads()
        tile_scheduler = self.tile_scheduler_cls.create(
            tile_sched_params, work_tile_cache=scheduler_coord
        )
        assert isinstance(tile_scheduler, TileSchedulerProtocol), f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY / CLC SCHEDULER WARP
        # ///////////////////////////////////////////////////////////////////////////////
        for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
            if warp_idx == self.empty_warp_ids[i]:
                cute.arch.setmaxregister_decrease(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load(
                thr_mma_qk,
                thr_mma_pv,
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
                pipeline_kv,
                cta_layout_vmnk,
                block_info,
                num_splits,
                SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            # Alloc tensor memory buffer
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                sP,
                tStS,
                tOtO,
                tOrP,
                pipeline_q,
                pipeline_kv,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_o_acc,
                is_leader_cta,
                block_info,
                num_splits,
                SeqlenInfoCls,
                blocksparse_tensors,
                tile_scheduler=tile_scheduler,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(not self.use_correction_warps_for_epi):
            if warp_idx >= self.epilogue_warp_ids[0] and warp_idx <= self.epilogue_warp_ids[-1]:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                self.epilogue_s2g(
                    mO,
                    sO,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    pipeline_o_epi,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    mma_tile_coord_v,
                    blocksparse_tensors=blocksparse_tensors,
                    tile_scheduler=tile_scheduler,
                )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            (const_expr(self.q_stage == 2) and warp_idx <= self.softmax1_warp_ids[-1]) or
            (const_expr(self.q_stage == 1) and warp_idx <= self.softmax0_warp_ids[-1])
        ):
            # increase register after decreasing
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            # sync with mma warp before retrieving tmem ptr
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
                descale_tensors=descale_tensors,
                thr_mma_qk=thr_mma_qk,
                sP=sP,
                sScale=sScale,
                mLSE=mLSE,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                pipeline_s0_s1_sequence=pipeline_s0_s1_sequence,
                learnable_sink=learnable_sink,
                block_info=block_info,
                num_splits=num_splits,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
                blocksparse_tensors=blocksparse_tensors,
                tile_scheduler=tile_scheduler,
            )

            if const_expr(not self.s0_s1_barrier):
                stage = Int32(0 if const_expr(self.q_stage == 1) or warp_idx < self.softmax1_warp_ids[0] else 1)
                softmax_loop(stage=stage, tStS=tStS)
            else:
                # If there's s0_s1_barrier, it's faster to have 2 WGs having different code
                if warp_idx < self.softmax1_warp_ids[0]:
                    softmax_loop(stage=0, tStS=tStS)
                if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax1_warp_ids[0]:
                    softmax_loop(stage=1, tStS=tStS)

            tmem_alloc_barrier.arrive()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_correction)
            # sync with mma warp before retrieving tmem ptr
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.correction_loop(
                thr_mma_qk,
                thr_mma_pv,
                tStS,
                tOtO,
                sScale,
                mO,
                mLSE,
                sO,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_o_acc,
                pipeline_sm_stats,
                sm_stats_barrier,
                pipeline_o_epi,
                learnable_sink,
                descale_tensors,
                gmem_tiled_copy_O,
                tma_atom_O,
                softmax_scale_log2,
                block_info,
                num_splits,
                SeqlenInfoCls,
                blocksparse_tensors,
                tile_scheduler,
                output_scale,
            )
            tmem_alloc_barrier.arrive()

        return

    @cute.jit
    def _kv_range_m_block(self, m_block: Int32) -> Int32:
        """Return the latest M tile in an independent-CTA multicast cluster.

        All CTAs must execute the same number of K/V pipeline transactions.
        Earlier causal tiles therefore process the cluster's union range; their
        extra diagonal tiles are fully removed by the normal causal mask.
        """
        if const_expr(self.use_causal_kv_multicast):
            group = 2
            return (m_block // group) * group + group - 1
        return m_block

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        cta_layout_vmnk: cute.Layout,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
    ):
        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        kv_mcast_mask = None
        kv_cta_coord = Int32(0)
        kv_cta_layout = cute.make_layout(1)
        if const_expr(self.use_causal_kv_multicast):
            cta_rank_in_cluster = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            block_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(
                cta_rank_in_cluster
            )
            kv_cta_coord = block_in_cluster_coord_vmnk[1]
            kv_cta_layout = cute.make_layout(
                cute.slice_(cta_layout_vmnk, (0, None, 0, 0)).shape
            )
            kv_mcast_mask = cpasync.create_tma_multicast_mask(
                cta_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None, head_idx])
            mV_cur = cute.domain_offset((0, seqlen.offset_k), mV[None, None, head_idx])
            gK = cute.local_tile(
                mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0)
            )
            gV = cute.local_tile(
                mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None)
            )
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)
            tiler_gQ = ((self.mma_tiler_qk[0] * self.q_stage), self.head_dim_padded)
            gQ = cute.local_tile(mQ_cur, tiler_gQ, (m_block, 0))
            gQ = layout_utils.select(
                cute.flat_divide(gQ, (self.mma_tiler_qk[0],)), mode=[0, 2, 1]
            )
            tSgQ = thr_mma_qk.partition_A(gQ)
            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
            )
            load_Q = partial(
                self.load_Q, load_Q_fn, pipeline_q=pipeline_q, phase=q_producer_phase
            )

            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K,
                kv_cta_coord,
                kv_cta_layout,
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgK, 0, 3),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V,
                kv_cta_coord,
                kv_cta_layout,
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tOgV, 0, 3),
            )

            load_K = partial(
                self.load_KV,
                tma_atom_K,
                tKgK,
                tKsK,
                pipeline_kv=pipeline_kv,
                K_or_V="K",
                mcast_mask=kv_mcast_mask,
            )
            load_V = partial(
                self.load_KV,
                tma_atom_V,
                tVgV,
                tVsV,
                pipeline_kv=pipeline_kv,
                K_or_V="V",
                mcast_mask=kv_mcast_mask,
            )

            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen,
                self._kv_range_m_block(m_block),
                split_idx,
                num_splits,
            )
            n_block_first = n_block_max - 1 if n_block_max > 0 else 0
            load_K(block=n_block_first, producer_state=kv_producer_state)
            load_Q(block=0, stage=0)
            kv_producer_state.advance()
            load_Q(block=1, stage=1)
            q_producer_phase ^= 1
            load_V(block=n_block_first, producer_state=kv_producer_state)
            kv_producer_state.advance()
            for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                n_block = n_block_max - 2 - i
                load_K(block=n_block, producer_state=kv_producer_state)
                kv_producer_state.advance()
                load_V(block=n_block, producer_state=kv_producer_state)
                kv_producer_state.advance()

            work_tile = tile_scheduler.advance_to_next_work()
            # End of persistent scheduler loop

        pipeline_kv.producer_tail(kv_producer_state)
        # This is equivalent to pipeline_q.producer_tail for the TMA-Q producer warp.
        pipeline_q.producer_acquire_w_index_phase(self.q_stage - 1, q_producer_phase)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.ThrMma,
        tiled_mma_pv: cute.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sP: Optional[cute.Tensor],
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        tOrP: cute.Tensor,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        is_leader_cta: Boolean,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        tile_scheduler=None,
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        if const_expr(self.q_stage == 2):
            tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 1])
        else:
            tSrQs = (tSrQ[None, None, None, 0],)

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op
        qk_mma_idesc, pv_mma_idesc = sm100_desc.mma_op_to_idesc(qk_mma_op), sm100_desc.mma_op_to_idesc(pv_mma_op)
        qk_mma_kind = sm100_utils._tcgen05_mma_kind(qk_mma_op)
        q_smem_base = sm100_desc.smem_desc_base_from_tensor(sQ, sm100_desc.Major.K)
        k_smem_base = sm100_desc.smem_desc_base_from_tensor(sK, sm100_desc.Major.K)
        v_smem_base = sm100_desc.smem_desc_base_from_tensor(sV, sm100_desc.Major.MN)
        q_smem_start = [sm100_desc.make_smem_desc_start_addr(sQ[None, None, None, stage].iterator) for stage in range(self.q_stage)]

        sm100_utils.declare_ptx_smem_desc(q_smem_start[self.q_stage - 1], q_smem_base, tSrQ[None, None, None, 0].layout, var_name_prefix="fa_fwd_q_smem_desc")
        sm100_utils.declare_ptx_idesc(qk_mma_op, var_name="fa_fwd_qk_mma_idesc")
        sm100_utils.declare_ptx_idesc(pv_mma_op, var_name="fa_fwd_pv_mma_idesc")

        sQ_stage_stride = (sQ.layout.stride[-1] * sQ.element_type.width // 8) >> 4
        if const_expr(self.q_stage == 1):
            sQ_stage_stride = 0
        gemm_Si = [
            partial(
                # sm100_utils.gemm_ptx_precomputed,
                # self.tmem_s_offset[stage],
                # smem_desc_start_a=q_smem_start[stage],
                # idesc=qk_mma_idesc,
                # smem_desc_base_a=q_smem_base,
                # smem_desc_base_b=k_smem_base,
                # tCrA_layout=tSrQ[None, None, None, 0].layout,
                sm100_utils.gemm_ptx_precomputed_varname,
                self.tmem_s_offset[stage],
                # idesc=qk_mma_idesc,
                smem_desc_base_b=k_smem_base,
                tCrB_layout=tSrK[None, None, None, 0].layout,
                smem_var_name_prefix="fa_fwd_q_smem_desc",
                idesc_var_name="fa_fwd_qk_mma_idesc",
                kind=qk_mma_kind,
                smem_offset=-sQ_stage_stride if stage == 0 else sQ_stage_stride,
                zero_init=True,
                cta_group=self.cta_group_size,
            )
            for stage in range(self.q_stage)
        ]
        # gemm_Si = [
        #     partial(
        #         sm100_utils.gemm,
        #         tiled_mma_qk,
        #         tStS[None, None, None, stage],
        #         tCrA=tSrQ[None, None, None, stage],
        #         zero_init=True,
        #     )
        #     for stage in range(self.q_stage)
        # ]
        gemm_Pi = [
            partial(
                # sm100_utils.gemm_ptx_precomputed,
                sm100_utils.gemm_ptx_partial,
                pv_mma_op,
                self.tmem_o_offset[stage],
                tOrP[None, None, None, stage],
                sA=sP[None, None, None, stage],
                split_arrive=None,
                # smem_desc_start_a=tOrP[None, None, None, stage].iterator.toint(),
                # smem_desc_start_a=self.tmem_p_offset[stage],
                # idesc=pv_mma_idesc,
                # smem_desc_base_a=None,
                # smem_desc_base_b=v_smem_base,
                # tCrA_layout=tOrP[None, None, None, 0].layout,
                # tCrB_layout=tOrV[None, None, None, 0].layout
                cta_group=self.cta_group_size,
            )
            for stage in range(self.q_stage)
        ]
        # gemm_Pi = [
        #     partial(
        #         sm100_utils.gemm, tOtO[None, None, None, stage], tCrA=tOrP[None, None, None, stage]
        #     )
        #     for stage in range(self.q_stage)
        # ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        P_full_O_rescaled_phase = Int32(0)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen,
                self._kv_range_m_block(m_block),
                split_idx,
                num_splits,
            )
            block_iter_count = n_block_max - n_block_min
            process_tile = True

            if process_tile and is_leader_cta:
                for stage in cutlass.range_constexpr(self.q_stage):
                    # GEMM_QK00 (Q0 * K0 -> S0) or GEMM_QK01 (Q1 * K0 -> S1)
                    # 1. wait for Q0 / Q1
                    pipeline_q.consumer_wait_w_index_phase(stage, mma_q_consumer_phase)
                    # 2. wait for K0
                    if const_expr(stage == 0):
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                    tSrKi = tSrK[None, None, None, Ki_index]
                    # We don't need to acquire empty S0 / S1.
                    # For the first iteration, we don't need to wait as we're guaranteed S0 / S1
                    # are empty. For subsequent iterations, the wait happened at the end
                    # of the while loop.
                    # 3. gemm
                    # sm100_utils.gemm(tiled_mma_qk, tStS[None, None, None, stage], tSrQ[None, None, None, stage], tSrKi, zero_init=True)
                    sK_cur = sK[None, None, None, Ki_index]
                    if const_expr(self.uneven_kv_smem):
                        sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                    # gemm_Si[stage](tCrB=tSrKi, sB=sK_cur)
                    gemm_Si[stage](
                        smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator)
                    )
                    # gemm_Si[stage](tCrB=tSrKi)
                    # 4. release S0 / S1
                    pipeline_s_p_o.producer_commit_w_index(stage)
                mma_q_consumer_phase ^= 1
                # 5. release K0
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM (Q1 * K0 -> S1)
                # Note: Q0 & Q1 are still needed in the seqlen_kv loop
                # so we need to release them after the seqlen_kv loop

                # O hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
                block_loop_count = block_iter_count - 1
                O_should_accumulate = False
                for i in cutlass.range(block_loop_count, unroll=1):
                    # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                    # 1. wait for V0
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    mma_kv_release_state = mma_kv_consumer_state.clone()
                    Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                    tOrVi = tOrV[None, None, None, Vi_index]
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # QK(i+1) only waits for softmax to finish reading S; P
                        # publication and O correction use a separate rendezvous.
                        pipeline_s_p_o.producer_acquire_w_index_phase(
                            stage, P_full_O_rescaled_phase
                        )
                        if const_expr(stage == 0):
                            mma_kv_consumer_state.advance()
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index, Ki_phase = (
                            mma_kv_consumer_state.index,
                            mma_kv_consumer_state.phase,
                        )
                        sK_cur = sK[None, None, None, Ki_index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                        gemm_Si[stage](
                            smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(
                                sK_cur.iterator
                            )
                        )
                        pipeline_s_p_o.producer_commit_w_index(stage)
                        pipeline_p_lastsplit.consumer_wait_w_index_phase(
                            stage, P_full_O_rescaled_phase
                        )
                        # 3. gemm
                        # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                        # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                        sV_cur = sV[None, None, None, Vi_index]
                        if const_expr(self.uneven_kv_smem):
                            sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                        gemm_Pi[stage](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            # smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sV_cur.iterator),
                            zero_init=not O_should_accumulate,
                            mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(stage) if self.split_P_arrive > 0 else None,
                            mbar_phase=P_full_O_rescaled_phase,
                        )
                        pipeline_p_lastsplit.consumer_release_w_index(stage)
                        # Don't need to signal O_full to the correction warps since the
                        # correction warps wait for the softmax warps anyway. By the time the softmax
                        # warps finished, S_i for the next iteration must have been done, so O_i-1
                        # must have been done as well.
                        # pipeline_o_acc.producer_commit_w_index(stage)
                        # 4. release V(i-1)
                        if const_expr(stage == self.q_stage - 1):
                            pipeline_kv.consumer_release(mma_kv_release_state)
                            mma_kv_release_state.advance()
                        # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                    # 4. release Ki
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    P_full_O_rescaled_phase ^= 1
                    O_should_accumulate = True
                # End of seqlen_kv loop

                # release Q0 & Q1
                for stage in cutlass.range(self.q_stage):
                    pipeline_q.consumer_release_w_index(stage)

                # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                # 1. wait for V0
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                tOrVi = tOrV[None, None, None, Vi_index]
                for stage in cutlass.range_constexpr(self.q_stage):
                    # Pre-acquire the S slot for the next persistent work tile,
                    # then wait for this tile's P/O inputs.
                    pipeline_s_p_o.producer_acquire_w_index_phase(
                        stage, P_full_O_rescaled_phase
                    )
                    pipeline_p_lastsplit.consumer_wait_w_index_phase(
                        stage, P_full_O_rescaled_phase
                    )
                    # 3. gemm
                    # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                    # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                    sV_cur = sV[None, None, None, Vi_index]
                    if const_expr(self.uneven_kv_smem):
                        sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                    gemm_Pi[stage](
                        tCrB=tOrVi,
                        sB=sV_cur,
                        # smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sV_cur.iterator),
                        zero_init=not O_should_accumulate,
                        mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(stage) if self.split_P_arrive > 0 else None,
                        mbar_phase=P_full_O_rescaled_phase,
                    )
                    pipeline_p_lastsplit.consumer_release_w_index(stage)
                    # 4. release accumulated O0_partial
                    # We do need O_full here since for the last tile, by the time the softmax warp
                    # has signaled to the correction warps, the softmax warp has just finished
                    # computing the row sum of the current tile. It does not guarantee that the 1st
                    # tile of the next work tile has been computed yet.
                    pipeline_o_acc.producer_commit_w_index(stage)
                    # End of GEMM_PV00 (P0 * V0 -> O0_partial)
                P_full_O_rescaled_phase ^= 1
                # 5. release Vi_end
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM_PV1(i_end) (P1 * Vi_end -> O1)

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop

        # We don't need pipeline_s_p_o.producer_tail() since there's no dangling mbarrier at the end
        # pipeline_s_p_o.producer_acquire_w_index_phase(self.q_stage - 1, P_full_O_rescaled_phase)
        # We don't need pipeline_o_acc.producer_tail() since we don't call
        # pipeline_o_acc.producer_acquire() inside the loop.

    # for both softmax0 and softmax1 warp group
    @cute.jit
    def _kv_head_idx(self, head_idx: Int32) -> Int32:
        """Map query-head tile index -> KV-head index (FA3 descale semantics)."""
        if cutlass.const_expr(self.pack_gqa):
            return head_idx
        return head_idx // self.qhead_per_kvhead

    @cute.jit
    def _load_effective_descales(
        self,
        descale_tensors: Optional[DescaleTensors],
        batch_idx: Int32,
        kv_head_idx: Int32,
    ) -> Tuple[Float32, Float32]:
        """Load effective QK and V descales, defaulting unspecified tensors to identity."""
        qk_descale = Float32(1.0)
        v_descale = Float32(1.0)
        if cutlass.const_expr(descale_tensors is not None):
            if cutlass.const_expr(descale_tensors.q_descale is not None):
                qk_descale = qk_descale * Float32(descale_tensors.q_descale[batch_idx, kv_head_idx])
            if cutlass.const_expr(descale_tensors.k_descale is not None):
                qk_descale = qk_descale * Float32(descale_tensors.k_descale[batch_idx, kv_head_idx])
            if cutlass.const_expr(descale_tensors.v_descale is not None):
                v_descale = Float32(descale_tensors.v_descale[batch_idx, kv_head_idx])
        return qk_descale, v_descale

    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        descale_tensors: Optional[DescaleTensors],
        thr_mma_qk: cute.ThrMma,
        sP: Optional[cute.Tensor],
        tStS: cute.Tensor,  # ((TILE_M, TILE_N), 1, 1, q_stage)
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_s0_s1_sequence: Optional[pipeline.PipelineAsync],
        learnable_sink: Optional[cute.Tensor],
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        tile_scheduler=None,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.
        """
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE
            # * (len(self.softmax0_warp_ids) if stage == 0 else len(self.softmax1_warp_ids)
            * (len(self.softmax0_warp_ids))
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        cta_qk_tiler = (self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape, self.mma_tiler_qk[1])
        tSAcc = tStS[(None, None), 0, 0, stage]  # (128, 128)
        tStScale = cute.composition(tSAcc, cute.make_layout((self.m_block_size, 1)))
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))

        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tStP_layout = cute.composition(
            tSAcc.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )
        tStP = cute.make_tensor(tSAcc.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.LdRed32x32bOp(
                tcgen05.copy.Repetition(32),
                redOp=tcgen05.TmemLoadRedOp.MAX,
            ),
            self.qk_acc_dtype,
        )
        tmem_load_tiled = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc)
        thr_tmem_load = tmem_load_tiled.get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)  # (((32,32),1),1,4)

        smem_store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.q_dtype, num_bits_per_copy=128
        )
        smem_store_tiled = cute.make_tiled_copy_D(smem_store_atom, tmem_load_tiled)
        thr_smem_store = smem_store_tiled.get_slice(tidx)
        sP_mnk = cute.composition(
            sP,
            cute.make_ordered_layout(
                (self.m_block_size, self.n_block_size, self.s_stage),
                order=(0, 1, 2),
            ),
        )
        tStP_r2s = thr_smem_store.partition_D(sP_mnk)

        tmem_store_scale_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)), Float32
        )
        thr_tmem_store_scale = tcgen05.make_tmem_copy(tmem_store_scale_atom, tStScale).get_slice(
            tidx
        )
        tStScale_r2t = thr_tmem_store_scale.partition_D(tStScale)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(8)),
            Float32,
        )
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)  # (((16,32),1),1,4)

        mma_si_consumer_phase = Int32(0)
        sm_stats_producer_phase = Int32(1)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)

        # self.warp_scheduler_barrier_init()

        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            kv_head_idx = self._kv_head_idx(head_idx)
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen,
                self._kv_range_m_block(m_block),
                split_idx,
                num_splits,
            )

            mask = AttentionMaskCls(seqlen)
            shared_mask_kwargs = dict(
                m_block=(self.q_stage * m_block + stage) * self.cta_group_size,
                thr_mma=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                batch_idx=batch_idx,
                head_idx=head_idx,
                aux_data=aux_data,
                vec_size=self.mask_vec_size,
            )

            mask_fn = partial(
                mask.apply_mask_sm100,
                mask_mod=None,
                fastdiv_mods=None,
                head_divmod=None,
                **shared_mask_kwargs,
            )

            qk_descale, _ = self._load_effective_descales(descale_tensors, batch_idx, kv_head_idx)

            max_offset = self.fp8_p_prescale_log2
            softmax_scale_log2_eff = softmax_scale_log2 * qk_descale

            softmax = SoftmaxSm103K3.create(
                softmax_scale_log2_eff,
                rescale_threshold=self.fp8_rescale_threshold,
                softmax_scale=None,
                max_offset=max_offset,
            )
            softmax.reset()

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                thr_mma_qk=thr_mma_qk,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                pipeline_s0_s1_sequence=pipeline_s0_s1_sequence,
                tiled_tmem_load=tmem_load_tiled,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                thr_smem_store=thr_smem_store,
                thr_tmem_store_scale=thr_tmem_store_scale,
                tStS_t2r=tStS_t2r,
                tStScale_r2t=tStScale_r2t,
                tStP_r2t=tStP_r2t,
                tStP_r2s=tStP_r2s,
                sScale=sScale,
                stage=stage,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=(self.q_stage * m_block + stage) * self.cta_group_size,
                seqlen=seqlen,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
            )

            pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
            sm_stats_producer_phase ^= 1

            mma_si_consumer_phase, sm_stats_producer_phase, s0_s1_sequence_phase = softmax_step(
                mma_si_consumer_phase,
                sm_stats_producer_phase,
                s0_s1_sequence_phase,
                n_block_max - 1,
                is_first=True,
                mask_fn=partial(mask_fn, mask_seqlen=True),
            )
            n_block_max -= 1
            n_block_min_causal_mask = block_info.get_n_block_min_causal_local_mask(
                seqlen, m_block, n_block_min
            )
            for n_tile in cutlass.range(n_block_max - n_block_min_causal_mask, unroll=1):
                n_block = n_block_max - 1 - n_tile
                mma_si_consumer_phase, sm_stats_producer_phase, s0_s1_sequence_phase = (
                    softmax_step(
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        s0_s1_sequence_phase,
                        n_block,
                        mask_fn=partial(mask_fn, mask_seqlen=False),
                    )
                )
            n_block_max = cutlass.min(n_block_max, n_block_min_causal_mask)
            n_block_min_unmasked = block_info.get_n_block_min_before_local_mask(
                seqlen, m_block, n_block_min
            )
            for n_tile in cutlass.range(n_block_max - n_block_min_unmasked, unroll=1):
                n_block = n_block_max - n_tile - 1
                mma_si_consumer_phase, sm_stats_producer_phase, s0_s1_sequence_phase = (
                    softmax_step(
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        s0_s1_sequence_phase,
                        n_block,
                    )
                )

            pipeline_sm_stats.producer_acquire_w_index_phase(
                stage, sm_stats_producer_phase
            )
            sm_stats_producer_phase ^= 1
            sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
            if const_expr(mLSE is not None):
                sScale[
                    tidx + stage * self.m_block_size + self.q_stage * self.m_block_size
                ] = softmax.row_max[0]
            sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

            # # Write LSE to gmem
            # if const_expr(mLSE is not None):
            #     acc_O_mn_row_is_zero_or_nan = softmax.row_sum[0] == 0.0 or softmax.row_sum[0] != softmax.row_sum[0]
            #     scale = (
            #         cute.arch.rcp_approx(softmax.row_sum[0] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            #     )
            #     LN2 = math.log(2.0)
            #     lse = (
            #         (softmax.row_max[0] * softmax.scale_log2 + cute.math.log2(softmax.row_sum[0], fastmath=True)) * LN2
            #         if not acc_O_mn_row_is_zero_or_nan else -Float32.inf
            #     )
            #     if const_expr(not seqlen.has_cu_seqlens_q):
            #         mLSE_cur = mLSE[None, head_idx, batch_idx]
            #     else:
            #         mLSE_cur = cute.domain_offset((seqlen.offset_q,), mLSE[None, head_idx])
            #     gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_block * 2 + stage,))
            #     if tidx < seqlen.seqlen_q - (m_block * 2 + stage) * self.m_block_size:
            #         gLSE[tidx] = lse

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_sm_stats.producer_tail
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
        # This is equivalent to pipeline_s0_s1.producer_tail
        if const_expr(self.s0_s1_barrier):
            if stage == 0:
                pipeline_s0_s1_sequence.sync_object_full.wait(stage, s0_s1_sequence_phase)

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm103K3,
        thr_mma_qk: cute.ThrMma,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_s0_s1_sequence: Optional[pipeline.PipelineAsync],
        tiled_tmem_load: cute.TiledCopy,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_smem_store: Optional[cute.CopyAtom],
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        tStP_r2s: Optional[cute.Tensor],
        sScale: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)
        # tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        cta_qk_tiler = (self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape, self.mma_tiler_qk[1])
        tScS_shape = cta_qk_tiler  # (128, 128)
        tScP_shape = (tScS_shape[0], tilePlikeFP32)  # (128, 64)

        # Wait for Si
        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_rmem_tensor(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype)
        tSrS_max = cute.make_rmem_tensor(
            cute.make_layout((1, cute.size(tSrS_t2r, mode=[2]))),
            self.qk_acc_dtype,
        )
        for i in cutlass.range_constexpr(cute.size(tSrS_t2r, mode=[2])):
            cute.copy_atom_call(
                tiled_tmem_load,
                tStS_t2r[None, 0, i],
                (tSrS_t2r[None, 0, i], tSrS_max[None, i]),
            )
        # QK may reuse this tmem S slot once the complete tile is in registers.
        cute.arch.fence_view_async_tmem_load()
        pipeline_s_p_o.consumer_release_w_index(stage)
        # tSrS_t2r = copy_utils.load_t2r(thr_tmem_load, tScS_shape, tStS_t2r)
        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)
        if const_expr(
            self.tmem_ldred_max
            and self.score_mod is None
            and mask_fn is None
        ):
            tile_row_max = tSrS_max.load().reduce(
                cute.ReductionOp.MAX, -cutlass.Float32.inf, 0
            )
            row_max_new = (
                tile_row_max
                if const_expr(is_first)
                else cute.arch.fmax(softmax.row_max[0], tile_row_max)
            )
            row_max, acc_scale = softmax.update_row_max_from_local(
                row_max_new, is_first
            )
        else:
            row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

        if const_expr(not is_first):
            if const_expr(self.stats_late_acquire):
                pipeline_sm_stats.producer_acquire_w_index_phase(
                    stage, sm_stats_producer_phase
                )
                sm_stats_producer_phase ^= 1
            # tSrScale_r2t = cute.make_rmem_tensor(thr_tmem_store_scale.partition_S(tScScale).shape, Float32)
            # tSrScale_r2t[0] = acc_scale
            # cute.copy(thr_tmem_store_scale, tSrScale_r2t, tStScale_r2t)
            # cute.arch.fence_view_async_tmem_store()
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
            # if thread_idx == 0: cute.printf("softmax acc_scale stage %d: %f, row_max = %f\n", stage, acc_scale, row_max)
        # Notify correction wg that row_max is ready
        # pipeline_sm_stats.producer_commit_w_index(stage)
        sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

        pipeline_p_lastsplit.producer_acquire_w_index_phase(
            stage, mma_si_consumer_phase ^ 1
        )

        # if thread_idx == 0 and stage == 0: cute.print_tensor(tSrS_t2r)
        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        # Sequence barrier wait
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.wait(stage, s0_s1_sequence_phase)
        tSrP_r2t_f32 = cute.make_rmem_tensor(
            thr_tmem_store.partition_S(cute.make_identity_tensor(tScP_shape)).shape, Float32
        )
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype), tSrS_t2r.layout
        )
        # softmax.scale_apply_exp2_convert(tSrS_t2r, row_max, tSrP_r2t)
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            ex2_emu_freq=self.ex2_emu_freq,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )
        # Sequence barrier arrive
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.arrive(1 - stage, dst=None)
        rP_smem_view = thr_smem_store.retile(tSrP_r2t)
        cute.copy(
            thr_smem_store,
            rP_smem_view,
            tStP_r2s[None, None, None, stage],
        )
        cute.arch.fence_view_async_shared()
        pipeline_p_lastsplit.producer_commit_w_index(stage)
        if const_expr(not self.stats_late_acquire):
            pipeline_sm_stats.producer_acquire_w_index_phase(
                stage, sm_stats_producer_phase
            )
            sm_stats_producer_phase ^= 1
        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        # acc_scale = cute.math.exp2(acc_scale_, fastmath=True)
        return mma_si_consumer_phase ^ 1, sm_stats_producer_phase, s0_s1_sequence_phase ^ 1

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_o_epi: pipeline.PipelineAsync,
        learnable_sink: Optional[cute.Tensor],
        descale_tensors: Optional[DescaleTensors],
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        tile_scheduler=None,
        output_scale: Optional[cute.Tensor] = None,
    ):
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mma_tile_coord_v = thr_mma_qk.thr_idx

        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tStScale_layout = cute.composition(tStS.layout, cute.make_layout((self.m_block_size, 1)))
        tStScales = tuple(
            cute.make_tensor(tStS.iterator + self.tmem_vec_offset[stage], tStScale_layout)
            for stage in range(self.q_stage)
        )
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        tmem_load_v_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)), self.qk_acc_dtype
        )
        thr_tmem_load_vec = tcgen05.make_tmem_copy(tmem_load_v_atom, tStScales[0]).get_slice(tidx)

        tStScales_t2r = [thr_tmem_load_vec.partition_S(tStScales[stage]) for stage in range(self.q_stage)]
        tSrScale_t2r_shape = thr_tmem_load_vec.partition_D(tScScale).shape

        # Load FP8 output scale and invert in-kernel.
        if const_expr(self.output_quant_key == "kFp8StaticTensorSym"):
            output_scale_inv = Float32(1.0) / Float32(output_scale[0])

        p_o_producer_phase = Int32(1)
        # First iter: no correction is required. Seed the O half of the
        # P-ready/O-credit rendezvous for each stage.
        for stage in cutlass.range(self.q_stage):
            pipeline_p_lastsplit.producer_acquire_w_index_phase(
                stage, p_o_producer_phase
            )
            pipeline_p_lastsplit.producer_commit_w_index(stage)
        p_o_producer_phase ^= 1

        sm_stats_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            kv_head_idx = self._kv_head_idx(head_idx)
            qk_descale, v_descale = self._load_effective_descales(descale_tensors, batch_idx, kv_head_idx)
            softmax_scale_log2_eff = softmax_scale_log2 * qk_descale

            max_offset = Float32(float(self.fp8_p_prescale_log2))
            max_offset_scale = Float32(float(2**self.fp8_p_prescale_log2))
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen,
                self._kv_range_m_block(m_block),
                split_idx,
                num_splits,
            )

            mO_cur = seqlen.offset_batch_Q(
                mO, batch_idx, dim=3, ragged=self.ragged_tma_O
            )[None, None, head_idx]
            tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
            gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))
            gO = layout_utils.select(
                cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
            )
            gO = cute.flat_divide(
                gO, (self.mma_tiler_pv[0] // self.cta_group_size,)
            )[None, mma_tile_coord_v, None, None]

            # Default LSE to -inf for invalid split_idx tiles
            stats = [(0.0, -Float32.inf if const_expr(mLSE is not None or learnable_sink is not None) else None, True)] * self.q_stage

            total_block_count = n_block_max - n_block_min
            has_work = True

            if has_work:
                # Ignore first signal from softmax as no correction is required
                # pipeline_sm_stats.consumer_wait_w_index_phase(0, sm_stats_consumer_phase)
                sm_stats_barrier.arrive_and_wait_w_index(index=0 * 4 + warp_idx)
                pipeline_sm_stats.consumer_release_w_index(0)
                if const_expr(self.q_stage == 2):
                    # pipeline_sm_stats.consumer_wait_w_index_phase(1, sm_stats_consumer_phase)
                    sm_stats_barrier.arrive_and_wait_w_index(index=1 * 4 + warp_idx)
                sm_stats_consumer_phase ^= 1

                tSrScale_t2r = cute.make_rmem_tensor(tSrScale_t2r_shape, Float32)
                for i in cutlass.range(total_block_count - 1, unroll=1):
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # wait for S0 / S1
                        # pipeline_sm_stats.consumer_wait_w_index_phase(stage, sm_stats_consumer_phase)
                        sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                        # cute.copy(tiled_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                        # cute.arch.fence_view_async_tmem_load()
                        # scale = tSrScale_t2r[0]
                        scale = sScale[tidx + stage * self.m_block_size]
                        should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                        pipeline_p_lastsplit.producer_acquire_w_index_phase(
                            stage, p_o_producer_phase
                        )
                        # should_rescale = True
                        # if tidx == 0: cute.printf("Correction scale i = %d, for stage %d: %f, should_rescale = %d\n", i, stage, scale, should_rescale)
                        # Don't need O_full anymore, since by the time softmax has signaled the correction
                        # warps, S_i must have been done, so O_i-1 must have been done as well.
                        # pipeline_o_acc.consumer_wait_w_index_phase(stage, o_corr_consumer_phase)
                        if should_rescale:
                            self.correction_rescale(thr_mma_pv, tOtO[None, None, None, stage], tidx, scale)
                        # Notify mma warp that O has been rescaled
                        pipeline_p_lastsplit.producer_commit_w_index(stage)
                        pipeline_sm_stats.consumer_release_w_index(self.q_stage - 1 - stage)
                    p_o_producer_phase ^= 1
                    sm_stats_consumer_phase ^= 1
                    # o_corr_consumer_phase ^= 1
                if const_expr(self.q_stage == 2):
                    pipeline_sm_stats.consumer_release_w_index(1)
                # End of seqlen_corr_loop_steps

                # Even in the case of self.overlap_sO_sQ, we can write to stage 0 of sO without
                # additional sync because the MMA in the top half must have been done.
                # Similarly we can write to stage 1 of sO without additional sync.
                learnable_sink_val = [None] * self.q_stage
                if const_expr(learnable_sink is not None):
                    if const_expr(not self.pack_gqa):
                        sink_val = Float32(learnable_sink[head_idx])
                        learnable_sink_val = [sink_val] * self.q_stage
                    else:  # Each thread might have a different sink value due to different q_head
                        for stage in cutlass.range_constexpr(self.q_stage):
                            q_head_idx = (
                                ((m_block * self.q_stage + stage) * self.cta_group_size + mma_tile_coord_v) * self.m_block_size + tidx
                            ) % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                            learnable_sink_val[stage] = Float32(learnable_sink[q_head_idx])
                for stage in cutlass.range_constexpr(self.q_stage):
                    # pipeline_sm_stats.consumer_wait_w_index_phase(stage, sm_stats_consumer_phase)
                    sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                    # cute.copy(tiled_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                    # cute.arch.fence_view_async_tmem_load()
                    # scale = tSrScale_t2r[0]
                    row_sum = sScale[tidx + stage * self.m_block_size]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        row_max = sScale[tidx + stage * self.m_block_size + self.q_stage * self.m_block_size]
                    else:
                        row_max = None
                    pipeline_sm_stats.consumer_release_w_index(stage)
                    if const_expr(learnable_sink is not None):
                        LOG2_E = math.log2(math.e)
                        sink_val = learnable_sink_val[stage]
                        if const_expr(not self.is_split_kv) or split_idx == 0:
                            if row_max == -Float32.inf:
                                # It's possible to have an empty row with splitKV.
                                row_max = sink_val * (LOG2_E / softmax_scale_log2_eff)
                                row_sum = max_offset_scale
                            else:
                                row_sum += cute.math.exp2(
                                    sink_val * LOG2_E - row_max * softmax_scale_log2_eff + max_offset, fastmath=True
                                )
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)
                    scale = scale * v_descale
                    # Fold per-tensor output scale into the existing per-row scale.
                    if const_expr(self.output_quant_key == "kFp8StaticTensorSym"):
                        scale = scale * output_scale_inv
                    # Wait for the last O to be ready from the MMA warp
                    pipeline_o_acc.consumer_wait_w_index_phase(stage, o_corr_consumer_phase)
                    if const_expr(not self.use_correction_warps_for_epi):
                        pipeline_o_epi.producer_acquire_w_index_phase(stage, corr_epi_producer_phase)
                    gO_stage = gO[None, None, stage] if const_expr(gO is not None) else None
                    self.correction_epilogue(
                        thr_mma_pv,
                        tOtO[None, None, None, stage],
                        tidx,
                        stage,
                        m_block,
                        seqlen.seqlen_q,
                        scale,
                        sO[None, None, stage],
                        mO_cur,
                        gO_stage,
                        gmem_tiled_copy_O,
                    )
                    # Signal for the next work tile that O buffers in tmem are already read, so
                    # mma warp can write to them
                    pipeline_p_lastsplit.producer_acquire_w_index_phase(
                        stage, p_o_producer_phase
                    )
                    pipeline_p_lastsplit.producer_commit_w_index(stage)
                    if const_expr(not self.use_correction_warps_for_epi):
                        pipeline_o_epi.producer_commit_w_index(stage)
                    # if tidx == 0: cute.printf("Correction final scale for stage %d: %f\n", stage, scale)

                p_o_producer_phase ^= 1
                o_corr_consumer_phase ^= 1
                sm_stats_consumer_phase ^= 1
                corr_epi_producer_phase ^= 1
            if const_expr(mLSE is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    if const_expr(self.is_split_kv):
                        mLSE_cur = mLSE[None, head_idx, batch_idx, split_idx]
                    else:
                        mLSE_cur = mLSE[None, head_idx, batch_idx]
                else:
                    offset = (
                        seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                    )
                    if const_expr(self.is_split_kv):
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx, split_idx])
                    else:
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx])

                for stage in cutlass.range_constexpr(self.q_stage):
                    m_tile_idx = (m_block * self.q_stage + stage) * self.cta_group_size + mma_tile_coord_v
                    row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
                    LN2 = math.log(2.0)
                    lse = (
                        (row_max * softmax_scale_log2_eff + (cute.math.log2(row_sum, fastmath=True) - max_offset)) * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    seqlen_q = (
                        seqlen.seqlen_q
                        if const_expr(not self.pack_gqa)
                        else seqlen.seqlen_q * self.qhead_per_kvhead
                    )
                    if const_expr(
                        not self.pack_gqa
                        or self.m_block_size % self.qhead_per_kvhead == 0
                    ):
                        gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_tile_idx,))
                        if tidx < seqlen_q - m_tile_idx * self.m_block_size:
                            gLSE[tidx] = lse
                    else:
                        idx = m_tile_idx * self.m_block_size + tidx
                        if idx < seqlen_q:
                            m_idx = idx // self.qhead_per_kvhead
                            h_idx = idx - m_idx * self.qhead_per_kvhead
                            lse_ptr_i64 = utils.elem_pointer(
                                mLSE_cur, ((h_idx, m_idx),)
                            ).toint()
                            lse_gmem_ptr = cute.make_ptr(
                                mLSE_cur.element_type,
                                lse_ptr_i64,
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            )
                            cute.make_tensor(lse_gmem_ptr, (1,))[0] = lse

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_o_epi.consumer_tail() for the correction warps
        if const_expr(not self.use_correction_warps_for_epi):
            pipeline_o_epi.producer_acquire_w_index_phase(self.q_stage - 1, corr_epi_producer_phase)

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        This method performs a crucial correction step in the attention computation pipeline.
        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.

        The implementation uses efficient tensor memory operations to:
        1. Load existing partial attention output from tensor memory
        2. Apply the scaling factor to all elements
        3. Store the rescaled results back to tensor memory
        """
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size = 16  # tuneable parameter
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)), self.pv_acc_dtype
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tOtO_i = cute.composition(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.composition(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i).get_slice(tidx)
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i).get_slice(tidx)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
        tOrO_t2r_shape = thr_tmem_load.partition_D(tOcO_i).shape
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i)

        frg_count = self.head_dim_v_padded // corr_tile_size
        tOrO_frg = cute.make_rmem_tensor((tOrO_t2r_shape, frg_count), self.pv_acc_dtype)
        for i in cutlass.range_constexpr(frg_count):
            tOrO_frg = cute.make_rmem_tensor(tOrO_t2r_shape, self.pv_acc_dtype)
            tOtO_t2r_i = cute.make_tensor(tOtO_t2r.iterator + i * corr_tile_size, tOtO_t2r.layout)
            cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                )
            tOtO_r2t_i = cute.make_tensor(tOtO_r2t.iterator + i * corr_tile_size, tOtO_r2t.layout)
            cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        stage: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        scale: Float32,
        sO: cute.Tensor,
        mO_cur: Optional[cute.Tensor] = None,
        gO: Optional[cute.Tensor] = None,
        gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
    ):
        """Apply final scaling and transformation to attention output before writing to global memory.

        This correction_epilogue function handles the final processing step for attention output values.
        It applies a scaling factor to the accumulated attention results and prepares the
        data for efficient transfer back to global memory.

        The method performs:
        1. Loading of accumulated attention results from tensor memory
        2. Application of the final output scaling factor
        3. Type conversion if necessary (typically from higher precision accumulator to output precision)
        4. Reorganization of data for optimal memory access patterns
        5. Preparation for efficient TMA store operations

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.ThrMma
        :param tOtO: Tensor containing accumulated attention output
        :type tOtO: cute.Tensor
        :param scale: Final scaling factor to apply to the output
        :type scale: Float32
        :param sO: Shared memory tensor for the final output
        :type sO: cute.Tensor
        """

        corr_tile_size = 8 * 32 // self.o_dtype.width
        # Use CTA 0 mapping for smem partitioning since sO is per-CTA sized
        tOsO = thr_mma.get_slice(0).partition_C(sO)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((self.m_block_size, corr_tile_size)))

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=self.use_2cta_instrs,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0])
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOsO_s2r = copy_utils.partition_D_position_independent(thr_tmem_load, tOsO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        for i in cutlass.range(self.head_dim_v_padded // corr_tile_size, unroll_full=True):
            tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            tOrO_frg = cute.make_rmem_tensor(tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                )
            copy_utils.cvt_copy(tiled_smem_store, tOrO_frg, tOsO_r2s_i)
        cute.arch.fence_view_async_shared()

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        pipeline_o_epi: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: int,
        SeqlenInfoCls: Callable,
        mma_tile_coord_v: Int32 = 0,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        tile_scheduler=None,
    ):
        epi_consumer_phase = Int32(0)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
            mO_cur = seqlen.offset_batch_Q(
                mO, batch_idx, dim=3, ragged=self.ragged_tma_O
            )[None, None, head_idx]
            tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
            gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))
            gO = layout_utils.select(
                cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
            )
            gO = cute.flat_divide(
                gO, (self.mma_tiler_pv[0] // self.cta_group_size,)
            )[None, mma_tile_coord_v, None, None]

            store_O, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_O, 0, cute.make_layout(1), sO, gO
            )
            for stage in cutlass.range(self.q_stage, unroll_full=True):
                pipeline_o_epi.consumer_wait_w_index_phase(stage, epi_consumer_phase)
                store_O(src_idx=stage, dst_idx=stage)
                cute.arch.cp_async_bulk_commit_group()
            for stage in cutlass.range_constexpr(self.q_stage):
                cute.arch.cp_async_bulk_wait_group(self.q_stage - 1 - stage, read=True)
                pipeline_o_epi.consumer_release_w_index(stage)

            epi_consumer_phase ^= 1

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

    def load_Q(
        self,
        load_Q_fn: Callable,
        pipeline_q: pipeline.PipelineAsync,
        block: Int32,
        stage: int,
        phase: Int32,
    ):
        pipeline_q.producer_acquire_w_index_phase(stage, phase)
        load_Q_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(stage))

    @cute.jit
    def load_KV(
        self,
        tma_atom: Optional[cute.CopyAtom],
        tXgX: Optional[cute.Tensor],
        tXsX: Optional[cute.Tensor],
        block: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        mcast_mask: Optional[Int32] = None,
    ):
        assert K_or_V in ("K", "V")
        stage, phase = producer_state.index, producer_state.phase
        extra_tx_count_kv = self.tma_copy_bytes[K_or_V] - self.tma_copy_bytes["K"]
        pipeline_kv.producer_acquire(
            producer_state, extra_tx_count=extra_tx_count_kv
        )
        if const_expr(K_or_V == "K" and self.uneven_kv_smem):
            # Before this round, the smem location was occupied by V, which is smaller than
            # K. So we need to wait for the stage after that (stage 1) to be empty as well.
            if stage == 0:
                pipeline_kv.sync_object_empty.wait(1, phase)

        assert tXgX is not None and tXsX is not None and tma_atom is not None
        tXsX_cur = tXsX[None, stage]
        if const_expr(self.uneven_kv_smem):
            # Since this is the producer_state, the phase starts at 1, so we have to invert it
            tXsX_cur = self.offset_kv_smem(tXsX_cur, stage, phase ^ 1)
        tXgX_cur = tXgX[None, block]
        copy_kwargs = (
            {"mcast_mask": mcast_mask}
            if const_expr(self.use_causal_kv_multicast)
            else {}
        )
        cute.copy(
            tma_atom,
            tXgX_cur,
            tXsX_cur,
            tma_bar_ptr=pipeline_kv.producer_get_barrier(producer_state),
            **copy_kwargs,
        )

    @cute.jit
    def offset_kv_smem(self, sX: cute.Tensor, stage: Int32, phase: Int32):
        if const_expr(self.uneven_kv_smem):
            # smem layout is [smem_large, smem_small, smem_large], and the current stride is
            # (smem_large + smem_small) // 2. So for stage == 1, move right by offset if
            # phase == 0, or left by offset if phase == 1.
            offset = 0 if stage != 1 else self.uneven_kv_smem_offset * (1 - 2 * phase)
            # Hint that the offset is 128-bit aligned so that
            # ptr + offset preserves the alignment needed by cp.async.
            offset = cute.assume(offset, divby=128 // self.k_dtype.width)
            return cute.make_tensor(sX.iterator + offset, sX.layout)
        else:
            return sX
