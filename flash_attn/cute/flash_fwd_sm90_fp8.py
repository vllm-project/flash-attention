# Copyright (c) 2026, The FlashAttention Authors.

"""Hopper E4M3 forward policy for static per-tensor descales."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, const_expr
from cutlass.cute.nvgpu import warpgroup
import cutlass.utils.hopper_helpers as sm90_utils

from flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
from flash_attn.cute.named_barrier import NamedBarrierFwd


class FlashAttentionForwardSm90Fp8(FlashAttentionForwardSm90):
    """E4M3 Q/K/V with BF16 output on the existing SM90 control plane.

    Scheduling, paging, SplitKV, masking, and epilogue ownership stay in
    :class:`FlashAttentionForwardSm90`; this policy owns only the D128 mixed
    input/output types and Hopper's FP8 PV operand permutation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.dtype != cutlass.Float8E4M3FN:
            raise TypeError("SM90 FP8 policy requires E4M3FN Q/K/V")
        if self.tile_hdim != 128 or self.tile_hdimv != 128:
            raise ValueError("SM90 FP8 requires D == Dv == 128")
        if self.has_qv:
            raise ValueError("SM90 FP8 does not support Qv")
        self.output_dtype = BFloat16
        self.has_static_descales = True
        # Shift softmax probabilities into E4M3's [0, 448] range.  The inverse
        # is folded into the final output normalization; LSE remains unshifted.
        self.output_acc_scale = 1.0 / 256.0
        self.transpose_v = True
        self.intra_wg_overlap = False
        self.use_paged_kv_overlap = False

    def _get_tiled_mma(self):
        tiled_mma_qk = sm90_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_n),
        )
        tiled_mma_pv = sm90_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_hdimv),
            a_source=warpgroup.OperandSource.RMEM,
        )
        return tiled_mma_qk, tiled_mma_pv, tiled_mma_qk

    @cute.jit
    def _prepare_v_for_mma(
        self,
        source: cute.Tensor,
        destination: cute.Tensor,
        stage: Int32,
        tidx: Int32,
    ) -> None:
        """Physically transpose D-contiguous V into WGMMA K-major storage."""
        elements = self.tile_n * self.tile_hdimv
        assert elements % self.num_mma_threads == 0
        for i in cutlass.range_constexpr(elements // self.num_mma_threads):
            linear_idx = tidx + i * self.num_mma_threads
            n_idx = linear_idx // self.tile_hdimv
            d_idx = linear_idx - n_idx * self.tile_hdimv
            n_group = (n_idx // 16) * 16
            n_in_group = n_idx - n_group
            # One-hot PV probes with identity placement observed output token n
            # reading physical V lane [0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15][n].
            # The inverse placement writes logical V[n] into that physical lane.
            destination_n_idx = (
                n_group
                + (n_in_group & 0x1)
                + ((n_in_group & 0x2) << 1)
                + ((n_in_group & 0x4) << 1)
                + ((n_in_group & 0x8) >> 2)
            )
            source_coord = (
                cute.idx2crd(n_idx, source.shape[0]),
                cute.idx2crd(d_idx, source.shape[1]),
                cute.idx2crd(stage, source.shape[2]),
            )
            destination_coord = (
                cute.idx2crd(d_idx, destination.shape[0]),
                cute.idx2crd(destination_n_idx, destination.shape[1]),
                cute.idx2crd(stage, destination.shape[2]),
            )
            destination[destination_coord] = source[source_coord]
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(
            # FP8 is RS-PV without the asymmetric P handoff, so PFull is free.
            barrier_id=int(NamedBarrierFwd.PFull),
            number_of_threads=self.num_mma_threads,
        )

    def _check_type(
        self,
        mQ_type,
        mK_type,
        mV_type,
        mO_type,
        mLSE_type,
        mCuSeqlensQ_type,
        mCuSeqlensK_type,
        mSeqUsedQ_type,
        mSeqUsedK_type,
        is_split_kv=False,
    ):
        if const_expr(not (mQ_type == mK_type == mV_type == self.dtype)):
            raise TypeError("SM90 FP8 Q, K, and V must all be E4M3FN")
        expected_o_type = Float32 if is_split_kv else self.output_dtype
        if const_expr(mO_type != expected_o_type):
            raise TypeError(
                "SM90 FP8 output must be BF16 (or FP32 for SplitKV partials)"
            )
        if const_expr(mLSE_type not in [None, Float32]):
            raise TypeError("LSE tensor must be Float32")
        for tensor_type, name in (
            (mCuSeqlensQ_type, "cu_seqlens_q"),
            (mCuSeqlensK_type, "cu_seqlens_k"),
            (mSeqUsedQ_type, "seqused_q"),
            (mSeqUsedK_type, "seqused_k"),
        ):
            if const_expr(tensor_type not in [None, Int32]):
                raise TypeError(f"{name} tensor must be Int32")

    @cute.jit
    def _convert_acc_to_p(self, acc: cute.Tensor, operand: cute.Tensor) -> None:
        """Convert range-shifted FP32 probabilities to E4M3."""
        operand.store((acc.load() * 256.0).to(self.dtype))

    @cute.jit
    def _reshape_acc_to_p(
        self, acc: cute.Tensor, operand: cute.Tensor
    ) -> cute.Tensor:
        frag_64 = cute.group_modes(
            cute.recast_tensor(acc, cutlass.Uint64), 1, 3
        )
        # One-hot C-register probes derive the RS-PV operand transform: exchange
        # (0,1,2i) with (0,0,2i+1), then reinterpret in the operand layout.
        for mi in cutlass.range_constexpr(cute.size(frag_64, mode=[1])):
            for i in cutlass.range_constexpr(cute.size(frag_64.shape[0][2]) // 2):
                lhs = frag_64[(0, 1, 2 * i), mi]
                rhs = frag_64[(0, 0, 2 * i + 1), mi]
                frag_64[(0, 1, 2 * i), mi] = rhs
                frag_64[(0, 0, 2 * i + 1), mi] = lhs
        return cute.make_tensor(acc.iterator, operand.layout)
