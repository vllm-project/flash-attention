/******************************************************************************
 * Copyright (c) 2025, Tri Dao, Samsung SDSA.
 ******************************************************************************/

#pragma once
#include "namespace_config.h"

#include <cute/tensor.hpp>

namespace FLASH_NAMESPACE {

using namespace cute;


template <bool Has_alibi>
struct TreeMask {

    const int max_seqlen_k, max_seqlen_q;
    const float alibi_slope;

    __forceinline__ __device__ TreeMask(const int max_seqlen_k, const int max_seqlen_q,
                                        const float alibi_slope=0.f)
        : max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q)
        , alibi_slope(!Has_alibi ? 0.0 : alibi_slope) {
    };

    // Causal_mask: whether this particular iteration needs causal masking
    template <bool Causal_mask=false, bool Is_even_MN=true, bool Tree_mask=false, typename Engine, typename Layout, typename MaskEngine, typename MaskLayout>
    __forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor_,
                                               const int col_idx_offset_,
                                               const int row_idx_offset,
                                               const int warp_row_stride,
                                               const Tensor<MaskEngine, MaskLayout> &mask) {
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
        static constexpr bool Need_masking = Has_alibi || Causal_mask || !Is_even_MN;
        // if (cute::thread0()) { printf("Has_alibi = %d, Causal_mask=%d, Is_even_MN = %d, Need_masking = %d\n", Has_alibi, Causal_mask, Is_even_MN, Need_masking); }
        if constexpr (Need_masking) {
            // Reshape tensor_ from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
            Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));
            // Do we need both row and column indices, or just column incides?
            // if (thread0()) {
            //     print(tensor_.layout()); print("\n");
            //     print(tensor.layout()); print("\n");
            // }
            static constexpr bool Col_idx_only = !Causal_mask;
            const int col_idx_offset = col_idx_offset_ + (threadIdx.x % 4) * 2;
            if constexpr (Col_idx_only) {
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j;
                        #pragma unroll
                        for (int mi = 0; mi < size<0>(tensor); ++mi) {
                            // No causal, no local
                            if constexpr (Has_alibi) {
                                tensor(mi, make_coord(j, nj)) += alibi_slope * col_idx;
                            }
                            if constexpr (!Is_even_MN) {
                                if (col_idx >= max_seqlen_k) { tensor(mi, make_coord(j, nj)) = -INFINITY; }
                            }
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                    const int row_idx_base = row_idx_offset + mi * warp_row_stride;
                    #pragma unroll
                    for (int i = 0; i < size<0, 0>(tensor); ++i) {
                        const int row_idx = row_idx_base + i * 8;
                        const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q);
                        #pragma unroll
                        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                            const int col_idx_base = col_idx_offset + nj * 8;
                            #pragma unroll
                            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                                const int col_idx = col_idx_base + j;
                                if constexpr (Has_alibi) {
                                    tensor(make_coord(i, mi), make_coord(j, nj)) += alibi_slope * col_idx;
                                }
                                if constexpr (Causal_mask) {
                                    if (col_idx >= col_idx_limit_right) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                    }
                                    if constexpr (Tree_mask) {
                                        if (row_idx >= (max_seqlen_q - size<0>(mask)) && row_idx < max_seqlen_q && col_idx >= (max_seqlen_k - size<0>(mask)) && col_idx < col_idx_limit_right) {
                                            tensor(make_coord(i, mi), make_coord(j, nj)) += mask(make_coord(row_idx + size<0>(mask)-max_seqlen_q)) & (1ULL << (max_seqlen_k-1-col_idx)) ? 0 : -INFINITY;
                                        }
                                    }
                                }
                                if constexpr (!Causal_mask && !Is_even_MN) {
                                    // Causal and Local already handles MN masking
                                    if (col_idx >= max_seqlen_k) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };
};

} // FLASH_NAMESPACE