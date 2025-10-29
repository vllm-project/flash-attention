#pragma once

#include <cute/tensor.hpp>

namespace flash {

/**
 * GmmaPromotionHelper: Fixes FP8 tensor core accumulation precision loss
 * 
 * Problem: FP8 tensor cores do not use full FP32 accumulators,
 * causing precision loss particularly if the contraction dimension is large.
 * 
 * Solution: Accumulate a few tiles <promotion_interval> in temporary fragment, then promote
 * to main accumulator using true FP32 arithmetic every promotion_interval tiles.
 * 
 * This significantly reduces the accuracy drop on very long context datasets.
 */
template<typename AccumFragment>
struct GmmaPromotionHelper {
    AccumFragment& accum_main;
    AccumFragment accum_temp;
    int mma_count = 0;
    static constexpr int promotion_interval = 2;  // Tunable: balance precision vs performance
    
    __device__ GmmaPromotionHelper(AccumFragment& main_accum) 
        : accum_main(main_accum), accum_temp(cute::make_fragment_like(main_accum)) {
        cute::clear(accum_temp);
    }
    
    template<typename MmaFunc>
    __device__ void step(MmaFunc&& mma_func) {
        mma_func(accum_temp);
        mma_count++;
        if (mma_count >= promotion_interval) {
            promote();
        }
    }
    
    __device__ void promote() {
        // Add temp accumulator to main accumulator (FP32 precision)
        #pragma unroll
        for (int i = 0; i < cute::size(accum_main); ++i) {
            accum_main(i) += accum_temp(i);
        }
        cute::clear(accum_temp);
        mma_count = 0;
    }
    
    __device__ void flush() {
        if (mma_count > 0) {
            promote();
        }
    }
};

} // namespace flash
