/******************************************************************************
 * Copyright (c) 2025, Tri Dao, Samsung SDSA.
 ******************************************************************************/

#pragma once

#include "flash.h"

namespace FLASH_NAMESPACE {

struct Flash_fwd_params_tree : public Flash_fwd_params {
    uint64_t * __restrict__ tree_mask_ptr; // For dynamically masking in tree attention.
    int * __restrict__ tree_mask_lens_ptr; // The length of each of the speculated sequences (batch_size + 1).
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Headdim> void run_mha_fwd_tree_(Flash_fwd_params_tree &params, cudaStream_t stream);
template<typename T, int Headdim> void run_mha_fwd_splitkv_dispatch_tree(Flash_fwd_params_tree &params, cudaStream_t stream);

}  // namespace FLASH_NAMESPACE