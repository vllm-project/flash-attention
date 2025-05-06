/******************************************************************************
 * Copyright (c) 2023, Tri Dao, Samsung SDSA.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include "block_info.h"

namespace FLASH_NAMESPACE {

template<bool Varlen=true>
struct BlockInfoTree : BlockInfo<Varlen> {

    template<typename Params>
    __device__ BlockInfoTree(const Params &params, const int bidb)
        : BlockInfo<Varlen>(params, bidb)
        , sum_s_tree(params.tree_mask_lens_ptr[bidb])
        , actual_tree_len(params.tree_mask_lens_ptr[bidb+1]-sum_s_tree)
        {
        }

    __forceinline__ __device__ uint32_t tree_offset() const {
        return uint32_t(sum_s_tree);
    }

    const int sum_s_tree;
    const int actual_tree_len;
};

} // FLASH_NAMESPACE