/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <vector>
#include "flash.h"

inline bool use_one_mma_wg(Flash_fwd_params const& params) {
    return params.arch >= 90 && params.d == 128 && 
        params.seqlen_q * (!params.pack_gqa ? 1 : params.h / params.h_k) <= 64;
};

inline bool should_pack_gqa(bool varlen_q, int seqlen_q, int qhead_per_khead, int blockM) {
    // If varlen, we don't actually know seqlen_q but only max_seqlen_q.
    if (varlen_q) return true;
    // Heuristic: PackGQA is a bit slower but can help if seqlen_q is small or not near a multiple of kBlockM
    auto round_up = [](int a, int b) { return (a + b - 1) / b * b; };
    float nopack_gqa_efficiency = float(seqlen_q) / float(round_up(seqlen_q, blockM));
    float pack_gqa_efficiency = float(seqlen_q * qhead_per_khead) / float(round_up(seqlen_q * qhead_per_khead, blockM));
    return nopack_gqa_efficiency < 0.9 * pack_gqa_efficiency;
};

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
inline int num_splits_heuristic(int total_mblocks, int num_SMs, int num_n_blocks, int num_m_blocks, int size_one_kv_head, bool is_causal_or_local, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    // However, in the case of super long seqlen where each head of KV doesn't even fit into
    // L2 (we assume that L2 size is 50MB), we want to split.
    if (total_mblocks >= 0.8f * num_SMs) {
        int const size_l2 = 50 * 1024 * 1024;
        // Only split if there are enough queries to go over the KV at least twice
        // Don't split if causal
        if (size_one_kv_head > size_l2 && num_m_blocks >= num_SMs * 2 && !is_causal_or_local) {
            return std::min((size_one_kv_head + size_l2 - 1) / size_l2, max_splits);
        } else {
            return 1;
        }
    }
    // If num_n_blocks is too small, use 1 split. For example, we never split for hdim = 128 and seqlen_k = 512.
    if (num_n_blocks <= 4) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        float n_waves = float(total_mblocks * num_splits) / num_SMs;
        float eff = n_waves / ceil(n_waves);
        // printf("num_splits = %d, eff = %f\n", num_splits, eff);
        if (eff > max_efficiency) { max_efficiency = eff; }
        efficiency.push_back(eff);
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

inline bool get_pack_gqa(Flash_fwd_params const& params) {
    // Always enable PackGQA for Sm8x or PagedKVNonTMA or Split to reduce compilation and binary size.
    // Has little effect on speed.
    if (params.arch < 90 || (params.page_table && !params.pagedkv_tma) || params.num_splits > 1) { return true; }
    #ifdef FLASHATTENTION_DISABLE_PACKGQA
    return false;
    #else
    // params.page_table must already be set
    if (params.h == params.h_k) { return false; }
    // This needs to match the kernel configs
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
    int const kBlockM = std::get<0>(kBlockMN_kernel_args_sm90);
    return should_pack_gqa(params.cu_seqlens_q || params.seqused_q, params.seqlen_q, params.h / params.h_k, kBlockM);
    #endif
}

inline int get_num_splits(Flash_fwd_params const& params) {
    #ifdef FLASHATTENTION_DISABLE_SPLIT
    return 1;
    #else
    // Always enable PackGQA for Split
    // params.page_table must already be set
    // This needs to match the kernel configs
    bool varlen = params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k || params.leftpad_k;

    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f, params.use_one_mma_wg);
    // Strictly speaking we need to pass in (varlen && params.num_splits > 1) but num_splits
    // has not been set here. It's OK though because we might just underestimate kBlockN a bit
    auto kBlockMN_kernel_args_sm8x = tile_size_fwd_sm8x(params.arch == 86 || params.arch == 89, params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, params.page_table, varlen, params.softcap > 0.f, params.knew_ptr);
    int const kBlockM = params.arch >= 90 ? std::get<0>(kBlockMN_kernel_args_sm90) : std::get<0>(kBlockMN_kernel_args_sm8x);
    int const kBlockN = params.arch >= 90 ? std::get<1>(kBlockMN_kernel_args_sm90) : std::get<1>(kBlockMN_kernel_args_sm8x);
    int seqlen_q_packgqa = params.seqlen_q * (params.h / params.h_k);
    // If is_local, we're not going to load all of seqlen_k
    int const seqlen_k_loaded = !params.is_local
        ? params.seqlen_k
        : std::max(0, std::min(params.seqlen_k, params.window_size_right + params.window_size_left + 1 + kBlockM));
    int const num_n_blocks = (seqlen_k_loaded + kBlockN - 1) / kBlockN;
    int const num_m_blocks = (seqlen_q_packgqa + kBlockM - 1) / kBlockM;
    int const size_one_kv_head = params.seqlen_k * (params.d + params.dv) * (params.is_e4m3 ? 1 : 2);
    // Always enable PackGQA for Split
    // If varlen, we use dynamic split, so this heuristic just needs to get an upper bound on num_splits.
    // We assume the case where there's 1 long sequence and the rest are short, i.e. pretending
    // that batch = 1.
    int total_mblocks = (params.num_splits_dynamic_ptr ? 1 : params.b) * params.h_k * num_m_blocks;
    return num_splits_heuristic(total_mblocks, params.num_sm, num_n_blocks, num_m_blocks, size_one_kv_head, params.is_causal || params.is_local, 128);
    #endif
}


inline void determine_pack_gqa_splits_and_mma_wgs(
    Flash_fwd_params &params, 
    int num_splits,
    std::optional<bool> const& pack_gqa,
    bool use_stream_k = false
) {
    assert(use_stream_k && (num_splits <= 0 || num_splits == 2));
    num_splits = use_stream_k ? 2 : num_splits;

    // Determine if we should pack GQA before num_splits since it impacts use_one_mma_wg (in get_num_splits)
    params.use_one_mma_wg = use_one_mma_wg(params);
    params.pack_gqa = pack_gqa.has_value() ? pack_gqa.value() : get_pack_gqa(params);
    params.num_splits = num_splits <= 0 ? get_num_splits(params) : num_splits;
    // Always enable PackGQA for Split
    params.pack_gqa = params.num_splits > 1;
}