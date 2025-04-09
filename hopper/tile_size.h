/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

struct TileConfig {
    int kBlockM;
    int kBlockN;
    int kNWarps;         // Only for SM80
    int kStages;         // Only for SM80
    bool Q_in_regs;      // Only for SM80
    bool MmaPV_is_RS;    // Only for SM90
    bool IntraWGOverlap; // Only for SM90
};

// A constexpr helper function for SM90 tile configurations.
constexpr TileConfig make_tile_config_sm90(
    int headdim, int headdim_v, bool is_causal, bool is_local, int element_size,
    int const_max_seqlen_q, bool v_colmajor = false, bool paged_kv = false, bool softcap = false)
{
    constexpr int kWarps = 8;
    constexpr int kStages = 2;

    if (element_size == 2) {
        if (headdim <= 64) {
            bool same_hdim = (headdim == headdim_v);  // if not same hdim, we're targeting hdimv=512
            // return {same_hdim ? 192 : 64, same_hdim ? 128 : 64, same_hdim, true};
            // With this workaround in Cutlass 3.8, tile size 192 x 128 got slower for non-causal, idk why
            // https://github.com/NVIDIA/cutlass/blob/833f6990e031b48b4cd2fcf55e0849c51ef6bac2/include/cute/container/tuple.hpp#L131
            // Switch to tile size 192 x 192 for now
            return TileConfig{ same_hdim ? 192 : 64, same_hdim ? 192 : 64, kWarps, kStages, false, false, true };
        }
        else if (headdim <= 96) {
            return TileConfig{192, (is_local || paged_kv ? 128 : 144), kWarps, kStages, false, false, true};
        }
        else if (headdim <= 128) {
            if (const_max_seqlen_q <= 64) {
                return TileConfig{64, (is_causal || is_local || paged_kv ? 128 : 176), kWarps, kStages, true, true, true};
            }
            else {
                return TileConfig{128, (is_causal || is_local || paged_kv ? 128 : 176), kWarps, kStages, true, true, true};
            }
        }
        else if (headdim <= 192) {
            return TileConfig{128, (paged_kv || is_local ? 96 : (headdim_v <= 128 ? 128 : 112)), kWarps, kStages, true, true, true};
        }
        else {
            return TileConfig{128, (is_local ? 64 : 80), kWarps, kStages, true, true, true};
        }
    }
    else {
        if (headdim <= 64) {
            return TileConfig{192, 160, kWarps, kStages, true, true, true};
        }
        else if (headdim <= 96) {
            return TileConfig{192, 128, kWarps, kStages, true, true, true};
        }
        else if (headdim <= 128) {
            return TileConfig{128, (paged_kv ? 160 : (v_colmajor || (softcap && is_local) ? 192 : 224)), kWarps, kStages, true, true, true};
        }
        else if (headdim <= 192) {
            return TileConfig{128, ((paged_kv || softcap) && is_local ? 128 : 160), kWarps, kStages, true, true, true};
        }
        else {
            return TileConfig{128, (is_local ? 64 : 128), kWarps, kStages, true, true, !paged_kv};
        }
    }
}

// A constexpr helper function for SM8x tile configurations.
constexpr TileConfig make_tile_config_sm8x(
    bool sm86_or_89, int headdim, int headdim_v, bool is_causal, bool is_local, int element_size = 2,
    bool paged_kv = false, bool varlen_and_split = false,
    bool softcap = false, bool append_kv = false)
{
    if (element_size == 2) {
        if (headdim <= 64) {
            return TileConfig{128, (varlen_and_split ? 80 : (is_local ? 96 : 112)), 4, 1, false, false, false};
        }
        else if (headdim <= 96) {
            return TileConfig{128, (varlen_and_split || is_local ? 48 : 64), 4, 1, false, false, false};
        }
        else if (headdim <= 128) {
            bool use_8_warps = sm86_or_89 || varlen_and_split;
            int kBlockN = use_8_warps ?
                (varlen_and_split ? (is_local ? 96 : 112) : (is_local ? 96 : 128)) :
                (is_local ? 48 : 64);
            return TileConfig{128, kBlockN, use_8_warps ? 8 : 4, 1, use_8_warps, false, false};
        }
        else if (headdim <= 192) {
            bool kBlockN_64 = append_kv || is_local || varlen_and_split || paged_kv;
            return TileConfig{128, (kBlockN_64 ? 64 : 96), 8, sm86_or_89 ? 1 : 2, (!kBlockN_64), false, false};
        }
        else {
            int kBlockN = sm86_or_89 ?
                (append_kv ? 32 : (varlen_and_split || is_local ? 48 : 64)) :
                (append_kv ? 48 : (varlen_and_split || is_local ? 64 : 96));
            return TileConfig{128, kBlockN, 8, 1, (sm86_or_89 && !append_kv), false, false};
        }
    }
    else {
        // Placeholder for now
        return TileConfig{128, 64, 8, 2, false, false, false};
    }
}

constexpr TileConfig get_tile_config(
    int Arch,
    int headdim, int headdim_v, bool is_causal, bool is_local, int element_size,
    int const_max_seqlen_q, bool v_colmajor = false, bool paged_kv = false,
    bool softcap = false, bool append_kv = false, bool varlen_and_split = false)
{
    if (Arch == 90) {
        return make_tile_config_sm90(
            headdim, headdim_v, is_causal, is_local, element_size,
            const_max_seqlen_q, v_colmajor, paged_kv, softcap);
    }
    else {
        return make_tile_config_sm8x(
            Arch == 86 || Arch == 89, headdim, headdim_v, is_causal, is_local,
            element_size, paged_kv, varlen_and_split, softcap, append_kv);
    }
}