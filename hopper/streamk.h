#pragma once

#include <stdint.h>
#include <torch/nn/functional.h>

#include "cute/container/alignment.hpp"
#include "tile_size.h"

struct CUTE_ALIGNAS(16) StreamKSchedulerDescisions {
    int  num_combine_blocks;
    int  num_work_tiles;
    int  max_num_peers;
    int  grid_size;
    int  m_block_size;
    bool pack_gqa;
    bool use_one_mma_wg;

    void print() const {
        printf("StreamKSchedulerDescisions:\n");
        printf("  num_combine_blocks: %d\n", num_combine_blocks);
        printf("  num_work_tiles: %d\n", num_work_tiles);
        printf("  max_num_peers: %d\n", max_num_peers);
        printf("  pack_gqa: %d\n", pack_gqa);
        printf("  use_one_mma_wg: %d\n", use_one_mma_wg);
    }
};

// Make sure this fits into 128bits
struct CUTE_ALIGNAS(16) StreamKWorkTile {
    int      m_block = -1;
    int      n_block_start = 0;
    uint16_t n_blocks = 0; // Max n_blocks per tile is 65535
    uint16_t bidb = 0;     // Max batch size is 65535
    uint16_t bidh = 0;     // Max num heads is 65535
    uint8_t  peer_id = 0;   // Max 255 peers
    uint8_t  num_peers = 0; // Max 255 peers

    void print() const {
        printf("StreamKWorkTile:\n");
        printf("  m_block: %d\n", m_block);
        printf("  n_block_start: %d\n", n_block_start);
        printf("  n_blocks: %d\n", n_blocks);
        printf("  bidb: %d\n", bidb);
        printf("  bidh: %d\n", bidh);
        printf("  peer_id: %d\n", peer_id);
        printf("  num_peers: %d\n", num_peers);
    }
};

struct CUTE_ALIGNAS(8) StreamKCombineTile {
    int const m_block;
    uint16_t const bidh;
    uint16_t const bidb;
    uint16_t const num_peers; // Number of peers / num_splits

    void print() const {
        printf("StreamKCombineTile:\n");
        printf("  m_block: %d\n", m_block);
        printf("  bidh: %d\n", bidh);
        printf("  bidb: %d\n", bidb);
        printf("  num_peers: %d\n", num_peers);
    }
};

struct StreamKMetadataByteOffsets {    
    int const work_tiles_offset;
    int const combine_tiles_offset;
    int const work_tiles_ind_ptr_offset;
};

inline std::tuple<StreamKMetadataByteOffsets, int> get_device_metadata_offsets_and_size(
    int num_sms,
    int num_work_tiles,
    int num_combine_tiles
) {
    auto round_up_to_16 = [](int size) {
        return cutlass::round_up(size, 16);
    };

    int work_tiles_offset = 0;
    int combine_tiles_offset = work_tiles_offset + round_up_to_16(num_work_tiles * sizeof(StreamKWorkTile));
    int work_tiles_ind_ptr_offset = combine_tiles_offset + round_up_to_16(num_combine_tiles * sizeof(StreamKCombineTile));
    int total_size = work_tiles_ind_ptr_offset + round_up_to_16((num_sms + 1) * sizeof(int));

    StreamKMetadataByteOffsets metadata_offsets{
        work_tiles_offset,
        combine_tiles_offset,
        work_tiles_ind_ptr_offset
    };

    return std::make_tuple(metadata_offsets, total_size);
}

std::tuple<torch::Tensor, torch::Tensor> streamk_schedule(
    int arch,
    int num_sms,
    int batch_size,
    std::optional<const at::Tensor> &cu_seqlens_q,
    const at::Tensor &seqused_k,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    int num_heads_k,
    int headdim, 
    int headdim_v, 
    bool is_causal, 
    bool is_local, 
    int element_size,
    bool v_colmajor, 
    bool paged_kv,
    bool paged_kv_non_TMA, 
    bool softcap,
    bool append_kv
);
