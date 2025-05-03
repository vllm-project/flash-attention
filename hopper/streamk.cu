#include <streamk.h>

std::tuple<torch::Tensor, torch::Tensor> streamk_schedule(
    int arch,
    int num_sms,
    int batch_size,
    std::optional<const at::Tensor> &cu_seqlens_q,
    const at::Tensor &seqused_k,
    int seqlen_q,
    int seqlen_k,
    int num_heads_q,
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
) {
    assert (is_local == false && "StreamK + Local attention not supported yet");

    std::optional<const at::Tensor> cu_seqlens_q_cpu;
    if (cu_seqlens_q) {
        cu_seqlens_q_cpu.emplace(cu_seqlens_q->cpu());
    }
    auto seqused_k_cpu = seqused_k.cpu();

    auto get_tile_sizes = [&](bool use_one_mma_wg) -> std::tuple<int, int> {
        if (arch == 90) {
            auto ts = tile_size_fwd_sm90(headdim, headdim_v, is_causal, is_local, element_size, v_colmajor, paged_kv_non_TMA, softcap, use_one_mma_wg);
            return std::make_tuple(std::get<0>(ts), std::get<1>(ts));
        } else if (arch < 90) {
            auto ts = tile_size_fwd_sm8x(headdim, headdim_v, is_causal, is_local, element_size, paged_kv, /* varlen_and_split */ true, softcap, append_kv);
            return std::make_tuple(std::get<0>(ts), std::get<1>(ts));
        } else {
            assert(false && "Unsupported architecture");
            return std::make_tuple(0, 0);
        }
    };

    auto get_seqlen_k = [&](int bidb) {
        return seqused_k_cpu.accessor<int32_t, 1>()[bidb];
    };


    auto get_seqlen_q = [&](int bidb) {
        if (cu_seqlens_q_cpu.has_value()) {
            auto cu_seqlens_q = cu_seqlens_q_cpu.value().accessor<int32_t, 1>();
            return cu_seqlens_q[bidb + 1] - cu_seqlens_q[bidb];
        } else {
            return seqlen_q;
        }
    };

    auto get_num_n_tiles = [&](int m_tile, int seqlen_k, int seqlen_q, int block_m, int block_n) {
        int m_tile_k_end = std::min(
            (seqlen_k - seqlen_q) + (m_tile + 1) * block_m,
            seqlen_k
        );
        return cutlass::ceil_div(m_tile_k_end, block_n);
    };

    auto tile_sizes = get_tile_sizes(false);
    auto tile_sizes_one_mma_wg = get_tile_sizes(true);

    auto compute_tiles = [&, get_num_n_tiles = get_num_n_tiles](
        int num_heads_q,
        int num_heads_k,
        int seqlen_q, 
        int seqlen_k,
        bool causal,
        bool pack_gqa,
        bool one_mma_wg
    ) {
        int block_m = one_mma_wg ? std::get<0>(tile_sizes_one_mma_wg) : std::get<0>(tile_sizes);
        int block_n = one_mma_wg ? std::get<1>(tile_sizes_one_mma_wg) : std::get<1>(tile_sizes);

        seqlen_q *= pack_gqa ? num_heads_q / num_heads_k : 1;
        int num_heads = pack_gqa ? num_heads_k : num_heads_q;
        int m_tiles = cutlass::ceil_div(seqlen_q, block_m);
        int tiles_total = 0;
        if (causal) {
            tiles_total += m_tiles * cutlass::ceil_div(seqlen_k, block_n);
        } else {
            for (int m_tile = 0; m_tile < m_tiles; m_tile++) {
                tiles_total += get_num_n_tiles(
                    m_tile, seqlen_k, seqlen_q, block_m, block_n);
            }
        }

        return tiles_total * num_heads;
    };

    bool pack_gqa = false;

    // Determine if we should pack GQA by determining the the amount of
    // available work that would benefit from packing GQA
    // Assume not `use_one_mma_wg` for now, we determine this later
    if (num_heads_q > num_heads_k) {
        assert (num_heads_q % num_heads_k == 0);

        int total_tiles_pack_gqa = 0;
        int total_tiles_no_pack_gqa = 0;

        for (int bidb = 0; bidb < batch_size; ++bidb) {
            int seqlen_k = get_seqlen_k(bidb);
            int seqlen_q = get_seqlen_q(bidb);

            total_tiles_pack_gqa += compute_tiles(
                num_heads_q, num_heads_k, seqlen_q, seqlen_k, is_causal, true, false);
            total_tiles_no_pack_gqa += compute_tiles(
                num_heads_q, num_heads_k, seqlen_q, seqlen_k, is_causal, false, false);
        }

        if (total_tiles_pack_gqa < (total_tiles_no_pack_gqa * 1.1f)) {
            pack_gqa = true;
        }
    }

    bool use_one_mma_wg = false;
    // Determine the amount of work that would benefit from using one MMA
    // workgroup

    int total_tiles = 0;
    int total_tiles_one_mma_wg = 0;

    for (int bidb = 0; bidb < batch_size; ++bidb) {
        int seqlen_k = get_seqlen_k(bidb);
        int seqlen_q = get_seqlen_q(bidb);

        total_tiles += compute_tiles(
            num_heads_q, num_heads_k, seqlen_q, seqlen_k, is_causal, pack_gqa, false);
        total_tiles_one_mma_wg += compute_tiles(
            num_heads_q, num_heads_k, seqlen_q, seqlen_k, is_causal, pack_gqa, true);
    }
    
    // if using one_mma_wg only increases the number of tiles by 50% or less
    // then we should use it (since it performs each tile computes 1/2 as much)
    // TODO(lucas): 50% is a guesstimate, we should do a more thorough analysis
    if (total_tiles_one_mma_wg < (total_tiles * 1.5f)) {
        use_one_mma_wg = true;
    }

    tile_sizes = use_one_mma_wg ? tile_sizes_one_mma_wg : tile_sizes;
    total_tiles = use_one_mma_wg ? total_tiles_one_mma_wg : total_tiles;

    int block_m = use_one_mma_wg ? std::get<0>(tile_sizes_one_mma_wg) : std::get<0>(tile_sizes);
    int block_n = use_one_mma_wg ? std::get<1>(tile_sizes_one_mma_wg) : std::get<1>(tile_sizes);

    int target_tiles_per_sm = cutlass::ceil_div(total_tiles, num_sms);

    std::vector<StreamKWorkTile> work_tiles;
    work_tiles.reserve(1024);
    std::vector<int> work_tiles_ind_ptr;
    work_tiles_ind_ptr.reserve(num_sms + 1);
    work_tiles_ind_ptr.push_back(0);
    std::vector<StreamKCombineTile> combine_tiles;
    combine_tiles.reserve(1024);

    int min_tiles = 1;
    int max_num_peers = 0;

    int current_tile = 0;
    int sm_target_tiles_remaining = target_tiles_per_sm;

    int tiles_total_allocated = 0;
    int num_heads = (pack_gqa) ? num_heads_k : num_heads_q;

    for (int bidb = 0; bidb < batch_size; ++bidb) {
        int seqlen_k = get_seqlen_k(bidb);
        int seqlen_q = get_seqlen_q(bidb);
        int m_tiles = cutlass::ceil_div(seqlen_q, block_m);
        for (int bidh = 0; bidh < num_heads; ++bidh) {
            for (int m_tile = 0; m_tile < m_tiles; m_tile++) {
                int n_tiles = get_num_n_tiles(
                    m_tile, seqlen_k, seqlen_q, block_m, block_n);

                int m_tile_start_idx = current_tile;
                int curr_n_tile_start = 0;
                int curr_n_tiles_remaining = n_tiles;
                int num_peers = 0;

                while (curr_n_tiles_remaining > 0) {
                    int n_tiles = std::min(curr_n_tiles_remaining, sm_target_tiles_remaining);

                    // if we would leave a residual tile that is less than the minimum tiles
                    // then we should just take the rest of the tiles
                    if (curr_n_tiles_remaining - n_tiles < min_tiles) {
                        n_tiles = curr_n_tiles_remaining;
                    }

                    curr_n_tiles_remaining -= n_tiles;
                    sm_target_tiles_remaining -= n_tiles;

                    work_tiles.emplace_back(StreamKWorkTile{
                        /* m_block:       */ m_tile,
                        /* n_block_start: */ curr_n_tile_start,
                        /* n_blocks:      */ uint16_t(n_tiles),
                        /* bidb:          */ uint16_t(bidb),
                        /* bidh:          */ uint16_t(bidh),
                        /* peer_id:       */ uint8_t(num_peers),
                        /* num_peers:     */ 0
                    });

                    current_tile += 1;
                    num_peers += 1;
                    curr_n_tile_start += n_tiles;
                    tiles_total_allocated += n_tiles;

                    if (sm_target_tiles_remaining <= 0) {
                        work_tiles_ind_ptr.push_back(current_tile);
                        sm_target_tiles_remaining = target_tiles_per_sm;
                    }
                }

                if (num_peers > 1) {
                    combine_tiles.emplace_back(StreamKCombineTile{
                        /* m_block:   */ m_tile,
                        /* bidh:      */ uint16_t(bidh),
                        /* bidb:      */ uint16_t(bidb),
                        /* num_peers: */ uint16_t(num_peers)
                    });
                }

                if (num_peers > max_num_peers) {
                    max_num_peers = num_peers;
                }

                for (int i = m_tile_start_idx; i < current_tile; ++i) {
                    work_tiles[i].num_peers = num_peers;
                }
            }
        }
    }

    auto [metadata_offsets, metadata_size] = get_device_metadata_offsets_and_size(
        num_sms,
        work_tiles.size(),
        combine_tiles.size()
    );

    auto device_metadata = torch::empty(
        {int(metadata_size)},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCPU)
    );

    uint8_t *device_metadata_ptr = device_metadata.data_ptr<uint8_t>();
    std::memcpy(device_metadata_ptr + metadata_offsets.work_tiles_offset, work_tiles.data(), work_tiles.size() * sizeof(StreamKWorkTile));
    std::memcpy(device_metadata_ptr + metadata_offsets.work_tiles_ind_ptr_offset, work_tiles_ind_ptr.data(), work_tiles_ind_ptr.size() * sizeof(int));
    std::memcpy(device_metadata_ptr + metadata_offsets.combine_tiles_offset, combine_tiles.data(), combine_tiles.size() * sizeof(StreamKCombineTile));

    auto host_metadata = torch::empty(
        {sizeof(StreamKSchedulerDescisions)},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCPU)
    );

    auto host_metadata_ptr = reinterpret_cast<StreamKSchedulerDescisions*>(host_metadata.data_ptr<uint8_t>());
    host_metadata_ptr->num_combine_blocks = combine_tiles.size();
    host_metadata_ptr->max_num_peers = max_num_peers;
    host_metadata_ptr->pack_gqa = pack_gqa;
    host_metadata_ptr->use_one_mma_wg = use_one_mma_wg;
    host_metadata_ptr->num_work_tiles = work_tiles.size();
    host_metadata_ptr->grid_size = work_tiles_ind_ptr.size() - 1;
    host_metadata_ptr->m_block_size = block_m;

    return {device_metadata, host_metadata};
}