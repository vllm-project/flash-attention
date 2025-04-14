/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cutlass/fast_math.h"
#include "cutlass/arch/barrier.h"

#include "named_barrier.hpp"
#include "block.h"
#include "utils.h"
#include "seqlen.h"

namespace flash {

///////////////////////////////////////////////////////////////////////////////

using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
using ShapePageTable = cute::Shape<int32_t, int32_t>;  // (batch, max_num_pages_per_seq)

// Host side kernel arguments
struct TileSchedulerArguments {
    // num_head is num_head_q if not PackGQA, else num_head_k
    int const num_blocks, num_head, num_batch, num_splits;
    int const qhead_per_khead;
    int const seqlen;  // Only used if Varlen and cu_seqlens == nullptr and seqused == nullptr
    int const seqlen_k, headdim, headdim_v, element_size;  // Used to calculate L2 swizzling
    int* const tile_count_semaphore = nullptr;
    int const* const cu_seqlens_q = nullptr;
    int const* const cu_seqlens_k = nullptr;
    int const* const cu_seqlens_k_new = nullptr;
    int const* const seqused_q = nullptr;
    int const* const seqused_k = nullptr;
    int const* const leftpad_k = nullptr;
    int const* const seqlens_rotary = nullptr;
    ShapeQKV const shape_Q;
    ShapeQKV const shape_K;
    ShapeQKV const shape_K_new;
    int const* const ptr_pagetable = nullptr;
    ShapePageTable const shape_pagetable;
    int const* const num_splits_dynamic_ptr = nullptr;
    int const window_size_left = -1;
    int const window_size_right = 0;
};

template<bool AppendKV>
struct BlockCoord {};

template<>
struct BlockCoord<false> {
    int const m_block = -1;
    int const bidh = -1;
    int const bidb = -1;
    int const n_block_min = 0;
    int const n_block_max = 0;
    int const peer_id = 0;   // Where to write the partial results / split_k index
    int const num_peers = 0; // Number of peers / num_splits
};

template<>
struct BlockCoord<true>: public BlockCoord<false> {
    int const n_block_new_min = 0;
    int const n_block_new_max = 0;
};

///////////////////////////////////////////////////////////////////////////////

template<typename SeqlenInfo_t, typename TileShape_MNK, bool Varlen=false, bool Split=false, bool PackGQA=false, bool AppendKV=false, bool Is_causal=false, bool Is_local=false>
struct TileSchedulerCommon {
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN, Is_causal, Is_local, PackGQA, Split>;

    struct Params {
        int const num_blocks, num_head, num_batch, num_splits;
        cutlass::FastDivmod qhead_per_khead;
        int const seqlen;
        int const* const cu_seqlens_q = nullptr;
        int const* const cu_seqlens_k = nullptr;
        int const* const cu_seqlens_k_new = nullptr;
        int const* const seqused_q = nullptr;
        int const* const seqused_k = nullptr;
        int const* const leftpad_k = nullptr;
        int const* const seqlens_rotary = nullptr;
        ShapeQKV shape_Q;
        ShapeQKV shape_K;
        ShapeQKV shape_K_new;
        int const* const ptr_pagetable = nullptr;
        ShapePageTable shape_pagetable;
        int const* const num_splits_dynamic_ptr = nullptr;
        int const window_size_left;
        int const window_size_right;
    };

    Params const& params;

    CUTLASS_DEVICE
    TileSchedulerCommon(Params const& params) : params(params) { }

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        assert(!Split || !Varlen || args.num_splits_dynamic_ptr != nullptr);
        assert(!Split || !Varlen || args.num_splits < (1 << 16)); // We use the top 16 bits to store num_splits in VarlenDynamic
        return {args.num_blocks, args.num_head, args.num_batch, !Split ? 1 : args.num_splits,
                cutlass::FastDivmod(args.qhead_per_khead),
                args.seqlen,
                args.cu_seqlens_q, args.cu_seqlens_k, args.cu_seqlens_k_new,
                args.seqused_q, args.seqused_k, args.leftpad_k, args.seqlens_rotary,
                args.shape_Q, args.shape_K, args.shape_K_new,
                args.ptr_pagetable,
                args.shape_pagetable,
                args.num_splits_dynamic_ptr,
                args.window_size_left, args.window_size_right};
    }
    CUTLASS_DEVICE
    SeqlenInfo_t 
    create_seqlen_info(int bidb) {
        return {
            bidb,
            get<0>(params.shape_Q),
            !params.ptr_pagetable ? size<0>(params.shape_K) : size<0>(params.shape_K) * size<1>(params.shape_pagetable),
            get<0>(params.shape_K_new),
            params.cu_seqlens_q, params.cu_seqlens_k, params.cu_seqlens_k_new,
            params.seqused_q, params.seqused_k, params.leftpad_k,
            params.seqlens_rotary
        };
    };

    CUTLASS_DEVICE
    BlockCoord<AppendKV>
    create_block_coord_split(SeqlenInfo_t const& seqlen_info, int m_block, int bidh, int bidb, int peer_id, int num_peers) {
        // Calculate n_block_min/max based on causality and local window
        auto n_block_min_max = BlockMN_t::get_n_block_min_max(
            seqlen_info, m_block, bidb, peer_id, num_peers,
            params.window_size_left, params.window_size_right, params.qhead_per_khead_divmod);
        
        if constexpr (AppendKV) {
            auto n_block_min_max_new = BlockMN_t::get_n_block_k_new_min_max(
                seqlen_info, m_block, bidb, peer_id, num_peers,
                params.window_size_left, params.window_size_right, params.qhead_per_khead_divmod);
            return {m_block, bidh, bidb, get<0>(n_block_min_max), get<1>(n_block_min_max), peer_id, num_peers, 
                    get<0>(n_block_min_max_new), get<1>(n_block_min_max_new)};
        } else {
            return {m_block, bidh, bidb, get<0>(n_block_min_max), get<1>(n_block_min_max), peer_id, num_peers};
        }
    };
};

template<typename SeqlenInfo_t, typename TileShape_MNK, bool Varlen=false, bool Split=false, bool PackGQA=false, bool AppendKV=false, bool Is_causal=false, bool Is_local=false>
class SingleTileScheduler: public TileSchedulerCommon<SeqlenInfo_t, TileShape_MNK, Varlen, Split, PackGQA, AppendKV, Is_causal, Is_local> {

public:
    using SharedStorage = int;
    using Super = TileSchedulerCommon<SeqlenInfo_t, TileShape_MNK, Varlen, Split, PackGQA, AppendKV, Is_causal, Is_local>;

    struct Params: public Super::Params {
        cutlass::FastDivmod nsplits_divmod;
    };

    using Super::create_seqlen_info;
    using Super::create_block_coord_split;

    Params const& params;

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        return {Super::to_underlying_arguments(args), cutlass::FastDivmod(!Split ? 1 : args.num_splits)};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(params.num_blocks), uint32_t((!Split ? 1 : params.num_splits) * params.num_head), uint32_t(params.num_batch)};
    }

    struct WorkTileInfo {
        BlockCoord<AppendKV> block_coord;
        SeqlenInfo_t seqlen_info;
    };

    CUTLASS_DEVICE
    bool
    is_valid(WorkTileInfo const& work_tile) const {
        return work_tile.block_coord.bidb >= 0;
    }

    CUTLASS_DEVICE
    BlockCoord<AppendKV>
    get_block_coord(WorkTileInfo const& work_tile) const {
        return work_tile.block_coord;
    }

    CUTLASS_DEVICE
    SeqlenInfo_t
    get_seqlen_info(WorkTileInfo const& work_tile) const {
        return work_tile.seqlen_info;
    }

    CUTLASS_DEVICE
    SingleTileScheduler(SharedStorage* const smem_scheduler, Params const& params) : Super(params), params(params) { }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        int const m_block = blockIdx.x;
        int bidb = blockIdx.z;
        int bidh = blockIdx.y;
        int peer_id = 0;
        int num_peers = 1;

        SeqlenInfo_t seqlen_info = create_seqlen_info(bidb);

        bool is_valid_tile = true;
        if constexpr (Split) {
            peer_id = params.nsplits_divmod.divmod(peer_id, bidh);
            num_peers = params.nsplits_divmod.divisor;
        }
        if constexpr (Varlen) {
            int seqlen_q_ = params.seqused_q
                ? params.seqused_q[bidb]
                : (params.cu_seqlens_q ? params.cu_seqlens_q[bidb + 1] - params.cu_seqlens_q[bidb] : params.seqlen);
            if constexpr (PackGQA) { seqlen_q_ *= params.qhead_per_khead_divmod.divisor; }
            is_valid_tile = m_block * size<0>(TileShape_MNK{}) < seqlen_q_;

            if constexpr (Split) {
                int num_splits_dynamic = params.num_splits_dynamic_ptr ? params.num_splits_dynamic_ptr[bidb] : num_peers;
                is_valid_tile &= peer_id < num_splits_dynamic;
                num_peers = num_splits_dynamic;
            }
        }
        if (!is_valid_tile) return {BlockCoord<AppendKV>{}, seqlen_info};

        return {
            create_block_coord_split(seqlen_info, m_block, bidh, bidb, peer_id, num_peers),
            seqlen_info
        };
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(WorkTileInfo& current_work) const {
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(WorkTileInfo const& current_work) const {
        return { BlockCoord<AppendKV>{}, {} };
    }

};

///////////////////////////////////////////////////////////////////////////////

template<typename SeqlenInfo_t,typename TileShape_MNK, bool Split=false, bool Is_causal=false, bool Is_local=false, bool Varlen=false, bool AppendKV=false, bool PackGQA=false>
class StaticPersistentTileScheduler: public TileSchedulerCommon<SeqlenInfo_t, TileShape_MNK, Varlen, Split, PackGQA, AppendKV, Is_causal, Is_local> {

public:
    using SharedStorage = int;
    using Super = TileSchedulerCommon<SeqlenInfo_t, TileShape_MNK, Varlen, Split, PackGQA, AppendKV, Is_causal, Is_local>;

    // Device side kernel params
    struct Params: public Super::Params {
        int total_blocks;
        cutlass::FastDivmod m_block_divmod, head_divmod;
        cutlass::FastDivmod nsplits_divmod;
    };

    Params const& params;

    using Super::create_seqlen_info;
    using Super::create_block_coord_split;

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        int num_splits_val = !Split ? 1 : args.num_splits;
        return { Super::to_underlying_arguments(args),
                 args.num_blocks * args.num_head * args.num_batch * num_splits_val,
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head * num_splits_val),
                cutlass::FastDivmod(num_splits_val)
               };
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        // This scheduler assumes that the grid shape is fixed and known at compile time.
        return {uint32_t(num_sm), 1u, 1u}; // Persistent kernel uses SM count as grid dim
    }

    struct WorkTileInfo {
        int tile_idx;
    };

    CUTLASS_DEVICE
    bool
    is_valid(WorkTileInfo const& work_tile) const {
        return work_tile.tile_idx < params.total_blocks;
    }

    CUTLASS_DEVICE
    BlockCoord<AppendKV>
    get_block_coord(WorkTileInfo const& work_tile) const {
        int m_block, bidh, bidb;
        bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(m_block, work_tile.tile_idx));
        int peer_id = 0;
        int num_peers = 1;
        if constexpr (Split) {
            num_peers = params.nsplits_divmod.divisor;
            if constexpr (Varlen) {
                 int num_splits_dynamic = params.num_splits_dynamic_ptr ? params.num_splits_dynamic_ptr[bidb] : num_peers;
                 if (peer_id >= num_splits_dynamic) return {}; // Invalid tile for this split
                 num_peers = num_splits_dynamic;
            }
        }

        SeqlenInfo_t seqlen_info = create_seqlen_info(bidb);
        return create_block_coord_split(seqlen_info, m_block, bidh, bidb, peer_id, num_peers);
    }

    CUTLASS_DEVICE
    SeqlenInfo_t
    get_seqlen_info(WorkTileInfo const& work_tile) const {
        bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(m_block, work_tile.tile_idx));
        return create_seqlen_info(bidb);
    }

    CUTLASS_DEVICE
    StaticPersistentTileScheduler(SharedStorage* const smem_scheduler, Params const& params)
        : Super(params), params(params) {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(WorkTileInfo& current_work) const {
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(WorkTileInfo const& current_work) const {
        return {current_work.tile_idx + int(gridDim.x)};
    }

};

template<typename SeqlenInfo_t, typename TileShape_MNK, int NumMmaThreads, int NumProducerThreads, bool Split, bool PackGQA, bool Is_causal, bool Is_local, bool AppendKV, bool WarpSpecialized>
class DynamicPersistentTileScheduler: public TileSchedulerCommon<SeqlenInfo_t, TileShape_MNK, /*Varlen=*/false, Split, PackGQA, AppendKV, Is_causal, Is_local> {

    // This scheduler targets the causal (or local) case where each tile takes different
    // amount of time. We use longest-processing-time-first scheduling:
    // the longest remaining tile is assigned to the first SM that's free.
    // SM indicates they are free by incrementing a semaphore.
    // However, we have to make sure K & V still fit into L2 cache, so we perform scheduling
    // on "sections" of the head & batch dimension, each section consisting of e.g. 8 heads.
    // This is the L2 swizzling part. The size of each section is precomputed based on the
    // size of K & V and the L2 cache size.

    static_assert(WarpSpecialized || NumProducerThreads == NumMmaThreads);
    static constexpr int NumThreads = NumMmaThreads; // Assuming non-warp-specialized usage for now

    using Super = TileSchedulerCommon<SeqlenInfo_t, TileShape_MNK, /*Varlen=*/false, Split, PackGQA, AppendKV, Is_causal, Is_local>;

protected:
    SharedStorage* const tile_count_smem;

public:

    struct Params : public Super::Params {
        int const total_blocks;
        cutlass::FastDivmod const m_block_divmod, head_divmod;
        cutlass::FastDivmod const l2_minor_divmod, l2_major_divmod;
        cutlass::FastDivmod const l2_minor_residual_divmod;
        int const num_hb_quotient;
        int* const tile_count_semaphore;
        // Params needed for L2 swizzling calculation
        int const seqlen_k;
        int const headdim;
        int const headdim_v;
        int const element_size;
    };

    Params const& params;

    using Super::create_seqlen_info;
    using Super::create_block_coord_split;

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        int const size_one_kv_head = args.seqlen_k * (args.headdim + args.headdim_v) * args.element_size * 2;
        int const size_l2 = 32 * 1024 * 1024;  // 32 MB for K & V
        // Swizzle is the size of each "section". Round swizzle to a power of 2
        // If not PackGQA already, the size of each section can increase by qhead_per_khead
        // Need to be careful about the case where only one head will fit
        int const swizzle = (size_l2 < size_one_kv_head ? 1 : (1 << cutlass::find_log2(size_l2 / size_one_kv_head))) * (PackGQA ? 1 : args.qhead_per_khead);
        // If we're in the last section (called residual), we don't want to divide by
        // swizzle. Instead we want to divide by the remainder.
        int const num_hb_remainder = (args.num_head * args.num_batch) % swizzle;
        int const num_split_blocks = args.num_blocks * (!Split ? 1 : args.num_splits);
        int const num_splits_val = !Split ? 1 : args.num_splits;
        // printf("num_split_blocks = %d, num_head = %d, num_batch = %d, swizzle = %d, PackGQA = %d, qhead_per_khead = %d, num_hb_remainder = %d\n", num_split_blocks, args.num_head, args.num_batch, swizzle, int(PackGQA), args.qhead_per_khead, num_hb_remainder);
        assert(args.tile_count_semaphore != nullptr);
        return {Super::to_underlying_arguments(args),
                num_split_blocks * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head),
                cutlass::FastDivmod(swizzle), cutlass::FastDivmod(swizzle * num_split_blocks),
                // don't divide by 0
                cutlass::FastDivmod(num_hb_remainder > 0 ? num_hb_remainder : 1),
                (args.num_head * args.num_batch) / swizzle,
                args.tile_count_semaphore,
                args.seqlen_k, args.headdim, args.headdim_v, args.element_size
               };
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx;
    };

    CUTLASS_DEVICE
    bool
    is_valid(WorkTileInfo const& work_tile) const {
        return work_tile.tile_idx < params.total_blocks;
    }

    CUTLASS_DEVICE
    BlockCoord<AppendKV>
    get_block_coord(WorkTileInfo const& work_tile) const {
        int m_block, bidh, bidb;
        int l2_mod, bidhb, bidhb_residual;
        bidhb = params.l2_major_divmod.divmod(l2_mod, work_tile.tile_idx);
        // If we're in the last section (called residual), we don't want to divide by
        // swizzle. Instead we want to divide by the remainder.
        if (bidhb < params.num_hb_quotient) {
            m_block = params.l2_minor_divmod.divmod(bidhb_residual, l2_mod);
        } else {
            m_block = params.l2_minor_residual_divmod.divmod(bidhb_residual, l2_mod);
        }
        bidb = params.head_divmod.divmod(bidh, bidhb * params.l2_minor_divmod.divisor + bidhb_residual);
        int peer_id = 0;
        int num_peers = 1;
        if constexpr (Split) {
            num_peers = params.num_splits; // Static splits for Dynamic scheduler
            peer_id = params.m_block_divmod.divmod(m_block, m_block); // This uses m_block_divmod's divisor (num_m_blocks)
        }
        // Longest-processing-time-first means we process m_blocks in reverse order
        m_block = params.m_block_divmod.divisor - 1 - m_block;

        return create_block_coord_split(seqlen_info, m_block, bidh, bidb, peer_id, num_peers);
    }


    CUTLASS_DEVICE
    SeqlenInfo_t
    get_seqlen_info(WorkTileInfo const& work_tile) const {
        int m_block, bidh, bidb;
        int l2_mod, bidhb, bidhb_residual;
        bidhb = params.l2_major_divmod.divmod(l2_mod, work_tile.tile_idx);
        if (bidhb < params.num_hb_quotient) {
            m_block = params.l2_minor_divmod.divmod(bidhb_residual, l2_mod);
        } else {
            m_block = params.l2_minor_residual_divmod.divmod(bidhb_residual, l2_mod);
        }
        bidb = params.head_divmod.divmod(bidh, bidhb * params.l2_minor_divmod.divisor + bidhb_residual);

        return create_seqlen_info(bidb);
    }

    CUTLASS_DEVICE
    DynamicPersistentTileScheduler(SharedStorage* const smem_scheduler, Params const& params)
        : Super(params), params(params), tile_count_smem(smem_scheduler) {
    }


    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {
        if (WarpSpecialized || cutlass::canonical_warp_idx_sync() > 0) {
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
        }
    }

    CUTLASS_DEVICE
    void
    prefetch_next_work(WorkTileInfo& current_work) const {
        if (threadIdx.x % NumProducerThreads == 0) {
            current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            // thread 0 already has the right tile_idx, just need to broadcast to the rest of warp 0
            int new_tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0 /*lane*/);
            WorkTileInfo work_info = tile_idx_to_work_tile(new_tile_idx, current_work);
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            if (threadIdx.x % NumProducerThreads == 0) {
                *tile_count_smem = current_work.tile_idx;
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            return work_info;
        } else {
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            int tile_idx = *tile_count_smem;
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            return {tile_idx};
        }
    }

};

template<typename SeqlenInfo_t, typename TileShape_MNK, int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=cutlass::NumThreadsPerWarp,
         bool Split=false, bool PackGQA=false,
         bool Is_causal=false, bool Is_local=false, bool AppendKV=false,
         bool WarpSpecialized=true>
class VarlenDynamicPersistentTileScheduler: public TileSchedulerCommon<SeqlenInfo_t, TileShape_MNK, /*Varlen=*/false, Split, PackGQA, AppendKV, Is_causal, Is_local> {

    static_assert(WarpSpecialized || NumProducerThreads == NumMmaThreads);
    static constexpr int NumThreads = WarpSpecialized ? NumMmaThreads + NumProducerThreads : NumMmaThreads;

protected:
    SharedStorage* const work_info_smem;
    using Super = TileSchedulerCommon<SeqlenInfo_t, TileShape_MNK, /*Varlen=*/false, Split, PackGQA, AppendKV, Is_causal, Is_local>;


public:

    // Device side kernel params
    struct Params: public Super::Params {
        cutlass::FastDivmod head_divmod;
        cutlass::FastDivmod nsplits_divmod; // Static num splits divisor
        int* const tile_count_semaphore;

    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        // If Split, for the purpose of scheduling, we pretend that instead there are
        // (args.num_splits * args.num_head) number of heads.
        assert(args.tile_count_semaphore != nullptr);
        assert(args.num_head < (1 << 16));  // We use the top 16 bits to store num_splits & split_idx
        assert(!Split || args.num_splits < (1 << 8)); // We use the top 8 bits to store num_splits
        int const num_splits_val = !Split ? 1 : args.num_splits;
        return {Super::to_underlying_arguments(args),
                cutlass::FastDivmod(args.num_head),
                cutlass::FastDivmod(num_splits_val),
                args.tile_count_semaphore
               };
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx, block, bidh, bidb;
    };

    Params const& params;

    CUTLASS_DEVICE
    bool
    is_valid(WorkTileInfo const& work_tile) const {
        // if (blockIdx.x >= 0 && (threadIdx.x == 128 || threadIdx.x == 0)) { printf("blockIdx.x = %d, threadIdx.x = %d, checking valid, bidb = %d, params.num_batch = %d\n", blockIdx.x, threadIdx.x, work_tile.bidb, params.num_batch); }
        return work_tile.bidb >= 0 && work_tile.bidb < params.num_batch;
    }

    CUTLASS_DEVICE
    BlockCoord<AppendKV>
    get_block_coord(WorkTileInfo const& work_tile) const {
        int m_block = work_tile.block;
        int bidh_in = work_tile.bidh; // This might be packed
        int bidb = work_tile.bidb;
        int peer_id = 0;
        int num_peers = 1;
        int bidh_actual = bidh_in;

        if constexpr (Split) {
            // the top 8 bits of bidh store num_splits and the next 8 bits store split_idx
            // reinterpret_cast to uint32_t to make sure we're not doing sign extension when we shift
            uint32_t bidh_packed = reinterpret_cast<uint32_t const&>(bidh_in);
            uint32_t bidh_actual_u = bidh_packed & 0x0000FFFF;
            bidh_actual = reinterpret_cast<int&>(bidh_actual_u);
            // Use the top 16 bits of split_idx to store num_splits and the next 16 bits to store split_idx
            // Extract split_idx (lower 8 bits of upper 16) and num_splits (upper 8 bits of upper 16)
            uint32_t split_idx_u = (bidh_packed >> 16) & 0xFF;
            uint32_t num_peers_u = (bidh_packed >> 24) & 0xFF;
            peer_id = reinterpret_cast<int&>(split_idx_u);
            num_peers = reinterpret_cast<int&>(num_peers_u);
        }

        SeqlenInfo_t seqlen_info = create_seqlen_info(bidb);
        return Super::create_block_coord(seqlen_info, m_block, bidh_actual, bidb, peer_id, num_peers);
    }

     CUTLASS_DEVICE
    SeqlenInfo_t
    get_seqlen_info(WorkTileInfo const& work_tile) const {
        return create_seqlen_info(work_tile.bidb);
    }


    CUTLASS_DEVICE
    VarlenDynamicPersistentTileScheduler(SharedStorage* const smem_scheduler, Params const& params) : work_info_smem(smem_scheduler), params(params) {};

    CUTLASS_DEVICE
    WorkTileInfo
    tile_idx_to_work_tile(int next_tile_idx, WorkTileInfo const& current_work) const {
        int lane = threadIdx.x % cutlass::NumThreadsPerWarp;
        auto get_num_m_blocks = [&] (int bidb_start) {
            int batch_idx = lane + bidb_start;
            int seqlen = params.seqlen * (!PackGQA ? 1 : params.qhead_per_khead_divmod.divisor);
            if (seqlen > get<0>(TileShape_MNK{})) {
                if (params.seqused_q) {
                    seqlen = batch_idx < params.num_batch ? params.seqused_q[batch_idx] : 0;
                } else if (params.cu_seqlens_q) {
                    int cur_cu_seqlen = batch_idx <= params.num_batch ? params.cu_seqlens_q[batch_idx] : 0;
                    int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
                    seqlen = next_cu_seqlen - cur_cu_seqlen;
                } else {
                    seqlen = params.seqlen;
                }
                if constexpr (PackGQA) { seqlen *= params.qhead_per_khead_divmod.divisor; }
            }
            return batch_idx < params.num_batch && lane < cutlass::NumThreadsPerWarp - 1
                ? cute::ceil_div(seqlen, get<0>(TileShape_MNK{})) : 0;
                // ? params.num_m_blocks_ptr[batch_idx] : 0;
        };

        auto get_num_splits = [&] (int bidb_start) {
            int batch_idx = lane + bidb_start;
            return batch_idx < params.num_batch && lane < cutlass::NumThreadsPerWarp - 1
                ? (!Split ? 1 : (params.num_splits_dynamic_ptr
                                ? params.num_splits_dynamic_ptr[batch_idx]
                                : params.nsplits_divmod.divisor))
                : 0;
        };

        int num_m_blocks = get_num_m_blocks(current_work.bidb);  // Different for each lane
        int num_splits = get_num_splits(current_work.bidb);
        int num_split_m_blocks = !Split ? num_m_blocks : num_m_blocks * num_splits;
        // Cumulative number of blocks for the next 31 batches
        int num_m_blocks_cumulative = warp_prefix_sum(num_split_m_blocks);
        // Total number of blocks for the next 31 batches
        int m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
        // Only the lower 16 bits are the actual bidh
        int current_bidh = !Split ? current_work.bidh : (current_work.bidh & 0x0000FFFF);
        int group_end_tile = current_work.tile_idx - current_work.block - current_bidh * __shfl_sync(0xffffffff, num_split_m_blocks, 0 /*lane*/) + m_blocks_in_group * params.num_head;  // Same for all lanes
        if constexpr (Split) {
            int current_split_idx = (current_work.bidh & 0x00FF0000) >> 16;
            group_end_tile -= current_split_idx * __shfl_sync(0xffffffff, num_m_blocks, 0 /*lane*/);
        }
        int bidb = current_work.bidb;
        // if (blockIdx.x <= 9 && threadIdx.x == 0) {
        //     printf("Before while, blockIdx.x = %d, threadIdx.x = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, cur tile_idx = %d, cur block = %d, cur bidh = %d, num_split_m_blocks = %d, group_end_tile = %d, m_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, current_work.bidb, num_m_blocks, next_tile_idx, current_work.tile_idx, current_work.block, current_bidh, num_split_m_blocks, group_end_tile, m_blocks_in_group);
        // }
        // if (threadIdx.x == 0 && blockIdx.x == 0) { printf("tile_idx = %d, group_end_tile = %d, num_m_blocks_cumulative = %d, m_blocks_in_group = %d\n", current_work.tile_idx, group_end_tile, num_m_blocks_cumulative, m_blocks_in_group); }
        while (group_end_tile <= next_tile_idx) {
            bidb += cutlass::NumThreadsPerWarp - 1;
            if (bidb >= params.num_batch) {
                // if (blockIdx.x <= 9 && threadIdx.x == 0) {
                //     printf("Returning early, blockIdx.x = %d, threadIdx.x = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d, m_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, bidb, num_m_blocks, next_tile_idx, group_end_tile, m_blocks_in_group);
                // }
                return {next_tile_idx, 0, 0, params.num_batch};
            }
            num_m_blocks = get_num_m_blocks(bidb);
            num_splits = get_num_splits(bidb);
            num_split_m_blocks = !Split ? num_m_blocks : num_m_blocks * num_splits;
            num_m_blocks_cumulative = warp_prefix_sum(num_split_m_blocks);
            m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
            group_end_tile += m_blocks_in_group * params.num_head;
            // if (blockIdx.x <= 9 && threadIdx.x == 0) {
            //     printf("Bottom of while, blockIdx.x = %d, threadIdx.x = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d, m_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, bidb, num_m_blocks, next_tile_idx, group_end_tile, m_blocks_in_group);
            // }
        }
        int group_start_tile = group_end_tile - m_blocks_in_group * params.num_head;
        // The next problem to process is the first one that does not have ending tile position
        // that is greater than or equal to tile index.
        int batch_idx_in_group = __popc(__ballot_sync(0xffffffff, group_start_tile + num_m_blocks_cumulative * params.num_head <= next_tile_idx));
        // if (threadIdx.x == 31 || threadIdx.x == 0) { printf("blockIdx.x = %d, tidx %d, group_start_tile = %d, num_m_blocks_cumulative = %d, num_head = %d, next_tile_idx = %d, ballot = %x, batch_idx_in_group = %d\n", blockIdx.x, threadIdx.x, group_start_tile, num_m_blocks_cumulative, params.num_head, next_tile_idx, tmp, batch_idx_in_group); }
        bidb += batch_idx_in_group;
        num_m_blocks = __shfl_sync(0xffffffff, num_m_blocks, batch_idx_in_group);
        if constexpr (Split) { num_splits = __shfl_sync(0xffffffff, num_splits, batch_idx_in_group); }
        int mh_block = next_tile_idx - group_start_tile - (batch_idx_in_group == 0 ? 0 : __shfl_sync(0xffffffff, num_m_blocks_cumulative, batch_idx_in_group - 1)) * params.num_head;
        int bidh = mh_block / num_m_blocks;
        int block = mh_block - bidh * num_m_blocks;
        if constexpr (Split) {
            int bidh_actual = bidh / num_splits;
            int split_idx = bidh - bidh_actual * num_splits;
            // TODO: idk why this gives wrong answer nondeterministically
            // int bidh_actual, split_idx;
            // split_idx = params.head_divmod.divmod(bidh_actual, bidh);
            // Use the top 8 bits to store num_splits and the next 8 bits to store split_idx
            // reinterpret_cast to uint32_t to make sure we're not doing sign extension when we shift
            uint32_t bidh_packed = reinterpret_cast<uint32_t&>(bidh_actual) + (reinterpret_cast<uint32_t&>(split_idx) << 16) + (reinterpret_cast<uint32_t&>(num_splits) << 24);
            // if (threadIdx.x == 0) {
            //     printf("blockIdx.x = %d, group_start_tiled = %d, bidb = %d, batch_idx_in_group = %d, mh_block = %d, num_m_blocks = %d, bidh = %d, bidh_actual = %d, split_idx = %d, num_splits = %d, bidh_packed = %d\n", blockIdx.x, group_start_tile, bidb, batch_idx_in_group, mh_block, num_m_blocks, bidh, bidh_actual, split_idx, num_splits, bidh_packed);
            // }
            bidh = reinterpret_cast<int&>(bidh_packed);
        }
        // if (blockIdx.x <= 9 && threadIdx.x == 0) {
        //     printf("Before returning, blockIdx.x = %d, threadIdx.x = %d, group_start_tile = %d, batch_idx_in_group = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d, m_blocks_in_group = %d, mh_block = %d, bidh = %d, block = %d\n", blockIdx.x, threadIdx.x, group_start_tile, batch_idx_in_group, bidb, num_m_blocks, next_tile_idx, group_end_tile, m_blocks_in_group, mh_block, bidh, block);
        // }
        return {next_tile_idx, block, bidh, bidb};
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        if constexpr (IsProducerWarp) {
            WorkTileInfo work_info = tile_idx_to_work_tile(int(blockIdx.x), {0, 0, 0, 0});
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            return work_info;
        } else {
            return get_next_work<false>({0, 0, 0, 0});
        }
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {
        // Don't arrive at the TileCountSmemEmpty barrier here, because get_initial_work will do that
    }

    CUTLASS_DEVICE
    void
    prefetch_next_work(WorkTileInfo& current_work) const {
        if (threadIdx.x % NumProducerThreads == 0) {
            current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            // thread 0 has the next tile_idx, just need to broadcast to the rest of warp 0
            int new_tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0 /*lane*/);
            WorkTileInfo work_info = {__shfl_sync(0xffffffff, current_work.tile_idx, 1 /*lane*/), current_work.block, current_work.bidh, current_work.bidb};
            work_info = tile_idx_to_work_tile(new_tile_idx, work_info);
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            return work_info;
        } else {
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            int4 work_info = *work_info_smem;
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            return WorkTileInfo{work_info.x, work_info.y, work_info.z, work_info.w};
        }
    }

};

} // flash
