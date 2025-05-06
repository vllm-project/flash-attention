/******************************************************************************
 * Copyright (c) 2025, Tri Dao, Samsung SDSA.
 ******************************************************************************/

#include <vector>

#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>  // For at::Generator and at::PhiloxCudaState
#include "philox_unpack.cuh"  // For at::cuda::philox::unpack

#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "hardware_info.h"
#include "flash_tree.h"
#include "static_switch.h"


#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace FLASH_NAMESPACE {


//
// Bit hacky but for now hook into the existing set_params_fprop, 
// set_params_splitkv, and set_params_alibi in flash_api.cpp
//
void set_params_fprop(Flash_fwd_params &params,
  // sizes
  const size_t b,
  const size_t seqlen_q,
  const size_t seqlen_k,
  const size_t seqlen_q_rounded,
  const size_t seqlen_k_rounded,
  const size_t h,
  const size_t h_k,
  const size_t d,
  const size_t d_rounded,
  // device pointers
  const at::Tensor q,
  const at::Tensor k,
  const at::Tensor v,
  at::Tensor out,
  void *cu_seqlens_q_d,
  void *cu_seqlens_k_d,
  void *seqused_k,
  void *p_d,
  void *softmax_lse_d,
  float p_dropout,
  float softmax_scale,
  int window_size_left,
  int window_size_right,
  const float softcap,
  bool seqlenq_ngroups_swapped=false,
  const bool unpadded_lse=false);

std::tuple<at::Tensor, at::Tensor>
set_params_splitkv(
  Flash_fwd_params &params,
  const int batch_size,
  const int num_heads,
  const int head_size,
  const int max_seqlen_k,
  const int max_seqlen_q,
  const int head_size_rounded,
  const float p_dropout,
  const int num_splits,
  const int num_sm,
  struct c10::TensorOptions opts);

void set_params_alibi(
  Flash_fwd_params &params,
  std::optional<at::Tensor> &alibi_slopes_,
  int batch_size,
  int num_heads);


void set_params_fprop_tree(
    Flash_fwd_params_tree &params,
    const size_t b,
    const size_t seqlen_q,
    const size_t seqlen_k,
    const size_t seqlen_q_rounded,
    const size_t seqlen_k_rounded,
    const size_t h,
    const size_t h_k,
    const size_t d,
    const size_t d_rounded,
    // device pointers
    const at::Tensor q,
    const at::Tensor k,
    const at::Tensor v,
    at::Tensor out,
    void *cu_seqlens_q_d,
    void *cu_seqlens_k_d,
    void *seqused_k,
    void *p_d,
    void *softmax_lse_d,
    float p_dropout,
    float softmax_scale,
    int window_size_left,
    int window_size_right,
    const float softcap,
    void *tree_mask,
    void *tree_mask_lens,
    bool seqlenq_ngroups_swapped=false,
    const bool unpadded_lse=false
)
{
    set_params_fprop(
        params, 
        b,
        seqlen_q, 
        seqlen_k,
        seqlen_q_rounded,
        seqlen_k_rounded,
        h,
        h_k, 
        d,
        d_rounded, 
        q, 
        k,
        v, 
        out, 
        cu_seqlens_q_d, 
        cu_seqlens_k_d, 
        seqused_k, 
        p_d, 
        softmax_lse_d, 
        p_dropout, 
        softmax_scale, 
        window_size_left, 
        window_size_right, 
        softcap,
        seqlenq_ngroups_swapped,
        unpadded_lse
    );
    params.tree_mask_ptr = static_cast<uint64_t *>(tree_mask);
    params.tree_mask_lens_ptr = static_cast<int *>(tree_mask_lens);
}

void run_mha_fwd_tree(Flash_fwd_params_tree &params, cudaStream_t stream, bool force_split_kernel=false) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                run_mha_fwd_tree_<elem_type, kHeadDim>(params, stream);
            } else {
                run_mha_fwd_splitkv_dispatch_tree<elem_type, kHeadDim>(params, stream);
            }
        });
    });
}


std::vector<at::Tensor>
tree_attention(
    at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    std::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor &cu_seqlens_q,  // b+1
    const at::Tensor &cu_seqlens_k,  // b+1
    std::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
    std::optional<const at::Tensor> &leftpad_k_, // batch_size
    std::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
    std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
    int max_seqlen_q,
    const int max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    const float softcap,
    const bool return_softmax,
    std::optional<at::Generator> gen_,
    const at::Tensor &tree_mask,
    const at::Tensor &tree_mask_lens)
{
    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(tree_mask.dtype() == torch::kUInt64, "TreeAttention only support uint64 data type for tree_mask");
    TORCH_CHECK(tree_mask_lens.dtype() == torch::kInt32, "TreeAttention only support i32 data type for tree_mask_lens"); 
    CHECK_DEVICE(tree_mask); CHECK_DEVICE(tree_mask_lens);
    CHECK_CONTIGUOUS(tree_mask); CHECK_CONTIGUOUS(tree_mask_lens);

    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes();

    const int batch_size = cu_seqlens_q.numel() - 1;
    int num_heads = sizes[1];
    const int head_size = sizes[2];
    const int num_heads_k = paged_KV ? k.size(2) : k.size(1);

    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks = !paged_KV ? 0 : k.size(0);
    const int page_block_size = !paged_KV ? 1 : k.size(1);
    TORCH_CHECK(!paged_KV || page_block_size % 16 == 0, "Paged KV cache block size must be divisible by 16");

    void *cu_seqlens_q_d = cu_seqlens_q.data_ptr();

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = max_seqlen_q == 1 && num_heads > num_heads_k && p_dropout == 0.f && head_size % 8 == 0 && !alibi_slopes_.has_value();
    const int ngroups = num_heads / num_heads_k;
    if (seqlenq_ngroups_swapped) {
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2).reshape({batch_size * ngroups, num_heads_k, head_size});
        max_seqlen_q = ngroups;
        num_heads = num_heads_k;
        cu_seqlens_q_d = nullptr;
    }

    const int total_q = q.sizes()[0];

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");


    CHECK_SHAPE(q, total_q, num_heads, head_size);
    if (!paged_KV) {
        const int total_k = k.size(0);
        CHECK_SHAPE(k, total_k, num_heads_k, head_size);
        CHECK_SHAPE(v, total_k, num_heads_k, head_size);
    } else {
        CHECK_SHAPE(k, num_blocks, page_block_size, num_heads_k, head_size);
        CHECK_SHAPE(v, num_blocks, page_block_size, num_heads_k, head_size);
        CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    }

    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    CHECK_SHAPE(tree_mask_lens, batch_size + 1);
    if (seqused_k.has_value()){
        auto seqused_k_ = seqused_k.value();
        TORCH_CHECK(seqused_k_.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        TORCH_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
        TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
        CHECK_SHAPE(seqused_k_, batch_size);
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, sizes[0], sizes[1], head_size);
        if (seqlenq_ngroups_swapped) {
            // NOTE(woosuk): We create a temporary buffer and copy the result to the `out_` tensor eventually.
            // This is because we reshaped the `q` tensor for the splik-KV optimization, and the `out_` tensor
            // has the same shape as the original `q` tensor, not the reshaped one.
            out = torch::empty_like(q);
        }
    } else {
        out = torch::empty_like(q);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = head_size <= 192 ? round_multiple(head_size, 32) : 256;
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    auto opts = q.options();
    auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));
    at::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
    }
    else {
        p = torch::empty({ 0 }, opts);
    }

    if (zero_tensors) {
        out.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if (return_softmax) {p.zero_();}
    }

    Flash_fwd_params_tree params;
    set_params_fprop_tree(
        params,
        batch_size,
        max_seqlen_q, max_seqlen_k,
        seqlen_q_rounded, seqlen_k_rounded,
        num_heads, num_heads_k,
        head_size, head_size_rounded,
        q, k, v, out,
        cu_seqlens_q_d,
        cu_seqlens_k.data_ptr(),
        seqused_k.has_value() ? seqused_k.value().data_ptr() : nullptr,
        return_softmax ? p.data_ptr() : nullptr,
        softmax_lse.data_ptr(),
        p_dropout,
        softmax_scale,
        -1,
        0,
        softcap,
        tree_mask.data_ptr(),
        tree_mask_lens.data_ptr(),
        seqlenq_ngroups_swapped,
        /*unpadded_lse*/true
    );
    params.total_q = total_q;

    if (paged_KV) {
        params.block_table = block_table.data_ptr<int>();
        params.block_table_batch_stride = block_table.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
    }
    params.page_block_size = page_block_size;
    // Keep references to these tensors to extend their lifetime
    at::Tensor softmax_lse_accum, out_accum;
    if (seqlenq_ngroups_swapped) {
        // Only apply split-k for decoding
        std::tie(softmax_lse_accum, out_accum) =
            set_params_splitkv(params, batch_size, num_heads, head_size,
                               max_seqlen_k, max_seqlen_q, head_size_rounded,
                               p_dropout, /*num_splits*/ 0, get_num_sm(get_current_device()), opts);
    }

    if (leftpad_k_.has_value()) {
        auto leftpad_k = leftpad_k_.value();
        TORCH_CHECK(!paged_KV, "We don't support Paged KV and leftpad_k running at the same time yet");
        TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
        CHECK_DEVICE(leftpad_k);
        CHECK_CONTIGUOUS(leftpad_k);
        CHECK_SHAPE(leftpad_k, batch_size);
        params.leftpad_k = static_cast<int *>(leftpad_k.data_ptr());
    }

    // NOTE(woosuk): Commented out because they are not used in inference.
    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    // int64_t counter_offset = params.b * params.h * 32;
    // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    // // Forward kernel will populate memory with the seed and offset.
    // params.rng_state = reinterpret_cast<uinpft64_t*>(rng_state.data_ptr());

    // if (p_dropout > 0.0)  {
    //     auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
    //         gen_, at::cuda::detail::getDefaultCUDAGenerator());
    //     // See Note [Acquire lock when using random generators]
    //     std::lock_guard<std::mutex> lock(gen->mutex_);
    //     params.philox_args = gen->philox_cuda_state(counter_offset);
    // }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (max_seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd_tree(params, stream, paged_KV);
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    if (seqlenq_ngroups_swapped) {
        int64_t size_before[] = {batch_size, max_seqlen_q, num_heads_k, head_size};
        int64_t size_after[] = {batch_size, num_heads_k * max_seqlen_q, head_size};
        out = out.reshape(size_before).transpose(1, 2);
        if (out_.has_value()) {
            // NOTE(woosuk): In this case, we should avoid `out.reshape(size_after)` because it causes
            // a redundant clone operation. Instead, we directly copy the result to the `out_` tensor.
            out_.value().view({batch_size, num_heads_k, max_seqlen_q, head_size}).copy_(out);
            out = out_.value();
        } else {
            out = out.reshape(size_after);
        }
        // NOTE(woosuk): The two lines are not needed because out_padded and q_padded are not used.
        // out_padded = out_padded.reshape(size_before).transpose(1, 2).reshape(size_after);
        // q_padded = q_padded.reshape(size_before).transpose(1, 2).reshape(size_after);
        int64_t lse_size_before[] = {num_heads, batch_size, max_seqlen_q};
        int64_t lse_size_after[] = {num_heads * max_seqlen_q, batch_size};
        softmax_lse = softmax_lse.reshape(lse_size_before).transpose(1, 2).reshape(lse_size_after);
    }

    return {out, softmax_lse};
}

} // FLASH_NAMESPACE