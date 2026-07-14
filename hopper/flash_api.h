#pragma once

#include "torch_api_shim.h"

#include <optional>
#include <vector>

using flash::torch_api::ScalarType;
using flash::torch_api::Tensor;

Tensor mha_fwd_get_scheduler_metadata(
    int batch_size,
    int max_seqlen_q,
    int max_seqlen_k,
    int num_heads,
    int num_heads_k,
    int headdim,
    int headdim_v,
    ScalarType qkv_dtype,
    const Tensor& seqused_k,
    std::optional<Tensor> cu_seqlens_q,
    std::optional<Tensor> cu_seqlens_k,
    std::optional<Tensor> cu_seqlens_k_new,
    std::optional<Tensor> seqused_q,
    std::optional<Tensor> leftpad_k,
    std::optional<int> page_size,
    int max_seqlen_k_new,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    bool has_softcap,
    int num_splits,
    std::optional<bool> pack_gqa,
    int sm_margin);

std::vector<Tensor> mha_fwd(
    Tensor q,
    const Tensor& k,
    const Tensor& v,
    std::optional<Tensor> k_new,
    std::optional<Tensor> v_new,
    std::optional<Tensor> q_v,
    std::optional<Tensor> out,
    std::optional<Tensor> cu_seqlens_q,
    std::optional<Tensor> cu_seqlens_k,
    std::optional<Tensor> cu_seqlens_k_new,
    std::optional<Tensor> seqused_q,
    std::optional<Tensor> seqused_k,
    std::optional<int> max_seqlen_q,
    std::optional<int> max_seqlen_k,
    std::optional<Tensor> page_table,
    std::optional<Tensor> kv_batch_idx,
    std::optional<Tensor> leftpad_k,
    std::optional<Tensor> rotary_cos,
    std::optional<Tensor> rotary_sin,
    std::optional<Tensor> seqlens_rotary,
    std::optional<Tensor> q_descale,
    std::optional<Tensor> k_descale,
    std::optional<Tensor> v_descale,
    float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float softcap,
    bool is_rotary_interleaved,
    std::optional<Tensor> scheduler_metadata,
    int num_splits,
    std::optional<bool> pack_gqa,
    int sm_margin,
    std::optional<Tensor> s_aux,
    int cp_world_size,
    int cp_rank,
    std::optional<Tensor> cp_tot_seqused_k);

std::vector<Tensor> mha_bwd(
    const Tensor& dout,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& out,
    const Tensor& softmax_lse,
    std::optional<Tensor> dq,
    std::optional<Tensor> dk,
    std::optional<Tensor> dv,
    std::optional<Tensor> cu_seqlens_q,
    std::optional<Tensor> cu_seqlens_k,
    std::optional<Tensor> seqused_q,
    std::optional<Tensor> seqused_k,
    std::optional<int> max_seqlen_q,
    std::optional<int> max_seqlen_k,
    float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float softcap,
    bool deterministic,
    int sm_margin);

std::vector<Tensor> mha_combine(
    const Tensor& out_partial,
    const Tensor& lse_partial,
    std::optional<Tensor> out,
    std::optional<ScalarType> out_dtype);
