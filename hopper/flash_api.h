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
    std::optional<const Tensor>& cu_seqlens_q,
    std::optional<const Tensor>& cu_seqlens_k,
    std::optional<const Tensor>& cu_seqlens_k_new,
    std::optional<const Tensor>& seqused_q,
    std::optional<const Tensor>& leftpad_k,
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
    Tensor& q,
    const Tensor& k,
    const Tensor& v,
    std::optional<const Tensor>& k_new,
    std::optional<const Tensor>& v_new,
    std::optional<const Tensor>& q_v,
    std::optional<Tensor>& out,
    std::optional<const Tensor>& cu_seqlens_q,
    std::optional<const Tensor>& cu_seqlens_k,
    std::optional<const Tensor>& cu_seqlens_k_new,
    std::optional<const Tensor>& seqused_q,
    std::optional<const Tensor>& seqused_k,
    std::optional<int> max_seqlen_q,
    std::optional<int> max_seqlen_k,
    std::optional<const Tensor>& page_table,
    std::optional<const Tensor>& kv_batch_idx,
    std::optional<const Tensor>& leftpad_k,
    std::optional<const Tensor>& rotary_cos,
    std::optional<const Tensor>& rotary_sin,
    std::optional<const Tensor>& seqlens_rotary,
    std::optional<Tensor>& q_descale,
    std::optional<Tensor>& k_descale,
    std::optional<Tensor>& v_descale,
    float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float softcap,
    bool is_rotary_interleaved,
    std::optional<Tensor>& scheduler_metadata,
    int num_splits,
    std::optional<bool> pack_gqa,
    int sm_margin,
    std::optional<const Tensor>& s_aux,
    int cp_world_size,
    int cp_rank,
    std::optional<const Tensor>& cp_tot_seqused_k);

std::vector<Tensor> mha_bwd(
    const Tensor& dout,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& out,
    const Tensor& softmax_lse,
    std::optional<Tensor>& dq,
    std::optional<Tensor>& dk,
    std::optional<Tensor>& dv,
    std::optional<const Tensor>& cu_seqlens_q,
    std::optional<const Tensor>& cu_seqlens_k,
    std::optional<const Tensor>& seqused_q,
    std::optional<const Tensor>& seqused_k,
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
