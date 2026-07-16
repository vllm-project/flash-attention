#include "registration.h"

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/library.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

using torch::stable::Tensor;
using torch::headeronly::ScalarType;

/**
 *  Externs for the flash_attn ops to be exposed as a pytorch library
 */

// b: batch_size
// b_k: batch_size_k
// s_q: seqlen_q
// s_k: seqlen_k
// s_k_new: seqlen_k_new
// h: num_heads
// h_k: num_heads_k
// d: head_size
std::vector<Tensor>
mha_fwd(Tensor &q,   // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        const Tensor &k,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table.
        const Tensor &v,  // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table.
        std::optional<const Tensor> &k_new_,  // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is cu_seqlens_k_new
        std::optional<const Tensor> &v_new_,  // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv) if there is cu_seqlens_k_new
        std::optional<const Tensor> &q_v_,  // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
        std::optional<Tensor> &out_,  // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        std::optional<const Tensor> &cu_seqlens_q_,  // b+1
        std::optional<const Tensor> &cu_seqlens_k_,  // b+1
        std::optional<const Tensor> &cu_seqlens_k_new_,  // b+1
        std::optional<const Tensor> &seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
        std::optional<const Tensor> &seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
        std::optional<int> max_seqlen_q_,
        // TODO: check if we need max_seqlen_k
        std::optional<int> max_seqlen_k_,
        std::optional<const Tensor> &page_table_, // (b_k, max_num_pages_per_seq)
        std::optional<const Tensor> &kv_batch_idx_, // b. indices to index into the KV cache
        std::optional<const Tensor> &leftpad_k_, // b
        std::optional<const Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
        std::optional<const Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
        std::optional<const Tensor> &seqlens_rotary_, // b
        std::optional<Tensor> &q_descale_,  // (b, h_k), not (b, h)
        std::optional<Tensor> &k_descale_,  // (b, h_k)
        std::optional<Tensor> &v_descale_,  // (b, h_k)
        float const softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        float const softcap,
        bool const is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
        std::optional<Tensor> &scheduler_metadata_,  // (b + 1)
        int num_splits,
        std::optional<bool> pack_gqa_,
        int const sm_margin,
        std::optional<const Tensor> &s_aux_,
        int const cp_world_size,
        int const cp_rank,
        std::optional<const Tensor> &cp_tot_seqused_k
);

// Only applicable to the case where seqused_k (i.e. cache_seqlens) is available
Tensor
mha_fwd_get_scheduler_metadata(
        int batch_size,
        int max_seqlen_q,
        int max_seqlen_k,
        int num_heads,
        int num_heads_k,
        int headdim,
        int headdim_v,
        ScalarType qkv_dtype,
        const Tensor &seqused_k, // b
        std::optional<const Tensor> &cu_seqlens_q_,  // b+1
        std::optional<const Tensor> &cu_seqlens_k_,  // b+1
        std::optional<const Tensor> &cu_seqlens_k_new_,  // b+1
        std::optional<const Tensor> &seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
        std::optional<const Tensor> &leftpad_k_, // b
        std::optional<int> page_size,
        int max_seqlen_k_new,  // 0 means we're not appending new KV
        bool is_causal,
        int window_size_left,
        int window_size_right,
        bool has_softcap,
        int num_splits,
        std::optional<bool> pack_gqa_,
        int const sm_margin
);

namespace {
// TODO: Create a clean adapter like pytorch_shim.h (legacy code) to convert types to 
// allowed TORCH_BOX types instead of creating wrappers for each kernel. 

inline int narrow_int64_to_int(int64_t v, const char* name) {
  STD_TORCH_CHECK(v <= std::numeric_limits<int>::max(),
                  name, " is too large to convert to int");
  STD_TORCH_CHECK(v >= std::numeric_limits<int>::min(),
                  name, " is too small to convert to int");
  return static_cast<int>(v);
}

inline float narrow_double_to_float(double v, const char* name) {
  STD_TORCH_CHECK(std::abs(v) <= static_cast<double>(std::numeric_limits<float>::max()),
                  name, " is too large to convert to float");
  return static_cast<float>(v);
}

// Bridge TORCH_BOX-friendly optional<Tensor> into the
// optional<const Tensor>& form expected by mha_* implementations.
inline std::optional<const Tensor>& as_const_tensor_opt(
    std::optional<Tensor>& opt) {
  return const_cast<std::optional<const Tensor>&>(
      reinterpret_cast<const std::optional<const Tensor>&>(opt));
}

inline std::optional<int> narrow_optional_int(const std::optional<int64_t>& v) {
  if (!v.has_value()) {
    return std::nullopt;
  }
  return narrow_int64_to_int(*v, "optional int");
}

std::vector<Tensor> mha_fwd_stable(
    Tensor q,
    Tensor k,
    Tensor v,
    std::optional<Tensor> k_new,
    std::optional<Tensor> v_new,
    std::optional<Tensor> q_v,
    std::optional<Tensor> out,
    std::optional<Tensor> cu_seqlens_q,
    std::optional<Tensor> cu_seqlens_k,
    std::optional<Tensor> cu_seqlens_k_new,
    std::optional<Tensor> seqused_q,
    std::optional<Tensor> seqused_k,
    std::optional<int64_t> max_seqlen_q,
    std::optional<int64_t> max_seqlen_k,
    std::optional<Tensor> page_table,
    std::optional<Tensor> kv_batch_idx,
    std::optional<Tensor> leftpad_k,
    std::optional<Tensor> rotary_cos,
    std::optional<Tensor> rotary_sin,
    std::optional<Tensor> seqlens_rotary,
    std::optional<Tensor> q_descale,
    std::optional<Tensor> k_descale,
    std::optional<Tensor> v_descale,
    double softmax_scale,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    double softcap,
    bool is_rotary_interleaved,
    std::optional<Tensor> scheduler_metadata,
    int64_t num_splits,
    std::optional<bool> pack_gqa,
    int64_t sm_margin,
    std::optional<Tensor> s_aux,
    int64_t cp_world_size,
    int64_t cp_rank,
    std::optional<Tensor> cp_tot_seqused_k) {
  return mha_fwd(
      q, k, v,
      as_const_tensor_opt(k_new),
      as_const_tensor_opt(v_new),
      as_const_tensor_opt(q_v),
      out,
      as_const_tensor_opt(cu_seqlens_q),
      as_const_tensor_opt(cu_seqlens_k),
      as_const_tensor_opt(cu_seqlens_k_new),
      as_const_tensor_opt(seqused_q),
      as_const_tensor_opt(seqused_k),
      narrow_optional_int(max_seqlen_q),
      narrow_optional_int(max_seqlen_k),
      as_const_tensor_opt(page_table),
      as_const_tensor_opt(kv_batch_idx),
      as_const_tensor_opt(leftpad_k),
      as_const_tensor_opt(rotary_cos),
      as_const_tensor_opt(rotary_sin),
      as_const_tensor_opt(seqlens_rotary),
      q_descale,
      k_descale,
      v_descale,
      narrow_double_to_float(softmax_scale, "softmax_scale"),
      is_causal,
      narrow_int64_to_int(window_size_left, "window_size_left"),
      narrow_int64_to_int(window_size_right, "window_size_right"),
      narrow_double_to_float(softcap, "softcap"),
      is_rotary_interleaved,
      scheduler_metadata,
      narrow_int64_to_int(num_splits, "num_splits"),
      pack_gqa,
      narrow_int64_to_int(sm_margin, "sm_margin"),
      as_const_tensor_opt(s_aux),
      narrow_int64_to_int(cp_world_size, "cp_world_size"),
      narrow_int64_to_int(cp_rank, "cp_rank"),
      as_const_tensor_opt(cp_tot_seqused_k));
}

Tensor mha_fwd_get_scheduler_metadata_stable(
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    int64_t num_heads,
    int64_t num_heads_k,
    int64_t headdim,
    int64_t headdim_v,
    ScalarType qkv_dtype,
    Tensor seqused_k,
    std::optional<Tensor> cu_seqlens_q,
    std::optional<Tensor> cu_seqlens_k,
    std::optional<Tensor> cu_seqlens_k_new,
    std::optional<Tensor> seqused_q,
    std::optional<Tensor> leftpad_k,
    std::optional<int64_t> page_size,
    int64_t max_seqlen_k_new,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    bool has_softcap,
    int64_t num_splits,
    std::optional<bool> pack_gqa,
    int64_t sm_margin) {
  return mha_fwd_get_scheduler_metadata(
      narrow_int64_to_int(batch_size, "batch_size"),
      narrow_int64_to_int(max_seqlen_q, "max_seqlen_q"),
      narrow_int64_to_int(max_seqlen_k, "max_seqlen_k"),
      narrow_int64_to_int(num_heads, "num_heads"),
      narrow_int64_to_int(num_heads_k, "num_heads_k"),
      narrow_int64_to_int(headdim, "headdim"),
      narrow_int64_to_int(headdim_v, "headdim_v"),
      qkv_dtype,
      seqused_k,
      as_const_tensor_opt(cu_seqlens_q),
      as_const_tensor_opt(cu_seqlens_k),
      as_const_tensor_opt(cu_seqlens_k_new),
      as_const_tensor_opt(seqused_q),
      as_const_tensor_opt(leftpad_k),
      narrow_optional_int(page_size),
      narrow_int64_to_int(max_seqlen_k_new, "max_seqlen_k_new"),
      is_causal,
      narrow_int64_to_int(window_size_left, "window_size_left"),
      narrow_int64_to_int(window_size_right, "window_size_right"),
      has_softcap,
      narrow_int64_to_int(num_splits, "num_splits"),
      pack_gqa,
      narrow_int64_to_int(sm_margin, "sm_margin"));
}

} // namespace

/**
 *  Torch Library Registration
 */
STABLE_TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("fwd(Tensor!  q,"
            "    Tensor   k,"
            "    Tensor   v,"
            "    Tensor?  k_new,"
            "    Tensor?  v_new,"
            "    Tensor?  q_v,"
            "    Tensor!? out,"
            "    Tensor?  cu_seqlens_q,"
            "    Tensor?  cu_seqlens_k,"
            "    Tensor?  cu_seqlens_k_new,"
            "    Tensor?  seqused_q,"
            "    Tensor?  seqused_k,"
            "    int?     max_seqlen_q,"
            "    int?     max_seqlen_k,"
            "    Tensor?  page_table,"
            "    Tensor?  kv_batch_idx,"
            "    Tensor?  leftpad_k,"
            "    Tensor?  rotary_cos,"
            "    Tensor?  rotary_sin,"
            "    Tensor?  seqlens_rotary,"
            "    Tensor?  q_descale,"
            "    Tensor?  k_descale,"
            "    Tensor?  v_descale,"
            "    float    softmax_scale,"
            "    bool     is_causal,"
            "    int      window_size_left,"
            "    int      window_size_right,"
            "    float    softcap,"
            "    bool     is_rotary_interleaved,"
            "    Tensor?  scheduler_metadata,"
            "    int      num_splits,"
            "    bool?    pack_gqa,"
            "    int      sm_margin,"
            "    Tensor?  s_aux,"
            "    int      cp_world_size,"
            "    int      cp_rank,"
            "    Tensor?  cp_tot_seqused_k) -> Tensor[]");

    ops.def("get_scheduler_metadata("
            "    int      batch_size,"
            "    int      max_seqlen_q,"
            "    int      max_seqlen_k,"
            "    int      num_heads,"
            "    int      num_heads_k,"
            "    int      headdim,"
            "    int      headdim_v,"
            "    ScalarType qkv_dtype,"
            "    Tensor   seqused_k,"
            "    Tensor?  cu_seqlens_q,"
            "    Tensor?  cu_seqlens_k,"
            "    Tensor?  cu_seqlens_k_new,"
            "    Tensor?  seqused_q,"
            "    Tensor?  leftpad_k,"
            "    int?     page_size,"
            "    int      max_seqlen_k_new," // 0 means we're not appending new KV
            "    bool     is_causal,"
            "    int      window_size_left,"
            "    int      window_size_right,"
            "    bool     has_softcap,"
            "    int      num_splits,"
            "    bool?    pack_gqa,"
            "    int      sm_margin) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
    ops.impl("fwd", TORCH_BOX(&mha_fwd_stable));
    ops.impl("get_scheduler_metadata",
             TORCH_BOX(&mha_fwd_get_scheduler_metadata_stable));
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME);
