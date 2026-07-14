#include <Python.h>

#include <torch/csrc/stable/library.h>

#include "flash_api.h"

extern "C" PyObject* PyInit__vllm_fa3_C(void) {
    static PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "_vllm_fa3_C", nullptr, -1, nullptr};
    return PyModule_Create(&module);
}

void boxed_mha_fwd(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs
) {
    auto q = torch::stable::detail::to<Tensor>(stack[0]);
    auto k = torch::stable::detail::to<Tensor>(stack[1]);
    auto v = torch::stable::detail::to<Tensor>(stack[2]);
    auto k_new = torch::stable::detail::to<std::optional<Tensor>>(stack[3]);
    auto v_new = torch::stable::detail::to<std::optional<Tensor>>(stack[4]);
    auto q_v = torch::stable::detail::to<std::optional<Tensor>>(stack[5]);
    auto out = torch::stable::detail::to<std::optional<Tensor>>(stack[6]);
    auto cu_seqlens_q = torch::stable::detail::to<std::optional<Tensor>>(stack[7]);
    auto cu_seqlens_k = torch::stable::detail::to<std::optional<Tensor>>(stack[8]);
    auto cu_seqlens_k_new = torch::stable::detail::to<std::optional<Tensor>>(stack[9]);
    auto seqused_q = torch::stable::detail::to<std::optional<Tensor>>(stack[10]);
    auto seqused_k = torch::stable::detail::to<std::optional<Tensor>>(stack[11]);
    auto max_seqlen_q = torch::stable::detail::to<std::optional<int64_t>>(stack[12]);
    auto max_seqlen_k = torch::stable::detail::to<std::optional<int64_t>>(stack[13]);
    auto page_table = torch::stable::detail::to<std::optional<Tensor>>(stack[14]);
    auto kv_batch_idx = torch::stable::detail::to<std::optional<Tensor>>(stack[15]);
    auto leftpad_k = torch::stable::detail::to<std::optional<Tensor>>(stack[16]);
    auto rotary_cos = torch::stable::detail::to<std::optional<Tensor>>(stack[17]);
    auto rotary_sin = torch::stable::detail::to<std::optional<Tensor>>(stack[18]);
    auto seqlens_rotary = torch::stable::detail::to<std::optional<Tensor>>(stack[19]);
    auto q_descale = torch::stable::detail::to<std::optional<Tensor>>(stack[20]);
    auto k_descale = torch::stable::detail::to<std::optional<Tensor>>(stack[21]);
    auto v_descale = torch::stable::detail::to<std::optional<Tensor>>(stack[22]);
    auto softmax_scale = torch::stable::detail::to<double>(stack[23]);
    auto is_causal = torch::stable::detail::to<bool>(stack[24]);
    auto window_size_left = torch::stable::detail::to<int64_t>(stack[25]);
    auto window_size_right = torch::stable::detail::to<int64_t>(stack[26]);
    auto softcap = torch::stable::detail::to<double>(stack[27]);
    auto is_rotary_interleaved = torch::stable::detail::to<bool>(stack[28]);
    auto scheduler_metadata = torch::stable::detail::to<std::optional<Tensor>>(stack[29]);
    auto num_splits = torch::stable::detail::to<int64_t>(stack[30]);
    auto pack_gqa = torch::stable::detail::to<std::optional<bool>>(stack[31]);
    auto sm_margin = torch::stable::detail::to<int64_t>(stack[32]);
    auto s_aux = torch::stable::detail::to<std::optional<Tensor>>(stack[33]);
    auto cp_world_size = torch::stable::detail::to<int64_t>(stack[34]);
    auto cp_rank = torch::stable::detail::to<int64_t>(stack[35]);
    auto cp_tot_seqused_k = torch::stable::detail::to<std::optional<Tensor>>(stack[36]);

    auto outputs = mha_fwd(
        q, k, v,
        flash::torch_api::as_optional_const(k_new),
        flash::torch_api::as_optional_const(v_new),
        flash::torch_api::as_optional_const(q_v),
        out,
        flash::torch_api::as_optional_const(cu_seqlens_q),
        flash::torch_api::as_optional_const(cu_seqlens_k),
        flash::torch_api::as_optional_const(cu_seqlens_k_new),
        flash::torch_api::as_optional_const(seqused_q),
        flash::torch_api::as_optional_const(seqused_k),
        max_seqlen_q, max_seqlen_k,
        flash::torch_api::as_optional_const(page_table),
        flash::torch_api::as_optional_const(kv_batch_idx),
        flash::torch_api::as_optional_const(leftpad_k),
        flash::torch_api::as_optional_const(rotary_cos),
        flash::torch_api::as_optional_const(rotary_sin),
        flash::torch_api::as_optional_const(seqlens_rotary),
        q_descale, k_descale, v_descale,
        softmax_scale, is_causal, window_size_left, window_size_right,
        softcap, is_rotary_interleaved, scheduler_metadata, num_splits,
        pack_gqa, sm_margin,
        flash::torch_api::as_optional_const(s_aux),
        cp_world_size, cp_rank,
        flash::torch_api::as_optional_const(cp_tot_seqused_k));

    stack[0] = torch::stable::detail::from(outputs);
}

void boxed_mha_fwd_get_scheduler_metadata(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs
) {
    auto batch_size = torch::stable::detail::to<int64_t>(stack[0]);
    auto max_seqlen_q = torch::stable::detail::to<int64_t>(stack[1]);
    auto max_seqlen_k = torch::stable::detail::to<int64_t>(stack[2]);
    auto num_heads = torch::stable::detail::to<int64_t>(stack[3]);
    auto num_heads_k = torch::stable::detail::to<int64_t>(stack[4]);
    auto headdim = torch::stable::detail::to<int64_t>(stack[5]);
    auto headdim_v = torch::stable::detail::to<int64_t>(stack[6]);
    auto qkv_dtype = torch::stable::detail::to<torch::headeronly::ScalarType>(stack[7]);
    auto seqused_k = torch::stable::detail::to<Tensor>(stack[8]);
    auto cu_seqlens_q = torch::stable::detail::to<std::optional<Tensor>>(stack[9]);
    auto cu_seqlens_k = torch::stable::detail::to<std::optional<Tensor>>(stack[10]);
    auto cu_seqlens_k_new = torch::stable::detail::to<std::optional<Tensor>>(stack[11]);
    auto seqused_q = torch::stable::detail::to<std::optional<Tensor>>(stack[12]);
    auto leftpad_k = torch::stable::detail::to<std::optional<Tensor>>(stack[13]);
    auto page_size = torch::stable::detail::to<std::optional<int64_t>>(stack[14]);
    auto max_seqlen_k_new = torch::stable::detail::to<int64_t>(stack[15]);
    auto is_causal = torch::stable::detail::to<bool>(stack[16]);
    auto window_size_left = torch::stable::detail::to<int64_t>(stack[17]);
    auto window_size_right = torch::stable::detail::to<int64_t>(stack[18]);
    auto has_softcap = torch::stable::detail::to<bool>(stack[19]);
    auto num_splits = torch::stable::detail::to<int64_t>(stack[20]);
    auto pack_gqa = torch::stable::detail::to<std::optional<bool>>(stack[21]);
    auto sm_margin = torch::stable::detail::to<int64_t>(stack[22]);

    auto scheduler_metadata = mha_fwd_get_scheduler_metadata(
        batch_size, max_seqlen_q, max_seqlen_k, num_heads, num_heads_k,
        headdim, headdim_v, qkv_dtype, seqused_k,
        flash::torch_api::as_optional_const(cu_seqlens_q),
        flash::torch_api::as_optional_const(cu_seqlens_k),
        flash::torch_api::as_optional_const(cu_seqlens_k_new),
        flash::torch_api::as_optional_const(seqused_q),
        flash::torch_api::as_optional_const(leftpad_k),
        page_size, max_seqlen_k_new, is_causal, window_size_left,
        window_size_right, has_softcap, num_splits, pack_gqa, sm_margin);

    stack[0] = torch::stable::detail::from(scheduler_metadata);
}

STABLE_TORCH_LIBRARY(_vllm_fa3_C, m) {
    m.def("fwd("
        "Tensor! q,"
        "Tensor k,"
        "Tensor v,"
        "Tensor? k_new,"
        "Tensor? v_new,"
        "Tensor? q_v,"
        "Tensor!? out,"
        "Tensor? cu_seqlens_q,"
        "Tensor? cu_seqlens_k,"
        "Tensor? cu_seqlens_k_new,"
        "Tensor? seqused_q,"
        "Tensor? seqused_k,"
        "int? max_seqlen_q,"
        "int? max_seqlen_k,"
        "Tensor? page_table,"
        "Tensor? kv_batch_idx,"
        "Tensor? leftpad_k,"
        "Tensor? rotary_cos,"
        "Tensor? rotary_sin,"
        "Tensor? seqlens_rotary,"
        "Tensor? q_descale,"
        "Tensor? k_descale,"
        "Tensor? v_descale,"
        "float softmax_scale,"
        "bool is_causal,"
        "int window_size_left,"
        "int window_size_right,"
        "float softcap,"
        "bool is_rotary_interleaved,"
        "Tensor? scheduler_metadata,"
        "int num_splits,"
        "bool? pack_gqa,"
        "int sm_margin,"
        "Tensor? s_aux,"
        "int cp_world_size,"
        "int cp_rank,"
        "Tensor? cp_tot_seqused_k) -> Tensor[]");
    m.def("get_scheduler_metadata("
        "int batch_size,"
        "int max_seqlen_q,"
        "int max_seqlen_k,"
        "int num_heads,"
        "int num_heads_k,"
        "int headdim,"
        "int headdim_v,"
        "ScalarType qkv_dtype,"
        "Tensor seqused_k,"
        "Tensor? cu_seqlens_q,"
        "Tensor? cu_seqlens_k,"
        "Tensor? cu_seqlens_k_new,"
        "Tensor? seqused_q,"
        "Tensor? leftpad_k,"
        "int? page_size,"
        "int max_seqlen_k_new,"
        "bool is_causal,"
        "int window_size_left,"
        "int window_size_right,"
        "bool has_softcap,"
        "int num_splits,"
        "bool? pack_gqa,"
        "int sm_margin) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(_vllm_fa3_C, CUDA, m) {
    m.impl("fwd", &boxed_mha_fwd);
    m.impl("get_scheduler_metadata", &boxed_mha_fwd_get_scheduler_metadata);
}
