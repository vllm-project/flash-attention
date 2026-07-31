#include <Python.h>

#include <torch/csrc/stable/library.h>

#include "flash_api.h"

extern "C" PyObject* PyInit__vllm_fa3_C(void) {
    static PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "_vllm_fa3_C", nullptr, -1, nullptr};
    return PyModule_Create(&module);
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
    m.impl("fwd", FLASH_STABLE_TORCH_BOX(mha_fwd));
    m.impl("get_scheduler_metadata",
           FLASH_STABLE_TORCH_BOX(mha_fwd_get_scheduler_metadata));
}
