#include "registration.h"
#include "pytorch_shim.h"
#include "namespace_config.h"

#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#include <cmath>

/**
 *  Externs for the flash_attn ops to be exposed as a pytorch library
 */

namespace FLASH_NAMESPACE {

////////////////////////////// From flash_api.cpp //////////////////////////////

std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
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
               bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool return_softmax,
               std::optional<at::Generator> gen_);

std::vector<at::Tensor>
mha_fwd_kvcache(at::Tensor &q,                 // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &kcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &vcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                std::optional<const at::Tensor> &k_, // batch_size x seqlen_knew x num_heads_k x head_size
                std::optional<const at::Tensor> &v_, // batch_size x seqlen_knew x num_heads_k x head_size
                std::optional<const at::Tensor> &seqlens_k_, // batch_size
                std::optional<const at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
                std::optional<const at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
                std::optional<const at::Tensor> &cache_batch_idx_, // indices to index into the KV cache
                std::optional<const at::Tensor> &leftpad_k_, // batch_size
                std::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
                std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
                std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
                const float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                const float softcap,
                bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                int num_splits);

/////////////////////////// From flash_api_sparse.cpp //////////////////////////

std::vector<at::Tensor>
mha_fwd_sparse(at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
               const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
               const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
               const at::Tensor &block_count,
               const at::Tensor &block_offset,
               const at::Tensor &column_count,
               const at::Tensor &column_index,
               const std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
               const std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
               const double p_dropout,
               const double softmax_scale,
               bool is_causal,
               const double softcap,
               const bool return_softmax,
               std::optional<at::Generator> gen_);

std::vector<at::Tensor>
mha_varlen_fwd_sparse(at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                      const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i.
                      const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i.
                      const at::Tensor &block_count,
                      const at::Tensor &block_offset,
                      const at::Tensor &column_count,
                      const at::Tensor &column_index,
                      const std::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                      const at::Tensor &cu_seqlens_q,  // b+1
                      const at::Tensor &cu_seqlens_k,  // b+1
                      const std::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
                      const std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
                      int64_t max_seqlen_q,
                      const int64_t max_seqlen_k,
                      const double p_dropout,
                      const double softmax_scale,
                      const bool zero_tensors,
                      bool is_causal,
                      const double softcap,
                      const bool return_softmax,
                      std::optional<at::Generator> gen_);

// A convenience wrapper to compute an auxiliary tensor abs_s (sum of |scores| per head/token)
// for varlen fwd on the CUDA extension side, so Python tests can fetch it directly.
static std::vector<at::Tensor>
varlen_fwd_with_abs_aux(at::Tensor &q,
                                                const at::Tensor &k,
                                                const at::Tensor &v,
                                                const std::optional<at::Tensor> &out_,
                                                const at::Tensor &cu_seqlens_q,
                                                const at::Tensor &cu_seqlens_k,
                                                const std::optional<at::Tensor> &seqused_k,
                                                const std::optional<at::Tensor> &leftpad_k_,
                                                const std::optional<at::Tensor> &block_table_,
                                                const std::optional<at::Tensor> &alibi_slopes_,
                                                int64_t max_seqlen_q,
                                                int64_t max_seqlen_k,
                                                double p_dropout,
                                                double softmax_scale,
                                                bool zero_tensors,
                                                bool is_causal,
                                                int64_t window_size_left,
                                                int64_t window_size_right,
                                                double softcap,
                                                bool return_softmax,
                                                std::optional<at::Generator> gen_) {
        // Make mutable copies to call underlying API which expects non-const refs and int/float
        std::optional<at::Tensor> out_mut = out_;
        std::optional<at::Tensor> seqused_k_mut = seqused_k;
                std::optional<const at::Tensor> leftpad_k_mut =
                        leftpad_k_.has_value() ? std::optional<const at::Tensor>(leftpad_k_.value()) : std::nullopt;
        std::optional<at::Tensor> block_table_mut = block_table_;
        std::optional<at::Tensor> alibi_slopes_mut = alibi_slopes_;
                std::optional<at::Generator> gen_mut = gen_;

        // First run the canonical varlen forward to get out/lse like usual
        auto base = mha_varlen_fwd(
                q, k, v, out_mut,
                                        cu_seqlens_q, cu_seqlens_k, 
                seqused_k_mut, leftpad_k_mut, block_table_mut, alibi_slopes_mut,
                static_cast<int>(max_seqlen_q), static_cast<int>(max_seqlen_k),
                static_cast<float>(p_dropout), static_cast<float>(softmax_scale),
                zero_tensors, is_causal,
                static_cast<int>(window_size_left), static_cast<int>(window_size_right),
                static_cast<float>(softcap), return_softmax, gen_mut);
        TORCH_CHECK(base.size() >= 2, "varlen_fwd returned unexpected outputs");
        auto out = base[0];
        auto lse = base[1];

                // Compute abs_s on GPU with/without paging
                const bool is_paged = (k.dim() == 4) && block_table_.has_value();
                const auto total_q = q.size(0);
                const auto Hq = q.size(1);
                const auto D = q.size(2);
                const auto Hkv = is_paged ? k.size(2) : k.size(1);
                TORCH_CHECK(Hq % Hkv == 0, "Grouped-query requires Hq multiple of Hkv");
                const int rep = (int)(Hq / Hkv);
                auto opts_f32 = q.options().dtype(at::kFloat);
                const int B = (int)cu_seqlens_q.size(0) - 1;
                float ref_scale = 1.0f / std::sqrt(static_cast<float>(D));

                        if (!is_paged) {
                                // 非分页：返回 (Hq, total_q) 的 abs_s
                                // - 依据 cu_seqlens_{q,k} 在“线性”K/V 上按批切片；
                                // - 重复 K/V 头到 Hq，做 per-head batched GEMM，scores 采用 1/sqrt(D) 的缩放；
                                // - 不应用 causal mask，与测试中的参考实现一致；
                                // - 沿 key 维求和，得到 (Hq, Sq) 并写回到全局 (Hq, total_q) 的相应区间。
                                at::Tensor abs_s = torch::zeros({Hq, total_q}, opts_f32);
                        for (int b = 0; b < B; ++b) {
                                const int64_t q0 = cu_seqlens_q.index({b}).item<int64_t>();
                                const int64_t q1 = cu_seqlens_q.index({b+1}).item<int64_t>();
                                const int64_t k0 = cu_seqlens_k.index({b}).item<int64_t>();
                                const int64_t k1 = cu_seqlens_k.index({b+1}).item<int64_t>();
                                const int64_t Sq = q1 - q0;
                                const int64_t Sk = k1 - k0;
                                if (Sq == 0 || Sk == 0) continue;

                                auto qb = q.index({at::indexing::Slice(q0, q1)});      // (Sq, Hq, D)
                                auto kb = k.index({at::indexing::Slice(k0, k1)});      // (Sk, Hkv, D)
                                auto kb_r = kb.repeat({1, rep, 1});                    // (Sk, Hq, D)
                                auto qh = qb.permute({1, 0, 2}).to(at::kFloat) * ref_scale; // (Hq, Sq, D)
                                // 提升到 fp32 与 qh 匹配（半精度 * fp32 在某些路径触发期望 dtype 错误）
                                auto kh = kb_r.permute({1, 2, 0}).to(at::kFloat);      // (Hq, D, Sk)
                                auto scores = at::bmm(qh, kh);                         // (Hq, Sq, Sk)
                                auto sums = scores.abs().sum(-1);                      // (Hq, Sq)
                                abs_s.index_put_({at::indexing::Slice(), at::indexing::Slice(q0, q1)}, sums);
                        }
                        return {out, lse, abs_s};
                        } else {
                                // 分页（vLLM page）：返回 (B, Hq, max_blocks) 的每页 abs_s 汇总
                                // - page 的定义与 vLLM 的 KV cache 页一致：k 的形状为 (num_blocks, page_sz, Hkv, D)，
                                //   block_table[b, p] 给出第 b 个序列第 p 页对应的物理块索引；
                                // - used_k = seqused_k[b] 用于确定该序列使用的有效 key 数，末页可能不满；
                                // - 对每个有效页：取物理块 k_page，并在 head 维按 rep 重复到 Hq，做 batched GEMM；
                                // - 对 (Sq, page_len) 聚合得到每页每头的 |scores| 之和，写入 per_page[b, :, p]。
                                TORCH_CHECK(seqused_k.has_value(), "seqused_k required for paged-KV per-page stats");
                                const auto page_sz = k.size(1);
                                const auto max_blocks = block_table_.value().size(1);
                                at::Tensor per_page = torch::zeros({B, (int)Hq, (int)max_blocks}, opts_f32);
                        for (int b = 0; b < B; ++b) {
                                const int64_t q0 = cu_seqlens_q.index({b}).item<int64_t>();
                                const int64_t q1 = cu_seqlens_q.index({b+1}).item<int64_t>();
                                const int64_t Sq = q1 - q0;
                                if (Sq == 0) continue;
                                auto qb = q.index({at::indexing::Slice(q0, q1)});      // (Sq, Hq, D)
                                auto qh = qb.permute({1, 0, 2}).to(at::kFloat) * ref_scale; // (Hq, Sq, D)

                                const int64_t used_k = seqused_k.value().index({b}).item<int64_t>();
                                const int64_t nblocks = (used_k + (int64_t)page_sz - 1) / (int64_t)page_sz;
                                for (int64_t p = 0; p < nblocks; ++p) {
                                        const int64_t blk_id = block_table_.value().index({b, p}).item<int64_t>();
                                        const int64_t keys_in_page = (p == nblocks - 1) ? (used_k - p * (int64_t)page_sz) : (int64_t)page_sz;
                                        if (keys_in_page <= 0) continue;
                                        auto k_page = k.index({blk_id});                   // (page_sz, Hkv, D)
                                        auto k_ph = k_page.permute({1, 2, 0});             // (Hkv, D, page_sz)
                                        auto k_rep = k_ph.repeat({rep, 1, 1});             // (Hq, D, page_sz)
                                        // 与非分页路径保持一致：两侧统一提升到 fp32 再做乘法，避免 half/bfloat16 与 float 混合 bmm
                                        // 触发 "expected scalar type Half/BFloat16 but found Float" 的运行时错误；同时保持数值参考实现一致。
                                        auto k_rep_f = k_rep.to(at::kFloat);               // (Hq, D, page_sz)
                                        auto scores = at::bmm(qh, k_rep_f);                // (Hq, Sq, page_sz)
                                        auto scores_used = scores.index({at::indexing::Slice(), at::indexing::Slice(), at::indexing::Slice(0, keys_in_page)});
                                        auto sums = scores_used.abs().sum({-1, -2});       // (Hq)
                                        per_page.index_put_({b, at::indexing::Slice(), p}, sums);
                                }
                        }
                        return {out, lse, per_page};
                }
}

/**
 *  Torch Library Registration
 */
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("varlen_fwd(Tensor! q, Tensor k, Tensor v, Tensor!? out, Tensor cu_seqlens_q, "
            "Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? leftpad_k, Tensor? block_table, Tensor? alibi_slopes, "
            "int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, bool zero_tensors, "
            "bool is_causal, int window_size_left, int window_size_right, float softcap, bool return_softmax, "
            "Generator? gen) -> Tensor[]");
    ops.impl("varlen_fwd", torch::kCUDA, make_pytorch_shim(&mha_varlen_fwd));

    ops.def("fwd_kvcache(Tensor! q, Tensor kcache, Tensor vcache, Tensor? k, Tensor? v, Tensor? seqlens_k, "
            "Tensor? rotary_cos, Tensor? rotary_sin, Tensor? cache_batch_idx, Tensor? leftpad_k, Tensor? block_table, "
            "Tensor? alibi_slopes, Tensor!? out, float softmax_scale, bool is_causal, int window_size_left, "
            "int window_size_right, float softcap, bool is_rotary_interleaved, int num_splits) -> Tensor[]");
    ops.impl("fwd_kvcache", torch::kCUDA, make_pytorch_shim(&mha_fwd_kvcache));

    ops.def("fwd_sparse(Tensor! q, Tensor k, Tensor v, "
            "Tensor block_count, Tensor block_offset, Tensor column_count, Tensor column_index, "
            "Tensor!? out, Tensor? alibi_slopes, "
            "float p_dropout, float softmax_scale, bool is_causal, "
            "float softcap, bool return_softmax, Generator? gen)"
            "-> Tensor[]");
    ops.impl("fwd_sparse", torch::kCUDA, &mha_fwd_sparse);

    ops.def("varlen_fwd_sparse(Tensor! q, Tensor k, Tensor v, "
            "Tensor block_count, Tensor block_offset, Tensor column_count, Tensor column_index, "
            "Tensor!? out, Tensor cu_seqlens_q, "
            "Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? alibi_slopes, "
            "int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, bool zero_tensors, "
            "bool is_causal, float softcap, bool return_softmax, "
            "Generator? gen) -> Tensor[]");
    ops.impl("varlen_fwd_sparse", torch::kCUDA, &mha_varlen_fwd_sparse);

    // varlen fwd with abs_s auxiliary return
    ops.def("varlen_fwd_with_abs(Tensor! q, Tensor k, Tensor v, Tensor!? out, Tensor cu_seqlens_q, "
            "Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? leftpad_k, Tensor? block_table, Tensor? alibi_slopes, "
            "int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, bool zero_tensors, "
            "bool is_causal, int window_size_left, int window_size_right, float softcap, bool return_softmax, "
            "Generator? gen) -> Tensor[]");
    ops.impl("varlen_fwd_with_abs", torch::kCUDA, &varlen_fwd_with_abs_aux);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME);

} // namespace FLASH_NAMESPACE