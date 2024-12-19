// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_sparse_<cutlass::bfloat16_t, 160, false>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_sparse_hdim160<cutlass::bfloat16_t, false>(params, stream);
}
