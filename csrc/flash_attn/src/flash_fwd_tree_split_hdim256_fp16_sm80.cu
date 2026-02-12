// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_tree_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_dispatch_tree<cutlass::half_t, 256>(Flash_fwd_params_tree &params, cudaStream_t stream);

} // namespace FLASH_NAMESPACE