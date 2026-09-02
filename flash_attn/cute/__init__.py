"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .interface import (
    compile_sm120_paged_decode_d256_plan,
    compile_sm120_paged_decode_d256_from_specs,
    compile_sm120_paged_decode_d256_warmup_plan,
    flash_attn_func,
    flash_attn_varlen_func,
    try_launch_sm120_paged_decode_d256,
)
from .sm120_paged_decode import (
    Sm120PagedDecodeCompileUnit,
    Sm120PagedDecodeD256Metadata,
    build_sm120_paged_decode_d256_compile_plan,
    build_sm120_paged_decode_d256_warmup_plan,
    select_sm120_paged_decode_d256_num_splits,
    sm120_paged_decode_d256_available,
    sm120_paged_decode_d256_enabled,
)

__all__ = [
    "Sm120PagedDecodeCompileUnit",
    "Sm120PagedDecodeD256Metadata",
    "build_sm120_paged_decode_d256_compile_plan",
    "build_sm120_paged_decode_d256_warmup_plan",
    "compile_sm120_paged_decode_d256_plan",
    "compile_sm120_paged_decode_d256_from_specs",
    "compile_sm120_paged_decode_d256_warmup_plan",
    "flash_attn_func",
    "flash_attn_varlen_func",
    "select_sm120_paged_decode_d256_num_splits",
    "sm120_paged_decode_d256_available",
    "sm120_paged_decode_d256_enabled",
    "try_launch_sm120_paged_decode_d256",
]
