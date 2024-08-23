cmake_minimum_required(VERSION 3.26)

project(vllm_flash_attn LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA by default, can be overridden by using -DVLLM_TARGET_DEVICE=... (used by setup.py)
set(VLLM_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for vLLM")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${VLLM_TARGET_DEVICE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${VLLM_PYTHON_PATH}")

# Supported python versions. These should be kept in sync with setup.py.
set(PYTHON_SUPPORTED_VERSIONS "3.8" "3.9" "3.10" "3.11" "3.12")

# Supported NVIDIA architectures. TODO(luka) this might be incorrect
set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0")

# Supported AMD GPU architectures.
set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100")

#
# Supported/expected torch versions for CUDA/ROCm.
#
# Currently, having an incorrect pytorch version results in a warning
# rather than an error.
#
# Note: the CUDA torch version is derived from pyproject.toml and various
# requirements.txt files and should be kept consistent.  The ROCm torch
# versions are derived from Dockerfile.rocm
#
set(TORCH_SUPPORTED_VERSION_CUDA "2.4.0")
set(TORCH_SUPPORTED_VERSION_ROCM "2.5.0")

find_python_constrained_versions(${PYTHON_SUPPORTED_VERSIONS})

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")

# Ensure the 'nvcc' command is in the PATH
find_program(NVCC_EXECUTABLE nvcc)
if (CUDA_FOUND AND NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found")
endif ()

#
# Import torch cmake configuration.
# Torch also imports CUDA (and partially HIP) languages with some customizations,
# so there is no need to do this explicitly with check_language/enable_language,
# etc.
#
find_package(Torch REQUIRED)

#
# Set up GPU language and check the torch version and warn if it isn't
# what is expected.
#
if (NOT HIP_FOUND AND CUDA_FOUND)
    set(VLLM_GPU_LANG "CUDA")

    # Check CUDA is at least 11.6
    if (CUDA_VERSION VERSION_LESS 11.6)
        message(FATAL_ERROR "CUDA version 11.6 or greater is required.")
    endif ()

    if (NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_CUDA})
        message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_CUDA} "
                "expected for CUDA build, saw ${Torch_VERSION} instead.")
    endif ()
elseif (HIP_FOUND)
    set(VLLM_GPU_LANG "HIP")

    # Importing torch recognizes and sets up some HIP/ROCm configuration but does
    # not let cmake recognize .hip files. In order to get cmake to understand the
    # .hip extension automatically, HIP must be enabled explicitly.
    enable_language(HIP)

    # ROCm 5.X and 6.X
    if (ROCM_VERSION_DEV_MAJOR GREATER_EQUAL 5 AND
            NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_ROCM})
        message(WARNING "Pytorch version >= ${TORCH_SUPPORTED_VERSION_ROCM} "
                "expected for ROCm build, saw ${Torch_VERSION} instead.")
    endif ()
else ()
    message(FATAL_ERROR "Can't find CUDA or HIP installation.")
endif ()

#
# Override the GPU architectures detected by cmake/torch and filter them by
# the supported versions for the current language.
# The final set of arches is stored in `VLLM_GPU_ARCHES`.
#
override_gpu_arches(VLLM_GPU_ARCHES
        ${VLLM_GPU_LANG}
        "${${VLLM_GPU_LANG}_SUPPORTED_ARCHS}")

#
# Query torch for additional GPU compilation flags for the given
# `VLLM_GPU_LANG`.
# The final set of arches is stored in `VLLM_GPU_FLAGS`.
#
get_torch_gpu_compiler_flags(VLLM_GPU_FLAGS ${VLLM_GPU_LANG})

#
# Set nvcc parallelism.
#
if (NVCC_THREADS AND VLLM_GPU_LANG STREQUAL "CUDA")
    list(APPEND VLLM_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif ()

# Other flags
list(APPEND VLLM_GPU_FLAGS --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math)

#
# _C extension
#

# TODO(luka): happy to hardcode, but these are generated so globbing seems fine.
file(GLOB FLASH_ATTN_GEN_SRCS "csrc/flash_attn/src/flash_fwd_*.cu")
message(DEBUG "FLASH_ATTN_GEN_SRCS: ${FLASH_ATTN_GEN_SRCS}")

define_gpu_extension_target(
        vllm_flash_attn_c
        DESTINATION vllm_flash_attn
        LANGUAGE ${VLLM_GPU_LANG}
        SOURCES csrc/flash_attn/flash_api.cpp ${FLASH_ATTN_GEN_SRCS}
        COMPILE_FLAGS ${VLLM_GPU_FLAGS}
        ARCHITECTURES ${VLLM_GPU_ARCHES}
        USE_SABI 3
        WITH_SOABI
)

target_include_directories(vllm_flash_attn_c PRIVATE
        csrc/flash_attn
        csrc/flash_attn/src
        csrc/cutlass/include)

# custom definitions
target_compile_definitions(vllm_flash_attn_c PRIVATE FLASHATTENTION_DISABLE_DROPOUT FLASHATTENTION_DISABLE_UNEVEN_K)

# Make sure RelWithDebInfo uses -03 and not -O2
target_compile_options(vllm_flash_attn_c PRIVATE $<$<CONFIG:RELWITHDEBINFO>:-O3>)

# Check for old generator
find_file(OLD_GENERATOR_FILE "ATen/CUDAGeneratorImpl.h" ${TORCH_INCLUDE_DIRS} NO_DEFAULT_PATH)
if (OLD_GENERATOR_FILE)
    target_compile_definitions(vllm_flash_attn_c PRIVATE -DOLD_GENERATOR_PATH)
endif ()
