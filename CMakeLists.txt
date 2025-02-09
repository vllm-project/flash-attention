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

# Supported NVIDIA architectures.
set(CUDA_SUPPORTED_ARCHS "8.0;8.6;8.9;9.0;10.0;10.1;12.0")

# Supported AMD GPU architectures.
set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100")

#
# Supported/expected torch versions for CUDA/ROCm.
#
# Currently, having an incorrect pytorch version results in a warning
# rather than an error.
#
# Note: these should be kept in sync with the torch version in setup.py.
# Likely should also be in sync with the vLLM version.
#
set(TORCH_SUPPORTED_VERSION_CUDA "2.4.0")

find_python_constrained_versions(${PYTHON_SUPPORTED_VERSIONS})

if (VLLM_PARENT_BUILD)
    # vLLM extracts the supported architectures from the global CMAKE_CUDA_FLAGS, which are set by torch.
    # Because CMAKE_CUDA_FLAGS has been modified, we cannot use the same logic.
    # Hence, we just use the parent's VLLM_GPU_ARCHES and VLLM_GPU_FLAGS.
    message(STATUS "Building vllm-flash-attn inside vLLM. Skipping flag detection and relying on parent build.")
    macro(check_found NAME VAR)
        if (NOT ${VAR})
            message(FATAL_ERROR "${NAME} must have been found by parent.")
        endif ()
    endmacro()

    check_found("Torch" TORCH_FOUND)

    set(VLLM_FA_GPU_FLAGS ${VLLM_GPU_FLAGS})
    set(VLLM_FA_GPU_ARCHES ${VLLM_GPU_ARCHES})

    # Allow direct override of GPU architectures.
    # These have to be in CMake syntax (75-real, 89-virtual, etc).
    if (DEFINED ENV{VLLM_FA_CMAKE_GPU_ARCHES})
        message(STATUS "Overriding GPU architectures to $ENV{VLLM_FA_CMAKE_GPU_ARCHES}")
        set(VLLM_FA_GPU_ARCHES $ENV{VLLM_FA_CMAKE_GPU_ARCHES})

        # Generally, we want to build with a subset of the parent arches.
        foreach (VLLM_FA_GPU_ARCH IN LISTS VLLM_FA_GPU_ARCHES)
            if (NOT VLLM_FA_GPU_ARCH IN_LIST VLLM_GPU_ARCHES)
                message(WARNING "Using GPU architecture ${VLLM_FA_GPU_ARCH}, "
                        "which is not included in the parent list.")
            endif ()
        endforeach ()
    endif ()

else ()
    message(STATUS "Standalone vllm-flash-attn build.")

    #
    # Update cmake's `CMAKE_PREFIX_PATH` with torch location.
    #
    append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
    message(DEBUG "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

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
        message(FATAL_ERROR "ROCm build is not currently supported for vllm-flash-attn.")

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
    override_gpu_arches(VLLM_FA_GPU_ARCHES
            ${VLLM_GPU_LANG}
            "${${VLLM_GPU_LANG}_SUPPORTED_ARCHS}")

    #
    # Query torch for additional GPU compilation flags for the given
    # `VLLM_GPU_LANG`.
    # The final set of arches is stored in `VLLM_FA_GPU_FLAGS`.
    #
    get_torch_gpu_compiler_flags(VLLM_FA_GPU_FLAGS ${VLLM_GPU_LANG})

    #
    # Set nvcc parallelism.
    #
    if (NVCC_THREADS AND VLLM_GPU_LANG STREQUAL "CUDA")
        list(APPEND VLLM_FA_GPU_FLAGS "--threads=${NVCC_THREADS}")
    endif ()
endif ()

# Other flags
list(APPEND VLLM_FA_GPU_FLAGS --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math)

# Replace instead of appending, nvcc doesn't like duplicate -O flags.
string(REPLACE "-O2" "-O3" CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")

#
# _C extension
#

file(GLOB FLASH_ATTN_GEN_SRCS "csrc/flash_attn/src/flash_fwd_*.cu")
message(DEBUG "FLASH_ATTN_GEN_SRCS: ${FLASH_ATTN_GEN_SRCS}")

define_gpu_extension_target(
        vllm_flash_attn_c
        DESTINATION vllm_flash_attn
        LANGUAGE ${VLLM_GPU_LANG}
        SOURCES csrc/flash_attn/flash_api.cpp ${FLASH_ATTN_GEN_SRCS}
        COMPILE_FLAGS ${VLLM_FA_GPU_FLAGS}
        ARCHITECTURES ${VLLM_FA_GPU_ARCHES}
        USE_SABI 3
        WITH_SOABI
)

target_include_directories(vllm_flash_attn_c PRIVATE
        csrc/flash_attn
        csrc/flash_attn/src
        csrc/cutlass/include)

# custom definitions
target_compile_definitions(vllm_flash_attn_c PRIVATE
        # FLASHATTENTION_DISABLE_BACKWARD
        FLASHATTENTION_DISABLE_DROPOUT
        # FLASHATTENTION_DISABLE_ALIBI
        # FLASHATTENTION_DISABLE_SOFTCAP
        FLASHATTENTION_DISABLE_UNEVEN_K
        # FLASHATTENTION_DISABLE_LOCAL
)

# Check for old generator
find_file(OLD_GENERATOR_FILE "ATen/CUDAGeneratorImpl.h" ${TORCH_INCLUDE_DIRS} NO_DEFAULT_PATH)
if (OLD_GENERATOR_FILE)
    target_compile_definitions(vllm_flash_attn_c PRIVATE -DOLD_GENERATOR_PATH)
endif ()
