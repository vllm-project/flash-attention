# Copyright (c) 2023, Tri Dao.
import logging
import sys
import os
import re
import ast
import sysconfig
from collections import namedtuple
from pathlib import Path
from typing import Dict
from shutil import which
from packaging.version import Version, parse

import subprocess

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDA_HOME,
)

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

logger = logging.getLogger(__name__)

# Enivronment variables
Envs = namedtuple("Envs", ["VERBOSE", "MAX_JOBS", "NVCC_THREADS", "VLLM_TARGET_DEVICE", "CMAKE_BUILD_TYPE"])
envs = Envs(
    VERBOSE=bool(int(os.getenv("VERBOSE", "0"))),
    MAX_JOBS=os.getenv("MAX_JOBS"),
    NVCC_THREADS=os.getenv("NVCC_THREADS"),
    VLLM_TARGET_DEVICE=os.getenv("VLLM_TARGET_DEVICE", "cuda"),
    CMAKE_BUILD_TYPE=os.getenv("CMAKE_BUILD_TYPE"),
)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        IS_ROCM = True
    else:
        IS_ROCM = False
else:
    if BUILD_TARGET == "cuda":
        IS_ROCM = False
    elif BUILD_TARGET == "rocm":
        IS_ROCM = True

PACKAGE_NAME = "flash_attn"

BASE_WHEEL_URL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/{tag_name}/{wheel_name}"
)

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FLASH_ATTENTION_FORCE_CXX11_ABI", "FALSE") == "TRUE"
USE_TRITON_ROCM = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
SKIP_CK_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CK_BUILD", "TRUE") == "TRUE" if USE_TRITON_ROCM else False
NVCC_THREADS = os.getenv("NVCC_THREADS") or "4"

@functools.lru_cache(maxsize=None)
def cuda_archs() -> str:
    return os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;110;120").split(";")


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return f'linux_{platform.uname().machine}'
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def add_cuda_gencodes(cc_flag, archs, bare_metal_version):
    """
    Adds -gencode flags based on nvcc capabilities:
      - sm_80/90 (regular)
      - sm_100/120 on CUDA >= 12.8
      - Use 100f on CUDA >= 12.9 (Blackwell family-specific)
      - Map requested 110 -> 101 if CUDA < 13.0 (Thor rename)
      - Embed PTX for newest arch for forward compatibility
    """
    # Always-regular 80
    if "80" in archs:
        cc_flag += ["-gencode", "arch=compute_80,code=sm_80"]

    # Hopper 9.0 needs >= 11.8
    if bare_metal_version >= Version("11.8") and "90" in archs:
        cc_flag += ["-gencode", "arch=compute_90,code=sm_90"]

    # Blackwell 10.x requires >= 12.8
    if bare_metal_version >= Version("12.8"):
        if "100" in archs:
            # CUDA 12.9 introduced "family-specific" for Blackwell (100f)
            if bare_metal_version >= Version("12.9"):
                cc_flag += ["-gencode", "arch=compute_100f,code=sm_100"]
            else:
                cc_flag += ["-gencode", "arch=compute_100,code=sm_100"]

        if "120" in archs:
            # sm_120 is supported in CUDA 12.8/12.9+ toolkits
            if bare_metal_version >= Version("12.9"):
                cc_flag += ["-gencode", "arch=compute_120f,code=sm_120"]
            else:
                cc_flag += ["-gencode", "arch=compute_120,code=sm_120"]


        # Thor rename: 12.9 uses sm_101; 13.0+ uses sm_110
        if "110" in archs:
            if bare_metal_version >= Version("13.0"):
                cc_flag += ["-gencode", "arch=compute_110f,code=sm_110"]
            else:
                # Provide Thor support for CUDA 12.9 via sm_101
                if bare_metal_version >= Version("12.8"):
                    cc_flag += ["-gencode", "arch=compute_101,code=sm_101"]
                # else: no Thor support in older toolkits

    # PTX for newest requested arch (forward-compat)
    numeric = [a for a in archs if a.isdigit()]
    if numeric:
        newest = max(numeric, key=int)
        cc_flag += ["-gencode", f"arch=compute_{newest},code=compute_{newest}"]

    return cc_flag


def get_hip_version():
    return parse(torch.version.hip.split()[-1].rstrip('-').replace('-', '+'))


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def check_if_rocm_home_none(global_option: str) -> None:
    if ROCM_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so hipcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but hipcc was not found."
    )


def detect_hipify_v2():
    try:
        from torch.utils.hipify import __version__
        from packaging.version import Version
        if Version(__version__) >= Version("2.0.0"):
            return True
    except Exception as e:
        print("failed to detect pytorch hipify version, defaulting to version 1.0.0 behavior")
        print(e)
    return False


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", NVCC_THREADS]


def rename_cpp_to_cu(cpp_files):
    for entry in cpp_files:
        shutil.copy(entry, os.path.splitext(entry)[0] + ".cu")


def validate_and_update_archs(archs):
    # List of allowed architectures
    allowed_archs = ["native", "gfx90a", "gfx950", "gfx942"]

    # Validate if each element in archs is in allowed_archs
    assert all(
        arch in allowed_archs for arch in archs
    ), f"One of GPU archs of {archs} is invalid or not supported by Flash-Attention"


cmdclass = {}
ext_modules = []

# We want this even if SKIP_CUDA_BUILD because when we run python setup.py sdist we want the .hpp
# files included in the source distribution, in case the user compiles from source.
if os.path.isdir(".git"):
    if not SKIP_CK_BUILD:
        subprocess.run(["git", "submodule", "update", "--init", "csrc/composable_kernel"], check=True)
        subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"], check=True)
else:
    if IS_ROCM:
        if not SKIP_CK_BUILD:
            assert (
                os.path.exists("csrc/composable_kernel/example/ck_tile/01_fmha/generate.py")
            ), "csrc/composable_kernel is missing, please use source distribution or git clone"
    else:
        assert (
            os.path.exists("csrc/cutlass/include/cutlass/cutlass.h")
        ), "csrc/cutlass is missing, please use source distribution or git clone"


    check_if_cuda_home_none("flash_attn")
    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version < Version("11.7"):
            raise RuntimeError(
                "FlashAttention is only supported on CUDA 11.7 and above.  "
                "Note: make sure nvcc has a supported version by running nvcc -V."
            )
        # Build -gencode (regular + PTX + family-specific 'f' when available)
        add_cuda_gencodes(cc_flag, set(cuda_archs()), bare_metal_version)
    else:
        # No nvcc present; warnings already emitted above
        pass

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    nvcc_flags = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    # "--ptxas-options=-v",
    # "--ptxas-options=-O2",
    # "-lineinfo",
    # "-DFLASHATTENTION_DISABLE_BACKWARD",
    # "-DFLASHATTENTION_DISABLE_DROPOUT",
    # "-DFLASHATTENTION_DISABLE_ALIBI",
    # "-DFLASHATTENTION_DISABLE_SOFTCAP",
    # "-DFLASHATTENTION_DISABLE_UNEVEN_K",
    # "-DFLASHATTENTION_DISABLE_LOCAL",
    ]

    compiler_c17_flag=["-O3", "-std=c++17"]
    # Add Windows-specific flags
    if sys.platform == "win32" and os.getenv('DISTUTILS_USE_SDK') == '1':
        nvcc_flags.extend(["-Xcompiler", "/Zc:__cplusplus"])
        compiler_c17_flag=["-O2", "/std:c++17", "/Zc:__cplusplus"]

    ext_modules.append(
        CUDAExtension(
            name="flash_attn_2_cuda",
            sources=[
                "csrc/flash_attn/flash_api.cpp",
                "csrc/flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim96_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim96_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim192_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim192_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim32_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim32_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim96_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim96_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim192_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim192_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim256_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_causal_sm80.cu",
            ],
            extra_compile_args={
                "cxx": compiler_c17_flag,
                "nvcc": append_nvcc_threads(nvcc_flags + cc_flag),
            },
            include_dirs=[
                Path(this_dir) / "csrc" / "flash_attn",
                Path(this_dir) / "csrc" / "flash_attn" / "src",
                Path(this_dir) / "csrc" / "cutlass" / "include",
            ],
        )
    )
elif not SKIP_CUDA_BUILD and IS_ROCM:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    # Skips CK C++ extension compilation if using Triton Backend
    if not SKIP_CK_BUILD:
        ck_dir = "csrc/composable_kernel"

def is_ninja_available() -> bool:
    return which("ninja") is not None

        optdim = os.getenv("OPT_DIM", "32,64,128,256")
        subprocess.run([sys.executable, f"{ck_dir}/example/ck_tile/01_fmha/generate.py", "-d", "fwd", "--output_dir", "build", "--receipt", "2", "--optdim", optdim], check=True)
        subprocess.run([sys.executable, f"{ck_dir}/example/ck_tile/01_fmha/generate.py", "-d", "fwd_appendkv", "--output_dir", "build", "--receipt", "2", "--optdim", optdim], check=True)
        subprocess.run([sys.executable, f"{ck_dir}/example/ck_tile/01_fmha/generate.py", "-d", "fwd_splitkv", "--output_dir", "build", "--receipt", "2", "--optdim", optdim], check=True)
        subprocess.run([sys.executable, f"{ck_dir}/example/ck_tile/01_fmha/generate.py", "-d", "bwd", "--output_dir", "build", "--receipt", "2", "--optdim", optdim], check=True)

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


        if archs != ['native']:
            cc_flag = [f"--offload-arch={arch}" for arch in archs]
        else:
            arch = torch.cuda.get_device_properties("cuda").gcnArchName.split(":")[0]
            cc_flag = [f"--offload-arch={arch}"]


def _is_cuda() -> bool:
    has_cuda = torch.version.cuda is not None
    return VLLM_TARGET_DEVICE == "cuda" and has_cuda

        # Check if torch is using hipify v2. Until CK is updated with HIPIFY_V2 macro,
        # we must replace the incorrect APIs.
        maybe_hipify_v2_flag = []
        if detect_hipify_v2():
            maybe_hipify_v2_flag = ["-DHIPIFY_V2"]

        rename_cpp_to_cu(sources)

def _is_hip() -> bool:
    return (VLLM_TARGET_DEVICE == "cuda"
            or VLLM_TARGET_DEVICE == "rocm") and torch.version.hip is not None

        cc_flag += ["-O3","-std=c++20",
                    "-DCK_TILE_FMHA_FWD_FAST_EXP2=1",
                    "-fgpu-flush-denormals-to-zero",
                    "-DCK_ENABLE_BF16",
                    "-DCK_ENABLE_BF8",
                    "-DCK_ENABLE_FP16",
                    "-DCK_ENABLE_FP32",
                    "-DCK_ENABLE_FP64",
                    "-DCK_ENABLE_FP8",
                    "-DCK_ENABLE_INT8",
                    "-DCK_USE_XDL",
                    "-DUSE_PROF_API=1",
                    # "-DFLASHATTENTION_DISABLE_BACKWARD",
                    "-D__HIP_PLATFORM_HCC__=1"]

def is_freethreaded():
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))

        # Imitate https://github.com/ROCm/composable_kernel/blob/c8b6b64240e840a7decf76dfaa13c37da5294c4a/CMakeLists.txt#L190-L214
        hip_version = get_hip_version()
        if hip_version > Version('5.5.00000'):
            cc_flag += ["-mllvm", "--lsr-drop-solution=1"]
        if hip_version > Version('5.7.23302'):
            cc_flag += ["-fno-offload-uniform-block"]
        if hip_version > Version('6.1.40090'):
            cc_flag += ["-mllvm", "-enable-post-misched=0"]
        if hip_version > Version('6.2.41132'):
            cc_flag += ["-mllvm", "-amdgpu-early-inline-all=true",
                        "-mllvm", "-amdgpu-function-calls=false"]
        if hip_version > Version('6.2.41133') and hip_version < Version('6.3.00000'):
            cc_flag += ["-mllvm", "-amdgpu-coerce-illegal-types=1"]

        extra_compile_args = {
            "cxx": ["-O3", "-std=c++20"] + generator_flag + maybe_hipify_v2_flag,
            "nvcc": cc_flag + generator_flag + maybe_hipify_v2_flag,
        }

    def __init__(self, name: str, cmake_lists_dir: str = '.', **kwa) -> None:
        super().__init__(name, sources=[], py_limited_api=not is_freethreaded(), **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    # A dict of extension directories that have been configured.
    did_config: Dict[str, bool] = {}

    #
    # Determine number of compilation jobs and optionally nvcc compile threads.
    #
    def compute_num_jobs(self):
        # `num_jobs` is either the value of the MAX_JOBS environment variable
        # (if defined) or the number of CPUs available.
        num_jobs = envs.MAX_JOBS
        if num_jobs is not None:
            num_jobs = int(num_jobs)
            logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
        else:
            try:
                # os.sched_getaffinity() isn't universally available, so fall
                #  back to os.cpu_count() if we get an error here.
                num_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                num_jobs = os.cpu_count()

        nvcc_threads = None
        if _is_cuda() and get_nvcc_cuda_version() >= Version("11.2"):
            # `nvcc_threads` is either the value of the NVCC_THREADS
            # environment variable (if defined) or 1.
            # when it is set, we reduce `num_jobs` to avoid
            # overloading the system.
            nvcc_threads = envs.NVCC_THREADS
            if nvcc_threads is not None:
                nvcc_threads = int(nvcc_threads)
                logger.info(
                    "Using NVCC_THREADS=%d as the number of nvcc threads.",
                    nvcc_threads)
            else:
                nvcc_threads = 1
            num_jobs = max(1, num_jobs // nvcc_threads)

        return num_jobs, nvcc_threads

    #
    # Perform cmake configuration for a single extension.
    #
    def configure(self, ext: CMakeExtension) -> None:
        # If we've already configured using the CMakeLists.txt for
        # this extension, exit early.
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        # Select the build type.
        # Note: optimization level + debug info are set by the build type
        default_cfg = "Debug" if self.debug else "RelWithDebInfo"
        cfg = envs.CMAKE_BUILD_TYPE or default_cfg

        cmake_args = [
            '-DCMAKE_BUILD_TYPE={}'.format(cfg),
            '-DVLLM_TARGET_DEVICE={}'.format(VLLM_TARGET_DEVICE),
        ]

        verbose = envs.VERBOSE
        if verbose:
            cmake_args += ['-DCMAKE_VERBOSE_MAKEFILE=ON']

        if is_sccache_available():
            cmake_args += [
                '-DCMAKE_CXX_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_C_COMPILER_LAUNCHER=sccache',
            ]
        elif is_ccache_available():
            cmake_args += [
                '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
                '-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache',
            ]

        # Pass the python executable to cmake so it can find an exact
        # match.
        cmake_args += ['-DPython_EXECUTABLE={}'.format(sys.executable)]

        # Pass the python path to cmake so it can reuse the build dependencies
        # on subsequent calls to python.
        cmake_args += ['-DVLLM_PYTHON_PATH={}'.format(":".join(sys.path))]

        #
        # Setup parallelism and build tool
        #
        num_jobs, nvcc_threads = self.compute_num_jobs()

        if nvcc_threads:
            cmake_args += ['-DNVCC_THREADS={}'.format(nvcc_threads)]

        if is_ninja_available():
            build_tool = ['-G', 'Ninja']
            cmake_args += [
                '-DCMAKE_JOB_POOL_COMPILE:STRING=compile',
                '-DCMAKE_JOB_POOLS:STRING=compile={}'.format(num_jobs),
            ]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []
        subprocess.check_call(
            ['cmake', ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp)

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError as e:
            raise RuntimeError('Cannot find CMake executable') from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        targets = []
        target_name = lambda s: remove_prefix(s, "vllm_flash_attn.")
        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)
            targets.append(target_name(ext.name))

        num_jobs, _ = self.compute_num_jobs()

        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]

        subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)

        # Install the libraries
        for ext in self.extensions:
            # Install the extension into the proper location
            outdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

            # Skip if the install directory is the same as the build directory
            if outdir == self.build_temp:
                continue

            # CMake appends the extension prefix to the install path,
            # and outdir already contains that prefix, so we need to remove it.
            prefix = outdir
            for i in range(ext.name.count('.')):
                prefix = prefix.parent

            # prefix here should actually be the same for all components
            install_args = [
                "cmake", "--install", ".", "--prefix", prefix, "--component",
                target_name(ext.name)
            ]
            subprocess.check_call(install_args, cwd=self.build_temp)


def get_package_version():
    with open(Path(this_dir) / PACKAGE_NAME / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FLASH_ATTN_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


PYTORCH_VERSION = "2.4.0"
MAIN_CUDA_VERSION = "12.1"


def get_nvcc_cuda_version() -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    assert CUDA_HOME is not None, "CUDA_HOME is not set"
    nvcc_output = subprocess.check_output([CUDA_HOME + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_version() -> str:
    version = get_package_version()
    cuda_version = str(get_nvcc_cuda_version())
    if cuda_version != MAIN_CUDA_VERSION:
        cuda_version_str = cuda_version.replace(".", "")[:3]
        version += f"+cu{cuda_version_str}"
    return version

            nvcc_threads = max(1, int(NVCC_THREADS))

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # free memory in GB
            # Assume worst-case peak observed memory usage of ~5GB per NVCC thread.
            # Limit: peak_threads = max_jobs * nvcc_threads and peak_threads * 5GB <= free_memory.
            max_num_jobs_memory = max(1, int(free_memory_gb / (5 * nvcc_threads)))

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            print(
                f"Auto set MAX_JOBS to `{max_jobs}`, NVCC_THREADS to `{nvcc_threads}`. "
                "If you see memory pressure, please use a lower `MAX_JOBS=N` or `NVCC_THREADS=N` value."
            )
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)

ext_modules.append(CMakeExtension(name="vllm_flash_attn._vllm_fa2_C"))
ext_modules.append(CMakeExtension(name="vllm_flash_attn._vllm_fa3_C"))

setup(
    name="vllm-flash-attn",
    version=get_version(),
    packages=find_packages(exclude=("build",
                                    "csrc",
                                    "include",
                                    "tests",
                                    "dist",
                                    "docs",
                                    "benchmarks",
                                    f"{PACKAGE_NAME}.egg-info",)),
    author="vLLM Team",
    description="Forward-only flash-attn",
    long_description=f"Forward-only flash-attn package built for PyTorch {PYTORCH_VERSION} and CUDA {MAIN_CUDA_VERSION}",
    url="https://github.com/vllm-project/flash-attention.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": cmake_build_ext} if len(ext_modules) > 0 else {},
    python_requires=">=3.8",
    install_requires=[f"torch == {PYTORCH_VERSION}"],
    setup_requires=["psutil"],
)
