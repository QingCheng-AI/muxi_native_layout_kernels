import shutil
import sys
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_version_from_gplusplus_executable(gplusplus_executable):
    return int(os.popen(gplusplus_executable + " -dumpversion").read().strip())


def find_gplusplus_version(version_high_bound):
    """
    Find the command of g++ version <= version_high_bound

    The reason of this version bound is from the fact that mxcc is based on clang++, while
    clang++ uses libstdc++ from g++, and clang++ from mxcc is too old for some new versions
    of libstdc++.
    """

    candidates = ["g++"]
    for v in range(version_high_bound, 1, -1):
        candidates.append(f"g++-{v}")
    for candidate in candidates:
        if shutil.which(candidate):
            if (
                version := get_version_from_gplusplus_executable(candidate)
            ) <= version_high_bound:
                return version
    return None


host_compile_flags = ["-std=c++20"]
device_compile_flags = [
    "-x",
    "maca",
    "-std=c++20",
    "-mllvm",
    "-metaxgpu-disable-bsm-offset=0",
    "-mllvm",
    "-metaxgpu-force-global-saddr=1",
    # "--res-usage",  # Please set `--verbose` to `pip install .` to see the message from `--res-usage`.
]
if (gcc_version := find_gplusplus_version(version_high_bound=11)) is not None:
    device_compile_flags += [f"--gcc-version={gcc_version}"]
else:
    print(
        f"WARNING: Failed to find g++ <= {version_high_bound}, which may not be compatible with mxcc.",
        file=sys.stderr,
    )

setup(
    name="muxi_layout_kernels",
    ext_modules=[
        CUDAExtension(
            name="muxi_layout_kernels_ext",
            sources=[
                "./csrc/gemm_kernel.cu",
                "./csrc/arg_selector.cu",
                "./csrc/layout_b.cu",
                "./csrc/re_layout_c.cu",
                "./csrc/gemm_layoutA_linear.cu",
                "./csrc/gemm_layout_A.cu",
                "./csrc/gemm_layout_A_soft_fp8.cu",
                "./csrc/gemm_layout_abc.cu",
                "./csrc/gemm_layout_ab_continuous_c.cu",
                "./csrc/gemm_ex.cu",
                "./csrc/muxi_hgemm.cu",
                "./csrc/muxi_hgemm_layout.cu",
                "./csrc/muxi_hgemm_layoutA.cu",
                "./csrc/muxi_hgemm_layoutC.cu",
                "./csrc/gemv_layout_a.cu",
                "./csrc/fp8_weight_repack.cu",
            ],
            extra_compile_args={
                "cxx": host_compile_flags,
                "nvcc": device_compile_flags,  # Yes, it's "nvcc" for "mxcc"
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=["muxi_layout_kernels"],
    package_data={
        "muxi_layout_kernels": [
            "layout_gemm_dispatch_arg.csv",
            "layout_gemv_dispatch_arg.csv",
            "continuous_gemm_dispatch_arg.csv",
            "layoutA_gemm_dispatch_arg.csv",
            "layoutA_Soft_Fp8_gemm_dispatch_arg.csv",
            "layout_gemv_soft_fp8_dispatch_arg.csv",
            "layout_gemv_bf16_dispatch_arg.csv",
        ]
    },
)
