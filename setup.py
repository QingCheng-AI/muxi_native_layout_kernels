from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
                "cxx": ["-std=c++17"],
                "nvcc": [  # Yes, it's "nvcc" for "mxcc"
                    "-x",
                    "maca",
                    "-std=c++17",
                    "-mllvm",
                    "-metaxgpu-disable-bsm-offset=0",
                    "-mllvm",
                    "-metaxgpu-force-global-saddr=1",
                    # "--res-usage",  # Please set `--verbose` to `pip install .` to see the message from `--res-usage`.
                ],
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
