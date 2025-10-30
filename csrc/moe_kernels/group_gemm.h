#pragma once

#include <mc_runtime.h>

#include <chrono>
#include <iostream>

#include "copyAndLayoutB.h"
#include "group_gemm_kernel.h"
#include "group_gemm_utils.h"
#include "silu_and_mul_kernel.h"

namespace muxi_layout_kernels {

template <typename Tab, typename Taccum, typename Tc, int BLOCK_DIM_X,
          int APerWarp, int splitN, int splitK, int N>
__global__ void __launch_bounds__(BLOCK_DIM_X)
    group_gemm_tn(Tab *A[], Tab *B[], Tc *C[], int m, int n, int k,
                  Taccum alpha, Taccum beta, int gemm_count,
                  int blockCountSum[], Tc *bias[]) {
    int gemm_id = __builtin_mxc_readfirstlane(-1);
    int gemm_warpId = __builtin_mxc_readfirstlane(-1);
    int global_warpId = __builtin_mxc_readfirstlane(
        (blockDim.x * blockIdx.x + threadIdx.x) / WARPSIZE);
    for (int i = 1; i < gemm_count + 1; i++) {
        if (global_warpId < blockCountSum[i] * (BLOCK_DIM_X / WARPSIZE)) {
            gemm_id = i - 1;
            gemm_warpId = __builtin_mxc_readfirstlane(
                global_warpId -
                blockCountSum[i - 1] * (BLOCK_DIM_X / WARPSIZE));
            break;
        }
    }

    Tc *bias_ = (bias == nullptr ? nullptr : bias[gemm_id]);
    GemmMmaLayoutABCReuseAKernelDispatch<Tab, Taccum, Tc, BLOCK_DIM_X, APerWarp,
                                         splitN, splitK, N>(
        A[gemm_id], B[gemm_id], C[gemm_id], m, n, k, alpha, beta, bias_,
        gemm_warpId);
}

template <typename Tab, typename Taccum, typename Tc>
void group_gemm_launcher(Tab *A[], Tab *B[], Tc *C[], int m, int n, int k,
                         Taccum alpha, Taccum beta, int gemm_count,
                         Tc *bias[] = nullptr) {
    constexpr int block_dim_x = 256;
    int *blockCountSum = (int *)malloc(sizeof(int) * (gemm_count + 1));
    blockCountSum[0] = 0;

    int APerWarp = 1, splitN = 1, splitK = 1;
    if (n <= 16)
        APerWarp = 1, splitN = 1, splitK = 1;
    else if (n <= 32)
        APerWarp = 1, splitN = 1, splitK = 1;
    else if (n <= 48)
        APerWarp = 1, splitN = 1, splitK = 1;
    else if (n <= 64)
        APerWarp = 1, splitN = 1, splitK = 1;
    else if (n <= 80)
        APerWarp = 1, splitN = 1, splitK = 1;
    else if (n <= 96)
        APerWarp = 1, splitN = 1, splitK = 1;
    else if (n <= 112)
        APerWarp = 1, splitN = 1, splitK = 1;
    else if (n <= 128)
        APerWarp = 1, splitN = 1, splitK = 1;
    else if (n <= 144)
        APerWarp = 1, splitN = 2, splitK = 2;
    else if (n <= 160)
        APerWarp = 2, splitN = 4, splitK = 2;
    else if (n <= 176)
        APerWarp = 2, splitN = 1, splitK = 2;
    else if (n <= 192)
        APerWarp = 2, splitN = 1, splitK = 2;
    else if (n <= 208)
        APerWarp = 2, splitN = 1, splitK = 2;
    else if (n <= 224)
        APerWarp = 2, splitN = 1, splitK = 2;
    else if (n <= 240)
        APerWarp = 2, splitN = 1, splitK = 2;
    else if (n <= 256)
        APerWarp = 1, splitN = 1, splitK = 4;

    // dispatch the warps to gemm_id
    for (int i = 1; i < gemm_count + 1; i++) {
        blockCountSum[i] =
            m / 16 / (block_dim_x / WARPSIZE * APerWarp) * splitN * splitK;
    }
    for (int i = 1; i < gemm_count + 1; i++) {
        blockCountSum[i] += blockCountSum[i - 1];
    }

    Tc **dev_bias = nullptr;
    if (bias != nullptr) {
        mcMalloc((void ***)&dev_bias, sizeof(Tc *) * gemm_count);
        mcMemcpy(dev_bias, bias, sizeof(Tc *) * gemm_count,
                 mcMemcpyHostToDevice);
    }
    int *dev_blockCountSum;
    mcMalloc((void **)&dev_blockCountSum, sizeof(int) * (gemm_count + 1));
    mcMemcpy(dev_blockCountSum, blockCountSum, sizeof(int) * (gemm_count + 1),
             mcMemcpyHostToDevice);

    // Launch kernel

#define DISPATCH_GEMM_OPT_KERNEL(APERWARP, SPLITN, SPLITK, N)                  \
    group_gemm_tn<Tab, Taccum, Tc, block_dim_x, APERWARP, SPLITN, SPLITK, N>   \
        <<<blockCountSum[gemm_count], block_dim_x, 0, stream>>>(               \
            A, B, C, m, n, k, alpha, beta, gemm_count, dev_blockCountSum,      \
            dev_bias);

    const mcStream_t stream =
        at::cuda::getCurrentCUDAStream(at::cuda::current_device());

    if (n <= 16) {
        DISPATCH_GEMM_OPT_KERNEL(1, 1, 1, 16);
    } else if (n <= 32) {
        DISPATCH_GEMM_OPT_KERNEL(1, 1, 1, 32);
    } else if (n <= 48) {
        DISPATCH_GEMM_OPT_KERNEL(1, 1, 1, 48);
    } else if (n <= 64) {
        DISPATCH_GEMM_OPT_KERNEL(1, 1, 1, 64);
    } else if (n <= 80) {
        DISPATCH_GEMM_OPT_KERNEL(1, 1, 1, 80);
    } else if (n <= 96) {
        DISPATCH_GEMM_OPT_KERNEL(1, 1, 1, 96);
    } else if (n <= 112) {
        DISPATCH_GEMM_OPT_KERNEL(1, 1, 1, 112);
    } else if (n <= 128) {
        DISPATCH_GEMM_OPT_KERNEL(1, 1, 1, 128);
    } else if (n <= 144) {
        DISPATCH_GEMM_OPT_KERNEL(1, 2, 2, 144);
    } else if (n <= 160) {
        DISPATCH_GEMM_OPT_KERNEL(2, 4, 2, 160);
    } else if (n <= 176) {
        DISPATCH_GEMM_OPT_KERNEL(2, 1, 2, 176);
    } else if (n <= 192) {
        DISPATCH_GEMM_OPT_KERNEL(2, 1, 2, 192);
    } else if (n <= 208) {
        DISPATCH_GEMM_OPT_KERNEL(2, 1, 2, 208);
    } else if (n <= 224) {
        DISPATCH_GEMM_OPT_KERNEL(2, 1, 2, 224);
    } else if (n <= 240) {
        DISPATCH_GEMM_OPT_KERNEL(2, 1, 2, 240);
    } else if (n <= 256) {
        DISPATCH_GEMM_OPT_KERNEL(1, 1, 4, 256);
    }

#undef DISPATCH_GEMM_OPT_KERNEL

    mcDeviceSynchronize();

    mcFree(dev_blockCountSum);
    free(blockCountSum);
    if (dev_bias != nullptr) {
        mcFree(dev_bias);
    }
}

template <typename Tab, typename Taccum, typename Tc, int BLOCK_DIM_X,
          int APerWarp, int splitN, int splitK, int N>
__global__ void __launch_bounds__(BLOCK_DIM_X)
    group_gemm_tn2(Tab *A[], Tab *B[], Tc *C, int m, int n, int k, Tc *alpha,
                   Taccum beta, int gemm_count, int blockCountSum[],
                   Tc *bias[]) {
    int gemm_id = __builtin_mxc_readfirstlane(-1);
    int gemm_warpId = __builtin_mxc_readfirstlane(-1);
    int global_warpId = __builtin_mxc_readfirstlane(
        (blockDim.x * blockIdx.x + threadIdx.x) / WARPSIZE);
    for (int i = 1; i < gemm_count + 1; i++) {
        if (global_warpId < blockCountSum[i] * (BLOCK_DIM_X / WARPSIZE)) {
            gemm_id = i - 1;
            gemm_warpId = __builtin_mxc_readfirstlane(
                global_warpId -
                blockCountSum[i - 1] * (BLOCK_DIM_X / WARPSIZE));
            break;
        }
    }

    Tc *bias_ = (bias == nullptr ? nullptr : bias[gemm_id]);

#define DISPATCH_GEMM_OPT_KERNEL(APERWARP, SPLITN, SPLITK, N)                  \
    GemmMmaLayoutAB_ContinuousCReuseA_and_mul_weights_KernelDispatch<          \
        Tab, Taccum, Tc, BLOCK_DIM_X, APERWARP, SPLITN, SPLITK, N>(            \
        A[gemm_id], B[gemm_id], C, m, n, k, alpha + n * gemm_id, beta, bias_,  \
        gemm_warpId)

    // launch the gemm device dispatcher
    DISPATCH_GEMM_OPT_KERNEL(APerWarp, splitN, splitK, N);

#undef DISPATCH_GEMM_OPT_KERNEL
}

template <typename Tab, typename Taccum, typename Tc>
void group_gemm_launcher2(Tab *A[], Tab *B[], Tc *C, int m, int n, int k,
                          Tc *dev_alpha, Taccum beta, int gemm_count,
                          Tc *bias[] = nullptr) {
    constexpr int block_dim_x = 256;
    int *blockCountSum = (int *)malloc(sizeof(int) * (gemm_count + 1));
    blockCountSum[0] = 0;

    int APerWarp = 1, splitN = 1, splitK = 1;
    if (n <= 16)
        APerWarp = 2, splitN = 1, splitK = 1;
    else if (n <= 32)
        APerWarp = 2, splitN = 1, splitK = 1;
    else if (n <= 48)
        APerWarp = 2, splitN = 1, splitK = 1;
    else if (n <= 64)
        APerWarp = 2, splitN = 1, splitK = 1;
    else if (n <= 80)
        APerWarp = 2, splitN = 1, splitK = 1;
    else if (n <= 96)
        APerWarp = 2, splitN = 1, splitK = 2;
    else if (n <= 112)
        APerWarp = 2, splitN = 1, splitK = 2;
    else if (n <= 128)
        APerWarp = 2, splitN = 1, splitK = 2;
    else if (n <= 144)
        APerWarp = 2, splitN = 1, splitK = 3;
    else if (n <= 160)
        APerWarp = 2, splitN = 1, splitK = 2;
    else if (n <= 176)
        APerWarp = 2, splitN = 1, splitK = 3;
    else if (n <= 192)
        APerWarp = 2, splitN = 1, splitK = 3;
    else if (n <= 208)
        APerWarp = 2, splitN = 1, splitK = 3;
    else if (n <= 224)
        APerWarp = 2, splitN = 1, splitK = 3;
    else if (n <= 240)
        APerWarp = 2, splitN = 1, splitK = 3;
    else if (n <= 256)
        APerWarp = 1, splitN = 1, splitK = 1;

    // dispatch the warps to gemm_id
    for (int i = 1; i < gemm_count + 1; i++) {
        blockCountSum[i] =
            m / 16 / (block_dim_x / WARPSIZE * APerWarp) * splitN * splitK;
    }
    for (int i = 1; i < gemm_count + 1; i++) {
        blockCountSum[i] += blockCountSum[i - 1];
    }

    Tc **dev_bias = nullptr;
    if (bias != nullptr) {
        mcMalloc((void ***)&dev_bias, sizeof(Tc *) * gemm_count);
        mcMemcpy(dev_bias, bias, sizeof(Tc *) * gemm_count,
                 mcMemcpyHostToDevice);
    }
    int *dev_blockCountSum;
    mcMalloc((void **)&dev_blockCountSum, sizeof(int) * (gemm_count + 1));
    mcMemcpy(dev_blockCountSum, blockCountSum, sizeof(int) * (gemm_count + 1),
             mcMemcpyHostToDevice);

    // Launch kernel
#define DISPATCH_GEMM_OPT_KERNEL(APERWARP, SPLITN, SPLITK, N)                  \
    group_gemm_tn2<Tab, Taccum, Tc, block_dim_x, APERWARP, SPLITN, SPLITK, N>  \
        <<<blockCountSum[gemm_count], block_dim_x, 0, stream>>>(               \
            A, B, C, m, n, k, dev_alpha, beta, gemm_count, dev_blockCountSum,  \
            dev_bias);

    const mcStream_t stream =
        at::cuda::getCurrentCUDAStream(at::cuda::current_device());

    if (n <= 16) {
        DISPATCH_GEMM_OPT_KERNEL(2, 1, 1, 16);
    } else if (n <= 32) {
        DISPATCH_GEMM_OPT_KERNEL(2, 1, 1, 32);
    } else if (n <= 48) {
        DISPATCH_GEMM_OPT_KERNEL(2, 1, 1, 48);
    } else if (n <= 64) {
        DISPATCH_GEMM_OPT_KERNEL(2, 1, 1, 64);
    } else if (n <= 80) {
        DISPATCH_GEMM_OPT_KERNEL(2, 1, 1, 80);
    } else if (n <= 96) {
        DISPATCH_GEMM_OPT_KERNEL(2, 1, 2, 96);
    } else if (n <= 112) {
        DISPATCH_GEMM_OPT_KERNEL(2, 1, 2, 112);
    } else if (n <= 128) {
        DISPATCH_GEMM_OPT_KERNEL(2, 1, 2, 128);
    } else if (n <= 144) {
        DISPATCH_GEMM_OPT_KERNEL(2, 1, 3, 144);
    } else if (n <= 160) {
        DISPATCH_GEMM_OPT_KERNEL(2, 1, 2, 160);
    } else if (n <= 176) {
        DISPATCH_GEMM_OPT_KERNEL(2, 1, 3, 176);
    } else if (n <= 192) {
        DISPATCH_GEMM_OPT_KERNEL(2, 1, 3, 192);
    } else if (n <= 208) {
        DISPATCH_GEMM_OPT_KERNEL(2, 1, 3, 208);
    } else if (n <= 224) {
        DISPATCH_GEMM_OPT_KERNEL(2, 1, 3, 224);
    } else if (n <= 240) {
        DISPATCH_GEMM_OPT_KERNEL(2, 1, 3, 240);
    } else if (n <= 256) {
        DISPATCH_GEMM_OPT_KERNEL(1, 1, 1, 256);
    }

#undef DISPATCH_GEMM_OPT_KERNEL
    mcDeviceSynchronize();

    // mcFree(dev_alpha);
    if (bias != nullptr) {
        mcFree(dev_bias);
    }
    mcFree(dev_blockCountSum);
    free(blockCountSum);
}

} // namespace muxi_layout_kernels
