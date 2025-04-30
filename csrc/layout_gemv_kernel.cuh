#pragma once
#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>

#include "utils.cuh"

namespace muxi_layout_kernels {

template <typename Tax, typename Taccum, typename Ty, int BLOCK_DIM_X,
          int APerWarp, int splitK, bool IsBetaZero, bool HasOneDimBias = false>
__global__ void __launch_bounds__(BLOCK_DIM_X)
    MvMmaLayoutKernel(const Tax *A, const Tax *x, Ty *y, int spatialDim,
                      int reducedDim, Taccum alpha, Taccum beta,
                      Ty *bias = nullptr) {
    constexpr int stages = 8;
    constexpr int elementsPerAccess = 8;
    constexpr int rowThreadsPerMma = 16;
    constexpr int colThreadsPerMma = 4;
    constexpr int elementsPerThreadPerMma = 4;
    constexpr int warpPerBlock = BLOCK_DIM_X / WARP_SIZE;
    const int rowsGroup = (spatialDim + rowThreadsPerMma * APerWarp - 1) /
                          rowThreadsPerMma / APerWarp;

    using yStgType = __NATIVE_VECTOR__(sizeof(Ty), uint);

    int nChunksPerRow = __builtin_mxc_readfirstlane(
        (reducedDim) / (stages * elementsPerAccess * colThreadsPerMma));
    int numWarps = __builtin_mxc_readfirstlane(gridDim.x * blockDim.x /
                                               WARP_SIZE / splitK);
    int warpId = __builtin_mxc_readfirstlane(
                     (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE) %
                 numWarps;
    int warpIdInBlock = (threadIdx.x / WARP_SIZE);
    int splitKId = __builtin_mxc_readfirstlane(
        ((blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE) / numWarps);
    int laneId = threadIdx.x & (WARP_SIZE - 1);
    int quarterWarpId = laneId / 16;
    int quarterLaneId = laneId & (16 - 1);

    int warpRowsGroupBegin =
        __builtin_mxc_readfirstlane(warpId * (rowsGroup / numWarps) +
                                    min(warpId, rowsGroup % numWarps)) *
        APerWarp;
    int warpRowsGroupEnd =
        __builtin_mxc_readfirstlane((warpId + 1) * (rowsGroup / numWarps) +
                                    min(warpId + 1, rowsGroup % numWarps)) *
        APerWarp;
    int splitKStart =
        __builtin_mxc_readfirstlane(splitKId * (nChunksPerRow / splitK) +
                                    min(splitKId, nChunksPerRow % splitK));
    int splitKEnd =
        __builtin_mxc_readfirstlane((splitKId + 1) * (nChunksPerRow / splitK) +
                                    min(splitKId + 1, nChunksPerRow % splitK));

    if (nChunksPerRow < splitK) {
        splitKStart = splitKId;
        splitKEnd = splitKId + 1;
        if (splitKEnd > nChunksPerRow) {
            splitKEnd = nChunksPerRow;
        }
    }

    int end = __builtin_mxc_readfirstlane(splitKEnd - splitKStart);

    int A_offset = ((warpRowsGroupBegin * rowThreadsPerMma) *
                    (reducedDim / elementsPerAccess)) +
                   laneId + splitKStart * stages * WARP_SIZE;

    int y_offset[APerWarp];
    for (int i = 0; i < APerWarp; ++i) {
        y_offset[i] = ((warpRowsGroupBegin + i) * rowThreadsPerMma) /
                          elementsPerThreadPerMma +
                      quarterWarpId;
    }

    const UINT4 *A_ptr[APerWarp];
    for (int i = 0; i < APerWarp; ++i) {
        A_ptr[i] = reinterpret_cast<const UINT4 *>(A) + A_offset +
                   i * (rowThreadsPerMma * (reducedDim / elementsPerAccess));
    }

    const UINT4 *x_ptr = reinterpret_cast<const UINT4 *>(x) + quarterWarpId +
                         splitKStart * stages * colThreadsPerMma;
    yStgType *y_ptr = reinterpret_cast<yStgType *>(y);

    UINT4 tmpA[stages][APerWarp], tmpX[stages];
    FLOAT4 y_f32[APerWarp];

    for (int i = 0; i < APerWarp; ++i) {
        y_f32[i] = {0.0f, 0.0f, 0.0f, 0.0f};
    }

    for (int index_A = 0; index_A < APerWarp; ++index_A) {
        if (warpRowsGroupBegin + index_A >
            (spatialDim / rowThreadsPerMma - 1)) {
            A_ptr[index_A] = reinterpret_cast<const UINT4 *>(A) + laneId +
                             splitKStart * stages * WARP_SIZE;
        }
    }

    int A_ptr_offset = 0;
    int B_ptr_offset = 0;

    // Load: 0
    if (end - 1 >= 0) {
#pragma unroll stages
        for (int k = 0; k < stages; k++) {
            asm("/* Stop compiler reordering (A) */");
            for (int index_A = 0; index_A < APerWarp; ++index_A) {
                tmpA[k][index_A] = *(A_ptr[index_A] + A_ptr_offset);
            }
            A_ptr_offset += WARP_SIZE;
            tmpX[k] = *(x_ptr + B_ptr_offset);
            B_ptr_offset += colThreadsPerMma;
        }
    }

    // Compute + save: [0, end - 2]
    // Load: [1, end - 1]
    for (int i = 0; i < end - 1; ++i) {
#pragma unroll stages
        for (int k = 0; k < stages; k++) {
            asm("/* Stop compiler reordering (B) */");
            for (int index_A = 0; index_A < APerWarp; ++index_A) {
                y_f32[index_A] = mma_16x16x16f16<Tax>(
                    tmpA[k][index_A][0], tmpA[k][index_A][1], tmpX[k][0],
                    tmpX[k][1], y_f32[index_A]);
                y_f32[index_A] = mma_16x16x16f16<Tax>(
                    tmpA[k][index_A][2], tmpA[k][index_A][3], tmpX[k][2],
                    tmpX[k][3], y_f32[index_A]);
            }
            for (int index_A = 0; index_A < APerWarp; ++index_A) {
                tmpA[k][index_A] = *(A_ptr[index_A] + A_ptr_offset);
            }
            A_ptr_offset += WARP_SIZE;
            tmpX[k] = *(x_ptr + B_ptr_offset);
            B_ptr_offset += colThreadsPerMma;
        }
    }

    // Compute + save: end - 1
    if (end - 1 >= 0) {
#pragma unroll stages
        for (int k = 0; k < stages; k++) {
            asm("/* Stop compiler reordering (C) */");
            for (int index_A = 0; index_A < APerWarp; ++index_A) {
                y_f32[index_A] = mma_16x16x16f16<Tax>(
                    tmpA[k][index_A][0], tmpA[k][index_A][1], tmpX[k][0],
                    tmpX[k][1], y_f32[index_A]);
                y_f32[index_A] = mma_16x16x16f16<Tax>(
                    tmpA[k][index_A][2], tmpA[k][index_A][3], tmpX[k][2],
                    tmpX[k][3], y_f32[index_A]);
            }
        }

        int remainingDim = reducedDim - nChunksPerRow * stages *
                                            elementsPerAccess *
                                            colThreadsPerMma;
        if ((remainingDim > 0) && (splitKId == splitK - 1)) {
            for (int i = 0;
                 i < remainingDim / (elementsPerAccess * colThreadsPerMma);
                 ++i) {
                for (int index_A = 0; index_A < APerWarp; ++index_A) {
                    tmpA[0][index_A] = *(A_ptr[index_A] + A_ptr_offset);
                }
                A_ptr_offset += WARP_SIZE;
                tmpX[0] = *(x_ptr + B_ptr_offset);
                B_ptr_offset += colThreadsPerMma;
                for (int index_A = 0; index_A < APerWarp; ++index_A) {
                    y_f32[index_A] = mma_16x16x16f16<Tax>(
                        tmpA[0][index_A][0], tmpA[0][index_A][1], tmpX[0][0],
                        tmpX[0][1], y_f32[index_A]);
                    y_f32[index_A] = mma_16x16x16f16<Tax>(
                        tmpA[0][index_A][2], tmpA[0][index_A][3], tmpX[0][2],
                        tmpX[0][3], y_f32[index_A]);
                }
            }
        }

        for (int index_A = 0; index_A < APerWarp; ++index_A) {
            if (warpRowsGroupBegin + index_A >
                (spatialDim / rowThreadsPerMma - 1)) {
                y_f32[index_A] = {0.0f, 0.0f, 0.0f, 0.0f};
                y_offset[index_A] = quarterWarpId; // Avoid memory out-of-bounds
            }
        }

        for (int index_A = 0; index_A < APerWarp; ++index_A) {
            if (quarterLaneId == 0) {
                y_f32[index_A][0] *= alpha;
                y_f32[index_A][1] *= alpha;
                y_f32[index_A][2] *= alpha;
                y_f32[index_A][3] *= alpha;
                Ty y_half_tmp[4] = {0};
                y_half_tmp[0] = static_cast<Ty>(y_f32[index_A][0]);
                y_half_tmp[1] = static_cast<Ty>(y_f32[index_A][1]);
                y_half_tmp[2] = static_cast<Ty>(y_f32[index_A][2]);
                y_half_tmp[3] = static_cast<Ty>(y_f32[index_A][3]);

                if constexpr (HasOneDimBias) {
                    if (splitKId == 0) {
                        yStgType bias_load =
                            *(reinterpret_cast<yStgType *>(bias) +
                              y_offset[index_A]);
                        Ty *bias_tc = reinterpret_cast<Ty *>(&bias_load);
#pragma unroll 4
                        for (int t = 0; t < 4; t++) {
                            y_half_tmp[t] = __hadd(y_half_tmp[t], bias_tc[t]);
                        }
                    }
                }

                if constexpr (std::is_same_v<Ty, __half>) {
                    atomicAdd(
                        reinterpret_cast<__half2 *>(&y_ptr[y_offset[index_A]]),
                        {y_half_tmp[0], y_half_tmp[1]});
                    atomicAdd(
                        reinterpret_cast<__half2 *>(&y_ptr[y_offset[index_A]]) +
                            1,
                        {y_half_tmp[2], y_half_tmp[3]});
                } else if constexpr (std::is_same_v<Ty, __maca_bfloat16>) {
                    atomicAdd(reinterpret_cast<__maca_bfloat162 *>(
                                  &y_ptr[y_offset[index_A]]),
                              {y_half_tmp[0], y_half_tmp[1]});
                    atomicAdd(reinterpret_cast<__maca_bfloat162 *>(
                                  &y_ptr[y_offset[index_A]]) +
                                  1,
                              {y_half_tmp[2], y_half_tmp[3]});
                }
            }
        }
    }
}

template <typename Tax, typename Taccum, typename Ty, int BLOCK_DIM_X,
          int THREADS_PER_ROW, bool IsBetaZero, bool HasOneDimBias = false>
__global__ void __launch_bounds__(BLOCK_DIM_X)
    MvSimtLayoutKernel(const Tax *A, const Tax *x, Ty *y, int spatialDim,
                       int reducedDim, Taccum alpha, Taccum beta,
                       Ty *bias = nullptr) {
    constexpr int stages = 8;
    constexpr int elementsPerAccess = 8;
    constexpr int rowThreadsPerMma = 16;
    constexpr int colThreadsPerMma = 4;
    constexpr int elementsPerThreadPerMma = 4;
    constexpr int warpPerBlock = BLOCK_DIM_X / WARP_SIZE;

    __shared__ UINT4
        shared_x[1024]; // Attention: just support k <= 8192 now !!!
    for (int i = threadIdx.x;
         (i < 1024) && (i < reducedDim / elementsPerAccess); i += blockDim.x) {
        shared_x[i] = reinterpret_cast<const UINT4 *>(x)[i];
    }

    const int nChunksPerRow = reducedDim / rowThreadsPerMma / stages;

    const int numWarps = (blockDim.x * gridDim.x) / WARP_SIZE;
    const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int laneId = threadIdx.x & (WARP_SIZE - 1);
    // const int quarterWarpId = laneId >> 2;
    // const int quarterLaneId = laneId & 15;

    const int numWorkGroups = numWarps * WARP_SIZE / THREADS_PER_ROW;
    const int workGroupId = (blockIdx.x * blockDim.x) / THREADS_PER_ROW +
                            (threadIdx.x & (BLOCK_DIM_X / THREADS_PER_ROW - 1));
    const int workIdInGroup = threadIdx.x / (BLOCK_DIM_X / THREADS_PER_ROW);

    int rowId = workGroupId;

    int chunks = reducedDim / elementsPerAccess / stages;

    int end = chunks / THREADS_PER_ROW;

    int tails = reducedDim / elementsPerAccess - end * THREADS_PER_ROW * stages;

    int tail_start = workIdInGroup * (tails / THREADS_PER_ROW) +
                     std::min(workIdInGroup, tails % THREADS_PER_ROW);
    int tail_end = (workIdInGroup + 1) * (tails / THREADS_PER_ROW) +
                   std::min(workIdInGroup + 1, tails % THREADS_PER_ROW);

    int A_offset =
        (rowId >> 4) * rowThreadsPerMma * (reducedDim / elementsPerAccess) +
        (rowId & (rowThreadsPerMma - 1)) + workIdInGroup * rowThreadsPerMma;

    const UINT4 *A_ptr = reinterpret_cast<const UINT4 *>(A) + A_offset;
    const UINT4 *x_ptr = &shared_x[workIdInGroup];

    UINT4 tmpA[stages], tmpX[stages];
    Taccum y_sum = 0.0f;

    int A_ptr_offset = 0;
    int B_ptr_offset = 0;

    __syncthreads();

    if (rowId >= spatialDim)
        return;

    // Load: 0
    if (end - 1 >= 0) {
#pragma unroll stages
        for (int k = 0; k < stages; k++) {
            asm("/* Stop compiler reordering (A) */");
            tmpA[k] = *(A_ptr + A_ptr_offset);
            A_ptr_offset += rowThreadsPerMma * THREADS_PER_ROW;
            tmpX[k] = *(x_ptr + B_ptr_offset);
            B_ptr_offset += THREADS_PER_ROW;
        }
    }

    // Compute + save: [0, end - 2]
    // Load: [1, end - 1]
    for (int i = 0; i < end - 1; ++i) {
#pragma unroll stages
        for (int k = 0; k < stages; k++) {
            asm("/* Stop compiler reordering (B) */");
            // compute
            if constexpr (std::is_same_v<Tax, __half>) {
                y_sum += static_cast<Taccum>(dotHalf(tmpA[k], tmpX[k]));
            } else {
                y_sum += static_cast<Taccum>(dotBf16(tmpA[k], tmpX[k]));
            }
            // prefetch
            tmpA[k] = *(A_ptr + A_ptr_offset);
            A_ptr_offset += rowThreadsPerMma * THREADS_PER_ROW;
            tmpX[k] = *(x_ptr + B_ptr_offset);
            B_ptr_offset += THREADS_PER_ROW;
        }
    }

    // Compute + save: end - 1
    if (end - 1 >= 0) {
#pragma unroll stages
        for (int k = 0; k < stages; k++) {
            asm("/* Stop compiler reordering (C) */");
            // compute
            if constexpr (std::is_same_v<Tax, __half>) {
                y_sum += static_cast<Taccum>(dotHalf(tmpA[k], tmpX[k]));
            } else {
                y_sum += static_cast<Taccum>(dotBf16(tmpA[k], tmpX[k]));
            }
        }
    }

    for (int i = tail_start; i < tail_end; i++) {
        tmpA[0] = *(A_ptr + A_ptr_offset);
        A_ptr_offset += rowThreadsPerMma * THREADS_PER_ROW;
        tmpX[0] = *(x_ptr + B_ptr_offset);
        B_ptr_offset += THREADS_PER_ROW;
        if constexpr (std::is_same_v<Tax, __half>) {
            y_sum += static_cast<Taccum>(dotHalf(tmpA[0], tmpX[0]));
        } else {
            y_sum += static_cast<Taccum>(dotBf16(tmpA[0], tmpX[0]));
        }
    }

    Taccum *y_shared = reinterpret_cast<Taccum *>(&shared_x[0]);
    __syncthreads();
    y_shared[threadIdx.x] = y_sum;
    for (int i = THREADS_PER_ROW / 2; i > 0; i /= 2) {
        __syncthreads();
        if (workIdInGroup < i) {
            y_shared[threadIdx.x] +=
                y_shared[threadIdx.x + i * (BLOCK_DIM_X / THREADS_PER_ROW)];
        }
    }

    if (workIdInGroup == 0) {
        if constexpr (HasOneDimBias) {
            y[rowId] = alpha * y_shared[threadIdx.x &
                                        (BLOCK_DIM_X / THREADS_PER_ROW - 1)] +
                       static_cast<Taccum>(bias[rowId]);
        } else {
            y[rowId] =
                alpha *
                y_shared[threadIdx.x & (BLOCK_DIM_X / THREADS_PER_ROW - 1)];
        }
    }
}

template <typename Tax, typename Taccum, typename Ty, int BLOCK_DIM_X>
void GemvMmaLayoutDispatch(const Tax *A, const Tax *x, Ty *y, int spatialDim,
                           int reducedDim, Taccum alpha, Taccum beta,
                           Ty *bias = nullptr, int KernelId = 1,
                           int KernelParam1 = 1, int KernelParam2 = 1) {
    bool IsBetaZero = (beta == static_cast<Taccum>(0));
    bool HasOneDimBias = !(bias == nullptr);
    int NUM_BLOCKS;
    if (KernelId == 1) {
        NUM_BLOCKS = (((spatialDim / 16) +
                       (BLOCK_DIM_X / WARP_SIZE * KernelParam1) - 1) /
                      (BLOCK_DIM_X / WARP_SIZE * KernelParam1)) *
                     KernelParam2;
    } else if (KernelId == 2) {
        NUM_BLOCKS = spatialDim / (BLOCK_DIM_X / KernelParam1);
    }

#define LAUNCH_GEMV_MMA(ISBETAZERO, HASONEDIMBIAS, KernelParam1, KernelParam2) \
    auto cur_device = at::cuda::current_device();                              \
    const mcStream_t stream = at::cuda::getCurrentCUDAStream(cur_device);      \
    if (ISBETAZERO) {                                                          \
        if (HASONEDIMBIAS) {                                                   \
            MvMmaLayoutKernel<Tax, Taccum, Ty, BLOCK_DIM_X, KernelParam1,      \
                              KernelParam2, true, true>                        \
                <<<NUM_BLOCKS, BLOCK_DIM_X, 0, stream>>>(                      \
                    A, x, y, spatialDim, reducedDim, alpha, beta, bias);       \
        } else {                                                               \
            MvMmaLayoutKernel<Tax, Taccum, Ty, BLOCK_DIM_X, KernelParam1,      \
                              KernelParam2, true, false>                       \
                <<<NUM_BLOCKS, BLOCK_DIM_X, 0, stream>>>(                      \
                    A, x, y, spatialDim, reducedDim, alpha, beta, bias);       \
        }                                                                      \
    } else {                                                                   \
        if (HASONEDIMBIAS) {                                                   \
            MvMmaLayoutKernel<Tax, Taccum, Ty, BLOCK_DIM_X, KernelParam1,      \
                              KernelParam2, false, true>                       \
                <<<NUM_BLOCKS, BLOCK_DIM_X, 0, stream>>>(                      \
                    A, x, y, spatialDim, reducedDim, alpha, beta, bias);       \
        } else {                                                               \
            MvMmaLayoutKernel<Tax, Taccum, Ty, BLOCK_DIM_X, KernelParam1,      \
                              KernelParam2, false, false>                      \
                <<<NUM_BLOCKS, BLOCK_DIM_X, 0, stream>>>(                      \
                    A, x, y, spatialDim, reducedDim, alpha, beta, bias);       \
        }                                                                      \
    }

#define LAUNCH_GEMV_SIMT(ISBETAZERO, HASONEDIMBIAS, KernelParam1)              \
    auto cur_device = at::cuda::current_device();                              \
    const mcStream_t stream = at::cuda::getCurrentCUDAStream(cur_device);      \
    if (ISBETAZERO) {                                                          \
        if (HASONEDIMBIAS) {                                                   \
            MvSimtLayoutKernel<Tax, Taccum, Ty, BLOCK_DIM_X, KernelParam1,     \
                               true, true>                                     \
                <<<NUM_BLOCKS, BLOCK_DIM_X, 0, stream>>>(                      \
                    A, x, y, spatialDim, reducedDim, alpha, beta, bias);       \
        } else {                                                               \
            MvSimtLayoutKernel<Tax, Taccum, Ty, BLOCK_DIM_X, KernelParam1,     \
                               true, false>                                    \
                <<<NUM_BLOCKS, BLOCK_DIM_X, 0, stream>>>(                      \
                    A, x, y, spatialDim, reducedDim, alpha, beta, bias);       \
        }                                                                      \
    } else {                                                                   \
        if (HASONEDIMBIAS) {                                                   \
            MvSimtLayoutKernel<Tax, Taccum, Ty, BLOCK_DIM_X, KernelParam1,     \
                               false, true>                                    \
                <<<NUM_BLOCKS, BLOCK_DIM_X, 0, stream>>>(                      \
                    A, x, y, spatialDim, reducedDim, alpha, beta, bias);       \
        } else {                                                               \
            MvSimtLayoutKernel<Tax, Taccum, Ty, BLOCK_DIM_X, KernelParam1,     \
                               false, false>                                   \
                <<<NUM_BLOCKS, BLOCK_DIM_X, 0, stream>>>(                      \
                    A, x, y, spatialDim, reducedDim, alpha, beta, bias);       \
        }                                                                      \
    }

    if (KernelId == 1) {
        if (KernelParam1 == 1 && KernelParam2 == 1) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 1, 1);
        } else if (KernelParam1 == 1 && KernelParam2 == 2) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 1, 2);
        } else if (KernelParam1 == 1 && KernelParam2 == 3) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 1, 3);
        } else if (KernelParam1 == 1 && KernelParam2 == 4) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 1, 4);
        } else if (KernelParam1 == 1 && KernelParam2 == 5) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 1, 5);
        } else if (KernelParam1 == 1 && KernelParam2 == 6) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 1, 6);
        } else if (KernelParam1 == 1 && KernelParam2 == 7) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 1, 7);
        } else if (KernelParam1 == 1 && KernelParam2 == 8) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 1, 8);
        } else if (KernelParam1 == 2 && KernelParam2 == 1) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 2, 1);
        } else if (KernelParam1 == 2 && KernelParam2 == 2) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 2, 2);
        } else if (KernelParam1 == 2 && KernelParam2 == 3) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 2, 3);
        } else if (KernelParam1 == 2 && KernelParam2 == 4) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 2, 4);
        } else if (KernelParam1 == 2 && KernelParam2 == 5) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 2, 5);
        } else if (KernelParam1 == 2 && KernelParam2 == 6) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 2, 6);
        } else if (KernelParam1 == 2 && KernelParam2 == 7) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 2, 7);
        } else if (KernelParam1 == 2 && KernelParam2 == 8) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 2, 8);
        } else if (KernelParam1 == 3 && KernelParam2 == 1) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 3, 1);
        } else if (KernelParam1 == 3 && KernelParam2 == 2) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 3, 2);
        } else if (KernelParam1 == 3 && KernelParam2 == 3) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 3, 3);
        } else if (KernelParam1 == 3 && KernelParam2 == 4) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 3, 4);
        } else if (KernelParam1 == 3 && KernelParam2 == 5) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 3, 5);
        } else if (KernelParam1 == 3 && KernelParam2 == 6) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 3, 6);
        } else if (KernelParam1 == 3 && KernelParam2 == 7) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 3, 7);
        } else if (KernelParam1 == 3 && KernelParam2 == 8) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 3, 8);
        } else if (KernelParam1 == 4 && KernelParam2 == 1) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 4, 1);
        } else if (KernelParam1 == 4 && KernelParam2 == 2) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 4, 2);
        } else if (KernelParam1 == 4 && KernelParam2 == 3) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 4, 3);
        } else if (KernelParam1 == 4 && KernelParam2 == 4) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 4, 4);
        } else if (KernelParam1 == 4 && KernelParam2 == 5) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 4, 5);
        } else if (KernelParam1 == 4 && KernelParam2 == 6) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 4, 6);
        } else if (KernelParam1 == 4 && KernelParam2 == 7) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 4, 7);
        } else if (KernelParam1 == 4 && KernelParam2 == 8) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 4, 8);
        } else if (KernelParam1 == 5 && KernelParam2 == 1) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 5, 1);
        } else if (KernelParam1 == 5 && KernelParam2 == 2) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 5, 2);
        } else if (KernelParam1 == 5 && KernelParam2 == 3) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 5, 3);
        } else if (KernelParam1 == 5 && KernelParam2 == 4) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 5, 4);
        } else if (KernelParam1 == 5 && KernelParam2 == 5) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 5, 5);
        } else if (KernelParam1 == 5 && KernelParam2 == 6) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 5, 6);
        } else if (KernelParam1 == 5 && KernelParam2 == 7) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 5, 7);
        } else if (KernelParam1 == 5 && KernelParam2 == 8) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 5, 8);
        } else if (KernelParam1 == 6 && KernelParam2 == 1) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 6, 1);
        } else if (KernelParam1 == 6 && KernelParam2 == 2) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 6, 2);
        } else if (KernelParam1 == 6 && KernelParam2 == 3) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 6, 3);
        } else if (KernelParam1 == 6 && KernelParam2 == 4) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 6, 4);
        } else if (KernelParam1 == 6 && KernelParam2 == 5) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 6, 5);
        } else if (KernelParam1 == 6 && KernelParam2 == 6) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 6, 6);
        } else if (KernelParam1 == 6 && KernelParam2 == 7) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 6, 7);
        } else if (KernelParam1 == 6 && KernelParam2 == 8) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 6, 8);
        } else if (KernelParam1 == 7 && KernelParam2 == 1) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 7, 1);
        } else if (KernelParam1 == 7 && KernelParam2 == 2) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 7, 2);
        } else if (KernelParam1 == 7 && KernelParam2 == 3) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 7, 3);
        } else if (KernelParam1 == 7 && KernelParam2 == 4) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 7, 4);
        } else if (KernelParam1 == 7 && KernelParam2 == 5) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 7, 5);
        } else if (KernelParam1 == 7 && KernelParam2 == 6) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 7, 6);
        } else if (KernelParam1 == 7 && KernelParam2 == 7) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 7, 7);
        } else if (KernelParam1 == 7 && KernelParam2 == 8) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 7, 8);
        } else if (KernelParam1 == 8 && KernelParam2 == 1) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 8, 1);
        } else if (KernelParam1 == 8 && KernelParam2 == 2) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 8, 2);
        } else if (KernelParam1 == 8 && KernelParam2 == 3) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 8, 3);
        } else if (KernelParam1 == 8 && KernelParam2 == 4) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 8, 4);
        } else if (KernelParam1 == 8 && KernelParam2 == 5) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 8, 5);
        } else if (KernelParam1 == 8 && KernelParam2 == 6) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 8, 6);
        } else if (KernelParam1 == 8 && KernelParam2 == 7) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 8, 7);
        } else if (KernelParam1 == 8 && KernelParam2 == 8) {
            LAUNCH_GEMV_MMA(IsBetaZero, HasOneDimBias, 8, 8);
        }
    } else if (KernelId == 2) {
        if (KernelParam1 == 1) {
            LAUNCH_GEMV_SIMT(IsBetaZero, HasOneDimBias, 1);
        } else if (KernelParam1 == 2) {
            LAUNCH_GEMV_SIMT(IsBetaZero, HasOneDimBias, 2);
        } else if (KernelParam1 == 4) {
            LAUNCH_GEMV_SIMT(IsBetaZero, HasOneDimBias, 4);
        } else if (KernelParam1 == 8) {
            LAUNCH_GEMV_SIMT(IsBetaZero, HasOneDimBias, 8);
        } else if (KernelParam1 == 16) {
            LAUNCH_GEMV_SIMT(IsBetaZero, HasOneDimBias, 16);
        } else if (KernelParam1 == 32) {
            LAUNCH_GEMV_SIMT(IsBetaZero, HasOneDimBias, 32);
        } else if (KernelParam1 == 64) {
            LAUNCH_GEMV_SIMT(IsBetaZero, HasOneDimBias, 64);
        } else if (KernelParam1 == 128) {
            LAUNCH_GEMV_SIMT(IsBetaZero, HasOneDimBias, 128);
        } else if (KernelParam1 == 256) {
            LAUNCH_GEMV_SIMT(IsBetaZero, HasOneDimBias, 256);
        } else if (KernelParam1 == 512) {
            LAUNCH_GEMV_SIMT(IsBetaZero, HasOneDimBias, 512);
        } else if (KernelParam1 == 1024) {
            LAUNCH_GEMV_SIMT(IsBetaZero, HasOneDimBias, 1024);
        }
    }

#undef LAUNCH_GEMV_MMA
#undef LAUNCH_GEMV_SIMT
}

} // namespace muxi_layout_kernels
