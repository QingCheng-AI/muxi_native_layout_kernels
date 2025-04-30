#pragma once

#include "group_gemm_utils.h"

template <typename T> __device__ __forceinline__ T silu(const T &x) {
    return (T)(((float)x) / (1.0f + exp(-(float)x)));
}

template <typename T>
__device__ void silu_and_mul_kernel(T *input, int n, int m, int warpId) {
    int laneId = threadIdx.x % WARPSIZE;
    int chunkSize =
        (n / ROWTHREADSPERMMA) * (m / 2 / COLTHREADSPERMMA / ELEMENTSPERACCESS);
    int numWarps = blockDim.x / WARPSIZE;
    int chunkStart =
        warpId * (chunkSize / numWarps) + min(warpId, chunkSize % numWarps);
    int chunkEnd = (warpId + 1) * (chunkSize / numWarps) +
                   min(warpId + 1, chunkSize % numWarps);

    UINT4 *A_ptr = reinterpret_cast<UINT4 *>(
                       input + chunkStart * ROWTHREADSPERMMA *
                                   COLTHREADSPERMMA * ELEMENTSPERACCESS) +
                   laneId;
    UINT4 *B_ptr = A_ptr + (n * m / 2 / ELEMENTSPERACCESS);

    for (int i = 0; i < chunkEnd - chunkStart; i++) {
        constexpr int offset = ROWTHREADSPERMMA * COLTHREADSPERMMA;
        UINT4 A = *(A_ptr + offset * i);
        UINT4 B = *(B_ptr + offset * i);
        T temp_t[8];
        for (int j = 0; j < 8; j++) {
            temp_t[j] = __hmul(silu(reinterpret_cast<T *>(&A)[j]),
                               reinterpret_cast<T *>(&B)[j]);
        }
        *(A_ptr + offset * i) = *reinterpret_cast<UINT4 *>(&temp_t[0]);
    }
}

template <typename T>
__global__ void silu_and_mul(T **input, int n, int m, int active_count) {
    int warpPerActiveExpert = blockDim.x / WARPSIZE;
    int global_warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
    int warpIdInExpert = global_warpId % warpPerActiveExpert;
    int activeId = global_warpId / warpPerActiveExpert;

    silu_and_mul_kernel<T>(input[activeId], n, m, warpIdInExpert);
}