#pragma once

#include "../utils.cuh"
#include "group_gemm_utils.h"

namespace muxi_layout_kernels {

inline int nextPow2_bit(int n) {
    if (n == 0)
        return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

template <typename T> __device__ __forceinline__ T my_silu(const T &x) {
    return fp_cast<T>((fp_cast<float>(x)) / (1.0f + exp(-fp_cast<float>(x))));
}

template <typename T>
__global__ void silu_and_mul_kernel(T *input, int n, int m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int chunkSize = m / 2 / ELEMENTSPERACCESS;
        UINT4 *A_ptr =
            reinterpret_cast<UINT4 *>(input) + tid * m / ELEMENTSPERACCESS;
        UINT4 *B_ptr = A_ptr + (m / 2 / ELEMENTSPERACCESS);
        for (int i = 0; i < chunkSize; i++) {
            UINT4 A = *(A_ptr + i);
            UINT4 B = *(B_ptr + i);
            T temp_t[8];
            for (int j = 0; j < 8; j++) {
                temp_t[j] = __hmul(my_silu(reinterpret_cast<T *>(&A)[j]),
                                   reinterpret_cast<T *>(&B)[j]);
            }
            *(A_ptr + i) = *reinterpret_cast<UINT4 *>(&temp_t[0]);
        }
        if ((m / 2) % ELEMENTSPERACCESS > 0) {
            for (int i = chunkSize * ELEMENTSPERACCESS; i < m / 2; i++) {
                T a = my_silu(input[tid * m + i]);
                T b = input[tid * m + i + m / 2];
                input[tid * m + i] = __hmul(a, b);
            }
        }
    }
}

template <typename T>
__global__ void silu_and_mul_kernel_block(T *input, int n, int m) {
    UINT4 A;
    UINT4 B;
    UINT4 C;
    int chunkSize = m / 2 / ELEMENTSPERACCESS;
    int tid = threadIdx.x;
    int row_id = blockIdx.x;

    if (row_id < n && tid < chunkSize) {
        UINT4 *A_ptr =
            reinterpret_cast<UINT4 *>(input) + row_id * m / ELEMENTSPERACCESS;
        UINT4 *B_ptr = A_ptr + (m / 2 / ELEMENTSPERACCESS);
        A = *(A_ptr + tid);
        B = *(B_ptr + tid);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            reinterpret_cast<T *>(&C)[j] =
                __hmul(my_silu(reinterpret_cast<T *>(&A)[j]),
                       reinterpret_cast<T *>(&B)[j]);
        }
        *(A_ptr + tid) = C;
    }
    int tail = (m % (2 * ELEMENTSPERACCESS)) / 2;
    for (int i = tid; i < tail; i += blockDim.x) {
        T a = my_silu(input[row_id * m + chunkSize * ELEMENTSPERACCESS + i]);
        T b = input[row_id * m + m / 2 + chunkSize * ELEMENTSPERACCESS + i];
        input[row_id * m + chunkSize * ELEMENTSPERACCESS + tid] = __hmul(a, b);
    }
}

} // namespace muxi_layout_kernels
