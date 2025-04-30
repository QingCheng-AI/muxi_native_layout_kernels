#pragma once
#include "group_gemm_utils.h"

template <typename T>
__device__ __forceinline__ T my_shfl_xor_sync(T var, int offset, int warpSize) {
    int index = (threadIdx.x % warpSize) ^ offset;
    int ret = __builtin_mxc_bsm_bpermute(index << 2, *((int *)&var));
    return *((T *)&ret);
}

template <typename T> __device__ T warp_reduce_max(T val) {
    // warp 级别的最大值
    for (int offset = 32; offset > 0; offset >>= 1) {
        T b = my_shfl_xor_sync(val, offset, WARPSIZE);
        val = __hgt(b, val) ? b : val;
    }
    return val;
}

template <typename T> __device__ T warp_reduce_sum(T val) {
    // warp级别的求和
    for (int offset = 32; offset > 0; offset >>= 1) {
        val = __hadd(val, my_shfl_xor_sync(val, offset, WARPSIZE));
    }
    return val;
}

template <typename T>
__global__ void softmax_kernel(T *out, const T *inp, int N, int C) {
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int i = C * blockIdx.x;
    // 计算每个线程块计算大小内的最大值->这里是每一个block计算一行
    T max_val = inp[i];
    for (int j = tid; j < C; j += block_size) {
        max_val = __hgt(max_val, inp[i + j]) ? max_val : inp[i + j];
    }
    max_val = warp_reduce_max(max_val);
    // if (idx == 0 && tid == 0) {
    //   printf("max_val: %f\n", static_cast<float>(max_val));
    // }
    T offset =
        __shfl_sync(0xFFFFFFFFFFFFFFFF, max_val, 0); // 广播到warp的全部位置
                                                     // 计算 sum求和

    // if (idx == 0 && tid == 0) {
    //   printf("max_val: %f\n", static_cast<float>(max_val));
    // }

    for (int j = tid; j < C; j += blockDim.x) {
        out[idx * C + j] = expf(__hsub(inp[i + j], offset));
    }
    T sum_val = 0.0f;
    for (int j = tid; j < C; j += blockDim.x) {
        sum_val = __hadd(sum_val, out[i + j]);
    }
    sum_val = warp_reduce_sum(sum_val);
    T sum = __shfl_sync(0xFFFFFFFFFFFFFFFF, sum_val, 0);
    T norm = __hdiv(1.0f, sum);

    // if (idx == 0 && tid == 0) {
    //   printf("sum: %f\n", static_cast<float>(sum));
    //   printf("norm: %f\n", static_cast<float>(norm));
    // }

    for (int j = tid; j < C; j += block_size) {
        out[i + j] = __hmul(out[i + j], norm);
    }
}

template <typename T>
__global__ void sigmoid_kernel(T *out, const T *inp, int N, int C) {
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int i = C * blockIdx.x;
    // 每一个block计算一行

    for (int j = tid; j < C; j += blockDim.x) {
        T input = inp[i + j];
        out[i + j] = (T)(((float)1.0f) / (1.0f + exp(-(float)input)));
    }
}