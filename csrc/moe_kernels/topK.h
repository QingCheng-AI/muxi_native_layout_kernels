#pragma once
#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>

#include "group_gemm_utils.h"

// CUDA kernel 实现分组 topK

template <typename T, int topK>
__device__ __forceinline__ void insert_value(T *value, int *index, T data,
                                             int data_index) {
    // if (data < value[topK - 1]) {
    if (__hlt(data, value[topK - 1])) {
        return;
    }
    for (int i = topK - 2; i >= 0; i--) {
        // if (data > value[i]) {
        if (__hgt(data, value[i])) {
            value[i + 1] = value[i];
            index[i + 1] = index[i];
        } else {
            value[i + 1] = data;
            index[i + 1] = data_index;
            return;
        }
    }

    value[0] = data;
    index[0] = data_index;
}

template <typename T, int topK, int BLOCK_SIZE>
__global__ void grouped_topK_kernel(const T *input, int *topK_indices,
                                    T *topK_values, int batch_size,
                                    int num_experts, int n_groups,
                                    int topK_groups, int *topK_group_index) {
    int bid = blockIdx.x;
    int block_offset = bid * num_experts;

    if (bid >= batch_size)
        return;

    __shared__ T shared_value[BLOCK_SIZE * topK];
    __shared__ int shared_index[BLOCK_SIZE * topK];
    T top_value[topK];
    int top_index[topK];

#pragma unroll
    for (int i = 0; i < topK; i++) {
        top_value[i] = -1.0;
        top_index[i] = -1;
    }

    for (int idx = threadIdx.x; idx < num_experts; idx += BLOCK_SIZE) {
        T value = 0.0f;
        if (topK_group_index != nullptr) {
            for (int i = 0; i < topK_groups; i++) {
                if (idx / (num_experts / n_groups) == topK_group_index[i]) {
                    value = input[block_offset + idx];
                    break;
                }
            }
        } else {
            value = input[block_offset + idx];
        }
        insert_value<T, topK>(top_value, top_index, value, idx);
    }

#pragma unroll
    for (int i = 0; i < topK; i++) {
        shared_value[topK * threadIdx.x + i] = top_value[i];
        shared_index[topK * threadIdx.x + i] = top_index[i];
    }
    __syncthreads();

#pragma unroll
    for (int i = BLOCK_SIZE / 2; i >= 1; i /= 2) {
        if (threadIdx.x < i) {
#pragma unroll
            for (int m = 0; m < topK; m++) {
                insert_value<T, topK>(
                    top_value, top_index,
                    shared_value[topK * (threadIdx.x + i) + m],
                    shared_index[topK * (threadIdx.x + i) + m]);
            }
        }
        __syncthreads();
        if (threadIdx.x < i) {
#pragma unroll
            for (int m = 0; m < topK; m++) {
                shared_value[topK * threadIdx.x + m] = top_value[m];
                shared_index[topK * threadIdx.x + m] = top_index[m];
            }
        }
        __syncthreads();
    }

    // sort and store
    if (blockIdx.x < batch_size) {
        if (threadIdx.x == 0) {
            for (int i = 0; i < topK - 1; ++i) {
                for (int j = 0; j < topK - 1 - i; ++j) {
                    if (shared_index[j] > shared_index[j + 1]) {
                        int temp = shared_index[j];
                        shared_index[j] = shared_index[j + 1];
                        shared_index[j + 1] = temp;
                        T temp_value = shared_value[j];
                        shared_value[j] = shared_value[j + 1];
                        shared_value[j + 1] = temp_value;
                    }
                }
            }

#pragma unroll
            for (int i = 0; i < topK; i++) {
                topK_values[topK * blockIdx.x + i] = shared_value[i];
                topK_indices[topK * blockIdx.x + i] = shared_index[i];
            }
        }
    }
}

template <typename T, int topK>
__device__ void insert_value_bias(T *value, int *index, T *value_bias, T data,
                                  int data_index, T data_bias) {
    // if (data_bias < value_bias[topK - 1]) {
    if (__hlt(data_bias, value_bias[topK - 1])) {
        return;
    }
    for (int i = topK - 2; i >= 0; i--) {
        // if (data_bias > value_bias[i]) {
        if (__hgt(data_bias, value_bias[i])) {
            value[i + 1] = value[i];
            index[i + 1] = index[i];
            value_bias[i + 1] = value_bias[i];
        } else {
            value[i + 1] = data;
            index[i + 1] = data_index;
            value_bias[i + 1] = data_bias;
            return;
        }
    }

    value[0] = data;
    index[0] = data_index;
    value_bias[0] = data_bias;
}

template <typename T, int topK, int BLOCK_SIZE>
__global__ void
grouped_topK_bias_kernel(const T *input, int *topK_indices, T *topK_values,
                         int batch_size, int num_experts, T *bias, int n_groups,
                         int topK_groups, int *topK_group_index) {
    int bid = blockIdx.x;
    int block_offset = bid * num_experts;

    __shared__ T shared_value[BLOCK_SIZE * topK];
    __shared__ int shared_index[BLOCK_SIZE * topK];
    __shared__ T shared_value_bias[BLOCK_SIZE * topK];
    T top_value[topK];
    int top_index[topK];
    T top_value_bias[topK];

    for (int i = 0; i < topK; i++) {
        top_value[i] = -1.0;
        top_index[i] = -1;
        top_value_bias[i] = -1.0;
    }

    for (int idx = threadIdx.x; idx < num_experts; idx += BLOCK_SIZE) {
        T value = 0.0f;
        T value_bias = 0.0f;
        if (topK_group_index != nullptr) {
            for (int i = 0; i < topK_groups; i++) {
                if (idx / (num_experts / n_groups) == topK_group_index[i]) {
                    value = input[block_offset + idx];
                    value_bias = __hadd(value, bias[idx]);
                    break;
                }
            }
        } else {
            value = input[block_offset + idx];
            value_bias = __hadd(value, bias[idx]);
        }
        insert_value_bias<T, topK>(top_value, top_index, top_value_bias, value,
                                   idx, value_bias);
    }
    for (int i = 0; i < topK; i++) {
        shared_value[topK * threadIdx.x + i] = top_value[i];
        shared_index[topK * threadIdx.x + i] = top_index[i];
        shared_value_bias[topK * threadIdx.x + i] = top_value_bias[i];
    }
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i >= 1; i /= 2) {
        if (threadIdx.x < i) {
            for (int m = 0; m < topK; m++) {
                insert_value_bias<T, topK>(
                    top_value, top_index, top_value_bias,
                    shared_value[topK * (threadIdx.x + i) + m],
                    shared_index[topK * (threadIdx.x + i) + m],
                    shared_value_bias[topK * (threadIdx.x + i) + m]);
            }
        }
        __syncthreads();
        if (threadIdx.x < i) {
            for (int m = 0; m < topK; m++) {
                shared_value[topK * threadIdx.x + m] = top_value[m];
                shared_index[topK * threadIdx.x + m] = top_index[m];
                shared_value_bias[topK * threadIdx.x + m] = top_value_bias[m];
            }
        }
        __syncthreads();
    }

    // sort and store
    if (blockIdx.x < batch_size) {
        if (threadIdx.x == 0) {
            for (int i = 0; i < topK - 1; ++i) {
                for (int j = 0; j < topK - 1 - i; ++j) {
                    if (shared_index[j] > shared_index[j + 1]) {
                        int temp = shared_index[j];
                        shared_index[j] = shared_index[j + 1];
                        shared_index[j + 1] = temp;
                        T temp_value = shared_value[j];
                        shared_value[j] = shared_value[j + 1];
                        shared_value[j + 1] = temp_value;
                    }
                }
            }

            for (int i = 0; i < topK; i++) {
                topK_values[topK * blockIdx.x + i] = shared_value[i];
                topK_indices[topK * blockIdx.x + i] = shared_index[i];
            }
        }
    }
}

template <typename T, int topK>
__device__ void insert_value_bias_without_index(T *value_bias, T data_bias) {
    // if (data_bias < value_bias[topK - 1]) {
    if (__hlt(data_bias, value_bias[topK - 1])) {
        return;
    }
    for (int i = topK - 2; i >= 0; i--) {
        // if (data_bias > value_bias[i]) {
        if (__hgt(data_bias, value_bias[i])) {
            value_bias[i + 1] = value_bias[i];
        } else {
            value_bias[i + 1] = data_bias;
            return;
        }
    }

    value_bias[0] = data_bias;
}

template <typename T, int topK, int BLOCK_SIZE>
__global__ void grouped_topK_bias_sum_kernel(const T *input, T *topK_values,
                                             int batch_size, int num_experts,
                                             T *bias = nullptr) {
    int bid = blockIdx.x;
    int block_offset = bid * num_experts;

    __shared__ T shared_value_bias[BLOCK_SIZE * topK];
    T top_value_bias[topK];

    for (int i = 0; i < topK; i++) {
        top_value_bias[i] = -1.0;
    }

    for (int idx = threadIdx.x; idx < num_experts; idx += BLOCK_SIZE) {
        insert_value_bias_without_index<T, topK>(
            top_value_bias, __hadd(input[block_offset + idx],
                                   (bias == nullptr ? ((T)(0.0)) : bias[idx])));
    }
    for (int i = 0; i < topK; i++) {
        shared_value_bias[topK * threadIdx.x + i] = top_value_bias[i];
    }
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i >= 1; i /= 2) {
        if (threadIdx.x < i) {
            for (int m = 0; m < topK; m++) {
                insert_value_bias_without_index<T, topK>(
                    top_value_bias,
                    shared_value_bias[topK * (threadIdx.x + i) + m]);
            }
        }
        __syncthreads();
        if (threadIdx.x < i) {
            for (int m = 0; m < topK; m++) {
                shared_value_bias[topK * threadIdx.x + m] = top_value_bias[m];
            }
        }
        __syncthreads();
    }

    // store group sum
    if (blockIdx.x < batch_size) {
        if (threadIdx.x == 0) {
            T sum = 0.0;
            for (int i = 0; i < topK; i++) {
                // sum = sum + shared_value_bias[i];
                sum = __hadd(sum, shared_value_bias[i]);
            }
            topK_values[blockIdx.x] = sum;
        }
    }
}

template <typename T, int BLOCK_SIZE>
void grouped_topK(const T *input, int *topK_indices, T *topK_values,
                  int batch_size, int num_experts, int topK, int n_groups,
                  int topK_groups, int *topK_group_index = nullptr) {
#define TOPK(K)                                                                \
    grouped_topK_kernel<T, K, BLOCK_SIZE>                                      \
        <<<batch_size, BLOCK_SIZE, 0, stream>>>(                               \
            input, topK_indices, topK_values, batch_size, num_experts,         \
            n_groups, topK_groups, topK_group_index);

    const mcStream_t stream =
        at::cuda::getCurrentCUDAStream(at::cuda::current_device());

    if (topK == 1) {
        TOPK(1);
    } else if (topK == 2) {
        TOPK(2);
    } else if (topK == 3) {
        TOPK(3);
    } else if (topK == 4) {
        TOPK(4);
    } else if (topK == 5) {
        TOPK(5);
    } else if (topK == 6) {
        TOPK(6);
    } else if (topK == 7) {
        TOPK(7);
    } else if (topK == 8) {
        TOPK(8);
    } else {
        std::cerr << "TopK value not supported. Supported values are 1-8 now."
                  << std::endl;
        exit(1);
    }

#undef TOPK
}

template <typename T, int BLOCK_SIZE>
void grouped_topK_bias(const T *input, int *topK_indices, T *topK_values,
                       int batch_size, int num_experts, T *bias, int topK,
                       int n_groups, int topK_groups,
                       int *topK_group_index = nullptr) {
#define TOPK_BIAS(K)                                                           \
    grouped_topK_bias_kernel<T, K, BLOCK_SIZE>                                 \
        <<<batch_size, BLOCK_SIZE, 0, stream>>>(                               \
            input, topK_indices, topK_values, batch_size, num_experts, bias,   \
            n_groups, topK_groups, topK_group_index);

    const mcStream_t stream =
        at::cuda::getCurrentCUDAStream(at::cuda::current_device());

    if (topK == 1) {
        TOPK_BIAS(1);
    } else if (topK == 2) {
        TOPK_BIAS(2);
    } else if (topK == 3) {
        TOPK_BIAS(3);
    } else if (topK == 4) {
        TOPK_BIAS(4);
    } else if (topK == 5) {
        TOPK_BIAS(5);
    } else if (topK == 6) {
        TOPK_BIAS(6);
    } else if (topK == 7) {
        TOPK_BIAS(7);
    } else if (topK == 8) {
        TOPK_BIAS(8);
    } else {
        std::cerr << "TopK value not supported. Supported values are 1-8 now."
                  << std::endl;
        exit(1);
    }

#undef TOPK_BIAS
}

template <typename T, int BLOCK_SIZE>
void grouped_topK_bias_sum(const T *input, T *topK_values, int batch_size,
                           int num_experts, T *bias, int topK) {
#define TOPK_BIAS_SUM(K)                                                       \
    grouped_topK_bias_sum_kernel<T, K, BLOCK_SIZE>                             \
        <<<batch_size, BLOCK_SIZE, 0, stream>>>(                               \
            input, topK_values, batch_size, num_experts, bias);

    const mcStream_t stream =
        at::cuda::getCurrentCUDAStream(at::cuda::current_device());

    if (topK == 1) {
        TOPK_BIAS_SUM(1);
    } else if (topK == 2) {
        TOPK_BIAS_SUM(2);
    } else if (topK == 3) {
        TOPK_BIAS_SUM(3);
    } else if (topK == 4) {
        TOPK_BIAS_SUM(4);
    } else if (topK == 5) {
        TOPK_BIAS_SUM(5);
    } else if (topK == 6) {
        TOPK_BIAS_SUM(6);
    } else if (topK == 7) {
        TOPK_BIAS_SUM(7);
    } else if (topK == 8) {
        TOPK_BIAS_SUM(8);
    } else {
        std::cerr << "TopK value not supported. Supported values are 1-8 now."
                  << std::endl;
        exit(1);
    }

#undef TOPK_BIAS_SUM
}
