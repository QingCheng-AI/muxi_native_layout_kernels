#pragma once
#include "group_gemm_utils.h"

// moe_align_blockSize kernel, compute comsum buffer for sorted_token_ids
template <int num_experts, int micro_batchsize>
__global__ void moe_align_tokens_kernel(
    const int *__restrict__ topk_ids, int *__restrict__ experts_ids,
    int *__restrict__ dev_padded_num_experts, int topk_ids_numel,
    int max_num_m_blocks, int *__restrict__ cumsum_buffer) {
    constexpr int padded_num_experts =
        (num_experts + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
    constexpr int experts_per_warp =
        num_experts <= 8
            ? 8
            : (num_experts <= 16 ? 16 : (num_experts <= 32 ? 32 : WARP_SIZE));
    constexpr int num_warps =
        (padded_num_experts + experts_per_warp - 1) / experts_per_warp;
    __shared__ int shared_counts[num_warps * experts_per_warp];

    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    const int warp_experts_start = warp_id * experts_per_warp;

    // init shared mem counter with zero
    if (warp_experts_start + lane_id < num_warps * experts_per_warp) {
        shared_counts[warp_experts_start + lane_id] = 0;
    }
    // init experts_ids with zero
    for (int i = threadIdx.x; i < max_num_m_blocks; i += blockDim.x) {
        experts_ids[i] = 0;
    }

    __syncthreads();

    const int tokens_per_thread =
        (topk_ids_numel + blockDim.x - 1) / blockDim.x;
    const int start_idx = threadIdx.x * tokens_per_thread;

    // calculate tokens count for each expert
    for (int i = start_idx;
         i < topk_ids_numel && i < start_idx + tokens_per_thread; i++) {
        int expert_id = topk_ids[i];
        atomicAdd(&shared_counts[expert_id], 1);
    }

    __syncthreads();

    // calculate cumsum
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        int expert_token_count = shared_counts[i];
        shared_counts[i] =
            ((expert_token_count + micro_batchsize - 1) / micro_batchsize) *
            micro_batchsize;
    }

    __syncthreads();

    // prefix sum shared_counts
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int t = 0;
        if (threadIdx.x < num_experts && threadIdx.x >= offset) {
            t = shared_counts[threadIdx.x - offset];
        }
        __syncthreads();
        if (threadIdx.x < num_experts && threadIdx.x >= offset) {
            shared_counts[threadIdx.x] += t;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *dev_padded_num_experts =
            shared_counts[num_experts - 1] / micro_batchsize;
        cumsum_buffer[0] = 0;
    }

    // write shared_counts to cumsum_buffer
    if (threadIdx.x < num_experts) {
        cumsum_buffer[threadIdx.x + 1] = shared_counts[threadIdx.x];
        int start_idx = threadIdx.x > 0 ? shared_counts[threadIdx.x - 1] : 0;
        int end_idx = shared_counts[threadIdx.x];
        for (int i = start_idx; i < end_idx; i += micro_batchsize) {
            experts_ids[i / micro_batchsize] = threadIdx.x;
        }
    }
}

// moe_align_blockSize kernel, compute sorted_token_ids
__global__ void moe_align_tokens_sorted_token_ids_kernel(
    const int *__restrict__ topk_ids, int *__restrict__ sorted_token_ids,
    int *__restrict__ cumsum_buffer, int topk_ids_numel,
    int max_num_tokens_padded) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // for relaxed sorted token ids
    for (int i = tid; i < topk_ids_numel; i += stride) {
        int expert_id = topk_ids[i];
        int rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
        sorted_token_ids[rank_post_pad] = i;
    }
}
