#pragma once
#include "group_gemm_first.h"
#include "group_gemm_second.h"
#include "group_gemm_utils.h"
#include "moe_align_tokens.h"
#include "silu_and_mul.h"
#include "soft_fp8_group_gemm_first.h"
#include "soft_fp8_group_gemm_second.h"

template <typename W, typename Taccum, typename A>
void fused_experts_compute(W *experts_weights_matrix1,
                           W *experts_weights_matrix2, A *activations, int m1,
                           int n1, int k1, int m2, int n2, int k2,
                           int batchsize, int num_experts, int topK,
                           int *expertsIds, A *activedExpertsWeights,
                           int *dev_sorted_token_ids, int *dev_cumsum_buffer,
                           int *dev_padded_num_experts, int *dev_experts_ids,
                           A *dev_C, A *y) {
    const mcStream_t stream =
        at::cuda::getCurrentCUDAStream(at::cuda::current_device());
    constexpr int micro_batchsize = 16;

    // first part, align tokens

    int max_num_tokens_padded =
        ((topK * batchsize) + num_experts * (micro_batchsize - 1));
    int max_num_m_blocks =
        (max_num_tokens_padded + micro_batchsize - 1) / micro_batchsize;

    int padded_ptr_number = topK * batchsize;

    int block_dim_x =
        num_experts <= 8
            ? 64
            : (num_experts <= 16 ? 128 : (num_experts <= 32 ? 128 : (256)));
    int block_dim_x_2 = std::min(256, block_dim_x);
    int num_blocks = (topK * batchsize + block_dim_x_2 - 1) / block_dim_x_2;

    if (num_experts == 256) {
        moe_align_tokens_kernel<256, micro_batchsize>
            <<<1, block_dim_x, 0, stream>>>(
                expertsIds, dev_experts_ids, dev_padded_num_experts,
                topK * batchsize, max_num_m_blocks, dev_cumsum_buffer);
    } else {
        assert(false &&
               "Unsupported number of experts, now just support 256 experts");
    }

    moe_align_tokens_sorted_token_ids_kernel<<<num_blocks, block_dim_x_2, 0,
                                               stream>>>(
        expertsIds, dev_sorted_token_ids, dev_cumsum_buffer, topK * batchsize,
        max_num_tokens_padded);

    // second part, group gemms and silu_and_mul

    constexpr int APerWarp = 2; // 2
    constexpr int splitK = 3;   // 3
    constexpr int tile_m = 128; // 128
    constexpr int tile_n = 16;  // 16
    constexpr int tile_k = 128; // 128
    constexpr int block_dim_x_gemm = 256;

    int max_gemm_count = max_num_m_blocks;
    int max_gridsize =
        (m1 / 16 / (block_dim_x_gemm / WARP_SIZE * APerWarp) * splitK) *
        max_gemm_count;
    int gemm2_gridsize_per_gemm = ((m2 + tile_m - 1) / tile_m) *
                                  ((micro_batchsize + tile_n - 1) / tile_n);

    fused_moe_first_group_gemm_kernel<W, A, Taccum, A, block_dim_x_gemm,
                                      micro_batchsize, APerWarp, splitK>
        <<<max_gridsize, block_dim_x_gemm, 0, stream>>>(
            experts_weights_matrix1, activations, dev_C, dev_sorted_token_ids,
            dev_experts_ids, m1, micro_batchsize, k1, 1.0f, topK,
            topK * batchsize, dev_padded_num_experts);

    int silu_blockSize = nextPow2_bit(512 / 2 / ELEMENTSPERACCESS);
    int silu_girdSize = batchsize * topK;

    silu_and_mul_kernel_block<A><<<silu_girdSize, silu_blockSize, 0, stream>>>(
        dev_C, batchsize * topK, m1);

    fused_moe_second_group_gemm_kernel<W, A, Taccum, A, block_dim_x_gemm,
                                       tile_m, tile_n, tile_k, micro_batchsize>
        <<<gemm2_gridsize_per_gemm * max_gemm_count, block_dim_x_gemm, 0,
           stream>>>(experts_weights_matrix2, dev_C, y, dev_sorted_token_ids,
                     dev_experts_ids, m2, micro_batchsize, k2,
                     activedExpertsWeights, topK * batchsize, topK,
                     dev_padded_num_experts);
}

// This is specifically for the case when using soft fp8
template <typename W, typename Taccum, typename A>
void fused_experts_compute(W *experts_weights_matrix1,
                           W *experts_weights_matrix2, A *activations, int m1,
                           int n1, int k1, int m2, int n2, int k2,
                           int batchsize, int num_experts, int topK,
                           int *expertsIds, A *activedExpertsWeights,
                           int *dev_sorted_token_ids, int *dev_cumsum_buffer,
                           int *dev_padded_num_experts, int *dev_experts_ids,
                           A *dev_C, A *y, Taccum *w1_scale, Taccum *w2_scale,
                           int w1_scale_m, int w1_scale_n, int w2_scale_m,
                           int w2_scale_n) {
    const mcStream_t stream =
        at::cuda::getCurrentCUDAStream(at::cuda::current_device());
    constexpr int micro_batchsize = 16;

    // first part, align tokens

    int max_num_tokens_padded =
        ((topK * batchsize) + num_experts * (micro_batchsize - 1));
    int max_num_m_blocks =
        (max_num_tokens_padded + micro_batchsize - 1) / micro_batchsize;

    int padded_ptr_number = topK * batchsize;

    int block_dim_x =
        num_experts <= 8
            ? 64
            : (num_experts <= 16 ? 128 : (num_experts <= 32 ? 128 : (256)));
    int block_dim_x_2 = std::min(256, block_dim_x);
    int num_blocks = (topK * batchsize + block_dim_x_2 - 1) / block_dim_x_2;

    if (num_experts == 256) {
        moe_align_tokens_kernel<256, micro_batchsize>
            <<<1, block_dim_x, 0, stream>>>(
                expertsIds, dev_experts_ids, dev_padded_num_experts,
                topK * batchsize, max_num_m_blocks, dev_cumsum_buffer);
    } else {
        assert(false &&
               "Unsupported number of experts, now just support 256 experts");
    }

    moe_align_tokens_sorted_token_ids_kernel<<<num_blocks, block_dim_x_2, 0,
                                               stream>>>(
        expertsIds, dev_sorted_token_ids, dev_cumsum_buffer, topK * batchsize,
        max_num_tokens_padded);

    // second part, group gemms and silu_and_mul

    constexpr int APerWarp = 2; // 2
    constexpr int splitK = 3;   // 3
    constexpr int tile_m = 128; // 128
    constexpr int tile_n = 16;  // 16
    constexpr int tile_k = 128; // 128
    constexpr int block_dim_x_gemm = 256;

    int max_gemm_count = max_num_m_blocks;
    int max_gridsize =
        (m1 / 16 / (block_dim_x_gemm / WARP_SIZE * APerWarp) * splitK) *
        max_gemm_count;
    int gemm2_gridsize_per_gemm = ((m2 + tile_m - 1) / tile_m) *
                                  ((micro_batchsize + tile_n - 1) / tile_n);

    fused_moe_first_group_gemm_kernel<W, A, Taccum, A, block_dim_x_gemm,
                                      micro_batchsize, APerWarp, splitK, 128,
                                      128>
        <<<max_gridsize, block_dim_x_gemm, 0, stream>>>(
            experts_weights_matrix1, activations, dev_C, dev_sorted_token_ids,
            dev_experts_ids, m1, micro_batchsize, k1, 1.0f, topK,
            topK * batchsize, dev_padded_num_experts, w1_scale, w1_scale_m,
            w1_scale_n);

    int silu_blockSize = nextPow2_bit(512 / 2 / ELEMENTSPERACCESS);
    int silu_girdSize = batchsize * topK;

    silu_and_mul_kernel_block<A><<<silu_girdSize, silu_blockSize, 0, stream>>>(
        dev_C, batchsize * topK, m1);

    fused_moe_second_group_gemm_kernel<W, A, Taccum, A, block_dim_x_gemm,
                                       tile_m, tile_n, tile_k, micro_batchsize,
                                       128, 128>
        <<<gemm2_gridsize_per_gemm * max_gemm_count, block_dim_x_gemm, 0,
           stream>>>(experts_weights_matrix2, dev_C, y, dev_sorted_token_ids,
                     dev_experts_ids, m2, micro_batchsize, k2,
                     activedExpertsWeights, topK * batchsize, topK,
                     dev_padded_num_experts, w2_scale, w2_scale_m, w2_scale_n);
}