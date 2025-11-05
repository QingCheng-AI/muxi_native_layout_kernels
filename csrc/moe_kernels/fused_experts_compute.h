#pragma once

#include <optional>

#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "group_gemm_first.h"
#include "group_gemm_second.h"
#include "group_gemm_utils.h"
#include "moe_align_tokens.h"
#include "silu_and_mul.h"
#include "soft_fp8_group_gemm_first.h"
#include "soft_fp8_group_gemm_second.h"

namespace muxi_layout_kernels {

template <int num_experts, int micro_batchsize = 16>
void batched_routed_activation_indexed_to_expert_block_indexed_inner(
    int batchsize, int topK, int *expertsIds, int *dev_sorted_token_ids,
    int *dev_cumsum_buffer, int *dev_padded_num_experts, int *dev_experts_ids) {
    const mcStream_t stream =
        at::cuda::getCurrentCUDAStream(at::cuda::current_device());

    int max_num_tokens_padded =
        ((topK * batchsize) + num_experts * (micro_batchsize - 1));
    int max_num_m_blocks =
        (max_num_tokens_padded + micro_batchsize - 1) / micro_batchsize;

    int block_dim_x =
        num_experts <= 8
            ? 64
            : (num_experts <= 16 ? 128 : (num_experts <= 32 ? 128 : 256));
    int block_dim_x_2 = std::min(256, block_dim_x);
    int num_blocks = (topK * batchsize + block_dim_x_2 - 1) / block_dim_x_2;

    moe_align_tokens_kernel<num_experts, micro_batchsize>
        <<<1, block_dim_x, 0, stream>>>(
            expertsIds, dev_experts_ids, dev_padded_num_experts,
            topK * batchsize, max_num_m_blocks, dev_cumsum_buffer);

    moe_align_tokens_sorted_token_ids_kernel<<<num_blocks, block_dim_x_2, 0,
                                               stream>>>(
        expertsIds, dev_sorted_token_ids, dev_cumsum_buffer, topK * batchsize,
        max_num_tokens_padded);
}

template <int num_experts, int micro_batchsize, typename W, typename Taccum,
          typename A>
void fused_experts_compute_inner(
    W *experts_weights_matrix1, W *experts_weights_matrix2, A *activations,
    int m1, int n1, int k1, int m2, int n2, int k2, int batchsize, int topK,
    int *expertsIds, A *activedExpertsWeights, int *dev_sorted_token_ids,
    int *dev_padded_num_experts, int *dev_experts_ids, A *dev_C, A *y,
    int APerWarp, int splitK, int tile_m_2, int tile_n_2, int tile_k_2,
    int block_dim_x_gemm) {
    const mcStream_t stream =
        at::cuda::getCurrentCUDAStream(at::cuda::current_device());

    int max_num_tokens_padded =
        ((topK * batchsize) + num_experts * (micro_batchsize - 1));
    int max_num_m_blocks =
        (max_num_tokens_padded + micro_batchsize - 1) / micro_batchsize;

    int max_gemm_count = max_num_m_blocks;
    dispatchToStaticInts<1, 2, 4>(APerWarp, [&]<int APERWARP>() {
        dispatchToStaticInts<1, 2, 3>(splitK, [&]<int SPLITK>() {
            dispatchToStaticInts<256, 512>(
                block_dim_x_gemm, [&]<int BLOCK_DIM_X_GEMM>() {
                    int max_gridsize =
                        (m1 / 16 / (256 / WARP_SIZE * APERWARP) * SPLITK) *
                        max_gemm_count;
                    fused_moe_first_group_gemm_kernel<
                        W, A, Taccum, A, BLOCK_DIM_X_GEMM, micro_batchsize,
                        APERWARP, SPLITK>
                        <<<max_gridsize, BLOCK_DIM_X_GEMM, 0, stream>>>(
                            experts_weights_matrix1, activations, dev_C,
                            dev_sorted_token_ids, dev_experts_ids, m1,
                            micro_batchsize, k1, 1.0f, topK, topK * batchsize,
                            dev_padded_num_experts);
                });
        });
    });

    int silu_blockSize = nextPow2_bit(m1 / 2 / ELEMENTSPERACCESS);
    int silu_girdSize = batchsize * topK;

    silu_and_mul_kernel_block<A><<<silu_girdSize, silu_blockSize, 0, stream>>>(
        dev_C, batchsize * topK, m1);

    dispatchToStaticInts<64, 128, 256>(tile_m_2, [&]<int TILE_M>() {
        dispatchToStaticInts<16>(tile_n_2, [&]<int TILE_N>() {
            dispatchToStaticInts<128>(tile_k_2, [&]<int TILE_K>() {
                dispatchToStaticInts<256, 512>(
                    block_dim_x_gemm, [&]<int BLOCK_DIM_X_GEMM>() {
                        int gemm2_gridsize_per_gemm =
                            ((m2 + TILE_M - 1) / TILE_M) *
                            ((micro_batchsize + TILE_N - 1) / TILE_N);
                        fused_moe_second_group_gemm_kernel<
                            W, A, Taccum, A, BLOCK_DIM_X_GEMM, TILE_M, TILE_N,
                            TILE_K, micro_batchsize>
                            <<<gemm2_gridsize_per_gemm * max_gemm_count,
                               BLOCK_DIM_X_GEMM, 0, stream>>>(
                                experts_weights_matrix2, dev_C, y,
                                dev_sorted_token_ids, dev_experts_ids, m2,
                                micro_batchsize, k2, activedExpertsWeights,
                                topK * batchsize, topK, dev_padded_num_experts);
                    });
            });
        });
    });
}

// This is specifically for the case when using soft fp8
template <int num_experts, int micro_batchsize, typename W, typename Taccum,
          typename A>
void fused_experts_compute_inner(
    W *experts_weights_matrix1, W *experts_weights_matrix2, A *activations,
    int m1, int n1, int k1, int m2, int n2, int k2, int batchsize, int topK,
    int *expertsIds, A *activedExpertsWeights, int *dev_sorted_token_ids,
    int *dev_padded_num_experts, int *dev_experts_ids, A *dev_C, A *y,
    Taccum *w1_scale, Taccum *w2_scale, int w1_scale_m, int w1_scale_n,
    int w2_scale_m, int w2_scale_n) {
    const mcStream_t stream =
        at::cuda::getCurrentCUDAStream(at::cuda::current_device());

    int max_num_tokens_padded =
        ((topK * batchsize) + num_experts * (micro_batchsize - 1));
    int max_num_m_blocks =
        (max_num_tokens_padded + micro_batchsize - 1) / micro_batchsize;

    constexpr int APerWarp = 2;
    constexpr int splitK = 3;
    constexpr int tile_m = 128;
    constexpr int tile_n = 16;
    constexpr int tile_k = 128;
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

    int silu_blockSize = nextPow2_bit(m1 / 2 / ELEMENTSPERACCESS);
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

void batched_routed_activation_indexed_to_expert_block_indexed(
    int batchSize, int expertCount, int topK, int microBatchSize,
    torch::Tensor &expertsIds, torch::Tensor &dev_sorted_token_ids,
    torch::Tensor &dev_cumsum_buffer, torch::Tensor &dev_padded_num_experts,
    torch::Tensor &dev_experts_ids);

void fused_experts_compute(
    torch::Tensor &experts_weights_matrix1,
    torch::Tensor &experts_weights_matrix2, torch::Tensor &activations,
    int64_t batchSize, int64_t expertCount, int64_t dynamicExpertsPerAct,
    torch::Tensor &expertsIds, torch::Tensor &activedExpertsWeights,
    torch::Tensor &dev_sorted_token_ids, torch::Tensor &dev_cumsum_buffer,
    torch::Tensor &dev_padded_num_experts, torch::Tensor &dev_experts_ids,
    torch::Tensor &dev_C, torch::Tensor &y, torch::Tensor &w1_scale,
    torch::Tensor &w2_scale, std::vector<int64_t> &block_shape, bool soft_fp8,
    int microBatchSize);

void fused_experts_compute(
    torch::Tensor &experts_weights_matrix1,
    torch::Tensor &experts_weights_matrix2, torch::Tensor &activations,
    int64_t batchSize, int64_t expertCount, int64_t dynamicExpertsPerAct,
    torch::Tensor &expertsIds, torch::Tensor &activedExpertsWeights,
    torch::Tensor &dev_sorted_token_ids, torch::Tensor &dev_cumsum_buffer,
    torch::Tensor &dev_padded_num_experts, torch::Tensor &dev_experts_ids,
    torch::Tensor &dev_C, torch::Tensor &y, int APerWarp, int splitK,
    int tile_m_2, int tile_n_2, int tile_k_2, int block_dim_x_gemm,
    int microBatchSize);

} // namespace muxi_layout_kernels
