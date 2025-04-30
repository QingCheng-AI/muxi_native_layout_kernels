#include "group_gemm_first.h"
#include "group_gemm_second.h"
#include "moe_align_tokens.h"
#include "silu_and_mul.h"

int main() {
    using TA = __half;
    using TB = __half;
    using TC = __half;

    constexpr int block_dim_x_gemm = 256;
    constexpr int APerWarp = 2;
    constexpr int splitK = 2;
    constexpr int micro_batchsize = 16;
    constexpr int topk = 8;
    constexpr int num_experts = 256;
    constexpr int tile_m = 128;
    constexpr int tile_n = 64;
    constexpr int tile_k = 128;

    const int batchsize = 8;

    const int m = 512;
    const int n = batchsize;
    const int k = 7168;

    // dim3 gridsize_gemm2((k + tile_m - 1) / tile_m,
    //                     (micro_batchsize + tile_n - 1) / tile_n);
    int gemm2_gridsize_per_gemm =
        ((k + tile_m - 1) / tile_m) * ((micro_batchsize + tile_n - 1) / tile_n);

    int padded_ptr_number = topk * batchsize;
    // int sorted_token_ids[] = {9,  3,  16, 40, 56, 64, 64, 64,
    //                           64, 64, 64, 64, 64, 64, 64, 64};
    // int sorted_token_ids[] = {20, 45, 38, 61, 64, 64, 64, 64,
    //                           64, 64, 64, 64, 64, 64, 64, 64};

    int topk_ids[] = {
        1,  19, 24, 45, 57,  78,  98,  231, 0,  45, 56, 57, 89, 90,  210, 255,
        45, 57, 68, 79, 100, 145, 178, 240, 12, 41, 53, 74, 78, 123, 157, 167,
        31, 42, 54, 65, 77,  89,  100, 121, 45, 56, 67, 78, 89, 100, 111, 122,
        32, 43, 55, 66, 76,  88,  99,  110, 45, 56, 67, 78, 89, 100, 111, 122};

    int max_num_tokens_padded =
        ((topk * batchsize) + num_experts * (batchsize - 1));
    int max_num_m_blocks =
        (max_num_tokens_padded + micro_batchsize - 1) / micro_batchsize;

    TA *A = (TA *)malloc(sizeof(TA) * m * k * num_experts);
    TB *B = (TB *)malloc(sizeof(TB) * n * k);
    initializeHostData(A, num_experts * m, k);
    initializeHostData(B, n, k);
    TC *C = (TC *)malloc(sizeof(TC) * topk * m * batchsize);
    TC *C_silu = (TC *)malloc(sizeof(TC) * topk * m * batchsize);
    TC *routing_weights = (TC *)malloc(sizeof(TC) * topk * batchsize);
    initializeHostData(routing_weights, topk, batchsize);
    TC *result = (TC *)malloc(sizeof(TC) * k * batchsize);

    int *dev_sorted_token_ids;
    mcMalloc((void **)&dev_sorted_token_ids,
             sizeof(int) * max_num_tokens_padded);
    int *dev_cumsum_buffer;
    mcMalloc((void **)&dev_cumsum_buffer, sizeof(int) * (num_experts + 1));
    int *dev_topk_ids;
    mcMalloc((void **)&dev_topk_ids, sizeof(int) * topk * batchsize);
    mcMemcpy(dev_topk_ids, topk_ids, sizeof(int) * topk * batchsize,
             mcMemcpyHostToDevice);
    TC *dev_routing_weights;
    mcMalloc((void **)&dev_routing_weights, sizeof(TC) * topk * batchsize);
    mcMemcpy(dev_routing_weights, routing_weights,
             sizeof(TC) * topk * batchsize, mcMemcpyHostToDevice);
    int *dev_experts_ids;
    mcMalloc((void **)&dev_experts_ids, sizeof(int) * max_num_m_blocks);
    int *dev_padded_num_experts;
    mcMalloc((void **)&dev_padded_num_experts, sizeof(int) * 1);
    // mcMemcpy(dev_sorted_token_ids, sorted_token_ids,
    //          sizeof(int) * micro_batchsize, mcMemcpyHostToDevice);
    TA *dev_A;
    mcMalloc((void **)&dev_A, sizeof(TA) * m * k * num_experts);
    mcMemcpy((void *)dev_A, A, sizeof(TA) * m * k * num_experts,
             mcMemcpyHostToDevice);
    TB *dev_B;
    mcMalloc((void **)&dev_B, sizeof(TB) * n * k);
    mcMemcpy((void *)dev_B, B, sizeof(TB) * n * k, mcMemcpyHostToDevice);
    TC *dev_C;
    mcMalloc((void **)&dev_C, sizeof(TC) * topk * m * batchsize);
    // TC *dev_result;
    // mcMalloc((void **)&dev_result, sizeof(TC) * k * batchsize);

    mcMemset(dev_C, 0, sizeof(TC) * topk * m * batchsize);

    {
        // first part, align tokens
        int block_dim_x =
            num_experts <= 8
                ? 64
                : (num_experts <= 16 ? 128 : (num_experts <= 32 ? 128 : (256)));
        int block_dim_x_2 = std::min(256, block_dim_x);
        int num_blocks = (topk * batchsize + block_dim_x_2 - 1) / block_dim_x_2;

        moe_align_tokens_kernel<num_experts, micro_batchsize>
            <<<1, block_dim_x>>>(dev_topk_ids, dev_experts_ids,
                                 dev_padded_num_experts, topk * batchsize,
                                 max_num_m_blocks, dev_cumsum_buffer);

        moe_align_tokens_sorted_token_ids_kernel<<<num_blocks, block_dim_x_2>>>(
            dev_topk_ids, dev_sorted_token_ids, dev_cumsum_buffer,
            topk * batchsize, max_num_tokens_padded);
    }

    int max_gemm_count = max_num_m_blocks;
    int max_gridsize =
        (m / 16 / (block_dim_x_gemm / WARP_SIZE * APerWarp) * splitK) *
        max_gemm_count;
    fused_moe_first_group_gemm_kernel<TA, TB, float, TC, block_dim_x_gemm,
                                      micro_batchsize, APerWarp, splitK>
        <<<max_gridsize, block_dim_x_gemm>>>(
            dev_A, dev_B, dev_C, dev_sorted_token_ids, dev_experts_ids, m,
            micro_batchsize, k, 1.0f, topk, topk * batchsize,
            dev_padded_num_experts);

    mcMemcpy(C, dev_C, sizeof(TC) * topk * m * batchsize, mcMemcpyDeviceToHost);

    int silu_girdsize = (batchsize * topk + WARP_SIZE - 1) / WARP_SIZE;
    int silu_blocksize = WARP_SIZE;

    silu_and_mul_kernel<TB>
        <<<silu_girdsize, silu_blocksize>>>(dev_C, batchsize * topk, m);

    mcMemcpy(C_silu, dev_C, sizeof(TC) * topk * m * batchsize,
             mcMemcpyDeviceToHost);

    mcMemset(dev_B, 0, sizeof(TB) * batchsize * k);

    // fused_moe_second_gemm_kernel<TA, TB, float, TC, block_dim_x_gemm, tile_m,
    //                              tile_n, tile_k, micro_batchsize>
    //     <<<gridsize_gemm2, block_dim_x_gemm>>>(
    //         dev_A, dev_C, dev_B, dev_sorted_token_ids, k, micro_batchsize,
    //         256, dev_routing_weights, topk * batchsize, topk);

    fused_moe_second_group_gemm_kernel<TA, TB, float, TC, block_dim_x_gemm,
                                       tile_m, tile_n, tile_k, micro_batchsize>
        <<<gemm2_gridsize_per_gemm * max_gemm_count, block_dim_x_gemm>>>(
            dev_A, dev_C, dev_B, dev_sorted_token_ids, dev_experts_ids, k,
            micro_batchsize, 256, dev_routing_weights, topk * batchsize, topk,
            dev_padded_num_experts);

    mcMemcpy(result, dev_B, sizeof(TB) * batchsize * k, mcMemcpyDeviceToHost);

    // Print the result non-zero elements
    float tmp = 0.0f;
    int non_zero_elements = 0;
    for (int i = 0; i < topk * m * batchsize; i++) {
        if (static_cast<float>(C[i]) != tmp) {
            std::cout << "C[" << i << "] = " << static_cast<float>(C[i])
                      << std::endl;
            std::cout << "repeated token id: " << i / m << std::endl;
            tmp = static_cast<float>(C[i]);
        }
        if (static_cast<float>(C[i]) != 0.0f) {
            non_zero_elements++;
        }
    }
    std::cout << "Non-zero elements: " << non_zero_elements << std::endl;

    // Print the result non-zero elements
    tmp = 0.0f;
    non_zero_elements = 0;
    for (int i = 0; i < topk * m * batchsize; i++) {
        if (static_cast<float>(C_silu[i]) != tmp) {
            std::cout << "C_silu[" << i
                      << "] = " << static_cast<float>(C_silu[i]) << std::endl;
            std::cout << "repeated token id: " << i / m << std::endl;
            tmp = static_cast<float>(C_silu[i]);
        }
        if (static_cast<float>(C_silu[i]) != 0.0f) {
            non_zero_elements++;
        }
    }
    std::cout << "Non-zero elements (silu applied): " << non_zero_elements
              << std::endl;

    // Print the result non-zero elements
    tmp = 0.0f;
    non_zero_elements = 0;
    for (int i = 0; i < topk * m * batchsize; i++) {
        if (static_cast<float>(result[i]) != tmp) {
            std::cout << "result[" << i
                      << "] = " << static_cast<float>(result[i]) << std::endl;
            std::cout << "repeated token id: " << i / k << std::endl;
            tmp = static_cast<float>(result[i]);
        }
        if (static_cast<float>(result[i]) != 0.0f) {
            non_zero_elements++;
        }
    }
    std::cout << "Non-zero elements (second gemm): " << non_zero_elements
              << std::endl;

    return 0;
}