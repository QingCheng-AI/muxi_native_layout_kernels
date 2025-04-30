#include <iostream>

#include "moe_align_tokens.h"

int main() {
    constexpr int num_experts = 256;
    constexpr int batch_size = 8;
    constexpr int micro_batchsize = 16;
    constexpr int topk = 8;
    // int topk_ids[] = {2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3};
    // int topk_ids[] = {1, 2, 3, 0, 1, 3, 0, 2, 3, 0, 1, 2};
    int topk_ids[] = {
        1,  19, 24, 45, 57,  78,  98,  231, 0,  45, 56, 57, 89, 90,  210, 255,
        45, 57, 68, 79, 100, 145, 178, 240, 12, 41, 53, 74, 78, 123, 157, 167,
        31, 42, 54, 65, 77,  89,  100, 121, 45, 56, 67, 78, 89, 100, 111, 122,
        32, 43, 55, 66, 76,  88,  99,  110, 45, 56, 67, 78, 89, 100, 111, 122};

    int max_num_tokens_padded =
        ((topk * batch_size) + num_experts * (batch_size - 1));
    int max_num_m_blocks =
        (max_num_tokens_padded + micro_batchsize - 1) / micro_batchsize;
    int *sorted_token_ids = (int *)malloc(sizeof(int) * max_num_tokens_padded);
    int *cumsum_buffer = (int *)malloc(sizeof(int) * (num_experts + 1));
    int *experts_ids = (int *)malloc(sizeof(int) * max_num_m_blocks);

    int *dev_topk_ids;
    int *dev_sorted_token_ids;
    int *dev_experts_ids;
    int *dev_padded_num_experts;
    int *dev_cumsum_buffer;
    mcMalloc((void **)&dev_topk_ids, sizeof(int) * topk * batch_size);
    mcMalloc((void **)&dev_sorted_token_ids,
             sizeof(int) * max_num_tokens_padded);
    mcMalloc((void **)&dev_experts_ids, sizeof(int) * max_num_m_blocks);
    mcMalloc((void **)&dev_padded_num_experts, sizeof(int) * 1);
    mcMalloc((void **)&dev_cumsum_buffer, sizeof(int) * (num_experts + 1));

    mcMemcpy(dev_topk_ids, topk_ids, sizeof(int) * topk * batch_size,
             mcMemcpyHostToDevice);

    int block_dim_x =
        num_experts <= 8
            ? 64
            : (num_experts <= 16 ? 128 : (num_experts <= 32 ? 128 : (256)));
    int block_dim_x_2 = std::min(256, block_dim_x);
    int num_blocks = (topk * batch_size + block_dim_x_2 - 1) / block_dim_x_2;

    moe_align_tokens_kernel<num_experts, micro_batchsize><<<1, block_dim_x>>>(
        dev_topk_ids, dev_experts_ids, dev_padded_num_experts,
        topk * batch_size, max_num_m_blocks, dev_cumsum_buffer);

    mcMemcpy(cumsum_buffer, dev_cumsum_buffer, sizeof(int) * (num_experts + 1),
             mcMemcpyDeviceToHost);

    moe_align_tokens_sorted_token_ids_kernel<<<num_blocks, block_dim_x_2>>>(
        dev_topk_ids, dev_sorted_token_ids, dev_cumsum_buffer,
        topk * batch_size, max_num_tokens_padded);

    mcMemcpy(sorted_token_ids, dev_sorted_token_ids,
             sizeof(int) * max_num_tokens_padded, mcMemcpyDeviceToHost);
    // mcMemcpy(cumsum_buffer, dev_cumsum_buffer, sizeof(int) * (num_experts +
    // 1),
    //          mcMemcpyDeviceToHost);
    mcMemcpy(experts_ids, dev_experts_ids, sizeof(int) * max_num_m_blocks,
             mcMemcpyDeviceToHost);

    // Print topk_ids
    std::cout << "Topk ids:" << std::endl;
    for (int i = 0; i < topk * batch_size; i++) {
        std::cout << topk_ids[i] << " ";
        if ((i + 1) % topk == 0) {
            std::cout << std::endl;
        }
    }

    // Print sorted_token_ids
    std::cout << "Sorted token ids:" << std::endl;
    for (int i = 0; i < max_num_tokens_padded; i++) {
        std::cout << sorted_token_ids[i] << " ";
        if ((i + 1) % micro_batchsize == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    // Print cumsum_buffer
    std::cout << "Cumsum buffer:" << std::endl;
    for (int i = 0; i < num_experts + 1; i++) {
        std::cout << cumsum_buffer[i] << " ";
        if ((i + 1) % (16) == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    // Print experts_ids
    std::cout << "Experts Ids:" << std::endl;
    for (int i = 0; i < max_num_m_blocks; i++) {
        std::cout << experts_ids[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}