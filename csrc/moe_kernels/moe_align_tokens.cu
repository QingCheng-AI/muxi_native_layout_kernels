#include "moe_align_tokens.h"

namespace muxi_layout_kernels {

__global__ void moe_align_tokens_sorted_token_ids_kernel(
    const int *__restrict__ topk_ids, int *__restrict__ sorted_token_ids,
    int *experts_map, int *__restrict__ cumsum_buffer, int topk_ids_numel,
    int max_num_tokens_padded) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // for relaxed sorted token ids
    for (int i = tid; i < topk_ids_numel; i += stride) {
        int expert_id = topk_ids[i];
        expert_id = experts_map != nullptr ? experts_map[expert_id] : expert_id;
        if (expert_id == -1)
            continue;
        int rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
        sorted_token_ids[rank_post_pad] = i;
    }
}

} // namespace muxi_layout_kernels
