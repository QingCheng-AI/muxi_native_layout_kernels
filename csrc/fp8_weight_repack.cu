#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>

#include "fp8_weight_repack.h"

namespace muxi_layout_kernels {
#define BLOCK_SIZE 256

__global__ void fp8_weight_repack_1324(uint32_t *W, uint32_t *C, const int N,
                                       const int M, const int K) {
    const uint32_t elements_per_fetch = 4;
    const uint32_t thread_numx = 8;
    const uint32_t thread_numy = blockDim.x / thread_numx;
    const uint32_t thread_idx = threadIdx.x % thread_numx;
    const uint32_t thread_idy = threadIdx.x / thread_numx;
    const uint32_t k_iter = BLOCK_SIZE / elements_per_fetch / thread_numx;
    const uint32_t m_iter = BLOCK_SIZE / thread_numy;
    const uint32_t block_num_k = K / BLOCK_SIZE;
    const uint32_t pid_k = blockIdx.x % block_num_k;
    const uint32_t pid_m = blockIdx.x / block_num_k;
    const uint32_t stride_m = K / elements_per_fetch;
    const uint32_t stride_n = stride_m * M;
    uint32_t first_off_set = (pid_m * BLOCK_SIZE + thread_idy) * stride_m +
                             pid_k * BLOCK_SIZE / elements_per_fetch +
                             thread_idx;
    uint32_t tmp32;

    for (int k = 0; k < N; k++) {
        for (int i = 0; i < m_iter; i++) {
            for (int j = 0; j < k_iter; j++) {
                tmp32 = *(W + first_off_set + k * stride_n +
                          i * thread_numy * stride_m + j * thread_numx);
                tmp32 = (tmp32 & 0xff0000ff) | ((tmp32 & 0x00ff0000) >> 8) |
                        ((tmp32 & 0x0000ff00) << 8);
                *(C + first_off_set + k * stride_n +
                  i * thread_numy * stride_m + j * thread_numx) = tmp32;
            }
        }
    }
}

torch::Tensor fp8_weight_repack(torch::Tensor weight) {
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    torch::Tensor C;
    if (weight.dim() == 3) {
        const int N = weight.size(0);
        const int M = weight.size(1);
        const int K = weight.size(2);
        C = torch::empty({N, M, K}, weight.options());
        fp8_weight_repack_1324<<<(M / BLOCK_SIZE) * (K / BLOCK_SIZE), 256>>>(
            reinterpret_cast<uint32_t *>(weight.data_ptr()),
            reinterpret_cast<uint32_t *>(C.data_ptr()), N, M, K);
    } else {
        const int N = 1;
        const int M = weight.size(0);
        const int K = weight.size(1);
        C = torch::empty({M, K}, weight.options());
        fp8_weight_repack_1324<<<(M / BLOCK_SIZE) * (K / BLOCK_SIZE), 256>>>(
            reinterpret_cast<uint32_t *>(weight.data_ptr()),
            reinterpret_cast<uint32_t *>(C.data_ptr()), N, M, K);
    }
    return C;
}
} // namespace muxi_layout_kernels