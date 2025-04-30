#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>

#include "layout_b.h"
#include "utils.cuh"

namespace muxi_layout_kernels {

template <typename T>
__global__ void GemmMmaLayoutBCReuseA(const T *input_B, T *output_B, int n,
                                      int k, int kSplit) {
    constexpr int elementsPerAccess = MEMORY_ACCESS_SIZE / sizeof(T);
    constexpr int rowThreadsPerMma = 16; // trans per 16 rows
    constexpr int colThreadsPerMma = 4;  // MMA shape is 16 * (4 * 4) * 16
    int accessPerRow = k / (elementsPerAccess * colThreadsPerMma);

    int nWarps = (gridDim.x * blockDim.x) / WARP_SIZE / kSplit;
    int warpId = (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE / kSplit;
    int kSplitWarpId =
        ((blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE) % kSplit;
    int laneId = threadIdx.x & (WARP_SIZE - 1);
    int quarterWarpId = laneId / 16;
    int quarterLaneId = laneId & (16 - 1);
    int numColsGroupsB = n / rowThreadsPerMma;
    int warpGroupBeginB = warpId * (numColsGroupsB / nWarps) +
                          std::min(warpId, numColsGroupsB % nWarps);
    int warpGroupEndB = (warpId + 1) * (numColsGroupsB / nWarps) +
                        std::min(warpId + 1, numColsGroupsB % nWarps);

    int4 tmpB;

    int offset_input_B = (warpGroupBeginB * rowThreadsPerMma + quarterLaneId) *
                             (k / elementsPerAccess) +
                         quarterWarpId;
    int offset_output_B = quarterWarpId * rowThreadsPerMma +
                          warpGroupBeginB * WARP_SIZE + quarterLaneId;
    int offset_warpGroup_B = rowThreadsPerMma * (k / elementsPerAccess);
    const int4 *input_B_ptr =
        reinterpret_cast<const int4 *>(input_B) + offset_input_B;
    int4 *output_B_ptr = reinterpret_cast<int4 *>(output_B) + offset_output_B;
    int output_chunk_offset_B =
        (colThreadsPerMma * elementsPerAccess) * (n / elementsPerAccess);
    int output_offset_warpGroup_B =
        (elementsPerAccess * colThreadsPerMma) / elementsPerAccess;

    int end_B = warpGroupEndB - warpGroupBeginB;
    int split_j_start = kSplitWarpId * (accessPerRow / kSplit) +
                        std::min(kSplitWarpId, accessPerRow % kSplit);
    int split_j_end = (kSplitWarpId + 1) * (accessPerRow / kSplit) +
                      std::min(kSplitWarpId + 1, accessPerRow % kSplit);

    for (int i = 0; i < end_B; ++i) {
        for (int j = split_j_start; j < split_j_end; ++j) {
            tmpB =
                *(input_B_ptr + i * offset_warpGroup_B + j * colThreadsPerMma);

            *(output_B_ptr + j * output_chunk_offset_B +
              i * output_offset_warpGroup_B) = tmpB;
        }
    }
}

torch::Tensor layoutB(torch::Tensor B) {
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    int n = B.size(0), k = B.size(1);
    TORCH_CHECK(n % 16 == 0, "n must be a multiple of 16");
    TORCH_CHECK(k % 32 == 0, "k must be a multiple of 32");
    torch::Tensor B_out = torch::empty({k / 32, n / 16, 4, 16, 8}, B.options());

    constexpr int block_dim_x = 64;
    int kSplit = (104 / ((n + 16 - 1) / 16) * 4);
    kSplit = std::max(4, kSplit);
    auto cur_device = at::cuda::current_device();
    const mcStream_t stream = at::cuda::getCurrentCUDAStream(cur_device);
    if (B.dtype() == torch::kFloat16) {
        GemmMmaLayoutBCReuseA<__half>
            <<<(n / 16 / (block_dim_x / WARP_SIZE)) * kSplit, block_dim_x, 0,
               stream>>>(reinterpret_cast<__half *>(B.data_ptr<at::Half>()),
                         reinterpret_cast<__half *>(B_out.data_ptr<at::Half>()),
                         n, k, kSplit);
    } else if (B.dtype() == torch::kBFloat16) {
        GemmMmaLayoutBCReuseA<__maca_bfloat16>
            <<<(n / 16 / (block_dim_x / WARP_SIZE)) * kSplit, block_dim_x, 0,
               stream>>>(
                reinterpret_cast<__maca_bfloat16 *>(B.data_ptr<at::BFloat16>()),
                reinterpret_cast<__maca_bfloat16 *>(
                    B_out.data_ptr<at::BFloat16>()),
                n, k, kSplit);
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }

    return B_out;
}

} // namespace muxi_layout_kernels
