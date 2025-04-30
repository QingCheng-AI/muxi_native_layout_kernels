#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>

#include "re_layout_c.h"
#include "utils.cuh"

namespace muxi_layout_kernels {

template <typename T>
__global__ void GemmMmaReLayoutCReuseA(T *input_B, T *output_B, int n, int k,
                                       int kSplit) {
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
    int4 *input_B_ptr = reinterpret_cast<int4 *>(input_B) + offset_input_B;
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
            tmpB = *(output_B_ptr + j * output_chunk_offset_B +
                     i * output_offset_warpGroup_B);
            *(input_B_ptr + i * offset_warpGroup_B + j * colThreadsPerMma) =
                tmpB;
        }
    }
}

torch::Tensor reLayoutC(torch::Tensor C_in) {
    TORCH_CHECK(C_in.is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(C_in.is_contiguous(), "C_in must be contiguous");

    TORCH_CHECK(C_in.dim() == 5, "C_in must be a 5D tensor");
    TORCH_CHECK(C_in.size(2) == 4, "C_in.shape[2] must be 4");
    TORCH_CHECK(C_in.size(3) == 16, "C_in.shape[3] must be 16");
    TORCH_CHECK(C_in.size(4) == 8, "C_in.shape[4] must be 8");
    int m = C_in.size(0) * 32, n = C_in.size(1) * 16;
    torch::Tensor C_out = torch::empty({n, m}, C_in.options());

    constexpr int block_dim_x = 64;
    int kSplit = (104 / ((n + 16 - 1) / 16) * 4);
    if (C_out.dtype() == torch::kFloat16) {
        GemmMmaReLayoutCReuseA<__half>
            <<<(n / 16 / (block_dim_x / WARP_SIZE) * kSplit), block_dim_x>>>(
                reinterpret_cast<__half *>(C_out.data_ptr<at::Half>()),
                reinterpret_cast<__half *>(C_in.data_ptr<at::Half>()), n, m,
                kSplit);
    } else if (C_out.dtype() == torch::kBFloat16) {
        GemmMmaReLayoutCReuseA<__maca_bfloat16>
            <<<(n / 16 / (block_dim_x / WARP_SIZE) * kSplit), block_dim_x>>>(
                reinterpret_cast<__maca_bfloat16 *>(
                    C_out.data_ptr<at::BFloat16>()),
                reinterpret_cast<__maca_bfloat16 *>(
                    C_in.data_ptr<at::BFloat16>()),
                n, m, kSplit);
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }

    return C_out;
}

} // namespace muxi_layout_kernels
