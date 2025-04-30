#pragma once
#include "group_gemm_utils.h"

// TODO: implement copyAndLayoutB kernel

template <typename T>
__device__ void copyAndLayoutBKernel(const T *input_B, T *output_B,
                                     int *dev_samplesForExpert, int n, int k,
                                     int kSplit, int group_warpId) {
    constexpr int elementsPerAccess = MEMORY_ACCESS_SIZE / sizeof(T);
    constexpr int rowThreadsPerMma = 16; // trans per 16 rows
    constexpr int colThreadsPerMma = 4;  // MMA shape is 16 * (4 * 4) * 16
    int accessPerRow = k / (elementsPerAccess * colThreadsPerMma);

    int nWarps = (n / 16 / (64 / WARPSIZE)) * kSplit;
    int warpId = group_warpId / kSplit;
    int kSplitWarpId = group_warpId % kSplit;
    int laneId = threadIdx.x & (WARP_SIZE - 1);
    int quarterWarpId = laneId / 16;
    int quarterLaneId = laneId & (16 - 1);
    int numColsGroupsB = n / rowThreadsPerMma;
    int warpGroupBeginB = warpId * (numColsGroupsB / nWarps) +
                          std::min(warpId, numColsGroupsB % nWarps);
    int warpGroupEndB = (warpId + 1) * (numColsGroupsB / nWarps) +
                        std::min(warpId + 1, numColsGroupsB % nWarps);

    bool fillZeros = true;
    for (int i = 1; i <= dev_samplesForExpert[0]; i++) {
        if ((warpId * rowThreadsPerMma + quarterLaneId) ==
            dev_samplesForExpert[i]) {
            fillZeros = false;
            break;
        }
    }

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

    if (!fillZeros) {
#pragma unroll
        for (int i = 0; i < end_B; ++i) {
            for (int j = split_j_start; j < split_j_end; ++j) {
                tmpB = *(input_B_ptr + i * offset_warpGroup_B +
                         j * colThreadsPerMma);
                *(output_B_ptr + j * output_chunk_offset_B +
                  i * output_offset_warpGroup_B) = tmpB;
            }
        }
    }
}

template <typename T>
__global__ void copyAndLayoutBLauncher(T *B, T *dev_B_expert_buffers,
                                       int *dev_samplesForExperts,
                                       int batchSize, int k, int expertCount,
                                       int kSplit) {
    int global_warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
    int warpsPerGroup = (batchSize / 16 / (64 / WARPSIZE)) * kSplit;
    int groupId = global_warpId / warpsPerGroup;
    int group_warpId = global_warpId - groupId * warpsPerGroup;
    int expertsId = -1, tmp_index = -1;
    for (int i = 0; i < expertCount; i++) {
        if (dev_samplesForExperts[i * (batchSize + 1)] > 0) {
            tmp_index++;
            if (tmp_index == groupId) {
                expertsId = i;
                break;
            }
        }
    }
    if ((groupId > (expertCount - 1)) || (groupId < 0))
        return;

    copyAndLayoutBKernel<T>(B, dev_B_expert_buffers + expertsId * batchSize * k,
                            dev_samplesForExperts + expertsId * (batchSize + 1),
                            batchSize, k, kSplit, group_warpId);
}

template <typename T>
void copyAndLayoutB(T *B, T *dev_B_expert_buffers, int *dev_samplesForExperts,
                    int batchSize, int k, int expertCount, int gemmCount) {
    int kSplit = (batchSize / 16);
    int blockSize = 64 * kSplit;
    int gridSize = gemmCount;
    while (blockSize > 1024) {
        kSplit = (kSplit + 1) / 2;
        blockSize = 64 * kSplit;
        gridSize *= 2;
    }
    const mcStream_t stream =
        at::cuda::getCurrentCUDAStream(at::cuda::current_device());

    copyAndLayoutBLauncher<T><<<gridSize, blockSize, 0, stream>>>(
        B, dev_B_expert_buffers, dev_samplesForExperts, batchSize, k,
        expertCount, 1);
}