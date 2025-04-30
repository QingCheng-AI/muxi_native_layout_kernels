#pragma once
#include "group_gemm_utils.h"

template <typename T>
__global__ void LayoutB(const T *input_B, T *output_B, int n, int k,
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

template <typename T>
__global__ void LayoutA(const T *input_A, T *output_A, int spatialDim,
                        int reducedDim) {
    constexpr int elementsPerAccess = MEMORY_ACCESS_SIZE / sizeof(T);
    constexpr int rowThreadsPerMma = 16; // trans per 16 rows
    constexpr int colThreadsPerMma = 4;  // MMA shape is 16 * (4 * 4) * 16
    int accessPerRow = reducedDim / (elementsPerAccess * colThreadsPerMma);

    int nWarps = (gridDim.x * blockDim.x) / WARP_SIZE;
    int warpId = (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;
    int laneId = threadIdx.x & (WARP_SIZE - 1);
    int quarterWarpId = laneId / 16;
    int quarterLaneId = laneId & (16 - 1);

    int nRowsGroups = spatialDim / rowThreadsPerMma;
    int warpGroupBegin = warpId * (nRowsGroups / nWarps) +
                         std::min(warpId, nRowsGroups % nWarps);
    int warpGroupEnd = (warpId + 1) * (nRowsGroups / nWarps) +
                       std::min(warpId + 1, nRowsGroups % nWarps);

    int4 tmpA;

    int offset_input = (warpGroupBegin * rowThreadsPerMma + quarterLaneId) *
                           (reducedDim / elementsPerAccess) +
                       quarterWarpId;
    int offset_output =
        (warpGroupBegin * rowThreadsPerMma) * (reducedDim / elementsPerAccess) +
        quarterWarpId * rowThreadsPerMma + quarterLaneId;
    int offset_warpGroup = rowThreadsPerMma * (reducedDim / elementsPerAccess);
    const int4 *input_A_ptr =
        reinterpret_cast<const int4 *>(input_A) + offset_input;
    int4 *output_A_ptr = reinterpret_cast<int4 *>(output_A) + offset_output;

    int end = warpGroupEnd - warpGroupBegin;

    for (int i = 0; i < end; ++i) {
        for (int k = 0; k < accessPerRow; ++k) {
            tmpA = *(input_A_ptr + i * offset_warpGroup + k * colThreadsPerMma);
            *(output_A_ptr + i * offset_warpGroup + k * WARP_SIZE) = tmpA;
        }
    }
}

template <typename T>
__global__ void ReLayoutC(T *input_B, T *output_B, int n, int k, int kSplit) {
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

template <typename T>
__global__ void transpose(T *A, T *A_transpose, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n && idy < m) {
        A_transpose[idx * m + idy] = A[idy * n + idx];
    }
}
