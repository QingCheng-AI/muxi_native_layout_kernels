#pragma once

#include "../utils.cuh"
#include "group_gemm_first.h"
#include "group_gemm_utils.h"

namespace muxi_layout_kernels {

template <typename _Ta, typename _Tscale>
struct FusedMoeFirstGemmAInputSoftFp8 {
    using Ta = _Ta;
    using Tscale = _Tscale;
    Ta *A;
    Tscale *A_scale;
    int A_scale_m, A_scale_n;

    __device__ __forceinline__ FusedMoeFirstGemmAInputSoftFp8(Ta *A,
                                                              Tscale *A_scale,
                                                              int A_scale_m,
                                                              int A_scale_n)
        : A(A), A_scale(A_scale), A_scale_m(A_scale_m), A_scale_n(A_scale_n) {}
};

template <typename...>
struct isFusedMoeFirstGemmAInputSoftFp8 : std::false_type {};
template <typename Ta, typename Tscale>
struct isFusedMoeFirstGemmAInputSoftFp8<
    FusedMoeFirstGemmAInputSoftFp8<Ta, Tscale>> : std::true_type {};

template <typename AInput, int ScaleBlockM, int ScaleBlockN,
          typename HyperParams>
    requires(isFusedMoeFirstGemmAInputSoftFp8<AInput>::value)
struct FusedMoeFirstGemmAIteratorSoftFp8 {
    using Ta = typename AInput::Ta;
    using Tscale = typename AInput::Tscale;
    static constexpr int APerWarp = HyperParams::APerWarp;
    static constexpr int stages = HyperParams::stages;
    static constexpr int elementsPerAccess = HyperParams::elementsPerAccess;
    static constexpr int rowThreadsPerMma = HyperParams::rowThreadsPerMma;
    static constexpr int colThreadsPerMma = HyperParams::colThreadsPerMma;

    UINT2 *A_ptr[APerWarp];
    Tscale *A_scale_ptr[APerWarp];
    UINT2 tmpA[stages][APerWarp];
    Tscale scale[stages][APerWarp];
    int A_ptr_offset = 0;
    int A_scale_col_offset = 0;

    __device__ __forceinline__ void init_ptr(AInput &A_input, int k,
                                             int warpRowsGroupBegin, int laneId,
                                             int quarterWarpId,
                                             int splitKStart) {
        int A_offset = ((warpRowsGroupBegin * rowThreadsPerMma) *
                        (k / elementsPerAccess)) +
                       laneId + splitKStart * stages * WARP_SIZE;
        int A_scale_offset =
            (warpRowsGroupBegin * rowThreadsPerMma / ScaleBlockM) *
                A_input.A_scale_n +
            (splitKStart * stages * elementsPerAccess * colThreadsPerMma +
             quarterWarpId * elementsPerAccess) /
                ScaleBlockN;
        for (int i = 0; i < APerWarp; i++) {
            A_ptr[i] = reinterpret_cast<UINT2 *>(A_input.A) + A_offset +
                       i * (rowThreadsPerMma * (k / elementsPerAccess));
            A_scale_ptr[i] =
                A_input.A_scale + A_scale_offset +
                (i * rowThreadsPerMma / ScaleBlockM) * A_input.A_scale_n;
        }
    }

    __device__ __forceinline__ void init_ptr_neutral(AInput &A_input) {
        for (int i = 0; i < APerWarp; i++) {
            A_ptr[i] = reinterpret_cast<UINT2 *>(A_input.A);
            A_scale_ptr[i] = A_input.A_scale;
        }
    }

    __device__ __forceinline__ void load_and_advance(int stage_id) {
        for (int i = 0; i < APerWarp; i++) {
            tmpA[stage_id][i] = mxc_read64_async(
                reinterpret_cast<INT64 *>(A_ptr[i] + A_ptr_offset));
            scale[stage_id][i] =
                *(A_scale_ptr[i] + A_scale_col_offset / ScaleBlockN);
        }
        A_ptr_offset += WARP_SIZE;
        A_scale_col_offset += elementsPerAccess * colThreadsPerMma;
    }
};

template <typename...>
struct isFusedMoeFirstGemmAIteratorSoftFp8 : std::false_type {};
template <typename AInput, int ScaleBlockM, int ScaleBlockN,
          typename HyperParams>
struct isFusedMoeFirstGemmAIteratorSoftFp8<FusedMoeFirstGemmAIteratorSoftFp8<
    AInput, ScaleBlockM, ScaleBlockN, HyperParams>> : std::true_type {};

template <typename Tb, typename AIter, typename HyperParams>
    requires(isFusedMoeFirstGemmAIteratorSoftFp8<AIter>::value)
struct FusedMoeFirstGemmMmaSoftFp8 {
    static constexpr int APerWarp = HyperParams::APerWarp;
    static constexpr int numCycleB = HyperParams::numCycleB;
    static constexpr int sharedNumCycleB = HyperParams::sharedNumCycleB;

    __device__ __forceinline__ void
    operator()(AIter &A_iter /* no const here: compiler bug */,
               UINT4 tmpB[numCycleB], FLOAT4 C_f32[numCycleB][APerWarp],
               int stage_id) {
        uint tmpA_bf16[4];
#pragma unroll
        for (int j = 0; j < sharedNumCycleB; j++) {
            for (int i = 0; i < APerWarp; i++) {
                // Convert fp8 to bf16.
                scalefp8tobf16(A_iter.tmpA[stage_id][i][0],
                               reinterpret_cast<uint16_t *>(tmpA_bf16),
                               A_iter.scale[stage_id][i]);
                scalefp8tobf16(A_iter.tmpA[stage_id][i][1],
                               reinterpret_cast<uint16_t *>(tmpA_bf16 + 2),
                               A_iter.scale[stage_id][i]);
                C_f32[j][i] =
                    mma_16x16x16f16<Tb>(tmpA_bf16[0], tmpA_bf16[1], tmpB[j][0],
                                        tmpB[j][1], C_f32[j][i]);
                C_f32[j][i] =
                    mma_16x16x16f16<Tb>(tmpA_bf16[2], tmpA_bf16[3], tmpB[j][2],
                                        tmpB[j][3], C_f32[j][i]);
            }
        }
    }
};

// first group gemm __global__ interface
template <typename Ta, typename Tb, typename Taccum, typename Tc,
          int BLOCK_DIM_X, int microBatchsize, int APerWarp, int splitK,
          int ScaleBlockM, int ScaleBlockN>
__global__ void __launch_bounds__(BLOCK_DIM_X)
    fused_moe_first_group_gemm_kernel(Ta *A, Tb *B, Tc *C,
                                      int *sorted_token_ids, int *experts_ids,
                                      int m, int n, int k, Taccum alpha,
                                      int topk, int padded_ptr_number,
                                      int *dev_padded_num_experts,
                                      Taccum *A_scale, int A_scale_m,
                                      int A_scale_n) {
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warps_per_gemm = (m / 16 / APerWarp * splitK);
    int gemm_id = global_warp_id / warps_per_gemm;
    int gemm_warp_id = global_warp_id % warps_per_gemm;
    if (gemm_id >= dev_padded_num_experts[0])
        return;
    using HyperParams =
        FusedMoeFirstGemmKernelHyperParams<BLOCK_DIM_X, microBatchsize,
                                           APerWarp, splitK>;
    using AInput = FusedMoeFirstGemmAInputSoftFp8<Ta, Taccum>;
    using AIterator =
        FusedMoeFirstGemmAIteratorSoftFp8<AInput, ScaleBlockM, ScaleBlockN,
                                          HyperParams>;
    using Mma = FusedMoeFirstGemmMmaSoftFp8<Tb, AIterator, HyperParams>;
    fused_moe_first_gemm_kernel<Ta, Tb, Taccum, Tc, HyperParams, AInput,
                                AIterator, Mma>(
        AInput(A + experts_ids[gemm_id] * m * k,
               A_scale + experts_ids[gemm_id] * A_scale_m * A_scale_n,
               A_scale_m, A_scale_n),
        B, C, sorted_token_ids + gemm_id * microBatchsize, m, n, k, alpha, topk,
        padded_ptr_number, gemm_warp_id);
}

} // namespace muxi_layout_kernels
