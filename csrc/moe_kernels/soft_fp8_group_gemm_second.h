#pragma once

#include "../utils.cuh"
#include "group_gemm_second.h"
#include "group_gemm_utils.h"

namespace muxi_layout_kernels {

template <typename _Ta, typename _Tscale>
struct FusedMoeSecondGemmAInputSoftFp8 {
    using Ta = _Ta;
    using Tscale = _Tscale;
    Ta *A;
    Tscale *A_scale;
    int A_scale_m, A_scale_n;

    __device__ __forceinline__ FusedMoeSecondGemmAInputSoftFp8(Ta *A,
                                                               Tscale *A_scale,
                                                               int A_scale_m,
                                                               int A_scale_n)
        : A(A), A_scale(A_scale), A_scale_m(A_scale_m), A_scale_n(A_scale_n) {}
};

template <typename...>
struct isFusedMoeSecondGemmAInputSoftFp8 : std::false_type {};
template <typename Ta, typename Tscale>
struct isFusedMoeSecondGemmAInputSoftFp8<
    FusedMoeSecondGemmAInputSoftFp8<Ta, Tscale>> : std::true_type {};

template <typename AInput, int ScaleBlockM, int ScaleBlockN,
          typename HyperParams>
    requires(isFusedMoeSecondGemmAInputSoftFp8<AInput>::value)
struct FusedMoeSecondGemmAIteratorSoftFp8 {
    using Ta = typename AInput::Ta;
    using Tscale = typename AInput::Tscale;
    static constexpr int tile_m = HyperParams::tile_m;
    static constexpr int elementsPerAccess = HyperParams::elementsPerAccess;
    static constexpr int rowThreadsPerMma = HyperParams::rowThreadsPerMma;
    static constexpr int colThreadsPerMma = HyperParams::colThreadsPerMma;
    static constexpr int stages = HyperParams::stages;
    static constexpr int warpPerBlock = HyperParams::warpPerBlock;
    static constexpr int m_times = HyperParams::m_times;
    static constexpr int loadAPerStage = HyperParams::loadAPerStage;

    UINT2 *A_ptr_stages[stages];
    Tscale *A_scale_ptr[stages];
    UINT2 tmpA[stages][m_times / warpPerBlock];
    Tscale scale[stages][m_times / warpPerBlock];

    __device__ __forceinline__ void init_ptr(AInput &A_input, int k,
                                             int stage_id, int laneId,
                                             int blockRowsGroup,
                                             int quarterWarpId) {
        A_ptr_stages[stage_id] =
            reinterpret_cast<UINT2 *>(A_input.A) +
            blockRowsGroup * tile_m * k / elementsPerAccess +
            stage_id * WARP_SIZE + laneId;
        A_scale_ptr[stage_id] =
            A_input.A_scale +
            (blockRowsGroup * tile_m) / ScaleBlockM * A_input.A_scale_n +
            (stage_id * colThreadsPerMma * elementsPerAccess +
             quarterWarpId * elementsPerAccess) /
                ScaleBlockN;
    }

    __device__ __forceinline__ void
    load_and_advance(AInput &A_input, int k, int readA_offset, int stage_id) {
        for (int j = 0; j < loadAPerStage; j++) {
            tmpA[stage_id][j] = mxc_read64_async(reinterpret_cast<INT64 *>(
                A_ptr_stages[stage_id] +
                (readA_offset + j) * rowThreadsPerMma * k / elementsPerAccess));
            scale[stage_id][j] = *(A_scale_ptr[stage_id] +
                                   (readA_offset + j) * rowThreadsPerMma /
                                       ScaleBlockM * A_input.A_scale_n);
        }
        A_ptr_stages[stage_id] += WARP_SIZE * stages;
        A_scale_ptr[stage_id] +=
            colThreadsPerMma * elementsPerAccess * stages / ScaleBlockN;
    }
};

template <typename...>
struct isFusedMoeSecondGemmAIteratorSoftFp8 : std::false_type {};
template <typename AInput, int ScaleBlockM, int ScaleBlockN,
          typename HyperParams>
struct isFusedMoeSecondGemmAIteratorSoftFp8<FusedMoeSecondGemmAIteratorSoftFp8<
    AInput, ScaleBlockM, ScaleBlockN, HyperParams>> : std::true_type {};

template <typename Tb, typename AIter, typename HyperParams>
    requires(isFusedMoeSecondGemmAIteratorSoftFp8<AIter>::value)
struct FusedMoeSecondGemmMmaSoftFp8 {
    static constexpr int warpPerBlock = HyperParams::warpPerBlock;
    static constexpr int m_times = HyperParams::m_times;
    static constexpr int n_times = HyperParams::n_times;

    __device__ __forceinline__ void
    operator()(AIter &A_iter, UINT4 tmpB[n_times],
               FLOAT4 C_f32[m_times / warpPerBlock][n_times], int stage_id) {
        uint tmpA_bf16[4];
        for (int j = 0; j < m_times / warpPerBlock; j++) {
            scalefp8tobf16(A_iter.tmpA[stage_id][j][0],
                           reinterpret_cast<uint16_t *>(tmpA_bf16),
                           A_iter.scale[stage_id][j]);
            scalefp8tobf16(A_iter.tmpA[stage_id][j][1],
                           reinterpret_cast<uint16_t *>(tmpA_bf16 + 2),
                           A_iter.scale[stage_id][j]);
            for (int k = 0; k < n_times; k++) {
                C_f32[j][k] =
                    mma_16x16x16f16<Tb>(tmpA_bf16[0], tmpA_bf16[1], tmpB[k][0],
                                        tmpB[k][1], C_f32[j][k]);
                C_f32[j][k] =
                    mma_16x16x16f16<Tb>(tmpA_bf16[2], tmpA_bf16[3], tmpB[k][2],
                                        tmpB[k][3], C_f32[j][k]);
            }
        }
    }
};

template <typename Ta, typename Tb, typename Taccum, typename Tc,
          int BLOCK_DIM_X, int tile_m, int tile_n, int tile_k,
          int micro_batchsize, int ScaleBlockM, int ScaleBlockN>
__global__ void fused_moe_second_group_gemm_kernel(
    Ta *A, Tb *B, Tc *C, int *sorted_token_ids, int *experts_ids, int m, int n,
    int k, Tc *routing_weights, int padded_ptr_number, int topk,
    int *dev_padded_num_experts, Taccum *A_scale, int A_scale_m,
    int A_scale_n) {
    int global_blockIdx = blockIdx.x;
    int blocks_per_gemm =
        (((m + tile_m - 1) / tile_m) * ((n + tile_n - 1) / tile_n));
    int gemm_id = global_blockIdx / blocks_per_gemm;
    int block_x_id_in_gemm =
        (global_blockIdx % blocks_per_gemm) % ((m + tile_m - 1) / tile_m);
    int block_y_id_in_gemm =
        (global_blockIdx % blocks_per_gemm) / ((m + tile_m - 1) / tile_m);
    if (gemm_id >= dev_padded_num_experts[0])
        return;
    using HyperParams =
        FusedMoeSecondGemmKernelHyperParams<BLOCK_DIM_X, tile_m, tile_n, tile_k,
                                            micro_batchsize>;
    using AInput = FusedMoeSecondGemmAInputSoftFp8<Ta, Taccum>;
    using AIterator =
        FusedMoeSecondGemmAIteratorSoftFp8<AInput, ScaleBlockM, ScaleBlockN,
                                           HyperParams>;
    using Mma = FusedMoeSecondGemmMmaSoftFp8<Tb, AIterator, HyperParams>;
    fused_moe_second_gemm_kernel<Ta, Tb, Taccum, Tc, HyperParams, AInput,
                                 AIterator, Mma>(
        AInput(A + experts_ids[gemm_id] * m * k,
               A_scale + experts_ids[gemm_id] * A_scale_m * A_scale_n,
               A_scale_m, A_scale_n),
        B, C, sorted_token_ids + gemm_id * micro_batchsize, m, n, k,
        routing_weights, padded_ptr_number, topk, block_x_id_in_gemm,
        block_y_id_in_gemm);
}

} // namespace muxi_layout_kernels
