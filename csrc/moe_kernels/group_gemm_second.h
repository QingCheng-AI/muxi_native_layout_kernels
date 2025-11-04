#pragma once

#include "../utils.cuh"
#include "group_gemm_utils.h"

namespace muxi_layout_kernels {

template <int _BLOCK_DIM_X, int _tile_m, int _tile_n, int _tile_k,
          int _micro_batchsize, int _elementsPerAccess = 8,
          int _rowThreadsPerMma = 16, int _colThreadsPerMma = 4,
          int _elementsPerThreadPerMma = 4>
    requires(_tile_k % (_elementsPerThreadPerMma * _elementsPerAccess) == 0 &&
             _BLOCK_DIM_X % WARP_SIZE == 0 &&
             _tile_m % _rowThreadsPerMma == 0 &&
             _tile_n % _rowThreadsPerMma == 0)
struct FusedMoeSecondGemmKernelHyperParams {
    static constexpr int BLOCK_DIM_X = _BLOCK_DIM_X;
    static constexpr int tile_m = _tile_m;
    static constexpr int tile_n = _tile_n;
    static constexpr int tile_k = _tile_k;
    static constexpr int micro_batchsize = _micro_batchsize;
    static constexpr int elementsPerAccess = _elementsPerAccess;
    static constexpr int rowThreadsPerMma = _rowThreadsPerMma;
    static constexpr int colThreadsPerMma = _colThreadsPerMma;
    static constexpr int elementsPerThreadPerMma = _elementsPerThreadPerMma;
    static constexpr int stages =
        tile_k / (elementsPerThreadPerMma * elementsPerAccess);
    static constexpr int warpPerBlock = BLOCK_DIM_X / WARP_SIZE;
    static constexpr int m_times = (tile_m / rowThreadsPerMma);
    static constexpr int n_times = (tile_n / rowThreadsPerMma);
    static constexpr int loadAPerStage =
        ((m_times + warpPerBlock - 1) / warpPerBlock);
    static constexpr int loadBPerStage =
        ((n_times + warpPerBlock - 1) / warpPerBlock);
};

template <typename _Ta> struct FusedMoeSecondGemmAInputFp16Bf16 {
    using Ta = _Ta;
    Ta *A;

    __device__ __forceinline__ FusedMoeSecondGemmAInputFp16Bf16(Ta *A) : A(A) {}
};

template <typename...>
struct isFusedMoeSecondGemmAInputFp16Bf16 : std::false_type {};
template <typename Ta>
struct isFusedMoeSecondGemmAInputFp16Bf16<FusedMoeSecondGemmAInputFp16Bf16<Ta>>
    : std::true_type {};

template <typename AInput, typename HyperParams>
    requires(isFusedMoeSecondGemmAInputFp16Bf16<AInput>::value)
struct FusedMoeSecondGemmAIteratorFp16Bf16 {
    using Ta = typename AInput::Ta;
    static constexpr int tile_m = HyperParams::tile_m;
    static constexpr int elementsPerAccess = HyperParams::elementsPerAccess;
    static constexpr int rowThreadsPerMma = HyperParams::rowThreadsPerMma;
    static constexpr int stages = HyperParams::stages;
    static constexpr int warpPerBlock = HyperParams::warpPerBlock;
    static constexpr int m_times = HyperParams::m_times;
    static constexpr int loadAPerStage = HyperParams::loadAPerStage;

    UINT4 *A_ptr_stages[stages];
    UINT4 tmpA[stages][m_times / warpPerBlock];

    __device__ __forceinline__ void init_ptr(AInput &A_input, int k,
                                             int stage_id, int laneId,
                                             int blockRowsGroup,
                                             int quarterWarpId) {
        A_ptr_stages[stage_id] =
            reinterpret_cast<UINT4 *>(A_input.A) +
            blockRowsGroup * tile_m * k / elementsPerAccess +
            stage_id * WARP_SIZE + laneId;
    }

    __device__ __forceinline__ void
    load_and_advance(AInput &A_input, int k, int readA_offset, int stage_id) {
        for (int j = 0; j < loadAPerStage; j++) {
            tmpA[stage_id][j] =
                __builtin_mxc_load_global_async128(reinterpret_cast<INT128 *>(
                    A_ptr_stages[stage_id] + (readA_offset + j) *
                                                 rowThreadsPerMma * k /
                                                 elementsPerAccess));
        }
        A_ptr_stages[stage_id] += WARP_SIZE * stages;
    }
};

template <typename...>
struct isFusedMoeSecondGemmAIteratorFp16Bf16 : std::false_type {};
template <typename AInput, typename HyperParams>
struct isFusedMoeSecondGemmAIteratorFp16Bf16<
    FusedMoeSecondGemmAIteratorFp16Bf16<AInput, HyperParams>> : std::true_type {
};

template <typename Tb, typename AIter, typename HyperParams>
    requires(isFusedMoeSecondGemmAIteratorFp16Bf16<AIter>::value)
struct FusedMoeSecondGemmMmaFp16Bf16 {
    static constexpr int warpPerBlock = HyperParams::warpPerBlock;
    static constexpr int m_times = HyperParams::m_times;
    static constexpr int n_times = HyperParams::n_times;

    __device__ __forceinline__ void
    operator()(AIter &A_iter, UINT4 tmpB[n_times],
               FLOAT4 C_f32[m_times / warpPerBlock][n_times], int stage_id) {
        for (int j = 0; j < m_times / warpPerBlock; j++) {
            for (int k = 0; k < n_times; k++) {
                C_f32[j][k] = mma_16x16x16f16<Tb>(
                    A_iter.tmpA[stage_id][j][0], A_iter.tmpA[stage_id][j][1],
                    tmpB[k][0], tmpB[k][1], C_f32[j][k]);
                C_f32[j][k] = mma_16x16x16f16<Tb>(
                    A_iter.tmpA[stage_id][j][2], A_iter.tmpA[stage_id][j][3],
                    tmpB[k][2], tmpB[k][3], C_f32[j][k]);
            }
        }
    }
};

template <typename Ta, typename Tb, typename Taccum, typename Tc,
          typename HyperParams, typename AInput, typename AIterator,
          typename Mma>
__device__ void fused_moe_second_gemm_kernel(AInput A_input, Tb *B, Tc *C,
                                             int *sorted_token_ids, int m,
                                             int n, int k, Tc *routing_weights,
                                             int padded_ptr_number, int topk,
                                             int block_x_id, int block_y_id) {
    constexpr int tile_m = HyperParams::tile_m;
    constexpr int tile_n = HyperParams::tile_n;
    constexpr int tile_k = HyperParams::tile_k;
    constexpr int micro_batchsize = HyperParams::micro_batchsize;
    constexpr int elementsPerAccess = HyperParams::elementsPerAccess;
    constexpr int rowThreadsPerMma = HyperParams::rowThreadsPerMma;
    constexpr int colThreadsPerMma = HyperParams::colThreadsPerMma;
    constexpr int elementsPerThreadPerMma =
        HyperParams::elementsPerThreadPerMma;
    constexpr int stages = HyperParams::stages;
    constexpr int warpPerBlock = HyperParams::warpPerBlock;
    constexpr int m_times = HyperParams::m_times;
    constexpr int n_times = HyperParams::n_times;
    constexpr int loadAPerStage = HyperParams::loadAPerStage;
    constexpr int loadBPerStage = HyperParams::loadBPerStage;

    const int warpIdInBlock =
        __builtin_mxc_readfirstlane(threadIdx.x / WARP_SIZE);
    const int laneId = threadIdx.x & (WARP_SIZE - 1);
    const int quarterWarpId = laneId / 16;
    const int quarterLaneId = laneId & (16 - 1);

    const int blockColsGroup = __builtin_mxc_readfirstlane(block_y_id);
    const int blockRowsGroup = __builtin_mxc_readfirstlane(block_x_id);
    const int kChunks = (k / tile_k);

    using CStgType = __NATIVE_VECTOR__(sizeof(Tc), uint);

    __shared__ UINT4 shared_data[micro_batchsize * tile_k / elementsPerAccess];
    AIterator A_iter;
    UINT4 *shared_B_stages[stages];
    UINT4 *B_ptr_stages[stages][loadBPerStage];
    CStgType *C_ptr = reinterpret_cast<CStgType *>(C);
    int C_write_ptr[n_times];
    Tc thread_token_weight[n_times];

    for (int j = 0;
         j < n_times && (j < (n / rowThreadsPerMma - blockColsGroup * n_times));
         j++) {
        C_write_ptr[j] = -1;
        int thread_token_ptr = padded_ptr_number;
        int thread_offset =
            blockColsGroup * tile_n + j * rowThreadsPerMma + quarterLaneId;
        if (thread_offset < micro_batchsize) {
            thread_token_ptr = sorted_token_ids[thread_offset];
            C_write_ptr[j] = thread_token_ptr / topk;
        }
        if (thread_token_ptr >= padded_ptr_number) {
            thread_token_ptr = 0;
            C_write_ptr[j] = -1;
            thread_token_weight[j] = zero<Tc>();
        } else {
            thread_token_weight[j] = routing_weights[thread_token_ptr];
        }
    }

    for (int j = 0; j < loadBPerStage; j++) {
        int thread_token_ptr = padded_ptr_number;
        int thread_offset =
            blockColsGroup * tile_n +
            (j * warpPerBlock + warpIdInBlock) * rowThreadsPerMma +
            quarterLaneId;
        if (thread_offset < micro_batchsize) {
            thread_token_ptr = sorted_token_ids[thread_offset];
        }
        if (thread_token_ptr >= padded_ptr_number) {
            thread_token_ptr = 0;
        }

        // k / elementsPerAccess * 2 for a part of space memory after silu and
        // mul
        B_ptr_stages[0][j] = reinterpret_cast<UINT4 *>(B) +
                             thread_token_ptr * k / elementsPerAccess * 2 +
                             quarterWarpId;
    }

    for (int i = 1; i < stages; i++) {
        for (int j = 0; j < loadBPerStage; j++) {
            B_ptr_stages[i][j] = B_ptr_stages[0][j] + i * colThreadsPerMma;
        }
    }

    for (int i = 0; i < stages; i++) {
        shared_B_stages[i] =
            &shared_data[tile_n * tile_k / elementsPerAccess / stages * i];
        A_iter.init_ptr(A_input, k, i, laneId, blockRowsGroup, quarterWarpId);
    }

    // return the threads without task
    if ((blockRowsGroup >= ((m + tile_m - 1) / tile_m)) ||
        (blockColsGroup >= (n + tile_n - 1) / tile_n)) {
        return;
    }

    UINT4 tmpB[n_times];
    FLOAT4 C_f32[m_times / warpPerBlock][n_times];
    Mma mma;
    for (int i = 0; i < m_times / warpPerBlock; i++) {
        for (int j = 0; j < n_times; j++) {
            C_f32[i][j] = {0, 0, 0, 0};
        }
    }

    const int readA_offset = warpIdInBlock * (m_times / warpPerBlock);

    // prefill the pipeline, prefetch stages data.
    for (int i = 0; i < stages; i++) {
        A_iter.load_and_advance(A_input, k, readA_offset, i);
        for (int j = 0; j < loadBPerStage; j++) {
            const int tmp_j = j * warpPerBlock + warpIdInBlock;
            const int predicatorB =
                (tmp_j < n_times)
                    ? (tmp_j < (n / rowThreadsPerMma - blockColsGroup * n_times)
                           ? 1
                           : 0)
                    : 0;
            LDG_B128_BSM_WITH_PREDICATOR_NORET0(
                shared_B_stages[i] + warpPerBlock * j * WARP_SIZE + threadIdx.x,
                B_ptr_stages[i][j], predicatorB, 1, MACA_ICMP_EQ);
            B_ptr_stages[i][j] += colThreadsPerMma * stages;
        }
    }

    // main loop in pipiline compute
    for (int i = 1; i < kChunks; i++) {
        // for every stage
        for (int stage = 0; stage < stages; stage++) {
            // wait for async data load
            __builtin_mxc_arrive_gvmcnt((loadAPerStage + loadBPerStage) *
                                        (stages - 1));
            __builtin_mxc_barrier_inst();

            for (int j = 0; j < n_times; j++) {
                tmpB[j] = *(shared_B_stages[stage] + (laneId + j * WARP_SIZE));
            }

            // MMA compute
            mma(A_iter, tmpB, C_f32, stage);
            __builtin_mxc_barrier_inst();

            // async load next stage data
            {
                A_iter.load_and_advance(A_input, k, readA_offset, stage);
                for (int j = 0; j < loadBPerStage; j++) {
                    const int tmp_j = j * warpPerBlock + warpIdInBlock;
                    const int predicatorB =
                        (tmp_j < n_times) ? (tmp_j < (n / rowThreadsPerMma -
                                                      blockColsGroup * n_times)
                                                 ? 1
                                                 : 0)
                                          : 0;
                    LDG_B128_BSM_WITH_PREDICATOR_NORET0(
                        shared_B_stages[stage] + warpPerBlock * j * WARP_SIZE +
                            threadIdx.x,
                        B_ptr_stages[stage][j], predicatorB, 1, MACA_ICMP_EQ);
                    B_ptr_stages[stage][j] += colThreadsPerMma * stages;
                }
            }
        }
    }

    // compute the last kChunk
    for (int stage = 0; stage < stages; stage++) {
        // wait for async data load
        if (stage == 0) {
            __builtin_mxc_arrive_gvmcnt((loadAPerStage + loadBPerStage) *
                                        (stages - 1));
        } else if (stage == 1) {
            __builtin_mxc_arrive_gvmcnt((loadAPerStage + loadBPerStage) *
                                        (stages - 2));
        } else if (stage == 2) {
            __builtin_mxc_arrive_gvmcnt((loadAPerStage + loadBPerStage) *
                                        (stages - 3));
        } else if (stage == 3) {
            __builtin_mxc_arrive_gvmcnt((loadAPerStage + loadBPerStage) *
                                        (stages - 4));
        }
        __builtin_mxc_barrier_inst();

        for (int j = 0; j < n_times; j++) {
            tmpB[j] = *(shared_B_stages[stage] + (laneId + j * WARP_SIZE));
        }

        // MMA compute
        mma(A_iter, tmpB, C_f32, stage);
    }

    // store result
    int warpRowsGroupBegin =
        blockRowsGroup * m_times + m_times / warpPerBlock * warpIdInBlock;

#pragma unroll
    for (int j = 0; (j < n_times) &&
                    (j < (n / rowThreadsPerMma - blockColsGroup * n_times));
         j++) {
        if (C_write_ptr[j] != -1) {
            int C_offset = C_write_ptr[j] * (m / elementsPerThreadPerMma) +
                           warpRowsGroupBegin * colThreadsPerMma +
                           quarterWarpId;
            for (int i = 0; i < m_times / warpPerBlock; i++) {
                Tc tc_tmp[4];
#pragma unroll
                for (int t = 0; t < 4; t++) {
                    tc_tmp[t] = __hmul(fp_cast<Tc>(C_f32[i][j][t]),
                                       thread_token_weight[j]);
                }

                if constexpr (std::is_same_v<Tc, __half>) {
                    atomicAdd(reinterpret_cast<__half2 *>(&C_ptr[C_offset]),
                              {tc_tmp[0], tc_tmp[1]});
                    atomicAdd(reinterpret_cast<__half2 *>(&C_ptr[C_offset]) + 1,
                              {tc_tmp[2], tc_tmp[3]});
                } else if constexpr (std::is_same_v<Tc, __maca_bfloat16>) {
                    atomicAdd(
                        reinterpret_cast<__maca_bfloat162 *>(&C_ptr[C_offset]),
                        {tc_tmp[0], tc_tmp[1]});
                    atomicAdd(
                        reinterpret_cast<__maca_bfloat162 *>(&C_ptr[C_offset]) +
                            1,
                        {tc_tmp[2], tc_tmp[3]});
                }
                C_offset += 4;
            }
        }
    }
}

template <typename Ta, typename Tb, typename Taccum, typename Tc,
          int BLOCK_DIM_X, int tile_m, int tile_n, int tile_k,
          int micro_batchsize>
__global__ void
fused_moe_second_group_gemm_kernel(Ta *A, Tb *B, Tc *C, int *sorted_token_ids,
                                   int *experts_ids, int m, int n, int k,
                                   Tc *routing_weights, int padded_ptr_number,
                                   int topk, int *dev_padded_num_experts) {
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
    using AInput = FusedMoeSecondGemmAInputFp16Bf16<Ta>;
    using AIterator = FusedMoeSecondGemmAIteratorFp16Bf16<AInput, HyperParams>;
    using Mma = FusedMoeSecondGemmMmaFp16Bf16<Tb, AIterator, HyperParams>;
    fused_moe_second_gemm_kernel<Ta, Tb, Taccum, Tc, HyperParams, AInput,
                                 AIterator, Mma>(
        AInput(A + experts_ids[gemm_id] * m * k), B, C,
        sorted_token_ids + gemm_id * micro_batchsize, m, n, k, routing_weights,
        padded_ptr_number, topk, block_x_id_in_gemm, block_y_id_in_gemm);
}

} // namespace muxi_layout_kernels
