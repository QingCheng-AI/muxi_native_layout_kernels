#pragma once

#include "../utils.cuh"
#include "group_gemm_utils.h"

namespace muxi_layout_kernels {

template <int _BLOCK_DIM_X, int _microBatchsize, int _APerWarp, int _splitK,
          int _stages = 4, int _elementsPerAccess = 8,
          int _rowThreadsPerMma = 16, int _colThreadsPerMma = 4,
          int _elementsPerThreadPerMma = 4>
    requires(_BLOCK_DIM_X % WARP_SIZE == 0 &&
             _microBatchsize % _rowThreadsPerMma == 0)
struct FusedMoeFirstGemmKernelHyperParams {
    static constexpr int BLOCK_DIM_X = _BLOCK_DIM_X;
    static constexpr int microBatchsize = _microBatchsize;
    static constexpr int APerWarp = _APerWarp;
    static constexpr int splitK = _splitK;
    static constexpr int stages = _stages;
    static constexpr int elementsPerAccess = _elementsPerAccess;
    static constexpr int rowThreadsPerMma = _rowThreadsPerMma;
    static constexpr int colThreadsPerMma = _colThreadsPerMma;
    static constexpr int elementsPerThreadPerMma = _elementsPerThreadPerMma;

    static constexpr int warpPerBlock = BLOCK_DIM_X / WARP_SIZE;
    static constexpr int numCycleB = microBatchsize / rowThreadsPerMma;
    static constexpr int numCycleBPerWarp =
        (numCycleB + warpPerBlock - 1) / warpPerBlock;
    static constexpr int sharedNumCycleB = numCycleB;
    static constexpr int sharedArriveCount =
        (sharedNumCycleB + warpPerBlock - 1) / warpPerBlock;
};

template <typename _Ta> struct FusedMoeFirstGemmAInputFp16Bf16 {
    using Ta = _Ta;
    Ta *A;

    __device__ __forceinline__ FusedMoeFirstGemmAInputFp16Bf16(Ta *A) : A(A) {}
};

template <typename...>
struct isFusedMoeFirstGemmAInputFp16Bf16 : std::false_type {};
template <typename Ta>
struct isFusedMoeFirstGemmAInputFp16Bf16<FusedMoeFirstGemmAInputFp16Bf16<Ta>>
    : std::true_type {};

template <typename AInput, typename HyperParams>
    requires(isFusedMoeFirstGemmAInputFp16Bf16<AInput>::value)
struct FusedMoeFirstGemmAIteratorFp16Bf16 {
    using Ta = typename AInput::Ta;
    static constexpr int APerWarp = HyperParams::APerWarp;
    static constexpr int stages = HyperParams::stages;
    static constexpr int elementsPerAccess = HyperParams::elementsPerAccess;
    static constexpr int rowThreadsPerMma = HyperParams::rowThreadsPerMma;

    UINT4 *A_ptr[APerWarp];
    UINT4 tmpA[stages][APerWarp];
    int A_ptr_offset = 0;

    __device__ __forceinline__ void init_ptr(AInput &A_input, int k,
                                             int warpRowsGroupBegin, int laneId,
                                             int quarterWarpId,
                                             int splitKStart) {
        int A_offset = ((warpRowsGroupBegin * rowThreadsPerMma) *
                        (k / elementsPerAccess)) +
                       laneId + splitKStart * stages * WARP_SIZE;
        for (int i = 0; i < APerWarp; i++) {
            A_ptr[i] = reinterpret_cast<UINT4 *>(A_input.A) + A_offset +
                       i * (rowThreadsPerMma * (k / elementsPerAccess));
        }
    }

    __device__ __forceinline__ void init_ptr_neutral(AInput &A_input) {
        for (int i = 0; i < APerWarp; i++) {
            A_ptr[i] = reinterpret_cast<UINT4 *>(A_input.A);
        }
    }

    __device__ __forceinline__ void load_and_advance(int stage_id) {
        for (int i = 0; i < APerWarp; i++) {
            tmpA[stage_id][i] = __builtin_mxc_load_global_async128(
                reinterpret_cast<INT128 *>(A_ptr[i] + A_ptr_offset));
        }
        A_ptr_offset += WARP_SIZE;
    }
};

template <typename...>
struct isFusedMoeFirstGemmAIteratorFp16Bf16 : std::false_type {};
template <typename AInput, typename HyperParams>
struct isFusedMoeFirstGemmAIteratorFp16Bf16<
    FusedMoeFirstGemmAIteratorFp16Bf16<AInput, HyperParams>> : std::true_type {
};

template <typename Tb, typename AIter, typename HyperParams>
    requires(isFusedMoeFirstGemmAIteratorFp16Bf16<AIter>::value)
struct FusedMoeFirstGemmMmaFp16Bf16 {
    static constexpr int APerWarp = HyperParams::APerWarp;
    static constexpr int numCycleB = HyperParams::numCycleB;
    static constexpr int sharedNumCycleB = HyperParams::sharedNumCycleB;

    __device__ __forceinline__ void
    operator()(AIter &A_iter, UINT4 tmpB[numCycleB],
               FLOAT4 C_f32[numCycleB][APerWarp], int stage_id) {
#pragma unroll
        for (int j = 0; j < sharedNumCycleB; j++) {
            for (int i = 0; i < APerWarp; i++) {
                C_f32[j][i] = mma_16x16x16f16<Tb>(
                    A_iter.tmpA[stage_id][i][0], A_iter.tmpA[stage_id][i][1],
                    tmpB[j][0], tmpB[j][1], C_f32[j][i]);
                C_f32[j][i] = mma_16x16x16f16<Tb>(
                    A_iter.tmpA[stage_id][i][2], A_iter.tmpA[stage_id][i][3],
                    tmpB[j][2], tmpB[j][3], C_f32[j][i]);
            }
        }
    }
};

template <typename Ta, typename Tb, typename Taccum, typename Tc,
          typename HyperParams, typename AInput, typename AIterator,
          typename Mma>
__device__ void
fused_moe_first_gemm_kernel(AInput A_input, Tb *B, Tc *C, int *sorted_token_ids,
                            int m, int n, int k, Taccum alpha, int topk,
                            int padded_ptr_number, int gemm_warp_id) {
    constexpr int microBatchsize = HyperParams::microBatchsize;
    constexpr int APerWarp = HyperParams::APerWarp;
    constexpr int splitK = HyperParams::splitK;
    constexpr int stages = HyperParams::stages;
    constexpr int elementsPerAccess = HyperParams::elementsPerAccess;
    constexpr int rowThreadsPerMma = HyperParams::rowThreadsPerMma;
    constexpr int colThreadsPerMma = HyperParams::colThreadsPerMma;
    constexpr int elementsPerThreadPerMma =
        HyperParams::elementsPerThreadPerMma;
    constexpr int warpPerBlock = HyperParams::warpPerBlock;
    constexpr int numCycleB = HyperParams::numCycleB;
    constexpr int numCycleBPerWarp = HyperParams::numCycleBPerWarp;
    constexpr int sharedNumCycleB = HyperParams::sharedNumCycleB;
    constexpr int sharedArriveCount = HyperParams::sharedArriveCount;

    const int rowsGroup = m / rowThreadsPerMma / APerWarp;

    using CStgType = __NATIVE_VECTOR__(sizeof(Tc), uint);

    int numWarps = __builtin_mxc_readfirstlane(m / 16 / APerWarp);
    int warpId = __builtin_mxc_readfirstlane(gemm_warp_id % numWarps);
    int splitKId = __builtin_mxc_readfirstlane(
        (((blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE) / numWarps) %
        splitK);
    int warpIdInBlock = (threadIdx.x / WARP_SIZE);
    int laneId = threadIdx.x & (WARP_SIZE - 1);
    int quarterWarpId = laneId / 16;
    int quarterLaneId = laneId & (16 - 1);

    int warpRowsGroupBegin =
        __builtin_mxc_readfirstlane(warpId * (rowsGroup / numWarps) +
                                    min(warpId, rowsGroup % numWarps)) *
        APerWarp;
    int numCycleA =
        __builtin_mxc_readfirstlane(k / (colThreadsPerMma * elementsPerAccess));

    int splitKStart = __builtin_mxc_readfirstlane(
        splitKId * (numCycleA / stages / splitK) +
        min(splitKId, (numCycleA / stages) % splitK));
    int splitKEnd = __builtin_mxc_readfirstlane(
        (splitKId + 1) * (numCycleA / stages / splitK) +
        min(splitKId + 1, (numCycleA / stages) % splitK));
    int end = __builtin_mxc_readfirstlane(splitKEnd - splitKStart);

    // modified here to support moe group gemm.
    // A ptr is no need modified, B ptr is need modified.

    int B_offset_in_token =
        quarterWarpId + splitKStart * stages * colThreadsPerMma;
    int C_offset[numCycleB];
    int C_write_flags[numCycleB];
    for (int i = 0; i < numCycleB; i++) {
        int thread_tokens_ptr =
            sorted_token_ids[i * rowThreadsPerMma + quarterLaneId];
        C_write_flags[i] = 1;
        if (thread_tokens_ptr >= padded_ptr_number) {
            thread_tokens_ptr = 0;
            C_write_flags[i] = 0;
        }
        C_offset[i] = thread_tokens_ptr * (m / elementsPerThreadPerMma) +
                      warpRowsGroupBegin * colThreadsPerMma + quarterWarpId;
    }

    AIterator A_iter;
    A_iter.init_ptr(A_input, k, warpRowsGroupBegin, laneId, quarterWarpId,
                    splitKStart);

    ////// maybe has bugs now, should debug first
    UINT4 *B_ptr_tokens[numCycleBPerWarp];
    for (int i = 0; i < numCycleBPerWarp; i++) {
        int thread_tokens_ptr = 0;
        if ((i * warpPerBlock + warpIdInBlock) * rowThreadsPerMma +
                quarterLaneId <
            microBatchsize) {
            thread_tokens_ptr =
                sorted_token_ids[(i * warpPerBlock + warpIdInBlock) *
                                     rowThreadsPerMma +
                                 quarterLaneId];
        }
        if (thread_tokens_ptr >= padded_ptr_number) {
            thread_tokens_ptr = 0;
        }
        thread_tokens_ptr = thread_tokens_ptr / topk;
        B_ptr_tokens[i] = reinterpret_cast<UINT4 *>(B) +
                          thread_tokens_ptr * (k / elementsPerAccess);
    }
    ////// maybe has bugs now, should debug first

    CStgType *C_ptr = reinterpret_cast<CStgType *>(C);

    __shared__ UINT4 sharedTmpB[stages][numCycleB][WARP_SIZE];
    UINT4 *shared_ptr =
        reinterpret_cast<UINT4 *>(&sharedTmpB[0][0][0]) + laneId;

    UINT4 tmpB[numCycleB];
    FLOAT4 C_f32[numCycleB][APerWarp];
    Mma mma;

    for (int i = 0; i < numCycleB; i++) {
        for (int index_A = 0; index_A < APerWarp; index_A++) {
            C_f32[i][index_A] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    if (end == 0) {
        A_iter.init_ptr_neutral(A_input);
    }
    int shared_ptr_offset = 0;

    // begin fill the pipeline
#pragma unroll stages
    for (int t = 0; t < stages; t++) {
        asm("/* Stop compiler reordering (begin) */");
        A_iter.load_and_advance(t);
#pragma unroll
        for (int j = 0; j < numCycleBPerWarp; j += 1) {
            LDG_B128_BSM_WITH_PREDICATOR_NORET0(
                (shared_ptr + shared_ptr_offset +
                 (j * warpPerBlock + warpIdInBlock) * WARP_SIZE),
                (B_ptr_tokens[j] + B_offset_in_token),
                (j * warpPerBlock + warpIdInBlock), numCycleB, MACA_ICMP_SLT);
        }
        shared_ptr_offset += sharedNumCycleB * WARP_SIZE;
        B_offset_in_token += colThreadsPerMma;
    }

    // pipeline main loop
    for (int i = 0; i < end - 1; i++) {
        shared_ptr_offset = 0;
#pragma unroll stages
        for (int t = 0; t < stages; t++) {
            asm("/* Stop compiler reordering (loop) */");
            __builtin_mxc_arrive_gvmcnt((sharedArriveCount + APerWarp) *
                                        (stages - 1));
            __builtin_mxc_barrier_inst();
#pragma unroll
            for (int j = 0; j < sharedNumCycleB; j++) {
                tmpB[j] = sharedTmpB[t][j][laneId];
            }

            mma(A_iter, tmpB, C_f32, t);
            __builtin_mxc_barrier_inst();
            A_iter.load_and_advance(t);
#pragma unroll
            for (int j = 0; j < sharedNumCycleB; j++) {
                LDG_B128_BSM_WITH_PREDICATOR_NORET0(
                    (shared_ptr + shared_ptr_offset +
                     (j * warpPerBlock + warpIdInBlock) * WARP_SIZE),
                    (B_ptr_tokens[j] + B_offset_in_token),
                    (j * warpPerBlock + warpIdInBlock), numCycleB,
                    MACA_ICMP_SLT);
            }
            shared_ptr_offset += sharedNumCycleB * WARP_SIZE;
            B_offset_in_token += colThreadsPerMma;
        }
    }

    // pipeline end loop
#pragma unroll stages
    for (int t = 0; t < stages; t++) {
        asm("/* Stop compiler reordering (end loop) */");
        if (t == 0) {
            __builtin_mxc_arrive_gvmcnt((sharedArriveCount + 1) * (stages - 1));
            __builtin_mxc_barrier_inst();
        } else if (t == 1) {
            __builtin_mxc_arrive_gvmcnt((sharedArriveCount + 1) * (stages - 2));
            __builtin_mxc_barrier_inst();
        } else if (t == 2) {
            __builtin_mxc_arrive_gvmcnt((sharedArriveCount + 1) * (stages - 3));
            __builtin_mxc_barrier_inst();
        } else if (t == 3) {
            __builtin_mxc_arrive_gvmcnt((sharedArriveCount + 1) * (stages - 4));
            __builtin_mxc_barrier_inst();
        }

#pragma unroll
        for (int j = 0; j < sharedNumCycleB; j++) {
            tmpB[j] = sharedTmpB[t][j][laneId];
        }

        mma(A_iter, tmpB, C_f32, t);
    }
    if (end == 0)
        return;

#pragma unroll 4
    for (int j = 0; j < sharedNumCycleB; j++) {
        if (C_write_flags[j] == 0)
            continue;
        for (int index_A = 0; index_A < APerWarp; index_A++) {
            float C_f32_res[4];
#pragma unroll 4
            for (int t = 0; t < 4; t++) {
                C_f32_res[t] = C_f32[j][index_A][t] * alpha;
            }
            Tc C_tc_tmp[4];
#pragma unroll 4
            for (int t = 0; t < 4; t++) {
                C_tc_tmp[t] = fp_cast<Tc>(C_f32_res[t]);
            }
            if constexpr (splitK > 1) {
                if constexpr (std::is_same_v<Tc, __half>) {
                    atomicAdd(reinterpret_cast<__half2 *>(&C_ptr[C_offset[j]]),
                              {C_tc_tmp[0], C_tc_tmp[1]});
                    atomicAdd(reinterpret_cast<__half2 *>(&C_ptr[C_offset[j]]) +
                                  1,
                              {C_tc_tmp[2], C_tc_tmp[3]});
                } else if constexpr (std::is_same_v<Tc, __maca_bfloat16>) {
                    atomicAdd(reinterpret_cast<__maca_bfloat162 *>(
                                  &C_ptr[C_offset[j]]),
                              {C_tc_tmp[0], C_tc_tmp[1]});
                    atomicAdd(reinterpret_cast<__maca_bfloat162 *>(
                                  &C_ptr[C_offset[j]]) +
                                  1,
                              {C_tc_tmp[2], C_tc_tmp[3]});
                }
            } else {
                C_ptr[C_offset[j]] =
                    *reinterpret_cast<CStgType *>(&C_tc_tmp[0]);
            }

            C_offset[j] += colThreadsPerMma;
        }
    }
}

// first group gemm __global__ interface
template <typename Ta, typename Tb, typename Taccum, typename Tc,
          int BLOCK_DIM_X, int microBatchsize, int APerWarp, int splitK>
__global__ void __launch_bounds__(BLOCK_DIM_X)
    fused_moe_first_group_gemm_kernel(Ta *A, Tb *B, Tc *C,
                                      int *sorted_token_ids, int *experts_ids,
                                      int m, int n, int k, Taccum alpha,
                                      int topk, int padded_ptr_number,
                                      int *dev_padded_num_experts) {
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warps_per_gemm = (m / 16 / APerWarp * splitK);
    int gemm_id = global_warp_id / warps_per_gemm;
    int gemm_warp_id = global_warp_id % warps_per_gemm;
    if (gemm_id >= dev_padded_num_experts[0])
        return;
    using HyperParams =
        FusedMoeFirstGemmKernelHyperParams<BLOCK_DIM_X, microBatchsize,
                                           APerWarp, splitK>;
    using AInput = FusedMoeFirstGemmAInputFp16Bf16<Ta>;
    using AIterator = FusedMoeFirstGemmAIteratorFp16Bf16<AInput, HyperParams>;
    using Mma = FusedMoeFirstGemmMmaFp16Bf16<Tb, AIterator, HyperParams>;
    fused_moe_first_gemm_kernel<Ta, Tb, Taccum, Tc, HyperParams, AInput,
                                AIterator, Mma>(
        AInput(A + experts_ids[gemm_id] * m * k), B, C,
        sorted_token_ids + gemm_id * microBatchsize, m, n, k, alpha, topk,
        padded_ptr_number, gemm_warp_id);
}

} // namespace muxi_layout_kernels
