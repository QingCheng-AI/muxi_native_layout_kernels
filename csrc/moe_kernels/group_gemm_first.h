#pragma once
#include "group_gemm_utils.h"

// first block gemm kernel for __device__ later
template <typename Ta, typename Tb, typename Taccum, typename Tc,
          int BLOCK_DIM_X, int microBatchsize, int APerWarp, int splitK>
__device__ void
fused_moe_first_gemm_kernel(Ta *A, Tb *B, Tc *C, int *sorted_token_ids, int m,
                            int n, int k, Taccum alpha, int topk,
                            int padded_ptr_number, int gemm_warp_id) {
    // some constexpr parameters just support FP16/BF16, if you want to support
    // fp8… , you need to adjust these parameters
    constexpr int stages = 4;
    constexpr int elementsPerAccess = 8;
    constexpr int rowThreadsPerMma = 16;
    constexpr int colThreadsPerMma = 4;
    constexpr int elementsPerThreadPerMma = 4;
    constexpr int warpPerBlock = BLOCK_DIM_X / WARP_SIZE;
    constexpr int numCycleB = microBatchsize / rowThreadsPerMma;
    constexpr int numCycleBPerWarp =
        (numCycleB + warpPerBlock - 1) / warpPerBlock;
    constexpr int sharedNumCycleB = numCycleB;
    constexpr int sharedArriveCount =
        (sharedNumCycleB + warpPerBlock - 1) / warpPerBlock;
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

    int A_offset =
        ((warpRowsGroupBegin * rowThreadsPerMma) * (k / elementsPerAccess)) +
        laneId + splitKStart * stages * WARP_SIZE;
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

    UINT4 *A_ptr[APerWarp];
    for (int i = 0; i < APerWarp; i++) {
        A_ptr[i] = reinterpret_cast<UINT4 *>(A) + A_offset +
                   i * (rowThreadsPerMma * (k / elementsPerAccess));
    }

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

    UINT4 tmpA[stages][APerWarp];
    UINT4 tmpB[numCycleB];
    FLOAT4 C_f32[numCycleB][APerWarp];

    for (int i = 0; i < numCycleB; i++) {
        for (int index_A = 0; index_A < APerWarp; index_A++) {
            C_f32[i][index_A] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    if (end == 0)
        for (int index_A = 0; index_A < APerWarp; index_A++) {
            A_ptr[index_A] = reinterpret_cast<UINT4 *>(A);
        }
    int A_ptr_offset = 0;
    int shared_ptr_offset = 0;

    // begin fill the pipeline
#pragma unroll stages
    for (int t = 0; t < stages; t++) {
        asm("/* Stop compiler reordering (begin) */");
        for (int index_A = 0; index_A < APerWarp; index_A++) {
            tmpA[t][index_A] = __builtin_mxc_load_global_async128(
                reinterpret_cast<INT128 *>(A_ptr[index_A] + A_ptr_offset));
        }
        A_ptr_offset += WARP_SIZE;
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

#pragma unroll
            for (int j = 0; j < sharedNumCycleB; j++) {
                for (int index_A = 0; index_A < APerWarp; index_A++) {
                    C_f32[j][index_A] = mma_16x16x16f16<Tb>(
                        tmpA[t][index_A][0], tmpA[t][index_A][1], tmpB[j][0],
                        tmpB[j][1], C_f32[j][index_A]);
                    C_f32[j][index_A] = mma_16x16x16f16<Tb>(
                        tmpA[t][index_A][2], tmpA[t][index_A][3], tmpB[j][2],
                        tmpB[j][3], C_f32[j][index_A]);
                }
            }
            __builtin_mxc_barrier_inst();
            for (int index_A = 0; index_A < APerWarp; index_A++) {
                tmpA[t][index_A] = __builtin_mxc_load_global_async128(
                    reinterpret_cast<INT128 *>(A_ptr[index_A] + A_ptr_offset));
            }
            A_ptr_offset += WARP_SIZE;
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

#pragma unroll
        for (int j = 0; j < sharedNumCycleB; j++) {
            for (int index_A = 0; index_A < APerWarp; index_A++) {
                C_f32[j][index_A] = mma_16x16x16f16<Tb>(
                    tmpA[t][index_A][0], tmpA[t][index_A][1], tmpB[j][0],
                    tmpB[j][1], C_f32[j][index_A]);
                C_f32[j][index_A] = mma_16x16x16f16<Tb>(
                    tmpA[t][index_A][2], tmpA[t][index_A][3], tmpB[j][2],
                    tmpB[j][3], C_f32[j][index_A]);
            }
        }
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
            Tc C_tc_tmp[4] = {0};
#pragma unroll 4
            for (int t = 0; t < 4; t++) {
                C_tc_tmp[t] = static_cast<Tc>(C_f32_res[t]);
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
    fused_moe_first_gemm_kernel<Ta, Tb, Taccum, Tc, BLOCK_DIM_X, microBatchsize,
                                APerWarp, splitK>(
        A + experts_ids[gemm_id] * m * k, B, C,
        sorted_token_ids + gemm_id * microBatchsize, m, n, k, alpha, topk,
        padded_ptr_number, gemm_warp_id);
}
