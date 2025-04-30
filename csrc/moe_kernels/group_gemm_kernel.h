#pragma once

#include <mc_runtime.h>

#include "group_gemm_utils.h"

template <typename Tab, typename Taccum, typename Tc, int BLOCK_DIM_X, int N,
          int APerWarp, int splitN, int splitK, bool IsBetaZero,
          bool HasOneDimBias = false>
__device__ void GemmMmaLayoutABCReuseAKernel(Tab *A, Tab *B, Tc *C, int m,
                                             int n, int k, Taccum alpha,
                                             Taccum beta, Tc *bias,
                                             int gemm_warpId) {
    constexpr int warpPerBlock = BLOCK_DIM_X / WARPSIZE;
    constexpr int numCycleB = N / ROWTHREADSPERMMA;
    constexpr int sharedNumCycleB = (numCycleB + splitN - 1) / splitN;
    constexpr int sharedArriveCount =
        (sharedNumCycleB + warpPerBlock - 1) / warpPerBlock;
    const int rowsGroup = m / ROWTHREADSPERMMA / APerWarp;

    using CStgType = __NATIVE_VECTOR__(sizeof(Tc), uint);

    int numWarps = __builtin_mxc_readfirstlane(m / 16 / APerWarp);
    int warpId = __builtin_mxc_readfirstlane(gemm_warpId) % numWarps;
    int splitNId =
        __builtin_mxc_readfirstlane((gemm_warpId / numWarps) % splitN);
    int splitKId =
        __builtin_mxc_readfirstlane((gemm_warpId / numWarps / splitN) % splitK);
    int warpIdInBlock = (threadIdx.x / WARPSIZE);
    int laneId = threadIdx.x & (WARPSIZE - 1);
    int quarterWarpId = laneId / 16;
    int quarterLaneId = laneId & (16 - 1);

    int warpRowsGroupBegin =
        __builtin_mxc_readfirstlane(warpId * (rowsGroup / numWarps) +
                                    min(warpId, rowsGroup % numWarps)) *
        APerWarp;
    int warpRowsGroupEnd =
        __builtin_mxc_readfirstlane((warpId + 1) * (rowsGroup / numWarps) +
                                    min(warpId + 1, rowsGroup % numWarps)) *
        APerWarp;
    int numCycleA =
        __builtin_mxc_readfirstlane(k / (COLTHREADSPERMMA * ELEMENTSPERACCESS));

    int splitNStart = __builtin_mxc_readfirstlane(
        splitNId * (numCycleB / splitN) + min(splitNId, (numCycleB % splitN)));
    int splitNEnd =
        __builtin_mxc_readfirstlane((splitNId + 1) * (numCycleB / splitN) +
                                    min(splitNId + 1, (numCycleB % splitN)));
    int splitKStart = __builtin_mxc_readfirstlane(
        splitKId * (numCycleA / STAGES / splitK) +
        min(splitKId, (numCycleA / STAGES) % splitK));
    int splitKEnd = __builtin_mxc_readfirstlane(
        (splitKId + 1) * (numCycleA / STAGES / splitK) +
        min(splitKId + 1, (numCycleA / STAGES) % splitK));
    int end = __builtin_mxc_readfirstlane(splitKEnd - splitKStart);

    int A_offset =
        ((warpRowsGroupBegin * ROWTHREADSPERMMA) * (k / ELEMENTSPERACCESS)) +
        laneId + splitKStart * STAGES * WARPSIZE;
    int B_offset = laneId + splitNStart * WARPSIZE +
                   splitKStart * STAGES * numCycleB * WARPSIZE;
    int warpStoreOffset =
        ((quarterWarpId > 1 ? (quarterWarpId + 30) : quarterWarpId)) +
        quarterLaneId * 2;
    int C_offset[APerWarp];
    for (int i = 0; i < APerWarp; i++) {
        C_offset[i] = ((warpRowsGroupBegin + i) / 2) *
                          (COLTHREADSPERMMA * ELEMENTSPERACCESS *
                           ROWTHREADSPERMMA / ELEMENTSPERTHREADPERMMA) *
                          numCycleB +
                      warpStoreOffset +
                      ((warpRowsGroupBegin + i) % 2) * WARPSIZE +
                      splitNStart * 2 * WARPSIZE;
    }

    UINT4 *A_ptr[APerWarp];
    for (int i = 0; i < APerWarp; i++) {
        A_ptr[i] = reinterpret_cast<UINT4 *>(A) + A_offset +
                   i * (ROWTHREADSPERMMA * (k / ELEMENTSPERACCESS));
    }
    UINT4 *B_ptr = reinterpret_cast<UINT4 *>(B) + B_offset;
    CStgType *C_ptr = reinterpret_cast<CStgType *>(C);

    __shared__ UINT4 sharedTmpB[STAGES][sharedNumCycleB][64];
    UINT4 *shared_ptr =
        reinterpret_cast<UINT4 *>(&sharedTmpB[0][0][0]) + laneId;

    int splitNumCycleB = __builtin_mxc_readfirstlane(splitNEnd - splitNStart);

    UINT4 tmpA[STAGES][APerWarp];
    UINT4 tmpB[sharedNumCycleB];
    FLOAT4 C_f32[sharedNumCycleB][APerWarp];

    for (int i = 0; i < sharedNumCycleB; i++) {
        for (int index_A = 0; index_A < APerWarp; index_A++) {
            C_f32[i][index_A] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    if (end == 0)
        for (int index_A = 0; index_A < splitNumCycleB; index_A++) {
            A_ptr[index_A] = reinterpret_cast<UINT4 *>(A);
        }
    int A_ptr_offset = 0;
    int B_ptr_offset = 0;
    int shared_ptr_offset = 0;

// begin fill the pipeline
#pragma unroll STAGES
    for (int t = 0; t < STAGES; t++) {
        asm("/* Stop compiler reordering (begin) */");
        for (int index_A = 0; index_A < APerWarp; index_A++) {
            tmpA[t][index_A] = __builtin_mxc_load_global_async128(
                reinterpret_cast<INT128 *>(A_ptr[index_A] + A_ptr_offset));
        }
        A_ptr_offset += WARPSIZE;
#pragma unroll
        for (int j = 0; j < sharedNumCycleB; j += warpPerBlock) {
            LDG_B128_BSM_WITH_PREDICATOR_NORET0(
                (shared_ptr + shared_ptr_offset +
                 (j + warpIdInBlock) * WARPSIZE),
                (B_ptr + B_ptr_offset + (j + warpIdInBlock) * WARPSIZE),
                (j + warpIdInBlock), splitNumCycleB, MACA_ICMP_SLT);
        }
        shared_ptr_offset += sharedNumCycleB * WARPSIZE;
        B_ptr_offset += numCycleB * WARPSIZE;
    }

    // pipeline main loop
    for (int i = 0; i < end - 1; i++) {
        shared_ptr_offset = 0;
#pragma unroll STAGES
        for (int t = 0; t < STAGES; t++) {
            asm("/* Stop compiler reordering (loop) */");
            __builtin_mxc_arrive_gvmcnt((sharedArriveCount + APerWarp) *
                                        (STAGES - 1));
            __builtin_mxc_barrier_inst();
#pragma unroll
            for (int j = 0; j < sharedNumCycleB; j++) {
                tmpB[j] = sharedTmpB[t][j][laneId];
            }

#pragma unroll
            for (int j = 0; j < sharedNumCycleB; j++) {
                for (int index_A = 0; index_A < APerWarp; index_A++) {
                    C_f32[j][index_A] = mma_16x16x16f16<Tab>(
                        tmpA[t][index_A][0], tmpA[t][index_A][1], tmpB[j][0],
                        tmpB[j][1], C_f32[j][index_A]);
                    C_f32[j][index_A] = mma_16x16x16f16<Tab>(
                        tmpA[t][index_A][2], tmpA[t][index_A][3], tmpB[j][2],
                        tmpB[j][3], C_f32[j][index_A]);
                }
            }
            __builtin_mxc_barrier_inst();
            for (int index_A = 0; index_A < APerWarp; index_A++) {
                tmpA[t][index_A] = __builtin_mxc_load_global_async128(
                    reinterpret_cast<INT128 *>(A_ptr[index_A] + A_ptr_offset));
            }
            A_ptr_offset += WARPSIZE;
#pragma unroll
            for (int j = 0; j < sharedNumCycleB; j += warpPerBlock) {
                LDG_B128_BSM_WITH_PREDICATOR_NORET0(
                    (shared_ptr + shared_ptr_offset +
                     (j + warpIdInBlock) * WARPSIZE),
                    (B_ptr + B_ptr_offset + (j + warpIdInBlock) * WARPSIZE),
                    (j + warpIdInBlock), splitNumCycleB, MACA_ICMP_SLT);
            }
            shared_ptr_offset += sharedNumCycleB * WARPSIZE;
            B_ptr_offset += numCycleB * WARPSIZE;
        }
    }

// pipeline end loop
#pragma unroll STAGES
    for (int t = 0; t < STAGES; t++) {
        asm("/* Stop compiler reordering (end loop) */");
        if (t == 0) {
            __builtin_mxc_arrive_gvmcnt((sharedArriveCount + 1) * (STAGES - 1));
            __builtin_mxc_barrier_inst();
        } else if (t == 1) {
            __builtin_mxc_arrive_gvmcnt((sharedArriveCount + 1) * (STAGES - 2));
            __builtin_mxc_barrier_inst();
        } else if (t == 2) {
            __builtin_mxc_arrive_gvmcnt((sharedArriveCount + 1) * (STAGES - 3));
            __builtin_mxc_barrier_inst();
        } else if (t == 3) {
            __builtin_mxc_arrive_gvmcnt((sharedArriveCount + 1) * (STAGES - 4));
            __builtin_mxc_barrier_inst();
        }

#pragma unroll
        for (int j = 0; j < sharedNumCycleB; j++) {
            tmpB[j] = sharedTmpB[t][j][laneId];
        }

#pragma unroll
        for (int j = 0; j < sharedNumCycleB; j++) {
            for (int index_A = 0; index_A < APerWarp; index_A++) {
                C_f32[j][index_A] = mma_16x16x16f16<Tab>(
                    tmpA[t][index_A][0], tmpA[t][index_A][1], tmpB[j][0],
                    tmpB[j][1], C_f32[j][index_A]);
                C_f32[j][index_A] = mma_16x16x16f16<Tab>(
                    tmpA[t][index_A][2], tmpA[t][index_A][3], tmpB[j][2],
                    tmpB[j][3], C_f32[j][index_A]);
            }
        }
    }
    if (end == 0)
        return;

    for (int index_A = 0; index_A < APerWarp; index_A++) {
        CStgType bias_load;
        if constexpr (HasOneDimBias) {
            if (splitKId == 0) {
                int bias_offset =
                    (warpRowsGroupBegin + index_A) * COLTHREADSPERMMA +
                    quarterWarpId;
                bias_load = (reinterpret_cast<CStgType *>(bias))[bias_offset];
            }
        }
        for (int j = 0; j < sharedNumCycleB; j++) {
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

            if constexpr (HasOneDimBias) {
                if (splitKId == 0) {
                    Tc *bias_tc = reinterpret_cast<Tc *>(&bias_load);
#pragma unroll 4
                    for (int t = 0; t < 4; t++) {
                        C_tc_tmp[t] = __hadd(C_tc_tmp[t], bias_tc[t]);
                    }
                }
            }

            if (j >= splitNumCycleB) {
                break;
            }

            if constexpr (std::is_same_v<Tc, __half>) {
                atomicAdd(
                    reinterpret_cast<__half2 *>(&C_ptr[C_offset[index_A]]),
                    {C_tc_tmp[0], C_tc_tmp[1]});
                atomicAdd(
                    reinterpret_cast<__half2 *>(&C_ptr[C_offset[index_A]]) + 1,
                    {C_tc_tmp[2], C_tc_tmp[3]});
            } else if constexpr (std::is_same_v<Tc, __maca_bfloat16>) {
                atomicAdd(reinterpret_cast<__maca_bfloat162 *>(
                              &C_ptr[C_offset[index_A]]),
                          {C_tc_tmp[0], C_tc_tmp[1]});
                atomicAdd(reinterpret_cast<__maca_bfloat162 *>(
                              &C_ptr[C_offset[index_A]]) +
                              1,
                          {C_tc_tmp[2], C_tc_tmp[3]});
            }

            C_offset[index_A] += WARPSIZE * 2;
        }
    }
}

template <typename Tab, typename Taccum, typename Tc, int BLOCK_DIM_X,
          int APerWarp, int splitN, int splitK, int N>
__device__ void
GemmMmaLayoutABCReuseAKernelDispatch(Tab *A, Tab *B, Tc *C, int m, int n, int k,
                                     Taccum alpha, Taccum beta, Tc *dev_bias,
                                     int gemm_warpId) {
    bool isBetaZero = (beta == static_cast<Taccum>(0));
    bool hasOneDimBias = !(dev_bias == nullptr);

#define LAUNCH_GEMM_OPT_KERNEL(IS_BETA_ZERO, HASONEDIMBIAS)                    \
    GemmMmaLayoutABCReuseAKernel<Tab, Taccum, Tc, BLOCK_DIM_X, N, APerWarp,    \
                                 splitN, splitK, true, false>(                 \
        A, B, C, m, n, k, alpha, beta, dev_bias, gemm_warpId);

    LAUNCH_GEMM_OPT_KERNEL(isBetaZero, hasOneDimBias);

#undef LAUNCH_GEMM_OPT_KERNEL
}

template <typename Tab, typename Taccum, typename Tc, int BLOCK_DIM_X, int N,
          int APerWarp, int splitN, int splitK, bool IsBetaZero,
          bool HasOneDimBias = false>
__device__ void GemmMmaLayoutAB_ContinuousCReuseA_and_mul_weights_Kernel(
    Tab *A, Tab *B, Tc *C, int m, int n, int k, Tc *score_weights, Taccum beta,
    Tc *bias, int gemm_warpId) {
    constexpr int stages = 4;
    constexpr int elementsPerAccess = 8;
    constexpr int rowThreadsPerMma = 16;
    constexpr int colThreadsPerMma = 4;
    constexpr int elementsPerThreadPerMma = 4;
    constexpr int warpPerBlock = BLOCK_DIM_X / WARP_SIZE;
    constexpr int numCycleB = N / rowThreadsPerMma;
    constexpr int sharedNumCycleB = (numCycleB + splitN - 1) / splitN;
    constexpr int sharedArriveCount =
        (sharedNumCycleB + warpPerBlock - 1) / warpPerBlock;
    const int rowsGroup = m / rowThreadsPerMma / APerWarp;

    using CStgType = __NATIVE_VECTOR__(sizeof(Tc), uint);

    int numWarps = __builtin_mxc_readfirstlane(m / 16 / APerWarp);
    int warpId = __builtin_mxc_readfirstlane(gemm_warpId) % numWarps;
    int splitNId =
        __builtin_mxc_readfirstlane((gemm_warpId / numWarps) % splitN);
    int splitKId =
        __builtin_mxc_readfirstlane((gemm_warpId / numWarps / splitN) % splitK);
    int warpIdInBlock = (threadIdx.x / WARP_SIZE);
    int laneId = threadIdx.x & (WARP_SIZE - 1);
    int quarterWarpId = laneId / 16;
    int quarterLaneId = laneId & (16 - 1);

    int warpRowsGroupBegin =
        __builtin_mxc_readfirstlane(warpId * (rowsGroup / numWarps) +
                                    min(warpId, rowsGroup % numWarps)) *
        APerWarp;
    int warpRowsGroupEnd =
        __builtin_mxc_readfirstlane((warpId + 1) * (rowsGroup / numWarps) +
                                    min(warpId + 1, rowsGroup % numWarps)) *
        APerWarp;
    int numCycleA =
        __builtin_mxc_readfirstlane(k / (colThreadsPerMma * elementsPerAccess));

    int splitNStart = __builtin_mxc_readfirstlane(
        splitNId * (numCycleB / splitN) + min(splitNId, (numCycleB % splitN)));
    int splitNEnd =
        __builtin_mxc_readfirstlane((splitNId + 1) * (numCycleB / splitN) +
                                    min(splitNId + 1, (numCycleB % splitN)));
    int splitKStart = __builtin_mxc_readfirstlane(
        splitKId * (numCycleA / stages / splitK) +
        min(splitKId, (numCycleA / stages) % splitK));
    int splitKEnd = __builtin_mxc_readfirstlane(
        (splitKId + 1) * (numCycleA / stages / splitK) +
        min(splitKId + 1, (numCycleA / stages) % splitK));
    int end = __builtin_mxc_readfirstlane(splitKEnd - splitKStart);

    int A_offset =
        ((warpRowsGroupBegin * rowThreadsPerMma) * (k / elementsPerAccess)) +
        laneId + splitKStart * stages * WARP_SIZE;
    int B_offset = laneId + splitNStart * WARP_SIZE +
                   splitKStart * stages * numCycleB * WARP_SIZE;
    int C_offset[APerWarp];
    for (int i = 0; i < APerWarp; i++) {
        C_offset[i] = (quarterLaneId + splitNStart * rowThreadsPerMma) *
                          (m / elementsPerThreadPerMma) +
                      (warpRowsGroupBegin + i) * colThreadsPerMma +
                      quarterWarpId;
    }

    UINT4 *A_ptr[APerWarp];
    for (int i = 0; i < APerWarp; i++) {
        A_ptr[i] = reinterpret_cast<UINT4 *>(A) + A_offset +
                   i * (rowThreadsPerMma * (k / elementsPerAccess));
    }
    UINT4 *B_ptr = reinterpret_cast<UINT4 *>(B) + B_offset;
    CStgType *C_ptr = reinterpret_cast<CStgType *>(C);

    __shared__ UINT4 sharedTmpB[stages][sharedNumCycleB][64];
    UINT4 *shared_ptr =
        reinterpret_cast<UINT4 *>(&sharedTmpB[0][0][0]) + laneId;

    int splitNumCycleB = __builtin_mxc_readfirstlane(splitNEnd - splitNStart);

    UINT4 tmpA[stages][APerWarp];
    UINT4 tmpB[sharedNumCycleB];
    FLOAT4 C_f32[sharedNumCycleB][APerWarp];

    for (int i = 0; i < sharedNumCycleB; i++) {
        for (int index_A = 0; index_A < APerWarp; index_A++) {
            C_f32[i][index_A] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    if (end == 0)
        for (int index_A = 0; index_A < splitNumCycleB; index_A++) {
            A_ptr[index_A] = reinterpret_cast<UINT4 *>(A);
        }
    int A_ptr_offset = 0;
    int B_ptr_offset = 0;
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
        for (int j = 0; j < sharedNumCycleB; j += warpPerBlock) {
            LDG_B128_BSM_WITH_PREDICATOR_NORET0(
                (shared_ptr + shared_ptr_offset +
                 (j + warpIdInBlock) * WARP_SIZE),
                (B_ptr + B_ptr_offset + (j + warpIdInBlock) * WARP_SIZE),
                (j + warpIdInBlock), splitNumCycleB, MACA_ICMP_SLT);
        }
        shared_ptr_offset += sharedNumCycleB * WARP_SIZE;
        B_ptr_offset += numCycleB * WARP_SIZE;
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
                    C_f32[j][index_A] = mma_16x16x16f16<Tab>(
                        tmpA[t][index_A][0], tmpA[t][index_A][1], tmpB[j][0],
                        tmpB[j][1], C_f32[j][index_A]);
                    C_f32[j][index_A] = mma_16x16x16f16<Tab>(
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
            for (int j = 0; j < sharedNumCycleB; j += warpPerBlock) {
                LDG_B128_BSM_WITH_PREDICATOR_NORET0(
                    (shared_ptr + shared_ptr_offset +
                     (j + warpIdInBlock) * WARP_SIZE),
                    (B_ptr + B_ptr_offset + (j + warpIdInBlock) * WARP_SIZE),
                    (j + warpIdInBlock), splitNumCycleB, MACA_ICMP_SLT);
            }
            shared_ptr_offset += sharedNumCycleB * WARP_SIZE;
            B_ptr_offset += numCycleB * WARP_SIZE;
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
                C_f32[j][index_A] = mma_16x16x16f16<Tab>(
                    tmpA[t][index_A][0], tmpA[t][index_A][1], tmpB[j][0],
                    tmpB[j][1], C_f32[j][index_A]);
                C_f32[j][index_A] = mma_16x16x16f16<Tab>(
                    tmpA[t][index_A][2], tmpA[t][index_A][3], tmpB[j][2],
                    tmpB[j][3], C_f32[j][index_A]);
            }
        }
    }
    if (end == 0)
        return;

    CStgType bias_load[APerWarp];
    if constexpr (HasOneDimBias) {
        if (splitKId == 0) {
            for (int index_A = 0; index_A < APerWarp; index_A++) {
                int bias_offset =
                    (warpRowsGroupBegin + index_A) * colThreadsPerMma +
                    quarterWarpId;
                bias_load[index_A] =
                    (reinterpret_cast<CStgType *>(bias))[bias_offset];
            }
        }
    }

    for (int j = 0; j < sharedNumCycleB; j++) {
        for (int index_A = 0; index_A < APerWarp; index_A++) {
            float C_f32_res[4];
            int weight_index =
                C_offset[index_A] * (sizeof(CStgType) / sizeof(Tc)) / m;
            Tc score_weight = score_weights[weight_index];
            // Tc score_weight = 1.0;
#pragma unroll 4
            for (int t = 0; t < 4; t++) {
                C_f32_res[t] =
                    C_f32[j][index_A][t] * static_cast<float>(score_weight);
            }
            Tc C_tc_tmp[4] = {0};
#pragma unroll 4
            for (int t = 0; t < 4; t++) {
                C_tc_tmp[t] = static_cast<Tc>(C_f32_res[t]);
            }

            if constexpr (HasOneDimBias) {
                if (splitKId == 0) {
                    if constexpr (std::is_same_v<Tc, __half>) {
                        reinterpret_cast<__half2 *>(&C_tc_tmp[0])[0] = __hadd2(
                            reinterpret_cast<__half2 *>(&C_tc_tmp[0])[0],
                            reinterpret_cast<__half2 *>(
                                &bias_load[index_A])[0]);
                        reinterpret_cast<__half2 *>(&C_tc_tmp[0])[1] = __hadd2(
                            reinterpret_cast<__half2 *>(&C_tc_tmp[0])[1],
                            reinterpret_cast<__half2 *>(
                                &bias_load[index_A])[1]);
                    } else if constexpr (std::is_same_v<Tc, maca_bfloat16>) {
                        reinterpret_cast<maca_bfloat162 *>(&C_tc_tmp[0])[0] =
                            __hadd2(reinterpret_cast<maca_bfloat162 *>(
                                        &C_tc_tmp[0])[0],
                                    reinterpret_cast<maca_bfloat162 *>(
                                        &bias_load[index_A])[0]);
                        reinterpret_cast<maca_bfloat162 *>(&C_tc_tmp[0])[1] =
                            __hadd2(reinterpret_cast<maca_bfloat162 *>(
                                        &C_tc_tmp[0])[1],
                                    reinterpret_cast<maca_bfloat162 *>(
                                        &bias_load[index_A])[1]);
                    }
                }
            }

            if (j >= splitNumCycleB) {
                break;
            }
            if constexpr (std::is_same_v<Tc, __half>) {
                atomicAdd(
                    reinterpret_cast<__half2 *>(&C_ptr[C_offset[index_A]]),
                    {C_tc_tmp[0], C_tc_tmp[1]});
                atomicAdd(
                    reinterpret_cast<__half2 *>(&C_ptr[C_offset[index_A]]) + 1,
                    {C_tc_tmp[2], C_tc_tmp[3]});
            } else if constexpr (std::is_same_v<Tc, __maca_bfloat16>) {
                atomicAdd(reinterpret_cast<__maca_bfloat162 *>(
                              &C_ptr[C_offset[index_A]]),
                          {C_tc_tmp[0], C_tc_tmp[1]});
                atomicAdd(reinterpret_cast<__maca_bfloat162 *>(
                              &C_ptr[C_offset[index_A]]) +
                              1,
                          {C_tc_tmp[2], C_tc_tmp[3]});
            }

            C_offset[index_A] +=
                rowThreadsPerMma * (m / elementsPerThreadPerMma);
        }
    }
}

template <typename Tab, typename Taccum, typename Tc, int BLOCK_DIM_X,
          int APerWarp, int splitN, int splitK, int N>
__device__ void
GemmMmaLayoutAB_ContinuousCReuseA_and_mul_weights_KernelDispatch(
    Tab *A, Tab *B, Tc *C, int m, int n, int k, Tc *alpha, Taccum beta,
    Tc *dev_bias, int gemm_warpId) {
    bool isBetaZero = (beta == static_cast<Taccum>(0));
    bool hasOneDimBias = !(dev_bias == nullptr);

#define LAUNCH_GEMM_OPT_KERNEL(IS_BETA_ZERO, HASONEDIMBIAS)                    \
    GemmMmaLayoutAB_ContinuousCReuseA_and_mul_weights_Kernel<                  \
        Tab, Taccum, Tc, BLOCK_DIM_X, N, APerWarp, splitN, splitK, true,       \
        false>(A, B, C, m, n, k, alpha, beta, dev_bias, gemm_warpId);

    LAUNCH_GEMM_OPT_KERNEL(isBetaZero, hasOneDimBias);

#undef LAUNCH_GEMM_OPT_KERNEL
}
