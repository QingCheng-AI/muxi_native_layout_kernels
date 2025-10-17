#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>

#include <type_traits>

#include "arg_selector.h"
#include "gemm_layout_A.h"
#include "utils.cuh"

namespace muxi_layout_kernels {

// multi M, splitK, splitN kernel
template <typename Ta, typename Tb, typename Taccum, typename Tc,
          int BLOCK_DIM_X, int N, int APerWarp, int splitN, int splitK,
          bool IsBetaZero, bool HasOneDimBias = false>
__global__ void __launch_bounds__(BLOCK_DIM_X)
    GemmMmaJustLayoutAKernel_Soft_Fp8(Ta *A, uint32_t *A_scale, Tb *B, Tc *C,
                                      int m, int n, int k, Taccum alpha,
                                      Taccum beta, Tc *bias) {
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

    int numWarps = __builtin_mxc_readfirstlane(gridDim.x * blockDim.x /
                                               WARP_SIZE / splitN / splitK);
    int warpId = __builtin_mxc_readfirstlane(
                     (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE) %
                 numWarps;
    int splitNId = __builtin_mxc_readfirstlane(
        (((blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE) / numWarps) %
        splitN);
    int splitKId = __builtin_mxc_readfirstlane(
        (((blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE) / numWarps /
         splitN) %
        splitK);
    int warpIdInBlock = (threadIdx.x / WARP_SIZE);
    int laneId = threadIdx.x & (WARP_SIZE - 1);
    int quarterWarpId = laneId / 16;
    int quarterLaneId = laneId & (16 - 1);
    int num_block_per_group = m / 128;
    int block_idx = (blockIdx.x * APerWarp / 2) % num_block_per_group;
    // if (threadIdx.x == 0){
    //   printf("(%d,%d)", blockIdx.x, block_idx);
    // }
    int warpRowsGroupBegin =
        __builtin_mxc_readfirstlane(warpId * (rowsGroup / numWarps) +
                                    min(warpId, rowsGroup % numWarps)) *
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
    int B_offset_conB =
        quarterWarpId + quarterLaneId * (k / elementsPerAccess) +
        splitNStart * rowThreadsPerMma * (k / elementsPerAccess) +
        splitKStart * stages * colThreadsPerMma;
    int C_offset[APerWarp];
    for (int i = 0; i < APerWarp; i++) {
        C_offset[i] = (quarterLaneId + splitNStart * rowThreadsPerMma) *
                          (m / elementsPerThreadPerMma) +
                      (warpRowsGroupBegin + i) * colThreadsPerMma +
                      quarterWarpId;
    }

    UINT2 *A_ptr[APerWarp];
    for (int i = 0; i < APerWarp; i++) {
        A_ptr[i] = reinterpret_cast<UINT2 *>(A) + A_offset +
                   i * (rowThreadsPerMma * (k / elementsPerAccess));
    }
    UINT4 *B_ptr_conB = reinterpret_cast<UINT4 *>(B) + B_offset_conB;
    CStgType *C_ptr = reinterpret_cast<CStgType *>(C);

    __shared__ UINT4 sharedTmpB[stages][sharedNumCycleB][64];
    UINT4 *shared_ptr =
        reinterpret_cast<UINT4 *>(&sharedTmpB[0][0][0]) + laneId;

    int splitNumCycleB = __builtin_mxc_readfirstlane(splitNEnd - splitNStart);

    UINT2 tmpA[stages][APerWarp];
    uint32_t tmpA_bf16[4];
    UINT4 tmpB[sharedNumCycleB];
    FLOAT4 C_f32[sharedNumCycleB][APerWarp];
    uint32_t *A_scale_ptr =
        A_scale + block_idx * k / 128 + splitKId * k / 128 / splitK;
    uint32_t scale = __builtin_mxc_readfirstlane(*A_scale_ptr);

    for (int i = 0; i < sharedNumCycleB; i++) {
        for (int index_A = 0; index_A < APerWarp; index_A++) {
            C_f32[i][index_A] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    if (end == 0)
        for (int index_A = 0; index_A < APerWarp; index_A++) {
            A_ptr[index_A] = reinterpret_cast<UINT2 *>(A);
        }
    int A_ptr_offset = 0;
    int B_ptr_offset_conB = 0;
    int shared_ptr_offset = 0;

    // begin fill the pipeline
#pragma unroll stages
    for (int t = 0; t < stages; t++) {
        asm("/* Stop compiler reordering (begin) */");
        for (int index_A = 0; index_A < APerWarp; index_A++) {
            tmpA[t][index_A] = mxc_read64_async(
                reinterpret_cast<INT64 *>(A_ptr[index_A] + A_ptr_offset));
        }
        A_ptr_offset += WARP_SIZE;
#pragma unroll
        for (int j = 0; j < sharedNumCycleB; j += warpPerBlock) {
            LDG_B128_BSM_WITH_PREDICATOR_NORET0(
                (shared_ptr + shared_ptr_offset +
                 (j + warpIdInBlock) * WARP_SIZE),
                (B_ptr_conB + B_ptr_offset_conB +
                 (j + warpIdInBlock) * (rowThreadsPerMma * k / 8)),
                (j + warpIdInBlock), splitNumCycleB, MACA_ICMP_SLT);
        }
        shared_ptr_offset += sharedNumCycleB * WARP_SIZE;
        B_ptr_offset_conB += colThreadsPerMma;
    }

    // pipeline main loop
    for (int i = 0; i < end - 1; i++) {
        shared_ptr_offset = 0;
#pragma unroll stages
        for (int t = 0; t < stages; t++) {
            asm("/* Stop compiler reordering (loop) */");
            __builtin_mxc_arrive_gvmcnt(
                (sharedArriveCount + APerWarp) * (stages - 1) - 1);
            __builtin_mxc_barrier_inst();
#pragma unroll
            for (int j = 0; j < sharedNumCycleB; j++) {
                tmpB[j] = sharedTmpB[t][j][laneId];
            }

#pragma unroll
            for (int index_A = 0; index_A < APerWarp; index_A++) {
                scalefp8tobf16(tmpA[t][index_A][0],
                               reinterpret_cast<uint16_t *>(tmpA_bf16), scale);
#pragma unroll
                for (int j = 0; j < sharedNumCycleB; j++) {
                    C_f32[j][index_A] = mma_16x16x16f16<__maca_bfloat16>(
                        tmpA_bf16[0], tmpA_bf16[1], tmpB[j][0], tmpB[j][1],
                        C_f32[j][index_A]);
                }
                scalefp8tobf16(tmpA[t][index_A][1],
                               reinterpret_cast<uint16_t *>(tmpA_bf16 + 2),
                               scale);
#pragma unroll
                for (int j = 0; j < sharedNumCycleB; j++) {
                    C_f32[j][index_A] = mma_16x16x16f16<__maca_bfloat16>(
                        tmpA_bf16[2], tmpA_bf16[3], tmpB[j][2], tmpB[j][3],
                        C_f32[j][index_A]);
                }
            }

            __builtin_mxc_barrier_inst();
            for (int index_A = 0; index_A < APerWarp; index_A++) {
                tmpA[t][index_A] = mxc_read64_async(
                    reinterpret_cast<INT64 *>(A_ptr[index_A] + A_ptr_offset));
            }
            A_ptr_offset += WARP_SIZE;
#pragma unroll
            for (int j = 0; j < sharedNumCycleB; j += warpPerBlock) {
                LDG_B128_BSM_WITH_PREDICATOR_NORET0(
                    (shared_ptr + shared_ptr_offset +
                     (j + warpIdInBlock) * WARP_SIZE),
                    (B_ptr_conB + B_ptr_offset_conB +
                     (j + warpIdInBlock) * (rowThreadsPerMma * k / 8)),
                    (j + warpIdInBlock), splitNumCycleB, MACA_ICMP_SLT);
            }
            shared_ptr_offset += sharedNumCycleB * WARP_SIZE;
            B_ptr_offset_conB += colThreadsPerMma;
        }
        scale = __builtin_mxc_readfirstlane(*(A_scale_ptr + i + 1));
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
        for (int index_A = 0; index_A < APerWarp; index_A++) {
            scalefp8tobf16(tmpA[t][index_A][0],
                           reinterpret_cast<uint16_t *>(tmpA_bf16), scale);
#pragma unroll
            for (int j = 0; j < sharedNumCycleB; j++) {
                C_f32[j][index_A] = mma_16x16x16f16<__maca_bfloat16>(
                    tmpA_bf16[0], tmpA_bf16[1], tmpB[j][0], tmpB[j][1],
                    C_f32[j][index_A]);
            }
            scalefp8tobf16(tmpA[t][index_A][1],
                           reinterpret_cast<uint16_t *>(tmpA_bf16 + 2), scale);
#pragma unroll
            for (int j = 0; j < sharedNumCycleB; j++) {
                C_f32[j][index_A] = mma_16x16x16f16<__maca_bfloat16>(
                    tmpA_bf16[2], tmpA_bf16[3], tmpB[j][2], tmpB[j][3],
                    C_f32[j][index_A]);
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
                    (warpRowsGroupBegin + index_A) * colThreadsPerMma +
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
                        // C_tc_tmp[t] = __hadd(C_tc_tmp[t], splitKFactor *
                        // bias_tc[t]);
                        C_tc_tmp[t] = __hadd(C_tc_tmp[t], bias_tc[t]);
                    }
                }
            }

            if (j >= splitNumCycleB) {
                break;
            }

            if constexpr (splitK > 1) {
                if constexpr (std::is_same_v<Tc, __half>) {
                    atomicAdd(
                        reinterpret_cast<__half2 *>(&C_ptr[C_offset[index_A]]),
                        {C_tc_tmp[0], C_tc_tmp[1]});
                    atomicAdd(
                        reinterpret_cast<__half2 *>(&C_ptr[C_offset[index_A]]) +
                            1,
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
            } else {
                C_ptr[C_offset[index_A]] =
                    *reinterpret_cast<CStgType *>(&C_tc_tmp[0]);
            }

            C_offset[index_A] +=
                rowThreadsPerMma * (m / elementsPerThreadPerMma);
        }
    }
}

template <typename Ta, typename Tb, typename Taccum, typename Tc,
          int BLOCK_DIM_X, int tile_m, int tile_n, int tile_k, bool IsBetaZero,
          bool HasOneDimBias = false>
__global__ void __launch_bounds__(BLOCK_DIM_X)
    GemmMmaJustLayoutAKernel2_Soft_Fp8(Ta *A, uint32_t *A_scale, Tb *B, Tc *C,
                                       int m, int n, int k, Taccum alpha,
                                       Taccum beta, Tc *bias) {
    constexpr int elementsPerAccess = 8;
    constexpr int rowThreadsPerMma = 16;
    constexpr int colThreadsPerMma = 4;
    constexpr int elementsPerThreadPerMma = 4;
    constexpr int stages =
        tile_k / (elementsPerThreadPerMma * elementsPerAccess);
    constexpr int warpPerBlock = BLOCK_DIM_X / WARP_SIZE;

    const int warpIdInBlock =
        __builtin_mxc_readfirstlane(threadIdx.x / WARP_SIZE);
    const int laneId = threadIdx.x & (WARP_SIZE - 1);
    const int quarterWarpId = laneId / 16;
    const int quarterLaneId = laneId & (16 - 1);
    int num_block_per_row = 128 / tile_m;
    int block_idx = blockIdx.x / num_block_per_row;

    const int blockColsGroup = __builtin_mxc_readfirstlane(blockIdx.y);
    const int blockRowsGroup = __builtin_mxc_readfirstlane(blockIdx.x);
    constexpr int m_times = (tile_m / rowThreadsPerMma);
    constexpr int n_times = (tile_n / rowThreadsPerMma);
    constexpr int loadAPerStage = ((m_times + warpPerBlock - 1) / warpPerBlock);
    constexpr int loadBPerStage = ((n_times + warpPerBlock - 1) / warpPerBlock);
    const int kChunks = (k / tile_k);

    __shared__ UINT2 shared_data_A[tile_m * tile_k / elementsPerAccess];
    __shared__ UINT4 shared_data_B[tile_n * tile_k / elementsPerAccess];
    UINT2 *shared_A_stages[stages], *A_ptr_stages[stages];
    UINT4 *shared_B_stages[stages], *B_ptr_stages[stages];
    CStgType *C_ptr = reinterpret_cast<CStgType *>(C);

    for (int i = 0; i < stages; i++) {
        shared_A_stages[i] =
            &shared_data_A[tile_m * tile_k / elementsPerAccess / stages * i];
        shared_B_stages[i] =
            &shared_data_B[tile_n * tile_k / elementsPerAccess / stages * i];
        A_ptr_stages[i] =
            reinterpret_cast<UINT2 *>(A) +
            blockRowsGroup * tile_m * k / elementsPerAccess + i * WARP_SIZE +
            warpIdInBlock * rowThreadsPerMma * k / elementsPerAccess + laneId;
        B_ptr_stages[i] = reinterpret_cast<UINT4 *>(B) +
                          blockColsGroup * tile_n * colThreadsPerMma +
                          i * colThreadsPerMma * n + warpIdInBlock * WARP_SIZE +
                          laneId;
        B_ptr_stages[i] =
            reinterpret_cast<UINT4 *>(B) +
            blockColsGroup * tile_n * (k / elementsPerAccess) +
            i * colThreadsPerMma +
            warpIdInBlock * (k / elementsPerAccess) * rowThreadsPerMma +
            quarterLaneId * (k / elementsPerAccess) + quarterWarpId;
    }

    // return the threads without task
    if ((blockRowsGroup >= ((m + tile_m - 1) / tile_m)) ||
        (blockColsGroup >= (n + tile_n - 1) / tile_n)) {
        return;
    }

    UINT2 tmpA[m_times / warpPerBlock];
    uint32_t tmpA_bf16[4];
    UINT4 tmpB[n_times];
    FLOAT4 C_f32[m_times / warpPerBlock][n_times];
    uint32_t *A_scale_ptr = A_scale + block_idx * k / 128;
    uint32_t scale = *A_scale_ptr;

    for (int i = 0; i < m_times / warpPerBlock; i++) {
        for (int j = 0; j < n_times; j++) {
            C_f32[i][j] = {0, 0, 0, 0};
        }
    }

    // prefill the pipeline, prefetch stages data.
    for (int i = 0; i < stages; i++) {
        // load A and B for every stage
        for (int j = 0; j < loadAPerStage - 1; j++) {
            LDG_B64_BSM_NO_PREDICATOR(
                shared_A_stages[i] + warpPerBlock * j * WARP_SIZE + threadIdx.x,
                A_ptr_stages[i] + j * rowThreadsPerMma * k * warpPerBlock /
                                      elementsPerAccess);
        }
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
                B_ptr_stages[i] + j * warpPerBlock * (k / elementsPerAccess) *
                                      rowThreadsPerMma,
                predicatorB, 1, MACA_ICMP_EQ);
        }
        // for the tail in every stage
        LDG_B64_BSM_WITH_PREDICATOR_NORET0(
            shared_A_stages[i] +
                warpPerBlock * (loadAPerStage - 1) * WARP_SIZE + threadIdx.x,
            A_ptr_stages[i] + rowThreadsPerMma * k * warpPerBlock /
                                  elementsPerAccess * (loadAPerStage - 1),
            (loadAPerStage - 1) * warpPerBlock + warpIdInBlock, m_times,
            MACA_ICMP_SLT);

        // Added A_ptr and B_ptr
        A_ptr_stages[i] += WARP_SIZE * stages;
        B_ptr_stages[i] += colThreadsPerMma * stages;
    }

    const int sharedA_offset = warpIdInBlock * (m_times / warpPerBlock);
    // main loop in pipiline compute
    for (int i = 1; i < kChunks; i++) {
        // for every stage
        for (int stage = 0; stage < stages; stage++) {
            // wait for async data load
            __builtin_mxc_arrive_gvmcnt((loadAPerStage + loadBPerStage) *
                                        (stages - 1));
            __builtin_mxc_barrier_inst();

            // prefetch data from shared to registers
            for (int j = 0; j < m_times / warpPerBlock; j++) {
                tmpA[j] = shared_A_stages[stage][laneId + (sharedA_offset + j) *
                                                              WARP_SIZE];
            }

            for (int j = 0; j < n_times; j++) {
                tmpB[j] = *(shared_B_stages[stage] + (laneId + j * WARP_SIZE));
            }

            // MMA compute
            for (int j = 0; j < m_times / warpPerBlock; j++) {
                scalefp8tobf16(tmpA[j][0],
                               reinterpret_cast<uint16_t *>(tmpA_bf16), scale);
                for (int k = 0; k < n_times; k++) {
                    C_f32[j][k] = mma_16x16x16f16<Tb>(tmpA_bf16[0],
                                                      tmpA_bf16[1], tmpB[k][0],
                                                      tmpB[k][1], C_f32[j][k]);
                }
                scalefp8tobf16(tmpA[j][1],
                               reinterpret_cast<uint16_t *>(tmpA_bf16 + 2),
                               scale);
                for (int k = 0; k < n_times; k++) {
                    C_f32[j][k] = mma_16x16x16f16<Tb>(tmpA_bf16[2],
                                                      tmpA_bf16[3], tmpB[k][2],
                                                      tmpB[k][3], C_f32[j][k]);
                }
                // if(blockIdx.x == 15 && threadIdx.x == 0){
                //   // printf("%f", C_f32[0][0][0]);
                //   uint tmp32 = tmpA_bf16[0] & 0xffff0000;
                //   float *tmpf = reinterpret_cast<float *>(&tmp32);
                //   printf("%f ", *tmpf);
                // }
            }

            __builtin_mxc_barrier_inst();

            // async load next stage data
            {
                for (int j = 0; j < loadAPerStage - 1; j++) {
                    LDG_B64_BSM_NO_PREDICATOR(
                        shared_A_stages[stage] + warpPerBlock * j * WARP_SIZE +
                            threadIdx.x,
                        A_ptr_stages[stage] + j * rowThreadsPerMma * k *
                                                  warpPerBlock /
                                                  elementsPerAccess);
                }
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
                        B_ptr_stages[stage] + j * warpPerBlock *
                                                  (k / elementsPerAccess) *
                                                  rowThreadsPerMma,
                        predicatorB, 1, MACA_ICMP_EQ);
                }
                // for the tail in every stage
                LDG_B64_BSM_WITH_PREDICATOR_NORET0(
                    shared_A_stages[stage] +
                        warpPerBlock * (loadAPerStage - 1) * WARP_SIZE +
                        threadIdx.x,
                    // A_ptr_stages[stage] + warpIdInBlock * (loadAPerStage -
                    // 1),
                    A_ptr_stages[stage] + rowThreadsPerMma * k * warpPerBlock /
                                              elementsPerAccess *
                                              (loadAPerStage - 1),
                    (loadAPerStage - 1) * warpPerBlock + warpIdInBlock, m_times,
                    MACA_ICMP_SLT);

                // Added A_ptr and B_ptr
                A_ptr_stages[stage] += WARP_SIZE * stages;
                B_ptr_stages[stage] += colThreadsPerMma * stages;
            }
        }
        scale = *(A_scale_ptr + i);
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

        // prefetch data from shared to registers
        for (int j = 0; j < m_times / warpPerBlock; j++) {
            tmpA[j] = shared_A_stages[stage][laneId +
                                             (sharedA_offset + j) * WARP_SIZE];
        }
        for (int j = 0; j < n_times; j++) {
            tmpB[j] = *(shared_B_stages[stage] + (laneId + j * WARP_SIZE));
        }

        // MMA compute
        for (int j = 0; j < m_times / warpPerBlock; j++) {
            scalefp8tobf16(tmpA[j][0], reinterpret_cast<uint16_t *>(tmpA_bf16),
                           scale);
            for (int k = 0; k < n_times; k++) {
                C_f32[j][k] =
                    mma_16x16x16f16<Tb>(tmpA_bf16[0], tmpA_bf16[1], tmpB[k][0],
                                        tmpB[k][1], C_f32[j][k]);
            }
            scalefp8tobf16(tmpA[j][1],
                           reinterpret_cast<uint16_t *>(tmpA_bf16 + 2), scale);
            for (int k = 0; k < n_times; k++) {
                C_f32[j][k] =
                    mma_16x16x16f16<Tb>(tmpA_bf16[2], tmpA_bf16[3], tmpB[k][2],
                                        tmpB[k][3], C_f32[j][k]);
            }
        }

        // __builtin_mxc_barrier_inst();
    }

    // store result
    int warpRowsGroupBegin =
        blockRowsGroup * m_times + m_times / warpPerBlock * warpIdInBlock;

    CStgType bias_load[m_times / warpPerBlock];
    if constexpr (HasOneDimBias) {
        for (int i = 0; i < m_times / warpPerBlock; i++) {
            int bias_offset =
                (warpRowsGroupBegin + i) * colThreadsPerMma + quarterWarpId;
            bias_load[i] = (reinterpret_cast<CStgType *>(bias))[bias_offset];
        }
    }

#pragma unroll
    for (int j = 0; (j < n_times) &&
                    (j < (n / rowThreadsPerMma - blockColsGroup * n_times));
         j++) {
        int C_offset = (quarterLaneId + blockColsGroup * tile_n) *
                           (m / elementsPerThreadPerMma) +
                       (warpRowsGroupBegin)*colThreadsPerMma + quarterWarpId +
                       j * rowThreadsPerMma * (m / elementsPerThreadPerMma);
        for (int i = 0; i < m_times / warpPerBlock; i++) {
            Tc tc_tmp[4];
#pragma unroll
            for (int t = 0; t < 4; t++) {
                C_f32[i][j][t] *= alpha;
            }
#pragma unroll
            for (int t = 0; t < 4; t++) {
                tc_tmp[t] = static_cast<Tc>(C_f32[i][j][t]);
            }

            if constexpr (HasOneDimBias) {
                Tc *bias_tc = reinterpret_cast<Tc *>(&bias_load[i]);
#pragma unroll
                for (int t = 0; t < 4; t++) {
                    tc_tmp[t] = __hadd(tc_tmp[t], bias_tc[t]);
                }
            }

            C_ptr[C_offset] = *reinterpret_cast<CStgType *>(&tc_tmp[0]);
            C_offset += 4;
        }
    }
}

template <typename Ta, typename Tb, typename Taccum, typename Tc, int APerWarp,
          int splitN, int splitK>
void GemmMmaJustLayoutAKernelDispatchN_Soft_Fp8(Ta *A, uint32_t *A_scale, Tb *B,
                                                Tc *C, int m, int n, int k,
                                                Taccum alpha, Taccum beta,
                                                Tc *dev_bias = nullptr) {
    constexpr int block_dim_x = 256;
    bool isBetaZero = (beta == static_cast<Taccum>(0));
    bool hasOneDimBias = !(dev_bias == nullptr);

#define LAUNCH_GEMM_OPT_KERNEL_SOFT_FP8(N, IS_BETA_ZERO, HASONEDIMBIAS)        \
    auto cur_device = at::cuda::current_device();                              \
    const mcStream_t stream = at::cuda::getCurrentCUDAStream(cur_device);      \
    if (IS_BETA_ZERO) {                                                        \
        if (HASONEDIMBIAS) {                                                   \
            GemmMmaJustLayoutAKernel_Soft_Fp8<Ta, Tb, Taccum, Tc, block_dim_x, \
                                              N, APerWarp, splitN, splitK,     \
                                              true, true>                      \
                <<<(m / 16 / (block_dim_x / WARP_SIZE * APerWarp) * splitN *   \
                    splitK),                                                   \
                   block_dim_x, 0, stream>>>(A, A_scale, B, C, m, n, k, alpha, \
                                             beta, dev_bias);                  \
        } else {                                                               \
            GemmMmaJustLayoutAKernel_Soft_Fp8<Ta, Tb, Taccum, Tc, block_dim_x, \
                                              N, APerWarp, splitN, splitK,     \
                                              true, false>                     \
                <<<(m / 16 / (block_dim_x / WARP_SIZE * APerWarp) * splitN *   \
                    splitK),                                                   \
                   block_dim_x, 0, stream>>>(A, A_scale, B, C, m, n, k, alpha, \
                                             beta, dev_bias);                  \
        }                                                                      \
    } else {                                                                   \
        if (HASONEDIMBIAS) {                                                   \
            GemmMmaJustLayoutAKernel_Soft_Fp8<Ta, Tb, Taccum, Tc, block_dim_x, \
                                              N, APerWarp, splitN, splitK,     \
                                              false, true>                     \
                <<<(m / 16 / (block_dim_x / WARP_SIZE * APerWarp) * splitN *   \
                    splitK),                                                   \
                   block_dim_x, 0, stream>>>(A, A_scale, B, C, m, n, k, alpha, \
                                             beta, dev_bias);                  \
        } else {                                                               \
            GemmMmaJustLayoutAKernel_Soft_Fp8<Ta, Tb, Taccum, Tc, block_dim_x, \
                                              N, APerWarp, splitN, splitK,     \
                                              false, false>                    \
                <<<(m / 16 / (block_dim_x / WARP_SIZE * APerWarp) * splitN *   \
                    splitK),                                                   \
                   block_dim_x, 0, stream>>>(A, A_scale, B, C, m, n, k, alpha, \
                                             beta, dev_bias);                  \
        }                                                                      \
    }

    if (n <= 16) {
        LAUNCH_GEMM_OPT_KERNEL_SOFT_FP8(16, isBetaZero, hasOneDimBias);
    } else if (n <= 32) {
        LAUNCH_GEMM_OPT_KERNEL_SOFT_FP8(32, isBetaZero, hasOneDimBias);
    } else if (n <= 48) {
        LAUNCH_GEMM_OPT_KERNEL_SOFT_FP8(48, isBetaZero, hasOneDimBias);
    } else if (n <= 64) {
        LAUNCH_GEMM_OPT_KERNEL_SOFT_FP8(64, isBetaZero, hasOneDimBias);
    } else if (n <= 80) {
        LAUNCH_GEMM_OPT_KERNEL_SOFT_FP8(80, isBetaZero, hasOneDimBias);
    } else if (n <= 96) {
        LAUNCH_GEMM_OPT_KERNEL_SOFT_FP8(96, isBetaZero, hasOneDimBias);
    } else if (n <= 112) {
        LAUNCH_GEMM_OPT_KERNEL_SOFT_FP8(112, isBetaZero, hasOneDimBias);
    } else if (n <= 128) {
        LAUNCH_GEMM_OPT_KERNEL_SOFT_FP8(128, isBetaZero, hasOneDimBias);
    } else if (n <= 144) {
        LAUNCH_GEMM_OPT_KERNEL_SOFT_FP8(144, isBetaZero, hasOneDimBias);
    } else if (n <= 160) {
        LAUNCH_GEMM_OPT_KERNEL_SOFT_FP8(160, isBetaZero, hasOneDimBias);
    } else if (n <= 176) {
        LAUNCH_GEMM_OPT_KERNEL_SOFT_FP8(176, isBetaZero, hasOneDimBias);
    } else if (n <= 192) {
        LAUNCH_GEMM_OPT_KERNEL_SOFT_FP8(192, isBetaZero, hasOneDimBias);
    } else if (n <= 208) {
        LAUNCH_GEMM_OPT_KERNEL_SOFT_FP8(208, isBetaZero, hasOneDimBias);
    } else if (n <= 224) {
        LAUNCH_GEMM_OPT_KERNEL_SOFT_FP8(224, isBetaZero, hasOneDimBias);
    } else if (n <= 240) {
        LAUNCH_GEMM_OPT_KERNEL_SOFT_FP8(240, isBetaZero, hasOneDimBias);
    } else if (n <= 256) {
        LAUNCH_GEMM_OPT_KERNEL_SOFT_FP8(256, isBetaZero, hasOneDimBias);
    }

#undef LAUNCH_GEMM_OPT_KERNEL_SOFT_FP8
} // namespace muxi_layout_kernels

template <typename Ta, typename Tb, typename Taccum, typename Tc>
void GemmMmaJustLayoutAKernelDispatch_Soft_Fp8(Ta *A, uint32_t *A_scale, Tb *B,
                                               Tc *C, int m, int n, int k,
                                               Taccum alpha, Taccum beta,
                                               Tc *dev_bias, int APerWarp,
                                               int splitN, int splitK) {
#define LAUNCH_GEMM_OPTS(APERWARP, SPLITN, SPLITK)                             \
    GemmMmaJustLayoutAKernelDispatchN_Soft_Fp8<Ta, Tb, Taccum, Tc, APERWARP,   \
                                               SPLITN, SPLITK>(                \
        A, A_scale, B, C, m, n, k, alpha, beta, dev_bias);

    if (APerWarp == 1 && splitN == 1 && splitK == 1) {
        LAUNCH_GEMM_OPTS(1, 1, 1);
    } else if (APerWarp == 1 && splitN == 1 && splitK == 2) {
        LAUNCH_GEMM_OPTS(1, 1, 2);
    } else if (APerWarp == 1 && splitN == 1 && splitK == 3) {
        LAUNCH_GEMM_OPTS(1, 1, 3);
    } else if (APerWarp == 1 && splitN == 1 && splitK == 4) {
        LAUNCH_GEMM_OPTS(1, 1, 4);
    } else if (APerWarp == 1 && splitN == 2 && splitK == 1) {
        LAUNCH_GEMM_OPTS(1, 2, 1);
    } else if (APerWarp == 1 && splitN == 2 && splitK == 2) {
        LAUNCH_GEMM_OPTS(1, 2, 2);
    } else if (APerWarp == 1 && splitN == 2 && splitK == 3) {
        LAUNCH_GEMM_OPTS(1, 2, 3);
    } else if (APerWarp == 1 && splitN == 2 && splitK == 4) {
        LAUNCH_GEMM_OPTS(1, 2, 4);
    } else if (APerWarp == 1 && splitN == 3 && splitK == 1) {
        LAUNCH_GEMM_OPTS(1, 3, 1);
    } else if (APerWarp == 1 && splitN == 3 && splitK == 2) {
        LAUNCH_GEMM_OPTS(1, 3, 2);
    } else if (APerWarp == 1 && splitN == 3 && splitK == 3) {
        LAUNCH_GEMM_OPTS(1, 3, 3);
    } else if (APerWarp == 1 && splitN == 3 && splitK == 4) {
        LAUNCH_GEMM_OPTS(1, 3, 4);
    } else if (APerWarp == 1 && splitN == 4 && splitK == 1) {
        LAUNCH_GEMM_OPTS(1, 4, 1);
    } else if (APerWarp == 1 && splitN == 4 && splitK == 2) {
        LAUNCH_GEMM_OPTS(1, 4, 2);
    } else if (APerWarp == 1 && splitN == 4 && splitK == 3) {
        LAUNCH_GEMM_OPTS(1, 4, 3);
    } else if (APerWarp == 1 && splitN == 4 && splitK == 4) {
        LAUNCH_GEMM_OPTS(1, 4, 4);
    } else if (APerWarp == 2 && splitN == 1 && splitK == 1) {
        LAUNCH_GEMM_OPTS(2, 1, 1);
    } else if (APerWarp == 2 && splitN == 1 && splitK == 2) {
        LAUNCH_GEMM_OPTS(2, 1, 2);
    } else if (APerWarp == 2 && splitN == 1 && splitK == 3) {
        LAUNCH_GEMM_OPTS(2, 1, 3);
    } else if (APerWarp == 2 && splitN == 1 && splitK == 4) {
        LAUNCH_GEMM_OPTS(2, 1, 4);
    } else if (APerWarp == 2 && splitN == 2 && splitK == 1) {
        LAUNCH_GEMM_OPTS(2, 2, 1);
    } else if (APerWarp == 2 && splitN == 2 && splitK == 2) {
        LAUNCH_GEMM_OPTS(2, 2, 2);
    } else if (APerWarp == 2 && splitN == 2 && splitK == 3) {
        LAUNCH_GEMM_OPTS(2, 2, 3);
    } else if (APerWarp == 2 && splitN == 2 && splitK == 4) {
        LAUNCH_GEMM_OPTS(2, 2, 4);
    } else if (APerWarp == 2 && splitN == 3 && splitK == 1) {
        LAUNCH_GEMM_OPTS(2, 3, 1);
    } else if (APerWarp == 2 && splitN == 3 && splitK == 2) {
        LAUNCH_GEMM_OPTS(2, 3, 2);
    } else if (APerWarp == 2 && splitN == 3 && splitK == 3) {
        LAUNCH_GEMM_OPTS(2, 3, 3);
    } else if (APerWarp == 2 && splitN == 3 && splitK == 4) {
        LAUNCH_GEMM_OPTS(2, 3, 4);
    } else if (APerWarp == 2 && splitN == 4 && splitK == 1) {
        LAUNCH_GEMM_OPTS(2, 4, 1);
    } else if (APerWarp == 2 && splitN == 4 && splitK == 2) {
        LAUNCH_GEMM_OPTS(2, 4, 2);
    } else if (APerWarp == 2 && splitN == 4 && splitK == 3) {
        LAUNCH_GEMM_OPTS(2, 4, 3);
    } else if (APerWarp == 2 && splitN == 4 && splitK == 4) {
        LAUNCH_GEMM_OPTS(2, 4, 4);
    }

#undef LAUNCH_GEMM_OPTS
}

template <typename Ta, typename Tb, typename Taccum, typename Tc>
void GemmMmaJustLayoutAKernelDispatch2_Soft_Fp8(Ta *A, uint32_t *A_scale, Tb *B,
                                                Tc *C, int m, int n, int k,
                                                Taccum alpha, Taccum beta,
                                                Tc *dev_bias, int tile_m,
                                                int tile_n, int tile_k) {
    constexpr int block_dim_x = 256;
    bool isBetaZero = (beta == static_cast<Taccum>(0));
    bool hasOneDimBias = !(dev_bias == nullptr);
    dim3 gridSize((m + tile_m - 1) / tile_m, (n + tile_n - 1) / tile_n);

#define LAUNCH_GEMM_LAYOUTA2_SOFT_FP8(tile_m, tile_n, tile_k, IS_BETA_ZERO,    \
                                      HASONEDIMBIAS)                           \
    auto cur_device = at::cuda::current_device();                              \
    const mcStream_t stream = at::cuda::getCurrentCUDAStream(cur_device);      \
    if (IS_BETA_ZERO) {                                                        \
        if (HASONEDIMBIAS) {                                                   \
            GemmMmaJustLayoutAKernel2_Soft_Fp8<Ta, Tb, Taccum, Tc,             \
                                               block_dim_x, tile_m, tile_n,    \
                                               tile_k, true, true>             \
                <<<gridSize, block_dim_x, 0, stream>>>(                        \
                    A, A_scale, B, C, m, n, k, alpha, beta, dev_bias);         \
        } else {                                                               \
            GemmMmaJustLayoutAKernel2_Soft_Fp8<Ta, Tb, Taccum, Tc,             \
                                               block_dim_x, tile_m, tile_n,    \
                                               tile_k, true, false>            \
                <<<gridSize, block_dim_x, 0, stream>>>(                        \
                    A, A_scale, B, C, m, n, k, alpha, beta, dev_bias);         \
        }                                                                      \
    } else {                                                                   \
        if (HASONEDIMBIAS) {                                                   \
            GemmMmaJustLayoutAKernel2_Soft_Fp8<Ta, Tb, Taccum, Tc,             \
                                               block_dim_x, tile_m, tile_n,    \
                                               tile_k, false, true>            \
                <<<gridSize, block_dim_x, 0, stream>>>(                        \
                    A, A_scale, B, C, m, n, k, alpha, beta, dev_bias);         \
        } else {                                                               \
            GemmMmaJustLayoutAKernel2_Soft_Fp8<Ta, Tb, Taccum, Tc,             \
                                               block_dim_x, tile_m, tile_n,    \
                                               tile_k, false, false>           \
                <<<gridSize, block_dim_x, 0, stream>>>(                        \
                    A, A_scale, B, C, m, n, k, alpha, beta, dev_bias);         \
        }                                                                      \
    }
    if (tile_m == 128 && tile_n == 16 && tile_k == 128) {
        LAUNCH_GEMM_LAYOUTA2_SOFT_FP8(128, 16, 128, isBetaZero, hasOneDimBias);
    } else if (tile_m == 128 && tile_n == 32 && tile_k == 128) {
        LAUNCH_GEMM_LAYOUTA2_SOFT_FP8(128, 32, 128, isBetaZero, hasOneDimBias);
    } else if (tile_m == 128 && tile_n == 48 && tile_k == 128) {
        LAUNCH_GEMM_LAYOUTA2_SOFT_FP8(128, 48, 128, isBetaZero, hasOneDimBias);
    } else if (tile_m == 128 && tile_n == 64 && tile_k == 128) {
        LAUNCH_GEMM_LAYOUTA2_SOFT_FP8(128, 64, 128, isBetaZero, hasOneDimBias);
    } else if (tile_m == 128 && tile_n == 80 && tile_k == 128) {
        LAUNCH_GEMM_LAYOUTA2_SOFT_FP8(128, 80, 128, isBetaZero, hasOneDimBias);
    } else if (tile_m == 128 && tile_n == 96 && tile_k == 128) {
        LAUNCH_GEMM_LAYOUTA2_SOFT_FP8(128, 96, 128, isBetaZero, hasOneDimBias);
    } else if (tile_m == 128 && tile_n == 112 && tile_k == 128) {
        LAUNCH_GEMM_LAYOUTA2_SOFT_FP8(128, 112, 128, isBetaZero, hasOneDimBias);
    } else if (tile_m == 128 && tile_n == 128 && tile_k == 128) {
        LAUNCH_GEMM_LAYOUTA2_SOFT_FP8(128, 128, 128, isBetaZero, hasOneDimBias);
    } else if (tile_m == 64 && tile_n == 16 && tile_k == 128) {
        LAUNCH_GEMM_LAYOUTA2_SOFT_FP8(64, 16, 128, isBetaZero, hasOneDimBias);
    } else if (tile_m == 64 && tile_n == 32 && tile_k == 128) {
        LAUNCH_GEMM_LAYOUTA2_SOFT_FP8(64, 32, 128, isBetaZero, hasOneDimBias);
    } else if (tile_m == 64 && tile_n == 48 && tile_k == 128) {
        LAUNCH_GEMM_LAYOUTA2_SOFT_FP8(64, 48, 128, isBetaZero, hasOneDimBias);
    } else if (tile_m == 64 && tile_n == 64 && tile_k == 128) {
        LAUNCH_GEMM_LAYOUTA2_SOFT_FP8(64, 64, 128, isBetaZero, hasOneDimBias);
    } else if (tile_m == 64 && tile_n == 80 && tile_k == 128) {
        LAUNCH_GEMM_LAYOUTA2_SOFT_FP8(64, 80, 128, isBetaZero, hasOneDimBias);
    } else if (tile_m == 64 && tile_n == 96 && tile_k == 128) {
        LAUNCH_GEMM_LAYOUTA2_SOFT_FP8(64, 96, 128, isBetaZero, hasOneDimBias);
    } else if (tile_m == 64 && tile_n == 112 && tile_k == 128) {
        LAUNCH_GEMM_LAYOUTA2_SOFT_FP8(64, 112, 128, isBetaZero, hasOneDimBias);
    } else if (tile_m == 64 && tile_n == 128 && tile_k == 128) {
        LAUNCH_GEMM_LAYOUTA2_SOFT_FP8(64, 128, 128, isBetaZero, hasOneDimBias);
    }

#undef LAUNCH_GEMM_LAYOUTA2_SOFT_FP8
}

torch::Tensor gemm_layoutA_soft_fp8_wapper(torch::Tensor A,
                                           std::optional<torch::Tensor> A_scale,
                                           torch::Tensor B, int m, int n, int k,
                                           float alpha, float beta,
                                           int kernelParam1, int kernelParam2,
                                           int kernelParam3, int kernelId,
                                           std::optional<torch::Tensor> bias) {
    torch::Tensor C;
    if (kernelId == 2 || (kernelId == 1 && kernelParam3 == 1)) {
        C = torch::empty({n, m}, B.options());
    } else {
        C = torch::zeros({n, m}, B.options());
    }
    if (B.dtype() == torch::kFloat16) {
        if (kernelId == 1) {
            GemmMmaJustLayoutAKernelDispatch_Soft_Fp8<uint8_t, __half, float,
                                                      __half>(
                reinterpret_cast<uint8_t *>(A.data_ptr()),
                reinterpret_cast<uint32_t *>(A_scale->data_ptr()),
                reinterpret_cast<__half *>(B.data_ptr<at::Half>()),
                reinterpret_cast<__half *>(C.data_ptr<at::Half>()), m, n, k,
                alpha, beta,
                bias == std::nullopt
                    ? nullptr
                    : reinterpret_cast<__half *>(bias->data_ptr<at::Half>()),
                kernelParam1, kernelParam2, kernelParam3);
        } else if (kernelId == 2) {
            GemmMmaJustLayoutAKernelDispatch2_Soft_Fp8<uint8_t, __half, float,
                                                       __half>(
                reinterpret_cast<uint8_t *>(A.data_ptr()),
                reinterpret_cast<uint32_t *>(A_scale->data_ptr()),
                reinterpret_cast<__half *>(B.data_ptr<at::Half>()),
                reinterpret_cast<__half *>(C.data_ptr<at::Half>()), m, n, k,
                alpha, beta,
                bias == std::nullopt
                    ? nullptr
                    : reinterpret_cast<__half *>(bias->data_ptr<at::Half>()),
                kernelParam1, kernelParam2, kernelParam3);
        } else {
            TORCH_CHECK(false, "Unsupported just layoutA gemm kernelId");
        }
    } else if (B.dtype() == torch::kBFloat16) {
        // std::cout << "ABCkernel is bf16 !!!" << std::endl;
        if (kernelId == 1) {
            GemmMmaJustLayoutAKernelDispatch_Soft_Fp8<uint8_t, __maca_bfloat16,
                                                      float, __maca_bfloat16>(
                reinterpret_cast<uint8_t *>(A.data_ptr()),
                reinterpret_cast<uint32_t *>(A_scale->data_ptr()),
                reinterpret_cast<__maca_bfloat16 *>(B.data_ptr<at::BFloat16>()),
                reinterpret_cast<__maca_bfloat16 *>(C.data_ptr<at::BFloat16>()),
                m, n, k, alpha, beta,
                bias == std::nullopt ? nullptr
                                     : reinterpret_cast<__maca_bfloat16 *>(
                                           bias->data_ptr<at::BFloat16>()),
                kernelParam1, kernelParam2, kernelParam3);
        } else if (kernelId == 2) {
            GemmMmaJustLayoutAKernelDispatch2_Soft_Fp8<uint8_t, __maca_bfloat16,
                                                       float, __maca_bfloat16>(
                reinterpret_cast<uint8_t *>(A.data_ptr()),
                reinterpret_cast<uint32_t *>(A_scale->data_ptr()),
                reinterpret_cast<__maca_bfloat16 *>(B.data_ptr<at::BFloat16>()),
                reinterpret_cast<__maca_bfloat16 *>(C.data_ptr<at::BFloat16>()),
                m, n, k, alpha, beta,
                bias == std::nullopt ? nullptr
                                     : reinterpret_cast<__maca_bfloat16 *>(
                                           bias->data_ptr<at::BFloat16>()),
                kernelParam1, kernelParam2, kernelParam3);
        } else {
            TORCH_CHECK(false, "Unsupported just layoutA gemm kernelId");
        }
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }
    return C;
}

torch::Tensor gemm_layoutA_soft_fp8(torch::Tensor A,
                                    std::optional<torch::Tensor> A_scale,
                                    torch::Tensor B, float alpha, float beta,
                                    std::optional<torch::Tensor> bias) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    TORCH_CHECK(A.dim() == 4, "A must be a 4D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");

    TORCH_CHECK(A.size(2) == 16, "A.shape[2] must be 16");
    TORCH_CHECK(A.size(3) == 8, "A.shape[3] must be 8");
    int m = A.size(0) * 16, A_k = A.size(1) * 8;

    // TORCH_CHECK(B.size(2) == 4, "B.shape[2] must be 4");
    // TORCH_CHECK(B.size(3) == 16, "B.shape[3] must be 16");
    // TORCH_CHECK(B.size(4) == 8, "B.shape[4] must be 8");
    int B_k = B.size(1), n = B.size(0);

    TORCH_CHECK(A_k == B_k, "A and B must have the same k");
    int k = A_k;
    TORCH_CHECK(m % 128 == 0,
                "GEMM m % 128 == 0, currently, will be revised soon")
    TORCH_CHECK(k % 128 == 0, "GEMM k % 128 must be 0")

    // get GEMM args
    auto &&[kernelId, kernelParam1, kernelParam2, kernelParam3] =
        getGlobalGemmJustLayoutASoftFp8ArgSelector().getArgs({m, n, k});

    torch::Tensor C = gemm_layoutA_soft_fp8_wapper(
        A, A_scale, B, m, n, k, alpha, beta, kernelParam1, kernelParam2,
        kernelParam3, kernelId, bias);

    return C;
}

} // namespace muxi_layout_kernels
