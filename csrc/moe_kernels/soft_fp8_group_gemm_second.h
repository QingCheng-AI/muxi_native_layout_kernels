#pragma once
#include "group_gemm_utils.h"

template <typename Ta, typename Tb, typename Taccum, typename Tc,
          int BLOCK_DIM_X, int tile_m, int tile_n, int tile_k,
          int micro_batchsize, int ScaleBlockM, int ScaleBlockN>
__device__ void fused_moe_second_gemm_kernel(
    Ta *A, Tb *B, Tc *C, int *sorted_token_ids, int m, int n, int k,
    Tc *routing_weights, int padded_ptr_number, int topk, int block_x_id,
    int block_y_id, Taccum *A_scale, int A_scale_m, int A_scale_n) {
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

    const int blockColsGroup = __builtin_mxc_readfirstlane(block_y_id);
    const int blockRowsGroup = __builtin_mxc_readfirstlane(block_x_id);
    constexpr int m_times = (tile_m / rowThreadsPerMma);
    constexpr int n_times = (tile_n / rowThreadsPerMma);
    constexpr int loadAPerStage = ((m_times + warpPerBlock - 1) / warpPerBlock);
    constexpr int loadBPerStage = ((n_times + warpPerBlock - 1) / warpPerBlock);
    const int kChunks = (k / tile_k);

    using CStgType = __NATIVE_VECTOR__(sizeof(Tc), uint);

    __shared__ UINT4 shared_data[micro_batchsize * tile_k / elementsPerAccess];
    UINT4 *shared_B_stages[stages];
    UINT2 *A_ptr_stages[stages];
    Taccum *A_scale_ptr[stages];
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
            thread_token_weight[j] = 0;
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
        A_ptr_stages[i] = reinterpret_cast<UINT2 *>(A) +
                          blockRowsGroup * tile_m * k / elementsPerAccess +
                          i * WARP_SIZE + laneId;
        A_scale_ptr[i] = A_scale +
                         (blockRowsGroup * tile_m) / ScaleBlockM * A_scale_n +
                         (i * colThreadsPerMma * elementsPerAccess +
                          quarterWarpId * elementsPerAccess) /
                             ScaleBlockN;
    }

    // return the threads without task
    if ((blockRowsGroup >= ((m + tile_m - 1) / tile_m)) ||
        (blockColsGroup >= (n + tile_n - 1) / tile_n)) {
        return;
    }

    UINT2 tmpA[stages][m_times / warpPerBlock];
    UINT4 tmpB[n_times];
    uint tmpA_bf16[4];
    Taccum scale[stages][m_times / warpPerBlock];
    FLOAT4 C_f32[m_times / warpPerBlock][n_times];
    for (int i = 0; i < m_times / warpPerBlock; i++) {
        for (int j = 0; j < n_times; j++) {
            C_f32[i][j] = {0, 0, 0, 0};
        }
    }

    const int readA_offset = warpIdInBlock * (m_times / warpPerBlock);

    // prefill the pipeline, prefetch stages data.
    for (int i = 0; i < stages; i++) {
        for (int j = 0; j < loadAPerStage; j++) {
            tmpA[i][j] = mxc_read64_async(reinterpret_cast<INT64 *>(
                A_ptr_stages[i] +
                (readA_offset + j) * rowThreadsPerMma * k / elementsPerAccess));
            scale[i][j] =
                *(A_scale_ptr[i] + (readA_offset + j) * rowThreadsPerMma /
                                       ScaleBlockM * A_scale_n);
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
                B_ptr_stages[i][j], predicatorB, 1, MACA_ICMP_EQ);
            B_ptr_stages[i][j] += colThreadsPerMma * stages;
        }

        // Added A_ptr and B_ptr
        A_ptr_stages[i] += WARP_SIZE * stages;
        A_scale_ptr[i] +=
            colThreadsPerMma * elementsPerAccess * stages / ScaleBlockN;
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
            for (int j = 0; j < m_times / warpPerBlock; j++) {
                scalefp8tobf16(tmpA[stage][j][0],
                               reinterpret_cast<uint16_t *>(tmpA_bf16),
                               scale[stage][j]);
                scalefp8tobf16(tmpA[stage][j][1],
                               reinterpret_cast<uint16_t *>(tmpA_bf16 + 2),
                               scale[stage][j]);
                for (int k = 0; k < n_times; k++) {
                    C_f32[j][k] = mma_16x16x16f16<Tb>(tmpA_bf16[0],
                                                      tmpA_bf16[1], tmpB[k][0],
                                                      tmpB[k][1], C_f32[j][k]);
                    C_f32[j][k] = mma_16x16x16f16<Tb>(tmpA_bf16[2],
                                                      tmpA_bf16[3], tmpB[k][2],
                                                      tmpB[k][3], C_f32[j][k]);
                }
            }

            __builtin_mxc_barrier_inst();

            // async load next stage data
            {
                for (int j = 0; j < loadAPerStage; j++) {
                    tmpA[stage][j] = mxc_read64_async(reinterpret_cast<INT64 *>(
                        A_ptr_stages[stage] + (readA_offset + j) *
                                                  rowThreadsPerMma * k /
                                                  elementsPerAccess));
                    scale[stage][j] = *(A_scale_ptr[stage] +
                                        (readA_offset + j) * rowThreadsPerMma /
                                            ScaleBlockN * A_scale_n);
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
                        B_ptr_stages[stage][j], predicatorB, 1, MACA_ICMP_EQ);
                    B_ptr_stages[stage][j] += colThreadsPerMma * stages;
                }

                // Added A_ptr and B_ptr
                A_ptr_stages[stage] += WARP_SIZE * stages;
                A_scale_ptr[stage] +=
                    colThreadsPerMma * elementsPerAccess * stages / ScaleBlockN;
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
        for (int j = 0; j < m_times / warpPerBlock; j++) {
            scalefp8tobf16(tmpA[stage][j][0],
                           reinterpret_cast<uint16_t *>(tmpA_bf16),
                           scale[stage][j]);
            scalefp8tobf16(tmpA[stage][j][1],
                           reinterpret_cast<uint16_t *>(tmpA_bf16 + 2),
                           scale[stage][j]);
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
                    tc_tmp[t] = __hmul(static_cast<Tc>(C_f32[i][j][t]),
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
    fused_moe_second_gemm_kernel<Ta, Tb, Taccum, Tc, BLOCK_DIM_X, tile_m,
                                 tile_n, tile_k, micro_batchsize, ScaleBlockM,
                                 ScaleBlockN>(
        A + experts_ids[gemm_id] * m * k, B, C,
        sorted_token_ids + gemm_id * micro_batchsize, m, n, k, routing_weights,
        padded_ptr_number, topk, block_x_id_in_gemm, block_y_id_in_gemm,
        A_scale + experts_ids[gemm_id] * A_scale_m * A_scale_n, A_scale_m,
        A_scale_n);
}
