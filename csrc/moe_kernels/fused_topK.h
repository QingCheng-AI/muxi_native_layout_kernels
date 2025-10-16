#pragma once
#include "group_gemm_utils.h"

namespace fused_softmax_topk {

#define DEBUG_ROW 0

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

template <typename T,
          /// Number of elements in the array
          int N,
          /// Alignment requirement in bytes
          int Alignment = sizeof(T) * N>
class alignas(Alignment) AlignedArray {
    T data[N];
};

template <typename T>
__device__ __forceinline__ T my_shfl_xor_sync(T var, int offset) {
    int index = (threadIdx.x % WARP_SIZE) ^ offset;
    int ret = __builtin_mxc_bsm_bpermute(index << 2, *((int *)&var));
    return *((T *)&ret);
}

template <typename T, int VPT, int NUM_EXPERTS, int BLOCK_SIZE,
          int BYTES_PER_LDG, int topK>
__global__ void __launch_bounds__(BLOCK_SIZE)
    fused_softmax_topk_kernel(const T *input, const int batchSize,
                              const int n_groups, const int topK_groups,
                              int *expertsIds, T *selectedExpertsWeights,
                              const T *bias) {
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_BLOCK =
        (BLOCK_SIZE / WARP_SIZE) * ROWS_PER_WARP;

    // ===== From this point, we finally start computing run-time variables.
    // =====
    const int blockBaseRow = blockIdx.x * ROWS_PER_BLOCK;
    const int warpBaseRow = blockBaseRow + threadIdx.y * ROWS_PER_WARP;
    const int threadRow = warpBaseRow + threadIdx.x / THREADS_PER_ROW;

    if (threadRow >= batchSize)
        return;

    // ===== compute self data ptr. =====
    const T *threadInputPtr = input + threadRow * NUM_EXPERTS;
    const int threadIdInGroup = threadIdx.x % THREADS_PER_ROW;
    const int firstEleReadByThread =
        threadIdInGroup * ELTS_PER_LDG * LDG_PER_THREAD;
    const T *threadReadPtr = threadInputPtr + firstEleReadByThread;

    // ===== Load data into register =====
    using AccessType = AlignedArray<T, ELTS_PER_LDG>;
    using INT4 = AlignedArray<int, 4>;
    T row_chunk[VPT];
    AccessType *row_chunk_vec_ptr = reinterpret_cast<AccessType *>(&row_chunk);
    const AccessType *vec_thread_read_ptr =
        reinterpret_cast<const AccessType *>(threadReadPtr);
    AccessType *weightOutputPtr = reinterpret_cast<AccessType *>(
        selectedExpertsWeights + threadRow * topK);
    INT4 *expertIdsOutputPtr =
        reinterpret_cast<INT4 *>(expertsIds + threadRow * topK);

#pragma unroll
    for (int i = 0; i < LDG_PER_THREAD; ++i) {
        // row_chunk_vec_ptr[i] = vec_thread_read_ptr[i * THREADS_PER_ROW];
        row_chunk_vec_ptr[i] = vec_thread_read_ptr[i];
    }

    // ===== find the max for safe-softmax. =====
    T thread_max_val = row_chunk[0];
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        thread_max_val =
            __hgt(thread_max_val, row_chunk[i]) ? thread_max_val : row_chunk[i];
    }
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
        T tmp = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, thread_max_val, mask,
                                THREADS_PER_ROW);
        thread_max_val = __hgt(thread_max_val, tmp) ? thread_max_val : tmp;
    }

    // ===== find the sum for safe-softmax. =====
    float thread_sum_val = 0.0;
    float row_chunk_float[VPT];
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        float tmp_float = expf((float)row_chunk[i] - (float)thread_max_val);
        row_chunk[i] = (T)tmp_float;
        row_chunk_float[i] = tmp_float;
        thread_sum_val = thread_sum_val + tmp_float;
    }
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
        thread_sum_val += __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, thread_sum_val,
                                          mask, THREADS_PER_ROW);
    }

    // ===== compute the softmax =====
    const float softmax_row_factor = 1.0f / thread_sum_val;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        row_chunk[i] = (T)(row_chunk_float[i] * softmax_row_factor);
    }

    // Add the bias to softmax values.
    T row_chunk_bias[VPT];
    if (bias != nullptr) {
        // const T *threadBiasPtr = bias + threadRow * NUM_EXPERTS;
        const T *biasReadPtr = bias + firstEleReadByThread;
        const AccessType *bias_vec_ptr =
            reinterpret_cast<const AccessType *>(biasReadPtr);
        AccessType *row_chunk_bias_vec_ptr =
            reinterpret_cast<AccessType *>(row_chunk_bias);
        for (int i = 0; i < LDG_PER_THREAD; ++i) {
            row_chunk_bias_vec_ptr[i] = bias_vec_ptr[i];
        }
        for (int i = 0; i < VPT; ++i) {
            row_chunk_bias[i] = __hadd(row_chunk[i], row_chunk_bias[i]);
        }
    } else {
        for (int i = 0; i < VPT; ++i) {
            row_chunk_bias[i] = row_chunk[i];
        }
    }

    // ===== try to support the grouped experts for deepseek. =====
    // ===== First, find the experts_group max (or sum max 2 with bias) for
    // group score. =====
    int topInGroup = (bias == nullptr) ? 1 : 2;
    const int experts_per_group = NUM_EXPERTS / n_groups;
    const int experts_group_id = threadIdInGroup * VPT / experts_per_group;
    int max_id = -1;
    T max_score_in_experts_group = 0;
    for (int k = 0; k < topInGroup; k++) {
        T max_score_in_experts_group_tmp = 0.0;
        int max_id_tmp = -1;
        for (int i = 0; i < VPT; ++i) {
            if ((threadIdInGroup * VPT + i) == max_id)
                continue;
            if (__hgt(row_chunk_bias[i], max_score_in_experts_group_tmp)) {
                max_score_in_experts_group_tmp = row_chunk_bias[i];
                max_id_tmp = threadIdInGroup * VPT + i;
            }
        }

        for (int mask = (experts_per_group / VPT) / 2; mask > 0; mask >>= 1) {
            T tmp = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF,
                                    max_score_in_experts_group_tmp, mask,
                                    THREADS_PER_ROW);
            int tmp_id = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, max_id_tmp, mask,
                                         THREADS_PER_ROW);

            // keep the lower expert_id "win"
            if (__hgt(tmp, max_score_in_experts_group_tmp) ||
                (__heq(tmp, max_score_in_experts_group_tmp) &&
                 tmp_id < max_id_tmp)) {
                max_score_in_experts_group_tmp = tmp;
                max_id_tmp = tmp_id;
            }
        }
        max_id = max_id_tmp;
        max_score_in_experts_group =
            __hadd(max_score_in_experts_group_tmp, max_score_in_experts_group);
    }
    // ===== Second, find the topK_groups. =====
    const int threadIdInexpertsGroup =
        threadIdInGroup % (experts_per_group / VPT);
    // Simply consider that topK is greater than or equal topK_groups
    T topK_groups_weights[topK];
    int topK_groups_id[topK];
    int max_experts_group_id = experts_group_id;
    T max_tmp = max_score_in_experts_group;
    for (int i = 0; i < topK_groups; i++) {
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
            T tmp = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, max_tmp, mask,
                                    THREADS_PER_ROW);
            int tmp_expert_id =
                __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, max_experts_group_id, mask,
                                THREADS_PER_ROW);
            if (__hgt(tmp, max_tmp) ||
                (__heq(tmp, max_tmp) && tmp_expert_id < max_experts_group_id)) {
                max_tmp = tmp;
                max_experts_group_id = tmp_expert_id;
            }
        }
        topK_groups_weights[i] = max_tmp;
        topK_groups_id[i] = max_experts_group_id;

        // TODO: clear the max value by set 0.0
        if (experts_group_id == max_experts_group_id) {
            max_score_in_experts_group = 0.0;
        }
        max_experts_group_id = experts_group_id;
        max_tmp = max_score_in_experts_group;
    }

    // ===== Third, set the softmax value not in topK_groups to 0. =====
    bool set_zero = true;
    for (int i = 0; i < topK_groups; ++i) {
        if (topK_groups_id[i] == experts_group_id) {
            set_zero = false;
            break;
        }
    }
    if (set_zero) {
        for (int i = 0; i < VPT; ++i) {
            row_chunk[i] = 0.0;
            row_chunk_bias[i] = 0.0;
        }
    }

    // ===== Now softmax compute finished, we can find the topK by argmax
    // first.
    int start_expert_id = firstEleReadByThread;
    static constexpr int EXPERTS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    T topK_weights[topK];
    T topK_weights_bias[topK];
    int topK_expert_ids[topK];

    for (int kid = 0; kid < topK; ++kid) {
        // First, each thread does the local argmax
        T max_val = row_chunk_bias[0];
        T max_val_no_bias = row_chunk[0];
        int expert_id = start_expert_id;
#pragma unroll
        for (int ldg = 0, experts = start_expert_id; ldg < LDG_PER_THREAD;
             ++ldg, experts += ELTS_PER_LDG) {
#pragma unroll
            for (int i = 0; i < ELTS_PER_LDG; ++i) {
                T val = row_chunk_bias[ldg * ELTS_PER_LDG + i];
                if (__hgt(val, max_val)) {
                    max_val = val;
                    expert_id = experts + i;
                    max_val_no_bias = row_chunk[ldg * ELTS_PER_LDG + i];
                }
            }
        }

// Second, use butterfly to find the global max
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
            T other_max_val = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, max_val, mask,
                                              THREADS_PER_ROW);
            int other_expert_id = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, expert_id,
                                                  mask, THREADS_PER_ROW);
            T other_max_val_no_bias = __shfl_xor_sync(
                0xFFFFFFFFFFFFFFFF, max_val_no_bias, mask, THREADS_PER_ROW);

            // keep the lower expert_id "win"
            if (__hgt(other_max_val, max_val) ||
                (__heq(other_max_val, max_val) &&
                 (other_expert_id < expert_id))) {
                max_val = other_max_val;
                expert_id = other_expert_id;
                max_val_no_bias = other_max_val_no_bias;
            }
        }

        // Third, write the result back to global memory
        if (threadIdInGroup == 0) {
            // const int output_offset = threadRow * topK + kid;
            // selectedExpertsWeights[output_offset] = max_val;
            // expertsIds[output_offset] = expert_id - start_expert_id;
            topK_weights[kid] = max_val_no_bias;
            topK_expert_ids[kid] = expert_id - start_expert_id;
            topK_weights_bias[kid] = max_val;
        }

        // Finally, clear the max value in buffer for next iteration
        if (kid + 1 < topK) {
            const int ldg_group_for_expert = expert_id / EXPERTS_PER_GROUP_LDG;
            const int thread_to_clear_in_group =
                (expert_id / ELTS_PER_LDG) % THREADS_PER_ROW;
            if (threadIdInGroup == thread_to_clear_in_group) {
                const int offset_for_expert = expert_id % ELTS_PER_LDG;
                row_chunk_bias[ldg_group_for_expert * ELTS_PER_LDG +
                               offset_for_expert] = 0.0f;
            }
        }
    }

    __syncthreads();

    // ===== sort the topK selected experts by expert id. =====
    // This is a simple sorting algorithm, but it should work well for our case.
    // A more efficient sorting algorithm may be needed if the dataset is very
    // large.
    if (threadIdInGroup == 0) {
        for (int kid = 0; kid < topK - 1; ++kid) {
            for (int jid = kid + 1; jid < topK; ++jid) {
                if (topK_expert_ids[kid] > topK_expert_ids[jid]) {
                    T tmpWeight = topK_weights[kid];
                    topK_weights[kid] = topK_weights[jid];
                    topK_weights[jid] = tmpWeight;
                    int tmpId = topK_expert_ids[kid];
                    topK_expert_ids[kid] = topK_expert_ids[jid];
                    topK_expert_ids[jid] = tmpId;
                }
            }
        }

        int index_t = 0, int_t = 0;
        int output_offset = threadRow * topK;
        for (index_t = 0; index_t < topK / ELTS_PER_LDG; ++index_t) {
            weightOutputPtr[index_t] = *reinterpret_cast<AccessType *>(
                &topK_weights[index_t * ELTS_PER_LDG]);
        }
        for (int_t = 0; int_t < topK / (BYTES_PER_LDG / sizeof(int)); ++int_t) {
            expertIdsOutputPtr[int_t] = *reinterpret_cast<INT4 *>(
                &topK_expert_ids[int_t * (BYTES_PER_LDG / sizeof(int))]);
        }
        for (index_t = index_t * ELTS_PER_LDG; index_t < topK; ++index_t) {
            selectedExpertsWeights[output_offset + index_t] =
                topK_weights[index_t];
        }
        for (int_t = int_t * (BYTES_PER_LDG / sizeof(int)); int_t < topK;
             ++int_t) {
            expertsIds[output_offset + int_t] = topK_expert_ids[int_t];
        }
    }
}

template <typename T, int VPT, int NUM_EXPERTS, int BLOCK_SIZE,
          int BYTES_PER_LDG, int topK>
__global__ void __launch_bounds__(BLOCK_SIZE)
    fused_sigmoid_topk_kernel(const T *input, const int batchSize,
                              const int n_groups, const int topK_groups,
                              int *expertsIds, T *selectedExpertsWeights,
                              const T *bias) {
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_BLOCK =
        (BLOCK_SIZE / WARP_SIZE) * ROWS_PER_WARP;

    // ===== From this point, we finally start computing run-time variables.
    // =====
    const int blockBaseRow = blockIdx.x * ROWS_PER_BLOCK;
    const int warpBaseRow = blockBaseRow + threadIdx.y * ROWS_PER_WARP;
    const int threadRow = warpBaseRow + threadIdx.x / THREADS_PER_ROW;

    if (threadRow >= batchSize)
        return;

    // ===== compute self data ptr. =====
    const T *threadInputPtr = input + threadRow * NUM_EXPERTS;
    const int threadIdInGroup = threadIdx.x % THREADS_PER_ROW;
    const int firstEleReadByThread =
        threadIdInGroup * ELTS_PER_LDG * LDG_PER_THREAD;
    const T *threadReadPtr = threadInputPtr + firstEleReadByThread;

    // ===== Load data into register =====
    using AccessType = AlignedArray<T, ELTS_PER_LDG>;
    using INT4 = AlignedArray<int, 4>;
    T row_chunk[VPT];
    AccessType *row_chunk_vec_ptr = reinterpret_cast<AccessType *>(&row_chunk);
    const AccessType *vec_thread_read_ptr =
        reinterpret_cast<const AccessType *>(threadReadPtr);
    AccessType *weightOutputPtr = reinterpret_cast<AccessType *>(
        selectedExpertsWeights + threadRow * topK);
    INT4 *expertIdsOutputPtr =
        reinterpret_cast<INT4 *>(expertsIds + threadRow * topK);

#pragma unroll
    for (int i = 0; i < LDG_PER_THREAD; ++i) {
        // row_chunk_vec_ptr[i] = vec_thread_read_ptr[i * THREADS_PER_ROW];
        row_chunk_vec_ptr[i] = vec_thread_read_ptr[i];
    }

    // ===== compute sigmoid values =====
    for (int i = 0; i < VPT; ++i) {
        row_chunk[i] = (T)(1.0f / (1.0f + expf(-(float)row_chunk[i])));
    }

    // Add the bias to sigmoid values.
    T row_chunk_bias[VPT];
    if (bias != nullptr) {
        // const T *threadBiasPtr = bias + threadRow * NUM_EXPERTS;
        const T *biasReadPtr = bias + firstEleReadByThread;
        const AccessType *bias_vec_ptr =
            reinterpret_cast<const AccessType *>(biasReadPtr);
        AccessType *row_chunk_bias_vec_ptr =
            reinterpret_cast<AccessType *>(row_chunk_bias);
        for (int i = 0; i < LDG_PER_THREAD; ++i) {
            row_chunk_bias_vec_ptr[i] = bias_vec_ptr[i];
        }
        for (int i = 0; i < VPT; ++i) {
            row_chunk_bias[i] = __hadd(row_chunk[i], row_chunk_bias[i]);
        }
    } else {
        for (int i = 0; i < VPT; ++i) {
            row_chunk_bias[i] = row_chunk[i];
        }
    }

    // ===== try to support the grouped experts for deepseek. =====
    // ===== First, find the experts_group max (or sum max 2 with bias) for
    // group score. =====
    int topInGroup = (bias == nullptr) ? 1 : 2;
    const int experts_per_group = NUM_EXPERTS / n_groups;
    const int experts_group_id = threadIdInGroup * VPT / experts_per_group;
    int start_thread_for_group = experts_group_id * (experts_per_group / VPT);
    int max_id = -1;
    int previous_max_id = -1;
    T max_score_in_experts_group = 0.0;
    for (int k = 0; k < topInGroup; k++) {
        T max_score_in_experts_group_tmp = 0.0;
        for (int i = 0; i < VPT; ++i) {
            // if (threadIdInGroup * VPT + i == max_id) continue;
            T current_score = row_chunk_bias[i];
            if ((threadIdInGroup - start_thread_for_group) * VPT + i ==
                previous_max_id) {
                current_score = max_score_in_experts_group_tmp;
            }
            if (__hgt(current_score, max_score_in_experts_group_tmp)) {
                max_score_in_experts_group_tmp = current_score;
                max_id = (threadIdInGroup - start_thread_for_group) * VPT + i;
            }
        }
        // printf("experts_per_group %d, VPT %d\n", experts_per_group, VPT);
        for (int mask = (experts_per_group / VPT) / 2; mask > 0; mask >>= 1) {
            T tmp = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF,
                                    max_score_in_experts_group_tmp, mask,
                                    THREADS_PER_ROW);
            int tmp_id = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, max_id, mask,
                                         THREADS_PER_ROW);
            if (__hgt(tmp, max_score_in_experts_group_tmp) ||
                (__heq(tmp, max_score_in_experts_group_tmp) &&
                 tmp_id < max_id)) {
                max_score_in_experts_group_tmp = tmp;
                max_id = tmp_id;
            }
        }
        previous_max_id = max_id;
        max_score_in_experts_group =
            __hadd(max_score_in_experts_group_tmp, max_score_in_experts_group);
    }

    // ===== Second, find the topK_groups. =====
    const int threadIdInexpertsGroup =
        threadIdInGroup % (experts_per_group / VPT);
    // Simply consider that topK is greater than or equal topK_groups
    T topK_groups_weights[topK];
    int topK_groups_id[topK];
    int max_experts_group_id = experts_group_id;
    T max_tmp = max_score_in_experts_group;
    for (int i = 0; i < topK_groups; i++) {
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
            T tmp = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, max_tmp, mask,
                                    THREADS_PER_ROW);
            int tmp_expert_id =
                __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, max_experts_group_id, mask,
                                THREADS_PER_ROW);
            if (__hgt(tmp, max_tmp) ||
                (__heq(tmp, max_tmp) && tmp_expert_id < max_experts_group_id)) {
                max_tmp = tmp;
                max_experts_group_id = tmp_expert_id;
            }
        }
        topK_groups_weights[i] = max_tmp;
        topK_groups_id[i] = max_experts_group_id;

        // TODO: clear the max value by set 0.0
        if (experts_group_id == max_experts_group_id) {
            max_score_in_experts_group = 0.0;
        }
        max_experts_group_id = experts_group_id;
        max_tmp = max_score_in_experts_group;
    }

    // ===== Third, set the sigmoid value not in topK_groups to 0. =====
    bool set_zero = true;
    for (int i = 0; i < topK_groups; ++i) {
        if (topK_groups_id[i] == experts_group_id) {
            set_zero = false;
            break;
        }
    }
    if (set_zero) {
        for (int i = 0; i < VPT; ++i) {
            row_chunk[i] = 0.0;
            row_chunk_bias[i] = 0.0;
        }
    }

    // ===== Now softmax compute finished, we can find the topK by argmax
    // first.
    int start_expert_id = firstEleReadByThread;
    static constexpr int EXPERTS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    T topK_weights[topK];
    T topK_weights_bias[topK];
    int topK_expert_ids[topK];

    for (int kid = 0; kid < topK; ++kid) {
        // First, each thread does the local argmax
        T max_val = row_chunk_bias[0];
        T max_val_no_bias = row_chunk[0];
        int expert_id = start_expert_id;
#pragma unroll
        for (int ldg = 0, experts = start_expert_id; ldg < LDG_PER_THREAD;
             ++ldg, experts += ELTS_PER_LDG) {
#pragma unroll
            for (int i = 0; i < ELTS_PER_LDG; ++i) {
                T val = row_chunk_bias[ldg * ELTS_PER_LDG + i];
                if (__hgt(val, max_val)) {
                    max_val = val;
                    expert_id = experts + i;
                    max_val_no_bias = row_chunk[ldg * ELTS_PER_LDG + i];
                }
            }
        }

// Second, use butterfly to find the global max
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
            T other_max_val = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, max_val, mask,
                                              THREADS_PER_ROW);
            int other_expert_id = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, expert_id,
                                                  mask, THREADS_PER_ROW);
            T other_max_val_no_bias = __shfl_xor_sync(
                0xFFFFFFFFFFFFFFFF, max_val_no_bias, mask, THREADS_PER_ROW);

            // keep the lower expert_id "win"
            if (__hgt(other_max_val, max_val) ||
                (__heq(other_max_val, max_val) &&
                 (other_expert_id < expert_id))) {
                max_val = other_max_val;
                expert_id = other_expert_id;
                max_val_no_bias = other_max_val_no_bias;
            }
        }

        // Third, write the result back to global memory
        if (threadIdInGroup == 0) {
            // const int output_offset = threadRow * topK + kid;
            // selectedExpertsWeights[output_offset] = max_val;
            // expertsIds[output_offset] = expert_id - start_expert_id;
            topK_weights[kid] = max_val_no_bias;
            topK_expert_ids[kid] = expert_id - start_expert_id;
            topK_weights_bias[kid] = max_val;
        }

        // Finally, clear the max value in buffer for next iteration
        if (kid + 1 < topK) {
            const int ldg_group_for_expert = expert_id / EXPERTS_PER_GROUP_LDG;
            const int thread_to_clear_in_group =
                (expert_id / ELTS_PER_LDG) % THREADS_PER_ROW;
            if (threadIdInGroup == thread_to_clear_in_group) {
                const int offset_for_expert = expert_id % ELTS_PER_LDG;
                // asm("/* check here 0.0! */");
                row_chunk_bias[ldg_group_for_expert * ELTS_PER_LDG +
                               offset_for_expert] = (T)0.0f;
                // asm("/* check here 0.0! */");
            }
        }
    }

    __syncthreads();

    // ===== sort the topK selected experts by expert id. =====
    // This is a simple sorting algorithm, but it should work well for our case.
    // A more efficient sorting algorithm may be needed if the dataset is very
    // large.
    if (threadIdInGroup == 0) {
        // for (int kid = 0; kid < topK - 1; ++kid) {
        //   for (int jid = kid + 1; jid < topK; ++jid) {
        //     if (topK_expert_ids[kid] > topK_expert_ids[jid]) {
        //       T tmpWeight = topK_weights[kid];
        //       topK_weights[kid] = topK_weights[jid];
        //       topK_weights[jid] = tmpWeight;
        //       int tmpId = topK_expert_ids[kid];
        //       topK_expert_ids[kid] = topK_expert_ids[jid];
        //       topK_expert_ids[jid] = tmpId;
        //     }
        //   }
        // }

        int index_t = 0, int_t = 0;
        int output_offset = threadRow * topK;
        for (index_t = 0; index_t < topK / ELTS_PER_LDG; ++index_t) {
            weightOutputPtr[index_t] = *reinterpret_cast<AccessType *>(
                &topK_weights[index_t * ELTS_PER_LDG]);
        }
        for (int_t = 0; int_t < topK / (BYTES_PER_LDG / sizeof(int)); ++int_t) {
            expertIdsOutputPtr[int_t] = *reinterpret_cast<INT4 *>(
                &topK_expert_ids[int_t * (BYTES_PER_LDG / sizeof(int))]);
        }
        for (index_t = index_t * ELTS_PER_LDG; index_t < topK; ++index_t) {
            selectedExpertsWeights[output_offset + index_t] =
                topK_weights[index_t];
        }
        for (int_t = int_t * (BYTES_PER_LDG / sizeof(int)); int_t < topK;
             ++int_t) {
            expertsIds[output_offset + int_t] = topK_expert_ids[int_t];
        }
    }
}

namespace detail {
// Constructs some constants needed to partition the work across threads at
// compile time.
template <typename T, int EXPERTS, int BYTES_PER_LDG> struct TopkConstants {
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 ||
                      EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0,
                  "");
    static constexpr int VECs_PER_THREAD =
        MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    //   static const int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};
} // namespace detail

template <typename T, int EXPERTS>
void fused_softmax_topk_dispatcher(const T *input, const int score_fun,
                                   const int batchSize, const int n_groups,
                                   const int topK_groups, int *expertsIds,
                                   T *selectedExpertsWeights, const int topK,
                                   const T *bias) {
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int BYTES_PER_LDG = 16;

    using Constants = detail::TopkConstants<T, EXPERTS, BYTES_PER_LDG>;

    static constexpr int A = Constants::ELTS_PER_LDG;
    static constexpr int VECs_PER_THREAD = Constants::VECs_PER_THREAD;
    static constexpr int THREADS_PER_ROW = Constants::THREADS_PER_ROW;
    static constexpr int VPT = Constants::VPT;
    const int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;

    const int numWarps = (batchSize + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int numBlocks =
        (numWarps + BLOCK_SIZE / WARP_SIZE - 1) / (BLOCK_SIZE / WARP_SIZE);

    dim3 block_dim(WARP_SIZE, BLOCK_SIZE / WARP_SIZE);

#define LAUNCH_FUSED_TOPK(TOPK)                                                \
    if (score_fun == 0) {                                                      \
        fused_softmax_topk_kernel<T, VPT, EXPERTS, BLOCK_SIZE, BYTES_PER_LDG,  \
                                  TOPK><<<numBlocks, block_dim, 0, stream>>>(  \
            input, batchSize, n_groups, topK_groups, expertsIds,               \
            selectedExpertsWeights, bias);                                     \
    } else if (score_fun == 1) {                                               \
        fused_sigmoid_topk_kernel<T, VPT, EXPERTS, BLOCK_SIZE, BYTES_PER_LDG,  \
                                  TOPK><<<numBlocks, block_dim, 0, stream>>>(  \
            input, batchSize, n_groups, topK_groups, expertsIds,               \
            selectedExpertsWeights, bias);                                     \
    }

    const mcStream_t stream =
        at::cuda::getCurrentCUDAStream(at::cuda::current_device());
    switch (topK) {
    // case 1:
    //   LAUNCH_FUSED_TOPK(1);
    //   break;
    // case 2:
    //   LAUNCH_FUSED_TOPK(2);
    //   break;
    // case 4:
    //   LAUNCH_FUSED_TOPK(4);
    //   break;
    case 8:
        LAUNCH_FUSED_TOPK(8);
        break;

    default:
        assert(false &&
               "Unsupported topK value, just 1, 2, 4, 8 are supported now.");
        break;
    }
} // namespace fused_softmax_topk

#define LAUNCH_SOFTMAX_TOPK(NUM_EXPERTS)                                       \
    fused_softmax_topk_dispatcher<T, NUM_EXPERTS>(                             \
        input, score_fun, batchSize, n_groups, topK_groups, expertsIds,        \
        selectedExpertsWeights, topK, bias)

template <typename T>
void fused_softmax_topk_launcher(const T *input, int score_fun,
                                 const int batchSize, const int n_groups,
                                 const int topK_groups, int *expertsIds,
                                 T *selectedExpertsWeights, const int topK,
                                 const int numExperts,
                                 const T *bias = nullptr) {
    switch (numExperts) {
    case 1:
        LAUNCH_SOFTMAX_TOPK(1);
        break;
    case 2:
        LAUNCH_SOFTMAX_TOPK(2);
        break;
    case 4:
        LAUNCH_SOFTMAX_TOPK(4);
        break;
    case 8:
        LAUNCH_SOFTMAX_TOPK(8);
        break;
    case 16:
        LAUNCH_SOFTMAX_TOPK(16);
        break;
    case 32:
        LAUNCH_SOFTMAX_TOPK(32);
        break;
    case 64:
        LAUNCH_SOFTMAX_TOPK(64);
        break;
    case 128:
        LAUNCH_SOFTMAX_TOPK(128);
        break;
    case 256:
        LAUNCH_SOFTMAX_TOPK(256);
        break;

    default:
        assert(
            false &&
            "fused_softmax_topK num_experts must be a power of 2 and <= 256.");
        break;
    }
}

} // namespace fused_softmax_topk
