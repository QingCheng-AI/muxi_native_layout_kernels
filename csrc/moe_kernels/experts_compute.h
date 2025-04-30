#include "copyAndLayoutB.h"
#include "group_gemm.h"
#include "group_gemm_utils.h"
#include "muxi_hgemm_layout.h"
#include "muxi_hgemm_layout_fused.h"
#include "silu_and_mul_kernel.h"

template <typename W, typename Taccum, typename A>
void experts_compute(W **experts_weights_matrix1, W **experts_weights_matrix2,
                     A *activations, int m1, int n1, int k1, int m2, int n2,
                     int k2, int batchSize, int expertCount,
                     int dynamicExpertsPerAct, int *expertsIds,
                     A *activedExpertsWeights, A **bias) {
    constexpr mcStream_t streamId = 0;

    A *dev_gemm_result_buffer;
    mcMalloc((void **)&dev_gemm_result_buffer,
             sizeof(A) * m1 * n1 * expertCount);
    A **B_group_gemm2_dev;
    mcMalloc((void ***)&B_group_gemm2_dev, sizeof(A *) * expertCount);
    int *samplesForExperts =
        (int *)malloc(expertCount * (batchSize + 1) * sizeof(int));
    int *dev_samplesForExperts;
    mcMalloc((void **)&dev_samplesForExperts,
             sizeof(int) * expertCount * (batchSize + 1));
    A *scoreWeightsForGEMMs = (A *)malloc(sizeof(A) * expertCount * batchSize);
    A *dev_B_expert_buffers;
    mcMalloc((void **)&dev_B_expert_buffers,
             (sizeof(A) * expertCount * n1 * k1));
    // for memcpy group gemm args from host to device
    W **dev_A1;
    mcMalloc((void ***)&dev_A1, sizeof(W *) * expertCount);
    A **dev_B1;
    mcMalloc((void ***)&dev_B1, sizeof(A *) * expertCount);
    A **dev_C1;
    mcMalloc((void ***)&dev_C1, sizeof(A *) * expertCount);
    W **dev_A2;
    mcMalloc((void ***)&dev_A2, sizeof(W *) * expertCount);

    // some Async device op can be added heres
    mcMemsetD32Async(dev_B_expert_buffers, 0,
                     sizeof(A) * expertCount * n1 * k1 / 4, streamId);
    mcMemsetD32Async(dev_gemm_result_buffer, 0,
                     sizeof(A) * m1 * n1 * expertCount / 4, streamId);

    /////////////////////////////
    // samplesForExperts :
    // 1  | 0
    // 3  | 4, 7, 9
    // 0  |
    // 2  | 5, 9
    // ……
    /////////////////////////////
    for (int i = 0; i < expertCount; i++) {
        samplesForExperts[i * (batchSize + 1)] = 0;
    }
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < dynamicExpertsPerAct; j++) {
            int index =
                samplesForExperts[(expertsIds[i * dynamicExpertsPerAct + j]) *
                                  (batchSize + 1)];
            samplesForExperts[(expertsIds[i * dynamicExpertsPerAct + j]) *
                                  (batchSize + 1) +
                              index + 1] = i;
            samplesForExperts[(expertsIds[i * dynamicExpertsPerAct + j]) *
                              (batchSize + 1)]++;
        }
    }

    mcMemcpyAsync(dev_samplesForExperts, samplesForExperts,
                  sizeof(int) * expertCount * (batchSize + 1),
                  mcMemcpyHostToDevice);

    int gemmCount = 0;
    for (int i = 0; i < expertCount; i++) {
        if (samplesForExperts[i * (batchSize + 1)] > 0) {
            gemmCount++;
        }
    }

    memset(scoreWeightsForGEMMs, 0, sizeof(A) * gemmCount * batchSize);

    int gemm_id = 0;
    for (int i = 0; i < expertCount; i++) {
        if (samplesForExperts[i * (batchSize + 1)] > 0) {
            for (int j = 1; j < samplesForExperts[i * (batchSize + 1)] + 1;
                 j++) {
                int sample_id = samplesForExperts[i * (batchSize + 1) + j];
                int expert_id = i;
                for (int k = 0; k < dynamicExpertsPerAct; k++) {
                    if (expertsIds[sample_id * dynamicExpertsPerAct + k] ==
                        expert_id) {
                        scoreWeightsForGEMMs[gemm_id * batchSize + sample_id] =
                            activedExpertsWeights[sample_id *
                                                      dynamicExpertsPerAct +
                                                  k];
                        break;
                    }
                }
            }
            gemm_id++;
        }
    }

    A *dev_scoreWeightsForGEMMs;
    mcMalloc((void **)&dev_scoreWeightsForGEMMs,
             sizeof(A) * gemmCount * batchSize);
    mcMemcpy(dev_scoreWeightsForGEMMs, scoreWeightsForGEMMs,
             sizeof(A) * gemmCount * batchSize, mcMemcpyHostToDevice);

    // make A and B pointer array for group gemm w1
    W **A_group_gemm = (W **)malloc(sizeof(W *) * gemmCount);
    A **B_group_gemm = (A **)malloc(sizeof(A *) * gemmCount);
    A **C_group_gemm = (A **)malloc(sizeof(A *) * gemmCount);
    A **A_group_gemm2 = (A **)malloc(sizeof(A *) * gemmCount);

    gemm_id = 0;
    for (int i = 0; i < expertCount; i++) {
        if (samplesForExperts[i * (batchSize + 1)] > 0) {
            A_group_gemm[gemm_id] = experts_weights_matrix1[i];
            B_group_gemm[gemm_id] = dev_B_expert_buffers + i * n1 * k1;
            C_group_gemm[gemm_id] = dev_gemm_result_buffer + i * n1 * m1;
            gemm_id++;
        }
    }

    // set w2 group gemm args
    gemm_id = 0;
    for (int i = 0; i < expertCount; i++) {
        if (samplesForExperts[i * (batchSize + 1)] > 0) {
            A_group_gemm2[gemm_id] = experts_weights_matrix2[i];
            gemm_id++;
        }
    }

    mcMemcpyAsync(dev_A1, A_group_gemm, sizeof(W *) * gemmCount,
                  mcMemcpyHostToDevice);
    mcMemcpyAsync(dev_A2, A_group_gemm2, sizeof(W *) * gemmCount,
                  mcMemcpyHostToDevice);
    mcMemcpyAsync(dev_B1, B_group_gemm, sizeof(A *) * gemmCount,
                  mcMemcpyHostToDevice);
    mcMemcpyAsync(dev_C1, C_group_gemm, sizeof(A *) * gemmCount,
                  mcMemcpyHostToDevice);

    // TODO: copy samples to expert's buffers and layout
    copyAndLayoutB<A>(activations, dev_B_expert_buffers, dev_samplesForExperts,
                      batchSize, k1, expertCount, gemmCount);

    mcMemsetD32Async(activations, 0, sizeof(A) * n1 * k1 / 4, streamId);
    const mcStream_t stream =
        at::cuda::getCurrentCUDAStream(at::cuda::current_device());

    // launch w1 group gemm
    if (batchSize <= 256) {
        // use group gemm
        group_gemm_launcher<A, Taccum, A>(dev_A1, dev_B1, dev_C1, m1, n1, k1,
                                          1.0, 0.0, gemmCount);
    } else {
        // use layout muxi hgemm
        for (int i = 0; i < gemmCount; i++) {
            dim3 grid(m1 / 128, (n1 + 127) / 128, 1);
            muxi_layout_kernels::
                layout_hgemm_tn_128x128x128_4m1n8k_256t_layoutC<A, A, Taccum,
                                                                true, false>
                <<<grid, 256, 0, stream>>>(A_group_gemm[i], B_group_gemm[i],
                                           C_group_gemm[i], m1, n1, k1, k1, k1,
                                           m1, 1.0f, 0.0f);
        }
    }

    // silu and mul
    silu_and_mul<A><<<gemmCount, 256, 0, stream>>>(dev_C1, n1, m1, gemmCount);

    // launch w2 group gemm
    if (batchSize <= 256) {
        // use group gemm
        group_gemm_launcher2<A, Taccum, A>(dev_A2, dev_C1, activations, m2, n2,
                                           k2, dev_scoreWeightsForGEMMs, 0.0,
                                           gemmCount);
    } else {
        // use layout muxi hgemm
        for (int i = 0; i < gemmCount; i++) {
            dim3 grid(m2 / 128, (n2 + 127) / 128, 1);
            muxi_layout_kernels_fused::
                layout_hgemm_tn_128x128x128_4m1n8k_256t_fused<A, A, Taccum,
                                                              false, false>
                <<<grid, 256, 0, stream>>>(
                    A_group_gemm2[i], C_group_gemm[i], activations, m2, n2, k2,
                    k2, k2, m2, dev_scoreWeightsForGEMMs + i * batchSize, 1.0f);
        }
    }

    mcDeviceSynchronize();

    // TODO: free buffers
    mcFree(dev_samplesForExperts);
    mcFree(dev_scoreWeightsForGEMMs);
    mcFree(dev_B_expert_buffers);
    mcFree(dev_gemm_result_buffer);
    mcFree(B_group_gemm2_dev);
    mcFree(dev_A1);
    mcFree(dev_A2);
    mcFree(dev_B1);
    mcFree(dev_C1);
    free(samplesForExperts);
    free(A_group_gemm);
    free(B_group_gemm);
    free(C_group_gemm);
    free(A_group_gemm2);
    free(scoreWeightsForGEMMs);
}