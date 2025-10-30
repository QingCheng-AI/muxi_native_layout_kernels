#include "fused_topK.h"
#include "group_gemm_utils.h"

int main() {
    // Test muxi_layout_kernels::fused_softmax_topk_kernel

    using TA = __half;
    TA *input = (TA *)malloc(256 * 16 * sizeof(TA));
    initializeHostData(input, 256, 16);
    TA *selectedExpertsWeights = (TA *)malloc(8 * 16 * sizeof(TA));
    int *expertsIds = (int *)malloc(8 * 16 * sizeof(int));

    TA *dev_input;
    mcMalloc((void **)&dev_input, sizeof(TA) * 256 * 16);
    mcMemcpy(dev_input, input, sizeof(TA) * 256 * 16, mcMemcpyHostToDevice);
    TA *dev_selectedExpertsWeights;
    mcMalloc((void **)&dev_selectedExpertsWeights, sizeof(TA) * 8 * 16);
    int *dev_expertsIds;
    mcMalloc((void **)&dev_expertsIds, sizeof(int) * 8 * 16);

    muxi_layout_kernels::fused_softmax_topk_launcher<TA>(
        dev_input, 16, 8, 4, dev_expertsIds, dev_selectedExpertsWeights, 8,
        256);

    mcMemcpy(selectedExpertsWeights, dev_selectedExpertsWeights,
             sizeof(TA) * 8 * 16, mcMemcpyDeviceToHost);
    mcMemcpy(expertsIds, dev_expertsIds, sizeof(int) * 8 * 16,
             mcMemcpyDeviceToHost);

    std::cout << "Selected Experts Weights:" << std::endl;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            std::cout << static_cast<float>(selectedExpertsWeights[i * 8 + j])
                      << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Experts Ids:" << std::endl;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            std::cout << static_cast<float>(expertsIds[i * 8 + j]) << " ";
        }
        std::cout << std::endl;
    }
}
