#include "experts_compute.h"

namespace muxi_layout_kernels {

void experts_compute(torch::Tensor &experts_weights_matrix1,
                     torch::Tensor &experts_weights_matrix2,
                     torch::Tensor &activations, int64_t batchSize,
                     int64_t expertCount, int64_t dynamicExpertsPerAct,
                     torch::Tensor &expertsIds,
                     torch::Tensor &activedExpertsWeights,
                     c10::optional<torch::Tensor> bias) {
    TORCH_CHECK(experts_weights_matrix1.dtype() ==
                    experts_weights_matrix2.dtype(),
                "experts_weights_matrix1 and experts_weights_matrix2 must have "
                "the same "
                "data type.");

    TORCH_CHECK(experts_weights_matrix1.dtype() == torch::kFloat16 ||
                    experts_weights_matrix1.dtype() == torch::kBFloat16,
                "experts_weights_matrix1 must be of type torch::kFloat16 or "
                "torch::kBFloat16.");

    TORCH_CHECK(expertsIds.dtype() == torch::kInt32,
                "expertsIds must be of type torch::kInt32.");

    TORCH_CHECK(activations.dtype() == activedExpertsWeights.dtype(),
                "activations and activedExpertsWeights must have the same "
                "data type.");

    TORCH_CHECK(
        activations.dtype() == torch::kFloat16 ||
            activations.dtype() == torch::kBFloat16,
        "activations must be of type torch::kFloat16 or torch::kBFloat16.");

    if (bias.has_value()) {
        TORCH_CHECK(bias->dtype() == activations.dtype(),
                    "bias must have the same data type as activations.");
    }

    TORCH_CHECK(
        expertsIds.device() == activedExpertsWeights.device(),
        "expertsIds must be on the same device as activedExpertsWeights.");

    // int n1 = experts_weights_matrix1.size(0);
    int n1 = batchSize;
    int m1 = experts_weights_matrix1.size(1);
    int k1 = experts_weights_matrix1.size(2);

    // int n2 = experts_weights_matrix2.size(0);
    int n2 = batchSize;
    int m2 = experts_weights_matrix2.size(1);
    int k2 = experts_weights_matrix2.size(2);

    auto experts_ids_ptr = reinterpret_cast<int *>(expertsIds.data_ptr());
    auto weightDtype = experts_weights_matrix1.dtype();
    auto ActDtype = activations.dtype();

    if (weightDtype == torch::kFloat16) {
        std::vector<half *> weights1;
        std::vector<half *> weights2;

        if (!experts_weights_matrix1.is_contiguous()) {
            for (int i = 0; i < expertCount; ++i) {
                weights1.push_back(reinterpret_cast<half *>(
                    experts_weights_matrix1[i].data_ptr()));
            }
        } else {
            half *w1_ptr =
                reinterpret_cast<half *>(experts_weights_matrix1.data_ptr());
            for (int i = 0; i < expertCount; i++) {
                weights1.push_back(w1_ptr + i * m1 * k1);
            }
        }
        if (!experts_weights_matrix2.is_contiguous()) {
            for (int i = 0; i < expertCount; ++i) {
                weights2.push_back(reinterpret_cast<half *>(
                    experts_weights_matrix2[i].data_ptr()));
            }
        } else {
            half *w2_ptr =
                reinterpret_cast<half *>(experts_weights_matrix2.data_ptr());
            for (int i = 0; i < expertCount; i++) {
                weights2.push_back(w2_ptr + i * m2 * k2);
            }
        }

        if (ActDtype == torch::kFloat16) {
            auto activations_ptr =
                reinterpret_cast<half *>(activations.data_ptr());
            auto activedExpertsWeights_ptr =
                reinterpret_cast<half *>(activedExpertsWeights.data_ptr());
            auto bias_ptr = bias.has_value()
                                ? reinterpret_cast<half **>(bias->data_ptr())
                                : nullptr;
            experts_compute_inner<half, float, half>(
                weights1.data(), weights2.data(), activations_ptr, m1, n1, k1,
                m2, n2, k2, static_cast<int>(batchSize),
                static_cast<int>(expertCount),
                static_cast<int>(dynamicExpertsPerAct), experts_ids_ptr,
                activedExpertsWeights_ptr, bias_ptr);
        } else {
            // auto activations_ptr =
            //     reinterpret_cast<__maca_bfloat16 *>(activations.data_ptr());
            // auto activedExpertsWeights_ptr =
            //     reinterpret_cast<__maca_bfloat16 *>(
            //         activedExpertsWeights.data_ptr());
            // auto bias_ptr =
            //     bias.has_value()
            //         ? reinterpret_cast<__maca_bfloat16 **>(bias->data_ptr())
            //         : nullptr;
            // experts_compute_inner<half, float, __maca_bfloat16>(
            //     weights1.data(), weights2.data(), activations_ptr, m1, n1,
            //     k1, m2, n2, k2, static_cast<int>(batchSize),
            //     static_cast<int>(expertCount),
            //     static_cast<int>(dynamicExpertsPerAct), experts_ids_ptr,
            //     activedExpertsWeights_ptr, bias_ptr);
        }
    } else {
        std::vector<__maca_bfloat16 *> weights1;
        std::vector<__maca_bfloat16 *> weights2;
        for (int i = 0; i < expertCount; ++i) {
            weights1.push_back(reinterpret_cast<__maca_bfloat16 *>(
                experts_weights_matrix1[i].data_ptr()));
        }
        for (int i = 0; i < expertCount; ++i) {
            weights2.push_back(reinterpret_cast<__maca_bfloat16 *>(
                experts_weights_matrix2[i].data_ptr()));
        }

        if (ActDtype == torch::kFloat16) {
            // auto activations_ptr =
            //     reinterpret_cast<half *>(activations.data_ptr());
            // auto activedExpertsWeights_ptr =
            //     reinterpret_cast<half *>(activedExpertsWeights.data_ptr());
            // auto bias_ptr = bias.has_value()
            //                     ? reinterpret_cast<half **>(bias->data_ptr())
            //                     : nullptr;
            // experts_compute_inner<__maca_bfloat16, float, half>(
            //     weights1.data(), weights2.data(), activations_ptr, m1, n1,
            //     k1, m2, n2, k2, static_cast<int>(batchSize),
            //     static_cast<int>(expertCount),
            //     static_cast<int>(dynamicExpertsPerAct), experts_ids_ptr,
            //     activedExpertsWeights_ptr, bias_ptr);
        } else {
            auto activations_ptr =
                reinterpret_cast<__maca_bfloat16 *>(activations.data_ptr());
            auto activedExpertsWeights_ptr =
                reinterpret_cast<__maca_bfloat16 *>(
                    activedExpertsWeights.data_ptr());
            auto bias_ptr =
                bias.has_value()
                    ? reinterpret_cast<__maca_bfloat16 **>(bias->data_ptr())
                    : nullptr;
            experts_compute_inner<__maca_bfloat16, float, __maca_bfloat16>(
                weights1.data(), weights2.data(), activations_ptr, m1, n1, k1,
                m2, n2, k2, static_cast<int>(batchSize),
                static_cast<int>(expertCount),
                static_cast<int>(dynamicExpertsPerAct), experts_ids_ptr,
                activedExpertsWeights_ptr, bias_ptr);
        }
    }
}

} // namespace muxi_layout_kernels
