#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>
#include <torch/all.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "dispatch_utils.h"
#include "fp8_weight_repack.h"
#include "gemm_ex.h"
#include "gemm_layout_A.h"
#include "gemm_layout_ab_continuous_c.h"
#include "gemm_layout_abc.h"
#include "gemv_layout_a.h"
#include "layout_b.h"
#include "layout_gemv_kernel.cuh"
#include "muxi_hgemm.h"
#include "muxi_hgemm_layout.h"
#include "muxi_hgemm_layoutA.h"
#include "muxi_hgemm_layoutC.h"
#include "ops.h"
#include "re_layout_c.h"
#include "utils.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using namespace muxi_layout_kernels;
    using namespace pybind11::literals;

    m.def("layoutB", &layoutB, "Layout B kernel for GEMM_LayoutABC");
    m.def("reLayoutC", &reLayoutC, "ReLayoutC after GEMM_LayoutABC");
    m.def("gemm_layoutABC", &gemm_layoutABC, "A"_a, "B"_a, "alpha"_a = 1,
          "beta"_a = 0, "bias"_a = std::nullopt, "GEMM LayoutABC kernel");
    m.def("gemm_layoutA_linear", &gemm_layoutA_linear, "A"_a, "B"_a,
          "alpha"_a = 1, "beta"_a = 0, "scale_matrix"_a = std::nullopt,
          "bias"_a = std::nullopt);
    m.def("gemm_layoutA", &gemm_layoutA, "A"_a, "B"_a, "alpha"_a = 1,
          "beta"_a = 0, "bias"_a = std::nullopt,
          "GEMM LayoutA and Continuous B C kernel");
    m.def("gemm_layoutA_soft_fp8", &gemm_layoutA_soft_fp8, "A"_a, "A_Scale"_a,
          "B"_a, "alpha"_a = 1, "beta"_a = 0, "bias"_a = std::nullopt,
          "GEMM LayoutA and Continuous B C kernel");
    m.def("gemm_layoutAB_ContinuousC", &gemm_layoutAB_ContinuousC, "A"_a, "B"_a,
          "alpha"_a = 1, "beta"_a = 0, "bias"_a = std::nullopt,
          "Gemm LayoutAB ContinuousC Kernel");
    m.def("gemmEx", &gemmEx, "A"_a, "B"_a, "alpha"_a = 1, "beta"_a = 0,
          "McblasGemmEx Interface kernel");
    m.def("muxi_hgemm", &muxi_hgemm, "A"_a, "B"_a, "alpha"_a = 1, "beta"_a = 0,
          "muxi_hgemm kernel");
    m.def("muxi_hgemm_layout", &muxi_hgemm_layout, "A"_a, "B"_a, "alpha"_a = 1,
          "beta"_a = 0, "bias"_a = std::nullopt,
          "muxi_hgemm LayoutAB ContinuousC kernel");
    m.def("muxi_hgemm_layoutA", &muxi_hgemm_layoutA, "A"_a, "B"_a,
          "alpha"_a = 1, "beta"_a = 0, "scale_matrix"_a = std::nullopt,
          "bias"_a = std::nullopt, "muxi_hgemm LayoutAB ContinuousC kernel");
    m.def("muxi_hgemm_layoutC", &muxi_hgemm_layoutC, "A"_a, "B"_a,
          "alpha"_a = 1, "beta"_a = 0, "bias"_a = std::nullopt,
          "muxi_hgemm LayoutABC kernel");
    m.def("gemv_layoutA", &gemv_layoutA, "A"_a, "B"_a, "alpha"_a = 1,
          "beta"_a = 0, "scale_matrix"_a = std::nullopt,
          "bias"_a = std::nullopt, "GEMV LayoutA kernel");
    m.def("gemm_layoutABC_wapper", &gemm_layoutABC_wapper, "A"_a, "B"_a, "m"_a,
          "n"_a, "k"_a, "alpha"_a = 1, "beta"_a = 0, "kernelParam1"_a = 1,
          "kernelParam2"_a = 1, "kernelParam3"_a = 1, "kernelId"_a = 1,
          "bias"_a = std::nullopt, "GEMM wapper for scan gemm kernel args");
    m.def("gemm_layoutA_wapper", &gemm_layoutA_wapper, "A"_a, "B"_a, "m"_a,
          "n"_a, "k"_a, "alpha"_a = 1, "beta"_a = 0, "kernelParam1"_a = 1,
          "kernelParam2"_a = 1, "kernelParam3"_a = 1, "kernelId"_a = 1,
          "bias"_a = std::nullopt,
          "GEMM wapper for scan layoutabc gemm kernel args");
    m.def("gemm_layoutA_soft_fp8_wapper", &gemm_layoutA_soft_fp8_wapper, "A"_a,
          "A_Scale"_a, "B"_a, "m"_a, "n"_a, "k"_a, "alpha"_a = 1, "beta"_a = 0,
          "kernelParam1"_a = 1, "kernelParam2"_a = 1, "kernelParam3"_a = 1,
          "kernelId"_a = 1, "bias"_a = std::nullopt,
          "GEMM wapper for scan layoutabc gemm kernel args");
    m.def("gemm_layoutAB_ContinuousC_wapper", &gemm_layoutAB_ContinuousC_wapper,
          "A"_a, "B"_a, "m"_a, "n"_a, "k"_a, "alpha"_a = 1, "beta"_a = 0,
          "kernelParam1"_a = 1, "kernelParam2"_a = 1, "kernelParam3"_a = 1,
          "kernelId"_a = 1, "bias"_a = std::nullopt,
          "GEMM wapper for scan gemm ContinuousC kernel args");
    m.def("gemv_layoutA_wapper", &gemv_layoutA_wapper, "A"_a, "B"_a, "m"_a,
          "k"_a, "alpha"_a = 1, "beta"_a = 0, "blockDimX"_a = 256,
          "kernelId"_a = 1, "kernelParam1"_a = 1, "kernelParam2"_a = 1,
          "scale_matrix"_a = std::nullopt, "bias"_a = std::nullopt,
          "GEMV wapper for scan gemv kernel args");
    m.def("fp8_weight_repack", &fp8_weight_repack, "weight"_a,
          "repack fp8_weight for GEMM");
    m.def("routing_gate", [](torch::Tensor &gating_output, int64_t score_fun,
                             int64_t batch_size, int64_t hidden_size,
                             int64_t n_groups, int64_t topK_groups,
                             torch::Tensor &experts_ids,
                             torch::Tensor &selected_experts_weights,
                             int64_t topK,
                             c10::optional<torch::Tensor> load_balance_bias =
                                 c10::nullopt) {
        int num_experts = gating_output.size(-1);
        auto dtype = gating_output.dtype();

        TORCH_CHECK(
            gating_output.dtype() == selected_experts_weights.dtype(),
            "Mismatched dtypes: gating_output and selected_experts_weights.");

        if (load_balance_bias.has_value()) {
            TORCH_CHECK(
                load_balance_bias->dtype() == gating_output.dtype(),
                "Mismatched dtypes: load_balance_bias and gating_output.");
        }

        TORCH_CHECK(
            gating_output.dtype() == torch::kFloat16 ||
                gating_output.dtype() == torch::kBFloat16,
            "Invalid dtype for gating_output: must be kFloat16 or kBFloat16.");

        TORCH_CHECK(experts_ids.dtype() == torch::kInt32,
                    "Invalid dtype for experts_ids: must be kInt32.");

        TORCH_CHECK(
            experts_ids.device() == selected_experts_weights.device(),
            "Device mismatch: experts_ids and selected_experts_weights.");

        auto experts_ids_ptr = reinterpret_cast<int *>(experts_ids.data_ptr());

        if (dtype == torch::kFloat16) {
            auto gating_output_ptr =
                reinterpret_cast<half *>(gating_output.data_ptr());
            auto selected_experts_weights_ptr =
                reinterpret_cast<half *>(selected_experts_weights.data_ptr());
            auto load_balance_bias_ptr =
                load_balance_bias.has_value()
                    ? reinterpret_cast<half *>(load_balance_bias->data_ptr())
                    : nullptr;
            routing_gate<float, half>(
                gating_output_ptr, static_cast<int>(score_fun), num_experts,
                static_cast<int>(batch_size), static_cast<int>(topK),
                static_cast<int>(n_groups), static_cast<int>(topK_groups),
                static_cast<int>(hidden_size), experts_ids_ptr,
                selected_experts_weights_ptr, load_balance_bias_ptr);
        } else {
            auto gating_output_ptr =
                reinterpret_cast<__maca_bfloat16 *>(gating_output.data_ptr());
            auto selected_experts_weights_ptr =
                reinterpret_cast<__maca_bfloat16 *>(
                    selected_experts_weights.data_ptr());
            auto load_balance_bias_ptr =
                load_balance_bias.has_value()
                    ? reinterpret_cast<__maca_bfloat16 *>(
                          load_balance_bias->data_ptr())
                    : nullptr;
            routing_gate<float, __maca_bfloat16>(
                gating_output_ptr, static_cast<int>(score_fun), num_experts,
                static_cast<int>(batch_size), static_cast<int>(topK),
                static_cast<int>(n_groups), static_cast<int>(topK_groups),
                static_cast<int>(hidden_size), experts_ids_ptr,
                selected_experts_weights_ptr, load_balance_bias_ptr);
        }
    });

    m.def("experts_compute", [](torch::Tensor &experts_weights_matrix1,
                                torch::Tensor &experts_weights_matrix2,
                                torch::Tensor &activations, int64_t batchSize,
                                int64_t expertCount,
                                int64_t dynamicExpertsPerAct,
                                torch::Tensor &expertsIds,
                                torch::Tensor &activedExpertsWeights,
                                c10::optional<torch::Tensor> bias =
                                    c10::nullopt) {
        TORCH_CHECK(
            experts_weights_matrix1.dtype() == experts_weights_matrix2.dtype(),
            "experts_weights_matrix1 and experts_weights_matrix2 must have "
            "the same "
            "data type.");

        TORCH_CHECK(
            experts_weights_matrix1.dtype() == torch::kFloat16 ||
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
                half *w1_ptr = reinterpret_cast<half *>(
                    experts_weights_matrix1.data_ptr());
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
                half *w2_ptr = reinterpret_cast<half *>(
                    experts_weights_matrix2.data_ptr());
                for (int i = 0; i < expertCount; i++) {
                    weights2.push_back(w2_ptr + i * m2 * k2);
                }
            }

            if (ActDtype == torch::kFloat16) {
                auto activations_ptr =
                    reinterpret_cast<half *>(activations.data_ptr());
                auto activedExpertsWeights_ptr =
                    reinterpret_cast<half *>(activedExpertsWeights.data_ptr());
                auto bias_ptr =
                    bias.has_value()
                        ? reinterpret_cast<half **>(bias->data_ptr())
                        : nullptr;
                experts_compute<half, float, half>(
                    weights1.data(), weights2.data(), activations_ptr, m1, n1,
                    k1, m2, n2, k2, static_cast<int>(batchSize),
                    static_cast<int>(expertCount),
                    static_cast<int>(dynamicExpertsPerAct), experts_ids_ptr,
                    activedExpertsWeights_ptr, bias_ptr);
            } else {
                // auto activations_ptr =
                //     reinterpret_cast<__maca_bfloat16
                //     *>(activations.data_ptr());
                // auto activedExpertsWeights_ptr =
                //     reinterpret_cast<__maca_bfloat16
                //     *>(activedExpertsWeights.data_ptr());
                // auto bias_ptr =
                //     bias.has_value()
                //         ? reinterpret_cast<__maca_bfloat16
                //         **>(bias->data_ptr()) : nullptr;
                // experts_compute<half, float, __maca_bfloat16>(
                //     weights1.data(), weights2.data(), activations_ptr, m1,
                //     n1, k1, m2, n2, k2, static_cast<int>(batchSize),
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
                // auto activations_ptr = reinterpret_cast<half
                // *>(activations.data_ptr()); auto activedExpertsWeights_ptr =
                //     reinterpret_cast<half
                //     *>(activedExpertsWeights.data_ptr());
                // auto bias_ptr = bias.has_value()
                //                     ? reinterpret_cast<half
                //                     **>(bias->data_ptr()) : nullptr;
                // experts_compute<__maca_bfloat16, float, half>(
                //     weights1.data(), weights2.data(), activations_ptr, m1,
                //     n1, k1, m2, n2, k2, static_cast<int>(batchSize),
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
                experts_compute<__maca_bfloat16, float, __maca_bfloat16>(
                    weights1.data(), weights2.data(), activations_ptr, m1, n1,
                    k1, m2, n2, k2, static_cast<int>(batchSize),
                    static_cast<int>(expertCount),
                    static_cast<int>(dynamicExpertsPerAct), experts_ids_ptr,
                    activedExpertsWeights_ptr, bias_ptr);
            }
        }
    });

    m.def("fused_routing_gate", [](torch::Tensor &gating_output,
                                   int64_t score_fun, int64_t batch_size,
                                   int64_t hidden_size, int64_t n_groups,
                                   int64_t topK_groups,
                                   torch::Tensor &experts_ids,
                                   torch::Tensor &selected_experts_weights,
                                   int64_t topK,
                                   c10::optional<torch::Tensor>
                                       load_balance_bias = c10::nullopt) {
        int num_experts = gating_output.size(-1);
        auto dtype = gating_output.dtype();

        TORCH_CHECK(
            gating_output.dtype() == selected_experts_weights.dtype(),
            "Mismatched dtypes: gating_output and selected_experts_weights.");

        if (load_balance_bias.has_value()) {
            TORCH_CHECK(
                load_balance_bias->dtype() == gating_output.dtype(),
                "Mismatched dtypes: load_balance_bias and gating_output.");
        }

        TORCH_CHECK(
            gating_output.dtype() == torch::kFloat16 ||
                gating_output.dtype() == torch::kBFloat16,
            "Invalid dtype for gating_output: must be kFloat16 or kBFloat16.");

        TORCH_CHECK(experts_ids.dtype() == torch::kInt32,
                    "Invalid dtype for experts_ids: must be kInt32.");

        TORCH_CHECK(
            experts_ids.device() == selected_experts_weights.device(),
            "Device mismatch: experts_ids and selected_experts_weights.");

        auto experts_ids_ptr = reinterpret_cast<int *>(experts_ids.data_ptr());

        if (dtype == torch::kFloat16) {
            auto gating_output_ptr =
                reinterpret_cast<half *>(gating_output.data_ptr());
            auto selected_experts_weights_ptr =
                reinterpret_cast<half *>(selected_experts_weights.data_ptr());
            auto load_balance_bias_ptr =
                load_balance_bias.has_value()
                    ? reinterpret_cast<half *>(load_balance_bias->data_ptr())
                    : nullptr;
            fused_routing_gate<float, half>(
                gating_output_ptr, static_cast<int>(score_fun), num_experts,
                static_cast<int>(batch_size), static_cast<int>(topK),
                static_cast<int>(n_groups), static_cast<int>(topK_groups),
                static_cast<int>(hidden_size), experts_ids_ptr,
                selected_experts_weights_ptr, load_balance_bias_ptr);
        } else {
            auto gating_output_ptr =
                reinterpret_cast<__maca_bfloat16 *>(gating_output.data_ptr());
            auto selected_experts_weights_ptr =
                reinterpret_cast<__maca_bfloat16 *>(
                    selected_experts_weights.data_ptr());
            auto load_balance_bias_ptr =
                load_balance_bias.has_value()
                    ? reinterpret_cast<__maca_bfloat16 *>(
                          load_balance_bias->data_ptr())
                    : nullptr;
            fused_routing_gate<float, __maca_bfloat16>(
                gating_output_ptr, static_cast<int>(score_fun), num_experts,
                static_cast<int>(batch_size), static_cast<int>(topK),
                static_cast<int>(n_groups), static_cast<int>(topK_groups),
                static_cast<int>(hidden_size), experts_ids_ptr,
                selected_experts_weights_ptr, load_balance_bias_ptr);
        }
    });

    m.def("fused_experts_compute", [](torch::Tensor &experts_weights_matrix1,
                                      torch::Tensor &experts_weights_matrix2,
                                      torch::Tensor &activations,
                                      int64_t batchSize, int64_t expertCount,
                                      int64_t dynamicExpertsPerAct,
                                      torch::Tensor &expertsIds,
                                      torch::Tensor &activedExpertsWeights,
                                      torch::Tensor &dev_sorted_token_ids,
                                      torch::Tensor &dev_cumsum_buffer,
                                      torch::Tensor &dev_padded_num_experts,
                                      torch::Tensor &dev_experts_ids,
                                      torch::Tensor &dev_C, torch::Tensor &y) {
        TORCH_CHECK(experts_weights_matrix1.dtype() ==
                        experts_weights_matrix2.dtype(),
                    "experts_weights_matrix1 and experts_weights_matrix2 "
                    "must have the same data type.");

        TORCH_CHECK(experts_weights_matrix1.dtype() == torch::kFloat16 ||
                        experts_weights_matrix1.dtype() == torch::kBFloat16,
                    "experts_weights_matrix1 must be of type torch::kFloat16 "
                    "or torch::kBFloat16.");

        TORCH_CHECK(expertsIds.dtype() == torch::kInt32,
                    "expertsIds must be of type torch::kInt32.");

        TORCH_CHECK(activations.dtype() == activedExpertsWeights.dtype(),
                    "activations and activedExpertsWeights must have the same "
                    "data type.");

        TORCH_CHECK(
            activations.dtype() == torch::kFloat16 ||
                activations.dtype() == torch::kBFloat16,
            "activations must be of type torch::kFloat16 or torch::kBFloat16.");

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
        auto dev_sorted_token_ids_ptr =
            reinterpret_cast<int *>(dev_sorted_token_ids.data_ptr());
        auto dev_cumsum_buffer_ptr =
            reinterpret_cast<int *>(dev_cumsum_buffer.data_ptr());
        auto dev_padded_num_experts_ptr =
            reinterpret_cast<int *>(dev_padded_num_experts.data_ptr());
        auto dev_experts_ids_ptr =
            reinterpret_cast<int *>(dev_experts_ids.data_ptr());

        auto weightDtype = experts_weights_matrix1.dtype();
        auto ActDtype = activations.dtype();

        if (weightDtype == torch::kFloat16) {
            half *w1_ptr =
                reinterpret_cast<half *>(experts_weights_matrix1.data_ptr());
            half *w2_ptr =
                reinterpret_cast<half *>(experts_weights_matrix2.data_ptr());

            if (ActDtype == torch::kFloat16) {
                auto activations_ptr =
                    reinterpret_cast<half *>(activations.data_ptr());
                auto activedExpertsWeights_ptr =
                    reinterpret_cast<half *>(activedExpertsWeights.data_ptr());
                dispatchToStaticInts<8, 16, 32, 64, 128, 256>(
                    expertCount, [&]<int expertCount>() {
                        fused_experts_compute<expertCount, half, float, half>(
                            w1_ptr, w2_ptr, activations_ptr, m1, n1, k1, m2, n2,
                            k2, static_cast<int>(batchSize),
                            static_cast<int>(dynamicExpertsPerAct),
                            experts_ids_ptr, activedExpertsWeights_ptr,
                            dev_sorted_token_ids_ptr, dev_cumsum_buffer_ptr,
                            dev_padded_num_experts_ptr, dev_experts_ids_ptr,
                            reinterpret_cast<half *>(dev_C.data_ptr()),
                            reinterpret_cast<half *>(y.data_ptr()));
                    });
            } else {
                TORCH_CHECK(
                    false,
                    "Weight date type and activation date type should be the "
                    "same now.");
            }
        } else {
            __maca_bfloat16 *w1_ptr = reinterpret_cast<__maca_bfloat16 *>(
                experts_weights_matrix1.data_ptr());
            __maca_bfloat16 *w2_ptr = reinterpret_cast<__maca_bfloat16 *>(
                experts_weights_matrix2.data_ptr());

            if (ActDtype == torch::kFloat16) {
                TORCH_CHECK(
                    false,
                    "Weight date type and activation date type should be the "
                    "same now.");
            } else {
                auto activations_ptr =
                    reinterpret_cast<__maca_bfloat16 *>(activations.data_ptr());
                auto activedExpertsWeights_ptr =
                    reinterpret_cast<__maca_bfloat16 *>(
                        activedExpertsWeights.data_ptr());
                dispatchToStaticInts<8, 16, 32, 64, 128, 256>(
                    expertCount, [&]<int expertCount>() {
                        fused_experts_compute<expertCount, __maca_bfloat16,
                                              float, __maca_bfloat16>(
                            w1_ptr, w2_ptr, activations_ptr, m1, n1, k1, m2, n2,
                            k2, static_cast<int>(batchSize),
                            static_cast<int>(dynamicExpertsPerAct),
                            experts_ids_ptr, activedExpertsWeights_ptr,
                            dev_sorted_token_ids_ptr, dev_cumsum_buffer_ptr,
                            dev_padded_num_experts_ptr, dev_experts_ids_ptr,
                            reinterpret_cast<__maca_bfloat16 *>(
                                dev_C.data_ptr()),
                            reinterpret_cast<__maca_bfloat16 *>(y.data_ptr()));
                    });
            }
        }
    });

    m.def("fused_experts_compute", [](torch::Tensor &experts_weights_matrix1,
                                      torch::Tensor &experts_weights_matrix2,
                                      torch::Tensor &activations,
                                      int64_t batchSize, int64_t expertCount,
                                      int64_t dynamicExpertsPerAct,
                                      torch::Tensor &expertsIds,
                                      torch::Tensor &activedExpertsWeights,
                                      torch::Tensor &dev_sorted_token_ids,
                                      torch::Tensor &dev_cumsum_buffer,
                                      torch::Tensor &dev_padded_num_experts,
                                      torch::Tensor &dev_experts_ids,
                                      torch::Tensor &dev_C, torch::Tensor &y,
                                      torch::Tensor &w1_scale,
                                      torch::Tensor &w2_scale,
                                      std::vector<int64_t> &block_shape,
                                      bool soft_fp8 = false) {
        TORCH_CHECK(experts_weights_matrix1.dtype() ==
                        experts_weights_matrix2.dtype(),
                    "experts_weights_matrix1 and experts_weights_matrix2 "
                    "must have the same data type.");

        TORCH_CHECK(experts_weights_matrix1.element_size() == 1,
                    "The element size of experts_weights_matrix1 must be 1 "
                    "byte, but got ",
                    experts_weights_matrix1.element_size(), " bytes.");

        TORCH_CHECK(expertsIds.dtype() == torch::kInt32,
                    "expertsIds must be of type torch::kInt32.");

        TORCH_CHECK(activations.dtype() == activedExpertsWeights.dtype(),
                    "activations and activedExpertsWeights must have the same "
                    "data type.");

        TORCH_CHECK(
            activations.dtype() == torch::kFloat16 ||
                activations.dtype() == torch::kBFloat16,
            "activations must be of type torch::kFloat16 or torch::kBFloat16.");

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
        auto dev_sorted_token_ids_ptr =
            reinterpret_cast<int *>(dev_sorted_token_ids.data_ptr());
        auto dev_cumsum_buffer_ptr =
            reinterpret_cast<int *>(dev_cumsum_buffer.data_ptr());
        auto dev_padded_num_experts_ptr =
            reinterpret_cast<int *>(dev_padded_num_experts.data_ptr());
        auto dev_experts_ids_ptr =
            reinterpret_cast<int *>(dev_experts_ids.data_ptr());

        auto ActDtype = activations.dtype();

        uint8_t *w1_ptr =
            reinterpret_cast<uint8_t *>(experts_weights_matrix1.data_ptr());
        uint8_t *w2_ptr =
            reinterpret_cast<uint8_t *>(experts_weights_matrix2.data_ptr());
        TORCH_CHECK(w1_scale.dtype() == torch::kFloat32 &&
                        w2_scale.dtype() == torch::kFloat32,
                    "w1_scale and w2_scale must be of type torch::kFloat32.");
        TORCH_CHECK(w1_scale.device() == w2_scale.device(),
                    "w1_scale and w2_scale must be on the same device.");
        TORCH_CHECK(block_shape.size() == 2, "block_shape must have size 2.");
        TORCH_CHECK(block_shape.at(0) == 128 && block_shape.at(1) == 128,
                    "block_shape must be (128, 128) for float8_e4m3fn.");
        TORCH_CHECK(soft_fp8 == true,
                    "soft_fp8 must be true for float8_e4m3fn.");

        if (ActDtype == torch::kBFloat16) {
            auto activations_ptr =
                reinterpret_cast<__maca_bfloat16 *>(activations.data_ptr());
            auto activedExpertsWeights_ptr =
                reinterpret_cast<__maca_bfloat16 *>(
                    activedExpertsWeights.data_ptr());
            dispatchToStaticInts<8, 16, 32, 64, 128, 256>(
                expertCount, [&]<int expertCount>() {
                    fused_experts_compute<expertCount, uint8_t, float,
                                          __maca_bfloat16>(
                        w1_ptr, w2_ptr, activations_ptr, m1, n1, k1, m2, n2, k2,
                        static_cast<int>(batchSize),
                        static_cast<int>(dynamicExpertsPerAct), experts_ids_ptr,
                        activedExpertsWeights_ptr, dev_sorted_token_ids_ptr,
                        dev_cumsum_buffer_ptr, dev_padded_num_experts_ptr,
                        dev_experts_ids_ptr,
                        reinterpret_cast<__maca_bfloat16 *>(dev_C.data_ptr()),
                        reinterpret_cast<__maca_bfloat16 *>(y.data_ptr()),
                        reinterpret_cast<float *>(w1_scale.data_ptr()),
                        reinterpret_cast<float *>(w2_scale.data_ptr()),
                        w1_scale.size(1), w1_scale.size(2), w2_scale.size(1),
                        w2_scale.size(2));
                });

        } else {
            TORCH_CHECK(false, "Weight date type and activation date type "
                               "should be bfloat16 ");
        }
    });
}
