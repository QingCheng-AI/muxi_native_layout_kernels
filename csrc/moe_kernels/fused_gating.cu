#include "fused_gating.h"

namespace muxi_layout_kernels {

void fused_routing_gate(torch::Tensor &gating_output, int64_t score_fun,
                        int64_t batch_size, int64_t hidden_size,
                        int64_t n_groups, int64_t topK_groups,
                        torch::Tensor &experts_ids,
                        torch::Tensor &selected_experts_weights, int64_t topK,
                        c10::optional<torch::Tensor> load_balance_bias) {
    int num_experts = gating_output.size(-1);
    auto dtype = gating_output.dtype();

    TORCH_CHECK(
        gating_output.dtype() == selected_experts_weights.dtype(),
        "Mismatched dtypes: gating_output and selected_experts_weights.");

    if (load_balance_bias.has_value()) {
        TORCH_CHECK(load_balance_bias->dtype() == gating_output.dtype(),
                    "Mismatched dtypes: load_balance_bias and gating_output.");
    }

    TORCH_CHECK(
        gating_output.dtype() == torch::kFloat16 ||
            gating_output.dtype() == torch::kBFloat16,
        "Invalid dtype for gating_output: must be kFloat16 or kBFloat16.");

    TORCH_CHECK(experts_ids.dtype() == torch::kInt32,
                "Invalid dtype for experts_ids: must be kInt32.");

    TORCH_CHECK(experts_ids.device() == selected_experts_weights.device(),
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
        fused_routing_gate_inner<float, half>(
            gating_output_ptr, static_cast<int>(score_fun), num_experts,
            static_cast<int>(batch_size), static_cast<int>(topK),
            static_cast<int>(n_groups), static_cast<int>(topK_groups),
            static_cast<int>(hidden_size), experts_ids_ptr,
            selected_experts_weights_ptr, load_balance_bias_ptr);
    } else {
        auto gating_output_ptr =
            reinterpret_cast<__maca_bfloat16 *>(gating_output.data_ptr());
        auto selected_experts_weights_ptr = reinterpret_cast<__maca_bfloat16 *>(
            selected_experts_weights.data_ptr());
        auto load_balance_bias_ptr = load_balance_bias.has_value()
                                         ? reinterpret_cast<__maca_bfloat16 *>(
                                               load_balance_bias->data_ptr())
                                         : nullptr;
        fused_routing_gate_inner<float, __maca_bfloat16>(
            gating_output_ptr, static_cast<int>(score_fun), num_experts,
            static_cast<int>(batch_size), static_cast<int>(topK),
            static_cast<int>(n_groups), static_cast<int>(topK_groups),
            static_cast<int>(hidden_size), experts_ids_ptr,
            selected_experts_weights_ptr, load_balance_bias_ptr);
    }
}

} // namespace muxi_layout_kernels
