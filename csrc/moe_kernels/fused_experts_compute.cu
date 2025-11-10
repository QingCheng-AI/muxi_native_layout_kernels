#include "../dispatch_utils.h"
#include "fused_experts_compute.h"

namespace muxi_layout_kernels {

void batched_routed_activation_indexed_to_expert_block_indexed(
    int batchSize, int expertCount, int topK, int microBatchSize,
    torch::Tensor &expertsIds, torch::Tensor &dev_sorted_token_ids,
    torch::Tensor &dev_cumsum_buffer, torch::Tensor &dev_padded_num_experts,
    torch::Tensor &dev_experts_ids, std::optional<torch::Tensor> experts_map) {
    TORCH_CHECK(expertsIds.dtype() == torch::kInt32,
                "expertsIds must be of type torch::kInt32.");
    TORCH_CHECK(
        expertsIds.device() == dev_sorted_token_ids.device(),
        "expertsIds must be on the same device as dev_sorted_token_ids.");
    TORCH_CHECK(expertsIds.device() == dev_cumsum_buffer.device(),
                "expertsIds must be on the same device as dev_cumsum_buffer.");
    TORCH_CHECK(
        expertsIds.device() == dev_padded_num_experts.device(),
        "expertsIds must be on the same device as dev_padded_num_experts.");
    TORCH_CHECK(expertsIds.device() == dev_experts_ids.device(),
                "expertsIds must be on the same device as dev_experts_ids.");

    auto experts_ids_ptr = reinterpret_cast<int *>(expertsIds.data_ptr());
    auto experts_map_ptr =
        experts_map != std::nullopt
            ? reinterpret_cast<int *>(experts_map->data_ptr())
            : nullptr;
    auto dev_sorted_token_ids_ptr =
        reinterpret_cast<int *>(dev_sorted_token_ids.data_ptr());
    auto dev_cumsum_buffer_ptr =
        reinterpret_cast<int *>(dev_cumsum_buffer.data_ptr());
    auto dev_padded_num_experts_ptr =
        reinterpret_cast<int *>(dev_padded_num_experts.data_ptr());
    auto dev_experts_ids_ptr =
        reinterpret_cast<int *>(dev_experts_ids.data_ptr());

    dispatchToStaticInts<8, 16, 32, 64, 128, 256>(
        expertCount, [&]<int expertCount>() {
            dispatchToStaticInts<16>(microBatchSize, [&]<int microBatchSize>() {
                batched_routed_activation_indexed_to_expert_block_indexed_inner<
                    expertCount, microBatchSize>(
                    batchSize, topK, experts_ids_ptr, experts_map_ptr,
                    dev_sorted_token_ids_ptr, dev_cumsum_buffer_ptr,
                    dev_padded_num_experts_ptr, dev_experts_ids_ptr);
            });
        });
}

void fused_experts_compute(
    torch::Tensor &experts_weights_matrix1,
    torch::Tensor &experts_weights_matrix2, torch::Tensor &activations,
    int64_t batchSize, int64_t expertCount, int64_t topK,
    torch::Tensor &expertsIds, torch::Tensor &activedExpertsWeights,
    torch::Tensor &dev_sorted_token_ids, torch::Tensor &dev_cumsum_buffer,
    torch::Tensor &dev_padded_num_experts, torch::Tensor &dev_experts_ids,
    torch::Tensor &dev_C, torch::Tensor &y, int APerWarp, int splitK,
    int tile_m_2, int tile_n_2, int tile_k_2, int block_dim_x_gemm,
    int microBatchSize, std::optional<torch::Tensor> experts_map) {
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

    int64_t experts_num = expertCount;
    if (experts_map != std::nullopt)
        experts_num = experts_map->size(0);
    batched_routed_activation_indexed_to_expert_block_indexed(
        batchSize, experts_num, topK, microBatchSize, expertsIds,
        dev_sorted_token_ids, dev_cumsum_buffer, dev_padded_num_experts,
        dev_experts_ids, experts_map);

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
                    dispatchToStaticInts<16>(
                        microBatchSize, [&]<int microBatchSize>() {
                            fused_experts_compute_inner<
                                expertCount, microBatchSize, half, float, half>(
                                w1_ptr, w2_ptr, activations_ptr, m1, n1, k1, m2,
                                n2, k2, static_cast<int>(batchSize),
                                static_cast<int>(topK), experts_ids_ptr,
                                activedExpertsWeights_ptr,
                                dev_sorted_token_ids_ptr,
                                dev_padded_num_experts_ptr, dev_experts_ids_ptr,
                                reinterpret_cast<half *>(dev_C.data_ptr()),
                                reinterpret_cast<half *>(y.data_ptr()),
                                APerWarp, splitK, tile_m_2, tile_n_2, tile_k_2,
                                block_dim_x_gemm);
                        });
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
            dispatchToStaticInts<8, 16, 32, 64, 128,
                                 256>(expertCount, [&]<int expertCount>() {
                dispatchToStaticInts<16>(
                    microBatchSize, [&]<int microBatchSize>() {
                        fused_experts_compute_inner<expertCount, microBatchSize,
                                                    __maca_bfloat16, float,
                                                    __maca_bfloat16>(
                            w1_ptr, w2_ptr, activations_ptr, m1, n1, k1, m2, n2,
                            k2, static_cast<int>(batchSize),
                            static_cast<int>(topK), experts_ids_ptr,
                            activedExpertsWeights_ptr, dev_sorted_token_ids_ptr,
                            dev_padded_num_experts_ptr, dev_experts_ids_ptr,
                            reinterpret_cast<__maca_bfloat16 *>(
                                dev_C.data_ptr()),
                            reinterpret_cast<__maca_bfloat16 *>(y.data_ptr()),
                            APerWarp, splitK, tile_m_2, tile_n_2, tile_k_2,
                            block_dim_x_gemm);
                    });
            });
        }
    }
}

void fused_experts_compute(
    torch::Tensor &experts_weights_matrix1,
    torch::Tensor &experts_weights_matrix2, torch::Tensor &activations,
    int64_t batchSize, int64_t expertCount, int64_t topK,
    torch::Tensor &expertsIds, torch::Tensor &activedExpertsWeights,
    torch::Tensor &dev_sorted_token_ids, torch::Tensor &dev_cumsum_buffer,
    torch::Tensor &dev_padded_num_experts, torch::Tensor &dev_experts_ids,
    torch::Tensor &dev_C, torch::Tensor &y, torch::Tensor &w1_scale,
    torch::Tensor &w2_scale, std::vector<int64_t> &block_shape, bool soft_fp8,
    int microBatchSize, std::optional<torch::Tensor> experts_map) {
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

    int64_t experts_num = expertCount;
    if (experts_map != std::nullopt)
        experts_num = experts_map->size(0);
    batched_routed_activation_indexed_to_expert_block_indexed(
        batchSize, experts_num, topK, microBatchSize, expertsIds,
        dev_sorted_token_ids, dev_cumsum_buffer, dev_padded_num_experts,
        dev_experts_ids, experts_map);

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
    TORCH_CHECK(soft_fp8 == true, "soft_fp8 must be true for float8_e4m3fn.");

    if (ActDtype == torch::kBFloat16) {
        auto activations_ptr =
            reinterpret_cast<__maca_bfloat16 *>(activations.data_ptr());
        auto activedExpertsWeights_ptr = reinterpret_cast<__maca_bfloat16 *>(
            activedExpertsWeights.data_ptr());
        dispatchToStaticInts<8, 16, 32, 64, 128,
                             256>(expertCount, [&]<int expertCount>() {
            dispatchToStaticInts<16>(microBatchSize, [&]<int microBatchSize>() {
                fused_experts_compute_inner<expertCount, microBatchSize,
                                            uint8_t, float, __maca_bfloat16>(
                    w1_ptr, w2_ptr, activations_ptr, m1, n1, k1, m2, n2, k2,
                    static_cast<int>(batchSize), static_cast<int>(topK),
                    experts_ids_ptr, activedExpertsWeights_ptr,
                    dev_sorted_token_ids_ptr, dev_padded_num_experts_ptr,
                    dev_experts_ids_ptr,
                    reinterpret_cast<__maca_bfloat16 *>(dev_C.data_ptr()),
                    reinterpret_cast<__maca_bfloat16 *>(y.data_ptr()),
                    reinterpret_cast<float *>(w1_scale.data_ptr()),
                    reinterpret_cast<float *>(w2_scale.data_ptr()),
                    w1_scale.size(1), w1_scale.size(2), w2_scale.size(1),
                    w2_scale.size(2));
            });
        });

    } else {
        TORCH_CHECK(false, "Weight date type and activation date type "
                           "should be bfloat16 ");
    }
}

} // namespace muxi_layout_kernels
