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

#include "fp8_weight_repack.h"
#include "gemm_ex.h"
#include "gemm_layout_A.h"
#include "gemm_layout_ab_continuous_c.h"
#include "gemm_layout_abc.h"
#include "gemv_layout_a.h"
#include "layout_b.h"
#include "layout_gemv_kernel.cuh"
#include "moe_kernels/experts_compute.h"
#include "moe_kernels/fused_experts_compute.h"
#include "moe_kernels/fused_gating.h"
#include "moe_kernels/routing_gate.h"
#include "muxi_hgemm.h"
#include "muxi_hgemm_layout.h"
#include "muxi_hgemm_layoutA.h"
#include "muxi_hgemm_layoutC.h"
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
    m.def("routing_gate", &routing_gate, "gating_output"_a, "score_fun"_a,
          "batch_size"_a, "hidden_size"_a, "n_groups"_a, "topK_groups"_a,
          "experts_ids"_a, "selected_experts_weights"_a, "topK"_a,
          "load_balance_bias"_a = c10::nullopt);
    m.def("experts_compute", &experts_compute, "experts_weights_matrix1"_a,
          "experts_weights_matrix2"_a, "activations"_a, "batch_size"_a,
          "expert_count"_a, "dynamic_experts_per_act"_a, "experts_ids"_a,
          "actived_experts_weights"_a, "bias"_a = c10::nullopt);
    m.def("fused_routing_gate", &fused_routing_gate, "gating_output"_a,
          "score_fun"_a, "batch_size"_a, "hidden_size"_a, "n_groups"_a,
          "topK_groups"_a, "experts_ids"_a, "selected_experts_weights"_a,
          "topK"_a, "load_balance_bias"_a = c10::nullopt);
    m.def("batched_routed_activation_indexed_to_expert_block_indexed",
          &batched_routed_activation_indexed_to_expert_block_indexed,
          "batchSize"_a, "expertCount"_a, "topK"_a, "microBatchSize"_a,
          "expertsIds"_a, "dev_sorted_token_ids"_a, "dev_cumsum_buffer"_a,
          "dev_padded_num_experts"_a, "dev_experts_ids"_a);
    m.def(
        "fused_experts_compute",
        static_cast<void (*)(
            torch::Tensor &, torch::Tensor &, torch::Tensor &, int64_t, int64_t,
            int64_t, torch::Tensor &, torch::Tensor &, torch::Tensor &,
            torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
            torch::Tensor &, int, int, int, int, int, int, int)>(
            &fused_experts_compute),
        "experts_weights_matrix1"_a, "experts_weights_matrix2"_a,
        "activations"_a, "batch_size"_a, "expert_count"_a,
        "dynamic_experts_per_act"_a, "topk_ids"_a, "actived_experts_weights"_a,
        "sorted_token_ids"_a, "cumsum_buffer"_a, "padded_num_experts"_a,
        "experts_ids"_a, "C"_a, "y"_a, "APerWarp"_a = 2, "splitK"_a = 3,
        "tile_m_2"_a = 128, "tile_n_2"_a = 16, "tile_k_2"_a = 128,
        "block_dim_x_gemm"_a = 256, "microBatchSize"_a = 16);
    m.def(
        "fused_experts_compute",
        static_cast<void (*)(
            torch::Tensor &, torch::Tensor &, torch::Tensor &, int64_t, int64_t,
            int64_t, torch::Tensor &, torch::Tensor &, torch::Tensor &,
            torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &,
            torch::Tensor &, torch::Tensor &, torch::Tensor &,
            std::vector<int64_t> &, bool, int)>(&fused_experts_compute),
        "experts_weights_matrix1"_a, "experts_weights_matrix2"_a,
        "activations"_a, "batch_size"_a, "expert_count"_a,
        "dynamic_experts_per_act"_a, "topk_ids"_a, "actived_experts_weights"_a,
        "sorted_token_ids"_a, "cumsum_buffer"_a, "padded_num_experts"_a,
        "experts_ids"_a, "C"_a, "y"_a, "w1_scale"_a, "w2_scale"_a,
        "block_shape"_a, "soft_fp8"_a = false, "microBatchSize"_a = 16);
}
