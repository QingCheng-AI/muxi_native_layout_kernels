#pragma once

#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <optional>

namespace muxi_layout_kernels {

torch::Tensor
gemm_layoutA_wapper(torch::Tensor A, torch::Tensor B, int m, int n, int k,
                    float alpha, float beta, int kernelParam1, int kernelParam2,
                    int kernelParam3, int kernelId,
                    std::optional<torch::Tensor> bias = std::nullopt);

torch::Tensor gemm_layoutA(torch::Tensor A, torch::Tensor B, float alpha = 1,
                           float beta = 0,
                           std::optional<torch::Tensor> bias = std::nullopt);

torch::Tensor gemm_layoutA_soft_fp8_wapper(
    torch::Tensor A, std::optional<torch::Tensor> A_scale, torch::Tensor B,
    int m, int n, int k, float alpha, float beta, int kernelParam1,
    int kernelParam2, int kernelParam3, int kernelId,
    std::optional<torch::Tensor> bias = std::nullopt);

torch::Tensor
gemm_layoutA_soft_fp8(torch::Tensor A, std::optional<torch::Tensor> A_scale,
                      torch::Tensor B, float alpha = 1, float beta = 0,
                      std::optional<torch::Tensor> bias = std::nullopt);

torch::Tensor
gemm_layoutA_linear(torch::Tensor A, torch::Tensor B, float alpha = 1,
                    float beta = 0,
                    std::optional<torch::Tensor> scale_matrix = std::nullopt,
                    std::optional<torch::Tensor> bias = std::nullopt);
} // namespace muxi_layout_kernels
