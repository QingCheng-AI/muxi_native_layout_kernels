#pragma once

#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <optional>

namespace muxi_layout_kernels {

torch::Tensor
gemv_layoutA_wapper(torch::Tensor A, torch::Tensor B, int m, int k, float alpha,
                    float beta, int blockDimX, int kernelId, int kernelParam1,
                    int kernelParam2,
                    std::optional<torch::Tensor> scale_matrix = std::nullopt,
                    std::optional<torch::Tensor> bias = std::nullopt);
torch::Tensor
gemv_layoutA(torch::Tensor A, torch::Tensor B, float alpha = 1, float beta = 0,
             std::optional<torch::Tensor> scale_matrix = std::nullopt,
             std::optional<torch::Tensor> bias = std::nullopt);

} // namespace muxi_layout_kernels
