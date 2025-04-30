#pragma once

#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <optional>

namespace muxi_layout_kernels {

torch::Tensor gemm_layoutAB_ContinuousC_wapper(
    torch::Tensor A, torch::Tensor B, int m, int n, int k, float alpha,
    float beta, int kernelParam1, int kernelParam2, int kernelParam3,
    int kernelId, std::optional<torch::Tensor> bias = std::nullopt);

torch::Tensor
gemm_layoutAB_ContinuousC(torch::Tensor A, torch::Tensor B, float alpha = 1,
                          float beta = 0,
                          std::optional<torch::Tensor> bias = std::nullopt);

} // namespace muxi_layout_kernels
