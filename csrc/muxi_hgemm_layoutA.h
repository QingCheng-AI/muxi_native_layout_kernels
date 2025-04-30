#pragma once

#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <optional>

namespace muxi_layout_kernels {

torch::Tensor
muxi_hgemm_layoutA(torch::Tensor A, torch::Tensor B, float alpha = 1,
                   float beta = 0,
                   std::optional<torch::Tensor> scale_matrix = std::nullopt,
                   std::optional<torch::Tensor> bias = std::nullopt);

} // namespace muxi_layout_kernels
