#pragma once

#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <optional>

namespace muxi_layout_kernels {

torch::Tensor gemmEx(torch::Tensor A, torch::Tensor B, float alpha = 1,
                     float beta = 0);

} // namespace muxi_layout_kernels
