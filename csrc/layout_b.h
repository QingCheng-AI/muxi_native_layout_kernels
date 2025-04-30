#pragma once

#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/types.h>

namespace muxi_layout_kernels {

torch::Tensor layoutB(torch::Tensor B);

} // namespace muxi_layout_kernels
