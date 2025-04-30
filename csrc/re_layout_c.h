#pragma once

#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/types.h>

namespace muxi_layout_kernels {

torch::Tensor reLayoutC(torch::Tensor C_in);

} // namespace muxi_layout_kernels
