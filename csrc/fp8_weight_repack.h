#include <c10/cuda/CUDAStream.h>
#include <maca.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "utils.cuh"

namespace muxi_layout_kernels {

torch::Tensor fp8_weight_repack(torch::Tensor weight);

}