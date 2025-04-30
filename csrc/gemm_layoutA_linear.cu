#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>

#include "gemm_layout_A.h"

namespace muxi_layout_kernels {
torch::Tensor gemm_layoutA_linear(torch::Tensor A, torch::Tensor B, float alpha,
                                  float beta,
                                  std::optional<torch::Tensor> scale_matrix,
                                  std::optional<torch::Tensor> bias) {
    torch::Tensor C;
    if (scale_matrix != std::nullopt) {
        C = gemm_layoutA_soft_fp8(A, scale_matrix, B, alpha, beta, bias);
    } else {
        C = gemm_layoutA(A, B, alpha, beta, bias);
    }
    return C;
}
} // namespace muxi_layout_kernels