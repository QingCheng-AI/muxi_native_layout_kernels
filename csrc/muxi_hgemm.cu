#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>

#include "muxi_hgemm.cuh"
#include "muxi_hgemm.h"
#include "utils.cuh"

namespace muxi_layout_kernels {

torch::Tensor muxi_hgemm(torch::Tensor A, torch::Tensor B, float alpha,
                         float beta) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have the same k");

    int m = A.size(0), n = B.size(0), k = A.size(1);
    torch::Tensor C = torch::empty({n, m}, B.options());

    const int lda = k;
    const int ldb = k;
    const int ldc = m;
    constexpr int tile_m = 128;
    constexpr int tile_n = 128;
    constexpr int tile_k = 128;
    uint32_t gridx = (m + tile_m - 1) / tile_m;
    uint32_t gridy = (n + tile_n - 1) / tile_n;
    uint32_t gridz = 1;
    dim3 gird = {gridx, gridy, gridz};

    bool isBetaZero = (beta == static_cast<float>(0));
    if (A.dtype() == torch::kFloat16) {
        if (isBetaZero) {
            hgemm_tn_128x128x128_4m1n8k_256t<__half, __half, float, true>
                <<<gird, 256>>>(
                    reinterpret_cast<__half *>(A.data_ptr<at::Half>()),
                    reinterpret_cast<__half *>(B.data_ptr<at::Half>()),
                    reinterpret_cast<__half *>(C.data_ptr<at::Half>()), m, n, k,
                    lda, ldb, ldc, alpha, beta);
        } else {
            hgemm_tn_128x128x128_4m1n8k_256t<__half, __half, float, false>
                <<<gird, 256>>>(
                    reinterpret_cast<__half *>(A.data_ptr<at::Half>()),
                    reinterpret_cast<__half *>(B.data_ptr<at::Half>()),
                    reinterpret_cast<__half *>(C.data_ptr<at::Half>()), m, n, k,
                    lda, ldb, ldc, alpha, beta);
        }
    } else if (A.dtype() == torch::kBFloat16) {
        if (isBetaZero) {
            hgemm_tn_128x128x128_4m1n8k_256t<maca_bfloat16, maca_bfloat16,
                                             float, true><<<gird, 256>>>(
                reinterpret_cast<__maca_bfloat16 *>(A.data_ptr<at::BFloat16>()),
                reinterpret_cast<__maca_bfloat16 *>(B.data_ptr<at::BFloat16>()),
                reinterpret_cast<__maca_bfloat16 *>(C.data_ptr<at::BFloat16>()),
                m, n, k, lda, ldb, ldc, alpha, beta);
        } else {
            hgemm_tn_128x128x128_4m1n8k_256t<maca_bfloat16, maca_bfloat16,
                                             float, false><<<gird, 256>>>(
                reinterpret_cast<__maca_bfloat16 *>(A.data_ptr<at::BFloat16>()),
                reinterpret_cast<__maca_bfloat16 *>(B.data_ptr<at::BFloat16>()),
                reinterpret_cast<__maca_bfloat16 *>(C.data_ptr<at::BFloat16>()),
                m, n, k, lda, ldb, ldc, alpha, beta);
        }
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }
    return C;
}

} // namespace muxi_layout_kernels
