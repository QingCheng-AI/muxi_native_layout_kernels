#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>

#include "muxi_hgemm_layout.cuh"
#include "muxi_hgemm_layout.h"
#include "utils.cuh"

namespace muxi_layout_kernels {

torch::Tensor muxi_hgemm_layout(torch::Tensor A, torch::Tensor B, float alpha,
                                float beta, std::optional<torch::Tensor> bias) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    TORCH_CHECK(A.dim() == 4, "A must be a 4D tensor");
    TORCH_CHECK(B.dim() == 5, "B must be a 5D tensor");

    TORCH_CHECK(A.size(2) == 16, "A.shape[2] must be 16");
    TORCH_CHECK(A.size(3) == 8, "A.shape[3] must be 8");
    int m = A.size(0) * 16, A_k = A.size(1) * 8;

    TORCH_CHECK(B.size(2) == 4, "B.shape[2] must be 4");
    TORCH_CHECK(B.size(3) == 16, "B.shape[3] must be 16");
    TORCH_CHECK(B.size(4) == 8, "B.shape[4] must be 8");
    int B_k = B.size(0) * 32, n = B.size(1) * 16;

    TORCH_CHECK(A_k == B_k, "A and B must have the same k");
    int k = A_k;

    torch::Tensor C = torch::empty({n, m}, B.options());

    const int lda = k;
    const int ldb = k;
    const int ldc = m;
    constexpr int tile_m = 128;
    constexpr int tile_n = 128;
    uint32_t gridx = m / tile_m;
    uint32_t gridy = (n + tile_n - 1) / tile_n;
    uint32_t gridz = 1;
    dim3 gird = {gridx, gridy, gridz};

    bool isBetaZero = (beta == static_cast<float>(0));
    bool hasOneDimBias = !(bias == std::nullopt);

    if (A.dtype() == torch::kFloat16) {
        auto cur_device = at::cuda::current_device();
        const mcStream_t stream = at::cuda::getCurrentCUDAStream(cur_device);
        if (isBetaZero) {
            if (hasOneDimBias) {
                layout_hgemm_tn_128x128x128_4m1n8k_256t<__half, __half, float,
                                                        true, true>
                    <<<gird, 256, 0, stream>>>(
                        reinterpret_cast<__half *>(A.data_ptr<at::Half>()),
                        reinterpret_cast<__half *>(B.data_ptr<at::Half>()),
                        reinterpret_cast<__half *>(C.data_ptr<at::Half>()), m,
                        n, k, lda, ldb, ldc, alpha, beta,
                        reinterpret_cast<__half *>(bias->data_ptr<at::Half>()));
            } else {
                layout_hgemm_tn_128x128x128_4m1n8k_256t<__half, __half, float,
                                                        true, false>
                    <<<gird, 256, 0, stream>>>(
                        reinterpret_cast<__half *>(A.data_ptr<at::Half>()),
                        reinterpret_cast<__half *>(B.data_ptr<at::Half>()),
                        reinterpret_cast<__half *>(C.data_ptr<at::Half>()), m,
                        n, k, lda, ldb, ldc, alpha, beta);
            }
        } else {
            if (hasOneDimBias) {
                layout_hgemm_tn_128x128x128_4m1n8k_256t<__half, __half, float,
                                                        false, true>
                    <<<gird, 256, 0, stream>>>(
                        reinterpret_cast<__half *>(A.data_ptr<at::Half>()),
                        reinterpret_cast<__half *>(B.data_ptr<at::Half>()),
                        reinterpret_cast<__half *>(C.data_ptr<at::Half>()), m,
                        n, k, lda, ldb, ldc, alpha, beta,
                        reinterpret_cast<__half *>(bias->data_ptr<at::Half>()));
            } else {
                layout_hgemm_tn_128x128x128_4m1n8k_256t<__half, __half, float,
                                                        false, false>
                    <<<gird, 256, 0, stream>>>(
                        reinterpret_cast<__half *>(A.data_ptr<at::Half>()),
                        reinterpret_cast<__half *>(B.data_ptr<at::Half>()),
                        reinterpret_cast<__half *>(C.data_ptr<at::Half>()), m,
                        n, k, lda, ldb, ldc, alpha, beta);
            }
        }
    } else if (A.dtype() == torch::kBFloat16) {
        auto cur_device = at::cuda::current_device();
        const mcStream_t stream = at::cuda::getCurrentCUDAStream(cur_device);
        if (isBetaZero) {
            if (hasOneDimBias) {
                layout_hgemm_tn_128x128x128_4m1n8k_256t<
                    maca_bfloat16, maca_bfloat16, float, true, true>
                    <<<gird, 256, 0, stream>>>(
                        reinterpret_cast<__maca_bfloat16 *>(
                            A.data_ptr<at::BFloat16>()),
                        reinterpret_cast<__maca_bfloat16 *>(
                            B.data_ptr<at::BFloat16>()),
                        reinterpret_cast<__maca_bfloat16 *>(
                            C.data_ptr<at::BFloat16>()),
                        m, n, k, lda, ldb, ldc, alpha, beta,
                        reinterpret_cast<__maca_bfloat16 *>(
                            bias->data_ptr<at::BFloat16>()));
            } else {
                layout_hgemm_tn_128x128x128_4m1n8k_256t<
                    maca_bfloat16, maca_bfloat16, float, true, false>
                    <<<gird, 256, 0, stream>>>(
                        reinterpret_cast<__maca_bfloat16 *>(
                            A.data_ptr<at::BFloat16>()),
                        reinterpret_cast<__maca_bfloat16 *>(
                            B.data_ptr<at::BFloat16>()),
                        reinterpret_cast<__maca_bfloat16 *>(
                            C.data_ptr<at::BFloat16>()),
                        m, n, k, lda, ldb, ldc, alpha, beta);
            }
        } else {
            if (hasOneDimBias) {
                layout_hgemm_tn_128x128x128_4m1n8k_256t<
                    maca_bfloat16, maca_bfloat16, float, false, true>
                    <<<gird, 256, 0, stream>>>(
                        reinterpret_cast<__maca_bfloat16 *>(
                            A.data_ptr<at::BFloat16>()),
                        reinterpret_cast<__maca_bfloat16 *>(
                            B.data_ptr<at::BFloat16>()),
                        reinterpret_cast<__maca_bfloat16 *>(
                            C.data_ptr<at::BFloat16>()),
                        m, n, k, lda, ldb, ldc, alpha, beta,
                        reinterpret_cast<__maca_bfloat16 *>(
                            bias->data_ptr<at::BFloat16>()));
            } else {
                layout_hgemm_tn_128x128x128_4m1n8k_256t<
                    maca_bfloat16, maca_bfloat16, float, false, false>
                    <<<gird, 256, 0, stream>>>(
                        reinterpret_cast<__maca_bfloat16 *>(
                            A.data_ptr<at::BFloat16>()),
                        reinterpret_cast<__maca_bfloat16 *>(
                            B.data_ptr<at::BFloat16>()),
                        reinterpret_cast<__maca_bfloat16 *>(
                            C.data_ptr<at::BFloat16>()),
                        m, n, k, lda, ldb, ldc, alpha, beta);
            }
        }
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }
    return C;
}

} // namespace muxi_layout_kernels
