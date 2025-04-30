#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>

#include "arg_selector.h"
#include "gemv_layout_a.h"
#include "layout_gemv_kernel.cuh"
#include "layout_gemv_kernel_soft_fp8.cuh"
#include "utils.cuh"

namespace muxi_layout_kernels {

torch::Tensor gemv_layoutA_wapper(torch::Tensor A, torch::Tensor B, int m,
                                  int k, float alpha, float beta, int blockDimX,
                                  int kernelId, int kernelParam1,
                                  int kernelParam2,
                                  std::optional<torch::Tensor> scale_matrix,
                                  std::optional<torch::Tensor> bias) {
    torch::Tensor C;
    if (kernelId == 1) {
        C = torch::zeros({1, m}, B.options());
    } else if (kernelId == 2) {
        C = torch::empty({1, m}, B.options());
    }

#define GEMV_KERNEL_LAUNCH(blockDimX)                                          \
    if (A.dtype() == torch::kFloat16) {                                        \
        GemvMmaLayoutDispatch<__half, float, __half, blockDimX>(               \
            reinterpret_cast<__half *>(A.data_ptr<at::Half>()),                \
            reinterpret_cast<__half *>(B.data_ptr<at::Half>()),                \
            reinterpret_cast<__half *>(C.data_ptr<at::Half>()), m, k, alpha,   \
            beta,                                                              \
            bias == std::nullopt                                               \
                ? nullptr                                                      \
                : reinterpret_cast<__half *>(bias->data_ptr<at::Half>()),      \
            kernelId, kernelParam1, kernelParam2);                             \
    } else if (A.dtype() == torch::kBFloat16) {                                \
        GemvMmaLayoutDispatch<__maca_bfloat16, float, __maca_bfloat16,         \
                              blockDimX>(                                      \
            reinterpret_cast<__maca_bfloat16 *>(A.data_ptr<at::BFloat16>()),   \
            reinterpret_cast<__maca_bfloat16 *>(B.data_ptr<at::BFloat16>()),   \
            reinterpret_cast<__maca_bfloat16 *>(C.data_ptr<at::BFloat16>()),   \
            m, k, alpha, beta,                                                 \
            bias == std::nullopt ? nullptr                                     \
                                 : reinterpret_cast<__maca_bfloat16 *>(        \
                                       bias->data_ptr<at::BFloat16>()),        \
            kernelId, kernelParam1, kernelParam2);                             \
    } else if (A.element_size() == 1 && B.dtype() == torch::kBFloat16) {       \
        GemvMmaLayoutDispatchSoftFp8<uint8_t, maca_bfloat16, float,            \
                                     maca_bfloat16, blockDimX>(                \
            reinterpret_cast<uint8_t *>(A.data_ptr()),                         \
            reinterpret_cast<__maca_bfloat16 *>(B.data_ptr<at::BFloat16>()),   \
            reinterpret_cast<__maca_bfloat16 *>(C.data_ptr<at::BFloat16>()),   \
            m, k, alpha, beta,                                                 \
            reinterpret_cast<float *>(scale_matrix->data_ptr()),               \
            bias == std::nullopt ? nullptr                                     \
                                 : reinterpret_cast<__maca_bfloat16 *>(        \
                                       bias->data_ptr<at::BFloat16>()),        \
            kernelId, kernelParam1, kernelParam2);                             \
    } else {                                                                   \
        TORCH_CHECK(false, "Unsupported data type");                           \
    }

    if (blockDimX == 64) {
        GEMV_KERNEL_LAUNCH(64);
    } else if (blockDimX == 128) {
        GEMV_KERNEL_LAUNCH(128);
    } else if (blockDimX == 256) {
        GEMV_KERNEL_LAUNCH(256);
    } else if (blockDimX == 512) {
        GEMV_KERNEL_LAUNCH(512);
    } else if (blockDimX == 1024) {
        GEMV_KERNEL_LAUNCH(1024);
    }

#undef GEMV_KERNEL_LAUNCH

    return C;
} // namespace muxi_layout_kernels

torch::Tensor gemv_layoutA(torch::Tensor A, torch::Tensor B, float alpha,
                           float beta,
                           std::optional<torch::Tensor> scale_matrix,
                           std::optional<torch::Tensor> bias) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    TORCH_CHECK(A.dim() == 4, "A must be a 4D tensor");

    TORCH_CHECK(A.size(2) == 16, "A.shape[2] must be 16");
    TORCH_CHECK(A.size(3) == 8, "A.shape[3] must be 8");
    int m = A.size(0) * 16, A_k = A.size(1) * 8;

    TORCH_CHECK(B.size(0) == 1, "B must be a vector");
    int B_k = B.size(1);

    TORCH_CHECK(A_k == B_k, "A and B must have the same k");
    int k = A_k;
    TORCH_CHECK(m % 16 == 0, "GEMV m % 16(at least) must be 0")
    TORCH_CHECK(k % 32 == 0, "GEMV k % 32(at least) must be 0")

    // if (A.element_size() == 1 && B.dtype() == torch::kBFloat16) {
    //   auto &&[BLOCK_DIM_X, KernelId, KernelParam1, KernelParam2] =
    //       getGlobalSoftFp8GemvArgSelector().getArgs({m, k});
    // } else {
    //   auto &&[BLOCK_DIM_X, KernelId, KernelParam1, KernelParam2] =
    //       getGlobalGemvArgSelector().getArgs({m, k});
    // }

    int BLOCK_DIM_X;
    int KernelId;
    int KernelParam1;
    int KernelParam2;

    if (A.element_size() == 1 && B.dtype() == torch::kBFloat16) {
        auto result = getGlobalSoftFp8GemvArgSelector().getArgs({m, k});
        BLOCK_DIM_X = std::get<0>(result);
        KernelId = std::get<1>(result);
        KernelParam1 = std::get<2>(result);
        KernelParam2 = std::get<3>(result);
    } else if (B.dtype() == torch::kFloat16) {
        auto result = getGlobalGemvArgSelector().getArgs({m, k});
        BLOCK_DIM_X = std::get<0>(result);
        KernelId = std::get<1>(result);
        KernelParam1 = std::get<2>(result);
        KernelParam2 = std::get<3>(result);
    } else if (B.dtype() == torch::kBFloat16) {
        auto result = getGlobalBf16GemvArgSelector().getArgs({m, k});
        BLOCK_DIM_X = std::get<0>(result);
        KernelId = std::get<1>(result);
        KernelParam1 = std::get<2>(result);
        KernelParam2 = std::get<3>(result);
    }

    if ((KernelId == 1) && (m % (16 * KernelParam1) != 0)) {
        BLOCK_DIM_X = 256;
        KernelId = 1;
        KernelParam1 = 1;
        KernelParam2 = 1;
    }
    if ((KernelId == 2) && (k > 8192)) {
        BLOCK_DIM_X = 256;
        KernelId = 1;
        KernelParam1 = 1;
        KernelParam2 = 1;
    }

    torch::Tensor C =
        gemv_layoutA_wapper(A, B, m, k, alpha, beta, BLOCK_DIM_X, KernelId,
                            KernelParam1, KernelParam2, scale_matrix, bias);

    return C;
}

} // namespace muxi_layout_kernels
