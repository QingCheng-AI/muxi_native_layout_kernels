#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>
#include <mcblas.h>

#include "gemm_ex.h"
#include "utils.cuh"

namespace muxi_layout_kernels {

class McblasHandleManager {
  public:
    McblasHandleManager() { mcblasCreate(&handle); }

    ~McblasHandleManager() { mcblasDestroy(handle); }

    mcblasHandle_t getHandle() { return handle; }

  private:
    mcblasHandle_t handle;
};

// 全局单例
McblasHandleManager &getGlobalHandleManager() {
    static McblasHandleManager instance;
    return instance;
}

template <typename Tab, typename Taccum, typename Tc>
void mcblasGemmExInterface(Tab *A, Tab *B, Tc *C, int m, int n, int k,
                           Taccum alpha, Taccum beta) {
    constexpr auto mcblasTransA = MCBLAS_OP_N;
    constexpr auto mcblasTransB = MCBLAS_OP_N;
    auto lda = m;
    auto ldb = k;
    auto ldc = m;
    auto handle = getGlobalHandleManager().getHandle();

    mcblasGemmEx(handle, mcblasTransA, mcblasTransB, m, n, k, &alpha, A,
                 mublasType<Tab>(), lda, B, mublasType<Tab>(), ldb, &beta, C,
                 mublasType<Tc>(), ldc, mublasComputeType<Taccum>(),
                 MCBLAS_GEMM_DEFAULT);
}

torch::Tensor gemmEx(torch::Tensor A, torch::Tensor B, float alpha,
                     float beta) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(1), "A and B must have the same k");
    int m = A.size(1), n = B.size(0), k = A.size(0);
    torch::Tensor C = torch::empty({n, m}, B.options());

    if (A.dtype() == torch::kFloat16) {
        mcblasGemmExInterface<__half, float, __half>(
            reinterpret_cast<__half *>(A.data_ptr<at::Half>()),
            reinterpret_cast<__half *>(B.data_ptr<at::Half>()),
            reinterpret_cast<__half *>(C.data_ptr<at::Half>()), m, n, k, alpha,
            beta);
    } else if (A.dtype() == torch::kBFloat16) {
        mcblasGemmExInterface<__maca_bfloat16, float, __maca_bfloat16>(
            reinterpret_cast<__maca_bfloat16 *>(A.data_ptr<at::BFloat16>()),
            reinterpret_cast<__maca_bfloat16 *>(B.data_ptr<at::BFloat16>()),
            reinterpret_cast<__maca_bfloat16 *>(C.data_ptr<at::BFloat16>()), m,
            n, k, alpha, beta);
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }

    return C;
}

} // namespace muxi_layout_kernels
