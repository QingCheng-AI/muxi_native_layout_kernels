#pragma once
#include <c10/cuda/CUDAStream.h>
#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>
#include <mcblas.h>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <iomanip>
#include <iostream>
#include <optional>
#include <type_traits>

namespace muxi_layout_kernels {

#define WARP_SIZE 64
#define MEMORY_ACCESS_SIZE 16 // int4 16 bytes

using FLOAT4 = __NATIVE_VECTOR__(4, float);
using UINT4 = __NATIVE_VECTOR__(4, uint);
using UINT2 = __NATIVE_VECTOR__(2, uint);
using INT128 = __NATIVE_VECTOR__(4, int);
using UINT2 = __NATIVE_VECTOR__(2, uint);
using INT64 = unsigned long;
using CStgType = __NATIVE_VECTOR__(2, uint);

// #define LDG_B128_BSM_NO_PREDICATOR(saddr, gaddr)                            \
//   __builtin_mxc_ldg_b128_bsm_predicator(saddr, gaddr, 0, true, true, false, \
//                                         true, 1, 1, MACA_ICMP_EQ);
#define LDG_B128_BSM_WITH_PREDICATOR_NORET0(saddr, gaddr, cmp_op1, cmp_op2,    \
                                            cmp_type)                          \
    __builtin_mxc_ldg_b128_bsm_predicator(saddr, gaddr, 0, false, true, false, \
                                          true, cmp_op1, cmp_op2, cmp_type);

#define LDG_B64_BSM_WITH_PREDICATOR_NORET0(saddr, gaddr, cmp_op1, cmp_op2,     \
                                           cmp_type)                           \
    __builtin_mxc_ldg_b64_bsm_predicator(saddr, gaddr, 0, false, true, false,  \
                                         true, cmp_op1, cmp_op2, cmp_type);

#define LDG_B128_BSM_NO_PREDICATOR(saddr, gaddr)                               \
    __builtin_mxc_ldg_b128_bsm_predicator(saddr, gaddr, 0, true, true, false,  \
                                          true, 1, 1, MACA_ICMP_EQ);

#define LDG_B64_BSM_NO_PREDICATOR(saddr, gaddr)                                \
    __builtin_mxc_ldg_b64_bsm_predicator(saddr, gaddr, 0, true, true, false,   \
                                         true, 1, 1, MACA_ICMP_EQ);

template <typename dataType>
__forceinline__ __device__ FLOAT4 mma_16x16x16f16(uint a0, uint a1, uint b0,
                                                  uint b1, FLOAT4 C) {
    UINT2 A;
    UINT2 B;

    A = UINT2{a0, a1};
    B = UINT2{b0, b1};

    if constexpr (std::is_same_v<dataType, __half>) {
        return __builtin_mxc_mma_16x16x16f16(A, B, C);
    } else if constexpr (std::is_same_v<dataType, __maca_bfloat16>) {
        return __builtin_mxc_mma_16x16x16bf16(A, B, C);
    }
}

__forceinline__ __device__ UINT2 mxc_read64_async(INT64 *tmp64ptr) {
    INT64 tmplong = __builtin_mxc_load_global_async64(tmp64ptr);
    UINT2 *tmpuint2 = reinterpret_cast<UINT2 *>(&tmplong);
    return *tmpuint2;
}

__forceinline__ __device__ void fp8tobf16(uint32_t fp8, uint *tmp4) {
    tmp4[0] = ((fp8 & 0x00800080) << 16) | ((fp8 & 0x007f007f) << 4);
    tmp4[1] = (fp8 & 0x80008000) | ((fp8 & 0x7f007f00) >> 4);
}

__forceinline__ __device__ void scalefp8tobf16(uint32_t fp8, uint16_t *tmp4,
                                               uint32_t scale) {
    uint32_t int32 = 0x7b800000;
    float *bias = reinterpret_cast<float *>(&int32);
    *bias = (*bias) * (*reinterpret_cast<float *>(&scale));
    float tmp;
    uint32_t interm;
    interm = (fp8 & 0x80000000) | ((fp8 & 0x7f000000) >> 4);
    tmp = (*bias) * (*reinterpret_cast<float *>(&interm));
    tmp4[3] = uint16_t((*reinterpret_cast<uint32_t *>(&tmp) >> 16));
    interm = ((fp8 & 0x00800000) << 8) | ((fp8 & 0x007f0000) << 4);
    tmp = (*bias) * (*reinterpret_cast<float *>(&interm));
    tmp4[2] = uint16_t((*reinterpret_cast<uint32_t *>(&tmp) >> 16));
    interm = ((fp8 & 0x00008000) << 16) | ((fp8 & 0x00007f00) << 12);
    tmp = (*bias) * (*reinterpret_cast<float *>(&interm));
    tmp4[1] = uint16_t((*reinterpret_cast<uint32_t *>(&tmp) >> 16));
    interm = ((fp8 & 0x00000080) << 24) | ((fp8 & 0x0000007f) << 20);
    tmp = (*bias) * (*reinterpret_cast<float *>(&interm));
    tmp4[0] = uint16_t((*reinterpret_cast<uint32_t *>(&tmp) >> 16));
}

template <typename T> auto mublasType() {
    if constexpr (std::is_same_v<T, __half>) {
        return MACA_R_16F;
    }
    if constexpr (std::is_same_v<T, maca_bfloat16>) {
        return MACA_R_16BF;
    }
    if constexpr (std::is_same_v<T, float>) {
        return MACA_R_32F;
    } else if constexpr (std::is_same_v<T, double>) {
        return MACA_R_64F;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return MACA_R_32I;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return MACA_R_64I;
    } else {
        assert(false); // Unsupported type
    }
}

template <typename T> auto mublasComputeType() {
    if constexpr (std::is_same_v<T, __half>) {
        return MCBLAS_COMPUTE_16F;
    }
    if constexpr (std::is_same_v<T, maca_bfloat16>) {
        return MCBLAS_COMPUTE_32F;
    }
    if constexpr (std::is_same_v<T, float>) {
        return MCBLAS_COMPUTE_32F;
    } else if constexpr (std::is_same_v<T, double>) {
        return MCBLAS_COMPUTE_64F;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return MCBLAS_COMPUTE_32I;
    } else {
        assert(false); // Unsupported type
    }
}

template <typename T> __device__ T zero();
template <> __device__ __attribute__((always_inline)) __half zero<__half>() {
    // Work around a mxcc bug on immediate values
    __half zero;
    asm("mov_b32 %0, 0" : "=r"(zero));
    return zero;
}
template <> __device__ __attribute__((always_inline)) float zero<float>() {
    // Work around a mxcc bug on immediate values
    float zero;
    asm("mov_b32 %0, 0" : "=r"(zero));
    return zero;
}

__device__ __attribute__((always_inline)) __half dotHalf(const UINT4 &lhs,
                                                         const UINT4 &rhs) {
    __half accum;
    asm("mov_b32 %0, 0\n\t"
        "fmac_f16 %0_lo, %2_lo, %3_lo\n\t"
        "fmac_f16 %0_lo, %2_hi, %3_hi\n\t"
        "fmac_f16 %0_lo, %4_lo, %5_lo\n\t"
        "fmac_f16 %0_lo, %4_hi, %5_hi\n\t"
        "fmac_f16 %0_lo, %6_lo, %7_lo\n\t"
        "fmac_f16 %0_lo, %6_hi, %7_hi\n\t"
        "fmac_f16 %0_lo, %8_lo, %9_lo\n\t"
        "fmac_f16 %0_lo, %8_hi, %9_hi"
        : "=r"(accum)
        : "0"(accum), "r"(lhs.x), "r"(rhs.x), "r"(lhs.y), "r"(rhs.y),
          "r"(lhs.z), "r"(rhs.z), "r"(lhs.w), "r"(rhs.w));
    return accum;
}

__device__ __attribute__((always_inline)) maca_bfloat16
dotBf16(const UINT4 &lhs, const UINT4 &rhs) {
    // MUXI can not compute BF16 without tensor core, so don`t have asm impl
    // here. maca_bfloat16 accum = static_cast<maca_bfloat16>(0.0f);
    maca_bfloat16 accum = 0x0000;
    accum += reinterpret_cast<const maca_bfloat16 *>(&lhs)[0] *
             reinterpret_cast<const maca_bfloat16 *>(&rhs)[0];
    accum += reinterpret_cast<const maca_bfloat16 *>(&lhs)[1] *
             reinterpret_cast<const maca_bfloat16 *>(&rhs)[1];
    accum += reinterpret_cast<const maca_bfloat16 *>(&lhs)[2] *
             reinterpret_cast<const maca_bfloat16 *>(&rhs)[2];
    accum += reinterpret_cast<const maca_bfloat16 *>(&lhs)[3] *
             reinterpret_cast<const maca_bfloat16 *>(&rhs)[3];
    accum += reinterpret_cast<const maca_bfloat16 *>(&lhs)[4] *
             reinterpret_cast<const maca_bfloat16 *>(&rhs)[4];
    accum += reinterpret_cast<const maca_bfloat16 *>(&lhs)[5] *
             reinterpret_cast<const maca_bfloat16 *>(&rhs)[5];
    accum += reinterpret_cast<const maca_bfloat16 *>(&lhs)[6] *
             reinterpret_cast<const maca_bfloat16 *>(&rhs)[6];
    accum += reinterpret_cast<const maca_bfloat16 *>(&lhs)[7] *
             reinterpret_cast<const maca_bfloat16 *>(&rhs)[7];
    return accum;
}

typedef struct alignas(16) half8 {
    __half elements[8];
} half8;

template <typename T>
__device__ __forceinline__ T my_shfl_xor_sync(T var, int offset) {
    int index = (threadIdx.x % WARP_SIZE) ^ offset;
    int ret = __builtin_mxc_bsm_bpermute(index << 2, *((int *)&var));
    return *((T *)&ret);
}

template <typename Taccum>
__device__ __forceinline__ Taccum warpReduceSum(Taccum sum,
                                                unsigned int threadNum) {
    if (threadNum >= 64)
        sum += my_shfl_xor_sync(sum, 32);
    if (threadNum >= 32)
        sum += my_shfl_xor_sync(sum, 16);
    if (threadNum >= 16)
        sum += my_shfl_xor_sync(sum, 8);
    if (threadNum >= 8)
        sum += my_shfl_xor_sync(sum, 4);
    if (threadNum >= 4)
        sum += my_shfl_xor_sync(sum, 2);
    if (threadNum >= 2)
        sum += my_shfl_xor_sync(sum, 1);
    return sum;
}

__forceinline__ __device__ void scalefp8tobf16(uint32_t fp8, uint16_t *tmp4,
                                               float &scale) {
    uint32_t int32 = 0x7b800000;
    float *bias = reinterpret_cast<float *>(&int32);
    *bias = (*bias) * scale;
    float tmp;
    uint32_t interm;
    interm = (fp8 & 0x80000000) | ((fp8 & 0x7f000000) >> 4);
    tmp = (*bias) * (*reinterpret_cast<float *>(&interm));
    tmp4[3] = uint16_t((*reinterpret_cast<uint32_t *>(&tmp) >> 16));
    interm = ((fp8 & 0x00800000) << 8) | ((fp8 & 0x007f0000) << 4);
    tmp = (*bias) * (*reinterpret_cast<float *>(&interm));
    tmp4[2] = uint16_t((*reinterpret_cast<uint32_t *>(&tmp) >> 16));
    interm = ((fp8 & 0x00008000) << 16) | ((fp8 & 0x00007f00) << 12);
    tmp = (*bias) * (*reinterpret_cast<float *>(&interm));
    tmp4[1] = uint16_t((*reinterpret_cast<uint32_t *>(&tmp) >> 16));
    interm = ((fp8 & 0x00000080) << 24) | ((fp8 & 0x0000007f) << 20);
    tmp = (*bias) * (*reinterpret_cast<float *>(&interm));
    tmp4[0] = uint16_t((*reinterpret_cast<uint32_t *>(&tmp) >> 16));
}

} // namespace muxi_layout_kernels
