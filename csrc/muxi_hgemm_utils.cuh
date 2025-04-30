#pragma once

#include <maca.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>

#include <cmath>
#include <type_traits>

namespace muxi_layout_kernels {

#define arrive_gvmcnt(num) __builtin_mxc_arrive_gvmcnt(num);
#define arrive_bsmcnt(num) __builtin_mxc_arrive_bsmcnt(num);

#define LDG_B128_BSM_NO_PREDICATOR(saddr, gaddr)                               \
    __builtin_mxc_ldg_b128_bsm_predicator(saddr, gaddr, 0, true, true, false,  \
                                          true, 1, 1, MACA_ICMP_EQ);
#define LDG_B128_BSM_WITH_PREDICATOR(saddr, gaddr, cmp_op1, cmp_op2, cmp_type) \
    __builtin_mxc_ldg_b128_bsm_predicator(saddr, gaddr, 0, true, true, false,  \
                                          true, cmp_op1, cmp_op2, cmp_type);
#define LDG_B64_BSM_NO_PREDICATOR(saddr, gaddr)                                \
    __builtin_mxc_ldg_b64_bsm_predicator(saddr, gaddr, 0, true, true, false,   \
                                         true, 1, 1, MACA_ICMP_EQ);
#define LDG_B64_BSM_with_PREDICATOR(saddr, gaddr, cmp_op1, cmp_op2, cmp_type)  \
    __builtin_mxc_ldg_b64_bsm_predicator(saddr, gaddr, 0, true, true, false,   \
                                         true, cmp_op1, cmp_op2, cmp_type);

using FLOAT4 = __NATIVE_VECTOR__(4, float);

template <typename T, bool SwapAB = false>
__forceinline__ __device__ FLOAT4 mma_16x16x16b16(uint a0, uint a1, uint b0,
                                                  uint b1, FLOAT4 C) {
    using UINT2 = __NATIVE_VECTOR__(2, uint);

    UINT2 A;
    UINT2 B;
    if constexpr (SwapAB) {
        A = UINT2{b0, b1};
        B = UINT2{a0, a1};
    } else {
        A = UINT2{a0, a1};
        B = UINT2{b0, b1};
    }

    if constexpr (std::is_same<T, __half>::value) {
        return __builtin_mxc_mma_16x16x16f16(A, B, C);
    } else {
        return __builtin_mxc_mma_16x16x16bf16(A, B, C);
    }
}

} // namespace muxi_layout_kernels
