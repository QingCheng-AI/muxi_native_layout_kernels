#pragma once

#include <c10/cuda/CUDAStream.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>

#include <algorithm>
#include <iostream>
#include <random>

#define WARPSIZE 64
#define WARP_SIZE 64
#define STAGES 4
#define ELEMENTSPERACCESS 8
#define ROWTHREADSPERMMA 16
#define COLTHREADSPERMMA 4
#define ELEMENTSPERTHREADPERMMA 4
#define MEMORY_ACCESS_SIZE 16

static float gemm_time = 0.0f;

using FLOAT4 = __NATIVE_VECTOR__(4, float);
using UINT4 = __NATIVE_VECTOR__(4, uint);
using INT128 = __NATIVE_VECTOR__(4, int);
using UINT2 = __NATIVE_VECTOR__(2, uint);
using INT64 = unsigned long;

#define LDG_B128_BSM_NO_PREDICATOR(saddr, gaddr)                               \
    __builtin_mxc_ldg_b128_bsm_predicator(saddr, gaddr, 0, true, true, false,  \
                                          true, 1, 1, MACA_ICMP_EQ);
#define LDG_B128_BSM_WITH_PREDICATOR_NORET0(saddr, gaddr, cmp_op1, cmp_op2,    \
                                            cmp_type)                          \
    __builtin_mxc_ldg_b128_bsm_predicator(saddr, gaddr, 0, false, true, false, \
                                          true, cmp_op1, cmp_op2, cmp_type);

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

template <typename Tax>
void initializeHostData(Tax *A, size_t m, size_t n, int add1 = 0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distA(-1.0, 1.0);
    float x = 0.0;
    for (int i = 0; i < m * n; ++i) {
        if (add1 == 0) {
#ifndef DEBUG
            A[i] = static_cast<Tax>(distA(gen));
#else
            A[i] = static_cast<Tax>(0.01f);
#endif
        } else {
            A[i] = static_cast<Tax>(x);
            x += 1.0f;
        }
    }
}

__forceinline__ __device__ void scalefp8tobf16(uint32_t fp8, uint16_t *tmp4,
                                               const float &scale) {
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

__forceinline__ __device__ UINT2 mxc_read64_async(INT64 *tmp64ptr) {
    INT64 tmplong = __builtin_mxc_load_global_async64(tmp64ptr);
    UINT2 *tmpuint2 = reinterpret_cast<UINT2 *>(&tmplong);
    return *tmpuint2;
}
