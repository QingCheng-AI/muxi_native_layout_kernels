// The development of this file is kindly supported by MetaX.

#pragma once

#include "muxi_hgemm_utils.cuh"

namespace muxi_layout_kernels {

template <typename T, typename Tc, typename Tscal, bool IsBetaZero,
          bool HasOneDimBias>
__forceinline__ __device__ void layoutA_hgemm_tn_128x128x128_4m1n8k_256t_device(
    const void *A, const void *B, void *C, int M, int N, int K, int lda,
    int ldb, int ldc, Tscal alpha, Tscal beta, const void *bias, int bidx,
    int bidy) {
    constexpr int TileM = 128;
    constexpr int TileN = 128;
    constexpr int Stage = 4;
    using ALdgType = __NATIVE_VECTOR__(4, uint);
    using BLdgType = __NATIVE_VECTOR__(4, uint);
    using CStgType = __NATIVE_VECTOR__(sizeof(Tc), uint);
    using ALdsType = ALdgType;
    using BLdsType = BLdgType;
    using FLOAT4 = __NATIVE_VECTOR__(4, float);

    uint8_t *APtr = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(A));
    uint8_t *BPtr = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(B));
    // CStgType *CPtr = reinterpret_cast<CStgType *>(C);

    lda /= sizeof(ALdgType) / sizeof(T);
    ldb /= sizeof(BLdgType) / sizeof(T);
    // ldc /= sizeof(CStgType) / sizeof(Tc);

    const int startRow = bidx * TileM;
    const int startCol = bidy * TileN;

    APtr += startRow * lda * sizeof(ALdgType);
    // BPtr += startCol / 128 * 64 * (128 / 16) * sizeof(BLdgType);

    // for Col major B
    BPtr += startCol * K * sizeof(T);

    // CPtr += startCol * ldc + startRow / (sizeof(CStgType) / sizeof(Tc));

    const int tid = threadIdx.x;
    const int slot = __builtin_mxc_readfirstlane(tid / 64);
    const int lane = tid & 63;
    const int quarterLaneId = lane & 15;
    const int quarterWarpId = lane / 16;

    // const int A_col_offset = tid;
    // const int A_row_offset = 16 * lda;

    int ALdgOffset[2][4];

    ALdgOffset[0][0] = (tid + 16 * lda * 0) * sizeof(ALdgType);
    ALdgOffset[0][1] = (tid + 16 * lda * 1) * sizeof(ALdgType);
    ALdgOffset[0][2] = (tid + 16 * lda * 2) * sizeof(ALdgType);
    ALdgOffset[0][3] = (tid + 16 * lda * 3) * sizeof(ALdgType);
    ALdgOffset[1][0] = (tid + 16 * lda * 4) * sizeof(ALdgType);
    ALdgOffset[1][1] = (tid + 16 * lda * 5) * sizeof(ALdgType);
    ALdgOffset[1][2] = (tid + 16 * lda * 6) * sizeof(ALdgType);
    ALdgOffset[1][3] = (tid + 16 * lda * 7) * sizeof(ALdgType);

    // const int B_row_offset = lane + slot * 64 * (N / 16);
    // const int B_col_offset = 64;

    // for Col major B
    const int B_row_offset =
        (quarterLaneId * K * sizeof(T) / sizeof(BLdgType)) + slot * 4 +
        quarterWarpId;

    int BLdgOffset[2][4];
    BLdgOffset[0][0] = B_row_offset * sizeof(BLdgType) + 0 * 16 * K * sizeof(T);
    BLdgOffset[0][1] = B_row_offset * sizeof(BLdgType) + 1 * 16 * K * sizeof(T);
    BLdgOffset[0][2] = B_row_offset * sizeof(BLdgType) + 2 * 16 * K * sizeof(T);
    BLdgOffset[0][3] = B_row_offset * sizeof(BLdgType) + 3 * 16 * K * sizeof(T);
    BLdgOffset[1][0] = B_row_offset * sizeof(BLdgType) + 4 * 16 * K * sizeof(T);
    BLdgOffset[1][1] = B_row_offset * sizeof(BLdgType) + 5 * 16 * K * sizeof(T);
    BLdgOffset[1][2] = B_row_offset * sizeof(BLdgType) + 6 * 16 * K * sizeof(T);
    BLdgOffset[1][3] = B_row_offset * sizeof(BLdgType) + 7 * 16 * K * sizeof(T);

    int ALdsOffset[4];
    int BLdsOffset[4];

#pragma unroll
    for (int i = 0; i < 4; i++) {
        ALdsOffset[i] = (lane + slot / 2 * 0x1000 / sizeof(ALdsType) +
                         i * 0x400 / sizeof(ALdsType)) *
                        sizeof(ALdsType);
        BLdsOffset[i] = (lane + 0x2000 / sizeof(BLdsType) +
                         (slot & 1) * 0x1000 / sizeof(BLdsType) +
                         i * 0x400 / sizeof(BLdsType)) *
                        sizeof(BLdgType);
    }

    __shared__ uint8_t WSM[0x10000]; // 64KB

    FLOAT4 C_f32[4][4] = {}; // = {} means all zeros
    ALdsType a[4][4];
    BLdsType b[4][4];

    uint8_t *WSM_Ldg = WSM + slot * 0x400;

    for (int stage_i = 0; stage_i < Stage; ++stage_i) {
        __builtin_mxc_ldg_b128_bsm_predicator(
            WSM_Ldg + 0x4000 * stage_i + 0x0000, APtr + ALdgOffset[0][stage_i],
            0, true, true, false, true, 0, K / (sizeof(ALdgType) / sizeof(T)),
            MACA_ICMP_SLT);
        __builtin_mxc_ldg_b128_bsm_predicator(
            WSM_Ldg + 0x4000 * stage_i + 0x1000, APtr + ALdgOffset[1][stage_i],
            0, true, true, false, true, 0, K / (sizeof(ALdgType) / sizeof(T)),
            MACA_ICMP_SLT);
        __builtin_mxc_ldg_b128_bsm_predicator(
            WSM_Ldg + 0x4000 * stage_i + 0x2000, BPtr + BLdgOffset[0][stage_i],
            0, true, true, false, true, startCol + stage_i * 16, N,
            MACA_ICMP_SLT);
        __builtin_mxc_ldg_b128_bsm_predicator(
            WSM_Ldg + 0x4000 * stage_i + 0x3000, BPtr + BLdgOffset[1][stage_i],
            0, true, true, false, true, startCol + (4 + stage_i) * 16, N,
            MACA_ICMP_SLT);
    }

    APtr += (128 / 8) * 16 * sizeof(ALdgType);
    // BPtr += 16 * N * sizeof(BLdgType);
    // for Col major B
    BPtr += 128 * sizeof(T);
    K -= 128;

    arrive_gvmcnt(4 * (Stage - 1));
    __builtin_mxc_barrier_inst();

    uint8_t *WSM_lds = reinterpret_cast<uint8_t *>(&WSM[0]);

    a[0][0] = *reinterpret_cast<ALdsType *>(WSM_lds + ALdsOffset[0]);
    a[0][1] = *reinterpret_cast<ALdsType *>(WSM_lds + ALdsOffset[1]);
    a[0][2] = *reinterpret_cast<ALdsType *>(WSM_lds + ALdsOffset[2]);
    a[0][3] = *reinterpret_cast<ALdsType *>(WSM_lds + ALdsOffset[3]);

    b[0][0] = *reinterpret_cast<ALdsType *>(WSM_lds + BLdsOffset[0]);
    b[0][1] = *reinterpret_cast<ALdsType *>(WSM_lds + BLdsOffset[1]);
    b[0][2] = *reinterpret_cast<ALdsType *>(WSM_lds + BLdsOffset[2]);
    b[0][3] = *reinterpret_cast<ALdsType *>(WSM_lds + BLdsOffset[3]);

    arrive_gvmcnt(4 * (Stage - 2));
    __builtin_mxc_barrier_inst();

    a[1][0] = *reinterpret_cast<ALdsType *>(WSM_lds + ALdsOffset[0] + 0x4000);
    a[1][1] = *reinterpret_cast<ALdsType *>(WSM_lds + ALdsOffset[1] + 0x4000);
    a[1][2] = *reinterpret_cast<ALdsType *>(WSM_lds + ALdsOffset[2] + 0x4000);
    a[1][3] = *reinterpret_cast<ALdsType *>(WSM_lds + ALdsOffset[3] + 0x4000);

    b[1][0] = *reinterpret_cast<ALdsType *>(WSM_lds + BLdsOffset[0] + 0x4000);
    b[1][1] = *reinterpret_cast<ALdsType *>(WSM_lds + BLdsOffset[1] + 0x4000);
    b[1][2] = *reinterpret_cast<ALdsType *>(WSM_lds + BLdsOffset[2] + 0x4000);
    b[1][3] = *reinterpret_cast<ALdsType *>(WSM_lds + BLdsOffset[3] + 0x4000);

    for (; K >= 128; K -= 128) {
        {
            C_f32[0][0] = mma_16x16x16b16<T, true>(
                b[0][0][0], b[0][0][1], a[0][0][0], a[0][0][1], C_f32[0][0]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 0 + 0x0000,
                                       APtr + ALdgOffset[0][0])
            C_f32[0][0] = mma_16x16x16b16<T, true>(
                b[0][0][2], b[0][0][3], a[0][0][2], a[0][0][3], C_f32[0][0]);
            C_f32[0][0] = mma_16x16x16b16<T, true>(
                b[0][1][0], b[0][1][1], a[0][1][0], a[0][1][1], C_f32[0][0]);
            C_f32[0][0] = mma_16x16x16b16<T, true>(
                b[0][1][2], b[0][1][3], a[0][1][2], a[0][1][3], C_f32[0][0]);
            C_f32[0][0] = mma_16x16x16b16<T, true>(
                b[0][2][0], b[0][2][1], a[0][2][0], a[0][2][1], C_f32[0][0]);
            C_f32[0][0] = mma_16x16x16b16<T, true>(
                b[0][2][2], b[0][2][3], a[0][2][2], a[0][2][3], C_f32[0][0]);
            C_f32[0][0] = mma_16x16x16b16<T, true>(
                b[0][3][0], b[0][3][1], a[0][3][0], a[0][3][1], C_f32[0][0]);
            C_f32[0][0] = mma_16x16x16b16<T, true>(
                b[0][3][2], b[0][3][3], a[0][3][2], a[0][3][3], C_f32[0][0]);

            C_f32[1][0] = mma_16x16x16b16<T, true>(
                b[0][0][0], b[0][0][1], a[1][0][0], a[1][0][1], C_f32[1][0]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 0 + 0x1000,
                                       APtr + ALdgOffset[1][0])
            C_f32[1][0] = mma_16x16x16b16<T, true>(
                b[0][0][2], b[0][0][3], a[1][0][2], a[1][0][3], C_f32[1][0]);
            C_f32[1][0] = mma_16x16x16b16<T, true>(
                b[0][1][0], b[0][1][1], a[1][1][0], a[1][1][1], C_f32[1][0]);
            C_f32[1][0] = mma_16x16x16b16<T, true>(
                b[0][1][2], b[0][1][3], a[1][1][2], a[1][1][3], C_f32[1][0]);
            C_f32[1][0] = mma_16x16x16b16<T, true>(
                b[0][2][0], b[0][2][1], a[1][2][0], a[1][2][1], C_f32[1][0]);
            C_f32[1][0] = mma_16x16x16b16<T, true>(
                b[0][2][2], b[0][2][3], a[1][2][2], a[1][2][3], C_f32[1][0]);
            C_f32[1][0] = mma_16x16x16b16<T, true>(
                b[0][3][0], b[0][3][1], a[1][3][0], a[1][3][1], C_f32[1][0]);
            arrive_gvmcnt(4 * (Stage - 3) + 2);
            __builtin_mxc_barrier_inst();
            C_f32[1][0] = mma_16x16x16b16<T, true>(
                b[0][3][2], b[0][3][3], a[1][3][2], a[1][3][3], C_f32[1][0]);

            a[2][0] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0x8000 + ALdsOffset[0]);
            C_f32[0][1] = mma_16x16x16b16<T, true>(
                b[1][0][0], b[1][0][1], a[0][0][0], a[0][0][1], C_f32[0][1]);
            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * 0 + 0x2000, BPtr + BLdgOffset[0][0], 0, true,
                true, false, true, startCol + 0, N, MACA_ICMP_SLT);
            C_f32[0][1] = mma_16x16x16b16<T, true>(
                b[1][0][2], b[1][0][3], a[0][0][2], a[0][0][3], C_f32[0][1]);
            a[2][1] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0x8000 + ALdsOffset[1]);
            C_f32[0][1] = mma_16x16x16b16<T, true>(
                b[1][1][0], b[1][1][1], a[0][1][0], a[0][1][1], C_f32[0][1]);
            C_f32[0][1] = mma_16x16x16b16<T, true>(
                b[1][1][2], b[1][1][3], a[0][1][2], a[0][1][3], C_f32[0][1]);
            a[2][2] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0x8000 + ALdsOffset[2]);
            C_f32[0][1] = mma_16x16x16b16<T, true>(
                b[1][2][0], b[1][2][1], a[0][2][0], a[0][2][1], C_f32[0][1]);
            C_f32[0][1] = mma_16x16x16b16<T, true>(
                b[1][2][2], b[1][2][3], a[0][2][2], a[0][2][3], C_f32[0][1]);
            a[2][3] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0x8000 + ALdsOffset[3]);
            C_f32[0][1] = mma_16x16x16b16<T, true>(
                b[1][3][0], b[1][3][1], a[0][3][0], a[0][3][1], C_f32[0][1]);
            C_f32[0][1] = mma_16x16x16b16<T, true>(
                b[1][3][2], b[1][3][3], a[0][3][2], a[0][3][3], C_f32[0][1]);
            b[2][0] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0x8000 + BLdsOffset[0]);

            C_f32[1][1] = mma_16x16x16b16<T, true>(
                b[1][0][0], b[1][0][1], a[1][0][0], a[1][0][1], C_f32[1][1]);
            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * 0 + 0x3000, BPtr + BLdgOffset[1][0], 0, true,
                true, false, true, startCol + 64, N, MACA_ICMP_SLT);
            C_f32[1][1] = mma_16x16x16b16<T, true>(
                b[1][0][2], b[1][0][3], a[1][0][2], a[1][0][3], C_f32[1][1]);
            b[2][1] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0x8000 + BLdsOffset[1]);
            C_f32[1][1] = mma_16x16x16b16<T, true>(
                b[1][1][0], b[1][1][1], a[1][1][0], a[1][1][1], C_f32[1][1]);
            C_f32[1][1] = mma_16x16x16b16<T, true>(
                b[1][1][2], b[1][1][3], a[1][1][2], a[1][1][3], C_f32[1][1]);
            b[2][2] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0x8000 + BLdsOffset[2]);
            C_f32[1][1] = mma_16x16x16b16<T, true>(
                b[1][2][0], b[1][2][1], a[1][2][0], a[1][2][1], C_f32[1][1]);
            C_f32[1][1] = mma_16x16x16b16<T, true>(
                b[1][2][2], b[1][2][3], a[1][2][2], a[1][2][3], C_f32[1][1]);
            b[2][3] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0x8000 + BLdsOffset[3]);
            C_f32[1][1] = mma_16x16x16b16<T, true>(
                b[1][3][0], b[1][3][1], a[1][3][0], a[1][3][1], C_f32[1][1]);
            C_f32[1][1] = mma_16x16x16b16<T, true>(
                b[1][3][2], b[1][3][3], a[1][3][2], a[1][3][3], C_f32[1][1]);

            C_f32[2][0] = mma_16x16x16b16<T, true>(
                b[0][0][0], b[0][0][1], a[2][0][0], a[2][0][1], C_f32[2][0]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 1 + 0x0000,
                                       APtr + ALdgOffset[0][1])
            C_f32[2][0] = mma_16x16x16b16<T, true>(
                b[0][0][2], b[0][0][3], a[2][0][2], a[2][0][3], C_f32[2][0]);
            C_f32[2][0] = mma_16x16x16b16<T, true>(
                b[0][1][0], b[0][1][1], a[2][1][0], a[2][1][1], C_f32[2][0]);
            C_f32[2][0] = mma_16x16x16b16<T, true>(
                b[0][1][2], b[0][1][3], a[2][1][2], a[2][1][3], C_f32[2][0]);
            C_f32[2][0] = mma_16x16x16b16<T, true>(
                b[0][2][0], b[0][2][1], a[2][2][0], a[2][2][1], C_f32[2][0]);
            C_f32[2][0] = mma_16x16x16b16<T, true>(
                b[0][2][2], b[0][2][3], a[2][2][2], a[2][2][3], C_f32[2][0]);
            C_f32[2][0] = mma_16x16x16b16<T, true>(
                b[0][3][0], b[0][3][1], a[2][3][0], a[2][3][1], C_f32[2][0]);
            C_f32[2][0] = mma_16x16x16b16<T, true>(
                b[0][3][2], b[0][3][3], a[2][3][2], a[2][3][3], C_f32[2][0]);

            C_f32[2][1] = mma_16x16x16b16<T, true>(
                b[1][0][0], b[1][0][1], a[2][0][0], a[2][0][1], C_f32[2][1]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 1 + 0x1000,
                                       APtr + ALdgOffset[1][1])
            C_f32[2][1] = mma_16x16x16b16<T, true>(
                b[1][0][2], b[1][0][3], a[2][0][2], a[2][0][3], C_f32[2][1]);
            C_f32[2][1] = mma_16x16x16b16<T, true>(
                b[1][1][0], b[1][1][1], a[2][1][0], a[2][1][1], C_f32[2][1]);
            arrive_gvmcnt(4 * (Stage - 4) + 6);
            __builtin_mxc_barrier_inst();
            C_f32[2][1] = mma_16x16x16b16<T, true>(
                b[1][1][2], b[1][1][3], a[2][1][2], a[2][1][3], C_f32[2][1]);
            a[3][0] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0xC000 + ALdsOffset[0]);
            C_f32[2][1] = mma_16x16x16b16<T, true>(
                b[1][2][0], b[1][2][1], a[2][2][0], a[2][2][1], C_f32[2][1]);
            C_f32[2][1] = mma_16x16x16b16<T, true>(
                b[1][2][2], b[1][2][3], a[2][2][2], a[2][2][3], C_f32[2][1]);
            a[3][1] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0xC000 + ALdsOffset[1]);
            C_f32[2][1] = mma_16x16x16b16<T, true>(
                b[1][3][0], b[1][3][1], a[2][3][0], a[2][3][1], C_f32[2][1]);
            C_f32[2][1] = mma_16x16x16b16<T, true>(
                b[1][3][2], b[1][3][3], a[2][3][2], a[2][3][3], C_f32[2][1]);
            a[3][2] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0xC000 + ALdsOffset[2]);

            C_f32[0][2] = mma_16x16x16b16<T, true>(
                b[2][0][0], b[2][0][1], a[0][0][0], a[0][0][1], C_f32[0][2]);
            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * 1 + 0x2000, BPtr + BLdgOffset[0][1], 0, true,
                true, false, true, startCol + 16, N, MACA_ICMP_SLT);
            C_f32[0][2] = mma_16x16x16b16<T, true>(
                b[2][0][2], b[2][0][3], a[0][0][2], a[0][0][3], C_f32[0][2]);
            a[3][3] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0xC000 + ALdsOffset[3]);
            C_f32[0][2] = mma_16x16x16b16<T, true>(
                b[2][1][0], b[2][1][1], a[0][1][0], a[0][1][1], C_f32[0][2]);
            C_f32[0][2] = mma_16x16x16b16<T, true>(
                b[2][1][2], b[2][1][3], a[0][1][2], a[0][1][3], C_f32[0][2]);
            b[3][0] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0xC000 + BLdsOffset[0]);
            C_f32[0][2] = mma_16x16x16b16<T, true>(
                b[2][2][0], b[2][2][1], a[0][2][0], a[0][2][1], C_f32[0][2]);
            C_f32[0][2] = mma_16x16x16b16<T, true>(
                b[2][2][2], b[2][2][3], a[0][2][2], a[0][2][3], C_f32[0][2]);
            b[3][1] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0xC000 + BLdsOffset[1]);
            C_f32[0][2] = mma_16x16x16b16<T, true>(
                b[2][3][0], b[2][3][1], a[0][3][0], a[0][3][1], C_f32[0][2]);
            C_f32[0][2] = mma_16x16x16b16<T, true>(
                b[2][3][2], b[2][3][3], a[0][3][2], a[0][3][3], C_f32[0][2]);
            b[3][2] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0xC000 + BLdsOffset[2]);

            C_f32[1][2] = mma_16x16x16b16<T, true>(
                b[2][0][0], b[2][0][1], a[1][0][0], a[1][0][1], C_f32[1][2]);
            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * 1 + 0x3000, BPtr + BLdgOffset[1][1], 0, true,
                true, false, true, startCol + 80, N, MACA_ICMP_SLT);
            C_f32[1][2] = mma_16x16x16b16<T, true>(
                b[2][0][2], b[2][0][3], a[1][0][2], a[1][0][3], C_f32[1][2]);
            b[3][3] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0xC000 + BLdsOffset[3]);
            C_f32[1][2] = mma_16x16x16b16<T, true>(
                b[2][1][0], b[2][1][1], a[1][1][0], a[1][1][1], C_f32[1][2]);
            C_f32[1][2] = mma_16x16x16b16<T, true>(
                b[2][1][2], b[2][1][3], a[1][1][2], a[1][1][3], C_f32[1][2]);
            C_f32[1][2] = mma_16x16x16b16<T, true>(
                b[2][2][0], b[2][2][1], a[1][2][0], a[1][2][1], C_f32[1][2]);
            C_f32[1][2] = mma_16x16x16b16<T, true>(
                b[2][2][2], b[2][2][3], a[1][2][2], a[1][2][3], C_f32[1][2]);
            C_f32[1][2] = mma_16x16x16b16<T, true>(
                b[2][3][0], b[2][3][1], a[1][3][0], a[1][3][1], C_f32[1][2]);
            C_f32[1][2] = mma_16x16x16b16<T, true>(
                b[2][3][2], b[2][3][3], a[1][3][2], a[1][3][3], C_f32[1][2]);

            C_f32[2][2] = mma_16x16x16b16<T, true>(
                b[2][0][0], b[2][0][1], a[2][0][0], a[2][0][1], C_f32[2][2]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 2 + 0x0000,
                                       APtr + ALdgOffset[0][2])
            C_f32[2][2] = mma_16x16x16b16<T, true>(
                b[2][0][2], b[2][0][3], a[2][0][2], a[2][0][3], C_f32[2][2]);
            C_f32[2][2] = mma_16x16x16b16<T, true>(
                b[2][1][0], b[2][1][1], a[2][1][0], a[2][1][1], C_f32[2][2]);
            C_f32[2][2] = mma_16x16x16b16<T, true>(
                b[2][1][2], b[2][1][3], a[2][1][2], a[2][1][3], C_f32[2][2]);
            C_f32[2][2] = mma_16x16x16b16<T, true>(
                b[2][2][0], b[2][2][1], a[2][2][0], a[2][2][1], C_f32[2][2]);
            C_f32[2][2] = mma_16x16x16b16<T, true>(
                b[2][2][2], b[2][2][3], a[2][2][2], a[2][2][3], C_f32[2][2]);
            C_f32[2][2] = mma_16x16x16b16<T, true>(
                b[2][3][0], b[2][3][1], a[2][3][0], a[2][3][1], C_f32[2][2]);
            C_f32[2][2] = mma_16x16x16b16<T, true>(
                b[2][3][2], b[2][3][3], a[2][3][2], a[2][3][3], C_f32[2][2]);

            C_f32[3][0] = mma_16x16x16b16<T, true>(
                b[0][0][0], b[0][0][1], a[3][0][0], a[3][0][1], C_f32[3][0]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 2 + 0x1000,
                                       APtr + ALdgOffset[1][2])
            C_f32[3][0] = mma_16x16x16b16<T, true>(
                b[0][0][2], b[0][0][3], a[3][0][2], a[3][0][3], C_f32[3][0]);
            C_f32[3][0] = mma_16x16x16b16<T, true>(
                b[0][1][0], b[0][1][1], a[3][1][0], a[3][1][1], C_f32[3][0]);
            C_f32[3][0] = mma_16x16x16b16<T, true>(
                b[0][1][2], b[0][1][3], a[3][1][2], a[3][1][3], C_f32[3][0]);
            C_f32[3][0] = mma_16x16x16b16<T, true>(
                b[0][2][0], b[0][2][1], a[3][2][0], a[3][2][1], C_f32[3][0]);
            C_f32[3][0] = mma_16x16x16b16<T, true>(
                b[0][2][2], b[0][2][3], a[3][2][2], a[3][2][3], C_f32[3][0]);
            C_f32[3][0] = mma_16x16x16b16<T, true>(
                b[0][3][0], b[0][3][1], a[3][3][0], a[3][3][1], C_f32[3][0]);
            C_f32[3][0] = mma_16x16x16b16<T, true>(
                b[0][3][2], b[0][3][3], a[3][3][2], a[3][3][3], C_f32[3][0]);
            arrive_gvmcnt(4 * (Stage - 5) + 10);
            __builtin_mxc_barrier_inst();

            C_f32[0][3] = mma_16x16x16b16<T, true>(
                b[3][0][0], b[3][0][1], a[0][0][0], a[0][0][1], C_f32[0][3]);
            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * 2 + 0x2000, BPtr + BLdgOffset[0][2], 0, true,
                true, false, true, startCol + 32, N, MACA_ICMP_SLT);
            C_f32[0][3] = mma_16x16x16b16<T, true>(
                b[3][0][2], b[3][0][3], a[0][0][2], a[0][0][3], C_f32[0][3]);
            b[0][0] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0 + BLdsOffset[0]);
            C_f32[0][3] = mma_16x16x16b16<T, true>(
                b[3][1][0], b[3][1][1], a[0][1][0], a[0][1][1], C_f32[0][3]);
            C_f32[0][3] = mma_16x16x16b16<T, true>(
                b[3][1][2], b[3][1][3], a[0][1][2], a[0][1][3], C_f32[0][3]);
            b[0][1] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0 + BLdsOffset[1]);
            C_f32[0][3] = mma_16x16x16b16<T, true>(
                b[3][2][0], b[3][2][1], a[0][2][0], a[0][2][1], C_f32[0][3]);
            C_f32[0][3] = mma_16x16x16b16<T, true>(
                b[3][2][2], b[3][2][3], a[0][2][2], a[0][2][3], C_f32[0][3]);
            b[0][2] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0 + BLdsOffset[2]);
            C_f32[0][3] = mma_16x16x16b16<T, true>(
                b[3][3][0], b[3][3][1], a[0][3][0], a[0][3][1], C_f32[0][3]);
            C_f32[0][3] = mma_16x16x16b16<T, true>(
                b[3][3][2], b[3][3][3], a[0][3][2], a[0][3][3], C_f32[0][3]);
            b[0][3] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0 + BLdsOffset[3]);

            C_f32[3][1] = mma_16x16x16b16<T, true>(
                b[1][0][0], b[1][0][1], a[3][0][0], a[3][0][1], C_f32[3][1]);
            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * 2 + 0x3000, BPtr + BLdgOffset[1][2], 0, true,
                true, false, true, startCol + 96, N, MACA_ICMP_SLT);
            C_f32[3][1] = mma_16x16x16b16<T, true>(
                b[1][0][2], b[1][0][3], a[3][0][2], a[3][0][3], C_f32[3][1]);
            a[0][0] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0 + ALdsOffset[0]);
            C_f32[3][1] = mma_16x16x16b16<T, true>(
                b[1][1][0], b[1][1][1], a[3][1][0], a[3][1][1], C_f32[3][1]);
            C_f32[3][1] = mma_16x16x16b16<T, true>(
                b[1][1][2], b[1][1][3], a[3][1][2], a[3][1][3], C_f32[3][1]);
            a[0][1] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0 + ALdsOffset[1]);
            C_f32[3][1] = mma_16x16x16b16<T, true>(
                b[1][2][0], b[1][2][1], a[3][2][0], a[3][2][1], C_f32[3][1]);
            C_f32[3][1] = mma_16x16x16b16<T, true>(
                b[1][2][2], b[1][2][3], a[3][2][2], a[3][2][3], C_f32[3][1]);
            a[0][2] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0 + ALdsOffset[2]);
            C_f32[3][1] = mma_16x16x16b16<T, true>(
                b[1][3][0], b[1][3][1], a[3][3][0], a[3][3][1], C_f32[3][1]);
            C_f32[3][1] = mma_16x16x16b16<T, true>(
                b[1][3][2], b[1][3][3], a[3][3][2], a[3][3][3], C_f32[3][1]);
            a[0][3] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0 + ALdsOffset[3]);

            C_f32[1][3] = mma_16x16x16b16<T, true>(
                b[3][0][0], b[3][0][1], a[1][0][0], a[1][0][1], C_f32[1][3]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 3 + 0x0000,
                                       APtr + ALdgOffset[0][3])
            C_f32[1][3] = mma_16x16x16b16<T, true>(
                b[3][0][2], b[3][0][3], a[1][0][2], a[1][0][3], C_f32[1][3]);
            C_f32[1][3] = mma_16x16x16b16<T, true>(
                b[3][1][0], b[3][1][1], a[1][1][0], a[1][1][1], C_f32[1][3]);
            C_f32[1][3] = mma_16x16x16b16<T, true>(
                b[3][1][2], b[3][1][3], a[1][1][2], a[1][1][3], C_f32[1][3]);
            C_f32[1][3] = mma_16x16x16b16<T, true>(
                b[3][2][0], b[3][2][1], a[1][2][0], a[1][2][1], C_f32[1][3]);
            C_f32[1][3] = mma_16x16x16b16<T, true>(
                b[3][2][2], b[3][2][3], a[1][2][2], a[1][2][3], C_f32[1][3]);
            C_f32[1][3] = mma_16x16x16b16<T, true>(
                b[3][3][0], b[3][3][1], a[1][3][0], a[1][3][1], C_f32[1][3]);
            C_f32[1][3] = mma_16x16x16b16<T, true>(
                b[3][3][2], b[3][3][3], a[1][3][2], a[1][3][3], C_f32[1][3]);

            C_f32[3][2] = mma_16x16x16b16<T, true>(
                b[2][0][0], b[2][0][1], a[3][0][0], a[3][0][1], C_f32[3][2]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 3 + 0x1000,
                                       APtr + ALdgOffset[1][3])
            C_f32[3][2] = mma_16x16x16b16<T, true>(
                b[2][0][2], b[2][0][3], a[3][0][2], a[3][0][3], C_f32[3][2]);
            C_f32[3][2] = mma_16x16x16b16<T, true>(
                b[2][1][0], b[2][1][1], a[3][1][0], a[3][1][1], C_f32[3][2]);
            C_f32[3][2] = mma_16x16x16b16<T, true>(
                b[2][1][2], b[2][1][3], a[3][1][2], a[3][1][3], C_f32[3][2]);
            C_f32[3][2] = mma_16x16x16b16<T, true>(
                b[2][2][0], b[2][2][1], a[3][2][0], a[3][2][1], C_f32[3][2]);
            arrive_gvmcnt(4 * (Stage - 6) + 14);
            __builtin_mxc_barrier_inst();
            C_f32[3][2] = mma_16x16x16b16<T, true>(
                b[2][2][2], b[2][2][3], a[3][2][2], a[3][2][3], C_f32[3][2]);
            a[1][0] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0x4000 + ALdsOffset[0]);
            C_f32[3][2] = mma_16x16x16b16<T, true>(
                b[2][3][0], b[2][3][1], a[3][3][0], a[3][3][1], C_f32[3][2]);
            C_f32[3][2] = mma_16x16x16b16<T, true>(
                b[2][3][2], b[2][3][3], a[3][3][2], a[3][3][3], C_f32[3][2]);
            a[1][1] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0x4000 + ALdsOffset[1]);

            C_f32[2][3] = mma_16x16x16b16<T, true>(
                b[3][0][0], b[3][0][1], a[2][0][0], a[2][0][1], C_f32[2][3]);
            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * 3 + 0x2000, BPtr + BLdgOffset[0][3], 0, true,
                true, false, true, startCol + 48, N, MACA_ICMP_SLT);
            C_f32[2][3] = mma_16x16x16b16<T, true>(
                b[3][0][2], b[3][0][3], a[2][0][2], a[2][0][3], C_f32[2][3]);
            a[1][2] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0x4000 + ALdsOffset[2]);
            C_f32[2][3] = mma_16x16x16b16<T, true>(
                b[3][1][0], b[3][1][1], a[2][1][0], a[2][1][1], C_f32[2][3]);
            C_f32[2][3] = mma_16x16x16b16<T, true>(
                b[3][1][2], b[3][1][3], a[2][1][2], a[2][1][3], C_f32[2][3]);
            a[1][3] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0x4000 + ALdsOffset[3]);
            C_f32[2][3] = mma_16x16x16b16<T, true>(
                b[3][2][0], b[3][2][1], a[2][2][0], a[2][2][1], C_f32[2][3]);
            C_f32[2][3] = mma_16x16x16b16<T, true>(
                b[3][2][2], b[3][2][3], a[2][2][2], a[2][2][3], C_f32[2][3]);
            b[1][0] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0x4000 + BLdsOffset[0]);
            C_f32[2][3] = mma_16x16x16b16<T, true>(
                b[3][3][0], b[3][3][1], a[2][3][0], a[2][3][1], C_f32[2][3]);
            C_f32[2][3] = mma_16x16x16b16<T, true>(
                b[3][3][2], b[3][3][3], a[2][3][2], a[2][3][3], C_f32[2][3]);
            b[1][1] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0x4000 + BLdsOffset[1]);

            C_f32[3][3] = mma_16x16x16b16<T, true>(
                b[3][0][0], b[3][0][1], a[3][0][0], a[3][0][1], C_f32[3][3]);
            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * 3 + 0x3000, BPtr + BLdgOffset[1][3], 0, true,
                true, false, true, startCol + 112, N, MACA_ICMP_SLT);
            C_f32[3][3] = mma_16x16x16b16<T, true>(
                b[3][0][2], b[3][0][3], a[3][0][2], a[3][0][3], C_f32[3][3]);
            b[1][2] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0x4000 + BLdsOffset[2]);
            C_f32[3][3] = mma_16x16x16b16<T, true>(
                b[3][1][0], b[3][1][1], a[3][1][0], a[3][1][1], C_f32[3][3]);
            C_f32[3][3] = mma_16x16x16b16<T, true>(
                b[3][1][2], b[3][1][3], a[3][1][2], a[3][1][3], C_f32[3][3]);
            b[1][3] =
                *reinterpret_cast<ALdsType *>(WSM_lds + 0x4000 + BLdsOffset[3]);
            C_f32[3][3] = mma_16x16x16b16<T, true>(
                b[3][2][0], b[3][2][1], a[3][2][0], a[3][2][1], C_f32[3][3]);
            C_f32[3][3] = mma_16x16x16b16<T, true>(
                b[3][2][2], b[3][2][3], a[3][2][2], a[3][2][3], C_f32[3][3]);
            C_f32[3][3] = mma_16x16x16b16<T, true>(
                b[3][3][0], b[3][3][1], a[3][3][0], a[3][3][1], C_f32[3][3]);
            C_f32[3][3] = mma_16x16x16b16<T, true>(
                b[3][3][2], b[3][3][3], a[3][3][2], a[3][3][3], C_f32[3][3]);
        }

        APtr += (128 / 8) * 16 * sizeof(ALdgType);
        // BPtr += 16 * N * sizeof(BLdgType);
        // for Col major B
        BPtr += 128 * sizeof(T);
    }

    if (K > 0) {
        for (int stage_i = 0; stage_i < Stage; ++stage_i) {
            const int ldsIdx = (stage_i + 1) % Stage;
            uint8_t *WSM_lds2 = WSM_lds + (0x4000 * ldsIdx);

            for (int i = 0; i < stage_i; ++i) {
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T, true>(
                    b[i][0][0], b[i][0][1], a[stage_i][0][0], a[stage_i][0][1],
                    C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T, true>(
                    b[i][0][2], b[i][0][3], a[stage_i][0][2], a[stage_i][0][3],
                    C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T, true>(
                    b[i][1][0], b[i][1][1], a[stage_i][1][0], a[stage_i][1][1],
                    C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T, true>(
                    b[i][1][2], b[i][1][3], a[stage_i][1][2], a[stage_i][1][3],
                    C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T, true>(
                    b[i][2][0], b[i][2][1], a[stage_i][2][0], a[stage_i][2][1],
                    C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T, true>(
                    b[i][2][2], b[i][2][3], a[stage_i][2][2], a[stage_i][2][3],
                    C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T, true>(
                    b[i][3][0], b[i][3][1], a[stage_i][3][0], a[stage_i][3][1],
                    C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T, true>(
                    b[i][3][2], b[i][3][3], a[stage_i][3][2], a[stage_i][3][3],
                    C_f32[stage_i][i + 0]);
            }
            for (int i = 0; i <= stage_i; ++i) {
                C_f32[i][stage_i + 0] = mma_16x16x16b16<T, true>(
                    b[stage_i][0][0], b[stage_i][0][1], a[i][0][0], a[i][0][1],
                    C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] = mma_16x16x16b16<T, true>(
                    b[stage_i][0][2], b[stage_i][0][3], a[i][0][2], a[i][0][3],
                    C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] = mma_16x16x16b16<T, true>(
                    b[stage_i][1][0], b[stage_i][1][1], a[i][1][0], a[i][1][1],
                    C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] = mma_16x16x16b16<T, true>(
                    b[stage_i][1][2], b[stage_i][1][3], a[i][1][2], a[i][1][3],
                    C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] = mma_16x16x16b16<T, true>(
                    b[stage_i][2][0], b[stage_i][2][1], a[i][2][0], a[i][2][1],
                    C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] = mma_16x16x16b16<T, true>(
                    b[stage_i][2][2], b[stage_i][2][3], a[i][2][2], a[i][2][3],
                    C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] = mma_16x16x16b16<T, true>(
                    b[stage_i][3][0], b[stage_i][3][1], a[i][3][0], a[i][3][1],
                    C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] = mma_16x16x16b16<T, true>(
                    b[stage_i][3][2], b[stage_i][3][3], a[i][3][2], a[i][3][3],
                    C_f32[i][stage_i + 0]);
            }

            arrive_gvmcnt(4 * (Stage - 2));
            __builtin_mxc_barrier_inst();

            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * stage_i + 0x0000,
                APtr + ALdgOffset[0][stage_i], 0, true, true, false, true, 0,
                K / (sizeof(ALdgType) / sizeof(T)), MACA_ICMP_SLT);
            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * stage_i + 0x1000,
                APtr + ALdgOffset[1][stage_i], 0, true, true, false, true, 0,
                K / (sizeof(ALdgType) / sizeof(T)), MACA_ICMP_SLT);
            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * stage_i + 0x2000,
                BPtr + BLdgOffset[0][stage_i], 0, true, true, false, true,
                startCol + stage_i * 16, N, MACA_ICMP_SLT);
            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * stage_i + 0x3000,
                BPtr + BLdgOffset[1][stage_i], 0, true, true, false, true,
                startCol + stage_i * 16 + 64, N, MACA_ICMP_SLT);

            a[ldsIdx][0] =
                *reinterpret_cast<ALdsType *>(WSM_lds2 + ALdsOffset[0]);
            a[ldsIdx][1] =
                *reinterpret_cast<ALdsType *>(WSM_lds2 + ALdsOffset[1]);
            a[ldsIdx][2] =
                *reinterpret_cast<ALdsType *>(WSM_lds2 + ALdsOffset[2]);
            a[ldsIdx][3] =
                *reinterpret_cast<ALdsType *>(WSM_lds2 + ALdsOffset[3]);
            b[ldsIdx][0] =
                *reinterpret_cast<ALdsType *>(WSM_lds2 + BLdsOffset[0]);
            b[ldsIdx][1] =
                *reinterpret_cast<ALdsType *>(WSM_lds2 + BLdsOffset[1]);
            b[ldsIdx][2] =
                *reinterpret_cast<ALdsType *>(WSM_lds2 + BLdsOffset[2]);
            b[ldsIdx][3] =
                *reinterpret_cast<ALdsType *>(WSM_lds2 + BLdsOffset[3]);
        }
    }

    {
#pragma unroll
        for (int stage_i = 0; stage_i < Stage; ++stage_i) {
            const int ldsIdx = (stage_i + 1) % Stage;
            uint8_t *WSM_lds2 = WSM_lds + (0x4000 * ldsIdx);

            for (int i = 0; i < stage_i; ++i) {
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T, true>(
                    b[i][0][0], b[i][0][1], a[stage_i][0][0], a[stage_i][0][1],
                    C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T, true>(
                    b[i][0][2], b[i][0][3], a[stage_i][0][2], a[stage_i][0][3],
                    C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T, true>(
                    b[i][1][0], b[i][1][1], a[stage_i][1][0], a[stage_i][1][1],
                    C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T, true>(
                    b[i][1][2], b[i][1][3], a[stage_i][1][2], a[stage_i][1][3],
                    C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T, true>(
                    b[i][2][0], b[i][2][1], a[stage_i][2][0], a[stage_i][2][1],
                    C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T, true>(
                    b[i][2][2], b[i][2][3], a[stage_i][2][2], a[stage_i][2][3],
                    C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T, true>(
                    b[i][3][0], b[i][3][1], a[stage_i][3][0], a[stage_i][3][1],
                    C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T, true>(
                    b[i][3][2], b[i][3][3], a[stage_i][3][2], a[stage_i][3][3],
                    C_f32[stage_i][i + 0]);
            }
            for (int i = 0; i <= stage_i; ++i) {
                C_f32[i][stage_i + 0] = mma_16x16x16b16<T, true>(
                    b[stage_i][0][0], b[stage_i][0][1], a[i][0][0], a[i][0][1],
                    C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] = mma_16x16x16b16<T, true>(
                    b[stage_i][0][2], b[stage_i][0][3], a[i][0][2], a[i][0][3],
                    C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] = mma_16x16x16b16<T, true>(
                    b[stage_i][1][0], b[stage_i][1][1], a[i][1][0], a[i][1][1],
                    C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] = mma_16x16x16b16<T, true>(
                    b[stage_i][1][2], b[stage_i][1][3], a[i][1][2], a[i][1][3],
                    C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] = mma_16x16x16b16<T, true>(
                    b[stage_i][2][0], b[stage_i][2][1], a[i][2][0], a[i][2][1],
                    C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] = mma_16x16x16b16<T, true>(
                    b[stage_i][2][2], b[stage_i][2][3], a[i][2][2], a[i][2][3],
                    C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] = mma_16x16x16b16<T, true>(
                    b[stage_i][3][0], b[stage_i][3][1], a[i][3][0], a[i][3][1],
                    C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] = mma_16x16x16b16<T, true>(
                    b[stage_i][3][2], b[stage_i][3][3], a[i][3][2], a[i][3][3],
                    C_f32[i][stage_i + 0]);
            }

            if (stage_i == 0) {
                arrive_gvmcnt(4 * (Stage - 2 - 0));
                __builtin_mxc_barrier_inst();
            } else if (stage_i == 1) {
                arrive_gvmcnt(4 * (Stage - 2 - 1));
                __builtin_mxc_barrier_inst();
            } else if (stage_i == 2) {
                arrive_gvmcnt(4 * (Stage - 2 - 2));
                __builtin_mxc_barrier_inst();
            }

            if (stage_i < Stage - 1) {
                a[ldsIdx][0] =
                    *reinterpret_cast<ALdsType *>(WSM_lds2 + ALdsOffset[0]);
                a[ldsIdx][1] =
                    *reinterpret_cast<ALdsType *>(WSM_lds2 + ALdsOffset[1]);
                a[ldsIdx][2] =
                    *reinterpret_cast<ALdsType *>(WSM_lds2 + ALdsOffset[2]);
                a[ldsIdx][3] =
                    *reinterpret_cast<ALdsType *>(WSM_lds2 + ALdsOffset[3]);
                b[ldsIdx][0] =
                    *reinterpret_cast<ALdsType *>(WSM_lds2 + BLdsOffset[0]);
                b[ldsIdx][1] =
                    *reinterpret_cast<ALdsType *>(WSM_lds2 + BLdsOffset[1]);
                b[ldsIdx][2] =
                    *reinterpret_cast<ALdsType *>(WSM_lds2 + BLdsOffset[2]);
                b[ldsIdx][3] =
                    *reinterpret_cast<ALdsType *>(WSM_lds2 + BLdsOffset[3]);
            }
        }
    }

    CStgType bias_load[4];
    if constexpr (HasOneDimBias) {
        for (int i = 0; i < 4; i++) {
            int bias_offset =
                startRow / 16 * 4 + (lane / 16) + slot / 2 * 4 * 4 + i * 4;
            bias_load[i] =
                (reinterpret_cast<const CStgType *>(bias))[bias_offset];
        }
    }

    CStgType *C_ptr = reinterpret_cast<CStgType *>(C);
    size_t C_row_offset = (size_t)(lane & 15) * (M / 4) + startRow / 16 * 4 +
                          (lane / 16) + slot / 2 * 4 * 4;
    size_t C_col_offset = (size_t)(startCol + (slot & 1) * 64) * M / 4;

#pragma unroll
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            size_t C_offset =
                C_row_offset + C_col_offset + i * 4 + j * (16 * M / 4);
            if ((startCol + (slot & 1) * 64 + j * 16) < N) {
                float C_f32_res[4];
#pragma unroll 4
                for (int t = 0; t < 4; t++) {
                    C_f32_res[t] = C_f32[i][j][t] * alpha;
                }
                if constexpr (!IsBetaZero) {
                    CStgType C_tmp = C_ptr[C_offset];
                    Tc *C_tmp_ptr = reinterpret_cast<Tc *>(&C_tmp);
                    C_f32_res[0] += beta * static_cast<Tscal>(C_tmp_ptr[0]);
                    C_f32_res[1] += beta * static_cast<Tscal>(C_tmp_ptr[1]);
                    C_f32_res[2] += beta * static_cast<Tscal>(C_tmp_ptr[2]);
                    C_f32_res[3] += beta * static_cast<Tscal>(C_tmp_ptr[3]);
                }
                Tc C_tc_tmp[4] = {0};
#pragma unroll 4
                for (int t = 0; t < 4; t++) {
                    C_tc_tmp[t] = static_cast<Tc>(C_f32_res[t]);
                }
                if constexpr (HasOneDimBias) {
                    Tc *bias_tc = reinterpret_cast<Tc *>(&bias_load[i]);
                    for (int t = 0; t < 4; t++) {
                        C_tc_tmp[t] = __hadd(C_tc_tmp[t], bias_tc[t]);
                    }
                }

                C_ptr[C_offset] = *reinterpret_cast<CStgType *>(&C_tc_tmp[0]);
            }
        }
    }
}

template <typename T, typename Tc, typename Tscal, bool IsBetaZero,
          bool HasOneDimBias>
__global__ void layoutA_hgemm_tn_128x128x128_4m1n8k_256t(
    const void *A, const void *B, void *C, int M, int N, int K, int lda,
    int ldb, int ldc, Tscal alpha, Tscal beta, const void *bias = nullptr) {
    layoutA_hgemm_tn_128x128x128_4m1n8k_256t_device<T, Tc, Tscal, IsBetaZero,
                                                    HasOneDimBias>(
        A, B, C, M, N, K, lda, ldb, ldc, alpha, beta, bias, blockIdx.x,
        blockIdx.y);
}

} // namespace muxi_layout_kernels
