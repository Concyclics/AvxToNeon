/*
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.

 */

#ifndef AVX2NEON_H
#error Never use <immintrin.h> directly; include " avx2neon.h" instead.
#endif

#include <arm_neon.h>
#include "typedefs.h"

# define RROTATE(a,n)     (((a)<<(n))|(((a)&0xffffffff)>>(32-(n))))
# define sigma_0(x)       (RROTATE((x),25) ^ RROTATE((x),14) ^ ((x)>>3))
# define sigma_1(x)       (RROTATE((x),15) ^ RROTATE((x),13) ^ ((x)>>10))
# define Sigma_0(x)       (RROTATE((x),30) ^ RROTATE((x),19) ^ RROTATE((x),10))
# define Sigma_1(x)       (RROTATE((x),26) ^ RROTATE((x),21) ^ RROTATE((x),7))

# define Ch(x,y,z)       (((x) & (y)) ^ ((~(x)) & (z)))
# define Maj(x,y,z)      (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

FORCE_INLINE __m128i _mm_sha256rnds2_epu32(__m128i a, __m128i b, __m128i k)
{
    __m128i res;
    uint32_t A[3];
    uint32_t B[3];
    uint32_t C[3];
    uint32_t D[3];
    uint32_t E[3];
    uint32_t F[3];
    uint32_t G[3];
    uint32_t H[3];
    uint32_t K[2];

    A[0] = vgetq_lane_u32(b.vect_u32, 3);
    B[0] = vgetq_lane_u32(b.vect_u32, 2);
    C[0] = vgetq_lane_u32(a.vect_u32, 3);
    D[0] = vgetq_lane_u32(a.vect_u32, 2);
    E[0] = vgetq_lane_u32(b.vect_u32, 1);
    F[0] = vgetq_lane_u32(b.vect_u32, 0);
    G[0] = vgetq_lane_u32(a.vect_u32, 1);
    H[0] = vgetq_lane_u32(a.vect_u32, 0);

    K[0] = vgetq_lane_u32(k.vect_u32, 0);
    K[1] = vgetq_lane_u32(k.vect_u32, 1);

    for (int i = 0; i < 2; i ++) {
        uint32_t T0 = Ch(E[i], F[i], G[i]) ;
        uint32_t T1 = Sigma_1(E[i]) + K[i] + H[i];
        uint32_t T2 = Maj(A[i], B[i], C[i]);

        A[i + 1] = T0 + T1 + T2 + Sigma_0(A[i]);
        B[i + 1] = A[i];
        C[i + 1] = B[i];
        D[i + 1] = C[i];
        E[i + 1] = T0 + T1 + D[i];
        F[i + 1] = E[i];
        G[i + 1] = F[i];
        H[i + 1] = G[i];
    }

    res.vect_u32 = vsetq_lane_u32(F[2], res.vect_u32, 0);
    res.vect_u32 = vsetq_lane_u32(E[2], res.vect_u32, 1);
    res.vect_u32 = vsetq_lane_u32(B[2], res.vect_u32, 2);
    res.vect_u32 = vsetq_lane_u32(A[2], res.vect_u32, 3);

    return res;
}

FORCE_INLINE __m128i _mm_sha256msg1_epu32(__m128i a, __m128i b)
{
    __asm__ __volatile__(
        "sha256su0 %[dst].4S, %[src].4S  \n\t"
        : [dst] "+w" (a)
        : [src] "w" (b)
    );
    return a;
}

FORCE_INLINE __m128i _mm_sha256msg2_epu32(__m128i a, __m128i b)
{
    __m128i res;
    uint32_t A = vgetq_lane_u32(b.vect_u32, 2);
    uint32_t B = vgetq_lane_u32(b.vect_u32, 3);

    uint32_t C = vgetq_lane_u32(a.vect_u32, 0) + sigma_1(A);
    uint32_t D = vgetq_lane_u32(a.vect_u32, 1) + sigma_1(B);
    uint32_t E = vgetq_lane_u32(a.vect_u32, 2) + sigma_1(C);
    uint32_t F = vgetq_lane_u32(a.vect_u32, 3) + sigma_1(D);

    res.vect_u32 = vsetq_lane_u32(C, res.vect_u32, 0);
    res.vect_u32 = vsetq_lane_u32(D, res.vect_u32, 1);
    res.vect_u32 = vsetq_lane_u32(E, res.vect_u32, 2);
    res.vect_u32 = vsetq_lane_u32(F, res.vect_u32, 3);

    return res;
}

FORCE_INLINE __m256i _mm256_setr_epi16(
    short e15, short e14, short e13, short e12,
    short e11, short e10, short e9,  short e8,
    short e7,  short e6,  short e5,  short e4,
    short e3,  short e2,  short e1,  short e0) {
    __m256i out;
    out.vect_s16[0] = (int16x8_t){e15, e14, e13, e12, e11, e10, e9, e8};
    out.vect_s16[1] = (int16x8_t){e7, e6, e5, e4, e3, e2, e1, e0};
    return out;
}

FORCE_INLINE __m256i _mm256_aesenc_epi128(__m256i a, __m256i rk) {
    __m256i out;
    out.vect_u8[0] = vaeseq_u8(a.vect_u8[0], rk.vect_u8[0]);
    out.vect_u8[0] = vaesmcq_u8(out.vect_u8[0]);
    out.vect_u8[0] = veorq_u8(out.vect_u8[0], rk.vect_u8[0]);

    out.vect_u8[1] = vaeseq_u8(a.vect_u8[1], rk.vect_u8[1]);
    out.vect_u8[1] = vaesmcq_u8(out.vect_u8[1]);
    out.vect_u8[1] = veorq_u8(out.vect_u8[1], rk.vect_u8[1]);

    return out;
}

FORCE_INLINE __m256i _mm256_slli_epi16(__m256i a, int imm8) {
    __m256i out;
    if (imm8 > 15) {
        out.vect_s16[0] = vdupq_n_s16(0);
        out.vect_s16[1] = vdupq_n_s16(0);
    } else {
        out.vect_s16[0] = vshlq_n_s16(a.vect_s16[0], imm8);
        out.vect_s16[1] = vshlq_n_s16(a.vect_s16[1], imm8);
    }
    return out;
}

FORCE_INLINE __m256i _mm256_srai_epi16(__m256i a, int imm8) {
    __m256i out;
    if (imm8 > 15) {
        out.vect_s16[0] = vshrq_n_s16(a.vect_s16[0], 15);
        out.vect_s16[0] = vsubq_s16(vdupq_n_s16(0), out.vect_s16[0]);

        out.vect_s16[1] = vshrq_n_s16(a.vect_s16[1], 15);
        out.vect_s16[1] = vsubq_s16(vdupq_n_s16(0), out.vect_s16[1]);
    } else {
        out.vect_s16[0] = vshrq_n_s16(a.vect_s16[0], imm8);
        out.vect_s16[1] = vshrq_n_s16(a.vect_s16[1], imm8);
    }
    return out;
}

FORCE_INLINE __m256i _mm256_permutexvar_epi8(__m256i idx, __m256i a) {
    __m256i out;
    uint8x16_t tbl0 = a.vect_u8[0];
    uint8x16_t tbl1 = a.vect_u8[1];

    uint8x16_t idx0 = idx.vect_u8[0];
    uint8x16_t idx1 = idx.vect_u8[1];

    uint8x16_t mask = vdupq_n_u8(0x1F); // restrict index within 0–31
    idx0 = vandq_u8(idx0, mask);
    idx1 = vandq_u8(idx1, mask);

    uint8x16x2_t tbl = { tbl0, tbl1 };

    out.vect_u8[0] = vqtbl2q_u8(tbl, idx0);
    out.vect_u8[1] = vqtbl2q_u8(tbl, idx1);

    return out;
}

FORCE_INLINE __m256i _mm256_permute2x128_si256(__m256i a, __m256i b, const int imm8) {
    __m256i out;

    int ctrl_lo = imm8 & 0x0F;
    int ctrl_hi = (imm8 >> 4) & 0x0F;

    // Low 128 bits
    if (ctrl_lo & 0x08) {
        out.vect_u64[0] = vdupq_n_u64(0);
    } else {
        switch (ctrl_lo & 0x03) {
            case 0: out.vect_u64[0] = a.vect_u64[0]; break;
            case 1: out.vect_u64[0] = a.vect_u64[1]; break;
            case 2: out.vect_u64[0] = b.vect_u64[0]; break;
            case 3: out.vect_u64[0] = b.vect_u64[1]; break;
        }
    }

    // High 128 bits
    if (ctrl_hi & 0x08) {
        out.vect_u64[1] = vdupq_n_u64(0);
    } else {
        switch (ctrl_hi & 0x03) {
            case 0: out.vect_u64[1] = a.vect_u64[0]; break;
            case 1: out.vect_u64[1] = a.vect_u64[1]; break;
            case 2: out.vect_u64[1] = b.vect_u64[0]; break;
            case 3: out.vect_u64[1] = b.vect_u64[1]; break;
        }
    }

    return out;
}

FORCE_INLINE __m256i _mm256_srli_epi16(__m256i a, int imm8) {
    __m256i out;
    if (imm8 > 15) {
        out.vect_u16[0] = vdupq_n_u16(0);
        out.vect_u16[1] = vdupq_n_u16(0);
    } else {
        out.vect_u16[0] = vshrq_n_u16(a.vect_u16[0], imm8);
        out.vect_u16[1] = vshrq_n_u16(a.vect_u16[1], imm8);
    }
    return out;
}

FORCE_INLINE __m256i _mm256_blendv_epi8(__m256i a, __m256i b, __m256i mask) {
    __m256i out;
    out.vect_u8[0] = vbslq_u8(mask.vect_u8[0], b.vect_u8[0], a.vect_u8[0]);
    out.vect_u8[1] = vbslq_u8(mask.vect_u8[1], b.vect_u8[1], a.vect_u8[1]);
    return out;
}

FORCE_INLINE __m256i _mm256_blend_epi32(__m256i a, __m256i b, const int imm8) {
    __m256i out;
    uint32x4_t va0 = a.vect_u32[0];
    uint32x4_t vb0 = b.vect_u32[0];
    uint32x4_t vo0;

    for (int i = 0; i < 4; i++) {
        vo0 = vsetq_lane_u32((imm8 & (1 << i)) ? vgetq_lane_u32(vb0, i) : vgetq_lane_u32(va0, i), vo0, i);
    }

    uint32x4_t va1 = a.vect_u32[1];
    uint32x4_t vb1 = b.vect_u32[1];
    uint32x4_t vo1;

    for (int i = 0; i < 4; i++) {
        vo1 = vsetq_lane_u32((imm8 & (1 << (i + 4))) ? vgetq_lane_u32(vb1, i) : vgetq_lane_u32(va1, i), vo1, i);
    }

    out.vect_u32[0] = vo0;
    out.vect_u32[1] = vo1;
    return out;
}

FORCE_INLINE __m256i _mm256_setr_m128i(__m128i lo, __m128i hi) {
    __m256i out;
    out.vect_u64[0] = lo.vect_u64;
    out.vect_u64[1] = hi.vect_u64;
    return out;
}

FORCE_INLINE __m256i _mm256_set1_epi16(short a) {
    __m256i out;
    out.vect_s16[0] = vdupq_n_s16(a);
    out.vect_s16[1] = vdupq_n_s16(a);
    return out;
}

FORCE_INLINE __m256i _mm256_sign_epi16(__m256i a, __m256i b) {
    __m256i out;
    int16x8_t a0 = a.vect_s16[0];
    int16x8_t a1 = a.vect_s16[1];
    int16x8_t b0 = b.vect_s16[0];
    int16x8_t b1 = b.vect_s16[1];

    // result[i] = (b[i] < 0) ? -a[i] : (b[i] == 0 ? 0 : a[i])
    out.vect_s16[0] = vbslq_s16(
        vcltq_s16(b0, vdupq_n_s16(0)),           // if b < 0
        vnegq_s16(a0),                          // then -a
        vbslq_s16(
            vceqq_s16(b0, vdupq_n_s16(0)),      // else if b == 0
            vdupq_n_s16(0),                     // then 0
            a0                                   // else a
        )
    );

    out.vect_s16[1] = vbslq_s16(
        vcltq_s16(b1, vdupq_n_s16(0)),
        vnegq_s16(a1),
        vbslq_s16(
            vceqq_s16(b1, vdupq_n_s16(0)),
            vdupq_n_s16(0),
            a1
        )
    );

    return out;
}

FORCE_INLINE __m256i _mm256_lddqu_si256(const __m256i* mem_addr) {
    __m256i out;
    uint8x16_t lo = vld1q_u8(((const uint8_t*)mem_addr) + 0);
    uint8x16_t hi = vld1q_u8(((const uint8_t*)mem_addr) + 16);
    out.vect_u8[0] = lo;
    out.vect_u8[1] = hi;
    return out;
}

FORCE_INLINE __m256i _mm256_zextsi128_si256(__m128i a) {
    __m256i out;
    out.vect_u64[0] = a.vect_u64;
    out.vect_u64[1] = vdupq_n_u64(0);
    return out;
}

FORCE_INLINE __m256i _mm256_insertf128_si256(__m256i a, __m128i b, int imm8) {
    __m256i out = a;
    if ((imm8 & 0x1) == 0) {
        out.vect_u64[0] = b.vect_u64;
    } else {
        out.vect_u64[1] = b.vect_u64;
    }
    return out;
}
