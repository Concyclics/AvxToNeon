#ifndef AVX2NEON_H
#error Never use <wmmintrin.h> directly; include " avx2neon.h" instead.
#endif

#include <arm_neon.h>
#include "typedefs.h"

FORCE_INLINE __m128i _mm_clmulepi64_si128(__m128i a, __m128i b, const int imm8)
{
    uint64_t a64 = (imm8 & 0x01) ? vgetq_lane_u64(a.vect_u64, 1) : vgetq_lane_u64(a.vect_u64, 0);
    uint64_t b64 = (imm8 & 0x10) ? vgetq_lane_u64(b.vect_u64, 1) : vgetq_lane_u64(b.vect_u64, 0);

    poly64_t a_poly = (poly64_t)a64;
    poly64_t b_poly = (poly64_t)b64;

    poly128_t result = vmull_p64(a_poly, b_poly); // carry-less multiplication

    __m128i out;
    out.vect_u64 = vreinterpretq_u64_p128(result);
    return out;
}

FORCE_INLINE __m128i _mm_aesenc_si128(__m128i a, __m128i RoundKey) {
    __m128i out;
    out.vect_u8 = vaeseq_u8(a.vect_u8, RoundKey.vect_u8);  // SubBytes + ShiftRows
    out.vect_u8 = vaesmcq_u8(out.vect_u8);                 // MixColumns
    out.vect_u8 = veorq_u8(out.vect_u8, RoundKey.vect_u8); // AddRoundKey
    return out;
}
