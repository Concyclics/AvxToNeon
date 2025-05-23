#ifndef AVX2NEON_H
#error Never use <smmintrin.h> directly; include " avx2neon.h" instead.
#endif

#include <arm_neon.h>
#include "typedefs.h"

FORCE_INLINE __m128i _mm_blend_epi16(__m128i a, __m128i b, const int imm8) {
    uint16x8_t va = a.vect_u16;
    uint16x8_t vb = b.vect_u16;
    uint16x8_t result;

    for (int i = 0; i < 8; ++i) {
        result = vsetq_lane_u16((imm8 & (1 << i)) ? vgetq_lane_u16(vb, i) : vgetq_lane_u16(va, i), result, i);
    }

    __m128i out;
    out.vect_u16 = result;
    return out;
}
