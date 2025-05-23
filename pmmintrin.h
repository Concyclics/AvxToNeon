#ifndef AVX2NEON_H
#error Never use <pmmintrin.h> directly; include " avx2neon.h" instead.
#endif

#include <arm_neon.h>
#include "typedefs.h"

FORCE_INLINE __m128i _mm_lddqu_si128(const __m128i* mem_addr) {
    __m128i out;
    out.vect_u8 = vld1q_u8((const uint8_t*)mem_addr);
    return out;
}
