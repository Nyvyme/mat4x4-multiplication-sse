#ifndef MAT4X4_MULTIPLICATION_SSE_H_
#define MAT4X4_MULTIPLICATION_SSE_H_

#include <memory.h>
#include <stdalign.h>
#include <xmmintrin.h>

typedef union mat4x4 {
  alignas(16) float data[16];
} mat4x4;

mat4x4 mat4x4_mul_sse(mat4x4 m1, mat4x4 m2) {
  mat4x4 out; memset(out, 0, sizeof(mat4x4));

  const float *pM1 = m1.data;
  const float *pM2 = m2.data;
  float *dst = out.data;

  __m128 row1 = _mm_load_ps(&pM2[0]);
  __m128 row2 = _mm_load_ps(&pM2[4]);
  __m128 row3 = _mm_load_ps(&pM2[8]);
  __m128 row4 = _mm_load_ps(&pM2[12]);
  for (int i = 0; i < 4; i++) {
    __m128 brod1 = _mm_set1_ps(pM1[4 * i + 0]);
    __m128 brod2 = _mm_set1_ps(pM1[4 * i + 1]);
    __m128 brod3 = _mm_set1_ps(pM1[4 * i + 2]);
    __m128 brod4 = _mm_set1_ps(pM1[4 * i + 3]);
    __m128 row = _mm_add_ps(
        _mm_add_ps(_mm_mul_ps(brod1, row1),
                   _mm_mul_ps(brod2, row2)),
        _mm_add_ps(_mm_mul_ps(brod3, row3),
                   _mm_mul_ps(brod4, row4)));
    _mm_store_ps(&dst[4 * i], row);
  }

  return out;
}

#endif // MAT4X4_MULTIPLICATION_SSE_H_
