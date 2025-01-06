// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/avgpool.h>


void xnn_f32_avgpool_minmax_ukernel_9x__sse_c4(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    const float* zero,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(kernel_elements <= 9);
  assert(channels != 0);

  const __m128 vscale = _mm_load_ps(params->sse.scale);
  const __m128 vmin = _mm_load_ps(params->sse.min);
  const __m128 vmax = _mm_load_ps(params->sse.max);

  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    const float* i1 = input[1];
    const float* i2 = input[2];
    const float* i3 = input[3];
    const float* i4 = input[4];
    const float* i5 = input[5];
    const float* i6 = input[6];
    const float* i7 = input[7];
    const float* i8 = input[8];
    input = (const float**) ((uintptr_t) input + input_increment);
    if (kernel_elements < 2) {
      i1 = zero;
    }
    assert(i1 != NULL);
    if (kernel_elements <= 2) {
      i2 = zero;
    }
    assert(i2 != NULL);
    if (kernel_elements < 4) {
      i3 = zero;
    }
    assert(i3 != NULL);
    if (kernel_elements <= 4) {
      i4 = zero;
    }
    assert(i4 != NULL);
    if (kernel_elements < 6) {
      i5 = zero;
    }
    assert(i5 != NULL);
    if (kernel_elements <= 6) {
      i6 = zero;
    }
    assert(i6 != NULL);
    if (kernel_elements < 8) {
      i7 = zero;
    }
    assert(i7 != NULL);
    if (kernel_elements <= 8) {
      i8 = zero;
    }
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }

    size_t c = channels;
    while (c >= 4) {
      const __m128 vi0 = _mm_loadu_ps(i0);
      i0 += 4;
      const __m128 vi1 = _mm_loadu_ps(i1);
      i1 += 4;
      const __m128 vi2 = _mm_loadu_ps(i2);
      i2 += 4;
      const __m128 vi3 = _mm_loadu_ps(i3);
      i3 += 4;
      const __m128 vi4 = _mm_loadu_ps(i4);
      i4 += 4;
      const __m128 vi5 = _mm_loadu_ps(i5);
      i5 += 4;
      const __m128 vi6 = _mm_loadu_ps(i6);
      i6 += 4;
      const __m128 vi7 = _mm_loadu_ps(i7);
      i7 += 4;
      const __m128 vi8 = _mm_loadu_ps(i8);
      i8 += 4;

      const __m128 vsum018 = _mm_add_ps(_mm_add_ps(vi0, vi1), vi8);
      const __m128 vsum23 = _mm_add_ps(vi2, vi3);
      const __m128 vsum45 = _mm_add_ps(vi4, vi5);
      const __m128 vsum67 = _mm_add_ps(vi6, vi7);

      const __m128 vsum2345 = _mm_add_ps(vsum23, vsum45);
      const __m128 vsum01678 = _mm_add_ps(vsum018, vsum67);
      const __m128 vsum = _mm_add_ps(vsum2345, vsum01678);

      __m128 vout = _mm_mul_ps(vsum, vscale);
      vout = _mm_max_ps(vout, vmin);
      vout = _mm_min_ps(vout, vmax);

      _mm_storeu_ps(output, vout); output += 4;

      c -= 4;
    }
    if (c != 0) {
      const __m128 vi0 = _mm_loadu_ps(i0);
      const __m128 vi1 = _mm_loadu_ps(i1);
      const __m128 vi2 = _mm_loadu_ps(i2);
      const __m128 vi3 = _mm_loadu_ps(i3);
      const __m128 vi4 = _mm_loadu_ps(i4);
      const __m128 vi5 = _mm_loadu_ps(i5);
      const __m128 vi6 = _mm_loadu_ps(i6);
      const __m128 vi7 = _mm_loadu_ps(i7);
      const __m128 vi8 = _mm_loadu_ps(i8);

      const __m128 vsum01 = _mm_add_ps(vi0, vi1);
      const __m128 vsum23 = _mm_add_ps(vi2, vi3);
      const __m128 vsum45 = _mm_add_ps(vi4, vi5);
      const __m128 vsum67 = _mm_add_ps(vi6, vi7);
      const __m128 vsum018 = _mm_add_ps(vsum01, vi8);
      const __m128 vsum2345 = _mm_add_ps(vsum23, vsum45);
      const __m128 vsum01678 = _mm_add_ps(vsum018, vsum67);
      const __m128 vsum = _mm_add_ps(vsum2345, vsum01678);

      __m128 vout = _mm_mul_ps(vsum, vscale);
      vout = _mm_max_ps(vout, vmin);
      vout = _mm_min_ps(vout, vmax);

      if (c & 2) {
        _mm_storel_pi((__m64*) output, vout);
        vout = _mm_movehl_ps(vout, vout);
        output += 2;
      }
      if (c & 1) {
        _mm_store_ss(output, vout);
        output += 1;
      }
    }
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
