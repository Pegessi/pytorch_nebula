// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                       \
      size_t n,                                                    \
      const void* input,                                           \
      const void* max,                                             \
      void* output,                                                \
      void* sum,                                                   \
      const union xnn_f16_expminus_params* params);

DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x32)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x32_acc2)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x32_acc4)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x40)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x40_acc2)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x40_acc5)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x48)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x48_acc2)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x48_acc3)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x64)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x64_acc2)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x64_acc4)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x72)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x72_acc3)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x80)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x80_acc2)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x80_acc5)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x96)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x96_acc2)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x96_acc3)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_x96_acc6)

DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_x32)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_x32_acc2)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_x32_acc4)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_x40)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_x40_acc2)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_x40_acc5)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_x48)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_x48_acc2)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_x48_acc3)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_x64)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_x64_acc2)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_x64_acc4)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_x72)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_x72_acc3)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_x80)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_x80_acc2)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_x80_acc5)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_x96)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_x96_acc2)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_x96_acc3)
DECLARE_F16_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_x96_acc6)


#define DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                       \
      size_t n,                                                    \
      const float* input,                                          \
      const float* max,                                            \
      float* output,                                               \
      float* sum,                                                  \
      const union xnn_f32_expminus_params* params);

DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x4)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x8)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x8_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x12)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x12_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x12_acc3)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x16)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x16_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x16_acc4)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x20)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x20_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x20_acc5)

DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x4)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x8)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x8_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x12)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x12_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x12_acc3)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x16)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x16_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x16_acc4)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x20)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x20_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x20_acc5)

DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x4)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x8)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x8_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x12)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x12_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x12_acc3)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x16)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x16_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x16_acc4)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x20)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x20_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x20_acc5)

DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x4)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x8)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x8_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x12)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x12_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x12_acc3)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x16)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x16_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x16_acc4)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x20)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x20_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x20_acc5)

DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x4)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x8)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x8_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x12)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x12_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x12_acc3)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x16)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x16_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x16_acc4)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x20)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x20_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x20_acc5)

DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x64)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x64_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x64_acc4)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x72)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x72_acc3)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x80)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x80_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x80_acc5)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x96)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x96_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x96_acc3)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x96_acc6)

DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x128)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x128_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x128_acc4)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x144)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x144_acc3)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x160)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x160_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x160_acc5)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x192)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x192_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x192_acc3)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x192_acc6)

DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x4)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x8)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x8_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x12)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x12_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x12_acc3)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x16)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x16_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x16_acc4)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x20)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x20_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x20_acc5)

DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x4)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x8)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x8_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x12)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x12_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x12_acc3)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x16)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x16_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x16_acc4)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x20)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x20_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x20_acc5)

DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_x1)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_x2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_x2_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_x4)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_x4_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_x4_acc4)

DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_x1)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_x2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_x2_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_x4)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_x4_acc2)
DECLARE_F32_RADDSTOREEXPMINUSMAX_UKERNEL_FUNCTION(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_x4_acc4)

#ifdef __cplusplus
} /* extern "C" */
#endif
