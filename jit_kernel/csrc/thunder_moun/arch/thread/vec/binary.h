/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace nvgpu {
namespace arch {

// vectorized intrinsic operations


// SIMD instruction

// NOTE (yiakwy) : there has been a common misunderstanding of NVDIA arch for a long time that NVIDIA GPU is purely SIMD arch,
// while Hopper and above actually employs a hybrid SIMD/SIMT design for low precision datatypes.
static __device__ __forceinline__ void simd_vadd(half2* dst, const half2* src) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) { // 128-bit = 8x half = 4x half2
        dst[i] = __hadd2(dst[i], src[i]);
    }
}

} // namespace arch
} // namespace nvgpu
