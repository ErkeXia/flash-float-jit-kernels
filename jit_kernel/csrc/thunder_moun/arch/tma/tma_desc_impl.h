/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include "tma_desc.h"

namespace nvgpu {
namespace arch{

template<int BLOCK_SIZE_K, bool Atom = false>
constexpr CUtensorMapSwizzle get_tma_swizzle_mode() {
    constexpr int N = BLOCK_SIZE_K;
#if CUDA_VERSION >= 12080
    if constexpr (Atom && N == 128) {
        return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B;
    }
#endif
    if constexpr (N == 0) {
        return CU_TENSOR_MAP_SWIZZLE_NONE;
    } else
    if constexpr (N == 32) {
        return CU_TENSOR_MAP_SWIZZLE_32B;
    } else
    if constexpr (N == 64) {
        return CU_TENSOR_MAP_SWIZZLE_64B;
    } else
    if constexpr (N == 128) {
        return CU_TENSOR_MAP_SWIZZLE_128B;
    } else {
        static_assert(N == 128 || N == 64 || N == 32, "Unsupported swizzling mode");
    }

    return CU_TENSOR_MAP_SWIZZLE_NONE;
}

} // namespace arch
} // namespace nvgpu
