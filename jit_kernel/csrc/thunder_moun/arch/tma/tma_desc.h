/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace nvgpu {
namespace arch {

template<int BLOCK_SIZE_K, bool Atom = false>
constexpr CUtensorMapSwizzle get_tma_swizzle_mode();

namespace tma {} // namespace tma

} // namespace arch
} // namespace nvgpu

#include "tma_desc_impl.h"
