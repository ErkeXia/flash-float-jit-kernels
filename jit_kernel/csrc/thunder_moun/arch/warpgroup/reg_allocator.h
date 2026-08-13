/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace nvgpu {
namespace arch {

template<uint32_t RegCount>
static __device__ inline
void reg_alloc_increase_registers(){
    static_assert(RegCount % 8 == 0, "n_reg must be a multiple of 8");
    asm volatile( "setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(RegCount) );
}

template<uint32_t RegCount>
static __device__ inline
void reg_dealloc_decrease_registers(){
    static_assert(RegCount % 8 == 0, "n_reg must be a multiple of 8");
    asm volatile( "setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(RegCount) );
}

} // namespace arch
} // namespace nvgpu
