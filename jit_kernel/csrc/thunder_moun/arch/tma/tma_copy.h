/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace nvgpu {
namespace arch {

__device__ __inline__ void tma2d_multicast_load_async(uint32_t smem_addr/*smem dest*/, const uint64_t tma_desc_addr/*gmem src*/,
                                                      uint32_t s_w_mbar_addr, /*mbarrier*/
                                                      const int32_t inner_dim_offset, const int32_t outter_dim_offset,
                                                      uint16_t cluster_mask, uint64_t cache_hint);

__device__ __inline__ void tma2d_multicast_load_async(void* smem_ptr/*smem dest*/, void const * tma_desc_ptr/*gmem src*/,
                                                      uint64_t* s_w_mbar_ptr, /*mbarrier*/
                                                      const int32_t inner_dim_offset, const int32_t outter_dim_offset,
                                                      uint16_t cluster_mask, uint64_t cache_hint);

} // namespace arch
} // namespace nvgpu

#include "tma_copy_impl.h"
