/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace nvgpu {
namespace arch {

enum class CacheHintSm90 : uint64_t {
  EVICT_NORMAL = 0x1000000000000000,
  EVICT_FIRST = 0x12F0000000000000,
  EVICT_LAST = 0x14F0000000000000,
};

// NOTE (yiakwy) : TMA bulk tensor (multi-dimension) copy (cp.async.bulk.tensor) exclusively used for bulk tensor copy between shared memory and global memory, with support of cluster multicast and L2 cache hint for TMA load operations.

__device__ __inline__ void tma2d_multicast_load_async(uint32_t smem_addr/*smem dest*/, const uint64_t tma_desc_addr/*gmem src*/,
                                                      uint32_t s_w_mbar_addr, /*mbarrier*/
                                                      const int32_t inner_dim_offset, const int32_t outter_dim_offset,
                                                      uint16_t cluster_mask, uint64_t cache_hint);

__device__ __inline__ void tma2d_multicast_load_async(void* smem_ptr/*smem dest*/, void const * tma_desc_ptr/*gmem src*/,
                                                      uint64_t* s_w_mbar_ptr, /*mbarrier*/
                                                      const int32_t inner_dim_offset, const int32_t outter_dim_offset,
                                                      uint16_t cluster_mask, uint64_t cache_hint);

__device__ __inline__ void tma2d_load_async(void* smem_ptr/*smem dest*/, void const * tma_desc_ptr/*gmem src*/,
                                            uint64_t* s_w_mbar_ptr, /*mbarrier*/
                                            const int32_t inner_dim_offset, const int32_t outter_dim_offset,
                                            uint64_t cache_hint);

__device__ __inline__ void tma2d_load_async(uint32_t smem_addr/*smem dest*/, const uint64_t tma_desc_addr/*gmem src*/,
                                            uint32_t s_w_mbar_addr, /*mbarrier*/
                                            const int32_t inner_dim_offset, const int32_t outter_dim_offset,
                                            uint64_t cache_hint);


__device__ __inline__ void tma2d_store(uint64_t tma_addr/*gmem dest*/, uint32_t smem_epilogue_addr /*smem src*/,
                                       const int32_t inner_dim_offset, const int32_t outter_dim_offset);

// NOTE (yiakwy) : 3D TMA bulk tensor copy helpers for batched (B, M, K) symmetric GEMM
//   - the tensor map is encoded with rank 3, shape (B, M, K) for inputs and (B, N, M) for output,
//   - dim0 (batch) always uses box size 1, dim1/dim2 are the 2D tile of the current batch.

__device__ __inline__ void tma3d_multicast_load_async(uint32_t smem_addr/*smem dest*/, const uint64_t tma_desc_addr/*gmem src*/,
                                                      uint32_t s_w_mbar_addr, /*mbarrier*/
                                                      const int32_t dim0_offset, const int32_t dim1_offset, const int32_t dim2_offset,
                                                      uint16_t cluster_mask, uint64_t cache_hint);

__device__ __inline__ void tma3d_multicast_load_async(void* smem_ptr/*smem dest*/, void const * tma_desc_ptr/*gmem src*/,
                                                      uint64_t* s_w_mbar_ptr, /*mbarrier*/
                                                      const int32_t dim0_offset, const int32_t dim1_offset, const int32_t dim2_offset,
                                                      uint16_t cluster_mask, uint64_t cache_hint);

__device__ __inline__ void tma3d_load_async(void* smem_ptr/*smem dest*/, void const * tma_desc_ptr/*gmem src*/,
                                            uint64_t* s_w_mbar_ptr, /*mbarrier*/
                                            const int32_t dim0_offset, const int32_t dim1_offset, const int32_t dim2_offset,
                                            uint64_t cache_hint);

__device__ __inline__ void tma3d_load_async(uint32_t smem_addr/*smem dest*/, const uint64_t tma_desc_addr/*gmem src*/,
                                            uint32_t s_w_mbar_addr, /*mbarrier*/
                                            const int32_t dim0_offset, const int32_t dim1_offset, const int32_t dim2_offset,
                                            uint64_t cache_hint);

__device__ __inline__ void tma3d_store(uint64_t tma_addr/*gmem dest*/, uint32_t smem_epilogue_addr /*smem src*/,
                                       const int32_t dim0_offset, const int32_t dim1_offset, const int32_t dim2_offset);

} // namespace arch
} // namespace nvgpu

#include "tma_copy_impl.h"
