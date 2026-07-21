/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace nvgpu {
namespace arch {

__device__ __inline__ uint32_t cluster_ctarank();

// NOTE (yiakwy) : use cluster.map_shared_rank api to resolve the shared memory address from block-level view to cluster-level view.
__device__ __inline__ uint32_t cluster_map_shared_rank(void* local_ptr, int target_block_rank);

__device__ __inline__ void cluster_arrive();

__device__ __inline__ void cluster_wait();

__device__ __inline__ void cluster_sync();

__device__ __inline__ uint32_t elect_one_sync();

} // namespace arch
} // namespace nvgpu
