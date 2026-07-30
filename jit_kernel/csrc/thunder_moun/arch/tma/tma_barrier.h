/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// NOTE (yiakwy)
// tma/tma_barrier.h defines mbarrier behaviors to cooperate with TMA operations, including barrier initialization, wait, and arrive (tma_expect_bytes) operations.

namespace nvgpu {
namespace arch {

/*
 * NOTE (yiakwy) : Usage:
 *   while (!tma_try_wait_once(current_barrier, tma_phases[stage])) {
 *       asm volatile("nanosleep.u32 64;\n");
 *   }
 */
__device__ __forceinline__ bool tma_try_wait_once(uint32_t bar_addr, int phase);

namespace internal {

__device__ __forceinline__ void tma_init_barrier(uint64_t* bar_ptr, int num_arrive);

} // namespace internal

template<bool MULTI_CAST = false>
__device__ __forceinline__ void tma_init_barrier(uint64_t* bar_ptr, int num_arrive = 1) {
    if constexpr (MULTI_CAST) {
        internal::tma_init_barrier(bar_ptr, num_arrive/*CLUSTER_SIZE_M*/);
    } else {
        internal::tma_init_barrier(bar_ptr, 1);
    }
}

__device__ __forceinline__ void tma_store_fence();

__device__ __forceinline__ void tma_wait(uint32_t bar, int tma_phase);

__device__ __forceinline__ void tma_expect_bytes(uint32_t bar_addr, uint32_t bytes);

__device__ __forceinline__ void tma_expect_bytes(uint64_t* bar, uint32_t bytes);

} // namespace arch
} // namespace nvgpu
