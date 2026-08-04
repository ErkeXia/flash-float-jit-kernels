/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "tma_barrier.h"

namespace nvgpu {
namespace arch {

__device__ __inline__ bool tma_try_wait_once(uint32_t bar_addr, int phase) {
  uint32_t success;
  asm volatile(
      "{\n\t"
      ".reg .pred P1; \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
      "selp.b32 %0, 1, 0, P1; \n\t"
      "}"
      : "=r"(success)
      : "r"(bar_addr), "r"(phase));
  return success;
}

namespace internal {

__device__ __inline__ void tma_init_barrier(uint64_t* bar_ptr, int num_arrive) {
    uint32_t s_bar_ptr = __cvta_generic_to_shared(bar_ptr);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(s_bar_ptr), "r"(num_arrive));
}

} // namespace internal


__device__ __inline__ void tma_store_fence() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_wait(uint32_t bar, int tma_phase) {

  asm volatile(
      "{\n"
      ".reg .pred P2;\n"
      "WAIT_LOOP:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P2, [%0], %1;\n"
      "@P2 bra.uni DONE;\n"
      "nanosleep.u32 64;\n"
      "bra.uni WAIT_LOOP;\n"
      "DONE:\n"
      "}\n"
      :: "r"(bar), "r"(tma_phase) : "memory"
  );

  // asm volatile(
  //     "{\n"
  //     ".reg .pred P2;\n"
  //     "WAIT_LOOP:\n"
  //     "mbarrier.try_wait.parity.shared::cta.b64 P2, [%0], %1;\n"
  //     "@P2 bra.uni DONE;\n"
  //     "bra.uni LAB_WAIT;\n"
  //     "DONE:\n"
  //     "}\n"
  //     :: "r"(bar), "r"(tma_phase) : "memory"
  // );
}

__device__ __forceinline__ void tma_expect_bytes(uint32_t bar_addr, uint32_t bytes) {
  asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
      :: "r"(bar_addr), "r"(bytes));
}

__device__ __forceinline__ void tma_expect_bytes(uint64_t* bar, uint32_t bytes) {
  uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  tma_expect_bytes(bar_addr, bytes);
}

} // namespace arch
} // namespace nvgpu
