/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#include "cluster.h"

namespace nvgpu {
namespace arch {

__device__ inline uint32_t cluster_ctarank() {
    uint32_t cluster_rank;
    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(cluster_rank) : );
    return cluster_rank;
}

__device__ inline uint32_t cluster_map_shared_rank(uint32_t local_addr, uint32_t cta_rank) {
    uint32_t mapped;
    asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(mapped) : "r"(local_addr), "r"(cta_rank));
    return mapped;
}

__device__ inline uint32_t cluster_map_shared_rank(void* local_ptr, int target_block_rank) {
    uint32_t local_smem_addr = __cvta_generic_to_shared(local_ptr);
    uint32_t cluster_smem_addr;

    cluster_smem_addr = cluster_map_shared_rank(local_smem_addr, target_block_rank);

    return cluster_smem_addr;
}


__device__ inline void cluster_arrive() {
    asm volatile("barrier.cluster.arrive.aligned;\n" : : );
}

__device__ inline void cluster_wait() {
    asm volatile("barrier.cluster.wait.aligned;\n" : : );
}

// NOTE (yiakwy) : refer to https://github.com/NVIDIA/cutlass/blob/25e252bdce504932d83f43f07c4b8cc7f9b8e2b6/include/cute/arch/cluster_sm90.hpp#L75
__device__ inline void cluster_sync() {
    cluster_arrive();
    cluster_wait();
}

__device__ inline uint32_t elect_one_sync()
{
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
    "{\n"
    ".reg .b32 %%rx;\n"
    ".reg .pred %%px;\n"
    "     elect.sync %%rx|%%px, %2;\n"
    "@%%px mov.s32 %1, 1;\n"
    "     mov.s32 %0, %%rx;\n"
    "}\n"
    : "+r"(laneid), "+r"(pred)
    : "r"(0xFFFFFFFF));
  return pred;
}

} // namespace arch
} // namespace nvgpu
