
/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#include <cuda_runtime.h>

#include "tensor/tuple.h"
#include "block/sched.h"

#define GROUP_SIZE_M 2

__global__ void test_swizzle_kernel(int num_blocks_m) {
    using namespace xpu;

    int local_task_id = blockIdx.x;

    auto idx = get_block_indices_tri_linear(local_task_id);
    int block_idx_m = xpu::get<0>(idx);
    int block_idx_n = xpu::get<1>(idx);

    // Grouping for better L2 cache locality in TMA load
    const uint32_t group_id = block_idx_m / GROUP_SIZE_M;

    get_block_indices_tri_linear_swizzled<GROUP_SIZE_M>(local_task_id, block_idx_m/*dest*/, block_idx_n/*dest*/, num_blocks_m, group_id);
}

int main() {
    int num_blocks_m = 8;

    // Launch exactly 10 Blocks (i.e., 4x4 symmetric gemm) with 32 Threads (1 Warp)
    test_swizzle_kernel<<<10, 32>>>(num_blocks_m);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Single-warp execution completed successfully.\n");
    return 0;
}
