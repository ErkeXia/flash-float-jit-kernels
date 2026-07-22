/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include "../tensor/tuple.h"
#include "../tensor/array_ref.h"

#include <cstddef>

#define XPU_SIM_DEVICE false

namespace xpu {

#define MAX_BLOCKS 128

static __host__ __device__ inline xpu::Tuple<int, int> get_block_indices_optimized(
    int task_id,
    int num_blocks_m
) {
#if defined(__CUDA_ARCH__) || !XPU_SIM_DEVICE // NOTE (yiakwy) : force device code path for CUDA simulation
    // NOTE (yiakwy) : we pair the first element with the last element
    int group_size = num_blocks_m + 1;

    // NOTE (yiakwy) : compute group index and remainder to determine the row and column
    int k = task_id / group_size;
    int rem = task_id % group_size;

    int row, col;
    int even_row_len = k + 1;

    if (rem < even_row_len) {
        row = k;
        col = rem;
    } else {
        row = num_blocks_m - 1 - k;
        col = rem - even_row_len;
    }

    return xpu::Tuple<int, int>(row, col);
#else

#error "Not Supported!"
    /*
    static_assert(m > 0);
    static_assert(m <= MAX_BLOCKS);
     */

    int total_rows = num_blocks_m;
    int row_order[MAX_BLOCKS];
    int left = 0, right = total_rows - 1;
    int idx = 0;
    while (left <= right) {
        row_order[idx++] = left++;
        if (left <= right) row_order[idx++] = right--;
    }

    int cumulative = 0;
    for (int i = 0; i < total_rows; ++i) {
        int row = row_order[i];
        int blocks_in_row = row + 1;
        if (task_id < cumulative + blocks_in_row) {
            return xpu::Tuple<int, int>(row, task_id - cumulative);
        }
        cumulative += blocks_in_row;
    }
    return xpu::Tuple<int, int>(0, 0);
#endif
}

// NOTE (yiakwy) : for device verification
static __host__ __device__ inline int get_task_id_from_block(
    int m,
    int n,
    int num_blocks_m
) {
#if defined(__CUDA_ARCH__) || !XPU_SIM_DEVICE // NOTE (yiakwy) : force device code path for CUDA simulation
    int group_size = num_blocks_m + 1;
    int k, cumulative;

    if (m < (num_blocks_m + 1) / 2) {
        k = m;
        cumulative = k * group_size;
    } else {
        k = num_blocks_m - 1 - m;
        cumulative = k * group_size + (k + 1);
    }

    return cumulative + n;
#else
    /*
    static_assert(n > 0);
    static_assert(n <= m);
    static_assert(m < num_blocks_m);
     */

    int total_rows = num_blocks_m;

    int row_order[MAX_BLOCKS];

    int left = 0, right = total_rows - 1;
    int idx = 0;
    while (left <= right) {
        row_order[idx++] = left++;
        if (left <= right) row_order[idx++] = right--;
    }

    int row_position = -1;
    for (int i = 0; i < total_rows; ++i) {
        if (row_order[i] == m) {
            row_position = i;
            break;
        }
    }
    if (row_position == -1) return -1;

    int cumulative = 0;
    for (int i = 0; i < row_position; ++i) {
        int r = row_order[i];
        cumulative += (r + 1);
    }

    return cumulative + n;
#endif
}

} // namespace xpu
