/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include "../tensor/tuple.h"
#include "../tensor/array_ref.h"

// NOTE (yiakwy) : enforce hostside simulation
#include <cmath>
#include <cstdint>


#define XPU_SIM_DEVICE false


#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif


namespace xpu {

#define MAX_BLOCKS 128

static __host__ __device__ inline xpu::Tuple<int, int> get_block_indices_tri_linear(
    int local_task_id
) {
    int block_idx_m = int((sqrt(8.0 * local_task_id + 1.0) - 1.0) / 2.0);
    int block_idx_n = local_task_id - (block_idx_m * (block_idx_m + 1)) / 2;

    return xpu::Tuple<int, int>(block_idx_m, block_idx_n);
}

template<int GROUP_SIZE_M>
static __host__ __device__ inline void get_block_indices_tri_linear_swizzled(int local_task_id, int& block_idx_m, int& block_idx_n, int num_blocks_m, int group_id) {
    // NOTE (yaikwy) : group swizzle after Down-Left Triangular Mapping for better L2 cache locality in TMA load
    if constexpr (GROUP_SIZE_M > 1) {
        const uint32_t group_off_row = group_id * GROUP_SIZE_M;
        // const uint32_t group_size_m = min(num_blocks_m - group_off_row, static_cast<uint32_t>(GROUP_SIZE_M));
        const uint32_t group_size_m = MIN(num_blocks_m - group_off_row, static_cast<uint32_t>(GROUP_SIZE_M));

        auto sum_tri = [](int h) {
            return h * (h + 1) / 2;
        };

        const uint32_t og_off = sum_tri(group_off_row);
        const uint32_t res = sum_tri(group_size_m);

        const uint32_t num_blocks_in_group = sum_tri(group_off_row + group_size_m) - og_off;
        const uint32_t in_group_idx = local_task_id - og_off;

        const uint32_t test_col = in_group_idx / group_size_m;

        if (test_col < group_off_row) {
            block_idx_m = group_off_row + (in_group_idx % group_size_m);
            block_idx_n = test_col;
        } else {
            const uint32_t ig_col_off = group_off_row;
            const uint32_t sub_in_group_id = in_group_idx - ig_col_off * group_size_m;

            // NOTE (yiakwy) : since int ( sqrt( gm^2 + gm - 2ig_id - 1/4) - 1/2 ) - 1 < c0 <= sqrt( gm^2 + gm - 2ig_id - 1/4) - 1/2 )
            const uint32_t c0 = int ( sqrt( group_size_m*group_size_m + group_size_m - 2*sub_in_group_id - 1.0/4) - 1.0/2 );
            const uint32_t c0_plus_1 = c0 + 1;
            const uint32_t c1 = group_size_m - c0_plus_1;

            block_idx_n = ig_col_off + c1;
            block_idx_m = group_off_row + (sub_in_group_id - (group_size_m + c0_plus_1 + 1) * (group_size_m - c0_plus_1) / 2) + c1;
        }
    }
}

static __host__ __device__ inline xpu::Tuple<int, int> get_block_indices_tri_linear_optimized(
    int local_task_id,
    int num_blocks_m
) {
#if defined(__CUDA_ARCH__) || !XPU_SIM_DEVICE // NOTE (yiakwy) : force device code path for CUDA simulation
    // NOTE (yiakwy) : we pair the first element with the last element
    int group_size = num_blocks_m + 1;

    // NOTE (yiakwy) : compute group index and remainder to determine the row and column
    int group_id = local_task_id / group_size;
    int col_in_group = local_task_id % group_size;

    int row, col;
    int even_row_len = group_id + 1;

    if (col_in_group < even_row_len) {
        row = group_id;
        col = col_in_group;
    } else {
        row = num_blocks_m - 1 - group_id;
        col = col_in_group - even_row_len;
    }

    return xpu::Tuple<int, int>(row, col);
#else

#error "Not Supported!"

    // static_assert(m > 0);
    // static_assert(m <= MAX_BLOCKS);

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
static __host__ __device__ inline int get_task_id_from_block_indices_tri_linear_optimized(
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

    // static_assert(n > 0);
    // static_assert(n <= m);
    // static_assert(m < num_blocks_m);

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

template<int GROUP_SIZE_M=2, bool Zig_Zag_Swizzle=true>
static __host__ __device__ inline void gaussian_folding_swizzled(int local_task_id, int& block_idx_m, int& block_idx_n, int num_blocks_m) {
    int row_size = num_blocks_m + 1;
    int group_size = row_size * GROUP_SIZE_M;

    int group_id = local_task_id / group_size;

    int task_id_group = local_task_id % group_size;

    int r = task_id_group % GROUP_SIZE_M;
    int c = task_id_group / GROUP_SIZE_M;

    if constexpr (Zig_Zag_Swizzle) {
        if (c % 2 == 1) {
            r = GROUP_SIZE_M - 1 - r;
        }
    }
    int mirror_task_id = r * row_size + c;

    auto idx = get_block_indices_tri_linear_optimized(mirror_task_id, num_blocks_m);
    block_idx_m = xpu::get<0>(idx);
    block_idx_n = xpu::get<1>(idx);
}

} // namespace xpu
