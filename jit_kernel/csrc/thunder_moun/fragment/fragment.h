/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once
#include <cuda_runtime.h>

#ifndef WARP_SIZE

#define WARP_SIZE 32

#endif

#ifndef SWIZZLE_64B_STORE

#define SWIZZLE_64B_STORE 0

#endif

namespace xpu {

enum class MemoryDomain { kShared, kRegister };

template <typename _T, int BM, int BN, MemoryDomain Domain>
struct FragmentView {

    using T = _T;

    static constexpr int VEC_SIZE = sizeof(T);

    T* shared_ptr;

    __device__ inline FragmentView(T* smem) : shared_ptr(smem) {}

    __device__ inline T& operator()(int m, int n) {
        return shared_ptr[m * BN + n];
    }

    // NOTE (yiakwy) : implement inplace transpose for 128x128 fp32 fragment
    // TODO (yaikwy) : add inplace transpose for 128x128 fp16/bf16 fragment
    __device__ inline void _transpose() {
        static_assert(BM == BN, "inplace transpose can be only applied to square fragment.");

        const int tid = threadIdx.x;
        const int threads_per_block = blockDim.x;

        const int lane_id = threadIdx.x % WARP_SIZE;
        const int warp_id = threadIdx.x / WARP_SIZE;

        constexpr int FRAG_M = 16;
        constexpr int TOTAL_ELEMENTS = BM * BN;

        constexpr int M_STEPS = BM / FRAG_M;

        constexpr int sub_tasks_0 = (M_STEPS * M_STEPS - M_STEPS) / 2;
        constexpr int sub_tasks_1 = M_STEPS;

        // if (threadIdx.x == 0 && blockIdx.x == 0) {
        //     printf("[_transpose] [Split#%d] [SM#%d] step 1 transpose non-diag data, M_STEPS=%d ...\n", blockIdx.x, blockIdx.y, M_STEPS);
        // }
        // __syncthreads();

        // NOTE (yiakwy) : process lower left sub fragment
        #pragma unroll
        for (int sub_frag_idx_m = 1; sub_frag_idx_m < M_STEPS; ++sub_frag_idx_m) {

            int sub_frag_idx_m_off = sub_frag_idx_m * FRAG_M;

            #pragma unroll
            for (int sub_frag_idx_n = 0; sub_frag_idx_n < sub_frag_idx_m; ++sub_frag_idx_n) {

                int sub_frag_idx_n_off = sub_frag_idx_n * FRAG_M;

                #pragma unroll
                for (int e_idx = tid; e_idx < FRAG_M * FRAG_M; e_idx += threads_per_block) {
                    int thr_col = e_idx % FRAG_M;
                    int thr_row = e_idx / FRAG_M;

                    // NOTE(yiakwy) : only valid for 16x16 fragment
#if SWIZZLE_64B_STORE
                    int swizzle_col = thr_col ^ (thr_row % 8);
                    int swizzle_row = thr_row ^ (thr_col % 8);

                    // (sub_frag_idx_m_off + thr_row, sub_frag_idx_n_off + swizzle_col)
                    T src_val = shared_ptr[(sub_frag_idx_m_off + thr_row) * BM + sub_frag_idx_n_off + swizzle_col];

                    // (sub_frag_idx_n_off + thr_row, sub_frag_idx_m_off + swizzle_col)
                    T dst_val = shared_ptr[(sub_frag_idx_n_off + thr_col) * BM + sub_frag_idx_m_off + swizzle_row];
#else
                    int swizzle_col = thr_col;
                    int swizzle_row = thr_row;

                    // (sub_frag_idx_m_off + thr_row, sub_frag_idx_n_off + swizzle_col)
                    T src_val = shared_ptr[(sub_frag_idx_m_off + thr_row) * BM + sub_frag_idx_n_off + thr_col];

                    // (sub_frag_idx_n_off + thr_row, sub_frag_idx_m_off + swizzle_col)
                    T dst_val = shared_ptr[(sub_frag_idx_n_off + thr_col) * BM + sub_frag_idx_m_off + thr_row];
#endif


#if SWIZZLE_64B_STORE
                    shared_ptr[(sub_frag_idx_m_off + thr_row) * BM + sub_frag_idx_n_off + swizzle_col] = dst_val;
                    shared_ptr[(sub_frag_idx_n_off + thr_col) * BM + sub_frag_idx_m_off + swizzle_row] = src_val;
#else
                    shared_ptr[(sub_frag_idx_m_off + thr_row) * BM + sub_frag_idx_n_off + thr_col] = dst_val;
                    shared_ptr[(sub_frag_idx_n_off + thr_col) * BM + sub_frag_idx_m_off + thr_row] = src_val;
#endif

                    // if (sub_frag_idx_m == 7 && sub_frag_idx_n == 0 && thr_row == 15 && thr_col == 0) {
                    //     printf("[Debug] [_transpose] src[%d, %d] -> dst[%d, %d], dst[%d, %d] -> src[%d, %d]\n",
                    //     sub_frag_idx_m_off + thr_row, sub_frag_idx_n_off + swizzle_col,
                    //     sub_frag_idx_n_off + thr_col, sub_frag_idx_m_off + thr_row,
                    //     sub_frag_idx_n_off + thr_col, sub_frag_idx_m_off + swizzle_row,
                    //     sub_frag_idx_m_off + thr_row, sub_frag_idx_n_off + thr_col);
                    // }

                } // end of e_idx

            } // end of sub_frag_idx_n

        } // end of sub_frag_idx_m
        __syncthreads();


        // if (threadIdx.x == 0 && blockIdx.x == 0) {
        //     printf("[_transpose] [Split#%d] [SM#%d] step 2 transpose diag data ...\n", blockIdx.x, blockIdx.y);
        // }
        // __syncthreads();


        #pragma unroll
        for (int task_idx = 0; task_idx < sub_tasks_1; task_idx++) {

            int sub_frag_idx_m_off = task_idx * FRAG_M;
            int sub_frag_idx_n_off = task_idx * FRAG_M;

            int pair_counter = 0;

            // for (int e_idx = tid; e_idx < (FRAG_M * FRAG_M - FRAG_M) / 2; e_idx += threads_per_block) {
            #pragma unroll
            for (int thr_row = 1; thr_row < FRAG_M; ++thr_row) {

                #pragma unroll
                for (int thr_col = 0; thr_col < thr_row; ++thr_col) {

                    if (pair_counter % threads_per_block == tid) {
#if SWIZZLE_64B_STORE
                        int swizzle_col = thr_col ^ (thr_row % 8);
                        int swizzle_row = thr_row ^ (thr_col % 8);

                        T src_val = shared_ptr[(sub_frag_idx_m_off + thr_row) * BM + sub_frag_idx_n_off + swizzle_col];
                        T dst_val = shared_ptr[(sub_frag_idx_n_off + thr_col) * BM + sub_frag_idx_m_off + swizzle_row];

                        shared_ptr[(sub_frag_idx_n_off + thr_col) * BM + sub_frag_idx_m_off + swizzle_row] = src_val;
                        shared_ptr[(sub_frag_idx_m_off + thr_row) * BM + sub_frag_idx_n_off + swizzle_col] = dst_val;
#else
                        T src_val = shared_ptr[(sub_frag_idx_m_off + thr_row) * BM + sub_frag_idx_n_off + thr_col];
                        T dst_val = shared_ptr[(sub_frag_idx_n_off + thr_col) * BM + sub_frag_idx_m_off + thr_row];

                        shared_ptr[(sub_frag_idx_n_off + thr_col) * BM + sub_frag_idx_m_off + thr_row] = src_val;
                        shared_ptr[(sub_frag_idx_m_off + thr_row) * BM + sub_frag_idx_n_off + thr_col] = dst_val;
#endif
                    }
                    pair_counter++;

                } // end of thr_col

            } // end of e_idx

        } // end of task_idx

        __syncthreads();
    }
};

} // namespace xpu
