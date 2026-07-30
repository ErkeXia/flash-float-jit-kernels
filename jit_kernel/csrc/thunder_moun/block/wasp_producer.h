/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include "../arch/tma/tma_copy.h"
#include "../arch/tma/tma_barrier.h"

#include "../arch/warpgroup/reg_allocator.h"

#include "block.h"

namespace xpu {

    template<int STAGES, int GROUP_SIZE_M,
             int BM, int BN, int BK,
             bool USE_CLUSTER_MULTICAST, bool USE_LINEAR_TO_TRIL_LAYOUT = true, bool FLIP_TMA_PHASE = false>
    struct producer {

        using fp8_t = __nv_fp8_e4m3;

        using DtypeA = SharedBlock<fp8_t, BM, BK>;
        using DtypeB = SharedBlock<fp8_t, BN, BK>;

        static __device__ __inline__ void setup() {
            // nvgpu::arch::reg_alloc_increase_registers<40>(); // increase registers for producers
        }

        // TODO (yiakwy) : move to setup and pass as Arg

        // NOTE (yiakwy) : load once w/o cluster multicast
        static __device__ __inline__ void load_once(
            const int& warpgroups_tid, const int& group_id,
            const int& current_k, const int& block_idx_m, const int& block_idx_n, const int& total_stage_bytes, // TODO (yiakwy) : move to setup and pass as Arg
            const CUtensorMap* tma_desc_X, const CUtensorMap* tma_desc_W,
            DtypeA* shmem_X/*dst*/, DtypeB* shmem_W/*dst*/, const uint64_t* barriers,
            const uint64_t& cache_hint_lhs, const uint64_t& cache_hint_rhs,
            int& write_stage/*src & dst*/, int& phase/*dst*/) {

            // Commit the async copy for the first few stages, and then we will wait on them in the main loop
            // NOTE (yiakwy) : see https://github.com/NVIDIA/cutlass/blob/5f06f5fc1a072bbe4815fae7ae8470b876ed603a/include/cute/arch/copy_sm90_tma.hpp#L117 for the recommended way to commit async copy with TMA and mbarrier synchronization.
            uint32_t smem_x_addr = __cvta_generic_to_shared(&shmem_X[write_stage]);
            uint32_t smem_w_addr = __cvta_generic_to_shared(&shmem_W[write_stage]);

            uint32_t s_w_bar_ptr = __cvta_generic_to_shared(&barriers[write_stage]);

            nvgpu::arch::tma_expect_bytes(s_w_bar_ptr, total_stage_bytes);

            nvgpu::arch::tma2d_load_async(
                smem_x_addr, reinterpret_cast<uint64_t>(tma_desc_X), s_w_bar_ptr,
                current_k * BK, block_idx_m * BM, cache_hint_lhs
            );

            nvgpu::arch::tma2d_load_async(
                smem_w_addr, reinterpret_cast<uint64_t>(tma_desc_W), s_w_bar_ptr,
                current_k * BK, block_idx_n * BN, cache_hint_rhs
            );

        }

        // NOTE (yiakwy) : load data once w/ cluster multicast
        static __device__ __inline__ void load_once(
            const int& warpgroups_tid, const int& group_id,
            const int& current_k, const int& block_idx_m, const int& block_idx_n, const int& total_stage_bytes,
            const CUtensorMap* tma_desc_X, const CUtensorMap* tma_desc_W,
            DtypeA* shmem_X/*dst*/, DtypeB* shmem_W/*dst*/, const uint64_t* barriers,
            const uint16_t& cluster_mask, const int& cluster_group_m_rank, const uint64_t& cache_hint_lhs, const uint64_t& cache_hint_rhs,
            int& write_stage/*src & dst*/, int& phase/*dst*/) {

            // Commit the async copy for the first few stages, and then we will wait on them in the main loop
            // NOTE (yiakwy) : see https://github.com/NVIDIA/cutlass/blob/5f06f5fc1a072bbe4815fae7ae8470b876ed603a/include/cute/arch/copy_sm90_tma.hpp#L117 for the recommended way to commit async copy with TMA and mbarrier synchronization.
            uint32_t smem_x_addr = __cvta_generic_to_shared(&shmem_X[write_stage]);
            uint32_t smem_w_addr = __cvta_generic_to_shared(&shmem_W[write_stage]);

            uint32_t s_w_bar_ptr = __cvta_generic_to_shared(&barriers[write_stage]);

            nvgpu::arch::tma_expect_bytes(s_w_bar_ptr, total_stage_bytes);

            static_assert(USE_LINEAR_TO_TRIL_LAYOUT && USE_CLUSTER_MULTICAST);

            // The 2d tma instruction needs cluster mask to specify the destination of the multicast
            if (block_idx_n < group_id * GROUP_SIZE_M) {

                nvgpu::arch::tma2d_load_async(
                    smem_x_addr, reinterpret_cast<uint64_t>(tma_desc_X), s_w_bar_ptr,
                    current_k * BK, block_idx_m * BM, cache_hint_lhs
                );

                if (cluster_group_m_rank == 0) {
                    nvgpu::arch::tma2d_multicast_load_async(
                        smem_w_addr, reinterpret_cast<uint64_t>(tma_desc_W), s_w_bar_ptr,
                        current_k * BK, block_idx_n * BN, cluster_mask, cache_hint_rhs
                    );
                } // end of cluster_group_m_rank == 0

            } else {

                // non-multicast version
                nvgpu::arch::tma2d_load_async(
                    smem_x_addr, reinterpret_cast<uint64_t>(tma_desc_X), s_w_bar_ptr,
                    current_k * BK, block_idx_m * BM, cache_hint_lhs
                );

                nvgpu::arch::tma2d_load_async(
                    smem_w_addr, reinterpret_cast<uint64_t>(tma_desc_W), s_w_bar_ptr,
                    current_k * BK, block_idx_n * BN, cache_hint_rhs
                );
            } // block_idx_n < group_id * GROUP_SIZE_M

        }


        // static __device__ __inline__ void load(
        //     const int warpgroups_lane_id, const int group_id, const int block_idx_m, const int block_idx_n,
        //     const int k_start, const int k_end, const int total_stage_bytes,
        //     const CUtensorMap* tma_desc_X, const CUtensorMap* tma_desc_W,
        //     DtypeA* shmem_X, DtypeB* shmem_W, const uint64_t * barriers,
        //     const uint64_t cache_hint_lhs, const uint64_t cache_hint_rhs,
        //     int& write_stage/*src & dst*/) {

        //     static_assert(FLIP_TMA_PHASE == false);

        //     int fake_tma_phase = 0;
        //     load(
        //         warpgroups_lane_id, group_id, block_idx_m, block_idx_n,
        //         k_start, k_end, total_stage_bytes,
        //         tma_desc_X, tma_desc_W,
        //         shmem_X, shmem_W, barriers,
        //         cache_hint_lhs, cache_hint_rhs,
        //         write_stage, fake_tma_phase);
        // }

        // static __device__ __inline__ void load(
        //     const int warpgroups_lane_id, const int group_id, const int block_idx_m, const int block_idx_n,
        //     const int k_start, const int k_end, const int total_stage_bytes,
        //     const CUtensorMap* tma_desc_X, const CUtensorMap* tma_desc_W,
        //     DtypeA* shmem_X, DtypeB* shmem_W, const uint64_t * barriers,
        //     const uint64_t cache_hint_lhs, const uint64_t cache_hint_rhs,
        //     int& write_stage, int& phase/*dst*/) {

        //     #pragma unroll
        //     for (int i = 0; i < STAGES - 1; ++i) {
        //         int current_k = k_start + i;
        //         if (current_k < k_end) {

        //             load_once(
        //                 warpgroups_lane_id, group_id,
        //                 current_k, block_idx_m, block_idx_n, total_stage_bytes,
        //                 tma_desc_X, tma_desc_W,
        //                 shmem_X, shmem_W, barriers,
        //                 cache_hint_lhs, cache_hint_rhs,
        //                 write_stage, phase
        //             );

        //         } // current_k < k_end

        //     }

        // }


        // // TODO (yiakwy) : move to setup and pass as Arg
        // // NOTE (yiakwy) : load data with cluster multicast
        // static __device__ __inline__ void load(
        //     const int& warpgroups_lane_id, const int& group_id, const int& block_idx_m, const int& block_idx_n,
        //     const int& k_start, const int& k_end, const int& total_stage_bytes,
        //     const CUtensorMap* tma_desc_X, const CUtensorMap* tma_desc_W,
        //     DtypeA* shmem_X, DtypeB* shmem_W, const uint64_t * barriers,
        //     const uint16_t& cluster_mask, const int& cluster_group_m_rank, const uint64_t& cache_hint_lhs, const uint64_t& cache_hint_rhs,
        //     int& write_stage/*src & dst*/) {

        //     static_assert(FLIP_TMA_PHASE == false);

        //     int fake_tma_phase = 0;
        //     load(
        //         warpgroups_lane_id, group_id, block_idx_m, block_idx_n,
        //         k_start, k_end, total_stage_bytes,
        //         tma_desc_X, tma_desc_W,
        //         shmem_X, shmem_W, barriers,
        //         cluster_mask, cluster_group_m_rank, cache_hint_lhs, cache_hint_rhs,
        //         write_stage, fake_tma_phase);
        // }

        // static __device__ __inline__ void load(
        //     const int& warpgroups_lane_id, const int& group_id, const int& block_idx_m, const int& block_idx_n,
        //     const int& k_start, const int& k_end, const int& total_stage_bytes,
        //     const CUtensorMap* tma_desc_X, const CUtensorMap* tma_desc_W,
        //     DtypeA* shmem_X, DtypeB* shmem_W, const uint64_t * barriers,
        //     const uint16_t& cluster_mask, const int& cluster_group_m_rank, const uint64_t& cache_hint_lhs, const uint64_t& cache_hint_rhs,
        //     int& write_stage/*src & dst*/, int& phase/*dst*/) {

        //     #pragma unroll
        //     for (int i = 0; i < STAGES - 1; ++i) {
        //         int current_k = k_start + i;
        //         if (current_k < k_end) {

        //             load_once(
        //                 warpgroups_lane_id, group_id,
        //                 current_k, block_idx_m, block_idx_n, total_stage_bytes,
        //                 tma_desc_X, tma_desc_W,
        //                 shmem_X, shmem_W, barriers,
        //                 cluster_mask, cluster_group_m_rank, cache_hint_lhs, cache_hint_rhs,
        //                 write_stage, phase
        //             );

        //         } // current_k < k_end

        //     }

        // }

        // === WASP producer ===

        static __device__ __inline__ void load(
            const int warpgroups_lane_id, const int group_id, const int block_idx_m, const int block_idx_n,
            const int k_start, const int k_end, const int total_stage_bytes,
            const CUtensorMap* tma_desc_X, const CUtensorMap* tma_desc_W,
            DtypeA* shmem_X, DtypeB* shmem_W, const uint64_t * barriers, const uint64_t * empty_barriers,
            const uint64_t cache_hint_lhs, const uint64_t cache_hint_rhs,
            int& write_stage, int& phase/*dst*/) {

            #pragma unroll
            for (int k=k_start; k < k_end; k++) {

                // wait for buffers to be ready to write
                nvgpu::arch::tma_wait(empty_barrieres[write_stage], phase);

                load_once(
                    warpgroups_lane_id, group_id,
                    k, block_idx_m, block_idx_n, total_stage_bytes,
                    tma_desc_X, tma_desc_W,
                    shmem_X, shmem_W, barriers,
                    cache_hint_lhs, cache_hint_rhs,
                    write_stage, phase
                );

                write_stage = (write_stage + 1) % STAGES;
                if (write_stage == 0) {
                    phase ^= 1;
                }

            }

        } //  load w/o cluster multicast


        static __device__ __inline__ void load(
            const int& warpgroups_lane_id, const int& group_id, const int& block_idx_m, const int& block_idx_n,
            const int& k_start, const int& k_end, const int& total_stage_bytes,
            const CUtensorMap* tma_desc_X, const CUtensorMap* tma_desc_W,
            DtypeA* shmem_X, DtypeB* shmem_W, const uint64_t * barriers, const uint64_t * empty_barriers,
            const uint16_t& cluster_mask, const int& cluster_group_m_rank, const uint64_t& cache_hint_lhs, const uint64_t& cache_hint_rhs,
            int& write_stage/*src & dst*/, int& phase/*dst*/) {

            #pragma unroll
            for (int k=k_start; k < k_end; k++) {

                // wait for buffers to be ready to write
                nvgpu::arch::tma_wait(empty_barrieres[write_stage], phase);

                load_once(
                    warpgroups_lane_id, group_id,
                    k, block_idx_m, block_idx_n, total_stage_bytes,
                    tma_desc_X, tma_desc_W,
                    shmem_X, shmem_W, barriers,
                    cluster_mask, cluster_group_m_rank, cache_hint_lhs, cache_hint_rhs,
                    write_stage, phase
                );

                write_stage = (write_stage + 1) % STAGES;
                if (write_stage == 0) {
                    phase ^= 1;
                }

            }

        } // load w/ cluster multicast

    };

} // namespace xpu
