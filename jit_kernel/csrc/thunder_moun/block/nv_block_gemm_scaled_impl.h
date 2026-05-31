/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once
#include <cooperative_groups.h>

#include <cuda/barrier>

namespace cg = cooperative_groups;

#include "../tensor/tensor_view_ref.h"
#include "../fragment/nv_frag_gemm_scaled_impl.h"
#include "block.h"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

namespace cg = cooperative_groups;


#define USE_LINEAR_TO_TRIL_LAYOUT 1


__device__ bool bar_try_wait(uint32_t bar_ptr, int phase) {
  uint32_t success;
  asm volatile(
      "{\n\t"
      ".reg .pred P1; \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
      "selp.b32 %0, 1, 0, P1; \n\t"
      "}"
      : "=r"(success)
      : "r"(bar_ptr), "r"(phase));
  return success;
}


namespace xpu {

template <int BM, int BN, int BK, int STAGES>
struct HopperPersistentSplitKPipeline {

    // NOTE (yiakwy) : use cluster.map_shared_rank api to resolve the shared memory address from block-level view to cluster-level view.
    static __device__ inline uint32_t resolve_noc_cluster_smem(void* local_ptr, int target_block_rank) {
        uint32_t local_smem_addr = __cvta_generic_to_shared(local_ptr);
        uint32_t cluster_smem_addr;
        asm("mapa.shared.u32 %0, %1, %2;\n" : "=r"(cluster_smem_addr) : "r"(local_smem_addr), "r"(target_block_rank));
        return cluster_smem_addr;
    }

    // sm90+
    static __device__ inline void run_persistent(
        HopperWGMMAAccumulator<BM, BN, BK>& accum,
        const CUtensorMap* tma_desc_X,
        const CUtensorMap* tma_desc_W,
        const float* scale_X,
        const float* scale_W,
        half* Out,
        int M, int N, int K,
        int total_symmetric_tiles,
        int num_blocks_n,
        uint8_t* smem_buffer
    ) {
        using fp8_t = __nv_fp8_e4m3;
        using fp32_t = float;

        using AccDtype = fp32_t;


        const int tid = threadIdx.x;
        // const int lane_id = threadIdx.x % WARP_SIZE;
        // const int warp_id = threadIdx.x / WARP_SIZE;

        // prepare
        auto* shmem_X = reinterpret_cast<SharedBlock<fp8_t, BM, BK>*>(smem_buffer);
        auto* shmem_W = reinterpret_cast<SharedBlock<fp8_t, BN, BK>*>(smem_buffer + STAGES * sizeof(*shmem_X));

        int threads_per_block = blockDim.x;

        AccDtype* epilogue_smem = reinterpret_cast<AccDtype *>(smem_buffer + STAGES * (BM * BK + BN * BK));
        FragmentView<AccDtype, BM, BN, MemoryDomain::kShared> frag_view(epilogue_smem);

        constexpr uint32_t tma_bytes_X = BM * BK;
        constexpr uint32_t tma_bytes_W = BN * BK;

        constexpr uint32_t total_stage_bytes = tma_bytes_X + tma_bytes_W;

        constexpr int SCLAE_BLOCK_SIZE_K = 128;
        constexpr int K_TILES_TOTAL = (8092 + SCLAE_BLOCK_SIZE_K - 1) / SCLAE_BLOCK_SIZE_K;

        __shared__ __align__(128) float shmem_XS[BM * K_TILES_TOTAL];
        __shared__ __align__(128) float shmem_WS[K_TILES_TOTAL];

        // TODO (yiakwy) : init full barriers for TMA, in mult-stages pipeline, we combine writer and reader barrieres in the same stage into one
        __shared__ __align__(128) uint64_t barriers[STAGES];

        if (threadIdx.x == 0) {
            #pragma unroll
            for (int s = 0; s < STAGES; ++s) {
                uint32_t s_bar_ptr = __cvta_generic_to_shared(&barriers[s]);
                asm volatile("mbarrier.init.shared.b64 [%0], 1;\n" :: "r"(s_bar_ptr));
            }
        }

        // NOTE (yiakwy) : ensure fences visible to all threads
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();


        int split_k_id = blockIdx.x;
        int split_k = gridDim.x;

        const int k_tiles_total = (K + BK - 1) / BK;

        int k_tiles_per_slice = (k_tiles_total + split_k - 1) / split_k;

        int k_start = split_k_id * k_tiles_per_slice;
        int k_end = min(k_tiles_total, (split_k_id + 1) * k_tiles_per_slice);

        alignas(128) __shared__ int local_task_id;
        if (threadIdx.x == 0) {
            local_task_id = blockIdx.y;
        }
        __syncthreads();


        // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("[Split#%d] [SM#%d] split_k = %d, k_tiles_total = %d, k_tiles_per_slice = %d, k_start=%d, k_end=%d\n", blockIdx.x, blockIdx.y, split_k, k_tiles_total, k_tiles_per_slice, k_start, k_end );
        // }
        // __syncthreads();

        while (local_task_id < total_symmetric_tiles) {
            accum.clear();

            // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
            //     printf("[Split#%d] [SM#%d] enter into persistent loop ...\n", blockIdx.x, blockIdx.y );
            // }

// NOTE (yiakwy) : we only support symmetric gemm, hence force to use linear to triangular mapping for block-level tile assignment.
// TODO (yiakwy) : precompute the block_idx_m and block_idx_n for each local_task_id and store in shared memory to avoid redundant computation on the fly and reduce the latency.
#ifdef USE_LINEAR_TO_TRIL_LAYOUT
            int block_idx_m = int((sqrt(8.0 * local_task_id + 1.0) - 1.0) / 2.0);
            int block_idx_n = local_task_id - (block_idx_m * (block_idx_m + 1)) / 2;
#else
            int block_idx_m = local_task_id / num_blocks_n;
            int block_idx_n = local_task_id % num_blocks_n;
#endif

            // if (threadIdx.x == 0 && blockIdx.x == 0) {
            //     printf("[Split#%d] [SM#%d] block#(%d, %d) initiate TAM loading ...\n", blockIdx.x, blockIdx.y, block_idx_m, block_idx_n);
            // }

            int write_stage = 0;
            int read_stage = 0;

            // 1. Ramp Up Fill : to initiate the pipeline, we will fill STAGES-1 stages of data before entering the main loop, and then maintain 1 stage ahead of the main loop to keep the pipeline full.
            #pragma unroll
            for (int i = 0; i < STAGES - 1; ++i) {
                int current_k = k_start + i;
                if (current_k < k_end) {

                    // if (threadIdx.x == 0 && blockIdx.y == 0) {
                    //     printf("[Prefetch] [Split#%d] [SM#%d] block#(%d, %d) Ramp up fill, stage#%d/%d\n", blockIdx.x, blockIdx.y, block_idx_m, block_idx_n, i, STAGES);
                    // }

                    // Commit the async copy for the first few stages, and then we will wait on them in the main loop
                    // NOTE (yiakwy) : see https://github.com/NVIDIA/cutlass/blob/5f06f5fc1a072bbe4815fae7ae8470b876ed603a/include/cute/arch/copy_sm90_tma.hpp#L117 for the recommended way to commit async copy with TMA and mbarrier synchronization.
                    // TODO (yiakwy) : use cache hint
                    if (tid == 0) {
                        uint32_t smem_x_addr = __cvta_generic_to_shared(&shmem_X[write_stage]);
                        uint32_t smem_w_addr = __cvta_generic_to_shared(&shmem_W[write_stage]);

                        uint32_t s_w_bar_ptr = __cvta_generic_to_shared(&barriers[write_stage]);

                        // NOTE (yiakwy) : call mbarrier.arrive once
                        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(s_w_bar_ptr), "r"(total_stage_bytes));

                        // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
                        //     printf("smem_x_addr : 0x%x (mod 128 = %d)\n", smem_x_addr, (int)(smem_x_addr % 128));
                        //     printf("smem_w_addr : 0x%x (mod 128 = %d)\n", smem_w_addr, (int)(smem_w_addr % 128));
                        // }

                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%3, %4}], [%2];\n"
                            :
                            : "r"(smem_x_addr), "l"(reinterpret_cast<uint64_t>(tma_desc_X)), "r"(s_w_bar_ptr),
                              "r"(current_k * BK), "r"(block_idx_m * BM)
                            : "memory"
                        );

                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%3, %4}], [%2];\n"
                            :: "r"(smem_w_addr), "l"(reinterpret_cast<uint64_t>(tma_desc_W)), "r"(s_w_bar_ptr),
                            "r"(current_k * BK), "r"(block_idx_n * BN)
                        );
                    } //  end of thread 0

                    write_stage = (write_stage + 1) % STAGES;
                }
            }

            // if (threadIdx.x == 0 && blockIdx.x == 0) {
            //     printf("[Prefetch] [Split#%d] [SM#%d] block#(%d, %d) enter into main loop ...\n", blockIdx.x, blockIdx.y, block_idx_m, block_idx_n);
            // }
            // __syncthreads();

            // NOTE (yiakwy) : prefetch all scale without TMA
            // NOTE (yiakwy) : rows of 1 or more BLOCKS share a scale
            const int stride_xs_m = K / SCLAE_BLOCK_SIZE_K;
            const int stride_ws_n = K / SCLAE_BLOCK_SIZE_K;

            constexpr int shares_per_scale = SCLAE_BLOCK_SIZE_K / BK;
            // static_assert(shares_per_scale == 2, "with BK=%d, each scale will be shared by 2 tiles, please adjust SCLAE_BLOCK_SIZE_K or BK to ensure that.");

            const int k_tiles = k_end - k_start;
            const int total_xs_elements = BM * k_tiles;
            const int total_ws_elements = k_tiles;

            #pragma unroll 4
            for (int i = tid; i < total_xs_elements; i += threads_per_block) {
                int s_row = i / k_tiles;
                int s_col = i % k_tiles;

                int g_row = block_idx_m * BM + s_row;
                int g_col = (k_start + s_col) / shares_per_scale;

                // shmem_XS[s_row * k_tiles + s_col] = scale_X[g_row * stride_xs_m + g_col];
                shmem_XS[s_col * BM + s_row] = scale_X[g_row * stride_xs_m + g_col];
            }
            __syncthreads();

            // TODO (yiakwy) : remap shmem_XS to per-thread registers to reduce the latency, since the scale load is on the critical path of the main loop.
            #pragma unroll 4
            for (int i = tid; i < total_ws_elements; i += threads_per_block) {
                int s_col = i;

                int g_row = block_idx_n;
                int g_col = (k_start + s_col) / shares_per_scale;

                shmem_WS[s_col] = scale_X[g_row * stride_ws_n + g_col];
            }
            __syncthreads();

            // TODO (yiakwy) : remap shmem_WS to per-thread registers to reduce the latency, since the scale load is on the critical path of the main loop.
            int tma_phase = 0;

            // 2. main loop
            for (int k_tile = k_start; k_tile < k_end; ++k_tile) {
                uint32_t current_barrier = __cvta_generic_to_shared(&barriers[read_stage]);

                // if (threadIdx.x == 0 && blockIdx.x == 1) {
                //     printf("[MainLoop] [Split#%d] [SM#%d] block#(%d, %d) k_tile=%d, k_start=%d, k_end=%d\n", blockIdx.x, blockIdx.y, block_idx_m, block_idx_n, k_tile, k_start, k_end);
                // }

                if (threadIdx.x == 0) {
                    // NOTE (yiakwy) : wait parity switch from phase (1 at prfetch stage) to ^phase (0 when TMA finish stage 0 transactions)
                    /*
                    while (!bar_try_wait(current_barrier, tma_phases[read_stage])) {
                        asm volatile("nanosleep.u32 64;\n");
                    }
                    */
                    asm volatile(
                        "{\n"
                        ".reg .pred P;\n"
                        "WAIT_LOOP:\n"
                        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
                        "@P bra DONE;\n"
                        "nanosleep.u32 64;\n"
                        "bra WAIT_LOOP;\n"
                        "DONE:\n"
                        "}\n"
                        :: "r"(current_barrier), "r"(tma_phase) : "memory"
                    );
                }
                __syncthreads();

                // if (threadIdx.x == 0 && blockIdx.x == 0) {
                //     printf("[Split#%d/%d] [SM#%d] block#(%d, %d) k_tile=%d, inputs are ready.\n", blockIdx.x, gridDim.x, blockIdx.y, block_idx_m, block_idx_n, k_tile);
                // }
                // __syncthreads();

                HopperWGMMAAccumulator<BM, BN, BK> local_step_accum;
                local_step_accum.clear();
                // asm volatile("" ::: "memory");

                uint32_t active_smem_x = __cvta_generic_to_shared(&shmem_X[read_stage]);
                uint32_t active_smem_w = __cvta_generic_to_shared(&shmem_W[read_stage]);

                // NOTE (yiakwy) : hopper (SM90a) does not support mma_scaled instruction, sx, sw will be ignored in the current implementation, and the scaling will be applied in the epilogue.
                HopperWGMMAExecutor::mma_scaled(local_step_accum, active_smem_x, active_smem_w);

                int next_k = k_tile + (STAGES - 1);
                if (next_k < k_end) {
                    if (threadIdx.x == 0) {
                        uint32_t smem_x_addr = __cvta_generic_to_shared(&shmem_X[write_stage]);
                        uint32_t smem_w_addr = __cvta_generic_to_shared(&shmem_W[write_stage]);

                        uint32_t s_w_bar_ptr = __cvta_generic_to_shared(&barriers[write_stage]);

                        // if (threadIdx.x == 0 && blockIdx.y == 0) {
                        //     printf("[Prefetch] [Split#%d] [SM#%d] block#(%d, %d) load k tile %d, stage#%d\n", blockIdx.x, blockIdx.y, block_idx_m, block_idx_n, next_k, write_stage);
                        // }

                        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(s_w_bar_ptr), "r"(total_stage_bytes));

                        // TODO (yiakwy) : add L2 cache locality .L2::cache_hint
                        // TODO (yiakwy) : add multicast support .multicast::cluster

                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%3, %4}], [%2];\n"
                            :: "r"(smem_x_addr), "l"(reinterpret_cast<uint64_t>(tma_desc_X)), "r"(s_w_bar_ptr),
                            "r"(next_k * BK), "r"(block_idx_m * BM)
                        );

                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%3, %4}], [%2];\n"
                            :: "r"(smem_w_addr), "l"(reinterpret_cast<uint64_t>(tma_desc_W)), "r"(s_w_bar_ptr),
                            "r"(next_k * BK), "r"(block_idx_n * BN)
                        );
                    }
                    write_stage = (write_stage + 1) % STAGES;
                }

                HopperWGMMAExecutor::commit_and_wait();
                // asm volatile("" ::: "memory");

                local_step_accum.mul_(&shmem_XS[0], &shmem_WS[0], k_tile);
                accum.add_(local_step_accum);
                __syncwarp();

                read_stage = (read_stage + 1) % STAGES;
                if (read_stage == 0) {
                    tma_phase ^= 1;
                }
            }

            // 3. Epilogue
            //   - first write data back to share memory for SPLIT-K reduction via NoC
            //   - applying successive operations upon tile results in the epilogue, such as bias add, activation, etc, can be fused in this step to save memory bandwidth.
            accum.store(epilogue_smem);
            __syncthreads();

            // if (threadIdx.x == 0 && blockIdx.x == 1) {
            //     printf("[Epilogue] [Split#%d] [SM#%d] write block <%d, %d> back to shared memory...\n", blockIdx.x, blockIdx.y, block_idx_m, block_idx_n);
            // }
            // __syncthreads();


#if __CUDA_ARCH__ >= 900 && ENABLE_HOPPER // Hopper 900+ GPU with TMA support
            // if (threadIdx.x == 0 && blockIdx.x == 1) {
            //     printf("[Epilogue] [Split#%d] [SM#%d] write split_k#%d block <%d, %d> on-chip reduce via NoC...\n", blockIdx.x, blockIdx.y, split_k, block_idx_m, block_idx_n);
            // }
            // __syncthreads();

            auto cluster = cooperative_groups::this_cluster();
            cluster.sync();

            if (split_k > 1) {
                if (split_k_id == 0) {
                    for (int r = 1; r < split_k; ++r) {
                        AccDtype* dst_epilogue_smem = cluster.map_shared_rank<AccDtype>(&epilogue_smem[0], r);

                        for (int idx = tid; idx < BM * BN; idx += threads_per_block) {
                            // // if (blockIdx.y == 1 && idx == (BM - 1) * BN) {
                            // if (blockIdx.y == 2 && idx == BM * BN - 1) {
                            //     printf("[Epilogue] [idx#%d] [Split#%d] [SM#%d] block#(%d, %d) read from split_k#%d partial result for reduction: epilogue_smem[idx]=%f, dst_epilogue_smem[idx]=%f\n", threadIdx.x, split_k_id, blockIdx.y, block_idx_m, block_idx_n, r, epilogue_smem[idx], dst_epilogue_smem[idx]);
                            // }
                            // __syncthreads();
                            atomicAdd(&epilogue_smem[idx], dst_epilogue_smem[idx]);
                        }
                    }
                }
            } //  split_k > 1
            cluster.sync();

            if (split_k_id == 0) {
                // write to lower left
                for (int idx = threadIdx.x; idx < BM * BN; idx += threads_per_block) {
                    int local_m = idx / BN;
                    int local_n = idx % BN;

                    // float val = frag_view(local_m, local_n);
                    float val = static_cast<float>(epilogue_smem[idx]);

                    // // if (blockIdx.y == 1 && idx == (BM - 1) * BN) {
                    // if (blockIdx.y == 2 && idx == BM * BN - 1) {
                    //     printf("[Epilogue] [idx#%d] [Split#%d] [SM#%d] block#(%d, %d) read from partial result for reduction: epilogue_smem[idx]=%f, val[%d, %d]=%f\n", threadIdx.x, split_k_id, blockIdx.y, block_idx_m, block_idx_n, epilogue_smem[idx], local_m, local_n, val);
                    // }
                    // __syncthreads();

                    int global_m = block_idx_m * BM + local_m;
                    int global_n = block_idx_n * BN + local_n;
                    if (global_m < M && global_n < N) {
                        Out[global_m * N + global_n] = static_cast<half>(val);
                    }
                }
                __syncthreads();

                // transpose copy to upper right
                if (block_idx_m > block_idx_n) {
                    for (int idx = threadIdx.x; idx < BM * BN; idx += threads_per_block) {
                        int local_m = idx / BN;
                        int local_n = idx % BN;

                        // float val = frag_view(local_m, local_n);
                        half val = static_cast<half>(epilogue_smem[idx]);

                        int sym_global_m = block_idx_n * BN + local_n;
                        int sym_global_n = block_idx_m * BM + local_m;
                        if (sym_global_m < M && sym_global_n < N) {
                            Out[sym_global_m * N + sym_global_n] = val;
                        }
                    }
                }
                __syncthreads();
            }
            cluster.sync();
#else
            for (int idx = threadIdx.x; idx < BM * BN; idx += threads_per_block) {
                int local_m = idx / BN;
                int local_n = idx % BN;

                // float val = frag_view(local_m, local_n);
                half val = static_cast<half>(epilogue_smem[idx]);

                // write to lower left
                int global_m = block_idx_m * BM + local_m;
                int global_n = block_idx_n * BN + local_n;
                if (global_m < M && global_n < N) {
                    if (split_k > 1) {
                        atomicAdd(&Out[global_m * N + global_n], val);
                    } else {
                        Out[global_m * N + global_n] = val;
                    }
                } // end of write to lower left

                // tranpose copy to upper right
                if (block_idx_m > block_idx_n) {
                    int sym_global_m = block_idx_n * BN + local_n;
                    int sym_global_n = block_idx_m * BM + local_m;
                    if (sym_global_m < M && sym_global_n < N) {
                        if (split_k > 1) {
                            atomicAdd(&Out[sym_global_m * N + sym_global_n], val);
                        } else {
                            Out[sym_global_m * N + sym_global_n] = val;
                        }
                    }
                } // end of tranpose copy to upper right
            }
            __syncthreads();
#endif // __CUDA_ARCH__ >= 900 && ENABLE_HOPPER

            // fetch next task
            if (threadIdx.x == 0) {
                local_task_id += gridDim.y;
            }
            __syncthreads();

        } // while
    }
};

} // namespace xpu
