#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace nvgpu {
namespace arch {

// NOTE (yiakwy) : use TMA multicast and L2 cache hint to optimize the TMA load for weight matrix.
__device__ __inline__ void tma2d_multicast_load_async(uint32_t smem_addr/*smem dest*/, const uint64_t tma_desc_addr/*gmem src*/,
                                                      uint32_t s_w_mbar_addr, /*mbarrier*/
                                                      const int32_t inner_dim_offset, const int32_t outter_dim_offset,
                                                      uint16_t cluster_mask, uint64_t cache_hint) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
        " [%0], [%1, {%3, %4}], [%2], %5, %6;\n"
        :: "r"(smem_addr), "l"(tma_desc_addr), "r"(s_w_mbar_addr),
        "r"(inner_dim_offset), "r"(outter_dim_offset), "h"(cluster_mask), "l"(cache_hint)
        // : "memory"
    );
}


__device__ __inline__ void tma2d_multicast_load_async(void* smem_ptr/*smem dest*/, void const * tma_desc_ptr/*gmem src*/,
                                                      uint64_t* s_w_mbar_ptr, /*mbarrier*/
                                                      const int32_t inner_dim_offset, const int32_t outter_dim_offset,
                                                      uint16_t cluster_mask, uint64_t cache_hint) {

    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    const uint64_t tma_desc_addr = reinterpret_cast<uint64_t>(tma_desc_ptr);
    uint32_t s_w_mbar_addr = __cvta_generic_to_shared(s_w_mbar_ptr);

    tma2d_multicast_load_async(smem_addr, tma_desc_addr, s_w_mbar_addr, inner_dim_offset, outter_dim_offset, cluster_mask, cache_hint);
}


__device__ __inline__ void tma2d_load_async(void* smem_ptr/*smem dest*/, void const * tma_desc_ptr/*gmem src*/,
                                            uint64_t* s_w_mbar_ptr, /*mbarrier*/
                                            const int32_t inner_dim_offset, const int32_t outter_dim_offset,
                                            uint64_t cache_hint) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    const uint64_t tma_desc_addr = reinterpret_cast<uint64_t>(tma_desc_ptr);
    uint32_t s_w_mbar_addr = __cvta_generic_to_shared(s_w_mbar_ptr);

    tma2d_load_async(smem_addr, tma_desc_addr, s_w_mbar_addr, inner_dim_offset, outter_dim_offset, cache_hint);
}

__device__ __inline__ void tma2d_load_async(uint32_t smem_addr/*smem dest*/, const uint64_t tma_desc_addr/*gmem src*/,
                                            uint32_t s_w_mbar_addr, /*mbarrier*/
                                            const int32_t inner_dim_offset, const int32_t outter_dim_offset,
                                            uint64_t cache_hint) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4}], [%2], %5;\n"
        :
        : "r"(smem_addr), "l"(tma_desc_addr), "r"(s_w_mbar_addr),
        "r"(inner_dim_offset), "r"(outter_dim_offset), "l"(cache_hint)
    );
}

__device__ __inline__ void tma2d_store(uint64_t tma_addr/*gmem dest*/, uint32_t smem_epilogue_addr /*smem src*/,
                                       const int32_t inner_dim_offset, const int32_t outter_dim_offset) {
    asm volatile (
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%2, %3}], [%1];"
        :
        : "l"(tma_addr), "r"(smem_epilogue_addr),
        "r"(inner_dim_offset), "r"(outter_dim_offset)
        : "memory"
    );
}

} // namespace arch
} // namespace nvgpu
