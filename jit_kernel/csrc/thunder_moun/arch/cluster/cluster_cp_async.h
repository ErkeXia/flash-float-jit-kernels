/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace nvgpu {
namespace arch {

// NOTE (yiakwy) : TMA bulk 1d copy (cp.async.bulk) via NoC (Dshmem)

static __device__ __forceinline__ void cluster_cp_async_bulk(
    void* dst_local_smem,
    const void* src_remote_smem,
    uint32_t bytes,
    uint64_t* s_mbar) {

    uint32_t dst_addr = __cvta_generic_to_shared(dst_local_smem);
    const uint32_t src_addr = __cvta_generic_to_shared(src_remote_smem);

    uint32_t mbar_addr = __cvta_generic_to_shared(s_mbar);

    asm volatile(
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :: "r"(dst_addr), "r"(src_addr), "r"(bytes), "r"(mbar_addr)
        : "memory"
    );
}


} // namespace arch
} // namespace nvgpu
