---
name: hopper-kernel
description: Hopper (SM 90+) CUDA kernel development patterns used in this repo. Covers WGMMA tile-level MMA (FP8/FP16), TMA producer/consumer pipelines (multi-stage, 1p2c, pingpong), mbarrier full/empty barrier synchronization across producer/consumer warp groups and NoC clusters, cluster multicast for L2 efficiency, and on-chip split-K reduction via NoC. Use when writing or modifying .cu files in jit_kernel/csrc/thunder_moun/.
---

# Hopper Kernel Development (SM 90+)

## Key Principle
This repo targets Hopper (H100/H800) exclusively for complex kernels. We do NOT fall back to Ampere.
All patterns below are Hopper-only. For Ampere/Turing, use the Triton fallback kernels.
We write zero-cutlass / zero-cute code: every abstraction (fragment, TMA, mbarrier,
WGMMA) is a thin header-only wrapper over raw PTX. See `fragment/nv_frag_gemm_scaled_impl.h`,
`arch/tma/*`, `block/nv_block_1p2c_gemm_scaled_impl.h` (and `-ref` copy).

## Target Platform
Kernels run on the H800 SuperPod connected via IB9700 (NDR 400G InfiniBand).
- H800 SXM compute peaks equal H100: 989.5 TFLOPS (FP16 dense), 3,352 GB/s (HBM3)
- **NVLink is halved vs H100: 400 GB/s** — intra-node multi-GPU sharing via NVLink is limited; design for cross-node communication over IB9700 where possible

## Kernel Entry: TVM-FFI

Use TVM FFI, NOT torch.utils.cpp_extension for production kernels.

### C++ Side
```cpp
#define USE_TVM_FFI 1
#ifdef USE_TVM_FFI
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#endif

extern "C" int my_kernel_entry(
    const void* __restrict__ X,
    void* __restrict__ Out,
    int M, int N,
    cudaStream_t stream = 0)
{
    // Setup and launch
    return 0;
}

#ifdef USE_TVM_FFI
namespace tvm_c_loader {
using namespace tvm::ffi;
void tvm_jit_launch(TensorView X, TensorView Out, int M, int N) {
    DLDevice device = X.device();
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(device.device_type, device.device_id));
    my_kernel_entry(X.data_ptr(), Out.data_ptr(), M, N, stream);
}
} // namespace tvm_c_loader
TVM_FFI_DLL_EXPORT_TYPED_FUNC(my_kernel, (tvm_c_loader::tvm_jit_launch));
#endif
```

### Python Side
```python
from tvm_ffi.cpp import load_inline

module = load_inline(
    "my_kernel",
    cuda_sources=['#include "path/to/kernel.cu"'],
    extra_cuda_cflags=["-O2", "-std=c++17", "-DENABLE_HOPPER=1",
                       "-arch=compute_90a", "-code=sm_90a"],
    extra_ldflags=["-lcudart", "-lcuda"],
)
module.my_kernel(x_tensor, out_tensor, M, N)
```

## WGMMA Tile-Level MMA (m64n128k32)

Hopper's tensor core MMA is `wgmma.mma_async`. A **warp group** (4 warps = 128
threads) cooperatively computes one `m64n128k32` fragment. It consumes A and B
**from shared memory via 64-bit SMEM descriptors** and accumulates into per-thread
registers. This is the FP8 path we use for scaled GEMM.

### Fragment Abstraction (`fragment/nv_frag_gemm_scaled_impl.h`)
```cpp
template <int BM, int BN, int BK>
struct HopperWGMMAAccumulator {
    static constexpr int FRAG_M = 64, FRAG_N = 128, FRAG_K = 32;
    static constexpr int kRegistersPerThread = 64;   // BM*BN/128 per warp group
    float regs[MAX_M_STEPS][64];
    void clear();
    void mul_(float scale);
    void mul_(float* xs, float* ws, int k_offset);    // per-block FP8 scale
    void add_(const HopperWGMMAAccumulator& b);       // on-chip split-K reduction
    int  getTargetWgmmaSmemOffset(...);               // reg <-> smem layout
    template<typename Dtype> void store(Dtype* smem);
};

struct HopperWGMMAExecutor {
    template <typename AccType>
    static void mma_scaled(AccType& accum, uint32_t smem_x_ptr, uint32_t smem_w_ptr);
    static void commit_and_wait();  // wgmma.commit_group + wait_group 0
};
```

### SMEM Descriptor Encoding
WGMMA reads A/B from SMEM using a 64-bit descriptor, NOT a raw address. We encode
it by hand (see `make_smem_desc` / `TmaDesc`):
```
bits  0-13  : start address          (>> 4)
bits 16-29  : leading byte offset    (>> 4)
bits 32-45  : stride byte offset     (>> 4)   e.g. 8 * BK (swizzle 128B)
bits 62-63  : layout type (SWIZZLE_128B = 1)
```
```cpp
static __device__ inline uint64_t make_smem_desc(
    uint32_t s_addr, const int layout_type,
    const uint32_t& leading_byte_offset = 16,
    const uint32_t& stride_byte_offset = 512) {
    TmaDesc desc;
    desc.bitfield.start_address_ = s_addr >> 4;
    desc.bitfield.leading_byte_offset_ = leading_byte_offset >> 4;
    desc.bitfield.stride_byte_offset_ = stride_byte_offset >> 4;
    desc.bitfield.base_offset_ = 0;
    desc.bitfield.layout_type_ = layout_type;
    return desc.desc_;
}
```
The A and B tiles must be swizzled in SMEM (128B swizzle) so that WGMMA's
descriptor-based load is bank-conflict free.

### Issuing the MMA (FP8 example)
```cpp
// 1. Fence
asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

// 2. For each (m_step, n_step, k_step): build descriptors and issue
uint64_t desc_x = make_smem_desc(addr_x, /*layout=*/1, 0, swizzle_stride_x);
uint64_t desc_w = make_smem_desc(addr_w, /*layout=*/1, 0, swizzle_stride_w);

asm volatile(
    "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
    "{ %0, %1, ..., %63 }, %64, %65, 1, 1, 1;\n"
    : "+f"(reg[0]), ..., "+f"(reg[63])
    : "l"(desc_x), "l"(desc_w));

// 3. Commit / wait
asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
```
Notes:
- FP8 `e4m3` block scaling is **not** a single `mma_scaled` instruction on SM90a —
  we multiply the per-k-block scale in the epilogue (`local_step_accum.mul_(scale_X, scale_W, k)`).
- A block that needs `BM=128` runs `MAX_M_STEPS=2` fragments per warp group, or
  multiple warp groups (`wgs = blockDim.x / 128`).

## TMA: Three Pipeline Writing Styles

TMA (`cp.async.bulk.tensor.2d`) is the Hopper+ bulk data mover. It is used by the
**producer warp group** to stream A/B tiles (and later the epilogue store) while the
**consumer warp groups** compute with WGMMA. We use three styles:

| Style | Producer | Consumers | Description |
|-------|----------|-----------|-------------|
| **multi-stage** | same CTA threads issue TMA (`tid==0`) | all threads | classic N-stage software pipeline, producer/consumer interleaved in the main loop |
| **1p2c** | dedicated producer warp group | 2 consumer warp groups | true async producer/consumer split, different register budgets |
| **pingpong** | STAGES=2, parity toggles | all consumer threads | minimal SMEM, strict alternation producer/consumer |

### Barrier Model (full / empty barriers)
Use two mbarrier sets per stage:
- `barriers[s]` — *full* barrier: TMA completes writing stage s (`expect_tx` bytes).
  Consumers wait on it before reading.
- `empty_barriers[s]` — *empty* barrier: consumers arrived after consuming stage s;
  producer waits on it before overwriting.

```cpp
__shared__ __align__(128) uint64_t barriers[STAGES];        // full
__shared__ __align__(128) uint64_t empty_barriers[STAGES];  // empty

if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) {
        tma_init_barrier<USE_CLUSTER_MULTICAST>(&barriers[s], 1);
        tma_init_barrier<USE_CLUSTER_MULTICAST>(&empty_barriers[s],
                                                CLUSTER_SIZE_M * CONSUMER_WARPGROUPS);
    }
}
tma_store_fence();
__syncthreads();
```
With multicast, full barriers use count = number of CTAs that receive the multicast;
empty barriers use count = CLUSTER_SIZE_M x number of consumer warp groups.

### Case A — multi-stage pipeline (ramp-up -> main loop -> epilogue)
Producer fills `STAGES-1` stages first, then the main loop keeps 1 stage ahead:
```cpp
// ramp-up: issue STAGES-1 TMA loads
#pragma unroll
for (int i = 0; i < STAGES - 1; ++i) {
    if (k_start + i < k_end) {
        if (tid == 0) {
            uint32_t s_w_bar_ptr = __cvta_generic_to_shared(&barriers[write_stage]);
            tma_expect_bytes(s_w_bar_ptr, total_stage_bytes);
            tma2d_load_async(smem_x_addr, tma_desc_X, s_w_bar_ptr,
                             (k_start + i) * BK, block_idx_m * BM, cache_hint);
            tma2d_load_async(smem_w_addr, tma_desc_W, s_w_bar_ptr,
                             (k_start + i) * BK, block_idx_n * BN, cache_hint);
        }
        write_stage = (write_stage + 1) % STAGES;
    }
}

// main loop
for (int k_tile = k_start; k_tile < k_end; ++k_tile) {
    if (tid == 0) tma_wait(__cvta_generic_to_shared(&barriers[read_stage]), tma_phase);
    __syncthreads();

    HopperWGMMAExecutor::mma_scaled(local_step_accum, smem_x(read_stage), smem_w(read_stage));
    HopperWGMMAExecutor::commit_and_wait();

    // prefetch next stage
    int next_k = k_tile + (STAGES - 1);
    if (next_k < k_end && tid == 0) { tma_expect_bytes(...); tma2d_load_async(...); }
    if (next_k < k_end) write_stage = (write_stage + 1) % STAGES;

    local_step_accum.mul_(&shmem_XS[0], &shmem_WS[0], k_tile - k_start);
    accum.add_(local_step_accum);

    read_stage = (read_stage + 1) % STAGES;
    if (read_stage == 0) tma_phase ^= 1;   // parity toggles on wrap
}
```
Phase toggling: `mbarrier.try_wait.parity` flips each time the barrier is reused.
Track `tma_phase` and XOR with 1 whenever `read_stage` wraps to 0.

### Case B — 1p2c pipeline (dedicated producer warp group)
Split threads into 1 producer warp group + 2 consumer warp groups
(see `block/nv_block_1p2c_gemm_scaled_impl.h`):
```cpp
const int wg_id = warp_id / WARP_GROUP;
bool is_consumer = wg_id < CONSUMER_WARPGROUPS;   // 2 warp groups compute
bool is_producer = !is_consumer;                   // 1 warp group loads
```

**Producer side** (leader thread = `CONSUMER_THREADS`):
```cpp
nvgpu::arch::reg_dealloc_decrease_registers<40>();  // producer needs fewer regs
if (warp_id == CONSUMER_WARPS && is_leader_thr_in_wgs) {
    // prefetch TMA descriptors (optional, reduces latency)
    asm volatile("prefetch.tensormap [%0];" :: "l"(gmem_int_desc) : "memory");
    while (local_task_id < total_symmetric_tiles) {
        // wait empty_barriers[write_stage] (producer may not overwrite a busy stage)
        producer<STAGES, GROUP_SIZE_M, BM, BN, BK,
                 USE_CLUSTER_MULTICAST, USE_LINEAR_TO_TRIL_LAYOUT>::load(
            tid, group_id, block_idx_m, block_idx_n,
            k_start, k_end, total_stage_bytes,
            tma_desc_X, tma_desc_W, shmem_X, shmem_W, barriers, empty_barriers,
            cluster_mask, cluster_group_m_rank, cache_hint_lhs, cache_hint_rhs,
            write_stage, tma_phase);
        local_task_id += gridDim.y;
    }
}
```

**Consumer side** (each consumer warp group, leader thread = `tid == 0`):
```cpp
nvgpu::arch::reg_alloc_increase_registers<232>();   // consumers need many regs
for (int k_tile = k_start; k_tile < k_end; ++k_tile) {
    if (threadIdx.x == 0)
        tma_wait(__cvta_generic_to_shared(&barriers[read_stage]), tma_phase);
    warpgroup_sync();                       // barrier.sync across the 2 consumer groups

    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    HopperWGMMAExecutor::mma_scaled(local_step_accum, smem_x(read_stage), smem_w(read_stage));
    HopperWGMMAExecutor::commit_and_wait();

    // notify producer this stage is free (cluster release with multicast)
    if (wg_lane_id < CLUSTER_SIZE_M) {
        mbar_arrive_cluster_release(&empty_barriers[read_stage], wg_lane_id);
    }
    local_step_accum.mul_(shmem_XS, shmem_WS, k_tile - k_start);
    accum.add_(local_step_accum);
    warpgroup_sync();

    read_stage = (read_stage + 1) % STAGES;
    if (read_stage == 0) tma_phase ^= 1;
}
```
Key: producer and consumer run **concurrently** in the same CTA. Only the producer
waits on `empty_barriers`; only consumers wait on `barriers`. Sync between consumer
warp groups is `barrier.sync N` (compile-time constant) — see `warpgroup_sync<128>`.

### Case C — pingpong pipeline (STAGES = 2)
Same producer/consumer structure but with only 2 stages and strict alternation.
Each stage toggles parity every iteration; because there are exactly 2 stages,
`tma_phase ^= 1` every iteration instead of only on wrap.
- SMEM usage is minimal: `2 * (BM*BK + BN*BK) + epilogue`.
- Overlap is one full stage behind (no deeper lookahead), so it is best when SMEM
  is the constraint.

## Synchronization Across Producer / Consumer / NoC Cluster

`cp.async.bulk` (TMA) operations complete asynchronously. Correctness requires
fencing between the async proxy and the generic/async proxy of threads:

```cpp
// after init, before any TMA issue:
asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
__syncthreads();

// before TMA store results are consumed by other CTAs (NoC reduction):
asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

// TMA store wait:
nvgpu::arch::tma_store_wait();  // cp.async.bulk.wait_group.read 0
```

Cluster-wide sync (`cooperative_groups::this_cluster()`):
```cpp
auto cluster = cooperative_groups::this_cluster();
cluster.sync();                    // barrier.cluster.arrive + wait
uint16_t cluster_mask = ...;       // bitmask of CTA ranks for multicast
```

Cross-CTA mbarrier arrive (NoC release) — resolve remote SMEM and arrive:
```cpp
static __device__ __forceinline__ uint32_t mapa_shared_cluster(
    uint32_t local_addr, uint32_t cta_rank) {
    uint32_t mapped;
    asm("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(mapped) : "r"(local_addr), "r"(cta_rank));
    return mapped;
}

static __device__ __forceinline__ void mbar_arrive_cluster_release(
    uint64_t* bar, uint32_t cta_rank) {
    const uint32_t mapped = mapa_shared_cluster(__cvta_generic_to_shared(bar), cta_rank);
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0], 1;"
                 ::"r"(mapped));
}
```

## Split-K On-Chip Reduction via NoC (no atomicAdd)

With split-K, each CTA (gridDim.x = split_k) computes a partial sum. The
`split_k_id == 0` CTA reduces all partials on chip through the NoC using
`cluster.map_shared_rank`, then stores once:

```cpp
// epilogue barriers
__shared__ __align__(128) uint64_t epilogue_barriers[1];        // partials arrived
__shared__ __align__(128) uint64_t epilogue_readable_barriers[1];
if (split_k > 1 && threadIdx.x == 0) {
    tma_init_barrier<false>(&epilogue_barriers[0], split_k - 1);   // others arrive
    tma_init_barrier<false>(&epilogue_readable_barriers[0], 1);
}
tma_store_fence();
__syncthreads();

// non-zero splits arrive on the leader CTA's barrier
if (split_k_id > 0) {
    if (threadIdx.x == 0)
        mbar_arrive_cluster_release(&epilogue_barriers[0], /*cta_rank=*/0);
}

// split_k_id == 0: wait, read remote smem via NoC, accumulate
if (split_k_id == 0) {
    if (threadIdx.x == 0) tma_wait(__cvta_generic_to_shared(&epilogue_barriers[0]), epilogue_phase);
    warpgroup_sync();
    epilogue_phase ^= 1;

    if (threadIdx.x == 0)
        for (int r = 1; r < split_k; ++r)
            dst[r] = cluster.map_shared_rank<OutDtype>(&shmem_epilogue[0], r);
    warpgroup_sync();

    for (int r = 1; r < split_k; ++r)
        for (int idx = tid; idx < BM * BN; idx += CONSUMER_THREADS)
            shmem_epilogue[idx] += dst[r][idx];   // on-chip add
    warpgroup_sync();

    // single TMA store of the reduced tile
    if (threadIdx.x == 0)
        cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group
            [tma_o_addr, {block_idx_n*BN, block_idx_m*BM}], [smem_epilogue_addr];
}
```
The `-ref` copy (`nv_block_1p2c_gemm_scaled_impl.h`) also demonstrates the
`cp.async.bulk.shared::cluster.shared::cta` variant that copies a remote partial
into local SMEM before adding (`cluster_cp_async_bulk`).

## L2 Cache Efficiency: Task Swizzling + NoC Cluster Multicast

We improve L2 reuse in two stacked layers:

1. **Task swizzling** (base layer, also described in the cuda-kernel skill):
   linear task id -> lower-triangular mapping, then group swizzle with
   `GROUP_SIZE_M` so consecutive CTAs consume overlapping operand rows.
   ```cpp
   get_block_indices_tri_linear_swizzled<GROUP_SIZE_M>(local_task_id, block_idx_m, block_idx_n, num_blocks_m, group_id);
   ```

2. **NoC cluster multicast** (Hopper-only, on top of the swizzle): instead of every
   CTA issuing its own TMA load of the *shared* operand (e.g. the W/weight tile reused
   by CTAs in the same column group), a **single CTA** (`cluster_group_m_rank == 0`)
   issues a multicast TMA load once, and the data is delivered to all CTAs in the
   cluster mask. This cuts redundant global->L2/L2->SMEM traffic and boosts L2 hit rate.

```cpp
// producer: build cluster mask over the GROUP_SIZE_M column group
uint16_t cluster_mask = 0;
for (int r = 0; r < clusterDim.y; ++r) {
    int target_rank = split_k_id + r * num_splits;
    cluster_mask |= (1 << target_rank);
}

// multicast load of the shared operand (W) — issued only by rank 0 of the group
if (cluster_group_m_rank == 0) {
    nvgpu::arch::tma2d_multicast_load_async(
        smem_w_addr, reinterpret_cast<uint64_t>(tma_desc_W), s_w_bar_ptr,
        current_k * BK, block_idx_n * BN, cluster_mask, cache_hint_rhs);
} else {
    // others in the group get W delivered via multicast; still load A locally
    tma2d_load_async(smem_x_addr, reinterpret_cast<uint64_t>(tma_desc_X), s_w_bar_ptr,
                     current_k * BK, block_idx_m * BM, cache_hint_lhs);
}
```
PTX: `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint`.

Combine with L2 cache hints (`CacheHintSm90`): EVICT_NORMAL for streaming operands,
EVICT_LAST for the operand you want to keep resident in L2.

## mbarrier Pipeline Synchronization

```cpp
// 1. Init barriers (thread 0 only)
for (int s = 0; s < STAGES; ++s) {
    uint32_t bar_ptr = __cvta_generic_to_shared(&barriers[s]);
    asm volatile("mbarrier.init.shared.b64 [%0], 1;\n" :: "r"(bar_ptr));
}

// 2. Fence to ensure visibility
asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
__syncthreads();

// 3. Wait with parity (toggle on each stage refill)
int tma_phase = 0;
// ...
asm volatile(
    "{\n"
    ".reg .pred P;\n"
    "WAIT_LOOP:\n"
    "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
    "@P bra DONE;\n"
    "nanosleep.u32 64;\n"
    "bra WAIT_LOOP;\n"
    "DONE:\n"
    "}\n" :: "r"(bar_ptr), "r"(tma_phase) : "memory"
);

// 4. Toggle phase when wrap around
if (read_stage == 0) {
    tma_phase ^= 1;
}
```

## Shared Memory Layout

```cpp
// 128-byte aligned blocks
template <typename T, int M, int N>
struct alignas(128) SharedBlock {
    T data[M * N];
};

// Multi-stage buffer layout in smem:
// [stage0_X | stage0_W | stage1_X | stage1_W | ... | epilogue]
uint32_t required_smem_bytes =
    sizeof(SharedBlock<fp8_t, BM, BK>) * STAGES +   // X buffers
    sizeof(SharedBlock<fp8_t, BN, BK>) * STAGES +   // W buffers
    sizeof(SharedBlock<fp16_t, BM, BN>);              // epilogue

cudaFuncSetAttribute(kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    required_smem_bytes);
```

## Register Management (1p2c)

Producer warp group and consumer warp groups have different register needs. Use
`setmaxnreg` so each group gets a different budget:
```cpp
// producer: fewer registers
nvgpu::arch::reg_dealloc_decrease_registers<40>();
// consumers: many registers (accumulator + operands)
nvgpu::arch::reg_alloc_increase_registers<232>();
```
This lets a 3-warp-group block (128 producer + 256 consumer threads) fit in fewer
SMs and improves occupancy.

## Persistent Kernel Pattern

```cpp
// Launch with gridDim.y = total_tiles, gridDim.x = split_k
dim3 grid(split_k, grid_mn, 1);
dim3 block(NUM_WARPS * WARP_SIZE, 1, 1);

// Inside kernel: task-based scheduling
int local_task_id = blockIdx.y;
while (local_task_id < total_symmetric_tiles) {
    // 1. Map linear task ID to (block_idx_m, block_idx_n) + group swizzle
    auto idx = get_block_indices_tri_linear(local_task_id);
    int block_idx_m = xpu::get<0>(idx);
    int block_idx_n = xpu::get<1>(idx);
    get_block_indices_tri_linear_swizzled<GROUP_SIZE_M>(local_task_id, block_idx_m, block_idx_n, num_blocks_m, group_id);

    // 2. Pipeline: ramp-up -> main loop -> epilogue

    // 3. Fetch next task
    local_task_id += gridDim.y;
}
```

## Compile Flags

```python
common_cuda_flags = ["-O2", "-std=c++17"]
if major >= 9:  # Hopper
    common_cuda_flags += [
        "-Xcompiler", "-fPIC",
        "-DENABLE_HOPPER=1",
        "-arch=compute_90a",
        "-code=sm_90a",
        "--ptxas-options=-v",
    ]
```

## Anti-Patterns (Do NOT Use)
- `cp.async` (use TMA instead)
- `wmma::` / `wmma::mma_sync` (use WGMMA instead)
- `atomicAdd` for split-K reduction (use cluster NoC on-chip reduction)
- cutlass/cute library calls (use our header-only abstractions)
- Fallback to Ampere/Turing for these kernels
- `__shared__` without `__align__(128)` for TMA buffers
- Reading TMA-written SMEM without waiting on the full mbarrier
- Overwriting a stage the producer has not freed (missing empty-barrier wait)
