---
name: cuda-kernel
description: CUDA C++ kernel development for NVIDIA GPUs (Ampere, Hopper, Blackwell). Covers JIT compilation via torch.utils.cpp_extension and TVM-FFI, tensor core fragment abstraction, L2 cache task swizzling, warp-level kernel swizzling (inplace/output transpose via shared memory), and shared memory optimization. For Hopper-only WGMMA/TMA/mbarrier/NoC cluster detail, see the hopper-kernel skill.
---

# CUDA Kernel Development

## CUDA Conventions in This Repo

### JIT Compilation Patterns

#### Pattern 1: torch.utils.cpp_extension (simple kernels)
```python
@functools.cache
def _jit_my_module():
    return load_jit(
        name="my_kernel",
        sources=[str(KERNEL_PATH / "csrc" / "my_kernel.cu")],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2", "-arch=sm_90", "-DENABLE_HOPPER=1"],
        verbose=True,
    )
```

#### Pattern 2: TVM-FFI (production kernels, lower overhead)
```python
from tvm_ffi.cpp import load_inline

cuda_sources = [f'#include "{path}"' for path in source_files]
module = load_inline(
    "my_kernel",
    cuda_sources=cuda_sources,
    extra_cuda_cflags=["-O2", "-std=c++17", "-DENABLE_HOPPER=1",
                       "-arch=compute_90a", "-code=sm_90a"],
    extra_ldflags=["-lcudart", "-lcuda"],
)
```
TVM-FFI is preferred for complex Hopper kernels (see thunder_moun.py).

### Platform Flags
- Hopper (SM 90a): `-DENABLE_HOPPER=1 -arch=compute_90a -code=sm_90a`
- ROCm: Check `jit_kernel/helper_rocm.py` for HIP flags, and roc device side implmentation shares the common interfaces and starts with `amd_`

## Tensor Core Programming (Zero cutlass/cute Dependency)

We develop pure cuda solutions with light weight abstraction based for the needs of applications (see
`jit_kernel/csrc/thunder_moun/fragment/nv_frag_gemm_scaled_impl.h`). We do NOT
link cutlass/cute and a light weight abstraction over PTX is intended to avoid complications.

Morover our pure cuda solutions can be easily adapted in cross platform solutions. One algorithm can truely 
run in different platforms.

### WGMMA Fragment Abstraction (Hopper tile-level MMA)

On Hopper we prefer WGMMA (`wgmma.mma_async`) for tile-level shape MMA, e.g. the
FP8 `m64n128k32` MMA demonstrated for our scaled GEMM. A warp group (4 warps =
128 threads) cooperatively produces one 64x128x32 fragment.

`HopperWGMMAAccumulator<BM, BN, BK>` holds the accumulator registers. Layout is a
hardware-fixed mapping (64 registers per thread for m64n128k32):
```cpp
template <int _BM, int _BN, int _BK>
struct HopperWGMMAAccumulator {
    static constexpr int FRAG_M = 64, FRAG_N = 128, FRAG_K = 32;
    static constexpr int kRegistersPerThread = 64;   // BM*BN/128
    static constexpr int MAX_M_STEPS = 1;
    float regs[MAX_M_STEPS][kRegistersPerThread] = {0.0f};

    __device__ inline void clear();
    __device__ inline void mul_(float scale);             // scalar scale
    __device__ inline void mul_(float* xs, float* ws, int k_offset); // block-scale
    __device__ inline void add_(const HopperWGMMAAccumulator& b);
    // Register -> shared memory layout mapping (16x8 per warp, float2, row stride 8)
    __device__ inline int getTargetWgmmaSmemOffset(
        int wg_id, int wg_lane_id, int reg_idx, int m_step, int M_STEPS,
        int* dest_row = nullptr, int* dest_col = nullptr);
    template <typename Dtype> __device__ inline void store(Dtype* smem);
};
```

`HopperWGMMAExecutor` builds the SMEM descriptors and issues the WGMMA:
```cpp
HopperWGMMAAccumulator<BM, BN, BK> accum;
accum.clear();

uint32_t smem_x_addr = __cvta_generic_to_shared(&shmem_X[stage]);
uint32_t smem_w_addr = __cvta_generic_to_shared(&shmem_W[stage]);

HopperWGMMAExecutor::mma_scaled(accum, smem_x_addr, smem_w_addr);
HopperWGMMAExecutor::commit_and_wait();
```

Key implementation points (from `nv_frag_gemm_scaled_impl.h`):
- SMEM descriptors are encoded by hand (`TmaDesc` / `make_smem_desc`), do NOT use
  `cute::make_gmma_desc`. 64-bit descriptor = start address (bits 0-13), leading
  byte offset (bits 16-29), stride byte offset (bits 32-45), layout type (bits 62-63).
- The `wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3` instruction consumes
  two SMEM descriptors (`desc_x`, `desc_w`) and the register accumulator.
- Always `wgmma.fence.sync.aligned` before issuing, then
  `wgmma.commit_group` + `wgmma.wait_group 0` after.
- FP8 scaled MMA is NOT a single instruction on SM90a — the per-k-block scale is
  applied in the epilogue via `mul_(&shmem_XS, &shmem_WS, k_offset)`.

For the full WGMMA/TMA/mbarrier/NoC patterns, see the **hopper-kernel** skill.

### MMA (Blackwell SM120/121)


## L2 Cache Task Swizzling

For symmetric GEMM we map a linear task id onto the lower triangle, then re-swizzle
tasks within a `GROUP_SIZE_M` group so that TMA loads reuse rows in L2.

```cpp
// 1. Linear -> lower-triangular mapping
int block_idx_m = int((sqrt(8.0 * local_task_id + 1.0) - 1.0) / 2.0);
int block_idx_n = local_task_id - (block_idx_m * (block_idx_m + 1)) / 2;

// 2. Group swizzle (GROUP_SIZE_M > 1) for L2 locality in TMA loads
const uint32_t group_id = block_idx_m / GROUP_SIZE_M;
// ... in-group re-mapping (see nv_block_gemm_scaled_impl.h) ...
```

This makes consecutive CTAs consume overlapping rows of A/B, so the L2-resident
operand is reused instead of re-fetched from HBM. On top of this task swizzle, the
Hopper skill shows how to go further with NoC cluster multicast.

## Warp-Level Kernel Swizzling (via Shared Memory)

When a kernel needs an inplace or output transpose (e.g. writing both the lower
triangle and its mirror to the upper right), do the transpose on the shared-memory
epilogue tile rather than in registers or global memory.

### Inplace Transpose (`FragmentView::_transpose`)
```cpp
if (threadIdx.x == 0) {
  // ... async TMA store
  // ... commit TMA event group firstly
}

if (block_idx_m > block_idx_n) {
  if (threadIdx.x == 0) {
    nvgpu::arch::tma_store_wait();  // cp.async.bulk.wait_group.read 0
  }
  frag_view._transpose(); // inplace has data dependency

  // ... TMA store
  if (threadIdx.x) {
    // ... async TMA store
  }
  // ... commit TMA event group secondly
}
```

- Non-diagonal 16x16 sub-fragments are pairwise swapped.
- Diagonal sub-fragments are transposed within themselves (upper/lower triangle).
- With `SWIZZLE_64B_STORE=1`, offsets are XOR-swizzled:
  `swizzle_col = col ^ (row % 8)` so consecutive rows land in distinct banks.

### Output Transpose
For a symmetric write of the transposed tile, use a dedicated swizzled TMA
descriptor (`tma_desc_O_swizzle`) and issue two 64B-chunk bulk stores:
```cpp
if (threadIdx.x == 0) {
  // ... async TMA store
}
if (block_idx_m > block_idx_n) {
    frag_view.transpose(trans_frag_view);  // outputplace transpose does not have data dependency
    
    // ... TMA store
    if (threadIdx.x) {
      // ... async TMA store
    }
}
// ... commit TMA event group
```

See `FragmentView` in `fragment/fragment.h` and the epilogue of
`nv_block_gemm_scaled_impl.h`.

## Bank Conflict Avoidance
- Pad shared memory arrays: `__shared__ float smem[BLOCK_SIZE][BLOCK_SIZE + 1]`
- If the shared memory is externally allocated, we should `xor` operations to calculate swizzled columns
- In the most of cases, we should avoid to specify shared memory inside the device codes
- Use `float4` / `half2` for vectorized access and deal with loop tails
- Swizzle patterns for multi-dimensional access

## Warp-Level Reduction
```cpp
#define FULL_MASK 0xffffffff // 32 lanes

// Butterfly shuffle reduction
for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(FULL_MASK, val, offset);
}
```

## Common CUDA Errors to Avoid
- `cudaErrorIllegalAddress`: Out-of-bounds memory access
- `cudaErrorMisalignedAddress`: Unaligned vector access
- `cudaErrorLaunchOutOfResources`: Too many registers/too much shared memory
- Undefined behavior from missing `__syncthreads()` after shared memory writes

## Performance Profiling
```bash
ncu --set full -o profile.ncu-rep python benchmark/bench_topk.py
nsys profile -o profile python benchmark/bench_topk.py
```

If necessary we can implement CUPTI to record on chip events.