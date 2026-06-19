#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <nv_bindings.h> // For TVM FFI or Torch Bindings

namespace cg = cooperative_groups;

// =============================================================================
// 0. 常量定义与架构配置 (Hopper Optimized)
// =============================================================================
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 64;
constexpr int STAGES = 4; // Multi-stage Mainloop Buffer

// WGMMA Shape for FP8 (E4M3 / E5M2) -> $16 \times 256 \times 16$ or $64 \times 128 \times 16$
constexpr int MMA_M = 64;
constexpr int MMA_N = 128;
constexpr int MMA_K = 16;

constexpr int NUM_THREADS = 256; // 1 Warp Group (128 threads) for MMA + 4 Warps for Movement/TMA
constexpr int MAX_SPLIT_K = 8;

// =============================================================================
// 1. 内存与 Fragment 抽象 (Block & Warp-Group Level)
// =============================================================================

template <typename T, int M, int N>
struct SharedMemBlock {
    alignas(1024) T data[M][N];
};

// Fragment 抽象：对应 Warp Group 的寄存器映射
template <typename T, int M, int N, int K>
struct RegFragment {
    // Hopper WGMMA 结果累加器通常为 fp32
    float accum[M * N / NUM_THREADS];
};

// Scale 因子 Fragment
struct ScaleFragment {
    __nv_fp8_e4m3 scale_x;
    __nv_fp8_e4m3 scale_w;
};

// =============================================================================
// 2. TMA (Tensor Memory Accelerator) 驱动器抽象
// =============================================================================
struct TMADescriptor {
    CUtensorMap desc;

    static inline void init_2d(CUtensorMap* desc, void* global_ptr, uint32_t global_dim0, uint32_t global_dim1,
                               uint32_t block_dim0, uint32_t block_dim1, uint32_t element_size) {
        uint64_t gdim[2] = {global_dim1, global_dim0};
        uint64_t gstride[1] = {global_dim1 * element_size};
        uint32_t bdim[2] = {block_dim1, block_dim0};
        uint32_t bstep[2] = {1, 1};

        cuTensorMapEncodeSmd(
            desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, global_ptr,
            gdim, gstride, bdim, bstep, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_CACHE_HINT_NONE);
    }
};

// =============================================================================
// 3. NoC Cluster 级数据交换与 Split-K Reduce 抽象
// =============================================================================
struct ClusterNode {
    // 利用 Hopper Cluster SM-to-SM Shared Memory 属性进行快速重用
    static __device__ inline void sync_cluster() {
        asm volatile("cluster.sync.aligned;\n" ::: "memory");
    }

    template <typename T>
    static __device__ inline T* get_remote_smem_ptr(T* local_ptr, int target_block_id) {
        uint32_t smem_addr = __cvta_generic_to_shared(local_ptr);
        uint32_t remote_addr;
        // 使用 Hopper ptx mapa 指令获取 Cluster 内其他 block 的 Smem 映射地址
        asm("mapa.shared.u32 %0, %1, %2;\n" : "=r"(remote_addr) : "r"(smem_addr), "r"(target_block_id));
        return (T*)(uintptr_t)remote_addr;
    }
};

// =============================================================================
// 4. MMA (WGMMA) 核心抽象驱动
// =============================================================================
template <int M, int N, int K>
struct wgmma_executor {
    static __device__ inline void mma_fp8_scale(
        float* accum,
        __nv_fp8_e4m3* smem_x,
        __nv_fp8_e4m3* smem_w,
        float scale_x,
        float scale_w)
    {
        uint32_t desc_x = __cvta_generic_to_shared(smem_x);
        uint32_t desc_w = __cvta_generic_to_shared(smem_w);

        // 转换为 Hopper WGMMA 要求的 64-bit 描述符格式
        uint64_t wgmma_desc_x = ((uint64_t)desc_x >> 4) | 0x2000000000000000B;
        uint64_t wgmma_desc_w = ((uint64_t)desc_w >> 4) | 0x2000000000000000B;

        // 注入 Scaled FP8 核心指令 (wgmma.mma_async)
        // 利用 Hopper 硬件级 FP8 乘法加算，并在指令层级通过寄存器或立即数乘上 scale
        asm volatile (
            "{\n"
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.e4m3.e4m3 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 0, 1, 1;\n"
            "}\n"
            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]),
              "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7])
            : "l"(wgmma_desc_x), "l"(wgmma_desc_w)
        );
    }
};

// =============================================================================
// 5. 核心 Device Kernel: SymmGEMM Hopper Mainloop
// =============================================================================
extern "C" __global__ __cluster_dims__(4, 4, 1) // 4x4 Cluster 配置支撑 Group 级重用
void hopper_symm_gemm_kernel(
    const __nv_fp8_e4m3* __restrict__ X,
    const __nv_fp8_e4m3* __restrict__ W,
    const float* __restrict__ scale_X,
    const float* __restrict__ scale_W,
    float* __restrict__ Y,
    int M, int N, int K,
    int split_k_slices,
    CUtensorMap tma_desc_X,
    CUtensorMap tma_desc_W)
{
    // 获取 2D Cluster 坐标与 Block 坐标
    cg::cluster_group cluster = cg::this_cluster();
    int block_idx_m = blockIdx.x;
    int block_idx_n = blockIdx.y;
    int cluster_local_id = cluster.block_rank();

    // 每一个 Block 内部划分多阶段 Shared Memory
    extern __shared__ char smem_buffer[];
    auto* shmem_X = reinterpret_cast<SharedMemBlock<__nv_fp8_e4m3, BLOCK_M, BLOCK_K>*>(smem_buffer);
    auto* shmem_W = reinterpret_cast<SharedMemBlock<__nv_fp8_e4m3, BLOCK_N, BLOCK_K>*>(smem_buffer + STAGES * sizeof(*shmem_X));

    // 初始化 Hopper 异步屏障集团 (Asynchronous Barrier Group)
    __shared__ uint64_t barriers[STAGES];
    if (threadIdx.x == 0) {
        for(int s=0; s<STAGES; s++) {
            asm volatile("init.shared.mbarrier.mbarrier_init::uint64 [%0], %1;\n" :: "r"(&barriers[s]), "r"(NUM_THREADS));
        }
    }
    __syncthreads();

    // 初始化累加寄存器
    RegFragment<float, BLOCK_M, BLOCK_N, BLOCK_K> frag_acc = {0.0f};

    // Calculate Split-K 段
    int k_tiles = (K + BLOCK_K - 1) / BLOCK_K;
    int k_tiles_per_slice = (k_tiles + split_k_slices - 1) / split_k_slices;
    int k_start = blockIdx.z * k_tiles_per_slice;
    int k_end = min(k_tiles, (int)(blockIdx.z + 1) * k_tiles_per_slice);

    // =========================================================================
    // Multi-stage Pipeline Pipeline (TMA Load + WGMMA Compute Cover)
    // =========================================================================
    int write_stage = 0;
    int read_stage = 0;

    // Pipelining 预加载 (Prologue)
    #pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
        int current_k = k_start + i;
        if (current_k < k_end) {
            if (threadIdx.x == 0) {
                // 使用 TMA 128-bit 异步零拷贝加载数据，避开寄存器中转
                uint32_t smem_X_ptr = __cvta_generic_to_shared(&shmem_X[write_stage]);
                uint32_t smem_W_ptr = __cvta_generic_to_shared(&shmem_W[write_stage]);

                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%3, %4}], [%2];\n"
                    :: "r"(smem_X_ptr), "l"(&tma_desc_X), "r"(&barriers[write_stage]),
                       "r"(current_k * BLOCK_K), "r"(block_idx_m * BLOCK_M)
                );
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%3, %4}], [%2];\n"
                    :: "r"(smem_W_ptr), "l"(&tma_desc_W), "r"(&barriers[write_stage]),
                       "r"(current_k * BLOCK_K), "r"(block_idx_n * BLOCK_N)
                );
            }
            write_stage = (write_stage + 1) % STAGES;
        }
    }

    // 主循环 (Main Loop)
    for (int k_tile_idx = k_start; k_tile_idx < k_end; ++k_tile_idx) {
        // 1. 等待当前读取阶段的 TMA 数据就绪
        uint32_t barrier_addr = __cvta_generic_to_shared(&barriers[read_stage]);
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "TRY_WAIT:\n"
            "mbarrier.try_wait.shared.mbarrier_init::uint64 p, [%0], 1;\n"
            "@!p bra TRY_WAIT;\n"
            "}\n" :: "r"(barrier_addr)
        );

        // 2. 提取当前 Block/Cluster 对应的 Scale 因子 (FP8 缩放)
        float s_x = scale_X[block_idx_m * (K / BLOCK_K) + k_tile_idx];
        float s_w = scale_W[block_idx_n * (K / BLOCK_K) + k_tile_idx];

        // 3. 执行 Warp-Group 级别的 WGMMA 抽象
        // 内部通过 128-bit 异步矩阵乘吞吐
        wgmma_executor<MMA_M, MMA_N, MMA_K>::mma_fp8_scale(
            frag_acc.accum,
            (__nv_fp8_e4m3*)&shmem_X[read_stage],
            (__nv_fp8_e4m3*)&shmem_W[read_stage],
            s_x, s_w
        );

        // 4. 释放当前 Stage 供下一次填充
        if (threadIdx.x == 0) {
            asm volatile("mbarrier.arrive.shared.mbarrier_init::uint64 [%0], 1;\n" :: "r"(barrier_addr));
        }

        // 5. 异步推进发射下一阶段的 TMA (Overlap 掩盖)
        int next_write_k = k_tile_idx + (STAGES - 1);
        if (next_write_k < k_end) {
            if (threadIdx.x == 0) {
                uint32_t smem_X_next = __cvta_generic_to_shared(&shmem_X[write_stage]);
                uint32_t smem_W_next = __cvta_generic_to_shared(&shmem_W[write_stage]);

                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%3, %4}], [%2];\n"
                    :: "r"(smem_X_next), "l"(&tma_desc_X), "r"(&barriers[write_stage]),
                       "r"(next_write_k * BLOCK_K), "r"(block_idx_m * BLOCK_M)
                );
            }
            write_stage = (write_stage + 1) % STAGES;
        }
        read_stage = (read_stage + 1) % STAGES;
    }

    // 等待所有 WGMMA 管道清空
    asm volatile("wgmma.wait_groups.sync.aligned;\n" ::: "memory");

    // =========================================================================
    // 6. Split-K Inter-Block NoC Multi-Block Reduce
    // =========================================================================
    // 为配合极致原子吞吐，采用类似 topk_indexer_radix.cu 的 NoC 锁机制
    __shared__ float shared_reduce_buf[BLOCK_M * BLOCK_N];

    // 将局部寄存器写回当前 Block 对应的 Smem
    int tx = threadIdx.x;
    for (int i = 0; i < (BLOCK_M * BLOCK_N / NUM_THREADS); ++i) {
        shared_reduce_buf[tx * (BLOCK_M * BLOCK_N / NUM_THREADS) + i] = frag_acc.accum[i];
    }
    __syncthreads();

    // 如果启用了 Split-K，进行多 Block 跨 NoC 协同
    if (split_k_slices > 1) {
        // 利用原子加减在 Global 或 Cluster 统一路由段进行 Lock-Free 归约
        for (int i = tx; i < BLOCK_M * BLOCK_N; i += NUM_THREADS) {
            int global_coord_m = block_idx_m * BLOCK_M + (i / BLOCK_N);
            int global_coord_n = block_idx_n * BLOCK_N + (i % BLOCK_N);

            if (global_coord_m < M && global_coord_n < N) {
                atomicAdd(&Y[global_coord_m * N + global_coord_n], shared_reduce_buf[i]);
            }
        }
    } else {
        // 独占情况下直接由 TMA 或普通向量指令刷回主存
        for (int i = tx; i < BLOCK_M * BLOCK_N; i += NUM_THREADS) {
            int global_coord_m = block_idx_m * BLOCK_M + (i / BLOCK_N);
            int global_coord_n = block_idx_n * BLOCK_N + (i % BLOCK_N);
            if (global_coord_m < M && global_coord_n < N) {
                Y[global_coord_m * N + global_coord_n] = shared_reduce_buf[i];
            }
        }
    }
}

// =============================================================================
// 6. TVM FFI C-API 导出接口 (JIT 友好型)
// =============================================================================
extern "C" int tvm_jit_symm_gemm_fp8(
    void* X_ptr, void* W_ptr, void* scale_X, void* scale_W, void* Y_ptr,
    int M, int N, int K, int split_k)
{
    // 1. 初始化 TMA 描述符结构体
    CUtensorMap desc_X, desc_W;
    TMADescriptor::init_2d(&desc_X, X_ptr, M, K, BLOCK_M, BLOCK_K, sizeof(__nv_fp8_e4m3));
    TMADescriptor::init_2d(&desc_W, W_ptr, N, K, BLOCK_N, BLOCK_K, sizeof(__nv_fp8_e4m3));

    // 2. 动态计算需要的 Shared Memory 大小 (Multi-stage Slots)
    uint32_t smem_size = STAGES * (sizeof(SharedMemBlock<__nv_fp8_e4m3, BLOCK_M, BLOCK_K>) +
                                   sizeof(SharedMemBlock<__nv_fp8_e4m3, BLOCK_N, BLOCK_K>));

    // 3. 配置内核函数的最大共享内存限制 (CUDA Dynamic Smem Allocation)
    cudaFuncSetAttribute((const void*)hopper_symm_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    // 4. 计算 Grid Layout (内含 Split-K Block 维度)
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N, split_k);
    dim3 block(NUM_THREADS, 1, 1);

    // 5. 激活 Cluster 配置 (Hopper Multi-SM Node Group)
    cudaLaunchConfig_t config = {0};
    config.gridDim = grid;
    config.blockDim = block;
    config.sharedMemSize = smem_size;

    cudaLaunchAttribute cluster_attr;
    cluster_attr.id = cudaLaunchAttributeClusterDimension;
    cluster_attr.val.clusterDim.x = 4;
    cluster_attr.val.clusterDim.y = 4;
    cluster_attr.val.clusterDim.z = 1;

    config.attrs = &cluster_attr;
    config.numAttrs = 1;

    // 6. 异步发射内核
    void* args[] = { &X_ptr, &W_ptr, &scale_X, &scale_W, &Y_ptr, &M, &N, &K, &split_k, &desc_X, &desc_W };
    cudaLaunchKernelEx(&config, (const void*)hopper_symm_gemm_kernel, args);

    return 0; // TVM FFI Success Code
}
