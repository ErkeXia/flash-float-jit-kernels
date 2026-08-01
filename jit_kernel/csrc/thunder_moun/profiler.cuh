/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace ffjk {

enum CudaProfilerEventKind : uint32_t {
    kProfilerEventInstant = 0,
    kProfilerEventBegin = 1,
    kProfilerEventEnd = 2,
};

enum CudaProfilerEventId : uint32_t {
    kProfilerEventKernelLaunch = 1,
    kProfilerEventPrefetchData = 20,
    kProfilerEventScaleLoad = 21,
    kProfilerEventWgmma = 30,
    kProfilerEventFmaScaled = 31,
    kProfilerEventAccumToSmem = 40,
    kProfilerEventStoreLower = 42,
    kProfilerEventInPlaceTranspose = 43,
    kProfilerEventStoreMirror = 44,
};

struct alignas(32) CudaProfilerHeader {
    uint32_t capacity;
    uint32_t grid_xy;
    uint32_t records_per_cta;
    uint32_t records_per_task;
    uint32_t max_tasks_per_cta;
    uint32_t max_k_tiles_per_task;
    uint32_t cta_slots_and_version;
    uint32_t per_k_slots;
};

// Event identity and coordinates are encoded by the record's fixed slot.
using CudaProfilerRecord = uint32_t;

struct alignas(32) CudaProfilerBuffer {
    CudaProfilerHeader header;
};

struct CudaProfilerLayout {
    CudaProfilerBuffer* buffer;
    uint32_t cta_id;
    uint32_t records_per_cta;
    uint32_t records_per_task;
    uint32_t max_tasks_per_cta;
};

static constexpr uint32_t kCudaProfilerCtaSlots = 1;
static constexpr uint32_t kCudaProfilerTaskSlots = 12;
static constexpr uint32_t kCudaProfilerPerKSlots = 4;
static constexpr uint32_t kCudaProfilerFormatVersion = 2;

static constexpr int32_t kCudaProfilerInvalidSlot = -1;

static constexpr uint32_t kProfilerCtaSlotKernelLaunch = 0;

static constexpr uint32_t kProfilerTaskSlotPrefetchDataBegin = 0;
static constexpr uint32_t kProfilerTaskSlotPrefetchDataEnd = 1;
static constexpr uint32_t kProfilerTaskSlotScaleLoadBegin = 2;
static constexpr uint32_t kProfilerTaskSlotScaleLoadEnd = 3;
static constexpr uint32_t kProfilerTaskSlotAccumToSmemBegin = 4;
static constexpr uint32_t kProfilerTaskSlotAccumToSmemEnd = 5;
static constexpr uint32_t kProfilerTaskSlotStoreLowerBegin = 6;
static constexpr uint32_t kProfilerTaskSlotStoreLowerEnd = 7;
static constexpr uint32_t kProfilerTaskSlotInPlaceTransposeBegin = 8;
static constexpr uint32_t kProfilerTaskSlotInPlaceTransposeEnd = 9;
static constexpr uint32_t kProfilerTaskSlotStoreMirrorBegin = 10;
static constexpr uint32_t kProfilerTaskSlotStoreMirrorEnd = 11;

static constexpr uint32_t kProfilerKSlotWgmmaBegin = 0;
static constexpr uint32_t kProfilerKSlotWgmmaEnd = 1;
static constexpr uint32_t kProfilerKSlotFmaScaledBegin = 2;
static constexpr uint32_t kProfilerKSlotFmaScaledEnd = 3;

static constexpr uint32_t kCudaProfilerHeaderU64Words =
    sizeof(CudaProfilerHeader) / sizeof(uint64_t);

static_assert(sizeof(CudaProfilerHeader) % sizeof(uint64_t) == 0,
              "CudaProfilerHeader must be uint64_t-aligned for Python tensor allocation.");
static_assert(sizeof(CudaProfilerHeader) == 32,
              "CudaProfilerHeader must stay 32 bytes to keep records aligned.");
static_assert(sizeof(CudaProfilerRecord) == 4,
              "CudaProfilerRecord must stay 4 bytes for timestamp-only profiling.");

__host__ __device__ __forceinline__ uint32_t cuda_profiler_pack_u16(
    uint32_t lo,
    uint32_t hi) {
    return (lo & 0xffffu) | ((hi & 0xffffu) << 16);
}

__host__ __device__ __forceinline__ uint32_t cuda_profiler_records_per_task(
    uint32_t max_k_tiles_per_task) {
    return kCudaProfilerTaskSlots + max_k_tiles_per_task * kCudaProfilerPerKSlots;
}

__host__ __device__ __forceinline__ uint32_t cuda_profiler_records_per_cta(
    uint32_t max_tasks_per_cta,
    uint32_t max_k_tiles_per_task) {
    return kCudaProfilerCtaSlots +
           max_tasks_per_cta * cuda_profiler_records_per_task(max_k_tiles_per_task);
}

__host__ __device__ __forceinline__ uint32_t cuda_profiler_required_records(
    uint32_t num_ctas,
    uint32_t max_tasks_per_cta,
    uint32_t max_k_tiles_per_task) {
    return num_ctas *
           cuda_profiler_records_per_cta(max_tasks_per_cta, max_k_tiles_per_task);
}

inline cudaError_t cuda_profiler_init(
    void* raw_buffer,
    uint32_t capacity,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t max_tasks_per_cta,
    uint32_t max_k_tiles_per_task,
    cudaStream_t stream) {
    if (raw_buffer == nullptr || capacity == 0) {
        return cudaSuccess;
    }

    cudaError_t memset_state = cudaMemsetAsync(
        raw_buffer,
        0,
        sizeof(CudaProfilerHeader) + static_cast<size_t>(capacity) * sizeof(CudaProfilerRecord),
        stream);
    if (memset_state != cudaSuccess) {
        return memset_state;
    }

    CudaProfilerHeader header{};
    header.capacity = capacity;
    header.grid_xy = cuda_profiler_pack_u16(grid_x, grid_y);
    header.max_tasks_per_cta = max_tasks_per_cta;
    header.max_k_tiles_per_task = max_k_tiles_per_task;
    header.records_per_task = cuda_profiler_records_per_task(max_k_tiles_per_task);
    header.records_per_cta = cuda_profiler_records_per_cta(
        max_tasks_per_cta, max_k_tiles_per_task);
    header.cta_slots_and_version = cuda_profiler_pack_u16(
        kCudaProfilerCtaSlots, kCudaProfilerFormatVersion);
    header.per_k_slots = kCudaProfilerPerKSlots;

    return cudaMemcpyAsync(
        raw_buffer, &header, sizeof(header), cudaMemcpyHostToDevice, stream);
}

#if defined(__CUDA_ARCH__)

__device__ __forceinline__ uint32_t cuda_profiler_read_timestamp() {
    uint32_t timestamp;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(timestamp));
    return timestamp;
}

__device__ __forceinline__ CudaProfilerRecord* cuda_profiler_records(
    CudaProfilerBuffer* buffer) {
    return reinterpret_cast<CudaProfilerRecord*>(
        reinterpret_cast<unsigned char*>(buffer) + sizeof(CudaProfilerHeader));
}

__device__ __forceinline__ CudaProfilerLayout cuda_profiler_make_layout(
    CudaProfilerBuffer* buffer,
    uint32_t total_symmetric_tiles,
    uint32_t max_k_tiles_per_task) {
    (void)max_k_tiles_per_task;
    CudaProfilerLayout layout{};
    layout.buffer = buffer;
    if (buffer == nullptr) {
        return layout;
    }

    layout.cta_id = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    layout.max_tasks_per_cta = (total_symmetric_tiles + gridDim.y - 1) / gridDim.y;
    layout.records_per_task = cuda_profiler_records_per_task(max_k_tiles_per_task);
    layout.records_per_cta = cuda_profiler_records_per_cta(
        layout.max_tasks_per_cta, max_k_tiles_per_task);
    return layout;
}

__device__ __forceinline__ int32_t cuda_profiler_cta_slot(
    uint32_t event_id,
    CudaProfilerEventKind kind) {
    if (kind != kProfilerEventInstant) {
        return kCudaProfilerInvalidSlot;
    }

    return event_id == kProfilerEventKernelLaunch
               ? static_cast<int32_t>(kProfilerCtaSlotKernelLaunch)
               : kCudaProfilerInvalidSlot;
}

__device__ __forceinline__ int32_t cuda_profiler_begin_end_slot(
    CudaProfilerEventKind kind,
    uint32_t begin_slot,
    uint32_t end_slot) {
    return kind == kProfilerEventBegin
               ? static_cast<int32_t>(begin_slot)
               : (kind == kProfilerEventEnd
                      ? static_cast<int32_t>(end_slot)
                      : kCudaProfilerInvalidSlot);
}

__device__ __forceinline__ int32_t cuda_profiler_task_slot(
    uint32_t event_id,
    CudaProfilerEventKind kind) {
    switch (event_id) {
    case kProfilerEventPrefetchData:
        return cuda_profiler_begin_end_slot(
            kind, kProfilerTaskSlotPrefetchDataBegin, kProfilerTaskSlotPrefetchDataEnd);
    case kProfilerEventScaleLoad:
        return cuda_profiler_begin_end_slot(
            kind, kProfilerTaskSlotScaleLoadBegin, kProfilerTaskSlotScaleLoadEnd);
    case kProfilerEventAccumToSmem:
        return cuda_profiler_begin_end_slot(
            kind, kProfilerTaskSlotAccumToSmemBegin, kProfilerTaskSlotAccumToSmemEnd);
    case kProfilerEventStoreLower:
        return cuda_profiler_begin_end_slot(
            kind, kProfilerTaskSlotStoreLowerBegin, kProfilerTaskSlotStoreLowerEnd);
    case kProfilerEventInPlaceTranspose:
        return cuda_profiler_begin_end_slot(
            kind, kProfilerTaskSlotInPlaceTransposeBegin, kProfilerTaskSlotInPlaceTransposeEnd);
    case kProfilerEventStoreMirror:
        return cuda_profiler_begin_end_slot(
            kind, kProfilerTaskSlotStoreMirrorBegin, kProfilerTaskSlotStoreMirrorEnd);
    default:
        return kCudaProfilerInvalidSlot;
    }
}

__device__ __forceinline__ int32_t cuda_profiler_k_slot(
    uint32_t event_id,
    CudaProfilerEventKind kind) {
    switch (event_id) {
    case kProfilerEventWgmma:
        return cuda_profiler_begin_end_slot(
            kind, kProfilerKSlotWgmmaBegin, kProfilerKSlotWgmmaEnd);
    case kProfilerEventFmaScaled:
        return cuda_profiler_begin_end_slot(
            kind, kProfilerKSlotFmaScaledBegin, kProfilerKSlotFmaScaledEnd);
    default:
        return kCudaProfilerInvalidSlot;
    }
}

__device__ __forceinline__ void cuda_profiler_record_slot(
    const CudaProfilerLayout& layout,
    uint64_t slot) {
    if (layout.buffer == nullptr || slot >= layout.buffer->header.capacity) {
        return;
    }

    CudaProfilerRecord* records = cuda_profiler_records(layout.buffer);
    records[slot] = cuda_profiler_read_timestamp();
}

__device__ __forceinline__ void cuda_profiler_record_cta_event(
    const CudaProfilerLayout& layout,
    uint32_t event_id,
    CudaProfilerEventKind kind) {
    int32_t slot_in_cta = cuda_profiler_cta_slot(event_id, kind);
    if (slot_in_cta < 0) {
        return;
    }

    uint64_t slot = static_cast<uint64_t>(layout.cta_id) * layout.records_per_cta +
                    static_cast<uint32_t>(slot_in_cta);
    cuda_profiler_record_slot(layout, slot);
}

__device__ __forceinline__ void cuda_profiler_record_task_event(
    const CudaProfilerLayout& layout,
    uint32_t task_iter,
    uint32_t event_id,
    CudaProfilerEventKind kind) {
    int32_t slot_in_task = cuda_profiler_task_slot(event_id, kind);
    if (slot_in_task < 0 || task_iter >= layout.max_tasks_per_cta) {
        return;
    }

    uint64_t cta_base = static_cast<uint64_t>(layout.cta_id) * layout.records_per_cta;
    uint64_t task_base = static_cast<uint64_t>(task_iter) * layout.records_per_task;
    uint64_t slot = cta_base + kCudaProfilerCtaSlots + task_base +
                    static_cast<uint32_t>(slot_in_task);
    cuda_profiler_record_slot(layout, slot);
}

__device__ __forceinline__ void cuda_profiler_record_k_event(
    const CudaProfilerLayout& layout,
    uint32_t task_iter,
    uint32_t k_iter,
    uint32_t event_id,
    CudaProfilerEventKind kind) {
    if (layout.buffer == nullptr) {
        return;
    }

    int32_t slot_in_k = cuda_profiler_k_slot(event_id, kind);
    if (slot_in_k < 0 || task_iter >= layout.max_tasks_per_cta ||
        k_iter >= layout.buffer->header.max_k_tiles_per_task) {
        return;
    }

    uint64_t cta_base = static_cast<uint64_t>(layout.cta_id) * layout.records_per_cta;
    uint64_t task_base = static_cast<uint64_t>(task_iter) * layout.records_per_task;
    uint64_t k_base = static_cast<uint64_t>(k_iter) * kCudaProfilerPerKSlots;
    uint64_t slot = cta_base + kCudaProfilerCtaSlots + task_base +
                    kCudaProfilerTaskSlots + k_base + static_cast<uint32_t>(slot_in_k);
    cuda_profiler_record_slot(layout, slot);
}

#endif // defined(__CUDA_ARCH__)

} // namespace ffjk

#ifdef FFJK_ENABLE_CUDA_PROFILER

#define FFJK_PROFILER_KERNEL_PARAMS , ffjk::CudaProfilerBuffer* ffjk_profiler
#define FFJK_PROFILER_KERNEL_ARGS , ffjk_profiler
#define FFJK_PROFILER_LAUNCH_ARG(profiler) , profiler

#define FFJK_PROFILER_DEFINE_LAYOUT(total_symmetric_tiles, max_k_tiles_per_task)    \
    ffjk::CudaProfilerLayout ffjk_prof_layout = ffjk::cuda_profiler_make_layout(    \
        ffjk_profiler,                                                              \
        static_cast<uint32_t>(total_symmetric_tiles),                               \
        static_cast<uint32_t>(max_k_tiles_per_task))

#define FFJK_PROF_CTA_EVENT(event_id)                                               \
    do {                                                                            \
        if (threadIdx.x == 0) {                                                     \
            ffjk::cuda_profiler_record_cta_event(                                   \
                ffjk_prof_layout, event_id, ffjk::kProfilerEventInstant);            \
        }                                                                           \
    } while (0)

#define FFJK_PROF_BEGIN(event_id)                                                    \
    do {                                                                            \
        if (threadIdx.x == 0) {                                                     \
            ffjk::cuda_profiler_record_task_event(                                  \
                ffjk_prof_layout, static_cast<uint32_t>(ffjk_prof_task_iter),       \
                event_id, ffjk::kProfilerEventBegin);                               \
        }                                                                           \
    } while (0)

#define FFJK_PROF_END(event_id)                                                      \
    do {                                                                            \
        if (threadIdx.x == 0) {                                                     \
            ffjk::cuda_profiler_record_task_event(                                  \
                ffjk_prof_layout, static_cast<uint32_t>(ffjk_prof_task_iter),       \
                event_id, ffjk::kProfilerEventEnd);                                 \
        }                                                                           \
    } while (0)

#define FFJK_PROF_K_BEGIN(event_id, k_iter)                                         \
    do {                                                                            \
        if (threadIdx.x == 0) {                                                     \
            ffjk::cuda_profiler_record_k_event(                                     \
                ffjk_prof_layout, static_cast<uint32_t>(ffjk_prof_task_iter),       \
                static_cast<uint32_t>(k_iter), event_id,                           \
                ffjk::kProfilerEventBegin);                                        \
        }                                                                           \
    } while (0)

#define FFJK_PROF_K_END(event_id, k_iter)                                           \
    do {                                                                            \
        if (threadIdx.x == 0) {                                                     \
            ffjk::cuda_profiler_record_k_event(                                     \
                ffjk_prof_layout, static_cast<uint32_t>(ffjk_prof_task_iter),       \
                static_cast<uint32_t>(k_iter), event_id,                           \
                ffjk::kProfilerEventEnd);                                          \
        }                                                                           \
    } while (0)

#else

#define FFJK_PROFILER_KERNEL_PARAMS
#define FFJK_PROFILER_KERNEL_ARGS
#define FFJK_PROFILER_LAUNCH_ARG(profiler)
#define FFJK_PROFILER_DEFINE_LAYOUT(total_symmetric_tiles, max_k_tiles_per_task)
#define FFJK_PROF_CTA_EVENT(event_id) do {} while (0)
#define FFJK_PROF_BEGIN(event_id) do {} while (0)
#define FFJK_PROF_END(event_id) do {} while (0)
#define FFJK_PROF_K_BEGIN(event_id, k_iter) do {} while (0)
#define FFJK_PROF_K_END(event_id, k_iter) do {} while (0)

#endif // FFJK_ENABLE_CUDA_PROFILER
