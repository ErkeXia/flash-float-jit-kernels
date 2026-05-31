/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once
#include <cuda_runtime.h>

namespace xpu {

enum class MemoryDomain { kShared, kRegister };

template <typename T, int BM, int BN, MemoryDomain Domain>
struct FragmentView {
    T* shared_ptr;

    __device__ inline FragmentView(T* smem) : shared_ptr(smem) {}

    // 将底层逻辑坐标 (m, n) 映射到共享内存暂存区，用于跨块或线程间重新对齐
    __device__ inline T& operator()(int m, int n) {
        return shared_ptr[m * BN + n];
    }
};

} // namespace xpu
