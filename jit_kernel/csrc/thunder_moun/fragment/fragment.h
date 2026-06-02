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

    __device__ inline T& operator()(int m, int n) {
        return shared_ptr[m * BN + n];
    }
};

} // namespace xpu
