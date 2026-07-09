/* Copyright 2025-2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once
#include <cuda_runtime.h>

namespace xpu {

template <typename T, int Rank>
struct TensorViewRef {
    T* ptr;
    int shape[Rank];
    int stride[Rank];

    __host__ __device__ constexpr TensorViewRef() : ptr(nullptr) {}
    __host__ __device__ constexpr TensorViewRef(T* p, const int (&sh)[Rank], const int (&st)[Rank]) : ptr(p) {
        #pragma unroll
        for (int i = 0; i < Rank; ++i) {
            shape[i] = sh[i];
            stride[i] = st[i];
        }
    }

    __host__ __device__ constexpr T& operator()(int i) const {
        return ptr[i * stride[0]];
    }

    __host__ __device__ constexpr T& operator()(int i, int j) const {
        return ptr[i * stride[0] + j * stride[1]];
    }
};

} // namespace xpu
