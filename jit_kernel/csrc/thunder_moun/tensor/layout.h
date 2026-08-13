/* Copyright 2025-2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include "common.h"

#include "tuple.h"
#include "array_ref.h"

namespace xpu {

template <class T>
inline constexpr bool is_integral_or_static_v = std::is_integral_v<T> || is_static_int<T>::value;

// NOTE (yiakwy) : Hopper Gemm wgmma typically won't use such layout, but there are cases,
// as I illustrated in our symgemm, we do need layout, hence layout transforms :
//   - store_acc_to_shmem_for_global_store : from regsiter memory layout (acc) to shard memory (epilogue_shmem)
//   - inplace_sotre_shmem_for_global_transpose_store : from shared memory layout (epilogue_shmem) to shared memory layout (epilogue_shmem)

template <class Shape, class Stride>
struct Layout {
    Shape shape;
    Stride stride;

    __host__ __device__ constexpr Layout(Shape const& shape, Stride const& stride)
        : shape(shape), stride(stride) {}

    // NOTE (yiakwy) : Compute the linear index from a multi-dimensional coordinate
    // ref : 1. Triton Linear Layout in F2, 2025.05.28, https://arxiv.org/html/2505.23819v1
    //       2. explaination for AMD legacy layout used in triton https://www.lei.chat/posts/triton-bespoke-layouts/
    //       3. explaination for Triton Linear Layout under the view of compiler : a. https://www.lei.chat/posts/triton-linear-layout-concept/, b. https://www.lei.chat/posts/triton-linear-layout-examples/
    template <class Coord>
    __host__ __device__ constexpr auto operator()(Coord const& coord) const {
        return cumdot(coord, stride);
    }

private:
    template <class Coord, class _Stride>
    __host__ __device__ constexpr auto cumdot(Coord const& c, _Stride const& s) const {
        if constexpr (is_integral_or_static_v<Coord>) {
        // if constexpr (std::is_integral_v<Coord> || is_static_int<Coord>::value) {
            return c * s;
        } else {
            return cumdot_flat(c, s, std::make_index_sequence<Coord::size()>{});
        }
    }

// TODO (yiakwy) : c++17 guard
    template <class Coord, class _Stride, size_t... Is>
    __host__ __device__ constexpr auto cumdot_flat(Coord const& c, _Stride const& s, std::index_sequence<Is...>) const {
        return (cumdot(get<Is>(c), get<Is>(s)) + ... + 0);
    }
};

/*
#if __cplusplus >= 201703L
template <typename T, typename U>
Layout(std::initializer_list<T>, std::initializer_list<U>) -> Layout<ArrayRef<const T>, ArrayRef<const U>>;
#endif

using Layout2D = Layout<ArrayRef<const int>, ArrayRef<const int>>;
*/

} // namespace xpu
