/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

namespace xpu {

template <typename... Args>
struct Tuple;

template <>
struct Tuple<> {};

template <typename Head, typename... Tail>
struct Tuple<Head, Tail...> {
    Head head;
    Tuple<Tail...> tail;

    __host__ __device__ constexpr Tuple() : head(), tail() {}
    __host__ __device__ constexpr Tuple(Head h, Tail... t) : head(h), tail(t...) {}
};

template <int I, typename T>
struct TupleElement;

template <typename Head, typename... Tail>
struct TupleElement<0, Tuple<Head, Tail...>> {
    using type = Head;
};

template <int I, typename Head, typename... Tail>
struct TupleElement<I, Tuple<Head, Tail...>> {
    using type = typename TupleElement<I - 1, Tuple<Tail...>>::type;
};

template <int I, typename Head, typename... Tail>
__host__ __device__ constexpr auto& get(Tuple<Head, Tail...>& t) {
    if constexpr (I == 0) return t.head;
    else return get<I - 1>(t.tail);
}

template <int I, typename Head, typename... Tail>
__host__ __device__ constexpr const auto& get(const Tuple<Head, Tail...>& t) {
    if constexpr (I == 0) return t.head;
    else return get<I - 1>(t.tail);
}

} // namespace xpu
