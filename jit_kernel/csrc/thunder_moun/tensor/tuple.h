/* Copyright 2025-2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include <cuda_runtime.h>

#include <ostream>

#define HOST_DEVICE __host__ __device__

#define USE_ARRAY_REF true

// TODO (yiakwy) : #include "common.h"

// TODO (yiakwy) : #include "array_ref.h"
#include "array_ref.h"

#include <type_traits>
#include <utility>

namespace xpu {

template <typename... Args>
struct Tuple;

template <>
struct Tuple<> {};


// NOTE (yiakwy) : static value placeholder
template <int v>
struct Int {
    static constexpr int value = v;
    __host__ __device__ constexpr operator int() const { return v; }
};


template <class T> struct is_static_int : std::false_type {};
template <int v> struct is_static_int<Int<v>> : std::true_type {};

template <int I, typename Head, typename... Tail>
HOST_DEVICE constexpr decltype(auto) get(Tuple<Head, Tail...>& t) noexcept;

template <int I, typename Head, typename... Tail>
HOST_DEVICE constexpr decltype(auto) get(const Tuple<Head, Tail...>& t) noexcept;

template <int I, typename Head, typename... Tail>
HOST_DEVICE constexpr decltype(auto) get(Tuple<Head, Tail...>&& t) noexcept;

template <int I, typename Head, typename... Tail>
HOST_DEVICE constexpr decltype(auto) get(const Tuple<Head, Tail...>&& t) noexcept;


// NOTE (yiakwy) : NVIDIA Cute style heterogeneous tuple in device side (our make_tuple actualy allocate homogenous tuple)
template <typename Head, typename... Tail>
struct Tuple<Head, Tail...> {
    Head head;
    Tuple<Tail...> tail;

    __host__ __device__ constexpr Tuple() : head(), tail() {}

    template <typename H, typename... T,
              typename = std::enable_if_t<!std::is_same_v<std::decay_t<H>, Tuple>>>
    HOST_DEVICE constexpr Tuple(H&& h, T&&... t)
        : head(std::forward<H>(h)), tail(std::forward<T>(t)...) {}

#if USE_ARRAY_REF

    template <typename U>
    HOST_DEVICE constexpr Tuple(ArrayRef<U> const& view, Tail... t)
        : head(view), tail(t...) {}

    template <typename... SubArgs, typename... ExtraArgs>
    HOST_DEVICE constexpr Tuple(Tuple<SubArgs...> const& sub_tree, ExtraArgs&&... extra)
        : head(sub_tree), tail(std::forward<ExtraArgs>(extra)...) {}

#endif // USE_ARRAY_REF

    // NOTE (yiakwy) : mutable return type, i.e., Head& get<0>(Tuple<Head, Tail...>& t) { return t.head; }
    template <int I>
    __host__ __device__ constexpr decltype(auto) operator[](Int<I>) & noexcept {
        return get<I>(*this);
    }

    // NOTE (yiakwy) : unmutable return type, i.e., const Head& get<0>(const Tuple<Head, Tail...>& t) { return t.head; }
    template <int I>
    __host__ __device__ constexpr decltype(auto) operator[](Int<I>) const & noexcept {
        return get<I>(*this);
    }


    template <int I>
    __host__ __device__ constexpr decltype(auto) operator[](Int<I>) && noexcept {
        return get<I>(std::move(*this));
    }

    template <int I>
    __host__ __device__ constexpr decltype(auto) operator[](Int<I>) const && noexcept {
        return get<I>(std::move(*this));
    }

    __host__ __device__ static constexpr size_t size() { return 1 + sizeof...(Tail); }

    // TODO (yiakwy) : return raw c pointer API
};

// NOTE (yiakwy) : for convenience, we can define a type alias for the TupleElement type to pass compiler complaints
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

// get api
template <int I, typename Head, typename... Tail>
HOST_DEVICE constexpr decltype(auto) get(Tuple<Head, Tail...>& t) noexcept {
    if constexpr (I == 0) return (t.head); else return get<I - 1>(t.tail);
}
template <int I, typename Head, typename... Tail>
HOST_DEVICE constexpr decltype(auto) get(const Tuple<Head, Tail...>& t) noexcept {
    if constexpr (I == 0) return (t.head); else return get<I - 1>(t.tail);
}
template <int I, typename Head, typename... Tail>
HOST_DEVICE constexpr decltype(auto) get(Tuple<Head, Tail...>&& t) noexcept {
    if constexpr (I == 0) return std::move(t.head); else return get<I - 1>(std::move(t.tail));
}
template <int I, typename Head, typename... Tail>
HOST_DEVICE constexpr decltype(auto) get(const Tuple<Head, Tail...>&& t) noexcept {
    if constexpr (I == 0) return std::move(t.head); else return get<I - 1>(std::move(t.tail));
}


// NOTE (yiakwy) : homogenous tuple in device side


// NOTE (yiakwy) : make_tuple factory
template <typename... Args>
HOST_DEVICE constexpr auto make_tuple(Args&&... args) {
    // if constexpr (sizeof...(Args) > 0 && all_same<std::decay_t<Args>...>::value) {
    if constexpr (false) {
        /*
        using FirstType = std::tuple_element_t<0, std::tuple<std::decay_t<Args>...>>;
        return Tuple<FirstType, std::integral_constant<size_t, sizeof...(Args)>>(std::forward<Args>(args)...);
         */
    } else {
        return Tuple<std::decay_t<Args>...>(std::forward<Args>(args)...);
    }
}

#if USE_ARRAY_REF

template <typename T>
HOST_DEVICE constexpr auto make_tuple(ArrayRef<T> const& view) {
    return Tuple<ArrayRef<T>>(view);
}

template <typename... SubArgs, typename T>
HOST_DEVICE constexpr auto make_tuple(Tuple<SubArgs...> const& sub_tree, T&& extra) {
    return Tuple<Tuple<SubArgs...>, std::decay_t<T>>(sub_tree, std::forward<T>(extra));
}

// NOTE (yiakwy) : bracket initialization support
template <typename T, typename U>
HOST_DEVICE constexpr auto make_tuple(std::initializer_list<T> const& sub, U&& extra) {
    return Tuple<Tuple<T, T>, std::decay_t<U>>(
        Tuple<T, T>(*(sub.begin()), *(sub.begin() + 1)),
        std::forward<U>(extra));
}

#if __cplusplus >= 201703L

template <typename T, typename U>
Tuple(std::initializer_list<T>, U) -> Tuple<Tuple<T, T>, U>;

#endif // __cplusplus >= 201703L

#endif // USE_ARRAY_REF

// test utility
template <int v>
std::ostream& operator<<(std::ostream& os, const Int<v>&) {
    os << v;
    return os;
}

inline std::ostream& print_tuple(std::ostream& os, const Tuple<>&) {
    return os;
}

template <typename Head, typename... Tail>
inline std::ostream& print_tuple(std::ostream& os, const Tuple<Head, Tail...>& t) {
    os << t.head;
    if constexpr (sizeof...(Tail) > 0) {
        os << ", ";
        print_tuple(os, t.tail);
    }
    return os;
}

template <typename... Args>
std::ostream& operator<<(std::ostream& os, const Tuple<Args...>& t) {
    os << "(";
    print_tuple(os, t);
    os << ")";
    return os;
}

} // namespace xpu
