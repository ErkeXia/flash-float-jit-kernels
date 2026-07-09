/* Copyright 2025 flashFloat authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "common.h"

namespace compat {

template <class T> inline constexpr auto data(T &&t) {
  return std::data(std::forward<T>(t));
}

template <class T> inline constexpr auto size(T &&t) {
  return std::size(std::forward<T>(t));
}

} // namespace compat


namespace traits {

template <class Type, class Container, class Traits = void>
struct IsContiguousContainer : std::false_type {};

// alow contianer with .data(), .size() interfaces, i.e. our Tuple and ArrayRef
template <class Type, class Container>
struct IsContiguousContainer<
    Type, Container,
    typename std::enable_if<
        std::is_pointer<decltype(std::declval<Container>().data())>::value &&
        std::is_integral<decltype(std::declval<Container>().size())>::value &&
        std::is_convertible<decltype(std::declval<Container>().data()),
                            typename std::add_pointer<Type>::type>::value
        >::type>
  : std::true_type {};

template <class Type, class Element, size_t N>
struct IsContiguousContainer<Type, Element (&)[N]>
  : std::is_convertible<typename std::add_pointer<Element>::type,
                        typename std::add_pointer<Type>::type> {};

template <class Type, class Element>
struct IsContiguousContainer<Type, std::initializer_list<Element>>
  : std::is_convertible<typename std::initializer_list<Element>::iterator,
                        typename std::add_pointer<Type>::type> {};

} // traits

// NOTE (yiakwy) : used for device tuple bracket initialization

namespace xpu {

// ArrayRef is array view of pre-allocated memory, we will use it in our cross static memory in devices.
template <class T> struct ArrayRef {
  static_assert(not std::is_const<T>::value,
                "T should be non-const type");

  using value_type = T;
  using ref_type = T &;
  using const_ref_type = const T &;
  using iterator = T *;
  using const_iterator = const T *;

  T *ptr = nullptr;
  std::size_t len = 0;

  HOST_DEVICE constexpr ArrayRef() = default;

  // Constructs a view to a buffer given its starting address and size.
  HOST_DEVICE constexpr ArrayRef(T *p, std::size_t size) noexcept : ptr(p), len(size) {}

  template <class U,
            typename = typename std::enable_if<traits::IsContiguousContainer<
                T, std::initializer_list<U>>::value>::type>
  HOST_DEVICE constexpr ArrayRef(std::initializer_list<U> ilist)
      : ArrayRef(ilist.begin(), ilist.size()) {}

  template <class U,
            typename = typename std::enable_if<
                traits::IsContiguousContainer<T, U &>::value>::type>
  HOST_DEVICE constexpr ArrayRef(U &t) : ArrayRef(compat::data(t), compat::size(t)) {}

  HOST_DEVICE T &operator[](std::size_t i) const noexcept {
    static_assert(i < len);
    return ptr[i];
  }

  HOST_DEVICE_INLINE constexpr std::size_t size() const { return len; }

  HOST_DEVICE_INLINE constexpr T *data() const noexcept { return ptr; }

  HOST_DEVICE_INLINE T *data() { return ptr; }
};

template <class T> struct ArrayRef<const T> {

  using value_type = const T;
  using reference = const T &;
  using const_reference = const T &;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;
  using iterator = const T *;
  using const_iterator = const T *;

  const T *ptr = nullptr;
  std::size_t len = 0;

  // HOST_DEVICE constexpr ArrayRef(std::nullptr) = delete;

  HOST_DEVICE constexpr ArrayRef() : ptr(nullptr), len(0) {}
  HOST_DEVICE constexpr ArrayRef(const T *p, std::size_t size) noexcept : ptr(p), len(size) {}

  HOST_DEVICE constexpr ArrayRef(const std::initializer_list<T> &list)
      : ArrayRef(list.begin(), list.size()) {}

  HOST_DEVICE T &operator[](std::size_t i) const noexcept {
    static_assert(i < len);
    return ptr[i];
  }

  HOST_DEVICE_INLINE constexpr std::size_t size() const { return len; }

  HOST_DEVICE_INLINE constexpr const T *data() const noexcept { return ptr; }

  HOST_DEVICE_INLINE const T *data() { return ptr; }

};

#if __cplusplus >= 201703L // Deduction guides are available
template <class T, size_t N> ArrayRef(const T (&)[N]) -> ArrayRef<const T>;
template <class T, size_t N> ArrayRef(T (&)[N]) -> ArrayRef<T>;

template <class T> ArrayRef(std::initializer_list<T>) -> ArrayRef<const T>;

#endif

} // namespace xpu
