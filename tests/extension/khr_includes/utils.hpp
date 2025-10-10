/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2025 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

// This header *must not* include anything from <sycl/> to make sure that
// khr_includes tests operate only on a single header to test it in isolation.

#include <type_traits>

namespace khr_includes {

template <typename T>
struct is_complete_helper {
  template <typename U>
  static auto test(U*) -> std::integral_constant<bool, sizeof(U) == sizeof(U)>;
  static auto test(...) -> std::false_type;
  using type = decltype(test((T*)nullptr));
};

template <typename T>
struct is_complete : is_complete_helper<T>::type {};

template <typename T>
inline constexpr bool is_complete_v = is_complete<T>::value;

template <typename T>
inline constexpr bool is_complete_class_v =
    std::conjunction_v<std::is_class<T>, is_complete<T>>;

}  // namespace khr_includes
