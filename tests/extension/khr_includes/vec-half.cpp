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

#include <catch2/catch_test_macros.hpp>

#include <sycl/khr/includes/half.hpp>
#include <sycl/khr/includes/vec.hpp>

#include "utils.hpp"

namespace khr_includes::tests {

TEST_CASE("The vec<half> class is a complete type", "[khr_includes][vec]") {
  STATIC_REQUIRE(is_complete_class_v<sycl::vec<sycl::half, 1>>);
  STATIC_REQUIRE(is_complete_class_v<sycl::vec<sycl::half, 2>>);
  STATIC_REQUIRE(is_complete_class_v<sycl::vec<sycl::half, 3>>);
  STATIC_REQUIRE(is_complete_class_v<sycl::vec<sycl::half, 4>>);
  STATIC_REQUIRE(is_complete_class_v<sycl::vec<sycl::half, 8>>);
  STATIC_REQUIRE(is_complete_class_v<sycl::vec<sycl::half, 16>>);
}

}  // namespace khr_includes::tests
