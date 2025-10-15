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

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <sycl/khr/includes/marray.hpp>

#include "utils.hpp"

namespace khr_includes::tests {

TEST_CASE("the implementation defines the SYCL_KHR_INCLUDES macro",
          "[khr_includes][marray]") {
#ifdef SYCL_KHR_INCLUDES
  constexpr bool macroIsDefined = true;
#else
  constexpr bool macroIsDefined = false;
#endif
  STATIC_REQUIRE(macroIsDefined);
}

namespace {

struct Foo {
  int a;
  char b;
  double c;
};

}  // namespace

TEMPLATE_TEST_CASE("The marray class is a complete type",
                   "[khr_includes][marray]", bool, char, signed char,
                   unsigned char, short int, unsigned short int, int,
                   unsigned int, long int, unsigned long int, long long int,
                   unsigned long long int, float, double, Foo
                   // TODO: sycl::half, sycl::byte
) {
  STATIC_REQUIRE(is_complete_class_v<sycl::marray<TestType, 1>>);
  STATIC_REQUIRE(is_complete_class_v<sycl::marray<TestType, 2>>);
  STATIC_REQUIRE(is_complete_class_v<sycl::marray<TestType, 3>>);
  STATIC_REQUIRE(is_complete_class_v<sycl::marray<TestType, 4>>);
  STATIC_REQUIRE(is_complete_class_v<sycl::marray<TestType, 7>>);
  STATIC_REQUIRE(is_complete_class_v<sycl::marray<TestType, 53>>);
}

}  // namespace khr_includes::tests
