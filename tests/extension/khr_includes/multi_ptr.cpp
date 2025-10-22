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

#include <sycl/khr/includes/multi_ptr.hpp>

#include "utils.hpp"

namespace khr_includes::tests {

TEST_CASE("the implementation defines the SYCL_KHR_INCLUDES macro",
          "[khr_includes][multi_ptr]") {
#ifdef SYCL_KHR_INCLUDES
  constexpr bool macroIsDefined = true;
#else
  constexpr bool macroIsDefined = false;
#endif
  STATIC_REQUIRE(macroIsDefined);
}

template <typename T, sycl::access::address_space AS>
void for_each_decorated() {
  STATIC_REQUIRE(
      is_complete_class_v<sycl::multi_ptr<T, AS, sycl::access::decorated::no>>);
  STATIC_REQUIRE(is_complete_class_v<
                 sycl::multi_ptr<T, AS, sycl::access::decorated::yes>>);
  STATIC_REQUIRE(is_complete_class_v<
                 sycl::multi_ptr<T, AS, sycl::access::decorated::legacy>>);
}

template <typename T>
void for_each_address_space_and_decorated() {
  for_each_decorated<T, sycl::access::address_space::global_space>();
  for_each_decorated<T, sycl::access::address_space::local_space>();
  for_each_decorated<T, sycl::access::address_space::constant_space>();
  for_each_decorated<T, sycl::access::address_space::private_space>();
  for_each_decorated<T, sycl::access::address_space::generic_space>();
}

namespace {

struct Foo {
  int a;
  char b;
  double c;
};

}  // namespace

TEMPLATE_TEST_CASE("The multi_ptr class is a complete type",
                   "[khr_includes][multi_ptr]", bool, char, signed char,
                   unsigned char, short int, unsigned short int, int,
                   unsigned int, long int, unsigned long int, long long int,
                   unsigned long long int, float, double) {
  for_each_address_space_and_decorated<TestType>();
}

}  // namespace khr_includes::tests
