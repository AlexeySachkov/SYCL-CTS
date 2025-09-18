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

#include <sycl/khr/includes/buffer.hpp>

#include <type_traits>

namespace khr_includes::tests {

TEST_CASE("the implementation defines the SYCL_KHR_INCLUDES macro",
          "[khr_includes][buffer]") {
#ifdef SYCL_KHR_INCLUDES
  constexpr bool macroIsDefined = true;
#else
  constexpr bool macroIsDefined = false;
#endif
  STATIC_REQUIRE(macroIsDefined);
}

namespace {
  struct Foo {
    int a; double b; int c;
  };
}

TEMPLATE_TEST_CASE("the implementation defines member types correctly",
                   "[khr_includes][buffer]", int, long long, Foo) {
  STATIC_REQUIRE(
      std::is_same_v<typename sycl::buffer<TestType>::value_type, TestType>);
  STATIC_REQUIRE(
      std::is_same_v<typename sycl::buffer<TestType>::reference, TestType&>);
  STATIC_REQUIRE(
      std::is_same_v<typename sycl::buffer<TestType>::const_reference,
                     const TestType&>);
  STATIC_REQUIRE(
      std::is_same_v<typename sycl::buffer<TestType>::allocator_type,
                     sycl::buffer_allocator<std::remove_const_t<TestType>>>);
}

TEST_CASE("the implementation defines constructors correctly",
    "[khr_includes][buffer][constructor]") {
  STATIC_REQUIRE(
      std::is_constructible_v<sycl::buffer<int>, const sycl::range<1>&>);
  STATIC_REQUIRE(
      std::is_constructible_v<sycl::buffer<int, 2>, const sycl::range<2>&>);
  STATIC_REQUIRE(
      std::is_constructible_v<sycl::buffer<int, 3>, const sycl::range<3>&>);

  STATIC_REQUIRE(
      std::is_constructible_v<sycl::buffer<int>, const sycl::range<1>&,
                              const sycl::property_list&>);
  STATIC_REQUIRE(
      std::is_constructible_v<sycl::buffer<int, 2>, const sycl::range<2>&,
                              const sycl::property_list&>);
  STATIC_REQUIRE(
      std::is_constructible_v<sycl::buffer<int, 3>, const sycl::range<3>&,
                              const sycl::property_list&>);



}

}  // namespace khr_includes::tests
