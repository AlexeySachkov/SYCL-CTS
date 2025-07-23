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

#include <sycl/khr/includes/platform.hpp>

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace khr_includes::tests {

TEST_CASE("the implementation defines the SYCL_KHR_INCLUDES macro",
          "[khr_includes][platform]") {
#ifdef SYCL_KHR_INCLUDES
  constexpr bool macroIsDefined = true;
#else
  constexpr bool macroIsDefined = false;
#endif
  STATIC_REQUIRE(macroIsDefined);
}

// TODO: check that methods are defined as 'const'
// TODO: check that methods are defined as 'noexcept'

TEST_CASE(
    "the implementation defines correct constructors for the platform class",
    "[khr_includes][platform][constructor]") {
  STATIC_REQUIRE(std::is_default_constructible_v<sycl::platform>);

  STATIC_REQUIRE(std::is_constructible_v<
                 sycl::platform,
                 decltype(std::declval<int (*)(const sycl::device&)>())>);
  auto lambda_device_selector = [](const sycl::device&) -> int { return 0; };
  STATIC_REQUIRE(std::is_constructible_v<sycl::platform,
                                         decltype(lambda_device_selector)>);
}

TEST_CASE(
    "the implementation defines platform class in full with the right APIs",
    "[khr_includes][platform][api]") {
  // TODO: /* -- common interface members -- */

  STATIC_REQUIRE(std::is_same_v<
                 decltype(std::declval<const sycl::platform>().get_backend()),
                 sycl::backend>);
  STATIC_REQUIRE(noexcept(std::declval<const sycl::platform>().get_backend()));

  STATIC_REQUIRE(
      std::is_invocable_v<decltype(&sycl::platform::get_devices),
                          const sycl::platform, sycl::info::device_type>);
  STATIC_REQUIRE(
      std::is_same_v<
          std::invoke_result_t<decltype(&sycl::platform::get_devices),
                               const sycl::platform, sycl::info::device_type>,
          std::vector<sycl::device>>);
  // It seems like is_invocable (and invoke_result_t) cannot be used to check
  // that a method can still be invoked without explicitly passing an optional
  // argument to it.
  STATIC_REQUIRE(std::is_same_v<
                 decltype(std::declval<const sycl::platform>().get_devices()),
                 std::vector<sycl::device>>);

  STATIC_REQUIRE(
      std::is_same_v<decltype(std::declval<const sycl::platform>().has(
                         std::declval<sycl::aspect>())),
                     bool>);

  // At intel/llvm we have:
  // public: bool has_extension(const std::string &);
  // private: bool has_extension(detail::string_view);
  // This makes the function overloaded and its address cannot be taken that
  // easily
#if 0
  STATIC_REQUIRE(std::is_invocable_v<decltype(&sycl::platform::has_extension),
                                     sycl::platform, const std::string&>);

  STATIC_REQUIRE(std::is_same_v<
                 std::invoke_result_t<decltype (&sycl::platform::has_extension)(
                     sycl::platform, const std::string&)>,
                 bool>);
#endif
  // Therefore, going with a decltype testing instead

  STATIC_REQUIRE(
      std::is_same_v<decltype(std::declval<const sycl::platform>()
                                  .has_extension(std::declval<std::string>())),
                     bool>);

  STATIC_REQUIRE(
      std::is_same_v<sycl::info::platform::version::return_type, std::string>);
  STATIC_REQUIRE(
      std::is_same_v<decltype(std::declval<const sycl::platform>()
                                  .get_info<sycl::info::platform::version>()),
                     sycl::info::platform::version::return_type>);

  STATIC_REQUIRE(
      std::is_same_v<sycl::info::platform::name::return_type, std::string>);
  STATIC_REQUIRE(
      std::is_same_v<decltype(std::declval<const sycl::platform>()
                                  .get_info<sycl::info::platform::name>()),
                     sycl::info::platform::name::return_type>);

  STATIC_REQUIRE(
      std::is_same_v<sycl::info::platform::vendor::return_type, std::string>);
  STATIC_REQUIRE(
      std::is_same_v<decltype(std::declval<const sycl::platform>()
                                  .get_info<sycl::info::platform::vendor>()),
                     sycl::info::platform::vendor::return_type>);

  STATIC_REQUIRE(std::is_same_v<sycl::info::platform::extensions::return_type,
                                std::vector<std::string>>);
  STATIC_REQUIRE(std::is_same_v<
                 decltype(std::declval<const sycl::platform>()
                              .get_info<sycl::info::platform::extensions>()),
                 sycl::info::platform::extensions::return_type>);

  // TODO: template <typename Param>
  //       typename Param::return_type get_backend_info() const;
  //       The test is omitted for now, because the SYCL 2020 specification does
  //       not document any Param's which could be queried through this API.

  STATIC_REQUIRE(std::is_same_v<decltype(sycl::platform::get_platforms()),
                                std::vector<sycl::platform>>);
}

}  // namespace khr_includes::tests
