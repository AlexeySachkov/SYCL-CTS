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

#include <sycl/khr/includes/device.hpp>

#include "utils.hpp"

#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <tuple>

namespace khr_includes::tests {

TEST_CASE("the implementation defines the SYCL_KHR_INCLUDES macro",
          "[khr_includes][device]") {
#ifdef SYCL_KHR_INCLUDES
  constexpr bool macroIsDefined = true;
#else
  constexpr bool macroIsDefined = false;
#endif
  STATIC_REQUIRE(macroIsDefined);
}

TEST_CASE(
    "The device class is a complete type"
    "[khr_includes][device]") {
  STATIC_REQUIRE(is_complete_class_v<sycl::device>);
}

TEMPLATE_TEST_CASE(
    "the implementation of predefined device selectors is correct",
    "[khr_includes][device][device_selector]",
    decltype(sycl::default_selector_v), decltype(sycl::gpu_selector_v),
    decltype(sycl::accelerator_selector_v), decltype(sycl::cpu_selector_v),
    sycl::default_selector, sycl::gpu_selector, sycl::accelerator_selector,
    sycl::cpu_selector) {
  using Selector = TestType;

  // Device selector is a callable that accepts const sycl::device & and returns
  // an integer
  STATIC_REQUIRE(std::is_invocable_v<Selector, const sycl::device &>);
  STATIC_REQUIRE(
      std::is_same_v<std::invoke_result_t<Selector, const sycl::device&>, int>);
}

TEST_CASE(
    "the implementation of aspect device selectors is correct",
    "[khr_includes][device][device_selector]") {
  // is_invocable, invoke_result do not work well with overloaded functions
  {
    using aspect_selector = decltype(sycl::aspect_selector(
        std::declval<std::vector<sycl::aspect>>(),
        std::declval<std::vector<sycl::aspect>>()));
    STATIC_REQUIRE(std::is_invocable_v<aspect_selector, const sycl::device&>);
    STATIC_REQUIRE(
        std::is_same_v<
            std::invoke_result_t<aspect_selector, const sycl::device&>, int>);
  }

  {
    using aspect_selector = decltype(sycl::aspect_selector(
        std::declval<std::vector<sycl::aspect>>()));
    STATIC_REQUIRE(std::is_invocable_v<aspect_selector, const sycl::device&>);
    STATIC_REQUIRE(
        std::is_same_v<
            std::invoke_result_t<aspect_selector, const sycl::device&>, int>);
  }

  {
    // This overload accepts arbitrary number of arguments, testing with just 3
    using aspect_selector = decltype(sycl::aspect_selector(
        std::declval<sycl::aspect>(), std::declval<sycl::aspect>(),
        std::declval<sycl::aspect>()));
    STATIC_REQUIRE(std::is_invocable_v<aspect_selector, const sycl::device&>);
    STATIC_REQUIRE(
        std::is_same_v<
            std::invoke_result_t<aspect_selector, const sycl::device&>, int>);
  }

#if 0
  // Looks like a gap in intel/llvm implementation (and SYCL-CTS)
  {
    // This overload accepts arbitrary number of arguments, testing with just 3
    using aspect_selector =
        decltype(sycl::aspect_selector<std::declval<sycl::aspect>(),
                                       std::declval<sycl::aspect>(),
                                       std::declval<sycl::aspect>()>());
    STATIC_REQUIRE(std::is_invocable_v<aspect_selector, const sycl::device&>);
    STATIC_REQUIRE(
        std::is_same_v<
            std::invoke_result_t<aspect_selector, const sycl::device&>, int>);
  }
#endif
}

// TODO: check for 'noexcept'

TEST_CASE(
    "the implementation defines device class in full with the right APIs",
    "[khr_includes][device]") {
  STATIC_REQUIRE(std::is_default_constructible_v<sycl::device>);

  STATIC_REQUIRE(std::is_constructible_v<
                 sycl::device,
                 decltype(std::declval<int (*)(const sycl::device&)>())>);
  auto lambda_device_selector = [](const sycl::device&) -> int { return 0; };
  STATIC_REQUIRE(std::is_constructible_v<sycl::device,
                                         decltype(lambda_device_selector)>);

  // TODO: /* -- common interface members -- */

  STATIC_REQUIRE(
      std::is_same_v<decltype(std::declval<const sycl::device>().get_backend()),
                     sycl::backend>);
  STATIC_REQUIRE(noexcept(std::declval<const sycl::device>().get_backend()));

  STATIC_REQUIRE(
      std::is_same_v<decltype(std::declval<const sycl::device>().is_cpu()),
                     bool>);
  STATIC_REQUIRE(
      std::is_same_v<decltype(std::declval<const sycl::device>().is_gpu()),
                     bool>);
  STATIC_REQUIRE(
      std::is_same_v<
          decltype(std::declval<const sycl::device>().is_accelerator()), bool>);

  STATIC_REQUIRE(std::is_same_v<
                 decltype(std::declval<const sycl::device>().get_platform()),
                 sycl::platform>);

  // template <typename Param> typename Param::return_type get_info() const;
  // is covered by a separate test case below

  // TODO: template <typename Param>
  //       typename Param::return_type get_backend_info() const;
  //       The test is omitted for now, because the SYCL 2020 specification does
  //       not document any Param's which could be queried through this API.

  STATIC_REQUIRE(std::is_same_v<decltype(std::declval<const sycl::device>().has(
                                    std::declval<sycl::aspect>())),
                                bool>);

  STATIC_REQUIRE(
      std::is_same_v<decltype(std::declval<const sycl::device>().has_extension(
                         std::declval<const std::string&>())),
                     bool>);

  STATIC_REQUIRE(
      std::is_same_v<
          decltype(std::declval<const sycl::device>()
                       .create_sub_devices<
                           sycl::info::partition_property::partition_equally>(
                           std::declval<size_t>())),
          std::vector<sycl::device>>);

  STATIC_REQUIRE(
      std::is_same_v<
          decltype(std::declval<const sycl::device>()
                       .create_sub_devices<
                           sycl::info::partition_property::partition_by_counts>(
                           std::declval<const std::vector<size_t>&>())),
          std::vector<sycl::device>>);

  STATIC_REQUIRE(
      std::is_same_v<
          decltype(std::declval<const sycl::device>()
                       .create_sub_devices<sycl::info::partition_property::
                                               partition_by_affinity_domain>(
                           std::declval<
                               sycl::info::partition_affinity_domain>())),
          std::vector<sycl::device>>);

  STATIC_REQUIRE(std::is_same_v<decltype(sycl::device::get_devices()),
                                std::vector<sycl::device>>);
  STATIC_REQUIRE(std::is_same_v<decltype(sycl::device::get_devices(
                                    std::declval<sycl::info::device_type>())),
                                std::vector<sycl::device>>);
}

namespace {
// /usr/lib/gcc/x86_64-linux-gnu/14/../../../../include/c++/14/tuple:78:23:
// error: incomplete type 'sycl::platform' used in type trait expression
//   78 |     = __conditional_t<__is_final(_Tp), false_type,
template <typename T1, typename T2>
struct PairOfTypes {
  using first = T1;
  using second = T2;
};
}  // namespace

TEMPLATE_TEST_CASE(
    "the implementation of device info desciptors and APIs is correct",
    "[khr_includes][device][info]",
    // tuple(info descriptor, expected return type)
    (PairOfTypes<sycl::info::device::device_type, sycl::info::device_type>),
    (PairOfTypes<sycl::info::device::vendor_id, uint32_t>),
    (PairOfTypes<sycl::info::device::max_compute_units, uint32_t>),
    (PairOfTypes<sycl::info::device::max_work_item_dimensions, uint32_t>),
    (PairOfTypes<sycl::info::device::max_work_item_sizes<1>, sycl::range<1>>),
    (PairOfTypes<sycl::info::device::max_work_item_sizes<2>, sycl::range<2>>),
    (PairOfTypes<sycl::info::device::max_work_item_sizes<3>, sycl::range<3>>),
    (PairOfTypes<sycl::info::device::max_work_group_size, size_t>),
    (PairOfTypes<sycl::info::device::max_num_sub_groups, uint32_t>),
    (PairOfTypes<sycl::info::device::sub_group_sizes, std::vector<size_t>>),
    (PairOfTypes<sycl::info::device::preferred_vector_width_char, uint32_t>),
    (PairOfTypes<sycl::info::device::preferred_vector_width_short, uint32_t>),
    (PairOfTypes<sycl::info::device::preferred_vector_width_int, uint32_t>),
    (PairOfTypes<sycl::info::device::preferred_vector_width_long, uint32_t>),
    (PairOfTypes<sycl::info::device::preferred_vector_width_float, uint32_t>),
    (PairOfTypes<sycl::info::device::preferred_vector_width_double, uint32_t>),
    (PairOfTypes<sycl::info::device::preferred_vector_width_half, uint32_t>),
    (PairOfTypes<sycl::info::device::native_vector_width_char, uint32_t>),
    (PairOfTypes<sycl::info::device::native_vector_width_short, uint32_t>),
    (PairOfTypes<sycl::info::device::native_vector_width_int, uint32_t>),
    (PairOfTypes<sycl::info::device::native_vector_width_long, uint32_t>),
    (PairOfTypes<sycl::info::device::native_vector_width_float, uint32_t>),
    (PairOfTypes<sycl::info::device::native_vector_width_double, uint32_t>),
    (PairOfTypes<sycl::info::device::native_vector_width_half, uint32_t>),
    (PairOfTypes<sycl::info::device::max_clock_frequency, uint32_t>),
    (PairOfTypes<sycl::info::device::address_bits, uint32_t>),
    (PairOfTypes<sycl::info::device::max_mem_alloc_size, uint64_t>),
    (PairOfTypes<sycl::info::device::image_support, bool>),
    (PairOfTypes<sycl::info::device::max_read_image_args, uint32_t>),
    (PairOfTypes<sycl::info::device::max_write_image_args, uint32_t>),
    (PairOfTypes<sycl::info::device::image2d_max_width, size_t>),
    (PairOfTypes<sycl::info::device::image2d_max_height, size_t>),
    (PairOfTypes<sycl::info::device::image3d_max_width, size_t>),
    (PairOfTypes<sycl::info::device::image3d_max_height, size_t>),
    (PairOfTypes<sycl::info::device::image3d_max_depth, size_t>),
    (PairOfTypes<sycl::info::device::image_max_buffer_size, size_t>),
    (PairOfTypes<sycl::info::device::max_samplers, uint32_t>),
    (PairOfTypes<sycl::info::device::max_parameter_size, size_t>),
    (PairOfTypes<sycl::info::device::mem_base_addr_align, uint32_t>),
    (PairOfTypes<sycl::info::device::half_fp_config,
                 std::vector<sycl::info::fp_config>>),
    (PairOfTypes<sycl::info::device::single_fp_config,
                 std::vector<sycl::info::fp_config>>),
    (PairOfTypes<sycl::info::device::double_fp_config,
                 std::vector<sycl::info::fp_config>>),
    (PairOfTypes<sycl::info::device::global_mem_cache_type,
                 sycl::info::global_mem_cache_type>),
    (PairOfTypes<sycl::info::device::global_mem_cache_line_size, uint32_t>),
    (PairOfTypes<sycl::info::device::global_mem_cache_size, uint64_t>),
    (PairOfTypes<sycl::info::device::global_mem_size, uint64_t>),
    (PairOfTypes<sycl::info::device::max_constant_buffer_size, uint64_t>),
    (PairOfTypes<sycl::info::device::max_constant_args, uint32_t>),
    (PairOfTypes<sycl::info::device::local_mem_size, uint64_t>),
    (PairOfTypes<sycl::info::device::error_correction_support, bool>),
    (PairOfTypes<sycl::info::device::host_unified_memory, bool>),
    (PairOfTypes<sycl::info::device::atomic_memory_order_capabilities,
                 std::vector<sycl::memory_order>>),
    (PairOfTypes<sycl::info::device::atomic_fence_order_capabilities,
                 std::vector<sycl::memory_order>>),
    (PairOfTypes<sycl::info::device::atomic_memory_scope_capabilities,
                 std::vector<sycl::memory_scope>>),
    (PairOfTypes<sycl::info::device::atomic_fence_scope_capabilities,
                 std::vector<sycl::memory_scope>>),
    (PairOfTypes<sycl::info::device::profiling_timer_resolution, size_t>),
    (PairOfTypes<sycl::info::device::is_endian_little, bool>),
    (PairOfTypes<sycl::info::device::is_compiler_available, bool>),
    (PairOfTypes<sycl::info::device::is_linker_available, bool>),
    (PairOfTypes<sycl::info::device::execution_capabilities,
                 std::vector<sycl::info::execution_capability>>),
    (PairOfTypes<sycl::info::device::queue_profiling, bool>),
    (PairOfTypes<sycl::info::device::built_in_kernel_ids,
                 std::vector<sycl::kernel_id>>),
    (PairOfTypes<sycl::info::device::built_in_kernels,
                 std::vector<std::string>>),
    (PairOfTypes<sycl::info::device::platform, sycl::platform>),
    (PairOfTypes<sycl::info::device::name, std::string>),
    (PairOfTypes<sycl::info::device::vendor, std::string>),
    (PairOfTypes<sycl::info::device::driver_version, std::string>),
    (PairOfTypes<sycl::info::device::profile, std::string>),
    (PairOfTypes<sycl::info::device::version, std::string>),
    (PairOfTypes<sycl::info::device::backend_version, std::string>),
    (PairOfTypes<sycl::info::device::aspects, std::vector<sycl::aspect>>),
    (PairOfTypes<sycl::info::device::extensions, std::vector<std::string>>),
    (PairOfTypes<sycl::info::device::printf_buffer_size, size_t>),
    (PairOfTypes<sycl::info::device::preferred_interop_user_sync, bool>),
    (PairOfTypes<sycl::info::device::parent_device, sycl::device>),
    (PairOfTypes<sycl::info::device::partition_max_sub_devices, uint32_t>),
    (PairOfTypes<sycl::info::device::partition_properties,
                 std::vector<sycl::info::partition_property>>),
    (PairOfTypes<sycl::info::device::partition_affinity_domains,
                 std::vector<sycl::info::partition_affinity_domain>>),
    (PairOfTypes<sycl::info::device::partition_type_property,
                 sycl::info::partition_property>),
    (PairOfTypes<sycl::info::device::partition_type_affinity_domain,
                 sycl::info::partition_affinity_domain>)) {
  using InfoDescriptor = typename TestType::first;
  using ExpectedRetType = typename TestType::second;
  STATIC_REQUIRE(
      std::is_same_v<typename InfoDescriptor::return_type, ExpectedRetType>);
  STATIC_REQUIRE(std::is_same_v<decltype(std::declval<const sycl::device>()
                                             .get_info<InfoDescriptor>()),
                                typename InfoDescriptor::return_type>);
}

}  // namespace khr_includes::tests
