#include <sycl/sycl.hpp>

#include "../common/common.h"
#include "../common/common_vec.h"

#include <algorithm>
#include <array>

namespace vector::operators {
enum arithmetic_operation_kind : int {
  vec_vec = 0,
  vec_swizzle,
  vec_scalar,
  swizzle_vec,
  swizzle_swizzle,
  swizzle_scalar,
  scalar_vec,
  scalar_swizzle,
  size
};

enum class arithmetic_operation {
  addition,
  multiplication,
  substraction,
  division
};

template <arithmetic_operation op, typename T, int N>
class arithmetic_kernel;

#define DO_VEC_SWIZZLE_OP(result, v1, v2, op)                              \
  if constexpr (v1.size() == 1) {                                          \
    result = v1 op v2.template swizzle<sycl::elem::s0>();                  \
  } else if constexpr (v1.size() == 2) {                                   \
    result = v1 op v2.template swizzle<sycl::elem::s0, sycl::elem::s1>();  \
  } else if constexpr (v1.size() == 3) {                                   \
    result = v1 op v2.template swizzle<sycl::elem::s0, sycl::elem::s1,     \
                                       sycl::elem::s2>();                  \
  } else if constexpr (v1.size() == 4) {                                   \
    result = v1 op v2.template swizzle<sycl::elem::s0, sycl::elem::s1,     \
                                       sycl::elem::s2, sycl::elem::s3>();  \
  } else if constexpr (v1.size() == 8) {                                   \
    result = v1 op v2.template swizzle<                                    \
        sycl::elem::s0, sycl::elem::s1, sycl::elem::s2, sycl::elem::s3,    \
        sycl::elem::s4, sycl::elem::s5, sycl::elem::s6, sycl::elem::s7>(); \
  } else if constexpr (v1.size() == 16) {                                  \
    result = v1 op v2.template swizzle<                                    \
        sycl::elem::s0, sycl::elem::s1, sycl::elem::s2, sycl::elem::s3,    \
        sycl::elem::s4, sycl::elem::s5, sycl::elem::s6, sycl::elem::s7,    \
        sycl::elem::s8, sycl::elem::s9, sycl::elem::sA, sycl::elem::sB,    \
        sycl::elem::sC, sycl::elem::sD, sycl::elem::sE, sycl::elem::sF>(); \
  }

#define DO_SCALAR_SWIZZLE_OP(result, s, v, op)                             \
  if constexpr (v.size() == 1) {                                           \
    result = s op v.template swizzle<sycl::elem::s0>();                    \
  } else if constexpr (v.size() == 2) {                                    \
    result = s op v.template swizzle<sycl::elem::s0, sycl::elem::s1>();    \
  } else if constexpr (v.size() == 3) {                                    \
    result = s op v.template swizzle<sycl::elem::s0, sycl::elem::s1,       \
                                     sycl::elem::s2>();                    \
  } else if constexpr (v.size() == 4) {                                    \
    result = s op v.template swizzle<sycl::elem::s0, sycl::elem::s1,       \
                                     sycl::elem::s2, sycl::elem::s3>();    \
  } else if constexpr (v.size() == 8) {                                    \
    result = s op v.template swizzle<                                      \
        sycl::elem::s0, sycl::elem::s1, sycl::elem::s2, sycl::elem::s3,    \
        sycl::elem::s4, sycl::elem::s5, sycl::elem::s6, sycl::elem::s7>(); \
  } else if constexpr (v.size() == 16) {                                   \
    result = s op v.template swizzle<                                      \
        sycl::elem::s0, sycl::elem::s1, sycl::elem::s2, sycl::elem::s3,    \
        sycl::elem::s4, sycl::elem::s5, sycl::elem::s6, sycl::elem::s7,    \
        sycl::elem::s8, sycl::elem::s9, sycl::elem::sA, sycl::elem::sB,    \
        sycl::elem::sC, sycl::elem::sD, sycl::elem::sE, sycl::elem::sF>(); \
  }

#define DO_SWIZZLE_SWIZZLE_OP(result, v1, v2, op)                             \
  if constexpr (v1.size() == 1) {                                             \
    result = v1.template swizzle<sycl::elem::s0>()                            \
                 op v2.template swizzle<sycl::elem::s0>();                    \
  } else if constexpr (v1.size() == 2) {                                      \
    result = v1.template swizzle<sycl::elem::s0, sycl::elem::s1>()            \
                 op v2.template swizzle<sycl::elem::s0, sycl::elem::s1>();    \
  } else if constexpr (v1.size() == 3) {                                      \
    result =                                                                  \
        v1.template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2>() \
            op v2.template swizzle<sycl::elem::s0, sycl::elem::s1,            \
                                   sycl::elem::s2>();                         \
  } else if constexpr (v1.size() == 4) {                                      \
    result = v1.template swizzle<sycl::elem::s0, sycl::elem::s1,              \
                                 sycl::elem::s2, sycl::elem::s3>()            \
                 op v2.template swizzle<sycl::elem::s0, sycl::elem::s1,       \
                                        sycl::elem::s2, sycl::elem::s3>();    \
  } else if constexpr (v1.size() == 8) {                                      \
    result =                                                                  \
        v1.template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2,   \
                            sycl::elem::s3, sycl::elem::s4, sycl::elem::s5,   \
                            sycl::elem::s6, sycl::elem::s7>() op v2           \
            .template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2, \
                              sycl::elem::s3, sycl::elem::s4, sycl::elem::s5, \
                              sycl::elem::s6, sycl::elem::s7>();              \
  } else if constexpr (v1.size() == 16) {                                     \
    result =                                                                  \
        v1.template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2,   \
                            sycl::elem::s3, sycl::elem::s4, sycl::elem::s5,   \
                            sycl::elem::s6, sycl::elem::s7, sycl::elem::s8,   \
                            sycl::elem::s9, sycl::elem::sA, sycl::elem::sB,   \
                            sycl::elem::sC, sycl::elem::sD, sycl::elem::sE,   \
                            sycl::elem::sF>() op v2                           \
            .template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2, \
                              sycl::elem::s3, sycl::elem::s4, sycl::elem::s5, \
                              sycl::elem::s6, sycl::elem::s7, sycl::elem::s8, \
                              sycl::elem::s9, sycl::elem::sA, sycl::elem::sB, \
                              sycl::elem::sC, sycl::elem::sD, sycl::elem::sE, \
                              sycl::elem::sF>();                              \
  }

#define DO_SWIZZLE_VEC_OP(result, v1, v2, op)                                 \
  if constexpr (v1.size() == 1) {                                             \
    result = v1.template swizzle<sycl::elem::s0>() op v2;                     \
  } else if constexpr (v1.size() == 2) {                                      \
    result = v1.template swizzle<sycl::elem::s0, sycl::elem::s1>() op v2;     \
  } else if constexpr (v1.size() == 3) {                                      \
    result =                                                                  \
        v1.template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2>() \
            op v2;                                                            \
  } else if constexpr (v1.size() == 4) {                                      \
    result = v1.template swizzle<sycl::elem::s0, sycl::elem::s1,              \
                                 sycl::elem::s2, sycl::elem::s3>() op v2;     \
  } else if constexpr (v1.size() == 8) {                                      \
    result =                                                                  \
        v1.template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2,   \
                            sycl::elem::s3, sycl::elem::s4, sycl::elem::s5,   \
                            sycl::elem::s6, sycl::elem::s7>() op v2;          \
  } else if constexpr (v1.size() == 16) {                                     \
    result = v1.template swizzle<                                             \
        sycl::elem::s0, sycl::elem::s1, sycl::elem::s2, sycl::elem::s3,       \
        sycl::elem::s4, sycl::elem::s5, sycl::elem::s6, sycl::elem::s7,       \
        sycl::elem::s8, sycl::elem::s9, sycl::elem::sA, sycl::elem::sB,       \
        sycl::elem::sC, sycl::elem::sD, sycl::elem::sE, sycl::elem::sF>()     \
                 op v2;                                                       \
  }

#define DO_SWIZZLE_SCALAR_OP(result, v, s, op)                               \
  if constexpr (v.size() == 1) {                                             \
    result = v.template swizzle<sycl::elem::s0>() op s;                      \
  } else if constexpr (v.size() == 2) {                                      \
    result = v.template swizzle<sycl::elem::s0, sycl::elem::s1>() op s;      \
  } else if constexpr (v.size() == 3) {                                      \
    result =                                                                 \
        v.template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2>() \
            op s;                                                            \
  } else if constexpr (v.size() == 4) {                                      \
    result = v.template swizzle<sycl::elem::s0, sycl::elem::s1,              \
                                sycl::elem::s2, sycl::elem::s3>() op s;      \
  } else if constexpr (v.size() == 8) {                                      \
    result =                                                                 \
        v.template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2,   \
                           sycl::elem::s3, sycl::elem::s4, sycl::elem::s5,   \
                           sycl::elem::s6, sycl::elem::s7>() op s;           \
  } else if constexpr (v.size() == 16) {                                     \
    result = v.template swizzle<                                             \
        sycl::elem::s0, sycl::elem::s1, sycl::elem::s2, sycl::elem::s3,      \
        sycl::elem::s4, sycl::elem::s5, sycl::elem::s6, sycl::elem::s7,      \
        sycl::elem::s8, sycl::elem::s9, sycl::elem::sA, sycl::elem::sB,      \
        sycl::elem::sC, sycl::elem::sD, sycl::elem::sE, sycl::elem::sF>()    \
                 op s;                                                       \
  }

template <typename T, int N>
sycl::buffer<sycl::vec<T, N>, 1> do_addition_test(sycl::queue q, T value_a,
                                                  T value_b) {
  sycl::buffer<sycl::vec<T, N>, 1> results(
      sycl::range{arithmetic_operation_kind::size});

  q.submit([&](sycl::handler& cgh) {
    sycl::accessor acc(results, cgh, sycl::write_only);
    cgh.single_task<arithmetic_kernel<arithmetic_operation::addition, T, N>>(
        [=]() {
          auto vec1 = sycl::vec<T, N>(value_a);
          auto vec2 = sycl::vec<T, N>(value_b);

          acc[vec_vec] = vec1 + vec2;
          DO_VEC_SWIZZLE_OP(acc[vec_swizzle], vec1, vec2, +);
          acc[vec_scalar] = vec1 + value_b;
          DO_SWIZZLE_VEC_OP(acc[swizzle_vec], vec1, vec2, +);
          DO_SWIZZLE_SWIZZLE_OP(acc[swizzle_swizzle], vec1, vec2, +);
          DO_SWIZZLE_SCALAR_OP(acc[swizzle_scalar], vec1, value_b, +);
          acc[scalar_vec] = value_a + vec2;
          DO_SCALAR_SWIZZLE_OP(acc[scalar_swizzle], value_a, vec2, +);
        });
  });

  return results;
}

template <typename T, int N>
sycl::buffer<sycl::vec<T, N>, 1> do_substraction_test(sycl::queue q, T value_a,
                                                      T value_b) {
  sycl::buffer<sycl::vec<T, N>, 1> results(
      sycl::range{arithmetic_operation_kind::size});

  q.submit([&](sycl::handler& cgh) {
    sycl::accessor acc(results, cgh, sycl::write_only);
    cgh.single_task<
        arithmetic_kernel<arithmetic_operation::substraction, T, N>>([=]() {
      auto vec1 = sycl::vec<T, N>(value_a);
      auto vec2 = sycl::vec<T, N>(value_b);

      acc[vec_vec] = vec1 - vec2;
      DO_VEC_SWIZZLE_OP(acc[vec_swizzle], vec1, vec2, -);
      acc[vec_scalar] = vec1 - value_b;
      DO_SWIZZLE_VEC_OP(acc[swizzle_vec], vec1, vec2, -);
      DO_SWIZZLE_SWIZZLE_OP(acc[swizzle_swizzle], vec1, vec2, -);
      DO_SWIZZLE_SCALAR_OP(acc[swizzle_scalar], vec1, value_b, -);
      acc[scalar_vec] = value_a - vec2;
      DO_SCALAR_SWIZZLE_OP(acc[scalar_swizzle], value_a, vec2, -);
    });
  });

  return results;
}

template <typename T, int N>
sycl::buffer<sycl::vec<T, N>, 1> do_division_test(sycl::queue q, T value_a,
                                                  T value_b) {
  sycl::buffer<sycl::vec<T, N>, 1> results(
      sycl::range{arithmetic_operation_kind::size});

  q.submit([&](sycl::handler& cgh) {
    sycl::accessor acc(results, cgh, sycl::write_only);
    cgh.single_task<arithmetic_kernel<arithmetic_operation::division, T, N>>(
        [=]() {
          auto vec1 = sycl::vec<T, N>(value_a);
          auto vec2 = sycl::vec<T, N>(value_b);

          acc[vec_vec] = vec1 / vec2;
          DO_VEC_SWIZZLE_OP(acc[vec_swizzle], vec1, vec2, /);
          acc[vec_scalar] = vec1 / value_b;
          DO_SWIZZLE_VEC_OP(acc[swizzle_vec], vec1, vec2, /);
          DO_SWIZZLE_SWIZZLE_OP(acc[swizzle_swizzle], vec1, vec2, /);
          DO_SWIZZLE_SCALAR_OP(acc[swizzle_scalar], vec1, value_b, /);
          acc[scalar_vec] = value_a / vec2;
          DO_SCALAR_SWIZZLE_OP(acc[scalar_swizzle], value_a, vec2, /);
        });
  });

  return results;
}

template <typename T, int N>
sycl::buffer<sycl::vec<T, N>, 1> do_multiplication_test(sycl::queue q,
                                                        T value_a, T value_b) {
  sycl::buffer<sycl::vec<T, N>, 1> results(
      sycl::range{arithmetic_operation_kind::size});

  q.submit([&](sycl::handler& cgh) {
    sycl::accessor acc(results, cgh, sycl::write_only);
    cgh.single_task<
        arithmetic_kernel<arithmetic_operation::multiplication, T, N>>([=]() {
      auto vec1 = sycl::vec<T, N>(value_a);
      auto vec2 = sycl::vec<T, N>(value_b);

      acc[vec_vec] = vec1 * vec2;
      DO_VEC_SWIZZLE_OP(acc[vec_swizzle], vec1, vec2, *);
      acc[vec_scalar] = vec1 * value_b;
      DO_SWIZZLE_VEC_OP(acc[swizzle_vec], vec1, vec2, *);
      DO_SWIZZLE_SWIZZLE_OP(acc[swizzle_swizzle], vec1, vec2, *);
      DO_SWIZZLE_SCALAR_OP(acc[swizzle_scalar], vec1, value_b, *);
      acc[scalar_vec] = value_a * vec2;
      DO_SCALAR_SWIZZLE_OP(acc[scalar_swizzle], value_a, vec2, *);
    });
  });

  return results;
}

template <typename Buf, typename T>
void check_results(Buf buf, T reference) {
  auto acc = buf.get_host_access();
  for (size_t i = 0; i < acc.size(); ++i) {
    for (size_t j = 0; j < acc[i].size(); ++j) {
      CHECK(acc[i][j] == reference);
    }
  }
}

template <typename T, int N>
void do_arithmetic_test() {
  auto q = sycl_cts::util::get_cts_object::queue();

  T value_a = static_cast<T>(42);
  T value_b = static_cast<T>(2);

  auto addition_results = do_addition_test<T, N>(q, value_a, value_b);
  auto substraction_results = do_substraction_test<T, N>(q, value_a, value_b);
  auto division_results = do_division_test<T, N>(q, value_a, value_b);
  auto multiplication_results =
      do_multiplication_test<T, N>(q, value_a, value_b);

  q.wait();

  check_results(addition_results, value_a + value_b);
  check_results(substraction_results, value_a - value_b);
  check_results(division_results, value_a / value_b);
  check_results(multiplication_results, value_a * value_b);
}

}  // namespace vector::operators
