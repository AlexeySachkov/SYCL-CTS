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

enum increment_decrement_check_index : int {
  post_inc_vec_return,
  post_inc_vec_afterwards,
  post_inc_swizzle_return,
  post_inc_swizzle_afterwards,
  pre_inc_vec_return,
  pre_inc_vec_afterwards,
  pre_inc_swizzle_return,
  pre_inc_swizzle_afterwards,
  post_dec_vec_return,
  post_dec_vec_afterwards,
  post_dec_swizzle_return,
  post_dec_swizzle_afterwards,
  pre_dec_vec_return,
  pre_dec_vec_afterwards,
  pre_dec_swizzle_return,
  pre_dec_swizzle_afterwards,
  total_inc_dec
};

enum class arithmetic_operation {
  addition,
  multiplication,
  substraction,
  division,
  increment_decrement
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

#define DO_SWIZZLE_POST_OP(result, v, op)                                     \
  if constexpr (v.size() == 1) {                                              \
    result = v.template swizzle<sycl::elem::s0>() op;                         \
  } else if constexpr (v.size() == 2) {                                       \
    result = v.template swizzle<sycl::elem::s0, sycl::elem::s1>() op;         \
  } else if constexpr (v.size() == 3) {                                       \
    result =                                                                  \
        v.template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2>()  \
            op;                                                               \
  } else if constexpr (v.size() == 4) {                                       \
    result = v.template swizzle<sycl::elem::s0, sycl::elem::s1,               \
                                sycl::elem::s2, sycl::elem::s3>() op;         \
  } else if constexpr (v.size() == 8) {                                       \
    result =                                                                  \
        v.template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2,    \
                           sycl::elem::s3, sycl::elem::s4, sycl::elem::s5,    \
                           sycl::elem::s6, sycl::elem::s7>() op;              \
  } else if constexpr (v.size() == 16) {                                      \
    result = v.template swizzle<                                              \
        sycl::elem::s0, sycl::elem::s1, sycl::elem::s2, sycl::elem::s3,       \
        sycl::elem::s4, sycl::elem::s5, sycl::elem::s6, sycl::elem::s7,       \
        sycl::elem::s8, sycl::elem::s9, sycl::elem::sA, sycl::elem::sB,       \
        sycl::elem::sC, sycl::elem::sD, sycl::elem::sE, sycl::elem::sF>() op; \
  }

#define DO_SWIZZLE_PRE_OP(result, v, op)                                      \
  if constexpr (v.size() == 1) {                                              \
    result = op v.template swizzle<sycl::elem::s0>();                         \
  } else if constexpr (v.size() == 2) {                                       \
    result = op v.template swizzle<sycl::elem::s0, sycl::elem::s1>();         \
  } else if constexpr (v.size() == 3) {                                       \
    result = op v.template swizzle<sycl::elem::s0, sycl::elem::s1,            \
                                   sycl::elem::s2>();                         \
  } else if constexpr (v.size() == 4) {                                       \
    result = op v.template swizzle<sycl::elem::s0, sycl::elem::s1,            \
                                   sycl::elem::s2, sycl::elem::s3>();         \
  } else if constexpr (v.size() == 8) {                                       \
    result =                                                                  \
        op v.template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2, \
                              sycl::elem::s3, sycl::elem::s4, sycl::elem::s5, \
                              sycl::elem::s6, sycl::elem::s7>();              \
  } else if constexpr (v.size() == 16) {                                      \
    result = op v.template swizzle<                                           \
        sycl::elem::s0, sycl::elem::s1, sycl::elem::s2, sycl::elem::s3,       \
        sycl::elem::s4, sycl::elem::s5, sycl::elem::s6, sycl::elem::s7,       \
        sycl::elem::s8, sycl::elem::s9, sycl::elem::sA, sycl::elem::sB,       \
        sycl::elem::sC, sycl::elem::sD, sycl::elem::sE, sycl::elem::sF>();    \
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

template <typename T, int N>
sycl::buffer<sycl::vec<T, N>, 1> do_inc_dec_test(sycl::queue q, T value_a) {
  sycl::buffer<sycl::vec<T, N>, 1> results(
      sycl::range{increment_decrement_check_index::total_inc_dec});

  q.submit([&](sycl::handler& cgh) {
    sycl::accessor acc(results, cgh, sycl::write_only);
    cgh.single_task<
        arithmetic_kernel<arithmetic_operation::increment_decrement, T, N>>(
        [=]() {
          auto vec1 = sycl::vec<T, N>(value_a);
          sycl::vec<T, N> copy;

          copy = vec1;
          acc[post_inc_vec_return] = copy++;
          acc[post_inc_vec_afterwards] = copy;

          copy = vec1;
          DO_SWIZZLE_POST_OP(acc[post_inc_swizzle_return], copy, ++);
          acc[post_inc_swizzle_afterwards] = copy;

          copy = vec1;
          acc[pre_inc_vec_return] = ++copy;
          acc[pre_inc_vec_afterwards] = copy;

          DO_SWIZZLE_PRE_OP(acc[pre_inc_swizzle_return], copy, ++);
          acc[pre_inc_swizzle_afterwards] = copy;

          copy = vec1;
          acc[post_dec_vec_return] = copy--;
          acc[post_dec_vec_afterwards] = copy;

          DO_SWIZZLE_POST_OP(acc[post_dec_swizzle_return], copy, --);
          acc[post_dec_swizzle_afterwards] = copy;

          copy = vec1;
          acc[pre_dec_vec_return] = --copy;
          acc[pre_dec_vec_afterwards] = copy;

          DO_SWIZZLE_PRE_OP(acc[pre_dec_swizzle_return], copy, --);
          acc[pre_dec_swizzle_afterwards] = copy;
        });
  });

  return results;
}

// C++17 does not allow ++ or -- (either prefix or postfix) for the bool type.
// Therefore, the test is essentially skipped for bool.
template <int N>
sycl::buffer<sycl::vec<bool, N>, 1> do_inc_dec_test(sycl::queue q, bool value_a) {
  sycl::buffer<sycl::vec<bool, N>, 1> results(
      sycl::range{increment_decrement_check_index::total_inc_dec});
  return results;
}

template <typename T, int N>
void check_all_of_vec(sycl::vec<T, N> v, T reference) {
  for (int i = 0; i < N; ++i) {
    CHECK(v[i] == reference);
  }
}

template <typename Buf, typename T>
void check_results(Buf buf, T reference) {
  auto acc = buf.get_host_access();
  for (size_t i = 0; i < acc.size(); ++i) {
    check_all_of_vec(acc[i], reference);
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
  auto inc_dec_results = do_inc_dec_test<T, N>(q, value_a);

  q.wait();

  check_results(addition_results, value_a + value_b);
  check_results(substraction_results, value_a - value_b);
  check_results(division_results, value_a / value_b);
  check_results(multiplication_results, value_a * value_b);
  if constexpr(!std::is_same_v<T, bool>) {
    auto acc = inc_dec_results.get_host_access();
    check_all_of_vec(acc[post_inc_vec_return], value_a);
    check_all_of_vec(acc[post_inc_vec_afterwards], value_a + 1);
    check_all_of_vec(acc[post_inc_swizzle_return], value_a);
    check_all_of_vec(acc[post_inc_swizzle_afterwards], value_a + 1);

    check_all_of_vec(acc[pre_inc_vec_return], value_a + 1);
    check_all_of_vec(acc[pre_inc_vec_afterwards], value_a + 1);
    check_all_of_vec(acc[pre_inc_swizzle_return], value_a + 1);
    check_all_of_vec(acc[pre_inc_swizzle_afterwards], value_a + 1);

    check_all_of_vec(acc[post_dec_vec_return], value_a);
    check_all_of_vec(acc[post_dec_vec_afterwards], value_a - 1);
    check_all_of_vec(acc[post_dec_swizzle_return], value_a);
    check_all_of_vec(acc[post_dec_swizzle_afterwards], value_a - 1);

    check_all_of_vec(acc[pre_dec_vec_return], value_a - 1);
    check_all_of_vec(acc[pre_dec_vec_afterwards], value_a - 1);
    check_all_of_vec(acc[pre_dec_swizzle_return], value_a - 1);
    check_all_of_vec(acc[pre_dec_swizzle_afterwards], value_a - 1);
  }
}

}  // namespace vector::operators
