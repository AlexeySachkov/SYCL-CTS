#include <sycl/sycl.hpp>

#include "../common/common.h"
#include "../common/common_vec.h"
#include "../common/type_coverage.h"

#include "kernel_helpers.h"

#include <algorithm>
#include <array>

namespace vector::operators {

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
  reminder,
  increment_decrement
};

std::string to_string(arithmetic_operation op,
                      arithmetic_operation_kind k = size) {
  std::string result;
  switch (op) {
    case arithmetic_operation::addition:
      result = "operator+";
      break;
    case arithmetic_operation::substraction:
      result = "operator-";
      break;
    case arithmetic_operation::division:
      result = "operator/";
      break;
    case arithmetic_operation::multiplication:
      result = "operator*";
      break;
    case arithmetic_operation::reminder:
      result = "operator%";
      break;
    default:
      result = "unknown operation";
  }

  switch (k) {
    case vec_vec:
      result += "(vec, vec)";
      break;
    case vec_swizzle:
      result += "(vec, swizzle)";
      break;
    case vec_scalar:
      result += "(vec, scalar)";
      break;
    case swizzle_vec:
      result += "(swizzle, vec)";
      break;
    case swizzle_swizzle:
      result += "(swizzle, swizzle)";
      break;
    case swizzle_scalar:
      result += "(swizzle, scalar)";
      break;
    case scalar_vec:
      result += "(scalar, vec)";
      break;
    case scalar_swizzle:
      result += "(scalar, swizzle)";
      break;
    case size:
      // do nothing
      break;
    default:
      result += "unknown arguments";
  }

  return result;
}

std::string to_string(arithmetic_operation_kind k) {
  switch (k) {
    case vec_vec:
      return "vec op vec";
    case vec_swizzle:
      return "vec op swizzle";
    case vec_scalar:
      return "vec op scalar";
    case swizzle_vec:
      return "swizzle op vec";
    case swizzle_swizzle:
      return "swizzle op swizzle";
    case swizzle_scalar:
      return "swizzle op scalar";
    case scalar_vec:
      return "scalar op vec";
    case scalar_swizzle:
      return "scalar op swizzle";
    default:
      return "unknown test case";
  }
}

template <typename T, int N>
void check_all_of_vec(sycl::vec<T, N> v, T reference) {
  for (int i = 0; i < N; ++i) {
    INFO("of vector element #" << i);
    CHECK(v[i] == reference);
  }
}

template <typename Buf, typename T>
void check_results(Buf buf, T reference) {
  auto acc = buf.get_host_access();
  for (size_t i = 0; i < acc.size(); ++i) {
    INFO("Validating results for operation #" << i);
    check_all_of_vec(acc[i], reference);
  }
}

template <arithmetic_binary_operator Op, typename Buf, typename T>
void check_binop_results(Buf buf, T reference) {
  auto acc = buf.get_host_access();
  for (size_t i = 0; i < acc.size(); ++i) {
    INFO(
        "of use case " << to_string(static_cast<arithmetic_operation_kind>(i)));
    check_all_of_vec(acc[i], reference);
  }
}

template <arithmetic_operation op, typename T, int N>
class arithmetic_kernel;

template <arithmetic_binary_operator Op, typename T>
typename result_type<T, Op>::type reference(T a, T b) {
  if constexpr (arithmetic_binary_operator::plus == Op)
    return a + b;
  else if constexpr (arithmetic_binary_operator::minus == Op)
    return a - b;
  else if constexpr (arithmetic_binary_operator::multiply == Op)
    return a * b;
  else if constexpr (arithmetic_binary_operator::divide == Op)
    return a / b;
  else if constexpr (arithmetic_binary_operator::reminder == Op) {
    return a % b;
  } else if constexpr (arithmetic_binary_operator::bitwise_and == Op) {
    return a & b;
  } else if constexpr (arithmetic_binary_operator::bitwise_or == Op) {
    return a | b;
  } else if constexpr (arithmetic_binary_operator::bitwise_xor == Op) {
    return a ^ b;
  } else if constexpr (arithmetic_binary_operator::logical_and == Op) {
    // FIXME: why - is required?
    return -(a && b);
  } else if constexpr (arithmetic_binary_operator::logical_or == Op) {
    // FIXME: why - is required?
    return -(a || b);
  } else if constexpr (arithmetic_binary_operator::logical_not == Op) {
    // FIXME: why - is required?
    return -!a;
  } else if constexpr (arithmetic_binary_operator::equal == Op) {
    return -(a == b);
  } else if constexpr (arithmetic_binary_operator::not_equal == Op) {
    return -(a != b);
  } else if constexpr (arithmetic_binary_operator::less_or_equal == Op) {
    return -(a <= b);
  } else if constexpr (arithmetic_binary_operator::greater_or_equal == Op) {
    return -(a >= b);
  } else if constexpr (arithmetic_binary_operator::less == Op) {
    return -(a < b);
  } else if constexpr (arithmetic_binary_operator::greater == Op) {
    return -(a > b);
  } else if constexpr (arithmetic_binary_operator::unary_plus == Op) {
    return +a;
  } else if constexpr (arithmetic_binary_operator::unary_minus == Op) {
    return -a;
  } else {
    assert(false && "unsupported operator");
  }
}

template <typename T, typename SizeT, typename OpT, typename Enable = void>
struct check_arithmetic_binary_operator {
  void operator()(const std::string&) {}
};

template <typename T, typename SizeT, typename OpT>
struct check_arithmetic_binary_operator<
    T, SizeT, OpT,
    std::enable_if_t<OpT::value != arithmetic_binary_operator::reminder ||
                     (!std::is_floating_point_v<T> &&
                      !std::is_same_v<T, sycl::half>)>> {
  static constexpr arithmetic_binary_operator Op = OpT::value;
  static constexpr int N = SizeT::value;
  void operator()(const std::string& operator_name) {
    INFO("Checking binary arithmetic operator " << operator_name);

    auto q = sycl_cts::util::get_cts_object::queue();

    T value_a = static_cast<T>(42);
    T value_b = static_cast<T>(2);

    sycl::buffer<sycl::vec<T, N>, 1> results(
        sycl::range{arithmetic_operation_kind::size});
    {
      INFO("Submitting a kernel");
      q.submit([&](sycl::handler& cgh) {
        sycl::accessor acc(results, cgh);
        arithmetic_binary_operator_kernel_functor<Op, T, N> f(acc, value_a,
                                                              value_b);
        cgh.single_task(f);
      });
    }
    INFO("Validating results");
    check_binop_results<Op>(results, reference<Op>(value_a, value_b));
  }
};

template <typename T, typename SizeT, typename OpT, typename Enable = void>
struct check_arithmetic_binary_assignment_operator {
  void operator()(const std::string&) {}
};

template <typename T, typename SizeT, typename OpT>
struct check_arithmetic_binary_assignment_operator<
    T, SizeT, OpT,
    std::enable_if_t<OpT::value != arithmetic_binary_operator::reminder ||
                     (!std::is_floating_point_v<T> &&
                      !std::is_same_v<T, sycl::half>)>> {
  static constexpr arithmetic_binary_operator Op = OpT::value;
  static constexpr int N = SizeT::value;
  void operator()(const std::string& operator_name) {
    INFO("Checking binary arithmetic asignment operator: " << operator_name);

    auto q = sycl_cts::util::get_cts_object::queue();

    T value_a = static_cast<T>(42);
    T value_b = static_cast<T>(2);

    sycl::buffer<sycl::vec<T, N>, 1> results(
        sycl::range{arithmetic_operation_kind::assignment_size});
    {
      INFO("Submitting a kernel");
      q.submit([&](sycl::handler& cgh) {
         sycl::accessor acc(results, cgh);
         arithmetic_binary_assignment_operator_kernel_functor<Op, T, N> f(
             acc, value_a, value_b);
         cgh.single_task(f);
       }).wait();
    }

    INFO("Validating results");
    check_binop_results<Op>(results, reference<Op>(value_a, value_b));
  }
};

template <typename T, typename SizeT, typename OpT>
struct check_bitwise_binary_operator {
  static constexpr arithmetic_binary_operator Op = OpT::value;
  static constexpr int N = SizeT::value;
  void operator()(const std::string& operator_name) {
    INFO("Checking bitwise operator: " << operator_name);

    auto q = sycl_cts::util::get_cts_object::queue();

    T value_a = static_cast<T>(42);
    T value_b = static_cast<T>(2);

    sycl::buffer<sycl::vec<T, N>, 1> results(
        sycl::range{arithmetic_operation_kind::size});
    {
      INFO("Submitting a kernel");
      q.submit([&](sycl::handler& cgh) {
        sycl::accessor acc(results, cgh);
        bitwise_binary_operator_kernel_functor<Op, T, N> f(acc, value_a,
                                                           value_b);
        cgh.single_task(f);
      });
    }
    INFO("Validating results");
    check_binop_results<Op>(results, reference<Op>(value_a, value_b));
  }
};

template <typename T, typename SizeT, typename OpT>
struct check_logical_operator {
  static constexpr arithmetic_binary_operator Op = OpT::value;
  static constexpr int N = SizeT::value;
  void operator()(const std::string& operator_name) {
    INFO("Checking bitwise operator: " << operator_name);

    auto q = sycl_cts::util::get_cts_object::queue();

    T value_a = static_cast<T>(42);
    T value_b = static_cast<T>(2);

    constexpr int size = arithmetic_binary_operator::logical_not == Op
                             ? unary_op_kind::total
                             : arithmetic_operation_kind::size;
    sycl::buffer<sycl::vec<typename result_type<T, Op>::type, N>, 1> results(
        sycl::range{size});
    {
      INFO("Submitting a kernel");
      q.submit([&](sycl::handler& cgh) {
        sycl::accessor acc(results, cgh);
        logical_operator_kernel_functor<Op, T, N> f(acc, value_a, value_b);
        cgh.single_task(f);
      });
    }
    INFO("Validating results");
    check_binop_results<Op>(results, reference<Op>(value_a, value_b));
  }
};

template <typename T, typename SizeT, typename OpT>
struct check_relational_operator {
  static constexpr arithmetic_binary_operator Op = OpT::value;
  static constexpr int N = SizeT::value;
  void operator()(const std::string& operator_name) {
    INFO("Checking bitwise operator: " << operator_name);

    auto q = sycl_cts::util::get_cts_object::queue();

    T value_a = static_cast<T>(42);
    T value_b = static_cast<T>(2);

    sycl::buffer<sycl::vec<typename result_type<T, Op>::type, N>, 1> results(
        sycl::range{arithmetic_operation_kind::size});
    {
      INFO("Submitting a kernel");
      q.submit([&](sycl::handler& cgh) {
        sycl::accessor acc(results, cgh);
        relational_operator_kernel_functor<Op, T, N> f(acc, value_a, value_b);
        cgh.single_task(f);
      });
    }
    INFO("Validating results");
    check_binop_results<Op>(results, reference<Op>(value_a, value_b));
  }
};

template <typename T, typename SizeT, typename OpT>
struct check_unary_arithmetic_operator {
  static constexpr arithmetic_binary_operator Op = OpT::value;
  static constexpr int N = SizeT::value;
  void operator()(const std::string& operator_name) {
    INFO("Checking unary operator: " << operator_name);

    auto q = sycl_cts::util::get_cts_object::queue();

    T value_a = static_cast<T>(42);
    T value_b = static_cast<T>(2);

    constexpr int size = unary_op_kind::total;
    sycl::buffer<sycl::vec<T, N>, 1> results(sycl::range{size});
    {
      INFO("Submitting a kernel");
      q.submit([&](sycl::handler& cgh) {
        sycl::accessor acc(results, cgh);
        unary_arithmetic_operator_kernel_functor<Op, T, N> f(acc, value_a,
                                                             value_b);
        cgh.single_task(f);
      });
    }
    INFO("Validating results");
    check_binop_results<Op>(results, reference<Op>(value_a, value_b));
  }
};

template <typename T, int N>
sycl::buffer<sycl::vec<T, N>, 1> do_inc_dec_test(sycl::queue q, T value_a) {
  INFO("Submitting kernel for testing increment/decrement operations");
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

          copy = vec1;
          DO_SWIZZLE_PRE_OP(acc[pre_inc_swizzle_return], copy, ++);
          acc[pre_inc_swizzle_afterwards] = copy;

          copy = vec1;
          acc[post_dec_vec_return] = copy--;
          acc[post_dec_vec_afterwards] = copy;

          copy = vec1;
          DO_SWIZZLE_POST_OP(acc[post_dec_swizzle_return], copy, --);
          acc[post_dec_swizzle_afterwards] = copy;

          copy = vec1;
          acc[pre_dec_vec_return] = --copy;
          acc[pre_dec_vec_afterwards] = copy;

          copy = vec1;
          DO_SWIZZLE_PRE_OP(acc[pre_dec_swizzle_return], copy, --);
          acc[pre_dec_swizzle_afterwards] = copy;
        });
  });

  return results;
}

// C++17 does not allow ++ or -- (either prefix or postfix) for the bool type.
// Therefore, the test is essentially skipped for bool.
template <int N>
sycl::buffer<sycl::vec<bool, N>, 1> do_inc_dec_test(sycl::queue q,
                                                    bool value_a) {
  sycl::buffer<sycl::vec<bool, N>, 1> results(
      sycl::range{increment_decrement_check_index::total_inc_dec});
  return results;
}

template <typename T, int N>
void check_all_operators() {
  auto q = sycl_cts::util::get_cts_object::queue();

  auto arithmetic_binary_ops = value_pack<
      arithmetic_binary_operator, arithmetic_binary_operator::plus,
      arithmetic_binary_operator::minus, arithmetic_binary_operator::multiply,
      arithmetic_binary_operator::divide,
      arithmetic_binary_operator::reminder>::generate_named("operator+",
                                                            "operator-",
                                                            "operator*",
                                                            "operator/",
                                                            "operator%");
  for_all_combinations<check_arithmetic_binary_operator, T,
                       std::integral_constant<int, N>>(arithmetic_binary_ops);

  auto arithmetic_binary_assignment_ops = value_pack<
      arithmetic_binary_operator, arithmetic_binary_operator::plus,
      arithmetic_binary_operator::minus, arithmetic_binary_operator::multiply,
      arithmetic_binary_operator::divide,
      arithmetic_binary_operator::reminder>::generate_named("operator+=",
                                                            "operator-=",
                                                            "operator*=",
                                                            "operator/=",
                                                            "operator%=");
  for_all_combinations<check_arithmetic_binary_assignment_operator, T,
                       std::integral_constant<int, N>>(
      arithmetic_binary_assignment_ops);

  auto bitwise_binary_ops = value_pack<
      arithmetic_binary_operator, arithmetic_binary_operator::bitwise_and,
      arithmetic_binary_operator::bitwise_or,
      arithmetic_binary_operator::bitwise_xor>::generate_named("operator&",
                                                               "operator|",
                                                               "operator^");
  for_all_combinations<check_bitwise_binary_operator, T,
                       std::integral_constant<int, N>>(bitwise_binary_ops);

  auto logical_ops = value_pack<
      arithmetic_binary_operator, arithmetic_binary_operator::logical_and,
      arithmetic_binary_operator::logical_or,
      arithmetic_binary_operator::logical_not>::generate_named("operator&&",
                                                               "operator||",
                                                               "operator!");
  for_all_combinations<check_logical_operator, T,
                       std::integral_constant<int, N>>(logical_ops);

  auto relational_ops =
      value_pack<arithmetic_binary_operator, arithmetic_binary_operator::equal,
                 arithmetic_binary_operator::not_equal,
                 arithmetic_binary_operator::less_or_equal,
                 arithmetic_binary_operator::greater_or_equal,
                 arithmetic_binary_operator::less,
                 arithmetic_binary_operator::greater>::
          generate_named("operator==", "operator!=", "operator<=", "operator>=",
                         "operator<", "operator>");
  for_all_combinations<check_relational_operator, T,
                       std::integral_constant<int, N>>(relational_ops);

  auto unary_ops = value_pack<
      arithmetic_binary_operator, arithmetic_binary_operator::unary_plus,
      arithmetic_binary_operator::unary_minus>::generate_named("operator+",
                                                               "operator-");
  for_all_combinations<check_unary_arithmetic_operator, T,
                       std::integral_constant<int, N>>(unary_ops);

  T value_a = static_cast<T>(42);
  T value_b = static_cast<T>(2);

  auto inc_dec_results = do_inc_dec_test<T, N>(q, value_a);
  // TODO: bitwise operators: >>, <<
  // TODO: bitwise assignment operators: |=, ^=, &=, >>=, <<=
  // TODO: subscript operator: []
  // TODO: conversion operators: vector_t(), DataT()
  // TODO: assignment operators: =
  // TODO: bitwise unary: ~

  q.wait();

  if constexpr (!std::is_same_v<T, bool>) {
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
