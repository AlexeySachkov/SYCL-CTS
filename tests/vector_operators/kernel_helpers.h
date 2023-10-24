#pragma once

#include "operator_helpers.h"

enum arithmetic_operation_kind : int {
  vec_vec = 0,
  vec_swizzle,
  vec_scalar,
  swizzle_vec,
  swizzle_swizzle,
  swizzle_scalar,
  scalar_vec,
  scalar_swizzle,
  size,
  assignment_size = swizzle_scalar
};

enum unary_op_kind : int { vec = 0, swizzle, total };

#define CHECK_BINARY_OP_KERNEL_BODY(op)                                     \
  auto vec1 = sycl::vec<T, N>(this->value_a);                               \
  auto vec2 = sycl::vec<T, N>(this->value_b);                               \
                                                                            \
  this->acc[vec_vec] = vec1 op vec2;                                        \
  DO_VEC_SWIZZLE_OP(this->acc[vec_swizzle], vec1, vec2, op);                \
  this->acc[vec_scalar] = vec1 op this->value_b;                            \
  DO_SWIZZLE_VEC_OP(this->acc[swizzle_vec], vec1, vec2, op);                \
  DO_SWIZZLE_SWIZZLE_OP(this->acc[swizzle_swizzle], vec1, vec2, op);        \
  DO_SWIZZLE_SCALAR_OP(this->acc[swizzle_scalar], vec1, this->value_b, op); \
  this->acc[scalar_vec] = this->value_a op vec2;                            \
  DO_SCALAR_SWIZZLE_OP(this->acc[scalar_swizzle], this->value_a, vec2, op);

#define CHECK_BINARY_ASSIGNMENT_OP_KERNEL_BODY(op)                             \
  auto vec1 = sycl::vec<T, N>(this->value_a);                                  \
  auto vec2 = sycl::vec<T, N>(this->value_b);                                  \
                                                                               \
  this->acc[vec_vec] = vec1;                                                   \
  this->acc[vec_vec] op vec2;                                                  \
  this->acc[vec_swizzle] = vec1;                                               \
  DO_VEC_SWIZZLE_ASSIGNMENT_OP(this->acc[vec_swizzle], vec2, op);              \
  this->acc[vec_scalar] = vec1;                                                \
  this->acc[vec_scalar] op this->value_b;                                      \
  this->acc[swizzle_vec] = vec1;                                               \
  DO_SWIZZLE_VEC_ASSIGNMENT_OP(this->acc[swizzle_vec], vec2, op);              \
  this->acc[swizzle_swizzle] = vec1;                                           \
  DO_SWIZZLE_SWIZZLE_ASSIGNMENT_OP(this->acc[swizzle_swizzle], vec2, op);      \
  this->acc[swizzle_scalar] = vec1;                                            \
  DO_SWIZZLE_SCALAR_ASSIGNMENT_OP(this->acc[swizzle_scalar], N, this->value_b, \
                                  op);

#define CHECK_UNARY_OP_KERNEL_BODY(op)        \
  auto vec1 = sycl::vec<T, N>(this->value_a); \
  this->acc[vec] = op vec1;                   \
  DO_SWIZZLE_UNARY_OP(this->acc[swizzle], vec1, op);

enum class arithmetic_binary_operator {
  plus,
  minus,
  multiply,
  divide,
  reminder,
  bitwise_and,
  bitwise_or,
  bitwise_xor,
  logical_and,
  logical_or,
  logical_not,
  equal,
  not_equal,
  less_or_equal,
  greater_or_equal,
  less,
  greater,
  unary_plus,
  unary_minus,
  bitwise_shift_left,
  bitwise_shift_right
};

template <typename T, int N>
class arithmetic_binary_operator_kernel_functor_base {
 public:
  arithmetic_binary_operator_kernel_functor_base(
      sycl::accessor<sycl::vec<T, N>> acc, T value_a, T value_b)
      : acc(acc), value_a(value_a), value_b(value_b) {}

 protected:
  sycl::accessor<sycl::vec<T, N>> acc;
  T value_a;
  T value_b;
};

// binary arithmetic: +, -, *, /, %

template <arithmetic_binary_operator Op, typename T, int N>
class arithmetic_binary_operator_kernel_functor
    : public arithmetic_binary_operator_kernel_functor_base<T, N> {};

template <typename T, int N>
class arithmetic_binary_operator_kernel_functor<
    arithmetic_binary_operator::plus, T, N>
    : public arithmetic_binary_operator_kernel_functor_base<T, N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      T, N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_OP_KERNEL_BODY(+) }
};

template <typename T, int N>
class arithmetic_binary_operator_kernel_functor<
    arithmetic_binary_operator::minus, T, N>
    : public arithmetic_binary_operator_kernel_functor_base<T, N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      T, N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_OP_KERNEL_BODY(-) }
};

template <typename T, int N>
class arithmetic_binary_operator_kernel_functor<
    arithmetic_binary_operator::divide, T, N>
    : public arithmetic_binary_operator_kernel_functor_base<T, N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      T, N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_OP_KERNEL_BODY(/) }
};

template <typename T, int N>
class arithmetic_binary_operator_kernel_functor<
    arithmetic_binary_operator::multiply, T, N>
    : public arithmetic_binary_operator_kernel_functor_base<T, N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      T, N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_OP_KERNEL_BODY(*) }
};

template <typename T, int N>
class arithmetic_binary_operator_kernel_functor<
    arithmetic_binary_operator::reminder, T, N>
    : public arithmetic_binary_operator_kernel_functor_base<T, N> {
  static_assert(!std::is_floating_point_v<T> && !std::is_same_v<T, sycl::half>);

 public:
  using arithmetic_binary_operator_kernel_functor_base<
      T, N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_OP_KERNEL_BODY(%) }
};

// binary assignment arithmetic: +=, -=, *=, /=, %=

template <arithmetic_binary_operator Op, typename T, int N>
class arithmetic_binary_assignment_operator_kernel_functor
    : public arithmetic_binary_operator_kernel_functor_base<T, N> {};

template <typename T, int N>
class arithmetic_binary_assignment_operator_kernel_functor<
    arithmetic_binary_operator::plus, T, N>
    : public arithmetic_binary_operator_kernel_functor_base<T, N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      T, N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_ASSIGNMENT_OP_KERNEL_BODY(+=) }
};

template <typename T, int N>
class arithmetic_binary_assignment_operator_kernel_functor<
    arithmetic_binary_operator::minus, T, N>
    : public arithmetic_binary_operator_kernel_functor_base<T, N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      T, N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_ASSIGNMENT_OP_KERNEL_BODY(-=) }
};

template <typename T, int N>
class arithmetic_binary_assignment_operator_kernel_functor<
    arithmetic_binary_operator::multiply, T, N>
    : public arithmetic_binary_operator_kernel_functor_base<T, N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      T, N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_ASSIGNMENT_OP_KERNEL_BODY(*=) }
};

template <typename T, int N>
class arithmetic_binary_assignment_operator_kernel_functor<
    arithmetic_binary_operator::divide, T, N>
    : public arithmetic_binary_operator_kernel_functor_base<T, N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      T, N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_ASSIGNMENT_OP_KERNEL_BODY(/=) }
};

template <typename T, int N>
class arithmetic_binary_assignment_operator_kernel_functor<
    arithmetic_binary_operator::reminder, T, N>
    : public arithmetic_binary_operator_kernel_functor_base<T, N> {
  static_assert(!std::is_floating_point_v<T> && !std::is_same_v<T, sycl::half>);

 public:
  using arithmetic_binary_operator_kernel_functor_base<
      T, N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_ASSIGNMENT_OP_KERNEL_BODY(%=) }
};

// bitwise binary: &, |, ^, <<, >>

template <arithmetic_binary_operator Op, typename T, int N>
class bitwise_binary_operator_kernel_functor
    : public arithmetic_binary_operator_kernel_functor_base<T, N> {};

template <typename T, int N>
class bitwise_binary_operator_kernel_functor<
    arithmetic_binary_operator::bitwise_and, T, N>
    : public arithmetic_binary_operator_kernel_functor_base<T, N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      T, N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_OP_KERNEL_BODY(&) }
};

template <typename T, int N>
class bitwise_binary_operator_kernel_functor<
    arithmetic_binary_operator::bitwise_or, T, N>
    : public arithmetic_binary_operator_kernel_functor_base<T, N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      T, N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_OP_KERNEL_BODY(|) }
};

template <typename T, int N>
class bitwise_binary_operator_kernel_functor<
    arithmetic_binary_operator::bitwise_xor, T, N>
    : public arithmetic_binary_operator_kernel_functor_base<T, N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      T, N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_OP_KERNEL_BODY(^) }
};

template <typename T, int N>
class bitwise_binary_operator_kernel_functor<
    arithmetic_binary_operator::bitwise_shift_left, T, N>
    : public arithmetic_binary_operator_kernel_functor_base<T, N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      T, N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_OP_KERNEL_BODY(<<) }
};

template <typename T, int N>
class bitwise_binary_operator_kernel_functor<
    arithmetic_binary_operator::bitwise_shift_right, T, N>
    : public arithmetic_binary_operator_kernel_functor_base<T, N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      T, N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_OP_KERNEL_BODY(>>) }
};

// logical: &&, ||, !

template <typename T, arithmetic_binary_operator Op, typename Enable = void>
struct result_type {
  using type = T;
};

template <typename T, typename T8 = std::int8_t, typename T16 = std::int16_t,
          typename T32 = std::int32_t, typename T64 = std::int64_t>
struct select_fixed_size_type {
  using type = std::conditional_t<
      sizeof(T) == 1, T8,
      std::conditional_t<sizeof(T) == 2, T16,
                         std::conditional_t<sizeof(T) == 4, T32, T64>>>;
};

template <typename T, arithmetic_binary_operator Op>
struct result_type<T, Op,
                   typename std::enable_if_t<(
                       Op == arithmetic_binary_operator::logical_or ||
                       Op == arithmetic_binary_operator::logical_and ||
                       Op == arithmetic_binary_operator::logical_not)>> {
  using type = typename select_fixed_size_type<T>::type;
};

template <arithmetic_binary_operator Op, typename T, int N>
class logical_operator_kernel_functor
    : public arithmetic_binary_operator_kernel_functor_base<
          typename result_type<T, Op>::type, N> {};

template <typename T, int N>
class logical_operator_kernel_functor<arithmetic_binary_operator::logical_and,
                                      T, N>
    : public arithmetic_binary_operator_kernel_functor_base<
          typename result_type<T,
                               arithmetic_binary_operator::logical_and>::type,
          N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      typename result_type<T, arithmetic_binary_operator::logical_and>::type,
      N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_OP_KERNEL_BODY(&&) }
};

template <typename T, int N>
class logical_operator_kernel_functor<arithmetic_binary_operator::logical_or, T,
                                      N>
    : public arithmetic_binary_operator_kernel_functor_base<
          typename result_type<T, arithmetic_binary_operator::logical_or>::type,
          N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      typename result_type<T, arithmetic_binary_operator::logical_or>::type,
      N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_OP_KERNEL_BODY(||) }
};

template <typename T, int N>
class logical_operator_kernel_functor<arithmetic_binary_operator::logical_not,
                                      T, N>
    : public arithmetic_binary_operator_kernel_functor_base<
          typename result_type<T,
                               arithmetic_binary_operator::logical_not>::type,
          N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      typename result_type<T, arithmetic_binary_operator::logical_not>::type,
      N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_UNARY_OP_KERNEL_BODY(!) }
};

// relational: ==, !=, <=, >=, <, >

template <arithmetic_binary_operator Op, typename T, int N>
class relational_operator_kernel_functor
    : public arithmetic_binary_operator_kernel_functor_base<
          typename result_type<T, Op>::type, N> {};

template <typename T, int N>
class relational_operator_kernel_functor<arithmetic_binary_operator::equal, T,
                                         N>
    : public arithmetic_binary_operator_kernel_functor_base<
          typename result_type<T, arithmetic_binary_operator::equal>::type, N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      typename result_type<T, arithmetic_binary_operator::equal>::type,
      N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_OP_KERNEL_BODY(==) }
};

template <typename T, int N>
class relational_operator_kernel_functor<arithmetic_binary_operator::not_equal,
                                         T, N>
    : public arithmetic_binary_operator_kernel_functor_base<
          typename result_type<T, arithmetic_binary_operator::not_equal>::type,
          N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      typename result_type<T, arithmetic_binary_operator::not_equal>::type,
      N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_OP_KERNEL_BODY(!=) }
};

template <typename T, int N>
class relational_operator_kernel_functor<
    arithmetic_binary_operator::less_or_equal, T, N>
    : public arithmetic_binary_operator_kernel_functor_base<
          typename result_type<T,
                               arithmetic_binary_operator::less_or_equal>::type,
          N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      typename result_type<T, arithmetic_binary_operator::less_or_equal>::type,
      N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_OP_KERNEL_BODY(<=) }
};

template <typename T, int N>
class relational_operator_kernel_functor<
    arithmetic_binary_operator::greater_or_equal, T, N>
    : public arithmetic_binary_operator_kernel_functor_base<
          typename result_type<
              T, arithmetic_binary_operator::greater_or_equal>::type,
          N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      typename result_type<T,
                           arithmetic_binary_operator::greater_or_equal>::type,
      N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_OP_KERNEL_BODY(>=) }
};

template <typename T, int N>
class relational_operator_kernel_functor<arithmetic_binary_operator::less, T, N>
    : public arithmetic_binary_operator_kernel_functor_base<
          typename result_type<T, arithmetic_binary_operator::less>::type, N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      typename result_type<T, arithmetic_binary_operator::less>::type,
      N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_OP_KERNEL_BODY(<) }
};

template <typename T, int N>
class relational_operator_kernel_functor<arithmetic_binary_operator::greater, T,
                                         N>
    : public arithmetic_binary_operator_kernel_functor_base<
          typename result_type<T, arithmetic_binary_operator::greater>::type,
          N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      typename result_type<T, arithmetic_binary_operator::greater>::type,
      N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_BINARY_OP_KERNEL_BODY(>) }
};

// unary arithmetic: +, -

template <arithmetic_binary_operator Op, typename T, int N>
class unary_arithmetic_operator_kernel_functor
    : public arithmetic_binary_operator_kernel_functor_base<T, N> {};

template <typename T, int N>
class unary_arithmetic_operator_kernel_functor<
    arithmetic_binary_operator::unary_plus, T, N>
    : public arithmetic_binary_operator_kernel_functor_base<T, N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      T, N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_UNARY_OP_KERNEL_BODY(+) }
};

template <typename T, int N>
class unary_arithmetic_operator_kernel_functor<
    arithmetic_binary_operator::unary_minus, T, N>
    : public arithmetic_binary_operator_kernel_functor_base<T, N> {
 public:
  using arithmetic_binary_operator_kernel_functor_base<
      T, N>::arithmetic_binary_operator_kernel_functor_base;
  void operator()() const { CHECK_UNARY_OP_KERNEL_BODY(-) }
};
