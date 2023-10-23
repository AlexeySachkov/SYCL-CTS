#pragma once

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
#define CHECK_BINARY_OP_KERNEL_BODY(op) \
            auto vec1 = sycl::vec<T, N>(this->value_a);\
            auto vec2 = sycl::vec<T, N>(this->value_b);\
\
            this->acc[vec_vec] = vec1 op vec2;\
            DO_VEC_SWIZZLE_OP(this->acc[vec_swizzle], vec1, vec2, op);\
            this->acc[vec_scalar] = vec1 op this->value_b;\
            DO_SWIZZLE_VEC_OP(this->acc[swizzle_vec], vec1, vec2, op);\
            DO_SWIZZLE_SWIZZLE_OP(this->acc[swizzle_swizzle], vec1, vec2, op);\
            DO_SWIZZLE_SCALAR_OP(this->acc[swizzle_scalar], vec1, this->value_b, op);\
            this->acc[scalar_vec] = this->value_a op vec2;\
            DO_SCALAR_SWIZZLE_OP(this->acc[scalar_swizzle], this->value_a, vec2, op);

enum class arithmetic_binary_operator {
  plus,
  minus,
  multiply,
  divide,
  reminder
};

template<typename T, int N>
class arithmetic_binary_operator_kernel_functor_base {
 public:
  arithmetic_binary_operator_kernel_functor_base(
      sycl::accessor<sycl::vec<T, N>, 1, sycl::access_mode::write> acc,
      T value_a, T value_b)
      : acc(acc), value_a(value_a), value_b(value_b) {}

 protected:
  sycl::accessor<sycl::vec<T, N>, 1, sycl::access_mode::write> acc;
  T value_a;
  T value_b;
};

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
