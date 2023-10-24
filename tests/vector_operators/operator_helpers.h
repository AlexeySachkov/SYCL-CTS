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

#define DO_VEC_SWIZZLE_ASSIGNMENT_OP(result, v, op)                           \
  if constexpr (v.size() == 1) {                                              \
    result op v.template swizzle<sycl::elem::s0>();                           \
  } else if constexpr (v.size() == 2) {                                       \
    result op v.template swizzle<sycl::elem::s0, sycl::elem::s1>();           \
  } else if constexpr (v.size() == 3) {                                       \
    result op                                                                 \
        v.template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2>(); \
  } else if constexpr (v.size() == 4) {                                       \
    result op v.template swizzle<sycl::elem::s0, sycl::elem::s1,              \
                                 sycl::elem::s2, sycl::elem::s3>();           \
  } else if constexpr (v.size() == 8) {                                       \
    result op v.template swizzle<                                             \
        sycl::elem::s0, sycl::elem::s1, sycl::elem::s2, sycl::elem::s3,       \
        sycl::elem::s4, sycl::elem::s5, sycl::elem::s6, sycl::elem::s7>();    \
  } else if constexpr (v.size() == 16) {                                      \
    result op v.template swizzle<                                             \
        sycl::elem::s0, sycl::elem::s1, sycl::elem::s2, sycl::elem::s3,       \
        sycl::elem::s4, sycl::elem::s5, sycl::elem::s6, sycl::elem::s7,       \
        sycl::elem::s8, sycl::elem::s9, sycl::elem::sA, sycl::elem::sB,       \
        sycl::elem::sC, sycl::elem::sD, sycl::elem::sE, sycl::elem::sF>();    \
  }

#define DO_SWIZZLE_SWIZZLE_ASSIGNMENT_OP(result, v, op)                        \
  if constexpr (v.size() == 1) {                                               \
    result.template swizzle<sycl::elem::s0>()                                  \
        op v.template swizzle<sycl::elem::s0>();                               \
  } else if constexpr (v.size() == 2) {                                        \
    result.template swizzle<sycl::elem::s0, sycl::elem::s1>()                  \
        op v.template swizzle<sycl::elem::s0, sycl::elem::s1>();               \
  } else if constexpr (v.size() == 3) {                                        \
    result                                                                     \
        .template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2>()    \
            op v                                                               \
        .template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2>();   \
  } else if constexpr (v.size() == 4) {                                        \
    result                                                                     \
        .template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2,      \
                          sycl::elem::s3>()                                    \
            op v.template swizzle<sycl::elem::s0, sycl::elem::s1,              \
                                  sycl::elem::s2, sycl::elem::s3>();           \
  } else if constexpr (v.size() == 8) {                                        \
    result                                                                     \
        .template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2,      \
                          sycl::elem::s3, sycl::elem::s4, sycl::elem::s5,      \
                          sycl::elem::s6, sycl::elem::s7>() op v               \
        .template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2,      \
                          sycl::elem::s3, sycl::elem::s4, sycl::elem::s5,      \
                          sycl::elem::s6, sycl::elem::s7>();                   \
  } else if constexpr (v.size() == 16) {                                       \
    result                                                                     \
        .template swizzle<                                                     \
            sycl::elem::s0, sycl::elem::s1, sycl::elem::s2, sycl::elem::s3,    \
            sycl::elem::s4, sycl::elem::s5, sycl::elem::s6, sycl::elem::s7,    \
            sycl::elem::s8, sycl::elem::s9, sycl::elem::sA, sycl::elem::sB,    \
            sycl::elem::sC, sycl::elem::sD, sycl::elem::sE, sycl::elem::sF>()  \
            op v                                                               \
        .template swizzle<                                                     \
            sycl::elem::s0, sycl::elem::s1, sycl::elem::s2, sycl::elem::s3,    \
            sycl::elem::s4, sycl::elem::s5, sycl::elem::s6, sycl::elem::s7,    \
            sycl::elem::s8, sycl::elem::s9, sycl::elem::sA, sycl::elem::sB,    \
            sycl::elem::sC, sycl::elem::sD, sycl::elem::sE, sycl::elem::sF>(); \
  }

#define DO_SWIZZLE_VEC_ASSIGNMENT_OP(result, v, op)                           \
  if constexpr (v.size() == 1) {                                              \
    result.template swizzle<sycl::elem::s0>() op v;                           \
  } else if constexpr (v.size() == 2) {                                       \
    result.template swizzle<sycl::elem::s0, sycl::elem::s1>() op v;           \
  } else if constexpr (v.size() == 3) {                                       \
    result.template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2>() \
        op v;                                                                 \
  } else if constexpr (v.size() == 4) {                                       \
    result.template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2,   \
                            sycl::elem::s3>() op v;                           \
  } else if constexpr (v.size() == 8) {                                       \
    result.template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2,   \
                            sycl::elem::s3, sycl::elem::s4, sycl::elem::s5,   \
                            sycl::elem::s6, sycl::elem::s7>() op v;           \
  } else if constexpr (v.size() == 16) {                                      \
    result.template swizzle<                                                  \
        sycl::elem::s0, sycl::elem::s1, sycl::elem::s2, sycl::elem::s3,       \
        sycl::elem::s4, sycl::elem::s5, sycl::elem::s6, sycl::elem::s7,       \
        sycl::elem::s8, sycl::elem::s9, sycl::elem::sA, sycl::elem::sB,       \
        sycl::elem::sC, sycl::elem::sD, sycl::elem::sE, sycl::elem::sF>()     \
        op v;                                                                 \
  }

#define DO_SWIZZLE_SCALAR_ASSIGNMENT_OP(result, n, s, op)                     \
  if constexpr (n == 1) {                                                     \
    result.template swizzle<sycl::elem::s0>() op s;                           \
  } else if constexpr (n == 2) {                                              \
    result.template swizzle<sycl::elem::s0, sycl::elem::s1>() op s;           \
  } else if constexpr (n == 3) {                                              \
    result.template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2>() \
        op s;                                                                 \
  } else if constexpr (n == 4) {                                              \
    result.template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2,   \
                            sycl::elem::s3>() op s;                           \
  } else if constexpr (n == 8) {                                              \
    result.template swizzle<sycl::elem::s0, sycl::elem::s1, sycl::elem::s2,   \
                            sycl::elem::s3, sycl::elem::s4, sycl::elem::s5,   \
                            sycl::elem::s6, sycl::elem::s7>() op s;           \
  } else if constexpr (n == 16) {                                             \
    result.template swizzle<                                                  \
        sycl::elem::s0, sycl::elem::s1, sycl::elem::s2, sycl::elem::s3,       \
        sycl::elem::s4, sycl::elem::s5, sycl::elem::s6, sycl::elem::s7,       \
        sycl::elem::s8, sycl::elem::s9, sycl::elem::sA, sycl::elem::sB,       \
        sycl::elem::sC, sycl::elem::sD, sycl::elem::sE, sycl::elem::sF>()     \
        op s;                                                                 \
  }

#define DO_SWIZZLE_UNARY_OP(result, v, op)                                    \
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
