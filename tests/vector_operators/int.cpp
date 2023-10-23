#include "common.h"

// TODO: catch2 include

#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE_SIG("vec::<int, N> operators", "[vec][operators]",
    ((int N), N), 1, 2, 3, 4, 8, 16
    ) {
  
  vector::operators::check_all_operators<int, N>();
}
