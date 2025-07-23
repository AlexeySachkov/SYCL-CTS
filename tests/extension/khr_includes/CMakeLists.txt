if(SYCL_CTS_ENABLE_KHR_INCLUDES_TESTS)
    file(GLOB test_cases_list *.cpp)

    add_cts_test(${test_cases_list})
endif()
