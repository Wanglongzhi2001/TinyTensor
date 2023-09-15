add_executable(test_tensor 
    test_tensor.cpp 
    ${PROJECT_SOURCE_DIR}/src/tensor.cpp
    ${PROJECT_SOURCE_DIR}/kernels/cpu/allocator.cpp)
target_link_libraries(test_tensor tensor_lib gtest fmt::fmt)
target_include_directories(test_tensor PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/third_party/googletest/include)
