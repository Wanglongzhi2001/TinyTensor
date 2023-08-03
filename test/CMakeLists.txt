add_executable(test test_vectorAdd.cpp)
target_link_libraries(test vectorAdd vectorAddKernel add gtest)