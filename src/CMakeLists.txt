add_library(tensor_lib tensor.cpp)
target_include_directories(tensor_lib PUBLIC ../kernels ../kernels/cpu)
add_subdirectory(test)