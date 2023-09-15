add_library(tensor_lib tensor.cpp basic_type.cpp)
target_include_directories(tensor_lib PUBLIC ../kernels ../kernels/cpu ${PROJECT_SOURCE_DIR}/third_party/fmt/include)
add_subdirectory(test)