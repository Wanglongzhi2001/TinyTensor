cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(TinyVSU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# 检查编译器是否支持 AVX2
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX)

# 设置编译选项
if(COMPILER_SUPPORTS_AVX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
else()
  message(WARNING "AVX2 is not supported by the compiler")
endif()

add_subdirectory(src/vector)
add_subdirectory(src/kernels/add)
add_subdirectory(src/simd)
add_subdirectory(third_party/googletest)
add_subdirectory(test)
add_subdirectory(benchmark)
