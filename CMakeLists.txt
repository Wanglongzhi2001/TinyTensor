cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(TinyVSU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TT_ENABLE_CUDA "Enable CUDA" OFF)

if (TT_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDA  REQUIRED)
  include_directories("${CUDA_INCLUDE_DIRS}")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
endif()


# # 检查编译器是否支持 AVX2
# include(CheckCXXCompilerFlag)
# CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX)

# # 设置编译选项
# if(COMPILER_SUPPORTS_AVX)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
# else()
#   message(WARNING "AVX2 is not supported by the compiler")
# endif()

add_subdirectory(src)
add_subdirectory(third_party)