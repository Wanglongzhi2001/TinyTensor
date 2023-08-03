
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
add_library(vectorAddKernel vectorAddKernel.cu)