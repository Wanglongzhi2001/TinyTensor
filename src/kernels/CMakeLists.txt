
set(CMAKE_CUDA_STANDARD 14)
add_library(vectorAddKernel vectorAddKernel.cu)
set_target_properties(vectorAddKernel PROPERTIES LINKER_LANGUAGE CXX)
