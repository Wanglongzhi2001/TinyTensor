#include "vectorAddKernel.cuh"

#define REGISTER_CUDA_VECTORADD_CALLER(dtype)                                                        \
    std::vector<dtype> kernel::vectorAddGPUCaller(std::vector<dtype> v_x, std::vector<dtype> v_y){   \
        int N = v_x.size();                                                                          \
        int nBytes = N * sizeof(dtype);                                                              \
        dtype *x = &v_x[0];                                                                          \
        dtype *y = &v_y[0];                                                                          \
        dtype *z;                                                                                    \
        z = (dtype*)malloc(nBytes);                                                                  \
        dtype *d_x;                                                                                  \
        dtype *d_y;                                                                                  \
        dtype *d_z;                                                                                  \
        cudaMalloc(&d_x, nBytes);                                                                    \
        cudaMalloc(&d_y, nBytes);                                                                    \
        cudaMalloc(&d_z, nBytes);                                                                    \
        cudaMemcpy(d_x, x, nBytes, cudaMemcpyHostToDevice);                                          \
        cudaMemcpy(d_y, y, nBytes, cudaMemcpyHostToDevice);                                          \
        dim3 blockSize(256);                                                                         \
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x);                                          \
        kernel::add<dtype> <<<gridSize, blockSize >>> (d_x, d_y, d_z, N);                            \
        cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyDeviceToHost);                            \
        cudaFree(d_x);                                                                               \
        cudaFree(d_y);                                                                               \
        cudaFree(d_z);                                                                               \
        std::vector<dtype> res(z, z + N);                                                            \
        return res;                                                                                  \
    }                                                                                                \

REGISTER_CUDA_VECTORADD_CALLER(int)
REGISTER_CUDA_VECTORADD_CALLER(float)
REGISTER_CUDA_VECTORADD_CALLER(double)