#include <chrono>
#include <iostream>
#include "../../src/kernels/add/vectorAddKernel.cuh"
#include "cublas_v2.h"

using namespace std::chrono;
using namespace kernel;

// refer to https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9956-best-practices-when-benchmarking-cuda-applications_V2.pdf
// to set proper environment to run benchmark
#define REGISTER_KERNEL_BENCHMARK(kernel) \
    void kernel##BenchMark() \
    {   \
        int N = 1 << 20;    \
        int iter_num = 100; \
        int nBytes = N * sizeof(float); \
        float* x_device;    \
        cudaMalloc(&x_device, nBytes);  \
        cudaMemset(&x_device, 0, nBytes);    \
        float* y_device;    \
        cudaMalloc(&y_device, nBytes);  \
        cudaMemset(&y_device, 0, nBytes);    \
        float* z_device;    \
        cudaMalloc(&z_device, nBytes);  \
        cudaMemset(&z_device, 0, nBytes);    \
        dim3 blockSize(256);    \
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x); \
        for (size_t i = 0; i < 10; i++) \
        {   \
            kernel<<<gridSize, blockSize>>>(x_device, y_device, z_device, N);   \
        }   \
        auto start = steady_clock::now();   \
        for (size_t i = 0; i < iter_num; i++)   \
        {   \
            kernel<<<gridSize, blockSize>>>(x_device, y_device, z_device, N);   \
        }   \
        auto end = steady_clock::now(); \
        auto usecs = duration_cast<duration<float, milliseconds::period>>(end - start); \
        std::cout << "kernel execution time: " << usecs.count() / iter_num << " ms" << std::endl;   \
        cublasHandle_t handle = 0;  \
        float alpha = 1.0f; \
        cublasCreate(&handle);  \
        for (size_t i = 0; i < 10; i++) \
        {   \
            cublasSaxpy_v2(handle, N, &alpha, x_device, 1, y_device, 1); \
        }   \
        start = steady_clock::now();   \
        for (size_t i = 0; i < iter_num; i++)   \
        {   \
            cublasSaxpy_v2(handle, N, &alpha, x_device, 1, y_device, 1); \
        }   \
        end = steady_clock::now(); \
        usecs = duration_cast<duration<float, milliseconds::period>>(end - start); \
        std::cout << "cublas execution time: " << usecs.count() / iter_num << " ms" << std::endl;   \
        cudaFree(x_device); \
        cudaFree(y_device); \
        cudaFree(z_device); \
        cublasDestroy(handle); \
    }   \

REGISTER_KERNEL_BENCHMARK(add);

int main()
{
    addBenchMark();
    return 0;
}