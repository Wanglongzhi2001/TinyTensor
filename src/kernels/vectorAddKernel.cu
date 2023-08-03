#include "vectorAddKernel.h"
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include <iostream>
#include <vector>

template<typename T>
__global__ void add(const T* x, const T* y, T* z, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i+=stride)
    {
        z[i] = x[i] + y[i];
    }
}

// template<typename T>
std::vector<double> vectorAddGPU(std::vector<double> v_x, std::vector<double> v_y)
{
    int N = v_x.size();
    int nBytes = N * sizeof(double);
    // 申请host内存
    double *x = &v_x[0];
    double *y = &v_y[0];
    double *z;
    z = (double*)malloc(nBytes);

    // 申请device内存
    double *d_x;
    double *d_y;
    double *d_z;
    cudaMalloc(&d_x, nBytes);
    cudaMalloc(&d_y, nBytes);
    cudaMalloc(&d_z, nBytes);

    // 将host数据拷贝到device
    cudaMemcpy(d_x, x, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, nBytes, cudaMemcpyHostToDevice);

    // 定义kernel的执行配置
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    // 执行kernel
    add << <gridSize, blockSize >> > (d_x, d_y, d_z, N);
    // 将device得到的结果拷贝到host
    cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyDeviceToHost);

    // 释放device内存
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    std::vector<double> res(z, z + N);
    return res;
}