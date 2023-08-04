#pragma once
#include <vector>
#include <iostream>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

namespace kernel
{
    template<typename T>
    __global__ void add(const T* x, const T* y, T* z, size_t n)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = index; i < n; i+=stride)
        {
            z[i] = x[i] + y[i];
        }
    }

    #define REGISTER_CUDA_VECTORADDCALLER_HEAD(dtype)   \
        std::vector<dtype> vectorAddGPUCaller(std::vector<dtype> v_x, std::vector<dtype> v_y);

    REGISTER_CUDA_VECTORADDCALLER_HEAD(int)
    REGISTER_CUDA_VECTORADDCALLER_HEAD(float)
    REGISTER_CUDA_VECTORADDCALLER_HEAD(double)

    template<typename T>
    std::vector<T> vectorAdd(std::vector<T> v_x, std::vector<T> v_y)
    {
        return vectorAddGPUCaller(v_x, v_y);
    }
}


