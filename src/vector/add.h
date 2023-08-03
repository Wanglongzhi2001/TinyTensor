#pragma once
#include <vector>
#include "../kernels/vectorAddKernel.cuh"
#include "../simd/vectorAdd.h"
#include <iostream>

namespace vsu
{
    enum class speedUpMethod
    {
        SIMD,
        GPU
    };
    
    template<typename T, speedUpMethod method = speedUpMethod::SIMD>
    std::vector<T> vcAdd(std::vector<T> x, std::vector<T> y)
    {
        std::vector<T> z;
        if constexpr(method == speedUpMethod::GPU)
        {
            z = kernel::vectorAdd(x, y);
        }
        else if constexpr (method == speedUpMethod::SIMD)
        {
            z = simd::vectorAdd(x, y);
        }
        return z;
    }
}
