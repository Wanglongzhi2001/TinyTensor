#pragma once
#include <vector>
#include "../kernels/vectorAddKernel.h"
#include "../simd/vectorAdd.h"
#include <iostream>

namespace vc
{
    template<typename T, bool useGPU = false>
    std::vector<T> vcAdd(std::vector<T> x, std::vector<T> y)
    {
        std::vector<T> z;
        // z.reserve(x.size());
        if (useGPU)
        {
            std::cout << "开始使用GPU加速计算" << std::endl;
            z = vectorAddGPU(x, y);
        }
        else
        {
            std::cout << "开始使用SIMD指令加速计算" << std::endl;
            z = vectorAdd(x, y);
        }
        return z;
    }
}
