#include "../../src/simd/add/vectorAdd.h"
#include <vector>
#include <chrono>
#include <iostream>

using namespace std::chrono;

void addBenchmark()
{
    const int N = 1 << 15;
    const int iter_num = 1000;
    std::vector<float> x(N, 0.0);
    std::vector<float> y(N, 1.0);
    std::vector<float> z(N, 0.0);

    // warm up stage
    for (int i = 0; i < 10 ; i++)
         z = simd::vectorAdd(x, y);

    // benchmark stage
    auto start = steady_clock::now();
    for (int i = 0; i < iter_num; i++)
        z = simd::vectorAdd(x, y);
    auto end = steady_clock::now();
    auto usecs = duration_cast<duration<float, milliseconds::period>>(end - start);
    std::cout << "simd(add) execution time: " << usecs.count() / iter_num << " ms" << std::endl;
    
    start = steady_clock::now();
    for (int i = 0; i < iter_num; i++)
    {
        for (int j = 0;  j < N; j++)
            z[j] = x[j] + y[j];
    }
    end = steady_clock::now();
    usecs = duration_cast<duration<float, milliseconds::period>>(end - start);
    std::cout << "naive(add) execution time: " << usecs.count() / iter_num << " ms" << std::endl;
}

int main()
{
    addBenchmark();
    return 0;
}