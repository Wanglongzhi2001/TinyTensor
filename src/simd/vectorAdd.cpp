#include "vectorAdd.h"
#include <iostream>
#include <immintrin.h>

void AddImpl(const double* x, const double* y, double* z, int N)
{
  __m256d vecArrx[N];
  __m256d vecArry[N];
  __m256d vecTemp[N];
  for (int i = 0; i < N; i+=4)
  {
    vecArrx[i / 4] = _mm256_set_pd(x[i + 3], x[i + 2], x[i + 1], x[i]);
    vecArry[i / 4] = _mm256_set_pd(y[i + 3], y[i + 2], y[i + 1], y[i]);
  }

  for (int i = 0; i < N / 4; ++i)
  {
    vecTemp[i] = _mm256_add_pd(vecArrx[i], vecArry[i]);
  }
  for(int i = 0; i < N / 4; ++i)
  {
    z[i * 4] = vecTemp[i][0];
    z[i * 4 + 1] = vecTemp[i][1];
    z[i * 4 + 2] = vecTemp[i][2];
    z[i * 4 + 3] = vecTemp[i][3];
  }
}

std::vector<double> vectorAdd(std::vector<double> x, std::vector<double> y)
{
  std::vector<double> z(x.size());
  AddImpl(x.data(), y.data(), z.data(), x.size());
  return z;
}