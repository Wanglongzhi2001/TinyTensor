#include "vectorAdd.h"
#include <iostream>
#include <immintrin.h>

void AddImpl(const double* x, const double* y, double* z, size_t N)
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

void AddImpl(const float* x, const float* y, float* z, size_t N)
{
  __m256 vecArrx[N];
  __m256 vecArry[N];
  __m256 vecTemp[N];
  for (int i = 0; i < N; i+=8)
  {
    vecArrx[i / 8] = _mm256_set_ps(x[i + 7], x[i + 6], x[i + 5], x[i + 4], x[i + 3], x[i + 2], x[i + 1], x[i]);
    vecArry[i / 8] =  _mm256_set_ps(y[i + 7], y[i + 6], y[i + 5], y[i + 4], y[i + 3], y[i + 2], y[i + 1], y[i]);
  }

  for (int i = 0; i < N / 8; ++i)
  {
    vecTemp[i] = _mm256_add_ps(vecArrx[i], vecArry[i]);
  }
  for(int i = 0; i < N / 8; ++i)
  {
    z[i * 8] = vecTemp[i][0];
    z[i * 8 + 1] = vecTemp[i][1];
    z[i * 8 + 2] = vecTemp[i][2];
    z[i * 8 + 3] = vecTemp[i][3];
    z[i * 8 + 4] = vecTemp[i][4];
    z[i * 8 + 5] = vecTemp[i][5];
    z[i * 8 + 6] = vecTemp[i][6];
    z[i * 8 + 7] = vecTemp[i][7];
  }
}


std::vector<double> simd::vectorAdd(std::vector<double> x, std::vector<double> y)
{
  std::vector<double> z(x.size());
  AddImpl(x.data(), y.data(), z.data(), x.size());
  return z;
}

std::vector<float> simd::vectorAdd(std::vector<float> x, std::vector<float> y)
{
  std::vector<float> z(x.size());
  AddImpl(x.data(), y.data(), z.data(), x.size());
  return z;
}
