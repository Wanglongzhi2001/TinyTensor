#pragma once

namespace types
{
    template<typename T>
    struct simd_register
    {
        struct register_type
        {
        };
    };

    #define DECLARE_SIMD_REGISTER(SCALAR_TYPE, VECTOR_TYPE) \
        template<>                                          \
        struct simd_register<SCALAR_TYPE>                   \
        {                                                   \
            using simd_type = VECTOR_TYPE;                  \
            simd_type data;                                 \
            inline operator simd_type() const noexcept      \
            {                                               \
                return data;                                \
            }                                               \
        };

    #include <immintrin.h>
    DECLARE_SIMD_REGISTER(bool, __m256i);
    DECLARE_SIMD_REGISTER(signed char, __m256i);
    DECLARE_SIMD_REGISTER(unsigned char, __m256i);
    DECLARE_SIMD_REGISTER(char, __m256i);
    DECLARE_SIMD_REGISTER(unsigned short, __m256i);
    DECLARE_SIMD_REGISTER(short, __m256i);
    DECLARE_SIMD_REGISTER(unsigned int, __m256i);
    DECLARE_SIMD_REGISTER(int, __m256i);
    DECLARE_SIMD_REGISTER(unsigned long int, __m256i);
    DECLARE_SIMD_REGISTER(long int, __m256i);
    DECLARE_SIMD_REGISTER(unsigned long long int, __m256i);
    DECLARE_SIMD_REGISTER(long long int, __m256i);
    DECLARE_SIMD_REGISTER(float, __m256);
    DECLARE_SIMD_REGISTER(double, __m256d);
}
