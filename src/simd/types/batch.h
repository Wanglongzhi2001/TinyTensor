#pragma once
#include "./type_register.h"

namespace vsu
{
    template<typename T>
    class batch : public types::simd_register<T>
    {
    public:
        using value_type = T;
        using simd_type = typename types::simd_register<T>::simd_type; // simd data type, such as __m256i

        inline batch() = default;
        // inline batch(const T&) = default;
        inline batch(T val) noexcept;
        inline batch(const batch&) = default;
        template<class... Ts>
        inline batch(T val0, T val1, Ts... vals) noexcept;
        inline batch(simd_type simd) noexcept : types::simd_register<T>{ simd } {};


        template<typename U>
        inline void store_aligned(U* mem) const noexcept;
        template<typename U>
        inline void store_unaligned(U* mem) const noexcept;

        // Update operators
        inline batch& operator=(const batch&) noexcept;
        inline batch& operator+=(const batch&) noexcept;
        inline batch& operator-=(const batch&) noexcept;
        inline batch& operator*=(const batch&) noexcept;
        inline batch& operator/=(const batch&) noexcept;
        inline batch& operator&=(const batch&) noexcept;
        inline batch& operator|=(const batch&) noexcept;
        inline batch& operator^=(const batch&) noexcept;

        // incr/decr operators
        inline batch& operator++() noexcept;
        inline batch& operator--() noexcept;
        inline batch operator++(int) noexcept;
        inline batch operator--(int) noexcept;

            /** Shorthand for xsimd::add() */
        friend inline batch operator+(batch const& self, batch const& other) noexcept
        {
            return batch(self) += other;
        }

        /** Shorthand for xsimd::sub() */
        friend inline batch operator-(batch const& self, batch const& other) noexcept
        {
            return batch(self) -= other;
        }

        /** Shorthand for xsimd::mul() */
        friend inline batch operator*(batch const& self, batch const& other) noexcept
        {
            return batch(self) *= other;
        }

        /** Shorthand for xsimd::div() */
        friend inline batch operator/(batch const& self, batch const& other) noexcept
        {
            return batch(self) /= other;
        }

        /** Shorthand for xsimd::bitwise_and() */
        friend inline batch operator&(batch const& self, batch const& other) noexcept
        {
            return batch(self) &= other;
        }

        /** Shorthand for xsimd::bitwise_or() */
        friend inline batch operator|(batch const& self, batch const& other) noexcept
        {
            return batch(self) |= other;
        }

        /** Shorthand for xsimd::bitwise_xor() */
        friend inline batch operator^(batch const& self, batch const& other) noexcept
        {
            return batch(self) ^= other;
        }

        /** Shorthand for xsimd::logical_and() */
        friend inline batch operator&&(batch const& self, batch const& other) noexcept
        {
            return batch(self).logical_and(other);
        }

        /** Shorthand for xsimd::logical_or() */
        friend inline batch operator||(batch const& self, batch const& other) noexcept
        {
            return batch(self).logical_or(other);
        }
    };
} // namespace vsu
