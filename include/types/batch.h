#pragma once
#include "./type_register.h"
#include <type_traits>
#include <exception>

namespace tinyvsu
{
    template<typename T>
    class batch : public types::simd_register<T>
    {
    public:
        static constexpr std::size_t size = sizeof(typename types::simd_register<T>::simd_type) / sizeof(T); ///< Number of scalar elements in this batch.

        using value_type = T;
        using simd_type = typename types::simd_register<T>::simd_type; // simd data type, such as __m256i

        inline batch() = default;
        // inline batch(const T&) = default;
        inline batch(T val) noexcept : types::simd_register<T>(broadcast(val)){};
        inline batch(const batch&) = default;
        
        template<class... Ts>
        inline batch(T val0, T val1, Ts... vals) noexcept : batch(set(val0, val1, vals...)) {}

        inline batch(simd_type simd) noexcept : types::simd_register<T>{ simd } {}


        template<typename U>
        inline void store_aligned(U* mem) const noexcept{
            static_assert(std::is_same<U, T>::value || std::is_same<U, bool>::value,
                "store_aligned() requires the memory to be aligned");
            _mm256_store_si256(reinterpret_cast<__m256i*>(mem), this->data);
        }

        /**
         * Copy content of this batch to the buffer \c mem. The
         * memory needs to be aligned.
         */
        template<typename U>
        void store_unaligned(U* mem) const noexcept{
            static_assert(std::is_same<U, T>::value || std::is_same<U, bool>::value,
                "store_unaligned() requires the memory to be aligned");
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(mem), this->data);
        }

        // Update operators
        inline batch& operator=(const batch&) noexcept;
        inline batch operator+=(const batch& other) noexcept{
            return add(*this, other);
        }
        inline batch operator-=(const batch& other) noexcept{
            return sub(*this, other);
        }
        inline batch operator*=(const batch& other) noexcept{
            return mul(*this, other);
        }
        inline batch operator/=(const batch&) noexcept;
        inline batch operator&=(const batch&) noexcept;
        inline batch operator|=(const batch&) noexcept;
        inline batch operator^=(const batch&) noexcept;

        // incr/decr operators
        inline batch operator++() noexcept;
        inline batch operator--() noexcept;
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

        T get(std::size_t i) const noexcept{
            alignas(8) T buffer[batch::size];
            store_aligned(&buffer[0]);
            return buffer[i];
        }

        batch<int> set(T val0, T val1, T val2, T val3) noexcept{
            return _mm256_set_epi64x(val3, val2, val1, val0);
        }

    private:
        // broadcastT
        template <class TT, class = typename std::enable_if<std::is_integral<TT>::value, void>::type>
        batch<TT> broadcast(TT val) noexcept
        {
            if constexpr(sizeof(TT) == 1)
            {
                return _mm256_set1_epi8(val);
            }
            else if constexpr(sizeof(TT) == 2)
            {
                return _mm256_set1_epi16(val);
            }
            else if constexpr(sizeof(TT) == 4)
            {
                return _mm256_set1_epi32(val);
            }
            else if constexpr(sizeof(TT) == 8)
            {
                return _mm256_set1_epi64x(val);
            }
            else
            {
                return {};
            }
        }

        inline batch<float> broadcast(float val) noexcept
        {
            return _mm256_set1_ps(val);
        }

        inline batch<double> broadcast(double val) noexcept
        {
            return _mm256_set1_pd(val);
        }

        // add
        template<typename TT, class = typename std::enable_if<std::is_integral_v<TT>, void>::type>
        batch<TT> add(const batch<TT>& self, const batch<TT>& other) noexcept
        {
            if constexpr(sizeof(TT) == 1)
            {
                return _mm256_add_epi8(self, other);
            }
            else if constexpr(sizeof(TT) == 2)
            {
                return _mm256_add_epi16(self, other);
            }
            else if constexpr(sizeof(TT) == 4)
            {
                return _mm256_add_epi32(self, other);
            }
            else if constexpr(sizeof(TT) == 8)
            {
                return _mm256_add_epi64(self, other);
            }
            else
            {
                throw std::exception("this size of batch have not been implemented!,"
                    "please ensure the size of your batch in [1, 2, 4, 8]");
            }
        }

        inline batch<float> add(batch<float> const& self, batch<float> const& other) noexcept
        {
            return _mm256_add_ps(self, other);
        }
      
        inline batch<double> add(batch<double> const& self, batch<double> const& other) noexcept
        {
            return _mm256_add_pd(self, other);
        }

        // eq
        template <class TT, class = typename std::enable_if<std::is_integral_v<TT>, void>::type>
        inline batch<TT> eq(batch<TT> const& self, batch<TT> const& other) noexcept
        {
            if constexpr(sizeof(TT) == 1)
            {
                return _mm256_cmpeq_epi8(self, other);
            }
            else if constexpr(sizeof(TT) == 2)
            {
                return _mm256_cmpeq_epi16(self, other);
            }
            else if constexpr(sizeof(TT) == 4)
            {
                return _mm256_cmpeq_epi32(self, other);
            }
            else if constexpr(sizeof(TT) == 8)
            {
                return _mm256_cmpeq_epi64(self, other);
            }
            else
            {
                throw std::exception("this size of batch have not been implemented!,"
                    "please ensure the size of your batch in [1, 2, 4, 8]");
            }
        }

        // lt
        template <class TT, class = typename std::enable_if<std::is_integral<TT>::value, void>::type>
        batch<TT> lt(batch<TT> const& self, batch<TT> const& other) noexcept
        {
            if (std::is_signed<TT>::value)
            {
                if constexpr(sizeof(TT) == 1)
                {
                    return _mm256_cmpgt_epi8(other, self);
                }
                else if constexpr(sizeof(TT) == 2)
                {
                    return _mm256_cmpgt_epi16(other, self);
                }
                else if constexpr(sizeof(TT) == 4)
                {
                    return _mm256_cmpgt_epi32(other, self);
                }
                else if constexpr(sizeof(TT) == 8)
                {
                    return _mm256_cmpgt_epi64(other, self);
                }
                else
                {
                    throw std::exception("this size of batch have not been implemented!,"
                        "please ensure the size of your batch in [1, 2, 4, 8]");
                }
            }
            else
            {
                throw std::exception("this size of batch have not been implemented!,"
                    "please ensure the size of your batch in [1, 2, 4, 8]");
            }
        }

        // mul
        template <class TT, class = typename std::enable_if<std::is_integral<TT>::value, void>::type>
        batch<TT> mul(batch<TT> const& self, batch<TT> const& other) noexcept
        {
            if constexpr(sizeof(TT) == 1)
            {
                __m256i mask_hi = _mm256_set1_epi32(0xFF00FF00);
                __m256i res_lo = _mm256_mullo_epi16(self, other);
                __m256i other_hi = _mm256_srli_epi16(other, 8);
                __m256i self_hi = _mm256_and_si256(self, mask_hi);
                __m256i res_hi = _mm256_mullo_epi16(self_hi, other_hi);
                __m256i res = _mm256_blendv_epi8(res_lo, res_hi, mask_hi);
                return res;
            }
            else if constexpr(sizeof(TT) == 2)
            {
                return _mm256_mullo_epi16(self, other);
            }
            else if constexpr(sizeof(TT) == 4)
            {
                return _mm256_mullo_epi32(self, other);
            }
            else
            {
                throw std::exception("this size of batch have not been implemented!,"
                    "please ensure the size of your batch in [1, 2, 4, 8]");
            }
        }

        // ssub
        template <class TT, class = typename std::enable_if<std::is_integral<TT>::value, void>::type>
        batch<TT> ssub(batch<TT> const& self, batch<TT> const& other) noexcept
        {
            if (std::is_signed<T>::value)
            {
                if constexpr(sizeof(TT) == 1)
                {
                    return _mm256_subs_epi8(self, other);
                }
                else if constexpr(sizeof(TT) == 2)
                {
                    return _mm256_subs_epi16(self, other);
                }
                else
                {
                    throw std::exception("this size of batch have not been implemented!,"
                        "please ensure the size of your batch in [1, 2, 4, 8]");
                }
            }
            else
            {
                if constexpr(sizeof(TT) == 1)
                {
                    return _mm256_subs_epu8(self, other);
                }
                else if constexpr(sizeof(TT) == 2)
                {
                    return _mm256_subs_epu16(self, other);
                }
                else
                {
                    throw std::exception("this size of batch have not been implemented!,"
                        "please ensure the size of your batch in [1, 2, 4, 8]");
                }
            }
        }

        // sub
        template <class TT, class = typename std::enable_if<std::is_integral<TT>::value, void>::type>
        batch<TT> sub(batch<TT> const& self, batch<TT> const& other) noexcept
        {
            if constexpr(sizeof(TT) == 1)
            {
                return _mm256_sub_epi8(self, other);
            }
            else if constexpr(sizeof(TT) == 2)
            {
                return _mm256_sub_epi16(self, other);
            }
            else if constexpr(sizeof(TT) == 4)
            {
                return _mm256_sub_epi32(self, other);
            }
            else if constexpr(sizeof(TT) == 8)
            {
                return _mm256_sub_epi64(self, other);
            }
            else
            {
                throw std::exception("this size of batch have not been implemented!,"
                    "please ensure the size of your batch in [1, 2, 4, 8]");
            }
        }
    };
} // namespace tinyvsu
