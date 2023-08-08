#include "../types/batch.h"

namespace vsu
{
    namespace kernel
    {
        template<typename T>
        batch<T> add(const batch<T>& self, const batch<T>& other) noexcept
        {
            if constexpr(sizeof(T) == 1)
            {
                return __m256_add_epi8(self, other);
            }
            else if constexpr(sizeof(T) == 2)
            {
                return __m256_add_epi16(self, other);
            }
            else if constexpr(sizeof(T) == 4)
            {
                return __m256_add_epi32(self, other);
            }
            else if constexpr(sizeof(T) == 8)
            {
                return __m256_add_epi64(self, other);
            }
            else
            {
                throw std::runtime_error("this size of batch have not been implemented!,"+
                    "please ensure the size of your batch in [1, 2, 4, 8]");
            }
        }
    } // namespace kernel
} // namespace vsu
