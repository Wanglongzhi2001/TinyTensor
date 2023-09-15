#include "basic_type.h"
#include <fmt/core.h>

namespace TT {

#if __cplusplus >= 201703L || __clang_major__ >= 4
#define TT_FALLTHRU [[fallthrough]];
#elif __GNUC__ >= 7
#define TT_FALLTHRU __attribute__((fallthrough));
#else
#define TT_FALLTHRU
#endif


/* ============ TensorShape ============*/
TensorShape::TensorShape(const std::vector<size_t>& shape) {
    TT_ASSERT(shape.size() <= MAX_NDIM, 
        fmt::format("TensorShape: shape size should be less than {}.", MAX_NDIM));
    ndim = shape.size();
    memcpy(this->shape, shape.data(), sizeof(size_t) * ndim);
}

TensorShape::TensorShape(std::initializer_list<size_t> shape)
    : TensorShape(std::vector<size_t>(shape)) {}

size_t TensorShape::num_elements() const {
    if (!ndim)
        return 0;
    return std::accumulate(shape, shape + ndim, 1, std::multiplies<size_t>());
}

bool TensorShape::eq_shape(const TensorShape& rhs) const {
    if (ndim == rhs.ndim) {
        size_t eq = 0;
        switch (ndim){
            case 7:
                eq += shape[6] == rhs.shape[6];
                TT_FALLTHRU
            case 6:
                eq += shape[5] == rhs.shape[5];
                TT_FALLTHRU
            case 5:
                eq += shape[4] == rhs.shape[4];
                TT_FALLTHRU
            case 4:
                eq += shape[3] == rhs.shape[3];
                TT_FALLTHRU
            case 3:
                eq += shape[2] == rhs.shape[2];
                TT_FALLTHRU
            case 2:
                eq += shape[1] == rhs.shape[1];
                TT_FALLTHRU
            case 1:
                eq += shape[0] == rhs.shape[0];
                break;
        }
        return eq == ndim;
    }
    return false;
}

std::string TensorShape::to_string() const {
    std::string s = "{";
    for (size_t i = 0; i < ndim; ++i) {
        s += std::to_string(shape[i]);
        if (i != ndim - 1) {
            s += ", ";
        }
    }
    s += "}";
    return s;
}

bool TensorShape::is_empty() const {
    for (size_t i = 0; i < ndim; ++i) {
        if (!shape[i]) {
            return true;
        }
    }
    return ndim == 0;
}

} // namespace TT