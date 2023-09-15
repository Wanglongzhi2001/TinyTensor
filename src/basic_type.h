#pragma once

#include "dtype.h"
#include <string>
#include <vector>
#include <numeric>
#include "check.h"

namespace TT {
struct TensorShape {
    static constexpr size_t MAX_NDIM = 7;
    size_t shape[MAX_NDIM], ndim = 0;

    // ctor
    TensorShape() = default;
    TensorShape(const TensorShape& rhs) = default;
    TensorShape(TensorShape&& rhs) = default;
    TensorShape(std::initializer_list<size_t> shape);
    TensorShape(const std::vector<size_t>& shape);
    TensorShape& operator=(const TensorShape& rhs) = default;

    std::string to_string() const;

    size_t num_elements() const;

    bool eq_shape(const TensorShape& rhs) const;

    bool is_scalar() const;

    bool is_empty() const;

    size_t& operator[](size_t i) { return shape[i]; }

    size_t operator[](size_t i) const { return shape[i]; }

};

/**
 * \brief Describing the tensor shape with its actual layout in memory and dtype
 *
 * x(i, j, ...) is stored at offset
 * stride[0]*i + stride[1]*j + ..., in number of elements; physical offset needs
 * to be multiplied by dtype size.
 */
struct TensorLayout : public TensorShape {
    ptrdiff_t stride[MAX_NDIM];
    DType dtype;

    // ctor
    TensorLayout() = default;
    TensorLayout(const TensorLayout& rhs) = default;
    // create empty layout with given dtype
    TensorLayout(DType dtype);
    // create layout from shape and dtype
    TensorLayout(const TensorShape& shape, DType dtype);
    // create layout with specified shape, stride and dtype
    TensorLayout(const TensorShape& shape, const ptrdiff_t* stride, DType dtype);

    TensorLayout& operator=(const TensorLayout& rhs) = default;

    std::string to_string() const;


};
} // namespace TT