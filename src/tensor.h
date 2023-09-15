#pragma once

#include <numeric>
#include <cassert>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>
#ifdef TT_ENABLE_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif
#include "check.h"
#include "dtype.h"
#include "basic_type.h"

namespace TT {


enum class Device {
    CPU,
    GPU
};

template <typename T> const DType dtype_v = DType::Float32;
// TODO: add float16 type
template <> const DType dtype_v<int> = DType::Int32;
template <> const DType dtype_v<int16_t> = DType::Int16;

inline int32_t elem_size(DType dtype) {
    switch (dtype) {
        case DType::Float32:
            return 4;
        case DType::Float16:
            return 2;
        case DType::Int32:
            return 4;
        case DType::Int16:
            return 2;
        default:
            TT_UNIMPLEMENTED();
    }
}

/// @brief specify how a subtensor reside in a larger one
class SubTensorSpec {

};


/*!
 * \brief slice along some axis; index as in Python, with negative indices
 *      supported. Scalar index can also be represented as a Slice, where
 *      m_begin = idx, m_end = idx+1 and m_step = 1. The flag m_is_scalar_idx
 *      indicates whether the Slice comes from a scalar index.
 */
class Slice {
    ptrdiff_t m_begin, m_end, m_step;
    bool m_is_scalar_idx;
public:
    Slice(ptrdiff_t begin, ptrdiff_t end, ptrdiff_t step = 1) noexcept
        : m_begin(begin), m_end(end), m_step(step), m_is_scalar_idx(false) {}

    /*!
     * \brief apply this slice on given tensor layout, and get corresponding
     *      subtensor
     * \param axis the axis to apply this slice; -1 can be used for
     *      flattened layout
     */
    SubTensorSpec apply(TensorLayout layout, int axis) const;
};


class TensorStorage {
public:
    using RawStorage = std::shared_ptr<dt_byte>;

    TensorStorage() = default;
    TensorStorage(size_t size, Device device);
    TensorStorage(void* ptr, Device device);
    TensorStorage(TensorStorage&&) noexcept = default;
    TensorStorage(const TensorStorage& rhs) { *this = rhs;}
    ~TensorStorage() = default;

    TensorStorage& operator=(TensorStorage&&) noexcept = default;
    TensorStorage& operator=(const TensorStorage& rhs);

    /*!
        * \brief apply lazy resize and get ptr
        * \return pointer to the data
        */
    dt_byte* ptr() const {
        return const_cast<TensorStorage*>(this)->apply_lazy_and_get_ptr();
    }

    std::shared_ptr<void*> get_ref_ptr() const {
        ptr();
        return m_ref_ptr;
    }

    TensorStorage sub(ptrdiff_t offset) const;

    size_t size() const { return m_size; }

    size_t offset() const { return m_offset; }

    bool empty() const { return !m_size; }

    const RawStorage raw_storage() const { 
        ptr();
        return m_data; 
    }

    Device device() const { return m_device; }

private:
    RawStorage m_data;
    size_t m_offset;
    size_t m_size; // size in bytes
    size_t m_capacity;
    Device m_device;
    std::shared_ptr<void*> m_ref_ptr = std::make_shared<void*>((void*)nullptr);
    //! used internally for returning a predefined TensorStorage
    TensorStorage(
        size_t size, size_t capacity, size_t offset, const RawStorage& data,Device device)
        : m_data(data),
        m_size(size),
        m_capacity(capacity),
        m_offset(offset),
        m_device(device) {}   
    
    dt_byte* apply_lazy_and_get_ptr();
};


class Tensor {
public:

    Tensor() = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor(const Tensor&) = default;

    Tensor(TensorShape shape, DType dtype = dtype_v<float>, Device device = Device::CPU) {
        size_t nbytes = shape.num_elements() * ::TT::elem_size(dtype);
        m_storage = std::make_shared<TensorStorage>(nbytes, device);
        m_shape = shape;
        m_dtype = dtype;
    }

    Tensor(void* ptr, TensorShape shape, DType dtype = dtype_v<float>, Device device = Device::CPU) {
        m_storage = std::make_shared<TensorStorage>(ptr, device);
        m_shape = shape;
        m_dtype = dtype;
    }

    Tensor(TensorStorage storage, TensorShape shape, DType dtype = dtype_v<float>) {
        m_storage = std::make_shared<TensorStorage>(std::move(storage));
        m_shape = shape;
        m_dtype = dtype;
    }

    template <typename T = void>
    const T* data_ptr() const {
        TT_CHECK(std::is_same_v<T, void> || m_dtype == dtype_v<T>);
        return static_cast<const T*>(m_storage->ptr());
    }

    template <typename T = void>
    T* data_ptr() {
        TT_CHECK(std::is_same_v<T, void> || m_dtype == dtype_v<T>);
        return static_cast<T*>(m_storage->ptr());
    }

    DType dtype() const { return m_dtype; }
    Device device() const { return m_device; }
    const TensorShape& size() const { return m_shape; }

    const TensorShape& shape() const { return m_shape; }
    size_t shape(size_t dim) const {
        TT_CHECK(dim < m_shape.ndim);
        return m_shape[dim];

    }
    
    Tensor operator[](std::initializer_list<Slice> slice) const;

    bool empty() const {return m_storage.get()->empty();}
    size_t ndim() const { return m_shape.ndim; }
    size_t size(int64_t dim) const { return m_shape[dim]; }
    size_t numel() const { return m_shape.num_elements(); }
    int32_t elem_size() const { return ::TT::elem_size(m_dtype); }

    Tensor& flatten() {
        m_shape = {numel()};
        return *this;
    }

    // Tensor &unsqueeze(int dim) {
    //     TT_ASSERT(dim >= 0 && dim <= ndim(), "invalid dim");
    //     m_shape.insert(m_shape.begin() + dim, 1);
    //     return *this;
    // }

    static Tensor empty(const TensorShape& shape, DType dtype, Device device);
    static Tensor from_ptr(void* ptr, const TensorShape& shape, DType dtype, Device device);

private:
    std::shared_ptr<TensorStorage> m_storage;
    DType m_dtype;
    TensorShape m_shape;
    Device m_device;
};

void print_tensor(const Tensor& t);
Tensor Copy(const Tensor &x, Device device, bool always_copy = true);
} // namespace TT