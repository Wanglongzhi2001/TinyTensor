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


namespace TT {
    enum class DType {
        Float32,
        Float16,
        Int32,
        Int16
    };

    enum class Device {
        CPU,
        GPU
    };

    using LengthType = int64_t;
    using Shape = std::vector<LengthType>;

    template <typename T> const DType dtype_v = DType::Float32;
    // TODO: add float16 type
    template <> const DType dtype_v<int> = DType::Int32;
    template <> const DType dtype_v<int16_t> = DType::Int16;

    inline LengthType num_elements(const std::vector<LengthType>& shape) {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<LengthType>());
    }

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

    class TensorStorage {
    public:
        using RawStorage = std::shared_ptr<dt_byte>;

        TensorStorage(size_t nbytes, Device device);
        TensorStorage(void* ptr, Device device);
        TensorStorage(TensorStorage&&) noexcept = default;
        TensorStorage(const TensorStorage& rhs) { *this = rhs;}
        ~TensorStorage() = default;

        TensorStorage& operator=(TensorStorage&&) noexcept = default;
        TensorStorage& operator=(const TensorStorage& rhs);


        void* ptr() const {
            return m_data;
        }

        std::shared_ptr<void*> get_ref_ptr() const {
            ptr();
            return m_ref_ptr;
        }

        size_t size() const { return m_size; }
        size_t offset() const { return m_offset; }
        bool empty() const { return !m_size; }

        const void* raw_storage() const { 
            ptr();
            return m_data; 
        }

        Device device() const { return m_device; }

    private:
        void* m_data;
        size_t m_offset;
        size_t m_size;
        size_t m_capacity;
        size_t nbytes_;
        bool _is_view = false;
        Device m_device;
        std::shared_ptr<void*> m_ref_ptr = std::make_shared<void*>((void*)nullptr);

    };
    

    class Tensor {
    public:

        Tensor() = default;
        Tensor(Tensor&&) noexcept = default;
        Tensor(const Tensor&) = default;

        Tensor(Shape shape, DType dtype = dtype_v<float>, Device device = Device::CPU) {
            size_t nbytes = num_elements(shape) * ::TT::elem_size(dtype);
            m_storage = std::make_shared<TensorStorage>(nbytes, device);
            m_shape = shape;
            m_dtype = dtype;
        }

        Tensor(void* ptr, Shape shape, DType dtype = dtype_v<float>, Device device = Device::CPU) {
            m_storage = std::make_shared<TensorStorage>(ptr, device);
            m_shape = shape;
            m_dtype = dtype;
        }

        Tensor(TensorStorage storage, Shape shape, DType dtype = dtype_v<float>) {
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
        const Shape& size() const { return m_shape; }

        const Shape& shape() const { return m_shape; }
        size_t shape(size_t dim) const {
            TT_CHECK(dim < m_shape.size());
            return m_shape[dim];

        }
        
        // TODO
        // Tensor<DeviceType> operator[](std::initializer_list)


        bool empty() const {return m_storage.get()->empty();}
        LengthType ndim() const { return m_shape.size(); }
        LengthType size(int64_t dim) const { return m_shape[dim]; }
        LengthType numel() const { return num_elements(m_shape); }
        int32_t elem_size() const { return ::TT::elem_size(m_dtype); }

        Tensor& flatten() {
            m_shape = {numel()};
            return *this;
        }

        Tensor &unsqueeze(int dim) {
            TT_ASSERT(dim >= 0 && dim <= ndim(), "invalid dim");
            m_shape.insert(m_shape.begin() + dim, 1);
            return *this;
        }

        static Tensor empty(const Shape& shape, DType dtype, Device device);
        static Tensor from_ptr(void* ptr, const Shape& shape, DType dtype, Device device);

    private:
        std::shared_ptr<TensorStorage> m_storage;
        DType m_dtype;
        Shape m_shape;
        Device m_device;
    };

    void print_tensor(const Tensor& t);
    Tensor Copy(const Tensor &x, Device device, bool always_copy = true);
} // namespace TT