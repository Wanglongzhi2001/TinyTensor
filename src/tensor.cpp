#include "tensor.h"
#include "check.h"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include "kernels.h"
#include <fmt/core.h>

namespace TT {

void print_tensor(const Tensor& t) {
    auto t_cpu = Copy(t, Device::CPU);
    std::cout << "Tensor(";
    for (int i = 0, n = std::min<int>(t_cpu.numel(), 20); i < n; ++i) {
        if (i == n - 1)
          std::cout << t.shape()[i];
        else
          std::cout << t.shape()[i] << ", ";
    }
    std::cout << ") ";
    int dtype_num = static_cast<int>(t_cpu.dtype());
    switch (dtype_num)
    {
    case 0:
      std::cout << "dtype: float32" << std::endl;
      break;
    case 1:
      std::cout << "dtype: float16" << std::endl;
      break;
    case 2:
      std::cout << "dtype: int32" << std::endl;
      break;
    case 3:
      std::cout << "dtype: int16" << std::endl;
      break;
    default:
      break;
    }
}

Tensor Copy(const Tensor &x, Device device, bool always_copy) {
    if (x.device() == device && !always_copy) {
      return x;
    }
    Tensor y = Tensor::empty(x.size(), x.dtype(), device);
    // TODO: registry
#ifdef TT_ENABLE_CUDA
    if (device == Device::CPU && x.device() == Device::CUDA) {
      cudaMemcpy(y.data_ptr(), x.data_ptr(), x.numel() * x.elem_size(),
                cudaMemcpyDeviceToHost);
      return y;
    }
    if (device == Device::CUDA && x.device() == Device::CPU) {
      cudaMemcpy(y.data_ptr(), x.data_ptr(), x.numel() * x.elem_size(),
                cudaMemcpyHostToDevice);
      return y;
    }
#endif

    if (device == Device::CPU && x.device() == Device::CPU) {
      std::memcpy(y.data_ptr(), x.data_ptr(), x.numel() * x.elem_size());
      return y;
    }
    throw std::runtime_error("unsupported device");
}

const int INVALID_AXIS = 7;
/* ================== Slice and SubTensorSpec =============*/

SubTensorSpec Slice::apply(TensorLayout layout, int axis) const {
    TT_ASSERT(layout.ndim > 0, "");
    if (axis == INVALID_AXIS) {
        axis = 0;
        TT_ASSERT(layout.ndim == 1, "apply Slice with axis==INVALID_AXIS on non-contig layout");
    }

    // axis in [-ndim, ndim) is available
    if (axis < 0)
        axis += layout.ndim;
    TT_ASSERT(axis >= 0 && static_cast<size_t>(axis) < layout.ndim, 
        fmt::format("invalid axis: {}; ndim={}", axis, layout.ndim));
}

/* ================== TensorStorage ==============*/
TensorStorage::TensorStorage(size_t size, Device device) {
    auto ptr = static_cast<dt_byte*>(allocator(device).Allocate(size));
    m_data = std::shared_ptr<dt_byte>(ptr);
    m_size = size;
    m_device = device;
}

TensorStorage::TensorStorage(void *external_ptr, Device device) {
    m_data = std::shared_ptr<dt_byte>(static_cast<dt_byte*>(external_ptr));
    m_device = device;
}

TensorStorage& TensorStorage::operator=(const TensorStorage& rhs) {
    if (rhs.m_size > rhs.m_capacity) {
        rhs.ptr();
    }
    m_data = rhs.m_data;
    m_size = rhs.m_size;
    m_capacity = rhs.m_capacity;
    m_offset = rhs.m_offset;
    m_device = rhs.m_device;
    m_ref_ptr = rhs.m_ref_ptr;
    return *this;
}

TensorStorage TensorStorage::sub(ptrdiff_t offset) const {
    ptr(); // apply lazy resize
    ptrdiff_t toff = offset + m_offset;
    if (offset == static_cast<ptrdiff_t>(m_size)) {
      return {0, 0, 0, RawStorage{}, m_device};
    }
    TT_ASSERT(toff >= 0 && toff < static_cast<ptrdiff_t>(m_size),
              "sub out of range");
    return {m_size - offset, m_capacity - offset, toff, m_data, m_device};
}

dt_byte* TensorStorage::apply_lazy_and_get_ptr() {
    if (m_size > m_capacity) {
      m_data.reset(); // free old ptr
      m_capacity = 0;  // to be exception safe
      auto ptr = static_cast<dt_byte*>(allocator(m_device).Allocate(m_size));
      TT_ASSERT(ptr, "failed to allocate memory");
      m_capacity = m_size;
      auto device = m_device;
      m_data.reset(ptr, [device](void* p){allocator(device).Deallocate(p);});
      m_ref_ptr = std::make_shared<void*>(static_cast<void*>(nullptr));
      m_capacity = m_size;
    }
    *m_ref_ptr = static_cast<void*>(m_data.get());
    return m_data.get() + m_offset;
}


/* ================== Tensor =====================*/
Tensor Tensor::empty(const TensorShape &shape, DType dtype, Device device) {
    auto storage = std::make_shared<TensorStorage>(
        shape.num_elements() * ::TT::elem_size(dtype), device);
    Tensor tensor;
    tensor.m_storage = storage;
    tensor.m_shape = shape;
    tensor.m_dtype = dtype;
    //   tensor.name = "tensor_" + std::to_string(unique_id());
    return tensor;
}

Tensor Tensor::from_ptr(void *dptr, const TensorShape &shape, DType dtype,
                       Device device) {
    auto storage = std::make_shared<TensorStorage>(dptr, device);
    Tensor tensor;
    tensor.m_storage = storage;
    tensor.m_shape = shape;
    tensor.m_dtype = dtype;
  //   tensor.name = "tensor_" + std::to_string(unique_id());
    return tensor;
}

Tensor Tensor::operator[](std::initializer_list<Slice> slice) const {

}



} // namespace TT

namespace {
    int uid() {
        static int _uid = 0;
        return _uid++;
    }
} // namespace

