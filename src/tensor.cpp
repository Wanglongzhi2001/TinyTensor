#include "tensor.h"
#include "check.h"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include "kernels.h"

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

Tensor Tensor::empty(const Shape &shape, DType dtype, Device device) {
  auto storage = std::make_shared<TensorStorage>(
      num_elements(shape) * ::TT::elem_size(dtype), device);
  Tensor tensor;
  tensor.m_storage = storage;
  tensor.m_shape = shape;
  tensor.m_dtype = dtype;
//   tensor.name = "tensor_" + std::to_string(unique_id());
  return tensor;
}

Tensor Tensor::from_ptr(void *dptr, const Shape &shape, DType dtype,
                       Device device) {
  auto storage = std::make_shared<TensorStorage>(dptr, device);
  Tensor tensor;
  tensor.m_storage = storage;
  tensor.m_shape = shape;
  tensor.m_dtype = dtype;
//   tensor.name = "tensor_" + std::to_string(unique_id());
  return tensor;
}


TensorStorage::TensorStorage(size_t nbytes, Device device) {
  m_data = allocator(device).Allocate(nbytes);
  m_device = device;
  _is_view = false;
}


TensorStorage::TensorStorage(void *external_ptr, Device device) {
  m_data = external_ptr;
  m_device = device;
  _is_view = true;
}

// TensorStorage::~TensorStorage() {
//   if (!_is_view) {
//     allocator(m_device).Deallocate(m_data);
//   }
// }


} // namespace TT

namespace {
    int uid() {
        static int _uid = 0;
        return _uid++;
    }
} // namespace

