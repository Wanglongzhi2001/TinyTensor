#include "allocator.h"
#include "registry.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace TT {
namespace cuda {

class CudaAsyncAllocator : public TT::Allocator {
public:
    void* DoAllocate(size_t size) override {
        void* ptr;
        cudaMallocAsync(&ptr, size, stream());
        return ptr;
    }

    void Deallocate(void* ptr) override {
        cudaFreeAsync(ptr, stream());
    }

    cudaStream_t stream() {
        static cudaStream_t stream;
        if (stream == nullptr) {
            cudaStreamCreate(&stream_);
        }
        return stream_;
    }
private:
    cudaStream_t stream_;
};
};


}   // namespace cuda
}   // namespace TT
