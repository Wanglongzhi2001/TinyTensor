#include <cstdlib>
#include "allocator.h"
#include "registry.h"

namespace TT {
namespace cpu {

class Allocator : public TT::Allocator {
public:
    void* DoAllocate(size_t size) override {
        return aligned_alloc(kAlignSize, size);
    }

    void Deallocate(void* ptr) override { free(ptr); }
};

TT::Allocator& allocator() {
    static Allocator allocator;
    return allocator;
}

KernelRegister allocator_reg("allocator", Device::CPU, allocator);

}   // namespace cpu
}   // namespace TT
