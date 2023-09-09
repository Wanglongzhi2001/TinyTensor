#pragma once
#include "allocator.h"
#include "registry.h"


namespace TT {


inline Allocator& allocator(Device device) {
    return KernelRegistry::Instance().Get<Allocator &(*)()>("allocator", device)();
}
} // namespace TT