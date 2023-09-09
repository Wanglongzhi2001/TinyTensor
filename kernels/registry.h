#ifndef _KERNELS_REGISTRY_H_
#define _KERNELS_REGISTRY_H_

#include <any>
#include <map>
#include <string>
#include <stdexcept>
#include <utility>
#include "../src/tensor.h"

namespace TT {

// Singleton KernelRegistry
class KernelRegistry {
public:
    static KernelRegistry& Instance() {
        static KernelRegistry registry;
        return registry;
    }

    void Register(const std::string& name, Device device, std::any kernel, int priority) {
        kernels_[std::make_pair(name, device)] = kernel;
    }

    template <typename T>
    T Get(const std::string& name, Device device) {
        if (kernels_.find(std::make_pair(name, device)) == kernels_.end()) {
            throw std::runtime_error("kernel " + name + " not found");
        }
        return std::any_cast<T>(kernels_[std::make_pair(name, device)]);
    }

    KernelRegistry(const KernelRegistry&) = delete;
    KernelRegistry& operator=(const KernelRegistry&) = delete;
private:
    std::map<std::pair<std::string, Device>, std::any> kernels_;
    KernelRegistry() = default;
};

struct KernelRegister {
    KernelRegister(const std::string& name, Device device, std::any kernel, int priority = 1) {
        KernelRegistry::Instance().Register(name, device, kernel, priority);
    }
};

} // namespace TT

#endif
