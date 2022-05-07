#ifndef CPP_EXAMPLE_IMAGING_KERNELREGISTRY_H
#define CPP_EXAMPLE_IMAGING_KERNELREGISTRY_H

#include <functional>

#include "imaging/Kernel.h"
#include "imaging/Operation.h"

namespace imaging {
class KernelRegistry {
public:
    using KernelFactory = std::function<Kernel::Handle(KernelConstructionContext &)>;

    static KernelRegistry getInstance() {
        static KernelRegistry instance;
        return instance;
    }

    Kernel::Handle createKernel(const Operation &op, KernelConstructionContext &ctx) {
        return kernels.at(op.getOpClassId())(ctx);
    }

    void registerKernelOpFactory(const Operation::OpClassId &classId, KernelFactory factory) {
        // TODO wrap it as a macro, that can be called in .cpp
        kernels.insert({classId, factory});
    }

private:
    // Factory functions.
    std::unordered_map<Operation::OpClassId, KernelFactory> kernels;
};

template<typename T> class RegisterKernelOpInitializer {
public:
    explicit RegisterKernelOpInitializer(const Operation::OpClassId id) {
        KernelRegistry::getInstance().registerKernelOpFactory(
            id, [](KernelConstructionContext &ctx) { return std::make_unique<T>(ctx); });
    }
};

#define REGISTER_KERNEL_OP(opClassId, KernelClass) \
namespace {                                        \
    static RegisterKernelOpInitializer<KernelClass> opInitializer{opClassId}; \
}

}// namespace imaging

#endif//CPP_EXAMPLE_IMAGING_KERNELREGISTRY_H
