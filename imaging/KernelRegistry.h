#ifndef CPP_EXAMPLE_IMAGING_KERNELREGISTRY_H
#define CPP_EXAMPLE_IMAGING_KERNELREGISTRY_H

#include <functional>

#include "imaging/Kernel.h"
#include "imaging/Operation.h"

namespace imaging {
// TODO implement
class KernelRegistry {
public:
    using KernelFactory = std::function<Kernel::Handle (KernelConstructionContext &)>;
    static KernelRegistry getInstance() {
        // TODO implement singleton
        return KernelRegistry{};
    }

    Kernel::Handle createKernel(const Operation &op, KernelConstructionContext& ctx) {
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

}

#endif//CPP_EXAMPLE_IMAGING_KERNELREGISTRY_H
