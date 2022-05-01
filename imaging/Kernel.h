#ifndef CPP_EXAMPLE_KERNELS_KERNEL_CUH
#define CPP_EXAMPLE_KERNELS_KERNEL_CUH

#include "KernelConstructionContext.h"
#include "KernelExecutionContext.h"

#include <memory>
namespace imaging {
class Kernel {
public:
    typedef std::unique_ptr<Kernel> Handle;

    explict Kernel(KernelConstructionContext &ctx) {}

    virtual void process(KernelExecutionContext &ctx) = 0;
};
}// namespace imaging
#endif//CPP_EXAMPLE_KERNELS_KERNEL_CUH
