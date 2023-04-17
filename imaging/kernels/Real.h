#ifndef CPP_EXAMPLE_IMAGING_KERNELS_REAL_H
#define CPP_EXAMPLE_IMAGING_KERNELS_REAL_H

#include "imaging/Kernel.h"
#include "imaging/Operation.h"
#include "imaging/ops/Real.h"
#include "imaging/KernelRegistry.h"

namespace arrus_example_imaging {

class RealFunctor {
public:
    void operator()(NdArray &output, const NdArray &input, cudaStream_t stream);
};

class RealKernel : public Kernel {
public:
    explicit RealKernel(KernelConstructionContext &ctx);
    void process(KernelExecutionContext &ctx) override;

private:
    RealFunctor impl;
};
}

#endif//CPP_EXAMPLE_IMAGING_KERNELS_REAL_H
