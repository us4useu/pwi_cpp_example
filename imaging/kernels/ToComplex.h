#ifndef CPP_EXAMPLE_IMAGING_KERNELS_TO_COMPLEX_H
#define CPP_EXAMPLE_IMAGING_KERNELS_TO_COMPLEX_H

#include "imaging/Kernel.h"
#include "imaging/Operation.h"
#include "imaging/ops/ToComplex.h"
#include "imaging/KernelRegistry.h"

namespace arrus_example_imaging {

class ToComplexFunctor {
public:
    void operator()(NdArray &output, const NdArray &input, cudaStream_t stream);
};

class ToComplexKernel : public Kernel {
public:
    explicit ToComplexKernel(KernelConstructionContext &ctx);
    void process(KernelExecutionContext &ctx) override;

private:
    ToComplexFunctor impl;
};
}

#endif//CPP_EXAMPLE_IMAGING_KERNELS_TO_COMPLEX_H
