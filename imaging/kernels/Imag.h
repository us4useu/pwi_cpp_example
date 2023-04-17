#ifndef CPP_EXAMPLE_IMAGING_KERNELS_IMAG_H
#define CPP_EXAMPLE_IMAGING_KERNELS_IMAG_H

#include "imaging/Kernel.h"
#include "imaging/Operation.h"
#include "imaging/ops/Imag.h"
#include "imaging/KernelRegistry.h"

namespace arrus_example_imaging {

class ImagFunctor {
public:
    void operator()(NdArray &output, const NdArray &input, cudaStream_t stream);
};

class ImagKernel : public Kernel {
public:
    explicit ImagKernel(KernelConstructionContext &ctx);
    void process(KernelExecutionContext &ctx) override;

private:
    ImagFunctor impl;
};
}

#endif//CPP_EXAMPLE_IMAGING_KERNELS_IMAG_H
