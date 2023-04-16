#ifndef CPP_EXAMPLE_IMAGING_KERNELS_AMPLITUDE_H
#define CPP_EXAMPLE_IMAGING_KERNELS_AMPLITUDE_H

#include "imaging/Kernel.h"
#include "imaging/Operation.h"
#include "imaging/ops/Amplitude.h"
#include "imaging/KernelRegistry.h"

namespace arrus_example_imaging {

class AmplitudeFunctor {
public:
    void operator()(NdArray &output, const NdArray &input, cudaStream_t stream);
};

class AmplitudeKernel : public Kernel {
public:
    explicit AmplitudeKernel(KernelConstructionContext &ctx);
    void process(KernelExecutionContext &ctx) override;

private:
    AmplitudeFunctor impl;
};
}

#endif//CPP_EXAMPLE_IMAGING_KERNELS_AMPLITUDE_H
