#ifndef CPP_EXAMPLE_IMAGING_KERNELS_PHASE_H
#define CPP_EXAMPLE_IMAGING_KERNELS_PHASE_H

#include "imaging/Kernel.h"
#include "imaging/Operation.h"
#include "imaging/ops/Phase.h"
#include "imaging/KernelRegistry.h"

namespace arrus_example_imaging {

class PhaseFunctor {
public:
    void operator()(NdArray &output, const NdArray &input, cudaStream_t stream);
};

class PhaseKernel : public Kernel {
public:
    explicit PhaseKernel(KernelConstructionContext &ctx);
    void process(KernelExecutionContext &ctx) override;

private:
    PhaseFunctor impl;
};
}

#endif//CPP_EXAMPLE_IMAGING_KERNELS_PHASE_H
