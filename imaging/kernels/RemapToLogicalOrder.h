#ifndef CPP_EXAMPLE_IMAGING_KERNELS_REMAPTOLOGICALORDER_H
#define CPP_EXAMPLE_IMAGING_KERNELS_REMAPTOLOGICALORDER_H

#include "imaging/Kernel.h"
#include "imaging/Operation.h"
#include "imaging/ops/RemapToLogicalOrder.h"
#include "imaging/KernelRegistry.h"

namespace imaging {

class RemapToLogicalOrderFunctor {
public:
    static constexpr int BLOCK_TILE_DIM = 32;
    void operator()();
};

class RemapToLogicalOrderKernel : public Kernel {
public:
    explicit RemapToLogicalOrderKernel(KernelConstructionContext &ctx);
    void process(KernelExecutionContext &ctx) override;

private:
    // Output shape
    //    unsigned nFrames, nSamples, nChannels;
    //    NdArray fcmChannels, fcmFrames;
    RemapToLogicalOrderFunctor impl;
    short *fcmFramesPtr;
    char *fcmChannelsPtr;
};



}// namespace imaging

#endif//CPP_EXAMPLE_IMAGING_KERNELS_REMAPTOLOGICALORDER_H
