#ifndef CPP_EXAMPLE_IMAGING_KERNELS_REMAPTOLOGICALORDER2_H
#define CPP_EXAMPLE_IMAGING_KERNELS_REMAPTOLOGICALORDER2_H

#include "imaging/Kernel.h"
#include "imaging/Operation.h"
#include "imaging/ops/RemapToLogicalOrder2.h"
#include "imaging/KernelRegistry.h"


#define REMAP2_BLOCK_TILE_DIM 32

namespace arrus_example_imaging {

class RemapToLogicalOrder2Functor {
public:
    void operator()(NdArray& output, const NdArray& input,
                    const NdArray& fcmFrames, const NdArray& fcmChannels, const NdArray &fcmUs4oems,
                    const NdArray& frameOffsets, const NdArray& nFramesUs4OEM,
                    unsigned nSequences, unsigned nFrames, unsigned nSamples, unsigned nChannels, unsigned nComponents,
                    cudaStream_t stream);
};

class RemapToLogicalOrder2Kernel : public Kernel {
public:
    explicit RemapToLogicalOrder2Kernel(KernelConstructionContext &ctx);
    void process(KernelExecutionContext &ctx) override;

private:
    NdArray fcmFrames;
    NdArray fcmChannels;
    NdArray fcmUs4oems;
    NdArray frameOffsets;
    NdArray nFramesUs4OEM;
    // number of logical sequences, frames, ...
    unsigned nSequences, nFrames, nSamples, nChannels, nComponents;
    RemapToLogicalOrder2Functor impl;
};



}// namespace arrus_example_imaging

#endif//CPP_EXAMPLE_IMAGING_KERNELS_REMAPTOLOGICALORDER2_H
