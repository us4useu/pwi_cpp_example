#include "RemapToLogicalOrder2.h"

#include <arrus/core/api/arrus.h>

namespace arrus_example_imaging {

RemapToLogicalOrder2Kernel::RemapToLogicalOrder2Kernel(KernelConstructionContext &ctx) : Kernel(ctx) {
    auto fcm = ctx.getInputMetadata()->getObject<arrus::devices::FrameChannelMapping>("frameChannelMapping");
    // Determining output dimensions.
    auto rawSequence = ctx.getInputMetadata()->getObject<arrus::ops::us4r::TxRxSequence>("rawSequence");

    // TODO Note: assumption that all TxRxs have the same number of samples. Validate that.
    auto [startSample, endSample] = rawSequence->getOps()[0].getRx().getSampleRange();
    nSequences = rawSequence->getNRepeats();
    nFrames = fcm->getNumberOfLogicalFrames();
    nSamples = endSample-startSample;
    nChannels = fcm->getNumberOfLogicalChannels();
    // Input: RF data: (total_n_samples, 32),
    // IQ data: (total_n_samples, 2, 32)
    auto inputOrder = ctx.getInput().getShape().size();
    nComponents = inputOrder == 2 ? 1: ctx.getInput().getShape()[1];

    // Prepare auxiliary arrays.
    std::vector<uint16_t> frames(nFrames*nChannels);
    std::vector<int8_t> channels(nFrames*nChannels);
    std::vector<uint8_t> us4oems(nFrames*nChannels);

    for(size_t frame = 0; frame < fcm->getNumberOfLogicalFrames(); ++frame) {
        for(int channel = 0; channel < fcm->getNumberOfLogicalChannels(); ++channel) {
            auto addr = fcm->getLogical(frame, channel);
            auto idx = frame*nChannels+channel;
            frames[idx] = addr.getFrame();
            channels[idx] = addr.getChannel();
            us4oems[idx] = addr.getUs4oem();
        }
    }
    fcmFrames = NdArray::asarray(frames, true).reshape({nFrames, nChannels});
    fcmChannels = NdArray::asarray(channels, true).reshape({nFrames, nChannels});
    fcmUs4oems = NdArray::asarray(us4oems, true).reshape({nFrames, nChannels});
    frameOffsets = NdArray::asarray(fcm->getFrameOffsets());
    nFramesUs4OEM = NdArray::asarray(fcm->getNumberOfFrames());

    ctx.setOutput(NdArrayDef{{nSequences, nFrames, nChannels, nSamples, nComponents}, ctx.getInput().getType()});
}

void RemapToLogicalOrder2Kernel::process(KernelExecutionContext &ctx) {
    impl(ctx.getOutput(), ctx.getInput(), fcmFrames, fcmChannels, fcmUs4oems, frameOffsets, nFramesUs4OEM,
         nSequences, nFrames, nChannels, nSamples, nComponents, ctx.getStream());
}

REGISTER_KERNEL_OP(OPERATION_CLASS_ID(RemapToLogicalOrder2), RemapToLogicalOrder2Kernel);

}// namespace arrus_example_imaging