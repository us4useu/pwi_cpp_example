#include "RemapToLogicalOrder.h"

namespace imaging {

RemapToLogicalOrderKernel::RemapToLogicalOrderKernel(KernelConstructionContext &ctx) : Kernel(ctx) {
    //    RemapToLogicalOrder(const NdArray fcmChannels, NdArray fcmFrames, unsigned nSamples) {
    //        : fcmChannels(std::move(fcmChannels)), fcmFrames(std::move(fcmFrames)), nSamples(nSamples) {
    //        nFrames = this->fcmFrames.getShape()[0];
    //        nChannels = this->fcmFrames.getShape()[1];
    //        fcmFramesPtr = this->fcmFrames.getPtr<short>();
    //        fcmChannelsPtr = this->fcmChannels.getPtr<char>();
    //    }
    //    KernelInitResult prepare(const KernelInitContext &ctx) override {
    //        return KernelInitResult({nFrames, nSamples, nChannels},
    //                                NdArray::DataType::INT16,
    //                                ctx.getInputSamplingFrequency());
    //    }
}

void RemapToLogicalOrderKernel::process(KernelExecutionContext &ctx) {
    //        dim3 block(BLOCK_TILE_DIM, BLOCK_TILE_DIM);
    //        dim3 grid((nChannels - 1) / block.x + 1,
    //                  (nSamples - 1) / block.y + 1,
    //                  nFrames);
    //        arrusRemap<<<grid, block, 0, stream>>>(
    //            output->getPtr<short>(), input->getConstPtr<short>(),
    //            fcmFramesPtr, fcmChannelsPtr,
    //            nFrames, nSamples, nChannels);
    //        CUDA_ASSERT(cudaGetLastError());
    impl();
}

//REGISTER_KERNEL(OPERATION_CLASS_ID(RemapToLogicalOrder), RemapToLogicalOrderKernel)

}// namespace imaging