#ifndef CPP_EXAMPLE_KERNELS_REMAPTOLOGICALORDER_CUH
#define CPP_EXAMPLE_KERNELS_REMAPTOLOGICALORDER_CUH

#include "imaging/kernels/RemapToLogicalOrder.h"
#include "imaging/CudaUtils.cu"
//#include "../NdArray.h"
//#include "../KernelInitResult.h"
//#include "../KernelConstructionContext.h"
//#include "../Kernel.h"

namespace imaging {

__global__ void arrusRemap(short *out, const short *in,
                           const short *fcmFrames,
                           const char *fcmChannels,
                           const unsigned nFrames,
                           const unsigned nSamples,
                           const unsigned nChannels) {
    int x = blockIdx.x * 32 + threadIdx.x; // logical channel
    int y = blockIdx.y * 32 + threadIdx.y; // logical sample
    int z = blockIdx.z; // logical frame
    if (x >= nChannels || y >= nSamples || z >= nFrames) {
        // outside the range
        return;
    }
    int indexOut = x + y * nChannels + z * nChannels * nSamples;
    int physicalChannel = fcmChannels[x + nChannels * z];
    if (physicalChannel < 0) {
        // channel is turned off
        return;
    }
    int physicalFrame = fcmFrames[x + nChannels * z];
    // 32 - number of channels in the physical mapping
    int indexIn = physicalChannel + y * 32 + physicalFrame * 32 * nSamples;
    out[indexOut] = in[indexIn];
}

void RemapToLogicalOrderFunctor::operator()() {
            dim3 block(BLOCK_TILE_DIM, BLOCK_TILE_DIM);
            // TODO
            dim3 grid(32, 32, 1);
//            dim3 grid((32 - 1) / block.x + 1,
//                      (nSamples - 1) / block.y + 1,
//                      nFrames);
            arrusRemap<<<grid, block, 0>>>(
                nullptr, nullptr,
                nullptr, nullptr,
                0, 0, 0
//                output->getPtr<short>(), input->getConstPtr<short>(),
//                fcmFramesPtr, fcmChannelsPtr,
//                nFrames, nSamples, nChannels
                );
            CUDA_ASSERT(cudaGetLastError());
}



}


#endif //CPP_EXAMPLE_KERNELS_REMAPTOLOGICALORDER_CUH
