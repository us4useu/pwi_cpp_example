#ifndef CPP_EXAMPLE_KERNELS_REMAPTOLOGICALORDER_CUH
#define CPP_EXAMPLE_KERNELS_REMAPTOLOGICALORDER_CUH

#include "imaging/CudaUtils.cuh"
#include "imaging/kernels/RemapToLogicalOrder2.h"

namespace arrus_example_imaging {

/**
 * Generic implementation of remapping the input data from us4OEM specific order (determined by the us4OEM
 * system architecture) to logical order, specific to TX/RX parameters set by user.
 * Due to its genericity, this kernel is not necessarily the most optimal way to implement the remapping.
 * NOTE: this operator also performs transposition from (..., nSamples, nChannels, ...) to (..., nChannels, nSamples...)
 *
 *
 * @param out: output array
 * @param in: input array
 * @param fcmFrames: frame channel mapping: frames
 * @param fcmChannels: frame channel mapping: channels
 * @param fcmUs4oems: frame channel mapping: us4oems
 * @parma frameOffsets: Number of frame (global), that starts given us4OEM data
 * @param nFramesUs4OEM: number of frames each us4OEM acquires
 * @param nSequences, nFrames, nSamples, nChannels, nComponents: output shape
 */
extern "C" __global__ void arrusRemap2(short *out,
                                       const short *in,
                                       const short *fcmFrames,
                                       const char *fcmChannels,
                                       const unsigned char *fcmUs4oems,
                                       const unsigned int *frameOffsets,
                                       const unsigned int *nFramesUs4OEM,
                                       const unsigned nSequences,
                                       const unsigned nFrames,
                                       const unsigned nSamples,
                                       const unsigned nChannels,
                                       const unsigned nComponents) {
    // TODO temporarily assuming, that maximum number of components == 2
    __shared__ short tile[REMAP2_BLOCK_TILE_DIM][REMAP2_BLOCK_TILE_DIM][2]; // NOTE: this is also the runtime block size.

    // gridDim.x*32 ~= number of channels
    // gridDim.y*32 ~= number of samples
    // Input.
    int channel = blockIdx.x*REMAP2_BLOCK_TILE_DIM + threadIdx.x; // logical channel
    int sample = blockIdx.y*REMAP2_BLOCK_TILE_DIM + threadIdx.y;  // logical sample
    int frame = blockIdx.z; // logical frame, global in the whole batch of sequences

    int sequence = frame/nFrames;
    int localFrame = frame%nFrames;
    if (channel >= nChannels || sample >= nSamples || localFrame >= nFrames || sequence >= nSequences) {
        return;
    }
    // FCM describes here a single sequence
    int physicalChannel = fcmChannels[localFrame*nChannels + channel];
    int physicalFrame = fcmFrames[localFrame*nChannels + channel];
    int us4oem = fcmUs4oems[localFrame*nChannels + channel];
    int us4oemOffset = frameOffsets[us4oem];
    int nPhysicalFrames = nFramesUs4OEM[us4oem];

    const int nus4OEMChannels = 32;
    // physical, input
    int pSampleSize = nus4OEMChannels*nComponents;  // total number of int16 values, for a single sampling time point
    int pFrameSize = pSampleSize*nSamples; // total number number of int16, for a single IQ frame

    // logical, output
    int lChannelSize = nSamples*nComponents; // total number of int16 values, for a single channel
    int lFrameSize = nChannels*lChannelSize; // total number of int16 values, for a single IQ frame
    int lSequenceSize = nFrames*lFrameSize; // total number of int16 values, for a single sequence of IQ frames
    short value = 0;

    // Copy -> transpose to the shared memory tile.
    for (unsigned component = 0; component < nComponents; ++component) {
        if (physicalChannel < 0) {
            value = 0;
        } else {
            // Note: optimistic assumption, that we get consecutive physical channel numbers
            // (and that is satisfied e.g. for full RX aperture on esaote3 adapter, like in plane wave imaging).
            size_t indexIn = us4oemOffset*pFrameSize
                + sequence*nPhysicalFrames*pFrameSize
                + physicalFrame*pFrameSize
                + sample*pSampleSize
                + component*nus4OEMChannels
                + physicalChannel;
            value = in[indexIn];
        }
        tile[threadIdx.y][threadIdx.x][component] = value;
    }
    __syncthreads();
    // Output
    int targetSample = blockIdx.y*REMAP2_BLOCK_TILE_DIM + threadIdx.x;
    int targetChannel = blockIdx.x*REMAP2_BLOCK_TILE_DIM + threadIdx.y;
    // Transpose -> copy to the output array.
    for (unsigned component = 0; component < nComponents; ++component) {
        // [sequence, frame, channel, sample, component]
        size_t indexOut =
            sequence*lSequenceSize
            + localFrame*lFrameSize
            + targetChannel*lChannelSize
            + targetSample*nComponents
            + component;
        out[indexOut] = tile[threadIdx.x][threadIdx.y][component];
    }
}

void RemapToLogicalOrder2Functor::operator()(NdArray &output,
                                            const NdArray &input,
                                            const NdArray &fcmFrames,
                                            const NdArray &fcmChannels,
                                            const NdArray &fcmUs4oems,
                                            const NdArray &frameOffsets,
                                            const NdArray &nFramesUs4OEM,
                                            const unsigned nSequences,
                                            const unsigned nFrames,
                                            const unsigned nSamples,
                                            const unsigned nChannels,
                                            const unsigned nComponents,
                                            cudaStream_t stream) {
    dim3 block(REMAP2_BLOCK_TILE_DIM, REMAP2_BLOCK_TILE_DIM);
    dim3 grid(
        (nChannels-1) / block.x + 1,
        (nSamples-1) / block.y + 1,
        nFrames*nSequences
    );
    arrusRemap2<<<grid, block, 0, stream>>>(
        output.getPtr<int16_t>(),
        input.getConstPtr<int16_t>(),
        fcmFrames.getConstPtr<int16_t>(),
        fcmChannels.getConstPtr<char>(),
        fcmUs4oems.getConstPtr<uint8_t>(),
        frameOffsets.getConstPtr<uint32_t>(),
        nFramesUs4OEM.getConstPtr<uint32_t>(),
        nSequences,
        nFrames,
        nSamples,
        nChannels,
        nComponents
    );
    CUDA_ASSERT(cudaGetLastError());
}

}// namespace arrus_example_imaging

#endif//CPP_EXAMPLE_KERNELS_REMAPTOLOGICALORDER_CUH
