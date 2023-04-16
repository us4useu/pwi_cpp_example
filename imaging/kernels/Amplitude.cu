#ifndef CPP_EXAMPLE_KERNELS_ENVELOPEDETECTION_CUH
#define CPP_EXAMPLE_KERNELS_ENVELOPEDETECTION_CUH

#include "Amplitude.h"

namespace arrus_example_imaging {
__global__ void gpuAmplitude(float *output, const int16_t *input, const unsigned totalNSamples) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= totalNSamples) {
        return;
    }
    int16_t i = input[2*idx];
    int16_t q = input[2*idx+1];
    output[idx] = hypotf(i, q);
}

void AmplitudeFunctor::operator()(NdArray &output, const NdArray &input, cudaStream_t stream) {
    dim3 block(512);
    unsigned totalNSamples = output.getNumberOfElements();
    dim3 grid((totalNSamples+block.x-1) / block.x);
    gpuAmplitude<<<grid, block, 0, stream>>>(output.getPtr<float>(), input.getConstPtr<int16_t>(),
                                                     totalNSamples);
    CUDA_ASSERT(cudaGetLastError());
}
}// namespace arrus_example_imaging

#endif
