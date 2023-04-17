#ifndef CPP_EXAMPLE_KERNELS_PHASE_CUH
#define CPP_EXAMPLE_KERNELS_PHASE_CUH

#include "Phase.h"

namespace arrus_example_imaging {
__global__ void gpuPhase(float *output, const float2 *input, const unsigned totalNSamples) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= totalNSamples) {
        return;
    }
    float2 value = input[idx];
    output[idx] = atan2f(value.y, value.x);
}

void PhaseFunctor::operator()(NdArray &output, const NdArray &input, cudaStream_t stream) {
    dim3 block(512);
    unsigned totalNSamples = input.getNumberOfElements();
    dim3 grid((totalNSamples + block.x - 1) / block.x);
    gpuPhase<<<grid, block, 0, stream>>>(output.getPtr<float>(), input.getConstPtr<float2>(),
                                                     totalNSamples);
    CUDA_ASSERT(cudaGetLastError());
}
}// namespace arrus_example_imaging

#endif
